from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from darwin_guard.config import pairs_per_round
from darwin_guard.online import atomic_json, run_online


class FakeWorker:

    def __init__(self, *, fail_at=None, bad_budget=False, missing_model=False):
        self.calls = []
        self.exclusions = {}
        self.previous_failures = {}
        self.database_counts = {}
        self.fail_at = fail_at
        self.bad_budget = bad_budget
        self.missing_model = missing_model

    def __call__(self, stage, config_path, round_dir, model_path, round_index):
        self.calls.append((stage, round_index, model_path))
        config = json.loads(config_path.read_text())
        if stage == "collect":
            self.exclusions[round_index] = json.loads((round_dir / "excluded_source_ids.json").read_text())
            self.previous_failures[round_index] = json.loads((round_dir / "previous_failures.json").read_text())
            with sqlite3.connect(round_dir / "strategies.sqlite") as db:
                self.database_counts[round_index] = db.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
                before = self.database_counts[round_index]
                remaining = config["online"]["rounds"] - round_index + 1
                added = (200 - before + remaining - 1) // remaining
                db.executemany("INSERT INTO strategies(status) VALUES (?)", [("active",)] * added)
            if self.fail_at == (stage, round_index):
                raise RuntimeError("injected collection failure")
            atomic_json(round_dir / "failures.json", [{"source_id": f"round-{round_index}-rejected"}])
            atomic_json(round_dir / "collection_complete.json", {
                "pair_count": pairs_per_round(config) - int(self.bad_budget),
                "used_source_ids": [f"round-{round_index}-retained", f"round-{round_index}-rejected"],
            })
        else:
            if self.fail_at == (stage, round_index):
                raise RuntimeError("injected training failure")
            checkpoint = round_dir / "model"
            checkpoint.mkdir(exist_ok=True)
            if not self.missing_model:
                atomic_json(checkpoint / "config.json", {"test_fixture": True})
            atomic_json(round_dir / "training_complete.json", {"checkpoint": str(checkpoint)})


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.initial = self.root / "seeds.sqlite"
        with sqlite3.connect(self.initial) as db:
            db.execute("CREATE TABLE strategies(id INTEGER PRIMARY KEY, status TEXT NOT NULL)")
            db.executemany("INSERT INTO strategies(status) VALUES (?)", [("active",)] * 50)
        self.config = {
            "runtime": {"output_dir": str(self.root / "run"), "seed": 42},
            "models": {"guard": {"path": str(self.root / "base-model")}},
            "attack": {"initial_database": str(self.initial)},
            "online": {"rounds": 2, "examples_per_round": 500, "examples_unit": "rows"},
        }

    def run_fake(self, worker, *, resume=False, config=None):
        with redirect_stdout(io.StringIO()):
            return run_online(config or self.config, resume=resume, worker=worker)

    def state(self):
        return json.loads((self.root / "run" / "state.json").read_text())

    def test_previous_checkpoint_strategy_state_and_source_exclusions_flow_forward(self):
        worker = FakeWorker()
        state = self.run_fake(worker)
        base = self.config["models"]["guard"]["path"]
        first = str(self.root / "run" / "round_0001" / "model")
        second = str(self.root / "run" / "round_0002" / "model")
        self.assertEqual(worker.calls, [("collect", 1, base), ("train", 1, base),
                                        ("collect", 2, first), ("train", 2, first)])
        self.assertEqual(worker.exclusions[1], [])
        self.assertEqual(set(worker.exclusions[2]), {"round-1-retained", "round-1-rejected"})
        self.assertEqual(worker.database_counts, {1: 50, 2: 125})
        self.assertEqual(worker.previous_failures[2], [{"source_id": "round-1-rejected"}])
        self.assertEqual(state["guard_checkpoint"], second)
        self.assertEqual(state["completed_rounds"], 2)
        self.assertEqual(len(state["used_source_ids"]), 4)
        with sqlite3.connect(self.initial) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM strategies").fetchone()[0], 50)

    def test_completed_resume_does_not_repeat_workers(self):
        expected = self.run_fake(FakeWorker())
        worker = FakeWorker()
        self.assertEqual(self.run_fake(worker, resume=True), expected)
        self.assertEqual(worker.calls, [])

    def test_inactive_seed_rows_fail_before_starting_workers(self):
        with sqlite3.connect(self.initial) as db:
            db.execute("INSERT INTO strategies(status) VALUES ('inactive')")
        worker = FakeWorker()
        with self.assertRaisesRegex(ValueError, "only active admitted"):
            self.run_fake(worker)
        self.assertEqual(worker.calls, [])

    def test_existing_state_requires_resume(self):
        self.run_fake(FakeWorker())
        with self.assertRaisesRegex(RuntimeError, "resume"):
            self.run_fake(FakeWorker())

    def test_resume_rejects_changed_config(self):
        self.run_fake(FakeWorker())
        changed = copy.deepcopy(self.config)
        changed["runtime"]["seed"] += 1
        worker = FakeWorker()
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            self.run_fake(worker, resume=True, config=changed)
        self.assertEqual(worker.calls, [])

    def test_resume_rejects_edited_attack_configuration_content(self):
        attack = self.root / "attack.yaml"
        attack.write_text("generation: first\n")
        self.config["attack"]["config_path"] = str(attack)
        self.run_fake(FakeWorker())
        attack.write_text("generation: second\n")
        with self.assertRaisesRegex(RuntimeError, "configuration differs"):
            self.run_fake(FakeWorker(), resume=True)

    def test_failure_does_not_commit_partial_round(self):
        for stage in ("collect", "train"):
            with self.subTest(stage=stage):
                config = copy.deepcopy(self.config)
                config["runtime"]["output_dir"] = str(self.root / stage)
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    self.run_fake(FakeWorker(fail_at=(stage, 2)), config=config)
                state = json.loads((self.root / stage / "state.json").read_text())
                self.assertEqual(state["completed_rounds"], 1)
                self.assertEqual(state["guard_checkpoint"], str(self.root / stage / "round_0001" / "model"))
                self.assertEqual(set(state["used_source_ids"]), {"round-1-retained", "round-1-rejected"})

    def test_training_failure_resume_reuses_completed_collection(self):
        with self.assertRaisesRegex(RuntimeError, "injected training"):
            self.run_fake(FakeWorker(fail_at=("train", 2)))
        worker = FakeWorker()
        self.assertEqual(self.run_fake(worker, resume=True)["completed_rounds"], 2)
        self.assertEqual(worker.calls, [("train", 2, str(self.root / "run" / "round_0001" / "model"))])

    def test_collection_failure_resume_discards_partial_strategy_mutations(self):
        with self.assertRaisesRegex(RuntimeError, "injected collection"):
            self.run_fake(FakeWorker(fail_at=("collect", 2)))
        worker = FakeWorker()
        self.run_fake(worker, resume=True)
        self.assertEqual(worker.database_counts[2], 125)
        self.assertEqual([item[:2] for item in worker.calls], [("collect", 2), ("train", 2)])

    def test_invalid_collection_budget_does_not_train_or_commit(self):
        worker = FakeWorker(bad_budget=True)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.run_fake(worker)
        self.assertEqual(self.state()["completed_rounds"], 0)
        self.assertEqual([item[:2] for item in worker.calls], [("collect", 1)])

    def test_missing_model_config_does_not_commit(self):
        with self.assertRaisesRegex(RuntimeError, "model configuration"):
            self.run_fake(FakeWorker(missing_model=True))
        self.assertEqual(self.state()["completed_rounds"], 0)

    def test_final_round_with_too_few_admitted_strategies_is_not_trained_or_committed(self):
        class ShortPoolWorker(FakeWorker):
            def __call__(self, stage, config_path, round_dir, model_path, round_index):
                super().__call__(stage, config_path, round_dir, model_path, round_index)
                if stage == "collect" and round_index == 2:
                    with sqlite3.connect(round_dir / "strategies.sqlite") as db:
                        db.execute("DELETE FROM strategies WHERE id > 150")

        worker = ShortPoolWorker()
        with self.assertRaisesRegex(RuntimeError, "200 admitted strategies; found 150"):
            self.run_fake(worker)
        self.assertEqual(self.state()["completed_rounds"], 1)
        self.assertNotIn(("train", 2), [call[:2] for call in worker.calls])

    def test_completed_resume_rechecks_actual_final_pool(self):
        state = self.run_fake(FakeWorker())
        with sqlite3.connect(state["strategy_database"]) as db:
            db.execute("DELETE FROM strategies WHERE id = 200")
        with self.assertRaisesRegex(RuntimeError, "200 admitted strategies; found 199"):
            self.run_fake(FakeWorker(), resume=True)

    def test_nonseed_initial_pool_is_rejected(self):
        with sqlite3.connect(self.initial) as db:
            db.execute("INSERT INTO strategies(status) VALUES ('active')")
        worker = FakeWorker()
        with self.assertRaisesRegex(ValueError, "50 admitted seeds"):
            self.run_fake(worker)
        self.assertEqual(worker.calls, [])

    def test_orphan_round_directory_requires_a_new_run_directory(self):
        orphan = self.root / "run" / "round_0001"
        orphan.mkdir(parents=True)
        atomic_json(orphan / "collection_complete.json", {"pair_count": 250, "used_source_ids": []})
        worker = FakeWorker()
        with self.assertRaises(RuntimeError):
            self.run_fake(worker)
        self.assertEqual(worker.calls, [])

    def test_completed_resume_checks_checkpoint_and_strategy_database(self):
        state = self.run_fake(FakeWorker())
        config_path = Path(state["guard_checkpoint"]) / "config.json"
        original_config = config_path.read_bytes()
        config_path.unlink()
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.run_fake(FakeWorker(), resume=True)
        config_path.write_bytes(original_config)
        Path(state["strategy_database"]).unlink()
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.run_fake(FakeWorker(), resume=True)


if __name__ == "__main__":
    unittest.main()
