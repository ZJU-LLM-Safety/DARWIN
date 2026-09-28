from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from darwin_attack import cli


class CliDatasetIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sandbox = self.root / "sandbox.jsonl"
        self.evaluation = self.root / "evaluation.jsonl"
        self.database = self.root / "strategies.sqlite"
        self.inputs = self.root / "candidates.jsonl"
        self.inputs.write_text(json.dumps({
            "key": "candidate", "name": "Candidate", "instruction": "Describe {input}",
            "mode": "template",
        }) + "\n", encoding="utf-8")
        self.config = SimpleNamespace(
            sandbox=SimpleNamespace(dataset_path=self.sandbox),
            attack=SimpleNamespace(dataset_path=self.evaluation),
            runtime=SimpleNamespace(database_path=self.database),
            pool=SimpleNamespace(target_size=1),
            embedding=SimpleNamespace(),
        )
        self.args = SimpleNamespace(
            config=self.root / "config.yaml", input=self.inputs,
            output=self.root / "result.jsonl", count=1, limit=1,
        )

    def write_datasets(self, sandbox_text="shared task", evaluation_text="shared task"):
        self.sandbox.write_text(json.dumps({
            "id": "sandbox-local-id", "goal": sandbox_text,
        }) + "\n", encoding="utf-8")
        self.evaluation.write_text("\n".join(json.dumps(row) for row in (
            {"id": "first", "goal": "another task"},
            {"id": "different-evaluation-id", "goal": evaluation_text},
        )) + "\n", encoding="utf-8")

    def test_overlap_rejected_before_admission_or_attack(self):
        self.write_datasets("shared  task\ntext", "shared task text")
        for handler in (cli.cmd_admit, cli.cmd_evolve, cli.cmd_reflect, cli.cmd_attack):
            with self.subTest(command=handler.__name__), \
                    patch.object(cli, "load_config", return_value=self.config), \
                    patch.object(cli, "Repository") as repository, \
                    patch.object(cli, "build_model") as build_model, \
                    patch.object(cli, "SentenceTransformerEmbedder") as embedder, \
                    self.assertRaisesRegex(ValueError, "overlapping prompts"):
                handler(self.args)
            repository.assert_not_called()
            build_model.assert_not_called()
            embedder.assert_not_called()
            self.assertFalse(self.database.exists())
            self.assertFalse(self.args.output.exists())

    def test_admission_helper_rejects_overlap_even_without_candidates(self):
        self.write_datasets()
        self.database.write_bytes(b"existing database must remain untouched")
        with patch.object(cli, "Repository") as repository, \
                patch.object(cli, "build_model") as build_model, \
                self.assertRaisesRegex(ValueError, "overlapping prompts"):
            cli._admit_candidates(self.config, [], self.args.output)
        repository.assert_not_called()
        build_model.assert_not_called()
        self.assertEqual(self.database.read_bytes(), b"existing database must remain untouched")

    def test_direct_sandbox_creation_checks_before_loading_models(self):
        self.write_datasets()
        with patch.object(cli, "build_model") as build_model, \
                self.assertRaisesRegex(ValueError, "overlapping prompts"):
            cli._sandbox(self.config)
        build_model.assert_not_called()

    def test_missing_evaluation_data_cannot_silently_approve_admission(self):
        self.write_datasets()
        self.evaluation.unlink()
        with patch.object(cli, "Repository") as repository, \
                self.assertRaises(FileNotFoundError):
            cli._admit_candidates(self.config, [], self.args.output)
        repository.assert_not_called()

    def test_disjoint_data_preserve_original_text_and_ignore_dataset_local_ids(self):
        self.sandbox.write_text(json.dumps({"id": "1", "goal": "sandbox  task"}) + "\n")
        self.evaluation.write_text(json.dumps({"id": "1", "goal": "evaluation task"}) + "\n")
        sandbox, evaluation = cli._checked_datasets(self.config)
        self.assertEqual(sandbox, [("1", "sandbox  task")])
        self.assertEqual(evaluation, [("1", "evaluation task")])

    def test_released_pool_import_is_independent_of_dataset_files(self):
        for dataset_state in ("unavailable", "overlapping", "malformed"):
            if self.database.exists():
                self.database.unlink()
            if dataset_state == "overlapping":
                self.write_datasets()
            elif dataset_state == "malformed":
                self.sandbox.write_text("not JSON", encoding="utf-8")
                self.evaluation.write_text("not JSON", encoding="utf-8")
            with self.subTest(dataset_state=dataset_state), \
                    patch.object(cli, "load_config", return_value=self.config), \
                    patch.object(cli, "load_goals", wraps=cli.load_goals) as load_goals, \
                    patch.object(cli, "SentenceTransformerEmbedder") as embedder, \
                    patch.object(cli, "build_model") as build_model, \
                    contextlib.redirect_stdout(io.StringIO()):
                embedder.return_value.encode.return_value = [[1.0, 0.0]]
                cli.cmd_load_released_pool(self.args)
            load_goals.assert_not_called()
            build_model.assert_not_called()
            with sqlite3.connect(self.database) as connection:
                rows = connection.execute(
                    "SELECT strategy_key, validation_status FROM strategies"
                ).fetchall()
            self.assertEqual(rows, [("candidate", "released_prevalidated")])


if __name__ == "__main__":
    unittest.main()
