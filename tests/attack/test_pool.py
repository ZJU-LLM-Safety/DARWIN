from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from darwin_attack.pool import StrategyPool
from darwin_attack.cli import _admit_candidates
from darwin_attack.schemas import SandboxReport, StrategyCandidate
from darwin_attack.storage import Repository
from darwin_attack.utils import read_jsonl

from fakes import FakeEmbedder


def candidate(key: str, instruction: str) -> StrategyCandidate:
    return StrategyCandidate(
        key=key,
        name=key,
        instruction=instruction,
        mode="template",
    )


def admission_config(database: Path) -> SimpleNamespace:
    sandbox = database.parent / "sandbox.jsonl"
    evaluation = database.parent / "evaluation.jsonl"
    sandbox.write_text('{"goal":"sandbox fixture"}\n', encoding="utf-8")
    evaluation.write_text('{"goal":"evaluation fixture"}\n', encoding="utf-8")
    return SimpleNamespace(
        runtime=SimpleNamespace(database_path=database),
        sandbox=SimpleNamespace(dataset_path=sandbox),
        attack=SimpleNamespace(dataset_path=evaluation),
    )


class PoolTest(unittest.TestCase):
    def test_cli_skips_sandbox_for_duplicates_and_full_pool(self):
        for target_size, key, instruction, reason in (
            (10, "alpha", "beta wrapper {input}", "duplicate_key"),
            (10, "alpha-copy", "alpha equivalent {input}", "semantic_duplicate"),
            (1, "beta", "beta wrapper {input}", "pool_full"),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "state.sqlite3"
                repository = Repository(database)
                repository.add_released_strategy(
                    candidate("alpha", "alpha wrapper {input}"),
                    np.asarray([1.0, 0.0], dtype=np.float32),
                )
                repository.close()
                config = admission_config(database)
                output = Path(directory) / "admission.jsonl"
                with patch(
                    "darwin_attack.cli._pool",
                    side_effect=lambda _, repo: StrategyPool(repo, FakeEmbedder(), 0.95, 0.80, target_size),
                ), patch("darwin_attack.cli._sandbox") as sandbox, redirect_stdout(StringIO()):
                    _admit_candidates(config, [candidate(key, instruction)], output)
                sandbox.assert_not_called()
                record = list(read_jsonl(output))[0]
                self.assertEqual(record["reason"], reason)
                self.assertTrue(record["sandbox_skipped"])
                self.assertIsNone(record["sandbox_success_rate"])
                self.assertIsNone(record["sandbox_average_score"])

    def test_cli_validates_unique_candidate_and_rejects_low_success_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            config = admission_config(database)
            output = Path(directory) / "admission.jsonl"
            sandbox = Mock()
            sandbox.validate.return_value = SandboxReport(3, 5, 3.0)
            unique = candidate("beta-low", "beta wrapper {input}")
            with patch(
                "darwin_attack.cli._pool",
                side_effect=lambda _, repo: StrategyPool(repo, FakeEmbedder(), 0.95, 0.80, 10),
            ), patch("darwin_attack.cli._sandbox", return_value=sandbox), redirect_stdout(StringIO()):
                _admit_candidates(config, [unique], output)
            sandbox.validate.assert_called_once_with(unique)
            record = list(read_jsonl(output))[0]
            self.assertFalse(record["admitted"])
            self.assertFalse(record["sandbox_skipped"])
            self.assertEqual(record["sandbox_success_rate"], 0.6)
            self.assertEqual(record["reason"], "below_sandbox_threshold")
            repository = Repository(database)
            self.assertEqual(repository.count_strategies(), 0)
            repository.close()

    def test_key_dedup_semantic_dedup_and_sandbox_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Repository(Path(directory) / "state.sqlite3")
            pool = StrategyPool(repository, FakeEmbedder(), 0.95, 0.80, 10)
            passing = SandboxReport(successes=4, trials=5, average_score=4.0)
            failing = SandboxReport(successes=3, trials=5, average_score=3.0)

            first = pool.consider(candidate("alpha", "alpha wrapper {input}"), passing)
            self.assertTrue(first.admitted)
            self.assertEqual(first.strategy_id, 1)

            duplicate_key = pool.consider(candidate("alpha", "beta wrapper {input}"), passing)
            self.assertEqual(duplicate_key.reason, "duplicate_key")

            semantic_duplicate = pool.consider(
                candidate("alpha-copy", "alpha equivalent {input}"), passing
            )
            self.assertEqual(semantic_duplicate.reason, "semantic_duplicate")

            below_gate = pool.consider(candidate("beta-low", "beta wrapper {input}"), failing)
            self.assertEqual(below_gate.reason, "below_sandbox_threshold")
            self.assertEqual(repository.count_strategies(), 1)

            second = pool.consider(candidate("beta", "beta wrapper {input}"), passing)
            self.assertTrue(second.admitted)
            self.assertEqual(repository.count_strategies(), 2)
            repository.close()

    def test_released_pool_entries_do_not_invent_sandbox_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Repository(Path(directory) / "state.sqlite3")
            repository.add_released_strategy(
                candidate("released", "released wrapper {input}"),
                np.asarray([1.0, 0.0], dtype=np.float32),
            )
            record = repository.active_strategies()[0]
            self.assertEqual(record.validation_status, "released_prevalidated")
            self.assertIsNone(record.sandbox_success_rate)
            self.assertIsNone(record.sandbox_average_score)
            repository.close()


if __name__ == "__main__":
    unittest.main()
