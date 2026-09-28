from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ReleaseBoundaryTest(unittest.TestCase):
    def test_release_contains_exact_strategy_artifacts_and_no_datasets(self):
        strategy_payloads = [
            path.name
            for path in (ROOT / "strategies").iterdir()
            if path.name not in {".gitkeep", "README.md"}
        ]
        data_payloads = [path for path in (ROOT / "data").iterdir() if path.name != ".gitkeep"]
        self.assertEqual(
            sorted(strategy_payloads),
            ["final_strategy_pool.jsonl", "mutation_operators.jsonl"],
        )
        self.assertEqual(data_payloads, [])

    def test_final_pool_has_200_released_unique_strategies(self):
        path = ROOT / "strategies" / "final_strategy_pool.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 200)
        self.assertEqual(len({record["key"] for record in records}), 200)
        self.assertEqual(
            [record["key"] for record in records],
            [f"strategy-{index:03d}" for index in range(1, 201)],
        )
        forbidden_fields = {
            "source",
            "source_group",
            "source_path",
            "source_id",
            "validation_run_id",
            "sandbox_success_rate",
            "sandbox_average_score",
        }
        for record in records:
            self.assertTrue(forbidden_fields.isdisjoint(record))
            if record["mode"] == "template":
                self.assertIn("{input}", record["instruction"])

    def test_release_contains_fifteen_mutation_operators(self):
        path = ROOT / "strategies" / "mutation_operators.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 15)
        self.assertEqual(len({record["name"] for record in records}), 15)
        self.assertEqual(len({record["dimension"] for record in records}), 5)

    def test_no_local_identity_or_secret_literal(self):
        forbidden = (
            "/" + "home" + "/",
            "sk" + "-ant-",
            "sk" + "-proj-",
            "sk" + "-sb",
        )
        for path in ROOT.rglob("*"):
            text_file = path.suffix in {
                ".cff",
                ".json",
                ".jsonl",
                ".md",
                ".py",
                ".toml",
                ".txt",
                ".yaml",
                ".yml",
            } or path.name in {".gitignore"}
            if not path.is_file() or not text_file:
                continue
            text = path.read_text(encoding="utf-8")
            self.assertFalse(any(token in text for token in forbidden), path)


if __name__ == "__main__":
    unittest.main()
