from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from darwin_attack.config import ConfigurationError, load_config
from darwin_attack.schemas import StrategyCandidate
from darwin_attack.storage import Repository
from darwin_guard.attack_bridge import DARWINAttackAdapter, validate_attack_profile
from darwin_guard.evaluation import check_evaluation_isolation


class GuardAttackConfigurationContextTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        release = Path(__file__).resolve().parents[2]
        self.raw = yaml.safe_load((release / "configs/darwin_attack.example.yaml").read_text())
        self.raw["runtime"].update(database_path="strategies.sqlite3", random_seed=7)
        for model in self.raw["models"].values():
            model["max_tokens"] = 32
        self.raw["models"]["target"].update(
            provider="transformers", model="Qwen3Guard-Gen-8B", temperature=0,
        )
        self.raw["embedding"]["history_threshold"] = 0.9
        self.raw["sandbox"].update(
            dataset_path="sandbox.jsonl", goals_per_candidate=1, trials_per_goal=1,
        )
        self.raw["attack"].update(
            dataset_path="evaluation.jsonl", dataset_id="evaluation", target_id="guard",
            target_kind="guardrail", guardrail_safe_pattern=r"^Safety: Safe$",
            guardrail_unsafe_pattern=r"^Safety: Unsafe$",
        )
        (self.root / "sandbox.jsonl").write_text('{"goal":"neutral sandbox fixture"}\n')
        (self.root / "evaluation.jsonl").write_text('{"goal":"neutral evaluation fixture"}\n')
        self.path = self.root / "attack.yaml"
        self.save()

    def save(self):
        self.path.write_text(yaml.safe_dump(self.raw))

    def test_standalone_target_contract_remains_required_by_default(self):
        with self.assertRaisesRegex(ConfigurationError, "guardrail_template"):
            load_config(self.path)
        config = load_config(self.path, validate_target=False)
        self.assertEqual(config.attack.target_kind, "guardrail")
        self.assertIsNone(config.attack.guardrail_template)

    def test_guard_context_keeps_core_configuration_and_paper_checks(self):
        self.assertEqual(validate_attack_profile(self.path)["numeric_profile"], "Table 5")
        self.raw["attack"]["success_score"] = 0
        self.save()
        with self.assertRaisesRegex(ConfigurationError, "success_score"):
            load_config(self.path, validate_target=False)
        self.raw["attack"]["success_score"] = 5
        self.raw["selection"]["gamma"] = 0.9
        self.save()
        with self.assertRaisesRegex(ValueError, "Table 5"):
            validate_attack_profile(self.path)

    def test_real_adapter_initializes_without_the_unused_target_contract(self):
        database = self.root / "strategies.sqlite3"
        repository = Repository(database)
        try:
            for index in range(50):
                repository.add_released_strategy(
                    StrategyCandidate(f"fixture-{index}", "Format", "Format {input}", "template"),
                    [1.0, 0.0],
                )
        finally:
            repository.close()
        with patch("darwin_attack.providers.build_model") as build_model:
            adapter = DARWINAttackAdapter(str(self.path), str(database), 1, 7)
            try:
                self.assertEqual(len(adapter.repository.active_strategies()), 50)
                build_model.assert_not_called()
            finally:
                adapter.close()

    def test_isolation_reader_skips_target_contract_but_checks_sandbox_text(self):
        config = {"attack": {"config_path": str(self.path)}}
        result = check_evaluation_isolation([{"text": "different neutral fixture"}], config)
        self.assertIn("attack.sandbox", result["checked_inputs"])
        with self.assertRaisesRegex(ValueError, "overlap with attack.sandbox"):
            check_evaluation_isolation([{"text": "neutral sandbox fixture"}], config)


if __name__ == "__main__":
    unittest.main()
