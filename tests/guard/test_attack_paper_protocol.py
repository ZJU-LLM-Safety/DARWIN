from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_attack_bridge as bridge_fixtures
from darwin_guard.attack_bridge import grow_pool_for_round, validate_attack_profile


class AttackPaperProtocolTests(unittest.TestCase):
    def setUp(self):
        self.fixture = bridge_fixtures.BridgeTest(methodName="test_initialization_does_not_load_chat_models")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.proposer = self.fixture.proposer
        self.counter = 0
        self.batch_seeds = []
        owner = self

        class Genetic:
            def __init__(self, **kwargs):
                owner.batch_seeds.append(kwargs["random_seed"])

            def generate(self, count):
                result = []
                for _ in range(count):
                    owner.counter += 1
                    result.append(bridge_fixtures.Strategy(
                        0, f"candidate-{owner.counter}",
                        instruction=f"distinct presentation {owner.counter}", metadata={},
                    ))
                return result

        self.fixture.api.GeneticStrategyEvolution = Genetic
        self.settings = {
            "genetic_candidates": 2, "reflection_candidates": 1,
            "max_evolution_batches": 100,
        }

    def test_twenty_round_default_can_reach_200_without_skipping_admission(self):
        sizes = [50]
        for round_index in range(1, 21):
            self.proposer.round_index = round_index
            report = grow_pool_for_round(self.proposer, self.settings, 20)
            sizes.append(report["after"])
            self.assertEqual(report["after"], report["target_size"])
            self.assertEqual(report["admitted"], report["after"] - report["before"])
        self.assertEqual(sizes[-1], 200)
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(self.fixture.sandbox_calls), 150)
        self.assertEqual(len(set(self.fixture.sandbox_calls)), 150)

    def test_last_batch_stops_admitting_at_explicit_round_target(self):
        targets = [53] * 19 + [200]
        report = grow_pool_for_round(self.proposer, dict(self.settings, round_pool_targets=targets), 20)
        self.assertEqual(report["after"], 53)
        self.assertEqual(report["generated"], 4)
        self.assertEqual(len(self.fixture.sandbox_calls), 3)
        self.assertEqual(report["batches"][-1]["decisions"][-1]["reason"], "pool_full")
        self.assertEqual(report["schedule"], "explicit_round_pool_targets")

    def test_rejected_candidates_cannot_fill_target_and_failure_is_bounded(self):
        class WeakGenetic:
            def __init__(self, **kwargs):
                pass

            def generate(self, count):
                return [bridge_fixtures.Strategy(0, "weak", instruction="weak variant", metadata={})]

        self.fixture.api.GeneticStrategyEvolution = WeakGenetic
        with self.assertRaisesRegex(RuntimeError, "exhausted 3 batches") as caught:
            grow_pool_for_round(self.proposer, dict(self.settings, max_evolution_batches=3), 20)
        report = caught.exception.evolution_report
        self.assertEqual(report["after"], 50)
        self.assertEqual(report["admitted"], 0)
        self.assertEqual(len(report["batches"]), 3)
        self.assertEqual(self.proposer.seed, 42)
        self.assertEqual(len(self.fixture.sandbox_calls), 3)

    def test_batches_vary_seed_reflection_and_extract_external_once(self):
        material_calls = []
        self.fixture.api.read_jsonl = lambda path: [{"text": "public research material"}]
        self.fixture.api.ExternalKnowledgeEvolution = lambda model: SimpleNamespace(
            extract=lambda text, source: material_calls.append(text) or [],
        )
        failures = [
            {"strategy_id": i, "failed_prompt": f"failure {i}", "feedback": "guard blocked"}
            for i in range(1, 6)
        ]
        self.proposer.failure_records = failures
        report = grow_pool_for_round(
            self.proposer, dict(self.settings, external_material="materials.jsonl"), 20,
        )
        self.assertGreater(len(report["batches"]), 1)
        self.assertEqual(len(material_calls), 1)
        self.assertEqual(len(set(self.batch_seeds)), len(self.batch_seeds))
        self.assertEqual([record[0] for record in self.fixture.reflection_calls], [1, 2, 3, 4])
        self.assertIs(self.proposer.failure_records, failures)
        self.assertEqual(self.proposer.seed, 42)

    def test_full_pool_does_not_generate_candidates_but_still_checks_isolation(self):
        for i in range(51, 201):
            self.fixture.repository.strategies.append(bridge_fixtures.Strategy(i, f"seed-{i}"))
        with patch.object(self.proposer, "validate_source_isolation", wraps=self.proposer.validate_source_isolation) as isolation:
            report = grow_pool_for_round(self.proposer, self.settings, 20)
        isolation.assert_called_once()
        self.assertEqual(report["after"], 200)
        self.assertEqual(report["batches"], [])
        self.assertEqual(self.batch_seeds, [])

    def test_explicit_schedule_rejects_nonmonotonic_and_wrong_final_values(self):
        for schedule in ([60, 55] + [200] * 18, [60] * 20, [200] * 19):
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, "round_pool_targets"):
                    grow_pool_for_round(self.proposer, dict(self.settings, round_pool_targets=schedule), 20)

    def test_table5_mismatch_is_rejected_before_models_load(self):
        config = copy.deepcopy(self.fixture.config)
        config.attack.max_chain_length = 3
        config.pool.mutation_operator_count = 15
        config.model = lambda role: bridge_fixtures.ModelConfig(temperature=0.7)
        validate_attack_profile(config)
        for section, key, value in (
            ("pool", "target_size", 100), ("pool", "admission_threshold", 0.1),
            ("selection", "alpha", 0.9), ("selection", "gamma", 0.9),
            ("attack", "max_chain_length", 2), ("attack", "success_score", 4),
            ("embedding", "similarity_threshold", 0.9),
        ):
            altered = copy.deepcopy(config)
            setattr(getattr(altered, section), key, value)
            with self.subTest(parameter=f"{section}.{key}"):
                with self.assertRaisesRegex(ValueError, "Table 5"):
                    validate_attack_profile(altered)
        self.assertEqual(self.fixture.builds, [])

    def test_declared_identity_is_checked_independently_of_deployment_alias(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.fixture.config.attack.max_chain_length = 3
        self.fixture.config.pool.mutation_operator_count = 15
        self.fixture.config.model = lambda role: bridge_fixtures.ModelConfig(temperature=0.7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack.yaml"
            path.write_text("models:\n  judge:\n    identity: GPT-4o\n    model: service-alias\n")
            report = validate_attack_profile(path)
            self.assertEqual(report["verified_model_declarations"], ["judge"])
            path.write_text("models:\n  judge:\n    identity: different-model\n    model: service-alias\n")
            with self.assertRaisesRegex(ValueError, "models.judge.identity"):
                validate_attack_profile(path)
            path.write_text("embedding:\n  identity: wrong-embedding\n")
            with self.assertRaisesRegex(ValueError, "embedding.identity"):
                validate_attack_profile(path)

    def test_sandbox_training_and_final_evaluation_compare_text_not_local_ids(self):
        sandbox = [("same-id", "sandbox text")]
        final = [("same-id", "final text")]
        self.proposer.source_records = {"same-id": "training text"}
        self.fixture.api.load_goals = lambda path: sandbox if path.name == "sandbox.jsonl" else final

        def disjoint(left, right):
            if {text for _, text in left} & {text for _, text in right}:
                raise ValueError("overlapping prompts")

        self.fixture.api.assert_disjoint_goals = disjoint
        self.proposer.validate_source_isolation()
        final[0] = ("different-id", "training \n text")
        with self.assertRaisesRegex(ValueError, "overlapping prompts"):
            self.proposer.validate_source_isolation()
        final[0] = ("different-id", "sandbox\ttext")
        with self.assertRaisesRegex(ValueError, "overlapping prompts"):
            self.proposer.validate_source_isolation()

    def test_missing_final_evaluation_file_fails_before_any_inference(self):
        def load(path):
            if path.name == "evaluation.jsonl":
                raise FileNotFoundError("final evaluation dataset is missing")
            return [("id", "sandbox")]

        self.fixture.api.load_goals = load
        with self.assertRaises(FileNotFoundError):
            self.proposer.validate_source_isolation()
        self.assertEqual(self.fixture.builds, [])


if __name__ == "__main__":
    unittest.main()
