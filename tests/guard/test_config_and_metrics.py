from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from darwin_guard.config import (
    ConfigurationError, PAPER_SETTINGS, get, load_config, missing_settings,
    pairs_per_round, put, validate, fingerprint,
)
from darwin_guard.evaluation import summarize


def complete_config():
    result = {}
    for key, value in PAPER_SETTINGS.items():
        put(result, key, value)
    values = {
        "runtime.output_dir": "runs/example", "runtime.seed": 42,
        "models.guard.path": "models/guard", "models.generator.provider": "transformers",
        "models.generator.model": "models/generator", "models.generator.temperature": 0.7,
        "models.generator.max_tokens": 384, "models.filter.provider": "transformers",
        "models.filter.model": "models/filter", "models.filter.temperature": 0.0,
        "models.filter.max_tokens": 192, "data.harmful_path": "data/harmful.jsonl",
        "data.benign_path": "data/benign.jsonl", "attack.config_path": "attack.yaml",
        "attack.initial_database": "seeds.sqlite", "attack.genetic_candidates": 2,
        "attack.reflection_candidates": 1, "attack.max_evolution_batches": 100,
        "online.examples_unit": "rows",
        "online.harmful_fraction": 0.1, "training.lambda_raw": 1.0,
        "training.max_length": 2048, "training.dtype": "bfloat16", "training.device": "cpu",
        "training.microbatch_pairs": 1, "training.weight_decay": 0.0,
        "training.gradient_checkpointing": True, "inference.max_new_tokens": 128,
    }
    for key, value in values.items():
        put(result, key, value)
    return result


class ConfigTests(unittest.TestCase):
    def test_complete_config_preserves_training_defaults(self):
        config = complete_config()
        self.assertEqual(validate(config), [])
        self.assertEqual(get(config, "training.lambda_raw"), 1.0)
        self.assertEqual(get(config, "online.examples_unit"), "rows")
        self.assertEqual(get(config, "online.examples_per_round"), 500)
        self.assertEqual(get(config, "online.harmful_fraction"), 0.1)
        self.assertEqual(get(config, "attack.genetic_candidates"), 2)
        self.assertEqual(get(config, "attack.reflection_candidates"), 1)
        self.assertEqual(pairs_per_round(config), 250)
        self.assertEqual(round(pairs_per_round(config) * get(config, "online.harmful_fraction")), 25)

    def test_table_7_raw_anchor_and_unsafe_safe_ratio_are_fixed(self):
        for key, value in (("training.lambda_raw", 0.25), ("online.harmful_fraction", 0.5)):
            for allow_unset in (False, True):
                config = complete_config()
                put(config, key, value)
                with self.subTest(setting=key, allow_unset=allow_unset), \
                        self.assertRaisesRegex(ConfigurationError, "Paper profile requires"):
                    validate(config, allow_unset=allow_unset)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "YAML parsing requires the package's PyYAML dependency")
    def test_distributed_template_defaults_and_unset_runtime_inputs(self):
        path = Path(__file__).resolve().parents[2] / "configs" / "darwin_guard.example.yaml"
        config = load_config(path, allow_unset=True)
        self.assertEqual(get(config, "training.lambda_raw"), 1.0)
        self.assertEqual(get(config, "online.examples_unit"), "rows")
        self.assertEqual(get(config, "online.examples_per_round"), 500)
        self.assertEqual(get(config, "online.harmful_fraction"), 0.1)
        self.assertEqual(get(config, "attack.genetic_candidates"), 2)
        self.assertEqual(get(config, "attack.reflection_candidates"), 1)
        missing = missing_settings(config)
        self.assertIn("models.guard.path", missing)
        self.assertIn("models.generator.model", missing)
        self.assertIn("attack.initial_database", missing)
        self.assertEqual(set(missing), {
            "runtime.output_dir", "models.guard.path", "models.generator.model",
            "models.filter.model", "data.harmful_path", "data.benign_path",
            "attack.config_path", "attack.initial_database",
        })
        with self.assertRaises(ConfigurationError):
            load_config(path)

    def test_unset_inputs_are_reported_but_never_approve_training(self):
        config = complete_config()
        put(config, "models.guard.path", None)
        put(config, "attack.genetic_candidates", None)
        self.assertEqual(set(validate(config, allow_unset=True)), {"models.guard.path", "attack.genetic_candidates"})
        with self.assertRaisesRegex(ConfigurationError, "required configuration values"):
            validate(config)

    def test_every_fixed_paper_setting_is_enforced_even_in_template_audit(self):
        for key, value in PAPER_SETTINGS.items():
            with self.subTest(setting=key):
                config = complete_config()
                put(config, key, value + 1 if isinstance(value, (int, float)) else "different")
                with self.assertRaisesRegex(ConfigurationError, "Paper profile"):
                    validate(config, allow_unset=True)

    def test_fixed_integer_paper_settings_reject_float_lookalikes(self):
        for key, value in PAPER_SETTINGS.items():
            if type(value) is int:
                with self.subTest(setting=key):
                    config = complete_config()
                    put(config, key, float(value))
                    with self.assertRaises(ConfigurationError):
                        validate(config)

    def test_invalid_runtime_parameters(self):
        cases = (
            ("runtime.seed", -1), ("runtime.seed", 1.0),
            ("training.lambda_raw", 0), ("training.lambda_raw", -1),
            ("training.lambda_raw", float("nan")), ("training.dtype", "float16"),
            ("training.microbatch_pairs", 17), ("training.microbatch_pairs", True),
            ("training.gradient_checkpointing", "true"), ("training.max_length", 0),
            ("training.max_length", 1),
            ("online.harmful_fraction", 0), ("online.harmful_fraction", 1),
            ("online.harmful_fraction", 0.333), ("online.examples_unit", "unknown"),
            ("online.examples_unit", "pairs"),
            ("models.generator.max_tokens", 0), ("models.generator.temperature", -0.1),
            ("models.generator.provider", "unknown"), ("models.filter.max_tokens", 2.5),
            ("models.filter.temperature", float("inf")), ("inference.max_new_tokens", False),
        )
        for key, value in cases:
            with self.subTest(setting=key, value=value):
                config = complete_config()
                put(config, key, value)
                with self.assertRaises(ConfigurationError):
                    validate(config)

    def test_each_evolution_channel_remains_enabled(self):
        for channel in ("genetic_candidates", "reflection_candidates"):
            config = complete_config()
            put(config, f"attack.{channel}", 0)
            put(config, "attack.external_material", "materials.jsonl")
            with self.subTest(channel=channel), self.assertRaises(ConfigurationError):
                validate(config)

    def test_growth_budget_and_optional_recorded_schedule(self):
        config = complete_config()
        put(config, "attack.round_pool_targets", [50] * 5 + [100] * 5 + [150] * 5 + [200] * 5)
        self.assertEqual(validate(config), [])
        for targets in ([200] * 19, [50] * 20, [200, 50] + [200] * 18, [200.0] * 20):
            put(config, "attack.round_pool_targets", targets)
            with self.subTest(targets=targets), self.assertRaises(ConfigurationError):
                validate(config)
        put(config, "attack.round_pool_targets", None)
        for budget in (0, -1, True, 1.5):
            put(config, "attack.max_evolution_batches", budget)
            with self.subTest(budget=budget), self.assertRaises(ConfigurationError):
                validate(config)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "YAML parsing requires PyYAML")
    def test_fingerprint_freezes_attack_dataset_and_operator_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "attack.yaml").write_text(
                "sandbox:\n  dataset_path: sandbox.jsonl\n"
                "attack:\n  dataset_path: evaluation.jsonl\n"
                "pool:\n  mutation_operators_file: operators.jsonl\n"
            )
            config = {"attack": {"config_path": str(root / "attack.yaml")}}
            for name in ("sandbox.jsonl", "evaluation.jsonl", "operators.jsonl"):
                (root / name).write_text("original input\n")
            original = fingerprint(config)
            for name in ("sandbox.jsonl", "evaluation.jsonl", "operators.jsonl"):
                with self.subTest(name=name):
                    (root / name).write_text("changed input\n")
                    self.assertNotEqual(original, fingerprint(config))
                    (root / name).write_text("original input\n")
                    self.assertEqual(original, fingerprint(config))
    def test_pair_count_cannot_silently_double_the_round_budget(self):
        config = complete_config()
        put(config, "online.examples_unit", "pairs")
        with self.assertRaises(ConfigurationError):
            pairs_per_round(config)
        put(config, "online.examples_unit", "rows")
        put(config, "online.examples_per_round", 501)
        with self.assertRaises(ConfigurationError):
            pairs_per_round(config)

    def test_api_model_requires_environment_variable_and_rejects_literal_credentials(self):
        for role in ("generator", "filter"):
            config = complete_config()
            put(config, f"models.{role}.provider", "openai_compatible")
            with self.subTest(role=role), self.assertRaises(ConfigurationError):
                validate(config)
            put(config, f"models.{role}.api_key_env", "MODEL_KEY")
            self.assertEqual(validate(config), [])
            for field in ("api_key", "base_url", "endpoint"):
                invalid = copy.deepcopy(config)
                put(invalid, f"models.{role}.{field}", "test-placeholder")
                with self.subTest(role=role, field=field), self.assertRaises(ConfigurationError):
                    validate(invalid)


class MetricTests(unittest.TestCase):
    def test_invalid_decisions_remain_in_each_denominator(self):
        summary = summarize([
            {"benchmark": "harm", "label": 1, "decision": 1},
            {"benchmark": "harm", "label": 1, "decision": 0},
            {"benchmark": "harm", "label": 1, "decision": None},
            {"benchmark": "benign", "label": 0, "decision": 0},
            {"benchmark": "benign", "label": 0, "decision": None},
        ])
        harmful = summary["benchmarks"]["harm"]["unsafe_recall"]
        self.assertEqual(harmful, {"value": 1 / 3, "count": 3, "invalid": 1})
        self.assertEqual(summary["benchmarks"]["benign"]["benign_pass_rate"],
                         {"value": 0.5, "count": 2, "invalid": 1})

    def test_macro_average_is_equal_weight_per_benchmark_not_micro(self):
        records = [{"benchmark": "large", "label": 1, "decision": 1}] * 9
        records += [{"benchmark": "small", "label": 1, "decision": None}]
        records += [{"benchmark": "benign-a", "label": 0, "decision": 0}]
        records += [{"benchmark": "benign-b", "label": 0, "decision": None}] * 3
        summary = summarize(records)
        self.assertEqual(summary["macro_unsafe_recall"], 0.5)
        self.assertEqual(summary["macro_benign_pass_rate"], 0.5)

    def test_mixed_benchmark_keeps_benign_and_harmful_denominators_separate(self):
        summary = summarize([
            {"benchmark": "mixed", "label": 1, "decision": 1},
            {"benchmark": "mixed", "label": 0, "decision": None},
        ])
        self.assertEqual(summary["macro_unsafe_recall"], 1)
        self.assertEqual(summary["macro_benign_pass_rate"], 0)

    def test_empty_metrics_are_unavailable_not_zero(self):
        result = summarize([])
        self.assertEqual(result["benchmarks"], {})
        self.assertIsNone(result["macro_unsafe_recall"])
        self.assertIsNone(result["macro_benign_pass_rate"])
        for group in ("harmful", "standard_benign", "over_refusal"):
            self.assertIsNone(result["paper_protocol"][group]["macro"])
            self.assertFalse(result["paper_protocol"][group]["complete"])

    def test_invalid_labels_decisions_and_names_are_rejected(self):
        for field, value in (("label", True), ("label", "1"), ("label", None),
                             ("decision", "unsafe"), ("decision", False), ("decision", 2),
                             ("benchmark", ""), ("benchmark", None)):
            record = {"benchmark": "fixture", "label": 1, "decision": 1, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize([record])


if __name__ == "__main__":
    unittest.main()
