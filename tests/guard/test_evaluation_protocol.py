import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from darwin_guard.cli import main
from darwin_guard.data import write_jsonl
from darwin_guard.evaluation import (
    PAPER_BENCHMARKS, check_evaluation_isolation, evaluation_input_settings, summarize,
)


class EvaluationProtocolTests(unittest.TestCase):
    def test_six_qa_average_excludes_both_over_refusal_sets(self):
        records = [dict(benchmark=name, label=0, decision=0)
                   for name in PAPER_BENCHMARKS["standard_benign"]]
        records += [dict(benchmark=name, label=0, decision=None)
                    for name in PAPER_BENCHMARKS["over_refusal"]]
        result = summarize(records)
        self.assertEqual(result["macro_benign_pass_rate"], 1)
        self.assertEqual(result["paper_protocol"]["standard_benign"]["macro"], 1)
        self.assertEqual(result["paper_protocol"]["over_refusal"]["macro"], 0)
        self.assertTrue(result["paper_protocol"]["standard_benign"]["complete"])
        self.assertTrue(result["paper_protocol"]["over_refusal"]["complete"])
        self.assertEqual(len(result["paper_protocol"]["harmful"]["missing_benchmarks"]), 9)

    def test_partial_known_and_custom_benchmarks_are_not_full_paper_results(self):
        result = summarize([
            dict(benchmark="arc_challenge", label=0, decision=0),
            dict(benchmark="custom", label=0, decision=None),
            dict(benchmark="custom-known-group", benchmark_group="standard_benign", label=0, decision=None),
        ])
        group = result["paper_protocol"]["standard_benign"]
        self.assertEqual(group["macro"], 1)
        self.assertFalse(group["complete"])
        self.assertEqual(group["present_benchmarks"], ["ARC Challenge"])
        self.assertEqual(group["additional_benchmarks"], ["custom-known-group"])
        self.assertEqual(result["paper_protocol"]["unclassified_benchmarks"], ["custom"])

    def test_jbb_subsets_and_aliases_remain_separate(self):
        result = summarize([
            dict(benchmark="JBB-Behaviors", label=1, decision=1),
            dict(benchmark="JBB-Behaviors", label=0, decision=None),
            dict(benchmark="jbb_benign", label=0, decision=0),
        ])
        self.assertEqual(result["paper_protocol"]["harmful"]["macro"], 1)
        self.assertEqual(result["paper_protocol"]["over_refusal"]["macro"], 0.5)
        self.assertIsNone(result["paper_protocol"]["standard_benign"]["macro"])
        self.assertEqual(result["benchmarks"]["JBB-Benign"]["benign_pass_rate"]["count"], 2)

    def test_conflicting_explicit_groups_fail(self):
        for name, label, group in (("PIQA", 0, "over_refusal"), ("custom", 1, "standard_benign"),
                                   ("custom", 0, "unknown")):
            with self.subTest(name=name, group=group), self.assertRaises(ValueError):
                summarize([dict(benchmark=name, label=label, decision=None, benchmark_group=group)])

    def test_paper_input_limits_cover_all_named_benchmarks_and_aliases(self):
        for group, names in PAPER_BENCHMARKS.items():
            for name in names:
                with self.subTest(benchmark=name):
                    result = evaluation_input_settings(dict(benchmark=name, label=int(group == "harmful")))
                    self.assertEqual(result["benchmark_group"], group)
                    self.assertEqual(result["max_input_tokens"], 2048 if group == "standard_benign" else 4096)
        self.assertEqual(evaluation_input_settings(dict(benchmark="JBB-Behaviors", label=0)), {
            "benchmark": "JBB-Benign", "benchmark_group": "over_refusal", "max_input_tokens": 4096,
        })

    def test_unknown_evaluation_benchmarks_require_a_group(self):
        for label in (0, 1):
            row = dict(benchmark="custom", label=label)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "set benchmark_group"):
                evaluation_input_settings(row)
            self.assertEqual(summarize([dict(row, decision=label)])["paper_protocol"]["unclassified_benchmarks"], ["custom"])
        self.assertEqual(evaluation_input_settings(dict(
            benchmark="custom", benchmark_group="standard_benign", label=0,
        ))["max_input_tokens"], 2048)

    def test_isolation_normalizes_whitespace_and_ignores_local_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            write_jsonl(path, [dict(id="1", text="same  task\ntext", label=1)])
            config = {"data": {"harmful_path": str(path)}}
            with self.assertRaisesRegex(ValueError, "overlap"):
                check_evaluation_isolation([dict(id="another", text="same task text")], config)
            result = check_evaluation_isolation([dict(id="1", text="different task")], config)
            self.assertEqual(result["status"], "partial")
            self.assertIn("data.harmful_path", result["checked_inputs"])
            self.assertIn("data.benign_path", result["unchecked_inputs"])

    def test_unset_and_missing_inputs_are_distinguished(self):
        self.assertEqual(check_evaluation_isolation([dict(text="task")], {})["status"], "not_checked")
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.jsonl")
            for config in ({"data": {"harmful_path": missing}}, {"attack": {"config_path": missing}}):
                with self.subTest(config=config), self.assertRaises(FileNotFoundError):
                    check_evaluation_isolation([dict(text="task")], config)

    def test_sandbox_isolation_uses_attack_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory) / "attack.yaml"
            settings.write_text("fixture")
            sandbox = Path(directory) / "sandbox.jsonl"
            sandbox.write_text("fixture")
            modules = {
                "darwin_attack.config": SimpleNamespace(load_config=lambda _, *, validate_target=True: SimpleNamespace(
                    sandbox=SimpleNamespace(dataset_path=sandbox))),
                "darwin_attack.datasets": SimpleNamespace(load_goals=lambda _: [("1", "sandbox   task")]),
            }
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "attack.sandbox"):
                check_evaluation_isolation([dict(text="sandbox task")], {"attack": {"config_path": str(settings)}})

    def test_cli_preserves_groups_and_records_evaluation_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input.jsonl", root / "predictions.jsonl"
            write_jsonl(inputs, [dict(benchmark="custom", benchmark_group="over_refusal", text="task", label=0)])
            config = {"training": {"max_length": 2048}, "inference": {"max_new_tokens": 128}}
            with patch("darwin_guard.cli.load_config", return_value=config), \
                    patch("darwin_guard.inference.HFGuard") as guard, contextlib.redirect_stdout(io.StringIO()):
                guard.return_value.predict.return_value = 0
                main(["evaluate", "--config", str(root / "config.yaml"), "--checkpoint", str(root / "model"),
                      "--input", str(inputs), "--output", str(output)])
            self.assertEqual(json.loads(output.read_text())["benchmark_group"], "over_refusal")
            metadata = json.loads((root / "predictions.jsonl.metadata.json").read_text())
            self.assertEqual(metadata["isolation_check"]["status"], "not_checked")
            guard.return_value.predict.assert_called_once_with("task", max_length=4096)
            self.assertEqual(metadata["input_limits_by_group"], {"over_refusal": 4096})
            self.assertEqual(metadata["benchmark_input_limits"], [{
                "benchmark": "custom", "benchmark_group": "over_refusal", "max_input_tokens": 4096,
            }])

    def test_cli_mixed_benchmarks_use_independent_limits_and_record_actual_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input.jsonl", root / "predictions.jsonl"
            rows = [
                dict(benchmark="HarmBench", text="harmful fixture", label=1),
                dict(benchmark="PIQA", text="qa fixture", label=0),
                dict(benchmark="XSTest", text="over-refusal fixture", label=0),
                dict(benchmark="JBB-Behaviors", text="jbb benign fixture", label=0),
            ]
            write_jsonl(inputs, rows)
            config = {"training": {"max_length": 1024}, "inference": {"max_new_tokens": 128}}
            seen = []

            def predict(text, *, max_length):
                seen.append((text, max_length))
                return 1 if text == "harmful fixture" else 0

            with patch("darwin_guard.cli.load_config", return_value=config), \
                    patch("darwin_guard.inference.HFGuard") as guard, contextlib.redirect_stdout(io.StringIO()):
                guard.return_value.predict.side_effect = predict
                main(["evaluate", "--config", str(root / "config.yaml"), "--checkpoint", str(root / "model"),
                      "--input", str(inputs), "--output", str(output)])
            self.assertEqual(seen, list(zip((row["text"] for row in rows), (4096, 2048, 4096, 4096))))
            self.assertEqual(config["training"]["max_length"], 1024)
            predictions = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["benchmark_group"] for row in predictions], [
                "harmful", "standard_benign", "over_refusal", "over_refusal",
            ])
            metadata = json.loads((root / "predictions.jsonl.metadata.json").read_text())
            self.assertEqual(metadata["input_limits_by_group"], {
                "harmful": 4096, "standard_benign": 2048, "over_refusal": 4096,
            })
            self.assertEqual({item["benchmark"]: item["max_input_tokens"]
                              for item in metadata["benchmark_input_limits"]}, {
                "HarmBench": 4096, "PIQA": 2048, "XSTest": 4096, "JBB-Benign": 4096,
            })
            self.assertNotIn("max_length", metadata)

    def test_cli_rejects_unknown_ungrouped_benchmarks_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input.jsonl", root / "predictions.jsonl"
            write_jsonl(inputs, [dict(benchmark="custom", text="task", label=0)])
            with patch("darwin_guard.cli.load_config", return_value={}), \
                    patch("darwin_guard.inference.HFGuard") as guard, \
                    self.assertRaisesRegex(ValueError, "set benchmark_group"):
                main(["evaluate", "--config", str(root / "config.yaml"), "--checkpoint", str(root / "model"),
                      "--input", str(inputs), "--output", str(output)])
            guard.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
