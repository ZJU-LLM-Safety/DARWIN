import json
from pathlib import Path
import tempfile
import unittest

from darwin_guard.data import deduplicate_sources, load_sources, normalize_source, select_sources, write_jsonl


def rows(label, count):
    return [{"id": f"{label}-{index}", "text": f"placeholder task {label} {index}", "label": label}
            for index in range(count)]


class DataTests(unittest.TestCase):
    def test_stable_content_id_and_label_validation(self):
        first = normalize_source({"text": " a neutral placeholder ", "label": "Safe"})
        second = normalize_source({"text": "a neutral placeholder", "label": 0})
        self.assertEqual(first, second)
        self.assertTrue(first["id"].startswith("sha256:"))
        for label in (None, True, 2, "controversial"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                normalize_source({"text": "placeholder", "label": label})

    def test_dedup_and_conflicts(self):
        self.assertEqual(len(deduplicate_sources([
            {"id": "a", "text": "same task", "label": 0},
            {"id": "b", "text": "same task", "label": 0},
        ])), 1)
        with self.assertRaisesRegex(ValueError, "conflicting ground-truth"):
            deduplicate_sources([{"id": "a", "text": "same task", "label": 0},
                                 {"id": "b", "text": "same task", "label": 1}])
        with self.assertRaisesRegex(ValueError, "different texts or labels"):
            deduplicate_sources([{"id": "a", "text": "first task", "label": 0},
                                 {"id": "a", "text": "second task", "label": 0}])

    def test_wildjailbreak_only_vanilla_harmful_train(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            write_jsonl(path, [
                {"id": "keep", "vanilla": "placeholder task A", "data_type": "vanilla_harmful", "split": "train"},
                {"id": "test", "vanilla": "placeholder task B", "data_type": "vanilla_harmful", "split": "test"},
                {"id": "variant", "vanilla": "placeholder task C", "data_type": "adversarial_harmful", "split": "train"},
                {"id": "benign", "vanilla": "placeholder task D", "data_type": "vanilla_benign", "split": "train"},
            ])
            self.assertEqual(load_sources(path, source_kind="wildjailbreak"), [
                {"id": "keep", "text": "placeholder task A", "label": 1}
            ])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            normalize_source({"data_type": "vanilla_harmful", "adversarial": "wrong field"}, source_kind="wildjailbreak")

    def test_orbench_benign_route_and_explicit_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps([{"prompt": "neutral placeholder", "prompt_id": 7}]), encoding="utf-8")
            self.assertEqual(load_sources(path, source_kind="orbench"), [{"id": "7", "text": "neutral placeholder", "label": 0}])
        for extra in ({"label": 1}, {"or_bench_config": "or-bench-toxic"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                normalize_source({"prompt": "neutral placeholder", **extra}, source_kind="orbench")

    def test_tsv_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.tsv"
            path.write_text("id\tdata_type\tvanilla\nsource\tvanilla_harmful\tneutral fixture\n", encoding="utf-8")
            self.assertEqual(load_sources(path, source_kind="wildjailbreak")[0]["label"], 1)

    def test_exact_quotas_determinism_and_no_replacement(self):
        harmful, benign = rows(1, 8), rows(0, 8)
        selected = select_sources(harmful, benign, total=6, harmful_fraction=0.5, seed=17, exclude_ids=["1-0"])
        reordered = select_sources(reversed(harmful), reversed(benign), total=6, harmful_fraction=0.5, seed=17, exclude_ids=["1-0"])
        self.assertEqual(selected, reordered)
        self.assertEqual(sum(row["label"] for row in selected), 3)
        self.assertEqual(len({row["id"] for row in selected}), 6)
        self.assertNotIn("1-0", {row["id"] for row in selected})

    def test_invalid_quotas(self):
        for total, fraction in ((3, 0.5), (6, -0.1), (0, 0.5), (6, float("nan"))):
            with self.subTest(total=total, fraction=fraction), self.assertRaises(ValueError):
                select_sources(rows(1, 5), rows(0, 5), total=total, harmful_fraction=fraction, seed=1)

    def test_missing_quota_is_not_filled_from_other_route(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_sources(rows(1, 1), rows(0, 10), total=6, harmful_fraction=0.5, seed=1)

    def test_declared_route_cannot_override_labels(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            select_sources(rows(0, 2), rows(0, 2), total=2, harmful_fraction=0.5, seed=1)

    def test_jsonl_roundtrip_and_invalid_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sources.jsonl"
            write_jsonl(path, rows(0, 2))
            self.assertEqual(load_sources(path), rows(0, 2))
            path.write_text('{"text":\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_sources(path)


if __name__ == "__main__":
    unittest.main()
