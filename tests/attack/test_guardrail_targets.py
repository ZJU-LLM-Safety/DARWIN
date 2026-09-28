from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from darwin_attack import cli
from darwin_attack.config import ConfigurationError, ModelConfig, load_config
from darwin_attack.guardrail_templates import (
    QWEN_TEMPLATE, YUFENG_TEMPLATE, YUFENG_POLICY_IDS, parse_builtin_guard_output,
)
from darwin_attack.providers import OpenAICompatibleChatModel, TransformersChatModel
from darwin_attack.targets import GuardrailTarget
from darwin_guard.prompts import format_guard_prompt
from test_config_and_extraction import CONFIG


class RecordingModel:
    def __init__(self, output="Safety: Safe\nCategories: None"):
        self.output = output
        self.prompts = []
        self.chats = []

    def generate_prompt(self, prompt):
        self.prompts.append(prompt)
        return self.output

    def generate(self, messages):
        self.chats.append(messages)
        return self.output


class YuFengTokenizer:
    init_kwargs = {"id2risk": {key: key for key in YUFENG_POLICY_IDS | {"sec"}}}

    def __init__(self):
        self.calls = []

    def get_chat_template(self):
        return "model-specific policy; reason_first; {{ messages }}"

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "official-template[" + messages[0]["content"] + "]assistant-prefix"


def guard_config(template=QWEN_TEMPLATE, provider="transformers", identity="Qwen3Guard-Gen-8B"):
    raw = yaml.safe_load(CONFIG)
    raw["models"]["target"] = {
        "provider": provider, "model": "deployment-alias", "identity": identity,
        "temperature": 0, "max_tokens": 128,
    }
    raw["attack"].update(target_kind="guardrail", guardrail_template=template)
    return raw


class GuardrailTargetTests(unittest.TestCase):
    def test_qwen_uses_shared_binary_template_once_and_preserves_query(self):
        model = RecordingModel()
        target = GuardrailTarget(model, template=QWEN_TEMPLATE)
        query = "A neutral query containing literal {{QUERY}} and {input}."
        result = target.query(query)
        self.assertEqual(model.prompts, [format_guard_prompt(query)])
        self.assertEqual(model.chats, [])
        self.assertEqual(model.prompts[0].count("<|im_start|>user"), 1)
        self.assertTrue(model.prompts[0].endswith("<think>\n\n</think>\n"))
        self.assertEqual(result.decision, "safe")
        self.assertEqual(result.raw["guardrail"]["template_id"], QWEN_TEMPLATE)

    def test_qwen_parser_does_not_accept_explanation_or_conflicting_labels(self):
        for output in ("I think this is safe", "Safety: Controversial", "Safety: Safe\nSafety: Unsafe"):
            with self.subTest(output=output):
                result = GuardrailTarget(RecordingModel(output), template=QWEN_TEMPLATE).query("query")
                self.assertEqual(result.decision, "unparseable")

    def test_yufeng_uses_native_policy_once_then_raw_completion(self):
        model = RecordingModel("mc\n<explanation>Details</explanation>")
        model.tokenizer = YuFengTokenizer()
        target = GuardrailTarget(model, template=YUFENG_TEMPLATE)
        result = target.query("neutral fixture")
        self.assertEqual(model.tokenizer.calls, [(
            [{"role": "user", "content": "neutral fixture"}],
            {"policy": None, "reason_first": False, "tokenize": False},
        )])
        self.assertEqual(model.prompts, ["official-template[neutral fixture]assistant-prefix"])
        self.assertEqual(model.chats, [])
        self.assertEqual(result.decision, "unsafe")
        self.assertEqual(result.raw["guardrail"]["policy"], "model_default")

    def test_yufeng_requires_official_taxonomy_and_strict_first_risk_id(self):
        model = RecordingModel("sec")
        model.tokenizer = SimpleNamespace(init_kwargs={})
        with self.assertRaisesRegex(ValueError, "official tokenizer"):
            GuardrailTarget(model, template=YUFENG_TEMPLATE)
        self.assertEqual(parse_builtin_guard_output("sec\n<explanation>mc</explanation>", YUFENG_TEMPLATE), "safe")
        for risk_id in YUFENG_POLICY_IDS:
            self.assertEqual(parse_builtin_guard_output(risk_id, YUFENG_TEMPLATE), "unsafe")
        for output in ("unknown", "sec mc", "explanation first\nsec", ""):
            self.assertEqual(parse_builtin_guard_output(output, YUFENG_TEMPLATE), "unparseable")

    def test_custom_template_replaces_only_one_slot_and_rejects_ambiguous_output(self):
        model = RecordingModel("YES NO")
        target = GuardrailTarget(
            model, r"\bNO\b", r"\bYES\b", template="custom",
            template_text="Classify {{QUERY}}\nVerdict:", template_id="custom-criterion-v1",
        )
        result = target.query("query {unrelated}")
        self.assertEqual(model.prompts, ["Classify query {unrelated}\nVerdict:"])
        self.assertEqual(result.decision, "unparseable")

    def test_endpoint_mode_is_explicit_and_never_locally_wraps(self):
        model = RecordingModel()
        target = GuardrailTarget(model, template="prewrapped_endpoint", template_id=QWEN_TEMPLATE)
        self.assertEqual(target.query("raw neutral query").decision, "safe")
        self.assertEqual(model.chats, [[{"role": "user", "content": "raw neutral query"}]])
        self.assertEqual(model.prompts, [])
        self.assertFalse(target.metadata["template_verified_locally"])
        with self.assertRaisesRegex(ValueError, "explicit"):
            GuardrailTarget(model, template="bare")
        with self.assertRaisesRegex(ValueError, "template_id"):
            GuardrailTarget(model, template="prewrapped_endpoint")

    def test_local_raw_prompt_bypasses_chat_template_and_special_token_insertion(self):
        provider = TransformersChatModel.__new__(TransformersChatModel)
        provider.config = ModelConfig("transformers", "fixture", 0.0, 16)
        provider.torch = SimpleNamespace(inference_mode=nullcontext)
        tensor = Mock(shape=(1, 7))
        tensor.to.return_value = tensor
        provider.tokenizer = Mock(return_value={"input_ids": tensor})
        provider.tokenizer.eos_token_id = 99
        provider.tokenizer.decode.return_value = "Safety: Safe"
        generated = Mock()
        generated.__getitem__ = Mock(return_value=[11, 12])
        provider.model = Mock()
        provider.model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
        provider.model.generate.return_value = generated
        target = GuardrailTarget(provider, template=QWEN_TEMPLATE)
        self.assertEqual(target.query("neutral").decision, "safe")
        provider.tokenizer.apply_chat_template.assert_not_called()
        provider.tokenizer.assert_called_once_with(
            format_guard_prompt("neutral"), return_tensors="pt", add_special_tokens=False,
        )
        self.assertFalse(provider.model.generate.call_args.kwargs["do_sample"])
        generated.__getitem__.assert_called_once_with((0, slice(7, None, None)))

    def test_remote_raw_prompt_calls_completions_and_not_chat_completions(self):
        provider = OpenAICompatibleChatModel.__new__(OpenAICompatibleChatModel)
        provider.config = ModelConfig("openai_compatible", "service-alias", 0.0, 16)
        provider.client = Mock()
        provider.client.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(text="Safety: Safe")],
        )
        target = GuardrailTarget(provider, template=QWEN_TEMPLATE)
        self.assertEqual(target.query("neutral").decision, "safe")
        provider.client.completions.create.assert_called_once_with(
            model="service-alias", prompt=format_guard_prompt("neutral"), temperature=0.0, max_tokens=16,
        )
        provider.client.chat.completions.create.assert_not_called()


class GuardrailConfigurationTests(unittest.TestCase):
    def load(self, raw, root):
        path = root / "run.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path)

    def test_explicit_builtin_model_contracts_and_remote_endpoint_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for template, identity in ((QWEN_TEMPLATE, "Qwen3Guard-Gen-8B"), (YUFENG_TEMPLATE, "YuFeng-XGuard-Reason-8B")):
                config = self.load(guard_config(template, identity=identity), root)
                self.assertEqual(config.attack.guardrail_template, template)
            raw = guard_config("prewrapped_endpoint", "openai_compatible")
            raw["attack"]["guardrail_template_id"] = QWEN_TEMPLATE
            self.assertEqual(self.load(raw, root).attack.guardrail_template_id, QWEN_TEMPLATE)

    def test_missing_mismatched_or_ambiguous_guard_contract_fails_before_loading_model(self):
        variants = []
        for template in (None, "unknown", "prewrapped_endpoint"):
            variants.append(guard_config(template))
        raw = guard_config()
        raw["models"]["target"]["identity"] = "YuFeng-XGuard-Reason-8B"
        variants.append(raw)
        raw = guard_config()
        raw["models"]["target"]["temperature"] = 0.7
        variants.append(raw)
        raw = guard_config()
        raw["attack"]["guardrail_safe_pattern"] = "safe"
        variants.append(raw)
        variants.append(guard_config(YUFENG_TEMPLATE, "openai_compatible", "YuFeng-XGuard-Reason-8B"))
        with tempfile.TemporaryDirectory() as directory:
            for raw in variants:
                with self.subTest(raw=raw["attack"]), self.assertRaises(ConfigurationError):
                    self.load(raw, Path(directory))

    def test_custom_requires_existing_complete_template_and_output_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = guard_config("custom", identity="custom-guard")
            raw["attack"].update(
                guardrail_template_file="template.txt", guardrail_template_id="criterion-v1",
                guardrail_safe_pattern=r"^NO$", guardrail_unsafe_pattern=r"^YES$",
            )
            with self.assertRaises(ConfigurationError):
                self.load(raw, root)
            for text in ("{{QUERY}}", "instructions without slot", "{{QUERY}} and {{QUERY}}"):
                (root / "template.txt").write_text(text)
                with self.assertRaises(ConfigurationError):
                    self.load(raw, root)
            (root / "template.txt").write_text("Classify {{QUERY}}\nAnswer:")
            self.assertEqual(self.load(raw, root).attack.guardrail_template_file, (root / "template.txt").resolve())

    def test_attack_cli_writes_template_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.load(guard_config(), root)
            output = root / "results.jsonl"
            engine = Mock()
            engine.attack.return_value = SimpleNamespace(to_dict=lambda: {"instance_id": "a"})
            with patch.multiple(
                cli, load_config=Mock(return_value=config), Repository=Mock(),
                SentenceTransformerEmbedder=Mock(), StrategyPool=Mock(),
                FeedbackGuidedEvolution=Mock(), build_model=Mock(return_value=RecordingModel()),
                AttackEngine=Mock(return_value=engine), load_goals=Mock(return_value=[("a", "neutral")]),
                assert_disjoint_goals=Mock(), summarize=Mock(return_value={}),
            ):
                cli.cmd_attack(SimpleNamespace(config="fixture", output=output, limit=None))
            metadata = json.loads(Path(str(output) + ".metadata.json").read_text())
            self.assertEqual(metadata["guardrail"]["template_id"], QWEN_TEMPLATE)
            self.assertEqual(metadata["guardrail"]["transport"], "raw_prompt_completion")
            self.assertEqual(metadata["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
