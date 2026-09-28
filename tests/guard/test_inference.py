import importlib.util
import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch, sentinel

from darwin_guard.inference import build_filter_model, GemmaIntentModel, HFGuard


class GuardInputBudgetTests(unittest.TestCase):
    def test_evaluation_override_does_not_change_online_collection_default(self):
        guard = object.__new__(HFGuard)
        guard.max_length = 1024
        guard.max_new_tokens = 128
        guard.device = "cpu"
        guard.torch = MagicMock()
        guard.model = MagicMock()
        guard.tokenizer = MagicMock()
        guard.tokenizer.decode.return_value = "Safety: Safe"
        with patch("darwin_guard.prompts.encode_guard_prompt", return_value=[1, 2, 3]) as encode:
            self.assertEqual(guard.predict("evaluation input", max_length=4096), 0)
            encode.assert_called_with(guard.tokenizer, "evaluation input", 4096, reserve_verdict=False)
            self.assertEqual(guard.predict("collection input"), 0)
            encode.assert_called_with(guard.tokenizer, "collection input", 1024, reserve_verdict=True)
            self.assertEqual(guard.max_length, 1024)
            for invalid in (True, 0, 1, 2048.0):
                with self.subTest(limit=invalid), self.assertRaisesRegex(ValueError, "Guard input length"):
                    guard.predict("input", max_length=invalid)


class GemmaProviderTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.bfloat16 = sentinel.bfloat16
        self.transformers = ModuleType("transformers")
        self.transformers.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)
        self.transformers.AutoProcessor = MagicMock()
        self.transformers.AutoModelForImageTextToText = MagicMock()
        self.processor = self.transformers.AutoProcessor.from_pretrained.return_value
        self.model = self.transformers.AutoModelForImageTextToText.from_pretrained.return_value
        self.inputs = MagicMock()
        self.inputs.__getitem__.return_value.shape = [1, 12]
        self.inputs.keys.return_value = ["input_ids"]
        self.processor.apply_chat_template.return_value.to.return_value = self.inputs
        self.processor.decode.return_value = '  {"preserves_intent": true}  '
        self.settings = {
            "provider": "transformers", "identity": "Gemma-4-31B-it",
            "model": "test-model", "temperature": 0, "max_tokens": 256,
        }

    def test_conditional_generation_interface_and_completion_only(self):
        with patch.dict(sys.modules, {"torch": self.torch, "transformers": self.transformers}):
            provider = build_filter_model(self.settings, "bfloat16")
            answer = provider.generate([{"role": "user", "content": "audit this pair"}])
        self.assertIsInstance(provider, GemmaIntentModel)
        self.transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(
            "test-model", dtype=sentinel.bfloat16, device_map="auto",
        )
        self.processor.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": [{"type": "text", "text": "audit this pair"}]}],
            tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        )
        self.processor.apply_chat_template.return_value.to.assert_called_once_with(self.model.device)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 256)
        self.assertFalse(self.model.generate.call_args.kwargs["do_sample"])
        self.assertNotIn("temperature", self.model.generate.call_args.kwargs)
        self.model.generate.return_value.__getitem__.assert_called_once_with((0, slice(12, None)))
        self.assertEqual(answer, '{"preserves_intent": true}')

    def test_sampling_and_model_load_overrides(self):
        settings = {**self.settings, "temperature": 0.4, "device": "cpu",
                    "extra": {"model_kwargs": {"dtype": "bfloat16", "low_cpu_mem_usage": True}}}
        with patch.dict(sys.modules, {"torch": self.torch, "transformers": self.transformers}):
            provider = build_filter_model(settings, "float32")
            provider.generate([{"role": "user", "content": "audit"}])
        self.transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once_with(
            "test-model", dtype=sentinel.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
        )
        self.assertTrue(self.model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(self.model.generate.call_args.kwargs["temperature"], 0.4)


if __name__ == "__main__":
    unittest.main()
