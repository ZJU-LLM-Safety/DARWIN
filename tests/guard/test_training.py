from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

from darwin_guard.prompts import BINARY_GUARD_PROMPT, encode_guard_prompt, format_guard_prompt, parse_safety
from darwin_guard.training import (
    cosine_learning_rate_scale,
    encode_verdict,
    optimizer_groups,
    paired_objective,
    train_round,
    validate_pairs,
    validate_settings,
    verdict_losses,
)


class TinyTokenizer:

    pad_token_id = 0
    eos_token_id = 3
    eos_token = "<eos>"
    pad_token = "<pad>"

    def encode(self, text, add_special_tokens=False):
        result = []
        index = 0
        before, after = BINARY_GUARD_PROMPT.split("{{QUERY}}")
        while index < len(text):
            for phrase, token in ((before, 8), (after, 9), (" Unsafe", 2), (" Safe", 1)):
                if text.startswith(phrase, index):
                    result.append(token)
                    index += len(phrase)
                    break
            else:
                result.append(ord(text[index]) + 4)
                index += 1
        return result

    def save_pretrained(self, path):
        Path(path, "tokenizer_config.json").write_text('{"test_only": true}\n')


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def test_template_is_binary_and_query_inserted_once(self):
        marker = "unique query {{QUERY}}"
        rendered = format_guard_prompt(marker)
        self.assertIn("USER: " + marker, rendered)
        self.assertNotIn("Controversial", rendered)
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n"))

    def test_safety_parser_invalid_outputs_remain_invalid(self):
        self.assertEqual(parse_safety("Safety: Safe\nCategories: None"), 0)
        self.assertEqual(parse_safety("<think>\n\n</think>\nSafety: Unsafe\nCategories: PII"), 1)
        for text in ("", "Safe", "Safety: Controversial", "Safety: Safe maybe", "reason\nSafety: Safe",
                     "Safety: Safe\nSafety: Unsafe", "Safety: Safe\nSafety: Controversial"):
            with self.subTest(text=text):
                self.assertIsNone(parse_safety(text))

    def test_only_verdict_token_is_supervised(self):
        for label, token_id in ((0, 1), (1, 2)):
            encoded = encode_verdict(self.tokenizer, "ordinary user query", label, 4096)
            self.assertEqual(encoded["labels"][-1], token_id)
            self.assertEqual(sum(value != -100 for value in encoded["labels"]), 1)
            self.assertTrue(all(value == -100 for value in encoded["labels"][:-1]))
            self.assertEqual(encoded["input_ids"][-1], token_id)
            self.assertEqual(encoded["truncated_tokens"], 0)

    def test_evaluation_uses_full_input_budget_while_training_reserves_verdict(self):
        query = "ordinary test query " * 500
        for maximum in (2048, 4096):
            with self.subTest(maximum=maximum):
                evaluation_ids = encode_guard_prompt(
                    self.tokenizer, query, maximum, reserve_verdict=False,
                )
                collection_ids = encode_guard_prompt(self.tokenizer, query, maximum)
                training = encode_verdict(self.tokenizer, query, 0, maximum)
                reserved = len(self.tokenizer.encode("Safety:")) + 1
                self.assertEqual(len(evaluation_ids), maximum)
                self.assertEqual(len(collection_ids), maximum - reserved)
                self.assertEqual(len(training["input_ids"]), maximum)
                self.assertEqual(training["input_ids"][:-reserved], collection_ids)
                self.assertEqual(evaluation_ids[0], 8)
                self.assertEqual(evaluation_ids[-1], 9)

    def test_left_truncation_matches_inference_and_keeps_verdict(self):
        maximum = 24
        query = "long query " * 20
        encoded = encode_verdict(self.tokenizer, query, 1, maximum)
        prefix_ids = self.tokenizer.encode("Safety:")
        inference_ids = encode_guard_prompt(self.tokenizer, query, maximum)
        self.assertEqual(encoded["input_ids"], inference_ids + prefix_ids + [2])
        self.assertEqual(len(encoded["input_ids"]), maximum)
        self.assertGreater(encoded["truncated_tokens"], 0)
        self.assertEqual(encoded["labels"], [-100] * (maximum - 1) + [2])
        self.assertEqual(inference_ids[0], 8)
        self.assertEqual(inference_ids[-1], 9)
        with self.assertRaisesRegex(ValueError, "complete fixed guard template"):
            encode_guard_prompt(self.tokenizer, "query", 9)

    def test_multitoken_verdict_is_rejected(self):
        tokenizer = TinyTokenizer()
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kwargs: [1, 7] if text == " Safe" else original(text, **kwargs)
        with self.assertRaisesRegex(ValueError, "single-token"):
            encode_verdict(tokenizer, "query", 0, 2048)

    def test_lambda_is_never_inferred(self):
        for settings in ({}, {"lambda_raw": None}, {"lambda_raw": float("nan")}, {"lambda_raw": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_settings(settings)
        config = validate_settings({"lambda_raw": 0.7})
        self.assertEqual(config["lambda_raw"], 0.7)
        self.assertEqual(config["learning_rate"], 5e-6)
        self.assertEqual(config["effective_batch_size"], 32)

    def test_batch_size_means_rows_and_requires_pairs(self):
        with self.assertRaisesRegex(ValueError, "even"):
            validate_settings({"lambda_raw": 1, "effective_batch_size": 31})
        with self.assertRaisesRegex(ValueError, "microbatch_pairs"):
            validate_settings({"lambda_raw": 1, "effective_batch_size": 4, "microbatch_pairs": 3})
        self.assertEqual(list(map(len, optimizer_groups(list(range(250)), 16))), [16] * 15 + [10])

    def test_only_bfloat16_or_float32_computation_is_supported(self):
        for dtype in ("bfloat16", "float32"):
            self.assertEqual(validate_settings({"lambda_raw": 1, "dtype": dtype})["dtype"], dtype)
        with self.assertRaisesRegex(ValueError, "bfloat16 or float32"):
            validate_settings({"lambda_raw": 1, "dtype": "float16"})

    def test_default_bfloat16_uses_bfloat16_parameters_and_cuda_autocast(self):
        torch = MagicMock()
        torch.float32 = sentinel.float32
        torch.bfloat16 = sentinel.bfloat16
        torch.device.return_value = SimpleNamespace(type="cuda")
        torch.cuda.is_available.return_value = True
        torch.isfinite.return_value = True
        torch.nn.utils.clip_grad_norm_.return_value = 1.0
        optimizer = torch.optim.AdamW.return_value
        optimizer.param_groups = [{"lr": 5e-6}]
        transformers = ModuleType("transformers")
        transformers.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)
        transformers.AutoTokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = self.tokenizer
        transformers.AutoModelForCausalLM = MagicMock()
        model = transformers.AutoModelForCausalLM.from_pretrained.return_value
        model.config.use_cache = True
        pairs = [{"source_id": "source", "raw_prompt": "raw", "disguised_prompt": "variant", "label": 0}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "round"
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                    patch("darwin_guard.training.collate_records", return_value={
                        "input_ids": sentinel.input_ids, "attention_mask": sentinel.mask, "labels": sentinel.labels,
                    }), patch("darwin_guard.training.verdict_losses", return_value=MagicMock()):
                train_round(pairs, "test-model", output, {"lambda_raw": 1})
            transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                "test-model", dtype=sentinel.bfloat16,
            )
            model.requires_grad_.assert_called_once_with(True)
            torch.autocast.assert_called_once_with(device_type="cuda", dtype=sentinel.bfloat16, enabled=True)
            self.assertFalse(torch.optim.AdamW.call_args.kwargs["foreach"])
            optimizer.step.assert_called_once_with()
            torch.amp.GradScaler.assert_not_called()
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(report["parameter_dtype"], "bfloat16")
            self.assertEqual(report["optimizer_state_dtype"], "bfloat16")
            self.assertEqual(report["autocast_dtype"], "bfloat16")

    def test_tail_uses_actual_pair_count_not_nominal_count(self):
        first = paired_objective(3.0, 7.0, 3, 0.5)
        second = paired_objective(5.0, 2.0, 3, 0.5)
        self.assertAlmostEqual(first + second, (8.0 + 0.5 * 9.0) / 3)
        self.assertNotAlmostEqual(first + second, (8.0 + 0.5 * 9.0) / 16)

    def test_no_duplicate_sources_or_predicted_label_substitution(self):
        pair = {"source_id": "a", "raw_prompt": "raw", "disguised_prompt": "variant", "label": 0}
        validate_pairs([pair])
        with self.assertRaisesRegex(ValueError, "Repeated source_id"):
            validate_pairs([pair, pair])
        with self.assertRaisesRegex(ValueError, "integer source label"):
            validate_pairs([{**pair, "label": "safe"}])

    def test_scheduler_matches_hugging_face_warmup_convention(self):
        scales = [cosine_learning_rate_scale(step, 16, 1) for step in range(16)]
        self.assertEqual(scales[0], 0.0)
        self.assertEqual(scales[1], 1.0)
        self.assertTrue(all(0 < scale <= 1 for scale in scales[1:]))
        self.assertLess(scales[-1], scales[1])
        self.assertEqual(cosine_learning_rate_scale(0, 1, 1), 0.0)
        self.assertEqual(cosine_learning_rate_scale(1, 100, 3), 1 / 3)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Optional torch runtime is not installed")
class RealTensorLossTests(unittest.TestCase):
    def test_explicit_float32_accumulates_small_full_parameter_updates(self):
        import torch

        class SingleParameterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                self.config = SimpleNamespace(use_cache=True)

            def forward(self, input_ids, attention_mask, use_cache):
                logits = torch.stack((self.weight * 0, self.weight, self.weight * 0))
                return SimpleNamespace(logits=logits.expand(*input_ids.shape, 3))

            def save_pretrained(self, path, **kwargs):
                Path(path, "single_parameter.json").write_text(json.dumps({"weight": self.weight.item()}))

        model = SingleParameterModel()
        transformers = ModuleType("transformers")
        transformers.__spec__ = importlib.util.spec_from_loader("transformers", loader=None)
        transformers.AutoTokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = TinyTokenizer()
        transformers.AutoModelForCausalLM = MagicMock()
        transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        pairs = [
            {"source_id": str(index), "raw_prompt": "raw", "disguised_prompt": "variant", "label": 0}
            for index in range(8)
        ]
        optimizers = []
        original_adamw = torch.optim.AdamW

        def record_optimizer(*args, **kwargs):
            optimizer = original_adamw(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "round"
            with patch.dict(sys.modules, {"transformers": transformers}), \
                    patch("torch.optim.AdamW", side_effect=record_optimizer):
                train_round(pairs, "test-model", output, {
                    "lambda_raw": 1, "device": "cpu", "dtype": "float32",
                    "effective_batch_size": 2, "microbatch_pairs": 1,
                    "max_length": 24, "gradient_checkpointing": False,
                })
            saved_weight = json.loads((output / "single_parameter.json").read_text())["weight"]
            self.assertGreater(saved_weight, 1.0 + 1e-5)
            self.assertLess(saved_weight, 1.0 + 1e-4)
            self.assertEqual(torch.tensor(saved_weight, dtype=torch.bfloat16).item(), 1.0)
            self.assertEqual(model.weight.dtype, torch.float32)
            state = optimizers[0].state[model.weight]
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(report["optimizer_steps"], 8)
            self.assertEqual(report["settings"]["learning_rate"], 5e-6)
            self.assertIsNone(report["autocast_dtype"])
            self.assertEqual(report["parameter_dtype"], "float32")
            self.assertEqual(report["optimizer_state_dtype"], "float32")

    def test_only_verdict_prediction_positions_receive_gradient(self):
        import torch

        logits = torch.zeros((2, 5, 4), dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([[-100, -100, -100, 1, -100], [-100, -100, 2, -100, -100]])
        losses = verdict_losses(logits, labels)
        self.assertTrue(torch.allclose(losses, torch.full((2,), math.log(4))))
        losses.mean().backward()
        active_positions = logits.grad.abs().sum(-1).ne(0)
        expected = torch.zeros((2, 5), dtype=torch.bool)
        expected[0, 2] = True
        expected[1, 1] = True
        self.assertTrue(torch.equal(active_positions, expected))

    def test_invalid_masks_cannot_train_categories_accidentally(self):
        import torch

        with self.assertRaisesRegex(ValueError, "exactly one"):
            verdict_losses(torch.zeros(1, 4, 5), torch.tensor([[-100, -100, 1, 2]]))

    def test_accumulated_ce_gradients_match_full_tail_batch(self):
        import torch
        import torch.nn.functional as functional

        values = torch.tensor([[0.3, -0.2], [-0.1, 0.5], [0.9, -0.7], [0.2, 0.8],
                               [-0.3, 0.4], [0.1, -0.4]], requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1, 0, 0])
        baseline_ce = functional.cross_entropy(values, targets, reduction="none")
        expected_loss = baseline_ce[::2].mean() + 0.4 * baseline_ce[1::2].mean()
        expected_gradient, = torch.autograd.grad(expected_loss, values)
        for start, stop in ((0, 4), (4, 6)):
            selected = values[start:stop]
            logits = torch.stack((selected, selected * 0), dim=1)
            labels = torch.stack((torch.full_like(targets[start:stop], -100), targets[start:stop]), dim=1)
            ce = verdict_losses(logits, labels)
            paired_objective(ce[::2].sum(), ce[1::2].sum(), 3, 0.4).backward()
        self.assertTrue(torch.allclose(values.grad, expected_gradient, atol=1e-7))

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "Optional transformers runtime is not installed")
    def test_cuda_round_trains_and_saves_bfloat16_parameters_and_moments(self):
        import random
        import torch
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM, GPT2Config

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("A CUDA device with BF16 support is required")
        original_random_state = random.getstate()
        self.addCleanup(random.setstate, original_random_state)
        optimizers = []
        original_adamw = torch.optim.AdamW

        def record_optimizer(parameters, **kwargs):
            parameters = list(parameters)
            self.assertTrue(parameters)
            self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in parameters))
            self.assertTrue(all(parameter.requires_grad for parameter in parameters))
            optimizer = original_adamw(parameters, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        with torch.random.fork_rng(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            output = root / "round"
            model = AutoModelForCausalLM.from_config(GPT2Config(
                vocab_size=256, n_positions=32, n_embd=8, n_layer=1, n_head=1,
                bos_token_id=3, eos_token_id=3, pad_token_id=0,
            ))
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            model.save_pretrained(initial)
            pairs = [
                {"source_id": str(index), "raw_prompt": "example", "disguised_prompt": "a variant", "label": index % 2}
                for index in range(3)
            ]
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                    patch("torch.optim.AdamW", side_effect=record_optimizer):
                result = train_round(pairs, str(initial), output, {
                    "lambda_raw": 0.4, "device": "cuda",
                    "effective_batch_size": 2, "microbatch_pairs": 1,
                    "max_length": 24, "gradient_checkpointing": False,
                })
            self.assertEqual(result, output)
            self.assertEqual(len(optimizers), 1)
            optimizer = optimizers[0]
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            self.assertEqual(sum(parameter.numel() for parameter in parameters), parameter_count)
            self.assertEqual(set(optimizer.state), set(parameters))
            for parameter in parameters:
                self.assertEqual(parameter.dtype, torch.bfloat16)
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertTrue(torch.isfinite(parameter.grad).all())
                state = optimizer.state[parameter]
                self.assertEqual(set(state), {"step", "exp_avg", "exp_avg_sq"})
                for name in ("exp_avg", "exp_avg_sq"):
                    self.assertEqual(state[name].dtype, torch.bfloat16)
                    self.assertEqual(state[name].shape, parameter.shape)
                    self.assertTrue(torch.isfinite(state[name]).all())
            with safe_open(output / "model.safetensors", framework="pt", device="cpu") as checkpoint:
                self.assertTrue(list(checkpoint.keys()))
                self.assertTrue(all(checkpoint.get_tensor(name).dtype == torch.bfloat16 for name in checkpoint.keys()))
            reloaded = AutoModelForCausalLM.from_pretrained(output, dtype="auto")
            self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in reloaded.parameters()))
            self.assertEqual(sum(parameter.numel() for parameter in reloaded.parameters()), parameter_count)
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(report["optimizer_steps"], 3)
            self.assertEqual(report["settings"]["learning_rate"], 5e-6)
            self.assertEqual(report["parameter_dtype"], "bfloat16")
            self.assertEqual(report["optimizer_state_dtype"], "bfloat16")
            self.assertEqual(report["autocast_dtype"], "bfloat16")
            self.assertFalse((output / "optimizer.pt").exists())

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "Optional transformers runtime is not installed")
    def test_cpu_round_exports_reloadable_full_model_and_consumes_tail(self):
        import torch
        from transformers import AutoModelForCausalLM, GPT2Config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            output = root / "round"
            model = AutoModelForCausalLM.from_config(GPT2Config(
                vocab_size=256, n_positions=32, n_embd=8, n_layer=1, n_head=1,
                bos_token_id=3, eos_token_id=3, pad_token_id=0,
            ))
            model.save_pretrained(initial)
            initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
            pairs = [
                {"source_id": str(index), "raw_prompt": "example", "disguised_prompt": "a variant", "label": index % 2}
                for index in range(3)
            ]
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()):
                result = train_round(pairs, str(initial), output, {
                    "lambda_raw": 0.4, "device": "cpu", "dtype": "float32",
                    "effective_batch_size": 4, "microbatch_pairs": 1,
                    "max_length": 24, "gradient_checkpointing": False,
                })
            self.assertEqual(result, output)
            reloaded = AutoModelForCausalLM.from_pretrained(output, dtype="auto")
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in reloaded.parameters()))
            self.assertTrue(all(parameter.requires_grad for parameter in reloaded.parameters()))
            self.assertTrue(any(not torch.equal(before, after) for before, after in zip(initial_parameters, reloaded.parameters())))
            report = json.loads((output / "training_report.json").read_text())
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(report["parameter_dtype"], "float32")
            self.assertEqual(report["optimizer_state_dtype"], "float32")
            self.assertIsNone(report["autocast_dtype"])
            self.assertEqual([update["pair_count"] for update in report["updates"]], [2, 1])
            self.assertTrue(report["optimizer_reset_each_round"])
            self.assertTrue((output / "model.safetensors").is_file())
            self.assertFalse((output / "optimizer.pt").exists())


if __name__ == "__main__":
    unittest.main()
