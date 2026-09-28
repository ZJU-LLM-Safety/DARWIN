from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

from .prompts import encode_guard_prompt, format_guard_prompt


DEFAULTS = {
    "learning_rate": 5e-6,
    "epochs": 1,
    "effective_batch_size": 32,
    "max_length": 2048,
    "dtype": "bfloat16",
    "device": "cuda:0",
    "seed": 42,
    "microbatch_pairs": 1,
    "weight_decay": 0.0,
    "gradient_checkpointing": True,
}


def validate_settings(settings: dict) -> dict:
    result = {**DEFAULTS, **settings}
    if "lambda_raw" not in settings or settings["lambda_raw"] is None:
        raise ValueError("lambda_raw must be set explicitly (Table 7 uses 1.0)")
    for name in ("lambda_raw", "learning_rate", "weight_decay"):
        value = result[name]
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"Invalid {name}: {value}")
        result[name] = value
    for name in ("epochs", "effective_batch_size", "max_length", "microbatch_pairs", "seed"):
        value = result[name]
        try:
            integral = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if isinstance(value, bool) or integral != value or (name != "seed" and integral <= 0):
            raise ValueError(f"Invalid {name}: {value}")
        result[name] = integral
    if result["effective_batch_size"] % 2:
        raise ValueError("effective_batch_size counts rows and must be even (two rows per source pair)")
    if result["max_length"] < 2:
        raise ValueError("max_length must leave room for a prefix and a verdict token")
    if result["microbatch_pairs"] > result["effective_batch_size"] // 2:
        raise ValueError("microbatch_pairs cannot exceed the optimizer step's pair budget")
    if result["dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("dtype must be bfloat16 or float32")
    if not isinstance(result["device"], str) or not result["device"]:
        raise ValueError("device must be a nonempty PyTorch device string")
    if not isinstance(result["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be a boolean")
    return result


def validate_pairs(pairs: list[dict]) -> None:
    if not pairs:
        raise ValueError("A training round must contain at least one retained source pair")
    identities: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Pair {index} must be a JSON object")
        source_id = pair.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Pair {index} has no source_id")
        if source_id in identities:
            raise ValueError(f"Repeated source_id within a training round: {source_id}")
        identities.add(source_id)
        if type(pair.get("label")) is not int or pair["label"] not in (0, 1):
            raise ValueError(f"Pair {index} must have an integer source label 0 or 1")
        for name in ("raw_prompt", "disguised_prompt"):
            if not isinstance(pair.get(name), str) or not pair[name].strip():
                raise ValueError(f"Pair {index} has an empty {name}")


def encode_verdict(tokenizer: Any, text: str, label: int, max_length: int) -> dict:
    if type(label) is not int or label not in (0, 1):
        raise ValueError("label must be the integer 0 or 1")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    prompt = format_guard_prompt(text)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    prefix_ids = list(tokenizer.encode("Safety:", add_special_tokens=False))
    safe_ids = list(tokenizer.encode(" Safe", add_special_tokens=False))
    unsafe_ids = list(tokenizer.encode(" Unsafe", add_special_tokens=False))
    if len(safe_ids) != 1 or len(unsafe_ids) != 1 or safe_ids == unsafe_ids:
        raise ValueError("This trainer requires distinct single-token ' Safe' and ' Unsafe' verdicts")
    verdict_text = " Unsafe" if label else " Safe"
    verdict_id = unsafe_ids[0] if label else safe_ids[0]
    combined = prompt_ids + prefix_ids + [verdict_id]
    if list(tokenizer.encode(prompt + "Safety:" + verdict_text, add_special_tokens=False)) != combined:
        raise ValueError("Tokenizer merges across the safety-verdict boundary; cannot isolate its label token")
    input_ids = encode_guard_prompt(tokenizer, text, max_length) + prefix_ids + [verdict_id]
    original_length = len(combined)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * (len(input_ids) - 1) + [verdict_id],
        "truncated_tokens": max(original_length - len(input_ids), 0),
    }


def optimizer_groups(order: list[int], pairs_per_step: int) -> Iterable[list[int]]:
    if pairs_per_step <= 0:
        raise ValueError("pairs_per_step must be positive")
    for offset in range(0, len(order), pairs_per_step):
        yield order[offset:offset + pairs_per_step]


def paired_objective(disguised_sum: Any, raw_sum: Any, pair_count: int, lambda_raw: float) -> Any:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    return (disguised_sum + lambda_raw * raw_sum) / pair_count


def cosine_learning_rate_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0 or not 0 <= step < total_steps or not 0 <= warmup_steps <= total_steps:
        raise ValueError("Invalid scheduler step")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def collate_records(records: list[dict], pad_token_id: int, device: str) -> dict:
    import torch

    width = max(len(record["input_ids"]) for record in records)
    result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for record in records:
        extra = width - len(record["input_ids"])
        result["input_ids"].append(record["input_ids"] + [pad_token_id] * extra)
        result["attention_mask"].append(record["attention_mask"] + [0] * extra)
        result["labels"].append(record["labels"] + [-100] * extra)
    return {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in result.items()}


def verdict_losses(logits: Any, labels: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    active = labels.ne(-100)
    if not torch.all(active.sum(dim=1).eq(1)):
        raise ValueError("Every training row must supervise exactly one verdict token")
    row_indices, token_indices = active.nonzero(as_tuple=True)
    if torch.any(token_indices.eq(0)):
        raise ValueError("The verdict requires at least one context token")
    selected_logits = logits[row_indices, token_indices - 1, :].float()
    targets = labels[row_indices, token_indices]
    return functional.cross_entropy(selected_logits, targets, reduction="none")


def train_round(pairs: list[dict], model_path: str, output_dir: Path, settings: dict) -> Path:
    config = validate_settings(settings)
    validate_pairs(pairs)
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path is required")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output_dir}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Training requires the torch and transformers runtime dependencies") from exc

    device = torch.device(config["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This release supports one CUDA GPU or CPU verification")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cpu" and config["dtype"] != "float32":
        raise ValueError("CPU verification requires dtype=float32")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either a padding or end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    encoded_pairs = [
        (
            encode_verdict(tokenizer, pair["disguised_prompt"], pair["label"], config["max_length"]),
            encode_verdict(tokenizer, pair["raw_prompt"], pair["label"], config["max_length"]),
        )
        for pair in pairs
    ]
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, config["dtype"]))
    model.to(device)
    model.requires_grad_(True)
    model.train()
    original_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
        foreach=False,
    )
    use_bfloat16 = device.type == "cuda" and config["dtype"] == "bfloat16"
    pairs_per_step = config["effective_batch_size"] // 2
    total_steps = math.ceil(len(pairs) / pairs_per_step) * config["epochs"]
    warmup_steps = min(total_steps, max(1, math.ceil(0.03 * total_steps)))
    optimizer_step = 0
    updates: list[dict] = []
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        order = list(range(len(pairs)))
        random.Random(config["seed"] + epoch).shuffle(order)
        for group in optimizer_groups(order, pairs_per_step):
            optimizer.zero_grad(set_to_none=True)
            learning_rate = config["learning_rate"] * cosine_learning_rate_scale(
                optimizer_step, total_steps, warmup_steps,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            step_disguised = 0.0
            step_raw = 0.0
            for micro_group in optimizer_groups(group, config["microbatch_pairs"]):
                rows = [record for index in micro_group for record in encoded_pairs[index]]
                batch = collate_records(rows, tokenizer.pad_token_id, str(device))
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    output = model(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
                    )
                    losses = verdict_losses(output.logits, batch["labels"])
                    disguised_sum = losses[0::2].sum()
                    raw_sum = losses[1::2].sum()
                    loss = paired_objective(disguised_sum, raw_sum, len(group), config["lambda_raw"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite verdict loss; the round checkpoint was not saved")
                loss.backward()
                step_disguised += float(disguised_sum.detach())
                step_raw += float(raw_sum.detach())
                del output, losses, loss, disguised_sum, raw_sum, batch
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer_step += 1
            record = {
                "epoch": epoch + 1,
                "optimizer_step": optimizer_step,
                "pair_count": len(group),
                "row_count": 2 * len(group),
                "learning_rate": learning_rate,
                "disguised_ce": step_disguised / len(group),
                "raw_ce": step_raw / len(group),
                "loss": paired_objective(step_disguised, step_raw, len(group), config["lambda_raw"]),
                "gradient_norm_before_clipping": float(gradient_norm),
            }
            updates.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    model.config.use_cache = original_use_cache
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)
    report = {
        "settings": config,
        "optimizer": "AdamW",
        "parameter_dtype": config["dtype"],
        "optimizer_state_dtype": config["dtype"],
        "autocast_dtype": "bfloat16" if use_bfloat16 else None,
        "optimizer_reset_each_round": True,
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
        "warmup_steps": warmup_steps,
        "gradient_clip_norm": 1.0,
        "supervision": "Safe/Unsafe verdict token only; full-vocabulary cross entropy",
        "objective": "mean(disguised CE) + lambda_raw * mean(raw CE)",
        "source_pairs": len(pairs),
        "sft_rows": 2 * len(pairs),
        "optimizer_steps": optimizer_step,
        "truncated_rows": sum(int(row["truncated_tokens"] > 0) for pair in encoded_pairs for row in pair),
        "truncated_tokens": sum(row["truncated_tokens"] for pair in encoded_pairs for row in pair),
        "elapsed_seconds": time.monotonic() - started,
        "updates": updates,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_dir
