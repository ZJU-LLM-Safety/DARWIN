from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


class ConfigurationError(ValueError):
    pass


PAPER_SETTINGS = {
    "method": "DARWIN-Guard",
    "models.guard.identity": "Qwen3Guard-Gen-8B",
    "models.generator.identity": "Mistral-7B-Instruct-v0.2",
    "models.filter.identity": "Gemma-4-31B-it",
    "online.rounds": 20,
    "online.examples_per_round": 500,
    "online.harmful_fraction": 0.1,
    "online.max_attempts": 20,
    "training.scope": "full",
    "training.optimizer": "AdamW",
    "training.learning_rate": 5e-6,
    "training.schedule": "cosine",
    "training.warmup_ratio": 0.03,
    "training.epochs": 1,
    "training.effective_batch_size": 32,
    "training.supervision": "safety_label_only",
    "training.lambda_raw": 1.0,
}

REQUIRED = (
    "runtime.output_dir", "runtime.seed", "models.guard.path",
    "models.generator.provider", "models.generator.model", "models.generator.temperature",
    "models.generator.max_tokens",
    "models.filter.provider", "models.filter.model", "models.filter.temperature",
    "models.filter.max_tokens", "data.harmful_path", "data.benign_path",
    "attack.config_path", "attack.initial_database", "attack.genetic_candidates",
    "attack.reflection_candidates", "attack.max_evolution_batches", "online.examples_unit", "online.harmful_fraction",
    "training.lambda_raw", "training.max_length", "training.dtype",
    "training.device", "training.microbatch_pairs", "training.weight_decay",
    "training.gradient_checkpointing", "inference.max_new_tokens",
)

PATH_KEYS = (
    "runtime.output_dir", "models.guard.path", "data.harmful_path",
    "data.benign_path", "attack.config_path", "attack.initial_database",
    "attack.external_material",
)


def get(config: dict, key: str):
    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def put(config: dict, key: str, value):
    parts = key.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value


def missing_settings(config: dict) -> list[str]:
    return [key for key in REQUIRED if get(config, key) is None or get(config, key) == ""]


def _number(config, key, minimum, *, integer=False, maximum=None):
    value = get(config, key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key} must be a number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ConfigurationError(f"{key} is outside its permitted range")
    if integer and type(value) is not int:
        raise ConfigurationError(f"{key} must be an integer")


def validate(config: dict, *, allow_unset=False) -> list[str]:
    for key, expected in PAPER_SETTINGS.items():
        actual = get(config, key)
        if isinstance(actual, bool) or actual != expected or (type(expected) is int and type(actual) is not int):
            raise ConfigurationError(f"Paper profile requires {key}={expected!r}; got {actual!r}")
    missing = missing_settings(config)
    if missing and not allow_unset:
        raise ConfigurationError("Set the required configuration values: " + ", ".join(missing))
    _number(config, "runtime.seed", 0, integer=True)
    for key in ("attack.genetic_candidates", "attack.reflection_candidates", "attack.max_evolution_batches"):
        _number(config, key, 1, integer=True)
    targets = get(config, "attack.round_pool_targets")
    if targets is not None:
        if (not isinstance(targets, list) or len(targets) != PAPER_SETTINGS["online.rounds"]
                or any(type(value) is not int or not 50 <= value <= 200 for value in targets)
                or targets != sorted(targets) or targets[-1] != 200):
            raise ConfigurationError("attack.round_pool_targets must contain 20 nondecreasing integers from 50 to 200, ending at 200")
    _number(config, "training.max_length", 2, integer=True)
    for key in ("training.microbatch_pairs", "inference.max_new_tokens", "models.filter.max_tokens"):
        _number(config, key, 1, integer=True)
    for key in ("training.lambda_raw", "training.weight_decay", "models.filter.temperature"):
        _number(config, key, 0)
    if get(config, "training.lambda_raw") == 0:
        raise ConfigurationError("lambda_raw must be positive to retain the raw-prompt term")
    _number(config, "online.harmful_fraction", 0, maximum=1)
    if get(config, "online.harmful_fraction") in (0, 1):
        raise ConfigurationError("The paper method requires both harmful and benign sources")
    units = get(config, "online.examples_unit")
    if units != "rows":
        raise ConfigurationError("DARWIN-Guard counts 500 raw-plus-disguised rows per round; online.examples_unit must be rows")
    dtype = get(config, "training.dtype")
    if dtype is not None and dtype not in {"bfloat16", "float32"}:
        raise ConfigurationError("training.dtype must be bfloat16 or float32")
    micro = get(config, "training.microbatch_pairs")
    if micro is not None and micro > get(config, "training.effective_batch_size") // 2:
        raise ConfigurationError("microbatch_pairs exceeds the effective pair batch size")
    gc = get(config, "training.gradient_checkpointing")
    if gc is not None and not isinstance(gc, bool):
        raise ConfigurationError("training.gradient_checkpointing must be boolean")
    for role in ("generator", "filter"):
        provider = get(config, f"models.{role}.provider")
        if provider is not None and provider not in {"transformers", "openai_compatible"}:
            raise ConfigurationError(f"Unsupported {role} provider")
        if provider == "openai_compatible" and not get(config, f"models.{role}.api_key_env"):
            raise ConfigurationError("API models need api_key_env, never a literal key")
        for key in ("api_key", "base_url", "endpoint"):
            if key in (get(config, f"models.{role}") or {}):
                raise ConfigurationError(f"Do not store {key} in YAML; name an environment variable instead")
        _number(config, f"models.{role}.max_tokens", 1, integer=True)
        _number(config, f"models.{role}.temperature", 0)
    if not missing:
        pairs = pairs_per_round(config)
        harmful = pairs * get(config, "online.harmful_fraction")
        if abs(harmful - round(harmful)) > 1e-8:
            raise ConfigurationError("harmful_fraction must give an integer number of pairs per round")
    return missing


def pairs_per_round(config: dict) -> int:
    if get(config, "online.examples_unit") != "rows":
        raise ConfigurationError("DARWIN-Guard examples_per_round counts raw-plus-disguised rows")
    size = int(get(config, "online.examples_per_round"))
    if size % 2:
        raise ConfigurationError("Each DARWIN-Guard source requires one raw and one disguised row")
    return size // 2


def load_config(path: str | Path, *, allow_unset=False) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install the package dependencies before reading YAML") from exc
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConfigurationError("The configuration must contain a mapping")
    validate(config, allow_unset=allow_unset)
    config = copy.deepcopy(config)
    for key in PATH_KEYS:
        value = get(config, key)
        if value:
            p = Path(value).expanduser()
            put(config, key, str(p if p.is_absolute() else (path.parent / p).resolve()))
    for role in ("generator", "filter"):
        value = get(config, f"models.{role}.model")
        if isinstance(value, str) and value.startswith("."):
            put(config, f"models.{role}.model", str((path.parent / value).resolve()))
    return config


def fingerprint(config: dict) -> str:
    inputs = {}
    input_paths = {key: get(config, key) for key in (
        "attack.config_path", "attack.initial_database", "attack.external_material",
        "data.harmful_path", "data.benign_path",
    )}
    attack_path = input_paths["attack.config_path"]
    if attack_path and Path(attack_path).is_file():
        import yaml

        attack_path = Path(attack_path)
        attack_config = yaml.safe_load(attack_path.read_text(encoding="utf-8")) or {}
        for key in ("sandbox.dataset_path", "attack.dataset_path", "pool.mutation_operators_file"):
            value = get(attack_config, key)
            if value:
                path = Path(value).expanduser()
                input_paths["attack." + key] = str(path if path.is_absolute() else attack_path.parent / path)
    for key, path in input_paths.items():
        if path and Path(path).is_file():
            digest = hashlib.sha256()
            with Path(path).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            inputs[key] = digest.hexdigest()
        else:
            inputs[key] = None
    payload = {"config": config, "input_sha256": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
