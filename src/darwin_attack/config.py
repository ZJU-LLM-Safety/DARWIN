from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


class ConfigurationError(ValueError):
    pass


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ConfigurationError(f"Missing required setting: {section}.{key}")
    return value


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key_env: str | None = None
    base_url_env: str | None = None
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    identity: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], section: str) -> "ModelConfig":
        provider = str(_required(data, "provider", section)).strip().lower()
        if provider not in {"openai_compatible", "transformers"}:
            raise ConfigurationError(
                f"{section}.provider must be openai_compatible or transformers"
            )
        max_tokens = int(_required(data, "max_tokens", section))
        if max_tokens <= 0:
            raise ConfigurationError(f"{section}.max_tokens must be positive")
        return cls(
            provider=provider,
            model=str(_required(data, "model", section)),
            temperature=float(_required(data, "temperature", section)),
            max_tokens=max_tokens,
            api_key_env=data.get("api_key_env"),
            base_url_env=data.get("base_url_env"),
            device=data.get("device"),
            extra=dict(data.get("extra") or {}),
            identity=data.get("identity"),
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    device: str | None
    similarity_threshold: float
    history_threshold: float


@dataclass(frozen=True)
class PoolConfig:
    target_size: int
    admission_threshold: float
    crossover_top_k: int
    crossover_probability: float
    mutation_probability: float
    mutation_operator_count: int
    mutation_operators_file: Path


@dataclass(frozen=True)
class SelectionConfig:
    alpha: float
    gamma: float


@dataclass(frozen=True)
class SandboxConfig:
    dataset_path: Path
    goals_per_candidate: int
    trials_per_goal: int


@dataclass(frozen=True)
class AttackConfig:
    dataset_path: Path
    dataset_id: str
    target_id: str
    target_kind: str
    max_target_queries: int
    chains_per_instance: int
    max_chain_length: int
    success_score: int
    guardrail_safe_pattern: str | None
    guardrail_unsafe_pattern: str | None
    guardrail_template: str | None = None
    guardrail_template_file: Path | None = None
    guardrail_template_id: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    database_path: Path
    random_seed: int


@dataclass(frozen=True)
class DarwinConfig:
    runtime: RuntimeConfig
    models: dict[str, ModelConfig]
    embedding: EmbeddingConfig
    pool: PoolConfig
    selection: SelectionConfig
    sandbox: SandboxConfig
    attack: AttackConfig

    def model(self, role: str) -> ModelConfig:
        try:
            return self.models[role]
        except KeyError as exc:
            raise ConfigurationError(f"Missing model role: models.{role}") from exc


def _resolve(base: Path, value: Any, section: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ConfigurationError(f"Missing required path: {section}")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _probability(value: Any, section: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ConfigurationError(f"{section} must be in [0, 1]")
    return parsed


def load_config(path: str | Path, *, validate_target: bool = True) -> DarwinConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent

    runtime_raw = raw.get("runtime") or {}
    embedding_raw = raw.get("embedding") or {}
    pool_raw = raw.get("pool") or {}
    selection_raw = raw.get("selection") or {}
    sandbox_raw = raw.get("sandbox") or {}
    attack_raw = raw.get("attack") or {}
    models_raw = raw.get("models") or {}

    models = {
        role: ModelConfig.from_dict(value or {}, f"models.{role}")
        for role, value in models_raw.items()
    }
    required_roles = {
        "strategy_generator",
        "reflection",
        "sandbox_target",
        "target",
        "judge",
        "intent_judge",
    }
    missing_roles = sorted(required_roles - models.keys())
    if missing_roles:
        raise ConfigurationError(f"Missing model roles: {', '.join(missing_roles)}")

    runtime = RuntimeConfig(
        database_path=_resolve(
            base, _required(runtime_raw, "database_path", "runtime"), "runtime.database_path"
        ),
        random_seed=int(_required(runtime_raw, "random_seed", "runtime")),
    )
    embedding = EmbeddingConfig(
        model=str(_required(embedding_raw, "model", "embedding")),
        device=embedding_raw.get("device"),
        similarity_threshold=_probability(
            _required(embedding_raw, "similarity_threshold", "embedding"),
            "embedding.similarity_threshold",
        ),
        history_threshold=_probability(
            _required(embedding_raw, "history_threshold", "embedding"),
            "embedding.history_threshold",
        ),
    )
    pool = PoolConfig(
        target_size=int(_required(pool_raw, "target_size", "pool")),
        admission_threshold=_probability(
            _required(pool_raw, "admission_threshold", "pool"),
            "pool.admission_threshold",
        ),
        crossover_top_k=int(_required(pool_raw, "crossover_top_k", "pool")),
        crossover_probability=_probability(
            _required(pool_raw, "crossover_probability", "pool"),
            "pool.crossover_probability",
        ),
        mutation_probability=_probability(
            _required(pool_raw, "mutation_probability", "pool"),
            "pool.mutation_probability",
        ),
        mutation_operator_count=int(_required(pool_raw, "mutation_operator_count", "pool")),
        mutation_operators_file=_resolve(
            base,
            _required(pool_raw, "mutation_operators_file", "pool"),
            "pool.mutation_operators_file",
        ),
    )
    selection = SelectionConfig(
        alpha=_probability(_required(selection_raw, "alpha", "selection"), "selection.alpha"),
        gamma=_probability(_required(selection_raw, "gamma", "selection"), "selection.gamma"),
    )
    sandbox = SandboxConfig(
        dataset_path=_resolve(
            base,
            _required(sandbox_raw, "dataset_path", "sandbox"),
            "sandbox.dataset_path",
        ),
        goals_per_candidate=int(_required(sandbox_raw, "goals_per_candidate", "sandbox")),
        trials_per_goal=int(_required(sandbox_raw, "trials_per_goal", "sandbox")),
    )
    target_kind = str(_required(attack_raw, "target_kind", "attack")).lower()
    if target_kind not in {"llm", "guardrail"}:
        raise ConfigurationError("attack.target_kind must be llm or guardrail")
    attack = AttackConfig(
        dataset_path=_resolve(
            base,
            _required(attack_raw, "dataset_path", "attack"),
            "attack.dataset_path",
        ),
        dataset_id=str(_required(attack_raw, "dataset_id", "attack")),
        target_id=str(_required(attack_raw, "target_id", "attack")),
        target_kind=target_kind,
        max_target_queries=int(_required(attack_raw, "max_target_queries", "attack")),
        chains_per_instance=int(_required(attack_raw, "chains_per_instance", "attack")),
        max_chain_length=int(_required(attack_raw, "max_chain_length", "attack")),
        success_score=int(_required(attack_raw, "success_score", "attack")),
        guardrail_safe_pattern=attack_raw.get("guardrail_safe_pattern"),
        guardrail_unsafe_pattern=attack_raw.get("guardrail_unsafe_pattern"),
        guardrail_template=attack_raw.get("guardrail_template"),
        guardrail_template_file=(
            _resolve(base, attack_raw["guardrail_template_file"], "attack.guardrail_template_file")
            if attack_raw.get("guardrail_template_file") else None
        ),
        guardrail_template_id=attack_raw.get("guardrail_template_id"),
    )

    positive_values = {
        "pool.target_size": pool.target_size,
        "pool.crossover_top_k": pool.crossover_top_k,
        "pool.mutation_operator_count": pool.mutation_operator_count,
        "sandbox.goals_per_candidate": sandbox.goals_per_candidate,
        "sandbox.trials_per_goal": sandbox.trials_per_goal,
        "attack.max_target_queries": attack.max_target_queries,
        "attack.chains_per_instance": attack.chains_per_instance,
        "attack.max_chain_length": attack.max_chain_length,
    }
    for label, value in positive_values.items():
        if value <= 0:
            raise ConfigurationError(f"{label} must be positive")
    if not 1 <= attack.success_score <= 5:
        raise ConfigurationError("attack.success_score must be in [1, 5]")
    if validate_target and attack.target_kind == "guardrail":
        _validate_guardrail(attack, models["target"])
    if attack.chains_per_instance * attack.max_chain_length < attack.max_target_queries:
        raise ConfigurationError(
            "chains_per_instance * max_chain_length must cover max_target_queries"
        )
    if sandbox.dataset_path == attack.dataset_path:
        raise ConfigurationError("Sandbox and final-evaluation datasets must be different files")

    return DarwinConfig(
        runtime=runtime,
        models=models,
        embedding=embedding,
        pool=pool,
        selection=selection,
        sandbox=sandbox,
        attack=attack,
    )


def _validate_guardrail(attack: AttackConfig, model: ModelConfig) -> None:
    from .guardrail_templates import QWEN_TEMPLATE, YUFENG_TEMPLATE, validate_custom_template

    template = attack.guardrail_template
    if template not in {QWEN_TEMPLATE, YUFENG_TEMPLATE, "custom", "prewrapped_endpoint"}:
        raise ConfigurationError(
            "guardrail targets require attack.guardrail_template: qwen3guard_binary, "
            "yufeng_xguard, custom, or prewrapped_endpoint"
        )
    if model.temperature != 0:
        raise ConfigurationError("Paper guardrail targets require greedy decoding (models.target.temperature=0)")
    template_id = template if template in {QWEN_TEMPLATE, YUFENG_TEMPLATE} else attack.guardrail_template_id
    if not isinstance(template_id, str) or not template_id.strip():
        raise ConfigurationError("custom/prewrapped_endpoint requires attack.guardrail_template_id")
    identity = str(model.identity or model.model).lower()
    expected = {QWEN_TEMPLATE: "qwen3guard", YUFENG_TEMPLATE: "yufeng-xguard"}.get(template_id)
    if expected and expected not in identity:
        raise ConfigurationError(
            f"Template {template_id} requires matching models.target.identity (or model name)"
        )
    if template == YUFENG_TEMPLATE and model.provider != "transformers":
        raise ConfigurationError(
            "yufeng_xguard uses the official local tokenizer; remote targets require "
            "custom with its complete template or prewrapped_endpoint with template_id=yufeng_xguard"
        )
    if template == "prewrapped_endpoint" and model.provider != "openai_compatible":
        raise ConfigurationError("prewrapped_endpoint requires an openai_compatible remote provider")
    if template == "custom":
        if not attack.guardrail_template_file or not attack.guardrail_template_file.is_file():
            raise ConfigurationError("custom requires an existing attack.guardrail_template_file")
        try:
            validate_custom_template(attack.guardrail_template_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ConfigurationError(str(exc)) from exc
    elif attack.guardrail_template_file is not None:
        raise ConfigurationError("guardrail_template_file is only used by the custom template mode")
    patterns = (attack.guardrail_safe_pattern, attack.guardrail_unsafe_pattern)
    if template_id in {QWEN_TEMPLATE, YUFENG_TEMPLATE}:
        if any(patterns):
            raise ConfigurationError("Built-in guardrails use paper decision rules; leave guardrail patterns null")
    else:
        if not all(isinstance(pattern, str) and pattern.strip() for pattern in patterns):
            raise ConfigurationError("Custom guardrail decision rules require safe and unsafe patterns")
        try:
            for pattern in patterns:
                re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ConfigurationError(f"Invalid guardrail output pattern: {exc}") from exc
