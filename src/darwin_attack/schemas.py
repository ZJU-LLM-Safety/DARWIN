from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StrategyCandidate:
    key: str
    name: str
    instruction: str
    mode: str = "instruction"
    tags: tuple[str, ...] = ()
    source_hash: str = ""
    generation: int = 0
    parent_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyCandidate":
        mode = str(data.get("mode", "instruction"))
        if mode not in {"instruction", "template"}:
            raise ValueError("strategy mode must be instruction or template")
        instruction = str(data["instruction"]).strip()
        if not instruction:
            raise ValueError("strategy instruction must not be empty")
        if mode == "template" and "{input}" not in instruction:
            raise ValueError("template strategies must contain the {input} placeholder")
        return cls(
            key=str(data["key"]).strip(),
            name=str(data["name"]).strip(),
            instruction=instruction,
            mode=mode,
            tags=tuple(str(item) for item in data.get("tags", [])),
            source_hash=str(data.get("source_hash", "")),
            generation=int(data.get("generation", 0)),
            parent_ids=tuple(int(item) for item in data.get("parent_ids", [])),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyRecord:
    id: int
    key: str
    name: str
    instruction: str
    mode: str
    status: str
    total_attempts: int
    total_successes: int
    sandbox_success_rate: float | None
    sandbox_average_score: float | None
    validation_status: str
    generation: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SandboxReport:
    successes: int
    trials: int
    average_score: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0


@dataclass(frozen=True)
class TargetObservation:
    output: str
    decision: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    success: bool
    score: float
    rationale: str


@dataclass(frozen=True)
class AttackAttempt:
    query_number: int
    chain_index: int
    step_index: int
    strategy_id: int
    strategy_name: str
    strategy_sequence: tuple[int, ...]
    disguised_prompt: str
    raw_target_response: str
    target_response: str
    response_extraction: dict[str, Any]
    target_decision: str | None
    score: float
    success: bool
    evaluator_output: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttackResult:
    instance_id: str
    goal: str
    target_id: str
    dataset_id: str
    success: bool
    query_count: int
    best_score: float
    harmfulness_score: float | None
    attempts: tuple[AttackAttempt, ...]
    harmfulness_evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
