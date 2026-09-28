from __future__ import annotations

from typing import Any, Protocol

from .data import normalize_source


class Proposer(Protocol):
    def start(self, source: dict[str, Any]) -> None: ...

    def propose(
        self, source: dict[str, Any], current_prompt: str, attempt: int
    ) -> dict[str, Any]: ...

    def feedback(
        self, source: dict[str, Any], candidate: dict[str, Any],
        decision: int | None, success: bool,
    ) -> None: ...


class Guard(Protocol):
    def predict(self, text: str) -> int | None: ...


class IntentFilter(Protocol):
    def keep(self, original: str, rewritten: str, label: int) -> bool: ...


def collect_pair(
    source: dict[str, Any],
    proposer: Proposer,
    guard: Guard,
    intent_filter: IntentFilter,
    max_attempts: int = 20,
) -> dict[str, Any] | None:
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    normalized = normalize_source(source)
    assert normalized is not None
    proposer.start(normalized)
    current_prompt = normalized["text"]
    for attempt in range(1, max_attempts + 1):
        candidate = proposer.propose(normalized, current_prompt, attempt)
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("The proposer must return a candidate with string text")
        strategy_ids = candidate.get("strategy_ids")
        if not isinstance(strategy_ids, list) or any(type(item) is not int for item in strategy_ids):
            raise ValueError("candidate.strategy_ids must be a list of integers")
        terminal = {**candidate, "strategy_ids": list(strategy_ids)}
        nonempty = bool(terminal["text"].strip())
        decision = guard.predict(terminal["text"]) if nonempty else None
        if decision is not None and (type(decision) is not int or decision not in (0, 1)):
            raise ValueError("The guard must return integer 0, integer 1, or None")
        misclassified = decision is not None and decision != normalized["label"]
        if misclassified or attempt == max_attempts:
            retained = nonempty and intent_filter.keep(
                normalized["text"], terminal["text"], normalized["label"]
            ) is True
            proposer.feedback(normalized, terminal, decision, misclassified and retained)
            if not retained:
                return None
            return {
                "source_id": normalized["id"],
                "raw_prompt": normalized["text"],
                "disguised_prompt": terminal["text"],
                "label": normalized["label"],
                "attempts": attempt,
                "misclassified": misclassified,
                "strategy_ids": terminal["strategy_ids"],
            }
        proposer.feedback(normalized, terminal, decision, False)
        if nonempty:
            current_prompt = terminal["text"]
    raise AssertionError("Positive attempt budget must reach a terminal candidate")
