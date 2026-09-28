from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import read_jsonl


def _goal(record: dict[str, Any]) -> str:
    for key in ("goal", "question", "prompt", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Dataset record has no goal/question/prompt/instruction field")


def load_goals(path: str | Path) -> list[tuple[str, str]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        records = list(read_jsonl(source))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = payload.get("data") or payload.get("records") or []
        else:
            records = payload
        if not isinstance(records, list):
            raise ValueError("Dataset JSON must contain a list of records")
    output = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset item {index} is not an object")
        instance_id = str(record.get("id", index))
        output.append((instance_id, _goal(record)))
    return output


def assert_disjoint_goals(
    sandbox_goals: list[tuple[str, str]], evaluation_goals: list[tuple[str, str]]
) -> None:
    sandbox_text = {goal.strip() for _, goal in sandbox_goals}
    overlap = sandbox_text.intersection(goal.strip() for _, goal in evaluation_goals)
    if overlap:
        raise ValueError("Sandbox and final-evaluation datasets contain overlapping prompts")
