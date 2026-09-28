from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable


SOURCE_KINDS = {"normalized", "wildjailbreak", "orbench"}


def _label(value: Any) -> int:
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip().lower() in {"safe", "unsafe", "0", "1"}:
        return int(value.strip().lower() in {"unsafe", "1"})
    raise ValueError("Source label must be 0/Safe or 1/Unsafe")


def normalize_source(
    record: dict[str, Any], expected_label: int | None = None,
    source_kind: str = "normalized",
) -> dict[str, Any] | None:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind: {source_kind}")
    if not isinstance(record, dict):
        raise ValueError("Every source record must be an object")
    if expected_label is not None and (type(expected_label) is not int or expected_label not in (0, 1)):
        raise ValueError("expected_label must be integer 0 or 1")
    if source_kind == "wildjailbreak":
        if str(record.get("data_type", "")).strip() != "vanilla_harmful":
            return None
        if str(record.get("split", "train")).strip() != "train":
            return None
        if expected_label not in (None, 1):
            raise ValueError("WildJailbreak vanilla_harmful requires label 1")
        expected_label = 1
        text = record.get("vanilla")
    elif source_kind == "orbench":
        if expected_label not in (None, 0):
            raise ValueError("OR-Bench benign sources require label 0")
        for key in ("config", "subset", "or_bench_config", "data_type"):
            if "toxic" in str(record.get(key, "")).lower():
                raise ValueError("OR-Bench toxic subsets cannot be used as benign sources")
        expected_label = 0
        text = record.get("prompt", record.get("text"))
    else:
        text = record.get("text", record.get("prompt"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A source must have non-empty text")
    text = text.strip()
    if "label" in record:
        label = _label(record["label"])
        if expected_label is not None and label != expected_label:
            raise ValueError("Source label conflicts with its declared dataset route")
    elif expected_label is not None:
        label = expected_label
    else:
        raise ValueError("Normalized sources require a ground-truth label")
    identifier = next((record[key] for key in ("id", "sample_id", "prompt_id")
                       if key in record and record[key] is not None and str(record[key]).strip()), None)
    if identifier is None:
        identifier = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    elif isinstance(identifier, (dict, list, bool)):
        raise ValueError("Source ids must be strings or numbers")
    return {"id": str(identifier).strip(), "text": text, "label": label}


def deduplicate_sources(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_text: dict[str, dict[str, Any]] = {}
    for record in records:
        item = normalize_source(record)
        assert item is not None
        previous_id = by_id.get(item["id"])
        if previous_id is not None and previous_id != item:
            raise ValueError("One source id refers to different texts or labels")
        previous_text = by_text.get(item["text"])
        if previous_text is not None and previous_text["label"] != item["label"]:
            raise ValueError("Identical source text has conflicting ground-truth labels")
        by_id[item["id"]] = item
        if previous_text is None:
            by_text[item["text"]] = item
            result.append(item)
    return result


def _read_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("A JSON source file must contain an array")
            yield from payload
        elif path.suffix.lower() in {".csv", ".tsv"}:
            yield from csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        else:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on source line {line_number}") from exc


def load_sources(
    path: str | Path, expected_label: int | None = None,
    source_kind: str = "normalized",
) -> list[dict[str, Any]]:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind: {source_kind}")
    if expected_label is not None and (type(expected_label) is not int or expected_label not in (0, 1)):
        raise ValueError("expected_label must be integer 0 or 1")
    normalized = (normalize_source(row, expected_label, source_kind) for row in _read_records(Path(path)))
    return deduplicate_sources(row for row in normalized if row is not None)


def select_sources(
    harmful: Iterable[dict[str, Any]], benign: Iterable[dict[str, Any]], *,
    total: int, harmful_fraction: float, seed: int,
    exclude_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    if type(total) is not int or total <= 0:
        raise ValueError("total must be a positive integer")
    if isinstance(harmful_fraction, bool) or not math.isfinite(harmful_fraction) or not 0 <= harmful_fraction <= 1:
        raise ValueError("harmful_fraction must be finite and between 0 and 1")
    requested_harmful = total * harmful_fraction
    if not math.isclose(requested_harmful, round(requested_harmful), abs_tol=1e-9):
        raise ValueError("The requested fraction must produce an integer harmful quota")
    harmful_rows = [normalize_source(row, expected_label=1) for row in harmful]
    benign_rows = [normalize_source(row, expected_label=0) for row in benign]
    combined = deduplicate_sources(harmful_rows + benign_rows)
    excluded = set(exclude_ids)
    quotas = {1: int(round(requested_harmful)), 0: total - int(round(requested_harmful))}
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    for label in (1, 0):
        pool = [row for row in combined if row["label"] == label and row["id"] not in excluded]
        pool.sort(key=lambda row: row["id"])
        if len(pool) < quotas[label]:
            raise ValueError(f"Insufficient unique sources for label {label}: need {quotas[label]}, have {len(pool)}")
        result.extend(rng.sample(pool, quotas[label]))
    rng.shuffle(result)
    return result


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
