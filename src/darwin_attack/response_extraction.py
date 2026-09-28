from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_target_response(
    raw_response: str,
    strategy_metadata: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw_response, str):
        raise ValueError("Target response must be a string")
    return raw_response, {
        "applied": False,
        "extractor": "",
        "marker_found": False,
        "evaluation_scope": "full_response",
        "legacy_extractor_ignored": bool((strategy_metadata or {}).get("response_extractor")),
    }


def recorded_target_response(attempt: Mapping[str, Any]) -> str:
    if "raw_target_response" not in attempt:
        extraction = attempt.get("response_extraction") or {}
        if isinstance(extraction, Mapping) and extraction.get("applied") is True:
            raise ValueError("Cannot rejudge an extracted response without raw_target_response")
    field = "raw_target_response" if "raw_target_response" in attempt else "target_response"
    response = attempt.get(field, "")
    if not isinstance(response, str):
        raise ValueError(f"Recorded {field} must be a string")
    return response
