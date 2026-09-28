from __future__ import annotations

import hashlib

from .prompts import STRATEGY_EXTRACTION_PROMPT
from .providers import ChatModel
from .schemas import StrategyCandidate
from .utils import parse_json_value


class ExternalKnowledgeEvolution:

    def __init__(self, model: ChatModel):
        self.model = model

    def extract(self, material: str, source_id: str = "") -> list[StrategyCandidate]:
        raw = self.model.generate(
            [
                {
                    "role": "user",
                    "content": STRATEGY_EXTRACTION_PROMPT.format(material=material),
                }
            ]
        )
        payload = parse_json_value(raw)
        if not isinstance(payload, list):
            raise ValueError("Strategy extractor must return a JSON array")
        source_hash = hashlib.sha256((source_id + "\0" + material).encode("utf-8")).hexdigest()
        candidates: list[StrategyCandidate] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item["source_hash"] = source_hash
            item["key"] = item.get("key") or f"external-{source_hash[:12]}-{index:02d}"
            candidates.append(StrategyCandidate.from_dict(item))
        return candidates
