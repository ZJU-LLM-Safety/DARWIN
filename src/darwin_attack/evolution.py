from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .prompts import CROSSOVER_PROMPT, MUTATION_PROMPT, STRATEGY_REFLECTION_PROMPT
from .providers import ChatModel
from .schemas import StrategyCandidate, StrategyRecord
from .storage import Repository
from .utils import parse_json_value, read_jsonl


def load_mutation_operators(path: str | Path) -> list[str]:
    operators = []
    names: set[str] = set()
    for record in read_jsonl(path):
        instruction = str(record.get("instruction", "")).strip()
        if not instruction:
            continue
        name = str(record.get("name", "")).strip()
        dimension = str(record.get("dimension", "")).strip()
        if name and name in names:
            raise ValueError(f"Duplicate mutation operator name: {name}")
        if name:
            names.add(name)
        label = " / ".join(item for item in (name, dimension) if item)
        operators.append(f"{label}: {instruction}" if label else instruction)
    if not operators:
        raise ValueError("The mutation-operator file contains no instructions")
    return operators


class GeneticStrategyEvolution:

    def __init__(
        self,
        repository: Repository,
        model: ChatModel,
        top_k: int,
        crossover_probability: float,
        mutation_probability: float,
        mutation_operators: list[str],
        random_seed: int,
    ):
        self.repository = repository
        self.model = model
        self.top_k = top_k
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_operators = mutation_operators
        self.rng = random.Random(random_seed)

    @staticmethod
    def _strategy_text(strategy: StrategyRecord) -> str:
        return f"Name: {strategy.name}\nInstruction: {strategy.instruction}"

    def generate(self, count: int) -> list[StrategyCandidate]:
        parents = self.repository.ranked_active(self.top_k)
        if not parents:
            raise RuntimeError("Genetic evolution requires an existing strategy pool")
        candidates: list[StrategyCandidate] = []
        for index in range(count):
            selected = [self.rng.choice(parents)]
            if len(parents) >= 2 and self.rng.random() < self.crossover_probability:
                selected = self.rng.sample(parents, 2)
                raw = self.model.generate(
                    [
                        {
                            "role": "user",
                            "content": CROSSOVER_PROMPT.format(
                                left=self._strategy_text(selected[0]),
                                right=self._strategy_text(selected[1]),
                            ),
                        }
                    ]
                )
                payload = parse_json_value(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Evolution model must return one JSON object")
            else:
                payload = {
                    "name": selected[0].name,
                    "instruction": selected[0].instruction,
                    "mode": selected[0].mode,
                    "tags": [],
                }

            if self.rng.random() < self.mutation_probability:
                raw = self.model.generate(
                    [
                        {
                            "role": "user",
                            "content": MUTATION_PROMPT.format(
                                candidate=str(payload),
                                operator=self.rng.choice(self.mutation_operators),
                            ),
                        }
                    ]
                )
                payload = parse_json_value(raw)
                if not isinstance(payload, dict):
                    raise ValueError("Mutation model must return one JSON object")

            serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
            payload = dict(payload)
            payload["key"] = payload.get("key") or f"evolved-{fingerprint}-{index:02d}"
            payload["generation"] = max(item.generation for item in selected) + 1
            payload["parent_ids"] = [item.id for item in selected]
            candidates.append(StrategyCandidate.from_dict(payload))
        return candidates


class ReflectionDrivenEvolution:

    def __init__(self, repository: Repository, model: ChatModel):
        self.repository = repository
        self.model = model

    def generate(self, strategy_id: int, failed_prompt: str, feedback: str) -> StrategyCandidate:
        parent = self.repository.strategy(strategy_id)
        if parent is None:
            raise ValueError(f"Unknown strategy id: {strategy_id}")
        raw = self.model.generate(
            [
                {
                    "role": "user",
                    "content": STRATEGY_REFLECTION_PROMPT.format(
                        strategy=GeneticStrategyEvolution._strategy_text(parent),
                        failed_prompt=failed_prompt,
                        feedback=feedback,
                    ),
                }
            ]
        )
        payload = parse_json_value(raw)
        if not isinstance(payload, dict):
            raise ValueError("Reflection model must return one JSON object")
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        payload = dict(payload)
        payload["key"] = payload.get("key") or f"reflected-{fingerprint}"
        payload["generation"] = parent.generation + 1
        payload["parent_ids"] = [parent.id]
        return StrategyCandidate.from_dict(payload)
