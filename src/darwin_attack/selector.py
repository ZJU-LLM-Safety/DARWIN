from __future__ import annotations

import random

import numpy as np

from .embeddings import Embedder, cosine_similarity
from .storage import Repository


class FeedbackGuidedEvolution:

    def __init__(
        self,
        repository: Repository,
        embedder: Embedder,
        scope: str,
        history_threshold: float,
        alpha: float,
        gamma: float,
        random_seed: int,
    ):
        self.repository = repository
        self.embedder = embedder
        self.scope = scope
        self.history_threshold = history_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(random_seed)

    def sync(self) -> list[int]:
        strategy_ids = [item.id for item in self.repository.active_strategies()]
        self.repository.ensure_transitions(self.scope, strategy_ids)
        return strategy_ids

    def encode_goal(self, goal: str) -> np.ndarray:
        return self.embedder.encode([goal])[0]

    def select_initial(self, goal_embedding: np.ndarray) -> int:
        active = {item.id for item in self.repository.active_strategies()}
        if not active:
            raise RuntimeError("The active strategy pool is empty")
        best_similarity = -1.0
        best_strategy: int | None = None
        for memory in self.repository.success_memories(self.scope):
            similarity = cosine_similarity(goal_embedding, memory["embedding"])
            sequence = memory["sequence"]
            candidate = int(sequence[0]) if sequence else None
            if candidate in active and similarity > best_similarity:
                best_similarity = similarity
                best_strategy = candidate
        if best_strategy is not None and best_similarity >= self.history_threshold:
            return best_strategy
        return self.rng.choice(sorted(active))

    def select_next(self, current_strategy_id: int) -> int:
        row = self.repository.transition_row(self.scope, current_strategy_id)
        if not row:
            strategy_ids = self.sync()
            return self.rng.choice(strategy_ids)
        strategy_ids = sorted(row)
        weights = np.asarray([max(row[item], 0.0) for item in strategy_ids], dtype=float)
        if float(weights.sum()) <= 0:
            weights = np.ones(len(strategy_ids), dtype=float)
        weights /= weights.sum()
        return self.rng.choices(strategy_ids, weights=weights.tolist(), k=1)[0]

    def update(self, from_strategy_id: int, to_strategy_id: int, reward: float) -> None:
        if reward not in {0, 1, 0.0, 1.0}:
            raise ValueError("Feedback Guided Evolution rewards must be binary")
        row = self.repository.transition_row(self.scope, from_strategy_id)
        if to_strategy_id not in row:
            self.sync()
            row = self.repository.transition_row(self.scope, from_strategy_id)
        current = row.get(to_strategy_id, 0.0)
        future = self.repository.max_transition(self.scope, to_strategy_id)
        updated = current + self.alpha * (reward + self.gamma * future - current)
        self.repository.update_transition(self.scope, from_strategy_id, to_strategy_id, updated)
