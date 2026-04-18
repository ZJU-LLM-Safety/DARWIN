"""Dynamic Markov selector with Q-learning-style probability updates."""
from __future__ import annotations

import numpy as np

from config.settings import ALPHA, BETA, GAMMA, TEMPERATURE
from database.sqlite_db import SQLiteDB


class MarkovSelector:
    """Dynamic-size Markov transition matrix over active strategies."""

    def __init__(self, sqlite: SQLiteDB):
        self.db = sqlite
        self.strategy_ids: list[int] = []
        self.matrix: np.ndarray | None = None
        self.q_table: np.ndarray | None = None
        self._load_or_init()

    def _load_or_init(self):
        state = self.db.load_markov_state()
        if state:
            self.strategy_ids = state["strategy_ids"]
            n = len(self.strategy_ids)
            self.matrix = np.array(state["matrix"]).reshape(n, n)
            self.q_table = np.array(state["q_table"]).reshape(n, n)

    def sync_strategies(self, active_ids: list[int]):
        if set(active_ids) == set(self.strategy_ids) and self.matrix is not None:
            return

        old_ids = self.strategy_ids
        old_matrix = self.matrix
        old_q = self.q_table

        self.strategy_ids = sorted(active_ids)
        n = len(self.strategy_ids)
        self.matrix = np.ones((n, n), dtype=float) / max(n, 1)
        self.q_table = np.ones((n, n), dtype=float) / max(n, 1)

        if old_matrix is None or old_q is None:
            return

        old_id_map = {sid: i for i, sid in enumerate(old_ids)}
        for i, sid_i in enumerate(self.strategy_ids):
            for j, sid_j in enumerate(self.strategy_ids):
                if sid_i in old_id_map and sid_j in old_id_map:
                    oi, oj = old_id_map[sid_i], old_id_map[sid_j]
                    self.matrix[i, j] = old_matrix[oi, oj]
                    self.q_table[i, j] = old_q[oi, oj]
        self._normalize_rows()

    def _normalize_rows(self):
        if self.matrix is None:
            return
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        self.matrix = self.matrix / row_sums

    def select_next(self, current_strategy_id: int) -> int:
        if not self.strategy_ids:
            raise ValueError("No strategies in Markov selector")
        if current_strategy_id not in self.strategy_ids or self.matrix is None:
            return int(np.random.choice(self.strategy_ids))

        idx = self.strategy_ids.index(current_strategy_id)
        probs = self.matrix[idx].copy()
        if self.q_table is not None:
            q_row = self.q_table[idx]
            q_shifted = q_row - q_row.max()
            q_probs = np.exp(q_shifted / max(TEMPERATURE, 1e-8))
            q_probs = q_probs / q_probs.sum()
            uniform = np.ones(len(self.strategy_ids)) / len(self.strategy_ids)
            probs = (1 - BETA) * (0.5 * probs + 0.5 * q_probs) + BETA * uniform
            probs = probs / probs.sum()

        chosen_idx = int(np.random.choice(len(self.strategy_ids), p=probs))
        return self.strategy_ids[chosen_idx]

    def update(self, from_strategy_id: int, to_strategy_id: int, reward: float):
        if (
            self.matrix is None
            or self.q_table is None
            or from_strategy_id not in self.strategy_ids
            or to_strategy_id not in self.strategy_ids
        ):
            return

        i = self.strategy_ids.index(from_strategy_id)
        j = self.strategy_ids.index(to_strategy_id)

        max_future = float(self.matrix[j].max()) if len(self.matrix[j]) > 0 else 0.0
        updated_value = self.matrix[i, j] + ALPHA * (
            reward + GAMMA * max_future - self.matrix[i, j]
        )
        self.matrix[i, j] = max(updated_value, 0.0)
        self.q_table[i, j] = self.matrix[i, j]
        self._normalize_rows()

    def save(self):
        if self.matrix is None or self.q_table is None:
            return
        self.db.save_markov_state(
            self.matrix.tolist(),
            self.q_table.tolist(),
            self.strategy_ids,
        )

    def select_initial(self, ucb_scores: list[tuple[dict, float]]) -> int:
        if not ucb_scores:
            raise ValueError("No strategies available")
        scores = np.array([s[1] for s in ucb_scores], dtype=float)
        scores_shifted = scores - scores.max()
        probs = np.exp(scores_shifted / max(TEMPERATURE, 1e-8))
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(ucb_scores), p=probs))
        return int(ucb_scores[idx][0]["id"])
