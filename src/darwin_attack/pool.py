from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder, cosine_similarity
from .schemas import SandboxReport, StrategyCandidate, StrategyRecord
from .storage import Repository


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    strategy_id: int | None = None
    maximum_similarity: float = 0.0


class StrategyPool:
    def __init__(
        self,
        repository: Repository,
        embedder: Embedder,
        similarity_threshold: float,
        admission_threshold: float,
        target_size: int,
    ):
        self.repository = repository
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.admission_threshold = admission_threshold
        self.target_size = target_size

    def active(self) -> list[StrategyRecord]:
        return self.repository.active_strategies()

    def precheck(self, candidate: StrategyCandidate) -> AdmissionDecision | None:
        return self._screen(candidate)[0]

    def _screen(
        self, candidate: StrategyCandidate
    ) -> tuple[AdmissionDecision | None, np.ndarray | None, float]:
        if self.repository.strategy_by_key(candidate.key):
            return AdmissionDecision(False, "duplicate_key"), None, 0.0
        if self.repository.count_strategies() >= self.target_size:
            return AdmissionDecision(False, "pool_full"), None, 0.0
        vector = self.embedder.encode([candidate.instruction])[0]
        similarities = [
            cosine_similarity(vector, existing_vector)
            for _, existing_vector in self.repository.embeddings()
        ]
        maximum = max(similarities, default=0.0)
        if maximum >= self.similarity_threshold:
            return (
                AdmissionDecision(False, "semantic_duplicate", maximum_similarity=maximum),
                vector,
                maximum,
            )
        return None, vector, maximum

    def consider(
        self, candidate: StrategyCandidate, sandbox_report: SandboxReport
    ) -> AdmissionDecision:
        rejection, vector, maximum = self._screen(candidate)
        if rejection is not None:
            return rejection
        assert vector is not None
        if sandbox_report.success_rate < self.admission_threshold:
            return AdmissionDecision(False, "below_sandbox_threshold", maximum_similarity=maximum)
        strategy_id = self.repository.add_strategy(candidate, vector, sandbox_report)
        return AdmissionDecision(True, "admitted", strategy_id, maximum)
