"""History memory — vector search for past successful attacks."""
from __future__ import annotations

from config.settings import HISTORY_MATCH_THRESHOLD
from database.chroma_db import ChromaDB
from database.sqlite_db import SQLiteDB


class HistoryMemory:
    """Manage the success history memory pool."""

    def __init__(self, sqlite: SQLiteDB, chroma: ChromaDB):
        self.db = sqlite
        self.vec = chroma

    def store_success(
        self,
        question: str,
        strategy_id: int,
        disguised_prompt: str = "",
        score: float = 0.0,
        target_model: str = "",
        metadata: dict | None = None,
    ):
        self.db.add_success_history(
            question,
            strategy_id,
            disguised_prompt,
            score,
            target_model,
            metadata=metadata,
        )
        rows = self.db.get_success_history()
        record_id = rows[-1]["id"] if rows else 0
        self.vec.add_history(
            record_id,
            question,
            strategy_id,
            metadata={"target_model": target_model, "score": score, **(metadata or {})},
        )

    def find_similar(
        self,
        question: str,
        n: int = 5,
        threshold: float = HISTORY_MATCH_THRESHOLD,
    ) -> list[dict]:
        return self.vec.find_similar_history(question, n, threshold)

    def get_strategy_for_question(self, question: str) -> dict | None:
        matches = self.find_similar(question, n=1)
        return matches[0] if matches else None
