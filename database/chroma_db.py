"""ChromaDB vector store for strategy dedup and history memory."""
import os
import chromadb
import numpy as np
from config.settings import CHROMA_DB_PATH, STRATEGY_DEDUP_THRESHOLD, HISTORY_MATCH_THRESHOLD
from database.embedding import EmbeddingEngine


class ChromaDB:
    """Manages two collections: strategies (dedup) and history_memory (recall)."""

    def __init__(self, persist_dir: str = CHROMA_DB_PATH):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_engine: EmbeddingEngine | None = None

        # Collections
        self.strategies = self.client.get_or_create_collection(
            name="strategies",
            metadata={"hnsw:space": "cosine"},
        )
        self.history_memory = self.client.get_or_create_collection(
            name="history_memory",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    # ── Strategy Dedup ────────────────────────────────────────────
    def is_duplicate_strategy(self, text: str,
                              threshold: float = STRATEGY_DEDUP_THRESHOLD) -> bool:
        """Check if a strategy text is too similar to existing ones."""
        if self.strategies.count() == 0:
            return False
        embedding = self.embedding_engine.embed_single(text).tolist()
        results = self.strategies.query(
            query_embeddings=[embedding], n_results=1,
        )
        if results["distances"] and results["distances"][0]:
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = 1.0 - results["distances"][0][0]
            return similarity >= threshold
        return False

    def add_strategy(self, strategy_id: int, text: str, metadata: dict | None = None):
        """Add a strategy embedding to the vector store."""
        embedding = self.embedding_engine.embed_single(text).tolist()
        meta = metadata or {}
        self.strategies.upsert(
            ids=[str(strategy_id)],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )

    def find_similar_strategies(self, text: str, n: int = 5) -> list[dict]:
        """Find the N most similar strategies."""
        if self.strategies.count() == 0:
            return []
        embedding = self.embedding_engine.embed_single(text).tolist()
        results = self.strategies.query(
            query_embeddings=[embedding], n_results=min(n, self.strategies.count()),
        )
        out = []
        for i, sid in enumerate(results["ids"][0]):
            sim = 1.0 - results["distances"][0][i]
            out.append({
                "strategy_id": int(sid),
                "similarity": sim,
                "text": results["documents"][0][i] if results["documents"] else "",
            })
        return out

    # ── History Memory ────────────────────────────────────────────
    def add_history(self, record_id: int, question: str, strategy_id: int,
                    metadata: dict | None = None):
        """Store a successful attack in history memory."""
        embedding = self.embedding_engine.embed_single(question).tolist()
        meta = metadata or {}
        meta["strategy_id"] = strategy_id
        self.history_memory.upsert(
            ids=[str(record_id)],
            embeddings=[embedding],
            documents=[question],
            metadatas=[meta],
        )

    def find_similar_history(self, question: str, n: int = 5,
                             threshold: float = HISTORY_MATCH_THRESHOLD) -> list[dict]:
        """Find similar past successful attacks above threshold."""
        if self.history_memory.count() == 0:
            return []
        embedding = self.embedding_engine.embed_single(question).tolist()
        results = self.history_memory.query(
            query_embeddings=[embedding],
            n_results=min(n, self.history_memory.count()),
        )
        out = []
        for i, rid in enumerate(results["ids"][0]):
            sim = 1.0 - results["distances"][0][i]
            if sim >= threshold:
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                out.append({
                    "record_id": int(rid),
                    "question": results["documents"][0][i] if results["documents"] else "",
                    "similarity": sim,
                    "strategy_id": meta.get("strategy_id"),
                    "metadata": meta,
                })
        return out

    def remove_strategy(self, strategy_id: int):
        """Remove a strategy from the vector store."""
        try:
            self.strategies.delete(ids=[str(strategy_id)])
        except Exception:
            pass
