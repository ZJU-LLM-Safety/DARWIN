from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .config import EmbeddingConfig


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        pass


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install DARWIN with the 'local' extra") from exc
        self.model = SentenceTransformer(config.model, device=config.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0
