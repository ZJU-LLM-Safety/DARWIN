"""BGE-small-en-v1.5 embedding wrapper."""
import torch
import numpy as np
from models.llm_manager import LLMManager


class EmbeddingEngine:
    """Compute embeddings using the BGE model via LLMManager."""

    def __init__(self):
        self._mgr = LLMManager()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings of shape (N, dim)."""
        tokenizer = self._mgr.bge_tokenizer
        model = self._mgr.bge_model
        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        # CLS pooling
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, return shape (dim,)."""
        return self.embed([text])[0]
