"""Singleton model manager — central access to Gemma, BGE, and API clients."""
from __future__ import annotations

import threading
from config.settings import (
    API_SECRET_KEY, API_BASE_URL, GEMMA_MODEL_PATH, BGE_MODEL_PATH,
    DEFAULT_TARGET_MODEL, DEFAULT_JUDGE_MODEL,
)
from models.api_model import APIModel


class LLMManager:
    """Thread-safe singleton that lazily loads all models."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # Lazy-loaded model holders
        self._gemma: LocalModel | None = None
        self._api: APIModel | None = None
        self._bge_model = None
        self._bge_tokenizer = None

    # ── Gemma (local attacker / optimizer) ────────────────────────
    @property
    def gemma(self):
        if self._gemma is None:
            from models.local_model import LocalModel
            self._gemma = LocalModel(GEMMA_MODEL_PATH, max_new_tokens=512)
        return self._gemma

    # ── API client (GPT-4o for judge / target / sandbox) ──────────
    @property
    def api(self) -> APIModel:
        if self._api is None:
            self._api = APIModel(API_SECRET_KEY, API_BASE_URL, DEFAULT_TARGET_MODEL)
        return self._api

    # ── BGE embedding model ───────────────────────────────────────
    @property
    def bge_tokenizer(self):
        self._load_bge()
        return self._bge_tokenizer

    @property
    def bge_model(self):
        self._load_bge()
        return self._bge_model

    def _load_bge(self):
        if self._bge_model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        print(f"[LLMManager] Loading BGE from {BGE_MODEL_PATH} ...")
        self._bge_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
        self._bge_model = AutoModel.from_pretrained(BGE_MODEL_PATH)
        self._bge_model.eval()
        if torch.cuda.is_available():
            self._bge_model = self._bge_model.cuda()
        print("[LLMManager] BGE loaded.")

    # ── Convenience methods ───────────────────────────────────────
    def gemma_generate(self, prompt: str) -> str:
        """Quick single-turn Gemma generation."""
        messages = [{"role": "user", "content": prompt}]
        return self.gemma.generate(messages)

    def api_generate(self, prompt: str, model: str | None = None, **kwargs) -> str:
        """Quick single-turn API generation."""
        messages = [{"role": "user", "content": prompt}]
        return self.api.generate(messages, model=model, **kwargs)

    def api_chat(self, messages: list[dict], model: str | None = None, **kwargs) -> str:
        """Multi-turn API generation."""
        return self.api.generate(messages, model=model, **kwargs)
