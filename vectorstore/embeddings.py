"""
vectorstore/embeddings.py — Compute and cache sentence embeddings.

Uses sentence-transformers (local, no API key required) for fast,
reproducible embeddings. The model is loaded once and reused.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import cfg


class EmbeddingModel:
    """
    Wraps a SentenceTransformer model.
    The underlying model is loaded lazily on first use and cached.
    """

    _instance: "EmbeddingModel | None" = None

    def __init__(self, model_name: str = cfg.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    # ── singleton helper ────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls(model_name=cfg.EMBEDDING_MODEL)
        return cls._instance

    # ── public API ──────────────────────────────────────────────────────────
    def load(self) -> None:
        """Explicitly load the model (called once at startup)."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised float32 embeddings.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), dtype float32
        """
        if self._model is None:
            self.load()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-norm → cosine = dot product
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single string."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        if self._model is None:
            self.load()
        return self._model.get_sentence_embedding_dimension()
