"""
vectorstore/faiss_store.py — FAISS-based vector store with persistence.

Stores chunk embeddings in a flat inner-product index (equivalent to
cosine similarity when embeddings are L2-normalised).

Persistence:
  - FAISS index  → <FAISS_INDEX_PATH>
  - Chunk metadata → <METADATA_PATH>  (JSON)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from config import cfg
from data.chunker import TextChunk
from vectorstore.embeddings import EmbeddingModel


# ── serialisation helper ────────────────────────────────────────────────────

def _chunk_to_dict(chunk: TextChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_estimate": chunk.token_estimate,
    }


def _dict_to_chunk(d: dict) -> TextChunk:
    c = TextChunk(
        chunk_id=d["chunk_id"],
        filename=d["filename"],
        page_number=d["page_number"],
        chunk_index=d["chunk_index"],
        text=d["text"],
    )
    return c


# ── main class ──────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Manages a FAISS IndexFlatIP (inner-product / cosine) index
    alongside a list of TextChunk objects.

    Usage
    -----
    store = FAISSVectorStore(embedding_model)
    store.add_chunks(chunks)          # build index
    results = store.search(query, k=5)
    store.save()                      # persist to disk
    store.load()                      # restore from disk
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[TextChunk] = []

    # ── index management ────────────────────────────────────────────────────

    def _init_index(self) -> None:
        dim = self.embedding_model.dimension
        self._index = faiss.IndexFlatIP(dim)

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Embed and add chunks to the FAISS index."""
        if not chunks:
            return

        if self._index is None:
            self._init_index()

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode(texts)

        self._index.add(embeddings)
        self._chunks.extend(chunks)

    def reset(self) -> None:
        """Clear the index and stored chunks."""
        self._index = None
        self._chunks = []

    # ── search ──────────────────────────────────────────────────────────────

    def search(
        self, query: str, k: int = cfg.TOP_K
    ) -> List[Tuple[TextChunk, float]]:
        """
        Return the top-k most similar chunks for *query*.

        Returns
        -------
        List of (TextChunk, score) sorted by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        query_emb = self.embedding_model.encode_single(query).reshape(1, -1)
        scores, indices = self._index.search(query_emb, k)

        results: List[Tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    # ── persistence ─────────────────────────────────────────────────────────

    def save(
        self,
        index_path: str = cfg.FAISS_INDEX_PATH,
        metadata_path: str = cfg.METADATA_PATH,
    ) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        if self._index is None:
            return

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self._index, index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([_chunk_to_dict(c) for c in self._chunks], f, ensure_ascii=False, indent=2)

    def load(
        self,
        index_path: str = cfg.FAISS_INDEX_PATH,
        metadata_path: str = cfg.METADATA_PATH,
    ) -> bool:
        """
        Load a previously saved FAISS index and metadata.

        Returns True if loading succeeded, False if files not found.
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return False

        self._index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._chunks = [_dict_to_chunk(d) for d in raw]
        return True

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def get_all_filenames(self) -> List[str]:
        return sorted(set(c.filename for c in self._chunks))
