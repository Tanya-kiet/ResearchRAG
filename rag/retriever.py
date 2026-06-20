"""
rag/retriever.py — Retrieval layer: query → top-K relevant chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from config import cfg
from data.chunker import TextChunk
from vectorstore.faiss_store import FAISSVectorStore


@dataclass
class RetrievedChunk:
    """A retrieved chunk paired with its similarity score."""
    chunk: TextChunk
    score: float

    @property
    def source_label(self) -> str:
        return f"{self.chunk.filename} (p.{self.chunk.page_number})"


class Retriever:
    """
    Retrieves the top-K most relevant TextChunks for a given query
    using cosine-similarity search on the FAISS index.
    """

    def __init__(self, vector_store: FAISSVectorStore, top_k: int = cfg.TOP_K):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Search the vector store for chunks most relevant to *query*.

        Returns
        -------
        List of RetrievedChunk sorted by descending similarity score.
        """
        if not self.vector_store.is_ready:
            return []

        raw_results: List[Tuple[TextChunk, float]] = self.vector_store.search(
            query, k=self.top_k
        )

        # Deduplicate by chunk_id (safety measure)
        seen_ids: set = set()
        results: List[RetrievedChunk] = []
        for chunk, score in raw_results:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                results.append(RetrievedChunk(chunk=chunk, score=score))

        return results
