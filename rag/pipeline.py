"""
rag/pipeline.py — End-to-end RAG pipeline orchestrator.

Ties together:
  PDFLoader → Chunker → EmbeddingModel → FAISSVectorStore
                                    ↓
                              Retriever → Generator → Answer
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

from config import cfg
from data.chunker import Chunker, TextChunk
from data.pdf_loader import PDFLoader
from llm.base import BaseLLM
from rag.generator import Generator
from rag.retriever import RetrievedChunk, Retriever
from vectorstore.embeddings import EmbeddingModel
from vectorstore.faiss_store import FAISSVectorStore


# ── result containers ────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    question: str
    baseline_answer: str
    rag_answer: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    context_used: str = ""


# ── pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    High-level API consumed by the Streamlit UI.

    Lifecycle
    ---------
    1. Instantiate with an LLM.
    2. Call `ingest_pdfs(filepaths)` to build/update the index.
    3. Call `query(question)` to get a QueryResult.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.embedding_model = EmbeddingModel.get_instance()
        self.vector_store = FAISSVectorStore(self.embedding_model)
        self.retriever = Retriever(self.vector_store, top_k=cfg.TOP_K)
        self.generator = Generator(llm)
        self.chunker = Chunker(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
        )
        self._loaded = False

    # ── ingestion ────────────────────────────────────────────────────────────

    def ingest_pdfs(
        self,
        filepaths: List[str],
        save_to_disk: bool = True,
        progress_callback=None,
    ) -> int:
        """
        Load PDFs, chunk, embed, and index all documents.
        Replaces any existing index.

        Parameters
        ----------
        filepaths        : list of absolute paths to PDF files.
        save_to_disk     : persist FAISS index after building.
        progress_callback: optional callable(step: int, total: int, msg: str)

        Returns
        -------
        int — total number of chunks indexed.
        """
        total_steps = len(filepaths)
        self.vector_store.reset()

        all_chunks: List[TextChunk] = []

        for i, fp in enumerate(filepaths):
            if progress_callback:
                progress_callback(i, total_steps, f"Loading: {os.path.basename(fp)}")

            # Save a copy to the documents directory
            dest = os.path.join(cfg.DOCUMENTS_DIR, os.path.basename(fp))
            os.makedirs(cfg.DOCUMENTS_DIR, exist_ok=True)
            if os.path.abspath(fp) != os.path.abspath(dest):
                shutil.copy2(fp, dest)

            doc = PDFLoader.load_file(fp)
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if progress_callback:
            progress_callback(total_steps, total_steps, "Building FAISS index…")

        self.vector_store.add_chunks(all_chunks)

        if save_to_disk:
            self.vector_store.save()

        self._loaded = True
        return len(all_chunks)

    def try_load_from_disk(self) -> bool:
        """Attempt to restore a previously saved FAISS index."""
        success = self.vector_store.load()
        if success:
            self._loaded = True
        return success

    # ── querying ─────────────────────────────────────────────────────────────

    def query(self, question: str, max_tokens: int = 512) -> QueryResult:
        """
        Run both Baseline and RAG generation for *question*.

        Returns a QueryResult with both answers and the retrieved context.
        """
        # 1. Baseline: no context
        baseline_ans = self.generator.baseline_answer(question, max_tokens=max_tokens)

        # 2. RAG: retrieve → build context → generate
        retrieved = self.retriever.retrieve(question)
        context = self._build_context(retrieved)
        rag_ans = self.generator.rag_answer(question, context, max_tokens=max_tokens)

        return QueryResult(
            question=question,
            baseline_answer=baseline_ans,
            rag_answer=rag_ans,
            retrieved_chunks=retrieved,
            context_used=context,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_context(retrieved: List[RetrievedChunk], max_chars: int = 6000) -> str:
        """
        Combine retrieved chunk texts into a single context string.
        Caps total length to avoid exceeding LLM context windows.
        """
        parts: List[str] = []
        total_chars = 0

        for rc in retrieved:
            header = f"[Source: {rc.source_label}]\n"
            body = rc.chunk.text.strip()
            block = header + body + "\n"

            if total_chars + len(block) > max_chars:
                break

            parts.append(block)
            total_chars += len(block)

        return "\n---\n".join(parts)

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._loaded and self.vector_store.is_ready

    @property
    def indexed_files(self) -> List[str]:
        return self.vector_store.get_all_filenames()

    @property
    def total_chunks(self) -> int:
        return self.vector_store.total_chunks
