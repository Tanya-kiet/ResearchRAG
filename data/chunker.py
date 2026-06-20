"""
data/chunker.py — Split document text into overlapping chunks.

Strategy:
  - Split on sentence/paragraph boundaries where possible.
  - Target 500–800 tokens per chunk (approximated as characters / 4).
  - Overlap of 10–20 % between consecutive chunks.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from data.pdf_loader import DocumentContent


@dataclass
class TextChunk:
    """A single chunk of text with its provenance metadata."""
    chunk_id: str
    filename: str
    page_number: int        # page where the chunk starts
    chunk_index: int        # position within the document
    text: str
    token_estimate: int = field(init=False)

    def __post_init__(self):
        # Rough approximation: 1 token ≈ 4 characters
        self.token_estimate = max(1, len(self.text) // 4)


class Chunker:
    """
    Splits DocumentContent objects into fixed-size, overlapping TextChunks.

    Parameters
    ----------
    chunk_size   : target chunk size in tokens (approx)
    chunk_overlap: overlap between consecutive chunks in tokens (approx)
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Characters corresponding to target token counts
        self._char_size = chunk_size * 4
        self._char_overlap = chunk_overlap * 4

    # ── public API ──────────────────────────────────────────────────────────

    def chunk_document(self, document: DocumentContent) -> List[TextChunk]:
        """Chunk a single DocumentContent into TextChunk objects."""
        chunks: List[TextChunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            page_chunks = self._split_text(page.text)
            for text in page_chunks:
                if not text.strip():
                    continue
                chunks.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        filename=document.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=text.strip(),
                    )
                )
                chunk_index += 1

        return chunks

    def chunk_documents(self, documents: List[DocumentContent]) -> List[TextChunk]:
        """Chunk a list of DocumentContent objects."""
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ── private helpers ─────────────────────────────────────────────────────

    def _split_text(self, text: str) -> List[str]:
        """
        Split *text* into overlapping character windows.
        Tries to break on sentence boundaries within the window.
        """
        text = self._clean(text)
        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self._char_size, length)

            # Try to snap the end to a sentence boundary
            if end < length:
                snap = self._find_sentence_boundary(text, end)
                if snap > start:
                    end = snap

            chunk = text[start:end]
            chunks.append(chunk)

            # Advance by (chunk_size - overlap)
            advance = max(1, self._char_size - self._char_overlap)
            start += advance

        return chunks

    @staticmethod
    def _find_sentence_boundary(text: str, near: int) -> int:
        """
        Search backwards from *near* for the last sentence-ending punctuation
        within a 200-character window.
        """
        window_start = max(0, near - 200)
        segment = text[window_start:near]
        # Find the last occurrence of '.', '!', or '?' followed by whitespace
        matches = list(re.finditer(r'[.!?]\s', segment))
        if matches:
            last_match = matches[-1]
            return window_start + last_match.end()
        return near  # fall back to hard cut

    @staticmethod
    def _clean(text: str) -> str:
        """Light normalisation: collapse excessive whitespace / newlines."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
