"""
data/pdf_loader.py — Load and extract raw text from PDF files.
Uses PyMuPDF (fitz) for reliable, fast text extraction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Raw text content from a single PDF page."""
    filename: str
    page_number: int
    text: str


@dataclass
class DocumentContent:
    """Aggregated content from a single PDF file."""
    filename: str
    filepath: str
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PDFLoader:
    """
    Loads one or more PDF files and extracts text page-by-page.
    """

    @staticmethod
    def load_file(filepath: str) -> DocumentContent:
        """Extract text from a single PDF file."""
        filename = os.path.basename(filepath)
        doc_content = DocumentContent(filename=filename, filepath=filepath)

        try:
            pdf = fitz.open(filepath)
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                text = page.get_text("text")
                doc_content.pages.append(
                    PageContent(
                        filename=filename,
                        page_number=page_idx + 1,
                        text=text,
                    )
                )
            pdf.close()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF '{filepath}': {e}") from e

        return doc_content

    @staticmethod
    def load_files(filepaths: List[str]) -> List[DocumentContent]:
        """Extract text from multiple PDF files."""
        documents = []
        for fp in filepaths:
            documents.append(PDFLoader.load_file(fp))
        return documents

    @staticmethod
    def load_directory(directory: str) -> List[DocumentContent]:
        """Load all PDFs from a directory."""
        pdf_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".pdf")
        ]
        return PDFLoader.load_files(pdf_files)
