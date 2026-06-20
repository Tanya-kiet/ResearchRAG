"""
utils/text_cleaning.py — Text normalisation utilities.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and newlines."""
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def remove_non_printable(text: str) -> str:
    """Remove non-printable / control characters."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == "\n")


def clean_pdf_text(text: str) -> str:
    """
    Apply a sensible cleaning pipeline for raw PDF-extracted text:
      1. Remove non-printable characters
      2. Normalize whitespace
      3. Remove hyphenation at line breaks (common in PDFs)
    """
    text = remove_non_printable(text)
    text = re.sub(r'-\n(\w)', r'\1', text)   # de-hyphenate line breaks
    text = normalize_whitespace(text)
    return text


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to *max_chars*, appending '…' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
