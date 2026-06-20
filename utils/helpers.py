"""
utils/helpers.py — Miscellaneous helpers.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import streamlit as st


def save_uploaded_files(uploaded_files) -> List[str]:
    """
    Save Streamlit UploadedFile objects to a temp directory and
    return a list of absolute file paths.
    """
    saved_paths: List[str] = []
    tmp_dir = tempfile.mkdtemp(prefix="researchrag_")

    for uf in uploaded_files:
        dest = os.path.join(tmp_dir, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved_paths.append(dest)

    return saved_paths


def format_score_badge(score: float) -> str:
    """Return a coloured emoji badge for a 1–10 score."""
    if score >= 8:
        return "🟢"
    if score >= 5:
        return "🟡"
    return "🔴"


def get_llm_from_config():
    """
    Instantiate the correct LLM based on the LLM_PROVIDER config value.
    Returns a BaseLLM instance or None if credentials are missing.
    """
    from config import cfg

    provider = cfg.LLM_PROVIDER.lower()

    try:
        if provider == "openai":
            from llm.openai_llm import OpenAILLM
            return OpenAILLM()
        elif provider == "claude":
            from llm.claude_llm import ClaudeLLM
            return ClaudeLLM()
        elif provider == "groq":
            from llm.groq_llm import GroqLLM
            return GroqLLM()
        else:
            st.error(f"Unknown LLM_PROVIDER: '{provider}'. Use 'openai', 'claude', or 'groq'.")
            return None
    except ValueError as exc:
        st.error(str(exc))
        return None
