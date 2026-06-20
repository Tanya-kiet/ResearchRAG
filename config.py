"""
config.py — Central configuration loader for ResearchRAG.
Reads from .env (via python-dotenv) with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ─────────────────────────────────────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # ── Embeddings ──────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Retrieval ───────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", 5))

    # ── Chunking ────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 700))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

    # ── Storage paths ───────────────────────────────────────
    FAISS_INDEX_PATH: str = os.getenv(
        "FAISS_INDEX_PATH", "storage/faiss_index/index.faiss"
    )
    METADATA_PATH: str = os.getenv(
        "METADATA_PATH", "storage/faiss_index/metadata.json"
    )
    EVALUATION_LOG_PATH: str = os.getenv(
        "EVALUATION_LOG_PATH", "storage/evaluation_log.csv"
    )
    DOCUMENTS_DIR: str = "storage/documents"


cfg = Config()
