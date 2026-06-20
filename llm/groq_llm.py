"""
llm/groq_llm.py — Groq API backend.

Groq provides ultra-fast LLM inference via an OpenAI-compatible API.
Supported models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma2-9b-it
"""

from __future__ import annotations

from groq import Groq

from config import cfg
from llm.base import BaseLLM


class GroqLLM(BaseLLM):
    """
    Backend that calls the Groq Chat Completions API.
    Groq uses an OpenAI-compatible interface with its own SDK.
    """

    def __init__(
        self,
        api_key: str = cfg.GROQ_API_KEY,
        model: str = cfg.GROQ_MODEL,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        self._client = Groq(api_key=api_key)
        self._model = model

    # ── BaseLLM interface ────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    @property
    def model_name(self) -> str:
        return f"Groq / {self._model}"
