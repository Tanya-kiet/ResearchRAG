"""
llm/claude_llm.py — Anthropic Claude backend.
"""

from __future__ import annotations

import anthropic

from config import cfg
from llm.base import BaseLLM


class ClaudeLLM(BaseLLM):
    """
    Backend that calls the Anthropic Messages API.
    """

    def __init__(
        self,
        api_key: str = cfg.ANTHROPIC_API_KEY,
        model: str = cfg.CLAUDE_MODEL,
    ):
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    # ── BaseLLM interface ────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    @property
    def model_name(self) -> str:
        return f"Anthropic / {self._model}"
