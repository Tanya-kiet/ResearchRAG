"""
llm/openai_llm.py — OpenAI ChatCompletion backend.
"""

from __future__ import annotations

from openai import OpenAI

from config import cfg
from llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    Backend that calls the OpenAI ChatCompletion API.
    """

    def __init__(
        self,
        api_key: str = cfg.OPENAI_API_KEY,
        model: str = cfg.OPENAI_MODEL,
    ):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model

    # ── BaseLLM interface ────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,   # low temperature for deterministic, grounded answers
        )
        return response.choices[0].message.content.strip()

    @property
    def model_name(self) -> str:
        return f"OpenAI / {self._model}"
