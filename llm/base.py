"""
llm/base.py — Abstract base class for all LLM backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Minimal interface every LLM backend must implement.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Send *prompt* to the language model and return the text response.

        Parameters
        ----------
        prompt     : Full prompt string (system + user combined, or just user).
        max_tokens : Maximum number of tokens in the completion.

        Returns
        -------
        str — The model's text response.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a human-readable identifier for this model."""
        ...
