"""
rag/generator.py — LLM-based answer generator for both modes.
"""

from __future__ import annotations

from llm.base import BaseLLM
from rag.prompt_templates import build_baseline_prompt, build_rag_prompt


class Generator:
    """
    Wraps a BaseLLM instance and exposes generate methods for
    Baseline and RAG modes.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def baseline_answer(self, question: str, max_tokens: int = 512) -> str:
        """
        Generate an answer using only the LLM's parametric knowledge.
        No external context is provided.
        """
        prompt = build_baseline_prompt(question)
        return self.llm.generate(prompt, max_tokens=max_tokens)

    def rag_answer(
        self, question: str, context: str, max_tokens: int = 512
    ) -> str:
        """
        Generate a grounded answer using retrieved context.
        The prompt strictly instructs the model to stay within the context.
        """
        prompt = build_rag_prompt(question, context)
        return self.llm.generate(prompt, max_tokens=max_tokens)
