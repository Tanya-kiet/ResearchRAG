"""
rag/prompt_templates.py — Prompt templates for Baseline and RAG modes.
"""

from __future__ import annotations


# ── Baseline prompt ──────────────────────────────────────────────────────────

BASELINE_TEMPLATE = """\
You are an expert in AI, Machine Learning, Deep Learning, NLP, and Computer Vision.
Answer the following academic question clearly and concisely.

Question: {question}

Answer:"""


# ── RAG prompt ───────────────────────────────────────────────────────────────

RAG_TEMPLATE = """\
You are an expert academic assistant. Your task is to answer a question \
using ONLY the context provided below from academic research papers.

Instructions:
- Use only information from the provided context.
- Do NOT use any external or prior knowledge.
- If the answer is not present in the context, respond with exactly:
  "Information not available in the provided documents."
- Be concise and precise.
- Cite the source filename when possible.

---
CONTEXT:
{context}
---

Question: {question}

Answer:"""


def build_baseline_prompt(question: str) -> str:
    """Return a formatted baseline prompt."""
    return BASELINE_TEMPLATE.format(question=question.strip())


def build_rag_prompt(question: str, context: str) -> str:
    """Return a formatted RAG prompt with grounded context."""
    return RAG_TEMPLATE.format(
        context=context.strip(),
        question=question.strip(),
    )
