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
using the context provided below from academic research papers.

Instructions:
- Base your answer on the provided context. Do not introduce facts, figures,
  or claims from outside the context.
- The retrieved context may only partially cover the question, or may be
  imperfectly matched — that is expected. Synthesize the best possible
  answer from whatever relevant information IS present, even if it only
  partially addresses the question. If your answer is partial, say so
  explicitly rather than refusing outright.
- Only respond with exactly "Information not available in the provided \
  documents." if the context contains nothing relevant to the question at \
  all — not merely an incomplete answer.
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
