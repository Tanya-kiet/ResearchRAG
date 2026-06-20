"""
evaluation/metrics.py — Manual scoring schema for ResearchRAG evaluation.

Metrics (1–10 scale):
  - Relevance     : How well does the answer address the question?
  - Completeness  : Does the answer cover all aspects of the question?
  - Clarity       : Is the answer easy to understand?
  - Source Coverage: Does the RAG answer correctly use the source material?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


METRIC_NAMES = ["Relevance", "Completeness", "Clarity", "Source Coverage"]
METRIC_DESCRIPTIONS = {
    "Relevance": "How well does the answer address the question? (1 = off-topic, 10 = perfectly on-topic)",
    "Completeness": "Does the answer cover all key aspects? (1 = missing critical info, 10 = fully complete)",
    "Clarity": "Is the answer easy to understand? (1 = confusing, 10 = crystal clear)",
    "Source Coverage": "Does the answer correctly use the source documents? (1 = ignores sources, 10 = excellent use)",
}


@dataclass
class EvaluationScore:
    """Holds manual scores for a single query-answer pair."""
    question: str
    mode: str                    # "baseline" or "rag"
    answer: str
    relevance: int = 0           # 1–10
    completeness: int = 0        # 1–10
    clarity: int = 0             # 1–10
    source_coverage: int = 0     # 1–10
    notes: str = ""

    @property
    def average_score(self) -> float:
        scores = [self.relevance, self.completeness, self.clarity, self.source_coverage]
        valid = [s for s in scores if s > 0]
        if not valid:
            return 0.0
        return round(sum(valid) / len(valid), 2)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "mode": self.mode,
            "answer_preview": self.answer[:200],
            "relevance": self.relevance,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "source_coverage": self.source_coverage,
            "average": self.average_score,
            "notes": self.notes,
        }
