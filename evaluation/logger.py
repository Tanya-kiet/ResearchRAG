"""
evaluation/logger.py — Persist evaluation scores to CSV and session log.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import List

from config import cfg
from evaluation.metrics import EvaluationScore


class EvaluationLogger:
    """
    Appends EvaluationScore records to a CSV file.
    Provides helpers to load the full log and compute summary statistics.
    """

    HEADERS = [
        "timestamp", "question", "mode", "answer_preview",
        "relevance", "completeness", "clarity", "source_coverage",
        "average", "notes",
    ]

    def __init__(self, log_path: str = cfg.EVALUATION_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._ensure_header()

    # ── writing ──────────────────────────────────────────────────────────────

    def log(self, score: EvaluationScore) -> None:
        """Append a single EvaluationScore row to the CSV."""
        row = {"timestamp": datetime.utcnow().isoformat()} | score.to_dict()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(row)

    # ── reading ──────────────────────────────────────────────────────────────

    def load_all(self) -> List[dict]:
        """Return all logged records as a list of dicts."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def summary(self) -> dict:
        """Compute mean scores per mode (baseline vs rag)."""
        records = self.load_all()
        summary: dict = {}
        for mode in ("baseline", "rag"):
            subset = [r for r in records if r.get("mode") == mode]
            if not subset:
                continue
            for metric in ("relevance", "completeness", "clarity", "source_coverage", "average"):
                values = []
                for r in subset:
                    try:
                        values.append(float(r[metric]))
                    except (ValueError, KeyError):
                        pass
                mean = round(sum(values) / len(values), 2) if values else 0.0
                summary.setdefault(mode, {})[metric] = mean
        return summary

    # ── private ──────────────────────────────────────────────────────────────

    def _ensure_header(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writeheader()
