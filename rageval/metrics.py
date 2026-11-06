"""Retrieval, correctness, and refusal metrics."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int) -> float:
    """Share of gold documents that appear in the top-k retrieved chunks."""
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("recall@k needs at least one gold document")
    return len(gold & set(retrieved_doc_ids[:k])) / len(gold)


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> float:
    gold = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def normalize(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = re.sub(r"[^a-z0-9.%$]+", " ", text)
    text = re.sub(r"\.(?=\s|$)", " ", text)
    return " ".join(text.split())


def fact_matches(answer: str, facts: Sequence[str]) -> list[bool]:
    padded = f" {normalize(answer)} "
    return [f" {normalize(fact)} " in padded for fact in facts]


def fact_recall(answer: str, facts: Sequence[str]) -> float:
    if not facts:
        raise ValueError("fact recall needs at least one expected fact")
    matches = fact_matches(answer, facts)
    return sum(matches) / len(matches)


def refusal_correct(refused: bool, answerable: bool) -> bool:
    return refused != answerable


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear interpolation between closest ranks."""
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
