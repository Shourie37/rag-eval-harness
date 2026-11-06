import math

import pytest

from rageval.cost import cost_usd, estimate_tokens
from rageval.metrics import (
    fact_matches,
    fact_recall,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
    refusal_correct,
)


def test_recall_at_k_counts_gold_docs_in_top_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["c", "z"], k=4) == 0.5
    assert recall_at_k(retrieved, ["c"], k=2) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(retrieved, [], k=2)


def test_reciprocal_rank_uses_first_gold_hit() -> None:
    assert reciprocal_rank(["x", "a", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_fact_match_normalizes_case_commas_and_punctuation() -> None:
    answer = "Each instant payout can be at most $10,000. [payouts-instant#c0]"
    assert fact_matches(answer, ["$10,000", "$10000", "at most"]) == [True, True, True]


def test_fact_match_needs_whole_tokens() -> None:
    assert fact_matches("The fee is $150.", ["$15"]) == [False]
    assert fact_recall("2.9% + 30 cents", ["2.9%", "30 cents", "1.5%"]) == pytest.approx(2 / 3)


def test_refusal_correct_truth_table() -> None:
    assert refusal_correct(refused=True, answerable=False)
    assert refusal_correct(refused=False, answerable=True)
    assert not refusal_correct(refused=True, answerable=True)
    assert not refusal_correct(refused=False, answerable=False)


def test_percentile_and_mean() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert math.isnan(percentile([], 50))
    assert mean([1.0, 2.0]) == 1.5


def test_cost_estimate() -> None:
    assert estimate_tokens("abcdefgh") == 2
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)
    with pytest.raises(KeyError):
        cost_usd("unknown-model", 1, 1)
