import math

import pytest

from rageval.bm25 import BM25Index


def test_score_matches_hand_computation() -> None:
    index = BM25Index(["refund fee", "payout schedule weekly"], k1=1.5, b=0.75)
    # "refund": df=1, N=2, doc length 2, avgdl 2.5.
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1 - 0.75 + 0.75 * 2 / 2.5
    expected = idf * 1 * 2.5 / (1 + 1.5 * norm)
    assert index.scores("refund")[0] == pytest.approx(expected)


def test_ranking_prefers_matching_document() -> None:
    index = BM25Index(["card pricing fee", "webhook retry", "refund policy refund fee"])
    ranked = index.search("refund fee", k=3)
    assert ranked[0][0] == 2
    assert {i for i, _ in ranked} == {0, 2}


def test_unknown_term_scores_nothing_and_has_max_idf() -> None:
    index = BM25Index(["alpha beta", "beta gamma"])
    assert index.scores("zeta") == {}
    assert index.idf("zeta") > index.idf("alpha") > index.idf("beta")


def test_stopwords_and_plurals_are_normalized() -> None:
    index = BM25Index(["the refunds are issued"])
    assert index.search("refund", k=1)[0][0] == 0
    assert index.scores("the are") == {}
