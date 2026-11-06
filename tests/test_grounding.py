from rageval.grounding import check_grounding, split_answer_sentences

RETRIEVED = {
    "disputes#c0": "Acme charges a $15 dispute fee for each dispute. The dispute fee is returned to you if you win.",
    "support#c0": "Phone support is available 24/7 for Enterprise accounts only.",
}


def test_split_keeps_citation_with_its_sentence() -> None:
    text = "First claim. [a#c0] Second claim [b#c1]. Third claim."
    assert split_answer_sentences(text) == ["First claim. [a#c0]", "Second claim [b#c1].", "Third claim."]


def test_verbatim_cited_sentence_is_supported() -> None:
    result = check_grounding("Acme charges a $15 dispute fee for each dispute. [disputes#c0]", RETRIEVED)
    assert result.fully_grounded
    assert result.total == 1


def test_changed_number_is_flagged() -> None:
    result = check_grounding("Acme charges a $16 dispute fee for each dispute. [disputes#c0]", RETRIEVED)
    assert not result.fully_grounded
    assert "16" in result.unsupported[0].reason


def test_claim_not_in_cited_chunk_is_flagged() -> None:
    answer = (
        "Acme charges a $15 dispute fee for each dispute. [disputes#c0] "
        "Acme waives this fee for registered nonprofit organizations. [disputes#c0]"
    )
    result = check_grounding(answer, RETRIEVED)
    assert result.supported == 1
    assert result.unsupported[0].reason.startswith("low term support")


def test_wrong_chunk_citation_is_flagged() -> None:
    result = check_grounding("Phone support is available 24/7 for Enterprise accounts only. [disputes#c0]", RETRIEVED)
    assert not result.fully_grounded


def test_missing_and_unknown_citations_are_flagged() -> None:
    result = check_grounding("Phone support is for Enterprise. Phone support exists. [ghost#c9]", RETRIEVED)
    assert [v.reason.split(":")[0] for v in result.unsupported] == ["no citation", "cites chunk not retrieved"]
