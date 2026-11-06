"""Text helpers shared by retrieval, generation, and grounding."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a an the and or but if of to in on for with by at from as is are was were be been being
    it its this that these those i you we they he she my your our their me us do does did
    can could will would should may might must not no so than then there here what which who
    whom whose when where why how about into over under up down out off per each any all both
    more most other some such only own same too very just also have has had having get
    """.split()
)

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _stem(term: str) -> str:
    # Plural folding only; enough to match "refunds" with "refund".
    if term[0].isdigit():
        return term
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return term


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase, split, drop stopwords, and fold plurals."""
    text = _THOUSANDS.sub("", text.lower())
    terms = []
    for raw in _TOKEN.findall(text):
        if not keep_stopwords and raw in STOPWORDS:
            continue
        if len(raw) == 1 and not raw.isdigit():
            continue
        terms.append(_stem(raw))
    return terms


def numbers_in(text: str) -> set[str]:
    """Return the numeric tokens in the text, with thousands separators removed."""
    return {t for t in tokenize(text, keep_stopwords=True) if t[0].isdigit()}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
