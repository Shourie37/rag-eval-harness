"""Check that every answer sentence is supported by the chunk it cites."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from rageval.generation import CITATION_RE
from rageval.text import numbers_in, tokenize

# A citation ends a sentence; a period followed by a citation does not.
_ANSWER_SPLIT = re.compile(r"(?<=\])\s+|(?<=[.!?])\s+(?!\[)")


@dataclass(frozen=True)
class SentenceVerdict:
    sentence: str
    citations: tuple[str, ...]
    supported: bool
    support: float
    reason: str


@dataclass
class GroundingResult:
    verdicts: list[SentenceVerdict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def supported(self) -> int:
        return sum(v.supported for v in self.verdicts)

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [v for v in self.verdicts if not v.supported]

    @property
    def fully_grounded(self) -> bool:
        return not self.unsupported


def split_answer_sentences(text: str) -> list[str]:
    return [s.strip() for s in _ANSWER_SPLIT.split(text.strip()) if s.strip()]


def check_sentence(sentence: str, retrieved: Mapping[str, str], threshold: float = 0.75) -> SentenceVerdict:
    cited = tuple(dict.fromkeys(CITATION_RE.findall(sentence)))
    claim = CITATION_RE.sub("", sentence)
    if not cited:
        return SentenceVerdict(sentence, cited, False, 0.0, "no citation")
    missing = [c for c in cited if c not in retrieved]
    if missing:
        return SentenceVerdict(sentence, cited, False, 0.0, f"cites chunk not retrieved: {', '.join(missing)}")
    evidence = " ".join(retrieved[c] for c in cited)
    terms = set(tokenize(claim))
    if not terms:
        return SentenceVerdict(sentence, cited, True, 1.0, "no content terms")
    support = len(terms & set(tokenize(evidence))) / len(terms)
    # Numbers carry most of the risk in pricing answers, so each one must appear verbatim.
    bad_numbers = sorted(numbers_in(claim) - numbers_in(evidence))
    if bad_numbers:
        return SentenceVerdict(sentence, cited, False, support, f"numbers not in cited text: {', '.join(bad_numbers)}")
    if support < threshold:
        return SentenceVerdict(sentence, cited, False, support, f"low term support {support:.2f}")
    return SentenceVerdict(sentence, cited, True, support, "supported")


def check_grounding(answer: str, retrieved: Mapping[str, str], threshold: float = 0.75) -> GroundingResult:
    return GroundingResult([check_sentence(s, retrieved, threshold) for s in split_answer_sentences(answer)])
