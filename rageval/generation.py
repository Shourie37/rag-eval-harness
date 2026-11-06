"""Answer generators: an offline extractive mock, a fault-injecting mock, and an LLM wrapper."""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rageval.corpus import Chunk
from rageval.cost import estimate_tokens
from rageval.llm import ChatModel
from rageval.text import split_sentences, tokenize

REFUSAL = "I don't know based on the provided documents."
_REFUSAL_MARKERS = ("i don't know", "i do not know", "not in the provided documents", "cannot answer")
CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+#c\d+)\]")

SYSTEM_PROMPT = f"""You answer questions about Acme Payments.
Use only the numbered context chunks in the user message.
Write one to three short sentences.
End each sentence with the id of the chunk that supports it, in square brackets, for example [refunds-policy#c0].
If the context does not contain the answer, reply exactly: {REFUSAL}"""


@dataclass(frozen=True)
class Answer:
    text: str
    input_tokens: int
    output_tokens: int
    usage_estimated: bool


class Generator(Protocol):
    name: str

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer: ...


def build_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    context = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def is_refusal(text: str) -> bool:
    lowered = text.strip().lower().replace("’", "'")
    return not lowered or any(marker in lowered for marker in _REFUSAL_MARKERS)


def citations(text: str) -> list[str]:
    return CITATION_RE.findall(text)


def _estimated(question: str, chunks: Sequence[Chunk], text: str) -> Answer:
    prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(build_prompt(question, chunks))
    return Answer(text, prompt_tokens, estimate_tokens(text), usage_estimated=True)


class ExtractiveMockGenerator:
    """Deterministic offline 'LLM': returns the retrieved sentences that best cover the question."""

    name = "mock-extractive"

    def __init__(self, idf: Callable[[str], float], min_coverage: float = 0.4, max_sentences: int = 2) -> None:
        self._idf = idf
        self.min_coverage = min_coverage
        self.max_sentences = max_sentences

    def coverage(self, question: str, sentence: str) -> float:
        """Share of the question's idf mass that the sentence contains."""
        q_terms = set(tokenize(question))
        total = sum(self._idf(t) for t in q_terms)
        if not total:
            return 0.0
        present = set(tokenize(sentence))
        return sum(self._idf(t) for t in q_terms if t in present) / total

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        candidates: list[tuple[float, int, str, str]] = []
        for rank, chunk in enumerate(chunks):
            for sentence in split_sentences(chunk.text):
                # Short fragments are usually headings or chunk-boundary debris.
                if len(sentence.split()) < 5:
                    continue
                candidates.append((self.coverage(question, sentence), rank, sentence, chunk.chunk_id))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        if not candidates or candidates[0][0] < self.min_coverage:
            return _estimated(question, chunks, REFUSAL)
        best = candidates[0][0]
        picked: list[tuple[str, str]] = []
        for score, _, sentence, chunk_id in candidates:
            if len(picked) == self.max_sentences or score < 0.75 * best:
                break
            if sentence not in {s for s, _ in picked}:
                picked.append((sentence, chunk_id))
        text = " ".join(f"{sentence} [{chunk_id}]" for sentence, chunk_id in picked)
        return _estimated(question, chunks, text)


_FABRICATED = "Acme waives this fee for registered nonprofit organizations."


class HallucinatingMockGenerator(ExtractiveMockGenerator):
    """Fault injection: corrupts a fixed share of answers so the grounding check has something to catch."""

    name = "mock-hallucinating"

    def __init__(self, idf: Callable[[str], float], rate: float = 0.3, **kwargs: float) -> None:
        super().__init__(idf, **kwargs)
        self.rate = rate

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        clean = super().answer(question, chunks)
        if is_refusal(clean.text) or (zlib.crc32(question.encode()) % 100) >= self.rate * 100:
            return clean
        text = clean.text
        number = re.search(r"\b\d+", text)
        if number and zlib.crc32(question.encode()) % 2 == 0:
            text = text[: number.start()] + str(int(number.group()) + 1) + text[number.end() :]
        else:
            text = f"{text} {_FABRICATED} [{citations(text)[0]}]"
        return _estimated(question, chunks, text)


class LLMGenerator:
    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self.name = model.name

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        completion = self._model.complete(SYSTEM_PROMPT, build_prompt(question, chunks))
        text = REFUSAL if completion.refused_by_provider else completion.text.strip()
        return Answer(text, completion.input_tokens, completion.output_tokens, usage_estimated=False)
