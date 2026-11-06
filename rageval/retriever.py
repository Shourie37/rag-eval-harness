"""Top-k retrieval: BM25 or BM25 + hashing-vector hybrid, with an optional reranker."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from rageval.bm25 import BM25Index
from rageval.corpus import Chunk
from rageval.embedding import HashingEmbedder, cosine
from rageval.text import tokenize

MODES = ("bm25", "hybrid")


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "bm25"
    k: int = 4
    rerank: bool = False
    hybrid_alpha: float = 0.5
    candidates: int = 20


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def _minmax(values: Sequence[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rerank(query: str, hits: Sequence[Hit], idf: Callable[[str], float], weight: float = 0.5) -> list[Hit]:
    """Blend the first-stage score with idf-weighted query coverage and bigram overlap."""
    q_terms = list(dict.fromkeys(tokenize(query)))
    if not hits or not q_terms:
        return list(hits)
    q_bigrams = set(pairwise(q_terms))
    total_idf = sum(idf(t) for t in q_terms)
    first_stage = _minmax([h.score for h in hits])
    rescored = []
    for hit, prior in zip(hits, first_stage, strict=True):
        terms = tokenize(hit.chunk.text)
        present = set(terms)
        coverage = sum(idf(t) for t in q_terms if t in present) / total_idf
        bigram = len(q_bigrams & set(pairwise(terms))) / len(q_bigrams) if q_bigrams else 0.0
        score = weight * prior + (1 - weight) * (0.8 * coverage + 0.2 * bigram)
        rescored.append(Hit(hit.chunk, score))
    return sorted(rescored, key=lambda h: -h.score)


class Retriever:
    def __init__(self, chunks: Sequence[Chunk], config: RetrievalConfig) -> None:
        if config.mode not in MODES:
            raise ValueError(f"unknown retrieval mode {config.mode!r}; expected one of {MODES}")
        if config.k <= 0:
            raise ValueError("k must be positive")
        self.chunks = list(chunks)
        self.config = config
        self.bm25 = BM25Index([c.text for c in self.chunks])
        self._embedder = HashingEmbedder()
        self._vectors = [self._embedder.embed(c.text) for c in self.chunks] if config.mode == "hybrid" else []

    def retrieve(self, query: str) -> list[Hit]:
        n = max(self.config.candidates, self.config.k)
        scored = self.bm25.search(query, n) if self.config.mode == "bm25" else self._hybrid(query, n)
        hits = [Hit(self.chunks[i], s) for i, s in scored]
        if self.config.rerank:
            hits = rerank(query, hits, self.bm25.idf)
        return hits[: self.config.k]

    def _hybrid(self, query: str, n: int) -> list[tuple[int, float]]:
        lexical = dict(self.bm25.search(query, n))
        q_vec = self._embedder.embed(query)
        dense_all = {i: cosine(q_vec, v) for i, v in enumerate(self._vectors)}
        dense_top = sorted(dense_all.items(), key=lambda item: (-item[1], item[0]))[:n]
        ids = sorted(set(lexical) | {i for i, _ in dense_top})
        if not ids:
            return []
        lex_norm = _minmax([lexical.get(i, 0.0) for i in ids])
        dense_norm = _minmax([dense_all[i] for i in ids])
        alpha = self.config.hybrid_alpha
        combined = {i: alpha * ln + (1 - alpha) * dn for i, ln, dn in zip(ids, lex_norm, dense_norm, strict=True)}
        return sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:n]
