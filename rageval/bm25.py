"""Okapi BM25 over an in-memory inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from rageval.text import tokenize


class BM25Index:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        term_counts = [Counter(tokenize(t)) for t in texts]
        self.doc_lengths = [sum(c.values()) for c in term_counts]
        self.n_docs = len(texts)
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, counts in enumerate(term_counts):
            for term, tf in counts.items():
                self._postings[term].append((i, tf))

    def idf(self, term: str) -> float:
        # The +1 keeps idf positive for terms that appear in most documents.
        df = len(self._postings.get(term, ()))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for i, tf in self._postings.get(term, ()):
                norm = 1 - self.b + self.b * self.doc_lengths[i] / self.avgdl
                out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return dict(out)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]
