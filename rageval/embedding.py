"""Feature-hashing vectors: a dependency-free stand-in for a dense embedding model."""

from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict
from itertools import pairwise

from rageval.text import tokenize

SparseVector = dict[int, float]


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, text: str) -> SparseVector:
        terms = tokenize(text)
        features = Counter(terms)
        features.update(f"{a} {b}" for a, b in pairwise(terms))
        vec: dict[int, float] = defaultdict(float)
        for feature, count in features.items():
            # crc32 is stable across runs; Python's hash() is salted per process.
            h = zlib.crc32(feature.encode("utf-8"))
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[h % self.dim] += sign * (1 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {i: v / norm for i, v in vec.items()} if norm else {}


def cosine(a: SparseVector, b: SparseVector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(i, 0.0) for i, v in a.items())
