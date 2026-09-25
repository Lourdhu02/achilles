# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/17_retrieval
"""Lab 17 -- retrieval for RAG: BM25, dense search, fusion, diversity, IR metrics, chunking, IVF.

Handout: labs/17_retrieval/README.md
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    """Okapi BM25 with Lucene's idf: log(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(d) for d in docs]
        self.df = Counter(t for d in docs for t in set(d))
        self.avgdl = sum(len(d) for d in docs) / len(docs)

    def idf(self, term: str) -> float:
        raise NotImplementedError("17_retrieval: implement idf")

    def score(self, query: list[str], i: int) -> float:
        raise NotImplementedError("17_retrieval: implement score")

    def search(self, query: list[str], k: int = 10) -> list[int]:
        scores = [self.score(query, i) for i in range(len(self.docs))]
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


def dense_topk(query: np.ndarray, docs: np.ndarray, k: int) -> list[int]:
    """Indices of the k docs with the highest cosine similarity."""
    raise NotImplementedError("17_retrieval: implement dense_topk")


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse ranked lists: score(d) = sum 1 / (k + rank), rank starting at 1."""
    raise NotImplementedError("17_retrieval: implement reciprocal_rank_fusion")


def mmr(query: np.ndarray, docs: np.ndarray, k: int, lam: float = 0.5) -> list[int]:
    """Maximal marginal relevance: greedily pick argmax lam*sim(q,d) - (1-lam)*max_{s picked} sim(d,s)."""
    raise NotImplementedError("17_retrieval: implement mmr")


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    raise NotImplementedError("17_retrieval: implement recall_at_k")


def mrr(ranked: list[int], relevant: set[int]) -> float:
    raise NotImplementedError("17_retrieval: implement mrr")


def ndcg_at_k(ranked: list[int], gains: dict[int, float], k: int) -> float:
    """DCG = sum (2^g - 1) / log2(rank + 1), normalized by the ideal ordering."""
    raise NotImplementedError("17_retrieval: implement ndcg_at_k")


def chunk(tokens: list, size: int, overlap: int) -> list[list]:
    """Sliding windows of `size` tokens with `overlap` shared tokens; the last window may be shorter."""
    raise NotImplementedError("17_retrieval: implement chunk")


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if np.any(assign == j):
                c[j] = x[assign == j].mean(0)
    return c


class IVFIndex:
    """Inverted file index: cluster vectors with k-means; search only the `nprobe` nearest lists."""

    def __init__(self, vectors: np.ndarray, nlist: int, seed: int = 0):
        self.vectors = vectors
        self.centroids = kmeans(vectors, nlist, seed=seed)
        assign = ((vectors[:, None] - self.centroids[None]) ** 2).sum(-1).argmin(1)
        self.lists = [np.flatnonzero(assign == j) for j in range(nlist)]

    def search(self, q: np.ndarray, k: int, nprobe: int) -> list[int]:
        raise NotImplementedError("17_retrieval: implement search")
