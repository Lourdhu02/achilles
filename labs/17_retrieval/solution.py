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
        # BEGIN SOLUTION
        n, df = len(self.docs), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))
        # END SOLUTION

    def score(self, query: list[str], i: int) -> float:
        # BEGIN SOLUTION
        tf, dl = self.tf[i], len(self.docs[i])
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return sum(self.idf(t) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)
        # END SOLUTION

    def search(self, query: list[str], k: int = 10) -> list[int]:
        scores = [self.score(query, i) for i in range(len(self.docs))]
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


def dense_topk(query: np.ndarray, docs: np.ndarray, k: int) -> list[int]:
    """Indices of the k docs with the highest cosine similarity."""
    # BEGIN SOLUTION
    sims = (docs / np.linalg.norm(docs, axis=1, keepdims=True)) @ (query / np.linalg.norm(query))
    return list(np.argsort(-sims, kind="stable")[:k])
    # END SOLUTION


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse ranked lists: score(d) = sum 1 / (k + rank), rank starting at 1."""
    # BEGIN SOLUTION
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])
    # END SOLUTION


def mmr(query: np.ndarray, docs: np.ndarray, k: int, lam: float = 0.5) -> list[int]:
    """Maximal marginal relevance: greedily pick argmax lam*sim(q,d) - (1-lam)*max_{s picked} sim(d,s)."""
    # BEGIN SOLUTION
    d = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    rel = d @ (query / np.linalg.norm(query))
    picked: list[int] = []
    while len(picked) < min(k, len(docs)):
        red = (d @ d[picked].T).max(1) if picked else np.zeros(len(docs))
        score = lam * rel - (1 - lam) * red
        score[picked] = -np.inf
        picked.append(int(np.argmax(score)))
    return picked
    # END SOLUTION


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    # BEGIN SOLUTION
    return len(set(ranked[:k]) & relevant) / len(relevant)
    # END SOLUTION


def mrr(ranked: list[int], relevant: set[int]) -> float:
    # BEGIN SOLUTION
    return next((1.0 / r for r, d in enumerate(ranked, start=1) if d in relevant), 0.0)
    # END SOLUTION


def ndcg_at_k(ranked: list[int], gains: dict[int, float], k: int) -> float:
    """DCG = sum (2^g - 1) / log2(rank + 1), normalized by the ideal ordering."""
    # BEGIN SOLUTION
    dcg = sum((2 ** gains.get(d, 0) - 1) / math.log2(r + 1) for r, d in enumerate(ranked[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
    # END SOLUTION


def chunk(tokens: list, size: int, overlap: int) -> list[list]:
    """Sliding windows of `size` tokens with `overlap` shared tokens; the last window may be shorter."""
    # BEGIN SOLUTION
    step = size - overlap
    out = []
    for start in range(0, len(tokens), step):
        out.append(tokens[start : start + size])
        if start + size >= len(tokens):
            break
    return out
    # END SOLUTION


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
        # BEGIN SOLUTION
        probe = np.argsort(((self.centroids - q) ** 2).sum(1))[:nprobe]
        cand = np.concatenate([self.lists[j] for j in probe])
        dist = ((self.vectors[cand] - q) ** 2).sum(1)
        return list(cand[np.argsort(dist, kind="stable")[:k]])
        # END SOLUTION
