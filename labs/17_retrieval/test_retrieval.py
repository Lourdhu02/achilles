import math

import numpy as np
import pytest

from labs._impl import load

rt = load(__file__)
DOCS = [["the", "cat", "sat"], ["the", "dog", "sat", "down"], ["cats", "and", "dogs"], ["cat", "cat", "food"]]


def test_bm25_by_hand():
    bm = rt.BM25(DOCS, k1=1.5, b=0.75)
    assert bm.idf("cat") == pytest.approx(math.log(1 + (4 - 2 + 0.5) / (2 + 0.5)))
    avgdl = 13 / 4
    tf_part = 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / avgdl))
    assert bm.score(["cat"], 3) == pytest.approx(bm.idf("cat") * tf_part)
    assert bm.search(["cat"], k=2) == [3, 0]
    assert bm.score(["zebra"], 0) == 0.0


def test_dense_mmr_and_fusion():
    docs = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.7, 0.7]])
    q = np.array([1.0, 0.2])
    assert rt.dense_topk(q, docs, 2) == [1, 0]
    diverse = rt.mmr(q, docs, 2, lam=0.3)
    assert diverse[0] == 1 and diverse[1] in (2, 3)
    fused = rt.reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert fused[:2] == [1, 3]


def test_metrics():
    ranked = [5, 2, 9, 1]
    assert rt.recall_at_k(ranked, {2, 1, 7}, 2) == pytest.approx(1 / 3)
    assert rt.mrr(ranked, {9}) == pytest.approx(1 / 3)
    assert rt.mrr(ranked, {42}) == 0.0
    dcg = (2**3 - 1) / math.log2(3)
    idcg = (2**3 - 1) + (2**1 - 1) / math.log2(3)
    assert rt.ndcg_at_k(ranked, {2: 3, 7: 1}, 3) == pytest.approx(dcg / idcg)


def test_chunking_covers_everything_with_overlap():
    toks = list(range(23))
    chunks = rt.chunk(toks, size=10, overlap=3)
    assert chunks[0] == list(range(10)) and chunks[1][:3] == [7, 8, 9]
    assert chunks[-1][-1] == 22
    assert all(len(c) <= 10 for c in chunks)


def test_ivf_recall_rises_with_nprobe():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(2000, 16))
    index = rt.IVFIndex(vecs, nlist=20)
    queries = rng.normal(size=(30, 16))

    def recall(nprobe):
        hits = 0
        for q in queries:
            exact = set(np.argsort(((vecs - q) ** 2).sum(1))[:10])
            hits += len(exact & set(index.search(q, 10, nprobe)))
        return hits / (10 * len(queries))

    low, full = recall(1), recall(20)
    assert full == 1.0 and low < full
