# Lab 17 — Retrieval for RAG

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 17_retrieval`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [applied systems §1](../../curriculum/10-applied-llm-systems.md#1-rag-that-actually-works)

## Implement and test
- `BM25` from scratch (k1, b). Test against a hand-computed example.
- Dense cosine top-k, reciprocal rank fusion (k = 60), and maximal marginal relevance (MMR).
- Metrics: recall@k, MRR, nDCG@k with graded relevance, and average precision, all tested against known values.
- `chunk(text, size, overlap)` on token boundaries.
- `IVFIndex`: k-means coarse quantizer and `nprobe` search. Test: recall rises to 1.0 as `nprobe` reaches `nlist`.

## GPU scale-up / stretch
Build an eval set of 100 real questions over your own documents. Compare BM25, dense and hybrid retrieval, with and without a reranker, and report recall@10 with CIs.
