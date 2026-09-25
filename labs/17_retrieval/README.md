# Lab 17 — Retrieval for RAG

**Run:** `pytest labs/17_retrieval` (your code) · `pytest labs/17_retrieval --impl=solution` (reference)

**Reads first:** [applied systems §1](../../curriculum/10-applied-llm-systems.md#1-rag-that-actually-works)

## What you implement (the tests check each property)
- `BM25` from scratch (k1, b). Test against a hand-computed example.
- Dense cosine top-k, reciprocal rank fusion (k = 60), and maximal marginal relevance (MMR).
- Metrics: recall@k, MRR, nDCG@k with graded relevance, and average precision, all tested against known values.
- `chunk(text, size, overlap)` on token boundaries.
- `IVFIndex`: k-means coarse quantizer and `nprobe` search. Test: recall rises to 1.0 as `nprobe` reaches `nlist`.

## GPU scale-up / stretch
Build an eval set of 100 real questions over your own documents. Compare BM25, dense and hybrid retrieval, with and without a reranker, and report recall@10 with CIs.
