# Prompt: finish FinSentinelAI

Paste everything below the line into a new Claude Code session with `Lourdhu02/fin-sentinal.ai` attached.

---

You are finishing **FinSentinelAI** (https://github.com/Lourdhu02/fin-sentinal.ai), a fully local RAG system for financial documents: FastAPI + React, ChromaDB, `all-MiniLM-L6-v2` embeddings, a `ms-marco-MiniLM-L-6-v2` cross-encoder reranker, Ollama for generation, and JWT-isolated workspaces. The goal is a project with a credible, reproducible evaluation that an ML hiring manager can trust.

## Where it stands

Branch `claude/magical-allen-wc99cc` (merge it first, or build on it) adds:
- `eval/`: a benchmark of 60 questions (10 per type) over the 1,000 PDFs in `test-data/sugar_dataset`, which covers invoices, bank statements, salary slips, GST returns, POs and credit/debit notes in 10 layouts. Gold answers are copied from each document. `eval/run_eval.py` indexes the corpus through the app's own parser, chunker, embedder and store, and the `Retrieval benchmark` workflow runs it in CI.
- **Measured results:**
  - Dense-only retrieval found the right document in the top 5 for **5%** of questions, and **15%** after reranking (CI run 36988217123). MiniLM blurs exact identifiers such as `INV-2023-8285` and ARNs.
  - BM25 alone gets **100% recall@1** on the same questions.
- **The fix, implemented on that branch:** `core/lexical.py` (BM25 with an identifier-preserving tokenizer), `Retriever.hybrid_search` (dense + BM25 fused with reciprocal-rank fusion), and `pipeline.query` now uses it. CI run 36989253719 measured hybrid + rerank; read its job summary first.

## What to do

1. **Make the benchmark harder and fairer.** The current questions are all exact-ID lookups, which BM25 solves trivially. Add at least 60 questions that need semantics or aggregation, for example:
   - "which vendor supplied lip gloss in June 2022?"
   - "total GST paid in FY 2023-24"
   - "who had the largest TDS deduction in March 2025?"
   - questions with paraphrased field names.

   Generate each gold answer by code from the documents, never by hand-guessing. Report results per question type.
2. **Report per stage:** dense, BM25, hybrid and hybrid + rerank, each with recall@1/5, MRR and 95% bootstrap CIs. Tune the RRF k and the candidate depth on a held-out split, never on the test split.
3. **Answer accuracy:** run `python -m eval.run_eval --llm` with a stated local model (e.g. `qwen2.5:7b-instruct` Q4_K_M). Score with exact match on normalized values. For aggregation questions, compare against the deterministic answer computed from the documents.
4. **Faithfulness:** check that every number in an answer appears in a cited chunk. Report the unsupported-number rate, and add a refusal path when retrieval confidence is low.
5. **Latency and footprint:** p50/p95 per stage on CPU-only hardware (state the machine); index build time and size for 1,000 docs.
6. **Security, since the project claims isolation:** a test proving user A can never retrieve user B's chunks through dense, BM25 or hybrid search. The BM25 cache must be keyed per filter.
7. **README:** an architecture diagram, a Results table (each number with its command, date, hardware and seed), and a "Known limitations" section.

## Rules

- No invented or projected numbers. Every number comes from `eval/` with its config committed. Report bad results honestly, along with what you changed in response.
- Commit as `Lourdhu Raju <b.lourdhuraju1234@gmail.com>`, with no Co-Authored-By or session trailers.
- Work on a branch, open a PR, and make sure the CI (both `CI` and `Retrieval benchmark`) is green.

## Done when

The benchmark has both lookup and semantic/aggregation questions, all four retrieval stages and answer accuracy are reported with CIs, the isolation test passes, and the README Results table reproduces from a clean clone. Then give me two resume bullets with measured numbers.
