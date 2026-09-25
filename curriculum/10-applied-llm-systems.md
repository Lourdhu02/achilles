# 10 — Applied LLM systems

Lab: [17 retrieval](../labs/17_retrieval/README.md). This is where your existing strengths (FinSentinelAI, ECHOME) live. The goal is to make them rigorous.

## 1. RAG that actually works
- **Choose deliberately:** prompting for behavior, RAG for knowledge that changes or must be cited, fine-tuning for format, style and skills, long context when the corpus is small and latency is acceptable.
- **Pipeline:** parse (layout-aware for PDFs; VLMs for scans and tables) → chunk (structure-aware, ~200–800 tokens, with parent-document links) → embed (contrastive bi-encoders; Matryoshka for tunable dimensions) → index (HNSW or IVF-PQ) → **hybrid retrieval** (BM25 plus dense, merged with RRF) → **rerank** (cross-encoder) → generate with citations → verify faithfulness.
- **Contextual chunk augmentation:** prepend a document-level summary to each chunk before embedding. This cheaply fixes chunks that lose meaning out of context.
- **Evaluate the two halves separately:** retrieval (recall@k, nDCG on a labeled set) and generation (faithfulness to sources, answer correctness), each with CIs. Most RAG failures are retrieval failures.
- **Multi-tenant security:** filter by tenant *inside* the retrieval query, never after generation. Watch for injection hidden in documents.

## 2. Agents
An agent is an LLM in a loop with tools and state. Know the mechanics: tool schemas (JSON schema), parallel tool calls, observation formatting, stop conditions and step budgets, and **context management** (summarization/compaction, memory stores, artifact files). MCP standardizes tool and context servers.
**Reliability math:** if each step succeeds with p = 0.95, a 20-step task succeeds with 0.95²⁰ ≈ 36%. Improve per-step reliability, add verification steps and checkpoints, and design for recovery.
**Evaluate agents** on task success over realistic environments, trajectory analysis (where did it go wrong?), cost and latency per task, and safety (irreversible actions, injection). Start simple: a single well-tooled agent usually beats a multi-agent graph.

## 3. Context engineering
What goes into the window, in what order, at what cost: system prompt, retrieved context, tool results, history. Use prompt caching for stable prefixes. Put instructions where the model attends reliably. Compress aggressively. Measure every change with evals.

## 4. Structured outputs and control
JSON schema / grammar-constrained decoding; validation plus retry; function-calling contracts. Constrained decoding can hurt quality if the schema fights the model's natural phrasing, so measure it.

## 5. LLMOps
Tracing every call (inputs, outputs, tokens, latency, cost); versioned prompts and configs; evals in CI; canary rollouts; fallbacks and model routing (cheap model first, escalate when confidence is low); caching; cost dashboards per feature and per customer. Treat a model upgrade as a deployment: run the evals before switching.

## 6. Upgrading your own projects
- **FinSentinelAI:** add a 200-question labeled eval, hybrid retrieval plus a reranker, faithfulness grading, and per-tenant filter tests. Report retrieval and answer metrics with CIs before and after.
- **ECHOME:** define measurable memory tasks (recall after N turns, contradiction handling), baseline against plain long context, and report the cost per session.
These upgrades turn "I built X" into "I measured X", which is what interviewers and customers believe.

**Read:** RAG (Lewis 2020); DPR; ColBERT; RRF (Cormack 2009); Lost in the Middle; ReAct; Toolformer; Anthropic's *Building Effective Agents* (2024) and *Contextual Retrieval* (2024); Chip Huyen, *AI Engineering* (2025); Hamel Husain on product evals.
