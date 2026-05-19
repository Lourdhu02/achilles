# Flagship Projects to Build (Months 4-8)

You have 3 strong existing projects (ECHOME, FinSentinelAI, Transformers-OCR). To get to "exceptional for 2 YOE", build 2 more flagships during Phase 2.

These should be PUBLIC, OPEN-SOURCE, BLOG-WORTHY.

---

## Flagship Project #1 (Months 4-6)

### Recommended: "RAGSilon" — Production RAG with Eval Harness

**Why this project:**
- Closes a real gap in OSS (most RAG repos have weak eval)
- Direct extension of FinSentinelAI experience
- Demonstrates production thinking + evaluation rigor
- Highly relevant to League A + League B targets

### Scope (3 months, ~150 hours)

**Core deliverables:**

1. **Modular RAG library** with:
   - Document ingestion (PDFs, DOCX, HTML, code)
   - Chunking (multiple strategies — fixed, semantic, layout-aware)
   - Embedding (pluggable: bge, OpenAI, Cohere)
   - Storage (Qdrant + OpenSearch hybrid)
   - Retrieval (dense + sparse + RRF + reranker)
   - Generation (multiple LLM backends)
   - Faithfulness verification post-generation

2. **Eval harness:**
   - RAGAS metrics + custom faithfulness checks
   - Golden set construction tools
   - LLM-as-judge with multiple judges
   - Comparison across models / chunking strategies / rerankers
   - Visualization of results

3. **Benchmark report:**
   - Run eval on 3 publicly available QA datasets
   - Report which configurations work best for which query types
   - Surface counterintuitive findings

4. **Deployment example:**
   - Docker compose for local
   - Kubernetes example
   - Cost estimator (per query, by config)

### Success criteria
- 50+ GitHub stars by end
- 1 blog post (3000+ words) about findings
- 1 talk submission (lightning talk at meetup or conference)
- Mentioned in at least 1 newsletter (Latent Space, etc.)

### Why this gets you interviews
- Shows: production engineering, ML evaluation, system design, code quality
- Directly relevant to roles at Cohere, Anthropic, Databricks, Google Vertex AI

---

## Flagship Project #2 (Months 6-8)

### Option A (Recommended): "AgentBench-India" — Agent Benchmark for Indian Languages / Tasks

**Why this project:**
- Novel angle: Indian-specific (uncrowded)
- Highly relevant to Google India, Microsoft, Sarvam, Krutrim
- Combines your agent expertise (ECHOME) with evaluation rigor

### Scope (2-3 months, ~100 hours)

**Deliverables:**

1. **Benchmark dataset:**
   - 10-15 agent tasks in Indian languages (Hindi, Tamil, Telugu, English-Indian)
   - Categories: customer service, navigation, code, math, factual QA
   - Ground truth + grading rubrics

2. **Agent harness:**
   - Run any agent (LangGraph, CrewAI, AutoGen) on the benchmark
   - Capture full traces for analysis
   - Metrics: task completion, # tool calls, cost, latency

3. **Baseline results:**
   - GPT-4o, Claude, Gemini, Llama, Mistral, Indian-language models
   - Comparative analysis

4. **Public leaderboard:**
   - Submit-your-agent format
   - Open submissions

### Success criteria
- 100+ stars
- Cited in at least 1 academic paper or tech blog
- Coverage in Indian AI media (Analytics India Mag, AI4India)

### Option B: "FastRAG" — Inference-Optimized RAG

If you'd rather lean into your TensorRT/quantization edge:

- Take a standard RAG stack
- Optimize for latency: cached embeddings, quantized reranker, streaming LLM with optimal batch
- Benchmark vs naive: aim for 10x latency improvement
- Document tradeoffs

### Option C: "MoE-LoRA" — Multi-LoRA Serving Framework

If you want to go deep on inference systems:

- Build on vLLM
- Add: serve N LoRA adapters on shared base efficiently
- Auto-loading from disk, LRU eviction, prefetch
- Open-source contribution path

---

## Project execution framework

### Phase 1 (week 1-2): Scope
- Write a 1-pager: problem, scope, deliverables, success criteria, timeline
- Get feedback on scope (from me, from peers)
- Cut features ruthlessly to fit in time budget

### Phase 2 (weeks 3-8): MVP
- Build the minimum that demonstrates the core value
- Daily commits
- Weekly retro: are you on track?

### Phase 3 (weeks 9-10): Polish
- README, docs, examples, install scripts
- Test from a clean machine
- CI/CD for tests

### Phase 4 (weeks 11-12): Launch
- Blog post with benchmarks
- LinkedIn announcement
- Submit to /r/MachineLearning, HN, relevant newsletters
- DM key people in the space ("hey, built this, would love your thoughts")

---

## What makes a flagship project DIFFERENT from a side project

A side project: code on GitHub, 2-line README, no docs, abandoned in 2 weeks.

A flagship project:
- ✅ Solves a real problem (not "I wanted to learn X")
- ✅ Has unique insight or angle (not just another RAG demo)
- ✅ Production-quality code (tested, documented, deployable)
- ✅ Reproducible benchmarks
- ✅ Public traction (stars, comments, mentions)
- ✅ Distribution plan (blog, social, communities)

If you're not committing 80+ hours, it's a side project, not a flagship.

---

## Anti-patterns

- **"I'll build the next ChatGPT":** scope too big; you'll abandon in week 4
- **"I'll build N small tools":** dilutes signal; pick ONE big one
- **"I'll build something competitive with $50M-funded startups":** lose by definition; pick a niche where YOU can be #1
- **"I'll publish in 6 months when it's perfect":** ship at week 8 with rough edges; iterate publicly
- **"It's open-source so it'll get noticed":** distribution IS the work; plan it

---

## Time budget across both projects

| Activity | Hours |
|---|---|
| Flagship #1 (RAGSilon) | 150 |
| Flagship #2 (AgentBench-India) | 100 |
| Blog posts (2) | 30 |
| Distribution / engagement | 20 |
| **Total** | **~300 hrs** |

Over Months 4-8 (5 months) = ~60 hrs/month = ~14 hrs/week.

This is the bulk of your portfolio investment. Treat it as a primary career investment, not "extra credit".

---

## Why this matters more than 50 more LeetCode problems

By Month 8, the marginal return of LC problem 351 is low. The marginal return of a strong flagship project is enormous: it's the differentiator that gets you offers vs. rejections among technically-competent candidates.

If you have to choose between an extra 30 LC problems vs. polishing your flagship: polish.

---

Next: [`07-kaggle-to-master.md`](./07-kaggle-to-master.md)
