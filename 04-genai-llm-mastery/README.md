# 04 — GenAI / LLM Mastery

**This is YOUR competitive edge.**

Your ECHOME, FinSentinelAI, and SVRT/OCR projects give you a head start most candidates don't have. But "I've used LangChain" is table stakes. To win at Anthropic, OpenAI, Cohere, or any GenAI-focused team at FAANG, you need depth that goes beyond using LangChain — you need to understand WHY each component exists and what's happening underneath.

By Month 8, you should be **Tier 4 (Expert)** on every topic in this folder.

---

## Files in this folder

1. [`01-llm-internals.md`](./01-llm-internals.md) — Tokenization, attention variants, positional encodings, training stack
2. [`02-rag-deepdive.md`](./02-rag-deepdive.md) — Beyond toy RAG: production patterns, eval, multi-modal
3. [`03-agents-deepdive.md`](./03-agents-deepdive.md) — ReAct, Plan-and-Execute, memory, eval, planning
4. [`04-fine-tuning.md`](./04-fine-tuning.md) — LoRA, QLoRA, RLHF, DPO, PEFT, when to fine-tune
5. [`05-inference-optimization.md`](./05-inference-optimization.md) — vLLM, FlashAttention, quantization, KV cache, speculative decoding
6. [`06-evaluation.md`](./06-evaluation.md) — Benchmarks, LLM-as-judge, custom evals, RAGAS, MTEB
7. [`07-must-know-papers.md`](./07-must-know-papers.md) — The 30 essential papers to know
8. [`08-genai-qa-bank.md`](./08-genai-qa-bank.md) — 100+ GenAI-specific interview questions

---

## Why GenAI mastery matters for your career

In 2026-2027, the hiring market is bifurcating:

- **Generalist ML engineers** — competition is fierce, supply has caught up to demand
- **GenAI/LLM specialists** — demand still exceeds supply; can command premium comp + better roles

Your projects already position you as the second. This folder makes that position defensible against a real interview.

---

## How to study (most efficient method)

### Layer 1: Conceptual understanding (Months 5-6)

For each topic:
1. Read 1-2 foundational papers
2. Read 2-3 explainer blogs
3. Watch 1-2 videos
4. Write your own 1-page summary

### Layer 2: Implementation (Months 6-7)

For each topic:
1. Implement from scratch (small scale)
2. Use the production tool (vLLM, TRL, etc.) on a real problem
3. Compare: your implementation vs production
4. Note differences, document why

### Layer 3: Production thinking (Months 7-8)

For each topic:
1. What's the failure mode at scale?
2. What's the monitoring strategy?
3. What's the cost tradeoff (compute, memory, latency)?
4. What's the eval strategy?

By Layer 3, you can answer almost any GenAI interview question, including the "design X at scale" questions.

---

## Your existing project map (lean into these in interviews)

| Concept | Your project that demonstrates it | What to articulate in interviews |
|---|---|---|
| Multi-agent orchestration | ECHOME (LangGraph) | Why LangGraph over alternatives; state mgmt; failure handling |
| Long-horizon agent memory | ECHOME (3-tier memory) | Tradeoffs of each tier; consolidation logic; retrieval |
| Voice/multimodal | ECHOME (XTTSv2) | Zero-shot voice cloning; quality/latency tradeoffs |
| Adaptive evaluation | ECHOME (CAT/IRT + Fisher Info) | Why adaptive testing; what made it 70% faster |
| RAG architecture | FinSentinelAI | Production decisions: Ollama vs hosted, ChromaDB choice, JWT isolation |
| Multimodal extraction | FinSentinelAI (VLM for PDFs) | Local VLM choice; cost/quality; fallback handling |
| OCR + transformers | Transformers-OCR (SVRT) | FocalCTCLoss reasoning; FRM custom modules; INT8 deployment |
| Edge ML deployment | Transformers-OCR (ONNX INT8) | Quantization strategy; calibration; accuracy/speed tradeoff |
| Production inference | Sujanix work (AWS Lambda) | Cold starts; autoscaling; serverless ML tradeoffs |

In every interview, **route follow-up questions back to these projects.** They are your "I've actually done this" credibility.

---

## The frontier of GenAI in 2026 (stay current)

Topics actively evolving — re-check every 3 months:

- **Reasoning models** (o1, o3 from OpenAI; Claude with extended thinking) — RL on chains of thought
- **Multi-modal models** (GPT-4o, Claude Opus 4.X multimodal) — unified audio/vision/text
- **Long context** (1M+ tokens) — Gemini, Claude
- **Agent frameworks** (Computer Use, browser agents, code agents)
- **Inference cost reduction** — distillation, quantization, MoE
- **Open-source LLMs** — Llama 4, DeepSeek, Mistral, Qwen all push frontier
- **Safety and alignment** — Constitutional AI, RSPs, mech interp

You don't need to know everything. But you should be reading 2-3 papers/month + following the firehose enough to have opinions.

---

## Communities + signals to follow

- **Latent Space podcast** (swyx) — best signal-to-noise for engineers
- **AI Engineer Summit / World's Fair** — conferences
- **Hugging Face Discord** — active community
- **OpenReview / arXiv-sanity** — papers
- **Karpathy YouTube** — long-form deep dives
- **Anthropic blog** — high-quality posts on alignment + capabilities
- **DeepMind blog** + Google DeepMind on YouTube
- **/r/LocalLLaMA** — community, mostly open-source focused

---

## Books worth buying (for this folder specifically)

- **"Hands-On Large Language Models" by Jay Alammar & Maarten Grootendorst** (2024) — best practical book
- **"Building LLMs for Production" by various authors** (2024) — production-focused
- **"AI Engineering" by Chip Huyen** (2025?) — system design for LLMs
- **"Generative AI on AWS" / "Generative AI on GCP"** — vendor-specific but useful

Avoid: any book that promises "master AI in 30 days" — those are surface-level.

---

## The "shipped a public flagship LLM project" goal

By Month 8, you should have shipped:

- **Flagship Project #1** (Months 5-6): an end-to-end RAG with eval framework (e.g., "GenAI-RAG-Eval — production-grade RAG with eval suite, benchmarked on RAGAS")
- **Flagship Project #2** (Months 7-8): an LLM agent or eval benchmark (e.g., "AgentBench-IND — Indian-language agent benchmark with 10 tasks")

Plans in `07-portfolio/06-new-projects-to-build.md`.

These two projects, plus your existing three (ECHOME, FinSentinelAI, Transformers-OCR), give you a portfolio of 5 substantial GenAI projects. **This is exceptional for 2 YOE.**

---

Now go to [`01-llm-internals.md`](./01-llm-internals.md).
