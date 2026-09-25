# ML system design (LLM era)

## The framework (45–60 min)
1. **Clarify** (5 min): users, the task, quality bar, latency SLO, scale, privacy and compliance, budget.
2. **Napkin math** (5 min): QPS, tokens in/out, FLOPs, KV memory, GPUs, $/request. Say the numbers out loud; this is where most candidates are weak.
3. **Architecture** (10 min): the data path end to end; draw it.
4. **Model choices** (10 min): API vs open weights vs fine-tuned; size vs latency; quantization; routing.
5. **Evaluation** (5–10 min): offline eval sets, online metrics, guardrails, a regression gate in CI.
6. **Failure modes and scaling** (5–10 min): bursts, long prompts, cache misses, provider outages, abuse, drift, cost blowups.
7. **Iterate**: what you'd build first, what you'd measure, what you'd defer.

## Worked designs (outline them yourself first, then compare)
**A. LLM serving platform** (70B, 1k req/s, 1k in / 300 out, p95 TTFT < 1 s). Output: 300k tokens/s. On H100s, an FP8 70B with TP = 4, continuous batching and paged KV might give ~2–4k output tokens/s per replica (measure; don't trust anyone's number), so ~100+ replicas. Prefill compute: 1k req/s × 1k tokens × 2·70e9 = 1.4e17 FLOP/s, ~350 H100s at 40% MFU. Consider disaggregating prefill and decode, prefix caching for shared system prompts, SLO-aware scheduling, autoscaling on queue depth, and speculative decoding for latency-tier users.
**B. Enterprise RAG** (10M documents, multi-tenant, citations). Ingestion (layout-aware parsing, contextual chunking), hybrid index per tenant or with an enforced tenant filter, reranker, a generator with a citation contract, faithfulness checks, per-tenant evals, incremental re-indexing, and access-control propagation from source systems.
**C. Pretraining run** (7B, 2T tokens, 256 GPUs). Data pipeline and mixture; tokenizer; FLOPs and days ([module 05 §8](../../curriculum/05-pretraining.md#8-exercise-plan-a-7b-run-on-paper)); FSDP/TP layout; checkpoint cadence and fault tolerance; a small-scale eval suite; loss-spike playbook; ablations at 1% scale first.
**D. RL post-training infrastructure** (GRPO on code). Prompt sampler → rollout fleet (vLLM) → sandboxed test execution (isolation, timeouts, determinism) → reward and advantage computation → trainer (FSDP) → weight sync to the rollouts (staleness budget) → eval and monitoring (reward hacking audits, length and entropy). Throughput is dominated by rollouts and sandboxes.
**E. Eval platform** for a lab or product. Versioned datasets, harness, judge models with calibration, CIs by default, contamination checks, dashboards, CI gates, cost controls.
**F. Classic ranking/recommendation** (still asked in Big Tech loops). Candidate generation (two-tower) → ranking (feature-rich model) → re-ranking (diversity, policy); features, logging, position bias, online A/B testing.

## Traps
Skipping the numbers; designing for 100× scale on day one; no eval story; ignoring data freshness and privacy; "fine-tune" as the answer to every quality problem; forgetting cost per request.
