# Design: LLM Inference Platform

**Problem:** Design ChatGPT's serving infrastructure: serve a 70B-parameter LLM to 100M users with high throughput, low latency, and acceptable cost.

This is the design problem at Nvidia, Anthropic, OpenAI. Master it.

---

## Step 1: Clarify (5 min)

1. **Use case:** Chat (long generations) or short completions (autocomplete)?
2. **Latency target:** TTFT (time to first token) and ITL (inter-token latency)?
3. **Throughput:** QPS or tokens/sec across all users?
4. **Model:** Fixed or can we choose? Fine-tuned variants per customer?
5. **Streaming:** Yes (most likely)
6. **Multi-tenant:** Per-customer rate limits / quotas?
7. **Hardware:** Available — H100? A100? Mix?
8. **Cost budget:** USD per million tokens served?

Assumptions:
- Chat use case, long responses (avg 500 output tokens)
- TTFT p99 < 1s, ITL < 50ms
- 1M concurrent users, ~30M QPM avg, peak 50% above
- 70B model, no per-customer fine-tunes (separate problem)
- H100 GPUs available
- Cost target: $1-2 per million tokens

---

## Step 2: Requirements (3 min)

### Functional
- Streaming text generation
- Support context windows up to 128k
- Multi-turn conversations (KV cache reuse helpful)
- Stop conditions (max tokens, custom stop strings)
- Logprobs returned optionally

### Non-functional
- Throughput: 100k+ tokens/sec aggregate
- TTFT p99: 1s
- ITL p99: 50ms
- Availability: 99.99%
- Per-query cost: <$0.05 for typical 500-token completion

### Operational
- Live model swaps (no downtime when upgrading model version)
- Multi-region (US, EU, Asia)
- Autoscaling based on load

---

## Step 3: Architecture (10 min)

```
                  CONTROL PLANE
        ┌─────────────────────────┐
        │ Auth + Quotas + Routing  │
        └─────────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                       │
        ▼                       ▼
DATA PLANE — REGION A    DATA PLANE — REGION B
                
       Load Balancer
            │
   ┌────────┼─────────┐
   ▼        ▼         ▼
 [Inference Server]  ... (many replicas)
         │
   ┌─────┴─────┐
   │ vLLM      │
   │ - PagedAttn│
   │ - Cont Batch│
   │ - Speculative│
   │ - Prefix Cache│
   └─────┬─────┘
         │
   ┌─────┴────┐
   │ 8x H100  │ (tensor parallel for 70B)
   └──────────┘

         │
         ▼
   Telemetry/Logs → Object storage
   Real-time metrics → Prometheus + Grafana
```

---

## Step 4: Deep Dive

### Component A: Inference engine (vLLM)

**Why vLLM:**
- PagedAttention (best-in-class KV cache management)
- Continuous batching (throughput win)
- Prompt caching (huge for chat with system prompts)
- Open-source, well-maintained

**Tensor parallelism for 70B:**
- Llama 70B in FP16: 140GB
- 8x H100 80GB → 640GB total, ~80GB/GPU after sharding
- Tensor parallel: split each matrix multiplication across 8 GPUs
- NCCL all-reduce after attention and FFN
- Latency cost: ~10-20% overhead from communication
- Required by model size (can't fit on fewer)

**Batching:**
- Continuous batching: each token step, in-flight requests advance together
- Max batch size: tune for KV cache memory (e.g., 64 concurrent at 4k context)
- TTFT for new request: ~50-100ms after slot opens
- ITL within batch: ~30ms

**KV cache management:**
- PagedAttention: virtual paging, no fragmentation
- Per-request cache: dynamic allocation
- Eviction: LRU when memory pressure (mid-generation kills are bad UX → only on idle)

**Quantization:**
- FP16 standard
- INT8 (SmoothQuant or AWQ) for 2x more throughput, minor quality loss
- INT4 (AWQ) for cost-aggressive, ~3% MMLU drop, 4x cheaper
- Tier: high-quality customers FP16, cost-sensitive INT8/4

### Component B: Routing and load balancing

**Geo-aware:**
- Route to closest region by user IP
- Anycast / GeoDNS

**Per-region:**
- Load balancer (e.g., AWS NLB or Envoy)
- Least-loaded routing: route to GPU with fewest in-flight requests
- Failover: if a GPU node fails, immediately drain and reroute

**Stickiness for KV cache reuse:**
- If user is in a multi-turn chat, route them to the same backend (same KV cache)
- Implement via session ID hashing
- Tradeoff: less perfect load balancing, but huge KV cache reuse win

### Component C: Prompt caching

For chat with long system prompts (e.g., Claude's system prompt is 4-8k tokens):

**Approach:**
- Compute KV cache for system prompt once
- Persist across requests on same backend
- Hash by prompt prefix
- Anthropic Claude API exposes this explicitly; OpenAI does it implicitly

**Savings:**
- TTFT: reduce by 80%+ for long system prompts
- Cost: typically 90% off cached portion

### Component D: Speculative decoding (optional)

**Pattern:**
- Small draft model (Llama 3 1B or similar) generates K tokens
- Big model verifies all K in one forward pass
- Accept matching tokens; reject rest

**Gains:**
- 1.5-3x speedup
- More for predictable outputs (code, formal text)
- Less for creative outputs

**Cost:**
- Extra GPU memory for draft model
- Extra compute (parallel verification)

### Component E: Observability

- **Per-request:** request ID, user, model, input/output tokens, latency, cost
- **Per-replica:** GPU util, KV cache util, queue depth, throughput
- **System:** error rate, p50/p99 latency, model output distribution drift
- **Tracing:** OpenTelemetry distributed traces

---

## Step 5: Scale (5 min)

### Going from 100M users to 1B

- More regions (8+ regions)
- More backend pools per region (model versioning)
- Hierarchical routing

### Cost optimization at scale

- Mix of model sizes (Llama 70B for premium, Llama 8B for free tier)
- Aggressive batching
- Quantize lower for cost-sensitive
- Off-peak inference: pre-compute embeddings, summarizations during low-traffic hours

### Hardware mix

- H100 for new requests (latency-critical)
- A100 for batch / async workloads
- Maybe TPU v5 in select regions (cost arbitrage)

---

## Step 6: Iterate (5 min)

### Eval and rollout

- New model version → shadow deploy (1% traffic, compare outputs)
- Then canary 5% → 25% → 100%
- Roll back if latency / cost / quality regression

### Monitoring drift

- Sample 0.1% of outputs → run through LLM-as-judge
- Track quality score over time
- Alert on degradation

### Cost optimization opportunities

- Optimize for tail latency (5% of requests cause 50% of cost)
- Push hard prompt caching adoption (90% savings)
- Tiered offering (price discrimination by speed/quality)

### Future improvements

- Per-customer KV cache pinning
- Multi-LoRA serving (different fine-tunes on same base)
- Adaptive routing (LLM-of-LLMs: route hard queries to bigger model)

---

## Common follow-ups

1. **"How do you handle a customer with extremely long context (128k)?"**
   - Dedicated pool with more KV cache memory (fewer concurrent)
   - Quantized KV cache (INT4 → 4x more capacity)
   - Chunked prefill for ultra-long inputs

2. **"What if H100 supply is constrained?"**
   - Fall back to A100 pools (slower)
   - Quantize more aggressively
   - Reduce concurrent batch
   - Implement rate limiting if needed

3. **"How would you support 1000 fine-tuned models?"**
   - Multi-LoRA serving: same base, different LoRA adapters loaded per request
   - vLLM supports this
   - Memory: small (LoRA adapters are MB-scale)

4. **"What about safety filtering?"**
   - Input: moderation API (Anthropic / OpenAI built-in or run a classifier)
   - Output: similar
   - Streaming: filter chunk-by-chunk with reasonable lookahead
   - Latency cost: 10-50ms

5. **"How do you handle a backend going down mid-stream?"**
   - Detect immediately
   - Re-route to another backend with same KV cache (replicated)
   - Or: gracefully end stream + tell client to retry (degraded UX)

6. **"How do you do rate limiting?"**
   - Token bucket per API key
   - Distributed via Redis
   - Per-second + per-minute + per-day buckets

---

## Tradeoffs to articulate

- **Quality vs cost:** Bigger model = better quality, higher cost. Offer tiers.
- **Latency vs throughput:** Smaller batch = lower latency per request, lower throughput. Tune for the SLA.
- **Quantization:** Lower precision = cheaper but worse quality. Use for cost-sensitive customers.
- **Multi-region:** More regions = lower geo-latency, more operational overhead. Start with 2-3.

---

## Numbers you should know

| Metric | Approximate value |
|---|---|
| Llama 70B FP16 memory | 140 GB |
| Llama 70B INT4 memory | ~35 GB |
| KV cache per token (70B GQA) | ~2 KB |
| KV cache for 32k context | ~80 GB |
| H100 SXM FLOPS (FP8) | ~2 PFLOPS |
| H100 HBM bandwidth | ~3 TB/s |
| Llama 70B inference throughput (single H100) | impossible (too big) |
| Llama 70B on 4xH100 tensor parallel | ~100 tokens/sec single-stream |
| Llama 70B aggregate throughput (continuous batching, 8xH100) | ~2000 tokens/sec |

Memorize roughly. Cite in interviews.

---

## Your unique angle

Your Sujanix work on AWS Lambda + ONNX + INT8 quantization is RELEVANT here:
- "Quantization in production: I've done INT8 on smaller models. The quality calibration approach matters — using a representative dataset for calibration. For LLMs, AWQ does this differently from CV models because activations are non-uniform."
- "Cost optimization: at Sujanix we reduced cost 30% via serverless + quantization. For 70B inference at scale, prompt caching alone can deliver similar wins."

---

Next: [`04-design-recommender.md`](./04-design-recommender.md)
