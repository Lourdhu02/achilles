# Inference Optimization — Make LLMs Fast and Cheap

Your TensorRT/ONNX INT8 work at Sujanix is your unique edge here. Lean into it.

This file is critical for Nvidia, Meta, Anthropic, and any "LLM Performance Engineer" role.

---

## The two phases of inference

```
Prefill: input prompt → process all tokens in parallel → produce first output token
         (compute-bound; FLOPS dominate)

Decode:  one token at a time → KV cache append → next token
         (memory-bound; HBM bandwidth dominates)
```

### Why this matters
- Prefill: optimize for FLOPS utilization (tensor parallelism, kernel fusion)
- Decode: optimize for memory bandwidth (KV cache management, quantization)

These are different optimization problems. Different solutions.

---

## Latency metrics

### TTFT (Time to First Token)
- Dominated by prefill
- Long prompts → high TTFT
- Optimize via: prefix caching, chunked prefill, speculative decoding (sort of)

### ITL (Inter-Token Latency) / TPOT (Time Per Output Token)
- Time between consecutive output tokens
- Dominated by decode
- Optimize via: smaller model, quantization, MoE active params, speculative decoding

### E2E latency
- TTFT + (output_tokens × ITL)
- What user experiences

### Throughput
- Tokens/second across all concurrent requests
- Optimize via: continuous batching, paged attention, prefix sharing

---

## KV Cache

### The math
- Per token: K and V vectors of dim d_model
- For L layers, H KV heads, head dim d_h: KV per token = 2 × L × H × d_h × 2 bytes (fp16)
- For Llama 70B (80 layers, 8 KV heads, 128 dim head, GQA): ~2KB per token per layer
- For 32k tokens × 80 layers: ~80GB

### Implementation
- Store K, V tensors growing over time
- On each new token: project to K_new, V_new; concatenate to cache
- Attention computes: Q_new × [K_cached, K_new], output_new × [V_cached, V_new]

### Quantization
- Cache in fp16 → 16 bits/element
- INT8 KV cache → 8 bits, 2x more capacity
- INT4 KV cache → 4 bits, 4x more capacity (some quality loss)

### Sharing
- Prefix cache: if multiple requests share a prefix, share the KV cache
- Especially useful for: system prompts, RAG context, chat history

### PagedAttention (vLLM)
- Manage KV cache like virtual memory pages
- Pages are non-contiguous blocks of fixed size
- Avoids memory fragmentation
- Enables copy-on-write for shared prefixes
- Massive throughput win (2-5x)

---

## FlashAttention (must know)

### Problem with naive attention
1. Q, K, V loaded from HBM (slow)
2. Compute Q @ K^T → write n×n matrix to HBM (slow)
3. Read n×n, apply softmax, write back
4. Read n×n, multiply by V, write output

HBM bandwidth is the bottleneck. For n=4096, the n×n matrix is 32MB which is huge.

### FlashAttention solution
- Tile Q, K, V into blocks
- Load blocks into SRAM (fast on-chip memory)
- Compute partial attention within SRAM
- Use online softmax (Welford-style numerical algorithm) to accumulate without storing full matrix
- Output: same result, way less HBM I/O

### Key ideas
1. **Tiling:** never materialize n×n matrix
2. **Recomputation:** backward pass recomputes attention rather than storing intermediates
3. **Online softmax:** can compute correct softmax incrementally

### Versions
- FlashAttention 1 (Dao 2022)
- FlashAttention 2 (Dao 2023) — better work partitioning
- FlashAttention 3 (Dao 2024) — H100-specific, FP8 support

### Adoption
- Built into PyTorch (`F.scaled_dot_product_attention`)
- Built into HuggingFace transformers
- Built into vLLM, TGI, SGLang

In production: you don't usually implement FlashAttention; you USE it. But interviewers will ask "why does it matter?" and you should know.

---

## Continuous batching

### Problem with static batching
- Wait until N requests arrive, batch them together
- All requests must produce same output length to batch efficiently
- Padding wastes compute; idle GPUs while batch fills

### Continuous batching (Orca, vLLM, etc.)
- Schedule per-token, not per-request
- When a request finishes a token, immediately add a new request to the batch
- All in-flight requests advance simultaneously
- Result: GPU stays full

### Benefits
- 2-5x throughput vs static batching
- Lower average latency at scale

---

## Quantization

### Numeric formats

| Format | Bits | Range | Use |
|---|---|---|---|
| FP32 | 32 | Wide | Training (reference) |
| FP16 | 16 | Narrow (overflow risk) | Standard training/inference |
| BF16 | 16 | Same as FP32 dynamic range | Modern training/inference |
| FP8 (E4M3, E5M2) | 8 | Narrow | H100 inference, some training |
| INT8 | 8 | -128 to 127 | Quantized inference |
| INT4 | 4 | -8 to 7 | Aggressive quantized inference |
| NF4 | 4 | Custom (normal-distrib optimal) | QLoRA training |

### Quantization methods

#### PTQ (Post-Training Quantization)
- After training, convert weights to lower precision
- Calibration: use a small dataset to determine optimal quantization params

Methods:
- **Naive:** scale + zero-point per tensor or per channel
- **GPTQ:** optimization-based; minimizes error in output activations
- **AWQ:** identifies salient weights (top 1%), keeps them in higher precision
- **SmoothQuant:** balance quantization error between weights and activations

#### QAT (Quantization-Aware Training)
- Train with simulated quantization in the loop
- Better quality than PTQ
- Expensive

### What to quantize in an LLM
- **Weights:** safe to quantize aggressively (INT4 often works)
- **Activations:** more sensitive; INT8 usually max
- **KV cache:** quantize to save memory
- **Embeddings + LM head:** often kept in higher precision

### Quality vs speedup tradeoff
- INT8: ~1% MMLU drop, 2x speedup
- INT4 (GPTQ/AWQ): 2-5% drop, 4x speedup
- INT4 + INT8 KV: 3-7% drop, 4-5x speedup + 2x memory

For most production: INT8 weights + FP16 activations is sweet spot.

---

## Speculative Decoding

### Idea
Use a small "draft" model to generate K candidate tokens. Then verify all K with the big model in one parallel forward pass. If they match, you saved K decode steps.

### Algorithm
1. Draft model generates K tokens (e.g., K=5)
2. Big model computes logits for ALL K positions in parallel (one forward pass)
3. For each position: did big model's top-1 match draft? If yes, accept; if no, take big model's choice and stop
4. Loop

### When it helps
- Output is "predictable" enough that the small model usually agrees with the big one
- Common in code generation, factual QA
- Less helpful for creative tasks

### Variants
- **Medusa:** add multiple LM heads to the big model itself (no separate draft)
- **EAGLE:** more efficient draft architecture
- **Lookahead decoding:** train-free, parallel-N-gram-based

### Real-world: 1.5-3x speedup typically

---

## Other inference tricks

### Continuous prefill
- Process prefill in chunks, interleaved with decode of other requests
- Reduces TTFT for new requests under load

### Prompt caching
- Cache prefix (e.g., system prompt) KV
- Reuse across multiple requests with same prefix
- Anthropic Claude has explicit prompt caching API; saves 90% on repeated prefixes

### Chunked attention
- Process long context in fixed-size chunks
- Avoid OOM on very long inputs

### Tensor parallelism
- Split a single layer across multiple GPUs
- Each GPU computes part of the matmul
- All-reduce to combine
- Used for: model that doesn't fit on one GPU

### Pipeline parallelism
- Different layers on different GPUs
- Forward pass: micro-batches flow through pipeline
- Used for: training (less common in inference)

### Expert parallelism (for MoE)
- Different experts on different GPUs
- Route tokens to GPUs based on gating
- DeepSeek-V3 style

---

## Inference serving frameworks

### vLLM (most popular open-source)
- PagedAttention, continuous batching
- HF-compatible
- High throughput
- Default for self-hosted

### TGI (Text Generation Inference, HuggingFace)
- Similar capabilities
- Better integration with HF ecosystem

### SGLang
- Newer, focuses on structured outputs and complex generation
- Often fastest for structured tasks

### TensorRT-LLM (Nvidia)
- Highly optimized for Nvidia GPUs
- Fast but Nvidia-only
- Used by Nvidia's NeMo, your Sujanix work? (TensorRT is a good signal)

### Llama.cpp / Ollama
- CPU + GPU, GGUF format
- Edge/local inference
- You used Ollama for FinSentinelAI — articulate this!

### Cloud APIs
- OpenAI, Anthropic, Google: managed
- AWS Bedrock, Vertex AI: cloud-hosted
- Together.ai, Anyscale, Fireworks: serverless

---

## Cost models

### Per-token pricing (OpenAI/Anthropic, 2026 estimates)
- GPT-4o-mini: $0.15/$0.60 per 1M tokens (input/output)
- GPT-4o: $2.50/$10
- Claude Haiku 4.X: ~$0.25/$1.25
- Claude Sonnet 4.X: ~$3/$15
- Claude Opus 4.X: ~$15/$75

### Self-hosted economics
- A100 80GB at ~$2-3/hr cloud
- Llama 70B inference: ~50 tokens/sec per GPU (decode)
- Cost per million output tokens: ~$10-20 (much cheaper than GPT-4o IF you have utilization)
- Break-even vs GPT-4o-mini: high volume, predictable load

---

## Lourdu's Sujanix experience — interview gold

Your work has:
- AWS Lambda serverless ML (cost optimization, cold starts)
- TensorRT/ONNX INT8 quantization (real production quantization)
- 99.9% uptime SLA (operational maturity)
- 30% cost reduction (real impact)

In interviews, articulate:
> "At Sujanix, we deployed Transformer-based OCR via FastAPI on AWS EC2 plus AWS Lambda for serverless inference. The challenge was: high throughput on government utility data + tight latency SLAs + cost.
>
> We approached it three ways:
> 1. **Model optimization:** SVRT architecture, then ONNX export, then INT8 post-training quantization. This let us run on smaller GPU instances.
> 2. **Serving:** FastAPI containers for high-throughput; Lambda for spiky workloads to avoid idle GPU cost.
> 3. **Quantization tradeoff:** we lost ~2% raw accuracy but with FocalCTCLoss and data augmentation gained 2.74% back, net positive.
>
> Result: 30% cost reduction with 99.9% uptime."

This shows: real understanding of inference economics, real production deployment, real tradeoff thinking. Use it.

---

## Interview questions

1. "Why is LLM inference memory-bound during decode?" — model weights must be loaded every token; only 1 token of compute per forward pass
2. "Explain KV cache. What's its memory footprint for a 70B model at 32k context?" — see math above
3. "How does PagedAttention help?" — non-contiguous memory pages, no fragmentation, shared prefixes
4. "When would you use INT4 vs INT8 quantization?" — INT4 for memory-constrained, INT8 for quality-sensitive
5. "What's speculative decoding? When does it help?" — draft + verify, helps for predictable outputs
6. "Walk through what FlashAttention does." — tiling + online softmax + no n×n materialization
7. "How would you reduce inference cost for a RAG application by 50%?" — smaller model, prompt caching, batching, quantization, distillation
8. "What's the bottleneck for prefilling 100k tokens?" — usually compute (matmul); GPU FLOPS

---

## Resources

- **vLLM paper** (Kwon 2023) — PagedAttention, must read
- **FlashAttention paper** (Dao 2022)
- **AWQ paper** (Lin 2023)
- **GPTQ paper** (Frantar 2022)
- **Speculative Decoding paper** (Leviathan 2023)
- **NVIDIA's TensorRT-LLM docs**
- **vLLM documentation**

---

Next: [`06-evaluation.md`](./06-evaluation.md)
