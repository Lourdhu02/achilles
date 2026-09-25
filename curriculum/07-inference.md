# 07 — Inference

Labs: [06 kernels](../labs/06_attention_kernels/README.md), [07 KV cache](../labs/07_kv_cache_sampling/README.md), [13 quantization](../labs/13_quantization/README.md), [14 speculative decoding](../labs/14_speculative_decoding/README.md).

## 1. Two phases, two bottlenecks
**Prefill** processes the prompt in parallel: large matmuls, compute-bound; it determines **TTFT**. **Decode** produces one token per step per sequence: memory-bound; it determines **TPOT/ITL**. Throughput comes from batching decode steps across users until the KV cache fills memory or latency SLOs break. Goodput is throughput *within* the SLOs.

## 2. Serving system mechanics
- **Continuous batching** (Orca): schedule at every iteration, adding and removing requests each step.
- **PagedAttention** (vLLM): the KV cache lives in fixed-size blocks with per-sequence block tables. No fragmentation, copy-on-write sharing, higher batch.
- **Prefix caching:** reuse KV for shared prefixes (system prompts, RAG context, multi-turn history). SGLang's RadixAttention keeps a radix tree of prefixes.
- **Chunked prefill** (Sarathi-Serve): split long prompts to interleave with decode and protect ITL. **Disaggregated prefill/decode** (DistServe, Splitwise): separate pools, KV transferred between them.
- Parallelism: TP within a node for latency, PP or EP across nodes for big and MoE models, DP replicas for throughput.
- **CUDA graphs** and torch.compile remove per-token launch overhead, which matters at small batch.

## 3. Speculative decoding, done correctly
The draft proposes k tokens from q; the target scores all k in one pass. Accept token x with probability `min(1, p(x)/q(x))`; on the first rejection, sample from `norm(max(0, p − q))`; if all are accepted, sample one bonus token from p. **This is exactly lossless:** the output distribution equals the target's. The acceptance rate is `α = Σ min(p, q)`, and expected tokens per target pass = `(1 − α^{k+1})/(1 − α)`. Draft options: a small model, extra heads (Medusa), feature-level drafts (EAGLE-2/3), or n-gram/prompt lookup (free for code and edits). It helps most at low batch, where decode is memory-bound and the target pass is nearly free per extra token.

## 4. Quantization
- **Weight-only** (INT4/INT8: GPTQ, AWQ, GGUF k-quants): cuts the bytes streamed per decode step, a direct speedup in the memory-bound regime. It does nothing for compute-bound prefill.
- **Weights plus activations** (INT8 SmoothQuant; FP8 E4M3 with scaling): also speeds prefill on hardware with fast low-precision math. Activation outliers are the problem, hence per-channel scales, SmoothQuant's migration of scale into the weights, and LLM.int8's outlier split.
- **KV-cache quantization** (FP8/INT8/INT4): more concurrent sequences and longer context.
- **Blackwell FP4** (MXFP4/NVFP4) with block scaling: the current frontier.
- Always evaluate quality on *your* task, with CIs. Perplexity hides task-specific damage.

## 5. Kernels: why FlashAttention wins
It tiles Q, K and V into SRAM and uses online softmax to accumulate outputs without materializing the T×T matrix; the backward pass recomputes P from the saved log-sum-exp. HBM traffic drops from Θ(T²) to Θ(T²d²/M). FlashAttention-2 improved work partitioning; FlashAttention-3 uses Hopper's asynchrony and FP8. For decode, **FlashDecoding** splits the KV sequence across programs. Also fused norms, rotary and activations, and paged-attention kernels. Triton makes writing these accessible ([lab 06](../labs/06_attention_kernels/README.md)).

## 6. Capacity planning (the interview question)
"Serve Llama-3-70B at 1,000 req/s, 1k in / 300 out, p95 TTFT < 1 s": work out the tokens/s needed, per-GPU decode throughput at a batch that fits the KV cache, and the prefill compute; then the number of GPUs, replicas and TP degree; then the cost per million tokens. Show the arithmetic and the failure modes (long-prompt bursts, cache thrash).

## 7. Engines
vLLM, SGLang, TensorRT-LLM (NVIDIA), llama.cpp/GGUF (CPU, edge, consumer GPUs), MLC. On your 8 GB card: llama.cpp for 7–8B at 4-bit; vLLM or SGLang for ≤ 3B in bf16 when supported on your CUDA build.

**Read:** Pope et al. 2022 (*Efficiently Scaling Transformer Inference*); Orca; vLLM/PagedAttention; SGLang; Sarathi-Serve; DistServe; Leviathan et al. and Chen et al. 2023 (speculative); Medusa; EAGLE; FlashAttention 1–3; LLM.int8; SmoothQuant; GPTQ; AWQ; MX formats.
