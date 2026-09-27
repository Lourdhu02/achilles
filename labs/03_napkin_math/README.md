# Lab 03 — Napkin math for large models

**Build:** a calculator for parameters, FLOPs, training and inference memory, KV cache, roofline time, MFU, communication and serving cost. Then measure your own GPU and check the predictions.<br>
**Time:** 5–7 h · **Reads first:** [compute & hardware](../../curriculum/02-compute-and-hardware.md)<br>
**Run:** `pytest labs/03_napkin_math` · `python tools/measure_gpu.py`

Frontier labs interview this directly ("how long does a 70B model take to train on 2,048 H100s?",
"why is decode slow?"). Founders need the same arithmetic for "what does a request cost us?".
The tests pin exact parameter counts for GPT-2, Llama-2-7B, Llama-3-8B and Mistral-7B, so your
formula is either right or it isn't.

---

## The fourteen rules

1. **A matmul `(m×k)·(k×n)` costs `2mkn` FLOPs**: one multiply plus one add per multiply-accumulate.
2. **Forward ≈ 2N FLOPs per token**, where N counts the weights that take part in matmuls, including the LM head (tied or not; the embedding lookup is free).
3. **Training ≈ 6N FLOPs per token.** For `Y = XW`, backward computes `dX = dY·Wᵀ` and `dW = Xᵀ·dY`, each the size of the forward matmul, so backward ≈ 2× forward.
4. **Attention adds `2·L·T·d` per token forward** (causal, averaged over positions). It is small at 2k context and comparable to the weights at 128k.
5. **Per layer ≈ 12d² weights** for MHA plus a 4× MLP. SwiGLU uses 3 matrices at ~8/3·d hidden to stay ≈ 12d². GQA shrinks K and V.
6. **Training memory ≈ 16 bytes/param** with mixed-precision Adam (bf16 weights 2 + bf16 grads 2 + fp32 master/m/v 12) plus activations. ZeRO/FSDP shard these across GPUs.
7. **KV cache = 2 · L · H_kv · d_head · T · B · bytes.** For Llama-2-70B at 32k context: 10 GiB with GQA (8 KV heads), 80 GiB if it had used MHA.
8. **Roofline:** time ≥ max(FLOPs / peak FLOP/s, bytes / bandwidth). *Arithmetic intensity* (FLOPs per byte) decides which bound wins. The *ridge point* is peak / bandwidth.
9. **Batch-1 decode is memory-bound:** every token streams every weight once, at an intensity of ~1 FLOP/byte in bf16. So `tokens/s ≤ bandwidth / weight bytes`. Batching amortizes the weight reads, until KV-cache reads dominate.
10. **Ring all-reduce moves `2(n−1)/n` of the buffer per device.** Its cost is almost independent of n; bandwidth decides it.
11. **MoE: FLOPs follow active parameters, memory follows total parameters.** A top-k mixture of experts costs about `2·N_active` FLOPs per token but must hold `N_total` in memory ([lab 18](../18_moe/README.md)).
12. **Tensor parallelism costs two all-reduces per layer in the forward pass** (after attention and after the MLP), each the size of the `B·T·d` activations, and two more in the backward. That is why tensor parallelism stays inside a node, on NVLink.
13. **Prefill is compute-bound, decode is memory-bound.** Time to first token ≈ `2·N·T_prompt / achieved FLOP/s`: a 2,000-token prompt to Llama-3-8B on an H100 at 50% MFU takes about 65 ms. Each later token costs about `weight bytes / bandwidth` (rule 9).
14. **Little's law for serving:** requests in flight = arrival rate × time per request. At 10 requests/s that each stream for 4 s, 40 sequences are in flight, and all 40 KV caches must fit in memory (rule 7).

## Worked examples (do them on paper first, then with your code)

**What can I train on an 8 GB RTX 5060?** At 16 B/param, 0.5B parameters already fill 8 GB before
any activations. Realistically ~100–150M parameters trains comfortably with bf16 autocast and
modest batches, or ~300M with activation checkpointing and 8-bit optimizer states. For
fine-tuning, LoRA on a frozen bf16 1.5B model (3 GB of weights) fits. QLoRA (4-bit base) reaches
7–8B with short sequences.

**How long to train GPT-2 (124M) on 10B tokens on that GPU?** 6 × 124M × 10B × 1.07 (attention)
≈ 8·10¹⁸ FLOPs. If you measure ~40 dense bf16 TFLOP/s and reach 40% MFU, that is 16 TFLOP/s,
so 5·10⁵ s ≈ 5.8 days. So plan 1B tokens (~14 h) or a smaller model for the scale-up runs.

**Decode speed of an 8B model in 4-bit on the 5060:** ~4 GB of weights ÷ 448 GB/s gives
≤ 112 tokens/s at batch 1. Real engines reach maybe 60–80% of that bound. Measure it and explain the gap.

**Serving memory for Llama-3-70B:** 70.6B × 2 bytes = 141 GB of bf16 weights, so at least two
80 GB GPUs before any KV cache. At 8k context the KV cache is 80·8·128·8192·2·2 ≈ 2.7 GB per
sequence, so a batch of 32 needs another ~86 GB. That is why engines quantize the KV cache, page
it, and share prefixes.

**Cost per million tokens:** `$/hour ÷ (tokens/s × 3600) × 10⁶`. At $2/GPU-hour and 1,000
tokens/s that is $0.56/M. Throughput, not the GPU's price, is the lever:
[founder track](../../tracks/founder/03-unit-economics.md).

## Measure your GPU

`python tools/measure_gpu.py` prints dense matmul TFLOP/s per dtype, copy bandwidth, a
decode-like mat-vec, and your ridge point. Record the results in
[journal/](../../journal/README.md). Then:

1. Predict the mat-vec's effective GB/s *before* running it. Why is it close to the copy bandwidth?
2. Find the matrix size at which bf16 matmul reaches 80% of its best TFLOP/s. Why do small matmuls underperform? (Kernel launch overhead, tile quantization, and too few tiles to fill the SMs.)
3. Compare fp32 with TF32 off and bf16. The ratio tells you how much the tensor cores give you.

## What to implement

| # | functions | tests | what the tests pin down |
|---|---|---|---|
| 1 | `attention_weights`, `mlp_weights`, `count_params` | `test_param_counts_match_released_models`, `test_llama2_70b_is_about_69b`, `test_classic_block_is_12_d_squared`, `test_gqa_shrinks_attention_weights` | exact counts: GPT-2 small 124,439,808; Llama-2-7B 6,738,415,616; Llama-3-8B 8,030,261,248; Mistral-7B 7,241,732,096; Llama-2-70B ≈ 68.98B; MHA plus a 4× MLP is exactly 12d² |
| 2 | `matmul_params`, `flops_per_token` | `test_flops_per_token_gpt2`, `test_attention_term_grows_with_context` | GPT-2 forward = 2 × 123,532,032 plus the attention term; training = 3 × forward ≈ 1.07 × 6N; Llama-3-8B at 128k context costs over 1.5× per token what it costs at 2k |
| 3 | `train_flops`, `chinchilla_optimal` | `test_six_n_d_and_chinchilla` | 70B params × 1.4T tokens = 5.88·10²³ FLOPs; Chinchilla's own budget of 5.76·10²³ gives 69.3B params and 1.386T tokens |
| 4 | `training_days`, `mfu` | `test_training_days_and_mfu` | 6 × 8e9 × 15e12 FLOPs on 1,024 GPUs at 1 PFLOP/s peak and 40% MFU = 20.3 days |
| 5 | `training_memory_bytes` | `test_zero_stages_match_the_zero_paper` | the ZeRO paper's 7.5B-parameter example on 64 GPUs: 120 GB, 31.4 GB, 16.6 GB and 1.875 GB per GPU for stages 0–3 |
| 6 | `activation_bytes_per_layer`, `kv_cache_bytes` | `test_activation_memory_formula`, `test_kv_cache_gqa_vs_mha` | `sbh(34 + 5as/h)` bytes per layer ([arXiv 2205.05198](https://arxiv.org/abs/2205.05198)), `34sbh` with FlashAttention; Llama-2-70B at 32k: 10 GiB with GQA, 80 GiB with MHA, 320 KiB per token |
| 7 | `arithmetic_intensity_matmul`, `ridge_point`, `roofline_seconds`, `decode_tokens_per_second` | `test_matmul_intensity`, `test_roofline_regimes`, `test_decode_upper_bounds` | a 4096³ matmul reaches 1,365 FLOP/byte, a batch-1 mat-vec ≈ 1, batch 64 ≈ 62; 4 GB of weights at 448 GB/s bound decode at 112 tokens/s; batch 32 is over 20× batch 1 |
| 8 | `ring_allreduce_seconds`, `cost_per_million_tokens` | `test_comms_and_cost` | 1 GB over 8 GPUs at 100 GB/s takes 17.5 ms; $2/hour at 1,000 tokens/s costs $0.556 per million tokens |

## Tips

> [!TIP]
> Keep a units column. Most napkin-math errors are units: GB versus GiB (7% apart), bits versus bytes, per-direction versus bidirectional bandwidth (the H100's "900 GB/s" NVLink figure adds both directions), and dense versus sparse peak FLOP/s (spec sheets often lead with 2:4-sparsity numbers, twice the dense rate).

- Count what multiplies. Embedding lookups cost no FLOPs, but the LM head is a matmul even when its weights are tied to the embedding. That is why `count_params` (what you store) and `matmul_params` (what you multiply) differ.
- Write every formula as a product of named factors (`2 · N · T`) before plugging in numbers. Interviewers follow the structure, and you catch a missing factor of 2 by reading it back.
- Sanity-check against a number you know: GPT-2 small is 124M, Llama-3-8B's KV cache is 128 KiB per token in bf16, and an H100 does about 1 PFLOP/s dense bf16.

## Common bugs

- **Counting a tied LM head twice** in `count_params` (GPT-2's 124,439,808 counts the token embedding once), or forgetting it in `matmul_params`.
- **Missing biases and LayerNorm weights** for GPT-2, which the exact counts need; the Llama models have no biases and only RMSNorm scales.
- **Two matrices for SwiGLU:** a gated MLP has three `d × d_ff` matrices.
- **Full-size K and V projections under GQA:** they are `d × (n_kv_heads · head_dim)`, not `d × d`.
- **Mixing GB and GiB:** the KV-cache test expects exactly 10 GiB (2³⁰ bytes each).
- **All-reduce as `bytes / bandwidth`:** a ring moves `2(n−1)/n` of the buffer per device, nearly twice that.

## Drills (answer in under 2 minutes each, then check)

1. A 1B model trained on 20B tokens: total FLOPs?
2. How many H100-days (989 dense bf16 TFLOP/s, 40% MFU) is that?
3. Memory to fine-tune a 7B model with full AdamW in mixed precision, excluding activations?
4. KV cache for Llama-3-8B (32 layers, 8 KV heads, head dim 128) at 128k context, bf16?
5. Batch-1 decode upper bound for Llama-3-8B in bf16 on an H100 (3.35 TB/s)?
6. Why does prefill hit high MFU while decode hits under 5%?
7. At what batch size does a bf16 4096×4096 matmul reach the H100's ridge point (~295 FLOP/byte)?
8. You all-reduce 14 GB of bf16 gradients over 8 GPUs with 450 GB/s per direction. Time?
9. Your 124M model trains at 15k tokens/s (context 1024) on a 40 TFLOP/s GPU. MFU?
10. A founder serves 30M output tokens/day on one $2.50/hour GPU at 800 tokens/s. Utilization and cost per M tokens?

<details><summary>Answers</summary>

1. 6 × 1e9 × 20e9 = 1.2·10²⁰ FLOPs.
2. 1.2e20 / (989e12 × 0.4) ≈ 3.0·10⁵ s ≈ 3.5 H100-days.
3. 16 B × 7e9 = 112 GB, so no single 80 GB GPU. Hence ZeRO/FSDP, 8-bit optimizers, or LoRA.
4. 2 × 32 × 8 × 128 × 131,072 × 2 B = 16 GiB, the same size as the weights.
5. 16 GB / 3.35 TB/s ≈ 4.8 ms per token, so ≲ 210 tokens/s.
6. Prefill multiplies the weights by a T-row matrix (intensity ~T), which is compute-bound. Decode multiplies by one row per sequence (intensity ~batch), which is memory-bound.
7. Intensity ≈ 2·m·k·n / (2(mk + kn + mn)) ≈ m when k = n ≫ m, so m ≈ 300. (Real kernels need a bit more.)
8. 2 × 7/8 × 14 GB / 450 GB/s ≈ 54 ms, about the same compute time as a small step. Hence overlap communication with backward.
9. 15e3 × 6 × 124e6 × 1.07 / 40e12 ≈ 30%. (If your numbers ever imply > 100% MFU, a FLOP count or a timer is wrong.)
10. Capacity is 800 × 86,400 ≈ 69M tokens/day, so 43% utilized. Cost: $60/day ÷ 30M = $2.00 per M tokens, versus $0.87/M at full utilization. Utilization is the business.
</details>

## CPU experiments (no GPU needed)

1. **Check the 12d² rule against PyTorch.** `sum(p.numel() for p in nn.TransformerEncoderLayer(1024, 16, 4096).parameters())` is 12,596,224, which is 13,312 more than 12 × 1024². Account for every extra parameter (the biases and the two LayerNorms).
2. **Count FLOPs with PyTorch's counter.** Run one forward pass of a small lab 05 model under `torch.utils.flop_counter.FlopCounterMode` and compare the total with `flops_per_token × tokens`. Explain any gap, for example whether the counter credits causal masking in attention.
3. **A CPU roofline.** Run `python tools/measure_gpu.py --device cpu`, predict batch-1 decode tokens/s for a 10M-parameter lab 05 model from your measured bandwidth, then time a greedy decode loop and explain the difference.

## GPU scale-up

1. **Decode against the bound.** Qwen2.5-0.5B in bf16 is about 0.99 GB of weights, so 448 GB/s bounds batch-1 decode at about 450 tokens/s. Measure it, then sweep batch sizes 1, 8, 32 and 64 and plot tokens/s. Small models at batch 1 usually land well below the bound because kernel launches, not bandwidth, limit them. Find where throughput stops scaling with batch size and say which roof you hit.
2. **Your MFU.** Take tokens/s from your S1 pretraining log (lab 05), compute MFU with `flops_per_token` and your measured peak, and write down where the rest went: data loading, small matmuls, the optimizer step, Python overhead.

## Stretch

- Add pipeline bubbles, `(p−1)/(m+p−1)`, and tensor-parallel communication per layer to the model. Plan a 3D-parallel layout for a 70B model on 512 GPUs and justify it.
- Add MoE to the calculator ([lab 18](../18_moe/README.md) has the per-layer counts): total vs active parameters, and all-to-all volume per token for expert parallelism. Apply it to DeepSeek-V3 (671B total, 37B active).
- Turn this module into a CLI: `python napkin.py --model llama3-8b --gpu h100 --batch 32 --ctx 8192`.
