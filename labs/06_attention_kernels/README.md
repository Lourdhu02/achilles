# Lab 06 — Attention kernels: online softmax → FlashAttention → Triton

**Build:** (A) the FlashAttention forward and backward algorithms in PyTorch, tile by tile;
(B) a fused softmax and a FlashAttention forward kernel in Triton that runs on your RTX 5060.
**Time:** 12–16 h · **Reads first:** [compute & hardware](../../curriculum/02-compute-and-hardware.md),
[inference §5](../../curriculum/07-inference.md#5-kernels-why-flashattention-wins)
**Run:** `pytest labs/06_attention_kernels` (Triton tests use the CPU interpreter when there is no GPU)

This is the lab that turns "I know FlashAttention exists" into "I can derive it, implement it,
and tell you when it won't help". Inference and performance teams (NVIDIA, frontier-lab inference,
vLLM/SGLang) interview for exactly this.

---

## 1. Why naive attention is slow: it's the memory, not the math

Naive attention writes the T×T score matrix S to HBM, reads it back for the softmax, writes P,
then reads P again to multiply by V. For T = 8192 in bf16 that is 128 MiB **per head per batch
element**, pushed through HBM several times. The FLOPs are fine; the bytes aren't.
On the roofline ([lab 03](../03_napkin_math/README.md)) the softmax is ~1 FLOP/byte, far below
the ridge.

## 2. Online softmax (Milakov & Gimelshein, 2018)

Keep a running max `m` and running sum `l = Σ exp(xᵢ − m)`. When a new block raises the max to
`m'`, every previous term was computed with the wrong offset, so rescale: `l ← l·e^{m−m'} + Σ e^{x−m'}`.
At the end, `logsumexp = m + log l`. One pass, O(1) state per row, no overflow.

## 3. FlashAttention forward (Dao et al., 2022)

Apply the same trick to the *output*. For each query tile keep `(m, l, acc)`, and for each key tile:

```
S = Q_i K_jᵀ · scale             (tile only, lives in SRAM/registers)
m' = max(m, rowmax S)
P̃ = exp(S − m')
acc ← acc · e^{m−m'} + P̃ V_j      (rescale what you had, add the new part)
l   ← l   · e^{m−m'} + rowsum P̃
```

At the end, `O = acc / l` and `LSE = m + log l`. The T×T matrix never exists. HBM traffic drops
from Θ(T·d + T²) to Θ(T²·d²/M) for SRAM size M (their Theorem 2), a large win when d² < M
(d = 64–128, M ≈ 100–200 KB per SM). Causal attention skips whole key tiles above the diagonal,
about half the work.

## 4. FlashAttention backward: the one-line trick

Given `dO`: `dV = Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ∘ (dP − D)` with `Dᵢ = Σⱼ Pᵢⱼ dPᵢⱼ`. Naively D
needs all of P. But `Σⱼ Pᵢⱼ (dOᵢ·Vⱼ) = dOᵢ · Σⱼ Pᵢⱼ Vⱼ = dOᵢ · Oᵢ`, so **D = rowsum(dO ∘ O)**:
O(T) work, no P needed. P itself is recomputed per tile as `exp(S − LSE)`. That recomputation
is why FlashAttention saves only `LSE` (T floats per head) instead of P (T² floats).
Recomputing costs extra FLOPs and saves far more in memory traffic: a trade you should now be
able to justify with the roofline.

## 5. Triton in one page

- You write a **program** that processes one tile. `tl.program_id(axis)` says which one; the launch grid says how many.
- `tl.arange(0, BLOCK)` builds index vectors (BLOCK must be a power of 2 and a `tl.constexpr`). Pointer arithmetic plus `tl.load(ptr, mask=..., other=...)` handles ragged edges.
- `tl.dot` targets the tensor cores. Accumulate in fp32 and cast on store.
- Everything inside a program lives in registers or SRAM. Global memory is touched only by `tl.load` and `tl.store`, so fusion comes for free.
- Debug on CPU with `TRITON_INTERPRET=1`: kernels run as NumPy, you can `print`, and the tests do this automatically when there is no GPU.

## What to implement

1. `online_softmax_stats`
2. `flash_attention_forward` → test against `naive_attention` for block sizes that do *not* divide T
3. `flash_attention_backward` → gradients must match autograd exactly (float64)
4. `_softmax_kernel` (Triton) → `test_triton_softmax`
5. `_flash_fwd_kernel` (Triton) → `test_triton_flash_attention`

## On your RTX 5060 (Blackwell, sm_120)

Benchmark your kernel against PyTorch's `scaled_dot_product_attention` (which dispatches to
FlashAttention or cuDNN kernels):

```python
import torch, triton
from labs._impl import load
ak = load("labs/06_attention_kernels/x.py", "exercise")
q, k, v = (torch.randn(8, 16, 2048, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
for name, fn in {"yours": lambda: ak.triton_flash_attention(q, k, v, causal=True, block_m=64, block_n=64),
                 "sdpa":  lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)}.items():
    ms = triton.testing.do_bench(fn)
    print(f"{name}: {ms:.3f} ms, {2 * 2 * 8 * 16 * 2048**2 * 64 / 2 / ms / 1e9:.1f} TFLOP/s (causal)")
```

Tune `block_m`, `block_n` and `num_warps`, and explain what changes. Typical findings: larger
tiles reuse more but spill registers. A 2× gap to the vendor kernel is normal for a first kernel;
closing it teaches you pipelining (`num_stages`), which is the next step.

## Check yourself

1. Why does FlashAttention not reduce FLOPs, yet run 2–4× faster?
2. For decoding one token against a 32k-token KV cache, why does plain FlashAttention underuse the GPU, and what does FlashDecoding (split-K) change?
3. Why does the backward pass save LSE but not the max m and sum l separately?
4. Would FlashAttention help a d = 1024 head dimension? Why or why not?
5. Your Triton kernel is correct on CPU but produces NaNs on GPU for causal masks with T not divisible by the block. Where do you look?

<details><summary>Answers</summary>

1. Attention at realistic sizes is memory-bound on the softmax and mask traffic. FlashAttention removes Θ(T²) HBM reads and writes, so the time drops even though the FLOPs rise slightly (from recomputation in backward).
2. With one query row, the only parallelism is over batch × heads, often fewer programs than there are SMs, and each program walks 32k keys sequentially. Split-K partitions the keys across programs, each computing a partial `(m, l, acc)`, then a small reduction merges them with the same rescaling rule.
3. LSE = m + log l is all the backward needs to recompute P = exp(S − LSE). It is one number per row instead of two, and it is numerically safe.
4. Much less. The Θ(T²d²/M) bound only beats Θ(T²) when d² < M. At d = 1024 a tile barely fits in SRAM, so reuse collapses.
5. Rows or tiles where every score is masked: `max = −inf` gives `exp(−inf − (−inf)) = NaN`. Check the loop bounds for causal skipping, the masked loads (`other=`), and whether padded rows beyond T are computed with fully masked scores.
</details>

## Stretch

- Write the backward pass in Triton (two kernels: dK/dV with key tiles outer, dQ with query tiles outer) and plug it into the autograd `Function`.
- FlashDecoding: split keys across programs for batch-1 decode and merge the partial `(m, l, acc)` in a second kernel.
- Try FP8 (E4M3) Q and K for the `QKᵀ` product on Blackwell and measure accuracy against speed. (Read FlashAttention-3 for how Hopper and Blackwell do this with warp specialization.)
