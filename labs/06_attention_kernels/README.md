# Lab 06 — Attention kernels: online softmax → FlashAttention → Triton

**Build:** (A) the FlashAttention forward and backward algorithms in PyTorch, tile by tile; (B) a fused softmax and a FlashAttention forward kernel in Triton that runs on your RTX 5060.<br>
**Time:** 12–16 h · **Reads first:** [compute & hardware](../../curriculum/02-compute-and-hardware.md), [inference §5](../../curriculum/07-inference.md#5-kernels-why-flashattention-wins)<br>
**Run:** `pytest labs/06_attention_kernels` (your code) · `pytest labs/06_attention_kernels --impl=solution` (reference). Triton tests use the CPU interpreter when there is no GPU.

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

Why the rescale is exact: `Σ e^{x−m'} = e^{m−m'} · Σ e^{x−m}`. Subtracting any constant from every
exponent and multiplying back is an identity; the max is just the constant that keeps every
exponent ≤ 0, so nothing overflows.

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

```
            keys →   tile 0   tile 1   tile 2   tile 3
queries  tile 0   [ diag  ][  skip  ][  skip  ][  skip  ]
   ↓     tile 1   [ full  ][ diag   ][  skip  ][  skip  ]
         tile 2   [ full  ][ full   ][ diag   ][  skip  ]
         tile 3   [ full  ][ full   ][ full   ][ diag   ]
full: no mask needed · diag: element-wise causal mask · skip: never loaded
```

## 4. FlashAttention backward: the one-line trick

Given `dO`: `dV = Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ∘ (dP − D)` with `Dᵢ = Σⱼ Pᵢⱼ dPᵢⱼ`. Naively D
needs all of P. But `Σⱼ Pᵢⱼ (dOᵢ·Vⱼ) = dOᵢ · Σⱼ Pᵢⱼ Vⱼ = dOᵢ · Oᵢ`, so **D = rowsum(dO ∘ O)**:
O(T) work, no P needed. P itself is recomputed per tile as `exp(S − LSE)`. That recomputation
is why FlashAttention saves only `LSE` (T floats per head) instead of P (T² floats).
Recomputing costs extra FLOPs and saves far more in memory traffic: a trade you should now be
able to justify with the roofline.

Then `dQ = dS K · scale` and `dK = dSᵀ Q · scale`. The reference loops over key tiles (outer) and
query tiles (inner), so `dK` and `dV` for a key tile accumulate in one place while `dQ` receives
contributions from every key tile.

## 5. Triton in one page

- You write a **program** that processes one tile. `tl.program_id(axis)` says which one; the launch grid says how many.
- `tl.arange(0, BLOCK)` builds index vectors (BLOCK must be a power of 2 and a `tl.constexpr`). Pointer arithmetic plus `tl.load(ptr, mask=..., other=...)` handles ragged edges.
- `tl.dot` targets the tensor cores. Accumulate in fp32 and cast on store.
- Everything inside a program lives in registers or SRAM. Global memory is touched only by `tl.load` and `tl.store`, so fusion comes for free.
- Debug on CPU with `TRITON_INTERPRET=1`: kernels run as NumPy, you can `print`, and the tests do this automatically when there is no GPU.

The reference kernel's layout: grid `(cdiv(T, BLOCK_M), B·H)`, so program `(m, bh)` owns `BLOCK_M`
query rows of one (batch, head). It loads its Q tile once, loops over K/V tiles, and writes O and
LSE once. With contiguous `(B, H, T, D)` inputs, `stride_bh = T·D` and `stride_t = D`.

## What to implement

Work in this order; each step has a test that must pass before you move on.

| # | function | test(s) | what the test pins down |
|---|---|---|---|
| 1 | `online_softmax_stats` | `test_online_softmax_matches_logsumexp`, `test_online_softmax_handles_growing_max` | `m + log l` equals `logsumexp`; the rescale survives a max that jumps from −40 to 1000 |
| 2 | `flash_attention_forward` | `test_flash_forward_matches_naive` (causal and not; block sizes 7, 11, 16, 32 that do *not* divide T), `test_flash_forward_cross_attention_lengths` | output and LSE match `naive_attention` in float64; T_q ≠ T_k works |
| 3 | `flash_attention_backward` | `test_flash_backward_matches_autograd`, `test_autograd_function_end_to_end` | dq, dk, dv match autograd exactly (float64), causal and not |
| 4 | `_softmax_kernel` (Triton) | `test_triton_softmax` | rows of 37, 128 and 1000 columns (not powers of two) |
| 5 | `_flash_fwd_kernel` (Triton) | `test_triton_flash_attention` | T = 16 and 37, causal and not, D = 32; LSE shape `(B, H, T)` |

`naive_attention`, the `FlashAttention` autograd wrapper and the Python launchers
(`triton_softmax`, `triton_flash_attention`) are given.

## Predict before you measure

Write these in your journal before running the benchmark below:

1. Bytes of the full score matrix for `B=8, H=16, T=2048` in bf16. (Answer: 8·16·2048²·2 B = 1 GiB.)
2. Forward FLOPs of causal attention at that shape: `4·B·H·T²·D / 2`. At what time would your GPU's measured peak finish it?
3. Which will be larger for `T = 512`: the gain from fusion or the launch overhead? Why does FlashAttention's advantage grow with T?

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

(`load` takes any path inside the lab directory; `x.py` only locates the folder.) Tune
`block_m`, `block_n` and `num_warps`, and explain what changes. Typical findings: larger
tiles reuse more but spill registers. A 2× gap to the vendor kernel is normal for a first kernel;
closing it teaches you pipelining (`num_stages`), which is the next step.

Scale-up sweep: repeat for T ∈ {512, 1024, 2048, 4096, 8192} and plot TFLOP/s for yours, SDPA and
`naive_attention` in bf16. The naive curve falls off (and eventually runs out of memory) as T²
bytes dominate; the tiled curves rise toward a plateau. That plot is the whole argument of this
lab in one picture.

On Windows, Triton needs WSL2 or the community `triton-windows` package (see
[SETUP.md](../../SETUP.md)). Part A needs only PyTorch.

> [!TIP]
> Debug Part B on CPU first: `TRITON_INTERPRET=1 pytest labs/06_attention_kernels -k triton`.
> In the interpreter you can `print` tensors inside the kernel. Once it passes, run on the GPU,
> where tf32 dot products make the tests use a looser tolerance (2e-3).

## Common bugs

- **Forgetting to rescale `acc`** when the max changes: outputs are right only when the first tile holds the row max. The growing-max test and odd block sizes catch it.
- **`exp(−inf − (−inf)) = NaN`:** a row whose every score in a tile is masked has `m = −inf`. In Part A this is safe only because key tiles are visited from 0 upward: every real row sees key 0 first, so `m` is already finite when a later tile masks the whole row. Iterate key tiles in another order (as a split-K kernel does) and you must guard it. In Triton, padded rows beyond T and masked columns must be handled with `mask=`/`other=` and `tl.where`, not by hoping.
- **Causal skip off by one:** the last key tile a query tile needs is the one containing its last row, `i1 − 1`. Skipping one tile too many passes small tests and fails `T=33, bq=7, bk=11`.
- **Masking only in the forward:** the backward must apply the same causal mask before `exp(S − LSE)`.
- **Using `m + log l` from the wrong dtype:** keep statistics in fp32 (or float64 in tests); bf16 LSE loses precision in the backward.
- **Non-power-of-two `BLOCK` in Triton:** `tl.arange` requires a power of two; the launcher uses `triton.next_power_of_2(n_cols)` for the softmax.
- **Wrong strides:** the launcher makes inputs contiguous. If you pass transposed views to your own launcher, `stride_t` is no longer `D`.

## Check yourself

1. Why does FlashAttention not reduce FLOPs, yet run 2–4× faster?
2. For decoding one token against a 32k-token KV cache, why does plain FlashAttention underuse the GPU, and what does FlashDecoding (split-K) change?
3. Why does the backward pass save LSE but not the max m and sum l separately?
4. Would FlashAttention help a d = 1024 head dimension? Why or why not?
5. Your Triton kernel is correct on CPU but produces NaNs on GPU for causal masks with T not divisible by the block. Where do you look?
6. Where does `D = rowsum(dO ∘ O)` come from, and why does it matter for memory?
7. With GQA (8 KV heads, 32 query heads), how would you change the kernel so a K/V tile loaded once serves several query heads?

<details><summary>Answers</summary>

1. Attention at realistic sizes is memory-bound on the softmax and mask traffic. FlashAttention removes Θ(T²) HBM reads and writes, so the time drops even though the FLOPs rise slightly (from recomputation in backward).
2. With one query row, the only parallelism is over batch × heads, often fewer programs than there are SMs, and each program walks 32k keys sequentially. Split-K partitions the keys across programs, each computing a partial `(m, l, acc)`, then a small reduction merges them with the same rescaling rule.
3. LSE = m + log l is all the backward needs to recompute P = exp(S − LSE). It is one number per row instead of two, and it is numerically safe.
4. Much less. The Θ(T²d²/M) bound only beats Θ(T²) when d² < M. At d = 1024 a tile barely fits in SRAM, so reuse collapses.
5. Rows or tiles where every score is masked: `max = −inf` gives `exp(−inf − (−inf)) = NaN`. Check the loop bounds for causal skipping, the masked loads (`other=`), and whether padded rows beyond T are computed with fully masked scores.
6. `Dᵢ = Σⱼ Pᵢⱼ dPᵢⱼ = Σⱼ Pᵢⱼ (dOᵢ · Vⱼ) = dOᵢ · Oᵢ`. It turns a T×T reduction into an O(T·d) one computed from tensors already saved, so the backward never needs P stored.
7. Map the program to a KV head and loop over (or stack) the query heads that share it, so each K/V tile loaded from HBM is reused by 4 query heads. That raises arithmetic intensity by the group size, which is the point of GQA at decode time.
</details>

## Stretch

- Write the backward pass in Triton (two kernels: dK/dV with key tiles outer, dQ with query tiles outer) and plug it into the autograd `Function`.
- FlashDecoding: split keys across programs for batch-1 decode and merge the partial `(m, l, acc)` in a second kernel. Measure against your forward kernel with T_q = 1 and T_k = 32k.
- Add `num_stages` software pipelining and autotuning (`@triton.autotune` over block sizes and warps). Record which configuration wins at each T and why.
- Try FP8 (E4M3) Q and K for the `QKᵀ` product on Blackwell and measure accuracy against speed. (Read FlashAttention-3 for how Hopper and Blackwell do this with warp specialization.)
