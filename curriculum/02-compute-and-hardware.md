# 02 — Compute and hardware

Every design decision in modern AI comes down to FLOPs, bytes and bandwidth. This module teaches you to predict runtime and memory within 2× before measuring: the GPU's memory hierarchy, the roofline model (including for attention), FLOP and memory accounting with the activation formula derived, and what each number-format bit buys.
Lab: [03 napkin math](../labs/03_napkin_math/README.md), plus `python tools/measure_gpu.py` to measure your own roofline.

**Contents:** [1 The GPU in one picture](#1-the-gpu-in-one-picture) · [2 Roofline](#2-roofline) · [3 Roofline for attention](#3-roofline-for-attention) · [4 Number formats](#4-number-formats-what-each-bit-buys) · [5 FLOP accounting](#5-flop-accounting) · [6 Memory accounting](#6-memory-accounting) · [7 Communication](#7-communication) · [8 Reference numbers](#8-reference-numbers) · [9 Worked drills](#9-worked-drills) · [Interview traps](#interview-traps) · [CPU vs GPU notes](#cpu-vs-gpu-notes) · [Check yourself](#check-yourself) · [Visual guides](#visual-guides) · [Read next](#read-next)

---

## 1. The GPU in one picture

A kernel launches a grid of **thread blocks**; each block runs on one **streaming multiprocessor (SM)** and is split into **warps** of 32 threads that execute in lockstep. **Tensor cores** inside each SM do small matrix multiply-accumulates at 8–16× the rate of the ordinary floating-point units (H100: ~990 dense bf16 TFLOP/s on tensor cores versus ~67 TFLOP/s of non-tensor fp32).

```
                    per SM                          whole chip (H100 SXM, 132 SMs)
 ┌──────────────────────────────────────┐
 │ registers        256 KB   ~1 cycle   │
 │ shared mem / L1  up to 228 KB        │  ← you control this (tiles in Triton/CUDA)
 │ tensor cores + fp32/int units        │
 └──────────────────────────────────────┘
                     │
            L2 cache, 50 MB, shared by all SMs
                     │
            HBM3, 80 GB, ~3.35 TB/s          ← most kernels are limited here
                     │
    NVLink ~900 GB/s (both directions) · PCIe 5.0 x16 ~64 GB/s per direction
                     │
            network: InfiniBand NDR 400 Gb/s ≈ 50 GB/s per port
```

Each level down is roughly an order of magnitude slower or smaller than the one above. **Most kernels are limited by how fast they move bytes, not by how fast they do math.** Every optimization you will meet (fusion, tiling, FlashAttention, quantization, KV-cache compression) is a way to move fewer bytes through the slow levels.

A third limit sits beside compute and bandwidth: **overhead**. Launching a kernel costs a few microseconds of CPU and driver time, and Python adds more per op. A 512×512 matmul needs ~0.3 µs of compute and ~0.5 µs of memory traffic on an H100, so it is dominated by launch overhead. That is why small models look slow on big GPUs and why CUDA graphs and `torch.compile` help them.

## 2. Roofline

$$t \;\ge\; \max\!\left(\frac{\text{FLOPs}}{\text{peak FLOP/s}},\; \frac{\text{bytes moved}}{\text{bandwidth}}\right)$$

**Arithmetic intensity** $I$ = FLOPs ÷ bytes moved. The **ridge point** = peak FLOP/s ÷ bandwidth: kernels with $I$ below it are memory-bound, above it compute-bound. H100 SXM in bf16: $989\text{e}12 / 3.35\text{e}12 \approx 295$ FLOP/byte.

```
 attainable
 FLOP/s   │                  ridge ≈ 295 (H100 bf16)
   peak ──┼────────────────────●━━━━━━━━━━━━━━━━━━━━━━  compute-bound
          │                 ╱
          │              ╱     slope = bandwidth
          │           ╱
          │        ╱   memory-bound
          │     ╱
          └──┴──────┴──────────┴──────────────┴──────▶ intensity (FLOP/byte)
             1      64        295            1365
         decode   naive     ridge        4096³ matmul
         matvec   attention
```

**Matmul intensity, derived.** $(m\times k)(k\times n)$ in bf16 does $2mkn$ FLOPs and, at minimum, reads $A$ and $B$ and writes $C$: $2(mk + kn + mn)$ bytes. So
$$I = \frac{mkn}{mk + kn + mn}.$$
- Square, $m = k = n$: $I = n/3$. For $n = 4096$, $I \approx 1365$: compute-bound. On an H100, compute takes $2\cdot4096^3/989\text{e}12 = 139$ µs; memory traffic $3\cdot4096^2\cdot2/3.35\text{e}12 = 30$ µs.
- **Decode** (one token, $m = 1$, $k = n = d$): $I \approx 1$ FLOP/byte. Every weight is read to do one multiply-add. Hopelessly memory-bound.
- **Batched decode** ($m = B \ll d$): $I = Bd/(2B + d) \approx B$. Batching is how you climb the roofline. For $d = 4096$ the H100 ridge needs $B \approx 345$ exactly, or ≈ 295 in the $B \ll d$ approximation.
- **Elementwise ops** (GELU, residual add, dropout, norms): a few FLOPs per element, 4+ bytes per element moved. $I < 1$. A GELU over a 4096×4096 bf16 tensor moves 67 MB, 20 µs on an H100, while its math would take nanoseconds. Hence **fusion**: do many elementwise ops per trip to memory.

> [!TIP]
> When profiling, compute *achieved* FLOP/s and *achieved* GB/s for each big kernel. If one of them is within 70–80% of its roof, that kernel is done; optimize something else. If both are low, the kernel is overhead-bound or badly tiled, and that is where the easy wins are.

## 3. Roofline for attention

One head, sequence length $T$, head dimension $d$ (128 in most LLMs), bf16, forward pass, non-causal for simplicity.

**FLOPs:** $S = QK^\top$ costs $2T^2d$ and $O = PV$ costs $2T^2d$, so $4T^2d$. The softmax adds ~5 elementwise ops per score, small in FLOPs but not free: FlashAttention-3 (Shah et al., 2024) notes that an H100 SXM5 has 989 TFLOP/s of bf16 matmul but only 3.9 TFLOP/s of special-function throughput for `exp`.

**Naive attention** writes $S$ to HBM, reads it for the softmax, writes $P$, reads it for $PV$:
$$\text{bytes} = \underbrace{3\cdot 2Td}_{Q,K,V} + \underbrace{4\cdot 2T^2}_{S,\,P\ \text{write+read}} + \underbrace{2Td}_{O} \quad\Rightarrow\quad I = \frac{4T^2d}{8T^2 + 8Td} \approx \frac{d}{2} = 64.$$
At $T = 4096$ the exact value is 62 FLOP/byte: memory-bound on an H100 (ridge 295), and the $T^2$ memory for $S$ and $P$ is often the bigger problem.

**FlashAttention** tiles $Q$, $K$, $V$ into SRAM and never writes $S$ or $P$ to HBM (online softmax, [lab 06](../labs/06_attention_kernels/README.md)):
- Ideal traffic, each of $Q$, $K$, $V$, $O$ moved once: $8Td$ bytes, so $I = 4T^2d/8Td = T/2$ = 2048 at $T = 4096$. Compute-bound.
- Honest traffic: each block of $B_r$ query rows streams all of $K$ and $V$ from memory, so $K$/$V$ are read $T/B_r$ times and $I \approx B_r$ (about 120 at $B_r = 128$, $T = 4096$). Thread blocks working on the same head run at the same time and share those $K$/$V$ tiles through L2, so real HBM traffic is far lower. That L2 reuse is why FlashAttention reaches a large fraction of the matmul peak.
- Causal masking halves the FLOPs, and skipping fully masked tiles roughly halves the traffic; the intensity barely changes.

**Decode attention** is different. One new query per sequence attends to a KV cache of length $T$: $4Td$ FLOPs per head against $2\cdot 2Td$ bytes of $K$ and $V$ read, so **$I \approx 1$**. Batching does not help, because every sequence has its own cache. Grouped-query attention with $g$ query heads per KV head reuses each loaded $K$/$V$ $g$ times, so $I \approx g$ (Llama-3-8B: $32/8 = 4$). This is the arithmetic behind GQA, MQA, MLA and KV-cache quantization ([04](04-transformers.md), [07](07-inference.md)).

**Worked: Llama-3-8B decode at 8k context on an H100** (bf16, 16.1 GB of weights, KV cache 1 GiB per sequence). Each step reads the weights once plus every sequence's cache:

| batch | bytes per step | step time at 3.35 TB/s | tokens/s (upper bound) | share of traffic that is KV cache |
|---|---|---|---|---|
| 1 | 17.1 GB | 5.1 ms | ~196 | 6% |
| 8 | 24.7 GB | 7.4 ms | ~1,090 | 35% |
| 32 | 50.4 GB | 15.1 ms | ~2,130 | 68% |
| 64 | 84.8 GB | 25.3 ms | ~2,530 | 81% (and it no longer fits in 80 GB) |

Weights are amortized by batching; KV-cache reads are not. Past batch ~32 the cache dominates, which is why serving engines quantize it, page it and share prefixes.

## 4. Number formats: what each bit buys

A float is sign · $2^{\text{exponent}}$ · $1.\text{mantissa}$. **Exponent bits buy range; mantissa bits buy precision.** "Epsilon" below is the gap between 1 and the next representable number.

| format | sign/exp/mantissa | max | epsilon | typical use |
|---|---|---|---|---|
| fp32 | 1/8/23 | 3.4·10³⁸ | 1.2·10⁻⁷ | master weights, optimizer state, reductions |
| tf32 | 1/8/10 | 3.4·10³⁸ | 9.8·10⁻⁴ | fp32 matmuls on tensor cores (internal only) |
| bf16 | 1/8/7 | 3.4·10³⁸ | 7.8·10⁻³ | default training and inference: fp32's range, low precision |
| fp16 | 1/5/10 | 65,504 | 9.8·10⁻⁴ | needs loss scaling for training; common for inference |
| fp8 e4m3 | 1/4/3 | 448 | 0.125 | forward activations and weights, with per-tensor or per-block scales |
| fp8 e5m2 | 1/5/2 | 57,344 | 0.25 | gradients (more range, less precision) |
| fp4 e2m1 (MXFP4, NVFP4) | 1/2/1 | 6 | 0.5 | Blackwell inference and emerging training; only 0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives, so a shared scale per small block does the heavy lifting (32 elements with a power-of-two scale in MXFP4; 16 elements with an fp8 scale in NVFP4) |
| int8 / int4 | integers + scales | | | weight-only and KV-cache quantization ([lab 13](../labs/13_quantization/README.md)) |

What the bits mean in practice (try these in a REPL):
- `torch.tensor(1.0, dtype=torch.bfloat16) + 1e-3` returns exactly 1.0, because the next bf16 number after 1 is 1.0078. A typical Adam update is ~1e-3 to 1e-4 of the weight's size, so **bf16 weights silently drop most updates**. Hence fp32 master weights.
- `torch.tensor(256.0, dtype=torch.bfloat16) + 1` returns 256.0. Summing thousands of values in bf16 stalls once the running total is large, so tensor cores and well-written kernels **accumulate in fp32**.
- fp16's smallest normal number is 6.1·10⁻⁵ and its smallest subnormal ~6·10⁻⁸. Small gradients flush to zero, so fp16 training multiplies the loss by a large scale before backward and divides afterwards (**loss scaling**). bf16 has fp32's exponent range and needs none.
- fp8 training (DeepSeek-V3 style) keeps fp32 accumulation, uses fine-grained block scales (per 1×128 activation tile and 128×128 weight block in DeepSeek-V3) so one outlier does not wreck a whole tensor, and keeps sensitive ops (norms, softmax, the optimizer) in higher precision.

## 5. FLOP accounting

**Forward ≈ 2N FLOPs per token**, where $N$ counts the weights that take part in matmuls, including the LM head. Each weight does one multiply and one add per token.

**Training ≈ 6N per token.** For each $Y = XW$, backward computes $\bar{X} = \bar{Y}W^\top$ and $\bar{W} = X^\top\bar{Y}$ ([01 §2.1](01-math.md#21-matmul-derived)), each the same size as the forward matmul. So backward ≈ 2× forward and training ≈ 3× forward. Activation checkpointing adds one more forward: ≈ 8N.

**Attention adds $2LTd$ per token forward** (causal). Per layer, the query at position $t$ dots with $t$ keys ($2td$ FLOPs) and mixes $t$ values ($2td$); averaged over positions $t \approx T/2$, that is $2Td$ per layer. Multiply by 3 for training.

| model, context | $2N$ (matmul weights) | attention term | attention share |
|---|---|---|---|
| GPT-2 small, $T$ = 1,024 | 247 MFLOP | 19 MFLOP | +7.6% (the "×1.07" in lab 03) |
| Llama-3-8B, $T$ = 2,048 | 15.0 GFLOP | 0.54 GFLOP | +3.6% |
| Llama-3-8B, $T$ = 8,192 | 15.0 GFLOP | 2.1 GFLOP | +14% |
| Llama-3-8B, $T$ = 131,072 | 15.0 GFLOP | 34.4 GFLOP | +229% |

**MFU** (model FLOPs utilization) = achieved model FLOP/s ÷ peak, counting only the FLOPs the model needs (6N plus attention). Well-run large training jobs reach roughly 35–55%; the Llama 3 paper reports 38–43% for its 405B model on up to 16k H100s. Small models and long sequences go lower. **HFU** (hardware FLOPs utilization) also counts recomputation, so HFU ≥ MFU; quote MFU when comparing runs.

## 6. Memory accounting

**Training state: 16 bytes/param** with mixed-precision Adam: bf16 weights (2) + bf16 grads (2) + fp32 master weights (4) + Adam $m$ (4) + Adam $v$ (4). PyTorch autocast with fp32 parameters also lands at 16 (fp32 weights 4 + fp32 grads 4 + Adam 8) plus transient bf16 copies. A 7B model needs 112 GB before activations: more than one 80 GB GPU, hence ZeRO/FSDP, 8-bit optimizers or LoRA.

**Activations, derived** (Korthikanti et al., 2022, [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)). For one GPT-3-style layer in 16-bit, with sequence $s$, micro-batch $b$, hidden $h$, heads $a$, counting what backward needs:

| stored tensor | bytes |
|---|---|
| attention: LayerNorm output (input to QKV) | $2sbh$ |
| $Q$ and $K$ (for $QK^\top$ backward) | $4sbh$ |
| softmax output $P$ | $2as^2b$ |
| attention dropout mask | $as^2b$ |
| attention dropout output (for $PV$ backward) | $2as^2b$ |
| $V$ | $2sbh$ |
| input to output projection | $2sbh$ |
| dropout mask after attention | $sbh$ |
| MLP: input to first linear | $2sbh$ |
| GELU input ($4h$ wide) | $8sbh$ |
| GELU output (input to second linear) | $8sbh$ |
| dropout mask after MLP | $sbh$ |
| two LayerNorm inputs | $4sbh$ |
| **total** | $\mathbf{34sbh + 5as^2b}$ |

FlashAttention never stores the three $s\times s$ tensors (it keeps only a per-row log-sum-exp and recomputes $P$ in backward), leaving about $34sbh$. Modern blocks (no dropout, SwiGLU, RMSNorm) change the constant; recount for your architecture using the same method.

**Worked: GPT-2 small, $s = 1024$, $b = 8$, 12 layers.** $34sbh$ = 204 MiB per layer; $5as^2b$ = 480 MiB per layer. Without FlashAttention: 12 × 684 MiB ≈ **8.0 GiB** of activations, which does not fit on an 8 GB card next to anything else. With it: 12 × 204 MiB ≈ **2.4 GiB**. The logits are a separate trap: $s\cdot b\cdot V$ in fp32 for the loss is $1024\cdot8\cdot50{,}257\cdot4$ bytes = 1.5 GiB, and its gradient is the same size again. Compute the loss in chunks when memory is tight.

**Activation checkpointing** stores only each layer's input ($2sbh$) and recomputes the rest during backward, for one extra forward pass (≈ +33% compute, 6N → 8N). **Selective recomputation** recomputes only the attention $s^2$ terms, which are large in memory but cheap in FLOPs.

**Inference memory:** weights (bytes/param × N) plus KV cache $2\cdot L\cdot H_{kv}\cdot d_h\cdot T\cdot B\cdot\text{bytes}$. Llama-3-8B in bf16: 128 KiB per token, 1 GiB per 8k-token sequence.

> [!TIP]
> `torch.cuda.max_memory_allocated()` after one training step, compared with your prediction of weights + grads + optimizer + activations, is the fastest way to find a memory bug. A gap of 2× usually means an fp32 copy you did not plan for (logits, a loss computed in fp32, an optimizer state in fp32 when you expected 8-bit). Use `torch.cuda.memory._record_memory_history()` and the memory-snapshot viewer when the gap is not obvious.

## 7. Communication

**Ring all-reduce** moves $2(n-1)/n$ of the buffer per device: a reduce-scatter ($n-1$ steps, each sending $1/n$ of the buffer) followed by an all-gather ($n-1$ more). Time ≈ $2\frac{n-1}{n}\cdot\frac{\text{bytes}}{\text{bandwidth}}$ + latency terms, nearly independent of $n$.

**Worked:** 2 GB of bf16 gradients (a 1B model) over 8 GPUs: 55 ms over PCIe 5.0 (~64 GB/s per direction), 7.8 ms over H100 NVLink (~450 GB/s per direction). A forward-backward step for that model on 8 GPUs can take tens of milliseconds, so on PCIe communication can dominate unless it overlaps with the backward pass.

| collective | what it does | where it appears |
|---|---|---|
| all-reduce | everyone ends with the sum | data-parallel gradients, tensor-parallel partial sums |
| reduce-scatter | everyone ends with one shard of the sum | ZeRO-2/3 and FSDP gradients |
| all-gather | everyone ends with all shards | ZeRO-3/FSDP weights before each layer |
| all-to-all | everyone sends a different piece to everyone | MoE expert parallelism |
| send / recv | point to point | pipeline parallelism |

An all-reduce is a reduce-scatter plus an all-gather, which is why ZeRO-2 costs the same communication as plain data parallelism ([lab 09](../labs/09_parallelism/README.md)).

## 8. Reference numbers

Order of magnitude, dense (no sparsity), as of September 2026. Verify spec sheets before quoting; vendors often headline the 2×-sparsity number.

| | dense bf16 | memory | bandwidth | ridge (FLOP/byte) |
|---|---|---|---|---|
| A100 80GB SXM | ~312 TFLOP/s | 80 GB HBM2e | ~2.0 TB/s | ~150 |
| H100 SXM | ~990 TFLOP/s | 80 GB HBM3 | ~3.35 TB/s | ~295 |
| B200 | ~2.2 PFLOP/s | ~180 GB HBM3e | ~8 TB/s | ~280 |
| **your RTX 5060** | measure it (`tools/measure_gpu.py`) | 8 GB GDDR7 | ~448 GB/s | your TFLOP/s ÷ 0.448 |

Interconnects: NVLink (H100) ~900 GB/s total per GPU (both directions); PCIe 5.0 x16 ~64 GB/s per direction; InfiniBand NDR 400 Gb/s ≈ 50 GB/s per port.

**RTX 5060 setup note:** Blackwell consumer cards are compute capability 12.0 (sm_120) and need PyTorch built for CUDA 12.8+ and an R570+ driver. See [SETUP.md](../SETUP.md). Laptop GPUs also run under a power limit, so expect measured peaks below desktop spec sheets, and re-measure when plugged in versus on battery.

## 9. Worked drills

Ten drills with answers are in [lab 03](../labs/03_napkin_math/README.md#drills-answer-in-under-2-minutes-each-then-check). Aim to answer each in under 2 minutes, out loud, before checking.

---

## Interview traps

- **"GPUs are fast at math, so make the math cheaper."** Most kernels are memory-bound; cut bytes first. FlashAttention does *more* FLOPs (recomputation in backward) and is faster.
- **Using peak FLOP/s for decode.** Batch-1 decode is bandwidth-bound: tokens/s ≤ bandwidth ÷ weight bytes.
- **"Batching makes decode attention efficient."** It amortizes weight reads, not KV-cache reads; each sequence's cache is read once per step regardless of batch.
- **Quoting sparse TFLOP/s.** H100 "1,979 TFLOP/s bf16" assumes 2:4 structured sparsity; the dense number is half.
- **GB vs GiB, bits vs bytes.** 8 GB is 7.45 GiB; InfiniBand is quoted in gigabits.
- **"6N per token" at long context.** At 128k tokens, attention exceeds the weight FLOPs for an 8B model.
- **Forgetting the logits** in activation memory, and the fp32 copies the loss creates.
- **MFU above 100%.** Your FLOP count or timer is wrong (usually a missing `synchronize`, or counting a tied embedding twice).

## CPU vs GPU notes

- **Timing.** CUDA calls are asynchronous: the CPU enqueues kernels and returns. Time GPU code with `torch.cuda.synchronize()` before reading the clock, or with CUDA events, and discard the first few iterations (cuBLAS initialization, autotuning, `torch.compile`). CPU code runs synchronously and needs no sync.
- **CPU rooflines exist too.** A laptop with 8 cores at 3 GHz, two 256-bit FMA units per core, has an fp32 peak of about $8\cdot3\text{e}9\cdot2\cdot8\cdot2 \approx 0.77$ TFLOP/s; dual-channel DDR5-5600 gives about 90 GB/s. The ridge is ~9 FLOP/byte, so CPUs become compute-bound at much lower intensity than GPUs. `python tools/measure_gpu.py` falls back to the CPU with smaller sizes; run it and compare with this estimate.
- **Everything in lab 03 is pure Python** and runs anywhere. The tests pin exact parameter counts, so no GPU is needed to master this module.
- **Windows laptops:** when VRAM is full, recent NVIDIA Windows drivers can spill allocations into system RAM instead of raising an out-of-memory error, and the run silently becomes several times slower. If a run slows down sharply near the memory limit, check the "CUDA - Sysmem Fallback Policy" setting in the NVIDIA Control Panel.

## Check yourself

<details><summary>1. Derive the arithmetic intensity of an (m×k)(k×n) bf16 matmul and evaluate it for decode.</summary>

FLOPs $2mkn$; minimum bytes $2(mk + kn + mn)$; $I = mkn/(mk + kn + mn)$. For $m = 1$, $k = n = 4096$: $I = 4096^2/(4096 + 4096^2 + 4096) \approx 1$. Memory-bound on every GPU.
</details>

<details><summary>2. What is the H100's ridge point in bf16, and what batch size makes a 4096×4096 decode matmul compute-bound?</summary>

$989/3.35 \approx 295$ FLOP/byte. With $I = Bd/(2B + d)$ and $d = 4096$, $I = 295$ at $B \approx 345$ (≈ 295 if you approximate $I \approx B$). Real kernels need somewhat more because of tiling and imperfect overlap.
</details>

<details><summary>3. Why is naive attention memory-bound while FlashAttention is compute-bound, for the same FLOPs?</summary>

Naive attention writes and reads the $T\times T$ scores and probabilities through HBM: $I \approx d/2 = 64$ for $d = 128$. FlashAttention keeps tiles in SRAM and moves only $Q$, $K$, $V$, $O$ (plus K/V re-reads that L2 largely absorbs), so $I$ grows with $T$ (up to $T/2$).
</details>

<details><summary>4. Why does GQA speed up decoding but not prefill?</summary>

Decode attention reads the whole KV cache for one query per head: $I \approx 1$, bandwidth-bound. Sharing each KV head across $g$ query heads cuts the bytes by $g$ with the same FLOPs. Prefill attention is already compute-bound (many queries per loaded $K$/$V$), and GQA does not change its FLOPs.
</details>

<details><summary>5. Derive the 34sbh activation constant, or at least its three biggest terms.</summary>

MLP: GELU input and output at $4h$ wide in 16-bit, $8sbh$ each ($16sbh$), plus the MLP input $2sbh$ and a dropout mask $sbh$ = $19sbh$. Attention: QKV input, Q, K, V, projection input ($2 + 4 + 2 + 2 = 10sbh$) plus a mask ($sbh$) = $11sbh$. Two LayerNorm inputs = $4sbh$. Total $34sbh$, plus $5as^2b$ for the softmax output, dropout mask and dropout output.
</details>

<details><summary>6. Why keep fp32 master weights if you compute in bf16?</summary>

bf16 has 8 bits of precision: the gap after 1.0 is 0.0078, so an update of 1e-3 of a weight's size rounds away. The fp32 master copy accumulates small updates; the bf16 copy is only for fast matmuls.
</details>

<details><summary>7. Why is bf16 preferred over fp16 for training, when fp16 has more mantissa bits?</summary>

fp16 has 5 exponent bits: values above 65,504 overflow and gradients below ~6e-8 underflow, so training needs dynamic loss scaling and still breaks on spikes. bf16 has fp32's range, so no scaling is needed; the lost precision is recovered by fp32 accumulation and master weights.
</details>

<details><summary>8. Gradient all-reduce of a 1B model on 8 GPUs over PCIe: roughly how long, and what do you do about it?</summary>

$2\cdot\frac78\cdot 2\text{ GB}/64\text{ GB/s} \approx 55$ ms. Overlap it with the backward pass (bucketed all-reduce, as DDP does), accumulate gradients over more micro-batches per all-reduce, or shard with ZeRO so each GPU communicates less state.
</details>

<details><summary>9. A small model trains at the same tokens/s on an H100 as on an A100. Why?</summary>

It is overhead-bound, not compute- or bandwidth-bound: kernels are too small to fill 132 SMs, and CPU launch overhead and Python dominate. Larger batches, `torch.compile`, CUDA graphs and fused kernels help; a faster GPU does not.
</details>

<details><summary>10. You measure 120% MFU. What went wrong?</summary>

Either the timer stopped before the GPU finished (missing `torch.cuda.synchronize()`), or the FLOP count is too high (counting a tied embedding twice, using total rather than active parameters for an MoE), or the denominator is the wrong peak (the fp32 or TF32 peak for a run that computes in bf16).
</details>

## Visual guides

- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): the compute / memory bandwidth / overhead split in §1–2, with diagrams. Read it first.
- Google DeepMind, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the [roofline chapter](https://jax-ml.github.io/scaling-book/roofline/) works through matmul and attention intensity with figures (TPU-flavoured, and the method is identical for GPUs).
- Modal, [GPU Glossary](https://modal.com/gpu-glossary): SMs, warps, tensor cores and the memory hierarchy, one illustrated entry each.
- kipply, [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/): §3's decode arithmetic, worked for real model sizes.

More per-topic visuals are collected in the [library](../library/README.md).

## Read next

- [03 Deep learning](03-deep-learning.md): where the bytes in §6 come from, and what mixed precision does to training.
- [Lab 03](../labs/03_napkin_math/README.md): implement every formula in this module against exact parameter counts.
- Deeper: NVIDIA, [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) (tile and wave quantization); Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (parallelism with measured numbers); Simon Boehm, [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM); Williams, Waterman & Patterson, *Roofline* (CACM, 2009).
