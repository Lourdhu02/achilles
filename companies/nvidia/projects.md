# Projects for NVIDIA's AI teams

Eight portfolio projects sized for an 8 GB RTX 5060 laptop GPU (consumer Blackwell, compute capability 12.0, `sm_120`) or a Colab GPU. Each maps to a kind of NVIDIA team, builds on labs in this repo, and ends in a public write-up with measured numbers.
Two finished projects with honest roofline analysis beat eight half-done ones.

> [!IMPORTANT]
> Every result must state its configuration: GPU, driver and CUDA versions, library versions, shapes, dtype, warm-up and iteration counts, and whether clocks were locked. Time GPU work with CUDA events (or `torch.cuda.synchronize()` before a host timer), never with a bare host timer around an asynchronous launch.

## Contents

| # | Project | Team it maps to | Time |
|---|---|---|---|
| 1 | [Tiled GEMM versus cuBLAS with a roofline](#1-a-tiled-gemm-benchmarked-against-cublas-with-a-roofline) | Math libraries, CUTLASS, DevTech | 3–4 weeks |
| 2 | [Fused memory-bound kernels in Triton and CUDA](#2-fused-memory-bound-kernels-in-triton-and-cuda-c) | Inference, DL frameworks, compilers | 1–2 weeks |
| 3 | [FP8 and INT4 quality versus speed](#3-fp8-and-int4-quality-versus-speed-study-on-a-small-model) | TensorRT-LLM, inference, research | 2–3 weeks |
| 4 | [Nsight profiling write-up of a training step](#4-an-nsight-profiling-write-up-of-a-real-training-step) | DevTech, frameworks, solutions architecture | 1–2 weeks |
| 5 | [Split-K decode attention kernel](#5-a-split-k-decode-attention-kernel) | TensorRT-LLM, inference kernels | 2–3 weeks |
| 6 | [Activation memory: formula versus measurement](#6-activation-memory-formula-versus-measurement) | Megatron-Core, NeMo, training | 1 week |
| 7 | [Speculative decoding: predicted versus measured speedup](#7-speculative-decoding-predicted-versus-measured-speedup) | Inference | 1–2 weeks |
| 8 | [A merged contribution to NVIDIA's open-source stack](#8-a-merged-contribution-to-nvidias-open-source-stack) | Any team that owns the repo | ongoing |

> [!NOTE]
> **Setup.** On Windows, use WSL2 for the CUDA toolkit, Nsight tools, Triton and TensorRT-LLM; see [SETUP.md](../../SETUP.md). Compile with `nvcc -arch=sm_120`. Datacenter-Blackwell code built for `sm_100a` does not run on an RTX 50-series card. On Colab, check which GPU you were given (`nvidia-smi`): a T4 (Turing) has no FP8 or bf16 tensor cores, an L4 (Ada) supports FP8, and an A100 supports bf16 but not FP8.

## 1. A tiled GEMM benchmarked against cuBLAS with a roofline

**Scope.** Write single-precision and then bf16 matrix-multiply kernels in CUDA C++, step by step: naive, coalesced, shared-memory tiled, 2D register-tiled, vectorized loads, then a tensor-core version with WMMA or `mma.sync`. Benchmark every step against cuBLAS (`cublasSgemm`, `cublasGemmEx`) and `torch.matmul` on square and LLM-shaped matrices (for example $M = 1..4096$ tokens against $4096 \times 4096$ and $4096 \times 14336$ weights).

**Roofline analysis.** For each kernel, compute global-memory arithmetic intensity from the tile sizes, $I = 2B_MB_N / ((B_M + B_N)\cdot\text{bytes})$, place the measured FLOP/s on a roofline built from *your measured* peak FLOP/s and bandwidth (`python tools/measure_gpu.py`), and confirm the bound with Nsight Compute's memory and compute throughput percentages. Explain the small-$M$ (decode-like) shapes, where every kernel, cuBLAS included, is memory-bound.

**Labs used.** [Lab 03](../../labs/03_napkin_math/README.md) (roofline), [curriculum 02](../../curriculum/02-compute-and-hardware.md).

**Success criteria.**
- Every kernel matches cuBLAS within a tolerance you justify for the dtype.
- A table and roofline plot for all steps, with Nsight Compute evidence for the two biggest jumps.
- A stated fraction of cuBLAS for your best fp32 and bf16 kernels at $4096^3$. Aim for a majority of cuBLAS on fp32 SIMT; tensor-core kernels are harder, and an honest gap with a diagnosis is a good result.

**Stretch.** Re-implement the best version with the CUTLASS CuTe DSL (Python) and compare effort and speed.

**Presentation.** A blog post in the style of Simon Boehm's matmul write-up: one section per step, what changed, the number, why. Put the repo and plot on your résumé as one line with the percentage of cuBLAS.

## 2. Fused memory-bound kernels in Triton and CUDA C++

**Scope.** Implement fused softmax, RMSNorm plus residual add, and SwiGLU in Triton and in CUDA C++. Compare with PyTorch eager and `torch.compile`.

**Why it matters.** These ops sit at about 1 FLOP/byte, far below any GPU's ridge point, so the only metric that matters is effective bandwidth: $\text{bytes read} + \text{bytes written}$ divided by time, compared with the bandwidth you measured (about 448 GB/s peak on the RTX 5060 spec sheet).

**Labs used.** [Lab 06](../../labs/06_attention_kernels/README.md) (fused softmax in Triton), [lab 05](../../labs/05_transformer/README.md) (where these ops sit in a model).

**Success criteria.** Correct against PyTorch for all shapes including non-power-of-two widths; effective bandwidth at or above 80% of your measured copy bandwidth for large rows; a plot of bandwidth against row width that shows where each implementation falls off, and why (register pressure, one row per block, and so on).

**Presentation.** A short post: "Three fused kernels, two languages, one roofline". Include the Triton versus CUDA code-size and performance comparison.

## 3. FP8 and INT4 quality versus speed study on a small model

**Scope.** Take a 0.5B–1.5B instruct model. Compare bf16, FP8 (weights and activations), INT8 weight-only and INT4 weight-only (AWQ or GPTQ), plus FP4 if your stack supports it, on quality and speed.

- **Engine.** TensorRT-LLM on Linux or WSL2 if its support matrix covers your GPU; check the current release notes for GeForce Blackwell support before you start. Otherwise use another engine that supports the format, and say so.
- **Quality.** A task accuracy (for example GSM8K-style exact match on a fixed subset) with 95% confidence intervals and a paired test between formats, plus perplexity. See [lab 15](../../labs/15_eval_stats/README.md).
- **Speed.** Prefill tokens/s at batch 1 and 16, decode tokens/s at batch 1 and 16, time to first token, and memory.

**Prediction first.** For batch-1 decode, $\text{tokens/s} \le \text{bandwidth} / \text{bytes read per step}$. A 1.24B-parameter model (Llama-3.2-1B) is about 2.5 GB in bf16, so the bound is about $448 / 2.5 \approx 180$ tokens/s before KV-cache reads; INT4 weights cut that to about 0.6–0.7 GB plus scales. Predict each format's decode speedup, then measure, then explain the gap (dequantization cost, kernel quality, launch overhead at small size).

**Labs used.** [Lab 13](../../labs/13_quantization/README.md), [lab 03](../../labs/03_napkin_math/README.md), [lab 15](../../labs/15_eval_stats/README.md), [curriculum 07 §4](../../curriculum/07-inference.md#4-quantization).

**Success criteria.** A single table of format × (accuracy with CI, perplexity, prefill tokens/s, decode tokens/s, memory), predictions next to measurements, and a recommendation per use case (latency-bound chat, throughput-bound batch) that follows from the numbers.

**Presentation.** Post plus a reproducible script. If you find a bug or a missing feature in the engine, file an issue with a minimal repro; that often becomes project 8.

## 4. An Nsight profiling write-up of a real training step

**Scope.** Profile one training step of the [lab 05](../../labs/05_transformer/README.md) GPT (`train.py`) with Nsight Systems, find the top three bottlenecks, fix them one at a time, and profile the slowest kernel with Nsight Compute.

Typical findings to look for: host-side gaps between kernels (launch overhead; fix with larger batches, `torch.compile` or CUDA graphs), data-loader stalls, unfused elementwise ops, fp32 paths that should be bf16, and matmul shapes that are not multiples of 8 or 16.

**Labs used.** [Lab 05](../../labs/05_transformer/README.md), [lab 02](../../labs/02_training_core/README.md), [lab 03](../../labs/03_napkin_math/README.md) (compute MFU before and after).

**Success criteria.** MFU before and after, computed from $6N$ FLOPs per token plus attention; timeline screenshots annotated with what each gap is; each fix attributed to a measured change; one thing you tried that did not help.

**Presentation.** A DevTech-style report: symptom, evidence, hypothesis, fix, result. This format is close to what performance engineers write for customers.

## 5. A split-K decode attention kernel

**Scope.** Write a Triton kernel for single-query (decode) attention over a long KV cache that splits the sequence across programs and combines partial results with the log-sum-exp trick (the FlashDecoding idea). Support grouped-query attention. Compare against PyTorch SDPA and against your lab 06 FlashAttention forward used with one query.

**Prediction first.** With Llama-3.2-1B's shape (16 layers, 8 KV heads, head dimension 64, bf16), the KV cache is $2 \cdot 16 \cdot 8 \cdot 64 \cdot 2 = 32{,}768$ bytes per token, so 256 MiB per sequence at 8,192 tokens. Decode attention must stream all of it every step, so the lower bound on time per layer is that layer's cache bytes divided by bandwidth.

**Labs used.** [Lab 06](../../labs/06_attention_kernels/README.md), [lab 07](../../labs/07_kv_cache_sampling/README.md).

**Success criteria.** Numerically matches reference attention; achieved bandwidth as a fraction of measured bandwidth across context lengths 512 to 32k and batch sizes 1 to 16; a clear explanation of why a non-split kernel underuses the GPU at batch 1 (too few programs to fill the SMs).

**Presentation.** Plot of achieved bandwidth against context length for each implementation; a paragraph on how paged KV caches change the memory-access pattern.

## 6. Activation memory: formula versus measurement

**Scope.** Korthikanti et al. give activation memory per transformer layer as about $sbh(34 + 5as/h)$ bytes in 16-bit precision without recomputation, where the $5as^2b$ attention term disappears with FlashAttention. Measure your lab 05 model's peak memory across sequence lengths and batch sizes, with and without activation checkpointing, and compare with the formula adapted to your architecture (SwiGLU, RMSNorm and GQA change the constant; derive the new one).

**Labs used.** [Lab 05](../../labs/05_transformer/README.md), [lab 03](../../labs/03_napkin_math/README.md), [lab 09](../../labs/09_parallelism/README.md) (extend to how tensor and sequence parallelism divide each term).

**Success criteria.** Your derived constant for your architecture, predictions within 10–15% of measured peak memory (explain the remainder: allocator fragmentation, temporary buffers, the logits tensor), and the compute overhead of checkpointing measured against the roughly 33% expected for full recomputation.

**Presentation.** A short note with the derivation, a predicted-versus-measured plot, and the table showing how TP and SP would divide each term. This is the kind of reasoning the Megatron and NeMo teams use every day.

## 7. Speculative decoding: predicted versus measured speedup

**Scope.** Use your [lab 14](../../labs/14_speculative_decoding/README.md) sampler with a small draft and a larger target from the same family (for example 0.5B drafting for 1.5B). Measure acceptance rate $\alpha$ per task, predict tokens per target pass as $(1 - \alpha^{k+1})/(1 - \alpha)$, then measure wall-clock speedup for $k = 1..8$ at batch 1 and batch 8.

**Labs used.** [Lab 14](../../labs/14_speculative_decoding/README.md), [lab 07](../../labs/07_kv_cache_sampling/README.md), [lab 03](../../labs/03_napkin_math/README.md).

**Success criteria.** A verified lossless sampler (lab 14 tests), predicted versus measured speedup per $k$ with an explanation of the gap (draft cost, verification overhead), and the batch size at which speculation stops paying off, with the reason (decode becomes less memory-bound as batch grows).

**Presentation.** One plot, speedup against $k$ for two batch sizes and two tasks (code and prose), with the predicted curve overlaid.

## 8. A merged contribution to NVIDIA's open-source stack

**Scope.** One merged pull request to TensorRT-LLM, Megatron-LM, NeMo, CUTLASS, Transformer Engine or Dynamo. Good first targets: a reproducible bug with a minimal failing test, a missing example for a model or GPU, documentation of a performance pitfall you hit in projects 1–7, or a small feature an open issue asks for.

**How.** Read the repository's `CONTRIBUTING.md` (several NVIDIA repos require a Developer Certificate of Origin sign-off, `git commit -s`). Comment on the issue before writing code, keep the PR small, include a test, and respond to review quickly.

**Success criteria.** Merged, with a test. A second PR to the same repo is worth more than a first PR to a second repo, because it shows you can work inside that team's codebase.

**Presentation.** List it on your résumé with the PR link and one line on the impact. Mention it when you apply to the team that owns the repository.

## Choosing two

| If you are aiming at | Do first | Then |
|---|---|---|
| CUDA libraries, CUTLASS, DevTech | 1 | 2 or 4 |
| TensorRT-LLM and inference | 3 | 5 or 7 |
| Megatron-Core, NeMo, training | 6 | 4 |
| Research (applied deep learning, Nemotron) | 3 | 6, and a small reproduction from the [reading list](reading-list.md) |

Project 8 runs in the background throughout: the other projects are where you find the bugs worth fixing.
