# 02 — Compute and hardware

Every design decision in modern AI comes down to FLOPs, bytes and bandwidth. Learn to reason in them. Lab: [03 napkin math](../labs/03_napkin_math/README.md).

## 1. The GPU in one picture
SMs (streaming multiprocessors) run warps of 32 threads. **Tensor cores** do small matrix multiply-accumulates at 8–16× the non-tensor rate. Memory hierarchy, fastest to slowest: registers → shared memory/L1 (~100–250 KB per SM, ~10+ TB/s aggregate) → L2 (tens of MB) → HBM/GDDR (GBs, ~0.4–8 TB/s) → PCIe/NVLink → network. **Most kernels are limited by how fast they move bytes, not by how fast they do math.**

## 2. Roofline
`time ≥ max(FLOPs / peak_FLOPs, bytes / bandwidth)`. Arithmetic intensity I = FLOPs/byte. Ridge point = peak/bandwidth.
- Large matmul: I ≈ n/3 for n×n×n in bf16, which is compute-bound.
- Elementwise ops, softmax, norms: I ≈ 1, which is memory-bound. Hence **fusion**: do many elementwise ops per trip to memory.
- Batch-1 decode: a matrix-vector product, I ≈ 1, memory-bound. Batch B raises I to ≈ B.

## 3. Accounting
- **FLOPs:** forward 2N per token, training 6N per token, plus attention `2·L·T·d` per token forward (causal).
- **Training memory:** 16 B/param (mixed-precision Adam) plus activations ≈ `34·s·b·h` bytes per layer with FlashAttention (Korthikanti et al.). Activation checkpointing trades roughly 33% more compute for storing only layer inputs.
- **Inference memory:** weights (bytes/param × N) plus KV cache `2·L·H_kv·d_h·T·B·bytes`.
- **MFU** = achieved model FLOP/s ÷ peak. Good large-scale training reaches 35–55%; small models and long sequences go lower.
- **Communication:** ring all-reduce moves `2(n−1)/n` of the buffer per device. Collectives: all-reduce (DP gradients, TP outputs), reduce-scatter and all-gather (ZeRO/FSDP), all-to-all (MoE expert parallelism).

## 4. Number formats
| format | exponent/mantissa bits | use |
|---|---|---|
| fp32 | 8/23 | master weights, optimizer state, reductions |
| tf32 | 8/10 | fp32 matmuls on tensor cores |
| bf16 | 8/7 | default training and inference: fp32's range, low precision |
| fp16 | 5/10 | needs loss scaling; still common for inference |
| fp8 e4m3 / e5m2 | 4/3, 5/2 | Hopper/Blackwell training and inference with per-tensor or per-block scales |
| fp4 (MXFP4/NVFP4) | 2/1 + block scales | Blackwell inference and emerging training |
| int8 / int4 | integers + scales | weight-only and KV-cache quantization |

## 5. Reference numbers (order of magnitude; verify spec sheets before quoting)
| | dense bf16 | memory | bandwidth |
|---|---|---|---|
| A100 80GB SXM | ~312 TFLOP/s | 80 GB HBM2e | ~2.0 TB/s |
| H100 SXM | ~990 TFLOP/s | 80 GB HBM3 | ~3.35 TB/s |
| B200 | ~2.2 PFLOP/s | ~180 GB HBM3e | ~8 TB/s |
| **your RTX 5060** | measure it (`tools/measure_gpu.py`) | 8 GB GDDR7 | ~448 GB/s |

Interconnects: NVLink (H100) ~900 GB/s bidirectional per GPU; PCIe 5.0 x16 ~64 GB/s per direction; InfiniBand NDR 400 Gb/s ≈ 50 GB/s per port.

**RTX 5060 setup note:** Blackwell consumer cards are compute capability 12.0 (sm_120) and need PyTorch built for CUDA 12.8+ and an R570+ driver. See [SETUP.md](../SETUP.md).

## 6. Worked drills
In [lab 03](../labs/03_napkin_math/README.md#drills-answer-in-under-2-minutes-each-then-check) with answers. Aim to answer each in under 2 minutes.

**Read:** Horace He, *Making Deep Learning Go Brrrr From First Principles*; *Transformer Inference Arithmetic* (kipply); *How to Scale Your Model* (Google DeepMind, 2025); Hugging Face *Ultra-Scale Playbook* (2025); Williams et al., *Roofline* (2009).
