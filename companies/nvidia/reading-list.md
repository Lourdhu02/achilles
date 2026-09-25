# NVIDIA reading list

Thirty papers, guides and whitepapers behind the work NVIDIA's AI teams do: GPU architecture and CUDA, GEMM and attention kernels, low-precision number formats, Megatron-style distributed training, inference, and the Nemotron models.
Each entry says what to extract, so you read with a question instead of skimming.

> [!TIP]
> Read with a notebook open. For every paper, write down one number you could re-derive on a whiteboard (a byte count, a FLOP count, a bubble fraction) and re-derive it. That is the form in which these ideas come up in interviews.

## Read first

If you only have two weeks, read these five in this order.

1. **CUDA C++ Programming Guide**, chapters on the programming model, the hardware implementation and performance guidelines. You cannot discuss anything else without the execution model.
2. **Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*** (2022). The shortest path from a naive kernel to a tiled one, with measured steps.
3. **FlashAttention** (Dao et al., 2022). The canonical example of an IO-aware kernel.
4. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** (PTD-P, 2021). How NVIDIA reasons about parallelism.
5. **FP8 Formats for Deep Learning** (2022). The number formats behind Hopper and Blackwell training and inference.

## GPU architecture and CUDA

| Item | Year | What to extract |
|---|---|---|
| CUDA C++ Programming Guide and the newer restructured *CUDA Programming Guide*, both linked from the [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/) | current (CUDA 13.x) | Thread hierarchy, memory spaces, warp execution, the compute-capability tables (look up 12.0 for an RTX 50-series card and 10.0 for B200) |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | current | The APOD cycle (assess, parallelize, optimize, deploy), coalescing rules, shared-memory bank conflicts, occupancy, how to time correctly |
| Hwu, Kirk and El Hajj, *Programming Massively Parallel Processors*, 4th ed. (Morgan Kaufmann) | 2022 | Chapters on tiling, reduction, scan and convolution; work the exercises in CUDA C++ |
| [Williams, Waterman and Patterson, *Roofline*](https://doi.org/10.1145/1498765.1498785) (CACM) | 2009 | The model itself, and ceilings below the roof (no tensor cores, poor coalescing) |
| [NVIDIA H100 Tensor Core GPU Architecture whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) | 2022 | Thread-block clusters, distributed shared memory, TMA, FP8 tensor cores, the Transformer Engine idea |
| [NVIDIA Blackwell architecture technical overview](https://resources.nvidia.com/en-us-blackwell-architecture) (B200) | 2024 | Fifth-generation tensor cores, FP4 and microscaling support, NVLink 5; contrast with Hopper |
| NVIDIA RTX Blackwell GPU architecture whitepaper (GeForce RTX 50 series; linked from NVIDIA's GeForce news pages) | 2025 | What your own sm_120 card has (fifth-generation tensor cores with FP4, GDDR7) and lacks relative to datacenter Blackwell |
| [Jia et al., *Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking*](https://arxiv.org/abs/1804.06826) | 2018 | How to measure latencies, cache sizes and bank behavior yourself; the method transfers to newer GPUs |

## GEMM and attention kernels

| Item | Year | What to extract |
|---|---|---|
| [Boehm, *How to Optimize a CUDA Matmul Kernel*](https://siboehm.com/articles/22/CUDA-MMM) | 2022 | Each optimization step and why it moved the kernel toward the roofline; reproduce the table on your GPU |
| [CUTLASS repository and docs](https://github.com/NVIDIA/cutlass) (CuTe layouts, CuTe DSL) | current | The hierarchical GEMM decomposition (thread block, warp, instruction), CuTe layouts as shape-and-stride algebra, why kernels are specialized per architecture |
| [Osama et al., *Stream-K*](https://arxiv.org/abs/2301.03598) | 2023 | Why tile-count quantization wastes SMs on some shapes, and work-splitting across the K dimension as the fix |
| [Tillet, Kung and Cox, *Triton*](https://doi.org/10.1145/3315508.3329973) (MAPL) | 2019 | Block-level programming model; what the compiler does for you (coalescing, shared memory) versus CUDA |
| [Dao et al., *FlashAttention*](https://arxiv.org/abs/2205.14135) | 2022 | The HBM-traffic analysis; online softmax in the forward pass; recomputation in the backward pass |
| [Dao, *FlashAttention-2*](https://arxiv.org/abs/2307.08691) | 2023 | Work partitioning across warps and the sequence dimension; fewer non-matmul FLOPs |
| [Shah et al., *FlashAttention-3*](https://arxiv.org/abs/2407.08608) | 2024 | Hopper-specific asynchrony (warp specialization, TMA, WGMMA) and FP8 attention; a template for architecture-specific kernel design |

## Low-precision numerics

| Item | Year | What to extract |
|---|---|---|
| [Micikevicius et al., *Mixed Precision Training*](https://arxiv.org/abs/1710.03740) | 2017 | fp32 master weights, loss scaling, fp32 accumulation; the reasons each is needed |
| [Micikevicius et al., *FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433) | 2022 | E4M3 versus E5M2, why forward uses one and gradients the other, per-tensor scaling, the special-value choices |
| [Rouhani et al., *Microscaling Data Formats for Deep Learning*](https://arxiv.org/abs/2310.10537) | 2023 | Block scaling (MX formats): one shared scale per 32 elements, and what it buys over per-tensor scales |
| [NVIDIA, *Pretraining Large Language Models with NVFP4*](https://arxiv.org/abs/2509.25149) | 2025 | What it takes to train in 4-bit: block scaling, which layers stay in higher precision, and how the loss curve compares with FP8 |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) docs and examples | current | Delayed versus current scaling for FP8, amax history, where FP8 is and is not applied in a transformer layer |

## Distributed training (Megatron)

| Item | Year | What to extract |
|---|---|---|
| [Shoeybi et al., *Megatron-LM*](https://arxiv.org/abs/1909.08053) | 2019 | Column-then-row parallel MLP and attention splits: two all-reduces per layer forward; derive them |
| [Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473) | 2021 | PTD-P: combining tensor, pipeline and data parallelism; the interleaved schedule and its bubble fraction $(p-1)/(v\cdot m)$; why TP stays inside a node |
| [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*](https://arxiv.org/abs/2205.05198) | 2022 | The per-layer activation-memory formula, sequence parallelism, selective recomputation; re-derive the $34sbh$ term |
| [Megatron-LM / Megatron-Core repository](https://github.com/NVIDIA/Megatron-LM) | current | How the papers became a library: parallel-state setup, context and expert parallelism, distributed checkpointing |

## Inference

| Item | Year | What to extract |
|---|---|---|
| [TensorRT-LLM repository and docs](https://github.com/NVIDIA/TensorRT-LLM) | current | The PyTorch-based LLM API, in-flight batching, paged KV cache, the supported quantization modes and their hardware requirements |
| [Kwon et al., *PagedAttention* (vLLM)](https://arxiv.org/abs/2309.06180) | 2023 | Block tables for the KV cache and why fragmentation limits batch size; the design every engine now shares |
| [NVIDIA Dynamo repository](https://github.com/ai-dynamo/dynamo) | current | Disaggregated prefill and decode, KV-aware routing; what problem sits above a single engine |

## Nemotron and NVIDIA's LLM research

| Item | Year | What to extract |
|---|---|---|
| [Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (Minitron)](https://arxiv.org/abs/2407.14679) | 2024 | Structured pruning of width and depth plus distillation, and the compute saved versus training from scratch |
| [NVIDIA, *Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*](https://arxiv.org/abs/2504.03624) | 2025 | Why replace most attention layers with Mamba layers: the inference-throughput argument at long context, and FP8 pretraining |
| [NVIDIA, *NVIDIA Nemotron 3: Efficient and Open Intelligence*](https://arxiv.org/abs/2512.20856), with the [Super](https://arxiv.org/abs/2604.12374) and [Ultra](https://arxiv.org/abs/2606.15007) reports | 2025–2026 | Hybrid Mamba-Transformer mixture-of-experts, LatentMoE, multi-token prediction as built-in speculative decoding, NVFP4 pretraining at scale; note each design choice's inference-cost justification |

## How to use this list

- Pair each kernel reading with a lab: Boehm and CUTLASS with [project 1](projects.md#1-a-tiled-gemm-benchmarked-against-cublas-with-a-roofline); FlashAttention with [lab 06](../../labs/06_attention_kernels/README.md); FP8 and MX with [lab 13](../../labs/13_quantization/README.md); Megatron with [lab 09](../../labs/09_parallelism/README.md).
- Write one journal note per item: the number you re-derived, one question you still have, one thing you would test.
- The general reading list for the repo is [curriculum/papers.md](../../curriculum/papers.md).
