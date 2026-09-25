# How to get into NVIDIA's AI teams

NVIDIA hires core-AI engineers into the software that makes its GPUs useful for deep learning: CUDA libraries, inference engines, training frameworks, compilers and a large research organization. The interviews reward people who can reason about bytes, FLOPs and warps as easily as about transformers.
This guide covers where those teams sit, what they test, how to prepare with this repo, and a 90-day plan for an ML engineer in India.

> [!IMPORTANT]
> Team names, org charts, office lists and hiring steps change often. Everything marked "as of September 2026" should be re-verified on NVIDIA's careers site and the linked repos before you act. Refresh this page each quarter with [agent-brief.md](agent-brief.md).

## Contents

- [At a glance](#at-a-glance)
- [Where core-AI people work](#where-core-ai-people-work)
- [What they value](#what-they-value)
- [The hiring process](#the-hiring-process)
- [How to prepare with this repo](#how-to-prepare-with-this-repo)
- [Signals, ranked](#signals-ranked)
- [90-day plan](#90-day-plan-for-a-2-yoe-ml-engineer-in-india)
- [Tips and common mistakes](#tips-and-common-mistakes)
- [From India](#from-india)
- [Sources](#sources)

Companion files: [reading-list.md](reading-list.md) · [projects.md](projects.md) · [agent-brief.md](agent-brief.md)

## At a glance

| | As of September 2026 |
|---|---|
| What NVIDIA sells to AI teams | GPUs and systems, plus a software stack (CUDA, libraries, inference and training frameworks) that most of the industry runs on |
| Where AI engineers sit | Product engineering orgs (CUDA libraries, deep-learning frameworks, inference, compilers), NVIDIA Research, developer-technology and solutions-architecture groups |
| How hiring works | You interview with the team you would join; technical candidates may get a HackerRank coding exercise; unapproved outside tools during interviews disqualify you (NVIDIA's "How We Hire" page) |
| Open-source footprint | TensorRT-LLM, Megatron-LM/Megatron-Core, the NeMo libraries, CUTLASS, Transformer Engine and Dynamo are public on GitHub, so you can contribute before you apply |
| Recent stack changes | Triton Inference Server was renamed Dynamo-Triton (March 2025); Dynamo reached 1.0 (March 2026); the NeMo monorepo was split, with LLM training recipes moving to separate repos such as Megatron Bridge; CUTLASS 4 added a Python CuTe DSL |
| Research models | Nemotron 3 family (Nano, December 2025; Super, April 2026; Ultra, June 2026): hybrid Mamba-Transformer mixture-of-experts, with Super and Ultra trained in NVFP4 |
| India | Engineering in Bengaluru, Hyderabad and Pune, plus offices in Gurugram, Mumbai and New Delhi; a roughly 760,000 sq ft Bengaluru lease starting April 2026 (see [From India](#from-india)) |

The practical consequence of per-team hiring: you are not interviewing for "NVIDIA", you are interviewing for one team's stack. Read the posting's "What you'll be doing", "What we need to see" and "Ways to stand out from the crowd" sections as the syllabus; NVIDIA postings use those headings consistently.

## Where core-AI people work

The table groups the public software projects by what the work is. Team boundaries inside NVIDIA are not public; the repo is the best proxy for the team.

| Area | Public projects (verify names) | The work | What proves you can do it |
|---|---|---|---|
| Math and DL libraries | [cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cudnn), [CUTLASS](https://github.com/NVIDIA/cutlass) (C++ templates and the Python CuTe DSL) | GEMM, attention and convolution kernels per architecture; tensor-core, TMA and pipelining details; heuristics that pick a kernel per shape | A tiled GEMM you wrote, profiled and compared against cuBLAS with a roofline |
| LLM inference | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [TensorRT](https://github.com/NVIDIA/TensorRT) | Fused attention and MoE kernels, quantization (FP8, NVFP4, INT4-AWQ), in-flight batching, KV-cache management, speculative decoding | A quantization study with quality CIs and measured latency; a merged PR |
| Distributed serving | [Dynamo](https://github.com/ai-dynamo/dynamo), [Dynamo-Triton](https://github.com/triton-inference-server/server) (formerly Triton Inference Server) | Dynamo orchestrates SGLang, TensorRT-LLM and vLLM across GPUs and nodes: disaggregated prefill and decode, KV-aware routing, SLA-based planning (Rust core, Python extensions) | A capacity-planning write-up; a serving benchmark with p50/p95 TTFT and ITL |
| Training frameworks | [Megatron-LM / Megatron-Core](https://github.com/NVIDIA/Megatron-LM), the [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) repos (including Megatron Bridge for Hugging Face ↔ Megatron checkpoints), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Tensor, pipeline, context, expert and data parallelism (including FSDP); FP8 and FP4 training; checkpointing; MFU at thousands of GPUs | Lab 09 plus a reproduced parallelism result; a Megatron-Core or TE contribution |
| Compilers and DSLs | CUDA compiler toolchain, [Triton](https://github.com/triton-lang/triton) GPU backend contributions, CuTe DSL | Code generation for new architectures, autotuning, graph-level fusion | A Triton kernel with an autotuning study; compiler-level reading (PTX, SASS) |
| Research | [NVIDIA Research](https://research.nvidia.com/), including applied deep-learning research and the [Nemotron](https://huggingface.co/nvidia) model family | LLM pretraining and post-training, efficient architectures (hybrid Mamba-Transformer), pruning and distillation, low-precision training | Papers or reproductions with error bars; a small-model study that extends a Nemotron or Minitron result |
| Developer technology (DevTech) | Customer and partner code, benchmarks such as MLPerf submissions | Optimizing other people's workloads on NVIDIA GPUs, upstreaming fixes to frameworks | Profiling write-ups that find and fix a real bottleneck in someone else's code |
| Solutions architecture | Customer-facing | Designing and debugging customers' training and inference deployments | Clear system-design communication plus hands-on depth |

> [!NOTE]
> NVIDIA Research's LLM work is public through papers and the Nemotron models on Hugging Face. The Nemotron technical reports (listed in the [reading list](reading-list.md)) are the most direct view of what the research side cares about: data, hybrid Mamba-Transformer architectures, mixture-of-experts, pruning and distillation, and 4-bit training. Nemotron 3 is also a hardware story: its reports tie architecture choices to inference throughput on NVIDIA GPUs, which is the lens the whole company uses.

## What they value

Primary sources first. NVIDIA's Code of Conduct lists five core values: **innovation, intellectual honesty, speed and agility, excellence and determination, and one team**. It glosses intellectual honesty as "seek truth, learn from mistakes, share learnings", speed and agility as "learn, adapt, shape the world", and one team as "do what's best for the company" (as of September 2026; re-check the current document before quoting it).

What that means for an engineer, read through the job postings and the public repos:

- **Measured performance, not claimed performance.** Library and inference postings ask for profiling experience (Nsight Systems and Nsight Compute) and knowledge of GPU architecture. Your résumé bullets should carry numbers, the baseline, and the tool that measured them.
- **Intellectual honesty.** In practice: say what you did not measure, report the configuration that lost, and know the roofline limit of your own result.
- **Depth in C++ and systems** for library, inference and compiler roles; Python and PyTorch depth for framework and research roles. Most postings list both.
- **Full-stack understanding.** The best candidates can go from "tokens per second dropped" to "this GEMM shape falls off a tensor-core-friendly tile size" in one conversation.

## The hiring process

### Officially described

NVIDIA's [How We Hire](https://www.nvidia.com/en-us/about-nvidia/careers/how-we-hire/) page (as of September 2026) says:

- Candidates for technical roles may be asked to complete a coding exercise, usually on HackerRank, on a whiteboard or on a laptop NVIDIA provides.
- Using unapproved outside tools, such as ChatGPT, during an interview disqualifies the candidacy.
- During the final interview every candidate is offered an optional 15-minute "Insider Chat" with a member of an employee Community Resource Group. It does not influence the hiring decision.

Openings are posted on the careers site, and each posting names the team's work and requirements. University and early-career hiring runs through separate [university recruiting](https://www.nvidia.com/en-us/about-nvidia/careers/university-recruiting/) postings for new college graduates and interns. NVIDIA does not publish a fixed interview loop; the hiring team sets the format.

### As reported by candidates (not official; varies by team)

These patterns recur in public candidate reports (Glassdoor, Blind) and in third-party interview-prep guides that summarise them. Treat them as a planning prior, not a guarantee, and ask your recruiter for the actual format. Reports commonly describe four to eight weeks from first call to decision.

| Stage | Commonly reported shape |
|---|---|
| Recruiter screen, then hiring-manager call | Background, why this team, a walk through one project in depth |
| Online assessment or technical phone screen | LeetCode-style problems in C++ or Python (often skipped for referred candidates); for library and inference roles, questions on the CUDA execution model and memory hierarchy |
| Loop of roughly three to six 45-minute interviews with team members | Coding (often C/C++), GPU architecture and CUDA, performance reasoning, deep-learning fundamentals, project deep dive, sometimes system design; senior roles may add a leadership interview |
| Team decision | Per team; a different team may interview you separately |

### What each round tests, and how to answer

**C++ and Python.** Expect pointers, memory layout, RAII, templates, move semantics, undefined behaviour, and bit manipulation for library roles; clean, tested Python and PyTorch for framework and research roles. Write C++ you would be comfortable having reviewed: `const`-correct, no leaks, bounds you have checked.

**CUDA and GPU architecture.** Know these cold, with the mechanism behind each:

- Thread hierarchy: threads, warps of 32, blocks, grid; blocks are scheduled onto SMs and never migrate.
- Memory hierarchy: registers, shared memory (per block), L1/L2, global memory (HBM or GDDR). Which is private to what, roughly how big, roughly how fast.
- Coalescing: a warp reading 32 consecutive 4-byte floats touches 128 bytes, four 32-byte sectors, which is the ideal. A stride-32 access pattern touches 32 different sectors for the same useful data.
- Shared-memory bank conflicts: 32 banks, 4 bytes wide. Reading a column of a `float tile[32][32]` puts all 32 threads on one bank (a 32-way conflict); padding to `tile[32][33]` spreads them across banks.
- Occupancy: 64K 32-bit registers per SM on recent architectures. A kernel using 128 registers per thread can keep at most 65,536 / 128 = 512 threads (16 warps) resident per SM, whatever the block size. Know why more occupancy is not always faster (a GEMM with large register tiles trades occupancy for reuse).
- Warp divergence, `__syncthreads()` semantics, atomics, streams and events, and asynchronous copies (`cp.async`, TMA on Hopper and later).
- Tensor cores: what matrix fragment shapes they consume, why dimensions that are multiples of 8 or 16 matter, and what FP8 and FP4 need (scales).

**Performance reasoning.** The question is almost always a roofline question in disguise. Worked example of the kind you should be able to do aloud:

> A thread block computes a $B_M \times B_N$ tile of $C = AB$, stepping through $K$ in chunks of $B_K$. Per step it loads $(B_M + B_N)B_K$ elements and does $2B_MB_NB_K$ FLOPs. Global-memory arithmetic intensity is therefore
> $$I = \frac{2B_MB_N}{(B_M + B_N)\cdot \text{bytes per element}}.$$
> For $128 \times 128$ tiles in bf16, $I = 2\cdot 16384 / (256 \cdot 2) = 64$ FLOP/byte. For $32 \times 32$ tiles in fp32, $I = 2 \cdot 1024 / (64 \cdot 4) = 8$. The ridge point of an H100 SXM is roughly 990 TFLOP/s / 3.35 TB/s ≈ 300 FLOP/byte, so small tiles leave a GEMM memory-bound; L2 reuse helps, but the kernel's job is to raise $I$ with bigger tiles until registers and shared memory run out. One stage of $128 \times 128$ tiles with $B_K = 32$ in bf16 needs $(128+128)\cdot 32 \cdot 2 = 16$ KB of shared memory; three pipeline stages need 48 KB.

Other classic prompts: why batch-1 decode is memory-bound and what batching does to it; why a fused softmax beats three kernels; what an Nsight Compute report tells you when "memory throughput" is 90% and "compute throughput" is 20%; how you would decide between FP8 and INT4 weight-only quantization for a given batch size.

**Deep-learning fundamentals.** Transformers end to end (attention, RoPE, GQA, MoE routing), backprop through a layer by hand, mixed-precision training and loss scaling, the KV cache, quantization error, and the parallelism strategies in Megatron (tensor, pipeline, sequence and context, expert).

**System design.** For inference teams: design a serving stack for a given model, traffic and latency SLO, and size it in GPUs. For training teams: plan a run's parallelism layout across nodes. Show the arithmetic first, then the failure modes. See [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md) and [curriculum 07 §6](../../curriculum/07-inference.md#6-capacity-planning-the-interview-question).

**Project deep dive.** You will be asked to defend every number. Have the profiler output, the baseline and the hardware for each claim.

## How to prepare with this repo

| Interview area | Do this in the repo | Done when |
|---|---|---|
| Roofline and napkin math | [Lab 03 napkin math](../../labs/03_napkin_math/README.md), [curriculum 02](../../curriculum/02-compute-and-hardware.md), `python tools/measure_gpu.py` | You predict any GEMM, softmax or decode step on your GPU within 2x before measuring, and you have your own measured peak FLOP/s and bandwidth written down |
| Kernels | [Lab 06 attention kernels](../../labs/06_attention_kernels/README.md) (online softmax, FlashAttention forward and backward, Triton) | Your Triton FlashAttention forward passes the tests, and you can explain its HBM traffic and why it fails to help at short sequence lengths |
| Inference systems | [Curriculum 07](../../curriculum/07-inference.md), [lab 07 KV cache](../../labs/07_kv_cache_sampling/README.md), [lab 14 speculative decoding](../../labs/14_speculative_decoding/README.md) | You can size a serving deployment on a whiteboard in 10 minutes |
| Quantization | [Lab 13 quantization](../../labs/13_quantization/README.md) | You can explain why weight-only INT4 speeds up decode but not prefill, and why FP8 needs scaling |
| Distributed training | [Lab 09 parallelism](../../labs/09_parallelism/README.md), [curriculum 05](../../curriculum/05-pretraining.md) | You can derive Megatron tensor parallelism's communication per layer and the pipeline bubble fraction |
| DL fundamentals | [Lab 01 autograd](../../labs/01_autograd/README.md), [lab 02 training core](../../labs/02_training_core/README.md), [lab 05 transformer](../../labs/05_transformer/README.md) | You can write attention and a training loop from memory in 20 minutes |
| Coding rounds | [coding-interviews.md](../../tracks/research-engineer/coding-interviews.md), [dsa-patterns.md](../../tracks/research-engineer/dsa-patterns.md) | Two mediums in 45 minutes, and the ML drills in C++ as well as Python |
| Question bank and loops | [question-bank.md](../../tracks/research-engineer/question-bank.md) (inference and pretraining sections), [interview-loops.md](../../tracks/research-engineer/interview-loops.md) | Every inference question answered aloud with numbers |
| Behavioral | [career/stories.md](../../career/stories.md) | Stories about a performance bug you found, a result you retracted, and a cross-team fix |

> [!TIP]
> The labs are written in Python and Triton. NVIDIA's library and inference teams also expect CUDA C++. Port one lab kernel (the fused softmax from lab 06 is the right size) to CUDA C++ and benchmark both. The comparison is itself a good interview story.

## Signals, ranked

Ranked by how directly they show the work NVIDIA's AI teams do. Start at the top of what you can ship in 30 days.

1. **Merged contributions to the stack the team owns**: TensorRT-LLM, Megatron-LM/Megatron-Core, NeMo, CUTLASS, Transformer Engine, Dynamo. A merged bug fix with a test outranks an unmerged feature. Start from issues labelled for newcomers or "help wanted", and read `CONTRIBUTING.md` first (TensorRT-LLM and others require signed-off commits).
2. **Published kernel benchmarks with a roofline**: a GEMM, attention or fused kernel, compared against cuBLAS, cuDNN or a strong open baseline, with Nsight Compute evidence of why it is fast or slow. Honest losses count.
3. **A reproduced or extended performance result**: for example an FP8 or INT4 quality-versus-latency study with confidence intervals, or a reproduction of a Megatron parallelism result at small scale.
4. **Talks**: GTC talks (GTC and GTC India/AI Summit sessions are recorded on NVIDIA On-Demand), GPU MODE talks, or a local meetup talk on a kernel you wrote. Speaking about your own measured work is the credible version.
5. **Deep Learning Institute (DLI) courses and certifications**: useful for structure if you are new to CUDA, weak as a hiring signal on their own. Some self-paced DLI courses are free and others are paid; check the current catalog. A certificate never substitutes for a public kernel.
6. **General ML credentials** (degrees, Kaggle, generic LLM apps): they help you pass résumé screens but do not distinguish you for these teams.

## 90-day plan for a 2-YOE ML engineer in India

Assumes about 12 hours a week alongside a job, a laptop with an RTX 5060 (8 GB, Blackwell sm_120) or Colab, and Python/PyTorch fluency but little C++ or CUDA.

| Weeks | Focus | Output |
|---|---|---|
| 1–2 | Roofline and measurement. [Curriculum 02](../../curriculum/02-compute-and-hardware.md), [lab 03](../../labs/03_napkin_math/README.md), run `tools/measure_gpu.py`. Install the CUDA toolkit and Nsight tools (WSL2 is easiest on Windows). | Journal entry: your GPU's measured bf16 FLOP/s and bandwidth, ridge point, and three predictions checked |
| 3–4 | CUDA C++ basics. Vector add, then a reduction, then a tiled matmul with shared memory, from the CUDA C++ Programming Guide and *Programming Massively Parallel Processors*. Learn `cudaMalloc`, `cudaMemcpy`, error checking, `nvcc -arch=sm_120`, and events for timing. | A repo with four kernels, each with a correctness test against PyTorch and a timing table |
| 5–6 | Profiling. Nsight Systems for timelines, Nsight Compute for one kernel. Walk your tiled matmul through the steps in Simon Boehm's matmul write-up (coalescing, shared-memory blocking, register tiling). | [Project 1](projects.md#1-a-tiled-gemm-benchmarked-against-cublas-with-a-roofline) write-up: GEMM versus cuBLAS on a roofline plot |
| 7–8 | Attention kernels. [Lab 06](../../labs/06_attention_kernels/README.md) end to end, then port the fused softmax to CUDA C++. Read FlashAttention 1 and 2. | Lab 06 tests pass; Triton versus CUDA softmax comparison |
| 9–10 | Inference and quantization. [Lab 13](../../labs/13_quantization/README.md), [curriculum 07](../../curriculum/07-inference.md). Run a small model with FP8 or INT4 using TensorRT-LLM if your setup supports it (Linux or WSL2), or with another engine if not; measure quality with CIs. | [Project 3](projects.md#3-fp8-and-int4-quality-versus-speed-study-on-a-small-model) write-up |
| 11 | Open source. Pick one issue in TensorRT-LLM, Megatron-LM, NeMo or CUTLASS; reproduce it, fix it, open the PR. | An open PR with a test |
| 12 | Interview loop. Mock rounds: C++ coding, CUDA concepts aloud, one napkin-math round, one serving design, one project deep dive. Apply to three to five specific postings whose "What we need to see" matches your artifacts. | Tailored résumé with numbers; applications with referrals where possible |

> [!WARNING]
> Do not spend the 90 days on certificates and course videos. Hiring managers for these teams look for code that runs fast and a write-up that explains why.

## Tips and common mistakes

- **Tailor to one team.** A résumé that says "LLMs" fits nobody. One that says "wrote a tiled GEMM reaching X% of cuBLAS on sm_120, profiled with Nsight Compute" fits a library or inference team.
- **Know your GPU.** An RTX 5060 is consumer Blackwell (compute capability 12.0). It is not a B200 (sm_100): some datacenter-only features and `sm_100a` kernels do not run on it. Saying this correctly signals you have read the documentation.
- **Quote measured numbers with their configuration**: GPU, clocks if locked, CUDA and driver versions, shapes, dtype, warm-up and iteration counts.
- **Common mistake: timing asynchronous work wrongly.** CUDA launches are asynchronous. Time with CUDA events or synchronize before reading a host timer, and exclude warm-up and compilation.
- **Common mistake: comparing against a weak baseline.** Beating naive PyTorch eager is not a result. Compare against cuBLAS, cuDNN or `torch.compile`.
- **Common mistake: memorizing spec numbers.** Interviewers care that you can derive the bound from peak FLOP/s and bandwidth, not that you remember the H100's TFLOP/s.
- **Apply to several specific postings**, and use referrals: NVIDIA's India teams are large, and an engineer on the team can route your résumé to the right manager.

## From India

As of September 2026 (re-verify on the careers site's location filter):

- **Offices**: NVIDIA's largest India engineering sites are in Bengaluru and Hyderabad, with further engineering in Pune and offices in Gurugram and other cities. Search postings by city on the careers portal to see which AI teams hire locally.
- **Roles in India**: postings in India cover CUDA libraries, deep-learning frameworks and inference, system software, and solutions architecture. Research-scientist openings are concentrated in the US and Europe, though some research engineers sit in India.
- **Campus hiring**: NVIDIA runs new-college-graduate and intern hiring for Indian centers (for example system software roles in Bengaluru and Hyderabad).
- **Moving abroad**: internal transfers after some time at the company are a common route to US or European teams. For direct hires abroad, see [career/visa-and-relocation.md](../../career/visa-and-relocation.md); US H-1B rules changed in 2025–2026, so check current rules rather than older blog posts.

## Sources

Primary, as of September 2026:

- NVIDIA careers: [nvidia.com/en-us/about-nvidia/careers](https://www.nvidia.com/en-us/about-nvidia/careers/)
- NVIDIA Research: [research.nvidia.com](https://research.nvidia.com/)
- Repositories: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [NeMo](https://github.com/NVIDIA/NeMo), [CUTLASS](https://github.com/NVIDIA/cutlass), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), [Dynamo](https://github.com/ai-dynamo/dynamo), [Triton Inference Server](https://github.com/triton-inference-server/server)
- NVIDIA models on Hugging Face: [huggingface.co/nvidia](https://huggingface.co/nvidia)
- Papers and docs: see [reading-list.md](reading-list.md)

Reported, not official: candidate interview reports on Glassdoor and Blind; use them only for the rough shape of a loop.
