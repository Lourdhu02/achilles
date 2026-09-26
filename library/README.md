# Library: visual guides and definitive papers

A curated reading library for the hardest topics in modern AI: 19 topics, 90 items, 2 to 5 per topic. For each topic you get the best visual explanation, the definitive paper, and course slides or notes where they add something, each with a reason it made the cut and the exact sections worth your time.

Everything is also listed in machine-readable form in [`manifest.json`](manifest.json); 59 items have a PDF that [`tools/fetch_library.py`](../tools/fetch_library.py) can download for offline study.

> [!NOTE]
> This library complements [`curriculum/papers.md`](../curriculum/papers.md), which lists many papers per module. Here the goal is the opposite: the few items per topic that give the most understanding per hour, and which pages to read. Links were last reviewed in September 2026. If one breaks, the arXiv ID or title will find it.

## Contents

- [How to use this library](#how-to-use-this-library)
- [Start here: a 10-item path](#start-here-a-10-item-path)
- Topics:
  - [Attention and transformers](#attention-and-transformers)
  - [RoPE and context extension](#rope-and-context-extension)
  - [Tokenization](#tokenization)
  - [GPU architecture, CUDA and Triton](#gpu-architecture-cuda-and-triton)
  - [FlashAttention](#flashattention)
  - [KV cache, PagedAttention and serving](#kv-cache-pagedattention-and-serving)
  - [Scaling laws](#scaling-laws)
  - [Distributed training (DP, FSDP/ZeRO, TP, PP)](#distributed-training-dp-fsdpzero-tp-pp)
  - [Mixed precision and FP8](#mixed-precision-and-fp8)
  - [Optimizers (Adam, AdamW, Muon)](#optimizers-adam-adamw-muon)
  - [RLHF, DPO and GRPO](#rlhf-dpo-and-grpo)
  - [Reasoning and test-time compute](#reasoning-and-test-time-compute)
  - [Quantization](#quantization)
  - [Speculative decoding](#speculative-decoding)
  - [Mixture of experts](#mixture-of-experts)
  - [State-space models (Mamba)](#state-space-models-mamba)
  - [Mechanistic interpretability](#mechanistic-interpretability)
  - [Evaluation and statistics](#evaluation-and-statistics)
  - [RAG and agents](#rag-and-agents)
- [How to download the PDFs](#how-to-download-the-pdfs)
- [Licensing and fair use](#licensing-and-fair-use)
- [Adding an item](#adding-an-item)

## How to use this library

1. **Picture first.** Read the visual explainer (article or video) before the paper; the paper then reads much faster.
2. **Read only the focus sections.** The *Focus on* column names the sections, figures or algorithms that carry the idea. Read those closely and skim the rest.
3. **Close the loop in a lab.** Each topic links the lab or curriculum module that uses it. Implement the mechanism and predict a number before you measure it.
4. **Write one paragraph.** After each item, write down the claim, the evidence and one thing you would test next (the format used in [`papers.md`](../curriculum/papers.md)).

Levels: **intro** needs only basic deep learning; **intermediate** assumes you have built a small transformer; **advanced** assumes the intermediate items in the same topic.

## Start here: a 10-item path

If you read nothing else, read these in order. Together they cover the stack from the transformer to evaluation, and each one sets up a lab. At one to two evenings per item, this is about a month of reading.

| # | Item | Why now | Then do |
|:-:|---|---|---|
| 1 | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) <sub>(article, intro)</sub> | The mechanism everything else builds on. | Lab 05 |
| 2 | [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) <sub>(video, intro)</sub> | Tokenization explains many model quirks; you will build one next. | Lab 04 |
| 3 | [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) <sub>(article, intro)</sub> | The performance model you need before any kernel or systems work. | Lab 03 |
| 4 | [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) <sub>(article, intro)</sub> | Turns that model into numbers for LLM inference. | Lab 03, then lab 07 |
| 5 | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) <sub>(paper, advanced)</sub> | The best example of IO-aware algorithm design. Read the online-softmax paper alongside it. | Lab 06 |
| 6 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) <sub>(paper, intermediate)</sub> | How to size a model for a compute budget. | Lab 08 |
| 7 | [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook) <sub>(article, intermediate)</sub> | How training is spread across many GPUs, and what each scheme costs. | Lab 09 |
| 8 | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) <sub>(paper, intermediate)</sub> | Preference optimisation from a derivation you can check by hand. | Lab 11 |
| 9 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) <sub>(paper, intermediate)</sub> | How reasoning models are trained with RL on verifiable rewards. | Lab 12 |
| 10 | [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640) <sub>(paper, intermediate)</sub> | How to tell whether a result is real. | Lab 15 |

> [!TIP]
> If you are aiming at interpretability, swap item 10 for [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) and do lab 16. If you are aiming at inference, add [PagedAttention](https://arxiv.org/abs/2309.06180) and [speculative decoding](https://arxiv.org/abs/2211.17192) after item 5.

## Attention and transformers

Get the mechanism from pictures first, then the specification, then what changed after 2017. Pairs with [lab 05](../labs/05_transformer/README.md), [module 04](../curriculum/04-transformers.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)<br><sub>Jay Alammar, 2018</sub> | article | The most widely used visual introduction. It builds attention one word-vector at a time before showing the matrix form, which is the order intuition needs. | The self-attention walkthrough (query, key and value vectors for one word, then the matrix form) and multi-head attention. Skim the encoder-decoder parts. | intro |
| [CS224N note: Self-Attention and Transformers (draft)](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) (PDF)<br><sub>John Hewitt (Stanford CS224N), 2023</sub> | notes | Short, rigorous notes that introduce each transformer component as the fix for a concrete failure of bare self-attention, instead of presenting the architecture as given. | The sequence of fixes: position representations, element-wise nonlinearities, causal masking. Then the full block with residuals and layer norm. | intermediate |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [PDF](https://arxiv.org/pdf/1706.03762)<br><sub>Vaswani et al., 2017</sub> | paper | The primary source, and still the cleanest specification of the architecture. | Section 3.2 (scaled dot-product and multi-head attention, and the footnote on why scores are divided by sqrt(d_k)), Figure 2, and Table 1 (per-layer cost vs recurrence and convolution). | intro |
| [CS336 lecture 3: architectures and hyperparameters](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | Covers what the 2017 paper cannot: the design choices modern LLMs converged on, argued from many published models. | Pre-norm vs post-norm, RMSNorm, gated MLPs, RoPE, attention variants (GQA/MQA), and the hyperparameter ratios that recur across models. | intermediate |
| [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)<br><sub>Lilian Weng, 2023</sub> | article | The best single index of transformer variants, each with its core equation and citation. A reference to return to, not a first read. | Positional encodings, long-context and efficient-attention variants. Read the section you need when a paper assumes it. | advanced |

3Blue1Brown's neural-network video series (the chapters on transformers and attention) is the best first visual intuition if you have not seen attention before.

## RoPE and context extension

RoPE is a rotation of queries and keys; context extension is mostly about which rotation frequencies you rescale. Pairs with [lab 05](../labs/05_transformer/README.md), [module 04](../curriculum/04-transformers.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)<br><sub>Christopher Fleetwood (Hugging Face), 2024</sub> | article | Derives RoPE step by step from the properties a positional encoding should have, with plots at every step. The best intuition-first route. | The progression from integer positions to sinusoids to rotations, and why a relative encoding should act on queries and keys rather than on the embeddings. | intro |
| [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)<br><sub>EleutherAI, 2021</sub> | article | Written by EleutherAI, who adopted RoPE early (GPT-J, GPT-NeoX); a short, correct derivation with the complex-number view and early experiments. | The derivation that rotating q and k makes their dot product depend only on relative position, and the implementation notes. | intermediate |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) · [PDF](https://arxiv.org/pdf/2104.09864)<br><sub>Su et al., 2021</sub> | paper | The primary source for RoPE. | Section 3.2 (the 2D case, then the general block-diagonal rotation matrix) and the long-term-decay property of the rotated inner product. | intermediate |
| [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) · [PDF](https://arxiv.org/pdf/2306.15595)<br><sub>Chen et al., 2023</sub> | paper | The first simple, effective way to extend a trained RoPE model's context, with the clearest explanation of why naive extrapolation fails. | The bound showing attention scores can blow up at unseen positions, and why squeezing positions into the trained range plus a short fine-tune works. | advanced |
| [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) · [PDF](https://arxiv.org/pdf/2309.00071)<br><sub>Peng et al., 2023</sub> | paper | The context-extension method adopted by several open model families, including DeepSeek-V3; its per-frequency view underlies later methods. | The frequency-dependent view: interpolate low-frequency dimensions, leave high-frequency ones alone (NTK-by-parts), plus the attention temperature correction. | advanced |

## Tokenization

Many strange model failures are tokenization bugs; build a tokenizer once and you will recognise them. Pairs with [lab 04](../labs/04_tokenizer/README.md), [module 04](../curriculum/04-transformers.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)<br><sub>Andrej Karpathy, 2024</sub> | video | Builds a working byte-level BPE tokenizer and then traces real model failures back to tokenization. Nothing else covers both the algorithm and its consequences as well. | The BPE training loop, GPT-2/GPT-4 regex pre-tokenization, special tokens, and the closing list of tokenization-caused failures (spelling, arithmetic, non-English text, trailing spaces). | intro |
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) · [PDF](https://arxiv.org/pdf/1508.07909)<br><sub>Sennrich, Haddow, Birch, 2015</sub> | paper | The paper that brought BPE to neural NLP. The whole merge-learning algorithm fits in about a dozen lines. | Section 3.2 and its Python listing (Algorithm 1). Implement it before reading anything else about BPE. | intro |
| [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) · [PDF](https://arxiv.org/pdf/2305.15425)<br><sub>Petrov et al., 2023</sub> | paper | Quantifies how much more the same content costs in some languages. Essential if you build for users who do not write English. | The tokenization-premium measurements across languages and their consequences: higher cost, higher latency and less usable context. | intermediate |

Companion code for the video: [karpathy/minbpe](https://github.com/karpathy/minbpe).

## GPU architecture, CUDA and Triton

Performance work starts with knowing whether you are limited by compute, memory bandwidth or overhead. Pairs with [lab 06](../labs/06_attention_kernels/README.md), [lab 03](../labs/03_napkin_math/README.md), [module 02](../curriculum/02-compute-and-hardware.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)<br><sub>Horace He, 2022</sub> | article | The best mental model of GPU performance in one article, with a way to diagnose which regime you are in. | The three regimes (compute-bound, memory-bandwidth-bound, overhead-bound), operator fusion, and how to tell which regime a workload is in. | intro |
| [How to Think About GPUs (How to Scale Your Model)](https://jax-ml.github.io/scaling-book/gpus/)<br><sub>Jacob Austin et al. (Google DeepMind), 2025</sub> | article | Quantitative: every hardware feature becomes a number you can plug into a roofline estimate, with GPUs and TPUs compared side by side. | SMs, tensor cores, the memory hierarchy and NVLink/InfiniBand networking, each expressed as numbers for roofline estimates. | intermediate |
| [CS336 lecture 5: GPUs](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | University lecture slides that tie GPU hardware directly to LLM workloads. | The execution model, the memory hierarchy, and why tiling, fusion and matmul shapes decide performance. | intermediate |
| [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)<br><sub>Simon Boehm, 2022</sub> | article | Shows with measurements which optimisations make a matmul kernel fast and why. The best bridge from CUDA basics to real kernels. | The kernel-by-kernel progression (coalescing, shared-memory tiling, 1D and 2D block tiling, vectorised loads, warp tiling) and the throughput gained at each step. | intermediate |
| [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/10.1145/3315508.3329973)<br><sub>Tillet, Kung, Cox, 2019</sub> | paper | The primary source for the tile-based programming model used in lab 06. | The core idea: program at the level of tiles and let the compiler handle scheduling and memory coalescing inside a tile. Then work through the official Triton tutorials (vector add, fused softmax, matmul). | advanced |

Also useful: the [Modal GPU Glossary](https://modal.com/gpu-glossary) (CC BY 4.0) for decoding terms, and the [rooflines chapter](https://jax-ml.github.io/scaling-book/roofline/) of How to Scale Your Model as the prerequisite for every estimate.

## FlashAttention

Online softmax makes tiled attention exact; tiling makes it IO-efficient. Read in this order. Pairs with [lab 06](../labs/06_attention_kernels/README.md), [module 07](../curriculum/07-inference.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) · [PDF](https://arxiv.org/pdf/1805.02867)<br><sub>Milakov, Gimelshein, 2018</sub> | paper | The key algorithmic ingredient of FlashAttention, isolated in a short paper. | The one-pass algorithm that keeps a running maximum and rescales the running sum. Prove to yourself that it is exact. | intermediate |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) · [PDF](https://arxiv.org/pdf/2205.14135)<br><sub>Dao et al., 2022</sub> | paper | The primary source, and a model of analysing an algorithm by memory traffic instead of FLOPs. | Figure 1 (tiling across HBM and SRAM), Algorithm 1, and the IO-complexity analysis in Section 3.2. Then the backward pass with recomputation in the appendix. | advanced |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) · [PDF](https://arxiv.org/pdf/2307.08691)<br><sub>Tri Dao, 2023</sub> | paper | Explains the engineering that turned the idea into the default attention kernel. | Section 3: fewer non-matmul FLOPs, parallelising over sequence length, and splitting work across warps to cut shared-memory traffic. | advanced |
| [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) · [PDF](https://arxiv.org/pdf/2407.08608)<br><sub>Shah et al., 2024</sub> | paper | Shows how new hardware features (asynchronous copies, FP8 tensor cores) change kernel design. | Warp specialisation with asynchronous loads, overlapping softmax with the matmuls, and FP8 with incoherent processing to limit quantization error. | advanced |

## KV cache, PagedAttention and serving

Decoding is usually memory-bandwidth-bound; serving systems are mostly about managing KV cache memory and batching. Pairs with [lab 07](../labs/07_kv_cache_sampling/README.md), [module 07](../curriculum/07-inference.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)<br><sub>kipply, 2022</sub> | article | Short, numeric and practical: teaches you to estimate inference memory and latency on paper. | KV cache bytes per token, why small-batch decoding is memory-bandwidth-bound, and the batch size where it turns compute-bound. Redo each number for Llama 3 8B. | intro |
| [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching)<br><sub>Hugging Face, 2025</sub> | article | Builds serving concepts in the right order with clear diagrams. | The sections on the KV cache, chunked prefill and continuous batching: how requests are packed into one batch and scheduled. | intro |
| [All About Transformer Inference (How to Scale Your Model)](https://jax-ml.github.io/scaling-book/inference/)<br><sub>Jacob Austin et al. (Google DeepMind), 2025</sub> | article | The most complete quantitative treatment of inference, from sampling one token to a sharded serving system. | Prefill vs decode rooflines, KV cache memory, the latency-throughput trade-off, and how to shard a model for serving. | intermediate |
| [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) · [PDF](https://arxiv.org/pdf/2309.06180)<br><sub>Kwon et al., 2023</sub> | paper | The primary source for PagedAttention, now standard in serving engines. Its memory-waste analysis explains why. | Section 3 (KV cache waste: reservation, internal and external fragmentation) and Section 4 (block tables; copy-on-write sharing for parallel sampling and beam search). | intermediate |
| [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) · [PDF](https://arxiv.org/pdf/2211.05102)<br><sub>Pope et al., 2022</sub> | paper | The paper behind most modern inference cost models; the scaling-book inference chapter builds on it. | The cost model for latency vs utilisation, the partitioning layouts for feedforward and attention layers, and why multiquery attention matters at long context. | advanced |

Going further: Lilian Weng's [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).

## Scaling laws

Learn to read, fit and distrust power laws; every model-sizing decision rests on them. Pairs with [lab 08](../labs/08_scaling_laws/README.md), [module 05](../curriculum/05-pretraining.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · [PDF](https://arxiv.org/pdf/2001.08361)<br><sub>Kaplan et al., 2020</sub> | paper | The paper that established loss as a smooth power law in parameters, data and compute. | The summary of results in Section 1 and Figure 1. Note which allocation conclusion Chinchilla later revised, and why (the learning-rate schedule was not matched to training length). | intermediate |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) · [PDF](https://arxiv.org/pdf/2203.15556)<br><sub>Hoffmann, Borgeaud, Mensch et al., 2022</sub> | paper | The compute-optimal result every model-sizing decision still starts from. | Section 3: three estimation approaches (fixed model sizes, IsoFLOP profiles, a parametric loss fit L(N, D) = E + A/N^alpha + B/D^beta), and the result that parameters and tokens should grow in equal proportion with compute. | intermediate |
| [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) · [PDF](https://arxiv.org/pdf/2404.10102)<br><sub>Besiroglu et al., 2024</sub> | paper | A worked lesson in fitting scaling laws carefully and checking published fits. | How refitting the parametric approach on data reconstructed from the paper's figure gives estimates consistent with the other two approaches, and the implausibly narrow original confidence intervals. | advanced |
| [CS336 lecture 9: scaling laws basics](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | A course-level synthesis of the scaling-law literature with the practical questions it answers. | Data and model scaling, how to run small sweeps and extrapolate, and the pitfalls in fitting. | intermediate |
| [CS336 lecture 11: scaling details](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | How labs actually use scaling experiments, which the papers rarely spell out. | Case studies from published model reports, and hyperparameter transfer across scale (muP, batch size and learning-rate scaling). | advanced |

## Distributed training (DP, FSDP/ZeRO, TP, PP)

Each parallelism scheme trades memory for communication; learn to compute both. Pairs with [lab 09](../labs/09_parallelism/README.md), [module 05](../curriculum/05-pretraining.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)<br><sub>Hugging Face (Nanotron team), 2025</sub> | article | The most complete visual treatment of LLM training parallelism, with interactive memory calculators and results measured on real clusters. | The memory breakdown (weights, gradients, optimizer states, activations), then DP, ZeRO stages 1-3, TP, sequence and context parallelism, pipeline schedules and expert parallelism. | intermediate |
| [How to Parallelize a Transformer for Training (How to Scale Your Model)](https://jax-ml.github.io/scaling-book/training/)<br><sub>Jacob Austin et al. (Google DeepMind), 2025</sub> | article | Derives, for each scheme, exactly when communication becomes the bottleneck. The analytical complement to the playbook. | For DP, FSDP, TP and PP: the communication cost per step and the batch size or model size at which each becomes communication-bound. | advanced |
| [CS336 lecture 7: parallelism basics](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | Compact lecture slides that cover the whole design space in one sitting. | Collective operations and their costs, then data, tensor and pipeline parallelism with the memory and communication each saves or adds. | intermediate |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) · [PDF](https://arxiv.org/pdf/1910.02054)<br><sub>Rajbhandari et al., 2019</sub> | paper | The primary source for sharded data parallelism (the idea behind FSDP) and the clearest memory accounting for mixed-precision Adam. | Figure 1 and Section 5: the 2 + 2 + 12 bytes-per-parameter accounting, and how stages 1, 2 and 3 partition optimizer states, gradients and parameters. | intermediate |
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) · [PDF](https://arxiv.org/pdf/1909.08053)<br><sub>Shoeybi et al., 2019</sub> | paper | The primary source for tensor parallelism as used in practice. | Section 3 and Figure 3: column-parallel then row-parallel splits of the MLP and attention, so each block needs one all-reduce forward and one backward. | intermediate |

Going further: Lilian Weng's [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/) (survey), [PTD-P](https://arxiv.org/abs/2104.04473) (pipeline schedules at cluster scale) and [PyTorch FSDP](https://arxiv.org/abs/2304.11277) (the implementation).

## Mixed precision and FP8

Low precision is a range and resolution budget; scaling is how you stay inside it. Pairs with [module 02](../curriculum/02-compute-and-hardware.md), [module 03](../curriculum/03-deep-learning.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) · [PDF](https://arxiv.org/pdf/1710.03740)<br><sub>Micikevicius, Narang, Alben et al., 2017</sub> | paper | The three techniques every half-precision training recipe still uses. | Section 3: an FP32 master copy of weights, loss scaling (and the gradient histogram that motivates it), and FP32 accumulation. | intro |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) · [PDF](https://arxiv.org/pdf/2209.05433)<br><sub>Micikevicius, Stosic et al., 2022</sub> | paper | The specification of the two FP8 formats that current accelerators implement. | The E4M3 and E5M2 encodings, why E4M3 gives up infinities to extend its range, and which tensors use which format (weights and activations vs gradients). | intermediate |
| [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) · [PDF](https://arxiv.org/pdf/2310.10537)<br><sub>Rouhani et al., 2023</sub> | paper | Block scaling is how FP8 and FP4 are made usable; this paper defines the MX formats. | The block structure (32 elements share one power-of-two scale) and the experiments on which element formats hold up for inference and for training. | advanced |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) · [PDF](https://arxiv.org/pdf/2412.19437)<br><sub>DeepSeek-AI, 2024</sub> | paper | The most detailed public account of FP8 training at frontier scale. | Section 3.3 (fine-grained tile- and block-wise scaling, and promoting partial sums to higher precision). Also Section 2 for multi-head latent attention and auxiliary-loss-free MoE load balancing. | advanced |

## Optimizers (Adam, AdamW, Muon)

Know Adam's update line by line, what decoupled weight decay changes, and why Muon orthogonalises its updates. Pairs with [lab 02](../labs/02_training_core/README.md), [module 03](../curriculum/03-deep-learning.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Why Momentum Really Works](https://distill.pub/2017/momentum/)<br><sub>Gabriel Goh, 2017</sub> | article | Interactive figures that make optimisation dynamics visible; the best intuition for step sizes, curvature and momentum. | The eigen-decomposition of the error on a quadratic, why the largest curvature limits the step size, and how momentum changes the convergence rate. | intermediate |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) · [PDF](https://arxiv.org/pdf/1412.6980)<br><sub>Kingma, Ba, 2014</sub> | paper | The primary source for the default optimizer. | Algorithm 1 and Section 3 (bias correction). Skip the convergence analysis, which was later shown to be flawed. | intro |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) · [PDF](https://arxiv.org/pdf/1711.05101)<br><sub>Loshchilov, Hutter, 2017</sub> | paper | Explains the one-line difference between Adam with L2 and AdamW, which matters in every LLM training run. | Algorithm 2 (Adam with L2 regularisation vs decoupled weight decay, differences highlighted) and why L2 is not weight decay under adaptive methods. | intro |
| [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)<br><sub>Keller Jordan, 2024</sub> | article | The original description of Muon by its author, with the speedrun evidence that made it popular. | The update (momentum, then approximate orthogonalisation with Newton-Schulz iterations), which parameters use Muon and which stay on AdamW, and the empirical results. | intermediate |
| [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) · [PDF](https://arxiv.org/pdf/2502.16982)<br><sub>Liu, Su, Yao et al. (Moonshot AI), 2025</sub> | paper | Shows what Muon needs to work at LLM scale. | The two changes: add weight decay, and rescale each matrix's update to match AdamW's update RMS so AdamW hyperparameters transfer. | advanced |

## RLHF, DPO and GRPO

Post-training is a KL-regularised optimisation against a learned or verifiable reward; the algorithms differ in how they estimate it. Pairs with [lab 11](../labs/11_dpo/README.md), [lab 12](../labs/12_grpo/README.md), [module 06](../curriculum/06-post-training.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)<br><sub>Nathan Lambert et al., 2022</sub> | article | The standard visual overview of the RLHF pipeline. | The three-stage diagram: pretrained model, reward model trained on comparisons, RL fine-tuning with a KL penalty to the reference model. | intro |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) · [PDF](https://arxiv.org/pdf/2203.02155)<br><sub>Ouyang, Wu, Jiang et al., 2022</sub> | paper | The definitive RLHF paper for language models. | Figure 2 (SFT, reward model, PPO) and Section 3.5 (reward-model loss and the PPO-ptx objective). Note that outputs of the 1.3B tuned model were preferred to those of 175B GPT-3. | intermediate |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) · [PDF](https://arxiv.org/pdf/2305.18290)<br><sub>Rafailov et al., 2023</sub> | paper | Removes the reward model and the RL loop with a short derivation you can check by hand. | Section 4: from the optimum of the KL-constrained objective to the DPO loss, and what the DPO gradient does. Then Section 5 on the reward reparameterisation. | intermediate |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) · [PDF](https://arxiv.org/pdf/2402.03300)<br><sub>Shao, Wang, Zhu et al., 2024</sub> | paper | The primary source for GRPO, the RL algorithm behind most open reasoning models. | Section 4 (GRPO: group-relative advantages instead of a value network, and the KL estimator) with its PPO-vs-GRPO diagram, and the unified view of SFT, rejection sampling, DPO, PPO and GRPO. | advanced |
| [Reinforcement Learning from Human Feedback (the RLHF Book)](https://arxiv.org/abs/2504.12501) · [PDF](https://arxiv.org/pdf/2504.12501)<br><sub>Nathan Lambert, 2025</sub> | notes | A free, book-length reference with consistent notation across reward modelling, policy gradients and direct alignment. | Use as the reference text: reward models, regularisation, policy-gradient algorithms and direct alignment algorithms. The web version is at rlhfbook.com. | intermediate |

Going further: [CS336 lecture 15 slides on RLHF and alignment](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf).

## Reasoning and test-time compute

Reasoning models spend more compute at inference, and are trained with RL to use it well. Pairs with [lab 12](../labs/12_grpo/README.md), [module 06](../curriculum/06-post-training.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)<br><sub>Sebastian Raschka, 2025</sub> | article | The clearest map of how reasoning models are built, organised around the DeepSeek-R1 pipeline. | The four approaches: inference-time scaling, pure RL, SFT plus RL, and SFT with distillation, and where each shows up in the R1 recipe. | intro |
| [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/)<br><sub>Lilian Weng, 2025</sub> | article | The most thorough survey of test-time compute, with citations for every claim. | Parallel sampling vs sequential revision, RL for reasoning, and whether chains of thought are faithful. | intermediate |
| [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) · [PDF](https://arxiv.org/pdf/2501.12948)<br><sub>DeepSeek-AI, 2025</sub> | paper | The open recipe that showed reasoning can be trained with RL on verifiable rewards. | R1-Zero (RL with rule-based rewards and no SFT; response length grows during training), the multi-stage R1 recipe, distillation into small models, and the unsuccessful attempts (process reward models, MCTS). | intermediate |
| [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) · [PDF](https://arxiv.org/pdf/2408.03314)<br><sub>Snell et al., 2024</sub> | paper | The careful study of how to spend inference compute, and when it substitutes for a bigger model. | How the best strategy (sequential revisions vs search against a process reward model) depends on question difficulty, and the compute-optimal allocation that is more than 4x more efficient than best-of-N. | advanced |
| [CS336 lecture 16: RL from verifiable rewards](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | Lecture slides that connect policy-gradient basics to current reasoning-model recipes. | Policy gradients to GRPO, and the published reasoning-model training recipes. | advanced |

Going further: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (process vs outcome reward models) and [s1](https://arxiv.org/abs/2501.19393) (budget forcing with 1,000 examples; a good small-budget reproduction).

## Quantization

Quantization error is dominated by outliers; each method is a different way of handling them. Pairs with [lab 13](../labs/13_quantization/README.md), [module 07](../curriculum/07-inference.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)<br><sub>Maarten Grootendorst, 2024</sub> | article | The best visual introduction, with more than 50 custom figures. | Number formats, symmetric vs asymmetric quantization, outliers and clipping, then post-training methods (GPTQ, GGUF) and quantization-aware training. | intro |
| [MIT 6.5940 TinyML and Efficient Deep Learning Computing (Fall 2024)](https://hanlab.mit.edu/courses/2024-fall-65940)<br><sub>Song Han (MIT), 2024</sub> | slides | The best university course on efficient ML, taught by the group behind SmoothQuant and AWQ. | The two quantization lectures: linear and k-means quantization, granularity, calibration, and quantization-aware training. Slides and recordings are linked from the course page. | intermediate |
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) · [PDF](https://arxiv.org/pdf/2208.07339)<br><sub>Dettmers et al., 2022</sub> | paper | Discovered why naive INT8 breaks large models: emergent outlier features. | The emergence of large-magnitude outlier features with scale, and mixed-precision decomposition (outlier dimensions in FP16, the rest in INT8). | intermediate |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) · [PDF](https://arxiv.org/pdf/2211.10438)<br><sub>Xiao et al., 2022</sub> | paper | A mathematically equivalent rescaling that makes W8A8 quantization work; implemented in lab 13. | The per-channel smoothing factor that moves quantization difficulty from activations to weights, controlled by alpha, and the figure of activation vs weight magnitudes before and after. | intermediate |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) · [PDF](https://arxiv.org/pdf/2210.17323)<br><sub>Frantar et al., 2022</sub> | paper | The standard second-order method for 3-4 bit weight quantization; implemented in lab 13. | The algorithm box: quantize columns in order and push each column's error onto the remaining columns using the inverse Hessian (via Cholesky), with lazy batched updates. | advanced |

Going further: [AWQ](https://arxiv.org/abs/2306.00978) (activation-aware scaling that protects salient weights).

## Speculative decoding

A cheap draft proposes, the target verifies in one pass, and rejection sampling keeps the output distribution exact. Pairs with [lab 14](../labs/14_speculative_decoding/README.md), [module 07](../curriculum/07-inference.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) · [PDF](https://arxiv.org/pdf/2211.17192)<br><sub>Leviathan, Kalman, Matias, 2022</sub> | paper | The primary source, with the proof that the output distribution is unchanged and a clean speedup analysis. | Speculative sampling and why it is exact, then the expected tokens per step and walltime improvement as functions of the acceptance rate alpha and draft length gamma. | intermediate |
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) · [PDF](https://arxiv.org/pdf/2302.01318)<br><sub>Chen et al., 2023</sub> | paper | Independent concurrent work; short, with the rejection-sampling rule stated compactly. | The modified rejection-sampling algorithm and its proof, and the measured speedups on a 70B model. | intermediate |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) · [PDF](https://arxiv.org/pdf/2401.10774)<br><sub>Cai et al., 2024</sub> | paper | Removes the separate draft model, and its tree attention verifies many candidates in one pass. | Extra decoding heads on the target model, and tree attention to verify many candidate continuations in one forward pass. | advanced |
| [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) · [PDF](https://arxiv.org/pdf/2401.15077)<br><sub>Li et al., 2024</sub> | paper | The feature-level drafting idea behind EAGLE-2 and EAGLE-3, which vLLM and SGLang support. | Drafting at the feature level (the second-to-top layer) instead of the token level, and why feeding in the sampled token resolves the uncertainty. | advanced |

## Mixture of experts

MoE decouples parameters from FLOPs per token; routing and load balancing are where it gets hard. Pairs with [module 04](../curriculum/04-transformers.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Mixture of Experts Explained](https://huggingface.co/blog/moe)<br><sub>Omar Sanseviero et al., 2023</sub> | article | The best single overview, covering training, fine-tuning and serving trade-offs. | Routing, load balancing and capacity factor, why MoEs are harder to fine-tune, and the serving trade-off (every expert must be in memory). | intro |
| [CS336 lecture 4: mixture of experts](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [PDF](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf)<br><sub>Percy Liang, Tatsunori Hashimoto (Stanford CS336), 2025</sub> | slides | Lecture slides that compare the routing and balancing choices of recent open MoE models. | Routing functions, balancing losses, and the design choices of recent open MoE models. | intermediate |
| [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) · [PDF](https://arxiv.org/pdf/1701.06538)<br><sub>Shazeer et al., 2017</sub> | paper | The origin of modern MoE routing. | Noisy top-k gating and the auxiliary losses that keep experts balanced. | intermediate |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) · [PDF](https://arxiv.org/pdf/2101.03961)<br><sub>Fedus, Zoph, Shazeer, 2021</sub> | paper | Made MoE simple enough to train reliably; its load-balancing loss is still the standard reference. | Top-1 routing, the capacity factor and dropped tokens, the auxiliary load-balancing loss, and selective precision for router stability. | intermediate |
| [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) · [PDF](https://arxiv.org/pdf/2401.06066)<br><sub>Dai, Deng, Zhao et al., 2024</sub> | paper | The architecture that DeepSeek-V2 and V3 build on, and the source of the fine-grained, shared-expert design. | Fine-grained expert segmentation plus always-active shared experts, and the evidence that experts specialise more. | advanced |

## State-space models (Mamba)

SSMs are linear recurrences that can also run as convolutions or as a form of attention; Mamba makes them input-dependent. Pairs with [module 04](../curriculum/04-transformers.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [The Annotated S4](https://srush.github.io/annotated-s4/)<br><sub>Sasha Rush, Sidd Karamcheti, 2022</sub> | article | Line-by-line code for the model, which makes the math concrete. | The continuous-time SSM, discretisation, and the recurrent vs convolutional views of the same model. | advanced |
| [State Space Duality (Mamba-2), Part I: The Model](https://goombalab.github.io/blog/2024/mamba2-part1-model/)<br><sub>Albert Gu, Tri Dao, 2024</sub> | article | The authors' own four-part explanation, clearer than the paper. | How a selective SSM and a form of masked linear attention compute the same thing. Read part I (the model) and part III (the algorithm). | advanced |
| [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) · [PDF](https://arxiv.org/pdf/2111.00396)<br><sub>Gu, Goel, Ré, 2021</sub> | paper | The paper that made SSMs competitive on long sequences. | Sections 1-2: SSMs as both recurrences and convolutions, and why naive parameterisations fail. The structured-kernel math is optional. | advanced |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) · [PDF](https://arxiv.org/pdf/2312.00752)<br><sub>Gu, Dao, 2023</sub> | paper | The primary source for selective SSMs. | Section 3: why input-dependent (selective) parameters break the convolution trick, the hardware-aware parallel scan, and the synthetic tasks (selective copying, induction heads) that motivate it. | advanced |
| [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) · [PDF](https://arxiv.org/pdf/2405.21060)<br><sub>Dao, Gu, 2024</sub> | paper | Unifies SSMs and attention, and gives a matmul-based algorithm that uses tensor cores. | The semiseparable-matrix view, and the chunked SSD algorithm. | advanced |

## Mechanistic interpretability

Start with the circuits vocabulary, then superposition, then sparse autoencoders and attribution graphs. Pairs with [lab 16](../labs/16_interpretability/README.md), [module 09](../curriculum/09-interpretability-and-safety.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)<br><sub>Olah et al., 2020</sub> | article | The clearest statement of the research programme, with vision-model examples you can see. | The three claims (features, circuits, universality) and the curve-detector examples. | intro |
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)<br><sub>Elhage, Nanda, Olsson et al. (Anthropic), 2021</sub> | article | The foundation for reasoning about transformers as circuits; most later work uses its vocabulary. | The residual stream as a communication channel, QK and OV circuits, and the zero-, one- and two-layer attention-only analyses that lead to induction heads. | intermediate |
| [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) · [PDF](https://arxiv.org/pdf/2209.10652)<br><sub>Elhage, Hume, Olsson et al., 2022</sub> | paper | Explains why individual neurons are hard to interpret, which motivates sparse autoencoders. | The small ReLU toy model, the phase diagram of when features are stored in superposition, and feature geometry. The HTML version on transformer-circuits.pub is the canonical one. | intermediate |
| [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)<br><sub>Templeton, Conerly, Marcus et al. (Anthropic), 2024</sub> | article | Sparse autoencoders applied to a production model, with the methods and evaluation described in detail. | How the sparse autoencoders are trained and evaluated, the feature examples, and feature steering. | advanced |
| [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)<br><sub>Lindsey, Gurnee, Ameisen et al. (Anthropic), 2025</sub> | article | The current frontier of circuit-level analysis on a production model. | Case studies with attribution graphs (multi-step reasoning, planning ahead when writing poetry) and the stated limitations of the method. | advanced |

Going further: [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895), the mechanism you build in lab 16.

## Evaluation and statistics

An eval score is an estimate with an error bar; most published comparisons ignore that. Pairs with [lab 15](../labs/15_eval_stats/README.md), [module 08](../curriculum/08-evaluation-and-research.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/abs/2411.00640) · [PDF](https://arxiv.org/pdf/2411.00640)<br><sub>Evan Miller, 2024</sub> | paper | The reference for doing statistics on eval results correctly; lab 15 implements it. | The formulas: standard error of an eval score, clustered standard errors for grouped questions, paired differences between two models, and sample-size planning. | intermediate |
| [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) · [PDF](https://arxiv.org/pdf/2107.03374)<br><sub>Chen, Tworek, Jun et al., 2021</sub> | paper | Defines the unbiased pass@k estimator used by every code benchmark. | Section 2.1: the unbiased pass@k estimator, why plugging the empirical pass rate into 1-(1-p)^k is biased, and the numerically stable implementation. | intro |
| [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) · [PDF](https://arxiv.org/pdf/2405.14782)<br><sub>Biderman, Schoelkopf, Sutawika et al., 2024</sub> | paper | Written by the maintainers of lm-evaluation-harness; explains why reported scores for the same model disagree. | The ways evaluations silently differ (prompt format, answer extraction, normalisation, few-shot selection) and the recommended practices. | intermediate |
| [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) · [PDF](https://arxiv.org/pdf/2403.04132)<br><sub>Chiang, Zheng, Sheng et al., 2024</sub> | paper | The statistics behind pairwise-preference leaderboards. | The Bradley-Terry model, confidence intervals, and how model pairs are sampled. | intermediate |
| [The LLM Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook)<br><sub>Clémentine Fourrier (Hugging Face), 2024</sub> | notes | Practical advice from people who ran the Open LLM Leaderboard. | The chapters on automatic benchmarks, human and model-as-judge evaluation, and troubleshooting. As of late 2025 the repository points to a maintained successor. | intro |

## RAG and agents

Retrieval quality and simple, observable control flow matter more than framework choice. Pairs with [lab 17](../labs/17_retrieval/README.md), [module 10](../curriculum/10-applied-llm-systems.md).

| Item | Type | Why it's the best | Focus on | Level |
|---|---|---|---|---|
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)<br><sub>Erik Schluntz, Barry Zhang (Anthropic), 2024</sub> | article | The most useful practical guide: simple composable patterns first, autonomous agents only when needed. | The workflow patterns (prompt chaining, routing, parallelisation, orchestrator-workers, evaluator-optimiser) vs agents, and when not to build an agent. | intro |
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)<br><sub>Lilian Weng, 2023</sub> | article | Still the clearest map of the agent design space. | The planning, memory and tool-use decomposition, with the key papers for each. | intro |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) · [PDF](https://arxiv.org/pdf/2005.11401)<br><sub>Lewis, Perez, Piktus et al., 2020</sub> | paper | The paper that named and defined RAG. | Figure 1 and Section 2: a retriever and generator trained end to end, and RAG-Sequence vs RAG-Token marginalisation. | intermediate |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) · [PDF](https://arxiv.org/pdf/2210.03629)<br><sub>Yao et al., 2022</sub> | paper | The thought-action-observation loop that most agent frameworks still use. | Figure 1 (reasoning-only vs acting-only vs interleaved traces) and the error analysis. | intro |
| [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) · [PDF](https://arxiv.org/pdf/2307.03172)<br><sub>Liu et al., 2023</sub> | paper | A simple, robust finding that changes how you order retrieved context. | The U-shaped accuracy curve over the position of the relevant document, and what it implies for placing retrieved chunks. | intro |

## How to download the PDFs

The downloader uses only the Python standard library, so it runs anywhere Python 3.10+ does, including Windows without WSL.

```bash
python tools/fetch_library.py --list                      # every item, with topic, kind, level and whether a PDF exists
python tools/fetch_library.py --list --topic quantization # one topic
python tools/fetch_library.py --dry-run                   # show what would be downloaded, and where
python tools/fetch_library.py --topic flashattention --topic kv-cache-serving
python tools/fetch_library.py                             # everything with a PDF
python tools/fetch_library.py --id chinchilla --force     # re-download one item
```

On Windows use `py` instead of `python` if `python` is not on your PATH.

What it does:

- Saves each file to `library/pdfs/<topic>/<id>.pdf`. That folder is listed in `.gitignore`.
- Skips files that already exist unless you pass `--force`.
- Uses a browser-like User-Agent, a 60-second timeout and up to 3 attempts with exponential backoff, and waits 1 second between downloads to be polite to arXiv.
- Checks that each response body starts with `%PDF` and refuses to save anything else (for example an HTML error or login page).
- Prints a summary table (ok, skipped, failed) and exits with status 1 if any download failed.

> [!TIP]
> If a download fails behind a corporate or campus proxy, open the item's landing page in a browser instead. Some hosts rate-limit scripts; re-running later usually works because finished files are skipped.

## Licensing and fair use

> [!IMPORTANT]
> The PDFs are for **personal study only**. They are downloaded to your machine and **never committed** to this repository: `library/pdfs/` is git-ignored, and it should stay that way. Do not re-host or redistribute them.

- **This repository links; it does not copy.** Titles, links and the short notes above are the only things stored here.
- **arXiv papers** keep the license their authors chose, shown on each abstract page. Many use arXiv's default non-exclusive license, which lets arXiv distribute the paper but gives you no right to redistribute it.
- **Articles and blogs** (distill.pub, transformer-circuits.pub, Hugging Face, personal blogs) are read online. Distill articles are CC BY 4.0.
- **Course slides** (Stanford CS336) come from the course's public GitHub repository, which is MIT-licensed; cite the course if you reuse a figure.
- The `license` field in `manifest.json` is filled in only where it is known. When it is missing, assume all rights are reserved and check the landing page.

## Adding an item

1. The bar: it must be the best available item for its topic at its level, not merely good. Replace an item rather than growing a topic past five.
2. Open the landing page and the PDF link yourself. Prefer stable URLs: `arxiv.org/abs/<id>` and `arxiv.org/pdf/<id>`, official course sites, and authors' own pages.
3. Add an object to `manifest.json` with `id` (unique, kebab-case), `topic`, `title`, `authors`, `year`, `kind` (`paper`, `notes`, `slides`, `article` or `video`), `pdf_url` (only if a PDF exists), `landing_url`, `license` (if known), `focus` and `level` (`intro`, `intermediate` or `advanced`).
4. Add the matching table row here, then run `python tools/fetch_library.py --list` to check that the manifest parses.

