# Reading list: Google DeepMind and Google's AI teams

Thirty primary sources, grouped by theme, that Google interviewers are likely to know well because their colleagues wrote them. For each: link, year, and what to extract so you can discuss it precisely.
Start with the five marked **read first**. Pair each with its lab in this repo; reading without building does not stick (see [module 00](../../curriculum/00-learning-os.md#reading-papers)).

## Contents

- [Read first](#read-first)
- [Transformer foundations](#transformer-foundations)
- [Scaling and pretraining](#scaling-and-pretraining)
- [Systems: TPUs, sharding, Pathways](#systems-tpus-sharding-pathways)
- [Efficient attention and decoding](#efficient-attention-and-decoding)
- [Mixture of experts](#mixture-of-experts)
- [Gemini and Gemma](#gemini-and-gemma)
- [Post-training, reasoning and test-time compute](#post-training-reasoning-and-test-time-compute)
- [Interpretability](#interpretability)
- [How to read these](#how-to-read-these)

## Read first

| # | Item | Why first |
|---|---|---|
| 1 | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (2025), chapters 1–4 and 7 | The shared mental model of GDM engineers: rooflines, TPU hardware, sharded matmuls, transformer FLOPs, inference. It turns lab 03 into TPU fluency. |
| 2 | [Chinchilla](https://arxiv.org/abs/2203.15556) (2022) | The compute-optimal scaling result every pretraining conversation starts from. Maps to [lab 08](../../labs/08_scaling_laws/README.md). |
| 3 | [PaLM](https://arxiv.org/abs/2204.02311) (2022) | The most detailed public account of a Google-scale TPU training run: architecture choices, Pathways, MFU, loss spikes. |
| 4 | [Gemma 3 technical report](https://arxiv.org/abs/2503.19786) (2025), then the [Gemma 4 report](https://arxiv.org/abs/2607.02770) (2026) | The open models you will fine-tune and inspect; read Gemma 3 in full, then read Gemma 4 for what changed. |
| 5 | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022) | Short, exact and from Google; you implement it in [lab 14](../../labs/14_speculative_decoding/README.md). |

## Transformer foundations

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Scaled dot-product attention and why the $1/\sqrt{d_k}$ factor keeps logit variance near 1; multi-head attention as parallel subspaces; the encoder–decoder layout that decoder-only models later dropped. | [05](../../labs/05_transformer/README.md) |
| [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | 2019 | Everything framed as text-to-text; span-corruption pretraining; the C4 dataset and its filtering heuristics; relative position biases; and above all the *method*: one controlled ablation per design choice. Use it as a template for your own write-ups. | [05](../../labs/05_transformer/README.md), [module 08](../../curriculum/08-evaluation-and-research.md) |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer) | 2020 | Where SwiGLU comes from; why the hidden size is scaled to about $\tfrac{2}{3}\cdot 4d$ to keep parameters matched with a 3-matrix MLP. | [05](../../labs/05_transformer/README.md) |

## Scaling and pretraining

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) | 2022 | Three independent estimation methods agreeing that parameters and tokens should scale roughly equally with compute; the ~20 tokens per parameter rule of thumb; Chinchilla (70B, 1.4T tokens) beating the much larger Gopher at the same compute. Know why inference cost pushes real models past the "optimal" token count. | [08](../../labs/08_scaling_laws/README.md) |
| [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | 2022 | 540B dense model trained on 6144 TPU v4 chips across two pods with Pathways. Architecture: SwiGLU, parallel attention and MLP in each block, multi-query attention, RoPE, shared input/output embeddings, no biases. The paper defines model FLOPs utilization (MFU). Loss spikes were handled by restarting from an earlier checkpoint and skipping a few hundred batches. | [03](../../labs/03_napkin_math/README.md), [05](../../labs/05_transformer/README.md) |
| [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) | 2023 | Two instabilities reproduced in small models at high learning rates: attention-logit growth (fixed by QK-layernorm) and output-logit divergence (fixed by z-loss). A method you can run on an 8 GB GPU: measure LR sensitivity across sizes. | [02](../../labs/02_training_core/README.md), [05](../../labs/05_transformer/README.md) |
| [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442) | 2023 | Where QK normalization entered mainstream large-model practice, plus parallel layers and omitted biases for efficiency. Read the stability section only. | [05](../../labs/05_transformer/README.md) |
| [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872) | 2024 | A large GDM sweep over parameterizations (standard, μP and others) and optimizers; per-layer learning-rate prescriptions; the finding that Adam's epsilon matters at scale and the Adam-atan2 variant that removes it. | [02](../../labs/02_training_core/README.md), [module 03](../../curriculum/03-deep-learning.md) |

## Systems: TPUs, sharding, Pathways

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Austin et al., GDM) | 2025 | Critical arithmetic intensity (~240 FLOPs/byte on TPU v5e); sharded-matmul notation and when each collective (AllGather, ReduceScatter, AllReduce) is needed; FSDP vs tensor vs pipeline parallelism on TPU topologies; KV-cache and latency analysis for inference; the profiling chapter. Do the worked problems. | [03](../../labs/03_napkin_math/README.md), [09](../../labs/09_parallelism/README.md) |
| [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) | 2021 | You annotate a few tensors with shardings and the XLA compiler propagates them and inserts collectives. This is what `jax.jit` with `NamedSharding` does today. | [09](../../labs/09_parallelism/README.md) |
| [Pathways: Asynchronous Distributed Dataflow for ML](https://arxiv.org/abs/2203.12533) | 2022 | Single-controller vs multi-controller designs; gang-scheduling of computations across TPU pods; how one client program drives thousands of chips. Useful background for infra interviews. | — |
| [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Pope et al.) | 2022 | How to choose partitioning layouts for inference by model size, batch and latency target; why multi-query attention pairs with sharding the batch for the KV cache; the prefill vs decode trade-off expressed in MFU and latency. The best single source for TPU serving questions. | [07](../../labs/07_kv_cache_sampling/README.md), [module 07](../../curriculum/07-inference.md) |

## Efficient attention and decoding

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) (Shazeer) | 2019 | The memory-bandwidth argument: in incremental decoding, reloading K and V for every head dominates; sharing one K/V head across all query heads cuts KV bytes by the head count. Be able to redo the ratio of memory access to arithmetic. | [05](../../labs/05_transformer/README.md), [07](../../labs/07_kv_cache_sampling/README.md) |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | 2023 | Groups of query heads share a K/V head, interpolating between MHA and MQA; existing MHA checkpoints can be uptrained to GQA with about 5% of the original pretraining compute by mean-pooling K/V heads. | [05](../../labs/05_transformer/README.md) |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan, Kalman, Matias) | 2022 | The modified rejection-sampling rule that makes the output distribution identical to the target model's; the expected tokens per target call, $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ for acceptance rate $\alpha$ and $\gamma$ draft tokens; how to choose $\gamma$ given the draft/target cost ratio. | [14](../../labs/14_speculative_decoding/README.md) |
| [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et al., DeepMind) | 2023 | The same idea discovered independently at DeepMind, tested on Chinchilla 70B; compare the two proofs of losslessness. | [14](../../labs/14_speculative_decoding/README.md) |

## Mixture of experts

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al.) | 2017 | Noisy top-k gating and auxiliary load-balancing losses; the original "conditional computation" argument. | [module 04](../../curriculum/04-transformers.md) |
| [GShard](https://arxiv.org/abs/2006.16668) | 2020 | Top-2 routing with expert capacity limits for a 600B-parameter translation MoE, and sharding annotations that later became GSPMD. | [09](../../labs/09_parallelism/README.md) |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | 2021 | Top-1 routing; the capacity factor and what happens to overflow tokens; selective precision (router computed in float32) for stability; the load-balancing loss you should be able to write down. | [module 04](../../curriculum/04-transformers.md) |
| [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) | 2022 | The router z-loss and why it stabilises training; why sparse models can overfit during fine-tuning; practical design recommendations. | [module 05](../../curriculum/05-pretraining.md) |

## Gemini and Gemma

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | 2023 | Natively multimodal training from the start; the Ultra/Pro/Nano split; the infrastructure section on training across TPU pods (failure handling, determinism, silent data corruption). | [module 05](../../curriculum/05-pretraining.md) |
| [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530) | 2024 | Long-context evaluation methodology (needle-in-a-haystack across text, audio and video, up to millions of tokens) and its limits; an MoE model at the frontier. | [15](../../labs/15_eval_stats/README.md) |
| [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261) | 2025 | How GDM describes "thinking" models and their evaluation; the cost/quality frontier across model sizes; safety and frontier-safety evaluation sections. For later Gemini generations, read the current model cards on the GDM site. | [12](../../labs/12_grpo/README.md), [15](../../labs/15_eval_stats/README.md) |
| [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) | 2024 | Interleaved local sliding-window and global attention; logit soft-capping; knowledge distillation from a larger teacher to train the smaller models; GQA. (The original [Gemma](https://arxiv.org/abs/2403.08295) report, 2024, is the baseline.) | [05](../../labs/05_transformer/README.md) |
| [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) | 2025 | Five local layers per global layer to shrink the KV cache at long context; QK-norm replacing soft-capping; 128K context (32K for the 1B model); a SigLIP vision encoder; distillation-based training; the 262K-entry vocabulary. Compute the KV-cache saving of the 5:1 layout yourself. | [05](../../labs/05_transformer/README.md), [07](../../labs/07_kv_cache_sampling/README.md) |
| [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) | 2026 | What changed from Gemma 3: dense and MoE variants (the 26B-A4B model has 25.2B total and 3.8B active parameters, with 8 of 128 experts active plus 1 shared); per-layer embeddings in the E2B/E4B models; unified keys and values and proportional RoPE on global layers; up to 256K context; Apache 2.0 licence. Read the architecture and post-training sections. | [05](../../labs/05_transformer/README.md), [10](../../labs/10_lora/README.md) |

## Post-training, reasoning and test-time compute

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Scaling Instruction-Finetuned Language Models (Flan-PaLM / Flan-T5)](https://arxiv.org/abs/2210.11416) | 2022 | Scaling the number of instruction tasks and model size; including chain-of-thought data in fine-tuning to keep reasoning ability; how the gains were measured. | [10](../../labs/10_lora/README.md) |
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | 2022 | Few-shot reasoning traces help only above a certain model scale; the evaluation setup on arithmetic and symbolic tasks. Historical root of "thinking" models. | [12](../../labs/12_grpo/README.md) |
| [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (Snell et al.) | 2024 | Two ways to spend inference compute (search against a verifier, revising the output distribution) and a "compute-optimal" policy that adapts to problem difficulty; the FLOPs-matched comparison with a much larger model. | [12](../../labs/12_grpo/README.md), [15](../../labs/15_eval_stats/README.md) |

## Interpretability

| Item | Year | What to extract | Lab |
|---|---|---|---|
| [Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2](https://arxiv.org/abs/2408.05147) | 2024 | JumpReLU SAEs trained on every layer of Gemma 2 2B and 9B; how they measure reconstruction vs sparsity; how to use released SAEs as a research substrate on a single GPU. | [16](../../labs/16_interpretability/README.md) |

## How to read these

- **Three passes.** Abstract, figures and conclusion first; then the method with a pen; then reproduce one number (a FLOP count, a memory saving, a speedup) with your own arithmetic or code.
- **One-paragraph notes** in your journal per paper: the claim, the evidence, one thing you doubt, one experiment you could run on 8 GB.
- **Interview framing.** For each read-first item, prepare a two-minute explanation and one follow-up you would try. Google interviewers often ask "what would you change?" rather than "what does it say?".
- Broader reading by module is in [curriculum/papers.md](../../curriculum/papers.md).
