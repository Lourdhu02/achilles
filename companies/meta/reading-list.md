# Meta AI reading list

Twenty-eight papers and documents that explain how Meta's AI organizations build models and the software under them: the Llama lineage, the current Muse models, FAIR's architecture research, efficiency work, and PyTorch's compiler and distributed stack.
Each entry says what to extract, meaning what an interviewer on that team would expect you to know, and which lab in this repo lets you verify it.

> [!TIP]
> Read the five marked **Read first** before anything else. They cover pretraining at scale, the training platform, the compiler, post-training and safety, and the current architecture. Together they are most of what a Meta core-AI interview will probe.

## Contents

- [Read first](#read-first)
- [The Llama and Muse lineage](#the-llama-and-muse-lineage)
- [Post-training, alignment and safety](#post-training-alignment-and-safety)
- [Architecture research from FAIR](#architecture-research-from-fair)
- [Efficiency and on-device](#efficiency-and-on-device)
- [PyTorch, compilers and distributed training](#pytorch-compilers-and-distributed-training)
- [Vision and world models](#vision-and-world-models)
- [Ranking and recommendation](#ranking-and-recommendation)
- [How to read these](#how-to-read-these)

## Read first

| # | Item | Year | Why first |
|---|---|---|---|
| 1 | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 2024 | The most detailed public account of a frontier-scale pretraining and post-training run from any lab |
| 2 | [TorchTitan](https://arxiv.org/abs/2410.06511) | 2024 (ICLR 2025) | How PyTorch composes 4D parallelism today; the codebase where PyTorch's LLM training is being consolidated |
| 3 | [PyTorch 2: TorchDynamo and TorchInductor](https://dl.acm.org/doi/10.1145/3620665.3640366) ([PDF](https://docs.pytorch.org/assets/pytorch2-2.pdf)) | 2024 (ASPLOS) | How `torch.compile` works; essential for PyTorch and performance roles |
| 4 | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | 2023 | The clearest public description of RLHF with separate helpfulness and safety reward models |
| 5 | [Muse Glimmer 30B model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 2026 | Meta's current open-weight architecture and deployment choices in one page |

## The Llama and Muse lineage

| Item | Year | What to extract |
|---|---|---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 2023 | Trained on public data only. Small models trained on far more tokens than Chinchilla-optimal because **inference** cost matters (7B on 1T tokens). The pre-norm RMSNorm + SwiGLU + RoPE recipe that became standard. Verify: [lab 08](../../labs/08_scaling_laws/README.md) inference-aware sizing, [lab 05](../../labs/05_transformer/README.md). |
| [Llama 2](https://arxiv.org/abs/2307.09288) (Read first) | 2023 | 2T tokens, 4K context, GQA in the larger models. Post-training: SFT quality over quantity, two reward models (helpfulness, safety), rejection sampling then PPO, the margin term in the reward-model loss, Ghost Attention for multi-turn system prompts. Verify: [lab 11](../../labs/11_dpo/README.md) for the preference-loss family. |
| [Effective Long-Context Scaling of Foundation Models](https://arxiv.org/abs/2309.16039) | 2023 | Long context via **continued** pretraining on longer sequences with a larger RoPE base frequency, rather than training long from scratch. Why the base matters: lower rotation frequencies keep distant positions distinguishable. |
| [Code Llama](https://arxiv.org/abs/2308.12950) | 2023 | Code specialization from a general base; fill-in-the-middle training; long-context fine-tuning with RoPE base raised from 10,000 to 1,000,000. Useful as a worked example of domain continued-pretraining. |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Read first) | 2024 | 405B dense on about 15.6T tokens; 128K vocabulary; RoPE base 500,000; staged context extension to 128K; annealing on high-quality data at the end; scaling laws that predict **downstream benchmark** performance, not just loss; 4D parallelism (TP, CP, PP, FSDP) and reported 38–43% BF16 MFU; 419 unexpected interruptions in a 54-day window; post-training with SFT, rejection sampling and DPO over several rounds (no PPO); contamination analysis and benchmark confidence intervals. Verify: [lab 03](../../labs/03_napkin_math/README.md) to recompute its compute budget. |
| [Llama 4 launch post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | 2025 | First Llama mixture-of-experts models (Scout, Maverick, both about 17B active parameters); early-fusion multimodality; interleaved attention layers without positional encoding ("iRoPE") for long context; FP8 training; a larger Behemoth teacher used for codistillation. At release Meta published a launch post and model cards rather than a full paper. A [third-party consolidation of the public details](https://arxiv.org/abs/2601.11659) exists; treat it as secondary. |
| [Muse Glimmer 30B model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Read first) | 2026 | Dense 29.6B with a 1.8B perception encoder; 3:1 local/global attention with a 2,048 window and RoPE only on local layers; 32 query / 2 KV heads; gated attention; distilled from the closed Muse Spark; about 4-bit quantization to fit 24–32 GB; a block-diffusion speculative drafter ([DFlash](https://arxiv.org/abs/2602.06036)) with 3.1x reported speedup on an RTX 5090. Compute its KV cache by hand (the worked example in the [guide](README.md#a-worked-example-of-the-thinking-these-teams-expect)). |

## Post-training, alignment and safety

| Item | Year | What to extract |
|---|---|---|
| [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | 2023 | 1,000 carefully curated SFT examples on a 65B base; the "superficial alignment hypothesis" that pretraining provides the knowledge and SFT mostly teaches format. Know where later work disagrees (reasoning and RL-heavy post-training). |
| [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | 2024 | The model judges its own samples (LLM-as-a-judge prompt) to build preference pairs, then iterative DPO. Failure mode to discuss: reward hacking and judge drift when the judge and policy are the same model. Verify: [lab 11](../../labs/11_dpo/README.md). |
| [Llama Guard](https://arxiv.org/abs/2312.06674) | 2023 | A safety classifier built as an instruction-tuned LLM, with the risk taxonomy in the prompt so policies can change without retraining; separate input and output classification. Relevant to system-level safety design questions. |

## Architecture research from FAIR

| Item | Year | What to extract |
|---|---|---|
| [Better and Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737) | 2024 | n output heads on a shared trunk each predict a future token; gains grow with model size and are largest on code; the extra heads enable self-speculative decoding. Note the memory trick: compute each head's forward and backward sequentially so logits for all heads are never held at once. Verify: [lab 14](../../labs/14_speculative_decoding/README.md). |
| [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818) | 2024 | Images tokenized into discrete tokens and modeled with text in one sequence. The stability section is the part to know: mixed-modality training diverged, and QK-norm plus changed norm placement fixed it. Verify: QK-norm in [lab 05](../../labs/05_transformer/README.md). |
| [Transfusion](https://arxiv.org/abs/2408.11039) | 2024 | One transformer trained with next-token loss on text and a diffusion loss on continuous image patches, with bidirectional attention inside each image. Compare with Chameleon's discrete approach. |
| [Mixture-of-Transformers](https://arxiv.org/abs/2411.04996) | 2024 | Modality-specific weights (FFN, attention projections, norms) with shared global self-attention; matches dense baselines at a fraction of the FLOPs. A sparsity pattern that is not token-routed MoE. |
| [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) | 2024 | Tokenizer-free: bytes grouped into dynamic patches where a small byte model's next-byte entropy is high; FLOP-matched scaling studies show patches can scale better than tokens. Verify: [lab 04](../../labs/04_tokenizer/README.md) to see what BPE does to non-English scripts. |
| [Large Concept Models](https://arxiv.org/abs/2412.08821) | 2024 | Autoregressive modeling in a sentence-embedding space instead of token space; diffusion-based variants. A research bet on higher-level representations. |
| [Training LLMs to Reason in a Continuous Latent Space (Coconut)](https://arxiv.org/abs/2412.06769) | 2024 | Feed the last hidden state back as the next input embedding ("continuous thought") instead of decoding a token; a curriculum replaces chain-of-thought steps gradually. Discuss what is lost: interpretability and monitorability of reasoning. |

## Efficiency and on-device

| Item | Year | What to extract |
|---|---|---|
| [LayerSkip](https://arxiv.org/abs/2404.16710) | 2024 | Layer dropout rising with depth plus an early-exit loss so early layers can predict tokens; self-speculative decoding with early layers as the draft and the remaining layers as verifier, sharing compute and cache. Verify: [lab 14](../../labs/14_speculative_decoding/README.md). |
| [SpinQuant](https://arxiv.org/abs/2405.16406) | 2024 | Rotating weights and activations by an orthogonal matrix leaves the network's function unchanged but removes outliers; learning the rotation (instead of a random Hadamard) makes 4-bit weights, activations and KV cache usable. Verify: [lab 13](../../labs/13_quantization/README.md), then add a random rotation before INT4 and measure the error. |
| [MobileLLM](https://arxiv.org/abs/2402.14905) | 2024 | For sub-billion models, deep-and-thin beats wide at equal parameters; embedding sharing, GQA and block-wise weight sharing. Useful for any on-device (ExecuTorch) discussion. |

## PyTorch, compilers and distributed training

| Item | Year | What to extract |
|---|---|---|
| [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703) | 2019 | The design principles: Python-first, define-by-run, the caching allocator, why eager mode won. Verify: [lab 01](../../labs/01_autograd/README.md) builds the same reverse-mode idea. |
| [PyTorch 2 (TorchDynamo and TorchInductor)](https://dl.acm.org/doi/10.1145/3620665.3640366) (Read first) | 2024 | Dynamo hooks CPython frame evaluation to capture FX graphs from bytecode, with **guards** that trigger recompilation and **graph breaks** where Python can't be traced; AOTAutograd traces the backward ahead of time; Inductor generates Triton (GPU) and C++ (CPU) with fusion. Practice: run `TORCH_LOGS=graph_breaks,recompiles` on lab 05's `train.py --compile`. |
| [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) | 2023 | Sharding parameters, gradients and optimizer state; all-gather before forward and backward, reduce-scatter after; prefetching and rate limiting to overlap communication; hybrid sharding. The paper describes the original flat-parameter design; FSDP2 (used in torchtitan) shards each parameter as a DTensor instead. Know why that matters: per-parameter dtype and freezing, and cleaner composition with tensor parallelism. Verify: [lab 09](../../labs/09_parallelism/README.md) ZeRO memory. |
| [TorchTitan](https://arxiv.org/abs/2410.06511) (Read first) | 2024 | Composable FSDP2, tensor parallelism (including async TP), pipeline and context parallelism; Float8 training; `torch.compile` per block; distributed checkpointing. Read the code alongside the paper: [pytorch/torchtitan](https://github.com/pytorch/torchtitan). |
| [TorchAO: PyTorch-native training-to-serving model optimization](https://openreview.net/attachment?id=HpqH0JakHf&name=pdf) | 2025 (CodeML @ ICML) | One library for quantization-aware training, float8 and MX training, weight-only int4/int8 inference and sparsity, designed to compose with `torch.compile` and FSDP2. The code is at [pytorch/ao](https://github.com/pytorch/ao). |

## Vision and world models

| Item | Year | What to extract |
|---|---|---|
| [DINOv2](https://arxiv.org/abs/2304.07193) | 2023 | Self-supervised visual features from a curated dataset; self-distillation objectives; the data-curation pipeline is the underrated part. |
| [Perception Encoder](https://arxiv.org/abs/2504.13181) | 2025 | Contrastive vision-language training alone yields general features, but the best ones sit in intermediate layers; alignment steps bring them to the output. This family is the vision encoder in Muse Glimmer. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 2025 | Predict masked video in representation space rather than pixels; an action-conditioned variant plans robot actions with little robot data. The core of FAIR's world-model agenda. |

## Ranking and recommendation

| Item | Year | What to extract |
|---|---|---|
| [DLRM](https://arxiv.org/abs/1906.00091) | 2019 | Embedding tables for sparse features plus MLPs, with explicit pairwise feature interactions; model-parallel embeddings and data-parallel MLPs. The basis of many ML system design questions at Meta. |
| [Actions Speak Louder than Words (HSTU, generative recommenders)](https://arxiv.org/abs/2402.17152) | 2024 | Recommendation reframed as sequential transduction over user action histories with a modified attention; shows recommendation quality scaling with compute in a way classic DLRMs did not. Bridges LLM and ranking work. |

## How to read these

1. **Three passes.** Abstract and figures; then method and main tables; then the appendix only for papers you will reproduce. The Llama 3 paper's infrastructure and evaluation sections deserve a full pass.
2. **Extract numbers into your journal.** Model sizes, token counts, context lengths, MFU, ablation deltas. Then recompute one of them with [lab 03](../../labs/03_napkin_math/README.md).
3. **Write one paragraph per paper** answering: what problem, what mechanism, what evidence, what would I test next. Use the [journal templates](../../journal/README.md).
4. **Tie papers to code.** For PyTorch items, read the matching source in `pytorch/pytorch` or `pytorch/torchtitan`; for model papers, find the corresponding config in torchtitan's model folder.

See also the repo-wide [papers.md](../../curriculum/papers.md), and [projects.md](projects.md) for how to turn several of these into portfolio work.
