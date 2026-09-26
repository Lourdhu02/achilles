# Portfolio projects for Meta AI

Eight projects sized for an 8 GB GPU (RTX 5060 class), free Colab or Kaggle, and in most cases a CPU. Each one maps to a Meta team, builds on labs in this repo, and ends in something a Meta engineer can check: a merged PR, a reproducible result with error bars, or a benchmark.
Pick two: one open-source contribution (project 1 or 4) and one research-style reproduction (project 2, 3 or 6).

> [!IMPORTANT]
> Check what is active before you start. As of September 2026, `pytorch/pytorch`, `pytorch/torchtitan` and `pytorch/ao` are active; **torchtune is no longer maintained** and **torchforge's development is paused**. Llama checkpoints on Hugging Face are gated: accept the license on the model page before downloading.

## Contents

| # | Project | Team it speaks to | Labs | Hardware |
|---|---|---|---|---|
| 1 | [A torchao contribution with a quantization study](#1-a-torchao-contribution-with-a-quantization-study) | PyTorch, inference | 13, 03 | 8 GB GPU or Colab |
| 2 | [Reproduce Llama 3's annealing result at small scale](#2-reproduce-llama-3s-annealing-result-at-small-scale) | Pretraining | 05, 02, 15 | 8 GB GPU |
| 3 | [Multi-token prediction at small scale](#3-multi-token-prediction-at-small-scale) | FAIR, pretraining | 05, 14, 15 | 8 GB GPU |
| 4 | [A torch.compile graph-break audit](#4-a-torchcompile-graph-break-audit) | PyTorch compiler | 05, 06 | 8 GB GPU (WSL2 on Windows) |
| 5 | [FSDP2 memory accounting, predicted then measured](#5-fsdp2-memory-accounting-predicted-then-measured) | Training infra, PyTorch distributed | 09, 03 | CPU, or Kaggle 2x T4 |
| 6 | [Local/global attention and KV memory](#6-localglobal-attention-and-kv-memory) | Architecture, inference | 05, 07, 03 | 8 GB GPU |
| 7 | [Lossless speculative decoding on Llama 3.2](#7-lossless-speculative-decoding-on-llama-32) | Inference, on-device | 14, 13, 07 | 8 GB GPU |
| 8 | [A Llama-3-style post-training loop on a 0.5–1B model](#8-a-llama-3-style-post-training-loop-on-a-051b-model) | Post-training | 10, 11, 12, 15 | 8 GB GPU with LoRA |

## What every project must include

- **A prediction before each measurement** (memory, speed or loss), written down, then compared. This is the repo's core habit and exactly what performance-minded interviewers probe.
- **Error bars.** At least three seeds for training results; bootstrap or paired confidence intervals for evals ([lab 15](../../labs/15_eval_stats/README.md)).
- **A one-page write-up**: question, setup, result table, what surprised you, what you would do with 1,000 GPUs. Use the [journal templates](../../journal/README.md).
- **A public repo** with a single command that reproduces the main table.

---

## 1. A torchao contribution with a quantization study

**Scope.** Two parts. (a) Quantize a small model (Llama 3.2 1B or a Qwen model of similar size) with torchao's `quantize_` API using several configs (int8 weight-only, int8 dynamic activation + int8 weight, int4 weight-only) and compare against your own INT8/INT4 code from lab 13. (b) Use what you learned to land a PR in `pytorch/ao`: start with an issue labelled `good first issue`, a docs gap you hit, a missing test, or a benchmark script. If a config fails on your GPU's architecture (the RTX 5060 is Blackwell, sm_120), a minimal reproduction in an issue is itself a useful contribution.

**Predict first.** Weight memory for each config: a 1.24B model is about 2.5 GB in BF16, 1.24 GB in int8 and about 0.62 GB plus scales in int4. Decode at batch size 1 is memory-bound, so predict tokens/s from your measured bandwidth ([lab 03](../../labs/03_napkin_math/README.md), `tools/measure_gpu.py`).

**Labs.** [13](../../labs/13_quantization/README.md), [03](../../labs/03_napkin_math/README.md).

**Success criteria.**
- Table: config × {weight memory, peak memory, decode tokens/s, prefill tokens/s, perplexity on a held-out set with a CI}.
- Your lab-13 INT8 matches torchao's int8 weight-only perplexity within the CI.
- One PR opened with tests, reviewed and merged (or in final review) by day 60.

**Presentation.** Lead with the PR link. Then the table, and one paragraph explaining why int4 speeds up decode but not prefill (decode is bandwidth-bound; prefill is compute-bound and pays a dequantization cost).

## 2. Reproduce Llama 3's annealing result at small scale

**Scope.** The Llama 3 paper anneals the learning rate to zero on a small amount of high-quality data at the end of pretraining, and also uses annealing to measure the value of a dataset cheaply. It reports sizable math-benchmark gains at 8B and negligible ones at 405B. Test the mechanism at 20–50M parameters: pretrain lab 05's model with a warmup-stable-decay schedule on a general corpus (for example a FineWeb-Edu sample), then compare three decay phases: (A) the same general data, (B) a mix with 30% high-quality domain data (for example grade-school math word problems with worked solutions), (C) the domain data alone.

**Predict first.** Which of A, B, C gives the best domain loss, and how much general-loss regression C causes.

**Labs.** [05](../../labs/05_transformer/README.md) (`train.py`), [02](../../labs/02_training_core/README.md) (WSD schedules), [15](../../labs/15_eval_stats/README.md).

**Success criteria.**
- Same checkpoint before decay for all arms; three seeds for the decay phase.
- Report general-validation loss and domain-validation loss with CIs for each arm, and the compute each phase used.
- A clear statement of whether B beats A beyond noise, and what that does and does not say about the 8B-vs-405B finding (a tiny model is closer to the regime where annealing helps).

**Presentation.** A two-panel plot (general loss, domain loss vs. step through the decay) and a short section, "What this would cost at 8B", using the $6ND$ rule.

## 3. Multi-token prediction at small scale

**Scope.** Add $n$ extra output heads to lab 05's model so that head $i$ predicts token $t+i$ from the shared trunk (the design of Gloeckle et al., 2024). Train $n \in \{1, 2, 4\}$ at equal compute on a code corpus (for example a Python subset) and on natural text. Then use the extra heads as a draft for self-speculative decoding with lab 14's verifier.

**Predict first.** The paper finds benefits grow with model size, so at 30M parameters you may see **no** next-token gain. Predict the direction, and predict the acceptance rate of head 2's drafts.

**Implementation detail that matters.** Compute each head's loss and backward one at a time, accumulating gradients into the trunk output, so you never hold $n$ vocabulary-sized logit tensors at once. With vocabulary $V=50{,}257$, batch 16 and sequence 512, one FP32 logit tensor is $16 \times 512 \times 50257 \times 4 \approx 1.6$ GB; four at once would not fit next to the model on 8 GB.

**Labs.** [05](../../labs/05_transformer/README.md), [14](../../labs/14_speculative_decoding/README.md), [15](../../labs/15_eval_stats/README.md).

**Success criteria.**
- Next-token validation loss for each $n$ with seed CIs, on code and on text separately.
- Measured peak memory with and without the sequential-head trick.
- Self-speculative decoding: acceptance rate, tokens/s, and a statistical check that outputs match the base model's distribution.

**Presentation.** Report the negative result plainly if that is what you find; a clean negative result at small scale, correctly interpreted, is a strong signal.

## 4. A torch.compile graph-break audit

**Scope.** Compile lab 05's model and one Hugging Face model (for example a Llama-architecture model of about 1B parameters) with `torch.compile`. Run with `TORCH_LOGS="graph_breaks,recompiles"`, list every graph break and recompilation, classify the causes (data-dependent control flow, unsupported Python, dynamic shapes, `.item()` calls), fix the ones in your code, and measure the speedup of each fix. If you find a break caused by PyTorch rather than user code, reduce it to a minimal reproduction and file an issue in `pytorch/pytorch`, or fix it.

**Predict first.** Which parts of training step time are memory-bound elementwise ops (fusion helps) and which are matmuls (fusion barely helps). Estimate the ceiling on compile's speedup from that split.

**Labs.** [05](../../labs/05_transformer/README.md) (`train.py --compile`), [06](../../labs/06_attention_kernels/README.md) (to read the Triton that Inductor generates; set `TORCH_LOGS=output_code`).

**Success criteria.**
- A table of breaks: location, cause, fix, step-time change.
- Eager vs. compiled step time and peak memory at three batch sizes, with the compile time reported separately.
- One upstream issue or PR with a minimal reproduction.

**Presentation.** Show one generated Triton kernel and explain which ops it fused and why that saved memory traffic.

> [!NOTE]
> On Windows, run this under WSL2: `torch.compile`'s GPU backend generates Triton kernels, and the lab's `train.py` notes `--compile` requires Linux or WSL2.

## 5. FSDP2 memory accounting, predicted then measured

**Scope.** Wrap a small transformer with FSDP2 (`fully_shard`) and compare per-rank memory against DDP at world sizes 1, 2 and 4. Use CPU processes with the `gloo` backend if your PyTorch build supports FSDP2 there, or Kaggle's free two-GPU (T4) notebooks. Then run torchtitan's small debug configuration and read how it applies FSDP2, activation checkpointing and `torch.compile` per block.

**Predict first.** With AdamW in mixed precision, per-parameter state is roughly 2 (BF16 weights) + 4 (FP32 master) + 8 (two FP32 moments) + 2 (BF16 gradients) = 16 bytes. Predict per-rank parameter-plus-optimizer memory as $16P/N$ for FSDP versus $16P$ for DDP, then add activations and the transient all-gathered layer. Lab 09's ZeRO calculator does this.

**Labs.** [09](../../labs/09_parallelism/README.md), [03](../../labs/03_napkin_math/README.md).

**Success criteria.**
- Predicted vs. measured memory within 20% for each world size, with the gap explained (allocator caching, communication buffers, prefetch).
- A timeline trace (`torch.profiler`) showing all-gather and reduce-scatter overlapping with compute, or an explanation of why they don't at this scale.

**Presentation.** A table of predicted and measured memory, one trace screenshot, and a paragraph on what changes at 405B on 16K GPUs (the Llama 3 paper's 4D layout).

## 6. Local/global attention and KV memory

**Scope.** Muse Glimmer uses three sliding-window layers (window 2,048) for every global layer, with RoPE only on the local layers; Llama 4 used a related interleaved design. Add a sliding-window mask and a "NoPE on global layers" option to lab 05's model, and a matching KV-cache layout to lab 07 (a ring buffer for local layers). Compare all-global, 3:1 local/global, and all-local at equal parameters.

**Predict first.** KV memory per token for each layout (see the worked example in the [guide](README.md#a-worked-example-of-the-thinking-these-teams-expect)), and whether all-local fails on a retrieval task beyond its window.

**Labs.** [05](../../labs/05_transformer/README.md), [07](../../labs/07_kv_cache_sampling/README.md), [03](../../labs/03_napkin_math/README.md).

**Success criteria.**
- Validation loss per layout (three seeds), KV memory measured vs. predicted, and decode tokens/s at several context lengths.
- A synthetic key-value retrieval test at distances inside and beyond the window: all-local should fail beyond it; 3:1 should not.
- Loss when evaluating at 2x the training context, per layout.

**Presentation.** One plot of KV memory vs. context length for the three layouts, one of retrieval accuracy vs. distance.

## 7. Lossless speculative decoding on Llama 3.2

**Scope.** Use Llama 3.2 1B as the draft for Llama 3.2 3B (quantize the 3B to int8 so both fit: about 3.2 GB + 2.5 GB of weights) with your exact rejection sampler from lab 14. As a second draft, try Meta's LayerSkip checkpoint of Llama 3.2 1B ([facebook/layerskip-llama3.2-1B](https://huggingface.co/facebook/layerskip-llama3.2-1B)) with early exit as a self-draft.

**Predict first.** From the acceptance rate $\alpha$ and draft length $k$, the expected tokens per target forward pass is $(1-\alpha^{k+1})/(1-\alpha)$. Predict the speedup from your measured draft and target step times.

**Labs.** [14](../../labs/14_speculative_decoding/README.md), [13](../../labs/13_quantization/README.md), [07](../../labs/07_kv_cache_sampling/README.md).

**Success criteria.**
- Speedup vs. $k$ and temperature, measured and predicted on the same plot.
- A statistical test that the output distribution matches the target model's (lab 14's test, applied to real models).
- Measured acceptance rate by task type (code, chat, math).

**Presentation.** Lead with the predicted-vs-measured plot. Note where the model breaks (for example, quantizing the target changes the distribution you are verifying against).

## 8. A Llama-3-style post-training loop on a 0.5–1B model

**Scope.** Llama 3's post-training repeats rounds of SFT, rejection sampling (generate several responses, keep the best by a reward model or verifier) and DPO. Implement two rounds on a 0.5–1B model with LoRA, on a task with a verifiable answer (grade-school math, or unit-tested code). Compare against GRPO from lab 12 at equal generation budget.

**Predict first.** How accuracy changes after each round, and where DPO on self-generated pairs plateaus.

**Labs.** [10](../../labs/10_lora/README.md), [11](../../labs/11_dpo/README.md), [12](../../labs/12_grpo/README.md), [15](../../labs/15_eval_stats/README.md).

**Success criteria.**
- Accuracy after each stage with paired bootstrap CIs on a held-out set of at least 500 problems.
- A contamination check between training prompts and the eval set.
- A failure analysis: 20 examples where the model got worse, categorized.

**Presentation.** A stage-by-stage table and a paragraph comparing iterated DPO with GRPO: sample efficiency, stability and what each is sensitive to.

---

## How to present any of these to Meta

- **Put the link to the merged PR or the reproduction command at the top** of your résumé line and write-up.
- **Quantify in the résumé bullet**: "Cut KV-cache memory 3.8x at 128K context with 3:1 local/global attention; loss within 0.01 nats (3 seeds)".
- **Prepare the 20-minute version** for the project deep dive: the mechanism, the prediction, the surprise, the next experiment. See [portfolio.md](../../tracks/research-engineer/portfolio.md).
