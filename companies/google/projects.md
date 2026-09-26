# Projects for a Google DeepMind / Google AI application

Nine projects that show Google-relevant depth: JAX and TPUs, Gemma, rigorous evaluation, and the papers Google wrote. Each fits an 8 GB GPU, a free Colab or Kaggle notebook, or TPU Research Cloud (TRC) quota.
Do two or three well rather than all nine. One JAX project plus one Gemma project plus one open-source PR is a strong set.

## Contents

- [Compute options](#compute-options)
- [Project summary](#project-summary)
- [1. Lab 05 GPT in JAX on a free TPU](#1-lab-05-gpt-in-jax-on-a-free-tpu)
- [2. A rigorous Gemma fine-tune](#2-a-rigorous-gemma-fine-tune)
- [3. Speculative decoding with a Gemma draft model](#3-speculative-decoding-with-a-gemma-draft-model)
- [4. Rooflines on TPU and on your GPU](#4-rooflines-on-tpu-and-on-your-gpu)
- [5. A Pallas attention kernel](#5-a-pallas-attention-kernel)
- [6. A miniature Chinchilla study in JAX](#6-a-miniature-chinchilla-study-in-jax)
- [7. GRPO on a small Gemma with a verifiable reward](#7-grpo-on-a-small-gemma-with-a-verifiable-reward)
- [8. Indic-language features with Gemma Scope](#8-indic-language-features-with-gemma-scope)
- [9. A merged contribution to the JAX or Gemma ecosystem](#9-a-merged-contribution-to-the-jax-or-gemma-ecosystem)
- [How to present the work](#how-to-present-the-work)

## Compute options

*As of September 2026; quotas and hardware change, so check each service's current page.*

| Option | What you get | Good for | Watch out for |
|---|---|---|---|
| RTX 5060 laptop (8 GB) | Local CUDA GPU | PyTorch work, Gemma inference and LoRA up to a few billion parameters in 4-bit | JAX GPU builds are Linux-only: use WSL2, and check that your `jaxlib` build supports Blackwell consumer GPUs |
| CPU only | Any machine | JAX correctness tests; simulate 8 devices with `XLA_FLAGS=--xla_force_host_platform_device_count=8` to test sharding code | Slow; keep models tiny |
| Kaggle notebooks | A free weekly quota of GPU and TPU time | Projects 1, 4, 6; public notebooks double as write-ups | Session time limits; save checkpoints often |
| Colab (free tier) | GPU and TPU runtimes when available | Quick experiments | Availability varies; runtimes reset |
| TPU Research Cloud | Free Cloud TPU quota for a limited period after acceptance ([apply](https://sites.research.google/trc/about/)) | Projects 1, 5, 6, 7 at real scale | You pay for VMs and storage used alongside; set a billing alert; you are expected to publish the work |

## Project summary

| # | Project | Maps to | Labs | Compute | Weeks |
|---|---|---|---|---|---|
| 1 | Lab 05 GPT in JAX | Pretraining, TPU infra | 01, 02, 05, 09 | CPU → Kaggle/TRC TPU | 3 |
| 2 | Rigorous Gemma fine-tune | Post-training, evals | 10, 15 | 8 GB GPU | 2–3 |
| 3 | Speculative decoding with Gemma | Serving | 07, 14 | 8 GB GPU | 2 |
| 4 | Rooflines on TPU and GPU | Infra, serving | 03 | Kaggle/Colab TPU + 8 GB GPU | 1 |
| 5 | Pallas attention kernel | Kernels, TPU infra | 06 | CPU (interpret mode) → TPU | 3 |
| 6 | Miniature Chinchilla | Pretraining, research | 05, 08, 15 | Kaggle/TRC TPU or 8 GB GPU | 3 |
| 7 | GRPO on a small Gemma | Post-training / RL | 12, 15 | 8 GB GPU or TRC | 3–4 |
| 8 | Indic features with Gemma Scope | Interpretability, safety | 04, 16 | 8 GB GPU | 2–3 |
| 9 | Merged OSS contribution | Any team | depends | any | ongoing |

---

## 1. Lab 05 GPT in JAX on a free TPU

**Scope.** Port your [lab 05](../../labs/05_transformer/README.md) GPT (RMSNorm, RoPE, GQA, SwiGLU, QK-norm) to JAX with Flax (NNX API) and Optax, train it on TinyStories as in `labs/05_transformer/train.py`, then shard it across 8 devices.

**Steps.**
1. Write the model as pure functions or Flax NNX modules. Load the weights of your trained PyTorch model into it and check logits match (fp32, same input) to about 1e-5. This parity test is the heart of the project.
2. Training step under `jax.jit`: loss, `jax.value_and_grad`, Optax AdamW with warmup and cosine or WSD schedule (compare with your [lab 02](../../labs/02_training_core/README.md) AdamW).
3. Fix shapes (pad to a fixed block size) so `jit` compiles once; log compile time separately from step time.
4. Shard with a `jax.sharding.Mesh` and `NamedSharding`: data parallel first, then shard the MLP weights across a "model" axis (the Megatron split from [lab 09](../../labs/09_parallelism/README.md)). Develop on CPU with 8 simulated devices, then run on a Kaggle or TRC TPU.
5. Profile one step with the JAX profiler and identify the top three ops by time.

**Success criteria.** Logit parity with PyTorch; loss curves from both frameworks overlap within seed-to-seed spread (3 seeds each); MFU measured on TPU with the FLOPs from [lab 03](../../labs/03_napkin_math/README.md); a one-paragraph explanation of what the profiler showed and one change that improved step time.

**Why Google cares.** It demonstrates JAX, XLA compilation behaviour and GSPMD sharding, which is the daily work of GDM pretraining and infra teams.

## 2. A rigorous Gemma fine-tune

**Scope.** Fine-tune a small Gemma model on one task with an automatic metric and report the result the way a GDM eval team would: seeds, confidence intervals, a paired comparison against the base model, and a contamination check.

**Model choices for 8 GB (predict the memory first, then measure).**

| Model | Method | Weight memory | Notes |
|---|---|---|---|
| Gemma 3 270M | Full fine-tune | ~268M params × ~16 bytes (bf16 weights, fp32 master, grads, Adam moments) ≈ 4.3 GB | Most of the parameters (~168M) are the 262,144 × 640 embedding table |
| Gemma 3 1B | LoRA, bf16 base | ~2 GB of frozen weights | Use gradient checkpointing for long sequences |
| Gemma 4 E2B | QLoRA (4-bit base) | 5.1B parameters including per-layer embeddings, so ~2.6 GB in 4-bit before overheads | Text-only tasks can skip the audio and vision encoders |

> [!WARNING]
> **The vocabulary is the memory trap.** Gemma's 262,144-entry vocabulary makes the logits huge: one sequence of 1,024 tokens produces $1024 \times 262144 \times 4$ bytes $\approx$ 1.07 GB of fp32 logits, and the backward pass needs a same-sized gradient. A batch of two sequences therefore needs ~4 GB for logits alone, on top of the 4.3 GB of optimizer state for the 270M full fine-tune. Compute the loss in chunks of tokens (or use a fused cross-entropy kernel) so the full logits tensor never exists. Explaining this in an interview shows you understand where memory goes.

**Task ideas.** A structured-output task with exact-match scoring; an Indic-language task (e.g. transliteration, or question answering in Telugu, Tamil or Hindi) with a held-out set you build yourself; a small maths task scored by exact answer.

**Method.**
1. Freeze a test set before training. Check for overlap between train and test (n-gram overlap) and with anything the base model may have seen.
2. Baseline: the base model with a fixed prompt, and the instruction-tuned model.
3. Train with three seeds. Report mean and a bootstrap 95% CI per condition, and a paired test (same test items) for "fine-tuned vs base", using [lab 15](../../labs/15_eval_stats/README.md).
4. One ablation that matters: LoRA rank, learning rate or data size. Plot the metric with error bars.
5. Error analysis: 30 failures, categorised by hand.

**Success criteria.** A results table where every number has a CI; a memory table where your predictions were within ~20% of measured peak memory (and an explanation where they were not); a model card listing data, limitations and intended use.

## 3. Speculative decoding with a Gemma draft model

**Scope.** Implement speculative decoding from [lab 14](../../labs/14_speculative_decoding/README.md) on real models: Gemma 3 270M drafting for Gemma 3 1B (they share the tokenizer), on the RTX 5060.

**Steps.**
1. Use your lab 14 acceptance rule with the real models' probabilities; verify losslessness statistically on a small prompt set (compare token distributions with and without speculation).
2. Measure the acceptance rate $\alpha$ per task type (code, chat, a translation task) and compare measured tokens per target call with $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ for $\gamma = 1\ldots 8$.
3. Measure the draft/target cost ratio $c$ directly (time per decode step), then find the best $\gamma$ from the expected-speedup formula and check it against wall-clock.

**The insight to find.** Decode is memory-bound, so the cost ratio tracks bytes read per step, not "parameter count". Gemma 3 270M reads its full 262,144 × 640 output projection every step (~168M of its ~268M parameters), so the draft is less cheap relative to the 1B target than the names suggest. Quantify this and try one fix (for example, a draft that restricts the output vocabulary and falls back to the target, then prove whether the result is still exact).

**Success criteria.** Losslessness check passes; predicted vs measured speedup within ~20% across $\gamma$; a clear explanation of why the speedup differs by task.

## 4. Rooflines on TPU and on your GPU

**Scope.** Measure the roofline knee on a TPU (Kaggle, Colab or TRC) and on the RTX 5060, and connect it to serving.

**Steps.**
1. Time bf16 matmuls $[B, D] \times [D, F]$ with $D = F = 8192$ for $B$ from 1 to 4096 under `jax.jit` (use `block_until_ready()` and exclude compile time). Plot achieved FLOP/s against $B$.
2. Predict the knee as peak FLOP/s ÷ HBM bandwidth (about 240 on TPU v5e per [How to Scale Your Model](https://jax-ml.github.io/scaling-book/roofline/); use `tools/measure_gpu.py` numbers for your GPU). Compare with the measured knee.
3. Repeat for a decode step of your project-1 model: tokens/s against batch size. Where does it stop scaling, and why?

**Success criteria.** Predicted and measured knees within a factor of ~1.5, with an explanation of the gap (clock, layout padding to 128 or 256, kernel overheads). This is lab 03 turned into evidence.

## 5. A Pallas attention kernel

**Scope.** Rewrite the FlashAttention forward from [lab 06](../../labs/06_attention_kernels/README.md) in Pallas, JAX's kernel language.

**Steps.**
1. Start from a Pallas softmax or matmul example. Run with `interpret=True` on CPU for correctness, just as lab 06 uses the Triton interpreter.
2. Implement the tiled forward with online softmax and causal masking. Test against a naive JAX reference across shapes and dtypes.
3. On a TPU, benchmark against the naive version and against the flash-attention kernel that ships in JAX (`jax/experimental/pallas/ops/tpu/`). Explain the gap by reading that kernel's block sizes and memory layout.

**Success criteria.** Correct to bf16 tolerance on all tested shapes; a benchmark table; a short note on how TPU tiling constraints (lane and sublane sizes, VMEM capacity) differ from GPU shared memory. A stretch goal is the backward pass.

## 6. A miniature Chinchilla study in JAX

**Scope.** Reproduce the *method* of Chinchilla's IsoFLOP approach at tiny scale using your project-1 model and [lab 08](../../labs/08_scaling_laws/README.md)'s fitting code.

**Steps.**
1. Choose three or four compute budgets (for example $10^{15}$ to $10^{17}$ FLOPs, computed with $6ND$). At each budget, train 5–7 model sizes with tokens set so $6ND$ equals the budget.
2. Fit a parabola in $\log N$ per budget to find the loss-minimising size, then fit $N_{opt} \propto C^{a}$.
3. Bootstrap over runs to put a CI on the exponent $a$. Compare with Chinchilla's ~0.5 and discuss why tiny models, a single dataset and fixed hyperparameters bias your estimate.

**Success criteria.** A plot of IsoFLOP curves, a fitted exponent with a CI, and a limitations section a reviewer would accept. The honest discussion is worth more than matching the paper.

## 7. GRPO on a small Gemma with a verifiable reward

**Scope.** Scale [lab 12](../../labs/12_grpo/README.md) from the toy counting task to Gemma 3 270M or 1B with LoRA on a task with an exact verifier (arithmetic, a countdown-style puzzle, or format-constrained answers). On TRC, try Google's JAX post-training library [Tunix](https://github.com/google/tunix), which supports GRPO.

**Method.** Group size, KL coefficient and clipping from lab 12; a held-out evaluation set; three seeds. Log reward, response length, KL to the reference and held-out accuracy.

**Success criteria.** Held-out accuracy improvement with a CI and a paired test against the starting model; at least one documented failure mode (reward hacking, length blow-up, entropy collapse) with the evidence and the fix you tried.

## 8. Indic-language features with Gemma Scope

**Scope.** Use the open [Gemma Scope](https://arxiv.org/abs/2408.05147) sparse autoencoders on Gemma 2 2B (about 5 GB of weights in bf16, so it fits for inference on 8 GB) to study how the model represents Indic scripts and languages.

**Steps.**
1. Tokenization first: with your [lab 04](../../labs/04_tokenizer/README.md) tooling, measure bytes per token and tokens per word for Hindi, Telugu, Tamil and English on the same parallel sentences.
2. Find SAE features that fire on a script or a language. Validate each with held-out text and with counterexamples (romanised Hindi, mixed-script text).
3. Intervene: ablate or amplify a feature and measure the effect on next-token loss and on the output language, with the activation-patching method from [lab 16](../../labs/16_interpretability/README.md).

**Success criteria.** Two or three features with quantitative evidence (activation statistics, intervention effects with CIs), and a clear statement of what the evidence does *not* show. This ties together multilingual work, interpretability and safety, all active areas at GDM, including in Bengaluru.

## 9. A merged contribution to the JAX or Gemma ecosystem

**Scope.** One substantive merged pull request to a Google open-source ML repository ([JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), [Orbax](https://github.com/google/orbax), [Grain](https://github.com/google/grain), [MaxText](https://github.com/AI-Hypercomputer/maxtext), [Tunix](https://github.com/google/tunix), [Gemma](https://github.com/google-deepmind/gemma)) or Gemma support in a widely used inference engine.

**How to find one.** Work through projects 1–7 and write down every bug, confusing error message and missing feature you hit. Search the repo's issues for it. Start with an issue that has a reproducible example; comment with your analysis before writing code; include a test and, for performance changes, a benchmark.

**Success criteria.** Merged, with a test. A second PR in the same repo is worth more than a first PR in a different one, because it shows you can work with a team's reviewers over time.

## How to present the work

Google's engineering culture runs on code review and design documents. Present your projects in that form.

- **Repository.** Installable package, tests that run in CI, a pinned environment, and one command that reproduces the headline number.
- **README results table.** Every metric with a CI and the number of seeds; hardware and wall-clock stated.
- **A two-page design doc** per project: context, goals and non-goals, design, alternatives considered, measurements, open questions. Link it from the README.
- **A write-up** (blog, Kaggle notebook or arXiv note) for the best one or two projects. Put the most surprising measurement in the first paragraph.
- **Résumé line.** Impact form with a number you can defend: "Ported a 20M-parameter GPT to JAX with logit parity to PyTorch; reached X% MFU on a TPU v5e-8 after fixing recompilation (profile in repo)".
- **Rehearse** each project at 1, 5 and 20 minutes, per [interview-loops.md](../../tracks/research-engineer/interview-loops.md). Expect "what would you do with 1,000 times the compute?"

More project ideas that fit 8 GB are in [module 08](../../curriculum/08-evaluation-and-research.md#research-projects-that-fit-an-8-gb-gpu) and the [portfolio plan](../../tracks/research-engineer/portfolio.md).
