# Portfolio projects for OpenAI

Nine projects that mirror OpenAI's published research agenda (RL on reasoning, process supervision, CoT monitoring, evaluation with honest statistics, weak-to-strong supervision, SAEs and efficient small models), each sized for an 8 GB GPU or free Colab/Kaggle, and one that is CPU-only. Each lists scope, which OpenAI work it maps to, the labs it builds on, what "done" means, and how to present it.

> [!IMPORTANT]
> One finished project with error bars beats three half-done ones. Pick project 3 or 1 as your flagship, then one smaller one. Write your prediction in `journal/` before every run.

**Contents:** [Summary table](#summary-table) · [1. pass@k done right](#1-passk-done-right) · [2. Process vs outcome reward models](#2-process-vs-outcome-reward-models-at-small-scale) · [3. Reward hacking and CoT monitoring](#3-reward-hacking-and-cot-monitoring-in-miniature) · [4. Weak-to-strong](#4-weak-to-strong-generalization-replication-plus-one-new-condition) · [5. TopK SAE scaling](#5-topk-sae-scaling-laws-on-a-small-model) · [6. Abstention-aware scoring](#6-abstention-aware-factuality-scoring) · [7. Parameter golf at home](#7-parameter-golf-at-home) · [8. Error bars on SWE-bench Verified](#8-error-bars-on-swe-bench-verified-cpu-only) · [9. Test-time compute curves](#9-test-time-compute-curves-on-a-small-reasoning-model) · [How to present](#how-to-present-any-of-these)

## Summary table

| # | Project | Hardware | Time | Best for |
|---|---|---|---|---|
| 1 | pass@k done right | 8 GB / Colab T4 | 2 weeks | Evals, Codex, post-training |
| 2 | Process vs outcome reward models | 8 GB (LoRA) | 3–4 weeks | Reasoning, post-training |
| 3 | Reward hacking and CoT monitoring | 8 GB (LoRA), tight | 4 weeks | Safety research, reasoning (**flagship**) |
| 4 | Weak-to-strong generalization | 8 GB / Colab T4 | 2–3 weeks | Alignment research |
| 5 | TopK SAE scaling laws | 8 GB | 3 weeks | Interpretability |
| 6 | Abstention-aware factuality scoring | 8 GB; CPU-possible for ≤1.5B models | 2 weeks | Evals, model behavior |
| 7 | Parameter golf at home | 8 GB | 3–4 weeks | Pretraining, efficiency |
| 8 | Error bars on SWE-bench Verified | **CPU-only** | 1–2 weeks | Evals, preparedness, agents |
| 9 | Test-time compute curves | 8 GB (+32 GB RAM for gpt-oss-20b) | 2–3 weeks | Inference, reasoning |

> [!WARNING]
> Projects 1, 2, 3 and 8 involve running model-generated code or grading it. Execute it in a sandbox (a container with no network and a time limit), never in your main environment.

---

## 1. pass@k done right

**Scope.** Measure a small code model on HumanEval ([dataset](https://huggingface.co/datasets/openai/openai_humaneval), [harness](https://github.com/openai/human-eval)) the way the Codex paper intended, then show what goes wrong when people do it casually. Model: Qwen2.5-Coder-0.5B or 1.5B Instruct. Draw $n = 100$–$200$ samples per problem at several temperatures.

- Compare the unbiased estimator $\widehat{\text{pass@}k} = 1 - \binom{n-c}{k}/\binom{n}{k}$ with the naive $1 - (1 - c/n)^k$ across $k$; plot the bias.
- Put CIs on pass@1 and pass@10 with a problem-level bootstrap (164 problems is a small sample; the CI is wide).
- Plot pass@k vs temperature: low temperature wins at $k=1$, higher temperature wins at large $k$. Find the crossover.
- Power: how many problems would you need to detect a 3-point difference at 80% power?

**Maps to.** [Codex paper](https://arxiv.org/abs/2107.03374); every pass@k number in OpenAI system cards.
**Labs.** [15 eval stats](../../labs/15_eval_stats/README.md) (`pass_at_k`, bootstrap, `required_n`), [07 sampling](../../labs/07_kv_cache_sampling/README.md).
**Success criteria.** Your pass@1 is within the CI of the model's published number, or you explain the gap (prompt format, stop sequences, sanitization). Bias plot plus a temperature-by-$k$ heat map. One-command reproduction.
**Presentation.** Post title in the form "How many samples do you need for pass@k?". Lead figure: the temperature-by-$k$ heat map.

## 2. Process vs outcome reward models at small scale

**Scope.** A small replication of [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). Train two verifiers on [PRM800K](https://github.com/openai/prm800k) with LoRA on a 0.5B–1.5B base: a process reward model (score per step, solution score = product or minimum of step scores) and an outcome reward model (final-answer correctness only, same solutions). Use them to rerank $N \in \{1, 4, 16, 64\}$ samples from a small math model on [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

**Maps to.** Process supervision, the verifier line from [GSM8K](https://arxiv.org/abs/2110.14168), reasoning-model reward design.
**Labs.** [10 LoRA](../../labs/10_lora/README.md), [15](../../labs/15_eval_stats/README.md); [module 06 §6](../../curriculum/06-post-training.md#6-reasoning-and-test-time-compute).
**Success criteria.** Best-of-$N$ accuracy curves for PRM, ORM and majority voting with paired CIs. State whether the paper's ordering holds at your scale, and at which $N$ it appears or breaks. An ablation on product vs minimum step aggregation.
**Presentation.** One figure: accuracy vs $N$ (log axis), three lines with bands. Note honestly that the PRM800K solutions come from a much stronger generator than yours: a distribution shift that you should measure, not hide.

## 3. Reward hacking and CoT monitoring in miniature

**Scope.** Reproduce the core phenomenon of [Baker et al. 2025](https://arxiv.org/abs/2503.11926) on one GPU.

1. Build a verifiable task with a planted loophole. Example: small Python functions graded by unit tests, where the visible tests can be passed by special-casing their inputs, and hidden tests measure real correctness.
2. Train Qwen2.5-0.5B-Instruct (or 1.5B with LoRA) with GRPO from [lab 12](../../labs/12_grpo/README.md), rewarding visible-test passes only. Prompt it to reason before answering.
3. Track the **hack rate** (visible pass, hidden fail) over training.
4. Build two monitors with a separate small instruct model: one reads the chain of thought and the code, one reads the code only. Label 200 episodes by hand and report each monitor's precision and recall with CIs.
5. Add the CoT monitor's flag as a reward penalty and retrain. Does hacking fall, or does the detected rate fall while the hidden-test hack rate stays high (obfuscation)?

**Maps to.** CoT monitoring, [monitorability](https://arxiv.org/abs/2507.11473), reward hacking in RL for reasoning. This is the most direct small-scale match to OpenAI's safety-research agenda.
**Labs.** [12 GRPO](../../labs/12_grpo/README.md) (S5 stretch), [15](../../labs/15_eval_stats/README.md) (precision/recall CIs, Cohen's kappa against your labels); [module 09](../../curriculum/09-interpretability-and-safety.md).
**Success criteria.** Hack rate vs step plot; a monitor comparison table with CIs; a clear yes, no or inconclusive on obfuscation, with 3 or more seeds. Qualitative examples of hacked CoTs in an appendix.
**Presentation.** Lead with the 2×2 finding: {CoT monitor, action monitor} × {with penalty, without}. Close with limits: a 0.5B model's CoT is not a frontier model's, so frame it as a testbed others can reuse, and release it.

> [!TIP]
> On 8 GB: keep completions under 512 tokens, group size 8, LoRA rank 16, gradient checkpointing, and run monitors offline on saved episodes rather than inside the training loop.

## 4. Weak-to-strong generalization: replication plus one new condition

**Scope.** Use OpenAI's archived [weak-to-strong](https://github.com/openai/weak-to-strong) code with small models: GPT-2 (124M) as the weak supervisor and GPT-2-medium or GPT-2-large as the strong student (the repo's model configs cover the GPT-2 family; adding a newer small model is a useful extension), on two or three of the repo's binary classification datasets. Reproduce performance gap recovered (PGR) with and without the auxiliary confidence loss, across 5 seeds. Then add one new condition, for example weak labels with structured (not random) errors, or a weak supervisor from a different model family.

**Maps to.** [Weak-to-strong generalization](https://arxiv.org/abs/2312.09390), superalignment-style research.
**Labs.** [02 training core](../../labs/02_training_core/README.md), [15](../../labs/15_eval_stats/README.md).
**Success criteria.** A PGR table with seed-level CIs; the new condition's effect with a paired test; a short section on why PGR can be unstable when the weak–strong gap is small (the denominator).
**Presentation.** "Weak-to-strong at 124M → 500M: what replicates and what doesn't."

## 5. TopK SAE scaling laws on a small model

**Scope.** Train TopK sparse autoencoders ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)) on the residual stream of GPT-2 small (layer 6) or your lab 05 model. Sweep latents $n \in \{2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}\}$ and $k \in \{16, 32, 64\}$. Fit normalized MSE as a function of $n$ and $k$. Measure dead latents with and without the auxiliary loss and the transpose initialization. Report downstream loss when the SAE reconstruction is spliced back into the model.

**Napkin math.** 10M tokens × 768 dims × 2 bytes (fp16) ≈ 15 GB of cached activations. Stream from disk, or regenerate activations on the fly in chunks.
**Maps to.** SAE scaling; [persona features](https://arxiv.org/abs/2506.19823) uses the same tool for model diffing.
**Labs.** [16 interpretability](../../labs/16_interpretability/README.md) (ReLU SAE → TopK), [08 scaling laws](../../labs/08_scaling_laws/README.md) (fitting).
**Success criteria.** A fitted scaling law with held-out points predicted within a stated error; a dead-latent comparison; 10 hand-inspected features with max-activating examples.
**Presentation.** Release the SAEs on the Hugging Face Hub with a model card and a notebook.

## 6. Abstention-aware factuality scoring

**Scope.** Test the incentive argument of [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) on [SimpleQA](https://arxiv.org/abs/2411.04368) (data via [simple-evals](https://github.com/openai/simple-evals)). Evaluate 3–5 small open models on 1,000 questions. Prompt with an explicit threshold $t \in \{0, 0.5, 0.75, 0.9\}$ ("answer only if more than $t$ confident; wrong answers cost $t/(1-t)$ points, abstaining costs 0"). Score each model under each rule.

**Maps to.** Hallucination, calibration, model behavior; how eval design shapes post-training incentives.
**Labs.** [15](../../labs/15_eval_stats/README.md) (paired tests, kappa for your grader), [07 sampling](../../labs/07_kv_cache_sampling/README.md).
**Success criteria.** Does the model ranking change with $t$? Paired-bootstrap CIs on every rank difference. Grader agreement with 200 hand labels (report kappa). Does stating $t$ in the prompt actually change abstention rates?
**Presentation.** A single table: models × thresholds, with the ranking flips highlighted. CPU-only is possible for 0.5B–1.5B models with llama.cpp; expect hours, not minutes.

## 7. Parameter golf at home

**Scope.** OpenAI's [Parameter Golf](https://github.com/openai/parameter-golf) challenge (March–April 2026) asked for the lowest bits per byte on FineWeb validation from a model whose artifact fits in 16,000,000 bytes and trains in 10 minutes on 8×H100. Run a scaled-down version on your GPU: same byte cap, a fixed wall-clock budget you choose (for example 60 minutes on the RTX 5060), a fixed FineWeb sample, and bits per byte on held-out text:

$$\text{bpb} = \frac{\sum_i -\ln p(x_i)}{\ln 2 \cdot \text{(number of UTF-8 bytes)}}$$

Ablate one idea at a time: vocabulary size (bytes vs BPE 1k/4k/8k: a bigger vocab costs embedding bytes), weight tying, depth recurrence (reuse layers), quantization-aware training to int8 or int4 (16M bytes holds about 16M int8 or 32M int4 parameters, minus everything else in the artifact), and Muon vs AdamW.

**Maps to.** Pretraining efficiency and scaling judgment; OpenAI framed the challenge as a way to spot research talent.
**Labs.** [02](../../labs/02_training_core/README.md) (Muon, schedules), [04](../../labs/04_tokenizer/README.md), [05](../../labs/05_transformer/README.md), [08](../../labs/08_scaling_laws/README.md), [13](../../labs/13_quantization/README.md).
**Success criteria.** An ablation table with 3 seeds per row and bpb differences larger than seed noise; an honest note that small-budget wins may not transfer to 8×H100 for 10 minutes. Stretch: port your best idea onto the public repo's baseline and report its effect there.
**Presentation.** "What matters under a 16 MB cap, at 1/100th the compute." Read the repository's rules and existing submissions first so you do not rediscover a known trick.

## 8. Error bars on SWE-bench Verified (CPU-only)

**Scope.** [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) is a 500-task human-validated subset that OpenAI's Preparedness team built with the SWE-bench authors ([dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)). Per-instance results for many public submissions are in [SWE-bench/experiments](https://github.com/SWE-bench/experiments). Using only those logs:

- Put 95% CIs on each submission's resolve rate, clustered by repository (tasks from the same repository are correlated).
- For adjacent leaderboard pairs, run paired tests (McNemar, paired bootstrap). How many "rank changes" are real?
- Fit per-task difficulty (Bradley–Terry or a simple item-response model) and find tasks that no system solves or that every system solves.
- Check whether resolve rates differ by issue creation date, a cheap contamination signal.

**Maps to.** Preparedness and frontier evals; how OpenAI reports agentic coding results.
**Labs.** [15](../../labs/15_eval_stats/README.md) (clustered SEs, McNemar, Holm–Bonferroni, Bradley–Terry).
**Success criteria.** A reproducible notebook, a figure of the leaderboard with clustered CIs, and a list of statistically indistinguishable groups.
**Presentation.** Short and factual; do not overclaim. Send it to the maintainers before posting.

## 9. Test-time compute curves on a small reasoning model

**Scope.** Measure accuracy vs thinking tokens and wall-clock cost. Options: [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) at low, medium and high reasoning effort (about 16 GB of weights: on 8 GB VRAM, keep attention on the GPU and offload experts to system RAM with llama.cpp; it is slow but works with 32 GB RAM), or a 1.5B–2B open reasoning model with a thinking budget. Evaluate on 200–500 GSM8K or MATH-500 problems. Compare longer thinking, majority voting over short samples, and best-of-$N$ with a verifier at equal token budgets.

**Maps to.** [Learning to reason](https://openai.com/index/learning-to-reason-with-llms/), [gpt-oss model card](https://arxiv.org/abs/2508.10925), inference cost.
**Labs.** [03 napkin math](../../labs/03_napkin_math/README.md) (predict tokens/s before measuring), [07](../../labs/07_kv_cache_sampling/README.md), [13](../../labs/13_quantization/README.md) (MXFP4), [14 speculative decoding](../../labs/14_speculative_decoding/README.md) (stretch: a draft model for the thinking phase).
**Success criteria.** Accuracy vs tokens per problem on a log axis for each strategy, with CIs; measured vs predicted decode throughput within 2×; a recommendation for which strategy wins at which budget.
**Presentation.** The equal-compute comparison is the finding; lead with it.

---

## How to present any of these

1. **Repository:** `README` with the question, the one-line answer, the key figure, and `make reproduce`. Pin seeds and package versions.
2. **Write-up (1,500–2,500 words):** question → prediction (written before the run) → setup → result with CIs → what surprised you → limitations → what you would do with 100× the compute.
3. **Figures:** one idea per figure, labeled axes, error bands, captions that state the takeaway.
4. **Distribution:** post it, then send it to one or two OpenAI researchers whose paper you built on, in under 120 words, asking one specific question ([career/outreach.md](../../career/outreach.md)).
5. **Interview readiness:** rehearse the project at 1, 5 and 20 minutes ([interview-loops.md](../../tracks/research-engineer/interview-loops.md)); expect "what would you try next?" and "which number are you least sure of?"

See [portfolio.md](../../tracks/research-engineer/portfolio.md) for how these fit the wider portfolio, and [README.md](README.md#signals-that-get-you-noticed-ranked) for how OpenAI weighs them.
