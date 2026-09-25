# Portfolio projects for Anthropic

Nine projects that map onto work Anthropic publishes, each sized for an 8 GB GPU or free Colab/Kaggle, and three that run on a CPU alone.
Each lists scope, which Anthropic work it connects to, which labs in this repo it builds on, what "done" means, and how to present it. Do one well rather than three halfway.

## Contents
- [How to choose](#how-to-choose)
- [What every project must have](#what-every-project-must-have)
- Projects [1](#1-superposition-phase-diagram-cpu-only) · [2](#2-sae-feature-study-on-a-small-open-model) · [3](#3-attribution-graphs-on-an-open-model) · [4](#4-a-sleeper-agent-model-organism-at-small-scale) · [5](#5-sycophancy-measure-it-find-its-direction-reduce-it) · [6](#6-reward-hacking-in-small-scale-grpo) · [7](#7-chain-of-thought-faithfulness-in-small-reasoning-models) · [8](#8-a-small-constitutional-classifier-including-indian-languages) · [9](#9-the-public-performance-take-home-cpu-only)
- [How to present](#how-to-present)

## How to choose

| Target team | Best projects | Second choice |
|---|---|---|
| Interpretability | 2, 3 | 1, 5 |
| Alignment science | 4, 6, 7 | 5 |
| RL / post-training | 6 | 4, 7 |
| Safeguards | 8 | 5, 4 |
| Inference / performance | 9 | lab 06 extension (see [module 07](../../curriculum/07-inference.md)) |
| Evaluation-heavy roles, Frontier Red Team | 7, 8 | any project, with rigorous stats |
| No GPU at all | 1, 9 | 8 (small encoder), 3 (hosted frontend) |

## What every project must have

These are what separate a portfolio project from a tutorial. Anthropic's own write-ups do all of them.

1. **A prediction written before the first run**, with a number. ("I expect the backdoor to survive LoRA safety fine-tuning at >80% trigger rate.")
2. **A baseline** that is embarrassingly simple, reported first.
3. **Error bars** on every headline number: bootstrap or binomial CIs, paired tests when comparing models, several seeds ([lab 15](../../labs/15_eval_stats/README.md), [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)).
4. **A causal check** where you claim a mechanism: ablate, patch, or steer, and show the behaviour changes.
5. **A limitations section** that names the most likely way you are wrong.
6. **Reproducible code**: one command per figure, pinned versions, seeds.

---

## 1. Superposition phase diagram (CPU-only)

**Scope.** Reimplement the toy model from [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html): $n$ sparse features, $m < n$ hidden dimensions, $h = Wx$, $\hat x = \mathrm{ReLU}(W^\top h + b)$, loss $\sum_i I_i (x_i - \hat x_i)^2$ with importance $I_i$. Reproduce (a) the $n=2, m=1$ phase diagram over sparsity $S$ and relative importance, and (b) "dimensions per feature" as sparsity rises for $n=20$, $m=5$, $I_i = 0.9^i$. Then extend: what happens with correlated features, or with a small amount of weight decay?

**Cost.** Each model has at most a few hundred parameters. A full sweep (say 40 × 40 grid, 3 seeds, a few thousand Adam steps each) runs in minutes to an hour on a laptop CPU.

**Maps to.** The interpretability team's core theory; the reason SAEs exist.

**Builds on.** [Lab 16](../../labs/16_interpretability/README.md) (the toy-superposition SAE), [lab 01](../../labs/01_autograd/README.md) (write the model in your own autograd for extra credit), [module 09 §1](../../curriculum/09-interpretability-and-safety.md#1-mechanistic-interpretability).

**Success.** Your phase diagram shows the same three regions as the paper (feature ignored, feature represented alone, superposition). Your extension has one clearly stated finding with seeds.

**Present.** Side-by-side figure: the paper's diagram and yours. One paragraph on where they differ and why.

## 2. SAE feature study on a small open model

**Scope.** Train sparse autoencoders on the residual stream of one layer of a small open model (Pythia-70M or GPT-2 small). Sweep dictionary size (for example 4x, 8x, 16x, 32x the model width) and sparsity penalty. Report the reconstruction-versus-L0 frontier, the fraction of dead latents, and **downstream loss** when you splice the SAE reconstruction back into the model. Hand-label 30 features at two dictionary sizes and document feature splitting.

**Worked numbers.** GPT-2 small has width 768. A 16x SAE has $768 \times 12{,}288 \approx 9.4$M weights in each of encoder and decoder, about 19M in total: trivial for 8 GB. The expensive part is activations: 10M tokens × 768 × 2 bytes (bf16) ≈ 15 GB, so **stream activations from the model in batches** instead of caching them to disk, or cache fewer tokens.

**Maps to.** [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

**Builds on.** [Lab 16](../../labs/16_interpretability/README.md) (your SAE), [lab 05](../../labs/05_transformer/README.md) (hooks into a transformer), [lab 15](../../labs/15_eval_stats/README.md).

**Success.** A frontier plot with at least four dictionary sizes; downstream loss reported, not just MSE; 30 labelled features with top-activating examples; one causal steering demo (clamp a feature, show the output change).

**Present.** A short post with an interactive or static feature browser (top examples per feature). Compare your dead-latent fraction to what the papers report and explain the difference.

**India angle.** Run the same SAE on Indian-language text (Hindi, Telugu, Tamil). Do the features split by script, or do multilingual features appear as they did in larger models?

## 3. Attribution graphs on an open model

**Scope.** Use Anthropic's open-sourced [circuit-tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing), which support open models such as Gemma-2-2B and Llama-3.2-1B, to study one behaviour in depth: two-hop factual recall, simple arithmetic, or a multilingual prompt. Form a hypothesis from the graph, then **test it with interventions** (suppress or inject features) and report how often the prediction holds across 20+ prompt variants.

**Hardware.** A 2B model in bf16 is about 5 GB of weights before transcoders and attribution memory, so use a free Colab or Kaggle GPU rather than an 8 GB laptop card, or start in the hosted Neuronpedia interface linked from the announcement (no GPU needed). Check the library's README for current memory needs.

**Maps to.** [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). The tools were built by Anthropic Fellows, so this is also the most direct preparation for the [Fellows Program](README.md#signals-that-get-you-noticed).

**Builds on.** [Lab 16](../../labs/16_interpretability/README.md) (activation patching), [module 09](../../curriculum/09-interpretability-and-safety.md).

**Success.** One mechanism claim, validated by interventions on held-out prompts, with a success rate and a clear account of where the graph misleads you.

**Present.** Share graph links, a single diagram of the circuit, and an intervention table. A merged fix or feature in the library itself is a strong bonus signal.

## 4. A sleeper-agent model organism at small scale

**Scope.** Following [Sleeper Agents](https://arxiv.org/abs/2401.05566), fine-tune a 0.5B–1.5B instruct model (LoRA) to behave normally except when a trigger appears (for example, write a harmless-looking but wrong answer, or insert a marker string in code). Then apply "safety training" (SFT on clean data, then DPO) and measure whether the backdoor survives. Finally, train a **linear probe** on activations to detect when the triggered behaviour is about to happen.

**Hardware.** LoRA on a 0.5B model in bf16 fits comfortably in 8 GB; 1.5B fits with LoRA, short sequences and gradient checkpointing.

**Maps to.** Alignment science: model organisms of misalignment and detecting them.

**Builds on.** [Lab 10](../../labs/10_lora/README.md), [lab 11](../../labs/11_dpo/README.md), [lab 16](../../labs/16_interpretability/README.md) (probing), [lab 15](../../labs/15_eval_stats/README.md).

**Success.** Trigger rate before and after each safety-training stage, with CIs over at least 3 seeds; probe AUROC on held-out prompts; a control showing the probe is not just detecting the trigger token.

**Present.** One figure: backdoor rate across training stages. One table: probe AUROC versus baselines. A careful limitations section: a small model organism is not evidence about frontier models, and say so.

> [!WARNING]
> Keep the backdoored behaviour harmless and do not publish backdoored weights under a name that invites use. The point is the method, not the payload.

## 5. Sycophancy: measure it, find its direction, reduce it

**Scope.** Build a sycophancy eval in the style of [Towards Understanding Sycophancy](https://arxiv.org/abs/2310.13548): the same factual question asked neutrally and with a user's stated (wrong) opinion. Measure the flip rate across three small open models. Extract a sycophancy direction by contrasting activations (as in [Persona Vectors](https://arxiv.org/abs/2507.21509)), show that steering along it changes the flip rate, then try to reduce sycophancy with DPO on preference pairs that reward disagreement when the user is wrong.

**Hardware.** 8 GB GPU, models up to about 1.5B.

**Maps to.** Alignment and post-training: reward signals that are subtly wrong, and representation-level monitoring.

**Builds on.** [Lab 11](../../labs/11_dpo/README.md), [lab 15](../../labs/15_eval_stats/README.md) (paired tests on flip rates), [lab 16](../../labs/16_interpretability/README.md).

**Success.** Flip rates with CIs per model; a dose–response curve for steering strength; DPO reduces flip rate without a significant drop on a general benchmark (check this with a paired test).

**Present.** The dose–response plot is the headline. Report capability cost next to safety gain.

## 6. Reward hacking in small-scale GRPO

**Scope.** Train a small model with GRPO on a verifiable task (arithmetic or short coding problems) where the verifier has a **deliberate loophole** (for example, tests that only check the output type, or a grader that accepts any answer containing the right number). Measure when hacking emerges, what it looks like, and which mitigations work: fixing the verifier, a KL penalty, a separate monitor, or telling the model the loophole is acceptable in training. Check whether hacking in one environment changes behaviour on a separate, unhackable eval.

**Hardware.** 0.5B model with LoRA and small group sizes on 8 GB, as in [lab 12](../../labs/12_grpo/README.md).

**Maps to.** RL environments and alignment: [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397). Anyone building RL environments deals with this daily.

**Builds on.** [Lab 12](../../labs/12_grpo/README.md), [module 06 §5](../../curriculum/06-post-training.md#5-rl-with-verifiable-rewards-grpo-and-friends), [lab 15](../../labs/15_eval_stats/README.md).

**Success.** A plot of true reward versus proxy reward over training showing where they diverge; a mitigation comparison with seeds; an honest answer to "does it generalize?" even if the answer is "not at this scale".

**Present.** Show sampled transcripts at three points in training. Reviewers remember concrete examples of a model gaming its grader.

## 7. Chain-of-thought faithfulness in small reasoning models

**Scope.** Replicate the hint methodology of [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) on small open reasoning models (1–2B distilled reasoning models fit on 8 GB with 4-bit weights). Insert a hint (a metadata tag, a "a professor says the answer is B" line), find the cases where the hint changes the answer, and measure how often the chain of thought mentions using it. Compare hint types and model sizes.

**Hardware.** 8 GB GPU with quantized weights, or free Colab. Generation-heavy, not training-heavy; budget the token count first with [lab 03](../../labs/03_napkin_math/README.md).

**Maps to.** Alignment science and evals: CoT monitoring and its limits.

**Builds on.** [Lab 07](../../labs/07_kv_cache_sampling/README.md), [lab 13](../../labs/13_quantization/README.md), [lab 15](../../labs/15_eval_stats/README.md).

**Success.** Faithfulness rate per hint type with CIs; a check that quantization does not change the conclusion (run a subset at bf16); a clear definition of "mentions the hint", with inter-rater agreement if you label by hand.

**Present.** A table in the same format as the paper, so a reader can compare directly, and a note on what differs at small scale.

## 8. A small constitutional classifier, including Indian languages

**Scope.** Write a short constitution for one narrow harm category that is safe to work on (for example, requests for step-by-step instructions to bypass a specific software licence check). Generate synthetic allowed and disallowed prompts from it with an open model, including Hindi, Telugu and code-mixed versions. Train a small input classifier (a fine-tuned multilingual encoder, or a linear probe on a small LM). Red-team it yourself, measure attack success rate and **over-refusal** on benign look-alike prompts, and study how performance changes across languages.

**Hardware.** A small encoder fine-tunes on CPU in hours; a GPU makes it minutes. Data generation is the costly step; use a free Colab GPU for it.

**Maps to.** Safeguards: [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) and the Indian-language evaluation work named in Anthropic's [Bengaluru announcement](https://www.anthropic.com/news/bengaluru-office-partnerships-across-india).

**Builds on.** [Lab 04](../../labs/04_tokenizer/README.md) (tokenization of Indic scripts), [lab 15](../../labs/15_eval_stats/README.md), [lab 17](../../labs/17_retrieval/README.md) (near-duplicate detection in synthetic data).

**Success.** A ROC curve and an operating point chosen for a stated over-refusal budget; per-language results with CIs; a list of the attacks that worked and why.

**Present.** Lead with the trade-off plot (catch rate versus over-refusal). The multilingual gap, if you find one, is the story.

## 9. The public performance take-home (CPU-only)

**Scope.** Attempt the performance-engineering take-home Anthropic released as an open challenge (`github.com/anthropics/original_performance_takehome`, described in [Designing AI-resistant technical evaluations](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations)). It is a Python simulator of a TPU-like accelerator; you optimize a parallel tree-traversal kernel using manual memory management, VLIW packing, SIMD and multiple cores. First do a strict 2-hour attempt, then an unlimited one.

**Hardware.** CPU only.

**Maps to.** Performance engineering and inference; also shows you can reason about a machine model you have never seen, which is the skill the post says it tests.

**Builds on.** [Lab 03](../../labs/03_napkin_math/README.md) (roofline thinking), [lab 06](../../labs/06_attention_kernels/README.md) (tiling, memory hierarchy), [module 02](../../curriculum/02-compute-and-hardware.md).

**Success.** A cycle count, a log of each optimization and its measured gain, and a lower-bound estimate of the best achievable cycles from the machine's resources (so you know how far you are from optimal).

**Present.** A write-up structured as "bottleneck → change → cycles saved", with the roofline-style bound. The post invites people who beat its benchmark to contact recruiting; read it for the current threshold before you claim anything.

---

## How to present

- **Repository.** README with the headline figure first, one command per figure, results table with CIs, and a limitations section. Link it from the top of your résumé.
- **Write-up.** 1,500–2,500 words. Claim → evidence → what would change your mind. Put the simplest baseline in the first table.
- **Talk track.** Prepare the project at three depths: 1 minute (claim and headline number), 5 minutes (method and the surprise), 20 minutes (every design decision and what you would do next). This is the research deep-dive round.
- **Honesty.** State what you did with tools, what you did not verify, and where your result could be an artifact. Anthropic's candidate guidance asks for your real experience; a small honest result beats a large inflated one.
- **Distribution.** Share where interpretability and alignment researchers read (the Alignment Forum or LessWrong, EleutherAI and GPU MODE communities, X). See [career/public-presence.md](../../career/public-presence.md).
