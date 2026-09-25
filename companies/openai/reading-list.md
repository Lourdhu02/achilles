# OpenAI reading list

Twenty-eight papers and reports that explain how OpenAI thinks about scaling, post-training, reasoning, evaluation and safety, with what to take from each. Five are marked **read first**: together they cover the questions an OpenAI interviewer is most likely to take three levels deep.

> [!NOTE]
> Links were checked in September 2026. All arXiv items are OpenAI-authored or co-authored. System cards for the newest models appear on openai.com first; check the [model release notes](https://help.openai.com/en/articles/9624314-model-release-notes) for anything newer than this list.

**Contents:** [How to read these](#how-to-read-these) · [Pretraining and scaling](#pretraining-and-scaling) · [Learning from human feedback](#learning-from-human-feedback) · [Code, verifiers and pass@k](#code-verifiers-and-passk) · [Reasoning models](#reasoning-models) · [Evaluations OpenAI built](#evaluations-openai-built) · [Alignment and CoT monitoring](#alignment-and-cot-monitoring) · [Interpretability](#interpretability) · [Open-weight models](#open-weight-models) · [A 4-week order](#a-4-week-order)

## How to read these

For each paper, write four lines in `journal/`: the claim, the key figure or equation, the number you would quote, and one experiment you could run on 8 GB to test it. [Module 00](../../curriculum/00-learning-os.md) describes the loop. If a "what to extract" line asks you to derive something, derive it on paper before you read the derivation.

| Mark | Meaning |
|---|---|
| **Read first** | Core to the interview; know it cold |
| (8 GB) | Has a small-scale experiment you can run; see [projects.md](projects.md) |

## Pretraining and scaling

| # | Item | Year | What to extract |
|---|---|---|---|
| 1 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2) | 2019 | WebText (outbound Reddit links with at least 3 karma) as a quality filter; byte-level BPE that blocks merges across character categories, the ancestor of the pre-tokenizer you build in [lab 04](../../labs/04_tokenizer/README.md); zero-shot task transfer as a function of model size. |
| 2 | **Read first.** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al.) | 2020 | Loss as power laws in non-embedding parameters $N$, data $D$ and compute $C$ ($\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$). The compute-optimal allocation they found ($N \propto C^{0.73}$) and why Chinchilla later found roughly $N \propto C^{0.5}$ (learning-rate schedules not matched to run length). Be able to fit one yourself: [lab 08](../../labs/08_scaling_laws/README.md). (8 GB) |
| 3 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3) | 2020 | In-context learning improving with scale; the 175B model trained on about 300B tokens; the data-contamination analysis (n-gram overlap between training data and benchmarks), which is still how many contamination checks work. |
| 4 | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023 | The "predictable scaling" section: final loss and a HumanEval metric predicted from much smaller runs before training. That is the pretraining team's core discipline. Also note what the report does not disclose, and why. |

## Learning from human feedback

| # | Item | Year | What to extract |
|---|---|---|---|
| 5 | [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | The Bradley–Terry reward model over pairs of trajectory segments, $P(a \succ b) = \sigma(r_a - r_b)$; asking humans about under 1% of the agent's interactions. Every RLHF system since descends from it. |
| 6 | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | 2017 | The clipped surrogate objective and when its gradient is exactly zero (advantage positive, ratio above $1+\epsilon$). You implement this in [lab 12](../../labs/12_grpo/README.md); GRPO is PPO without a value network. |
| 7 | [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) | 2020 | The full recipe at small scale: SFT, a reward model from comparisons, PPO with a KL penalty to the SFT policy. The figure where optimizing the reward model too hard makes true quality fall. |
| 8 | **Read first.** [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (InstructGPT) | 2022 | The three steps (SFT, RM, PPO); why a 1.3B InstructGPT was preferred over 175B GPT-3; the "alignment tax" and the PPO-ptx fix (mixing in pretraining gradients); how comparisons from one prompt are batched in the RM loss to avoid overfitting. Derive the RM loss $-\log \sigma(r(x,y_w) - r(x,y_l))$ and connect it to DPO in [lab 11](../../labs/11_dpo/README.md). |
| 9 | [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) | 2022 | A synthetic "gold" reward model lets you measure Goodhart's law. Gold score vs $d = \sqrt{\mathrm{KL}}$ follows $d(\alpha - \beta d)$ for best-of-$n$ and $d(\alpha - \beta \log d)$ for RL. Explains why KL budgets exist. (8 GB) |

## Code, verifiers and pass@k

| # | Item | Year | What to extract |
|---|---|---|---|
| 10 | **Read first.** [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Codex, HumanEval) | 2021 | The unbiased pass@k estimator $1 - \binom{n-c}{k} / \binom{n}{k}$ and why $1-(1-\hat p)^k$ is biased; HumanEval's 164 hand-written problems; Codex solved 28.8% with one sample and 70.2% with 100 samples per problem, the first clear evidence that sampling plus a verifier multiplies capability. Implemented in [lab 15](../../labs/15_eval_stats/README.md). (8 GB) |
| 11 | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (GSM8K) | 2021 | Generate many solutions, rank with a trained verifier: the paper reports a gain comparable to a 30× larger model. The seed of test-time compute. |

## Reasoning models

| # | Item | Year | What to extract |
|---|---|---|---|
| 12 | **Read first.** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | 2023 | Process supervision (a reward per reasoning step) beats outcome supervision for best-of-$N$ on MATH; the process reward model solves about 78% of a representative MATH test subset; PRM800K (800K step labels) is [public](https://github.com/openai/prm800k); active learning to pick convincing wrong answers. Know the trade-off: process labels cost more but give denser credit assignment. (8 GB) |
| 13 | [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) and the [OpenAI o1 System Card](https://arxiv.org/abs/2412.16720) | 2024 | Performance rising smoothly with both RL training compute and test-time thinking compute. From the system card: how a reasoning model is evaluated before launch (jailbreaks, hallucination, CoT deception monitoring, Preparedness categories). |
| 14 | [Competitive Programming with Large Reasoning Models](https://arxiv.org/abs/2502.06807) | 2025 | A specialized system with hand-engineered test-time strategies (o1-ioi) versus a general model (o3) that reached IOI gold-medal level without them. The lesson OpenAI draws: scale general RL rather than hand-built inference pipelines. |
| 15 | [OpenAI GPT-5 System Card](https://arxiv.org/abs/2601.03267) | 2025 | A unified system with a router between a fast model and a thinking model; "safe-completions" instead of hard refusals; how sycophancy, deception and hallucination are measured; precautionary High-capability treatment in biology. |

## Evaluations OpenAI built

| # | Item | Year | What to extract |
|---|---|---|---|
| 16 | [Measuring Short-Form Factuality in Large Language Models](https://arxiv.org/abs/2411.04368) (SimpleQA) | 2024 | 4,326 short questions collected adversarially against GPT-4, graded correct / incorrect / not attempted; calibration measured from stated confidence and answer frequency. Small enough to evaluate a local model with CIs. (8 GB) |
| 17 | [MLE-bench](https://arxiv.org/abs/2410.07095), [PaperBench](https://arxiv.org/abs/2504.01848), [SWE-Lancer](https://arxiv.org/abs/2502.12115) | 2024–25 | Three agentic benchmarks from the Preparedness side: 75 Kaggle competitions; replicating 20 ICML 2024 papers graded by author-agreed rubrics; 1,400+ freelance tasks worth $1M in payouts. Extract how each handles grading, contamination and cost. |
| 18 | [BrowseComp](https://arxiv.org/abs/2504.12516) | 2025 | 1,266 questions that are hard to find but easy to verify: the asymmetry that makes browsing agents measurable. Useful template for your own agent evals. |
| 19 | [HealthBench](https://arxiv.org/abs/2505.08775) | 2025 | 5,000 multi-turn conversations graded against physician-written rubrics; rubric-based grading with a model grader and the checks on its agreement with physicians (compare Cohen's kappa in lab 15). |
| 20 | [GDPval](https://arxiv.org/abs/2510.04374) | 2025 | Tasks from 44 occupations, graded by blinded expert comparison with human deliverables. Shows where evaluation is heading: economic value, not exam questions. |
| 21 | [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) | 2025 | Pretraining hallucination as a consequence of classification error; post-training evaluations that give zero credit for "I don't know" reward guessing. The fix: state a confidence threshold in the task and penalize wrong answers accordingly. A good interview topic that joins statistics and incentives. (8 GB) |

## Alignment and CoT monitoring

| # | Item | Year | What to extract |
|---|---|---|---|
| 22 | [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390) | 2023 | The superalignment analogy: a small model supervises a large one. Performance gap recovered, $\mathrm{PGR} = \frac{\text{weak-to-strong} - \text{weak}}{\text{strong ceiling} - \text{weak}}$; the auxiliary confidence loss; why reward modeling recovers less than NLP tasks. [Code](https://github.com/openai/weak-to-strong) (archived). (8 GB) |
| 23 | [Deliberative Alignment](https://arxiv.org/abs/2412.16339) and [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541) | 2024–25 | Training a reasoning model to recall and reason over the written safety spec in its chain of thought, without human-written CoTs. The follow-up (with Apollo Research) measures covert actions and shows large reductions that are partly confounded by the model recognizing it is being evaluated. |
| 24 | **Read first.** [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926) | 2025 | A weaker model reading a frontier reasoning model's chain of thought catches reward hacking in coding tasks far better than one watching actions alone. Penalizing "bad thoughts" during RL makes the agent keep hacking while hiding intent. The "monitorability tax" argument follows. Reproducible in miniature: [projects.md](projects.md) project 3. (8 GB) |
| 25 | [Chain of Thought Monitorability: A New and Fragile Opportunity](https://arxiv.org/abs/2507.11473) and [Monitoring Monitorability](https://arxiv.org/abs/2512.18311) | 2025 | The multi-lab position paper on why CoT is monitorable (hard tasks force externalized reasoning) and what would break it (outcome-only RL pressure, CoT supervision, latent-space reasoning architectures). The follow-up gives three evaluation archetypes and a metric; longer CoTs tend to be more monitorable. |
| 26 | [Persona Features Control Emergent Misalignment](https://arxiv.org/abs/2506.19823) | 2025 | Narrow fine-tuning on bad data (insecure code, bad advice) causes broad misalignment; model diffing with SAEs finds a "toxic persona" latent that predicts and steers it; a small amount of benign fine-tuning restores behavior. Connects interpretability to a training-time safety problem. |

## Interpretability

| # | Item | Year | What to extract |
|---|---|---|---|
| 27 | [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) and [Weight-Sparse Transformers Have Interpretable Circuits](https://arxiv.org/abs/2511.13653) | 2024–25 | TopK SAEs set the sparsity $L_0 = k$ directly (no L1 penalty, no shrinkage); clean scaling laws in latents and $k$; dead-latent fixes (encoder initialized as decoder transpose, an auxiliary loss); evaluation beyond MSE (downstream loss, probes, ablation sparsity). The 2025 paper trains models whose weights are mostly zero so circuits are small by construction. Extend the ReLU SAE from [lab 16](../../labs/16_interpretability/README.md) to TopK. (8 GB) |

## Open-weight models

| # | Item | Year | What to extract |
|---|---|---|---|
| 28 | [gpt-oss-120b and gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) | 2025 | Mixture-of-experts with about 5.1B (120b) and 3.6B (20b) active parameters; MXFP4 expert weights so 120b fits one 80 GB GPU and 20b about 16 GB; the [harmony](https://github.com/openai/harmony) response format; adjustable reasoning effort; the adversarial fine-tuning study used to judge open-weight release risk. Connect MXFP4 to [lab 13](../../labs/13_quantization/README.md). |

## A 4-week order

| Week | Read | Build alongside |
|---|---|---|
| 1 | 2, 10, 4 | Lab 08 fit; lab 15 pass@k with CIs |
| 2 | 5, 6, 8, 9 | Lab 11 and lab 12 core |
| 3 | 11, 12, 13, 14, 21 | A best-of-$N$ verifier on GSM8K with a 0.5B model |
| 4 | 22, 24, 25, 27 | Project 3 or 4 from [projects.md](projects.md) |

Then use items 15–20, 23, 26 and 28 to prepare for the specific team you apply to.
