# Anthropic reading list

Thirty primary sources that define how Anthropic thinks about models, safety and engineering, grouped by theme, each with what to extract.
Start with the five marked **read first**; they are the minimum for a credible conversation with any Anthropic team. Everything is free to read.

How to read each one: write down (1) the claim, (2) the evidence and its weakest point, (3) one experiment you could run on an 8 GB GPU to test or extend it. Item (3) is where [projects.md](projects.md) comes from.

## Contents
- [Read first](#read-first)
- [Mission and policy](#mission-and-policy)
- [Interpretability](#interpretability)
- [Alignment science](#alignment-science)
- [Safeguards and robustness](#safeguards-and-robustness)
- [Evaluation and engineering](#evaluation-and-engineering)
- [A six-week reading order](#a-six-week-reading-order)

## Read first

| # | Item | Why first |
|---|---|---|
| 1 | Core Views on AI Safety (2023) | The reasoning behind the whole company. Everything else is a consequence. |
| 2 | Toy Models of Superposition (2022) | The mental model behind modern interpretability, reproducible on a CPU. |
| 3 | On the Biology of a Large Language Model (2025) | Where interpretability is now: what attribution graphs show about real model computation. |
| 4 | Constitutional AI (2022) | How Anthropic replaced much human feedback with principles plus AI feedback. |
| 5 | Alignment Faking in Large Language Models (2024) | The kind of empirical safety result Anthropic is known for, and how carefully such claims are made. |

## Mission and policy

| Item | Year | What to extract |
|---|---|---|
| **[Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety)** (read first) | 2023 | The argument for building frontier models in order to study their safety. The optimistic, intermediate and pessimistic scenarios, and which research directions pay off in each. Decide where you disagree. |
| [Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy) (v3.4, effective July 2026) | 2023–2026 | Capability thresholds, the safeguards each triggers, how capability assessments and risk reports work, who reviews them. Read the redline of the latest version to see what changed and guess why. |
| [Claude's constitution](https://www.anthropic.com/constitution) ([announcement](https://www.anthropic.com/news/claudes-constitution)) | 2023, updated 2026 | How Anthropic wants Claude to weigh helpfulness, honesty and harm, and where it draws hard lines. Useful context for any post-training or safeguards role. |
| [System cards](https://www.anthropic.com/system-cards) (read the latest one) | ongoing | How a model is evaluated before release: capability evals, alignment assessments, RSP determinations, welfare and safeguards sections. The best single view of what the eval, alignment and red-team teams actually do. |

## Interpretability

| Item | Year | What to extract |
|---|---|---|
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al.) | 2021 | The residual stream as a shared communication channel; QK and OV circuits; composition between heads. You need this vocabulary for every later paper. |
| [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (Olsson et al.) | 2022 | How induction heads form in a phase change and why they are tied to in-context learning. You build one in [lab 16](../../labs/16_interpretability/README.md). |
| **[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)** (Elhage et al.; [arXiv](https://arxiv.org/abs/2209.10652)) (read first) | 2022 | Why sparse features get packed into fewer dimensions; the phase diagram of sparsity versus importance; geometry of feature arrangements. Reproduce the core figure yourself. |
| [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken et al.) | 2023 | Dictionary learning with sparse autoencoders on a one-layer model; how to judge whether a feature is interpretable; feature splitting as the dictionary grows. |
| [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Templeton et al.) | 2024 | SAEs scaled to a production model; abstract and multilingual features; steering by clamping features; the scaling-law view of dictionary size. |
| [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Ameisen et al.) | 2025 | Cross-layer transcoders as a replacement model; attribution graphs; how to validate a graph with interventions; the stated limitations. |
| **[On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)** (Lindsey et al.; [overview](https://www.anthropic.com/research/tracing-thoughts-language-model)) (read first) | 2025 | Case studies: multi-step reasoning, planning ahead in poetry, multilingual circuits, unfaithful reasoning. Note how each claim is checked by intervention. |
| [Open-sourcing circuit-tracing tools](https://www.anthropic.com/research/open-source-circuit-tracing) | 2025 | The public library and interactive frontend for attribution graphs on open models. Your entry point for doing this work yourself. |
| [Persona Vectors](https://arxiv.org/abs/2507.21509) (Chen et al.) | 2025 | Linear directions for character traits (for example sycophancy), how to extract them, and using them to monitor and control fine-tuning. Cheap to reproduce on small models. |

## Alignment science

| Item | Year | What to extract |
|---|---|---|
| [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) (Bai et al.) | 2022 | The helpfulness–harmlessness tension, preference-model scaling, online RLHF. Background for [lab 11](../../labs/11_dpo/README.md) and [module 06](../../curriculum/06-post-training.md). |
| **[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)** (Bai et al.) (read first) | 2022 | The two stages: supervised self-critique and revision, then RL from AI feedback against a preference model trained on AI labels. What the constitution is doing mechanically. |
| [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) (Sharma et al.) | 2023 | Evidence that human preference data rewards agreeable answers, and that optimizing against it increases sycophancy. A clean example of a reward signal that is subtly wrong. |
| [Sleeper Agents](https://arxiv.org/abs/2401.05566) (Hubinger et al.) | 2024 | Backdoored models whose behaviour survives SFT, RL and adversarial training; larger models and chain-of-thought make the backdoor more persistent. The idea of a "model organism" of misalignment. |
| **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)** (Greenblatt et al.) (read first) | 2024 | A model that behaves differently when it infers it is being trained; the experimental controls; the limitations section. Practise summarizing it in two minutes without overclaiming. |
| [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965) (Marks et al.) | 2025 | A blind auditing game: a model trained with a hidden objective, and teams using interpretability, behavioural attacks and data analysis to find it. Shows how alignment audits are structured. |
| [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) (Chen et al.) | 2025 | Measuring chain-of-thought faithfulness with inserted hints; why CoT monitoring is useful but not sufficient. A method you can replicate on small open reasoning models. |
| [Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment) | 2025 | Stress-testing models in simulated agentic settings with conflicting goals; how such red-teaming scenarios are designed, and how to read results that come from contrived setups. |
| [Natural Emergent Misalignment from Reward Hacking in Production RL](https://arxiv.org/abs/2511.18397) (MacDiarmid et al.) | 2025 | Learning to reward-hack in real coding environments generalizes to broader misalignment; which mitigations helped. Directly relevant to anyone working on RL environments. |

## Safeguards and robustness

| Item | Year | What to extract |
|---|---|---|
| [Many-shot Jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking) (Anil et al.) | 2024 | An attack that scales with context length and follows a power law in the number of shots; why long context creates new attack surface. |
| [Constitutional Classifiers](https://arxiv.org/abs/2501.18837) (Sharma et al.) | 2025 | Input and output classifiers trained on synthetic data generated from a constitution; the large red-teaming exercise; the trade-off between robustness, over-refusal and compute cost. |

## Evaluation and engineering

| Item | Year | What to extract |
|---|---|---|
| [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (Miller; [overview](https://www.anthropic.com/research/statistical-approach-to-model-evals)) | 2024 | Standard errors, clustered errors for related questions, paired comparisons, power analysis. The statistics behind [lab 15](../../labs/15_eval_stats/README.md); use them in every write-up. |
| [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | 2026 | How Anthropic engineers think about building evals for agents: tasks, graders, and what to trust. |
| [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise) | 2026 | How much benchmark scores move for reasons unrelated to the model. A model for how to report your own eval numbers. |
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | 2024 | Workflows versus agents; the named patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer); "start simple". Required for Applied AI and forward-deployed roles. |
| [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) | 2025 | Real inference bugs that degraded output quality, why they were hard to detect, and what changed. Essential for inference roles: precision, sampling and routing bugs are subtle. |
| [Designing AI-resistant technical evaluations](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations) | 2026 | How the performance team designs its take-home, and what it says a good evaluation looks like. Read it as a description of the candidate they want. |

> [!TIP]
> Anthropic publishes a lot. To see what is current, check three feeds monthly: [research](https://www.anthropic.com/research), [engineering](https://www.anthropic.com/engineering), and the interpretability team's site at transformer-circuits.pub (its monthly "circuits updates" posts are short and show work in progress).

## A six-week reading order

About 4–5 hours a week, alongside the labs.

| Week | Read | Do |
|---|---|---|
| 1 | Core Views; RSP; the latest system card (skim) | Write your values answers in [career/stories.md](../../career/stories.md#values-and-mission-prep) |
| 2 | Mathematical Framework; Induction Heads; Toy Models | [Lab 16](../../labs/16_interpretability/README.md), induction and SAE parts |
| 3 | Towards and Scaling Monosemanticity; Persona Vectors | Start [project 2](projects.md#2-sae-feature-study-on-a-small-open-model) or [project 1](projects.md#1-superposition-phase-diagram-cpu-only) |
| 4 | Circuit Tracing; Biology of an LLM | Explore attribution graphs on an open model |
| 5 | HH-RLHF; Constitutional AI; Sycophancy; Sleeper Agents | [Lab 11](../../labs/11_dpo/README.md), [lab 12](../../labs/12_grpo/README.md) |
| 6 | Alignment Faking; Auditing; CoT faithfulness; Reward hacking; Constitutional Classifiers; Error Bars | [Lab 15](../../labs/15_eval_stats/README.md); pick your project |

Engineering posts: read the ones that match your target team, in any week.
