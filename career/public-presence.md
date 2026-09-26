# Public presence: making your work legible

Without a well-known employer or university name, public work has to supply the signal: code people can run, write-ups with honest numbers, and contributions to projects core-AI teams use. This file is the plan for producing that work alongside the roadmap, and the standards that make it credible.
See also the [portfolio guide](../tracks/research-engineer/portfolio.md) for what to build; this page covers how to publish it.

**Contents:** [Twelve-month plan](#twelve-month-proof-of-work-plan) · [GitHub](#github) · [Writing](#writing) · [Open source](#open-source) · [Hugging Face](#hugging-face) · [LinkedIn and X](#linkedin-and-x) · [Kaggle](#kaggle) · [Don'ts](#donts)

## Twelve-month proof-of-work plan

Aligned with the [roadmap](../ROADMAP.md) stages. Each row is a minimum; do more only if the labs stay on schedule.

| when | stage | writing | open source | Hugging Face | other |
|---|---|---|---|---|---|
| Oct–Dec 2026 | 1–2 | set up the blog and GitHub profile README; **post 1** from lab 03, 04 or 06 | pick 2 projects you use; read their contributing guides; file one well-reproduced issue or doc fix | profile; publish the tokenizer study as a small dataset or Space | join 1–2 communities |
| Jan–Mar 2027 | 3–4 | **write-up 2** (scaling-law fit or GRPO) | **first merged PR** (docs, tests or a small fix) | publish the small model from the pretraining run with an honest model card | Gate A: record indicators |
| Apr–Jun 2027 | 5–7 | **write-up 3** (inference benchmark); research project proposal | second PR, ideally a bug fix or small feature | dataset card for the research project's data | one meetup talk or lightning talk |
| Jul–Sep 2027 | 8–9 | **research write-up**, workshop-ready | 2–3 merged PRs in total; start owning one small area | a collection linking every artifact | Gate B, then applications |

> [!TIP]
> Publish on a fixed day each month even when the result is small. A short, honest post about a negative result ("I expected X, measured Y, here's why") builds more credibility with researchers than a long post that hides the uncertainty.

## GitHub

- **Pin:** this repo with *your* implementations, 2–3 experiment repos, FinSentinelAI and ECHOME (after their eval upgrades).
- **Every pinned README:** what it is (one sentence), a result figure, one-command reproduction, hardware used, what you learned or what didn't work.
- **Profile README:** a "Now" section with the current lab and experiment, and links to the three best artifacts.

## Writing

One post per month. Posts that come straight out of this repo:
1. "The KV-cache number most blog posts get wrong" (GQA vs MHA napkin math, lab 03).
2. "Why GPT-4's tokenizer shatters Telugu, and what o200k fixed" (lab 04, with measurements).
3. "I wrote FlashAttention in Triton on an RTX 5060" (lab 06, with benchmarks).
4. "GRPO from scratch: the details the equations hide" (lab 12).
5. "Error bars on a public leaderboard" (lab 15).

Post structure:

```text
Title: the claim or question, specific
1. The question and why it matters (3–4 sentences)
2. Prediction before measuring (napkin math)
3. Setup: code link, hardware, versions, seeds
4. Result: one clear figure, with error bars or seed spread
5. What surprised me / what didn't work
6. Limitations and what I'd do with more compute
7. Reproduce: one command
```

Quality checklist: one claim per post; compute and uncertainty stated; every number reproducible from the linked code; no claim broader than the evidence. Host on your own domain (so links outlive platforms) and cross-post.

## Open source

**Where:** projects you already use in the labs, so your contributions come from real use. Examples: PyTorch, Triton, Hugging Face Transformers, TRL and PEFT, vLLM, llama.cpp, EleutherAI's lm-evaluation-harness, TransformerLens.

**The ladder:**
1. A well-reproduced issue: minimal script, versions, expected vs actual.
2. Docs or tests: fix what confused you while doing a lab.
3. A bug fix with a test.
4. A small feature or performance improvement with a benchmark.
5. Ownership of a small area: reviewing others' PRs there, answering issues.

**Etiquette:** read the contributing guide; comment on the issue before large work so maintainers can say whether they want it; keep PRs small and focused; include tests (and benchmarks for performance changes); respond to review quickly and without defensiveness. A few substantive PRs are worth far more than many typo fixes.

## Hugging Face

- **Models:** publish models from your experiments with a model card that states intended use, training data, evaluation with intervals, limitations and license.
- **Datasets:** publish evaluation sets or processed data you are allowed to share, with a dataset card (source, collection method, license, known issues). Never publish data you do not have rights to, including client data.
- **Spaces:** small demos that run on free CPU hardware where possible, so anyone can try them.
- **Collections:** group each project's model, dataset, Space and write-up in one place.

## LinkedIn and X

A headline that matches your target (for example, "ML engineer building LLM systems from first principles | GRPO, kernels, evals"). Share each write-up with its key figure and a two-line summary; comment substantively on researchers' and engineers' posts. Consistency over volume.

## Kaggle

Maintenance mode. Keep the Expert tier visible; invest only if a competition directly serves a research question.

## Don'ts

Claims your code doesn't support; results without the hardware and seeds; engagement bait; publishing anything derived from an employer's or client's data or code; letting a pinned repo rot (a broken install command undoes the signal).
