# Portfolio: proof of core-AI ability

Recruiters and interviewers believe artifacts, not adjectives. This page sets the portfolio to have by the time you apply, the standard every artifact must meet, templates for write-ups and model cards, a way into open-source projects, and a flagship project that fits an 8 GB GPU.

## Contents

- [The target](#the-target)
- [Artifact standards](#artifact-standards)
- [Experiment write-up template](#experiment-write-up-template)
- [Repo hygiene checklist](#repo-hygiene-checklist)
- [Model card template](#model-card-template)
- [Open-source contributions](#open-source-contributions)
- [Flagship: Telugu-first small LM](#flagship-telugu-first-small-lm)
- [Distribution](#distribution)
- [Strong vs weak portfolios and bullets](#strong-vs-weak-portfolios-and-bullets)

## The target

By the time you apply:

1. **This repo, public, with your implementations** of labs 01–17 passing (`python tools/progress.py`) and a short README of what you learned. It shows you built the stack from scratch. Make your own work easy to find: it lives in your `exercise.py` files and your commits, not in the reference `solution.py`. Paste the progress table into the README and add the three hardest bugs you hit and how you found them; bug stories make good deep-dive material.
2. **Three experiment write-ups** with predictions, CIs and ablations from the [research list](../../curriculum/08-evaluation-and-research.md#research-projects-that-fit-an-8-gb-gpu), published as blog posts. Each should contain one surprising finding. Choose them to cover your target team and one neighbour:

   | target team | project from the research list |
   |---|---|
   | Pretraining | [1. tokenizer fairness for Telugu](../../curriculum/08-evaluation-and-research.md#1-tokenizer-fairness-for-telugu) or [2. byte-level vs BPE scaling](../../curriculum/08-evaluation-and-research.md#2-scaling-laws-for-byte-level-vs-bpe-models) |
   | Post-training | [3. GRPO ablations](../../curriculum/08-evaluation-and-research.md#3-grpo-ablations-on-small-models) |
   | Inference | [4. KV-cache quantization](../../curriculum/08-evaluation-and-research.md#4-kv-cache-quantization-quality-vs-speed-on-consumer-blackwell) or [5. speculative decoding by domain](../../curriculum/08-evaluation-and-research.md#5-speculative-decoding-acceptance-vs-domain) |
   | Interpretability | [6. SAE features](../../curriculum/08-evaluation-and-research.md#6-sae-features-of-a-tinystories-model) |
   | Evals, and useful to every team | [7. error bars on a leaderboard](../../curriculum/08-evaluation-and-research.md#7-error-bars-on-a-public-leaderboard) |

3. **One paper reproduction** with an honest note on the discrepancies. Pick a paper whose central claim is a trend you can test at small scale: the reward–KL trade-off of DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) on a 0.5B model; the IsoFLOP curves of Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) at 1M–30M parameters; the speedup model of speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) with your lab 14 sampler. The note says what matched (for example, the direction of an effect), what did not (for example, its size), which candidate causes you tested, and what closing the gap would take. See [reproducing papers](../../curriculum/08-evaluation-and-research.md#5-reproducing-papers).
4. **Two or three merged OSS PRs** in core projects: vLLM, SGLang, TRL, transformers, llama.cpp, lm-evaluation-harness, torchtitan or torchao. Start with issues labeled "good first issue" or with a bug you hit during the labs; small, correct fixes count. How to find and land them is [below](#open-source-contributions).
5. **One flagship that ties it together.** Recommended, because it plays to a real advantage: **"Telugu-first small LM"**. A tokenizer study → pretrain a small bilingual model on your GPU → SFT and GRPO on a Telugu extraction task → evals with CIs → a public model card on the Hugging Face Hub. Milestones and a compute budget are [below](#flagship-telugu-first-small-lm).
6. **Upgraded existing projects** (FinSentinelAI, ECHOME) with measured evals ([module 10 §6](../../curriculum/10-applied-llm-systems.md#6-upgrading-your-own-projects)). For FinSentinelAI: a held-out question set with gold answers and source passages, retrieval metrics from [lab 17](../../labs/17_retrieval/README.md) (recall@k, nDCG), answer faithfulness, and latency and cost per query, measured before and after each change with paired CIs. For ECHOME: a task suite with pass/fail criteria, memory-recall probes across sessions, and a taxonomy of the failures you saw. This turns "built a RAG app" into evidence that applied and evals teams recognise.

## Artifact standards

Every artifact, from a blog post to a PR, should pass all of these.

| standard | what it means | how to check |
|---|---|---|
| Predicted | the prediction was written before the first run, and the post links it | a journal entry dated before the results |
| Reproducible | one command per figure; pinned environment; seeds; hardware and run time stated | a stranger reruns it from the README |
| Honest statistics | 95% CIs on headline numbers; n stated; paired tests for comparisons; several seeds for training runs | no bare point estimates |
| Ablated | each claimed ingredient removed once | an ablation table |
| Traceable | every number links to the script, config and commit that produced it | click through from the post to the code |
| Scoped | one question; the title states the finding | the TL;DR fits in two sentences |
| Legible | one figure a non-specialist can read; about ten minutes to read | a friend can tell you what you found |
| Clean rights | data and model licenses checked; no client or private data | licenses listed in the README |

## Experiment write-up template

The journal's [experiment template](../../journal/templates/experiment.md) is the private lab notebook; this is the public version. CIs come from [lab 15](../../labs/15_eval_stats/README.md) and [module 08 §2](../../curriculum/08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen). For training runs, include seed-to-seed variation, not only the sampling error over eval items.

```markdown
# <The finding, as a sentence>

**TL;DR:** <headline number with its 95% CI, and why it matters>
**Code:** <repo link at a commit> · **Prediction:** <journal link, dated>

## Question
One question, and who would care about the answer.

## Prediction (written before the first run; never edited)
The number you expected, with a range, and the reasoning behind it.

## Setup
- Model, tokenizer and data: sources, sizes, splits, licenses
- Config file link; seeds (at least 3 for training runs); hardware; GPU-hours
- Metrics and eval sets with item counts; how CIs are computed (bootstrap over items, clustered by prompt, over seeds)
- Baselines, and why they are the right ones

## Results
| variant | metric (95% CI) | difference vs baseline (paired 95% CI) | n |
|---|---|---|---|
One figure per claim; units on the axes; error bars defined in the caption.

## Ablations
Change one ingredient at a time; same table format.

## Surprises
Where results differed from the prediction, your best explanation, and the check you ran (or would run) to test it.

## Limitations
Scale, data, metric validity, and what this does not show.

## Next steps
The one to three experiments you would run next, and the result that would change your conclusion.

## Reproduce
One command per figure.
```

## Repo hygiene checklist

- [ ] The README opens with a two-sentence summary, the headline figure and a results table.
- [ ] A "Reproduce" section: exact commands, expected run time, hardware.
- [ ] Environment pinned: `pyproject.toml` with a lockfile (or `requirements.txt` with versions), plus Python, CUDA and driver versions.
- [ ] Configs in files (YAML or TOML), not constants in code; every run saves its config and the git commit hash next to its outputs.
- [ ] Seeds set and logged; nondeterministic operations noted.
- [ ] Tests for the core logic (losses, masking, tokenizer round-trips) running in CI on a CPU.
- [ ] Per-item eval outputs saved as JSONL, so others can recompute your intervals.
- [ ] Large files kept out of git: weights and datasets on the Hugging Face Hub or in releases, fetched by a script that verifies checksums.
- [ ] No secrets: `.env` in `.gitignore`, and no API key anywhere in the history, not only in the last commit.
- [ ] A license for the code, and the licenses of all data and base models listed.
- [ ] A clear layout (`src/`, `configs/`, `scripts/`, `tests/`, `results/`); notebooks for exploration only, with the logic in modules.
- [ ] Linted and formatted (for example with ruff); type hints on public functions.
- [ ] A tag or release for the exact version each write-up used.
- [ ] A `CITATION.cff` if you hope people will cite it.

## Model card template

On the Hugging Face Hub, the model card is the repository's `README.md`, with YAML metadata at the top ([public presence](../../career/public-presence.md#hugging-face)). Keep it short; the evaluation and limitations sections matter most. Choose the license only after checking the licenses of your data and of any base model; the one below is an example.

```markdown
---
license: apache-2.0
language:
  - te
  - en
datasets:
  - <Hub dataset ids you trained on>
pipeline_tag: text-generation
---

# <model name>

One paragraph: what it is, its size, what it is for.

## Intended use and out-of-scope use
## Training data
Sources and licenses; tokens per language; normalization, filtering, deduplication; decontamination against the eval sets.
## Training procedure
Architecture, parameters, tokenizer and vocabulary size, tokens seen, hardware, GPU-hours, key hyperparameters; link to the config and commit.
## Evaluation
| eval | metric | this model (95% CI) | baseline (95% CI) | n |
|---|---|---|---|---|
## Limitations and risks
Known failure modes (hallucination, script mixing, dialect coverage) and what was not evaluated (for example safety or toxicity).
## Citation and contact
```

## Open-source contributions

A merged PR is code reviewed by the maintainers of a project that core teams use every day, and a stranger can verify it in a minute. The [open-source ladder](../../career/public-presence.md#open-source) covers etiquette; this section covers where to start and how to get merged.

| project | what it is | entry points after the labs | labs |
|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | high-throughput LLM serving engine | bugs reproduced with a minimal script; model-support fixes; benchmarks; docs | 07, 13, 14 |
| [SGLang](https://github.com/sgl-project/sglang) | serving engine with prefix caching (RadixAttention) and structured generation | as for vLLM; reproductions of benchmark numbers | 07, 14 |
| [TRL](https://github.com/huggingface/trl) | post-training library with SFT, DPO and GRPO trainers | tests for loss and masking correctness; examples; docs | 10, 11, 12 |
| [transformers](https://github.com/huggingface/transformers) | model definitions, tokenizers and generation | model-specific bugs; generation and tokenizer edge cases | 04, 05, 07 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ inference with GGUF quantization | model conversion scripts; quantization quality checks with its perplexity tool; backend bugs | 13, 14 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | EleutherAI's eval harness with tasks defined in YAML | a new task (for example a Telugu benchmark) with its README; fixes to existing task definitions | 15 |
| [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch-native platform for large-scale pretraining (FSDP2, tensor and pipeline parallelism, float8) | docs and tests; small features; reported numbers reproduced at small scale | 05, 09 |
| [torchao](https://github.com/pytorch/ao) | PyTorch-native quantization and sparsity | tests and benchmarks for quantization schemes; docs | 13 |

**Steps**
1. **Pick two projects you already use** in the labs or at work, matched to your target team. Contributions that come from real use are easier to get right and easier to talk about.
2. **Build from source and run the tests** following the contributing guide, which may recommend a different development setup from the user install. On Windows, use WSL2 for projects that assume Linux.
3. **Learn the norms:** read `CONTRIBUTING.md`, ten recently merged PRs, and the issue labels (look for "good first issue" or similar; names vary by repo).
4. **Find a candidate:** a bug you hit during a lab (the best source), an issue with a reproduction but no PR, missing tests, or confusing docs. Telugu text is a good stress test for tokenizers and generation, because Indic scripts exercise Unicode normalization and byte fallback.
5. **Comment before coding** anything larger than a small fix: say what you found and what you plan, and wait for a maintainer to agree.
6. **Reproduce, write a failing test, then make the smallest fix that passes it.** For performance changes, add a benchmark with before and after numbers, hardware and versions.
7. **Write the PR description:** problem, root cause, fix, test, benchmark where relevant, linked issue. Run the project's formatters and pre-commit hooks first.
8. **Respond to review within a day or two,** without arguing about style. If a PR stalls for a week or two, one polite ping is fine.
9. **After a merge, stay in the same area:** review others' PRs there and answer issues. Ownership of one small area is worth more than scattered one-off fixes.

Review in busy repositories can take days to weeks, so keep two PRs in flight in different projects.

## Flagship: Telugu-first small LM

It is uncrowded, it spans the full stack (tokenizer, pretraining, post-training, evals, release), it matters to Indian AI companies and global labs alike, and it can become a product ([founder track](../founder/README.md)). Every milestone is a publishable artifact on its own, so the project pays off even if you stop early.

**Budget assumptions:** an RTX 5060 laptop GPU with 8 GB, the $6ND$ training estimate, and module 08's planning figure of 20 TFLOP/s achieved in bf16 for small models. Measure your own throughput with `python tools/measure_gpu.py` over a long run (laptop GPUs can throttle under sustained load) and rescale every number below.

| milestone | deliverable | success metric | GPU budget | weeks |
|---|---|---|---|---|
| M0 Data | a Telugu–English corpus with a data card; held-out splits; NFC normalization; exact and near-duplicate removal | licenses documented per source; bytes per token of 2–3 existing tokenizers measured on Telugu and English | CPU only | 2–3 |
| M1 Tokenizer study | [research project 1](../../curriculum/08-evaluation-and-research.md#1-tokenizer-fairness-for-telugu): BPE tokenizers with two pre-tokenizers at 2–3 vocabulary sizes | bits per byte against vocabulary size, with seed CIs; the final tokenizer chosen for a stated reason | ~9 h | 3 |
| M2 Scaling sweep | 4–5 models from ~5M to ~40M parameters; $L(C)$ fitted as in [lab 08](../../labs/08_scaling_laws/README.md) | the main run's final loss predicted, with an interval, before launch | 4–8 h | 2–3 |
| M3 Pretraining | a ~100M-parameter bilingual model on ~2B tokens (about 20 tokens per parameter), with the [lab 05](../../labs/05_transformer/README.md) architecture | no unrecovered loss spikes; held-out bits per byte inside the M2 interval, or the miss explained | ~17 h of compute; budget ~35 h for restarts and evals | 3–4 |
| M4 Adaptation | SFT, then GRPO, on a template-generated Telugu extraction task with a programmatic checker; the same recipe on Qwen2.5-0.5B-Instruct with LoRA as a baseline ([module 06 §8](../../curriculum/06-post-training.md#8-an-8-gb-practical-path)) | held-out exact match beats the base model with a paired bootstrap CI that excludes zero; forgetting and response-length drift reported | SFT under 1 h; GRPO about 6 runs of a few hours | 4–5 |
| M5 Release | model, tokenizer, eval set (if licenses allow) and a CPU demo on the Hugging Face Hub; model card; write-up | a stranger reproduces the main table from the README; the eval task proposed to lm-evaluation-harness | 2–4 h | 2–3 |
| **Total** | | | **~70–90 GPU-hours** | **16–21** |

**M3 arithmetic.** $6 \times 10^8 \times 2\times10^9 = 1.2\times10^{18}$ FLOPs; at $2\times10^{13}$ FLOP/s that is $6\times10^4$ s, about 17 hours, or two to three nights. Weights, gradients and AdamW states take about 16 bytes per parameter, so ~1.6 GB; activations take the rest. Use bf16 autocast, small micro-batches with gradient accumulation, and activation checkpointing if needed, and read `torch.cuda.max_memory_allocated()` after the first steps. The embedding matrix takes a large share of a small model: a 32k vocabulary at width 768 is ~25M parameters, so tie input and output embeddings. If 17 hours is too much, 50M parameters on 1B tokens is $3\times10^{17}$ FLOPs, about 4 hours.

**Data.** Telugu Wikipedia and the Telugu portions of open multilingual web corpora (for example FineWeb-2 or AI4Bharat's Sangraha) are natural starting points; check each source's license and terms before training on it or redistributing it. If clean Telugu text is scarce, repeating it for up to about four epochs costs little ([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)); beyond that, shrink the model rather than repeat more.

**The task.** Generate Telugu sentences from templates that mention a name, a date, a place and an amount, and train the model to return them as JSON. The generator gives unlimited labelled data and an exact checker, which is what GRPO needs. Keep the held-out templates disjoint from the training templates, so the eval measures generalization rather than template recall.

**Evals.** Bits per byte on held-out Telugu and English text (never per-token loss, which is not comparable across tokenizers); exact match on the held-out templates; a small general set to catch forgetting; every number with a CI from [lab 15](../../labs/15_eval_stats/README.md). Decontaminate: check that no eval text, including sentences from any public benchmark you use, appears in the pretraining corpus.

**Where it fits in the roadmap.** M0–M1 alongside lab 04 (stage 2), M2 with lab 08 (stage 3), M3 in January–February, M4 with labs 10–12 (stage 4), and M5 with lab 15 (stage 6). See [Plan A](README.md#plan-a-12-months-from-scratch).

**Without a GPU.** Run M0 and M1 at 2–5M parameters on a CPU or a free notebook GPU; the tokenizer study alone is a publishable result. Free notebook tiers have weekly GPU quotas that change, so check the current limits.

> [!WARNING]
> The usual ways this project goes wrong: comparing per-token loss across tokenizers; mixing NFC and NFD text; romanized Telugu silently mixed into the corpus (decide whether it is in scope, and measure it separately); judging GRPO by training reward instead of held-out accuracy.

## Distribution

An artifact nobody sees does not get you interviews. See [public presence](../../career/public-presence.md#writing) for the writing process.

- **Every artifact gets a post** with the finding in the title, one clear figure, and links to the code and data.
- **Post where people who hire for core roles read:** GPU MODE, EleutherAI and Hugging Face communities, X and LinkedIn.
- **Group each project on the Hugging Face Hub** as a collection: model, tokenizer, dataset, demo Space and write-up.
- **Send it to three to five people whose work you built on,** with one specific question each, following [outreach](../../career/outreach.md).
- **Answer every substantive comment,** and publish corrections as a dated update rather than a silent edit.
- **Measure what matters:** replies from practitioners, reuse, citations and inbound messages from recruiters, not likes or stars.
- **Turn the best piece into a workshop paper,** the roadmap's stage 9 target.

## Strong vs weak portfolios and bullets

Values in angle brackets are placeholders for your own measured numbers; never fill them with estimates. Bullets follow the [bullet formula](../../career/resume/guide.md#bullet-formula) in the résumé guide.

| kind | weak | strong | why the strong one works |
|---|---|---|---|
| Portfolio | ten tutorial notebooks (MNIST, Titanic, a PDF chatbot) | labs 01–17 passing, three write-ups with predictions and CIs, one flagship, two merged PRs | depth and verification over volume |
| Portfolio | a model on the Hub with an empty card | a card with data sources, training compute, evals with CIs, and limitations | shows judgment, not only training |
| Skills | "Familiar with PyTorch, Transformers, CUDA" | "Wrote FlashAttention's forward pass in Triton: <x>% of PyTorch SDPA's throughput at 1k–8k tokens on an RTX 5060 (benchmark linked)" | a claim with a measurement and a link |
| Research | "Reproduced the DPO paper" | "Reproduced DPO's reward–KL trade-off on a 0.5B model; the trend matched and the absolute rewards did not; traced the gap to <cause> (write-up)" | honest about discrepancies |
| Post-training | "Fine-tuned LLMs with RLHF" | "GRPO with LoRA on Qwen2.5-0.5B for Telugu extraction: held-out exact match <a>% → <b>% (paired 95% CI <c> to <d> points); length drift reported" | method, metric, eval set and uncertainty |
| Open source | "Open-source contributor" | "Fixed <bug> in vLLM's <component>, with a regression test (merged PR #<n>)" | reviewed by maintainers; verifiable in a minute |
| Applied | "Built a RAG chatbot" | "Raised FinSentinelAI answer accuracy from <a>% to <b>% on a <n>-question held-out set with hybrid retrieval and reranking; p95 latency <t> ms" | before and after on a fixed eval |
| Production | "Improved OCR accuracy significantly" | "Raised OCR accuracy from <a>% to <b>% on <n> held-out documents; INT8 ONNX on AWS Lambda cut cost per page by <x>%" | only when the numbers come from your own records |
