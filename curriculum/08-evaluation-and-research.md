# 08 — Evaluation and research method

Lab: [15 eval statistics](../labs/15_eval_stats/README.md). This module is what turns an engineer into a member of technical staff.

## 1. What an eval measures
Construct validity: does the score track the capability you care about? Benchmark types: knowledge (MMLU-style), expert reasoning (GPQA), math and code with verifiers (competition math, LiveCodeBench, SWE-bench Verified), agentic tasks (long-horizon; METR's time-horizon metric), preference and arena rankings, safety evals, and **your product's own evals**. Benchmarks saturate and leak into training data: prefer fresh, dynamic or private sets. Contamination checks: n-gram overlap, canary strings, performance on post-cutoff items.
**LLM-as-judge:** position bias, verbosity bias, self-preference. Mitigate with swapped order, rubrics, reference answers and length control, and **measure judge–human agreement** (Cohen's κ) before trusting a judge. **Prompt sensitivity:** formatting and few-shot choices can move scores by points. Fix the harness and report it.

## 2. Statistics: error bars or it didn't happen
- For n items at accuracy p: `SE = √(p(1−p)/n)`. On 500 items at 60%, ±4.3 points (95%).
- **Paired comparisons** (both models on the same items) have much lower variance. Report the mean difference with a CI from a paired bootstrap or permutation test; use McNemar for binary outcomes.
- **Clustered SEs** when items share context (several questions per passage). Naive SEs are too small.
- **Seeds:** training noise is real. Report ≥ 3 seeds for small-scale claims.
- **Power:** before collecting data, compute how many items you need to detect the effect you care about.
- **pass@k:** use the unbiased estimator from n ≥ k samples (Codex paper), not naive resampling.
- **Multiple comparisons:** ten ablations at p < 0.05 give one false positive on average. Use Holm–Bonferroni, or better, a held-out confirmation run.
Reference: Miller 2024, *Adding Error Bars to Evals*.

## 3. Product evals (founder-critical)
Collect real traces, read them, cluster the failure modes into a taxonomy, build a targeted eval set per failure mode, and write graders: code-based first, a model judge only when needed, calibrated against human labels. Run evals in CI on every prompt, model or retrieval change. A new frontier model release becomes a 30-minute decision rather than a debate. See [tracks/founder/02](../tracks/founder/02-ai-product-engineering.md).

## 4. The experiment discipline
1. Write the **hypothesis and your predicted result before running**.
2. Use a strong, tuned baseline at **matched compute**. Most "wins" are an untuned baseline.
3. Change one thing at a time; keep an ablation table.
4. Check that the effect holds at two scales and on two datasets.
5. Report the variance, the compute, and the negative results.
6. Log everything (config, git hash, data version) so any number can be reproduced from the journal.
Template: [journal/templates/experiment.md](../journal/templates/experiment.md).

## 5. Reproducing papers
Pick papers with a small-compute core claim. Reproduce one key figure. Document every unstated detail you had to guess. A clean reproduction that exposes a gap in a paper is valuable, publishable-quality work and strong hiring signal.

## 6. Writing
A paper or post is an argument: claim → evidence → limits. Draft the figures first. One idea per paragraph. State the compute and the uncertainty. Put the main result in the first two sentences.

## 7. Research taste
Work on problems where you have an unfair advantage (Indic languages, document AI, consumer-GPU efficiency). Notice anomalies and chase them. Prefer simple ideas that scale over clever ones that don't. Read broadly; implement narrowly.

## Research projects that fit an 8 GB GPU
1. **Tokenizer fairness for Telugu:** compression and downstream loss at equal compute, GPT-4 vs o200k-style pre-tokenization ([lab 04](../labs/04_tokenizer/README.md)).
2. **Scaling laws for byte-level vs BPE models** on TinyStories-scale data ([lab 08](../labs/08_scaling_laws/README.md)).
3. **GRPO ablations on small models:** loss aggregation, std normalization, KL on or off. Which variant matters at 0.5B? ([lab 12](../labs/12_grpo/README.md))
4. **KV-cache quantization quality vs speed** on consumer Blackwell (FP8/INT4 KV) ([lab 13](../labs/13_quantization/README.md)).
5. **Speculative decoding acceptance vs domain** (code vs chat vs Telugu) ([lab 14](../labs/14_speculative_decoding/README.md)).
6. **SAE features of a TinyStories model** and how they split with dictionary size ([lab 16](../labs/16_interpretability/README.md)).
7. **Error bars on a public leaderboard:** which rankings are real? ([lab 15](../labs/15_eval_stats/README.md))
Any of these, done carefully with CIs, is blog-worthy; two of them become a workshop paper.
