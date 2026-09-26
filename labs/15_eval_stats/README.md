# Lab 15 — Evaluation statistics

**Build:** normal and bootstrap confidence intervals, two paired tests (sign-flip permutation and exact McNemar), cluster-robust standard errors, the unbiased pass@k estimator, a power calculation, Holm–Bonferroni, Cohen's kappa for judge agreement, and a Bradley–Terry fit for arena votes. Then put error bars on the eval results of labs 10–12.
**Time:** 2–3 h for the tests, 3–6 h for the scale-up · **Reads first:** [evaluation §2](../../curriculum/08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen)
**Run:** `pytest labs/15_eval_stats` (your code) · `pytest labs/15_eval_stats --impl=solution` (reference). The six tests run on CPU in a few seconds.

Most decisions in model development (ship the fine-tune, keep the ablation, claim the gain) come down to a difference of a few points on a few hundred items, and many such differences are noise. Research-engineer interviews test this directly: "is 62.1 vs 60.8 significant?", "how many items would you need?", "how do you estimate pass@k?". This lab turns each answer into a function you have tested, and the scale-up makes you use them on your own results.

---

## 1. One model, one score: the standard error

A benchmark score is the mean of $n$ per-item scores. For binary items with accuracy $\hat p$, one item has variance $\hat p(1-\hat p)$, so

$$\text{SE} = \sqrt{\hat p(1-\hat p)/n}, \qquad \text{95\% CI} = \hat p \pm 1.96\,\text{SE}.$$

**Worked example.** 500 questions at 70%: $\text{SE} = \sqrt{0.7 \cdot 0.3/500} = 0.0205$ and the 95% half-width is $1.96 \times 0.0205 = 0.040$. The score is **70.0 ± 4.0 points**, [66.0, 74.0]; a model at 73% sits inside that interval (section 2 shows how to compare the two properly). The test's 1,000-item benchmark at 70% gives ±2.84 points. Halving an interval always costs 4× the items.

- **Bootstrap.** Resample the $n$ items with replacement, recompute the mean, repeat 10,000 times, and take the 2.5th and 97.5th percentiles. On the test's benchmark it gives [67.2, 72.8] against the normal [67.16, 72.84]. For a plain mean the two agree; the bootstrap earns its place for statistics without a simple SE formula (F1, nDCG, a median, a ratio).
- **Near 0% or 100%** the normal interval fails. At $n = 100$ and a true 95%, the nominal 95% normal interval covers the truth only 87.7% of the time (exact binomial calculation); the Wilson interval covers 96.6%. Zero successes out of 50 gives a normal interval of [0, 0] and a Wilson interval of [0, 7.1%].
- **Several samples per item** (say 5 generations per question): average within each item first, then compute the SE over items. Treating 5 × 500 generations as 2,500 independent items understates the SE, because generations for one question are correlated.

## 2. Two models, same items: pair the comparison

The variance of a difference is

$$\operatorname{Var}(a - b) = \operatorname{Var}(a) + \operatorname{Var}(b) - 2\operatorname{Cov}(a, b).$$

An unpaired comparison (two separate CIs, or a two-sample test) assumes the covariance is zero. On shared items it is large and positive, because hard questions are hard for both models. Pairing subtracts it.

**Worked example.** 500 items; baseline B scores 70.0%, new model A scores 73.0%:

|  | B right | B wrong |
|---|---|---|
| **A right** | 335 | 30 |
| **A wrong** | 15 | 120 |

- **Unpaired:** $\text{SE} = \sqrt{0.73 \cdot 0.27/500 + 0.70 \cdot 0.30/500} = 0.0285$, so the difference is +3.0 ± 5.6 points (z-test p = 0.29).
- **Paired:** the per-item differences $d_i$ are +1 on 30 items, −1 on 15 and 0 on 455. $\text{SD}(d) = 0.299$ and $\text{SE} = 0.299/\sqrt{500} = 0.0134$, so the difference is **+3.0 ± 2.6 points**, [+0.4, +5.6]. Per-item correctness of A and B has correlation 0.78, which cuts the variance of the difference from 0.407 (unpaired) to 0.089.
- **Paired bootstrap:** resample item indices and recompute mean(a) − mean(b) on the same indices. That is exactly `bootstrap_ci(a - b)`, which gives [+0.4, +5.6].
- **McNemar exact test:** only the 45 discordant items carry information about which model is better. Under the null each is equally likely to favor either model, so the A-only count is Binomial(45, ½) and $p = 2\,P(X \le 15) = 0.036$. The sign-flip permutation test gives 0.038.

`test_paired_tests_detect_small_consistent_gains` is the same logic at its extreme: model b fixes 17 of a's 500 items and breaks none (55.0% → 58.4%). McNemar gives $p = 2 \cdot 2^{-17} = 1.5 \times 10^{-5}$ and the permutation test 0.0002 (its floor with 10,000 permutations is $1/10{,}001$). An unpaired z-test on the same data gives p = 0.28.

**Permutation test mechanics.** If the two models are exchangeable on every item, each $d_i$ is as likely to be $+|d_i|$ as $-|d_i|$. Flip signs at random 10,000 times, count how often $|\text{mean}|$ reaches the observed value, and report $(\text{count} + 1)/(n_\text{perm} + 1)$, which is never exactly zero. Unlike McNemar it works for any per-item score (graded answers, reward-model scores), not only 0/1.

## 3. Items that share context: clustered standard errors

Five questions about one passage, twenty test cases from one repository, several turns of one conversation: outcomes inside a cluster are correlated, and the naive SE, which assumes $n$ independent items, is too small. Write the error of the mean as a sum over clusters, $\bar x - \mu = \frac1n \sum_c S_c$ with $S_c = \sum_{i \in c}(x_i - \mu)$. Clusters are independent, so $\operatorname{Var}(\bar x) = \frac{1}{n^2}\sum_c \operatorname{Var}(S_c)$, and plugging in each observed $S_c^2$ gives

$$\text{SE}_\text{cluster} = \frac{1}{n}\sqrt{\sum_c \Big(\sum_{i \in c}(x_i - \bar x)\Big)^2}.$$

With one item per cluster this reduces to the naive SE (with ddof = 0).

**Design effect.** With clusters of size $m$ and intra-cluster correlation $\rho$, the variance of the mean is multiplied by $1 + (m-1)\rho$. With $m = 5$ and $\rho = 0.3$ that is 2.2× the variance and 1.48× the SE: 1,000 questions carry the information of about 455 independent ones. The test is the extreme case: 8 passages × 10 questions with identical outcomes inside each passage ($\rho = 1$), a design effect of 10, and an SE ratio of $\sqrt{10} = 3.16$ (0.177 against a naive 0.056). Eighty questions are worth eight.

With few clusters (tens rather than hundreds) the cluster-robust SE is itself noisy and biased low; the usual small-sample correction multiplies it by $\sqrt{G/(G-1)}$ for $G$ clusters. The bootstrap alternative resamples **clusters**, never items.

## 4. pass@k without bias

pass@k is the probability that at least one of $k$ samples solves a problem. Draw $n \ge k$ samples, of which $c$ pass. The estimator from Chen et al. 2021 ([arXiv 2107.03374](https://arxiv.org/abs/2107.03374)) is

$$\widehat{\text{pass@}k} = 1 - \binom{n-c}{k} \Big/ \binom{n}{k},$$

the fraction of the $\binom nk$ size-$k$ subsets of your samples that contain at least one pass. Each subset is a legitimate draw of $k$ samples, so its indicator has expectation pass@k, and so does the average over all subsets: the estimator is unbiased.

The plug-in $1 - (1 - \hat p)^k$ with $\hat p = c/n$ is not. $f(p) = 1 - (1-p)^k$ is concave for $k \ge 2$, so by Jensen's inequality $\mathbb{E}[f(\hat p)] \le f(p)$: the plug-in is biased **low**. Exact expectations over $c \sim \text{Binomial}(n, p)$:

| true p | n | k | true pass@k | mean of plug-in | mean of unbiased estimator |
|---|---|---|---|---|---|
| 0.15 | 20 | 5 | 0.556 | 0.518 | 0.556 |
| 0.05 | 20 | 10 | 0.401 | 0.336 | 0.401 |

That is 3.8 points of bias in the first row and 6.5 in the second. Compute the estimator as a product, $1 - \prod_{i=n-c+1}^{n}(1 - k/i)$: no huge integers ($\binom{2000}{1000}$ has 601 digits) and it vectorizes. If $n - c < k$, every $k$-subset contains a pass and the answer is exactly 1. Average the per-problem estimates over problems and get the CI from those per-problem values, as in section 1. Use $n \gg k$: the Codex paper used $n = 200$ for $k \le 100$. For agents, also report pass^k (all $k$ tries succeed), estimated per task by $\binom{c}{k}/\binom{n}{k}$; it is the reliability users actually experience.

## 5. Power: decide n before you collect

Power is the probability of detecting an effect of a given size when it is real. For an unpaired two-proportion z-test at level α and power $1-\beta$, which is what `required_n` computes:

$$n_\text{per model} = \frac{\Big(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{1-\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)}\Big)^2}{(p_1 - p_2)^2}, \qquad \bar p = \tfrac{p_1 + p_2}{2}.$$

The first term is the rejection threshold under the null (both models at $\bar p$); the second is the margin needed so that the estimate under the alternative clears that threshold 80% of the time. With $z_{0.975} = 1.960$ and $z_{0.8} = 0.842$:

| Detect (α = 0.05, power 0.8) | 50% → 60% (the test) | 80% → 85% | 70% → 75% | 90% → 92% | **70% → 72%** | 50% → 52% |
|---|---|---|---|---|---|---|
| Items per model, unpaired | 388 | 906 | 1,251 | 3,213 | **8,080** | 9,806 |

A 2-point gain at 70% needs 8,080 items per model unpaired. For paired binary outcomes the requirement depends on the discordance rate ψ, the fraction of items where the models disagree: $n \approx \big(z_{1-\alpha/2}\sqrt\psi + z_{1-\beta}\sqrt{\psi - \delta^2}\big)^2/\delta^2$ for a gap δ. For the same 2-point gain, ψ = 16% needs 3,137 items and ψ = 8% needs 1,567. Similar models disagree less, and that is where pairing pays most.

**Minimum detectable effect (MDE).** Run the calculation backwards. Near 70% accuracy, the smallest gap you can reliably detect unpaired is about $(z_{1-\alpha/2} + z_{1-\beta})\sqrt{2p(1-p)/n}$: 12.8 points at $n = 200$, 8.1 at 500, 5.7 at 1,000. Paired, at ψ = 10% and 500 items, it is about $(1.96 + 0.84)\sqrt{0.10/500} = 4.0$ points. State the MDE before you run an experiment on a small eval set, not after.

## 6. Many comparisons, many seeds, and judges

**Multiplicity.** Ten ablations that do nothing, each tested at α = 0.05, give at least one "significant" result with probability $1 - 0.95^{10} = 40\%$; twenty give 64%. Holm–Bonferroni controls this family-wise error: sort the $m$ p-values, compare the $r$-th smallest ($r = 0, 1, \dots$) with $\alpha/(m - r)$, and stop at the first failure. For the test's [0.01, 0.04, 0.03, 0.005]: 0.005 ≤ 0.0125 and 0.01 ≤ 0.0167 are rejected, then 0.03 > 0.025 stops the procedure, so 0.04 is not rejected although it is below 0.05 on its own.

**Seeds.** Sampling items is one source of variance; training (initialization, data order) is another, and more eval items do nothing about it. With $k$ training seeds, the CI on the mean uses Student's t with $k - 1$ degrees of freedom: $t_{0.975} = 4.30$ for 3 seeds and 2.78 for 5, not 1.96. Evaluate every seed on the same items and compare the seed-to-seed spread with the item-level SE: if the seed spread dominates, add seeds, not items.

**Judges.** Raw agreement between an LLM judge and human labels flatters the judge when one label dominates. Cohen's kappa corrects for chance: $\kappa = (p_o - p_e)/(1 - p_e)$, where $p_e = \sum_\ell P_a(\ell)\,P_b(\ell)$ is the agreement two independent raters with the same label frequencies would reach. Worked example on 100 items: both say pass on 65, both say fail on 20, only the human says pass on 5, only the judge on 10. Agreement is 85%, but $p_e = 0.70 \cdot 0.75 + 0.30 \cdot 0.25 = 0.60$, so $\kappa = 0.25/0.40 = 0.625$.

## 7. Arena data: Bradley–Terry and Elo

Pairwise votes ("A's answer is better than B's") are modeled by Bradley–Terry: $P(i \text{ beats } j) = s_i/(s_i + s_j) = \sigma(\theta_i - \theta_j)$ with $\theta = \ln s$. Elo is the same model on the scale $R = 400 \log_{10} s$:

| Elo gap | 30 | 100 | 200 |
|---|---|---|---|
| Expected win rate | 54.3% | 64.0% | 76.0% |

The maximum-likelihood fit has a simple fixed-point iteration, the MM algorithm (Hunter 2004). With $W_i$ the total wins of $i$ and $n_{ij}$ the number of games between $i$ and $j$:

$$s_i \leftarrow \frac{W_i}{\sum_j n_{ij}/(s_i + s_j)}, \qquad \text{then rescale so that } \textstyle\sum_i s_i = 1.$$

Each update increases the likelihood, and the rescaling matters because the likelihood depends only on ratios. The test builds an expected win matrix from strengths 4 : 2 : 1, i.e. [0.571, 0.286, 0.143]: the top model sits 120 Elo above the second and 241 above the third.

Online Elo updates depend on the order of the games and on the K-factor; a Bradley–Terry fit on all votes at once does not, and it is what the Chatbot Arena paper uses ([arXiv 2403.04132](https://arxiv.org/abs/2403.04132)). Get CIs by bootstrapping the votes and refitting. Small gaps are expensive: to see a 30-point gap (54.3%) in head-to-head votes with a 95% CI that excludes 50% takes about 520 votes, and about 1,060 for 80% power. Models whose CIs overlap are tied, however the table is sorted.

## 8. Reporting a result

A comparison you can defend reports each model's interval, the paired difference with its interval and test, and the setup that produced them. For the section 2 example:

| Model | Accuracy, % (95% CI) | Δ vs baseline, points (paired 95% CI) | McNemar p |
|---|---|---|---|
| Baseline (B) | 70.0 (66.0–74.0) | — | — |
| Fine-tuned (A) | 73.0 (69.1–76.9) | +3.0 (+0.4 to +5.6) | 0.036 |

*n = 500 items (500 clusters), greedy decoding, one training seed. Per-model CI: normal. Difference: paired bootstrap, 10,000 resamples. Variants tried: 1. Per-item outputs: `runs/<name>/items.jsonl`.*

The per-model intervals overlap heavily, yet the paired interval excludes zero: overlapping per-model CIs are not a test. With one seed the claim is about this checkpoint, not about the method. The full reporting checklist is §2.9 of the [evaluation module](../../curriculum/08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen).

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `mean_ci(scores, conf=0.95) -> (mean, low, high)` | `test_ci_width_for_a_1000_item_benchmark` | 700 ones and 300 zeros: mean exactly 0.7, half-width 0.0284 ± 0.0005 (sample std, `z` from `NormalDist`) |
| 2 | `bootstrap_ci(scores, n_boot=10000, conf=0.95, seed=0) -> (low, high)` | same | percentile bounds within 0.004 of the normal bounds |
| 3 | `paired_permutation_test(a, b, n_perm=10000, seed=0) -> float` | `test_paired_tests_detect_small_consistent_gains` | two-sided p < 0.01 when b fixes 17 of a's 500 items and breaks none |
| 4 | `mcnemar_exact(a_correct, b_correct) -> float` | same | p < 0.01 on the same data; exactly 1.0 with no discordant pairs |
| 5 | `clustered_se(scores, clusters) -> float` | `test_clustered_se_exceeds_naive_when_clusters_correlate` | more than 2.5× the naive SE for 8 clusters of 10 identical outcomes (exact ratio √10) |
| 6 | `pass_at_k(n, c, k) -> float` | `test_pass_at_k` | 0 when c = 0; 0.3 for (10, 3, 1); 1 − C(7,2)/C(10,2) = 0.533 for (10, 3, 2); 1.0 for (5, 4, 2), where n − c < k |
| 7 | `required_n(p1, p2, alpha=0.05, power=0.8) -> int` | `test_power_and_multiple_comparisons` | exactly 388 for 50% vs 60% |
| 8 | `holm_bonferroni(pvals, alpha=0.05) -> list[bool]` | same | [0.01, 0.04, 0.03, 0.005] → [True, False, False, True], in input order |
| 9 | `cohens_kappa(a, b) -> float` | `test_kappa_and_bradley_terry` | 1.0 for identical labels; −1.0 for [1,0,1,0] vs [0,1,0,1] |
| 10 | `bradley_terry(wins, iters=500) -> np.ndarray` | same | recovers strengths ∝ (4, 2, 1), summing to 1, within rtol 1e-4 |

Every function starts as a stub; there is no given code in this lab. `wins[i, j]` is the number of times `i` beat `j`.

## Tips

> [!TIP]
> Before running pytest, build `a` and `b` from the four cells of the section 2 table and check your paired functions against known answers: `mcnemar_exact(a, b)` = 0.036, `paired_permutation_test(a, b)` ≈ 0.038, `bootstrap_ci(a - b)` ≈ (0.004, 0.056). A fixture you understand beats a failing assertion you don't.

- Take `z = NormalDist().inv_cdf(0.5 + conf / 2)` (1.95996 at 95%) rather than hard-coding 1.96, so other confidence levels work.
- `bootstrap_ci`: one index matrix `rng.integers(0, n, (n_boot, n))` and a mean along axis 1 is short and correct. It holds `n_boot × n` int64 indices (80 MB for 10,000 × 1,000), so chunk it for large `n`.
- `mcnemar_exact`: with $n = n_{01} + n_{10}$ discordant pairs, the tail is $\sum_{i \le \min(n_{01}, n_{10})} \binom{n}{i} / 2^n$ and $p = \min(1, 2 \cdot \text{tail})$. Use `math.comb`, which is exact.
- `paired_permutation_test`: count null statistics with `>= obs - 1e-12`. For graded scores, sums that are equal on paper can differ in the last bit.
- `bradley_terry`: `n = w + w.T` (the diagonal of `wins` is zero), and the denominator vectorizes as `(n / (s[:, None] + s[None, :])).sum(1)`.

## Common bugs

- **Pooled variance in both terms of `required_n`:** using $\sqrt{2\bar p(1-\bar p)}$ in the power term too gives 388.5, which rounds up to 389; the correct 387.3 rounds up to the tested 388. Forgetting `math.ceil` also fails.
- **A permutation p-value of exactly 0:** without the +1 in numerator and denominator, a strong effect reports p = 0, which is not a valid p-value.
- **Holm flags in sorted order** instead of input order. The expected `[True, False, False, True]` follows the input.
- **McNemar without the `min(1, ·)`:** with one discordant pair each way the doubled tail is 1.5.
- **Bradley–Terry without rescaling:** the likelihood is unchanged when all $s$ are multiplied by a constant, so the iterates drift and the absolute comparison in the test fails.
- **Generations counted as items** (section 1). The tests do not catch it, and it is the most common error in real eval code.

## CPU experiments (no GPU needed)

1. **Coverage.** Simulate 10,000 benchmarks of 100 items at a true 95% and count how often `mean_ci` contains 0.95. Predict first: exact calculation gives 87.7% for the normal interval and 96.6% for Wilson. Repeat at 70% with 500 items, where the normal interval is fine (94.9%).
2. **Discordance.** Rebuild the section 2 table, then grow both discordant cells while keeping their difference at 15 (30/15 → 45/30 → 75/60). The marginals stay at 73% and 70%, so the unpaired CI does not move at all. Predict how the paired CI and McNemar's p change from $\text{SD}(d) \approx \sqrt{\psi}$, then check.
3. **pass@k bias.** Reproduce the section 4 table by simulation, then plot the bias of the plug-in against $k$ for $n = 20$ and a few values of $p$.
4. **Seeds.** `train_sae(seed=s)` from [lab 16](../16_interpretability/README.md) returns the fraction of recovered features and takes a few seconds on a CPU. Run 5 seeds and report the mean with a t-interval ($t_{0.975,4} = 2.78$).
5. **Arena.** Give 8 models true Elo ratings 30 points apart, simulate votes between random pairs, fit `bradley_terry`, bootstrap the votes 200 times, and count how many adjacent pairs have non-overlapping 95% CIs at 1k, 10k and 100k votes.

## GPU scale-up (8 GB)

Put error bars on results you already have. A free Colab or Kaggle T4 works too; use fp16 there, since T4 has no bf16 tensor cores.

1. **Lab 10 (LoRA SFT on GSM8K):** score the base and the fine-tuned model on the same 500 held-out questions with greedy decoding, save per-item correctness, and fill in the section 8 table: per-model CIs, the paired difference from `bootstrap_ci(a - b)`, and `mcnemar_exact`. Compute the MDE for your observed discordance before you believe any gain below it.
2. **pass@k on your GRPO policy ([lab 12](../12_grpo/README.md)):** sample $n = 16$ completions per question at temperature 1.0 for 200 questions (3,200 generations; batch 32–64 prompts per call). Report pass@1, pass@4 and pass@8 with the unbiased estimator and with the plug-in, each with a CI over questions. Compare base and GRPO models: does the pass@1 gain persist at pass@8?
3. **Seeds:** rerun the lab 10 fine-tune with 3 seeds on the same data and compare the seed spread with the item-level SE from step 1. With 3 seeds, the t multiplier is 4.30.
4. **Judge agreement (optional):** have a small instruct model grade 100 answers as correct or incorrect, grade the same 100 yourself, and report Cohen's kappa next to raw agreement.

## Check yourself

1. A model scores 70% on 500 questions. What is the 95% CI, and how many questions would give ±2 points?
2. On 1,000 shared items, only A is right on 90 and only B on 70. Is A better at α = 0.05?
3. Why is $1 - (1 - c/n)^k$ a biased estimate of pass@k, in which direction, and how does the unbiased estimator avoid it?
4. 1,000 reading-comprehension questions, 5 per passage, intra-passage correlation 0.3. What is the effective sample size, and how should you bootstrap?
5. You want to detect a 2-point gain at 70%. How many items unpaired, and what property of the two models makes a paired design much cheaper?
6. An LLM judge agrees with human labels on 85% of items. What do you need before trusting it?
7. Two arena models differ by 30 Elo. What is the expected win rate, and roughly how many head-to-head votes before a 95% CI on it excludes 50%?

<details><summary>Answers</summary>

1. SE = √(0.21/500) = 0.0205, so 70.0 ± 4.0 points. For ±2 points, n = 0.21 × (1.96/0.02)² = 2,017: four times the items to halve the interval.
2. Not shown. McNemar on the 160 discordant items (90 vs 70) gives p = 0.13; the paired 95% CI on the 2-point gap is about ±2.5 points. Either collect more items (compute the power first) or report it as inconclusive.
3. $1 - (1-p)^k$ is concave in $p$ for $k \ge 2$, so by Jensen's inequality the plug-in underestimates on average (0.518 vs a true 0.556 at p = 0.15, n = 20, k = 5). The unbiased estimator averages the pass indicator over all $k$-subsets of the $n$ samples actually drawn, so it never extrapolates from an estimated rate.
4. Design effect 1 + 4 × 0.3 = 2.2, so about 455 effective questions and an SE 1.48× the naive one. Bootstrap by resampling passages, keeping each passage's 5 questions together.
5. 8,080 items per model unpaired (α = 0.05, power 0.8). Low discordance: at ψ = 8% the paired design needs about 1,570.
6. Base rates and chance agreement: Cohen's kappa (0.625 in the section 6 example), a CI on it, and a check for systematic biases such as position or length. Agreement of 85% is what two raters reach when chance agreement is 60% and kappa is 0.625.
7. 1/(1 + 10^(−30/400)) = 54.3%. The CI half-width 1.96 × 0.5/√n must drop below 0.043, so about 520 votes, and about 1,060 for 80% power.
</details>

## Stretch

- Re-score a public leaderboard with CIs, from per-item results where they are published or from $n$ and the scores alone: which rank orderings survive? That makes a strong blog post.
- Implement the cluster bootstrap and compare it with `clustered_se` on simulated passages as ρ goes from 0 to 1.
- Add Benjamini–Hochberg and compare its rejections with Holm on 20 simulated ablations, 3 of them real.
- Add the Wilson interval and the BCa bootstrap and rerun the coverage experiment.
- Estimate an LLM judge's position bias: judge each pair in both orders and add a first-position advantage parameter to the Bradley–Terry fit.
- Read Miller (2024), *Adding Error Bars to Evals* ([arXiv 2411.00640](https://arxiv.org/abs/2411.00640)), and Card et al. (2020), *With Little Power Comes Great Responsibility* ([arXiv 2010.06595](https://arxiv.org/abs/2010.06595)).
