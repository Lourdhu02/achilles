# Lab 15 — Evaluation statistics

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 15_eval_stats`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [evaluation §2](../../curriculum/08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen)

## Implement and test
- Normal-approximation CI, percentile bootstrap CI, clustered standard errors (questions that share a passage).
- Paired comparison of two models on the same items: paired bootstrap, permutation test, McNemar exact test.
- `pass_at_k(n, c, k)`: the unbiased estimator `1 − C(n−c,k)/C(n,k)`, computed stably.
- `required_n(p1, p2, α, power)`: how many eval items you need to detect a 2-point difference.
- `holm_bonferroni`, `cohens_kappa` (judge–human agreement), and Bradley–Terry fitting for arena-style leaderboards.

## GPU scale-up / stretch
Re-score a public leaderboard table with CIs: which rank orderings survive? That makes a strong blog post.
