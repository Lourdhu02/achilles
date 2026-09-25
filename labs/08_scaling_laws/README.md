# Lab 08 — Scaling laws

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 08_scaling_laws`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [pretraining §3](../../curriculum/05-pretraining.md#3-scaling-laws)

## Implement and test
- `fit_power_law(N, L)`: fit `L(N) = E + A·N^-α` (grid over E, then least squares in log space). Test: recover known E, A, α from noisy synthetic data.
- `chinchilla_loss(N, D)`: `E + A/N^α + B/D^β`.
- `compute_optimal(C, A, B, α, β)`: closed form under `C = 6ND`: `N* = G·(C/6)^(β/(α+β))`, `G = (αA/βB)^(1/(α+β))`. Test it against brute-force grid search.
- `inference_aware_optimum(target_loss, inference_tokens)`: minimize training plus inference FLOPs (Sardana et al.). Test: more inference demand gives a smaller N.

## GPU scale-up / stretch
**S2 (RTX 5060, overnight):** train 5 model sizes (1–30M parameters) on TinyStories at 3 compute budgets using `labs/05_transformer/train.py` with a WSD schedule. Fit your own IsoFLOP curves and compare the fitted exponents with Chinchilla's. Write it up.
