# Lab 11 — DPO

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 11_dpo`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [post-training §4](../../curriculum/06-post-training.md#4-dpo-and-its-family)

## Implement and test
- `sequence_logprobs(logits, labels, mask)`: shift by one, gather, and sum over response tokens only. This is the most common off-by-one bug in alignment code.
- `dpo_loss(πc, πr, refc, refr, β, label_smoothing)` returns the loss and the implicit rewards `β·(logπ − logπ_ref)`. Test against hand-computed values.
- `ipo_loss`, `simpo_loss` (length-normalized, reference-free, margin γ).
- **The theory test:** a tabular policy over 6 responses, with Bradley–Terry soft labels from hidden rewards r. Minimizing DPO must converge to `π* ∝ π_ref·exp(r/β)`. You verify the derivation numerically.

## GPU scale-up / stretch
After S4, build 500 preference pairs (a strong model as judge), run DPO with β ∈ {0.05, 0.1, 0.5}, and track win rate against KL to the reference.
