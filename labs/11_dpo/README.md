# Lab 11 — DPO

**Run:** `pytest labs/11_dpo` (your code) · `pytest labs/11_dpo --impl=solution` (reference)

**Reads first:** [post-training §4](../../curriculum/06-post-training.md#4-dpo-and-its-family)

## What you implement (the tests check each property)
- `sequence_logprobs(logits, labels, mask)`: shift by one, gather, and sum over response tokens only. This is the most common off-by-one bug in alignment code.
- `dpo_loss(πc, πr, refc, refr, β, label_smoothing)` returns the loss and the implicit rewards `β·(logπ − logπ_ref)`. Test against hand-computed values.
- `ipo_loss`, `simpo_loss` (length-normalized, reference-free, margin γ).
- **The theory test:** a tabular policy over 6 responses, with Bradley–Terry soft labels from hidden rewards r. Minimizing DPO must converge to `π* ∝ π_ref·exp(r/β)`. You verify the derivation numerically.

## GPU scale-up / stretch
After S4, build 500 preference pairs (a strong model as judge), run DPO with β ∈ {0.05, 0.1, 0.5}, and track win rate against KL to the reference.
