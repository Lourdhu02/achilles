# Lab 16 — Mechanistic interpretability

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 16_interpretability`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [interpretability](../../curriculum/09-interpretability-and-safety.md)

## Implement and test
- Train a 2-layer attention-only transformer on repeated random sequences. Compute an `induction_score` per head (attention from position i to i − T + 1). Test: a layer-2 head scores > 0.5 (chance ≈ 1/T).
- `activation_patching(model, clean, corrupt, site)` with hooks. Test: patching the full residual stream at the last layer restores 100% of the clean logit difference.
- `SparseAutoencoder` (ReLU encoder, unit-norm decoder, MSE + λ‖f‖₁) trained on toy superposition data: 32 sparse features packed into 8 dimensions. Test: most true feature directions are recovered with cosine > 0.9.

## GPU scale-up / stretch
Train an SAE on the residual stream of your TinyStories model from S1 and label its top 20 features by hand. Look for dead latents and feature splitting as the width changes.
