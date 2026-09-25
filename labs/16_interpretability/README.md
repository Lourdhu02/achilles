# Lab 16 — Mechanistic interpretability

**Run:** `pytest labs/16_interpretability` (your code) · `pytest labs/16_interpretability --impl=solution` (reference)

**Reads first:** [interpretability](../../curriculum/09-interpretability-and-safety.md)

## What you implement (the tests check each property)
- Train a 2-layer attention-only transformer on repeated random sequences. Compute an `induction_score` per head (attention from position i to i − T + 1). Test: a layer-2 head scores > 0.5 (chance ≈ 1/T).
- `activation_patching(model, clean, corrupt, site)` with hooks. Test: patching the full residual stream at the last layer restores 100% of the clean logit difference.
- `SparseAutoencoder` (ReLU encoder, unit-norm decoder, MSE + λ‖f‖₁) trained on toy superposition data: 32 sparse features packed into 16 dimensions (then try 8 dims, and l1 = 0.01 vs 0.2, and watch recovery collapse). Test: most true feature directions are recovered with cosine > 0.9.

## GPU scale-up / stretch
Train an SAE on the residual stream of your TinyStories model from S1 and label its top 20 features by hand. Look for dead latents and feature splitting as the width changes.
