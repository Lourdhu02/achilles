# Lab 02 — The training toolkit

**Build:** initialization, LayerNorm/RMSNorm, masked cross-entropy, AdamW, Muon, cosine and WSD schedules,
gradient clipping, and gradient accumulation that is actually correct, all from tensor ops.
**Time:** 8–10 h · **Reads first:** [deep learning §2–6](../../curriculum/03-deep-learning.md#2-initialization-keeping-signals-alive)
**Run:** `pytest labs/02_training_core`

The tests compare against PyTorch. Your AdamW must match `torch.optim.AdamW` to 1e-10 in float64,
which means matching the order of operations, not just the idea.

---

## 1. Initialization is signal propagation

For `y = W x` with independent zero-mean entries, `Var(yᵢ) = n_in · Var(w) · Var(x)`.
ReLU zeroes half the mass, which halves the second moment. To keep activations O(1) through
depth, you need `Var(w) = 2 / n_in` (Kaiming). Anything else compounds geometrically:

| init | per-layer gain on the std (width 256) | after 30 layers |
|---|---|---|
| N(0, 1) | √(256/2) ≈ 11.3 | ~10³¹ (explodes) |
| N(0, 0.01²) | 0.01·√128 ≈ 0.11 | ~10⁻²⁹ (vanishes) |
| Kaiming N(0, 2/256) | 1.0 | O(1) |

`test_init_scale_decides_signal_propagation` shows exactly this. Transformers add one more rule.
The residual stream is a *sum* of 2L branch outputs, so GPT-2 scales each branch's output
projection by `1/√(2L)` to keep the stream's variance independent of depth
(implemented in [lab 05](../05_transformer/README.md)).

## 2. Normalization

- **LayerNorm:** `(x − μ)/√(σ² + ε) · γ + β` over features. Unlike BatchNorm it does not depend on the batch, so it works for batch size 1, variable-length sequences and autoregressive decoding.
- **RMSNorm:** drops the centering and β: `x/√(mean(x²) + ε) · γ`. It is cheaper, works as well in practice, and is the norm used in Llama, Qwen and Mistral.
- **Scale invariance:** `RMSNorm(10x) = RMSNorm(x)`. Weights that feed a norm have no preferred scale, so weight decay on them changes the *effective learning rate* rather than the function. This interaction is subtle and matters at scale.

## 3. Masked cross-entropy

SFT trains only on assistant tokens: prompt positions get `ignore_index = −100`, and the mean
divides by the **count of valid tokens**. Label smoothing mixes in a uniform target:
`(1−ε)·NLL + ε·mean_c(−log p_c)`.

## 4. AdamW, derived

```
m ← β₁m + (1−β₁)g            v ← β₂v + (1−β₂)g²
m̂ = m/(1−β₁ᵗ)                v̂ = v/(1−β₂ᵗ)          # bias correction: E[m_t] = (1−β₁ᵗ)·E[g]
θ ← θ(1 − ηλ) − η·m̂/(√v̂ + ε)                        # decay is DECOUPLED from the gradient
```

- **Bias correction** is needed because m and v start at 0. Without it, early steps are tiny. With it, the first step is exactly `η·sign(g)` whatever the gradient's scale (see `test_first_adam_step_has_magnitude_lr`). Adam is a *sign-like* optimizer with per-parameter trust.
- **Why decoupled:** in Adam with L2 regularization, the decay gradient `λθ` gets divided by `√v̂`, so parameters with large gradients are barely regularized. AdamW applies `θ(1−ηλ)` directly, which gives uniform shrinkage.
- **LLM defaults:** β₂ = 0.95 (not 0.999) so v adapts quickly after gradient spikes; λ = 0.1; ε = 1e-8. Memory is two fp32 states, **8 bytes/param**, which is half the "16 bytes/param" figure in [lab 03](../03_napkin_math/README.md).

## 5. Muon: steepest descent under the spectral norm

For a weight matrix, Muon takes the momentum `M = U S Vᵀ` and replaces it with `U Vᵀ`: every
singular direction gets the same step size. Rare but important directions stop being drowned out
by a few dominant ones. Newton–Schulz iterations compute `U Vᵀ` with a handful of matmuls (no
SVD), which is cheap on tensor cores. Rules of use:

- Use it only for 2-D hidden weights. Embeddings, the LM head, norms and biases stay on AdamW.
- It keeps one state (momentum), so **4 bytes/param** of optimizer state instead of 8.
- It takes fixed-size steps, like signSGD. In `test_muon_reduces_loss`, lr = 0.1 converges smoothly, while lr = 0.2–0.3 *oscillates* near the optimum. **Sign-like optimizers need learning-rate decay.**

Reference: Keller Jordan's Muon write-up (2024) and Moonshot's *Muon is Scalable for LLM Training* (2025).

## 6. Schedules

- **Warmup:** early on Adam's `v̂` estimates are noisy and the loss surface is sharp, so full-size steps can push the model into an unstable region it never recovers from. Warmup costs almost nothing and saves runs.
- **Cosine:** decay smoothly to ~10% of peak by the final step. You must know the step count in advance.
- **WSD (warmup–stable–decay):** hold the peak LR, then decay over the last 10–20%. You can branch several decay phases off one long stable run, which gives several "finished" models at different token counts for the price of one run. That makes it ideal for scaling-law sweeps ([lab 08](../08_scaling_laws/README.md)).

## 7. Clipping and accumulation

- **Global-norm clipping:** scale *all* grads by `min(1, c/‖g‖)` (c ≈ 1.0). Log the **pre-clip** norm every step; it is the best early warning of instability you have.
- **Accumulation bug:** summing per-micro-batch *mean* losses over-weights micro-batches with few valid tokens. Hugging Face Transformers shipped this bug until late 2024. The fix is to count valid tokens across all micro-batches first, then backprop `sum_loss / total_tokens`. `test_accumulation_equals_full_batch_with_uneven_masks` checks it exactly.

## 8. Mixed precision, in napkin math

bf16 has fp32's 8 exponent bits, so its range matches and gradients never underflow, but it has
only 8 bits of precision: `1 + 1e-3` rounds back to `1`. A typical update is 1e-4 relative to its
weight, so keeping the weights in bf16 silently discards training. Hence **fp32 master weights**.
fp16 has more mantissa but 5 exponent bits: gradients below ~6e-8 flush to zero, hence **loss
scaling**. Try both in a REPL: `torch.tensor(1.0, dtype=torch.bfloat16) + 1e-3`.

---

## What to implement

1. `kaiming_normal_`, `xavier_uniform_` → init tests
2. `LayerNorm.forward`, `RMSNorm.forward`
3. `cross_entropy` (ignore_index, label smoothing, three reductions; the gradient must match too)
4. `AdamW.step` → bit-exact trajectory test
5. `newton_schulz`, `Muon.step`
6. `lr_cosine`, `lr_wsd`
7. `clip_grad_norm_`, `accumulate_gradients`

## Check yourself

1. Your fresh 50k-vocab LM starts at loss 10.8. Is that a bug? What should it be?
2. Why does a warmup of 2,000 steps barely matter for SGD but matter a lot for Adam?
3. You double the batch size. What should you do to the Adam learning rate, and why is there no simple rule?
4. Loss is flat, gradient norm is 1e-9 at every layer. List three causes.
5. Gradient norm spikes 50× for a few steps and then the loss spikes. What would you log to find the cause?

<details><summary>Answers</summary>

1. A uniform predictor has loss ln(50,000) ≈ 10.8, so this is correct. If you saw 30, your output logits would be too large at init (for example no final norm, or a large LM-head init).
2. SGD's step is proportional to the gradient, which is small in flat regions. Adam normalizes the step to ~η per parameter from step 1, even where its curvature estimates are garbage.
3. For SGD the linear scaling rule is a decent start. For Adam, scaling by √k is more common, but the correct answer depends on where you are relative to the *critical batch size* (the gradient noise scale). Past it, larger batches stop helping. Measure it (McCandlish et al., 2018).
4. Detached graph (`.detach()`, `torch.no_grad()`, a `.data` assignment); dead ReLUs or saturated tanh; a loss multiplied by 0 (a mask that is all ignore_index); LR of 0 from a schedule bug; parameters not passed to the optimizer.
5. Per-layer gradient norms, max attention logit, max output logit, update-to-weight ratios, and the data batch indices. Spikes often come from one bad batch or from attention logits growing without bound. QK-norm and z-loss are the usual fixes ([curriculum 05](../../curriculum/05-pretraining.md#6-stability-at-scale)).
</details>

## Stretch

- Implement Lion (`sign(β₁m + (1−β₁)g)`) and Adafactor (factored second moments). Compare their optimizer-state memory with AdamW's.
- Train [lab 05](../05_transformer/README.md)'s GPT with AdamW, then with Muon on the hidden matrices plus AdamW elsewhere. Compare loss at equal steps and equal wall-clock time.
- μP: train widths 64/128/256/512 with a fixed LR and plot the loss. Then apply μP scaling and show the optimal LR stops moving (Tensor Programs V).
