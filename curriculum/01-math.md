# 01 — Math you actually use

Assumes undergraduate comfort. The goal is fluency in the handful of identities that modern ML rests on, each checked numerically.

## 1. Linear algebra
- **Matmul as the unit of compute:** `(m×k)(k×n)` costs 2mkn FLOPs. Everything in a transformer is a matmul or a memory-bound elementwise op.
- **SVD** `A = UΣVᵀ`: the best rank-r approximation keeps the top r singular values (Eckart–Young). LoRA bets that fine-tuning updates are approximately low rank. Muon replaces a gradient with `UVᵀ`, giving every direction equal step size.
- **Norms:** Frobenius (sum of squares) vs spectral (largest singular value). Spectral norms bound how much a layer can stretch inputs, which is why they appear in stability and μP arguments.
- **High-dimensional geometry:** random unit vectors in d dims have dot product ~N(0, 1/d), so they are nearly orthogonal. A d-dim space holds exponentially many *almost*-orthogonal directions, which is the basis of superposition ([09](09-interpretability-and-safety.md)). The dot product of two random vectors with unit-variance entries has variance d, hence attention's 1/√d.

## 2. Matrix calculus: the only calculus you need
Work with **VJPs**: given `ḡ = ∂L/∂y`, compute `∂L/∂x`. Use the trace trick `dL = tr(ḡᵀ dy)`.

| op | VJP |
|---|---|
| `Y = XW` | `X̄ = ȲWᵀ`, `W̄ = XᵀȲ` |
| `y = softmax(z)` | `z̄ = y ⊙ (ȳ − ⟨ȳ, y⟩)` |
| `L = −log softmax(z)_t` | `z̄ = softmax(z) − onehot(t)` |
| `y = LN(x)` (no affine), `x̂ = (x−μ)/σ` | `x̄ = (ȳ − mean(ȳ) − x̂·mean(ȳ⊙x̂)) / σ` |
| attention `P = softmax(S)`, `O = PV` | `V̄ = PᵀŌ`, `P̄ = ŌVᵀ`, `S̄ = P ⊙ (P̄ − rowsum(Ō⊙O))` |

Derive each one once on paper, then check it with finite differences ([lab 01](../labs/01_autograd/README.md)). The last row is the FlashAttention backward trick ([lab 06](../labs/06_attention_kernels/README.md)).

## 3. Probability and information
- **MLE = minimizing cross-entropy = minimizing forward KL** to the data distribution. That is all pretraining is.
- **KL is asymmetric.** Forward `KL(p‖q)` is mode-covering (q must put mass wherever p does); reverse `KL(q‖p)` is mode-seeking. RLHF's KL penalty is reverse KL to the reference model; on-policy distillation uses reverse KL; SFT is forward KL.
- **Log-derivative trick:** `∇θ E_{x∼pθ}[f(x)] = E[f(x) ∇θ log pθ(x)]`, the basis of REINFORCE, PPO and GRPO. Subtracting a baseline `b` keeps the estimate unbiased (`E[∇ log p] = 0`) and cuts variance.
- **Importance sampling:** `E_p[f] = E_q[f·p/q]`. PPO's ratio `π_new/π_old` and speculative decoding's acceptance rule `min(1, p/q)` are both importance-sampling ideas.
- **Bradley–Terry:** `P(a ≻ b) = σ(r_a − r_b)`. Reward models, DPO and arena leaderboards all use it.
- **KL estimators** from samples x ∼ q, with `ρ = p(x)/q(x)`: k1 = −log ρ (unbiased, high variance); k3 = ρ − 1 − log ρ (unbiased, always ≥ 0, used in GRPO).

## 4. Optimization
- Gradient descent on a quadratic with Hessian eigenvalues in [μ, L] converges if `η < 2/L`; speed depends on the condition number L/μ. Adaptive methods (Adam) approximate a diagonal preconditioner; Shampoo and SOAP use Kronecker factors; Muon orthogonalizes the update.
- Momentum damps oscillation along high-curvature directions. Warmup exists because early curvature is high and Adam's statistics are unreliable.
- SGD noise scales roughly with η/B. Past the *critical batch size*, larger batches only waste compute.

## 5. Statistics for ML
Standard error `σ/√n`. For accuracy p on n items: `SE = √(p(1−p)/n)`, so on 1,000 items near 70% the 95% CI is ±2.8 points. Most "improvements" on small benchmarks are noise. Paired tests on the same items are far more powerful than comparing two means. The bootstrap works when formulas don't. Full treatment: [08 §2](08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen), [lab 15](../labs/15_eval_stats/README.md).

## Check yourself
1. Derive `∂L/∂z` for softmax cross-entropy in 3 lines. 2. Why is the KL penalty in RLHF estimated per token from samples rather than computed exactly? 3. Why does the REINFORCE baseline not bias the gradient? 4. Show that DPO's optimal policy is `π_ref·exp(r/β)/Z` ([06 §4](06-post-training.md#4-dpo-and-its-family)).

**Resources:** *Mathematics for Machine Learning* (Deisenroth et al., free); 3Blue1Brown; *The Matrix Cookbook*; Stanford CS109.
