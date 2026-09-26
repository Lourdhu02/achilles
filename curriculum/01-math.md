# 01 — Math you actually use

Fluency in the handful of identities that modern ML rests on: matmul and SVD, vector–Jacobian products, KL and the log-derivative trick, the convergence of gradient descent, and the error bars on a benchmark score. Every identity here is derived step by step and checked numerically.
Assumes undergraduate calculus, linear algebra and probability. Mastery target: derive the backward pass of any layer on a whiteboard, and derive PPO/DPO/GRPO from their objectives.

Lab: [01 autograd](../labs/01_autograd/README.md). Used everywhere after.

**Contents:** [1 Linear algebra](#1-linear-algebra) · [2 Matrix calculus](#2-matrix-calculus-the-only-calculus-you-need) · [3 Probability and information](#3-probability-and-information) · [4 Optimization](#4-optimization) · [5 Statistics](#5-statistics-for-ml) · [Interview traps](#interview-traps) · [CPU vs GPU notes](#cpu-vs-gpu-notes) · [Check yourself](#check-yourself) · [Visual guides](#visual-guides) · [Read next](#read-next)

---

## 1. Linear algebra

- **Matmul is the unit of compute.** `(m×k)(k×n)` costs $2mkn$ FLOPs: $mn$ outputs, each a length-$k$ dot product of $k$ multiplies and $k$ adds. Everything in a transformer is either a matmul or a memory-bound elementwise op ([02](02-compute-and-hardware.md)).
- **SVD** $A = U\Sigma V^\top$. The best rank-$r$ approximation in Frobenius or spectral norm keeps the top $r$ singular values (Eckart–Young), with error $\sqrt{\sum_{i>r}\sigma_i^2}$ in Frobenius norm. LoRA bets that fine-tuning updates are approximately low rank ([lab 10](../labs/10_lora/README.md) tests that bet). Muon replaces a gradient $G = U\Sigma V^\top$ with $UV^\top$, giving every singular direction the same step size ([lab 02](../labs/02_training_core/README.md)).
- **Norms.** Frobenius $\|A\|_F = \sqrt{\sum a_{ij}^2} = \sqrt{\sum\sigma_i^2}$; spectral $\|A\|_2 = \sigma_{\max}$, the most a layer can stretch any input: $\|Ax\| \le \|A\|_2\|x\|$. Always $\|A\|_2 \le \|A\|_F \le \sqrt{r}\,\|A\|_2$. Spectral norms appear in stability arguments, μP and Muon; Frobenius norms are cheap and are what `clip_grad_norm_` uses.
- **High-dimensional geometry.** Two random unit vectors in $d$ dimensions have dot product with mean 0 and variance $1/d$: at $d = 4096$ the standard deviation is $1/64 \approx 0.016$, so they are nearly orthogonal. A $d$-dimensional space holds exponentially many *almost*-orthogonal directions, which is the geometric basis of superposition ([09](09-interpretability-and-safety.md)). For vectors with unit-variance entries (not unit length), the dot product has variance $d$, hence attention's $1/\sqrt{d}$.
- **Eigen vs singular.** Eigenvalues describe repeated application of a square map (they govern gradient descent on a quadratic, §4); singular values describe one application of any map (they govern how much a layer stretches activations). For symmetric PSD matrices, such as a Hessian at a minimum or a covariance, the two coincide.

> [!TIP]
> Before trusting any matrix identity, check it with `np.random.randn` inputs of *different* sizes (say 3×5 and 5×2). Square test matrices hide transpose mistakes because the shapes happen to line up.

## 2. Matrix calculus: the only calculus you need

Backpropagation never forms a Jacobian. It needs **vector–Jacobian products (VJPs)**: given the upstream gradient $\bar{y} = \partial L/\partial y$ (same shape as $y$), compute $\bar{x} = \partial L/\partial x$ (same shape as $x$). The tool is the **differential with the trace trick**: for scalar $L$,
$$dL = \langle \bar{y}, dy\rangle = \mathrm{tr}(\bar{y}^\top dy),$$
so write $dy$ in terms of $dx$, rearrange into the form $\mathrm{tr}(\,\cdot^\top dx)$, and read off $\bar{x}$.

### 2.1 Matmul, derived

$Y = XW$ with $X\in\mathbb{R}^{B\times n}$, $W\in\mathbb{R}^{n\times m}$. Then $dY = dX\,W + X\,dW$ and
$$dL = \mathrm{tr}(\bar{Y}^\top dX\,W) + \mathrm{tr}(\bar{Y}^\top X\,dW) = \mathrm{tr}\big((\bar{Y}W^\top)^\top dX\big) + \mathrm{tr}\big((X^\top\bar{Y})^\top dW\big),$$
using the cyclic property $\mathrm{tr}(ABC) = \mathrm{tr}(CAB)$. So $\bar{X} = \bar{Y}W^\top$ and $\bar{W} = X^\top\bar{Y}$.

**Shape check:** $\bar{X}$ must be $B\times n$; $\bar{Y}$ is $B\times m$ and $W^\top$ is $m\times n$. There is exactly one way to combine $\bar{Y}$, $X$, $W$ into each required shape, which is why you can "derive" matmul VJPs by shape-matching in an interview. Say so, then show the trace derivation anyway.

### 2.2 Softmax Jacobian, derived

$y_i = e^{z_i}/\sum_k e^{z_k}$. Differentiate the log: $\log y_i = z_i - \log\sum_k e^{z_k}$, so $\partial\log y_i/\partial z_j = \delta_{ij} - y_j$, and
$$\frac{\partial y_i}{\partial z_j} = y_i(\delta_{ij} - y_j), \qquad J = \mathrm{diag}(y) - yy^\top.$$

Properties you can check: $J$ is symmetric, its rows sum to zero ($J\mathbf{1} = y - y(y^\top\mathbf{1}) = 0$, because adding a constant to all logits does not change the softmax), it is positive semidefinite with rank at most $n-1$, and it goes to zero when $y$ is one-hot (a saturated softmax passes no gradient).

**VJP:** $\bar{z} = J^\top\bar{y} = y\odot\bar{y} - y\,(y^\top\bar{y}) = y\odot(\bar{y} - \langle\bar{y}, y\rangle)$. Cost $O(n)$, never $O(n^2)$.

**Worked numbers.** $z = (2, 1, 0.1)$ gives $y = (0.659, 0.242, 0.099)$ and
$$J = \begin{pmatrix} 0.225 & -0.160 & -0.065\\ -0.160 & 0.184 & -0.024\\ -0.065 & -0.024 & 0.089\end{pmatrix},$$
whose rows sum to 0.

### 2.3 Softmax cross-entropy, in three lines

$L = -\log y_t = -z_t + \log\sum_k e^{z_k}$. Then $\partial L/\partial z_j = -\delta_{jt} + y_j$, so
$$\bar{z} = \mathrm{softmax}(z) - \mathrm{onehot}(t).$$
With the numbers above and $t = 0$: $L = 0.417$ and $\bar{z} = (-0.341, 0.242, 0.099)$, which sums to zero. For a batch mean over $N$ rows, divide by $N$. This is the most-asked "show me calculus" interview question; the answer should take under a minute, and the fused form is also why frameworks compute `log_softmax` rather than `log(softmax)`.

### 2.4 LayerNorm backward, derived

Over one row of $N$ features (no affine): $\mu = \frac1N\sum x_i$, $\sigma = \sqrt{\frac1N\sum(x_i-\mu)^2 + \epsilon}$, $\hat{x}_i = (x_i - \mu)/\sigma$.

1. $\partial\mu/\partial x_j = 1/N$.
2. $\partial\sigma/\partial x_j = \frac{1}{2\sigma}\cdot\frac{2}{N}(x_j - \mu) = \hat{x}_j/N$ (the $\sum(x_i-\mu)$ term from differentiating $\mu$ vanishes).
3. Quotient rule: $\dfrac{\partial\hat{x}_i}{\partial x_j} = \dfrac{\delta_{ij} - 1/N}{\sigma} - \dfrac{x_i - \mu}{\sigma^2}\cdot\dfrac{\hat{x}_j}{N} = \dfrac{1}{\sigma}\Big(\delta_{ij} - \dfrac1N - \dfrac{\hat{x}_i\hat{x}_j}{N}\Big)$.
4. Contract with $\bar{y}$: 
$$\bar{x}_j = \sum_i\bar{y}_i\frac{\partial\hat{x}_i}{\partial x_j} = \frac{1}{\sigma}\Big(\bar{y}_j - \mathrm{mean}(\bar{y}) - \hat{x}_j\,\mathrm{mean}(\bar{y}\odot\hat{x})\Big).$$

With the affine part $y = \gamma\odot\hat{x} + \beta$: $\bar\beta = \sum_\text{rows}\bar{y}$, $\bar\gamma = \sum_\text{rows}\bar{y}\odot\hat{x}$, and use $\bar{y}\odot\gamma$ in place of $\bar{y}$ in step 4.

**Interpretation:** the gradient is $\bar{y}$ with two components projected out: its mean (LayerNorm ignores shifts of $x$) and its component along $\hat{x}$ (it ignores rescaling of $x$). So $\sum_j\bar{x}_j = 0$ exactly, and $\sum_j\bar{x}_j\hat{x}_j \approx 0$ (exactly 0 when $\epsilon = 0$). These are two free unit tests.

**RMSNorm** ($r = \sqrt{\mathrm{mean}(x^2) + \epsilon}$, $\hat{x} = x/r$) drops the centering, so only one projection remains: $\bar{x} = (\bar{y} - \hat{x}\,\mathrm{mean}(\bar{y}\odot\hat{x}))/r$.

Check it (runs on CPU in a second; both errors print around 1e-16):
```python
import torch
torch.manual_seed(0)
x = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
gy, eps = torch.randn(4, 16, dtype=torch.float64), 1e-5
torch.nn.functional.layer_norm(x, (16,), eps=eps).backward(gy)
xh = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
sigma = torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
gx = (gy - gy.mean(-1, keepdim=True) - xh * (gy * xh).mean(-1, keepdim=True)) / sigma
print((gx - x.grad).abs().max().item(), gx.sum(-1).abs().max().item())
```

### 2.5 The VJP table

| op | VJP (given $\bar{y}$) | derivation hint |
|---|---|---|
| `Y = XW` | `X̄ = ȲWᵀ`, `W̄ = XᵀȲ` | §2.1 |
| elementwise `y = f(x)` | `x̄ = ȳ ⊙ f′(x)` | diagonal Jacobian |
| broadcast / sum | sum ↔ broadcast are each other's VJP | a copied value's gradient is the sum over copies |
| gather `y = x[idx]` | scatter-**add** `ȳ` into zeros | repeated indices accumulate |
| `y = softmax(z)` | `z̄ = y ⊙ (ȳ − ⟨ȳ, y⟩)` | §2.2 |
| `L = −log softmax(z)_t` | `z̄ = softmax(z) − onehot(t)` | §2.3 |
| `y = LN(x)` (no affine), `x̂ = (x−μ)/σ` | `x̄ = (ȳ − mean(ȳ) − x̂·mean(ȳ⊙x̂)) / σ` | §2.4 |
| `y = RMSNorm(x)` (no weight) | `x̄ = (ȳ − x̂·mean(ȳ⊙x̂)) / r` | §2.4 |
| attention `P = softmax(S)`, `O = PV` | `V̄ = PᵀŌ`, `P̄ = ŌVᵀ`, `S̄ = P ⊙ (P̄ − rowsum(Ō⊙O))` | below |

**The attention row, and why FlashAttention can use it.** Row by row, the softmax VJP gives $\bar{S}_i = P_i\odot(\bar{P}_i - \langle\bar{P}_i, P_i\rangle)$. The inner product needs the full row of $\bar{P}$, which FlashAttention never materializes. But
$$\langle\bar{P}_i, P_i\rangle = \sum_j P_{ij}\sum_k\bar{O}_{ik}V_{jk} = \sum_k\bar{O}_{ik}\sum_j P_{ij}V_{jk} = \sum_k\bar{O}_{ik}O_{ik} = \langle\bar{O}_i, O_i\rangle,$$
a length-$d$ dot product of two things already in memory. That scalar per row ($D_i$ in the FlashAttention-2 paper) is the whole trick ([lab 06](../labs/06_attention_kernels/README.md)).

Derive each row once on paper, then check it with finite differences in [lab 01](../labs/01_autograd/README.md).

### 2.6 Gradient checking, and choosing ε

Central differences $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$ have truncation error $O(\epsilon^2)$ and rounding error about $u/\epsilon$, where $u$ is the unit roundoff. The total is minimized near $\epsilon \approx u^{1/3}$.

| dtype | $u$ | best ε | best achievable error (d/dx sin at x = 1, measured) |
|---|---|---|---|
| float64 | 1.1e-16 | ~1e-5 to 1e-6 | ~3e-11 at ε = 1e-6 |
| float32 | 6e-8 | ~1e-2 to 1e-3 | ~1e-5 at ε = 1e-2 |

So gradient checks belong in float64: in float32 you cannot tell a subtle bug from rounding. Check a VJP with a *random* upstream gradient $G$, comparing against finite differences of $\sum f(x)\odot G$; that tests every row of the Jacobian at once.

## 3. Probability and information

- **MLE = minimizing cross-entropy = minimizing forward KL to the data.** $\mathbb{E}_{x\sim p}[-\log q_\theta(x)] = H(p) + \mathrm{KL}(p\|q_\theta)$ and $H(p)$ does not depend on $\theta$. That is all pretraining is. Loss in nats per token; divide by $\ln 2$ for bits.
- **KL is asymmetric.** Forward $\mathrm{KL}(p\|q)$ is mode-covering ($q$ is punished wherever $p > 0$ and $q \approx 0$); reverse $\mathrm{KL}(q\|p)$ is mode-seeking. Worked: $p = (0.5, 0.5)$, $q = (0.9, 0.1)$ gives $\mathrm{KL}(p\|q) = 0.511$ nats but $\mathrm{KL}(q\|p) = 0.368$. RLHF's KL penalty is reverse KL from the policy to the reference; SFT and pretraining are forward KL; on-policy distillation uses reverse KL.
- **Log-derivative trick.** $\nabla_\theta\mathbb{E}_{x\sim p_\theta}[f(x)] = \sum_x f(x)\nabla_\theta p_\theta(x) = \sum_x f(x)p_\theta(x)\nabla_\theta\log p_\theta(x) = \mathbb{E}[f(x)\nabla_\theta\log p_\theta(x)]$. It needs samples and $\log p$, not a differentiable $f$: the basis of REINFORCE, PPO and GRPO.
- **Baselines are free.** $\mathbb{E}[\nabla\log p_\theta] = \sum_x\nabla p_\theta(x) = \nabla\sum_x p_\theta(x) = \nabla 1 = 0$, so subtracting any $b$ that does not depend on the sampled $x$ leaves the gradient unbiased and can cut its variance enormously. GRPO's group-mean reward is such a baseline (it depends on the other samples, which introduces a small bias that vanishes as the group grows).
- **Importance sampling:** $\mathbb{E}_p[f] = \mathbb{E}_q[f\,p/q]$. PPO's ratio $\pi_\text{new}/\pi_\text{old}$ and speculative decoding's acceptance rule $\min(1, p/q)$ are both importance-sampling ideas. Variance explodes when $q$ puts little mass where $p$ is large, which is why PPO clips the ratio.
- **Bradley–Terry:** $P(a\succ b) = \sigma(r_a - r_b)$. Reward models, DPO and arena leaderboards all use it.
- **KL estimators** from samples $x\sim q$, with $\rho = p(x)/q(x)$: $k_1 = -\log\rho$ (unbiased for $\mathrm{KL}(q\|p)$, high variance, can be negative); $k_3 = \rho - 1 - \log\rho$ (unbiased because $\mathbb{E}_q[\rho] = 1$, always $\ge 0$, lower variance; used in GRPO).

## 4. Optimization

**Gradient descent on a quadratic** $f(x) = \frac12 x^\top Hx$ with Hessian eigenvalues in $[\mu, L]$: each eigen-direction evolves as $x_i \leftarrow (1-\eta\lambda_i)x_i$. It converges iff $|1-\eta\lambda_i| < 1$ for all $i$, i.e. $\eta < 2/L$. The best fixed step $\eta = 2/(L+\mu)$ gives a contraction of $(\kappa-1)/(\kappa+1)$ per step, with condition number $\kappa = L/\mu$. Heavy-ball momentum with tuned parameters improves that to $(\sqrt\kappa-1)/(\sqrt\kappa+1)$.

**Worked numbers, $\kappa = 100$:** plain GD contracts by 0.980 per step, about 50 steps per factor of $e$; momentum contracts by 0.818, about 5 steps per factor of $e$. That $\sqrt{\kappa}$ speed-up is why every practical optimizer has momentum.

- **Preconditioning** changes the effective $\kappa$. Adam approximates a diagonal preconditioner; Shampoo and SOAP use Kronecker-factored ones; Muon orthogonalizes the update so all singular directions of a weight matrix move equally.
- **Edge of stability.** In practice, full-batch GD on neural networks drives the sharpness (top Hessian eigenvalue) up until it hovers near $2/\eta$ and the loss keeps decreasing non-monotonically (Cohen et al., 2021, [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)). The quadratic picture predicts the threshold; networks adapt to it.
- **Warmup** exists because early curvature is high and Adam's second-moment estimates are unreliable ([03 §5](03-deep-learning.md#5-schedules-and-batch-size)).
- **Noise and batch size.** SGD noise scales roughly with $\eta/B$. Past the *critical batch size* (McCandlish et al., 2018, [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)), larger batches give almost no reduction in steps and only waste compute.

## 5. Statistics for ML

Standard error of a mean: $\sigma/\sqrt{n}$. For accuracy $p$ on $n$ items, $\mathrm{SE} = \sqrt{p(1-p)/n}$. **Worked:** $p = 0.70$, $n = 1{,}000$ gives $\mathrm{SE} = 0.0145$, so the 95% interval is $\pm 1.96\cdot 0.0145 = \pm 2.8$ points. A "+1.5 points" claim on that benchmark is noise unless the test is paired.

- **Paired tests** on the same items are far more powerful than comparing two means: most per-item variance is shared by both models and cancels in the difference.
- **Bootstrap** when there is no formula (pass@k, medians, ratios): resample items with replacement, recompute, take percentiles.
- **Clustered items** (several questions per document) need clustered standard errors, or the interval is too narrow.
- **Seeds are samples too.** Report the mean and spread over seeds for training results, not the best seed.

Full treatment: [08 §2](08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen), [lab 15](../labs/15_eval_stats/README.md).

---

## Interview traps

- **"Backprop computes the Jacobian."** It computes VJPs. For a 4096×4096 matmul the Jacobian with respect to the weights alone would have $4096^2$ columns per output element; nobody forms it.
- **Transposes by guesswork.** Say "the gradient has the shape of the variable" and derive with the trace trick; do not pattern-match on square matrices.
- **Forgetting the mean over the batch** in the cross-entropy gradient (the $1/N$), or dividing twice.
- **Softmax Jacobian as `y(1−y)`.** That is only the diagonal; the off-diagonal terms $-y_iy_j$ are what make rows sum to zero.
- **"KL is a distance."** It is not symmetric and does not satisfy the triangle inequality. Know which direction each method uses and why.
- **"The baseline reduces bias."** It reduces variance; the estimator is unbiased with or without it (if $b$ does not depend on the sampled action).
- **Gradient checks in float32** that "almost" pass. Switch to float64 before concluding anything.
- **Confusing $\sigma/\sqrt{n}$ with $\sigma$** when reading an error bar: ask whether a plot shows standard deviation, standard error or a confidence interval.

## CPU vs GPU notes

- Everything in this module is checked on a CPU in float64, which is the right place for correctness checks. `torch.autograd.gradcheck` requires double precision for the same reason as §2.6.
- GPUs default to lower precision (TF32 matmuls on Ampere and later when enabled, bf16 under autocast). A VJP that passes in float64 on CPU can show 1e-3 relative differences on GPU; that is precision, not a bug. Compare against a float64 CPU reference, not against another GPU run.
- GPU reductions (sums in `index_add_`, atomics in scatter-add) are not always deterministic. Run-to-run differences at the 1e-7 level in float32 are expected; set `torch.use_deterministic_algorithms(True)` when you need bitwise reproducibility.

## Check yourself

<details><summary>1. Derive ∂L/∂z for softmax cross-entropy in three lines.</summary>

$L = -z_t + \log\sum_k e^{z_k}$. $\partial L/\partial z_j = -\delta_{jt} + e^{z_j}/\sum_k e^{z_k}$. So $\bar{z} = \mathrm{softmax}(z) - \mathrm{onehot}(t)$ (divide by $N$ for a batch mean).
</details>

<details><summary>2. Derive the VJP of Y = XW with the trace trick, and check the shapes.</summary>

$dL = \mathrm{tr}(\bar{Y}^\top(dX\,W + X\,dW)) = \mathrm{tr}((\bar{Y}W^\top)^\top dX) + \mathrm{tr}((X^\top\bar{Y})^\top dW)$. So $\bar{X} = \bar{Y}W^\top$ ($B\times m$ times $m\times n$ = $B\times n$, the shape of $X$) and $\bar{W} = X^\top\bar{Y}$ ($n\times B$ times $B\times m$ = $n\times m$, the shape of $W$).
</details>

<details><summary>3. Why do the rows of the softmax Jacobian sum to zero, and what does that imply for z̄?</summary>

Adding a constant $c$ to every logit leaves the softmax unchanged, so the directional derivative along $\mathbf{1}$ is zero: $J\mathbf{1} = 0$. Since $J$ is symmetric, $\mathbf{1}^\top\bar{z} = \mathbf{1}^\top J\bar{y} = 0$: the logit gradient always sums to zero. The cross-entropy gradient $y - \mathrm{onehot}(t)$ sums to $1 - 1 = 0$, consistent with this.
</details>

<details><summary>4. State two exact properties of the LayerNorm input gradient you can use as unit tests.</summary>

$\sum_j\bar{x}_j = 0$ (LayerNorm is invariant to shifting $x$) and $\sum_j\bar{x}_j\hat{x}_j = 0$ when $\epsilon = 0$ (it is invariant to scaling $x$). RMSNorm keeps only the second.
</details>

<details><summary>5. Why can FlashAttention's backward avoid materializing the T×T matrix dP?</summary>

The softmax VJP needs $\langle\bar{P}_i, P_i\rangle$ per row, and that equals $\langle\bar{O}_i, O_i\rangle$ (swap the sums: $\sum_j P_{ij}V_j = O_i$). One dot product of length $d$ per row, computed before the backward loop, replaces a full row of $\bar{P}$.
</details>

<details><summary>6. Why does the REINFORCE baseline not bias the gradient?</summary>

$\mathbb{E}_{x\sim p_\theta}[b\,\nabla\log p_\theta(x)] = b\sum_x\nabla p_\theta(x) = b\,\nabla\sum_x p_\theta(x) = b\,\nabla 1 = 0$, for any $b$ that does not depend on the sampled $x$.
</details>

<details><summary>7. Why is the RLHF KL penalty estimated per token from samples instead of computed exactly?</summary>

The exact KL between two sequence distributions sums over all possible sequences, which is exponential in length. Per token, the exact KL over the vocabulary is possible but costs a full-vocabulary pass for both models at every position; with samples already drawn from the policy, the log-ratio (k1) or k3 on the sampled tokens is an unbiased estimate at almost no extra cost.
</details>

<details><summary>8. Gradient descent on a quadratic with L = 10, μ = 0.1. Largest stable step, best step, and roughly how many steps to shrink the error by 1,000×?</summary>

Stable iff $\eta < 2/L = 0.2$. Best $\eta = 2/(L+\mu) \approx 0.198$. $\kappa = 100$, contraction 0.980 per step, so $\ln(1000)/0.0198 \approx 350$ steps. With tuned momentum (0.818 per step): about 35.
</details>

<details><summary>9. Model A scores 71.2% and model B 70.0% on 1,000 items. Is A better?</summary>

Unpaired, the 95% interval on each score is about ±2.8 points, so the difference is not established. Paired (same items), compute the per-item difference and its standard error; that depends on how often the two models disagree and can be much smaller. Report the paired interval, not two overlapping bars.
</details>

<details><summary>10. Show that DPO's optimal policy is π_ref·exp(r/β)/Z.</summary>

Maximize $\mathbb{E}_{\pi}[r] - \beta\,\mathrm{KL}(\pi\|\pi_\text{ref})$ per prompt. Rewrite as $-\beta\,\mathrm{KL}(\pi\,\|\,\pi_\text{ref}e^{r/\beta}/Z) + \beta\log Z$, which is maximized when the KL is zero. Full derivation in [06 §4](06-post-training.md#4-dpo-and-its-family).
</details>

## Visual guides

- 3Blue1Brown, [Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra): matrices as transformations, determinants, eigenvectors. Watch before §1 if SVD feels abstract.
- 3Blue1Brown, [Neural networks](https://www.3blue1brown.com/topics/neural-networks): the backpropagation chapters draw the chain rule on a computational graph.
- Christopher Olah, [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/): why reverse mode wins, in pictures.
- Gabriel Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/) (distill.pub): interactive version of §4, including the $\sqrt\kappa$ speed-up.

More per-topic visuals are collected in the [library](../library/README.md).

## Read next

- [02 Compute and hardware](02-compute-and-hardware.md): turning matmul FLOPs into seconds and bytes.
- [03 Deep learning](03-deep-learning.md): the same VJPs inside autodiff, initialization and optimizers.
- [Lab 01](../labs/01_autograd/README.md): implement the VJP table and check every row with finite differences.
- References: Parr & Howard, [*The Matrix Calculus You Need for Deep Learning*](https://explained.ai/matrix-calculus/); Deisenroth et al., [*Mathematics for Machine Learning*](https://mml-book.github.io/) (free); Petersen & Pedersen, *The Matrix Cookbook*; Stanford CS109 for probability.
