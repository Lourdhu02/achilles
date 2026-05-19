# Math Essentials for ML Interviews

Just enough math to handle interview questions. Not a complete reference.

If you find yourself rusty on basics: spend 1 week (Month 1 or 2) on a refresher. Recommendations at bottom.

---

## Linear Algebra

### Vectors and matrices
- **Vector:** ordered list of numbers. Has dimension.
- **Matrix:** 2D array. Shape (rows, cols).
- **Dot product:** `a·b = Σ a_i b_i = |a||b|cos(θ)`. Captures "similarity".
- **Matrix multiplication:** `(AB)_{ij} = Σ_k A_{ik} B_{kj}`. Shapes: `(m,k) × (k,n) = (m,n)`.

### Key concepts
- **Norm:** `||x||_p = (Σ|x_i|^p)^(1/p)`. L2 = Euclidean.
- **Cosine similarity:** `a·b / (||a||·||b||)`. Range [-1, 1].
- **Orthogonal:** `a·b = 0`.
- **Linear independence:** none of the vectors can be written as a combination of the others.
- **Rank:** number of linearly independent rows/columns.

### Eigenvalues and eigenvectors
- For square matrix A: `Av = λv`
- v: eigenvector, λ: eigenvalue
- Matrix has at most n eigenvalues (n = size)
- For symmetric matrices: eigenvalues are real, eigenvectors are orthogonal
- Applications: PCA (eigenvectors of covariance matrix = principal components)

### SVD (Singular Value Decomposition)
- `A = UΣV^T` where U, V orthogonal, Σ diagonal with non-negative entries
- Generalization of eigendecomposition to non-square / non-symmetric
- Applications: PCA, recommender systems, low-rank approximation (LoRA!)

### Matrix calculus (you need this for backprop)
- ∂(Ax)/∂x = A^T
- ∂(x^T Ax)/∂x = (A + A^T)x
- ∂(||x||²)/∂x = 2x

### Practical questions you might get
- "Walk through how matrix multiplication is implemented and why GPU-parallelization is so effective." → it's outer-product sums; each output element is independent → embarrassingly parallel.
- "What is a positive semi-definite matrix? Why does it appear in covariance matrices?" → x^T M x ≥ 0 for all x; covariance is by construction PSD.
- "Explain PCA mathematically." → center data, compute cov matrix, take top-k eigenvectors, project.

---

## Probability and Statistics

### Probability basics
- **Probability:** P(A) ∈ [0, 1]
- **Joint:** P(A, B)
- **Conditional:** P(A|B) = P(A, B) / P(B)
- **Independence:** P(A, B) = P(A) · P(B)
- **Bayes:** P(A|B) = P(B|A) · P(A) / P(B)

### Random variables
- **Discrete:** PMF, e.g., Bernoulli, Binomial, Poisson
- **Continuous:** PDF, e.g., Gaussian, Exponential, Beta
- **Expectation:** E[X] = Σ x·P(x) (discrete) or ∫ x·p(x) dx (continuous)
- **Variance:** Var(X) = E[(X - E[X])²]
- **Covariance:** Cov(X, Y) = E[(X - E[X])(Y - E[Y])]

### Gaussian distribution
- `p(x) = (1/√(2πσ²)) exp(-(x - μ)² / (2σ²))`
- Multivariate: `p(x) ∝ exp(-(1/2)(x-μ)^T Σ^(-1) (x-μ))`
- Why it appears everywhere: Central Limit Theorem

### Central Limit Theorem
- Sum/mean of large number of i.i.d. random variables → Gaussian distribution
- Why: explains why noise is often Gaussian; why MSE works

### KL Divergence
- `KL(p || q) = Σ p(x) log(p(x)/q(x))`
- Measures "how different" two distributions are
- NOT symmetric: KL(p||q) ≠ KL(q||p)
- KL ≥ 0; = 0 iff p = q
- Appears in: VAEs, variational inference, distillation, RLHF

### Cross-entropy
- H(p, q) = -Σ p(x) log q(x)
- Cross-entropy = H(p) + KL(p || q)
- For one-hot p, equals -log(predicted_prob_of_correct_class)
- Why we use it as a loss: equivalent to maximizing log-likelihood

### Information / Entropy
- H(p) = -Σ p(x) log p(x)
- Max entropy: uniform distribution
- Min entropy: deterministic

### Mutual Information
- I(X; Y) = KL(P(X,Y) || P(X)P(Y))
- "How much knowing Y tells you about X"
- 0 iff X, Y independent

### Bayesian thinking
- Prior: P(θ)
- Likelihood: P(data | θ)
- Posterior: P(θ | data) = P(data | θ) · P(θ) / P(data)
- MAP estimate: argmax_θ P(θ | data)
- MLE estimate: argmax_θ P(data | θ)

### Hypothesis testing (you'll see this in A/B testing questions)
- p-value: probability of observing data as extreme, assuming null hypothesis
- Type I error: false positive (reject null when true)
- Type II error: false negative (fail to reject when false)
- Power: 1 - P(Type II)

---

## Calculus

### Derivatives
- ∂f/∂x: how f changes per unit change in x
- Chain rule: ∂f/∂x = ∂f/∂g · ∂g/∂x
- Backprop is REPEATED chain rule

### Common derivatives
- d/dx (sigmoid(x)) = sigmoid(x) · (1 - sigmoid(x))
- d/dx (tanh(x)) = 1 - tanh²(x)
- d/dx (ReLU(x)) = 1 if x > 0 else 0
- d/dx (softmax(x)_i) = softmax(x)_i · (δ_ij - softmax(x)_j)
- d/dx (log(σ(x))) = 1 - σ(x)  ← useful for binary CE loss gradient

### Gradient
- Vector of partial derivatives
- Points in direction of steepest ascent
- Used in gradient descent: θ ← θ - η · ∇L

### Hessian
- Matrix of second derivatives
- Eigenvalues of Hessian → curvature
- Newton's method uses inverse Hessian (too expensive in DL → approximate via Adam etc.)

### Taylor series
- f(x + Δx) ≈ f(x) + f'(x)·Δx + (1/2)·f''(x)·Δx²
- Used to derive Newton's method, learning rate analysis
- Justifies first-order methods at small learning rates

---

## Optimization theory (for ML)

### Convex functions
- A function f is convex if line segment between any two points is above the function
- Convex optimization has unique global minimum
- Linear regression: convex. NN: non-convex (multiple local minima)

### Stochastic gradient noise
- SGD: gradient estimate has noise ∝ 1/√batch_size
- Larger batch = lower noise but slower convergence per step
- Noise can help escape local minima (folklore + some theory)

---

## Common math gotchas in interviews

### "What's the gradient of softmax+cross-entropy combined?"
For one-hot target y, prediction p = softmax(z):
- Loss: -log(p_y) = -z_y + log(Σ exp(z_j))
- ∂L/∂z_j = p_j - 1(j == y)

Very clean. This is why frameworks combine these for numerical stability.

### "What does L2 regularization do to gradients?"
- L = original_loss + (λ/2) · ||θ||²
- ∂L/∂θ = ∂original/∂θ + λ·θ
- Effect: every step, θ shrinks toward 0 by factor (1 - η·λ)

### "Derive the gradient of sigmoid"
- σ(z) = 1/(1+e^(-z))
- σ'(z) = σ(z)·(1 - σ(z))

This is the most common "show me you know calculus" question.

### "Why does layer norm work mathematically?"
- Normalizing changes the optimization landscape
- Forward: invariant to scale of pre-norm output
- Backward: gradients of scaled inputs are bounded; less exploding/vanishing
- LearnableScale (γ) and shift (β) restore representational capacity

---

## How to self-test

After reading a math concept, can you:
1. State it in your own words
2. Give a concrete numeric example
3. Identify where it appears in ML
4. Compute by hand on a 2x2 / 3-element example

If you can do all 4: good. If not, study more.

---

## When you find a gap

Process for any math gap:
1. Identify what specifically you don't understand (e.g., "I can't derive the gradient of CE+softmax")
2. Pick ONE resource (Khan Academy, 3Blue1Brown, textbook chapter)
3. Spend 30-60 min on it
4. Implement a small numeric verification in numpy
5. Add to your math cheatsheet

---

## Resources

- **3Blue1Brown** (YouTube)
  - "Essence of Linear Algebra" series
  - "Essence of Calculus" series
  - Best math intuition videos ever made

- **Khan Academy** — fill any gaps in calculus, linear algebra, probability

- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong) — free online textbook

- **Probability for Computer Scientists** (Stanford CS109) — lectures online

- **Linear Algebra MIT 18.06 (Strang)** — gold standard

---

End. Next: `06-interview-qa-bank.md` (already written) — drill the Q&A there.
