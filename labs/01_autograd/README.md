# Lab 01 — Autograd from scratch

**Build:** a NumPy `Tensor` with reverse-mode automatic differentiation, then train an MLP with it.
**Time:** 8–12 h · **Reads first:** [math §2](../../curriculum/01-math.md#2-matrix-calculus-the-only-calculus-you-need), [deep learning §1](../../curriculum/03-deep-learning.md#1-autodiff-what-backward-actually-does)
**Run:** `pytest labs/01_autograd` (your code) · `pytest labs/01_autograd --impl=solution` (reference)

After this lab you should be able to derive, on a whiteboard, the backward pass of any layer you meet,
and to explain why frameworks are built the way they are. Every later lab rests on this one.

---

## 1. The ideas

### A computation is a DAG
`loss = cross_entropy(relu(x @ W1 + b1) @ W2 + b2, y)` is a directed acyclic graph: leaves are
parameters and data, internal nodes are ops, the root is a scalar. Each `Tensor` you create keeps
`_parents` and a `_vjp` closure. That record *is* the graph.

### Vector–Jacobian products (VJPs), not Jacobians
For an op `y = f(x)` with upstream gradient `ḡ = ∂L/∂y`, the backward pass needs
`∂L/∂x = Jᵀ ḡ`. You never build `J` (for a 4096×4096 matmul it would have 2.8·10¹⁴ entries).
Every op supplies a closed-form VJP:

| op | forward | VJP (given ḡ = ∂L/∂out) |
|---|---|---|
| add | `a + b` | `ḡ`, `ḡ` (then unbroadcast) |
| mul | `a * b` | `ḡ·b`, `ḡ·a` |
| div | `a / b` | `ḡ/b`, `−ḡ·a/b²` |
| matmul | `C = A @ B` | `Ā = C̄ Bᵀ`, `B̄ = Aᵀ C̄` |
| exp / log | `eˣ` / `ln x` | `ḡ·eˣ` / `ḡ/x` |
| tanh / relu | | `ḡ·(1 − tanh²x)` / `ḡ·𝟙[x>0]` |
| sum | `Σ` over axes | broadcast `ḡ` back to the input shape |
| logsumexp | `log Σ eˣ` | `ḡ · softmax(x)` |
| gather `x[idx]` | pick entries | **scatter-add** `ḡ` into zeros (`np.add.at`) |

Derive the matmul rule once, by hand: `dL = tr(C̄ᵀ dC)` and `dC = dA·B + A·dB`, so
`dL = tr((C̄Bᵀ)ᵀ dA) + tr((AᵀC̄)ᵀ dB)`. Then check yourself: the shapes must match
`A` and `B`, and there is exactly one arrangement of `C̄, A, B` that does.

### Why reverse mode
For `f: ℝⁿ → ℝ` (a loss over *n* parameters), forward mode costs *n* passes, one per input
direction. Reverse mode costs **one** backward pass, about 2–3× a forward pass, whatever *n* is.
The price is memory: the backward pass needs the forward pass's intermediates. That trade-off
leads to activation checkpointing ([lab 02](../02_training_core/README.md), [curriculum 05](../../curriculum/05-pretraining.md)).

### Topological order and accumulation
A node can feed several children (`x*x + x`). Its gradient is the **sum** over all paths, so you
can only call its VJP once every child has contributed. Process nodes in reverse topological order
and accumulate into a pending dict. Use an iterative DFS: a 5,000-op chain overflows Python's
recursion limit, and the test for exactly that is on purpose.

### Broadcasting: copies forward, sums backward
`x (3,4) + b (4,)` copies `b` onto each of 3 rows. The gradient of a copied value is the sum of
the copies' gradients, so `unbroadcast` sums away prepended axes and axes where the original had
size 1. Half of all "my gradients are wrong" bugs in real code are this.

### Gather forward, scatter-add backward
`x[[0, 0, 2]]` reads element 0 twice, so element 0 must receive **two** gradient contributions.
`full[idx] += g` silently drops the duplicate; `np.add.at` does not. The same duality explains
embedding-layer gradients (sparse rows) and why `index_add_` exists in PyTorch.

### Numerically stable logsumexp
`log Σ exp(x)` overflows for `x = 1000`. Use `m + log Σ exp(x − m)` with `m = max(x)`. The
gradient is `softmax(x)`, which you already have from the forward pass. Cross-entropy then
reduces to `logsumexp(z) − z_y`, and its gradient is `(softmax(z) − onehot(y)) / N`. That identity
is the most-asked "show me you can do calculus" interview question.

### Gradient checking
Central differences `(f(x+ε) − f(x−ε)) / 2ε` have error O(ε²), versus O(ε) for one-sided ones.
Use float64 and ε ≈ 1e-6: much smaller and floating-point cancellation dominates. The tests
backprop a *random* upstream gradient `G` and compare against finite differences of
`Σ f(x)·G`, which checks the full VJP rather than one row of it.

---

## 2. What to implement (in this order)

1. `unbroadcast` → `test_unbroadcast_matches_sum`
2. `__add__`, `__mul__`, `backward` → `test_node_used_twice_accumulates` (your first real milestone)
3. `__truediv__`, `__pow__`, `exp`, `log`, `relu`, `tanh` → `test_unary_ops`, `test_add_mul_div_broadcasting`
4. `__matmul__` → `test_matmul` (includes batched and broadcast batch dims)
5. `sum`, `reshape`, `transpose`, `__getitem__`, `logsumexp` → shape and reduction tests
6. `log_softmax`, `cross_entropy`, `numerical_gradient`, `sgd_step` → `test_mlp_learns_spirals` (≥ 97% accuracy)

`pytest labs/01_autograd -x` stops at the first failure. Work top to bottom.

## 3. Check yourself (answer without looking)

1. Why does `backward()` need a topological order? Construct a graph where a naive DFS order gives a wrong answer.
2. What is the memory cost of the backward pass for an L-layer MLP with batch B and width d? What would you store and what could you recompute?
3. Your loss is a scalar. What changes if you want the full Jacobian of a vector output? What does `torch.func.jacrev` do under the hood?
4. Why is `softmax` followed by `log` numerically worse than `log_softmax`? Give a concrete input where it fails in float32.
5. `relu` has no derivative at 0. Why does it not matter in practice, and when could it?

<details><summary>Answers (after you try)</summary>

1. A node's VJP must see the *total* gradient from all its children. If you visit `x` before one of its children has pushed its contribution, `x` propagates an incomplete gradient and the missing part never arrives. Example: `y = x*x + x`, visiting `x` right after the `+` node.
2. It stores each layer's input activations (B·d per layer, so O(L·B·d)) and the ReLU masks. With checkpointing you store every k-th activation and recompute the rest in the backward pass, trading about one extra forward pass for √L-scale memory.
3. You need one VJP per output coordinate, that is, one backward pass per row of the Jacobian. `jacrev` vectorizes those VJPs with `vmap`. For wide outputs and narrow inputs, forward mode (`jacfwd`) is cheaper.
4. `softmax` can underflow to exactly 0 for very negative logits, and then `log(0) = −inf`. In float32, `exp(−110)` is already 0, so logits `[0, −200]` produce `log p₂ = −inf`, while `log_softmax` returns −200 correctly.
5. The kink is a measure-zero set: activations are almost never exactly 0 under continuous noise. It matters for quantized or saturated activations, or when a bug initializes everything to exactly 0.
</details>

## 4. Stretch goals

- Add `max(axis)` with a correct tie-breaking VJP, then build `conv1d` using only existing ops and check its gradient.
- Add a `no_grad()` context manager and measure the memory saved (`tracemalloc`).
- Time your MLP step against the same MLP in PyTorch. Where does your time go? (Profile with `cProfile`: the answer is Python overhead per op, which is why frameworks fuse kernels.)
- Read [micrograd](https://github.com/karpathy/micrograd) and [tinygrad's `Tensor`](https://github.com/tinygrad/tinygrad). What does tinygrad do that you don't?

## 5. Common bugs

| symptom | likely cause |
|---|---|
| gradient right for `(3,4)+(3,4)` but wrong for `(3,4)+(4,)` | missing or incorrect `unbroadcast` |
| gradient doubles every time you call `backward` | `.grad` never cleared between steps |
| `x*x` gives `x.grad == x` instead of `2x` | you overwrite rather than accumulate |
| `RecursionError` on long chains | recursive topological sort |
| NaN in cross-entropy | softmax computed without subtracting the max |
