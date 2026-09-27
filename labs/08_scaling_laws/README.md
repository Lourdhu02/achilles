# Lab 08 — Scaling laws

**Build:** a power-law fitter, the Chinchilla loss surface, the closed-form compute-optimal allocation,
and inference-aware model sizing. Then run your own IsoFLOP study on your GPU and fit your own exponent.
**Time:** 3–4 h for the tests + an overnight scale-up run · **Reads first:** [pretraining §3](../../curriculum/05-pretraining.md#3-scaling-laws)<br>
**Run:** `pytest labs/08_scaling_laws` (your code) · `pytest labs/08_scaling_laws --impl=solution` (reference) · CPU only, seconds

---

## Why this matters

Every large training run starts with a scaling-law argument: how big a model, how many tokens,
what loss to expect, and whether a change that helps at 100M parameters will still help at 10B.
Getting it wrong is expensive: by Chinchilla's analysis, GPT-3 (175B parameters, 300B tokens) and
Gopher (280B parameters, 300B tokens) were much larger than optimal for their compute. This lab gives
you the tools to make and check these arguments yourself.

## The math

**The loss surface** (Hoffmann et al. 2022, approach 3):

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},\qquad E = 1.69,\ A = 406.4,\ B = 410.7,\ \alpha = 0.34,\ \beta = 0.28 .$$

**Fitting a power law with an offset.** $L(N) = E + A N^{-\alpha}$ is nonlinear in $E$, but for a
*fixed* $E$ it is linear in log space:

$$\log(L - E) = \log A - \alpha \log N .$$

So grid-search the single nonlinear parameter $E$ over $[0, \min L)$, solve ordinary least squares
for $(\log A, \alpha)$ at each grid point, and keep the $E$ with the smallest residual. Fitting in
log space weights relative errors equally, which is what you want when losses span a range.
This is more robust than a generic nonlinear optimizer, which often lands in a bad local minimum
with $E$ and $A$ trading off against each other.

**Compute-optimal allocation.** With $C = 6ND$, substitute $D = C/(6N)$ and set $dL/dN = 0$:

$$N^* = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}},\qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\qquad D^* = \frac{C}{6N^*} .$$

**Inference-aware sizing** (Sardana et al. 2023). A model that will serve $I$ tokens costs
$6ND$ to train and about $2NI$ to serve. For a target loss $\ell$, the tokens needed at size $N$ are

$$D(N) = \left(\frac{B}{\ell - E - A N^{-\alpha}}\right)^{1/\beta}\quad\text{(defined only when } \ell > E + AN^{-\alpha}\text{)},$$

and you minimize $6N D(N) + 2NI$ over $N$. More inference pushes the optimum to smaller models
trained on more tokens.

## What to implement

| function | contract | test |
|---|---|---|
| `fit_power_law(N, L, n_grid)` | return `(E, A, alpha)`; grid over E in `[0, min(L))`, least squares in log space | `test_fit_recovers_known_law`: recovers E = 1.8 (±0.1) and α = 0.34 (±0.04) from 12 points over 4 decades with 0.2% noise |
| `chinchilla_loss(N, D, ...)` | `E + A/N^α + B/D^β`, vectorized over NumPy arrays | `test_loss_decreases_in_both_axes` |
| `compute_optimal(C, ...)` | closed form above; return `(N*, D*)` with `6·N*·D* = C` | `test_closed_form_matches_grid_search` at C = 1e20 and 5.76e23: within 1% of a 20,000-point grid search |
| `inference_aware_optimum(target_loss, inference_tokens, ...)` | over `n_grid` (default 1e7–1e12), skip N that cannot reach the target, solve D(N), minimize `6ND + 2NI`; `ValueError` if nothing reaches the target | `test_more_inference_means_smaller_models_trained_longer`: I = 1e13 gives smaller N and larger D than I = 0, and the loss hits 2.2 to 1e-6 |

## Worked numbers (check yours against these)

`compute_optimal` with the default constants:

| C (FLOPs) | N* | D* | tokens/param |
|---|---|---|---|
| 1e20 | 0.645B | 25.9B | 40 |
| 1e22 | 5.16B | 323B | 63 |
| 5.76e23 | 32.2B | 2.98T | 93 |

Chinchilla itself (5.76e23 FLOPs) was 70B parameters on 1.4T tokens. The gap is real: the published
approach-3 fit implies $N^* \propto C^{0.45}$ and a drifting tokens/param ratio, while Hoffmann et al.'s
approaches 1 and 2 give $\approx C^{0.5}$ and ~20 tokens/param. A replication by Epoch AI
(Besiroglu et al. 2024) traced the problem to the approach-3 fit itself. See
[curriculum 05 §3](../../curriculum/05-pretraining.md#3-scaling-laws).

`inference_aware_optimum(2.2, I)`:

| I (tokens served) | N | D | tokens/param |
|---|---|---|---|
| 0 | 3.54B | 204B | 58 |
| 1e12 | 1.73B | 511B | 295 |
| 1e13 | 0.97B | 1.80T | 1,856 |
| 1e14 | 0.65B | 8.17T | 12,628 |

## Common bugs

- **Grid reaching min(L).** `log(L − E)` is undefined at $E = \min L$. Stop the grid just below it.
- **Fitting in linear space.** A least-squares fit of $L$ itself is dominated by the largest losses
  (the smallest models); fit $\log(L - E)$.
- **Sign of α.** The slope of $\log(L-E)$ against $\log N$ is $-\alpha$.
- **Forgetting the 6.** $N^*$ uses $C/6$, and $D^* = C/(6N^*)$, not $C/N^*$.
- **Unreachable targets.** For small $N$, $\ell - E - AN^{-\alpha} \le 0$: no amount of data
  reaches the target. Skip those N instead of taking the log or power of a negative number.
- **Inference cost of 6N per token.** Serving is a forward pass only: about $2N$ FLOPs per token.

> [!TIP]
> Write the brute-force version first (a dense grid over N for `compute_optimal`, as the test
> does) and use it to check the closed form. Keep both: the grid is how you will sanity-check any
> scaling-law claim you meet later.

## Scale-up run S2: your own IsoFLOP curves (RTX 5060, overnight)

Train a family of byte-level models from lab 05 at three compute budgets, find the loss-minimizing
size at each budget, and fit the exponent $a$ in $N^* \propto C^a$. Chinchilla's approaches 1 and 2
give $a \approx 0.5$; what does your setup give?

**1. Define compute with the script's own count.** `labs/05_transformer/train.py` prints
MFLOPs/token, including attention and the LM head, which matter at this size (attention is ~13% of
the total for the smallest model below). Use $C = \text{FLOPs/token} \times \text{tokens}$, not $6N$.

**2. Pick sizes and budgets.** Byte-level vocabulary (256) keeps embeddings negligible. With
`--block-size 256 --batch-size 64` (16,384 tokens/step), steps = C / (FLOPs/token × 16,384):

| `--d-model` / `--n-layer` / `--n-head` | params | MFLOP/token | steps at C = 1e15 | 3e15 | 1e16 |
|---|---|---|---|---|---|
| 128 / 4 / 2 | 0.89M | 6.1 | 10,003 | 30,009 | 100,029 |
| 192 / 6 / 3 | 2.71M | 18.0 | 3,390 | 10,170 | 33,900 |
| 256 / 8 / 4 | 6.49M | 42.1 | 1,450 | 4,349 | 14,498 |
| 384 / 8 / 6 | 14.3M | 90.3 | 676 | 2,028 | 6,760 |
| 512 / 10 / 8 | 32.3M | 201.4 | 303 | 909 | 3,031 |

Each budget needs points on both sides of its minimum. Drop cells with very few steps (under ~500)
and add a size if a budget's minimum sits at the edge of your range.

**3. Run each cell with its own schedule.** `train.py` warms up for `--warmup` steps and then decays
the LR with a cosine that ends exactly at `--max-steps`, so setting `--max-steps` per run gives each model a
full schedule sized to its own run length, as in Chinchilla's IsoFLOP protocol. Keep `--warmup`
small relative to the shortest run (e.g. 100 steps): Porian et al. found overlong warmup biases
scaling fits against small models. Set `--eval-interval` to `--max-steps` so the script evaluates
only at the start and the end, and raise `--eval-iters` (e.g. 50) for a less noisy final
validation loss. Give each run its own `--out` directory.

```bash
python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt \
    --d-model 256 --n-layer 8 --n-head 4 --block-size 256 --batch-size 64 \
    --max-steps 4349 --warmup 100 --eval-interval 4349 --eval-iters 50 \
    --out runs/isoflop/C3e15_d256
```

**4. Predict the wall-clock time first.** The whole grid is about $5 \times (1 + 3 + 10)\times10^{15}
= 7\times10^{16}$ FLOPs. Divide by your measured throughput (from S1) to get hours; tiny models run
at low MFU, so expect the 100k-step cells to be dominated by per-step overhead.

**5. Fit.** For each budget, fit a parabola to final validation loss against $\ln N$ and take the
vertex as $N^*(C)$. Then fit a line to $\ln N^*$ against $\ln C$; the slope is $a$. Also fit
`fit_power_law` to the lower envelope of loss against N. Report the exponent with a bootstrap
interval (resample the runs within each budget) and at least two seeds for one budget, so you know
your noise floor.

**6. Write it up** with the [experiment template](../../journal/templates/experiment.md): setup,
your predicted exponent, the measured one with its interval, and what limits the comparison with
Chinchilla (byte-level tokens, one dataset, tiny models, a fixed batch size and LR).

> [!TIP]
> Before the full grid, run the smallest and largest model of one budget at two learning rates
> each. If the best LR differs across sizes, per-size LR tuning matters for your fit, which is one
> of the three causes of the Kaplan–Chinchilla disagreement.

## CPU vs GPU notes

The four functions and their tests are NumPy and run on any CPU in seconds. S2 needs a GPU for the
budgets above. On CPU, scale everything down about 30x (budgets of 3e13–3e14 FLOPs, the three
smallest sizes, a shorter `--block-size`), and expect noisier fits; the method is the same.

## Check yourself

1. Why grid-search E rather than fit all three parameters with a generic nonlinear optimizer?
2. Derive $N^*(C)$ from the loss surface and the constraint $C = 6ND$.
3. With the default constants, what is the exponent $a$ in $N^* \propto C^a$, and why does it make tokens/param drift upward with compute?
4. Why must each IsoFLOP run have its own LR schedule sized to its own length?
5. Your fitted exponent is 0.62 with a bootstrap interval of ±0.15. What do you report, and what do you change?
6. Why does `inference_aware_optimum` sometimes skip grid points, and when does it raise?
7. A startup will serve 1e13 tokens. The compute-optimal model for their training budget is 3.5B. What does the inference-aware view suggest, and what does it cost in training?

<details><summary>Answers</summary>

1. For fixed E, the problem is linear least squares in log space with a closed-form solution, so a 1-D grid over E finds the global optimum reliably. Joint nonlinear fits are sensitive to initialization because E, A and α trade off, especially when N spans few decades.
2. Substituting $D = C/(6N)$ and setting the derivative to zero gives $\alpha A N^{-\alpha} = \beta B (C/(6N))^{-\beta}$, so $N^{\alpha+\beta} = (\alpha A/\beta B)(C/6)^\beta$ and $N^* = G(C/6)^{\beta/(\alpha+\beta)}$.
3. $a = 0.28/0.62 = 0.45$, so $D^* \propto C^{0.55}$ and $D^*/N^* \propto C^{0.10}$, which rises slowly with compute (40 at 1e20, 93 at 5.76e23).
4. A run cut from a longer cosine schedule is evaluated before its LR has decayed, so its loss is worse than a properly finished run of that length. Shorter (larger-model) runs are penalized more, which biases the optimum toward smaller models. Hoffmann et al. suspected this biased Kaplan et al.'s fits; Porian et al. later found it was not the main cause, but matched schedules remain the clean protocol.
5. Report 0.62 ± 0.15 and say it is consistent with 0.5. To tighten it: add budgets to widen the compute range, add sizes near each minimum, run more seeds, and increase `--eval-iters`.
6. For small N, $E + AN^{-\alpha}$ is already above the target, so no finite D reaches it; those N are skipped. If no N on the grid can reach the target, it raises `ValueError`.
7. At loss 2.2 the table gives ~0.97B trained on ~1.8T tokens instead of 3.5B on ~204B. Training FLOPs rise (6·0.97e9·1.8e12 ≈ 1.05e22 vs 6·3.54e9·2.04e11 ≈ 4.3e21), but serving 1e13 tokens costs 2·0.97e9·1e13 ≈ 1.9e22 instead of 7.1e22, so total cost falls from ~7.5e22 to ~3.0e22 FLOPs, about 60% less.
</details>

## Stretch

- **Approach 3 on your own data.** Fit the full $L(N, D)$ surface to all S2 runs by minimizing a Huber loss on $\log L$ (as Hoffmann et al. did), and compare its implied exponent with your IsoFLOP estimate.
- **Reproduce the Kaplan effect in miniature.** Rerun a few S2 cells with `--tokenizer gpt2`, where the embedding table (50,257 × d) is most of a small model's parameters. Fit $N^*(C)$ once counting all parameters and once counting only non-embedding parameters. Which exponent looks more like Kaplan's?
- **WSD instead of cosine.** Copy `train.py`, change `lr_at` to warmup–stable–decay, and branch cooldowns from one long run at several token counts. Compare the cost of your IsoFLOP study both ways.
- **Data-constrained scaling.** Restrict the data file to 20M bytes and train past 4 epochs. Where does repetition start to hurt compared with Muennighoff et al.'s findings?
