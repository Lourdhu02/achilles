# Lab 11 — DPO

**Build:** response-only sequence log-probabilities, the DPO loss with implicit rewards and label smoothing, IPO and SimPO, and a numerical proof that DPO converges to the closed-form optimum of the KL-regularized RLHF objective. Then DPO on your lab 10 model with 500 preference pairs.<br>
**Time:** 3–4 h for the tests, 4–8 h for the scale-up · **Reads first:** [post-training §3–4](../../curriculum/06-post-training.md#4-dpo-and-its-family)<br>
**Run:** `pytest labs/11_dpo` (your code) · `pytest labs/11_dpo --impl=solution` (reference). All five tests run on CPU in seconds.

DPO is the most-asked alignment derivation in research-engineer interviews, and `sequence_logprobs` is where most real alignment code has its bugs. After this lab you can derive the loss, implement it without an off-by-one, and show numerically that it optimizes what the derivation says.

---

## 1. The loss in one screen

From the curriculum's derivation: the RLHF optimum is $\pi^* \propto \pi_\text{ref}\, e^{r/\beta}$, so $r = \beta \log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$, and substituting into Bradley–Terry cancels $Z$. With the policy as the parameter:

$$\hat r(y) = \beta\,\big(\log\pi_\theta(y\mid x) - \log\pi_\text{ref}(y\mid x)\big), \qquad h = \hat r(y_w) - \hat r(y_l), \qquad \mathcal{L} = -\log\sigma(h)$$

With label smoothing $\epsilon$ (conservative DPO: assume a fraction $\epsilon$ of the labels are flipped):

$$\mathcal{L} = -(1-\epsilon)\log\sigma(h) - \epsilon\log\sigma(-h)$$

The family members you also implement ($\ell$ are log-ratios, $\bar\ell$ are length-normalized log-probs):

| loss | formula in the code | idea |
|---|---|---|
| `dpo_loss` | $-(1-\epsilon)\log\sigma(h) - \epsilon\log\sigma(-h)$ | Bradley–Terry on implicit rewards |
| `ipo_loss` | $\big((\ell_w - \ell_l) - \tfrac{1}{2\beta}\big)^2$ | regress the log-ratio margin to a finite target; `beta` here is IPO's $\tau$ |
| `simpo_loss` | $-\log\sigma\big(\beta(\bar\ell_w - \bar\ell_l) - \gamma\big)$ | no reference model; average log-prob; margin $\gamma$ |

## 2. Sequence log-probs: the off-by-one

A causal LM's logits at position $t$ predict the token at $t+1$. For a sequence `[prompt tokens | response tokens]`:

```
position t:     0     1     2     3     4
labels:        p0    p1    r0    r1    r2        mask: 0 0 1 1 1  (1 = response token)
logits[t] ->  p1?   r0?   r1?   r2?   (next)
use:          logits[:, :-1]  scored against  labels[:, 1:]  weighted by  mask[:, 1:]
```

So: `log_softmax(logits[:, :-1])`, `gather` at `labels[:, 1:]`, multiply by `mask[:, 1:]`, sum over time (or divide by the mask count when `average=True`, which SimPO uses). The mask marks which **label** tokens are response tokens; the shift moves it along with the labels.

## 3. The theory test

`tabular_dpo` (given) builds a policy over K = 6 responses with logits `theta`, a random reference, and hidden rewards `r`. It trains on **all ordered pairs** with Bradley–Terry soft labels: for the pair (i, j), label smoothing $\epsilon = 1 - \sigma(r_i - r_j)$, so the target probability that i beats j is $\sigma(r_i - r_j)$. The loss is minimized when $\sigma(h_{ij}) = \sigma(r_i - r_j)$ for every pair, i.e. $\beta(\ell_i - \ell_j) = r_i - r_j$, i.e.

$$\log\pi(i) = \log\pi_\text{ref}(i) + r_i/\beta - \log Z$$

`test_dpo_converges_to_the_closed_form_optimum` checks that 3,000 Adam steps land within 0.02 of this in every coordinate. That is the DPO derivation verified end to end, including the claim that $\beta$ controls how far the optimum sits from the reference.

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `sequence_logprobs(logits, labels, mask, average=False)` | `test_sequence_logprobs_shift_and_mask` | shift by one, gather, masked sum; `average=True` divides by the number of response tokens (3 and 2 in the test) |
| 2 | `dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta, label_smoothing)` | `test_dpo_loss_values`, `test_gradient_pushes_chosen_up_and_rejected_down` | returns `(mean loss, chosen rewards, rejected rewards)` with rewards `β(logπ − logπ_ref)` **detached**; loss is `log 2` when policy equals reference; gradient is negative on chosen and positive on rejected |
| 3 | `ipo_loss`, `simpo_loss` | `test_ipo_and_simpo` | IPO at zero margin with β = 0.5 is `(0 − 1)² = 1`; SimPO with a margin exactly equal to γ gives `log 2` |
| 4 | (given) `tabular_dpo` | `test_dpo_converges_to_the_closed_form_optimum` | uses your `dpo_loss` with per-pair label smoothing |

Hand-check the main test case before coding: `pi_c = −1, pi_r = −3, ref = −2, β = 0.5` gives rewards `+0.5` and `−0.5`, `h = 1`, loss `log(1 + e^{−1}) ≈ 0.313`.

## Tips

> [!TIP]
> Use `F.logsigmoid(h)`, never `torch.log(torch.sigmoid(h))`. For a badly wrong pair (h = −100) the second form computes `log(0) = −inf`; `logsigmoid` returns −100 exactly. The same applies to `log_softmax` vs `log(softmax)` in `sequence_logprobs`.

- Keep everything vectorized over the batch: inputs are tensors of shape `(B,)`, and `dpo_loss` returns the batch mean.
- `label_smoothing` can be a tensor of per-pair values (the theory test passes one), so write the formula with broadcasting, not an `if eps > 0` branch.
- Return the implicit rewards with `.detach()`: they are for logging, and gradients through them would double-count.

## Common bugs

- **No shift:** scoring `logits[:, t]` against `labels[:, t]` makes the model "predict" the token it was given; values are wrong and the test catches it.
- **Mask from the prompt side:** masking `mask[:, :-1]` instead of `mask[:, 1:]` scores the last prompt token and drops the last response token.
- **Mean instead of sum** in plain DPO: `average=False` is the DPO definition. Averaging changes the effective β per example and is SimPO's choice, not DPO's.
- **Swapped sign of label smoothing:** `ε` multiplies `log σ(−h)`, the log-probability that the *rejected* response is better.
- **IPO target** written as `1/β` or `β/2` instead of `1/(2β)`.
- **SimPO margin** applied outside the β scaling in the wrong direction: the test value `log 2` only comes out when `β·(avg_c − avg_r) − γ = 0`.

## GPU scale-up: DPO on your SFT model

Start from the lab 10 scale-up model (Qwen2.5-0.5B-Instruct after LoRA SFT on GSM8K).

**1. Build 500 pairs from your own model.** For each training question, sample 4–8 answers at temperature 0.8–1.0. With a verifiable task you do not need a judge: chosen = an answer with the correct final number, rejected = one with a wrong number. Skip questions where all samples agree. On-policy pairs like these avoid the off-policy gap of pairs written by another model. (For open-ended tasks, use a stronger model as a judge and check its agreement with your own labels on 50 pairs.)

**2. Get the reference for free.** Merge the SFT adapter into the base (lab 10's `merge()`), attach a fresh LoRA for DPO, and compute reference log-probs by setting `scale = 0.0` on every `LoRALinear` (the given forward multiplies the adapter path by `self.scale`), then restoring it. Or precompute all reference log-probs once before training and store them with the pairs; with 500 pairs that is cheap.

**3. Train.** One forward for chosen and one for rejected per pair (or concatenate them in one batch), `sequence_logprobs` on response tokens only, your `dpo_loss`. Learning rate around 5e-6 to 5e-5 for LoRA, 1–2 epochs. Watch memory: the logits of a 512-token sequence are 512 × 151,936 floats (0.31 GB in fp32) per sequence; compute log-probs in chunks or one example at a time and accumulate gradients.

**4. Sweep β ∈ {0.05, 0.1, 0.5}.** For each, log every step:

| metric | healthy | warning sign |
|---|---|---|
| preference accuracy (margin > 0) | rises from ~50% | ~100% within a few steps: pairs too easy or a length shortcut |
| chosen implicit reward | small positive or flat | strongly negative together with rejected: likelihood displacement |
| raw chosen log-prob | roughly flat | falls by tens of nats |
| mean response length on held-out prompts | flat | rising steadily: length exploitation |

**5. Evaluate** each β on held-out GSM8K exact match with a paired bootstrap CI against the SFT model ([lab 15](../15_eval_stats/README.md)), and estimate KL to the reference by sampling from the policy on held-out prompts and averaging `log π − log π_ref` over the sampled tokens. Plot accuracy against KL, one point per β. Smaller β should move further (higher KL); whether it helps is the experiment.

> [!TIP]
> Save 20 fixed prompts and print the model's answers to them at every evaluation. Most DPO pathologies (verbosity, repetition, refusals, format drift) are obvious to a reader long before they show up in an aggregate metric.

## Check yourself

1. Why does $Z(x)$ cancel, and why would the method be impractical if it did not?
2. What is the DPO loss and its gradient when the policy equals the reference?
3. Why does `dpo_loss` need the reference log-probs at all, when SimPO does not?
4. Accuracy is 95% but both chosen and rejected log-probs are falling. What is happening and what do you try?
5. Why can DPO drive $\pi(y_l) \to 0$ on a pair that is always preferred, regardless of β, and how does IPO avoid it?
6. In the theory test, what would the learned policy converge to with β = 5 instead of 0.5?

<details><summary>Answers</summary>

1. Bradley–Terry depends only on the reward difference for the same prompt, and $\beta\log Z(x)$ is the same for both responses. Without the cancellation you would need $Z(x)$, a sum over all possible responses.
2. $h = 0$, so the loss is $\log 2$, and the gradient is $-\tfrac{\beta}{2}\big(\nabla\log\pi(y_w) - \nabla\log\pi(y_l)\big)$: push chosen up and rejected down with weight 1/2.
3. The reference anchors the implicit reward and provides the KL regularization implied by the derivation. SimPO drops it and uses the average log-prob directly, trading the KL anchor for simplicity and length normalization.
4. Likelihood displacement: the margin grows because rejected falls faster than chosen, and probability mass moves to other responses. Add an NLL term on chosen responses, filter near-duplicate pairs, raise β, or switch to on-policy pairs.
5. With target probability 1, the Bradley–Terry likelihood keeps improving as $h \to \infty$, which on a finite dataset can be achieved by sending $\pi(y_l)$ to zero; β only rescales $h$. IPO regresses $h$ to the finite target $1/(2\tau)$, so the optimum stays a finite distance from the reference.
6. $\log\pi = \log\pi_\text{ref} + r/5 - \log Z$: much closer to the reference, since a larger β means a stronger KL penalty.
</details>

## Stretch

- **DPO vs IPO on hard labels:** copy `tabular_dpo`, set `label_smoothing = 0` with every pair ordered by `r`, and track `log π` of the worst response for DPO and IPO over training. Show DPO driving it down without bound while IPO settles.
- **Likelihood displacement in miniature:** on the tabular policy, make two responses nearly identical in a shared feature parameterization (instead of free logits) and show the chosen probability falling during DPO.
- **KTO:** implement the unpaired loss from the curriculum table and train on the same data with the pairing thrown away.
- **Online iterative DPO:** after one epoch, resample pairs from the current policy, relabel with the checker, and train again. Compare with a single offline pass at equal compute.
