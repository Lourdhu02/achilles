# Lab 16 — Mechanistic interpretability

**Build:** an induction-head detector for a two-layer attention-only transformer trained on repeated random tokens, an activation-patching function, and a sparse autoencoder (SAE) that recovers 32 ground-truth features packed into 16 dimensions. Then train an SAE on the residual stream of your own lab 05 model.
**Time:** 3–4 h for the tests, 4–8 h for the scale-up · **Reads first:** [interpretability](../../curriculum/09-interpretability-and-safety.md)
**Run:** `pytest labs/16_interpretability` (your code) · `pytest labs/16_interpretability --impl=solution` (reference). The eight tests train two small models on CPU in about 15 seconds.

Interpretability asks what computation a trained network actually performs. Anthropic, Google DeepMind and OpenAI all publish research on it, and its tools double as everyday debugging instruments: attention patterns show what a head reads, patching shows where information lives, SAEs split activations into parts you can name. Interviews ask you to explain an induction head, to say what a patching result does and does not prove, and to train and judge an SAE. This lab builds each tool small enough to check against ground truth, and it shows a trap worth knowing: a test that passes while the model learns something other than what the test name says.

---

## 1. Induction heads

**The task.** `repeated_batch` makes sequences $[x_1 \dots x_T, x_1 \dots x_T]$ of random tokens (vocabulary 16, period $T$). The first half is unpredictable; in the second half every next token has appeared before, so only in-context copying can push the loss below $\ln 16 = 2.77$ nats. `train_induction` trains `AttnOnly` (2 layers × 2 heads, $d = 64$, learned absolute positions, no MLPs or norms) for 1,000 steps, drawing $T$ from 5–10 for each batch, with the loss on the second half only (`copy_loss`).

**The circuit.** The textbook solution (Elhage et al. 2021, [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html); Olsson et al. 2022, [arXiv 2209.11895](https://arxiv.org/abs/2209.11895)) uses two heads:

```text
position:   j-1   j   ...   i
token:       A    B   ...   A        -> predict B

layer 0   previous-token head at j: attends to j-1, writes "previous token = A" into resid[j]
layer 1   induction head at i: query "my token is A" matches key "previous token = A" at j,
          so it attends to j; its OV circuit copies "B" into resid[i] -> logit(B) rises
```

The induction head's keys are built from what the previous-token head wrote: that is **K-composition**. One layer cannot do it, because a layer-0 key at position j knows only token B and position j, not what preceded it.

**Detection.** On a sequence with period $T$, an induction head at second-half position $i$ attends to $i - T + 1$, the token after the previous occurrence of the current token; a previous-token head attends to $i - 1$. Average those attention weights over the batch and the second-half positions to get one score per head. Uniform attention would score about 0.07 at $T = 10$ (the mean of $1/(i+1)$ for $i = 10 \dots 19$); a clean induction head scores near 1.

**Why the period varies: the shortcut.** Train with a fixed period instead, `train_induction(steps=600, min_half=10)`, and the model passes the induction-head test without having an induction circuit. Its scores on fresh sequences:

| Score | L0 H0 | L0 H1 | L1 H0 | L1 H1 |
|---|---|---|---|---|
| Induction, period 10 | 0.89 | 0.77 | 0.88 | 0.90 |
| Induction, period 7 | 0.04 | 0.03 | 0.04 | 0.06 |
| Previous-token, period 10 | 0.08 | 0.04 | 0.08 | 0.10 |

Layer 0 scores as high as layer 1, and no head is a previous-token head (all score 0.10 or less). With the period always 10 and learned positions, "attend 9 positions back" is a purely positional rule that solves the task. It even beats real induction there: with 16 tokens, about 97% of first halves repeat some token, which makes induction ambiguous. Evaluate at period 7 and the second-half loss goes from 0.00 to 16.3 nats, far worse than guessing.

**What the lab's model learns.** With the period drawn from 5–10 for each batch (the default), no single offset works, and the textbook circuit appears:

| Score | L0 H0 | L0 H1 | L1 H0 | L1 H1 |
|---|---|---|---|---|
| Induction, period 10 | 0.29 | 0.05 | 0.87 | 0.77 |
| Induction, period 7 | 0.15 | 0.00 | 0.82 | 0.88 |
| Previous-token, period 10 | 0.38 | 0.76 | 0.04 | 0.04 |
| Previous-token, period 7 | 0.56 | 0.97 | 0.04 | 0.04 |

The second-half loss is 0.10 at period 10 and 0.09 at period 7. L0 H1 is a previous-token head (L0 H0 partly one), and both layer-1 heads are induction heads that work at any period. Zero-ablating L0 H1 (zeroing its 32 columns of `model.out[0].weight`) raises the loss from 0.10 to 0.23 at period 10 and from 0.09 to 0.46 at period 7, and drops the layer-1 induction scores from 0.87 and 0.77 to 0.68 and 0.50; ablating L0 H0 costs less (losses 0.16 and 0.12). The dependence is real but partial: in a model this small, other paths carry part of the signal. `test_copying_works_at_other_periods` and `test_a_previous_token_head_forms_in_layer_1` check exactly what the shortcut model fails.

The lesson generalizes: a metric that passes is not a mechanism. Test the claimed mechanism on inputs a shortcut cannot solve, and look at every layer, not only the one your hypothesis is about.

## 2. Activation patching

**Clean, corrupted, metric.** Run two inputs that differ minimally: **clean** (behavior present) and **corrupted** (behavior absent). Pick a metric $m$, usually the logit difference between the correct answer and a counterfactual one (in the IOI task of Wang et al. 2022, [arXiv 2211.00593](https://arxiv.org/abs/2211.00593), the indirect object's name versus the subject's); this lab uses the answer's logit. The normalized effect of a patch is

$$\text{effect} = \frac{m_\text{patched} - m_\text{corrupt}}{m_\text{clean} - m_\text{corrupt}},$$

0 if the patched site carries none of the difference, 1 if it restores all of it. Logits beat probabilities as a metric because probabilities saturate: moving the logit from 10 to 20 barely changes $p$ (Zhang and Nanda 2023, [arXiv 2309.16042](https://arxiv.org/abs/2309.16042)).

**Denoising vs noising.** Denoising (this lab) runs the corrupted input and patches in a clean activation: it tests whether a site is *sufficient*. Noising runs the clean input and patches in a corrupted activation: it tests whether a site is *necessary*. They disagree exactly where it matters. If two paths are redundant, denoising either one restores the behavior while noising either one does nothing; if both are required, the reverse holds. Run both before concluding that a site is unimportant (Heimersheim and Nanda 2024, [arXiv 2404.15255](https://arxiv.org/abs/2404.15255)).

**What `patching_effect` does.** It runs clean and keeps `model.resid[layer + 1]` (the residual stream after `layer`), runs corrupted, then runs corrupted again with that whole stream substituted, and reads the logit of `answer` at `pos` in all three. The test patches after the last layer, so the unembedding reads the clean stream directly and the effect is exactly 1.0. Full-stream patches are trivial at every layer: patching all positions after layer 0 also gives 1.0, since everything downstream is a function of that stream. Localization starts when you patch narrower sites.

**A patching table, by hand.** The test's pair: the clean first half is [8, 15, 13, 8, 6, 11, 2, 11, 8, 7], repeated; the corrupted input replaces the first half with the random tokens [2, 1, 15, 11, 5, 15, 15, 15, 10, 4]. Read at position 11 (token 15, which occurs once in the clean first half); the answer is 13, the token after that earlier 15. Patch the residual stream at one position at a time:

| Site: stream after layer, at position | Shortcut model (fixed period) | Lab model (variable period) |
|---|---|---|
| layer 0, position 2 (token 13, after the earlier 15) | 0.37 | 0.65 |
| layer 0, position 11 (the query) | 0.55 | 0.00 |
| layer 0, each other position | 0.00 | −0.11 to 0.01 |
| layer 1, position 11 | 1.00 | 1.00 |
| layer 1, each other position | 0.00 | 0.00 |
| logit of 13: clean / corrupted | 20.3 / −5.6 | 9.4 / −0.8 |

The two models keep the answer in different places, and patching shows it. The shortcut model's layer-0 heads at the query already look 9 positions back and fetch the 13, so the query's own stream carries 55% of the effect. The circuit model's query stream after layer 0 carries nothing: its layer-0 heads read the previous token, an 8 in both inputs. Everything sits at position 2, which holds the value to copy (13) and, written there by the previous-token head, the key "previous token = 15". Patching it restores 65%, not 100%, because in the corrupted input positions 3, 6, 7 and 8 also follow a 15, and their keys still compete for the induction heads' attention. The −0.11 is real as well: effects outside [0, 1] carry information, so do not clip them.

A patching map over layers (or heads) and positions tells you **where** the difference between the two inputs is carried, not **how** it is used. For that, patch narrower sites: single heads, then only a head's query, key or value input. Patching only the layer-1 key input at position 2 tests K-composition directly.

## 3. Sparse autoencoders

**The objective.** Models represent more sparse features than they have dimensions, in almost-orthogonal directions (superposition; Elhage et al. 2022, [arXiv 2209.10652](https://arxiv.org/abs/2209.10652)), which is why single neurons are polysemantic. An SAE learns an overcomplete dictionary to undo this:

$$f = \text{ReLU}\big((x - b_\text{dec})W_\text{enc} + b_\text{enc}\big), \qquad \hat x = f\,W_\text{dec} + b_\text{dec}, \qquad \mathcal L = \lVert x - \hat x\rVert_2^2 + \lambda \lVert f \rVert_1,$$

with every row of $W_\text{dec}$ kept at unit norm. The L1 term is a convex stand-in for L0, the number of active latents. The lab's toy data: 32 features, each active with probability 0.05 (1.6 active per input on average) at a uniform random magnitude in [0, 1), embedded along random unit directions in 16 dimensions. The SAE has 64 latents and trains for 3,000 steps. Sweeping λ:

| λ (`l1`) | Features recovered (cosine > 0.9) | L0 | FVU | Dead latents |
|---|---|---|---|---|
| 0.01 | 1 of 32 | 14.8 | 0.001 | 1 of 64 |
| 0.05 | 15 of 32 | 8.9 | 0.025 | 0 |
| **0.2** (default) | **31 of 32** | 3.1 | 0.097 | 1 |
| 0.5 | 31 of 32 | 1.6 | 0.258 | 7 |

FVU is the fraction of variance unexplained, $\lVert x - \hat x\rVert^2 / \lVert x - \bar x\rVert^2$.

- **Weak penalty (0.01):** near-perfect reconstruction with 15 active latents per input, and almost nothing recovered. 64 directions in 16 dimensions can reconstruct anything densely; the sparsity penalty is the only thing that prefers the true basis.
- **Strong penalty (0.5):** L0 matches the true 1.6, but a quarter of the variance is lost. L1 pulls every active latent toward zero (shrinkage), and more latents die. TopK SAEs (Gao et al. 2024, [arXiv 2406.04093](https://arxiv.org/abs/2406.04093)) and JumpReLU SAEs ([arXiv 2407.14435](https://arxiv.org/abs/2407.14435)) exist largely to escape this trade-off.
- **Dead latents** never fire, so the ReLU passes them no gradient and they stay dead. Initializing $W_\text{dec} = W_\text{enc}^\top$ (as the lab does), resampling dead latents toward badly reconstructed inputs, or an auxiliary loss keeps them alive.

**How the test checks recovery.** Toy data comes with ground truth: `superposition_data` returns the 32 true directions, and `train_sae` counts a feature as recovered if some decoder row has cosine > 0.9 with it. The test needs more than 70%; the reference gets 97%. The pre-bias $b_\text{dec}$ is frozen at 0 here, because the data mean mixes every feature.

**Why decoder norms must be constrained.** Scaling latent $i$ by $1/c$ and decoder row $i$ by $c$ leaves $\hat x$ unchanged and divides that latent's L1 cost by $c$. Unconstrained, training grows the decoder and shrinks the codes until the penalty means nothing. Remove `normalize_decoder()` from the default run and the decoder rows grow to an average norm of 5.5 (maximum 14.1), the mean active latent shrinks from 0.197 to 0.015, and recovery falls from 31 to 9 of 32. The equivalent alternative puts the norm in the penalty, $\lambda \sum_i f_i \lVert W_{\text{dec},i}\rVert$. Superposition has limits too: the same 32 features in 8 dimensions instead of 16 give 8 of 32 recovered and 13 dead latents.

**Reading an SAE feature** in a real model, where there is no ground truth. A latent is a hypothesis to test:

1. Look at activating contexts across the whole range, not only the top 20: sample them at several activation quantiles.
2. Check the activation density, the fraction of tokens on which it fires. Latents that fire on a large fraction of tokens are rarely interpretable.
3. Check the output side: $W_{\text{dec},i} W_U$ lists the tokens the direction pushes up and down (approximately, since it skips the final norm).
4. Test it causally: clamp or ablate the latent and check that behavior changes the way your label predicts.

**Limits.**

- **Feature splitting.** Widths 32, 64 and 128 on the toy recover 31, 31 and 32 of 32 features, with 0, 1 and 19 dead latents, and 0, 5 and 7 features covered by two or more live latents. The toy has nothing finer to split into, so extra width shows up as duplicates and dead latents. In real models extra width splits a feature into finer ones, such as "math" into "algebra" and "geometry"; there is no single true dictionary size.
- **Reconstruction error.** At the default, 9.7% of the variance is not reconstructed. In a real model, splice $\hat x$ back in and measure the increase in next-token loss: whatever the SAE misses is computation your features do not explain, and it is structured, not noise.
- **Absorption and top-activation illusions**: see the pitfalls in the [curriculum](../../curriculum/09-interpretability-and-safety.md).

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `induction_scores(attn, half) -> Tensor` | `test_induction_score_indexing`, `test_an_induction_head_forms_in_layer_2` | `attn` is (B, H, T, T); returns (H,), the mean attention from each i in [half, 2·half) to i − half + 1. Exactly 1.0 on a synthetic pattern; some layer-1 head of the trained model scores > 0.5 |
| 2 | `previous_token_scores(attn) -> Tensor` | `test_previous_token_score_indexing`, `test_a_previous_token_head_forms_in_layer_1` | returns (H,), the mean attention from each i ≥ 1 to i − 1. Exactly 1.0 on a synthetic pattern; some layer-0 head of the trained model scores > 0.5 |
| 3 | `patching_effect(model, clean, corrupt, layer, answer, pos=-1) -> float` | `test_patching_the_final_residual_restores_everything` | patching the stream after the last layer restores 1.0 ± 1e-4 of the logit gap, read at `pos=11` of 19-token inputs |
| 4 | `SparseAutoencoder.forward(x) -> (x_hat, f)` | `test_sae_recovers_features_from_superposition` | subtracts `b_dec` before encoding; returns the reconstruction and the latents |
| 5 | `SparseAutoencoder.loss(x, l1) -> Tensor` | same | squared error summed over dimensions plus `l1` × the L1 norm of `f` summed over latents, both averaged over the batch; `train_sae()` must recover more than 70% of the 32 directions |

Given: `AttnOnly` (it records `model.attn` and `model.resid`, and `forward(idx, patch={layer: tensor})` replaces the stream after `layer`), `repeated_batch`, `copy_loss`, `train_induction`, the SAE constructor and `normalize_decoder`, `superposition_data`, `train_sae`. `test_model_learns_in_context_copying` and `test_copying_works_at_other_periods` check the given training loop (second-half loss below 0.5 at periods 10 and 7) and pass once the module imports.

## Tips

> [!TIP]
> Plot before you score: `plt.imshow(model.attn[L][0, h])` on one repeated sequence. A previous-token head is a stripe just below the diagonal; an induction head is a stripe T − 1 below it, in the second half. Train the fixed-period model too (`train_induction(steps=600, min_half=10)`): its layer 0 shows the T − 1 stripe as well, the first sign of the positional shortcut.

- `induction_scores`: with `idx = torch.arange(half, 2 * half)`, `attn[:, :, idx, idx - half + 1]` picks one entry per query position (the two index tensors pair up elementwise); then average over the batch and position dimensions.
- `model.resid[0]` is the embedding and `model.resid[l + 1]` is the stream after layer `l`. Run the clean input first and `.clone()` the stream you need before the next forward.
- Wrap the three forwards of `patching_effect` in `torch.no_grad()`.

## Common bugs

- **Induction offset off by one:** `i - half` is the previous occurrence of the same token (a duplicate-token pattern), not the token after it. The indexing test catches it.
- **`resid[layer]` instead of `resid[layer + 1]`:** you substitute the stream from one layer earlier and the effect is no longer 1.0.
- **Wrong read position:** the test reads position 11 of a 19-token input, where the model predicts token 12.
- **Loss scaled differently from the reference:** averaging the squared error over dimensions (as `F.mse_loss` does) divides the reconstruction term by 16, and averaging the L1 over latents divides the penalty by 64. Both change the effective λ, and each drops `train_sae()` to 0 of 32 recovered.
- **Clipping patching effects** to [0, 1]: values outside it are real (section 2).

## CPU experiments (no GPU needed)

1. **Make the shortcut, then break it.** Train `train_induction(steps=600, min_half=10)` and the default model, and reproduce both tables of section 1. Then zero-ablate each layer-0 head in each model and measure the loss and the layer-1 induction scores at period 7.
2. **Watch the heads form.** Log the second-half loss and the best layer-1 induction score every 20 steps of the default run. Olsson et al. describe induction heads forming in an abrupt phase change: does your loss curve show a plateau and a drop, and does the score jump at the same step?
3. **Patching map.** Reproduce the section 2 table for every layer and position and plot it. Then change `AttnOnly.forward` so you can patch only the key input of layer 1 at position 2. How much of the effect does that alone restore?
4. **SAE sweeps.** Reproduce the λ table, then 8 dimensions, widths 32 and 128, and a run without `normalize_decoder()`. Plot recovery and FVU against L0 across all runs.
5. **Toy models of superposition.** Train $x' = \text{ReLU}(W^\top W x + b)$ with 5 features in 2 dimensions and plot the columns of $W$ as the features get sparser.

## GPU scale-up (8 GB): an SAE on your lab 05 model

Train an SAE on the residual stream of the S1 model from [lab 05](../05_transformer/README.md): byte-level, $d_\text{model} = 384$, 6 layers, saved by `train.py` to `runs/gpt/ckpt.pt`. A free Colab or Kaggle T4 works too. Napkin math first:

- A 16× SAE has 6,144 latents: $2 \times 384 \times 6{,}144 + 6{,}144 + 384 = 4.73$ M parameters, 76 MB with fp32 weights, gradients and two Adam moments.
- The latents for a batch of 4,096 tokens take $4{,}096 \times 6{,}144 \times 4$ bytes = 101 MB.
- Caching 10 M tokens of activations in fp16 takes $10^7 \times 384 \times 2$ bytes = 7.7 GB. Keep such a cache on disk, or run the 11 M-parameter model on the fly and shuffle activations through a buffer.

```python
import torch
from labs._impl import load
g = load("labs/05_transformer/x.py", "exercise")        # the implementation you trained S1 with
mi = load("labs/16_interpretability/x.py", "exercise")
ck = torch.load("runs/gpt/ckpt.pt", map_location="cpu")
model = g.GPT(g.GPTConfig(**ck["cfg"]))
model.load_state_dict(ck["model"])
model.cuda().eval()

acts = []
model.blocks[3].register_forward_hook(lambda mod, inp, out: acts.append(out.detach().float()))
sae = mi.SparseAutoencoder(384, 16 * 384).cuda()
opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
# for each batch of byte ids (B, 256) from TinyStories:
#     acts.clear()
#     with torch.no_grad(): model(ids)
#     x = acts[0].reshape(-1, 384)                       # shuffle across batches in practice
#     loss = sae.loss(x, l1=...); opt.zero_grad(); loss.backward(); opt.step(); sae.normalize_decoder()
```

- Unlike the toy, set `b_dec` to the mean of a sample of activations and let it train, and rescale activations to a fixed average norm so that λ means the same thing across layers and checkpoints.
- Sweep λ on a log grid and compare runs at matched L0 (tens of active latents per token is a common target). For each, report L0, FVU, the fraction of dead latents over about 1 M tokens, and the spliced loss: a forward hook that returns a tensor replaces the module's output, so return the SAE reconstruction from a hook on `blocks[3]` and measure the increase in next-token loss in nats.
- Label the top 20 features by hand: decode their activating contexts from bytes, note their density, and list the tokens each promotes through `lm_head`.
- Repeat at widths 4× and 64× and look for feature splitting and dead latents.

## Check yourself

1. Why does an induction head need a second layer in an attention-only model, and what is K-composition?
2. A model trained at a fixed period of 10 passes the induction test but fails at period 7. What did it learn, and which evidence gives it away?
3. Patching the full residual stream after layer 0 also restores 100%. Why, and what should you patch instead?
4. Denoising a site restores 90% of the logit difference, but noising it changes nothing. What does that suggest?
5. An SAE with λ = 0.01 explains 99.9% of the variance but recovers 1 of 32 features. Why?
6. Why must decoder norms be constrained, and what happened when the toy run skipped it?
7. You double an SAE's width and its "math" latent becomes "algebra" and "geometry" latents. Which is the real feature?

<details><summary>Answers</summary>

1. A layer-0 key at position j is built from the token and position at j only, so it cannot say "the token before me was A". A previous-token head in layer 0 writes the identity of token j − 1 into position j; the layer-1 head's keys read it and can match the query "my token is A". A later head whose keys read an earlier head's output is K-composition.
2. A fixed positional offset, "attend 9 back", built from learned position embeddings. The evidence: layer-0 heads have induction scores as high as layer 1 (0.89 and 0.77), no head has a previous-token pattern, at period 7 every induction score falls to 0.06 or less, and patching finds the answer already in the query's own stream after layer 0.
3. Everything downstream is a function of that stream at all positions, so replacing all of it reproduces the clean run exactly. Patch one position at a time, then single heads, then a head's query, key or value input.
4. The site is sufficient but not necessary: another path carries the same information (redundancy, backup heads). Noise pairs of sites, or noise the second path while denoising the first, to find it.
5. 64 latents in 16 dimensions can reconstruct anything with dense codes (L0 of 14.8 against a true 1.6), and the weak penalty does not prefer the sparse, true basis. Good reconstruction alone does not identify features.
6. Scaling a latent down and its decoder row up keeps $\hat x$ and shrinks the L1 cost, so the penalty can be evaded. Without normalization, decoder norms grew to 5.5 on average, active latents shrank about 13×, and recovery fell from 31 to 9 of 32.
7. Neither alone. The dictionary depends on width, data and λ; the split reveals a hierarchy in how the model represents the concept. Pick the width for the question you are asking, and check causally that the latents you rely on matter.
</details>

## Stretch

- **TopK SAE:** keep the k largest pre-activations instead of ReLU plus L1, and compare FVU and dead latents with your SAE at matched L0 on the toy.
- **Attribution patching:** approximate the whole section 2 table with one backward pass, $(a_\text{clean} - a_\text{corr}) \cdot \partial m/\partial a$, and compare it with exact patching.
- **Composition scores:** compute the K-composition score between each layer-0 OV circuit and each layer-1 QK circuit (Elhage et al. 2021) and check that, in the lab's model, the largest links the previous-token head to an induction head.
- **IOI on GPT-2 small:** reproduce the main patching results of Wang et al. 2022 with forward hooks on a Hugging Face model.
- Read [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Bricken et al. 2023) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Templeton et al. 2024) and compare their SAE training choices with yours.
