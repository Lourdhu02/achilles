# Experiment: does lab 16's model really learn an induction circuit?

**Date:** 2026-09-26 · **Code version:** after the lab 16 fix (September 2026) · **Hardware:** any laptop CPU, about 5 s per run · **Code:** `labs/16_interpretability/solution.py` (`train_induction`, `induction_scores`, `previous_token_scores`, `copy_loss`)

> [!NOTE]
> A worked example of the [experiment template](../templates/experiment.md), written from a real investigation of this repo's lab 16. Every number below reproduces on a CPU in under a minute with the snippet at the end. Use it as the standard for your own entries: a prediction written first, one change at a time, several seeds, and the lesson stated plainly.

## Question and hypothesis

Lab 16's test `test_an_induction_head_forms_in_layer_2` passed. Does that mean the model implements the two-head induction circuit: a previous-token head in layer 0 whose output the layer-1 induction head reads (K-composition)?

**Prediction, written before running.** If the circuit is real, then:

1. the model copies at a repeat period it never trained on (7) almost as well as at its training period (10);
2. some layer-0 head attends to the previous token (score above 0.5);
3. layer-0 heads do not score as induction heads, since one layer cannot compose.

If instead the model learned a positional shortcut, "attend 9 positions back", which is possible because the period was always 10 and positions are learned, then (1) fails badly and (2) fails.

## Setup

- **Model:** `AttnOnly`, 2 layers × 2 heads, $d = 64$, learned absolute positions, vocabulary 16, loss on the second half only.
- **The one change:** the repeat period during training.
  - *Fixed:* period 10, 600 steps, the lab's original setting: `train_induction(steps=600, min_half=10, seed=s)`.
  - *Variable:* period drawn from 5–10 for each batch, 1,000 steps: `train_induction(seed=s)`, the lab's current default.
- **Seeds:** 0, 1, 2 for each regime.
- **Metrics, on fresh sequences:** second-half copy loss in nats (256 sequences) at periods 10 and 7; the best head's induction score in each layer; the best layer-0 previous-token score (32 sequences). For scale: guessing costs $\ln 16 = 2.77$ nats, and uniform attention scores about 0.07.

## Results

Mean of three seeds, with the per-seed range in brackets.

| metric | fixed period | variable period |
|---|---|---|
| copy loss at period 10 (nats) | 0.00 | 0.10 [0.09–0.11] |
| copy loss at period 7 (nats) | **15.7** [15.0–16.3] | 0.10 [0.06–0.16] |
| best layer-1 induction score, period 10 | 0.90 [0.90–0.90] | 0.85 [0.82–0.87] |
| best layer-1 induction score, period 7 | **0.07** [0.02–0.14] | 0.88 [0.86–0.90] |
| best layer-0 induction score, period 10 | **0.89** [0.88–0.90] | 0.26 [0.20–0.30] |
| best layer-0 previous-token score | **0.11** [0.08–0.16] | 0.73 [0.70–0.76] |

## Prediction vs result

For the fixed-period model all three predictions of a real circuit fail, on every seed. At period 7 its loss is 15.7 nats, over five times worse than guessing. No head attends to the previous token, and its layer-0 heads already score 0.89 as "induction heads", because they attend 9 positions back by position alone. The passing test measured attention at the training period, where the shortcut and the circuit look the same.

The variable-period model matches all three predictions: it copies equally well at periods 7 and 10, has a clear previous-token head in layer 0 (0.73), and its induction heads are in layer 1. For seed 0, zero-ablating the previous-token head raises the period-7 loss from 0.09 to 0.46, so layer 1 uses that head's output. The effect is causal, though only partial.

**What surprised me:** the shortcut model is *better* at the training period (0.00 against 0.10 nats). With 16 tokens, about 97% of 10-token first halves repeat some token, which makes content-based lookup ambiguous; the positional rule never is. A metric measured only on the training distribution cannot separate the two mechanisms, and it even rewards the wrong one. An input the shortcut cannot solve can.

## Threats to validity

- Three seeds, one model size, one vocabulary. The shortcut's advantage depends on the vocabulary size and the range of periods.
- Attention scores show where heads look, not that the result is used. The ablation is the causal check, and I ran it for one seed only.
- The variable model trained for 1,000 steps and the fixed one for 600. The fixed model is already at 0.00 loss, so more steps are unlikely to add a circuit, but a run with matched steps would remove the doubt.

## What changed as a result

- Lab 16 now trains with a variable period, and two new tests check exactly what the shortcut fails: copying at period 7, and a previous-token head in layer 0.
- The [lab 16 handout](../../labs/16_interpretability/README.md) now teaches the shortcut and shows how activation patching tells the two models apart.

## Next

Train with periods 5–10 and a longer position table (`max_len` of 40), then evaluate at periods 12–20. Do learned absolute positions stop the circuit from working at periods longer than any it has seen?

## Reproduce

```python
import torch
from labs._impl import load

mi = load("labs/16_interpretability/x.py", "solution")
for name, kw in [("fixed", dict(steps=600, min_half=10)), ("variable", {})]:
    model, _ = mi.train_induction(seed=0, **kw)
    with torch.no_grad():
        seq = mi.repeated_batch(256, 7, 16, torch.Generator().manual_seed(3))
        print(name, "loss at period 7:", round(mi.copy_loss(model, seq, 7).item(), 2))
```
