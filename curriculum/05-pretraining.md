# 05 — Pretraining

How a base model gets made: the data pipeline, scaling laws that set model size and token count, the parallelism that fits the job on a cluster, and the failures that interrupt it.
Every section ends in numbers you can recompute. The mastery target is §8: plan a 7B run end to end and defend each number.

Labs: [08 scaling laws](../labs/08_scaling_laws/README.md), [09 parallelism](../labs/09_parallelism/README.md), and scale-up run S1 in [lab 05](../labs/05_transformer/README.md#scale-up-run-s1-pretrain-on-your-rtx-5060).

**Contents:** [1 Data](#1-data-is-the-model) · [2 Tokenization](#2-tokenization-choices) · [3 Scaling laws](#3-scaling-laws) · [4 Recipe](#4-the-standard-recipe-202426-era) · [5 Distributed training](#5-distributed-training) · [6 Stability](#6-stability-at-scale) · [7 Long context](#7-long-context) · [8 Plan a 7B run](#8-exercise-plan-a-7b-run-on-paper) · [Interview traps](#interview-traps) · [CPU vs GPU notes](#cpu-vs-gpu-notes) · [Check yourself](#check-yourself) · [Visual guides](#visual-guides) · [Read next](#read-next)

---

## 1. Data is the model

At a fixed architecture and compute budget, the data decides most of what the model knows. Modern pipelines look like this:

```mermaid
flowchart LR
    CC["Common Crawl WARC<br/>(HTML)"] --> EX["text extraction<br/>(trafilatura)"] --> LID["language ID<br/>(fastText)"]
    LID --> HF["heuristic filters<br/>(Gopher, C4 rules)"] --> DD["dedup<br/>(MinHash, exact)"]
    DD --> QF["model-based quality<br/>classifier"] --> PII["PII / toxicity"]
    PII --> DC["decontamination<br/>vs eval sets"] --> MIX["mixing, upsampling<br/>+ code, math, books"]
    MIX --> TOK["tokenize, pack<br/>into sequences"]
```

| stage | typical method | what it removes or fixes | pitfall |
|---|---|---|---|
| extraction | trafilatura on raw WARC HTML | boilerplate, menus, markup | Common Crawl's pre-extracted WET text is noisier; FineWeb found WARC extraction worked better |
| language ID | fastText classifier, keep score ≥ threshold (FineWeb: English ≥ 0.65) | wrong-language pages | low-resource languages (Telugu, Odia) score poorly and get dropped silently |
| heuristic filters | Gopher/MassiveText quality and repetition rules, C4 rules | spam, lists of keywords, repeated lines | C4's terminal-punctuation rule removes ~30% of tokens; FineWeb kept every C4 rule except that one |
| deduplication | MinHash-LSH (fuzzy), suffix arrays or hashes (exact) | mirrors, templates, near-copies | global dedup across all crawls can *hurt* (see below) |
| quality classifier | FineWeb-Edu: a classifier trained on Llama-3-70B-Instruct ratings of educational value; DCLM: a fastText classifier with instruction-style positives | low-information text | narrows diversity; tune thresholds with ablations, not taste |
| decontamination | n-gram overlap against benchmark test sets | leaked test items | paraphrased or translated items evade n-gram checks |

**A real funnel (FineWeb, 2024).** 96 Common Crawl snapshots → base filtering (URL blocklist, fastText English ≥ 0.65, Gopher quality and repetition filters) gave ~36T GPT-2 tokens → MinHash dedup *per snapshot* gave ~20T → C4-style and custom heuristic filters gave the released **15T**. The FineWeb-Edu subset, kept by the educational-quality classifier, is ~1.3T tokens and trains much better on knowledge and reasoning benchmarks per token.

### MinHash near-dedup, the math

Represent each document as a set of shingles (FineWeb uses word 5-grams). Similarity is Jaccard: $J(A,B) = |A\cap B|/|A\cup B|$. Comparing all pairs is impossible at billions of documents, so:

1. **MinHash.** For a random hash function $h$, $\Pr[\min_{a\in A}h(a) = \min_{b\in B}h(b)] = J(A,B)$. Proof: the minimum over $A\cup B$ is equally likely to be any element of $A\cup B$, and the two minima agree exactly when that element lies in $A\cap B$. With $k$ hash functions, the fraction of matching minima estimates $J$.
2. **Banding (LSH).** Split the $k = b\cdot r$ minhashes into $b$ bands of $r$. Two documents become a candidate pair if **all** $r$ values match in **any** band:

$$P(\text{candidate}\mid J = s) = 1 - (1 - s^r)^b ,$$

an S-curve whose steepest point is near $s^* \approx (1/b)^{1/r}$.

FineWeb uses 112 hashes as $b = 14$ bands of $r = 8$, targeting documents at least ~75% similar. Check the curve yourself:

| $(b, r)$ | $s$ = 0.5 | 0.6 | 0.7 | 0.75 | 0.8 | 0.9 | $(1/b)^{1/r}$ |
|---|---|---|---|---|---|---|---|
| (14, 8), FineWeb | 0.05 | 0.21 | 0.57 | 0.77 | 0.92 | 1.00 | 0.72 |
| (20, 5), loose | 0.47 | 0.80 | 0.98 | 1.00 | 1.00 | 1.00 | 0.55 |
| (9, 13), strict | 0.00 | 0.01 | 0.08 | 0.20 | 0.40 | 0.93 | 0.84 |

More rows per band sharpens the threshold upward; more bands pulls it down and catches more pairs. Candidates are then clustered transitively (A~C and B~C puts A, B, C in one cluster) and one document per cluster is kept.

**The FineWeb surprise.** Deduplicating globally across all 96 snapshots removed up to 90% of the oldest crawls, and the model trained on the result barely beat undeduplicated data. For one old snapshot, the 10% that survived global dedup was *worse* than the 90% removed: the survivors were disproportionately spam that appears only once, while good pages recur across crawls. Deduplicating each snapshot independently worked better. Lesson: dedup is for removing huge duplicate clusters, not for making every document unique.

**Exact dedup** complements MinHash: suffix arrays find long repeated substrings (Lee et al. 2021 removed repeats of 50+ tokens), and line- or paragraph-level hashing strips boilerplate that repeats inside otherwise distinct pages. Deduplication also reduces memorization of training text.

### Mixtures, repetition, decontamination

- **Mixtures matter as much as volume.** Llama 3's final pretraining mix was roughly 50% general knowledge, 25% math and reasoning, 17% code and 8% multilingual tokens. Learned weights are an option: DoReMi trains a 280M proxy with group DRO over domains to set weights for an 8B model. Always compute **epochs per source**: a 50B-token math corpus at 10% of a 2T-token run is seen $0.1\cdot2\text{T}/50\text{B} = 4$ times.
- **Repetition.** Muennighoff et al. (2023) found that up to ~4 epochs of repeated data is nearly as good as fresh data at fixed compute; beyond that the value of extra compute decays towards zero. Scarce high-quality sources (math, code, Indic text) hit this limit first.
- **Decontamination.** Collect all $n$-grams (GPT-3 used 13-grams) from benchmark test sets, and drop or flag training documents that overlap. Report results on clean vs contaminated subsets. Paraphrases, translations and solutions posted in forums still leak, so prefer benchmarks created after your data cutoff, and hold out private evals.
- **Synthetic data** (rephrased web pages, textbook-style generations, reasoning traces) is now central, but it inherits the generator's blind spots and narrows diversity. Keep a real-data baseline and measure diversity, not only benchmark scores.
- **Packing.** Concatenate documents with an end-of-text token into fixed-length sequences so no compute is wasted on padding. Add document masking so tokens do not attend across document boundaries (§7).

> [!TIP]
> Before any mixture debate, build a table: source, unique tokens, weight, resulting epochs, and a small-model ablation score. Most bad mixtures are visible in the epochs column.

## 2. Tokenization choices

Vocabulary size (32k → 128k → 200k+), the pre-tokenization regex, digit handling and multilingual coverage are fixed for the model's life. A larger vocabulary shortens sequences (cheaper attention, more text per context window) but adds $Vd$ parameters per table: at $V = 128{,}256$ and $d = 4096$, 0.53B for the embedding and another 0.53B for an untied head, and $2dV$ LM-head FLOPs per token. Measure **bytes per token on each language you care about**; a tokenizer trained on English-heavy data can charge several times more tokens for Indic scripts. See [lab 04](../labs/04_tokenizer/README.md) for the Telugu case. A storage detail with a real bug behind it: token ids above 65,535 do not fit in `uint16`, so 128k vocabularies need 32-bit id files (8 TB for 2T tokens).

## 3. Scaling laws

The Chinchilla parametric form:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

$E$ is the irreducible loss of the data (its entropy under a perfect model), $A/N^\alpha$ the penalty for a finite model, $B/D^\beta$ the penalty for finite data. Hoffmann et al.'s published fit: $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.

### Compute-optimal allocation, derived

With $C = 6ND$, substitute $D = C/(6N)$ and set $dL/dN = 0$:

$$-\alpha A N^{-\alpha-1} + \beta B\left(\frac{C}{6}\right)^{-\beta} N^{\beta-1} = 0 \;\Rightarrow\; N^* = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}},\quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\quad D^* = \frac{C}{6N^*}.$$

So $N^* \propto C^{a}$ with $a = \beta/(\alpha+\beta)$ and $D^* \propto C^{1-a}$. You implement and grid-check this in lab 08.

### The same fit, in numbers (lab 08 reproduces these)

| compute $C$ (FLOPs) | $N^*$ | $D^*$ | tokens/param | predicted loss |
|---|---|---|---|---|
| 1e20 | 0.64B | 26B | 40 | 2.60 |
| 1e21 | 1.8B | 91B | 50 | 2.33 |
| 1e22 | 5.2B | 323B | 63 | 2.14 |
| 5.76e23 (Chinchilla's budget) | 32B | 3.0T | 93 | 1.93 |

**Trap:** Chinchilla itself was 70B parameters on 1.4T tokens (20 tokens/param). Hoffmann et al. used three methods. Approaches 1 and 2 give $a \approx 0.5$, which means a roughly constant ~20 tokens/param. The published approach-3 parametric fit gives $a = 0.28/0.62 = 0.45$ (0.46 in the paper), so its tokens/param drifts upward with compute, as the table shows. Epoch AI's replication (Besiroglu et al. 2024) traced this to the fit (an optimizer stopped before convergence, and rounded constants) and found a re-fit consistent with approaches 1 and 2. When someone quotes "20 tokens per parameter", they mean approaches 1 and 2.

### IsoFLOP fitting (Chinchilla's approach 2)

1. Pick 5–9 compute budgets spanning at least 1.5–2 orders of magnitude.
2. At each budget, train 6–10 model sizes, each for $D = C/(6N)$ tokens, **with its own LR schedule sized to its own run length**.
3. Plot final loss against $\log N$. Fit a parabola; its vertex is $N^*(C)$.
4. Fit a line to $\log N^*$ against $\log C$; the slope is $a$. Do the same for $D^*$.

Simulate it with the lab's loss function and 0.3% noise:

```python
import math, numpy as np
from labs._impl import load
sl = load("labs/08_scaling_laws/test_scaling_laws.py", "solution")
rng = np.random.default_rng(0)
Cs, n_opt = np.array([1e18, 3e18, 1e19, 3e19, 1e20]), []
for C in Cs:
    N = np.geomspace(1e7, 1e10, 25); D = C / (6 * N)
    keep = (D / N > 1) & (D / N < 2000); N, D = N[keep], D[keep]
    L = sl.chinchilla_loss(N, D) * (1 + 0.003 * rng.normal(size=N.size))
    a, b, _ = np.polyfit(np.log(N), L, 2)
    n_opt.append(math.exp(-b / (2 * a)))
print(np.polyfit(np.log(Cs), np.log(n_opt), 1)[0])  # 0.446; true value 0.28/0.62 = 0.452
```

Even with perfect functional form and 0.3% noise, the vertex at the smallest budget is off by 7%. Real sweeps are noisier; report confidence intervals (bootstrap over runs) on the exponent.

### Kaplan vs Chinchilla: why they disagreed

Kaplan et al. (2020) found $N^* \propto C^{0.73}$, $D^* \propto C^{0.27}$: spend most extra compute on parameters. Chinchilla found $\approx C^{0.5}$ each. The disagreement mattered: GPT-3 and Gopher were undertrained by Chinchilla's standard. Porian et al. (2024) reproduced Kaplan's result and removed it by fixing three things:

1. **Last-layer compute.** Kaplan counted non-embedding parameters and FLOPs, omitting the LM head, a large share of real compute for small models.
2. **Warmup duration.** A fixed, long warmup penalized small models.
3. **Scale-dependent optimizer tuning.** Learning rate, batch size and AdamW $\beta_2$ must be tuned per scale; a single setting favours some sizes.

With these corrected they matched Chinchilla. Contrary to a hypothesis in Hoffmann et al., careful learning-rate decay was not essential for the Chinchilla law to hold in their experiments. Pearce and Song (2024) independently attributed much of the gap to non-embedding parameter counting at small scale. Takeaway: a scaling exponent is a property of your whole experimental protocol, not only of the model family.

### Inference-aware sizing

If a model will serve $I$ tokens over its life, minimize $6ND + 2NI$ at a target loss instead of training FLOPs alone (Sardana et al. 2023). Lab 08's `inference_aware_optimum`, target loss 2.2, published fit:

| lifetime inference tokens $I$ | $N$ | $D$ | tokens/param |
|---|---|---|---|
| 0 | 3.5B | 204B | 58 |
| 1e12 | 1.7B | 511B | 295 |
| 1e13 | 0.97B | 1.8T | 1,856 |
| 1e14 | 0.65B | 8.2T | 12,628 |

This is why Llama 3 8B saw ~15T tokens (~1,900 tokens/param): it is far past compute-optimal, and that is correct for a model that will be run billions of times. The cost of over-training is small: for §8's run (6.72B parameters, 2T tokens), the published fit predicts loss 2.023, against 2.016 for the compute-optimal 13.2B model at the same training compute, at half the serving cost.

**What else scales.** Optimal learning rate and batch size also follow power laws in compute (Porian et al. fit both). μP makes the optimal LR nearly width-independent, so you tune on a small proxy. Downstream task metrics scale much more noisily than loss, and discontinuous metrics (exact match) can make smooth loss improvements look like sudden jumps. Predict loss first, then map loss to tasks with a separate fit.

> [!TIP]
> In an interview, derive $N^* \propto C^{\beta/(\alpha+\beta)}$ on the whiteboard in five lines, then say which experimental choices change $\alpha$ and $\beta$. That second part is what separates people who ran sweeps from people who read the abstract.

## 4. The standard recipe (2024–26 era)

| choice | common setting | why |
|---|---|---|
| optimizer | AdamW, $\beta = (0.9, 0.95)$, wd 0.1, clip 1.0; or Muon for matrices plus AdamW for the rest | $\beta_2 = 0.95$ adapts faster to gradient-scale shifts than 0.999 |
| schedule | warmup (~2k steps), then cosine to 10% of peak, or WSD (warmup, stable, decay the last 10–20%) | WSD lets you branch cooldowns and extend training |
| precision | bf16 compute, fp32 master weights and optimizer state; FP8 matmuls at the frontier | bf16 has fp32's exponent range, so no loss scaling |
| init | $\mathcal{N}(0, 0.02)$ with depth scaling, or μP | keeps updates comparable across depth and width |
| batch | ~1–4M tokens early, ramped up during training | critical batch size grows as loss falls |
| sequences | packed, with document masking | no padding waste, no cross-document attention |
| evaluation | held-out loss per data source plus a small benchmark suite every few thousand steps | catches data-source regressions early |

**Mid-training (annealing).** During the final LR decay, upsample the highest-quality data (math, code, instruction-like text, curated long documents). Per token, it moves benchmarks more than anything else, and it is where many labs add capabilities they will build on in post-training.

## 5. Distributed training

| strategy | shards | communication | where |
|---|---|---|---|
| DP | batch | all-reduce grads (overlap with backward) | everywhere |
| ZeRO-1/2/3, FSDP | optimizer / + grads / + params | reduce-scatter + all-gather | memory-bound models |
| TP (Megatron) | each matmul (column then row) | 2 all-reduces per layer per pass | within a node (NVLink) |
| SP | layernorm/dropout activations along the sequence | pairs with TP | long sequences |
| PP | layers into stages | point-to-point activations; bubble `(p−1)/(m+p−1)` | across nodes |
| CP / ring attention | sequence for attention | pass KV blocks around a ring | very long context |
| EP | MoE experts | all-to-all tokens | MoE models |

### Collectives and their cost

For a buffer of $S$ bytes over $n$ ranks, ring algorithms send per rank:

| collective | bytes sent per rank | used by |
|---|---|---|
| all-reduce (= reduce-scatter + all-gather) | $2\frac{n-1}{n}S$ | DP gradients, TP activations |
| reduce-scatter | $\frac{n-1}{n}S$ | ZeRO-2/3 gradients |
| all-gather | $\frac{n-1}{n}S$ | ZeRO-3/FSDP parameters |
| all-to-all | $\frac{n-1}{n}S$ | MoE token dispatch, Ulysses-style sequence parallelism |

The per-rank cost is almost independent of $n$: that is why ring all-reduce scales (lab 09 proves the $2(n-1)/n$ factor). Time ≈ bytes / link bandwidth, plus a latency term per step that matters for small messages.

**Typical H100 links (spec; measure achieved bandwidth with nccl-tests):** NVLink 4 inside a node, 900 GB/s bidirectional per GPU (450 GB/s each way); InfiniBand NDR between nodes, 400 Gb/s = 50 GB/s per GPU per direction. The 9x gap is the reason for the design order below.

### Communication per strategy, with numbers

For the 6.72B model of §8 (bf16, $\Psi = 6.72\times10^9$ parameters, $T = 4096$, micro-batch of 1 sequence, 40% MFU on H100s):

| layout | what moves | bytes per GPU | time | compare with compute |
|---|---|---|---|---|
| DP over 256 GPUs | all-reduce bf16 grads once per step | $2\cdot\frac{255}{256}\cdot 2\Psi$ = 26.8 GB | 0.54 s over IB | step compute 1.67 s: overlappable, but 16 B/param = 107.5 GB per GPU does not fit |
| FSDP full shard over 256 | all-gather params (fwd and bwd) + reduce-scatter grads, per micro-batch | $3\cdot\frac{255}{256}\cdot 2\Psi$ = 40.2 GB | 0.80 s over IB | ×2 micro-batches ≈ the whole step: too tight |
| **HSDP**: shard within the node, replicate across 32 nodes | intra-node FSDP traffic per micro-batch; inter-node all-reduce of each GPU's 1/8 gradient shard per step | 35.3 GB intra + 3.3 GB inter | 0.08 s (NVLink) + 0.07 s (IB) | small; this is §8's choice |
| TP = 8 in a node | 4 all-reduces per layer (2 fwd, 2 bwd) of $T\cdot d\cdot 2$ = 33.5 MB | 7.5 GB per micro-batch | 17 ms over NVLink, 150 ms over IB | compute per micro-batch per GPU: 52 ms. Fine in a node, 3x the compute across nodes |
| PP | activations at each stage boundary | 33.5 MB per micro-batch per direction | < 1 ms | the cost is the bubble: $p = 4$, $m = 16$ gives 3/19 = 16% idle; $m = 32$ gives 8.6% |
| CP = 8 at 128k context | K and V chunks passed around a ring ($H_{kv} = 8$) | 67 MB per hop, 7 hops = 470 MB per layer forward | overlapped with blockwise attention | needed when one sequence's activations do not fit |
| EP (MoE) | two all-to-alls per MoE layer | ≈ $2kd$ elements per token per layer | — | see [04 §6](04-transformers.md#6-mixture-of-experts) |

### Memory

Mixed-precision Adam holds $2 + 2 + 12 = 16$ bytes per parameter: bf16 weights, bf16 grads, and fp32 master weights, momentum and variance. ZeRO shards the $12\Psi$ optimizer state (stage 1), then the grads (stage 2), then the weights (stage 3). For 7.5B parameters on 64 GPUs: 120 → 31.4 → 16.6 → 1.9 GB per GPU (lab 09's `test_zero_paper_numbers`).

Activations are the other half. Korthikanti et al. give, per layer of a GPT-3-style block (MHA, 4d GELU MLP, dropout), with 16-bit activations and no parallelism:

$$\text{activation bytes per layer} = s\,b\,h\left(34 + 5\frac{a\,s}{h}\right),$$

with $s$ sequence length, $b$ micro-batch, $h$ hidden size, $a$ heads. The $5as/h$ term is the attention matrix; FlashAttention never stores it. At $s = h = 4096$, $a = 32$, $b = 1$: 3.25 GB per layer naively, 0.57 GB with FlashAttention (18 GB for 32 layers), 34 MB per layer if you store only each layer's input and recompute the rest (full activation checkpointing, costing an extra forward pass: ~8N instead of 6N FLOPs per token). Llama-style blocks change the constant (SwiGLU stores more, no dropout masks), so treat this as a ±20% estimate.

### Design order

```
1. Does one replica fit?     params+grads+optimizer (16 B/param) + activations
   no  -> shard state inside the node first (FSDP/ZeRO over NVLink), or TP <= GPUs per node
   still no -> PP across nodes (or FSDP across nodes if bandwidth allows)
2. Does one sequence fit?    long context -> SP with TP, then CP
3. Scale throughput          DP (or HSDP replicas) across everything else
4. Check the arithmetic      per-GPU micro-batch big enough to be compute-bound;
                             every collective overlapped with compute or < ~10% of step time
```

Llama 3 405B, for reference: TP 8 inside the node, PP 16, CP for long-context stages, DP outermost, on up to 16k H100s at 38–43% bf16 MFU.

> [!TIP]
> Global batch = DP degree × micro-batch × gradient-accumulation steps. When an interviewer changes the GPU count, recompute this first; a plan that silently changes the global batch changes the optimization, not just the speed.

## 6. Stability at scale

| symptom | likely causes | check | fix |
|---|---|---|---|
| sharp loss spike that recovers | bad or unusual batch interacting with the current state; LR too high | grad-norm spike one or two steps earlier; which layers | rewind ~100 steps, skip the offending batches; lower LR |
| attention entropy collapse, then divergence | attention-logit growth | max $\lvert q\cdot k\rvert$ per layer climbing | QK-norm; lower LR |
| divergence to NaN/inf | output-logit divergence; bf16/FP8 overflow | max logit, $\log Z$ of the output softmax | z-loss ($10^{-4}\log^2 Z$); keep softmax and norms in fp32 |
| loss plateaus higher than a smaller proxy predicted | Adam's $\epsilon$ comparable to gradient RMS in large models; LR not transferred correctly | per-layer grad RMS vs $\epsilon$; update/weight ratio | smaller $\epsilon$; μP or re-tuned LR |
| slow upward drift late in training | LR too high for the current loss scale; data-order shift | loss per data source | shorten the stable phase, start decay; check the loader |
| one rank slower than the rest | straggler GPU, thermal throttling, bad link | per-rank step time | drain the node, restart elastically |

**What large runs report.** PaLM saw loss spikes roughly 20 times during training. Restarting from about 100 steps before the spike and skipping 200–500 batches fixed them, and replaying the same batches from a different checkpoint did not reproduce the spike. The cause was the interaction of particular batches with a particular model state, not "bad data" alone. Wortsman et al. (2023) reproduced attention-logit growth and output-logit divergence in small models at high learning rates, which makes the fixes (QK-norm, z-loss) testable on one GPU.

**The dashboard to watch:** loss (overall and per data source), global and per-layer grad norm, fraction of clipped steps, max attention logit per layer, output $\log Z$, update-to-weight ratio per layer, residual-stream RMS, tokens/s per rank.

**Hardware failures are routine.** In a 54-day snapshot of Llama 3 405B pretraining on up to 16k H100s, there were 466 job interruptions: 47 planned and 419 unexpected, about 78% of them attributed to confirmed or suspected hardware issues, with GPU faults the largest category. That is $419/(54 \times 16{,}384) \approx 4.7\times10^{-4}$ unexpected interruptions per GPU-day. For §8's 256 GPUs over 9.2 days, expect about one. Mitigations: fast asynchronous checkpointing, automated restart, a deterministic data loader so a restarted job sees the same batches, and detection of silent data corruption (a rank whose loss or grad norm disagrees with its replicas).

**How often to checkpoint.** Young's approximation: interval $\tau \approx \sqrt{2\delta M}$ for checkpoint cost $\delta$ and mean time between failures $M$. With $\delta = 1$ min of blocking time and $M \approx 8.4$ days (12,000 min), $\tau \approx 155$ min. Checkpoint every ~2 hours.

## 7. Long context

Train short, then extend, because attention cost grows with $T$: for an 8B model at 128k context, attention is ~68% of training FLOPs versus ~6% at 4k ([04 §5](04-transformers.md#5-parameter-and-flop-accounting)).

1. **Pretrain at 4–8k.** Pick a RoPE base with extension in mind (Llama 3 used 500,000).
2. **Extend in stages.** Raise the base or apply YaRN ([04 §3](04-transformers.md#3-positional-information)), then continue pretraining on long documents. Llama 3 went from 8k to 128k in six stages over ~800B tokens. Xiong et al. (2023) found continued pretraining from a short-context model as effective as training long from scratch, and cheaper.
3. **Data.** Long documents are rare on the web; upsample books, long papers, code repositories and synthetic long-range tasks. Keep short data in the mix so short-context quality does not regress.
4. **Masking.** With packing, a 128k sequence holds many unrelated documents. Llama 3 found cross-document masking had limited effect at standard lengths but mattered for very long sequences.
5. **Systems.** Activation memory grows with $T$: use context parallelism (ring attention) and keep tokens per batch roughly constant (fewer, longer sequences).
6. **Evaluation.** Needle-in-a-haystack is necessary but not sufficient. Use multi-needle, aggregation and variable-tracking tasks (RULER-style), and real long-document QA. Report the *effective* context length, where quality drops, not the configured one.

## 8. Exercise: plan a 7B run on paper

Plan it yourself first; then compare with this worked version. Budget: **2T tokens, 256 H100 80 GB GPUs (32 nodes × 8), NVLink inside nodes, 400 Gb/s InfiniBand per GPU.**

**1. Architecture (6.72B parameters).** $d = 4096$, $L = 32$, $H = 32$, $H_{kv} = 8$, $d_h = 128$, SwiGLU $f = 11{,}008$, $V = 128{,}256$ untied, RoPE base 500,000, RMSNorm, QK-norm, context 4,096.
- Attention per layer: $2\cdot4096^2 + 2\cdot4096\cdot1024 = 41.9$M. MLP: $3\cdot4096\cdot11008 = 135.3$M. Per layer 177.2M; 32 layers: 5.67B.
- Embedding + head: $2\cdot128256\cdot4096 = 1.05$B. **Total 6.72B.**

**2. Tokens.** Chinchilla-optimal for 6.72B is ~134B tokens; we train 2T (~300 tokens/param) because the model will be served heavily (§3, inference-aware). The published fit predicts loss 2.023, vs 2.016 for the compute-optimal 13.2B model at equal compute.

**3. FLOPs.** Per token: $6\times$ matmul parameters (everything except the input embedding lookup, 6.20B) = 37.2 GFLOP, plus attention $6LTd$ = 3.2 GFLOP, total 40.4 GFLOP (the familiar $6N$ = 40.3 GFLOP lands in the same place by coincidence). **Total $C$ = 40.4e9 × 2e12 = 8.08e22 FLOPs.**

**4. Time and cost.** 256 × 989 TFLOP/s (bf16 dense) × 40% MFU = 1.01e17 FLOP/s → 7.98e5 s = **9.2 days, 56.7k GPU-hours**; add ~10% for restarts, evals and checkpoints. Price it with your provider's current rate (at an assumed $2–3 per H100-hour, roughly $125k–190k; cloud prices change often, so re-check).

**5. Batch and schedule.** Global batch 1,024 sequences × 4,096 = 4.19M tokens → 476,837 steps at 1.67 s/step. AdamW (0.9, 0.95), wd 0.1, clip 1.0, peak LR 3e-4 (Llama 2 7B's value; confirm with a μP or small-proxy sweep), 2k warmup steps, WSD with the final ~15% as a decay that upsamples high-quality data.

**6. Memory per GPU.** Model state 16 B/param = 107.5 GB → sharded 8 ways inside the node: 13.4 GB. Activations with FlashAttention at micro-batch 1: ~18–22 GB. Logits: $4096 \times 128{,}256$ in fp32 is 2.1 GB (use a chunked or fused cross-entropy). Gathered layer weights and buffers: ~1–2 GB. Total ~40 GB of 80. Choose **micro-batch 2 with selective recomputation, gradient accumulation 2**: 256 × 2 × 2 = 1,024 sequences.

**7. Parallel layout.** HSDP: FSDP full-shard within each node, 32 replicas across nodes. No TP or PP needed at this size and context. Communication (§5): 0.08 s intra-node per micro-batch and 0.07 s inter-node per step, both overlappable with 1.67 s of compute. Add TP = 2 or CP only for a later 32k+ context stage.

**8. Data.** An illustrative 2T mix: 55% filtered web (FineWeb-Edu/DCLM-style), 20% code, 10% math and science, 8% multilingual (including Indic languages, with a tokenizer checked on them), 5% books and papers, 2% instruction-like data saved for the decay phase. Check epochs per source (≤ ~4), decontaminate against every eval you will report, and store ids as `uint32` (8 TB).

**9. Checkpoints, evals, failures.** Full state for restart is ~94 GB (fp32 master + Adam moments + bf16 weights); write it asynchronously every ~2 h (~4,300 steps; §6's Young estimate is 155 min). Expect about one hardware interruption. Evaluate held-out loss per source every ~5k steps plus a small benchmark suite; keep the data loader deterministic so any spike can be replayed.

**10. De-risk first.** Spend ~5–10% of compute on proxies: 100–400M models on 10–30B tokens to set the LR, check the mixture and run a mini IsoFLOP sweep. Run the full pipeline for one day at full scale before committing.

Now defend every number in an interview. Likely challenges:

| challenge | answer |
|---|---|
| "Why not 13B, the compute-optimal size?" | Inference cost; the fit says we lose ~0.007 nats for half the serving FLOPs. |
| "Why is MFU only 40%?" | Communication, memory-bound ops (norms, softmax, cross-entropy), pipeline gaps and restarts. 40% is realistic for a well-tuned bf16 job. |
| "Half the GPUs?" | 18.5 days; either keep the global batch by doubling gradient accumulation, or halve it and re-tune the LR. |
| "Context 32k from the start?" | Attention FLOPs grow from 8% to ~40% of the total, activation memory 8x; do a staged extension instead. |
| "What if the loss spikes on day 6?" | Check grad norm and max-logit history, rewind to the last checkpoint, skip the batches, lower LR if it repeats; QK-norm and z-loss make it rarer. |

---

## Interview traps

- **"20 tokens per parameter is a law."** It is the compute-optimal ratio from Chinchilla's approaches 1–2 under their protocol, for training compute only. Inference-heavy models should train far longer.
- **Using Chinchilla's published $A, B, \alpha, \beta$ and expecting 20 tokens/param.** The approach-3 fit gives 40–100 at common budgets (see the table).
- **"Kaplan was wrong because of the LR schedule."** Porian et al. found last-layer FLOPs, warmup and per-scale tuning explain it; careful LR decay was not essential.
- **Counting 6N and ignoring attention** at long context, or counting the input embedding as FLOPs.
- **"FSDP costs the same as DP."** Same bytes for ZeRO-1/2; ZeRO-3/FSDP adds a parameter all-gather in the backward pass: 1.5x DP traffic, repeated per micro-batch.
- **Putting TP across nodes.** Its all-reduces sit on the critical path of every layer; over InfiniBand they cost more than the compute.
- **Global dedup is always better.** FineWeb's result says otherwise.
- **Forgetting that data epochs differ per source.** A "2T-token" run can still see its math data 8 times.
- **Treating hardware failures as rare events.** At thousands of GPUs they happen daily; the restart path is part of the design.

## CPU vs GPU notes

- **CPU is enough for** everything in §3 and §5's arithmetic: lab 08 (fits and closed forms) and lab 09 (simulated collectives, ZeRO memory, bubbles) are NumPy. Real multi-process DDP also runs on CPU with PyTorch's `gloo` backend.
- **An 8 GB GPU is enough for** a small IsoFLOP study: lab 08's S2 trains 1–30M-parameter models on TinyStories. On CPU, shrink to models under ~2M parameters and a few million tokens per run, and expect the fitted exponents to be noisier.
- **What you cannot do locally:** measure real interconnect costs. Use §5's formulas with published link speeds, and when you get cluster access, run nccl-tests before anything else.

## Check yourself

<details><summary>1. Derive P(candidate) for MinHash-LSH and pick (b, r) for a 0.8 threshold with 128 hashes.</summary>

A band matches with probability $s^r$; the pair is a candidate if any of $b$ bands matches: $1-(1-s^r)^b$. The threshold is near $(1/b)^{1/r}$. With $b\cdot r = 128$: $(b, r) = (8, 16)$ gives $(1/8)^{1/16} = 0.88$, $(16, 8)$ gives $(1/16)^{1/8} = 0.71$; so for 0.8 use something in between, e.g. $(11, 11)$ with 121 hashes, which gives $(1/11)^{1/11} = 0.80$.
</details>

<details><summary>2. Derive the compute-optimal N for L = E + A/N^α + B/D^β under C = 6ND.</summary>

Substitute $D = C/(6N)$, differentiate, set to zero: $\alpha A N^{-\alpha} = \beta B (C/6N)^{-\beta}$, so $N^{\alpha+\beta} = \frac{\alpha A}{\beta B}(C/6)^\beta$ and $N^* = G(C/6)^{\beta/(\alpha+\beta)}$ with $G = (\alpha A/\beta B)^{1/(\alpha+\beta)}$.
</details>

<details><summary>3. Give the three causes Porian et al. found for the Kaplan–Chinchilla discrepancy.</summary>

Not counting last-layer (LM-head) compute, a warmup too long for small models, and not tuning optimizer hyperparameters per scale. With these fixed, a Kaplan-style study reproduces Chinchilla.
</details>

<details><summary>4. A 50B-token Telugu corpus gets 5% weight in a 4T-token run. Is that a problem?</summary>

$0.05 \times 4\text{T}/50\text{B} = 4$ epochs, right at the ~4-epoch point where repetition starts losing value. Acceptable, but do not raise the weight without more unique data (or better filtering and rephrasing of what exists).
</details>

<details><summary>5. Why does TP stay inside a node while DP can span the cluster?</summary>

TP does several all-reduces of activations per layer on the critical path (for §8's model, 7.5 GB per micro-batch per GPU, which is 17 ms over NVLink but 150 ms over InfiniBand, against 52 ms of compute). DP all-reduces gradients once per step and overlaps them with the backward pass.
</details>

<details><summary>6. Compute ZeRO-3 memory per GPU for a 70B model on 512 GPUs, and name what it leaves out.</summary>

$16 \times 70\text{e}9 / 512 = 2.19$ GB of model state per GPU. It leaves out activations, temporary gathered parameters for the current layer, communication buffers and fragmentation, which usually dominate at that point.
</details>

<details><summary>7. Pipeline with p = 8 stages: how many micro-batches for a bubble under 10%?</summary>

$(p-1)/(m+p-1) < 0.1 \Rightarrow 7 < 0.1(m+7) \Rightarrow m > 63$. Use $m \ge 64$ micro-batches per step, which constrains the micro-batch size given the global batch. Interleaved schedules reduce the bubble further.
</details>

<details><summary>8. PaLM's loss spikes did not reproduce when replayed from an earlier checkpoint. What does that tell you, and what do you do?</summary>

The spike came from the interaction of those batches with that model state, not from corrupt data alone. Rewind to before the spike, skip the batches around it and continue; if spikes recur, reduce the LR or add QK-norm/z-loss. Keep the data order deterministic so you can do this.
</details>

<details><summary>9. Why is attention 68% of training FLOPs at 128k context for an 8B model?</summary>

Matmul FLOPs are $6N \approx 48$ GFLOP per token regardless of context, but attention is $6LTd = 6\cdot32\cdot131072\cdot4096 \approx 103$ GFLOP per token. $103/151 \approx 68\%$.
</details>

<details><summary>10. Your 256-GPU run should checkpoint how often, and why not every 10 minutes?</summary>

Young: $\tau \approx \sqrt{2\delta M} = \sqrt{2 \cdot 1 \cdot 12{,}000} \approx 155$ min for a 1-minute blocking checkpoint and a 12,000-minute mean time between failures. Checkpointing every 10 minutes would waste 10% of the run in blocking writes to save, on average, a few minutes of recomputation per failure.
</details>

## Visual guides

- Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook): DP, ZeRO, TP, SP, CP, PP and EP with memory and communication diagrams and measured benchmarks. The best companion to §5.
- Google DeepMind, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): rooflines, collectives and sharding worked from first principles (TPU-centric, but the arithmetic transfers to GPUs).

More per-topic visuals are collected in the [library](../library/README.md).

## Read next

- [06 Post-training](06-post-training.md): what happens to the base model next.
- [08 Evaluation and research](08-evaluation-and-research.md): how to run the ablations this module keeps asking for.
- [Lab 08](../labs/08_scaling_laws/README.md) and [lab 09](../labs/09_parallelism/README.md) to make §3 and §5 concrete.

**Papers** (full list in [papers.md](papers.md)): Kaplan et al. ([2001.08361](https://arxiv.org/abs/2001.08361)); Hoffmann et al., Chinchilla ([2203.15556](https://arxiv.org/abs/2203.15556)); Besiroglu et al., Chinchilla replication ([2404.10102](https://arxiv.org/abs/2404.10102)); Porian et al., resolving the discrepancies ([2406.19146](https://arxiv.org/abs/2406.19146)); Sardana et al., beyond Chinchilla-optimal ([2401.00448](https://arxiv.org/abs/2401.00448)); Muennighoff et al., data-constrained scaling ([2305.16264](https://arxiv.org/abs/2305.16264)); Penedo et al., FineWeb ([2406.17557](https://arxiv.org/abs/2406.17557)); Li et al., DCLM ([2406.11794](https://arxiv.org/abs/2406.11794)); Lee et al., deduplication ([2107.06499](https://arxiv.org/abs/2107.06499)); Xie et al., DoReMi ([2305.10429](https://arxiv.org/abs/2305.10429)); Shoeybi et al., Megatron-LM ([1909.08053](https://arxiv.org/abs/1909.08053)); Rajbhandari et al., ZeRO ([1910.02054](https://arxiv.org/abs/1910.02054)); Korthikanti et al., activation recomputation ([2205.05198](https://arxiv.org/abs/2205.05198)); Liu et al., Ring Attention ([2310.01889](https://arxiv.org/abs/2310.01889)); Zhao et al., PyTorch FSDP ([2304.11277](https://arxiv.org/abs/2304.11277)); Chowdhery et al., PaLM ([2204.02311](https://arxiv.org/abs/2204.02311)); Wortsman et al., small-scale proxies ([2309.14322](https://arxiv.org/abs/2309.14322)); Xiong et al., effective long-context scaling ([2309.16039](https://arxiv.org/abs/2309.16039)); Llama 3 herd ([2407.21783](https://arxiv.org/abs/2407.21783)); OLMo 2 ([2501.00656](https://arxiv.org/abs/2501.00656)).
