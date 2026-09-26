# Coding interviews

Three kinds of coding round show up in core-AI loops: ML coding (implement a model component from memory), practical engineering (build a small, growing system in a real editor) and classic data structures and algorithms.
This page has timed drills for the first two with solution sketches and the bugs interviewers watch for; DSA has its own page, [dsa-patterns.md](dsa-patterns.md).

## Contents

- [How these rounds are scored](#how-these-rounds-are-scored)
- [ML coding drills](#ml-coding-drills) (18 drills with sketches)
- [Practical engineering drills](#practical-engineering-drills) (7 drills with sketches)
- [DSA](#dsa)
- [A four-week drill plan](#a-four-week-drill-plan)

## How these rounds are scored

Interviewers across companies look for roughly the same things, in roughly this order:

1. **Correctness,** including edge cases you raise yourself (empty input, padding, numerical overflow, ties).
2. **Clarity:** names, small functions, shapes written in comments. Someone should be able to read your code once.
3. **Communication:** you state the plan before typing, narrate decisions, and test as you go.
4. **Speed:** finishing the core early leaves time for the extensions where strong candidates separate themselves.
5. **Depth when pushed:** complexity, memory, numerical stability, and what changes at scale or on a GPU.

**Rules for every drill:** write from memory with no references; start the timer; state the input and output shapes first; write a tiny test (a hand-computed case or a comparison with PyTorch) before calling it done; finish by stating complexity and one failure mode.

> [!TIP]
> Say the shapes out loud and write them as comments: `# q: (B, H, T, hd)`. Most ML coding bugs are shape or axis bugs, and a comment makes them visible to you and the interviewer.

> [!TIP]
> After each drill, write down the bug you made in a "bug log". After a month, the log shows your personal failure modes, and those are what to drill.

---

## ML coding drills

| # | Drill | Target | Source |
|---|---|---|---|
| 1 | [Softmax cross-entropy, forward and backward](#1-softmax-cross-entropy-forward-and-backward) | 10 min | [lab 01](../../labs/01_autograd/README.md) |
| 2 | [MLP with manual backprop and SGD](#2-mlp-with-manual-backprop-and-sgd) | 20 min | [lab 01](../../labs/01_autograd/README.md) |
| 3 | [AdamW step](#3-adamw-step) | 10 min | [lab 02](../../labs/02_training_core/README.md) |
| 4 | [Causal multi-head attention with GQA](#4-causal-multi-head-attention-with-gqa) | 15 min | [lab 05](../../labs/05_transformer/README.md) |
| 5 | [RoPE](#5-rope) | 10 min | [lab 05](../../labs/05_transformer/README.md) |
| 6 | [KV-cache decode loop](#6-kv-cache-decode-loop) | 20 min | [lab 07](../../labs/07_kv_cache_sampling/README.md) |
| 7 | [Top-k, top-p and min-p sampling](#7-top-k-top-p-and-min-p-sampling) | 10 min | [lab 07](../../labs/07_kv_cache_sampling/README.md) |
| 8 | [Beam search with length penalty](#8-beam-search-with-length-penalty) | 25 min | extends lab 07 |
| 9 | [BPE training and encoding](#9-bpe-training-and-encoding) | 30 min | [lab 04](../../labs/04_tokenizer/README.md) |
| 10 | [Online softmax and tiled attention](#10-online-softmax-and-tiled-attention) | 20 min | [lab 06](../../labs/06_attention_kernels/README.md) |
| 11 | [LoRA layer with merge](#11-lora-layer-with-merge) | 10 min | [lab 10](../../labs/10_lora/README.md) |
| 12 | [DPO loss with masking](#12-dpo-loss-with-masking) | 10 min | [lab 11](../../labs/11_dpo/README.md) |
| 13 | [GRPO advantages and clipped loss](#13-grpo-advantages-and-clipped-loss) | 15 min | [lab 12](../../labs/12_grpo/README.md) |
| 14 | [Group-wise INT4 quantization](#14-group-wise-int4-quantization) | 10 min | [lab 13](../../labs/13_quantization/README.md) |
| 15 | [Speculative decoding acceptance](#15-speculative-decoding-acceptance) | 15 min | [lab 14](../../labs/14_speculative_decoding/README.md) |
| 16 | [Bootstrap CI and pass@k](#16-bootstrap-ci-and-passk) | 10 min | [lab 15](../../labs/15_eval_stats/README.md) |
| 17 | [BM25 and nDCG](#17-bm25-and-ndcg) | 20 min | [lab 17](../../labs/17_retrieval/README.md) |
| 18 | [k-means and logistic regression](#18-k-means-and-logistic-regression) | 15 min each | classic |

All sketches assume these imports:

```python
import math, time, random, heapq, bisect, itertools, threading, asyncio
from collections import Counter, OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
```

### 1. Softmax cross-entropy, forward and backward

<details>
<summary>Solution sketch</summary>

```python
def softmax_ce(logits, y):
    """logits: (N, V) float, y: (N,) int. Returns mean loss and dL/dlogits."""
    z = logits - logits.max(axis=1, keepdims=True)           # stability: exp never overflows
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))  # log-softmax
    n = logits.shape[0]
    loss = -logp[np.arange(n), y].mean()
    dlogits = np.exp(logp)
    dlogits[np.arange(n), y] -= 1.0                          # p - onehot(y)
    return loss, dlogits / n                                  # divide by N because of the mean
```

**Common bugs:** no max subtraction (overflow for logits above ≈ 88 in fp32); computing `log(softmax(z))` in two steps (log of 0); forgetting the `/ n`; integer logits; fancy indexing with `y` as a float array.

**Test:** finite differences on a 3×5 example; the gradient rows must sum to zero.

</details>

### 2. MLP with manual backprop and SGD

<details>
<summary>Solution sketch</summary>

```python
def mlp_train_step(params, x, y, lr):
    """params: [W1 (D,H), b1 (H,), W2 (H,C), b2 (C,)], updated in place. x: (N, D), y: (N,)."""
    W1, b1, W2, b2 = params
    h_pre = x @ W1 + b1                     # (N, H)
    h = np.maximum(h_pre, 0.0)              # ReLU
    logits = h @ W2 + b2                    # (N, C)
    loss, dlogits = softmax_ce(logits, y)
    dW2 = h.T @ dlogits                     # (H, C)
    db2 = dlogits.sum(0)
    dh = dlogits @ W2.T                     # (N, H)
    dh_pre = dh * (h_pre > 0)               # ReLU gradient
    dW1 = x.T @ dh_pre
    db1 = dh_pre.sum(0)
    for p, g in zip(params, (dW1, db1, dW2, db2)):
        p -= lr * g
    return loss
```

**Common bugs:** summing the bias gradient over the wrong axis; using `h > 0` instead of `h_pre > 0` (the same for ReLU, wrong for other activations); reassigning `p = p - lr * g`, which does not update the list; initializing weights with std 1 (use ≈ √(2/fan_in)).

**Test:** gradient check against finite differences, then overfit 32 examples to near-zero loss.

</details>

### 3. AdamW step

<details>
<summary>Solution sketch</summary>

```python
def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """In-place update of numpy arrays p, m, v. t is the 1-based step count."""
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)                # bias correction
    v_hat = v / (1 - b2 ** t)
    p *= 1 - lr * wd                         # decoupled weight decay (AdamW, not L2)
    p -= lr * m_hat / (np.sqrt(v_hat) + eps)
```

**Common bugs:** `t` starting at 0 (division by zero in bias correction); adding `wd * p` to the gradient (that is Adam with L2); applying weight decay to norms and biases; `eps` inside the square root (a different optimizer).

**Test:** matches `torch.optim.AdamW` to float precision over several steps.

</details>

### 4. Causal multi-head attention with GQA

<details>
<summary>Solution sketch</summary>

```python
def gqa_attention(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
    """x: (B, T, D). Wq, Wo: (D, D). Wk, Wv: (D, n_kv_heads * hd)."""
    B, T, D = x.shape
    hd = D // n_heads
    q = (x @ Wq).view(B, T, n_heads, hd).transpose(1, 2)        # (B, H, T, hd)
    k = (x @ Wk).view(B, T, n_kv_heads, hd).transpose(1, 2)     # (B, Hkv, T, hd)
    v = (x @ Wv).view(B, T, n_kv_heads, hd).transpose(1, 2)
    rep = n_heads // n_kv_heads
    k = k.repeat_interleave(rep, dim=1)                          # query head h uses KV head h // rep
    v = v.repeat_interleave(rep, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(hd)             # (B, H, T, T)
    causal = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    out = scores.softmax(dim=-1) @ v                             # (B, H, T, hd)
    return out.transpose(1, 2).reshape(B, T, D) @ Wo
```

**Common bugs:** `repeat` instead of `repeat_interleave` (maps query heads to the wrong KV heads when loading real weights); `reshape` straight to `(B, H, T, hd)` without the transpose (scrambles heads and time); mask with `diagonal=0` (a token cannot see itself); softmax over the wrong axis; forgetting `/ sqrt(hd)`.

**Test:** compare with `F.scaled_dot_product_attention(q, k, v, is_causal=True)` on the expanded K and V. **Say:** O(T²·D) time, O(T²) memory per head for the scores, which FlashAttention removes.

</details>

### 5. RoPE

<details>
<summary>Solution sketch</summary>

```python
def rope(x, pos0=0, base=10000.0):
    """x: (..., T, d), d even. Rotates pairs (2i, 2i+1) by angle pos * base^(-2i/d).
    pos0 is the absolute position of the first token (the cache length during decoding)."""
    T, d = x.shape[-2], x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)       # (d/2,)
    pos = torch.arange(pos0, pos0 + T, dtype=torch.float32)
    ang = pos[:, None] * inv_freq[None, :]                                     # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
```

**Common bugs:** mixing the interleaved-pair convention above with the "rotate half" convention (pairs i and i + d/2) used by Llama checkpoints in Hugging Face format: both are valid, but weights trained with one break under the other; forgetting `pos0` during cached decoding (every new token gets position 0); applying RoPE to V; computing angles in bf16 (large positions lose precision).

**Test:** the dot product of `rope(q)` at position m and `rope(k)` at position n depends only on m − n.

</details>

### 6. KV-cache decode loop

<details>
<summary>Solution sketch</summary>

```python
class KVCache:
    def __init__(self, n_layers):
        self.k, self.v = [None] * n_layers, [None] * n_layers

    def update(self, layer, k_new, v_new):               # (B, Hkv, t_new, hd)
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        return self.k[layer], self.v[layer]

    @property
    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]


def attend(q, k, v):
    """q: (B, H, tq, hd); k, v: (B, H, tk, hd), tk >= tq. Query i sits at absolute position tk - tq + i."""
    tq, tk = q.shape[2], k.shape[2]
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    mask = torch.ones(tq, tk, dtype=torch.bool).triu(diagonal=tk - tq + 1)
    return s.masked_fill(mask, float("-inf")).softmax(-1) @ v


@torch.no_grad()
def generate(model, prompt, max_new, sample_fn):
    """model(tokens (B, t), cache, pos0) -> logits (B, t, V), appending to the cache."""
    cache = KVCache(model.n_layers)
    logits = model(prompt, cache, pos0=0)                 # prefill the whole prompt in one pass
    out = []
    for _ in range(max_new):
        nxt = sample_fn(logits[:, -1])                    # (B,)
        out.append(nxt)
        logits = model(nxt[:, None], cache, pos0=cache.length)   # one token per step
    return torch.stack(out, dim=1)
```

**Common bugs:** a causal mask built for `(tq, tq)` instead of offset for `(tq, tk)` (with one query token the mask must allow every cached key); RoPE positions not offset by the cache length; caching K before applying RoPE in one place and after in another; growing the cache with `torch.cat` in production (reallocates every step: preallocate or use paged blocks); feeding the whole sequence again each step (correct but quadratic).

**Test:** logits from cached decoding equal logits from running the full sequence without a cache.

</details>

### 7. Top-k, top-p and min-p sampling

<details>
<summary>Solution sketch</summary>

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, generator=None):
    """logits: (B, V). Returns (B,) token ids. Order: temperature, top-k, min-p, top-p."""
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]              # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    if min_p is not None:
        probs = probs * (probs >= min_p * probs.max(-1, keepdim=True).values)
    if top_p is not None:
        sp, idx = probs.sort(-1, descending=True)
        mass_before = sp.cumsum(-1) - sp
        sp = sp.masked_fill(mass_before >= top_p * sp.sum(-1, keepdim=True), 0.0)  # keep smallest prefix reaching p
        probs = torch.zeros_like(probs).scatter(-1, idx, sp)
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
```

**Common bugs:** top-p that drops the token which crosses the threshold (then a very confident distribution keeps nothing); sorting but scattering back to the wrong indices; temperature applied after the softmax; forgetting to renormalize; top-k ties (`<` keeps all tied values; fine, but say so).

**Test:** `top_k=1` and tiny `top_p` both equal argmax; sampled frequencies on a 5-token vocabulary match the renormalized truncated distribution.

</details>

### 8. Beam search with length penalty

<details>
<summary>Solution sketch</summary>

```python
def beam_search(step_logprobs, bos, eos, beam=4, max_len=20, alpha=0.6):
    """step_logprobs(tokens: list[int]) -> 1D tensor of next-token log-probs.
    Final score = sum of log-probs / lp(length), with the GNMT penalty lp(n) = ((5 + n) / 6) ** alpha."""
    lp = lambda n: ((5 + n) / 6) ** alpha
    alive, finished = [([bos], 0.0)], []
    for _ in range(max_len):
        cand = []
        for seq, s in alive:
            logp = step_logprobs(seq)
            top = logp.topk(min(beam, logp.numel()))
            cand += [(seq + [t], s + l) for l, t in zip(top.values.tolist(), top.indices.tolist())]
        cand.sort(key=lambda c: c[1], reverse=True)
        alive = []
        for seq, s in cand:
            if seq[-1] == eos:
                finished.append((seq, s / lp(len(seq) - 1)))
            else:
                alive.append((seq, s))
            if len(alive) == beam:
                break
        if not alive:
            break
        # Sums only fall and lp only grows, so an alive beam can at best reach s / lp(max_len).
        if finished and max(f[1] for f in finished) >= alive[0][1] / lp(max_len):
            break
    finished += [(seq, s / lp(len(seq) - 1)) for seq, s in alive]
    return max(finished, key=lambda f: f[1])[0]
```

**Common bugs:** comparing finished hypotheses by raw sums (favors short outputs); stopping as soon as `beam` hypotheses finish without checking whether alive ones could still win; expanding finished beams; forgetting that `alpha = 0` means no normalization.

**Test:** with a beam wider than the number of possible sequences, the result equals brute-force search over all sequences. **Say:** O(max_len · beam · V) time with full top-k over V; beam search suits translation and ASR, while chat models use sampling because beam outputs are generic and repetitive.

</details>

### 9. BPE training and encoding

<details>
<summary>Solution sketch</summary>

```python
def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def bpe_train(text, vocab_size):
    """Byte-level BPE on one string (a real tokenizer pre-splits with a regex first; see lab 04)."""
    ids, merges = list(text.encode("utf-8")), {}           # merges: (a, b) -> new id, in learned order
    for new_id in range(256, vocab_size):
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        best = max(pairs, key=pairs.get)                   # ties go to the first pair seen
        merges[best] = new_id
        ids = merge(ids, best, new_id)
    return merges


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = min(set(zip(ids, ids[1:])), key=lambda p: merges.get(p, math.inf))  # earliest-learned merge
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def bpe_decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():                     # insertion order = merge order
        vocab[idx] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

**Common bugs:** encoding by applying the most *frequent* pair in the new text instead of the earliest-learned merge; merging overlapping pairs (`aaa` with pair `aa` must give `[X, a]`); decoding each token to a string separately (breaks multi-byte characters such as Telugu); merges that cross pre-tokenization boundaries.

**Test:** `bpe_decode(bpe_encode(s)) == s` for training text and unseen text in another script. **Say:** this trainer is O(merges × corpus); real trainers count pairs over unique pre-tokenized chunks and update counts incrementally.

</details>

### 10. Online softmax and tiled attention

<details>
<summary>Solution sketch</summary>

```python
def tiled_attention(q, k, v, block=64):
    """One head, non-causal. q: (Tq, d); k, v: (Tk, d). Never forms the (Tq, Tk) score matrix."""
    Tq, d = q.shape
    m = torch.full((Tq, 1), float("-inf"))           # running row max
    l = torch.zeros(Tq, 1)                           # running denominator
    o = torch.zeros(Tq, v.shape[1])                  # unnormalized output
    for j in range(0, k.shape[0], block):
        s = q @ k[j:j + block].T / math.sqrt(d)      # (Tq, block)
        m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        scale = torch.exp(m - m_new)                 # rescale what was accumulated under the old max
        l = l * scale + p.sum(-1, keepdim=True)
        o = o * scale + p @ v[j:j + block]
        m = m_new
    return o / l, m + torch.log(l)                   # output, and the log-sum-exp saved for backward
```

**Common bugs:** forgetting to rescale `o` as well as `l`; dividing by `l` inside the loop; with causal masking, a block in which a row is fully masked gives `m_new = -inf` and `exp(-inf - (-inf)) = nan` (skip such blocks or guard the max).

**Test:** equals `softmax(q kᵀ / √d) v` for several block sizes, including one that does not divide Tk.

</details>

### 11. LoRA layer with merge

<details>
<summary>Solution sketch</summary>

```python
class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                              # frozen base
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # random A
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))  # zero B: starts at the base model
        self.scale = alpha / r
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if not self.merged:
            y = y + (x @ self.A.T @ self.B.T) * self.scale       # never materializes B @ A
        return y

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.scale * (self.B @ self.A)   # (out, r) @ (r, in) = (out, in)
            self.merged = True
```

**Common bugs:** initializing both A and B to zero (gradients of both stay zero); forgetting the `alpha / r` scale in `merge` but not in `forward`; computing `x @ (B @ A).T` (materializes a full d×k matrix per step); merging twice; merging into a quantized base without dequantizing.

**Test:** output equals the base layer at init; after setting B to random values, merged and unmerged outputs match.

</details>

### 12. DPO loss with masking

<details>
<summary>Solution sketch</summary>

```python
def sequence_logprob(logits, labels, mask):
    """logits: (B, T, V) where position t predicts labels[:, t] (already shifted by one).
    mask: (B, T), 1 on response tokens only. Returns (B,) summed log-probs."""
    logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # (B, T)
    return (logp * mask).sum(-1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    """Each input: (B,) summed response log-probs; ref_* computed under torch.no_grad()."""
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -F.logsigmoid(margin).mean()
    accuracy = (margin > 0).float().mean()           # implicit-reward accuracy, a useful metric
    return loss, accuracy
```

**Common bugs:** not shifting (`logits[:, :-1]` predicts `input_ids[:, 1:]`); including prompt or padding tokens in the mask; averaging over tokens instead of summing (a different, length-normalized objective); `torch.log(torch.sigmoid(x))` instead of `logsigmoid` (underflows); gradients flowing into the reference model.

**Test:** loss equals log 2 when policy equals reference; hand-computed margin on a two-example batch.

</details>

### 13. GRPO advantages and clipped loss

<details>
<summary>Solution sketch</summary>

```python
def grpo_advantages(rewards, G, eps=1e-6):
    """rewards: (N * G,), the G samples of each prompt stored consecutively."""
    r = rewards.view(-1, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
    return adv.view(-1)


def grpo_loss(logp, logp_old, logp_ref, adv, mask, clip=0.2, beta=0.04):
    """logp*: (B, T) log-probs of the sampled tokens; adv: (B,); mask: (B, T) response tokens."""
    ratio = torch.exp(logp - logp_old)
    a = adv[:, None]
    pg = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    log_r = logp_ref - logp
    kl = torch.exp(log_r) - log_r - 1                        # k3 estimator, always >= 0
    per_token = pg + beta * kl
    return (per_token * mask).sum() / mask.sum()             # token-level mean over the whole batch
```

**Common bugs:** normalizing advantages across the whole batch instead of within each group; groups not stored consecutively; `logp_old` not detached (or recomputed after the update); `torch.max` instead of `torch.min` in the clipped objective (sign error: the loss is the negative); a per-sequence mean followed by a batch mean (length bias; see [question bank Q70–Q71](question-bank.md#7-post-training-and-rl)).

**Test:** advantages have zero mean per group; with `logp == logp_old == logp_ref` the loss equals the negative token-weighted mean advantage and the KL term is zero.

</details>

### 14. Group-wise INT4 quantization

<details>
<summary>Solution sketch</summary>

```python
def quantize_int4(w, group=128):
    """Symmetric per-group INT4 along the input dimension. w: (out, in), in divisible by group."""
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // group, group)
    scale = (g.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-8)   # map max |w| to 7
    q = torch.clamp(torch.round(g / scale), -8, 7).to(torch.int8)  # int4 range, stored in int8 here
    return q, scale


def dequantize_int4(q, scale):
    return (q.float() * scale).reshape(q.shape[0], -1)
```

**Common bugs:** grouping along the output dimension when the kernel expects input-dimension groups; a zero scale for an all-zero group (division by zero); forgetting that real kernels pack two 4-bit values per byte; measuring only mean error (outliers matter more).

**Test:** elementwise error ≤ scale / 2. **Say:** storage is 4 + 16/128 = 4.125 bits per weight with fp16 scales.

</details>

### 15. Speculative decoding acceptance

<details>
<summary>Solution sketch</summary>

```python
def speculative_accept(p, q, draft, generator=None):
    """p: (k + 1, V) target probabilities at each draft position plus one bonus position;
    q: (k, V) draft probabilities; draft: (k,) tokens sampled from q.
    Returns the tokens to append; their distribution equals sampling from the target alone."""
    out = []
    for i, t in enumerate(draft.tolist()):
        if torch.rand((), generator=generator) < torch.clamp(p[i, t] / q[i, t], max=1.0):
            out.append(t)                                   # accept with probability min(1, p/q)
        else:
            residual = torch.clamp(p[i] - q[i], min=0.0)    # resample from norm(max(0, p - q))
            out.append(torch.multinomial(residual / residual.sum(), 1, generator=generator).item())
            return out
    out.append(torch.multinomial(p[-1], 1, generator=generator).item())   # all accepted: bonus token
    return out
```

**Common bugs:** accepting when the argmaxes match (lossless only for greedy decoding); resampling from p instead of the residual (biases the output toward the draft); forgetting the bonus token; comparing probabilities from different temperatures or top-p settings for target and draft.

**Test:** over many trials, the first returned token's frequencies match p[0] for random p and q.

</details>

### 16. Bootstrap CI and pass@k

<details>
<summary>Solution sketch</summary>

```python
def bootstrap_ci(scores, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of per-item scores."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    idx = rng.integers(0, len(scores), size=(n_boot, len(scores)))
    means = scores[idx].mean(axis=1)
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])


def paired_diff_ci(a, b, **kw):
    """CI for mean(a - b) when both models were scored on the same items."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), **kw)


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k), as a stable product."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
```

**Common bugs:** bootstrapping two models independently when items are shared (use the paired difference); resampling samples instead of items (or items instead of clusters when items share a passage); `1 - (1 - c/n) ** k` (biased); the full `(n_boot, n)` index matrix for large n (chunk it).

**Test:** `pass_at_k(n, c, 1) == c / n`; the CI width is close to 2 × 1.96 × SE for binary scores.

</details>

### 17. BM25 and nDCG

<details>
<summary>Solution sketch</summary>

```python
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [d.lower().split() for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / self.N
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}  # never negative
        self.k1, self.b = k1, b

    def score(self, query, i):
        tf, dl, s = self.tf[i], len(self.docs[i]), 0.0
        for t in query.lower().split():
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=10):
        return sorted(range(self.N), key=lambda i: self.score(query, i), reverse=True)[:k]


def ndcg_at_k(ranked_rels, judged_rels, k):
    """ranked_rels: graded relevance of the returned docs in rank order; judged_rels: all judged docs."""
    dcg = lambda rels: sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = dcg(sorted(judged_rels, reverse=True))
    return dcg(ranked_rels) / ideal if ideal > 0 else 0.0
```

**Common bugs:** the classic Robertson IDF without the `1 +`, which goes negative for terms in over half the documents; no length normalization; computing the ideal DCG from the returned list instead of all judged documents; `log2(i + 1)` with 0-based `i` (divides by zero at rank 1).

**Test:** a document containing the rare query term outranks one with only common terms; a perfect ranking gives nDCG = 1. **Say:** real systems score only documents in the posting lists of the query terms (an inverted index), not all N.

</details>

### 18. k-means and logistic regression

<details>
<summary>Solution sketch</summary>

```python
def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]                  # k-means++ is the better init
    for _ in range(iters):
        assign = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    assign = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)  # assignments for the final centers
    return C, assign


def logreg_gd(X, y, lr=0.1, steps=1000, l2=0.0):
    """Binary logistic regression with mean cross-entropy; y in {0, 1}."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        z = X @ w + b
        p = np.exp(-np.logaddexp(0.0, -z))                       # stable sigmoid
        g = p - y                                                # dL/dz
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b
```

**Common bugs:** k-means: empty clusters producing NaN centers; the (N, k, d) broadcast running out of memory at scale (use ‖x‖² − 2x·c + ‖c‖²). Logistic regression: sigmoid overflow; forgetting to divide by N; regularizing the bias.

**Test:** k-means recovers three well-separated blobs; logistic regression reaches the accuracy of `sklearn` or a closed-form check on separable toy data.

</details>

---

## Practical engineering drills

Many frontier-lab coding rounds are like these: 60–90 minutes in a real editor, a small system whose requirements grow in stages ("now add expiry", "now make it thread-safe", "now it must survive restarts"), judged on clean, working code. The target times cover the first two stages.

| Drill | Target | Typical extensions |
|---|---|---|
| [LRU cache with TTL](#lru-cache-with-ttl) | 20 min | Evict expired entries first; thread safety; size-aware capacity |
| [Token-bucket rate limiter](#token-bucket-rate-limiter) | 15 min | Per-key buckets; retry-after; token-based limits for LLM APIs |
| [Request micro-batcher](#request-micro-batcher) | 30 min | Backpressure; cancellation; pipelining batches |
| [Streaming SSE parser](#streaming-sse-parser) | 20 min | Reconnect with last event id; incremental JSON |
| [Retry with backoff and jitter](#retry-with-backoff-and-jitter) | 10 min | Retry-After; retry budgets; circuit breaker |
| [Key-value store with transactions and snapshots](#key-value-store-with-transactions-and-snapshots) | 30 min | Persistence (write-ahead log); TTL; compaction |
| [Interval scheduler](#interval-scheduler) | 20 min | Cancellation; jitter; running on a thread pool |

> [!TIP]
> Inject the clock (`clock=time.monotonic`) and randomness into anything time-dependent. It makes tests deterministic and shows the interviewer you have tested real systems. Use `time.monotonic`, not `time.time`, for durations.

### LRU cache with TTL

<details>
<summary>Solution sketch</summary>

```python
class LRUCacheTTL:
    def __init__(self, capacity, ttl_s, clock=time.monotonic):
        self.cap, self.ttl, self.clock = capacity, ttl_s, clock
        self.data = OrderedDict()                        # key -> (value, expires_at); oldest first

    def get(self, key):
        item = self.data.get(key)
        if item is None:
            return None
        value, expires_at = item
        if self.clock() >= expires_at:
            del self.data[key]                           # lazy expiry
            return None
        self.data.move_to_end(key)                       # mark most recently used
        return value

    def put(self, key, value):
        self.data[key] = (value, self.clock() + self.ttl)
        self.data.move_to_end(key)
        while len(self.data) > self.cap:
            self.data.popitem(last=False)                # evict least recently used
```

**Watch for:** expired-but-unread entries still count toward capacity (extension: evict expired entries first with a min-heap on expiry, or sweep periodically); thread safety needs a lock around each method, since `move_to_end` mutates on reads; `get` returning `None` is ambiguous if `None` is a valid value (use a sentinel).

</details>

### Token-bucket rate limiter

<details>
<summary>Solution sketch</summary>

```python
class TokenBucket:
    def __init__(self, rate_per_s, burst, clock=time.monotonic):
        self.rate, self.cap, self.clock = rate_per_s, burst, clock
        self.tokens, self.last = float(burst), clock()
        self.lock = threading.Lock()

    def allow(self, cost=1.0):
        """Returns (allowed, retry_after_seconds)."""
        with self.lock:
            now = self.clock()
            self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate)   # refill lazily
            self.last = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True, 0.0
            return False, (cost - self.tokens) / self.rate
```

**Watch for:** refilling with a background thread (unnecessary; compute lazily); integer division in the refill; a cost larger than the burst can never succeed (reject it explicitly). **Extensions:** per-key buckets in a dict with LRU eviction of idle keys; a distributed limiter (an atomic script in a shared store); for LLM APIs, limit tokens per minute where the cost is unknown until the response finishes: reserve an estimate, then reconcile.

</details>

### Request micro-batcher

Collect requests for at most 10 ms or 32 items, whichever comes first, run them as one batch, and return each caller its own result.

<details>
<summary>Solution sketch</summary>

```python
class MicroBatcher:
    def __init__(self, process_batch, max_batch=32, max_wait_s=0.010):
        self.process_batch = process_batch               # async: list[input] -> list[output], same order
        self.max_batch, self.max_wait = max_batch, max_wait_s
        self.queue = asyncio.Queue()
        self.worker = None

    async def submit(self, item):
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]             # wait for the first item
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            items, futs = zip(*batch)
            try:
                results = await self.process_batch(list(items))
                for f, r in zip(futs, results):
                    if not f.done():                     # the caller may have been cancelled
                        f.set_result(r)
            except Exception as e:                       # fail this batch's callers, keep serving
                for f in futs:
                    if not f.done():
                        f.set_exception(e)
```

**Watch for:** starting the 10 ms timer at the wrong moment (it starts when the first item arrives, not when the previous batch ended); one failed batch killing the worker loop; setting a result on a cancelled future (raises); unbounded queues (extension: `asyncio.Queue(maxsize=...)` for backpressure). **Extension:** while one batch runs, collect the next (dispatch `process_batch` as a task), which is what inference servers do.

</details>

### Streaming SSE parser

LLM APIs stream server-sent events. Parse a byte stream delivered in arbitrary chunks.

<details>
<summary>Solution sketch</summary>

```python
class SSEParser:
    """feed(bytes) -> list of {"event", "data", "id"} dicts, however the stream is chunked."""
    def __init__(self):
        self.buf, self.last_id = b"", None
        self.event, self.data = "message", []

    def feed(self, chunk: bytes):
        self.buf += chunk
        events = []
        while (nl := self.buf.find(b"\n")) >= 0:
            line, self.buf = self.buf[:nl].rstrip(b"\r"), self.buf[nl + 1:]
            line = line.decode("utf-8")                  # safe: a UTF-8 character never contains b"\n"
            if line == "":                               # blank line ends an event
                if self.data:
                    events.append({"event": self.event, "data": "\n".join(self.data), "id": self.last_id})
                self.event, self.data = "message", []
            elif line.startswith(":"):
                continue                                 # comment or keep-alive
            else:
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "data":
                    self.data.append(value)
                elif field == "event":
                    self.event = value
                elif field == "id":
                    self.last_id = value                 # persists across events, used to reconnect
        return events
```

**Watch for:** decoding each network chunk as UTF-8 (a multi-byte character split across chunks raises or corrupts; buffer bytes and decode whole lines); assuming one chunk is one event; dropping multi-line `data:` fields; not handling `\r\n`. **Extension:** parse each event's JSON and assemble streamed tool-call arguments that arrive as partial JSON strings.

</details>

### Retry with backoff and jitter

<details>
<summary>Solution sketch</summary>

```python
def retry(fn, *, attempts=5, base=0.1, cap=10.0, retry_on=(TimeoutError, ConnectionError),
          sleep=time.sleep, rng=random.random):
    """Exponential backoff with "full jitter": sleep uniformly in [0, min(cap, base * 2^i)]."""
    for i in range(attempts):
        try:
            return fn()
        except retry_on:
            if i == attempts - 1:
                raise
            sleep(rng() * min(cap, base * 2 ** i))
```

**Watch for:** retrying non-idempotent operations (use idempotency keys); retrying on every exception, including bugs and 4xx errors; no jitter, so thousands of clients retry in lockstep; retrying at every layer of a stack (multiplicative retry storms). **Extensions:** honor a server's Retry-After; a per-client retry budget; a circuit breaker that stops calling a failing dependency for a while; propagating an overall deadline.

</details>

### Key-value store with transactions and snapshots

A classic multi-stage problem: stage 1 `set/get/delete`; stage 2 nested `begin/commit/rollback`; stage 3 point-in-time snapshots.

<details>
<summary>Solution sketch</summary>

```python
class KVStore:
    """Stages 1 and 2: each open transaction keeps an undo log of first-touched keys."""
    _MISSING = object()

    def __init__(self):
        self.data, self.undo = {}, []                    # undo: stack of {key: previous value}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def _record(self, key):
        if self.undo and key not in self.undo[-1]:
            self.undo[-1][key] = self.data.get(key, self._MISSING)

    def set(self, key, value):
        self._record(key)
        self.data[key] = value

    def delete(self, key):
        self._record(key)
        self.data.pop(key, None)

    def begin(self):
        self.undo.append({})

    def rollback(self):
        if not self.undo:
            raise RuntimeError("no open transaction")
        for key, prev in self.undo.pop().items():
            if prev is self._MISSING:
                self.data.pop(key, None)
            else:
                self.data[key] = prev

    def commit(self):
        if not self.undo:
            raise RuntimeError("no open transaction")
        inner = self.undo.pop()
        if self.undo:                                    # nested: the parent keeps its older values
            for key, prev in inner.items():
                self.undo[-1].setdefault(key, prev)


class SnapshotKV:
    """Stage 3: every write gets a version; reads can ask for the value as of a snapshot."""
    def __init__(self):
        self.hist, self.version = {}, 0                  # key -> ([versions], [values])

    def set(self, key, value):                           # use a tombstone value for deletes
        self.version += 1
        versions, values = self.hist.setdefault(key, ([], []))
        versions.append(self.version)
        values.append(value)

    def snapshot(self):
        return self.version

    def get(self, key, at=None):
        versions, values = self.hist.get(key, ([], []))
        i = bisect.bisect_right(versions, self.version if at is None else at) - 1
        return values[i] if i >= 0 else None
```

**Watch for:** copying the whole dictionary on `begin` (O(n) per transaction; the undo log is O(keys touched)); a nested commit that overwrites the parent's older undo value; deletes inside transactions; unbounded version history (extension: drop versions older than the oldest live snapshot). **Say:** get is O(log v) per key with v versions.

</details>

### Interval scheduler

Run registered functions every N seconds; `run_pending()` is called by a loop.

<details>
<summary>Solution sketch</summary>

```python
class IntervalScheduler:
    def __init__(self, clock=time.monotonic):
        self.clock, self.heap, self.seq = clock, [], itertools.count()

    def every(self, period_s, fn):
        heapq.heappush(self.heap, (self.clock() + period_s, next(self.seq), period_s, fn))

    def run_pending(self):
        now = self.clock()
        while self.heap and self.heap[0][0] <= now:
            due, _, period, fn = heapq.heappop(self.heap)
            try:
                fn()
            finally:                                     # a failing job is still rescheduled
                nxt = due + period                       # schedule from the due time: no drift
                if nxt <= now:
                    nxt = now + period                   # fell behind: skip missed runs, no burst
                heapq.heappush(self.heap, (nxt, next(self.seq), period, fn))
```

**Watch for:** heap tuples that compare functions when times tie (the sequence counter prevents it); scheduling from `now` (drifts) vs from `due` (does not); catch-up bursts after a pause; long jobs blocking the loop (extension: run them on a thread pool and skip a job whose previous run is still going).

</details>

---

## DSA

Big Tech loops (and some labs) include classic algorithm rounds: typically one or two medium problems in 45 minutes.

- **Target:** a LeetCode medium in ≤ 25 minutes with narration: clarify, examples, brute force and its complexity, the better approach, code, test, complexity.
- **Volume:** ≈ 150 well-understood problems beat 500 skimmed ones. Log each with its pattern and the insight you missed.
- **Maintenance:** 3–4 hours a week once you reach the target.
- **Plan and patterns:** [dsa-patterns.md](dsa-patterns.md), including a 12-week plan that fits around a full-time job.

> [!TIP]
> When stuck, say what you know: "Brute force is O(n²) because of the pair loop; to do better I need the complement quickly, so a hash map." Interviewers give hints to people who show their reasoning, and hints taken well still score.

---

## A four-week drill plan

For someone with evenings and weekends; each week also keeps 3–4 hours of DSA ([plan](dsa-patterns.md#a-12-week-plan-for-people-with-full-time-jobs)).

| Week | ML coding (timed, from memory) | Practical drill | Review |
|---|---|---|---|
| 1 | Drills 1–5: softmax CE, MLP, AdamW, GQA attention, RoPE | LRU with TTL; token bucket | Redo your two slowest drills |
| 2 | Drills 6–10: KV cache, sampling, beam search, BPE, tiled attention | Micro-batcher; SSE parser | Bug-log review |
| 3 | Drills 11–18: LoRA, DPO, GRPO, INT4, speculative, stats, BM25, classics | Retry; KV store with transactions; scheduler | One mock with a peer |
| 4 | Random drills from the list, two per session, under target time | One full 90-minute multi-stage problem | Second mock; rewrite anything over time |

Pass criterion: every drill under its target time, with a test, twice in a row on different days.

For how these rounds fit into a full loop, see [interview-loops.md](interview-loops.md); for company-specific emphasis, the [company guides](../../companies/README.md).
