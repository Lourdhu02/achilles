# Transformers — Deep Dive

The single most important architecture for your career. Master this file.

For LLM-specific topics (RAG, agents, fine-tuning, inference), see `04-genai-llm-mastery/`.

---

## The full transformer (encoder-decoder; original 2017)

```
Encoder block (×N):
  ┌─ Multi-Head Self-Attention ─┐
  │           +                  │  (residual)
  │       LayerNorm              │
  └──────────────────────────────┘
           ↓
  ┌─ Feed-Forward (2 linear)    ┐
  │           +                  │  (residual)
  │       LayerNorm              │
  └──────────────────────────────┘

Decoder block (×N):
  ┌─ Masked Self-Attention      ┐
  │       + LayerNorm            │
  └──────────────────────────────┘
           ↓
  ┌─ Cross-Attention (Q from dec, K,V from enc) ┐
  │       + LayerNorm                            │
  └──────────────────────────────────────────────┘
           ↓
  ┌─ Feed-Forward + LayerNorm   ┐
  └──────────────────────────────┘
```

Modern decoder-only LLMs (GPT, Llama) use ONLY the decoder side with masked self-attention.

---

## Scaled dot-product attention (THE core operation)

```
Attention(Q, K, V) = softmax( QK^T / √d_k ) V
```

### Step by step:

Given:
- Q (queries): shape `[seq_len, d_k]`
- K (keys): shape `[seq_len, d_k]`
- V (values): shape `[seq_len, d_v]`

Steps:
1. **Compute attention scores:** `QK^T` → `[seq_len, seq_len]`. Each entry [i,j] is "how much should query i attend to key j".
2. **Scale by √d_k:** prevents large dot products from saturating softmax.
3. **Apply mask (if causal):** set future positions to `-inf` before softmax.
4. **Softmax over rows:** convert scores to probabilities; each row sums to 1.
5. **Multiply by V:** weighted sum of values → output, shape `[seq_len, d_v]`.

### Why the √d_k scaling?
Without scaling, for large d_k, dot products have variance d_k. Large scores → softmax saturates → near-zero gradients. Scaling normalizes variance.

### Implementing single-head attention in PyTorch
```python
import torch
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    """
    Q, K: (batch, seq_len, d_k)
    V:    (batch, seq_len, d_v)
    mask: (seq_len, seq_len) or None
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (batch, seq, seq)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ V  # (batch, seq, d_v)
```

---

## Multi-Head Attention (MHA)

Idea: instead of one attention, run H attentions in parallel with different learned projections, then concatenate.

```
MultiHead(X) = concat(head_1, head_2, ..., head_h) · W_O

where head_i = Attention(X·W_Q_i, X·W_K_i, X·W_V_i)
```

In practice:
- `d_model` is the model dim (e.g., 4096)
- `h` heads, each with dim `d_k = d_model / h`
- Implementation: project full d_model, reshape into (h, d_k), apply attention in parallel

### Why multi-head?
- Different heads can attend to different "kinds of relationships" (syntactic, semantic, positional)
- More expressive than single big head

### Implementation (efficient)
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)  # combined for efficiency
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, d_k)
        
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.W_o(out)
```

---

## Variants of multi-head attention (modern)

### MHA (vanilla)
- Each head has its own Q, K, V projections
- # KV heads = # Q heads = h
- Standard, but inference-memory-heavy (KV cache scales with h)

### MQA (Multi-Query Attention)
- All heads share a SINGLE K and V
- # KV heads = 1, # Q heads = h
- Drastically reduces KV cache memory
- Slight accuracy drop in some cases

### GQA (Grouped-Query Attention)
- K, V are shared in groups
- E.g., # KV heads = 8, # Q heads = 32 (4:1 ratio)
- Compromise between MHA and MQA
- Llama 2 70B, Llama 3, many modern models

### MLA (Multi-Latent Attention) — DeepSeek
- Compress K, V into a smaller latent space, then expand
- Reduces KV cache further
- Reportedly maintains quality

---

## Positional Encoding

Transformers are order-invariant by design (attention is permutation-equivariant). Position info must be injected.

### Sinusoidal (original paper)
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Added to embeddings before first layer.

Why sinusoidal? In theory, allows model to attend to relative positions via `PE(pos+k) = linear combination of PE(pos)`.

### Learned absolute positional embeddings
- Per-position learned vector
- Limited to training-seen positions (no length generalization)
- Used in BERT, GPT-1/2

### Relative position encoding (T5-style)
- Bias added to attention scores based on relative distance i-j
- Generalizes to longer sequences

### RoPE (Rotary Position Embedding) — MOST POPULAR NOW
- Rotates Q and K by position-dependent angle
- For dim pair (2i, 2i+1): rotation by `m·θ_i` at position m
- Encodes relative position implicitly
- Used in GPT-Neo, Llama 1/2/3, most modern models
- Allows length extrapolation (somewhat)

### ALiBi (Attention with Linear Biases)
- Don't add to embeddings; add to attention scores: `score_ij - m·|i-j|` where m is head-specific slope
- No positional embedding needed
- Strong length generalization

---

## Feed-Forward Network (FFN)

The other half of a transformer block. After attention, each token is independently processed:

```
FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2
```

Typical dim: hidden = 4 · d_model.
Most params of a transformer are in FFN, not attention!

### Modern variants
- **SwiGLU:** `FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down`
- Used in PaLM, Llama, etc.
- 3 matrices instead of 2; better performance per param

---

## LayerNorm placement: Pre-norm vs Post-norm

### Post-norm (original 2017)
```
x → SubLayer(x) → +x (residual) → LayerNorm → ...
```
Hard to train deeply; gradient flow issues.

### Pre-norm (modern, standard)
```
x → LayerNorm(x) → SubLayer → +x (residual) → ...
```
Much more stable for deep networks. Standard in GPT-2 onwards.

### RMSNorm
Simplified LayerNorm: only scale by RMS, no centering or shift.
```
RMSNorm(x) = (x / RMS(x)) · γ
where RMS(x) = sqrt(mean(x²))
```
- Faster (1 less stat to compute)
- Used in Llama, etc.

---

## Embeddings

### Token embedding
- Look-up table: vocab_size × d_model
- Each token ID → dense vector
- Standard initialization: ~N(0, 1/√d_model)

### Output projection (LM head)
- Often **tied** to input embedding (W_out = W_in^T)
- Saves params, sometimes regularizes

### Embedding scaling
- Original paper scales embeddings by √d_model before adding positional encoding
- Modern: often skip this

---

## Causal Masking (for decoder/GPT-style)

Make attention triangular: token i can only attend to tokens 0..i.

```python
mask = torch.tril(torch.ones(T, T))  # lower triangular, T = seq_len
# mask[i, j] = 1 if j <= i, else 0
scores = scores.masked_fill(mask == 0, float('-inf'))
```

Why: training in parallel for next-token prediction, while ensuring no "future leakage".

---

## Training dynamics

### Pre-training (causal LM)
- Objective: predict next token, average cross-entropy loss
- Massive datasets (CommonCrawl, Wiki, code, etc.)
- Trillions of tokens
- Months of TPU/GPU time

### What's "scaling laws"?
- Chinchilla paper (Hoffmann et al., 2022): for compute budget C:
  - Optimal model size N ∝ √C
  - Optimal data D ∝ √C
- Old wisdom (GPT-3) under-trained models on data
- New wisdom (Chinchilla): much more data, comparatively smaller models

### Key hyperparameters for transformer training
- Learning rate (peak): 1e-4 to 6e-4 typical for AdamW
- Warmup steps: 1-5% of total steps
- LR schedule: cosine to 10% of peak
- Batch size: as large as fits + gradient accumulation; "global batch" up to 4M tokens
- Weight decay: 0.1 typical
- Gradient clipping: norm 1.0 standard
- Dropout: 0.0 - 0.1 (decreasing in modern very large models — they don't need it)

---

## Inference (covered deeper in `04-genai-llm-mastery/05-inference-optimization.md`)

### Two phases
1. **Prefill:** process entire prompt in parallel (compute-bound)
2. **Decode:** generate tokens one at a time (memory-bound; KV cache crucial)

### KV cache
- Each generated token, recompute K and V for ONLY that token, append to cached K and V from prior tokens
- Avoids recomputing prior positions
- Memory cost: O(2 · n_layers · seq_len · d_model)
- For 70B model with 32k context: ~80GB for KV cache alone

---

## Common interview questions on transformers

These appear in EVERY top company:

1. **"Walk me through attention. Explain each step."** — must do in 5 min, on whiteboard.
2. **"Why divide by √d_k?"** — variance explanation.
3. **"Why multi-head and not one big head?"** — different representational subspaces.
4. **"Why is layer norm preferred over batch norm here?"** — sequence length variability, no batch dependence.
5. **"How does KV cache work?"** — explain the memory and compute reuse.
6. **"What's the inference complexity of a transformer?"** — quadratic in seq length, linear in model dim.
7. **"What is FlashAttention solving?"** — IO complexity, materializing the n×n attention matrix.
8. **"What's the difference between GPT and BERT architecturally?"** — decoder only vs encoder only, causal vs bidirectional.

Drill these until you can answer each in 90 seconds, clear and complete.

---

## Implementation exercise (do this in Month 2-3)

Write a from-scratch transformer (decoder-only, GPT-style) in numpy or PyTorch:

```
[ ] Tokenizer (BPE — use a library; understand inputs/outputs)
[ ] Embedding (token + positional)
[ ] Attention (single head, then multi-head)
[ ] FFN with GELU/SwiGLU
[ ] LayerNorm / RMSNorm
[ ] Decoder block stacking
[ ] LM head (tied with embedding)
[ ] Training loop on a small dataset (Tiny Shakespeare)
[ ] Generation with temperature/top-k
```

This single exercise will teach you more than 20 hours of reading. Andrej Karpathy's "nanoGPT" is the reference implementation.

---

## Resources

- **"Attention is All You Need"** (original paper, 2017) — read it
- **The Annotated Transformer** (Harvard NLP) — line-by-line code walkthrough
- **The Illustrated Transformer** (Jay Alammar) — best visuals
- **nanoGPT** (Karpathy, GitHub) — read every line, modify, retrain
- **"Let's build GPT from scratch"** (Karpathy YouTube) — 2-hour video, lifetime value
- **Llama 3 architecture paper** — modern transformer reference

---

End. Next file: `05-math-essentials.md`
