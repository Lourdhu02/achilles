# LLM Internals — The Real Deep Dive

The single most important file in this folder. If you can answer everything here cold, you'll pass any GenAI engineering interview's technical bar.

---

## 1. Tokenization

### Why tokenize at all
LLMs operate on sequences of integers (token IDs). The choice of tokenizer affects:
- Vocabulary size (memory + LM head dim)
- Sequence length (input/output cost)
- Language coverage
- Subword granularity

### Algorithms

#### BPE (Byte-Pair Encoding)
Original from 1994 (data compression), adapted for NLP in 2015.

Training:
1. Start with character-level vocabulary
2. Find most-frequent adjacent pair (e.g., "th")
3. Merge into new token, add to vocab
4. Repeat until vocab size hit

Encoding: greedy left-to-right, applying merges learned in training.

Used in: GPT-2/3/4, RoBERTa, Llama (byte-level BPE).

#### WordPiece
Like BPE, but uses likelihood (data prob) instead of frequency for merges.

Used in: BERT, DistilBERT.

#### SentencePiece (unigram or BPE)
- Operates directly on raw text (no pre-tokenization)
- Treats whitespace as a token
- Unigram model: train via EM
- BPE model: same as BPE

Used in: T5, mT5, Llama (sentencepiece variant).

### Practical details
- **Vocab size:** typically 32k-128k. Larger vocab = shorter sequences but bigger LM head.
- **Byte-level BPE:** can represent any unicode (GPT-2 approach). No "unknown token" needed.
- **Special tokens:** `<bos>`, `<eos>`, `<pad>`, `<sep>`, `<mask>`, plus role markers in chat formats.

### Tokenization gotchas
- **Indian/Asian languages get penalized:** non-Latin scripts produce many more tokens per character than English. E.g., Hindi text might be 3-5x longer in tokens than equivalent English.
- **Numbers:** GPT tokenizers often split numbers oddly: "1234" might be 2-3 tokens.
- **Code:** tokenizers trained on text often produce 2-3x more tokens for code vs natural language.
- **Whitespace + punctuation:** can cause subtle bugs in templates (e.g., trailing space affects generation).

### "Tokens per character" rule of thumb
- English: ~0.25 tokens/char (4 chars/token)
- Code: ~0.5 tokens/char
- Hindi, Arabic, Chinese: 1+ tokens/char

---

## 2. Embeddings

### What they are
A lookup table: `[vocab_size, d_model]`. Each token ID gets a dense vector.

### Training
- Trained jointly with the LM (no separate "embedding training" step in standard LLM)
- Random init, gradient updates

### Tied embeddings
- LM head (output projection) often = transpose of input embedding
- Saves ~half a billion params for big models
- Helps generalization

### What about "sentence embeddings"?
- Different concept: a single vector per sentence/document
- Trained via contrastive objectives (BERT-style, sentence-transformers)
- Models: text-embedding-3-large (OpenAI), embed-english-v3 (Cohere), bge-large (open)
- Used in: RAG retrieval, semantic search, clustering

### Embedding eval (MTEB benchmark)
- Massive Text Embedding Benchmark
- Tasks: retrieval, classification, clustering, reranking, STS
- Leaderboard: huggingface.co/spaces/mteb/leaderboard

---

## 3. Attention internals (review + advanced)

(Foundations covered in `03-ml-fundamentals/04-transformers-deepdive.md`. This is the **deep** version.)

### Why softmax (vs alternatives)
- Softmax: max-attention with continuous gradients, all positions get nonzero weight (if all scores finite)
- Alternatives explored: sparse attention (k-largest), entmax (sparse but differentiable), Performer (kernel-based linear attention)
- Tradeoff: softmax has full O(n²) cost; sparse can be O(n) but representational tradeoff

### Causal mask details
- Lower-triangular: position i attends to 0..i only
- Applied BEFORE softmax (set future positions to -inf, become 0 after softmax)
- For batch: shape `[seq_len, seq_len]` typically broadcast

### Cross-attention (encoder-decoder)
- Q from decoder, K and V from encoder
- Encoder attends bidirectionally; decoder attends to encoder via cross-attention
- Used in: T5, original transformer, multimodal models (vision encoder → text decoder)

### Sliding window attention (e.g., Mistral)
- Each token attends only to last W positions
- Linear complexity O(n·W)
- Local context only — combine with global tokens for long-range

### Grouped attention variants (advanced)

#### Why we have so many variants
The fundamental cost of attention:
- Compute: O(n² · d) per head per layer
- Memory (KV cache during inference): O(n · d · L · h) where L=layers, h=heads
- KV cache dominates inference memory at long context

Reducing # of distinct KV heads → less memory:
- MHA: h Q heads, h KV heads → reference
- GQA: h Q heads, h/g KV heads (g groups) → /g memory
- MQA: h Q heads, 1 KV head → /h memory
- MLA: latent compression of KV → even less memory

### Attention in production (FlashAttention)
- Standard attention materializes n×n matrix in HBM
- HBM bandwidth is the bottleneck (slow vs compute)
- FlashAttention: tile + recompute → never materialize full n×n in HBM
- Computes attention IN SRAM (fast on-chip memory)
- Result: 2-4x faster on common shapes, exact same outputs

Standard:
```
Q, K, V load from HBM → compute QK^T (n×n) write to HBM → softmax read+write n×n → multiply by V → output
```

FlashAttention:
```
Load tile of Q, K, V into SRAM → compute partial QK^T → online softmax (Welford-style) → accumulate output → next tile
```

Key idea: never store the n×n matrix in HBM. Recompute in backward if needed.

### PagedAttention (vLLM)
- Treats KV cache like virtual memory (paged)
- Allows non-contiguous KV cache → no memory fragmentation
- Enables: prefix caching across requests, dynamic batching, prompt caching
- vLLM gets 2-5x throughput vs naive KV cache management

---

## 4. Positional encodings (deep version)

### Why position matters
Attention is permutation-invariant. Without position, "the cat ate the dog" and "the dog ate the cat" give same outputs.

### Absolute (sinusoidal, original)
PE depends only on position, added to embeddings. Drawback: no inherent length generalization.

### Relative position (T5-style)
Bias added to attention scores: `scores_ij + b(i - j)`. b is learned per-relative-position (often bucketed for distant pairs).

### RoPE (Rotary Position Embedding)
Rotate Q and K by position-dependent angle in 2D subspaces.

For dim pair (2k, 2k+1) at position m:
```
[Q'_{2k}]   [cos(mθ_k)  -sin(mθ_k)] [Q_{2k}]
[Q'_{2k+1}] = [sin(mθ_k)   cos(mθ_k)] [Q_{2k+1}]
```

Where `θ_k = 10000^(-2k/d)`.

Key property: `Q'_m · K'_n = f(Q_m, K_n, m-n)` — encodes RELATIVE position naturally.

Used in: GPT-Neo, GPT-J, Llama family, most modern open models.

### Length extension via RoPE manipulation
- Position Interpolation: scale position indices down (m → m / k)
- NTK-aware: scale θ_k differently per dimension
- YaRN: more refined NTK with attention scale correction

These let you take a Llama trained on 4k context and use it at 32k with finetuning (or even without).

### ALiBi (Attention with Linear Biases)
- Add bias to attention scores: `score_ij - m·|i-j|`
- m: head-specific slope (geometric progression: 1/2¹, 1/2², ...)
- No positional embedding needed
- Strong length extrapolation

---

## 5. Activation functions in modern LLMs

### Why SwiGLU is standard
Original FFN: `FFN(x) = W_2 · ReLU(W_1·x + b_1) + b_2`

SwiGLU: `FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down`
- SiLU(x) = x · σ(x) = x · sigmoid(x)
- Gating mechanism: x·W_up is filtered by SiLU(x·W_gate)
- 3 matrices instead of 2; ~50% more params per FFN
- But: typically reduce hidden dim by 2/3 to keep param count similar
- Empirically: 0.5-1% perplexity improvement

Used in: PaLM, Llama 2, Llama 3, Mistral, Qwen, DeepSeek.

### Other modern activations
- **GeGLU:** GELU + gating (similar to SwiGLU)
- **ReGLU:** ReLU + gating

---

## 6. Normalization

### Why LayerNorm not BatchNorm
- BN computes statistics across batch dim → fails when seq lengths vary or batch is small
- LN computes statistics within each sample → batch-size independent

### Pre-norm vs Post-norm
Original transformer: post-norm (LayerNorm after residual add). Unstable for deep training.

Modern: pre-norm
```
x → LN → SubLayer → +x → ...
```

Better gradient flow, allows deeper networks.

### RMSNorm
Simplification of LayerNorm:
```
RMSNorm(x) = x · γ / RMS(x)
where RMS(x) = sqrt(mean(x²))
```

- Drops centering and bias terms
- ~10% faster
- Used in: Llama, Mistral, Qwen

---

## 7. Training pipeline

### Pre-training
- **Data:** trillions of tokens (CommonCrawl, refined; books; code; Wikipedia)
- **Objective:** next-token prediction (causal LM)
- **Duration:** months on thousands of GPUs/TPUs
- **Compute:** Llama-3 70B used ~10M GPU-hours

### Mid-training (chinchilla-optimal vs over-training)
- Chinchilla scaling laws (2022): for fixed compute, data ∝ √compute and model ∝ √compute
- Pre-Chinchilla LLMs (GPT-3) under-trained on data
- Post-Chinchilla (Llama 2, 3) train far more tokens (often >Chinchilla optimal because inference cost dominates total cost over a model's life)

### Supervised Fine-Tuning (SFT)
- After pre-training, fine-tune on instruction-following examples
- Format: `{instruction, input, output}` triples
- Examples: Dolly-15k, OpenAssistant, Alpaca
- Standard LM loss

### RLHF (Reinforcement Learning from Human Feedback)
Phase 1: Train reward model on human preferences (A vs B comparisons)
Phase 2: Use reward model to fine-tune via PPO

PPO challenges: instability, mode collapse, reward hacking, hyperparameter sensitivity.

### DPO (Direct Preference Optimization)
- Reformulates RLHF without separate reward model
- Loss directly on preferred vs rejected outputs
- More stable, simpler, often comparable quality
- Standard for modern OSS LLMs

### KTO, IPO, others
- KTO: only positive feedback (no need for paired prefs)
- IPO: addresses DPO's reward over-optimization

---

## 8. Decoding strategies

### Greedy
- Pick argmax at each step
- Deterministic, but repetitive

### Beam search
- Maintain top-k partial sequences
- Better for translation (where there's "one right answer")
- Less used for open-ended generation (causes bland outputs)

### Sampling
- Sample from output distribution
- Temperature: scale logits by 1/T before softmax
  - T=1: as-is
  - T<1: sharper, more deterministic
  - T>1: flatter, more random

### Top-k sampling
- Restrict to top k highest-probability tokens
- Then sample (re-normalized)

### Top-p (nucleus) sampling
- Restrict to smallest set whose cumulative prob ≥ p
- Then sample
- Adapts to distribution shape

### Typical sampling, mirostat, contrastive search
- More sophisticated; rarely needed

### Common practice
- Temperature: 0.7-1.0 for creative; 0.1-0.3 for code/factual
- Top-p: 0.9-0.95
- Both at once (top-p then temperature)

### Repetition penalty
- Multiplicative penalty on already-seen tokens
- Prevents loops
- Tradeoff: too high → unnatural

---

## 9. Inference compute and memory

### Phases
1. **Prefill:** process all input tokens in parallel
   - Compute-bound (lots of matmul, used efficiently)
   - Fast (compared to decode)
2. **Decode:** generate one token at a time
   - Memory-bound (must load whole model weights per token)
   - Slow (per token)
   - KV cache crucial to avoid recompute

### Memory breakdown (for 70B Llama, fp16)
- Weights: 140GB
- KV cache (per request): ~80GB at 32k context
- Activations during forward: ~1GB
- → Multi-GPU required, e.g., 4x H100 (320GB total)

### Latency considerations
- TTFT (time to first token): dominated by prefill
- ITL (inter-token latency): dominated by decode
- Throughput vs latency tradeoff: continuous batching helps throughput, slightly hurts ITL

### Quantization
- FP16/BF16: standard
- INT8: post-training (AWQ, GPTQ, SmoothQuant)
- INT4: Q4_K_M (GGUF), GPTQ-INT4
- FP8: emerging, used in inference on H100

---

## 10. What's "emergent ability"?

Term from Wei et al. 2022 paper.

Claim: certain abilities (math, in-context learning, chain-of-thought) "emerge" suddenly at certain model scales — they're absent in smaller models, then sharply appear above a threshold.

Skeptical view (Schaeffer et al. 2023): emergence may be an artifact of how we measure. Continuous improvement on smooth metrics; emergence visible only with hard binary metrics.

What to know:
- Term is contested in the research community
- Cite the skeptical view to show depth
- Don't say "LLMs magically gain abilities" — say "performance improves with scale; sometimes sharply on specific metrics"

---

## 11. The "reasoning models" frontier (o1, o3, Claude with extended thinking, DeepSeek R1)

- Train via RL on chain-of-thought outputs
- Model spends "thinking tokens" before answering
- Trades inference time for accuracy
- Strong on math/code, weaker on creative tasks
- DeepSeek R1 (Jan 2025): showed RL-only training (no SFT) can produce reasoning

What to know:
- This is the most active research direction in 2026
- Concepts: process reward models, search at inference time, verifier-guided generation
- Read DeepSeek R1 paper if going deep

---

## Implementation exercises

Do these once during Months 5-6. They will teach you more than 30 papers.

### Exercise 1: nanoGPT clone (Months 5)
- Implement decoder-only transformer from scratch
- Use Karpathy's `nanoGPT` repo as reference (read every line)
- Train on Tiny Shakespeare
- Implement: tokenizer (use tiktoken), embedding, MHA, FFN with GELU, LN, causal mask, generation

### Exercise 2: KV cache (Months 6)
- Take your nanoGPT
- Add KV cache to inference
- Measure speedup
- Compare memory usage

### Exercise 3: LoRA fine-tune (Months 6-7)
- Use a small pre-trained model (Llama 3.2 1B or similar)
- Fine-tune on a small dataset (e.g., your blog posts)
- Implement LoRA from scratch (matrix factorization with rank 8)
- Compare to full fine-tuning

### Exercise 4: Quantize and serve (Months 7)
- Take your fine-tuned model
- Quantize to INT8 (AWQ or GPTQ)
- Serve via vLLM
- Benchmark throughput

These four exercises = strongest possible portfolio of "I have ACTUALLY done this end to end."

---

## How to drill this file for interviews

Pick 3-5 topics, write your own 200-word answer to "explain this to a colleague." Practice it out loud. Time yourself: under 90 seconds per explanation.

Topics most-likely to be asked:
1. Attention math + multi-head
2. KV cache (the universal "test your LLM systems knowledge" question)
3. RoPE / positional encoding
4. Why LayerNorm not BatchNorm
5. Difference between SFT and RLHF (and DPO)
6. FlashAttention motivation
7. Quantization strategies (and tradeoffs)
8. Token vocabulary size tradeoffs
9. MoE vs dense
10. The bottleneck in LLM inference (memory bandwidth)

---

Next: [`02-rag-deepdive.md`](./02-rag-deepdive.md)
