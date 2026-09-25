# Lab 05 — A modern GPT from scratch

**Build:** a Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, optional QK-norm,
pre-norm residual blocks, tied embeddings, depth-scaled init. Then pretrain it on your GPU.
**Time:** 10–14 h + the scale-up run · **Reads first:** [transformers](../../curriculum/04-transformers.md)
**Run:** `pytest labs/05_transformer` · then `python labs/05_transformer/train.py --help`

---

## The forward pass, with shapes

```
idx (B,T) ─ tok_emb ─▶ x (B,T,d)
for each block:                                   # residual stream: blocks READ from and WRITE to x
    h = RMSNorm(x)
    q = h Wq → (B,H,T,hd)   k,v = h Wk, h Wv → (B,H_kv,T,hd)
    [QK-norm]  q,k ← RoPE(q,k, positions)
    k,v ← repeat_kv(k,v, H/H_kv)
    a = softmax(q kᵀ/√hd + causal) v → (B,H,T,hd) → merge heads → Wo → (B,T,d)
    x = x + a
    x = x + W_down( SiLU(W_gate RMSNorm(x)) ⊙ W_up RMSNorm(x) )
logits = RMSNorm(x) · Eᵀ  (tied)  → (B,T,V)
```

## Component notes, the parts interviews probe

**Attention and the √d scaling.** If q and k have i.i.d. unit-variance entries, `q·k` has variance
d. Unscaled, the logits grow like √d and the softmax saturates into a near one-hot with vanishing
gradients. Dividing by √d restores unit variance.

**RoPE.** Treat each pair of dims as a complex number and multiply position m's query by
`e^{i m θ_k}`, and position n's key likewise. Then `Re⟨q_m e^{imθ}, k_n e^{inθ}⟩` depends only on
`m − n`: relative position comes out of absolute rotations, with no learned parameters and no
change to the norm. The frequencies `θ_k = base^(−2k/d)` range from fast (local syntax) to slow
(long range). Raising the base, or rescaling positions (PI, NTK, YaRN), is how context gets
extended ([curriculum 04](../../curriculum/04-transformers.md#3-positional-information)).
`test_rope_preserves_norm_and_encodes_relative_position` checks both properties.

**GQA.** H query heads share H_kv key/value heads. Quality stays close to MHA while the **KV cache
shrinks by H/H_kv** (4× for Llama-3-8B), and at serving time the KV cache is the memory that
limits batch size ([lab 03](../03_napkin_math/README.md), [lab 07](../07_kv_cache_sampling/README.md)).
`test_gqa_with_shared_kv_equals_mha` proves GQA is MHA with tied K/V heads.

**SwiGLU.** `SiLU(xW_gate) ⊙ xW_up` is a learned multiplicative gate. It uses three matrices, so
the hidden size drops to ~8/3·d to keep the FLOPs of a 4d two-matrix MLP. Llama-3-8B uses
14336 = 3.5·4096: its config multiplies the 8/3 default by `ffn_dim_multiplier = 1.3` and rounds
up to a multiple of 1024, a deliberate widening.

**Pre-norm plus the residual stream.** Each block adds its output into `x`. Gradients flow
through the identity path unimpeded, which is why deep pre-norm transformers train without
warmup heroics. The cost: the stream's norm grows with depth, so a final norm before the head is
mandatory.

**Initialization.** Weights ~N(0, 0.02). The two projections that write into the stream (`Wo`,
`W_down`) get 0.02/√(2L), because 2L branches sum into the stream.

**The self-logit effect (found while building this lab).** With *tied* embeddings, the final hidden
state at init is roughly the normalized embedding of the *input* token, so the model already gives
its own input a logit up to ≈ `0.02·d`. With d = 64 that is 1.3, and `loss(idx, idx)` starts at
2.9 instead of ln 64 = 4.16. Always measure initial loss against real next-token targets. The
correct value is ≈ ln V; anything far off means your init or your targets are wrong.

**QK-norm.** Normalizing q and k per head bounds the attention logits. That prevents the "attention
logit growth" instability behind many large-scale loss spikes (Wortsman et al., 2023; used in
OLMo 2 and Gemma 3).

## What to implement

`rope_cache` → `apply_rope` → `causal_attention` → `repeat_kv` → `Attention.forward` → `SwiGLU.forward`
→ `Block.forward` → `GPT._init_residual_projections` → `GPT.forward`.
The overfit test is your integration test: a correct model memorizes 4 random sequences in 150 steps.

## Scale-up run S1: pretrain on your RTX 5060

```python
from huggingface_hub import hf_hub_download  # pip install huggingface_hub
hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt", repo_type="dataset", local_dir="data")
```
```bash
python tools/measure_gpu.py            # note your bf16 TFLOP/s
python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt \
    --d-model 384 --n-layer 6 --n-head 6 --block-size 256 --batch-size 64 \
    --max-steps 5000 --peak-tflops <your number> --compile
```

On an 8 GB card this ~11M-parameter byte-level model fits easily and learns grammatical stories in
about an hour. Then run **one controlled ablation** and write it up with the
[experiment template](../../journal/templates/experiment.md):
GQA (`--n-kv-head 2`) vs MHA at equal steps, QK-norm on/off at a deliberately high LR, depth vs
width at equal parameters, or byte-level vs GPT-2 BPE at equal *compute* (not equal steps; explain why).

Before you run, predict tokens/s from your measured TFLOP/s and the printed FLOPs/token. Then
explain the gap: MFU on small models is limited by kernel launch overhead and the attention and
softmax memory traffic. `--compile` fuses kernels; measure how much it helps.

## Check yourself

1. Why is the causal mask applied before the softmax and not after?
2. GQA with H = 32, H_kv = 8: how many times smaller is the KV cache? Is attention compute smaller?
3. What breaks if you apply RoPE to `v` as well?
4. Why do most modern LLMs have no bias terms in their linear layers?
5. Your model trains fine at 512 context and outputs garbage at 2,048. List the reasons and the fixes.

<details><summary>Answers</summary>

1. Masking after the softmax leaves rows that do not sum to 1, and the probability mass has already been spent on future tokens. Setting logits to −∞ first gives an exact distribution over the allowed positions.
2. 4× smaller. Attention FLOPs are unchanged: every query head still scores every key.
3. The output would mix position into content additively and lose the relative-position property. RoPE works because the rotation cancels in the q·k dot product; v is not dotted with anything positional.
4. Biases add little capacity, hurt stability at scale, and complicate sharding and quantization. With a norm before each block, they are mostly redundant.
5. The model never saw positions beyond 512, so RoPE angles fall out of distribution, and attention entropy changes with length. Fixes: train at longer context, increase the RoPE base, apply position interpolation or YaRN with a short fine-tune, and check your block_size and position caches.
</details>

## Stretch

- Load GPT-2 small's weights into your architecture variant (learned positions, LayerNorm, GELU MLP, biases) and match Hugging Face logits to 1e-4. This proves you understand the checkpoint format.
- Add weight-tied multi-token prediction (predict t+1 and t+2) as in DeepSeek-V3's MTP, and measure its effect at equal compute.
- Replace softmax attention in one layer with a linear-attention or state-space (Mamba-style) block. Where does it fail? (Associative recall, see [lab 16](../16_interpretability/README.md).)
