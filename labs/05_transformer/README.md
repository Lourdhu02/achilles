# Lab 05 — A modern GPT from scratch

**Build:** a Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU, optional QK-norm,
pre-norm residual blocks, tied embeddings, depth-scaled init. Then pretrain it on your GPU.
**Time:** 10–14 h + the scale-up run · **Reads first:** [transformers](../../curriculum/04-transformers.md)<br>
**Run:** `pytest labs/05_transformer` (your code) · `pytest labs/05_transformer --impl=solution` (reference) · then `python labs/05_transformer/train.py --help`

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
gradients. Dividing by √d restores unit variance. ([Curriculum 04 §2](../../curriculum/04-transformers.md#2-attention)
has a table of how saturated it gets.)

**RoPE.** Treat each pair of dims as a complex number and multiply position m's query by
`e^{i m θ_k}`, and position n's key likewise. Then `Re⟨q_m e^{imθ}, k_n e^{inθ}⟩` depends only on
`m − n`: relative position comes out of absolute rotations, with no learned parameters and no
change to the norm. The frequencies `θ_k = base^(−2k/d)` range from fast (local syntax) to slow
(long range). Raising the base, or rescaling positions (PI, NTK, YaRN), is how context gets
extended ([curriculum 04](../../curriculum/04-transformers.md#3-positional-information)).
`test_rope_preserves_norm_and_encodes_relative_position` checks both properties, and
`test_rope_matches_complex_multiplication` checks your real-valued code against `torch.polar`.
This lab uses the interleaved-pair convention `(x[2i], x[2i+1])`; Hugging Face checkpoints rotate
halves `(x[i], x[i + D/2])`, so porting weights needs a permutation of `Wq` and `Wk`.

**GQA.** H query heads share H_kv key/value heads. Quality stays close to MHA while the **KV cache
shrinks by H/H_kv** (4× for Llama-3-8B), and at serving time the KV cache is the memory that
limits batch size ([lab 03](../03_napkin_math/README.md), [lab 07](../07_kv_cache_sampling/README.md)).
`test_gqa_with_shared_kv_equals_mha` proves GQA is MHA with tied K/V heads.

**SwiGLU.** `SiLU(xW_gate) ⊙ xW_up` is a learned multiplicative gate. It uses three matrices, so
the hidden size drops to ~8/3·d to keep the FLOPs of a 4d two-matrix MLP. Llama-3-8B uses
14336 = 3.5·4096: its config multiplies the 8/3 default by `ffn_dim_multiplier = 1.3` and rounds
up to a multiple of 1024, a deliberate widening. This lab's `GPTConfig` rounds 8/3·d up to a
multiple of 64 (d = 64 gives 192; d = 384 gives 1024).

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

| step | contract | tests |
|---|---|---|
| `rope_cache(head_dim, max_seq, theta)` | `cos`, `sin` of shape `(max_seq, head_dim/2)`, angle `m·θ^(−2i/head_dim)` | `test_rope_tables` |
| `apply_rope(x, cos, sin)` | rotate pairs `(x[..., 2i], x[..., 2i+1])` | `test_rope_matches_complex_multiplication`, `test_rope_preserves_norm_and_encodes_relative_position` |
| `causal_attention(q, k, v)` | `softmax(qkᵀ/√D + mask)v` without `F.scaled_dot_product_attention` | `test_causal_attention_matches_sdpa` |
| `repeat_kv(x, n_rep)` | each KV head serves `n_rep` *consecutive* query heads; return `x` itself when `n_rep = 1` | `test_repeat_kv_groups_consecutive_heads` |
| `Attention`, `SwiGLU`, `Block` forwards | as in the shape diagram | `test_model_is_causal` (MHA, GQA, MQA), `test_gqa_with_shared_kv_equals_mha` |
| `_init_residual_projections` | `Wo`, `W_down` ~ N(0, 0.02/√(2L)) | `test_residual_projection_init_is_depth_scaled` |
| `GPT.forward(idx, targets)` | logits `(B,T,V)`; mean cross-entropy with `ignore_index=-100` | `test_initial_loss_is_near_uniform`, `test_ignore_index_in_targets`, `test_overfits_a_batch` (with and without QK-norm), `test_generate_extends_and_is_greedy_with_top_k_1` |

`test_param_count_matches_formula` and `test_tied_embeddings_share_storage` check the provided
constructor; read them to see the exact parameter formula
(`V·d + L·(2d² + 2d·H_kv·hd + 3·d·d_ff + 2d) + d` with a tied head).

## Common bugs

- **`view` after `transpose`.** `y.transpose(1, 2).view(B, T, d)` fails or silently scrambles on a
  non-contiguous tensor; use `reshape` (or `.contiguous().view`).
- **Mask after softmax, or a mask of zeros.** Fill future positions with `-inf` *before* the softmax.
  Multiplying probabilities by a 0/1 mask afterwards leaves rows that do not sum to 1.
- **`repeat` instead of `repeat_interleave`.** `repeat` tiles KV heads as `[0,1,0,1]`; GQA groups
  consecutive query heads, `[0,0,1,1]`. The model still trains, but it breaks weight compatibility
  and `test_repeat_kv_groups_consecutive_heads`.
- **RoPE on the wrong axis or wrong positions.** Rotate along the head dimension with angles
  indexed by sequence position, sliced to the current `T` (`rope_cos[:T]`).
- **Scaling by √d_model instead of √head_dim.**
- **Targets not shifted.** `targets = idx` makes the "loss" small for the wrong reason (the
  self-logit effect above). Next-token targets are `data[:, 1:]` for inputs `data[:, :-1]`.
- **Norm weight decay and init.** RMSNorm gains start at 1; do not re-initialize them with the
  N(0, 0.02) init, and do not decay them (train.py decays only ≥ 2-D parameters).

> [!TIP]
> Test pieces in isolation before the full model: compare `causal_attention` with
> `F.scaled_dot_product_attention(q, k, v, is_causal=True)` on random tensors, and check that
> changing token 7 changes no logits at positions 0–6. Those two checks catch most attention bugs.

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

### What the script does

- Tokenizes the text once (`--tokenizer byte`, the default, or `gpt2` via `tiktoken`) and caches the
  ids as a `.bin` memmap next to the text file. The last 1% of tokens is the validation split.
- Builds your model (`--impl exercise`, the default) or the reference (`--impl solution`).
- Trains with AdamW (β = 0.9, 0.95; weight decay 0.1 on matrices only), linear warmup
  (`--warmup`, default 200 steps) then cosine decay from `--lr` to `--min-lr`, gradient clipping
  at 1.0, bf16 autocast on CUDA.
- Every `--eval-interval` steps it prints train/val loss and overwrites `runs/gpt/ckpt.pt`
  (`--out` changes the directory); every 10 steps it prints loss, LR, grad norm, tokens/s and (with `--peak-tflops`) MFU.
- At the end it samples 300 tokens, prompted with the start of the validation split.

Run `--help` for the full list; if your copy of the script offers device or preset options, they
are documented there.

### Predict before you run

| quantity | how to predict | S1 default |
|---|---|---|
| parameters | `V·d + L·(4d² + 3·d·d_ff + 2d) + d` for MHA | 10.72M (d_ff = 1024) |
| FLOPs/token (as printed) | `6·(non-embedding + V·d) + 6·L·T·d` | 67.9 MFLOP |
| tokens seen | steps × batch × block | 5000 × 64 × 256 = 81.9M (~4% of one epoch of the ~2 GB file) |
| total compute | tokens × FLOPs/token | 5.6e15 FLOP |
| time | total / (peak × MFU) | your number: write it down first |
| initial loss | ln 256 | 5.55 |

A model this small on 82M tokens will not be Chinchilla-optimal in either direction; the point is
to check your throughput model and see a real loss curve. Expect validation loss (nats per byte)
to fall fast in the first few hundred steps, then slowly; samples become grammatical before they
become coherent.

### CPU-only version

No GPU? The same script runs on CPU with a smaller model. Try
`--d-model 128 --n-layer 4 --n-head 4 --block-size 128 --batch-size 16 --max-steps 1000`
(0.89M parameters, 5.7 MFLOP/token) and drop `--compile` and `--peak-tflops`. Measure tokens/s at
step 100 and extrapolate before committing to a longer run. On Windows, `--compile` needs Triton;
use WSL2 or leave it off.

### Running the ablation well

- **Change one thing.** GQA with `--n-kv-head 2` also removes parameters (9.54M vs 10.72M at the
  S1 shape). Either report it as "GQA at fewer parameters" or compensate the width.
- **Compare in bits per byte across tokenizers.** Loss per token is not comparable between byte-level
  and GPT-2 BPE, because a BPE token covers several bytes. Convert:
  `bpb = loss_nats_per_token × n_tokens / (n_bytes × ln 2)`. And the GPT-2 vocabulary adds
  50,257 × d embedding parameters and a large LM-head cost, so match *compute*, not steps.
- **Seeds.** Run at least 2 seeds per arm; the seed-to-seed spread is your noise floor
  ([curriculum 08](../../curriculum/08-evaluation-and-research.md)).
- **High-LR QK-norm test.** Raise `--lr` until the baseline shows grad-norm spikes or diverges, then
  rerun with `--qk-norm` at the same LR. Plot grad norm, not only loss.

## Check yourself

1. Why is the causal mask applied before the softmax and not after?
2. GQA with H = 32, H_kv = 8: how many times smaller is the KV cache? Is attention compute smaller?
3. What breaks if you apply RoPE to `v` as well?
4. Why do most modern LLMs have no bias terms in their linear layers?
5. Your model trains fine at 512 context and outputs garbage at 2,048. List the reasons and the fixes.
6. Your first logged loss on byte-level data is 3.1. What do you suspect?
7. Why does the overfit test use 4 random sequences instead of real text?
8. The S1 model reports 20% MFU. Name three reasons it is not 60%.

<details><summary>Answers</summary>

1. Masking after the softmax leaves rows that do not sum to 1, and the probability mass has already been spent on future tokens. Setting logits to −∞ first gives an exact distribution over the allowed positions.
2. 4× smaller. Attention FLOPs are unchanged: every query head still scores every key.
3. The output would mix position into content additively and lose the relative-position property. RoPE works because the rotation cancels in the q·k dot product; v is not dotted with anything positional.
4. Biases add little capacity, hurt stability at scale, and complicate sharding and quantization. With a norm before each block, they are mostly redundant.
5. The model never saw positions beyond 512, so RoPE angles fall out of distribution, and attention entropy changes with length. Fixes: train at longer context, increase the RoPE base, apply position interpolation or YaRN with a short fine-tune, and check your block_size and position caches.
6. The expected value is ln 256 = 5.55. A value near 3 suggests the targets equal the inputs (the self-logit effect with tied embeddings) or leakage through a broken causal mask. Check the target shift and `test_model_is_causal`.
7. Random sequences have no structure to generalize from, so driving the loss near zero proves the whole forward and backward path can memorize, with no confound from the data. Real text would plateau at its entropy.
8. Kernel launch overhead with small matrices (d = 384), memory-bound ops (RMSNorm, softmax, SiLU, the naive T×T attention matrix written to memory), and the Python-side data loading and logging; `--compile` and a fused attention kernel (lab 06) address the first two.
</details>

## Stretch

- Load GPT-2 small's weights into your architecture variant (learned positions, LayerNorm, GELU MLP, biases) and match Hugging Face logits to 1e-4. This proves you understand the checkpoint format.
- Add weight-tied multi-token prediction (predict t+1 and t+2) as in DeepSeek-V3's MTP, and measure its effect at equal compute.
- Replace softmax attention in one layer with a linear-attention or state-space (Mamba-style) block. Where does it fail? (Associative recall, see [lab 16](../16_interpretability/README.md).)
- Implement YaRN (NTK-by-parts plus the attention temperature) in `rope_cache`, train at block size 256 and evaluate validation loss at 512 and 1,024 with and without it. Build the evaluation model with the larger `block_size`; the RoPE tables are not saved in the checkpoint, so the trained weights load unchanged.
- Swap in a sliding-window mask for alternate layers and measure the loss cost at equal steps.
