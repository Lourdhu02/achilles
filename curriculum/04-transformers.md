# 04 — Transformers

Labs: [04 tokenizer](../labs/04_tokenizer/README.md), [05 GPT](../labs/05_transformer/README.md), [06 kernels](../labs/06_attention_kernels/README.md).

## 1. Tokens in, logits out
Byte-level BPE turns text into ids; an embedding table maps ids to d-dim vectors; L blocks transform the residual stream; a final norm and the LM head produce V logits. Next-token cross-entropy trains everything. The tokenizer's quirks (digit grouping, Indic combining marks, special tokens) are permanent properties of the model.

## 2. Attention
`softmax(QKᵀ/√d_h + mask)V`, per head. You should be able to explain:
- **Why √d_h:** q·k has variance d_h, and without scaling the softmax saturates.
- **Heads as low-rank bilinear forms:** head h computes `x_i W_Q^h (W_K^h)ᵀ x_jᵀ`, a rank-d_h QK circuit, and writes through a rank-d_h OV circuit.
- **Cost:** O(T²·d) FLOPs; the KV cache holds O(T·L·H_kv·d_h) memory. **MQA/GQA** share K/V across heads (KV memory ÷ H/H_kv). **MLA** (DeepSeek-V2/V3) caches a compressed latent (~512 dims plus a small decoupled RoPE part per token per layer) and up-projects, which shrinks the cache far below GQA's.
- **Sliding-window / local attention** (Mistral, Gemma interleaved local and global layers) caps the KV cache for local layers.

## 3. Positional information
Attention is permutation-equivariant, so position must be injected. **RoPE** rotates dimension pairs by `m·θ_k`, so q·k depends only on m − n. `θ_k = base^(−2k/d)`. Context extension: raise the base, apply position interpolation (scale m down), NTK-aware scaling, or **YaRN** (per-frequency interpolation plus an attention temperature), followed by a short long-context fine-tune. Alternatives: ALiBi (linear distance bias) and NoPE (no positions; causality leaks order). Long-context quality requires training on long documents, not just a positional trick.

## 4. The MLP and the block
SwiGLU `W_down(SiLU(W_g x) ⊙ W_u x)` with ~8/3·d hidden. The MLP holds about 2/3 of the parameters and acts as key-value memory for facts. The block is pre-norm (RMSNorm), attention, residual, MLP, residual.

## 5. Parameter and FLOP accounting
Per layer ≈ 12d² (MHA, 4× MLP); embeddings V·d. Training ≈ 6N per token. Know Llama-3-8B cold: d = 4096, L = 32, 32 heads / 8 KV heads, d_ff = 14336, V = 128,256 → 8.03B parameters ([lab 03](../labs/03_napkin_math/README.md)).

## 6. Mixture of experts
The router picks the top-k of E expert MLPs per token, so **total ≫ active** parameters (DeepSeek-V3: 671B total, 37B active). Issues: load balancing (Switch aux loss `E·Σ f_i·P_i`; DeepSeek-V3's aux-loss-free per-expert bias), token dropping and capacity factor, router z-loss, fine-grained plus shared experts, and expert parallelism's all-to-all communication. The win: more capacity per FLOP. The costs: memory and communication.

## 7. Stability tricks you will see in configs
QK-norm (bounds attention logits), z-loss `1e-4·log²Z` on the output softmax, logit soft-capping (Gemma 2), no biases, embedding scaling, and removing weight decay from norms and embeddings.

## 8. Beyond vanilla attention
Linear attention and **state-space models** (Mamba: input-dependent recurrence, O(T) and a constant-size state) are fast but weaker at exact recall. **Hybrids** (interleaved SSM and attention layers) are increasingly common. Other directions: multi-token prediction (DeepSeek-V3), diffusion language models, and encoder-decoder models for some modalities. Vision: ViT patches into tokens, CLIP (contrastive image-text), LLaVA-style projectors into an LLM.

## Check yourself
Why GQA over MQA? Why does the KV cache, not the weights, often limit serving batch size? What does increasing the RoPE base do to low-frequency dimensions? Why are MoE models cheap to train but expensive to serve on few GPUs?

**Read:** Vaswani 2017; RoFormer; GQA; Shazeer 2020 (GLU variants); Llama 3 herd paper; DeepSeek-V2 (MLA) and V3; Switch Transformer; Mamba; YaRN.
