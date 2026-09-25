# Lab 07 — KV cache and sampling

**Build:** a preallocated KV cache, a cache-aware causal mask, chunked prefill, cached generation, and temperature / top-k / top-p / min-p sampling.
**Time:** 6–8 h · **Reads first:** [inference](../../curriculum/07-inference.md)
**Run:** `pytest labs/07_kv_cache_sampling`

## Ideas
- **Why a cache:** without one, step t recomputes K and V for all t previous tokens, so generation costs O(T²) projections. With a cache each step projects one token and *reads* the cached K and V. Decode becomes memory-bound: every step streams the weights plus the cache ([lab 03](../03_napkin_math/README.md)).
- **The offset mask:** a query at absolute position `pos + i` may attend to keys `0 … pos + i`. Get this wrong and cached logits drift from full-forward logits. `test_incremental_logits_match_full_forward` is the contract.
- **Chunked prefill:** prefilling a long prompt in chunks gives identical logits and lets a server interleave prefill with other users' decode steps (Sarathi-Serve). Your test proves the equivalence.
- **Sampling order:** temperature, then top-k, then top-p, then min-p, then sample. Top-p keeps the token that *crosses* p, and always keeps the argmax. Min-p scales its cutoff with the model's confidence, so it stays permissive when the distribution is flat and strict when it is peaked.

## Implement
`KVCache.update` → `cache_attention_mask` → `top_k_filter`, `top_p_filter`, `min_p_filter` → `sample_next` → `generate`.

## Measure (stretch)
Time `generate(use_cache=True)` against `use_cache=False` for 32 to 512 new tokens and plot both. Explain the curve with FLOP and byte counts. Then implement a paged cache (fixed 16-token blocks plus a block table, as in vLLM) and prefix sharing between two prompts with the same system prompt.
