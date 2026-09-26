# Lab 07 — KV cache and sampling

**Build:** a preallocated KV cache, a cache-aware causal mask, chunked prefill, cached generation, and temperature / top-k / top-p / min-p sampling, on a small Llama-style decoder (RoPE, GQA, RMSNorm) that mirrors lab 05.
**Time:** 6–8 h · **Reads first:** [inference §1–2](../../curriculum/07-inference.md#1-two-phases-two-bottlenecks)
**Run:** `pytest labs/07_kv_cache_sampling` (your code) · `pytest labs/07_kv_cache_sampling --impl=solution` (reference). All 13 tests run on CPU in seconds.

Every serving engine is built on the two things in this lab: a KV cache whose indexing is exactly right, and a sampler whose filters compose correctly. Both fail silently when they are wrong: the model still produces fluent text, just not the text it should.

---

## 1. Why a cache

Without a cache, generating token t recomputes K and V for all t previous tokens, so producing n tokens after a P-token prompt costs about `Σ (P + t)` token-forwards: quadratic in length. With a cache, each step runs the model on **one** new token, appends its K and V, and *reads* the cached K and V of everything before it.

The price: decode becomes memory-bound. Every step streams all the weights plus the whole cache from memory to produce one token per sequence ([lab 03](../03_napkin_math/README.md)). Cache size per token is `2 · n_layer · n_kv_head · head_dim · bytes`: the 2 is K and V, and GQA (`n_kv_head < n_head`) is the main lever that shrinks it.

```
prefill (prompt of 8 tokens, one forward)      decode (one token per forward)
tokens:  t0 t1 t2 t3 t4 t5 t6 t7                 new token t8
K/V:     [written at positions 0..7]  pos = 8    K/V: [0..7 read] + [8 written]   pos = 9
queries: 8 rows, causal mask 8×8                 queries: 1 row, sees keys 0..8
```

## 2. The offset mask

With a cache, the queries in a forward pass do not start at position 0. A forward over `t_new` tokens when `pos` tokens are already cached has queries at absolute positions `pos … pos + t_new − 1` and keys at `0 … pos + t_new − 1`. Query `i` may attend to key `j` iff `j ≤ pos + i`.

```
cache_attention_mask(t_new=2, pos=3)       (True = NOT allowed)
            keys:  0     1     2     3     4
query at 3:     [ F     F     F     F     T ]
query at 4:     [ F     F     F     F     F ]
```

With `pos = 0` it reduces to the usual upper-triangular causal mask. RoPE must use the same absolute positions: the given `Attention.forward` builds `positions = arange(pos, pos + T)`. Get the mask or the positions wrong and the cached logits drift from the full-forward logits. `test_incremental_logits_match_full_forward` is the contract.

## 3. Chunked prefill

Prefilling a 17-token prompt as chunks of 5, 5, 5 and 2 tokens must give the same logits as one 17-token forward: each chunk writes its K/V at the current `pos`, and its queries see everything cached plus the causal part of the chunk. Servers use this to interleave a long prefill with other users' decode steps (Sarathi-Serve), so decoding users never see a stall. Your test proves the equivalence; the speed trade-off (re-reading the earlier chunks' KV) is what you measure in the stretch.

## 4. The sampling stack

`sample_next` applies, in order: **temperature** (divide logits), **top-k**, **top-p**, **min-p**, then samples from the softmax of what is left. `temperature == 0` means greedy (argmax) and skips everything else.

| filter | keeps | behaviour |
|---|---|---|
| top-k | the k largest logits (ties at the boundary kept) | fixed number of candidates regardless of confidence |
| top-p (nucleus) | the smallest set of top tokens whose mass reaches p, **including the token that crosses p**; the argmax is always kept | adapts to the shape: few candidates when peaked, many when flat |
| min-p | tokens with probability ≥ `min_p × p_max` | cutoff scales with the model's confidence: strict when peaked, permissive when flat |

Worked example with probabilities `[0.5, 0.25, 0.15, 0.06, 0.04]` (the tests' `LOGITS`):

- top-p 0.5 keeps `[0.5]`: the mass *before* token 2 is 0.5, already ≥ 0.5, so it is dropped.
- top-p 0.6 keeps `[0.5, 0.25]`: token 2 crosses 0.6.
- min-p 0.2: threshold 0.2 × 0.5 = 0.1, keeps `[0.5, 0.25, 0.15]`.
- top-k 2 then sampling: the kept tokens are renormalized to `[2/3, 1/3]`.

The rule "drop a token if the mass strictly before it is already ≥ p" gives both the crossing behaviour and the always-keep-the-argmax behaviour for free (the argmax has zero mass before it).

## What to implement

| # | function | tests | what the tests pin down |
|---|---|---|---|
| 1 | `KVCache.update(layer, k_new, v_new)` | `test_cache_update_writes_and_returns_prefix`, `test_cache_bytes_match_formula` | writes `(B, H_kv, T, hd)` at `[pos, pos+T)` for that layer and returns keys/values `[0, pos+T)`; does **not** advance `pos` |
| 2 | `cache_attention_mask(t_new, pos)` | `test_mask_offsets_by_cache_position` | shape `(t_new, pos + t_new)`, True where attention is not allowed |
| 3 | (given model uses 1 and 2) | `test_incremental_logits_match_full_forward`, `test_chunked_prefill_matches_single_prefill` | prefill 8 then decode 12 one at a time, or chunks of 5, both equal a full forward (atol 1e-5) |
| 4 | `top_k_filter`, `top_p_filter`, `min_p_filter` | `test_top_k`, `test_top_p_keeps_the_token_that_crosses_the_threshold`, `test_top_p_works_on_unsorted_rows`, `test_min_p` | filtered positions become `-inf`; top-p works on rows that are not sorted |
| 5 | `sample_next` | `test_greedy_and_low_temperature`, `test_sampling_distribution_after_top_k`, `test_temperature_flattens` | order of operations; empirical frequencies match the filtered distribution |
| 6 | `generate(model, prompt, max_new_tokens, use_cache, prefill_chunk, **sampling)` | `test_cached_generation_equals_uncached` | greedy output identical with and without the cache, and with `prefill_chunk=4` |

Given: `Config`, `KVCache.__init__`, `advance` and `nbytes`, `_rope`, `RMSNorm`, `Attention`, `Block`, `TinyLM`. Note who advances the cache: `TinyLM.forward` calls `cache.advance(T)` once, **after** all layers have called `update`, so every layer writes at the same `pos`.

## Tips

> [!TIP]
> Build `generate` in two phases with the cache: (1) prefill the prompt, in chunks if `prefill_chunk` is set, keeping only the last position's logits; (2) loop: sample from those logits, append the token, run the model on that one token to get the next logits. The uncached path simply re-runs the model on the whole sequence each step and is your correctness baseline.

- `Config.max_seq` is 64, and the cache is preallocated to it: prompt plus new tokens must fit. The reference `update` raises `ValueError` on overflow instead of silently writing past the end; do the same.
- For top-p, sort descending, compute `cumsum − probs` (the mass before each token), mark drops in sorted order, then `scatter` the mask back to the original order. Filtering the sorted logits and forgetting to unsort is the bug `test_top_p_works_on_unsorted_rows` exists for.
- Filters should return logits with `-inf` in dropped positions, not probabilities, so they compose: top-p after top-k sees the renormalized top-k distribution.

## Common bugs

- **Advancing `pos` inside `update`:** layer 1 then writes at the wrong offset. Symptom: prefill logits correct, decode logits wrong from the second layer on.
- **Mask built for `(t_new, t_new)`** instead of `(t_new, pos + t_new)`: shape error at best, a silently wrong mask when broadcasting at worst.
- **RoPE positions restarting at 0 for each chunk** (in your own models): chunked prefill then disagrees with the full forward.
- **Top-p that drops the crossing token** (`cumsum > p` instead of "mass before ≥ p"): nucleus sets are one token too small; the 0.6 and 0.9 test cases fail.
- **Temperature applied after the filters**, or to probabilities instead of logits.
- **Dropping the `generator` argument** on the way to `torch.multinomial`: the distribution tests still pass by luck most of the time, but your runs stop being reproducible.

## GPU scale-up: cached vs uncached decode

Time `generate` with and without the cache on a larger config. The given `generate` builds its cache in float32, so keep the model in float32 (or make your `generate` create the cache with the model's dtype and device).

```python
import time, torch
from labs._impl import load
kv = load("labs/07_kv_cache_sampling/x.py", "exercise")   # any path in the lab dir
cfg = kv.Config(vocab_size=4096, max_seq=1024, n_layer=8, n_head=8, n_kv_head=2, d_model=512)
model = kv.TinyLM(cfg).cuda().eval()
prompt = torch.randint(0, cfg.vocab_size, (8, 128), device="cuda")
for new in (32, 64, 128, 256, 512):
    for use_cache in (True, False):
        torch.cuda.synchronize(); t = time.perf_counter()
        kv.generate(model, prompt, new, use_cache=use_cache, temperature=0)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        print(f"new={new:4d} cache={use_cache!s:5}: {dt * 1e3:8.1f} ms  {8 * new / dt:9.0f} tok/s")
```

Predict first, then plot time against `new` for both modes:

1. Uncached work grows like `Σ (128 + t)` token-forwards, quadratic in `new`; cached work grows linearly. Where do the curves cross, and why not at `new = 1`?
2. A model this small is **launch-overhead-bound** at batch 8: each decode step is hundreds of small kernels. Estimate the per-step time from the cached curve and compare it with the time to read the weights at your measured bandwidth. The gap is the case for CUDA graphs and fusion ([inference §2.6](../../curriculum/07-inference.md#26-cuda-graphs)).
3. Repeat with batch 64. Per-token cost should barely move for the cached path: that is why batching decode is nearly free until memory or compute runs out.

CPU-only: remove `.cuda()` and the `torch.cuda.synchronize()` calls and use `device="cpu"`; pick a smaller config (`n_layer=4, d_model=256`) and `new` up to 256. The crossover still appears.

## Check yourself

1. How many bytes does `KVCache` allocate for `Config()` with batch 3 in float32? Derive it from the shape before running `nbytes()`.
2. Why can the cache store K and V after RoPE is applied, and what would break if it stored them before?
3. Chunked prefill gives identical logits. What does it cost relative to one big prefill, and what does it buy a server?
4. Why is top-p applied after top-k, and does the order matter?
5. Why does min-p stay permissive on a flat distribution while top-k does not?
6. Your cached and uncached greedy outputs agree for 10 tokens and then diverge. Name two likely causes.

<details><summary>Answers</summary>

1. `2 (K, V) × 2 layers × 3 × 2 KV heads × 64 positions × 16 head dim × 4 bytes = 98,304` bytes.
2. RoPE depends only on the token's own absolute position, which never changes once written, so the rotated K can be cached. Caching un-rotated K would force re-rotating every cached key at every step (and caching rotated Q is never needed).
3. It re-reads the KV of earlier chunks and launches more, smaller forwards, so total prefill time rises slightly. It buys bounded step time: long prompts no longer stall other users' decode steps, which protects inter-token latency.
4. Each filter renormalizes over what the previous one kept, so the order changes the result in general. The conventional order (temperature, top-k, top-p, min-p) is what libraries use; the important part is being consistent between experiments.
5. Min-p's cutoff is a fraction of the top probability: when the top token has 0.05, the cutoff is small and many tokens survive. Top-k keeps exactly k regardless.
6. The mask or RoPE positions are wrong at some offset (for example, only after the prefill length), or `pos` is advanced at the wrong time. Numerical near-ties in argmax can also flip in low precision; the tests use float32 and small models to avoid that.
</details>

## Stretch

- **Paged cache:** replace the contiguous cache with fixed 16-token blocks plus a per-sequence block table, as in vLLM. Keep `test_incremental_logits_match_full_forward` passing.
- **Prefix sharing:** two prompts with the same 32-token system prompt share its blocks (reference counts); measure the memory saved.
- **Continuous batching:** a loop that admits new requests into free batch slots at every step and retires finished ones. Plot throughput against arrival rate.
- **Static shapes for CUDA graphs:** always attend over the full preallocated cache with a mask instead of slicing `[:end]`, then try `torch.compile(mode="reduce-overhead")` on the decode step and measure the batch-1 speedup.
- **Speculative decoding** on top of your cache: see [lab 14](../14_speculative_decoding/README.md).
