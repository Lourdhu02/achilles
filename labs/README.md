# Labs

Eighteen test-driven labs that rebuild the modern LLM stack. Each lab has a handout (`README.md`), a stub you implement (`exercise.py`), a test suite, and a reference solution (`solution.py`) to read after an honest attempt. Every test suite runs on a laptop CPU. The scale-up runs are optional, and sized for one 8 GB GPU or a free Colab or Kaggle T4.

| # | Lab | You build | Tests | Time (tests) | Scale-up run |
|---|---|---|:-:|---|---|
| 01 | [Autograd from scratch](01_autograd/README.md) | Reverse-mode autodiff in NumPy; an MLP trained with it | 37 | 8–12 h | — |
| 02 | [The training toolkit](02_training_core/README.md) | AdamW matching PyTorch to 1e-10, Muon, WSD schedules, clipping and accumulation | 24 | 8–10 h | — |
| 03 | [Napkin math](03_napkin_math/README.md) | Parameter, FLOP, KV-cache, roofline and MFU calculators | 18 | 5–7 h | `tools/measure_gpu.py` |
| 04 | [Byte-level BPE tokenizer](04_tokenizer/README.md) | Training and encoding with GPT-4 and o200k pre-tokenization | 33 | 6–8 h | Telugu tokenizer study |
| 05 | [A modern GPT](05_transformer/README.md) | RoPE, GQA, SwiGLU, QK-norm; a pretraining script | 17 | 10–14 h | **S1:** pretrain on TinyStories (`train.py`) |
| 06 | [Attention kernels](06_attention_kernels/README.md) | Online softmax, FlashAttention forward and backward, Triton kernels | 21 | 12–16 h | benchmark your kernel against SDPA |
| 07 | [KV cache and sampling](07_kv_cache_sampling/README.md) | KV cache, chunked prefill, top-k, top-p, min-p | 13 | 6–8 h | cached vs uncached decode |
| 08 | [Scaling laws](08_scaling_laws/README.md) | Power-law fits, Chinchilla and inference-aware sizing | 6 | 3–4 h | **S2:** your own IsoFLOP sweep |
| 09 | [Parallelism](09_parallelism/README.md) | Ring all-reduce, tensor parallelism, ZeRO memory; DDP on two CPU processes | 7 | 2–3 h | multi-GPU on a Kaggle T4 x2 |
| 10 | [LoRA](10_lora/README.md) | Adapters, merging, a test of the low-rank hypothesis | 5 | 3–4 h | **S4:** LoRA SFT of Qwen2.5-0.5B |
| 11 | [DPO](11_dpo/README.md) | DPO, IPO, SimPO; DPO's closed-form optimum, checked numerically | 6 | 3–4 h | DPO on your SFT model |
| 12 | [Policy gradients to GRPO](12_grpo/README.md) | REINFORCE, GRPO with clipping and a k3 KL penalty | 5 | 4–5 h | **S5:** GRPO on a verifiable task |
| 13 | [Quantization](13_quantization/README.md) | INT8, INT4 packing, NF4, SmoothQuant, GPTQ | 8 | 3–4 h | quality vs tokens/s |
| 14 | [Speculative decoding](14_speculative_decoding/README.md) | Exact rejection sampling, verified lossless | 6 | 2–3 h | 0.5B draft → 1.5B target |
| 15 | [Evaluation statistics](15_eval_stats/README.md) | CIs, paired tests, clustered SEs, pass@k, power, Bradley–Terry | 6 | 2–3 h | a leaderboard with error bars |
| 16 | [Mechanistic interpretability](16_interpretability/README.md) | Induction heads, activation patching, sparse autoencoders | 9 | 3–4 h | an SAE on your S1 model |
| 17 | [Retrieval for RAG](17_retrieval/README.md) | BM25, RRF, MMR, nDCG, an IVF index | 7 | 2–3 h | evaluate your own RAG |
| 18 | [Mixture of experts](18_moe/README.md) | Top-k routing, load balancing, capacity and dropping, total vs active params | 10 | 3–4 h | an MoE version of your S1 model |

Times are for a first honest attempt at the tests; the scale-up runs add a few hours to an overnight run each. [SETUP.md](../SETUP.md#what-you-can-do-on-each-tier) lists the minimum hardware for every scale-up.

## How to work a lab

1. **Read the handout** up to "What to implement". Derive anything it derives, on paper.
2. **Predict** the number the lab is about (a loss, a memory size, a speedup) and write it in your [journal](../journal/README.md).
3. **Implement** `exercise.py` one function at a time, running its tests as you go. Hints (`# HINT:`) sit where the code goes.
4. **Stuck for more than 30 minutes?** Re-read the handout's "Common bugs", then print shapes and compare against the formula. Read `solution.py` only for the one function you are stuck on.
5. **Measure and write up.** Compare your prediction with the measurement, and explain the gap.

## Commands

```bash
pytest labs/05_transformer -x                     # your implementation (exercise.py); stop at the first failure
pytest labs/05_transformer --impl=solution        # the reference
pytest labs/05_transformer -k rope                # only the tests whose names match
python tools/progress.py                          # scoreboard across all labs
python tools/make_exercises.py --force 08_scaling_laws   # regenerate a stub (destroys your work in it)
```

`LABS_IMPL=solution` works too, for example in an IDE's test runner. In `solution.py`, code between `# BEGIN SOLUTION` and `# END SOLUTION` is stripped into `exercise.py`, and `# HINT:` lines survive. Existing exercise files are never overwritten without `--force`, because they hold your work.

> [!TIP]
> Commit your `exercise.py` work to your own fork as you go. Then `git diff` shows exactly what you changed, your progress is backed up, and a public fork with all 18 labs passing is a portfolio piece in itself.
