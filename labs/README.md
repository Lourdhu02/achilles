# Labs

| # | lab | status | GPU scale-up (RTX 5060) |
|---|---|---|---|
| 01 | [Autograd from scratch](01_autograd/README.md) | complete | — |
| 02 | [Training toolkit: AdamW, Muon, schedules](02_training_core/README.md) | complete | — |
| 03 | [Napkin math](03_napkin_math/README.md) | complete | `tools/measure_gpu.py` |
| 04 | [Byte-level BPE tokenizer](04_tokenizer/README.md) | complete | Telugu tokenizer study |
| 05 | [Modern GPT (RoPE, GQA, SwiGLU)](05_transformer/README.md) | complete | **S1** pretrain on TinyStories (`train.py`) |
| 06 | [FlashAttention in PyTorch and Triton](06_attention_kernels/README.md) | complete | benchmark your kernel vs SDPA |
| 07 | [KV cache and sampling](07_kv_cache_sampling/README.md) | complete | cached vs uncached decode |
| 08 | [Scaling laws](08_scaling_laws/README.md) | complete | **S2** your own IsoFLOP sweep |
| 09 | [Parallelism, simulated](09_parallelism/README.md) | complete | DDP/FSDP |
| 10 | [LoRA](10_lora/README.md) | complete | **S4** LoRA SFT Qwen2.5-0.5B |
| 11 | [DPO](11_dpo/README.md) | complete | DPO on your SFT model |
| 12 | [Policy gradients → GRPO](12_grpo/README.md) | complete | **S5** GRPO on a verifiable task |
| 13 | [Quantization (INT8/INT4/NF4/GPTQ)](13_quantization/README.md) | complete | quality vs tokens/s |
| 14 | [Speculative decoding](14_speculative_decoding/README.md) | complete | 0.5B draft → 1.5B target |
| 15 | [Evaluation statistics](15_eval_stats/README.md) | complete | leaderboard with CIs |
| 16 | [Mechanistic interpretability](16_interpretability/README.md) | complete | SAE on your S1 model |
| 17 | [Retrieval for RAG](17_retrieval/README.md) | complete | eval your own RAG |

## Workflow
```bash
pytest labs/05_transformer -x                  # your implementation (exercise.py)
pytest labs/05_transformer --impl=solution     # the reference
python tools/progress.py                       # scoreboard
python tools/make_exercises.py --force 08_scaling_laws  # regenerate a stub (destroys your work in it)
```
In `solution.py`, code between `# BEGIN SOLUTION` and `# END SOLUTION` is stripped into `exercise.py`, and `# HINT:` lines survive. Existing exercise files are never overwritten without `--force`, because they hold your work.
