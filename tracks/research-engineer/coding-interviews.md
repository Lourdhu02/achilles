# Coding interviews

## ML coding drills (from memory, timed, no references)
| drill | target time | source |
|---|---|---|
| Softmax cross-entropy forward and backward in NumPy | 10 min | lab 01 |
| MLP with manual backprop and SGD | 20 min | lab 01 |
| AdamW step | 10 min | lab 02 |
| Causal multi-head attention with GQA | 15 min | lab 05 |
| RoPE | 10 min | lab 05 |
| KV-cache decode loop | 20 min | lab 07 |
| Top-k / top-p / min-p sampling | 10 min | lab 07 |
| Beam search with length penalty | 25 min | extend lab 07 |
| BPE training and encoding | 30 min | lab 04 |
| Online softmax / tiled attention | 20 min | lab 06 |
| LoRA layer with merge | 10 min | lab 10 |
| DPO loss with correct log-prob masking | 10 min | lab 11 |
| GRPO advantages and clipped loss | 15 min | lab 12 |
| Bootstrap CI and pass@k | 10 min | lab 15 |
| BM25 and nDCG | 20 min | lab 17 |
| k-means; logistic regression with gradient descent | 15 min each | classic |

Rules: talk while you code, write a tiny test first, and state the complexity plus one failure mode at the end.

## Practical engineering drills
LRU cache with TTL · token-bucket rate limiter · thread-safe request batcher (collect for ≤ 10 ms or ≤ 32 items) · streaming JSON/SSE parser · retry with exponential backoff and jitter · a simple KV store with snapshots · an interval scheduler. Many frontier-lab coding rounds are like these: incremental, practical, judged on clean code.

## DSA (for Big Tech and some labs)
Target: LeetCode mediums in ≤ 25 min with narration. Maintenance: 3–4 h/week, pattern-based ([dsa-patterns.md](dsa-patterns.md)), ~150 well-understood problems beat 500 skimmed ones. Log each problem with its pattern and the insight you missed.
