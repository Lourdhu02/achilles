# Lab 09 — Parallelism, simulated

**Run:** `pytest labs/09_parallelism` (your code) · `pytest labs/09_parallelism --impl=solution` (reference)

**Reads first:** [pretraining §5](../../curriculum/05-pretraining.md#5-distributed-training)

## What you implement (the tests check each property)
- `ring_all_reduce(chunks)`: simulate reduce-scatter then all-gather over N ranks (NumPy). Test: every rank ends with the sum, and bytes sent per rank = `2(N−1)/N·size`.
- `column_parallel` / `row_parallel` linear layers and the Megatron MLP (column → GeLU → row → one all-reduce). Test: equal to the unsharded result.
- `zero_memory_bytes(Ψ, N, stage)`: test against the ZeRO paper (7.5B params, 64 GPUs: 120 / 31.4 / 16.6 / 1.9 GB).
- `pipeline_bubble(p, m) = (p−1)/(m+p−1)`.
- Data-parallel equivalence: averaged shard gradients equal the full-batch gradient.

## GPU scale-up / stretch
Run real DDP on CPU with `torchrun --nproc_per_node=2` and the `gloo` backend. Then FSDP-wrap your lab-05 GPT on one GPU and read the memory numbers in `torch.cuda.memory_summary`.
