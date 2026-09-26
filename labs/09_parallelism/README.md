# Lab 09 — Parallelism, simulated

**Build:** a step-by-step ring all-reduce, Megatron column- and row-parallel layers, the ZeRO memory model, the pipeline-bubble formula and data-parallel gradient averaging, all in NumPy. Then real DDP across two processes on your laptop CPU.
**Time:** 2–3 h for the tests, 2–4 h for the experiments · **Reads first:** [pretraining §5](../../curriculum/05-pretraining.md#5-distributed-training)
**Run:** `pytest labs/09_parallelism` (your code) · `pytest labs/09_parallelism --impl=solution` (reference). The tests are NumPy only and finish in about a second.

Every frontier model is trained on more GPUs than fit in one server, so large-scale jobs and research-engineer interviews both assume you can size a run on paper: which collective moves how many bytes, what each GPU holds in memory, and how much time idle pipeline stages waste. None of this needs a cluster. The arithmetic is exact, and this lab makes you derive it by building each piece.

---

## 1. Ring all-reduce

All-reduce: N ranks each hold a vector of S bytes; afterwards every rank holds the elementwise sum. The ring algorithm splits each vector into N chunks and runs two phases of N − 1 steps. In every step each rank sends one chunk to its right neighbour, all at the same time:

```
N = 3; rank r starts with chunks (a_r, b_r, c_r) and sends to rank r+1 (mod 3)

                 rank 0            rank 1            rank 2
reduce-scatter 0 c += c2           a += a0           b += b1
reduce-scatter 1 b += b1+b2        c += c0+c2        a += a0+a1
                 owns sum(b)       owns sum(c)       owns sum(a)
all-gather 0     a = sum (from 2)  b = sum (from 0)  c = sum (from 1)
all-gather 1     c = sum (from 2)  a = sum (from 0)  b = sum (from 1)
```

Each step sends S/N bytes per rank, so over 2(N − 1) steps each rank sends (and receives)

$$2(N-1)\cdot\frac{S}{N} = 2\,\frac{N-1}{N}\,S < 2S .$$

**Why this is bandwidth-optimal.** Follow one element. Its N contributions start on N ranks, and a one-way send can merge at most one partial sum into another, so the first complete sum exists only after at least N − 1 sends. At that moment the other N − 1 ranks lack it, and each must receive at least one more message: N − 1 more sends. So any algorithm needs at least 2(N − 1)S bytes in total, which is at least 2(N − 1)S/N on the busiest rank. The ring meets that bound on every rank at once.

Compare gathering on rank 0 and broadcasting back: rank 0 receives (N − 1)S, which grows with N. For the test's buffer (64 float64 values, S = 512 bytes) the ring sends 512, 768 and 896 bytes per rank for N = 2, 4, 8; rank 0 of the naive scheme would receive 512, 1,536 and 3,584.

The ring's weakness is latency: 2(N − 1) sequential steps, each with a fixed start-up cost, so time ≈ 2(N − 1)·latency + 2(N − 1)/N · S / bandwidth. For small messages and very large N, NCCL switches to tree algorithms.

**Worked:** all-reducing the bf16 gradients of a 7B model (S = 14 GB) over 8 GPUs sends 2 · 7/8 · 14 = 24.5 GB per GPU: 54 ms at 450 GB/s (NVLink 4, one direction), 0.49 s at 50 GB/s (400 Gb/s InfiniBand).

## 2. Data parallelism: why averaging is exact

Each rank holds the whole model and 1/N of the batch. For a loss that is a mean over examples and N equal shards $B_k$ of the batch $B$,

$$\nabla L_B = \frac{1}{|B|}\sum_{i\in B}\nabla \ell_i = \frac{1}{N}\sum_{k=1}^{N}\nabla L_{B_k}, \qquad \nabla L_{B_k} = \frac{N}{|B|}\sum_{i\in B_k}\nabla \ell_i ,$$

so one all-reduce (sum, then divide by N) gives every replica the full-batch gradient, and all replicas take the identical step. `data_parallel_grad` checks this on least squares, where the gradient of ½·mean((Xw − y)²) is Xᵀ(Xw − y)/n.

- **Equal shards matter.** With 65 rows over 8 ranks, the plain average missed the full-batch gradient by 1.4% of its largest component in one random draw. `DistributedSampler` pads or drops the remainder. With variable-length sequences, weight by tokens: sum the per-token losses and divide by the global token count.
- **Overlap.** PyTorch DDP groups gradients into buckets (25 MB by default) and all-reduces each bucket as soon as backward has produced it, hiding communication behind the rest of backward.

## 3. Tensor parallelism: the Megatron MLP

The MLP computes Z = GeLU(XA)B with A of shape (d, 4d) and B of shape (4d, d). Megatron-LM (Shoeybi et al., [1909.08053](https://arxiv.org/abs/1909.08053)) splits it over n GPUs with no communication in the middle:

```
X (replicated) ──┬── X A_1 ── GeLU ── Y_1 B_1 ──┐
                 ├── X A_2 ── GeLU ── Y_2 B_2 ──┼── all-reduce (sum) ──► Z (replicated)
                 └── ...                        ┘
A = [A_1 | A_2 | ...] split by columns        B = [B_1 ; B_2 ; ...] split by rows
```

- **Column-parallel first:** rank i computes X A_i, a column block of XA. GeLU is elementwise, so GeLU(X A_i) is exactly the i-th column block of GeLU(XA).
- **Row-parallel second:** that block is exactly the input slice that multiplies row block B_i, and YB = Σᵢ Yᵢ Bᵢ. One all-reduce sums the partial products.

The order matters. Split A by rows and you hold partial sums of XA, and GeLU does not distribute over a sum: with the lab's GeLU, GeLU(1) + GeLU(1) = 1.68 but GeLU(2) = 1.95. You would need an extra all-reduce before the nonlinearity.

Megatron writes this with two operators: f before the block (identity forward, all-reduce backward, which sums the partial gradients with respect to X) and g after it (all-reduce forward, identity backward). So an MLP block costs **one all-reduce in forward and one in backward**. Attention splits the same way (Q, K, V column-parallel by heads, output projection row-parallel), giving 2 + 2 all-reduces per layer per micro-batch.

**Worked:** each all-reduce carries the activation, b·s·h·2 bytes: 33.5 MB for s = h = 4096, b = 1. With TP = 8 each GPU sends 2 · 7/8 · 33.5 = 58.7 MB per all-reduce; 4 per layer over 32 layers is 7.5 GB per micro-batch: 17 ms over NVLink, 150 ms over InfiniBand. This communication sits on the critical path, which is why TP stays inside a node.

## 4. ZeRO: where 16 bytes per parameter go

Mixed-precision Adam holds 2 bytes of bf16 weights, 2 of bf16 gradients and 12 of fp32 optimizer state (master weights, momentum, variance) per parameter: $2\Psi + 2\Psi + 12\Psi = 16\Psi$. ZeRO (Rajbhandari et al., [1910.02054](https://arxiv.org/abs/1910.02054)) shards these across the N data-parallel ranks:

| stage | shards | bytes per GPU | 7.5B, N = 64 (paper and test) | 7B, N = 8 |
|---|---|---|---|---|
| 0 (plain DP) | nothing | 16Ψ | 120 GB | 112 GB |
| 1 | optimizer state | 4Ψ + 12Ψ/N | 31.4 GB | 28 + 10.5 = 38.5 GB |
| 2 | + gradients | 2Ψ + 14Ψ/N | 16.6 GB | 14 + 12.25 = 26.25 GB |
| 3 | + weights | 16Ψ/N | 1.9 GB | 14 GB |

Read the 7B column against 40 GB cards: plain DP is hopeless, stage 1 leaves 1.5 GB for activations (not enough in practice), stage 2 leaves 13.75 GB and stage 3 leaves 26 GB. These are model states only; activations, temporary buffers and fragmentation come on top.

Stages 1 and 2 move the same bytes as plain DP, because an all-reduce already is a reduce-scatter plus an all-gather: reduce-scatter the gradients so each rank gets the slice it owns, update that slice, all-gather the new weights. Stage 3 all-gathers the weights again in backward: 1.5× DP traffic. PyTorch FSDP implements the same idea (full sharding is roughly stage 3; see Zhao et al., [2304.11277](https://arxiv.org/abs/2304.11277)). The `k` argument is the optimizer's bytes per parameter: 12 for Adam with fp32 master weights, 8 for SGD with momentum.

## 5. Pipeline parallelism and the bubble

Split the layers into p stages and the batch into m micro-batches. In a GPipe schedule (Huang et al., [1811.06965](https://arxiv.org/abs/1811.06965)) every stage runs all forwards, then all backwards:

```
time     0   1   2   3   4   5   6   7   8   9   10  11  12  13
stage 0  F1  F2  F3  F4  .   .   .   .   .   .   B4  B3  B2  B1
stage 1  .   F1  F2  F3  F4  .   .   .   .   B4  B3  B2  B1  .
stage 2  .   .   F1  F2  F3  F4  .   .   B4  B3  B2  B1  .   .
stage 3  .   .   .   F1  F2  F3  F4  B4  B3  B2  B1  .   .   .
```

The schedule lasts (m + p − 1)(t_F + t_B) and each stage works m(t_F + t_B) of it, so the idle fraction is

$$\text{bubble} = \frac{p-1}{m+p-1}\qquad (p = 4,\ m = 4:\ 3/7 = 43\%).$$

| idle fraction | m = 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| p = 2 | 50% | 20% | 11.1% | 5.9% | 3.0% | 1.5% |
| p = 4 | 75% | 42.9% | 27.3% | 15.8% | 8.6% | 4.5% |
| p = 8 | 87.5% | 63.6% | 46.7% | 30.4% | 18.0% | 9.9% |

1F1B schedules alternate one forward and one backward in steady state: same bubble, but at most p micro-batches of activations alive instead of m. Interleaved 1F1B (Narayanan et al., [2104.04473](https://arxiv.org/abs/2104.04473)) gives each GPU v chunks of layers and divides the bubble by v, at the cost of more point-to-point messages. That paper defines the bubble as idle time over ideal time, (p − 1)/m; this lab uses idle over total. They agree when m ≫ p; say which one you mean.

## 6. Choosing a layout

| strategy | splits | memory per GPU | communication | use when |
|---|---|---|---|---|
| DP (DDP) | batch | full 16Ψ + activations | all-reduce grads once per step, overlapped | model and optimizer fit on one GPU |
| FSDP / ZeRO-3 | batch; states sharded | 16Ψ/N + activations + gathered layers | all-gather weights, reduce-scatter grads: 1.5× DP | model does not fit; links are fast |
| TP | each matmul | weights and activation work ÷ n | 2 all-reduces per layer per pass, on the critical path | layers too big, or latency matters; within NVLink |
| PP | layers | layers ÷ p | small point-to-point activations | across nodes with slow links; costs the bubble |

Typical large runs combine them: TP inside a node, PP across nodes when needed, DP or FSDP outermost. The curriculum's [design order](../../curriculum/05-pretraining.md#5-distributed-training) and its [7B planning exercise](../../curriculum/05-pretraining.md#8-exercise-plan-a-7b-run-on-paper) apply this.

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `ring_all_reduce(arrays)` → `(results, sent)` | `test_ring_all_reduce_sums_and_moves_2_n_minus_1_over_n` | for N = 2, 4, 8 every rank ends with the elementwise sum, and `sent[r] == 2(N−1)/N · 64 · 8` bytes for every rank |
| 2 | `column_parallel(x, w_shards)` | `test_tensor_parallel_linear_layers` | concatenating `x @ W_i` over 4 column shards of a 12 × 8 `W` equals `x @ W` |
| 3 | `row_parallel(x_shards, w_shards)` | same | summing `x_i @ W_i` over 3 input slices and matching row blocks equals `x @ W` |
| 4 | `megatron_mlp(x, A, B, n)` | `test_megatron_mlp_equals_unsharded` | with n = 4, A 16 × 64 and B 64 × 16, equals `gelu(x @ A) @ B` |
| 5 | `zero_memory_bytes(n_params, n_gpus, stage, k=12)` | `test_zero_paper_numbers` | 7.5B parameters on 64 GPUs: 120, 31.4, 16.6, 1.875 GB (1 GB = 10⁹ bytes) |
| 6 | `pipeline_bubble(stages, micro_batches)` | `test_pipeline_bubble_and_data_parallel` | 0.75 for p = 4, m = 1 and 3/32 for p = 4, m = 29 |
| 7 | `data_parallel_grad(X, y, w, n)` | same | the mean of 8 shard gradients equals `Xᵀ(Xw − y)/64` |

Given: `gelu` (tanh form), and in `ring_all_reduce` the float64 chunk lists, the `sent` counters and the return line.

## Tips

> [!TIP]
> In each ring step, first build every rank's message (sender, chunk index, a copy of the data), then deliver them all. That is what "simultaneous" means on real hardware, and it keeps the simulation honest whatever schedule you choose.

- Index bookkeeping: in reduce-scatter step t, rank r sends chunk (r − t) mod N; afterwards rank r owns the finished chunk (r + 1) mod N, so in all-gather step t it sends chunk (r + 1 − t) mod N. Trace N = 3 against the diagram in §1 first, and count bytes with `data.nbytes` on what you send.
- `megatron_mlp` is two `np.array_split` calls (`axis=1` for A, `axis=0` for B) and one `row_parallel` call.
- `zero_memory_bytes`: start from 2Ψ, 2Ψ and kΨ and divide one more term by N at each stage.

## Common bugs

- **Starting the all-gather from the wrong chunk** (chunk r instead of the finished (r + 1) mod N): receivers overwrite correct sums with partial ones.
- **Counting bytes per step for the whole array**, or counting received and sent bytes together: `sent` comes out N× or 2× too large.
- **GeLU after the sum**, or A split by rows: `test_megatron_mlp_equals_unsharded` fails because GeLU is not additive.
- **Sharding the wrong term per stage** (weights at stage 2), or using GiB (2³⁰) where the paper and the test use 10⁹.
- **Using (p − 1)/m**, the other bubble convention: 3/29 instead of 3/32.
- **Summing shard gradients instead of averaging** (N× too large), or dividing each shard gradient by the full batch size and then averaging (N× too small).

## CPU experiments (no GPU needed)

**Real DDP on two processes.** Save this as `ddp_demo.py` and run `torchrun --nproc_per_node=2 ddp_demo.py`:

```python
import time
import torch
import torch.distributed as dist
import torch.nn as nn

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                  # same data and init on every rank
X, y = torch.randn(64, 32), torch.randn(64, 1)
model = nn.Linear(32, 1)
ddp = nn.parallel.DistributedDataParallel(model)

shard = slice(rank * 64 // world, (rank + 1) * 64 // world)
loss = ((ddp(X[shard]) - y[shard]) ** 2).mean()
loss.backward()                                       # DDP all-reduces (averages) grads here

ref = nn.Linear(32, 1)
ref.load_state_dict(model.state_dict())
((ref(X) - y) ** 2).mean().backward()
assert torch.allclose(model.weight.grad, ref.weight.grad, atol=1e-6)

buf = torch.zeros(25_000_000)                         # 100 MB of fp32
dist.barrier(); t = time.time()
for _ in range(5):
    dist.all_reduce(buf)
dt = (time.time() - t) / 5
sent = 2 * (world - 1) / world * buf.numel() * 4       # ring all-reduce bytes per rank
if rank == 0:
    print(f"grads match full batch; all-reduce 100 MB: {dt * 1e3:.0f} ms, ~{sent / dt / 1e9:.1f} GB/s per rank")
dist.destroy_process_group()
```

Rank 0 prints one line. On one cloud CPU container it printed 83 ms and about 1.2 GB/s per rank; your laptop will differ. The GB/s figure is what nccl-tests calls bus bandwidth: bytes actually sent per rank, not buffer size over time. Warnings about `OMP_NUM_THREADS` or IPv6 sockets are harmless. On native Windows, gloo is the only backend; if torchrun fails with an error that mentions libuv, set `USE_LIBUV=0` in the environment, or run it in WSL2.

Three follow-ups:

1. **Sweep the message size** from 4 KB to 400 MB and fit time = a + bytes/B. The intercept a is the latency term; find the size below which it dominates. Repeat with `--nproc_per_node=4` and predict the change first (per-rank bytes go from 1.0S to 1.5S).
2. **Break the equal-shard assumption:** call `data_parallel_grad` with 65 rows, then fix it by weighting each shard's gradient by its size.
3. **Draw the schedule:** write a 20-line simulator that prints the GPipe grid above for any p and m, counts idle slots, and checks `pipeline_bubble`. Add 1F1B and count the peak number of micro-batches whose activations are alive.

## GPU scale-up (8 GB)

One GPU cannot show communication, but it can check the memory model, which decides whether a run fits. A free Colab or Kaggle T4 (16 GB) works too; T4 has no bf16 tensor cores, so use fp16 autocast with a `GradScaler`, or fp32.

1. **Predict, then measure, model-state memory.** Train your [lab 05](../05_transformer/README.md) GPT for a few steps with AdamW under bf16 autocast. Autocast keeps parameters in fp32, so model states are 4 + 4 + 8 = 16 bytes per parameter, the same total as 2 + 2 + 12. Compare `torch.cuda.max_memory_allocated()` with 16Ψ plus your activation estimate, then enable activation checkpointing and see which term moved.
2. **Why full fine-tuning a 0.5B model does not fit.** Qwen2.5-0.5B has 494 M parameters: 7.9 GB of model state before any activation. Confirm it, then see [lab 10](../10_lora/README.md) for what does fit.
3. **FSDP on one GPU.** On Linux, WSL2 or Colab (native Windows builds of PyTorch have no NCCL), launch with `torchrun --nproc_per_node=1`, apply `fully_shard` from `torch.distributed.fsdp` to each block and then the model, and read `torch.cuda.memory_summary()`. At N = 1 there is nothing to shard, so memory matches the plain model: 16Ψ/N at N = 1. Then pass `offload_policy=CPUOffloadPolicy()` and measure the memory saved and the step time lost.
4. **Two real GPUs.** Kaggle has offered a two-T4 accelerator option (check the current menu). Run `ddp_demo.py` with `"nccl"`, tensors on `cuda:{LOCAL_RANK}`, and compare the bandwidth with gloo; then run the Megatron MLP across the two GPUs with `dist.all_reduce` and check it against the unsharded result.

## Check yourself

1. Ring all-reduce of a 1 GB buffer: bytes sent per GPU with 8 GPUs, and with 64? What does grow with N?
2. Why is the first Megatron MLP layer column-parallel and the second row-parallel, and not the other way round?
3. A 7B model on 8 GPUs with mixed-precision Adam: model-state memory per GPU under ZeRO-2 and ZeRO-3, and what ZeRO-3 costs you.
4. p = 8, m = 16: what fraction of time is idle, and how can you reduce it without changing the global batch?
5. Why does TP usually stay inside a node while DP and FSDP span nodes?
6. Each DDP rank computes the mean loss over its own non-padding tokens, and ranks have different token counts. What is wrong with averaging those gradients, and what is the fix?

<details><summary>Answers</summary>

1. 2 · 7/8 · 1 GB = 1.75 GB with 8 GPUs; 2 · 63/64 = 1.97 GB with 64. The number of sequential steps grows, from 14 to 126, so the latency term grows with N while the bandwidth term barely changes.
2. Column-parallel output is already split along the hidden dimension, and GeLU is elementwise, so each rank applies it locally and the result is exactly the input slice the row-parallel layer needs. The block then needs one all-reduce. The other order produces partial sums before the nonlinearity, which needs an extra all-reduce because GeLU(a + b) ≠ GeLU(a) + GeLU(b).
3. ZeRO-2: 2Ψ + 14Ψ/8 = 14 + 12.25 = 26.25 GB. ZeRO-3: 16Ψ/8 = 14 GB. ZeRO-3 all-gathers the weights in forward and again in backward: 1.5× the communication of plain DP, and the gathers must be prefetched to hide them.
4. 7/23 = 30.4%. Use an interleaved schedule (the bubble falls by the number of chunks per GPU), use more, smaller micro-batches if the GPU stays efficient at the smaller size, or use fewer pipeline stages by sharding with FSDP or TP instead.
5. TP puts two all-reduces per layer per pass on the critical path: 7.5 GB per micro-batch in the §3 example, 17 ms over NVLink but 150 ms over InfiniBand, against ~50 ms of compute. DP's single gradient all-reduce per step overlaps with backward, so slower links are acceptable.
6. Each rank's gradient is a mean over a different number of tokens, so averaging ranks weights tokens on short-sequence ranks more heavily than the full-batch gradient would. Sum the per-token losses on each rank, all-reduce the global token count, and divide by it.
</details>

## Stretch

- **Ring all-reduce for real:** implement it with `dist.isend` and `dist.irecv` under gloo and compare its time with `dist.all_reduce`.
- **Tensor-parallel autograd:** write f and g as `torch.autograd.Function`s and check that the gradients of a 2-process Megatron MLP match the unsharded MLP.
- **Hierarchical all-reduce:** reduce-scatter inside each node, all-reduce across nodes, all-gather inside the node. Compute the inter-node bytes and compare with a flat ring (the idea behind HSDP).
- **Sequence parallelism** (Korthikanti et al., [2205.05198](https://arxiv.org/abs/2205.05198)): replace each TP all-reduce by a reduce-scatter and an all-gather, shard LayerNorm and dropout activations along the sequence, and show the communication volume is unchanged while activation memory falls.
