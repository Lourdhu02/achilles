# 03 — Deep learning, the parts that matter at scale

Labs: [01 autograd](../labs/01_autograd/README.md), [02 training core](../labs/02_training_core/README.md).

## 1. Autodiff: what backward() actually does
A program is a DAG of ops, and reverse mode applies each op's VJP in reverse topological order, summing gradients where paths merge. For a scalar loss, one backward pass costs about 2× a forward pass whatever the parameter count, which is why deep learning is possible. The price is storing activations, hence checkpointing, recomputation (FlashAttention) and activation offloading. Forward mode (JVP) wins when there are few inputs and many outputs.

## 2. Initialization: keeping signals alive
`Var(y) = n_in·Var(w)·Var(x)`: choose `Var(w) = 2/n_in` for ReLU (Kaiming) or `1/n_in` for linear/tanh-like layers (LeCun/Xavier). In residual networks, scale branch outputs by `1/√(2L)` so the residual stream's variance stays independent of depth. **μP** (Tensor Programs V) goes further: it scales init and per-layer learning rates with width so the best hyperparameters found on a small model transfer to a large one. That is how labs tune 100B-parameter models on 100M-parameter proxies.

## 3. Normalization and the residual stream
Pre-norm (`x + f(norm(x))`) keeps an identity gradient path through depth; post-norm needs careful warmup. RMSNorm drops the centering. Normalized weights are scale-invariant, so weight decay acts as a learning-rate controller on them. Think of the residual stream as a shared communication channel that every layer reads from and writes to ([09](09-interpretability-and-safety.md)).

## 4. Optimizers
- **AdamW:** momentum plus a per-parameter RMS normalization, with bias correction and *decoupled* weight decay. LLM defaults: β = (0.9, 0.95), wd = 0.1, clip = 1.0. State: 8 bytes/param.
- **Muon:** orthogonalized momentum for 2-D weights (Newton–Schulz), AdamW for the rest. State: 4 bytes/param. Used at scale since 2025 (Moonshot's Kimi models).
- **Also know:** Lion (sign of momentum), Adafactor (factored second moments), Shampoo/SOAP (Kronecker preconditioners), and 8-bit optimizer states.

## 5. Schedules and batch size
Warmup (~1–2% of steps), then cosine to ~10% of the peak LR, or **WSD** (hold the peak, decay over the last 10–20%), which lets you branch cooldowns and continue training later. There is a critical batch size (gradient noise scale) past which bigger batches stop helping; it grows during training, hence batch-size ramps. For Adam, scaling the LR by about √k when the batch grows k× is a starting guess, not a law.

## 6. Mixed precision
bf16 compute with fp32 master weights and fp32 reductions (loss, softmax statistics, norms). fp16 needs dynamic loss scaling. FP8 training (DeepSeek-V3 style) needs fine-grained block scaling and high-precision accumulation.

## 7. Debugging a training run (the playbook)
1. Initial loss ≈ ln V? If not, check the init, the targets, and label shifting.
2. Overfit one batch to ~0 loss. If you can't, there is a bug, not a hyperparameter problem.
3. Check that data and labels line up by decoding a batch and reading it.
4. Log per-layer gradient norms, update/weight ratios (~1e-3 is healthy), max attention logits, and output logit norms.
5. LR range test. Loss flat means the LR is too low or the graph is detached. Oscillation or spikes mean the LR is too high, or there is a bad batch or logit growth.
6. Most bugs are data bugs: truncation, shuffling, masking, tokenizer mismatch, duplicated shards, eval leakage.
7. Gradient accumulation normalization: divide by total tokens, not by micro-batch count ([lab 02](../labs/02_training_core/README.md#7-clipping-and-accumulation)).

**Read:** Karpathy, *A Recipe for Training Neural Networks* (2019); Google *Deep Learning Tuning Playbook*; Wortsman et al. 2023 (small-scale proxies for instabilities); Yang et al. 2022 (μP).
