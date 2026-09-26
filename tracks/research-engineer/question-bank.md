# Question bank: core AI

136 questions that core-AI interviews at frontier labs and Big Tech actually probe, from fundamentals to open problems. Each has a compact, checked answer, a difficulty tag and the follow-up an interviewer is likely to ask next.
Answer out loud first, then open the answer. The follow-up is where interviews are decided.

**Difficulty tags.** `L1` fundamentals: expected of anyone in an ML role. `L2` practitioner: expected of a research engineer who has trained and served models. `L3` frontier: open or recent material where a strong answer shows you read and ran things.

## Contents

- [How to practice this bank](#how-to-practice-this-bank)
- [1. Math and probability](#1-math-and-probability) (Q1–Q10)
- [2. Deep learning and optimization](#2-deep-learning-and-optimization) (Q11–Q23)
- [3. Transformers and architecture](#3-transformers-and-architecture) (Q24–Q35)
- [4. Tokenization](#4-tokenization) (Q36–Q43)
- [5. Pretraining, data and scaling](#5-pretraining-data-and-scaling) (Q44–Q54)
- [6. Distributed training](#6-distributed-training) (Q55–Q63)
- [7. Post-training and RL](#7-post-training-and-rl) (Q64–Q77)
- [8. Inference and serving](#8-inference-and-serving) (Q78–Q88)
- [9. Quantization and number formats](#9-quantization-and-number-formats) (Q89–Q96)
- [10. GPU kernels and performance](#10-gpu-kernels-and-performance) (Q97–Q106)
- [11. Evals and statistics](#11-evals-and-statistics) (Q107–Q116)
- [12. Interpretability and safety](#12-interpretability-and-safety) (Q117–Q124)
- [13. Applied LLM systems](#13-applied-llm-systems) (Q125–Q131)
- [14. Napkin math](#14-napkin-math) (Q132–Q136)
- [Probes about your own work](#probes-about-your-own-work)

Reference hardware used in the numbers below (verify spec sheets before quoting in an interview): H100 SXM ≈ 989 dense bf16 TFLOP/s, 80 GB HBM3 at ≈ 3.35 TB/s, so the ridge point is ≈ 295 FLOP/byte. NVLink ≈ 900 GB/s bidirectional per GPU; InfiniBand NDR ≈ 50 GB/s per port.

## How to practice this bank

1. **Out loud, timed, closed book.** Read the question, set a 2-minute timer, answer as if at a whiteboard. Only then open the answer. Silent "yes, I know that" does not count.
2. **Grade yourself 0–2.** 0: wrong or blank. 1: right idea, missing a formula, number or caveat. 2: matches the answer and you could handle the follow-up. Log the score with the date in your journal.
3. **Spaced repetition.** Revisit 0s after 2 days, 1s after a week, 2s after a month. A spreadsheet with a "next review" column is enough; Anki works if you already use it.
4. **Do the follow-up every time.** Interviewers go three "why"s deep on one topic rather than asking ten topics once. The follow-up is the second "why"; invent the third yourself.
5. **Tie each answer to a lab.** If an answer names a mechanism you implemented (attention, DPO, GPTQ, pass@k), re-derive it from your own code, not from memory of this page. If you have not implemented it, that is your next lab session.
6. **Mix sections.** After the first pass, draw 10 random question numbers per session. Real interviews jump from KV caches to p-values in one round.
7. **Mock with a partner monthly.** One person reads a question and the follow-up, the other answers, then swap. Record it; listen for hedging and missing numbers.

> [!TIP]
> Before your first pass, skim the tags. If you are two years into an ML job, aim for all `L1` at score 2 within four weeks, `L2` within three months, and pick one section's `L3` questions to own completely; that becomes the topic you steer deep dives toward.

> [!TIP]
> When you do not know an answer in a real interview, say what you would compute or measure to find out. "I'd expect decode to be memory-bound; I'd check by comparing achieved bandwidth with the spec" earns more credit than a confident guess.

---

## 1. Math and probability

<details>
<summary><b>Q1</b> · <code>L1</code> · What is the gradient of softmax cross-entropy with respect to the logits?</summary>

**Answer.** For logits z, p = softmax(z) and true class y: ∂L/∂z = p − onehot(y). With a mean over N examples, divide by N. The expression is cheap and stable, which is why frameworks fuse softmax and cross-entropy into one op (computed via log-sum-exp) instead of differentiating through a separate softmax.

**Follow-up:** *What changes with label smoothing ε over V classes?* The target becomes q = (1 − ε)·onehot(y) + ε/V, and the gradient is p − q. The model can no longer drive the correct logit to infinity.

</details>

<details>
<summary><b>Q2</b> · <code>L1</code> · Write the Jacobian of softmax and its vector-Jacobian product.</summary>

**Answer.** ∂p_i/∂z_j = p_i(δ_ij − p_j), i.e. J = diag(p) − ppᵀ. For an upstream gradient g, the VJP is dz = p ⊙ (g − ⟨g, p⟩). You never build J: it is V×V (for V = 128k that is 16 billion entries), while the VJP is O(V).

**Follow-up:** *J is singular. Why, and what does that mean?* J·1 = p − p·(pᵀ1) = 0, so adding a constant to all logits does not change the output. That is why you can subtract the max before exponentiating.

</details>

<details>
<summary><b>Q3</b> · <code>L1</code> · Backward pass of Y = XW, with X of shape N×d and W of shape d×k. How many FLOPs relative to the forward pass?</summary>

**Answer.** dX = dY·Wᵀ (N×k · k×d) and dW = Xᵀ·dY (d×N · N×k). Each is a matmul with the same 2Ndk FLOPs as the forward, so backward ≈ 2× forward and training ≈ 3× forward. That is where 6N FLOPs per token for training comes from (2N forward + 4N backward).

**Follow-up:** *Which of the two backward matmuls can you skip, and when?* dX for the first layer (inputs need no gradient) and dW for frozen weights. In LoRA fine-tuning, the frozen base still needs dX to propagate the gradient, so backward cost drops by roughly a third, not to zero.

</details>

<details>
<summary><b>Q4</b> · <code>L2</code> · Derive the backward pass of RMSNorm.</summary>

**Answer.** y = γ ⊙ x̂ with x̂ = x/r, r = √(mean(x²) + ε). Let g = dy ⊙ γ. Then dx = (1/r)·(g − x̂·mean(g ⊙ x̂)) and dγ = Σ over tokens of dy ⊙ x̂. For LayerNorm (x̂ = (x − μ)/σ) there is one more centering term: dx = (1/σ)·(g − mean(g) − x̂·mean(g ⊙ x̂)).

**Follow-up:** *Why do fused norm kernels save the reciprocal r (or σ) per row?* Recomputing it in backward means rereading x, and norms are memory-bound, so saving one scalar per row is cheaper than an extra pass over d elements.

</details>

<details>
<summary><b>Q5</b> · <code>L1</code> · Forward vs reverse KL: which is which, and where does each appear in LLM training?</summary>

**Answer.** KL(p‖q) = E_p[log p/q], and it is not symmetric. Minimizing forward KL(p_data‖q_θ) over θ is maximum likelihood; it is mass-covering (q is penalized heavily wherever p has mass and q does not). Minimizing reverse KL(q_θ‖p) is mode-seeking (q is penalized for putting mass where p has none, so it can collapse onto one mode). The RLHF penalty KL(π_θ‖π_ref) is reverse KL with respect to the policy.

**Follow-up:** *Why is minimizing cross-entropy the same as minimizing KL?* H(p, q) = H(p) + KL(p‖q), and H(p) of the data does not depend on θ.

</details>

<details>
<summary><b>Q6</b> · <code>L2</code> · Derive the policy-gradient (log-derivative) estimator and explain why a baseline is allowed.</summary>

**Answer.** ∇_θ E_{x∼π_θ}[R(x)] = Σ_x R(x)∇π_θ(x) = Σ_x R(x)π_θ(x)∇log π_θ(x) = E[R(x)∇log π_θ(x)]. Subtracting a baseline b that does not depend on x leaves it unbiased, because E[∇log π_θ(x)] = ∇Σ_x π_θ(x) = ∇1 = 0. A good baseline (≈ E[R]) reduces variance a lot.

**Follow-up:** *How does GRPO choose its baseline?* The mean reward of the G samples for the same prompt (then it divides by their standard deviation). PPO instead learns a value function.

</details>

<details>
<summary><b>Q7</b> · <code>L3</code> · Compare the k1, k2 and k3 estimators of KL(π‖π_ref) from samples of π. What gradient do you get if you put k3 in the loss?</summary>

**Answer.** With x ∼ π and r = π_ref(x)/π(x): k1 = −log r (unbiased, high variance, can be negative); k2 = ½(log r)² (biased, low variance); k3 = (r − 1) − log r (unbiased because E_π[r − 1] = 0, always ≥ 0, low variance). GRPO uses k3.

**Follow-up:** *Differentiate E_π[k3] treating the samples as fixed, as a loss term does.* ∇k3 = −r∇log π + ∇log π, so the expected gradient is E_π[(1 − r)∇log π] = Σ(π − π_ref)∇log π = −Σ π_ref ∇log π = ∇_θ KL(π_ref‖π). That is the gradient of the forward KL, not the reverse KL you estimated. The two agree to first order when π ≈ π_ref, which is why it works in practice, but it is a real bias to know about.

</details>

<details>
<summary><b>Q8</b> · <code>L1</code> · Why compute log-sum-exp with the max subtracted?</summary>

**Answer.** LSE(z) = m + log Σ exp(z_i − m) with m = max z. Without it, exp overflows (fp32 overflows above ≈ 88.7; bf16 has the same exponent range). With it, the largest term is exp(0) = 1 and nothing overflows. The gradient of LSE is softmax(z).

**Follow-up:** *How do you compute it in one pass over a stream?* Keep a running max m and running sum s; for a new x, m' = max(m, x), s' = s·exp(m − m') + exp(x − m'). That is online softmax, the core of FlashAttention.

</details>

<details>
<summary><b>Q9</b> · <code>L2</code> · For gradient descent on a quadratic with Hessian H, what learning rate is stable, and what sets the convergence speed?</summary>

**Answer.** The error evolves as e ← (I − ηH)e, so it converges iff |1 − ηλ_i| < 1 for all eigenvalues, i.e. η < 2/λ_max. The slowest direction contracts by (1 − ηλ_min) per step, so speed is governed by the condition number κ = λ_max/λ_min.

**Follow-up:** *What happens in real networks?* Full-batch GD tends to raise the top Hessian eigenvalue until it hovers near 2/η, the "edge of stability" (Cohen et al., 2021), and training keeps making progress non-monotonically. Warmup and adaptive optimizers are partly about managing this sharpness.

</details>

<details>
<summary><b>Q10</b> · <code>L2</code> · What is the best rank-r approximation of a matrix, and how does that connect to LoRA?</summary>

**Answer.** Eckart–Young: the truncated SVD, keeping the top r singular values and vectors, is optimal in both Frobenius and spectral norm. The squared Frobenius error is Σ_{i>r} σ_i². LoRA assumes the fine-tuning update ΔW has most of its energy in a few singular directions.

**Follow-up:** *How would you test that assumption?* Full-fine-tune a small model, compute the SVD of ΔW per layer and plot cumulative energy vs rank; compare with LoRA at several ranks on held-out loss. Lab 10 does exactly this.

</details>

---

## 2. Deep learning and optimization

<details>
<summary><b>Q11</b> · <code>L1</code> · Why does Adam's first update have magnitude ≈ lr regardless of gradient scale?</summary>

**Answer.** After one step, bias-corrected m̂ = g and v̂ = g², so the update is lr·g/(|g| + ε) ≈ lr·sign(g). Adam normalizes each coordinate by its own recent gradient scale, which makes it behave like sign descent with a per-parameter trust region.

**Follow-up:** *When does ε matter?* When √v̂ is comparable to ε (tiny or rarely updated parameters, e.g. embeddings of rare tokens). A large ε makes Adam act like momentum SGD with learning rate lr/ε for those parameters.

</details>

<details>
<summary><b>Q12</b> · <code>L1</code> · Adam with L2 regularization vs AdamW: what is the difference?</summary>

**Answer.** With L2, the decay term λθ is added to the gradient and then divided by √v̂, so parameters with large gradient history are barely regularized. AdamW applies decay separately: θ ← θ − ηλθ − η·m̂/(√v̂ + ε), shrinking every parameter by the same factor (1 − ηλ).

**Follow-up:** *Which parameters do you exclude from decay?* Usually norm gains and biases, often embeddings. LLM pretraining commonly uses λ = 0.1 on matrices. Note that the effective decay is ηλ, so it changes with the schedule.

</details>

<details>
<summary><b>Q13</b> · <code>L1</code> · Why use learning-rate warmup?</summary>

**Answer.** Early in training the loss surface is sharp and Adam's second-moment estimates are based on few steps. Full-size steps can overshoot into unstable regions (loss spikes, divergence, attention-logit blowup). Warmup lets the statistics settle and the sharpness drop before the LR reaches its peak.

**Follow-up:** *Why have warmup-stable-decay (WSD) schedules become popular?* A constant LR phase lets you branch cooldowns from any checkpoint, so one run gives you models at several token budgets (useful for scaling-law fits and continued training). Most of the loss improvement from annealing arrives in the short decay phase.

</details>

<details>
<summary><b>Q14</b> · <code>L1</code> · Pre-LN vs post-LN transformers?</summary>

**Answer.** Pre-LN (x + f(LN(x))) keeps an identity path through the residual stream, so gradients reach early layers without attenuation and deep models train stably with modest warmup. Post-LN (LN(x + f(x))) normalizes the stream itself; it can reach slightly better quality when it trains, but is fragile at depth. Pre-LN needs a final norm before the LM head because the stream's scale grows with depth.

**Follow-up:** *What else is used to stabilize deep pre-LN stacks?* Scaling residual-branch initialization by 1/√(2L), QK-norm, and in some models a second norm after each sublayer ("sandwich" norm).

</details>

<details>
<summary><b>Q15</b> · <code>L1</code> · Why do Xavier and He initialization scale by 1/fan_in?</summary>

**Answer.** For y = Σ_i w_i x_i with independent zero-mean terms, Var(y) = fan_in · Var(w) · Var(x). Keeping Var(y) = Var(x) requires Var(w) = 1/fan_in. ReLU zeroes half of its inputs, halving the second moment, so He init uses 2/fan_in. (Xavier averages fan_in and fan_out to balance the forward and backward passes.)

**Follow-up:** *Why do GPT-style models scale the output projections of residual branches by 1/√(2L)?* Each of the 2L branches adds variance to the residual stream; scaling keeps the final stream variance roughly independent of depth.

</details>

<details>
<summary><b>Q16</b> · <code>L1</code> · At initialization, what should the loss of a language model be? It stays there. What is wrong?</summary>

**Answer.** About ln V for a uniform prediction: ln 50,257 ≈ 10.82 for GPT-2's vocabulary, ln 128,256 ≈ 11.76 for Llama 3's. Stuck at ln V means the model outputs near-uniform logits: labels not shifted by one (or shifted twice), learning rate zero or scheduler bug, optimizer not given the parameters, gradients detached, or loss mask all zeros.

**Follow-up:** *Loss at init is 40, not 11. Why?* Initial logits are too large (output layer not scaled down, or an embedding with std 1 tied to the head), so the model is confidently wrong. Fix the init before tuning anything else.

</details>

<details>
<summary><b>Q17</b> · <code>L2</code> · Loss is NaN at step 3,000 after being fine. Walk through the diagnosis.</summary>

**Answer.** Look at the history, not the step. Plot the grad norm before clipping, max attention logit and max output logit per layer: steady growth points to logit blowup; a single jump points to the batch. Dump the offending batch (empty sequences, a loss mask producing 0/0, extreme token repetition). Check the precision: fp16 overflows at 65,504; bf16 does not overflow but loses precision. Usual fixes: rewind to the last good checkpoint and skip the batch, lower the LR, add QK-norm or z-loss, fix the masking division.

**Follow-up:** *How do you tell an optimizer instability from a data problem?* Replay the same checkpoint on the same batch (does it reproduce deterministically?) and on a different batch. A data problem follows the batch; an instability recurs at a similar point with different data.

</details>

<details>
<summary><b>Q18</b> · <code>L2</code> · What does μP (maximal update parametrization) buy you?</summary>

**Answer.** In standard parametrization the optimal learning rate shifts as width grows, so you would have to re-tune at full scale. μP rescales initialization and per-layer learning rates (for Adam, hidden-layer LR ∝ 1/width and a 1/width multiplier on output logits) so the optimal hyperparameters stay roughly constant across width. You tune on a small proxy and transfer.

**Follow-up:** *What does μP not transfer?* Depth (it needs depth-specific extensions), training duration and batch size. Teams still confirm transferred hyperparameters at two or three scales before a large run.

</details>

<details>
<summary><b>Q19</b> · <code>L2</code> · Why is a bigger batch not free throughput?</summary>

**Answer.** Past the critical batch size (roughly the gradient noise scale, B ≈ tr(Σ)/|G|², from McCandlish et al., 2018), extra samples per step add little new information: steps to a given loss stop falling, so total compute rises. Below it, doubling the batch nearly halves the steps. Large batches also need LR retuning and more memory.

**Follow-up:** *Does the critical batch size change during training?* It grows as the loss falls (gradients get noisier relative to their mean), which is why large runs ramp the batch size up.

</details>

<details>
<summary><b>Q20</b> · <code>L1</code> · bf16 vs fp16 for training: what is the difference and what still needs fp32?</summary>

**Answer.** bf16 has 8 exponent bits (fp32's range) and 7 mantissa bits; fp16 has 5 and 10. fp16 needs loss scaling because small gradients underflow; bf16 does not. bf16's machine epsilon is 2⁻⁷ ≈ 0.008, so keep in fp32: matmul accumulation, softmax and norm statistics, the loss, and the master weights and optimizer states.

**Follow-up:** *Why fp32 master weights?* An update of relative size 10⁻⁴ is below bf16's rounding step, so adding it to a bf16 weight does nothing and training stalls. Alternatives are stochastic rounding or Kahan-compensated updates.

</details>

<details>
<summary><b>Q21</b> · <code>L2</code> · You use gradient accumulation over 8 micro-batches with variable-length sequences. What is the classic bug?</summary>

**Answer.** Computing a mean loss per micro-batch and then averaging the 8 means. Micro-batches with few tokens get the same weight as long ones, so the gradient is not the gradient of the per-token mean over the full batch. Fix: sum token losses, count non-masked tokens across all micro-batches (and all data-parallel ranks), and divide once.

**Follow-up:** *Where else does this bug appear?* Data-parallel training where each rank averages over its own tokens before the all-reduce. Lab 02 tests the correct version.

</details>

<details>
<summary><b>Q22</b> · <code>L3</code> · What does the Muon optimizer do, and where is it applied?</summary>

**Answer.** For 2D hidden weight matrices, Muon takes the momentum-averaged gradient and replaces it by an approximation of its orthogonal polar factor (UVᵀ from its SVD) using a few Newton–Schulz iterations, then scales and applies it. All singular values of the update become ≈ 1, so no single direction dominates. Embeddings, the LM head, norms and biases stay on AdamW.

**Follow-up:** *What makes it harder to use at scale than AdamW?* It needs the whole matrix to orthogonalize, which conflicts with sharded (ZeRO/FSDP) optimizer states and adds matmuls per step; update-RMS matching and weight decay also need care when transferring AdamW hyperparameters.

</details>

<details>
<summary><b>Q23</b> · <code>L1</code> · Why do most LLM pretraining runs use no dropout?</summary>

**Answer.** They train for about one epoch on huge data, so they are not in the overfitting regime that dropout addresses. Dropout would slow learning per token and cost throughput.

**Follow-up:** *When would you turn it on?* Fine-tuning on small datasets, multi-epoch training on scarce data, or small models trained many epochs on a fixed corpus.

</details>

---

## 3. Transformers and architecture

<details>
<summary><b>Q24</b> · <code>L1</code> · Why divide attention scores by √d_k?</summary>

**Answer.** If q and k have independent unit-variance components, q·k has variance d_k. Without scaling, logits grow with head dimension, the softmax saturates to near one-hot and its gradients vanish.

**Follow-up:** *What does QK-norm add?* Normalizing q and k (RMSNorm per head) bounds the logits regardless of how weights grow during training, removing a common source of loss spikes. A learnable scale or temperature keeps the model able to sharpen attention.

</details>

<details>
<summary><b>Q25</b> · <code>L2</code> · Count the parameters of Llama-3-8B from its config (d = 4096, 32 layers, 32 heads, 8 KV heads, FFN 14,336, vocab 128,256, untied).</summary>

**Answer.** Embedding 128,256 × 4,096 ≈ 525.3M, doubled for the untied LM head ≈ 1.051B. Per layer: attention Wq and Wo 2 × 4,096² = 33.6M, Wk and Wv 2 × 4,096 × 1,024 = 8.4M, total 41.9M; SwiGLU MLP 3 × 4,096 × 14,336 = 176.2M; norms 8,192. Per layer ≈ 218.1M, × 32 ≈ 6.98B. Total ≈ 8.03B.

**Follow-up:** *Forward FLOPs per token at 8k context?* ≈ 2N plus attention: 2N ≈ 16.1 GFLOP (the embedding lookup is free, but the head is a matmul, so use ≈ 7.5B matmul weights → ≈ 15 GFLOP), and causal attention adds ≈ 2 · n_layers · T · d = 2 · 32 · 8,192 · 4,096 ≈ 2.1 GFLOP averaged over positions. Attention is ≈ 14% of the total at 8k context and grows linearly with T.

</details>

<details>
<summary><b>Q26</b> · <code>L1</code> · How does RoPE encode relative position?</summary>

**Answer.** Treat pairs of dimensions as complex numbers and rotate q at position m by angle mθ_i and k at position n by nθ_i, with θ_i = base^(−2i/d). Then Re[(q e^{imθ})·conj(k e^{inθ})] = Re[q·conj(k)·e^{i(m−n)θ}], which depends only on m − n. It has no parameters, preserves norms, and low-frequency pairs carry long-range position.

**Follow-up:** *How do you extend a RoPE model's context?* Position interpolation (scale positions down), raising the base (NTK-aware scaling) or YaRN (frequency-dependent interpolation plus an attention temperature), followed by a short fine-tune on long documents.

</details>

<details>
<summary><b>Q27</b> · <code>L2</code> · MQA vs GQA vs MLA?</summary>

**Answer.** MQA: all query heads share one K/V head (smallest cache, some quality loss). GQA: query heads are split into H_kv groups sharing K/V (the common default; Llama 3 8B uses 8 KV heads for 32 query heads, a 4× cache reduction). MLA (DeepSeek-V2/V3): cache one low-rank latent vector per token and up-project to K and V; the up-projections can be absorbed into the query and output projections at inference, giving a cache smaller than typical GQA at similar or better quality.

**Follow-up:** *Why does MLA need a separate RoPE component?* RoPE applies a position-dependent rotation between the up-projection and the dot product, which would block absorbing the up-projection into the query side. MLA adds a small decoupled key dimension, shared across heads, that carries RoPE.

</details>

<details>
<summary><b>Q28</b> · <code>L1</code> · Why is the SwiGLU hidden size about 8/3·d?</summary>

**Answer.** SwiGLU uses three matrices (gate, up, down) instead of two. A hidden size of 8/3·d keeps 3 · d · (8/3)d = 8d² parameters, the same as a 4d two-matrix MLP. At equal parameters and FLOPs, gated MLPs consistently train better.

**Follow-up:** *Llama-3-8B uses 14,336 = 3.5·d. Why not 8/3?* Sizes are rounded to hardware-friendly multiples and often enlarged deliberately; parameter budgets are a design choice, not a rule.

</details>

<details>
<summary><b>Q29</b> · <code>L2</code> · Mixture of experts: why are total parameters much larger than active parameters, and what is hard about it?</summary>

**Answer.** Each token is routed to the top-k of E expert MLPs, so FLOPs scale with active parameters while capacity scales with total parameters. Hard parts: load balancing (auxiliary loss or bias-based balancing), all-to-all communication for expert parallelism, memory to hold every expert, routing instability and token dropping when experts overflow capacity, and serving efficiency at small batch (each expert sees few tokens).

**Follow-up:** *How does auxiliary-loss-free balancing work (DeepSeek-V3)?* Add a per-expert bias to the routing scores used only for top-k selection, nudging it down for overloaded experts and up for underloaded ones after each step. The gating weights themselves are unbiased, so balancing does not distort the training objective the way an auxiliary loss can.

</details>

<details>
<summary><b>Q30</b> · <code>L1</code> · Batched generation with a decoder-only model gives garbage for short prompts. What is wrong?</summary>

**Answer.** Padding. Decoder-only models generate from the last position, so pad on the left; the attention mask must exclude pad tokens, and position ids must start at the first real token (or RoPE positions shift). Right-padding makes the model continue from pad tokens.

**Follow-up:** *In pretraining, you pack several documents into one sequence. What is the issue?* Tokens can attend across document boundaries. Either accept it (common, mild effect) or use document masking with variable-length attention kernels; also reset position ids per document if you mask.

</details>

<details>
<summary><b>Q31</b> · <code>L1</code> · Weight tying: what is it, and why do large models often skip it?</summary>

**Answer.** Use the input embedding matrix as the output projection. It saves V·d parameters: for GPT-2 small, 50,257 × 768 ≈ 38.6M of 124M, a large fraction. For an 8B model it is a much smaller fraction, and separate matrices give the model more freedom, so large models are often untied (Llama-3-8B is untied; the small Llama 3.2 1B and 3B are tied).

**Follow-up:** *Any initialization interaction?* With tied weights, the embedding scale also sets the initial logit scale; a unit-variance embedding produces huge initial logits unless you scale the output.

</details>

<details>
<summary><b>Q32</b> · <code>L2</code> · What are z-loss and logit soft-capping for?</summary>

**Answer.** z-loss (PaLM) adds a small term, e.g. 10⁻⁴·(log Z)², where Z is the softmax normalizer over the vocabulary; it keeps the log-partition near zero and the output logits bounded. Soft-capping (Gemma 2) replaces logits x with c·tanh(x/c), bounding attention or output logits smoothly. Both address instabilities from growing logits.

**Follow-up:** *Why does soft-capping complicate kernels?* The tanh is inside the softmax, so standard FlashAttention kernels need a variant that applies it to scores; early kernel support lagged for exactly this reason.

</details>

<details>
<summary><b>Q33</b> · <code>L1</code> · What is the time and memory complexity of attention, and what did FlashAttention change?</summary>

**Answer.** Compute is O(T²·d) per layer. A naive implementation also materializes the T×T score matrix, so memory is O(T²). FlashAttention computes exactly the same result in tiles with online softmax, so memory is O(T) and HBM traffic drops sharply; the FLOPs are unchanged.

**Follow-up:** *What changes the O(T²) compute?* Sliding-window attention (O(T·w)), sparse patterns, linear attention and state-space models; each trades some exact long-range recall for cost.

</details>

<details>
<summary><b>Q34</b> · <code>L2</code> · What is the residual-stream view of a transformer, and why does it matter?</summary>

**Answer.** Each attention and MLP block reads from and adds to a shared d-dimensional stream. The final logits are a sum of contributions from every block plus the embedding. This makes the stream a communication channel between layers, gives gradients a direct path, and lets interpretability decompose outputs into per-component contributions (direct logit attribution, logit lens).

**Follow-up:** *What does it imply for layer pruning?* Blocks that write little to the stream can often be removed with small loss increase, and later layers of many trained models are more redundant than early ones.

</details>

<details>
<summary><b>Q35</b> · <code>L3</code> · State-space models (Mamba) vs attention: what do you gain and lose? Why hybrid models?</summary>

**Answer.** SSMs keep a fixed-size recurrent state: linear time in sequence length, constant memory per generated token, no growing KV cache. The cost is that everything must be compressed into that state, which hurts exact retrieval and copying from far back in context (associative recall, long-range lookup) where attention excels. Hybrids interleave a few attention layers with many SSM or linear-attention layers to recover recall while keeping most of the efficiency.

**Follow-up:** *How would you test the recall weakness cheaply?* Synthetic associative recall or multi-query key-value lookup at increasing context lengths, comparing accuracy at matched parameters and FLOPs.

</details>

---

## 4. Tokenization

<details>
<summary><b>Q36</b> · <code>L1</code> · Describe BPE training and encoding.</summary>

**Answer.** Training: start from 256 byte tokens; pre-tokenize text into chunks; repeatedly count adjacent pairs within chunks (weighted by chunk frequency), merge the most frequent pair into a new token, and record the merge; stop at the target vocab size. Encoding: split into chunks the same way, then within each chunk repeatedly apply the lowest-rank (earliest-learned) merge present until none apply.

**Follow-up:** *Why is naive training slow, and how do you speed it up?* Recounting all pairs after each merge is O(corpus) per merge. Count unique pre-tokenized chunks once, and update only the pair counts affected by each merge (with an index from pair to chunks).

</details>

<details>
<summary><b>Q37</b> · <code>L1</code> · Why byte-level BPE?</summary>

**Answer.** Every string is a sequence of UTF-8 bytes, so there is no unknown token and any input round-trips exactly. The base vocabulary is only 256.

**Follow-up:** *What breaks in streaming detokenization?* A single character can span several tokens (Telugu and most Indic characters are 3 bytes in UTF-8), so a token boundary can fall inside a character. Buffer bytes until they form valid UTF-8 before emitting text.

</details>

<details>
<summary><b>Q38</b> · <code>L2</code> · What does the pre-tokenization regex do, and how does it affect arithmetic?</summary>

**Answer.** It splits text into chunks (words with a leading space, numbers, punctuation runs, whitespace) and merges never cross chunk boundaries, so tokens do not mix words and punctuation arbitrarily. GPT-4's cl100k pattern splits digits into groups of up to 3; Llama 1 and 2 split numbers into individual digits.

**Follow-up:** *Why does digit grouping matter?* With left-to-right 3-digit chunks, the same number aligns differently depending on its length, so place value is inconsistent across examples; single-digit or right-aligned grouping gives the model a consistent representation and tends to help arithmetic.

</details>

<details>
<summary><b>Q39</b> · <code>L2</code> · What is the trade-off in choosing vocabulary size?</summary>

**Answer.** Larger V: shorter sequences (less compute and more text per context window), better coverage of languages and code. Costs: embedding and head parameters V·d (the head matmul is 2·V·d FLOPs per token), more rare tokens that are undertrained, and a bigger softmax. Llama 2 used 32k; Llama 3 moved to 128k, largely for multilingual and code efficiency.

**Follow-up:** *How would you choose V for a bilingual English–Telugu model of 100M parameters?* Measure fertility (tokens per word) on held-out text in both languages for several V; at 100M parameters a 128k vocab at d = 768 would be ≈ 98M parameters in embeddings alone, so a smaller vocab with good Telugu coverage wins.

</details>

<details>
<summary><b>Q40</b> · <code>L2</code> · What is tokenizer fertility and why does it matter for Indic languages?</summary>

**Answer.** Fertility is the average number of tokens per word (or per character). A tokenizer whose merges were learned mostly on English text splits Telugu into many byte-level pieces. Higher fertility means higher API cost per word, less text per context window, more decode steps per answer, and usually worse quality.

**Follow-up:** *How do you measure it fairly across languages?* Use a parallel corpus (the same content in each language, e.g. FLORES) and compare tokens per sentence, not per word, since word boundaries differ across scripts.

</details>

<details>
<summary><b>Q41</b> · <code>L2</code> · What are "glitch tokens"?</summary>

**Answer.** Tokens that exist in the vocabulary (because they were frequent in the tokenizer's training data) but almost never appear in the model's training data. Their embeddings stay near initialization, so prompts containing them produce erratic behavior.

**Follow-up:** *How would you find them?* Look for tokens with unusually small embedding norms or near-identical embeddings, and check each candidate by prompting the model to repeat it.

</details>

<details>
<summary><b>Q42</b> · <code>L2</code> · Two models use different tokenizers. How do you compare their language-modeling loss?</summary>

**Answer.** Per-token loss is not comparable because a token covers different amounts of text. Normalize to bits per byte: bpb = (total loss in nats over the text) / (number of UTF-8 bytes × ln 2). Equivalently, loss per token × tokens per byte / ln 2.

**Follow-up:** *What else must be matched?* The exact same evaluation text, document boundaries and context handling (sliding window or not), since loss depends on how much context each token sees.

</details>

<details>
<summary><b>Q43</b> · <code>L1</code> · Why do chat templates and special tokens cause bugs?</summary>

**Answer.** The model learns the exact template it was fine-tuned on. A different system prompt format, a missing BOS, or an extra newline at inference measurably degrades output. Separately, if user text can be tokenized into control tokens (for example a literal end-of-turn string), a user can inject fake turns.

**Follow-up:** *How do you prevent control-token injection?* Tokenize user content with special tokens disabled (treated as plain text), and only insert control tokens from trusted template code.

</details>

---

## 5. Pretraining, data and scaling

<details>
<summary><b>Q44</b> · <code>L1</code> · Chinchilla in one sentence, and why do labs train far past it?</summary>

**Answer.** At fixed training compute, loss is minimized by scaling parameters N and tokens D roughly equally, about 20 tokens per parameter (Hoffmann et al., 2022). Labs over-train because inference cost scales with N: a smaller model trained on more tokens is cheaper to serve for the same quality. Llama 3 8B saw about 15T tokens, roughly 1,900 tokens per parameter.

**Follow-up:** *What is the fitted form?* L(N, D) = E + A/N^α + B/D^β, an irreducible loss plus power-law terms in parameters and data. Lab 08 fits it and adds inference cost to the optimization.

</details>

<details>
<summary><b>Q45</b> · <code>L1</code> · How long does it take to train a 7B model on 2T tokens with 256 H100s at 40% MFU?</summary>

**Answer.** 6 × 7·10⁹ × 2·10¹² = 8.4·10²² FLOPs. Throughput 256 × 989·10¹² × 0.4 ≈ 1.01·10¹⁷ FLOP/s. Time ≈ 8.3·10⁵ s ≈ 9.6 days.

**Follow-up:** *You use full activation recomputation. Does MFU change?* MFU counts only 6ND model FLOPs, so it is unchanged in definition, but wall-clock time rises because the hardware does ≈ 8ND (an extra forward). Hardware FLOPs utilization (HFU) counts the recomputation.

</details>

<details>
<summary><b>Q46</b> · <code>L2</code> · Why MinHash deduplication, and what do the band and row parameters control?</summary>

**Answer.** Near-duplicates waste compute, increase memorization and leak evaluation data. MinHash estimates the Jaccard similarity s of shingle sets; LSH groups b·r hashes into b bands of r rows, and two documents become candidates if any band matches: P = 1 − (1 − s^r)^b. This is an S-curve with threshold near (1/b)^(1/r). Example: b = 20, r = 5 gives threshold ≈ 0.55, P(s = 0.8) ≈ 0.9996, P(s = 0.5) ≈ 0.47.

**Follow-up:** *Exact vs fuzzy vs semantic dedup?* Exact hashing catches copies; MinHash catches near-copies with edits; embedding-based dedup catches paraphrases but is costlier and can remove legitimately distinct documents.

</details>

<details>
<summary><b>Q47</b> · <code>L2</code> · How are web documents filtered for quality, and how do you know a filter helps?</summary>

**Answer.** Stages: URL and language filtering, text extraction, heuristic rules (length, symbol and stopword ratios, repetition), dedup, model-based quality classifiers (e.g. a small classifier trained on LLM-annotated "educational value" labels, as in FineWeb-Edu), PII removal and toxicity filtering, and benchmark decontamination.

**Follow-up:** *How do you validate a filter?* Train small proxy models (hundreds of millions of parameters, billions of tokens) on filtered vs unfiltered data at matched token counts, and compare a fixed suite of downstream evals with error bars. Check that gains are not just contamination by rerunning with decontaminated evals.

</details>

<details>
<summary><b>Q48</b> · <code>L2</code> · How do you pick a data mixture, and how many epochs of repeated data are acceptable?</summary>

**Answer.** Domain weights are set by small-scale ablations or learned methods (DoReMi, RegMix), with code and math often upsampled, and high-quality data concentrated in a final annealing phase. For repetition, Muennighoff et al. (2023) found that up to about 4 epochs of repeated data is nearly as good as fresh data, with rapidly diminishing value after that.

**Follow-up:** *Why anneal on high-quality data at the end?* As the LR decays, the model consolidates what it sees last; putting the best data there improves benchmark performance more per token than mixing it uniformly. It also makes annealing a cheap way to evaluate a candidate dataset.

</details>

<details>
<summary><b>Q49</b> · <code>L2</code> · Why do loss spikes happen, and how are they handled at scale?</summary>

**Answer.** Causes: attention-logit or output-logit growth, bad batches, too-high LR, Adam's second-moment estimate lagging a sudden gradient change, and numeric overflow. Mitigations: QK-norm, z-loss, lower peak LR or longer warmup, lower Adam β2 or larger ε in some cases, careful init, and operationally: skip the batch and rewind to a checkpoint. Monitor grad norm and max logits continuously.

**Follow-up:** *A spike recovers by itself. Do you care?* Yes, if it recurs: each spike costs tokens and can leave lasting damage. Log it, check whether the same data or layer is involved, and fix the cause before scaling up.

</details>

<details>
<summary><b>Q50</b> · <code>L2</code> · How do you fit a scaling law you can trust?</summary>

**Answer.** Train a grid of small models across several compute budgets with the LR schedule matched to each run's length, fit IsoFLOP curves (loss vs N at fixed C), find the minimum per budget, and fit power laws for N_opt(C) and D_opt(C). Report the fit's uncertainty and check it by predicting a held-out larger run.

**Follow-up:** *Why did Kaplan et al. (2020) and Chinchilla disagree on the optimal ratio?* Differences in experimental setup, including learning-rate schedules not matched to each run's token budget in the earlier work and how parameters were counted (embeddings excluded or not) at small scale. Lesson: small-scale details bias extrapolations.

</details>

<details>
<summary><b>Q51</b> · <code>L2</code> · How is long context usually trained, and how should it be evaluated?</summary>

**Answer.** Most tokens are trained at a short context (e.g. 4–8k) because attention cost grows as T². A final stage extends the context: raise the RoPE base or apply YaRN, and train on long documents at progressively longer lengths, often with context parallelism.

**Follow-up:** *Why is needle-in-a-haystack insufficient?* It tests single-fact retrieval, which models saturate easily. Use tasks that require aggregation, multi-hop reasoning and tracking across the whole context (e.g. RULER-style suites), plus real long-document tasks.

</details>

<details>
<summary><b>Q52</b> · <code>L2</code> · How often should a large run checkpoint?</summary>

**Answer.** Failures are routine at scale: the Llama 3 paper reports 419 unexpected interruptions during a 54-day period on 16k H100s. With checkpoint cost C (time the job is blocked) and mean time between failures M, Young's approximation gives an optimal interval ≈ √(2·C·M). Reduce C with sharded, asynchronous checkpointing (copy to host memory, write in the background).

**Follow-up:** *What else must be checkpointed for exact resumption?* Optimizer state, LR scheduler step, data-loader position and RNG states on every rank. Without the data position you silently repeat or skip data.

</details>

<details>
<summary><b>Q53</b> · <code>L1</code> · Why is "tokens seen" the x-axis for pretraining curves rather than steps?</summary>

**Answer.** Steps change meaning when batch size changes (ramps, different configs); tokens are comparable across runs and map directly onto compute (≈ 6N per token). Compare runs at matched tokens or matched FLOPs.

**Follow-up:** *When would you plot against FLOPs instead?* When comparing models of different sizes, such as when choosing N for a fixed budget.

</details>

<details>
<summary><b>Q54</b> · <code>L3</code> · Synthetic data for pretraining: what can go wrong?</summary>

**Answer.** Loss of diversity (generator mode collapse into a narrow style), amplified errors and biases of the generator, contamination (the generator memorized benchmarks), and repeated recursive training on model outputs losing the tails of the distribution ("model collapse" when real data is replaced rather than accumulated).

**Follow-up:** *How do you use synthetic data safely?* Mix it with real data rather than replacing it, diversify seeds and prompts, filter with verifiers where possible (code execution, math checkers), dedup against evals, and measure on held-out real distributions.

</details>

---

## 6. Distributed training

<details>
<summary><b>Q55</b> · <code>L2</code> · Tensor, pipeline and fully sharded data parallelism: when do you use each?</summary>

**Answer.** Tensor parallelism (TP) splits matmuls within a layer and needs collectives every layer, so it stays within a node's NVLink domain. Pipeline parallelism (PP) splits layers into stages and sends activations point to point, which tolerates slower links but creates bubbles. FSDP/ZeRO-3 shards parameters, gradients and optimizer states and all-gathers each layer's weights just in time; it is the default until communication dominates. Large runs combine them: TP within a node, PP and FSDP across nodes, data parallel for the rest.

**Follow-up:** *Which one reduces activation memory?* TP (with sequence parallelism) and PP do; plain FSDP does not, since each rank still processes full layers for its micro-batch.

</details>

<details>
<summary><b>Q56</b> · <code>L1</code> · What is the cost of a ring all-reduce?</summary>

**Answer.** It is a reduce-scatter followed by an all-gather, each in n − 1 steps. Every device sends and receives 2(n − 1)/n of the buffer, nearly independent of n. Time ≈ 2(n − 1)/n · S/B + 2(n − 1)·α, where S is buffer size, B link bandwidth and α per-step latency.

**Follow-up:** *When is the ring a bad choice?* For small messages or very large n, the latency term 2(n − 1)α dominates; tree or hierarchical algorithms have O(log n) latency.

</details>

<details>
<summary><b>Q57</b> · <code>L2</code> · How much memory do ZeRO stages 1, 2 and 3 save for a 7B model on 64 GPUs?</summary>

**Answer.** Mixed-precision Adam uses 16 bytes per parameter: 2 (bf16 weights) + 2 (bf16 gradients) + 12 (fp32 master weights, m and v). For 7B that is 112 GB per GPU without sharding. ZeRO-1 shards the 12-byte optimizer states: 4 + 12/64 ≈ 4.19 bytes/param ≈ 29 GB. ZeRO-2 also shards gradients: 2 + 14/64 ≈ 2.22 bytes/param ≈ 16 GB. ZeRO-3 shards everything: 16/64 = 0.25 bytes/param = 1.75 GB.

**Follow-up:** *What is missing from this count?* Activations (often the largest term at long sequence lengths), temporary buffers for gathered weights and communication, and memory fragmentation.

</details>

<details>
<summary><b>Q58</b> · <code>L2</code> · How big are activations, and how does recomputation trade memory for compute?</summary>

**Answer.** Korthikanti et al. (2022) estimate activation memory per transformer layer as s·b·h·(34 + 5·a·s/h) bytes in 16-bit without parallelism (s sequence length, b micro-batch, h hidden size, a heads). The 5as/h term is the attention score matrix, which FlashAttention avoids storing. Full recomputation stores only each layer's input and reruns the forward in backward (≈ 33% more compute); selective recomputation reruns only cheap, memory-heavy parts.

**Follow-up:** *For a 7B-class layer (h = 4096) at s = 4096, b = 1, without the attention term, how much is that?* 4096 · 4096 · 34 bytes ≈ 570 MB per layer, ≈ 18 GB for 32 layers: more than the sharded model states in Q57.

</details>

<details>
<summary><b>Q59</b> · <code>L2</code> · What is the pipeline bubble and how do you shrink it?</summary>

**Answer.** With p stages and m micro-batches, the GPipe/1F1B schedule has idle time equal to (p − 1) micro-batch slots at the start and end. Bubble time relative to ideal compute is (p − 1)/m; as a fraction of total time it is (p − 1)/(m + p − 1). Increase m (more micro-batches per step), use interleaved schedules (each device holds v chunks, reducing the bubble by v at the cost of more communication), or zero-bubble schedules that split the backward pass.

**Follow-up:** *Why not just make m huge?* The global batch is fixed by optimization (critical batch size), so more micro-batches means smaller ones, which lowers per-kernel efficiency.

</details>

<details>
<summary><b>Q60</b> · <code>L2</code> · How does Megatron-style tensor parallelism split an MLP?</summary>

**Answer.** The first matrix is split by columns (each GPU computes a slice of the hidden units), so the nonlinearity is applied locally with no communication. The second matrix is split by rows, each GPU produces a partial sum, and one all-reduce combines them. Attention splits by heads the same way. That is one all-reduce per block in forward and one in backward.

**Follow-up:** *What does sequence parallelism add?* The regions between TP blocks (norms, dropout, residual adds) are sharded along the sequence, and the all-reduce becomes a reduce-scatter plus all-gather of the same total volume, cutting activation memory in those regions by the TP degree.

</details>

<details>
<summary><b>Q61</b> · <code>L1</code> · Estimate the gradient all-reduce time per step for a 7B model under plain data parallelism across nodes with 50 GB/s per GPU.</summary>

**Answer.** bf16 gradients are 14 GB. Ring all-reduce sends ≈ 2 × 14 = 28 GB per GPU. At 50 GB/s that is ≈ 0.56 s per step, which must be overlapped with the backward pass (bucketed all-reduce as layers finish) to avoid stalling.

**Follow-up:** *How does FSDP's communication compare?* All-gather weights in forward, again in backward, then reduce-scatter gradients: ≈ 3 × parameter bytes vs ≈ 2× for DDP, so about 1.5× the volume, in exchange for large memory savings.

</details>

<details>
<summary><b>Q62</b> · <code>L3</code> · What makes expert parallelism for MoE hard?</summary>

**Answer.** Every MoE layer needs two all-to-all exchanges (dispatch tokens to experts, combine results), whose volume scales with tokens × k × d and which cross nodes. Load imbalance means some GPUs wait for the busiest expert. Techniques: capacity factors, balancing losses or bias balancing, limiting how many nodes a token can be sent to, and overlapping all-to-all with computation of another micro-batch.

**Follow-up:** *Why did DeepSeek-V3 restrict each token to experts on at most 4 nodes?* To bound cross-node (InfiniBand) traffic, which is slower than intra-node NVLink, so communication can be hidden behind computation.

</details>

<details>
<summary><b>Q63</b> · <code>L3</code> · How does context (sequence) parallelism with ring attention work?</summary>

**Answer.** Split the sequence across GPUs. Each GPU keeps its query block and passes K/V blocks around a ring; at each step it computes attention between its queries and the current K/V block and merges results with online softmax (running max and sum). Communication of the next block overlaps with computation on the current one.

**Follow-up:** *Causal masking creates imbalance. How is it fixed?* With a contiguous split, the first GPU's queries attend to little and the last GPU's to everything. Assign each GPU two chunks from opposite ends (a "zig-zag" split) so work is balanced.

</details>

---

## 7. Post-training and RL

<details>
<summary><b>Q64</b> · <code>L1</code> · In SFT, which tokens contribute to the loss?</summary>

**Answer.** Usually only the assistant's response tokens; prompt, system and user tokens are masked out. With packing, the mask must also respect sample boundaries. The loss is averaged over unmasked tokens (see Q21 for the averaging trap).

**Follow-up:** *What happens if you train on prompt tokens too?* The model spends capacity learning to generate user turns, and with short answers the gradient is dominated by prompts; sometimes harmless, sometimes it degrades instruction following.

</details>

<details>
<summary><b>Q65</b> · <code>L1</code> · How is a reward model trained from preferences?</summary>

**Answer.** Bradley–Terry: P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l)). Train a scalar head on a language model with loss −log σ(r_w − r_l). Only differences matter, so the reward is defined up to a per-prompt constant.

**Follow-up:** *What is reward over-optimization?* As the policy is optimized harder (KL from the reference grows), the proxy reward keeps rising but true quality (measured by a stronger "gold" reward model or humans) rises then falls (Gao et al., 2022).

</details>

<details>
<summary><b>Q66</b> · <code>L2</code> · Derive DPO.</summary>

**Answer.** The KL-regularized objective max_π E[r(x, y)] − β·KL(π‖π_ref) has the closed-form optimum π*(y|x) = π_ref(y|x)·exp(r(x, y)/β)/Z(x). Solve for the reward: r = β·log(π*/π_ref) + β·log Z(x). Substitute into Bradley–Terry; Z(x) cancels in r_w − r_l. Replacing π* with the trainable π gives the loss −log σ(β·[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)]).

**Follow-up:** *What does β control, and what is the "implicit reward"?* β sets how far π may move from π_ref (smaller β, larger moves). β·log(π/π_ref) acts as a reward you can read off the trained policy and evaluate like a reward model.

</details>

<details>
<summary><b>Q67</b> · <code>L2</code> · Why can DPO lower the likelihood of the chosen response?</summary>

**Answer.** The loss depends only on the margin between chosen and rejected log-ratios. The margin can grow while both log-probabilities fall, as long as the rejected one falls faster. The freed probability mass goes to other outputs, sometimes undesirable ones ("likelihood displacement").

**Follow-up:** *Mitigations?* Add an SFT (NLL) term on chosen responses, use a larger β, keep preference pairs close to the policy's own outputs (on-policy or iterative DPO), and monitor chosen log-probabilities during training.

</details>

<details>
<summary><b>Q68</b> · <code>L2</code> · DPO vs IPO vs SimPO?</summary>

**Answer.** IPO replaces the log-sigmoid with a squared loss pulling the log-ratio margin to a fixed target (1/(2τ)), which prevents the margin from growing without bound on deterministic preferences. SimPO drops the reference model and uses the length-normalized average log-probability as the implicit reward, with a target margin γ: −log σ((β/|y_w|)·log π(y_w) − (β/|y_l|)·log π(y_l) − γ).

**Follow-up:** *What risk does SimPO's reference-free design add?* Nothing anchors the policy to its starting point, so it can drift further and forget capabilities; hyperparameters (β, γ, LR) matter more.

</details>

<details>
<summary><b>Q69</b> · <code>L2</code> · Write the PPO clipped objective and explain the clip.</summary>

**Answer.** With ratio ρ = π_θ(a|s)/π_old(a|s) and advantage A: L = E[min(ρA, clip(ρ, 1 − ε, 1 + ε)A)]. For A > 0 the objective stops rewarding increases of ρ beyond 1 + ε; for A < 0, decreases below 1 − ε. It is a cheap trust region that lets you take several gradient steps on one batch of rollouts. In RLHF, the per-token reward is usually −β·KL plus the reward model's score at the final token, and advantages come from a learned value function with GAE.

**Follow-up:** *What does the clip not protect against?* It only removes the incentive to move further; the ratio can still leave the range through other tokens' gradients or shared parameters, which is why implementations also monitor the approximate KL and clip fraction.

</details>

<details>
<summary><b>Q70</b> · <code>L2</code> · GRPO vs PPO?</summary>

**Answer.** GRPO drops the value network. For each prompt it samples G responses and uses the normalized reward A_i = (r_i − mean(r))/std(r) as the advantage for every token of response i. It keeps the clipped ratio and adds a KL penalty to the reference policy (k3 estimator) in the loss. It is cheaper (no critic) and fits verifiable, sequence-level rewards naturally.

**Follow-up:** *What biases did "Dr. GRPO" point out?* Dividing by the group's standard deviation up-weights prompts that are nearly always solved or nearly always failed, and normalizing the loss by each response's length changes the per-token weight with length (short correct answers get stronger updates, long wrong answers weaker penalties). Removing both normalizations removes these biases.

</details>

<details>
<summary><b>Q71</b> · <code>L3</code> · Name three GRPO failure modes and the fixes introduced in DAPO.</summary>

**Answer.** (1) Entropy collapse: the policy becomes deterministic early and stops exploring; fix with "clip-higher" (a larger upper clip bound so low-probability tokens can grow). (2) Zero-advantage groups: when all G samples are right or all wrong, the group contributes no gradient; fix with dynamic sampling (filter such prompts and resample to keep the batch full). (3) Length effects from per-sequence averaging; fix with a token-level loss average across the batch, plus overlong reward shaping for truncated responses. Reward hacking is the fourth, handled by better verifiers and audits.

**Follow-up:** *DAPO removes the KL penalty entirely. Why is that reasonable there?* With a verifiable rule-based reward there is no learned reward model to exploit, and reasoning training is supposed to move the policy far from the base model; the KL mostly slows that down.

</details>

<details>
<summary><b>Q72</b> · <code>L1</code> · Why is there a KL penalty in RLHF?</summary>

**Answer.** It keeps the policy near the reference model, where the learned reward model is accurate, which limits reward hacking and preserves general capabilities and fluency. KL(π‖π_ref) is reverse KL with respect to the policy, so it is mode-seeking: it tolerates the policy dropping modes of the reference.

**Follow-up:** *Penalty in the reward vs a term in the loss?* InstructGPT-style PPO puts per-token −β·log(π/π_ref) into the reward; GRPO adds a KL estimate directly to the loss. They differ in how the penalty interacts with advantage estimation and clipping.

</details>

<details>
<summary><b>Q73</b> · <code>L1</code> · LoRA: why is B initialized to zero, what is α/r, and why does it work?</summary>

**Answer.** W' = W + (α/r)·BA with A ∈ ℝ^{r×k} random and B ∈ ℝ^{d×r} zero, so training starts exactly at the base model and the update is nonzero after the first step. α/r keeps the update scale roughly stable when you change r. It works because fine-tuning updates have low effective rank, so rank 8–64 on attention and MLP projections captures most of the benefit with ≈ 1% of the trainable parameters.

**Follow-up:** *What does QLoRA change?* The frozen base is stored in 4-bit NF4 with double-quantized scales and dequantized on the fly; adapters and their gradients stay in bf16. A 7B base then needs ≈ 4 GB for weights instead of 14 GB.

</details>

<details>
<summary><b>Q74</b> · <code>L2</code> · Process vs outcome reward models?</summary>

**Answer.** Outcome reward models (ORMs) score the final answer; process reward models (PRMs) score each step. PRMs give denser credit assignment and better guidance for search, but step labels are expensive and PRMs are easier to exploit. Much of the recent progress in reasoning models used outcome rewards from verifiers (unit tests, exact-match math) with RL.

**Follow-up:** *How would you get step labels cheaply?* Monte Carlo rollouts: from each prefix, sample completions and use the fraction that reach a correct answer as that step's value estimate.

</details>

<details>
<summary><b>Q75</b> · <code>L2</code> · Give concrete reward hacks in RL on code or math, and defenses.</summary>

**Answer.** Code: special-casing visible tests, editing or deleting the tests, exiting the process with a success code before tests run, catching all exceptions. Math: exploiting a lenient answer parser (printing several answers), formatting tricks. Defenses: hidden held-out tests, read-only test files and a locked-down sandbox, strict answer extraction, penalties for known exploits, and regular human audits of the highest-reward samples.

**Follow-up:** *How do you detect hacking you have not anticipated?* Track reward alongside an independent held-out metric; divergence (reward up, held-out flat or down) is the signal. Also read samples whose reward jumped fastest.

</details>

<details>
<summary><b>Q76</b> · <code>L3</code> · In an asynchronous RL system, the rollout engine and trainer disagree on log-probabilities even for the same weights. Why does it matter and what do you do?</summary>

**Answer.** Inference engines use different kernels, precision and batching than the trainer, so π_engine ≠ π_trainer numerically. If you compute the ratio with the engine's log-probabilities as π_old, "on-policy" data is actually slightly off-policy, and with asynchronous rollouts it is also stale by some steps. Fixes: recompute π_old with the trainer on the rollout data, apply a (truncated) importance-sampling correction for the engine–trainer mismatch, bound staleness, and monitor the mismatch KL.

**Follow-up:** *What staleness is acceptable?* It is empirical: measure learning curves at 0, 1, 2… steps of lag. Clipping bounds the damage per update, but large lag slows learning or destabilizes it.

</details>

<details>
<summary><b>Q77</b> · <code>L2</code> · Distillation: logit, sequence-level and on-policy. When do you use each?</summary>

**Answer.** Logit distillation minimizes KL(teacher‖student) per token on a fixed dataset; it needs the teacher's full distributions (same tokenizer). Sequence-level distillation is SFT on teacher-generated outputs; it works through an API. On-policy distillation samples from the student and has the teacher score each token (reverse KL on the student's own outputs), which fixes the train–test mismatch of learning only from teacher trajectories.

**Follow-up:** *Why does on-policy distillation help?* The student learns to recover from its own mistakes, the states it will actually visit at inference, much as DAgger does in imitation learning.

</details>

---

## 8. Inference and serving

<details>
<summary><b>Q78</b> · <code>L1</code> · Why is decode memory-bound?</summary>

**Answer.** Each decode step multiplies every weight matrix by one vector per sequence. At batch 1 in bf16 that is 2 FLOPs per 2 bytes of weights, about 1 FLOP/byte, far below an H100's ridge point of ≈ 295. So time per token ≈ weight bytes / memory bandwidth.

**Follow-up:** *At what batch size does the weight matmul become compute-bound?* Intensity grows roughly linearly with batch, so around batch ≈ 300 in bf16 on an H100. But attention over the KV cache stays memory-bound at any batch, because each sequence reads its own cache.

</details>

<details>
<summary><b>Q79</b> · <code>L1</code> · Upper bound on batch-1 decode speed for an 8B model in bf16 on one H100?</summary>

**Answer.** 16 GB of weights / 3.35 TB/s ≈ 4.8 ms per token, so at most ≈ 210 tokens/s, ignoring the KV cache and overheads. Real systems reach perhaps 70–85% of peak bandwidth.

**Follow-up:** *The same model in INT4 weights?* ≈ 4.5 GB including scales, so ≈ 1.3 ms, ≈ 700+ tokens/s bound, if the dequantizing kernel keeps up.

</details>

<details>
<summary><b>Q80</b> · <code>L1</code> · KV cache size for Llama-2-70B at 32k context, batch 1, bf16?</summary>

**Answer.** 2 (K and V) × 80 layers × 8 KV heads × 128 head dim × 2 bytes = 320 KiB per token. × 32,768 tokens = 10 GiB. With multi-head attention (64 KV heads) it would be 80 GiB; GQA is why it is not.

**Follow-up:** *On 8 × H100 80 GB with bf16 weights, how many concurrent 4k-token sequences fit?* 640 GB − 140 GB weights − ≈ 60 GB activations and overhead ≈ 440 GB ≈ 410 GiB for KV. One 4k sequence needs 1.25 GiB, so ≈ 330 sequences.

</details>

<details>
<summary><b>Q81</b> · <code>L2</code> · Is speculative decoding lossless? What speedup should you expect?</summary>

**Answer.** Yes, with the rejection-sampling rule: accept a draft token with probability min(1, p/q) (p target, q draft); on rejection, sample from normalize(max(0, p − q)). The output distribution equals the target's exactly. The "accept if argmax matches" variant is lossless only for greedy decoding. With per-token acceptance rate α and k draft tokens, the expected tokens per target forward pass is (1 − α^(k+1))/(1 − α).

**Follow-up:** *When does it stop helping?* At large batch sizes, where the target model is already compute-bound, verifying k extra tokens per sequence costs real FLOPs; and when α is low (draft poorly matched to the domain). Speedup depends on α, k and the draft's cost relative to the target.

</details>

<details>
<summary><b>Q82</b> · <code>L1</code> · What does continuous batching fix?</summary>

**Answer.** Static batching waits for the whole batch to finish, so short sequences idle while the longest one generates. Continuous (iteration-level) batching, introduced in Orca, re-forms the batch every step: finished sequences leave and waiting requests join immediately. Throughput rises several-fold on realistic length distributions.

**Follow-up:** *What new problem does it create?* Prefills of new requests interrupt ongoing decodes and spike inter-token latency, which motivates chunked prefill (Q85).

</details>

<details>
<summary><b>Q83</b> · <code>L1</code> · What does PagedAttention fix?</summary>

**Answer.** Allocating a contiguous KV buffer per sequence for its maximum length wastes memory through fragmentation and over-reservation. PagedAttention stores KV in fixed-size blocks mapped by a per-sequence block table, like virtual memory pages, so waste is near zero, batches are larger, and sequences can share prefix blocks copy-on-write.

**Follow-up:** *What does the attention kernel have to do differently?* Gather K and V through the block table (indirect addressing) instead of reading one contiguous tensor.

</details>

<details>
<summary><b>Q84</b> · <code>L1</code> · TTFT vs inter-token latency: what drives each?</summary>

**Answer.** Time to first token (TTFT) = queueing + prefill of the prompt, which is compute-bound and grows with prompt length (and quadratically in the attention part). Inter-token latency (ITL, or TPOT) = one decode step, which is memory-bound and grows with batch size and total KV cache read per step.

**Follow-up:** *Throughput vs latency trade-off?* Larger batches raise tokens/s per GPU but increase ITL. Serving systems pick a batch size that meets the ITL SLO, and that sets cost per token.

</details>

<details>
<summary><b>Q85</b> · <code>L2</code> · Chunked prefill vs prefill–decode disaggregation?</summary>

**Answer.** Chunked prefill splits long prompts into chunks and mixes them with decode steps in the same batch, smoothing ITL on shared GPUs. Disaggregation runs prefill and decode on separate GPU pools and transfers the KV cache between them, so each pool can use its own parallelism and batch size; it pays off at large scale with strict TTFT and ITL targets.

**Follow-up:** *How much KV must move for a 70B model (Q80) with 1k-token prompts at 1,000 requests/s?* 320 KiB × 1,024 tokens = 320 MiB per request, ≈ 312 GiB/s across the cluster. That needs RDMA-class interconnect, and it is why KV transfer is overlapped with the prefill layer by layer.

</details>

<details>
<summary><b>Q86</b> · <code>L1</code> · What is prefix caching and why does it matter for agents and chat?</summary>

**Answer.** Reuse the KV cache of a shared prefix (system prompt, few-shot examples, earlier conversation turns) across requests, keyed by hashes of token blocks or a radix tree (SGLang's RadixAttention). It skips that prefill, cutting TTFT and compute. Multi-turn agents resend a growing context every step, so hit rates are high.

**Follow-up:** *What design choice keeps hit rates high?* Put stable content first and volatile content (timestamps, retrieved documents, user input) last, since any change invalidates everything after it.

</details>

<details>
<summary><b>Q87</b> · <code>L1</code> · Explain temperature, top-k, top-p and min-p.</summary>

**Answer.** Temperature divides logits by T before softmax (T → 0 is greedy; larger T flattens). Top-k keeps the k most likely tokens. Top-p (nucleus) keeps the smallest set whose cumulative probability reaches p. Min-p keeps tokens with probability ≥ min_p × p_max, so the cutoff adapts to the model's confidence. Renormalize after filtering, then sample.

**Follow-up:** *Does the order of temperature and filtering matter?* Yes: top-p after temperature selects a different set than before it. Engines differ, so fix the order when comparing results.

</details>

<details>
<summary><b>Q88</b> · <code>L2</code> · How does constrained decoding for JSON or grammars work, and does it hurt quality?</summary>

**Answer.** Compile the schema or grammar into a finite-state machine or pushdown automaton over tokens; at each step, mask logits of tokens that would leave the valid language, then sample. Precompute per-state token masks for speed. Token boundaries are the tricky part, since a token can span several grammar symbols.

**Follow-up:** *When can it hurt?* Masking redistributes probability onto tokens the model found unlikely, which can distort content, and forcing an answer field before any reasoning removes the model's chance to think. Put a free-text reasoning field before the answer fields when allowed.

</details>

---

## 9. Quantization and number formats

<details>
<summary><b>Q89</b> · <code>L1</code> · Why does weight-only INT4 speed up decode but not prefill?</summary>

**Answer.** Decode is memory-bound, so reading 4× fewer weight bytes makes it up to ≈ 4× faster (minus dequantization overhead). Prefill is compute-bound, and dequantizing to bf16 before the matmul adds work without reducing FLOPs, unless the hardware has low-precision matmul units for the format (e.g. FP8 or FP4 tensor cores with quantized activations too).

**Follow-up:** *What changes when both weights and activations are quantized to FP8?* Then the matmul itself runs on FP8 tensor cores at ≈ 2× bf16 throughput on Hopper, so prefill speeds up too.

</details>

<details>
<summary><b>Q90</b> · <code>L1</code> · Symmetric vs asymmetric, and per-tensor vs per-channel vs per-group quantization?</summary>

**Answer.** Symmetric INT8: scale = max|w|/127, q = round(w/scale). Asymmetric adds a zero point to use the full range for skewed distributions. Granularity: one scale per tensor (cheapest, most error), per output channel, or per group of g consecutive weights (g = 128 is common for INT4), which isolates outliers.

**Follow-up:** *What is the storage overhead of group-wise INT4 with g = 128 and fp16 scales?* 16/128 = 0.125 extra bits per weight, so ≈ 4.125 bits; ≈ 4.25 with an fp16 zero point as well.

</details>

<details>
<summary><b>Q91</b> · <code>L2</code> · Why are LLM activations hard to quantize, and what does SmoothQuant do?</summary>

**Answer.** Activations in LLMs have a few outlier channels with magnitudes far above the rest, so a per-tensor INT8 scale wastes resolution on everything else. SmoothQuant rescales per input channel: XW = (X·diag(s)⁻¹)(diag(s)·W), with s_j = max|X_j|^α / max|W_j|^(1−α) and α ≈ 0.5, moving difficulty from activations to weights, which quantize easily. The scale folds into the previous layer offline.

**Follow-up:** *Why not just use per-channel scales for activations?* Per-input-channel activation scales cannot be factored out of the matmul's reduction dimension, so INT8 GEMMs cannot use them efficiently; per-token scales along the other axis are fine.

</details>

<details>
<summary><b>Q92</b> · <code>L2</code> · How does GPTQ work?</summary>

**Answer.** It quantizes a layer's weights column by column and, after each column, updates the not-yet-quantized columns to compensate for the error, using the inverse of the Hessian H = 2XXᵀ of the layer's reconstruction loss computed from calibration activations X (an Optimal Brain Surgeon-style update). A Cholesky-based formulation makes it fast enough for 100B-parameter models.

**Follow-up:** *Why does the calibration set matter?* The Hessian encodes which input directions matter; calibrating only on English text can hurt other languages or code. Use data from the target distribution.

</details>

<details>
<summary><b>Q93</b> · <code>L2</code> · What is NF4?</summary>

**Answer.** A 4-bit data type from QLoRA whose 16 levels are placed at quantiles of a standard normal, so each level covers equal probability mass for normally distributed weights (after per-block absmax scaling, blocks of 64). QLoRA also double-quantizes the block scales to save memory.

**Follow-up:** *Why is NF4 good for storage but not for fast inference kernels?* Its levels are non-uniform, so dequantization needs a lookup, and there is no hardware matmul for it; it suits memory-bound fine-tuning more than high-throughput serving.

</details>

<details>
<summary><b>Q94</b> · <code>L2</code> · FP8 formats and scaling: E4M3 vs E5M2, per-tensor vs block scaling?</summary>

**Answer.** E4M3 has more precision and a max of 448; E5M2 has more range (max 57,344). A common recipe uses E4M3 for weights and activations and E5M2 for gradients. With only 3–4 bits of mantissa, scaling is essential: per-tensor scales from an amax history (delayed scaling) or fine-grained block scaling (DeepSeek-V3 used 1×128 tiles for activations and 128×128 blocks for weights, with higher-precision accumulation).

**Follow-up:** *What are MXFP4 and NVFP4?* Microscaling formats: FP4 (E2M1) values in small blocks with a shared scale: 32 elements with a power-of-two (E8M0) scale for MXFP4, 16 elements with an FP8 (E4M3) scale plus a per-tensor scale for NVFP4. Blackwell tensor cores run them natively.

</details>

<details>
<summary><b>Q95</b> · <code>L2</code> · How do you quantize the KV cache?</summary>

**Answer.** INT8 or FP8 KV halves cache memory relative to bf16, doubling feasible batch or context. Keys have outlier channels (much like activations), so quantize keys per channel and values per token (the KIVI observation), or keep a small window of recent tokens in full precision.

**Follow-up:** *What does KV quantization cost at long context?* Errors in keys shift attention weights across many positions, so check long-context retrieval tasks, not just short-prompt perplexity.

</details>

<details>
<summary><b>Q96</b> · <code>L1</code> · How do you measure quantization quality properly?</summary>

**Answer.** Perplexity alone is not enough. Measure downstream tasks (with CIs), long-context tasks, code and math, each language you serve (low-resource languages often degrade more), and the KL divergence of the quantized model's next-token distribution from the original on real prompts. Compare against the bf16 model on identical prompts and seeds.

**Follow-up:** *A quantized model matches perplexity but users complain. Where do you look?* Tail behavior: rare tokens, long generations drifting, structured output errors, and the specific languages or domains in the complaints. Token-level KL on those inputs shows where it diverges.

</details>

---

## 10. GPU kernels and performance

<details>
<summary><b>Q97</b> · <code>L1</code> · State the roofline model and use it on a square matmul.</summary>

**Answer.** Attainable FLOP/s = min(peak compute, arithmetic intensity × memory bandwidth), where intensity is FLOPs per byte moved. For an n×n by n×n bf16 matmul: 2n³ FLOPs over 3 × 2n² bytes, intensity n/3. For n = 1,024 that is ≈ 341 FLOP/byte, above the H100's ridge (≈ 295), so compute-bound; for n = 256 it is ≈ 85, memory-bound.

**Follow-up:** *What is the intensity of an elementwise op like GELU?* About 1 FLOP per 4 bytes (read and write bf16), so it is always memory-bound; fuse it into a neighbor.

</details>

<details>
<summary><b>Q98</b> · <code>L2</code> · Why does FlashAttention help if the FLOPs do not drop?</summary>

**Answer.** Attention in the standard implementation is limited by HBM traffic, not FLOPs: it writes the T×T score matrix, reads it for softmax, writes P and reads it again for PV. FlashAttention tiles Q, K and V into on-chip SRAM, computes the softmax online, and never writes the score matrix to HBM. The backward pass recomputes P from Q, K and the saved per-row log-sum-exp.

**Follow-up:** *What does FlashAttention save for backward instead of P?* The output O and the log-sum-exp per query row (one scalar per row per head), O(T) instead of O(T²).

</details>

<details>
<summary><b>Q99</b> · <code>L2</code> · Write the online softmax update used inside FlashAttention.</summary>

**Answer.** For each query row keep a running max m, a running denominator ℓ and an unnormalized output accumulator o. For a new block of scores s with values V: m' = max(m, max(s)); ℓ' = ℓ·e^(m − m') + Σ e^(s − m'); o' = o·e^(m − m') + e^(s − m')·V. At the end, output = o/ℓ and log-sum-exp = m + log ℓ.

**Follow-up:** *Why is FlashAttention-2 faster than the first version?* It reorganizes the loops so the outer loop runs over query blocks (parallel across thread blocks without synchronization), defers the division by ℓ to the end, and reduces non-matmul FLOPs, which run far slower than tensor-core matmuls.

</details>

<details>
<summary><b>Q100</b> · <code>L1</code> · Why fuse kernels? Estimate one unfused elementwise op.</summary>

**Answer.** Each unfused elementwise op reads its inputs from HBM and writes its output back. A 4096 × 4096 bf16 tensor is 32 MiB; read plus write is ≈ 67 MB, ≈ 20 µs at 3.35 TB/s, for almost no arithmetic. Fusing a chain (bias, activation, residual add, norm) into one kernel does one read and one write total.

**Follow-up:** *How do you get fusion without writing CUDA?* torch.compile (which generates Triton kernels), or writing the fused op directly in Triton.

</details>

<details>
<summary><b>Q101</b> · <code>L1</code> · MFU vs HFU, and what numbers are realistic?</summary>

**Answer.** Model FLOPs utilization = (6·N·tokens/s, plus attention FLOPs if you count them) / peak FLOP/s. Hardware FLOPs utilization also counts recomputation. Large dense runs have reported roughly 35–50% MFU (PaLM reported 46%; Llama 3 405B reported 38–43% in bf16).

**Follow-up:** *Your MFU is 15%. Where do you look first?* A profile: GPU idle gaps (data loading, host sync), exposed communication, small or misaligned matmuls, unfused memory-bound ops, and the micro-batch size.

</details>

<details>
<summary><b>Q102</b> · <code>L2</code> · Why pad the vocabulary to a multiple of 64 or 128?</summary>

**Answer.** Tensor-core matmuls run most efficiently when dimensions are multiples of the tile sizes; an awkward size like 50,257 forces boundary handling and can use slower paths. Padding to 50,304 (= 64 × 786, also a multiple of 128) measurably speeds the LM head, which is one of the largest matmuls in a small model. The padded logits are never targets, so they learn to be low.

**Follow-up:** *Which other dimensions deserve the same care?* Hidden sizes, head dimensions, FFN sizes and the micro-batch × sequence product that forms the matmul's M dimension.

</details>

<details>
<summary><b>Q103</b> · <code>L2</code> · Describe the Triton programming model and how it differs from CUDA.</summary>

**Answer.** In Triton you write a program for one tile (a block of the output), launched over a grid; you compute pointer offsets for the tile, load with masks for boundaries, operate on whole blocks (tl.dot, reductions) and store. The compiler handles thread mapping, shared memory, coalescing and tensor-core instructions. CUDA is written per thread and you manage those yourself, which gives more control and more ways to be slow.

**Follow-up:** *What does @triton.autotune do?* It benchmarks a list of configurations (block sizes, num_warps, num_stages) the first time a kernel sees a given key (e.g. the input shapes), and caches the fastest.

</details>

<details>
<summary><b>Q104</b> · <code>L1</code> · Describe the GPU memory hierarchy with rough H100 numbers.</summary>

**Answer.** Registers per thread; shared memory/L1 on each SM (up to 228 KB shared memory per SM on H100, 132 SMs on the SXM part); a 50 MB L2 cache shared by all SMs; 80 GB HBM3 at ≈ 3.35 TB/s. Each level down is much larger and slower, so fast kernels reuse data from the upper levels as much as possible.

**Follow-up:** *What is memory coalescing?* Threads in a warp (32 threads) accessing consecutive addresses get served by a few wide transactions; scattered accesses need many, cutting effective bandwidth.

</details>

<details>
<summary><b>Q105</b> · <code>L2</code> · A training step is slower than your roofline estimate. How do you profile it?</summary>

**Answer.** Use the PyTorch profiler or Nsight Systems and read the timeline. Look for GPU idle gaps (data loading, CPU-side Python, synchronizations like .item() or printing a tensor), many tiny kernels (launch overhead; fix with fusion or CUDA graphs), communication not overlapped with compute, and kernels far below their roofline (check achieved bandwidth or FLOP/s per kernel).

**Follow-up:** *How do you find hidden host-device syncs?* torch.cuda.set_sync_debug_mode("warn") reports operations that synchronize; they show up as gaps on the GPU timeline.

</details>

<details>
<summary><b>Q106</b> · <code>L3</code> · Why is batch-1, long-context decode attention hard to parallelize, and what does Flash-Decoding do?</summary>

**Answer.** With one query token per sequence and few heads, standard attention kernels launch too few thread blocks (roughly batch × heads) to occupy all SMs, even though each must read a long KV cache. Flash-Decoding splits the KV sequence into chunks processed in parallel, each producing a partial output and log-sum-exp, then combines them with a small reduction using the log-sum-exp rescaling (split-K).

**Follow-up:** *How does GQA change decode attention's arithmetic intensity?* All query heads in a group share one K/V head, so each loaded K/V element is used by several queries, raising intensity by the group size; kernels should load each K/V block once per group.

</details>

---

## 11. Evals and statistics

<details>
<summary><b>Q107</b> · <code>L1</code> · Model A scores 71.2% and B 69.8% on the same 1,000 items. Is A better?</summary>

**Answer.** Each score has a standard error of ≈ √(p(1 − p)/n) ≈ 1.4 points. Treated as independent, the difference has SE ≈ √(1.43² + 1.45²) ≈ 2.0 points, so a 1.4-point gap is well within noise (z ≈ 0.7). Since both models answered the same items, run a paired test on per-item results: if they mostly succeed and fail on the same items, the paired SE is much smaller and the difference may be real. Report the difference with a CI, not a ranking.

**Follow-up:** *Write the paired SE.* With d_i = a_i − b_i per item, SE = sd(d)/√n. Equivalently, McNemar's test on the discordant items (A right/B wrong vs A wrong/B right).

</details>

<details>
<summary><b>Q108</b> · <code>L2</code> · How many items do you need to detect a 2-point difference around 70% accuracy?</summary>

**Answer.** For an unpaired comparison at α = 0.05 (two-sided) and 80% power: n ≈ (1.96 + 0.84)² · 2p(1 − p)/δ² = 7.84 × 0.42/0.0004 ≈ 8,200 items per model. Most benchmarks are smaller, which is why small leaderboard gaps are often noise. Pairing reduces this in proportion to how correlated the models' per-item results are.

**Follow-up:** *How else can you reduce variance without more items?* Sample several generations per item and average (reduces within-item sampling noise, not between-item variance), and use paired designs.

</details>

<details>
<summary><b>Q109</b> · <code>L2</code> · When do you need clustered standard errors on an eval?</summary>

**Answer.** When items are not independent: several questions about the same passage, the same code repository, or templated variants of one problem. Treating them as independent overstates precision. Use cluster-robust SEs (sum residuals within each cluster) or bootstrap by resampling clusters. See Miller, "Adding Error Bars to Evals" ([arXiv:2411.00640](https://arxiv.org/abs/2411.00640)).

**Follow-up:** *How big can the effect be?* It depends on within-cluster correlation; with strong correlation and large clusters, the true SE can be several times the naive one. Lab 15 computes both.

</details>

<details>
<summary><b>Q110</b> · <code>L2</code> · How do you validate an LLM-as-judge?</summary>

**Answer.** Hand-label a stratified sample and measure agreement with the judge (Cohen's κ, or accuracy on pairwise preferences), compared with human–human agreement. Test for position bias (swap order), length and verbosity bias (length-controlled pairs) and self-preference (judge favoring its own family). Use rubrics and reference answers. Re-validate whenever the judge model or prompt changes.

**Follow-up:** *The judge agrees with humans 85% overall. Is that enough?* Look per slice: agreement can be much lower on the hard cases you care about. Also check whether the judge's errors correlate with the models being compared.

</details>

<details>
<summary><b>Q111</b> · <code>L1</code> · What is the unbiased pass@k estimator?</summary>

**Answer.** Sample n ≥ k completions per problem, c of them correct: pass@k = 1 − C(n − c, k)/C(n, k), averaged over problems. Compute it as 1 − Π_{i = n−c+1}^{n} (1 − k/i) to avoid huge binomials.

**Follow-up:** *Why not 1 − (1 − ĉ/n)^k?* Plugging the estimated success rate into a nonlinear function gives a biased estimate (it overestimates pass@k).

</details>

<details>
<summary><b>Q112</b> · <code>L2</code> · What makes an ablation convincing?</summary>

**Answer.** A tuned baseline at matched compute and tokens; one change at a time; several seeds with CIs; the effect replicating at two scales or on two datasets; the metric and prediction written down before the run; and negative results reported.

**Follow-up:** *The effect appears at 125M parameters but not at 350M. What do you conclude?* Nothing general yet. Check whether the baseline at 350M was tuned as well, then run a third scale; many small-scale tricks disappear with scale, and that itself is a useful finding.

</details>

<details>
<summary><b>Q113</b> · <code>L2</code> · How do you detect benchmark contamination?</summary>

**Answer.** Search training data for n-gram overlap with test items (e.g. 13-gram matches); compare performance on the original vs freshly rewritten or newly collected items; membership-inference signals (unusually low perplexity on test items or their exact ordering); canary strings; and time-split evaluations on problems published after the training cutoff.

**Follow-up:** *Contamination is found. What do you report?* Scores on the clean subset and on the full set, with the contamination rate, so readers can judge.

</details>

<details>
<summary><b>Q114</b> · <code>L2</code> · How are pairwise-battle leaderboards (Elo-style) computed, and what are the pitfalls?</summary>

**Answer.** Fit a Bradley–Terry model, P(i beats j) = σ(θ_i − θ_j), by logistic regression over all battles (more stable than online Elo, which depends on order); get CIs by bootstrapping battles. Pitfalls: the prompt distribution (who asks what), style effects (length, formatting) that sway voters, and non-transitive preferences. Style control adds length and formatting features as covariates.

**Follow-up:** *Two models' intervals overlap heavily. What do you say?* That the data do not distinguish them; a rank is not meaningful there.

</details>

<details>
<summary><b>Q115</b> · <code>L1</code> · Why do eval scores change when you rerun them, and what do you fix?</summary>

**Answer.** Sampling temperature, prompt format and few-shot examples, answer-extraction rules, max generation length (truncation), inference engine and batch size (non-deterministic kernels), and data version. Pin all of them, log them with each result, and report means over seeds with the variance.

**Follow-up:** *Greedy decoding is deterministic, right?* Not necessarily: batched kernels with different batch compositions change floating-point reduction order, which can flip near-tied argmaxes and diverge from there.

</details>

<details>
<summary><b>Q116</b> · <code>L3</code> · A benchmark is saturated at 95%. How do you decide whether to keep reporting it?</summary>

**Answer.** Estimate the label-error rate (audit a sample of the remaining "failures"); if many are wrong labels, the ceiling is reached and differences are noise. Check construct validity: does the benchmark still measure the capability you care about? Replace it with a harder or refreshed set, keep it only as a regression check, and report the audit.

**Follow-up:** *How would you design its successor?* Start from what the old one failed to distinguish, collect items where current models fail for reasons experts agree on, build in a private held-out split and refresh schedule, and pilot to confirm the scores spread out.

</details>

---

## 12. Interpretability and safety

<details>
<summary><b>Q117</b> · <code>L2</code> · What is an induction head?</summary>

**Answer.** A circuit of two attention heads in different layers. A previous-token head writes "the token before me was X" into each position. The induction head, in a later layer, queries with the current token, attends to earlier positions whose previous token matches it (K-composition with the first head), and copies the token found there. The result: [A][B] … [A] → predict [B]. It is a core mechanism of in-context copying.

**Follow-up:** *How would you show a head is an induction head?* On sequences of random tokens repeated twice, measure its attention from each second-half position to the token after the earlier occurrence, and ablate it to see in-context loss on the repeated half rise.

</details>

<details>
<summary><b>Q118</b> · <code>L2</code> · What is activation patching?</summary>

**Answer.** Run the model on a clean input and a corrupted one that changes the answer. Replace (patch) one component's activation in the corrupted run with its clean value and measure how much of the clean behavior (e.g. logit difference between correct and incorrect answers) is restored. Components with large recovery are causally important for that behavior.

**Follow-up:** *Patching every component is expensive. What approximates it?* Attribution patching: a first-order Taylor estimate using gradients, (clean − corrupted activation) · ∂metric/∂activation, computed for all components in one backward pass; then verify the top candidates with real patching.

</details>

<details>
<summary><b>Q119</b> · <code>L2</code> · What is superposition?</summary>

**Answer.** Models represent more features than they have dimensions by assigning them nearly orthogonal directions. It works because features are sparse (rarely active together), so interference is usually small. A consequence is polysemantic neurons, which respond to several unrelated features, so individual neurons are poor units of analysis.

**Follow-up:** *What experiment shows it?* Toy models (Elhage et al., 2022): train a small autoencoder with fewer hidden dimensions than input features and vary feature sparsity; beyond a sparsity level the model stores more features than dimensions in geometric arrangements.

</details>

<details>
<summary><b>Q120</b> · <code>L2</code> · How do sparse autoencoders (SAEs) find features, and how do you evaluate one?</summary>

**Answer.** Train an overcomplete autoencoder on activations: f = ReLU(W_e(x − b_d) + b_e), x̂ = W_d f + b_d, with loss ‖x − x̂‖² + λ‖f‖₁ (or a TopK activation instead of L1). Each latent should correspond to an interpretable feature. Evaluate with reconstruction error, sparsity (average L0), the language-model loss when the SAE's reconstruction replaces the activation, dead or always-on latents, and interpretability of sampled latents.

**Follow-up:** *What is feature splitting?* Larger SAEs split one broad feature into several finer ones, so the "features" found depend on dictionary size; there is no single true set.

</details>

<details>
<summary><b>Q121</b> · <code>L1</code> · A linear probe decodes "is this sentence true?" from layer 20 with 90% accuracy. What does that show?</summary>

**Answer.** That the information is linearly decodable there, not that the model uses it. The probe might exploit correlated features. Causal evidence requires intervening: steer along the probe direction or ablate it and check whether behavior changes as predicted.

**Follow-up:** *What is the logit lens?* Apply the final norm and unembedding to an intermediate residual stream to see which tokens it already predicts; the tuned lens learns a per-layer affine map to correct for representation drift.

</details>

<details>
<summary><b>Q122</b> · <code>L1</code> · What is prompt injection and how do you defend an agent against it?</summary>

**Answer.** Untrusted content in the model's context (a web page, email, tool output, document) contains instructions that the model follows as if they came from the user or developer. No filter is reliable, so design for containment: least-privilege tools and credentials, treat tool outputs as data, isolate untrusted content from privileged actions (e.g. a quarantined model processes untrusted text and cannot call tools), require confirmation for irreversible or external actions, restrict network egress, and log everything.

**Follow-up:** *What is the most dangerous combination?* An agent with access to private data, exposure to untrusted content, and a way to send data out (for example making web requests): injection can then exfiltrate data. Remove at least one of the three for any given task.

</details>

<details>
<summary><b>Q123</b> · <code>L2</code> · Reward hacking, sycophancy and deceptive alignment: how do they differ, and how would you test for sycophancy?</summary>

**Answer.** Reward hacking: the policy exploits flaws in the reward signal. Sycophancy: the model tells users what they seem to want (agreeing with stated opinions, caving under pushback), often a learned side-effect of human preference training. Deceptive alignment: a hypothesized failure where a model behaves well when it believes it is evaluated and differently otherwise. The first two are well documented; the third is harder to test and an active research area.

**Follow-up:** *Design a sycophancy eval.* Pairs of prompts that differ only in the user's stated opinion or identity, plus follow-ups where the user pushes back on a correct answer. Measure how often the answer changes, with CIs.

</details>

<details>
<summary><b>Q124</b> · <code>L3</code> · You are asked to evaluate whether a model has a dangerous capability. What makes the evaluation credible?</summary>

**Answer.** Elicitation effort: a negative result counts only if you tried hard to bring out the capability (good prompting, tools, scaffolding, fine-tuning where relevant, many samples), because under-elicitation gives false reassurance. Clear thresholds set before the run and tied to decisions (as in responsible-scaling-style frameworks), expert-built tasks close to real-world risk, uplift comparisons against a baseline (e.g. internet access alone), and reporting of uncertainty.

**Follow-up:** *How do you keep such evals from leaking into training data?* Keep them private, use canary strings, restrict access, and rotate items.

</details>

---

## 13. Applied LLM systems

<details>
<summary><b>Q125</b> · <code>L2</code> · How do you evaluate a RAG system?</summary>

**Answer.** Separately for each stage. Retrieval: recall@k and nDCG against labeled relevant passages. Generation: answer correctness, faithfulness (every claim supported by retrieved context), citation accuracy and refusal when evidence is missing. End to end on a fixed set of real queries, with slices by query type and CIs. A generation failure with good retrieval needs a different fix from a retrieval failure.

**Follow-up:** *No labels exist. How do you start?* Sample 100–200 real queries, label relevance for the top results of your current system and a strong baseline (pooling), and grow the set from production failures.

</details>

<details>
<summary><b>Q126</b> · <code>L1</code> · How does hybrid search with reciprocal rank fusion work?</summary>

**Answer.** Run lexical (BM25) and dense retrieval separately and combine ranks: score(d) = Σ over systems of 1/(k + rank_s(d)), with k = 60 as in the original paper. It uses ranks only, so no score calibration between systems is needed. BM25 catches exact identifiers and rare terms; dense retrieval catches paraphrases.

**Follow-up:** *When would you prefer learned score fusion?* When you have labeled data and the systems' scores carry useful confidence information that ranks throw away.

</details>

<details>
<summary><b>Q127</b> · <code>L1</code> · Bi-encoder vs cross-encoder reranker?</summary>

**Answer.** A bi-encoder embeds queries and documents separately, so documents are indexed once and search is fast nearest-neighbor lookup. A cross-encoder reads query and document together and scores relevance much more accurately, but costs a forward pass per pair, so it only reranks the top 50–200 candidates.

**Follow-up:** *What does reranking cost in latency?* Roughly one short forward pass per candidate, batched; with 100 candidates on a small model this is typically tens of milliseconds on a GPU. Measure it against your latency budget.

</details>

<details>
<summary><b>Q128</b> · <code>L1</code> · What does maximal marginal relevance (MMR) do?</summary>

**Answer.** Selects results one at a time, maximizing λ·sim(q, d) − (1 − λ)·max over already-selected s of sim(d, s), trading relevance for diversity. It prevents the context from filling with near-duplicate chunks.

**Follow-up:** *How do you set λ?* Tune it on your eval set; answer quality, not diversity, is the target metric.

</details>

<details>
<summary><b>Q129</b> · <code>L2</code> · Design the core loop of a tool-using agent. What goes wrong in practice?</summary>

**Answer.** Loop: the model receives the task, tool schemas and history; it emits a tool call or a final answer; the runtime validates arguments, executes in a sandbox with timeouts, truncates or summarizes the observation, appends it and repeats, within step, token and cost budgets. Common failures: loops that repeat the same failing call, context overflow from large observations, malformed arguments, giving up or declaring success early, and unsafe actions. Fix with structured errors returned to the model, observation limits, explicit stop conditions and permissions per tool.

**Follow-up:** *How would you evaluate it?* Tasks in a reproducible sandbox with programmatic success checks, measuring success rate over several trials, cost and steps, and failure categories from trajectory review (Q130).

</details>

<details>
<summary><b>Q130</b> · <code>L2</code> · Why measure agent reliability with pass^k rather than pass@k?</summary>

**Answer.** pass@k asks whether at least one of k attempts succeeds, which suits settings where you can check and retry. pass^k (introduced with τ-bench) asks whether all k independent attempts succeed, which is what a user relying on an agent experiences. An agent with 60% per-attempt success has pass^8 near 0.6⁸ ≈ 1.7% if attempts are independent.

**Follow-up:** *How do you estimate pass^k without bias from n ≥ k trials?* C(c, k)/C(n, k), the probability that k draws without replacement from the n trials are all successes.

</details>

<details>
<summary><b>Q131</b> · <code>L2</code> · Long context vs RAG: when do you use which?</summary>

**Answer.** Long context is simpler and better for tasks needing the whole corpus at once (cross-document synthesis) when the corpus fits and cost is acceptable, especially with prefix caching. RAG is necessary when the corpus is far larger than the context, for access control per document, freshness, citations and cost per query. Many systems combine them: retrieve generously, then let a long-context model read.

**Follow-up:** *What is the cost difference for a 200k-token context vs 8k of retrieved context per query?* About 25× more input tokens per query, and prefill time grows faster than linearly because of attention. Prefix caching helps only if the long context is the same across queries.

</details>

---

## 14. Napkin math

Do these on paper in under two minutes each. State assumptions out loud.

<details>
<summary><b>Q132</b> · <code>L1</code> · Can you fully fine-tune a 7B model on one 80 GB GPU? With LoRA? With QLoRA on your 8 GB laptop GPU?</summary>

**Answer.** Full fine-tuning with mixed-precision Adam: 16 bytes/param × 7B = 112 GB before activations, so no (without offloading or 8-bit optimizers). LoRA with a frozen bf16 base: 14 GB of weights plus small adapter states and activations, so yes on 80 GB. QLoRA: ≈ 4 GB of 4-bit weights with scales, plus adapters and activations; possible on 8 GB only with short sequences, small micro-batches and gradient checkpointing. Measure peak memory rather than trusting the estimate.

**Follow-up:** *What dominates memory at 4k sequence length in QLoRA?* Activations, which scale with sequence length and batch; checkpointing trades them for recomputation.

</details>

<details>
<summary><b>Q133</b> · <code>L2</code> · How long is prefill for a 100k-token prompt on Llama-2-70B-shaped weights across 8 H100s at 50% utilization?</summary>

**Answer.** Weight FLOPs: 2 × 70·10⁹ × 10⁵ = 1.4·10¹⁶. Causal attention: 2 · n_layers · T² · d = 2 × 80 × 10¹⁰ × 8,192 ≈ 1.3·10¹⁶, about as much as the weights at this length. Total ≈ 2.7·10¹⁶ FLOPs; throughput 8 × 989·10¹² × 0.5 ≈ 4·10¹⁵ FLOP/s; ≈ 7 s.

**Follow-up:** *What is the KV cache for that prompt?* 320 KiB × 100,000 ≈ 30.5 GiB (Q80), before any batching.

</details>

<details>
<summary><b>Q134</b> · <code>L1</code> · What is the compute-optimal model for a budget of 10²¹ FLOPs?</summary>

**Answer.** With C = 6ND and D = 20N: C = 120N², so N = √(10²¹/120) ≈ 2.9·10⁹ parameters and D ≈ 58·10⁹ tokens. Check: 6 × 2.9·10⁹ × 5.8·10¹⁰ ≈ 1.0·10²¹.

**Follow-up:** *You expect to serve 10¹² tokens in the model's lifetime. How does that change the choice?* Inference adds ≈ 2N FLOPs per served token, 2·10¹² × N in total, which exceeds the training budget for N in the billions; minimize training plus inference cost and you pick a smaller N trained on more tokens (lab 08).

</details>

<details>
<summary><b>Q135</b> · <code>L2</code> · What is the batch-1 decode speed ceiling for a 70B model in FP8 with TP = 4 on H100s?</summary>

**Answer.** 70 GB of weights split over 4 GPUs: 17.5 GB each at 3.35 TB/s ≈ 5.2 ms per token. TP adds two all-reduces per layer, 160 per token for 80 layers; these are small messages, latency-bound at roughly 10 µs each in practice, so ≈ 1.6 ms more. Ceiling ≈ 7 ms per token, ≈ 140 tokens/s, before KV reads and other overheads.

**Follow-up:** *Why not TP = 8 for lower latency?* Weight time halves to ≈ 2.6 ms, but communication does not shrink, so its share grows; the gain is real but less than 2×, and cost per token rises.

</details>

<details>
<summary><b>Q136</b> · <code>L1</code> · What does it cost to serve a million output tokens of an 8B model?</summary>

**Answer.** State your assumptions: one H100 at $2/hour (an assumption; check current prices), and an engine sustaining 5,000 output tokens/s aggregate across a large batch (measure it). That is 18M tokens/hour, so ≈ $0.11 per million output tokens at full utilization. At 30% average utilization the cost is ≈ 3.3× higher. Input tokens are cheaper per token because prefill is compute-efficient.

**Follow-up:** *Which single number most changes this estimate?* Utilization (traffic shape and autoscaling), then achieved throughput at your latency SLO.

</details>

> [!TIP]
> Build your own napkin sheet from lab 03 and keep it to one page: FLOPs per token, bytes per parameter for training and inference, the KV formula, your GPU's measured bandwidth and FLOP/s, and H100 reference numbers. Rehearse from it until you do not need it.

---

## Probes about your own work

Interviewers will ask about your projects with the same precision as the questions above. For every project on your résumé, prepare answers to:

- **Data:** What exactly was the test set? How many examples, how was it split, and could information leak from train to test?
- **Numbers:** What is the headline metric, with what confidence interval? What was the baseline, and was it tuned as carefully as your method?
- **Choices:** Why this loss, architecture or model size? What did you try that did not work?
- **Costs:** If you quantized or optimized something, what did it cost in accuracy and gain in latency or memory, measured how and on what hardware?
- **Next:** What would you do with ten times the compute? What result would change your mind?

For example, for an OCR model: which test set and how many images, character or word error rate with a CI, why a focal variant of CTC loss and what it changed, and the measured accuracy and latency before and after INT8 quantization. If the answer is "I don't remember", recompute it now, before an interviewer asks.

> [!TIP]
> Keep a one-page "numbers sheet" per project in your journal, with the exact command or notebook that produced each number. It turns deep-dive rounds from memory tests into conversations.
