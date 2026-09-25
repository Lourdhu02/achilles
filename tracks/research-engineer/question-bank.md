# Question bank: core AI

Answer out loud first, then open the answer. Mark misses and revisit them in 2 weeks. The answers are compressed; each should expand to a 2–3 minute whiteboard explanation.

## Math and deep learning
<details><summary>1. Gradient of softmax cross-entropy w.r.t. the logits?</summary>`softmax(z) − onehot(y)` (divided by N for a mean). That's why frameworks fuse the two: it's stable and cheap.</details>
<details><summary>2. Why does Adam's first update have magnitude ≈ lr regardless of gradient scale?</summary>Bias-corrected m̂ = g and v̂ = g², so the step is lr·g/|g| = lr·sign(g). Adam is sign-like with per-parameter trust.</details>
<details><summary>3. AdamW vs Adam + L2?</summary>With L2, the decay gradient λθ gets divided by √v̂, so heavily-updated parameters are barely regularized. AdamW shrinks θ by (1 − ηλ) directly, uniformly.</details>
<details><summary>4. Why warmup?</summary>Early curvature is high and Adam's second-moment estimates are noisy. Full-size steps can push the model into unstable regions (loss spikes, divergence). Warmup lets the statistics settle and the loss surface smooth out.</details>
<details><summary>5. Pre-LN vs post-LN?</summary>Pre-LN keeps an identity residual path, so gradients reach early layers unattenuated and training is stable without careful warmup. Post-LN normalizes the stream itself and can give slightly better quality but is fragile when deep. Pre-LN needs a final norm.</details>
<details><summary>6. Loss is NaN at step 3,000 after being fine. Walk through it.</summary>Check the pre-clip grad-norm history, max attention and output logits, and the offending batch (data bug?). Check fp16 overflow vs bf16. Usual fixes: rewind to a checkpoint and skip the batch, lower the LR, add QK-norm or z-loss, check that the loss mask isn't producing 0/0.</details>
<details><summary>7. What does μP buy you?</summary>Width-independent optimal hyperparameters: tune the LR and init on a small proxy and transfer them to the full model, instead of sweeping at full scale.</details>
<details><summary>8. Why is batch size not "free throughput"?</summary>Beyond the critical batch size (the gradient noise scale), more samples per step add little information, so you need proportionally more compute for the same loss. Also memory and LR-scaling issues.</details>

## Transformers
<details><summary>9. Why divide by √d_k?</summary>For unit-variance q and k, q·k has variance d_k. Unscaled logits saturate the softmax and kill the gradients.</details>
<details><summary>10. Parameter count of Llama-3-8B from its config?</summary>Embeddings 128,256·4096 ≈ 525M, ×2 because the head is untied; per layer: attention 2·4096² + 2·4096·1024 = 41.9M, MLP 3·4096·14336 = 176.2M; ×32 ≈ 6.98B. Total ≈ 8.03B.</details>
<details><summary>11. KV cache for Llama-2-70B at 32k context, batch 1, bf16?</summary>2·80·8·128·32768·2 B = 10 GiB. (It would be 80 GiB with MHA's 64 KV heads; GQA is the reason it isn't.)</details>
<details><summary>12. How does RoPE encode relative position?</summary>Rotating q at m and k at n by angles proportional to position makes q·k depend only on m − n: Re⟨q e^{imθ}, k e^{inθ}⟩ = Re(q k̄ e^{i(m−n)θ}). It has no parameters and preserves norms.</details>
<details><summary>13. MQA vs GQA vs MLA?</summary>MQA: one KV head (smallest cache, some quality loss). GQA: H_kv groups (a good compromise, now the default). MLA: cache a low-rank latent per token and up-project, with a separate small RoPE component. Its cache is even smaller than GQA's at similar or better quality.</details>
<details><summary>14. Why SwiGLU at 8/3·d?</summary>The gated MLP has 3 matrices, so ~8/3·d hidden keeps its parameters and FLOPs equal to a 4d two-matrix MLP. Gating gives consistent quality gains.</details>
<details><summary>15. MoE: why are total parameters ≫ active, and what's hard?</summary>Each token uses top-k experts, so compute scales with active parameters and capacity with total. Hard parts: load balancing (aux loss, bias balancing), all-to-all communication, memory to hold every expert, and routing instability.</details>
<details><summary>16. What is an induction head?</summary>A two-layer circuit: a previous-token head writes "the token before me" into each position; an induction head matches the current token against those and copies the following token. It is a core mechanism of in-context copying.</details>

## Pretraining and scaling
<details><summary>17. Chinchilla in one sentence, and why labs over-train anyway.</summary>At fixed compute, loss is minimized with N and D scaled about equally (~20 tokens/param). But inference cost scales with N, so models that will serve many tokens are trained smaller and longer.</details>
<details><summary>18. Training time for 7B on 2T tokens with 256 H100s at 40% MFU?</summary>6·7e9·2e12 = 8.4e22 FLOPs ÷ (256 · 989e12 · 0.4) ≈ 8.3e5 s ≈ 9.6 days.</details>
<details><summary>19. Why MinHash dedup, and what does the band/row choice control?</summary>Near-duplicates waste compute, cause memorization and contaminate evals. P(candidate) = 1 − (1 − s^r)^b sets where the S-curve thresholds on Jaccard similarity s, trading recall against false positives.</details>
<details><summary>20. TP vs PP vs FSDP: when do you use each?</summary>TP splits matmuls and needs an all-reduce every layer, so it stays within NVLink. PP splits layers and sends small point-to-point messages across nodes, at the cost of bubbles. FSDP shards state and all-gathers weights per layer, a good default until communication dominates. Combine them: TP inside a node, PP/FSDP across nodes, DP for throughput.</details>
<details><summary>21. Ring all-reduce cost?</summary>Each device sends and receives 2(n−1)/n of the buffer: nearly independent of n, bandwidth-bound. Hide it by overlapping with backward.</details>
<details><summary>22. Why do loss spikes happen and how are they handled at scale?</summary>Attention or output logit growth, bad batches, LR too high, numeric overflow. Mitigations: QK-norm, z-loss, a lower LR, skipping the batch and rewinding, careful init. Monitor grad-norm and max-logit continuously.</details>

## Post-training
<details><summary>23. Derive DPO.</summary>Maximize E[r] − βKL(π‖π_ref), whose optimum is π* = π_ref·e^{r/β}/Z. Then r = β log(π*/π_ref) + β log Z. Substitute into Bradley–Terry; Z cancels in r_w − r_l. The loss is −log σ(β(Δ log-ratio_w − Δ log-ratio_l)).</details>
<details><summary>24. Why can DPO lower the likelihood of the chosen response?</summary>The loss only depends on the *margin*; it can widen by pushing both down, with the rejected one falling faster. Probability mass moves to unseen outputs ("likelihood displacement"). Mitigations: an SFT term, conservative β, online data.</details>
<details><summary>25. GRPO vs PPO?</summary>GRPO drops the value network: the advantage is the reward normalized within G samples per prompt. It uses a clipped ratio and a KL to the reference (k3 estimator). It is cheaper and a natural fit for verifiable rewards.</details>
<details><summary>26. Name three GRPO failure modes and their fixes.</summary>Length bias from per-sequence averaging (use a token-level mean or constant normalizer); entropy collapse (clip-higher); zero-advantage groups (dynamic sampling). Plus reward hacking, handled by audits and a better verifier.</details>
<details><summary>27. Why a KL penalty in RLHF?</summary>It keeps the policy near the reference, where the reward model is accurate, which limits reward hacking and preserves capabilities. It is reverse KL, so it is mode-seeking.</details>
<details><summary>28. LoRA: why B = 0 at init, and why does it work?</summary>B = 0 means training starts exactly at the base model. Fine-tuning updates have low intrinsic rank, so rank 8–64 captures most of the gain with ~1% of the parameters.</details>
<details><summary>29. Process vs outcome reward models?</summary>ORMs score the final answer; PRMs score each step. PRMs give denser credit and better search guidance but are costly to label and easy to hack. Verifiable outcomes plus RL have dominated recent reasoning gains.</details>

## Inference
<details><summary>30. Why is decode memory-bound?</summary>Each step multiplies every weight matrix by a vector per sequence: ~1 FLOP per byte at batch 1, far below the ridge point (~300 on an H100). Time ≈ bytes / bandwidth.</details>
<details><summary>31. Batch-1 upper bound for an 8B model in bf16 on an H100?</summary>16 GB / 3.35 TB/s ≈ 4.8 ms, so ≤ ~210 tokens/s.</details>
<details><summary>32. Is speculative decoding lossless?</summary>Yes, with rejection sampling: accept with probability min(1, p/q), resample from norm(max(0, p − q)). Only the greedy "argmax matches" variant is merely lossless under greedy decoding. Expected tokens per pass = (1 − α^{k+1})/(1 − α).</details>
<details><summary>33. Why does weight-only INT4 speed up decode but not prefill?</summary>Decode is memory-bound, so fewer bytes means proportionally faster. Prefill is compute-bound, and dequantizing adds work without cutting FLOPs (unless you have low-precision matmul hardware).</details>
<details><summary>34. What does PagedAttention fix?</summary>Fragmentation and over-reservation of contiguous KV buffers. Fixed-size blocks and block tables give near-zero waste, larger batches, and copy-on-write prefix sharing.</details>
<details><summary>35. Why does FlashAttention help if FLOPs don't drop?</summary>It removes the Θ(T²) HBM reads and writes of the score matrix; attention is IO-bound. The backward pass recomputes P from the saved log-sum-exp.</details>
<details><summary>36. Chunked prefill vs disaggregation?</summary>Chunked prefill interleaves prompt chunks with decode steps on the same GPUs, smoothing ITL. Disaggregation runs prefill and decode on separate pools with a KV transfer, so each is optimized independently; it wins at large scale.</details>

## Evaluation and research
<details><summary>37. Model A scores 71.2%, B 69.8% on 1,000 items. Is A better?</summary>The unpaired SE is ≈ 1.45 points per model, so the difference is within noise. Run a paired test on per-item results; if the models mostly fail on the same items the paired CI is much tighter. Report the CI, not the ranking.</details>
<details><summary>38. How do you validate an LLM judge?</summary>Label a stratified sample by hand and measure agreement (κ). Check position and length bias by swapping order and controlling length. Use rubrics and references. Re-validate whenever the judge model changes.</details>
<details><summary>39. What makes an ablation convincing?</summary>A tuned baseline at matched compute; one change at a time; multiple seeds with CIs; the effect holding at two scales or datasets; a prediction written before the result.</details>
<details><summary>40. Unbiased pass@k?</summary>With n ≥ k samples, c correct: 1 − C(n−c, k)/C(n, k), computed as a stable product.</details>

## Behavioral probes you must be able to answer with numbers
Your OCR model's accuracy: on what test set, how many images, how was it split, what is the CI? Why FocalCTCLoss? What did the INT8 quantization cost in accuracy and gain in latency, measured how? If the answer is "I don't remember", fix it now, before an interviewer finds it.
