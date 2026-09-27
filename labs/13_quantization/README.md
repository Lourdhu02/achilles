# Lab 13 — Quantization

**Build:** symmetric INT8 and INT4 quantization at per-tensor, per-channel and group-wise granularity, INT4 packing, the NF4 codebook from normal quantiles, SmoothQuant's outlier migration, and GPTQ's error-feedback solver for one layer. Then quantize real models on your 8 GB GPU and check the bandwidth prediction.<br>
**Time:** 3–4 h for the tests, 3–6 h for the scale-up · **Reads first:** [inference §4](../../curriculum/07-inference.md#4-quantization)<br>
**Run:** `pytest labs/13_quantization` (your code) · `pytest labs/13_quantization --impl=solution` (reference). The eight tests run on CPU in a few seconds.

Nearly every deployed LLM runs quantized. Weight bytes set decode speed and decide which model fits which card: on an 8 GB laptop GPU a 7B model exists only at 4 bits. Interviewers ask what an outlier does to a scale, why weight-only quantization speeds up decode but not prefill, how NF4 and GPTQ work, and what E4M3 means. This lab turns each answer into something you have computed.

---

## 1. Uniform quantization and its error

**Symmetric (absmax)**, b bits, as the lab implements it:

$$s = \frac{\max|w|}{2^{b-1}-1}, \qquad q = \mathrm{clamp}\big(\mathrm{round}(w/s),\ -(2^{b-1}-1),\ 2^{b-1}-1\big), \qquad \hat w = s\,q$$

INT8 uses [−127, 127] and INT4 [−7, 7]; the lab leaves −128 and −8 unused so the grid is symmetric.

**Asymmetric (zero point)**, for skewed data such as post-GeLU activations, maps [min, max] onto [0, 2ᵇ − 1]:

$$s = \frac{\max w - \min w}{2^{b}-1}, \qquad z = \mathrm{round}(-\min w / s), \qquad q = \mathrm{clamp}\big(\mathrm{round}(w/s) + z,\ 0,\ 2^{b}-1\big), \qquad \hat w = s\,(q - z)$$

The zero point is an integer, so 0.0 stays exact. Symmetric quantization of all-positive data wastes the negative half of the codes.

**Error.** Rounding error lies in [−s/2, s/2]. When values span many steps it is close to uniform there, so its mean square is $\frac{1}{s}\int_{-s/2}^{s/2} e^2\,de = s^2/12$ (the Δ²/12 rule). Each extra bit halves s and cuts the MSE by 4×.

Worked (the first test): 64 × 128 standard normal weights have max|w| = 4.10, so s = 4.10/127 = 0.0323 and s²/12 = 8.69e-5. The measured MSE is 8.66e-5, a signal-to-noise ratio of 40.6 dB.

## 2. Granularity: one outlier sets the scale

Every value that shares a scale pays for the largest one. In the second test, row 3 of that matrix is multiplied by 50. With one scale per tensor, max|w| = 205 and s = 1.61, so every ordinary weight (standard deviation 1) gets noise of variance s²/12 = 0.22, and 58% of them round to zero: the 63 normal rows use only 6 of 255 levels. One scale per row gives an MSE of 0.0031, 71× lower (the test asks for 10×).

| granularity | one scale per | typical use |
|---|---|---|
| per-tensor | tensor | static activation scales, FP8 |
| per-channel | output row of W | INT8 weights |
| per-group | g consecutive weights in a row (32–128) | INT4 weights: GPTQ, AWQ, GGUF |
| per-token | activation row, computed at runtime | dynamic activation quantization |

In the third test the columns are scaled by a ramp from 0.1 to 5. INT4 with one scale per tensor gives an MSE of 0.42; one scale per 64 columns gives 0.10, 4.1× lower. Scales cost storage: an fp16 scale per 64 weights is 4 + 16/64 = 4.25 bits per weight, per 128 is 4.125.

Why per-channel weights but per-token activations? The scale must factor out of the integer matmul. For $Y_{ij} = \sum_k X_{ik} W_{jk}$, a scale per token i and per output channel j pulls out as $s^x_i s^w_j \sum_k q^x_{ik} q^w_{jk}$. A scale per input channel k sits inside the sum and cannot, which is the problem SmoothQuant solves (§5).

## 3. INT4 packing

PyTorch has no 4-bit storage type, so store two values per `uint8`. `q & 0xF` maps −8…7 to the two's-complement nibbles 0…15 (−1 becomes 15). Low nibble first:

```
byte = (q[0] & 0xF) | ((q[1] & 0xF) << 4)        # [-3, 5] -> 0xD | 0x50 = 0x5D = 93
lo, hi = byte & 0xF, byte >> 4                   # then sign-extend: v - 16 if v > 7
```

Production kernels often permute values before packing so the GPU can unpack them quickly, so packed formats from different libraries are not interchangeable.

## 4. NF4 and double quantization

Trained weights are roughly Gaussian. After per-block absmax scaling to [−1, 1], a uniform grid spends levels on the tails where few weights live. NF4 (QLoRA, Dettmers et al., [2305.14314](https://arxiv.org/abs/2305.14314)) puts the 16 levels at quantiles of N(0, 1), so each level is used about equally often. The construction the test checks to 1e-4:

1. 8 positive levels: $\Phi^{-1}$ of `linspace(δ, 0.5, 9)[:-1]` (dropping 0.5, whose quantile is 0).
2. 7 negative levels: $-\Phi^{-1}$ of `linspace(δ, 0.5, 8)[:-1]`.
3. Add an exact 0, sort, and divide by the largest magnitude so the ends are ±1.

δ = 0.9677083 keeps the end quantiles finite, since $\Phi^{-1}(1) = \infty$; $\Phi^{-1}(δ) = 1.848$. Numerically, δ = 1 − ½(1/30 + 1/32).

**Why asymmetric:** 16 is even, and a grid symmetric about 0 contains 0 only if it has an odd number of levels. Symmetric INT4 gives up the code −8 to keep 0; NF4 keeps all 16 codes and an exact 0 by giving the positive side one more level. The result runs −1, −0.696, …, −0.091, 0, 0.080, …, 0.723, 1.

On 256 × 256 Gaussian weights with blocks of 64 (the sixth test), NF4's MSE is 0.0084 against 0.0115 for INT4 with group 64, 1.36× lower. The entropy of code usage shows why: 3.89 of 4 bits for NF4, 3.51 for INT4. NF4 is equal-probability, which is not quite MSE-optimal (Lloyd–Max would be).

**Double quantization.** The fp32 absmax of each 64-weight block adds 32/64 = 0.5 bits per weight. QLoRA quantizes those constants to 8 bits with one 32-bit scale per 256 blocks: 8/64 + 32/(64·256) = 0.127 bits, a saving of 0.373 bits per parameter, about 3 GB on a 65B model. QLoRA stores the base in NF4 but dequantizes to bf16 for every matmul, so NF4 saves memory, not compute ([lab 10](../10_lora/README.md)).

## 5. SmoothQuant: move activation outliers into the weights

For W8A8, activations must be quantized too, but a few input channels carry values 10–100× larger than the rest, in every token, so per-token scales do not help. For any positive per-input-channel vector s, $XW^\top = (X\,\mathrm{diag}(s)^{-1})(W\,\mathrm{diag}(s))^\top$. SmoothQuant (Xiao et al., [2211.10438](https://arxiv.org/abs/2211.10438)) picks

$$s_j = \frac{\max|X_{:,j}|^{\alpha}}{\max|W_{:,j}|^{1-\alpha}}$$

which leaves $\max|\hat X_j| = (\max|X_j| \max|W_j|)^{1-\alpha}$ and $\max|\hat W_j| = (\max|X_j| \max|W_j|)^{\alpha}$. At α = 0.5 both equal the geometric mean, splitting the difficulty evenly. The division by s is folded offline into the preceding LayerNorm, so it costs nothing at runtime.

Worked, on the seventh test's setup (32 × 16 activations with channel 2 multiplied by 40, 8 × 16 weights), simulating W8A8 with a per-tensor activation scale and per-row weight scales:

| smoothing | max \|X̂\| | max \|Ŵ\| | relative output MSE |
|---|---|---|---|
| none (s = 1) | 72.1 | 3.39 | 5.8e-4 |
| α = 0.25 | 35.4 | 3.29 | 8.5e-5 |
| α = 0.5 | 10.8 | 10.8 | 3.9e-5 |
| α = 0.75 | 3.29 | 35.4 | 4.6e-5 |

α = 0.5 cuts the error 15× here. The paper tunes α per model; 0.5 is the usual starting point.

## 6. GPTQ: let the remaining columns absorb each error

Round-to-nearest minimizes each weight's error. What matters is the layer's output on real inputs, $\lVert WX - \hat W X\rVert^2$ (paper notation: W is rows × columns, X is columns × samples). Per row this is a quadratic with Hessian $H = 2XX^\top$, the same for every row because it depends only on the inputs. When column j is rounded, Optimal Brain Surgeon gives the best change to the not-yet-quantized columns F:

$$\delta_F = -\frac{w_j - q_j}{[H_F^{-1}]_{jj}}\,(H_F^{-1})_{j,:}$$

GPTQ (Frantar et al., [2210.17323](https://arxiv.org/abs/2210.17323)) makes this practical at 175B parameters with three changes:

1. **One column order for all rows.** Earlier OBS-based methods pick a greedy order per row and need a separate inverse per row. Quantizing columns left to right for every row lets all rows share $H^{-1}$, computed once per layer.
2. **Cholesky of the inverse.** Column j needs only row j of $H_F^{-1}$, and with $H^{-1} = U^\top U$ (U upper triangular) that row is $U_{jj}\,U_{j,j:}$. So the update is `err = (w_j − q_j) / U[j, j]` then `W[:, j+1:] −= err ⊗ U[j, j+1:]`, exactly the lab's docstring. Factorizing once is numerically stable; repeatedly downdating $H^{-1}$ accumulates rounding error that at billions of parameters can make it indefinite. Dampening (1% of the mean diagonal, the lab's `damp`) is added to H first.
3. **Lazy batch updates.** Updating the whole remaining matrix after every column is memory-bound. GPTQ processes 128 columns at a time and applies the accumulated update to the rest in one matrix multiply. Same result, far better GPU utilization.

Worked (the eighth test): inputs of rank about 8 in 32 dimensions plus small noise, 16 × 32 weights, 4 bits. RTN's output MSE is 1.96, GPTQ's 0.30: 6.5× lower (the test asks for 2×). Yet GPTQ's weight MSE is 2× RTN's (0.017 against 0.0086): it moves weights along directions the inputs never excite. On i.i.d. inputs, where H is nearly diagonal, the gain disappears (output MSE ratio 0.97 in one run). AWQ (Lin et al., [2306.00978](https://arxiv.org/abs/2306.00978)) takes another route: it scales up the ~1% of weight channels that see the largest activations before quantizing, with scales searched on calibration data, and needs no Hessian.

## 7. FP8: E4M3 and E5M2

| format | sign, exponent, mantissa bits | max | smallest normal | smallest subnormal | eps |
|---|---|---|---|---|---|
| E4M3 (`torch.float8_e4m3fn`) | 1, 4, 3 | 448 | 2⁻⁶ = 0.0156 | 2⁻⁹ ≈ 0.00195 | 0.125 |
| E5M2 (`torch.float8_e5m2`) | 1, 5, 2 | 57,344 | 2⁻¹⁴ ≈ 6.1e-5 | 2⁻¹⁶ ≈ 1.5e-5 | 0.25 |

E4M3 spends the bit on precision and serves weights and activations; E5M2 spends it on range and serves gradients. The `fn` variant has no infinities and keeps only one NaN pattern per sign, which raises its max from 240 to 448. Floating point keeps relative error roughly constant across magnitudes, while integers keep absolute error constant. On a well-scaled Gaussian tensor INT8 is actually more accurate: with a per-tensor scale over 10⁶ samples, relative MSE is 1.2e-4 for INT8, 7.0e-4 for E4M3 and 2.8e-3 for E5M2. FP8 wins on range and on native tensor-core matmuls (Hopper, Ada, Blackwell, including your RTX 5060).

## 8. Napkin math: what 4 bits buy at decode

At batch 1, each generated token reads every weight once, so tokens/s ≤ bandwidth / weight bytes ([lab 03](../03_napkin_math/README.md), rule 9). For a 7B model (Ψ = 7 × 10⁹) on a GPU with 448 GB/s of memory bandwidth (the desktop RTX 5060's specification):

| weights | bytes | fits in 8 GB? | upper bound at 448 GB/s |
|---|---|---|---|
| bf16 | 14.0 GB | no | 32.0 tokens/s |
| INT8 | 7.0 GB | barely, little room for KV cache | 64.0 tokens/s |
| INT4, group 128 (4.125 bits) | 3.61 GB | yes | 124.1 tokens/s |

These are upper bounds. They ignore KV-cache reads (which grow with context), dequantization work and kernel efficiency. Laptop GPUs are often clocked lower than the desktop card, so replace 448 with the number you measured with [`tools/measure_gpu.py` in lab 03](../03_napkin_math/README.md#measure-your-gpu). Prefill is compute-bound and gains nothing from weight-only quantization, and at large batch decode nears the compute roof and the gain shrinks.

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `quantize_symmetric(w, bits=8, dim=None)` → `(q, s)` | `test_int8_roundtrip_error_is_bounded`, `test_per_channel_beats_per_tensor_with_outlier_rows` | `q` is `int8` with \|q\| ≤ 127; error ≤ s/2 everywhere; with `dim=1`, MSE < 0.1 × per-tensor MSE when one row is 50× larger |
| 2 | `quantize_groupwise(w, bits=4, group=64)` → `(q, s)` | `test_groupwise_int4_beats_per_tensor_int4` | lower MSE than per-tensor INT4 on ramp-scaled columns; `s` must broadcast in the given `dequantize_groupwise` |
| 3 | `pack_int4(q)`, `unpack_int4(packed, shape)` | `test_int4_packing_roundtrip` | 60 values in [−8, 7] become 30 `uint8` bytes and round-trip exactly |
| 4 | `nf4_codebook()` | `test_nf4_codebook_matches_qlora` | the 16 published QLoRA values to 1e-4 |
| 5 | `quantize_nf4(w, block=64)` → `(idx, s)` | `test_nf4_beats_uniform_int4_on_gaussian_weights` | lower MSE than group-64 INT4 on Gaussian weights |
| 6 | `smoothquant(w, act_absmax, alpha=0.5)` → `(w * s, s)` | `test_smoothquant_preserves_output_and_tames_outliers` | `(x / s) @ ws.T` equals `x @ w.T`, and max\|x/s\| < half of max\|x\| |
| 7 | `gptq_quantize(w, x, bits=4, damp=0.01)` → dequantized W | `test_gptq_beats_round_to_nearest_on_correlated_inputs` | output MSE below half of RTN's on correlated inputs |

Given: `dequantize`, `dequantize_groupwise` and `dequantize_nf4`. Read them first: they fix the shapes your scales must have.

## Tips

> [!TIP]
> Group-wise quantization is per-channel quantization after a reshape: view `(rows, cols)` as `(rows, cols // group, group)`, call `quantize_symmetric(..., dim=-1)` (which must use `keepdim=True`), and reshape `q` back. The scale keeps shape `(rows, cols // group, 1)`, which is what `dequantize_groupwise` expects.

- Clamp scales away from zero (`clamp(min=1e-12)`): an all-zero row otherwise gives 0/0.
- Do the bit operations in `int16`, then cast to `uint8`; sign-extend on unpacking with `torch.where(u > 7, u - 16, u)`.
- Build NF4 in float64 with `torch.special.ndtri`, then cast. Nearest-level lookup is one broadcast: `((x / s)[..., None] - code).abs().argmin(-1)`.
- In GPTQ, work in float64, compute the per-row scales from the original weights before any error feedback (as the docstring says), and get U with `torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)`.

## Common bugs

- **Scales without `keepdim`:** shape `(64,)` against a `(64, 128)` tensor fails, or on a square matrix silently scales the wrong axis.
- **qmax = 2ᵇ⁻¹ instead of 2ᵇ⁻¹ − 1:** 128 overflows `int8` and wraps to −128.
- **Truncating instead of rounding** (`.to(torch.int8)` on floats): errors reach s instead of s/2, and the MSE roughly quadruples (s²/3 instead of s²/12).
- **No sign extension when unpacking:** −8…−1 come back as 8…15.
- **SmoothQuant over the wrong axis:** `w` is (out, in); the weight maximum per input channel is `w.abs().amax(0)`, and `w * s` scales columns.
- **GPTQ with `x @ x.T`** (samples × samples) instead of `x.T @ x`, with the lower Cholesky factor, or without dampening, which can make the factorization fail on near-singular H.

## CPU experiments (no GPU needed)

1. **Bits sweep:** quantize a Gaussian tensor at 2–8 bits and plot the MSE against s²/12. Where does the uniform-noise model break?
2. **Zero point:** implement `quantize_asymmetric` and compare it with symmetric quantization on `torch.nn.functional.gelu(torch.randn(...))` at 8 and 4 bits. Predict the MSE ratio first.
3. **Outliers and groups:** sweep the outlier multiplier (1–100) in test 2's setup, and the group size (16–256) in test 3's, plotting MSE against bits per weight including scales.
4. **FP8 on CPU:** `x.to(torch.float8_e4m3fn).float()` works without a GPU. On a log-uniform tensor spanning 10⁻³ to 10², plot relative error against magnitude for E4M3 and for per-tensor INT8.
5. **GPTQ on a real layer:** record the inputs to one MLP layer of your lab-05 GPT on 128 sequences, compare RTN and GPTQ output error, then quantize every linear layer both ways and compare validation loss.

## GPU scale-up (8 GB)

Measure your bandwidth first (lab 03). A free Colab or Kaggle T4 works too; it has no bf16 or FP8 tensor cores, so use fp16 there.

1. **Baseline:** load Qwen2.5-1.5B-Instruct in bf16 (1.54 B parameters, 3.1 GB, a bound of about 145 tokens/s at 448 GB/s, lower on a laptop part). Measure batch-1 decode tokens/s over 256 greedy tokens after a warm-up, with `torch.cuda.synchronize()` around the timer, and compare with your bound. Also time Qwen2.5-0.5B: small models at batch 1 are often limited by kernel launches, not bandwidth.
2. **Your own quantizers:** replace each linear layer's weight with your INT8 per-channel and INT4 group-128 versions, dequantized back to bf16, and measure perplexity on held-out text (same text and context length for every variant). This measures quality only: dequantizing a whole matrix per call writes a bf16 copy, so it cannot be faster.
3. **A real low-bit kernel:** load the same model with bitsandbytes (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`), torchao, or a GGUF file in llama.cpp. Report memory, perplexity and tokens/s against bf16. Speed depends on whether the kernel fuses dequantization into the matmul, not only on bytes.
4. **A 7B-class model at 4 bits:** it fits with a few GB to spare. Count which layers stayed in 16 bits (embeddings and `lm_head` often do) and redo the bound with the real byte count.
5. **Beyond perplexity:** evaluate bf16 against INT4 on 200 or more questions of a task you can grade (for example GSM8K exact match) with a paired confidence interval ([lab 15](../15_eval_stats/README.md)).

## Check yourself

1. INT8 absmax on a tensor with max|w| = 2.54: what are the step size and the expected MSE?
2. Why are weights quantized per output channel but activations per token, and not per input channel?
3. How many bits per weight does INT4 with group 64, an fp16 scale and an fp16 zero point cost?
4. NF4 has 16 levels and an exact zero, while symmetric INT4 in this lab uses 15. How?
5. SmoothQuant with α = 0.5 on a channel with max|X_j| = 64 and max|W_j| = 0.25: what are s_j and the new maxima?
6. GPTQ raised the weight MSE compared with RTN but lowered the output error 6.5×. How is that possible?
7. Your INT4 model decodes no faster than bf16. Give two plausible reasons.

<details><summary>Answers</summary>

1. s = 2.54/127 = 0.02; the MSE is about s²/12 = 3.3e-5.
2. The scales must factor out of the integer matmul: per-token and per-output-channel scales become an outer product applied to the int32 accumulator, while a per-input-channel scale sits inside the sum over k. SmoothQuant moves that per-input-channel variation into the weights instead.
3. 4 + (16 + 16)/64 = 4.5 bits.
4. NF4 is asymmetric: 8 positive levels, 7 negative levels and an exact 0. A symmetric grid with 0 needs an odd number of levels.
5. s_j = 64^0.5 / 0.25^0.5 = 8 / 0.5 = 16. The activation maximum becomes 64/16 = 4 and the weight maximum 0.25 × 16 = 4, both √(64 × 0.25).
6. Output error is weighted by H = 2XXᵀ. GPTQ pushes each rounding error onto later columns along directions where the inputs have little energy, so the weights move further but the outputs move less.
7. The kernel dequantizes to a bf16 copy instead of fusing into the matmul, so bytes read do not fall; the model is small enough that batch-1 decode is launch-bound; the workload is dominated by prefill, which is compute-bound; or the 4-bit kernel is not tuned for your GPU.
</details>

## Stretch

- **AWQ for one layer:** search per-input-channel scales of the form mean|X_j|^a over a ∈ [0, 1] to minimize output error, and compare with RTN and GPTQ.
- **Lazy-batch GPTQ:** process 128 columns at a time, check the result matches your column-by-column version, and time both on a 4096 × 4096 layer.
- **Act-order:** quantize columns in order of decreasing diag(H) and compare output error.
- **Rotations:** multiply W by a random orthogonal (for example Hadamard) matrix Q and X by Qᵀ; show that outliers spread across channels and INT4 error falls (the idea behind QuaRot).
- **INT8 KV cache:** quantize the cache in your [lab 07](../07_kv_cache_sampling/README.md) decoder per token and measure memory and quality.
