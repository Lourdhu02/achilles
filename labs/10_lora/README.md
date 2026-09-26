# Lab 10 — LoRA

**Build:** a LoRA adapter around a frozen `nn.Linear`, adapter merging, a function that retrofits LoRA into any model by layer name, and a direct numerical test of the low-rank hypothesis. Then LoRA SFT of a real 0.5 B model on your 8 GB GPU with your own adapter code.
**Time:** 3–4 h for the tests, 4–8 h for the scale-up · **Reads first:** [post-training §1–2](../../curriculum/06-post-training.md#2-parameter-efficient-fine-tuning)
**Run:** `pytest labs/10_lora` (your code) · `pytest labs/10_lora --impl=solution` (reference). The four tests run on CPU in a few seconds.

LoRA is the default way to fine-tune on a budget and the way most products serve thousands of customer-specific models on one base. Interviewers ask for its initialization, its parameter count, its merge, and when it is the wrong tool; this lab makes each of those something you have checked.

---

## 1. The idea

Freeze the pretrained weight $W_0$ (shape out × in) and learn a low-rank update:

```
            ┌──────────── W0 x (frozen) ─────────────┐
x (in) ─────┤                                        (+) ──► h (out)
            └── A (r × in) ──► B (out × r) ──► ·α/r ─┘
                 Kaiming init     zeros at init
```

$$h = W_0 x + \tfrac{\alpha}{r} B A x$$

- **Init:** `A` Kaiming-uniform, `B = 0`, so `BA = 0` and the first forward equals the base model exactly. `B` still gets a gradient (it is multiplied by the nonzero `A x`); if both were zero, neither would.
- **Parameters:** `r · (in + out)` per adapted matrix, instead of `in · out`.
- **Scale α/r:** keeps the size of the update roughly constant when you change `r` at fixed `α`.
- **Merge:** `W = W0 + (α/r)·B A` is an ordinary matrix. A merged adapter adds zero inference cost.

## 2. The low-rank hypothesis, tested

`fit_low_rank_delta(r)` (given) builds a frozen 16 × 16 base `W0` and a target `W0 + Δ` where `Δ` has rank 2, then trains `LoRALinear(base, r, alpha=r)` (scale 1) to match it with Adam. Theory says: with `r ≥ 2` the adapter can represent `Δ` exactly, so the loss goes to ~0; with `r = 1` the best it can do is the best rank-1 approximation of `Δ` (Eckart–Young), so the loss plateaus at the energy in the second singular value. `test_low_rank_hypothesis` asserts exactly that: `r = 2` reaches MSE < 1e-3 and `r = 1` stays more than 20× higher.

LoRA works on real models because the *update* that fine-tuning needs is often close to low-rank, even though the weights themselves are full-rank. When the update is not low-rank (large new domains, lots of data), LoRA underfits: that is the "learns less, forgets less" trade in the curriculum.

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `LoRALinear.forward(x)` | `test_identity_at_init_and_only_adapters_train` | output equals `base(x)` at init; only `A` and `B` require grad (the constructor, given, freezes the base) |
| 2 | `LoRALinear.merge()` | `test_merge_matches_adapter` | returns a plain `nn.Linear` (bias copied if present) whose output equals the adapter's, after `B` is randomized |
| 3 | `apply_lora(model, targets, r, alpha)` | `test_apply_lora_and_param_count` | freezes **all** parameters, then replaces every `nn.Linear` whose attribute name is in `targets`, in place, at any depth; trainable count is `2 · r · (32 + 32)` for two 32 × 32 targets |
| 4 | (given) `fit_low_rank_delta` | `test_low_rank_hypothesis` | uses your `forward`; passes only if the scale and the init are right |

Given: `LoRALinear.__init__`, `trainable_params`, `fit_low_rank_delta`.

## Tips

> [!TIP]
> In `apply_lora`, iterate over `list(model.modules())` and `list(module.named_children())`: you are mutating the tree while walking it, and taking a snapshot first avoids visiting the `LoRALinear` you just inserted (and its inner `base`). Match on the **attribute name** (`"q_proj"`), not the full dotted path, so the same targets work at every layer.

- Write the forward as `base(x) + (x @ A.T @ B.T) * scale`, applying `A` first: `(x A^T) B^T` costs `r·(in + out)` per token, while forming `B A` first costs `in·out·r` and materializes a full matrix.
- `merge()` runs under `@torch.no_grad()` (already decorated); copy into the new layer's parameters with `.copy_` so the result is a fresh, independent `nn.Linear`.

## Common bugs

- **Changing the given init** (both matrices random, or re-initializing inside `forward`): the output at init is no longer the base model and `test_identity_at_init_and_only_adapters_train` fails.
- **Scale applied twice**, or `α` used instead of `α/r`: the low-rank test still converges (Adam adapts) but merge and adapter outputs disagree if you apply it in only one place.
- **`B @ A` shape confusion:** `B` is `(out, r)` and `A` is `(r, in)`, so `B @ A` is `(out, in)`, the same shape as `base.weight`.
- **Forgetting to freeze non-target parameters** in `apply_lora`: the trainable count includes `mlp` and the test fails.
- **Dtype and device mismatch on a real model:** the given constructor creates `A` and `B` in float32 on the CPU. If the base is bf16 on the GPU, a plain forward fails with a dtype error. Apply LoRA *before* `model.to("cuda")`, and run the forward under `torch.autocast("cuda", dtype=torch.bfloat16)` so the matmuls run in bf16 while the adapter weights (and their Adam state) stay in fp32.

## CPU experiments (no GPU needed)

1. **Rank sweep:** call `fit_low_rank_delta(r)` for `r = 1 … 4` and `true_rank = 1 … 4`. Plot the final MSE. Predict each value from the singular values of `Δ` first (compute them with `torch.linalg.svdvals`).
2. **What did it learn?** Modify a copy of `fit_low_rank_delta` to return the trained layer and compute the singular values of `scale · B @ A` for `r = 4`. You should see two large values and two near zero: extra rank goes unused.
3. **Where to put adapters:** on the lab 05 GPT, apply LoRA to attention only vs MLP only vs all linear layers with the same total parameter budget, fine-tune on a small shifted dataset, and compare held-out loss.

## GPU scale-up (S4): LoRA SFT of Qwen2.5-0.5B-Instruct on 8 GB

Install `transformers` (and optionally `datasets`). Use your own `apply_lora`: the Qwen2 layers are named `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from labs._impl import load
lo = load("labs/10_lora/x.py", "exercise")        # any path in the lab dir

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
lo.apply_lora(model, targets, r=16, alpha=32)     # adapters are fp32, on CPU
model.to("cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print(lo.trainable_params(model))                 # predict this number first

END = tok.convert_tokens_to_ids("<|im_end|>")
def encode(prompt: str, answer: str):
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    a = tok(answer, add_special_tokens=False).input_ids + [END]   # train the stop token
    return p + a, [-100] * len(p) + a                             # loss on the answer only

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
# for each (padded) batch of ids, labels, attention_mask:
#     with torch.autocast("cuda", dtype=torch.bfloat16):
#         loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
#     loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
```

Plan it on paper first ([post-training §8](../../curriculum/06-post-training.md#8-an-8-gb-practical-path)):

- **Trainable parameters:** 24 layers × 366,592 = 8,798,208 (1.8% of 494 M). With fp32 weights, grads and two Adam moments that is ~0.14 GB.
- **Logits:** batch × sequence × 151,936 × 4 bytes in fp32. Batch 4 × 1,024 is 2.5 GB. Start with batch 4 × 512 and gradient accumulation.
- **Data:** 1–2 k examples of a task you can grade automatically. The GSM8K train split (`openai/gsm8k` on the Hugging Face Hub) works well: its answers end in `#### <number>`, so exact match on the final number is your metric, and the same task carries into DPO (lab 11) and GRPO (lab 12).
- **Hyperparameters to start from:** `r = 16`, `α = 32`, learning rate 1e-4 to 2e-4 with a short warmup, 1–3 epochs, sequence length 512.

Evaluate before and after on 300–500 held-out questions with greedy decoding, and report the accuracy difference with a paired bootstrap confidence interval ([lab 15](../15_eval_stats/README.md)). Also evaluate on a small general-chat set to see what, if anything, got worse.

Then **merge**: replace every `LoRALinear` with `module.merge()` (cast the result to bf16), check that logits match the unmerged model to bf16 tolerance, and compare decode tokens/s merged vs unmerged at batch 1.

> [!TIP]
> Decode one batch back to text with `tok.decode`, printing the label-masked positions as `_`, before the first training step. Template, masking and stop-token bugs are visible in ten seconds this way and invisible in the loss curve.

**Pitfalls specific to the scale-up:**

- With all base parameters frozen and **reentrant** gradient checkpointing, no input to a checkpointed block requires grad, so the adapters silently get no gradient. Use `use_reentrant=False` (as above) or `model.enable_input_require_grads()`. Check that `A.grad` is non-zero after the first backward.
- Qwen2.5's chat template inserts a default system prompt when you give none. Use the same template call for training and evaluation.
- Pad with a token that is masked in `labels` via `-100`, not by ID: if `pad_token_id` equals the end-of-turn ID and your collator masks pad IDs, the model never learns to stop.

## Check yourself

1. Why initialize `B = 0` rather than `A = 0`?
2. LoRA with `r = 8` on a 4096 × 4096 matrix: how many trainable parameters, and what fraction of the matrix?
3. Why does merging make LoRA free at inference, and when would you *not* merge?
4. Your LoRA fine-tune matches full fine-tuning on a 2 k-example task but falls far behind on 2 B tokens of code. Why?
5. The r = 1 fit in `test_low_rank_hypothesis` plateaus. What determines the plateau value?
6. Why does QLoRA not update the 4-bit weights, and how do gradients reach the adapters?

<details><summary>Answers</summary>

1. Either choice gives `BA = 0`, so training starts at the base model. With `B = 0` and `A` random, the first step updates `B` (its gradient `scale · g (A x)ᵀ` is non-zero) while `A`'s gradient `scale · Bᵀ g xᵀ` is zero; the mirror choice updates `A` first. What you must avoid is both zero (neither ever gets a gradient) or both random (step 0 is a perturbed model). `B = 0` is the LoRA paper's convention and what the given constructor does.
2. `8 · (4096 + 4096) = 65,536`, which is 0.39% of 16.8 M.
3. `W0 + (α/r) B A` is one dense matrix of the original shape, so the merged model has exactly the base architecture. Do not merge when serving many adapters on one base (S-LoRA style batching), or when you want to hot-swap adapters.
4. Adapter capacity: a large, diverse dataset needs an update that is not low-rank. That is the "learns less, forgets less" regime; use higher rank, more target layers, or full fine-tuning.
5. The best rank-1 approximation of Δ leaves the energy of its second singular direction unexplained (Eckart–Young); the MSE is that residual averaged over the random inputs.
6. The frozen base is stored in NF4 and dequantized to bf16 on the fly for each matmul. The backward pass flows through the dequantized weights into the activations and on to the bf16 adapters; the base has no gradient or optimizer state.
</details>

## Stretch

- **rsLoRA:** use scale `α/√r` and repeat the rank sweep at a fixed α; compare how final loss changes with `r`.
- **DoRA:** decompose `W` into magnitude and direction and train LoRA on the direction only; compare with plain LoRA at the same rank on the S4 task.
- **Multi-adapter batching:** two adapters on one base in one forward pass, where each row of the batch selects its adapter (the idea behind S-LoRA and Punica). Verify outputs match running each adapter separately.
- **QLoRA:** quantize the frozen base with your NF4 code from [lab 13](../13_quantization/README.md), dequantize per matmul, and fine-tune Qwen2.5-1.5B in under 8 GB.
