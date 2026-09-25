# Lab 10 — LoRA

**Run:** `pytest labs/10_lora` (your code) · `pytest labs/10_lora --impl=solution` (reference)

**Reads first:** [post-training §2](../../curriculum/06-post-training.md#2-parameter-efficient-fine-tuning)

## What you implement (the tests check each property)
- `LoRALinear(base, r, alpha)`: frozen base, `A` Kaiming-initialized, `B = 0`, scale α/r. Test: output equals the base at init; only A and B get gradients.
- `merge()` returns an `nn.Linear` with `W + (α/r)·BA`. Test: merged output equals the LoRA output.
- `apply_lora(model, targets)` replaces named `nn.Linear` layers in place.
- **The low-rank test:** a target equal to the base plus a rank-2 delta. r = 2 fits it to near-zero loss; r = 1 plateaus. That is the hypothesis behind LoRA, demonstrated.

## GPU scale-up / stretch
**S4:** LoRA SFT of Qwen2.5-0.5B-Instruct in bf16 on 8 GB (r = 16 on all linear layers, 1–2k examples). Evaluate before and after on a held-out set, with confidence intervals ([lab 15](../15_eval_stats/README.md)).
