# Lab 10 — LoRA

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 10_lora`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [post-training §2](../../curriculum/06-post-training.md#2-parameter-efficient-fine-tuning)

## Implement and test
- `LoRALinear(base, r, alpha)`: frozen base, `A` Kaiming-initialized, `B = 0`, scale α/r. Test: output equals the base at init; only A and B get gradients.
- `merge()` returns an `nn.Linear` with `W + (α/r)·BA`. Test: merged output equals the LoRA output.
- `apply_lora(model, targets)` replaces named `nn.Linear` layers in place.
- **The low-rank test:** a target equal to the base plus a rank-2 delta. r = 2 fits it to near-zero loss; r = 1 plateaus. That is the hypothesis behind LoRA, demonstrated.

## GPU scale-up / stretch
**S4:** LoRA SFT of Qwen2.5-0.5B-Instruct in bf16 on 8 GB (r = 16 on all linear layers, 1–2k examples). Evaluate before and after on a held-out set, with confidence intervals ([lab 15](../15_eval_stats/README.md)).
