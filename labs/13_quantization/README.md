# Lab 13 — Quantization

> **Status: spec lab.** Labs 01–07 ship with reference solutions and tests. For this lab, you write both,
> following the same pattern: `solution.py` with `# BEGIN SOLUTION` / `# END SOLUTION` markers,
> `test_*.py` that imports it with `load(__file__)`, then `python tools/make_exercises.py 13_quantization`.
> Writing the tests yourself is part of the training: every test below states a property you must understand.

**Reads first:** [inference §4](../../curriculum/07-inference.md#4-quantization)

## Implement and test
- Symmetric absmax INT8 per-tensor and per-channel; asymmetric with a zero point; group-wise INT4 (group 64); `pack_int4` / `unpack_int4`.
- `nf4_codebook()` built from normal quantiles with `torch.special.ndtri`. Test against QLoRA's published 16 values.
- Error tests: per-channel beats per-tensor when a channel has outliers; NF4 beats uniform INT4 on Gaussian weights.
- `smoothquant(W, act_absmax, α)`: test `(X/s)(sW) = XW` with smaller activation outliers.
- `gptq_quantize(W, X)`: column-by-column quantization with Hessian-inverse error feedback. Test: lower output error than round-to-nearest on correlated inputs.

## GPU scale-up / stretch
Quantize your lab-05 GPT and Qwen2.5-0.5B to INT8 and INT4 and measure perplexity against decode tokens/s on the RTX 5060. Compare with your lab-03 bandwidth prediction.
