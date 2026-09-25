# Résumé guide

Source: [resume.tex](resume.tex) (Jake Gutierrez / sb2nov template, compile on Overleaf) · current PDF: [LourduRaju_Resume.pdf](LourduRaju_Resume.pdf).

## Open issues: fix before sending anything
1. **Inconsistent numbers.** The résumé says OCR exact-match went from **82.74% → 97.04%**; the old story bank and feedback said **89% → 97%** and mentioned a "1.2M-image" set and "~200 req/s". Several of those details were drafted by an assistant, not taken from your records. **Keep only numbers you can source**, and know the test-set size, the split and the measurement method.
2. **SVRT vs SVTR.** The Sujanix bullet says *SVRT (a Swin-V2-based regression transformer)*; the Transformers-OCR project says *SVTR backbone*. SVTR is a published scene-text architecture (Du et al., IJCAI 2022). Make the naming intentional and consistent, and be ready to explain the difference.
3. **Claims you would have to defend:** "strong CS foundations: DSA, ML system design, distributed systems", Kubernetes, SageMaker, Vertex AI, vLLM, C++, gRPC. Every keyword is an interview question. Keep only what you can go three "why"s deep on.
4. **Experience length:** say "~2 years" (internship plus founder plus Sujanix), not "2+".
5. **Add a "Selected technical work" section** as the labs and experiments land: e.g. "Implemented GRPO from scratch; on Qwen2.5-0.5B improved task accuracy X→Y (95% CI ±Z)"; "Triton FlashAttention kernel at N% of cuDNN throughput on RTX 5060"; merged PRs with links.

## Bullet formula
**Verb + what you built + how (the technical decision) + measured result with context.** Example: "Reduced OCR inference cost 30% by deploying INT8 ONNX on AWS Lambda; accuracy change −0.3 pts on a 5k-image held-out set." (Use your real numbers.)

## Rules
One page. PDF, single column, ATS-safe. Tailor the ordering per role (core-AI roles: lead with from-scratch and research work; applied roles: lead with shipped systems). Version every change and log which version went to which company.
