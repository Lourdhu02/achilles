# 04 — Moats and data flywheels

## Not moats
Prompts, a UI on top of a model, "we fine-tuned Llama", first-mover advantage without distribution, a benchmark score.

## Real moats for AI applications
- **Workflow depth and integration:** embedded in systems of record, approvals and audit trails. Switching costs are real.
- **Proprietary, accumulating data**, *if* it measurably improves the product: corrections, edge cases, domain labels. Prove it with evals over time.
- **Evals and domain know-how encoded as graders and test sets.** Hard to copy, and they let you adopt new models fastest.
- **Distribution and trust:** relationships in a vertical, compliance (on-prem, certifications), brand in a community.
- **Cost and latency advantages** from your own small models on your own data, where volume justifies them.

## Designing a data flywheel
User action → captured signal (accept, edit, reject, correction) → labeling queue (active learning: label the low-confidence and disagreement cases first) → training or prompt and retrieval updates → eval gate → deploy → better product → more usage. Measure the flywheel's velocity: time from a new failure mode to a fix in production. Get explicit data-use consent in contracts from day one.

## When to train your own model
When (a) the evals show a small tuned model matches the API on your task, and (b) volume makes the cost difference material, or privacy/on-prem requires it, or no available model does the job (niche scripts, domain OCR, speech in low-resource languages). Your core-AI skills from the curriculum make this a real option rather than a pitch-deck line.
