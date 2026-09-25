# 02 — AI product engineering

## The eval-driven loop
1. Collect 50–200 **real** examples of the task (with consent); write down what "good" means.
2. Build the dumbest baseline: one prompt to a strong frontier model.
3. **Read the outputs.** Cluster the failures into a taxonomy; count them.
4. Write graders per failure mode: code checks first (schema, exact fields, citations), model judges only when needed, calibrated against your own labels.
5. Change one thing (prompt, retrieval, tools, model), rerun the evals, keep it only if it wins beyond noise ([module 08 §2](../../curriculum/08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen)).
6. Run the evals in CI on every change and on every new model release.

## Climb the architecture ladder only when the evals demand it
prompt → structured output plus validation → RAG → tools / agent loop → fine-tuning (format, cost, latency) → your own model (privacy/on-prem, cost at scale, capabilities you can't buy, e.g. niche Indic OCR).

## Reliability for probabilistic systems
Per-step success compounds (0.95²⁰ ≈ 36%): minimize steps, verify each step programmatically, checkpoint, retry idempotently. Estimate confidence and **abstain or escalate to a human** when it's low; in document workflows a human-review queue is a feature. Show sources. Make errors cheap to spot and fix (side-by-side review UIs).

## Data, privacy, security
Log inputs, outputs and user corrections with consent; corrections are your future training data. Keep tenant isolation in the retrieval layer. Treat all retrieved or uploaded text as untrusted (prompt injection). Know India's DPDP Act obligations for personal data (check the current rules with a lawyer) and your customers' data-residency needs; on-prem or VPC deployment can be the product itself.

## Model upgrades
Keep a model-agnostic interface and a fixed eval suite. When a new model ships, run the suite, compare quality, cost and latency with CIs, and switch in hours. That speed advantage compounds.
