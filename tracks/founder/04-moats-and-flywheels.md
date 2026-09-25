# 04: Moats and data flywheels

A moat is whatever makes you harder to replace next year than today. For AI applications it rarely comes from the model; it comes from workflow depth, accumulated data that measurably improves the product, evals and domain know-how, distribution and trust.
This file gives a test for whether something is a moat, how to design and measure a data flywheel, and when training your own model becomes a real advantage.

## Not moats

Prompts; a UI on top of a model; "we fine-tuned an open model"; first-mover advantage without distribution; a benchmark score; a waitlist.

## Real moats for AI applications

| moat | what it looks like | how to measure it |
|---|---|---|
| **Workflow depth and integration** | embedded in systems of record (ERP, core banking, billing), approvals, audit trails | integration count per customer; switching effort a customer would face |
| **Proprietary, accumulating data** | corrections, edge cases, domain labels no competitor can buy | eval score on your hardest cases, over time, with and without the new data |
| **Evals and domain know-how** | graders and test sets encoding what "correct" means in the domain | time to adopt a new model; regressions caught before customers see them |
| **Distribution and trust** | relationships in a vertical, compliance (on-prem, security reviews passed), references | sales-cycle length; share of deals from referrals |
| **Cost and latency advantages** | small models tuned on your data, running cheaper than a general API | cost per unit versus the best API option at equal quality |

## The absorption test

Ask two questions about every product decision.

1. **When the next frontier model ships, do we get stronger or redundant?** A product that routes, verifies, integrates and learns from corrections gets cheaper and better with each model release. A product whose value is "the model can now do X" gets absorbed when the model provider, or a general assistant, does X directly.
2. **If a well-funded competitor copied our product surface tomorrow, what would they still lack?** If the answer is "nothing", you have a feature. If it is "our integrations, our 50,000 corrected documents, our evals and our customers' trust", you have the beginning of a moat.

> [!TIP]
> Write your absorption-test answers into every quarterly plan. If an answer weakens two quarters in a row, move deeper into the workflow before a competitor forces you to.

## Designing a data flywheel

```text
user action ─▶ captured signal ─▶ labeling queue ─▶ model / prompt / retrieval update
   ▲           (accept, edit,     (low-confidence      │
   │            reject, correct)   and disagreement    ▼
   │                               cases first)     eval gate ─▶ deploy
   └────────────── better product ◀── more usage ◀──────┘
```

Design rules:
- **Capture corrections as structured data** (field, old value, new value, reviewer, time), not free text.
- **Label the informative cases first:** low confidence, model disagreement, new document formats (active learning).
- **Gate every update on the eval suite.** A flywheel without an eval gate can spin backwards.
- **Get explicit data-use consent in contracts from day one,** including whether corrections from one customer may improve the product for others.

**Measure the flywheel's velocity:** the time from a new failure mode appearing in production to a fix deployed. Hypothetical example of a flywheel that is working:

| month | documents processed | share sent to review | median days from new failure to fix |
|---|---|---|---|
| 1 | 40,000 | 22% | 21 |
| 3 | 110,000 | 16% | 9 |
| 6 | 260,000 | 11% | 4 |

If volume rises and the review share does not fall, the data is not improving the product, and the flywheel is only a diagram.

## When to train your own model

Train when (a) your evals show that a small tuned model matches or beats the API on your task, **and** at least one of: (b) volume makes the cost difference material ([03](03-unit-economics.md#levers-roughly-in-order-of-impact)), (c) privacy or on-prem deployment requires it, or (d) no available model does the job (niche scripts, domain OCR, speech in low-resource languages). The full decision sequence is in [02 § Build vs buy vs train](02-ai-product-engineering.md#build-vs-buy-vs-train).

The core-AI skills from this repo (fine-tuning in [lab 10](../../labs/10_lora/README.md), quantization in [lab 13](../../labs/13_quantization/README.md), evaluation in [lab 15](../../labs/15_eval_stats/README.md)) make this a real option rather than a pitch-deck line.

## Failure modes

- **Claiming a data moat you cannot measure.** If you cannot show an eval curve that improves with your data, investors and competitors will assume there is none.
- **Collecting data without consent terms.** Unusable data, and a legal risk.
- **A specialist model overtaken by a general one.** Re-run the API baseline on every major release; keep the cheaper path if it wins.
- **Integration depth without product depth.** Custom integrations for each customer are services work unless they become reusable connectors.
