# 02: AI product engineering

Build the product around an evaluation suite, start with the simplest architecture a frontier API allows, and climb to fine-tuning or your own models only when measured evidence demands it.
This file covers the eval-driven loop, a build vs buy vs train framework with a worked example, reliability design (including how to set a human-review threshold), document-AI specifics, and data and privacy obligations.

**Contents:** [Eval-driven loop](#the-eval-driven-loop) · [Architecture ladder](#climb-the-architecture-ladder-only-when-the-evals-demand-it) · [Build vs buy vs train](#build-vs-buy-vs-train) · [Reliability](#reliability-for-probabilistic-systems) · [Document AI notes](#document-ai-notes) · [Data, privacy, security](#data-privacy-security) · [Model upgrades](#model-upgrades) · [Failure modes](#failure-modes) · [Further reading](#further-reading)

## The eval-driven loop

1. **Collect 100–300 real examples** of the task, with the customer's written consent. Write down what "good" means per field or per output, in the customer's words.
2. **Build the dumbest baseline:** one prompt to a strong frontier model, structured output, no retrieval.
3. **Read the outputs yourself.** Cluster the failures into a taxonomy and count them. This step is the one teams skip, and it is the one that tells you what to build.
4. **Write graders per failure mode:** code checks first (schema validity, exact-match fields, arithmetic consistency, citation present), model-based judges only where code cannot judge, and calibrate every judge against your own labels before trusting it.
5. **Change one thing at a time** (prompt, retrieval, tool, model), rerun the suite, and keep the change only if it wins by more than the noise. See [module 08](../../curriculum/08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen) and [lab 15](../../labs/15_eval_stats/README.md) for paired tests and confidence intervals.
6. **Run the suite in CI** on every change and on every new model release.

Starter failure taxonomy for a document-extraction product (adapt to yours):

| failure | example | grader |
|---|---|---|
| missing field | invoice date left empty | code: required-field check |
| wrong value | total off by a digit | code: exact match against label |
| inconsistent values | line items don't sum to total | code: arithmetic check |
| hallucinated value | a GSTIN that isn't on the page | code: value must appear in OCR text (fuzzy) |
| wrong format | date as 3/4/25 when the spec says ISO | code: regex / parser |
| wrong page or document type | a delivery note treated as an invoice | code: classifier label vs gold |
| unreadable input | blurred photo | code: abstain flag expected |

> [!TIP]
> Keep a "golden 50": the fifty hardest real examples you have seen, each with a note on why it is hard. Run it on every change in under two minutes. It catches most regressions before the full suite does.

## Climb the architecture ladder only when the evals demand it

prompt → structured output with validation → retrieval (RAG) → tools or an agent loop → fine-tuning (for format, cost or latency) → your own model (privacy or on-prem requirements, cost at scale, or a capability you cannot buy, such as a niche script or domain OCR).

Each rung adds cost, latency and failure surface. Move up only when a specific failure cluster from step 3 cannot be fixed on the current rung. [Module 10](../../curriculum/10-applied-llm-systems.md) covers RAG, agents and structured outputs in depth.

## Build vs buy vs train

Three options for each capability in your product (OCR, extraction, classification, generation):

| option | what it means | choose when | watch out for |
|---|---|---|---|
| **Buy** | call a frontier or specialist API | default for anything new; volume is low or unknown; quality leader matters | data-residency and consent terms; price changes; provider outages; vendor lock-in (keep a model-agnostic interface) |
| **Build on open weights** | run an open model, prompted or lightly fine-tuned, on your infra or the customer's | privacy or on-prem is a hard requirement; volume is steady and high; a small model meets the eval bar | GPU utilization (idle GPUs are the cost), MLOps burden, security patching, licenses (read each model's license) |
| **Train** | train or substantially fine-tune a model for your task | no available model meets the bar (niche script, domain OCR, low-resource language); you have enough labeled data from your flywheel | data collection time, eval rigor, the risk that the next general model overtakes your specialist model |

**Decision sequence:**
1. Does any API meet the quality bar on your eval suite? If no, can a fine-tuned open model? If neither, you have a training problem, and a potential moat.
2. Is there a hard constraint (on-prem, residency, air-gap) that rules out APIs? If yes, build on open weights.
3. At your *measured* monthly volume, is the self-hosted cost per unit lower than the API cost per unit, including engineering time? Use the formula in [03](03-unit-economics.md#levers-roughly-in-order-of-impact).
4. If the answers are "yes, no, no": buy, and revisit every quarter.

### Worked example (hypothetical numbers)

A bank-statement extraction product, 100,000 pages per month, eval metric = field-level exact match on a 300-document test set.

| option | field accuracy (hypothetical) | cost per 1,000 pages (hypothetical) | fixed monthly cost (hypothetical) | notes |
|---|---|---|---|---|
| Frontier API, prompt only | 96.5% ± 0.8 | ₹900 | ₹0 | fastest to ship |
| Open 7B VLM, fine-tuned on 2,000 corrected pages, self-hosted on one rented GPU | 96.0% ± 0.9 | ₹150 marginal | ₹60,000 (GPU rental plus ops time) | needed if a customer demands on-prem |
| Classic OCR + small layout model + rules | 91.0% ± 1.2 | ₹60 | ₹40,000 | cheapest, brittle on new formats |

Arithmetic at 100,000 pages per month: the API costs 100 × ₹900 = ₹90,000; self-hosting costs ₹60,000 + 100 × ₹150 = ₹75,000. The self-hosted option is cheaper only above the break-even volume, where ₹900 × *V* = ₹60,000 + ₹150 × *V*, so *V* = 80 thousand-page units, or **80,000 pages per month**. Below that, buy. The accuracy difference between the first two rows (0.5 points) is inside the error bars, so accuracy does not decide it; volume, privacy and engineering time do.

> [!NOTE]
> The ± values are illustrative 95% confidence intervals. With 300 documents and several fields each, compute the interval with a clustered standard error by document, because fields on the same page are correlated ([lab 15](../../labs/15_eval_stats/README.md)).

## Reliability for probabilistic systems

**Compounding.** Per-step success multiplies: 20 steps at 95% each succeed end-to-end about 0.95²⁰ ≈ 36% of the time. Minimize steps, verify each step programmatically, checkpoint, and make retries idempotent.

**Abstain and escalate.** Estimate confidence per field or per document and route low-confidence cases to a human. In document workflows a review queue is a feature customers pay for, not an embarrassment.

**Choosing the review threshold.** Plot accuracy of auto-approved items against coverage (the share auto-approved) as you move the confidence cut-off. Pick the cut-off where auto-approved accuracy meets the customer's written requirement.

| confidence cut-off (hypothetical) | share auto-approved | accuracy on auto-approved fields | share sent to review |
|---|---|---|---|
| 0.50 | 97% | 97.1% | 3% |
| 0.80 | 88% | 99.0% | 12% |
| 0.90 | 80% | 99.5% | 20% |
| 0.97 | 60% | 99.8% | 40% |

If the pilot's success criterion is "99.5% accuracy on auto-approved fields", the cut-off is 0.90 and 20% of documents go to review. That 20% is a direct cost line in [03](03-unit-economics.md#worked-example-document-extraction). Confidence scores must be **calibrated** on held-out data; raw model log-probabilities or self-reported confidence are often miscalibrated, so measure before trusting them.

**Make errors cheap to catch.** Side-by-side review UI (source image next to extracted fields), keyboard shortcuts, highlighted low-confidence fields, and every correction saved as training data ([04](04-moats-and-flywheels.md#designing-a-data-flywheel)).

## Document AI notes

For a founder coming from production OCR, three architectures cover most document products:

| pipeline | strengths | weaknesses |
|---|---|---|
| OCR engine → text → LLM extraction | cheap, auditable (every value traces to OCR text), works with text-only models | OCR errors propagate; layout (tables, checkboxes) is lost unless you keep coordinates |
| Vision-language model directly on page images | handles layout, handwriting and stamps well; fewer components | higher cost per page; harder to prove where a value came from; needs a hallucination check |
| Specialist models (detection, recognition, layout) + rules | fastest and cheapest at high volume; can run on CPU or edge | brittle to new formats; engineering-heavy |

Practical points: keep bounding boxes for every extracted value so the review UI can highlight the source; check values against the OCR text to catch hallucinations; handle rotated, multi-page and mixed-language documents explicitly; and quantize specialist models for cost (see [lab 13](../../labs/13_quantization/README.md)).

## Data, privacy, security

- **Log with consent.** Inputs, outputs and user corrections are your future training data. Put data-use rights in the contract from the first pilot, including whether you may use the customer's data to improve models for other customers.
- **Tenant isolation** in storage and retrieval: one customer's documents must never appear in another's context window.
- **Treat all uploaded or retrieved text as untrusted.** A document can contain instructions aimed at your model (prompt injection). Never let extracted text trigger actions without validation.
- **India's data-protection law.** The Digital Personal Data Protection Act, 2023 and the DPDP Rules notified in November 2025 phase in obligations through 2027. Most document-AI startups act as a *data processor* for customers who are the *data fiduciaries*, which shapes your contracts and security duties. Details and dates: [06 § Data protection](06-fundraising-and-structure.md#data-protection-dpdp-act-2023). Not legal advice; confirm with a lawyer.
- **Deployment as product.** On-prem or customer-VPC deployment can be what the customer is buying. Price it accordingly.

## Model upgrades

Keep a model-agnostic interface and a fixed eval suite. When a new model ships, run the suite and compare quality, cost and latency with confidence intervals; switch within a day if it wins. A team that can adopt a better model in a day compounds that advantage with every release.

## Failure modes

- **No eval set until a customer complains.** Build it in week one from real data.
- **Judging by demo.** A great demo on five cherry-picked documents says nothing about the 95th percentile document.
- **Uncalibrated judges.** A model grader that agrees with you 70% of the time is noise with a confident tone. Measure agreement against your labels first.
- **Premature agents.** Autonomy multiplies failure points. Use fixed workflows until the evals show that the task needs dynamic decisions.
- **Training too early.** A fine-tuned model that took a month to build can be overtaken by the next general model release. Train when evals and volume justify it, and keep the API path alive as a baseline.

## Further reading

- Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/index.html) (2024).
- [What We've Learned From A Year of Building with LLMs](https://applied-llms.org/) (2024).
- Schluntz and Zhang, [Building effective agents](https://www.anthropic.com/research/building-effective-agents) (Anthropic, 2024).
- In this repo: [module 08](../../curriculum/08-evaluation-and-research.md) (evaluation), [module 10](../../curriculum/10-applied-llm-systems.md) (applied systems), [lab 17](../../labs/17_retrieval/README.md) (retrieval).
