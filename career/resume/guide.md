# Résumé guide

How to make the résumé survive a technical deep dive: fix the open issues first, rewrite bullets with a formula that forces evidence, and check the result against a list built for core-AI roles.
Source: [resume.tex](resume.tex) (Jake Gutierrez / sb2nov template, compile on Overleaf) · current PDF: [LourduRaju_Resume.pdf](LourduRaju_Resume.pdf).

**Contents:** [Open issues](#open-issues-verify-before-sending-anything) · [Bullet formula](#bullet-formula) · [Before and after](#before-and-after-rewrites) · [Checklist for core-AI roles](#checklist-for-core-ai-roles) · [Section order](#section-order-by-target) · [Rules and versioning](#rules-and-versioning)

## Open issues: verify before sending anything

> [!WARNING]
> Several numbers below appeared in earlier drafts (an old story bank and feedback notes) that were written from memory, not from your records. Until you have checked each against a source you could show an interviewer (an experiment log, a report, a dashboard, an email), treat it as **unverified** and leave it off.

| # | claim (where it appears) | the problem | what you need before keeping it |
|---|---|---|---|
| 1 | OCR exact-match **82.74% → 97.04%** (Sujanix bullet) vs **89% → 97%**, a "1.2M-image" set and "~200 req/s" (old story bank, feedback) | the numbers disagree with each other | one consistent pair of numbers; the test-set size, how it was split from training data, what "exact match" means (whole string? per digit?), and who measured it |
| 2 | **SVRT** ("a Swin-V2-based regression transformer", Sujanix bullet) vs **SVTR backbone** (Transformers-OCR project) | SVTR is a published scene-text architecture (Du et al., IJCAI 2022); "SVRT" reads like a typo or a different model | intentional, consistent naming, and a 2-minute explanation of the architecture and why you chose it |
| 3 | inference cost **−30%** at **99.9% uptime** (Sujanix bullet) | a percentage needs a baseline; uptime needs a measurement window | the before/after cost per request or per month, and where the uptime figure came from (monitoring tool, period) |
| 4 | assessment time **−70% (30 min → 9 min)** (ECHOME) | a time saving needs users and a method | how many sessions or users, measured how, compared with what |
| 5 | forecasting accuracy **+15%** (internship bullet) | "accuracy" for forecasting is ambiguous; relative or absolute? | the metric (MAPE, RMSE, …), the baseline model, the evaluation period |
| 6 | skills line: "strong CS foundations: DSA, ML system design, distributed systems", Kubernetes, SageMaker, Vertex AI, vLLM, C++, gRPC | every keyword is an interview question | keep only what you can go three "why"s deep on; move the rest out or into "exposure to" |
| 7 | experience length | "2+" overstates it | say "about 2 years" (internship, founder and Sujanix together) |
| 8 | no from-scratch or research work listed | core-AI screeners look for it first | add a **Selected technical work** section as labs and experiments land (examples below) |

## Bullet formula

**Verb + what you built + how (the technical decision) + measured result with context (baseline, dataset, error bar where it applies).**

A number without its denominator invites the question "out of what?". Give the test-set size, the baseline and, for model quality, an interval or at least the variance across seeds.

## Before and after rewrites

Generic examples to show the pattern. **All numbers in the "after" column are illustrative;** use only your own measured numbers.

| before | after (illustrative numbers) | what changed |
|---|---|---|
| Worked on LLM fine-tuning for a chatbot. | Fine-tuned a 1B-parameter model with LoRA (rank 16) on 12k support transcripts; task success on a 400-conversation held-out set rose from 61% to 74% (95% CI ±4 pts), scored by a rubric grader checked against 200 human labels. | method, data size, baseline, held-out result with an interval, how it was graded |
| Responsible for the RAG pipeline. | Rebuilt retrieval for an internal-docs assistant as BM25 + dense hybrid with reciprocal-rank fusion; recall@10 on 300 labeled queries rose from 0.62 to 0.81 with p95 latency under 800 ms. | ownership shown by a decision, a retrieval metric, a latency constraint |
| Optimized model inference. | Cut GPU cost per request 38% by moving a 7B model to INT4 weight-only quantization and continuous batching; product-eval accuracy changed −0.4 pts, within its ±0.6 CI. | the levers, the cost metric, and the quality check that makes the saving credible |
| Implemented a transformer from scratch. | Implemented a Llama-style decoder (RoPE, GQA, SwiGLU) and a Triton FlashAttention forward kernel; matches PyTorch SDPA to 1e-3 and reaches N% of its throughput at 2k context on an 8 GB laptop GPU. Code and benchmarks: link. | specific components, a correctness check, a measured comparison, a link |
| Achieved 99% accuracy on OCR. | Raised exact-match accuracy on a frozen 5,000-image held-out set of meter photos from X% to Y% by replacing a CRNN with a transformer recognizer and a class-balanced CTC loss. | the test set is named and frozen; the change that caused the gain is stated |
| Managed a team to deliver client projects. | Scoped, priced and delivered N fixed-price ML projects; brought in 3 contractors under written scopes when volume exceeded one person; <sourced outcome, e.g. repeat clients>. | the decisions you owned, a count, a sourced outcome |
| Proficient in PyTorch, TensorFlow, JAX, Kubernetes, Spark, CUDA, … | PyTorch (daily; wrote custom autograd functions and Triton kernels), Hugging Face, ONNX Runtime; working knowledge of Docker and AWS Lambda. | fewer tools, each with depth you can defend |

> [!TIP]
> After writing each bullet, ask the three questions an interviewer will ask: *How did you measure that? What was the baseline? What would you do differently?* If you can't answer all three in under a minute each, rewrite or cut the bullet.

## Checklist for core-AI roles

Run it before every version you send.

**Content**
- [ ] The top third of the page shows the strongest core-AI evidence (from-scratch work, experiments, research-style write-ups), not the skills list.
- [ ] A **Selected technical work** section with 3–5 items, each linking to code or a write-up. Examples as labs land: "Implemented GRPO from scratch; on a 0.5B model improved accuracy on a verifiable task from X to Y (95% CI ±Z)"; "Triton FlashAttention kernel at N% of PyTorch SDPA throughput on an RTX 5060"; merged PRs with links.
- [ ] Every quality number has its dataset, its baseline and, where it applies, an interval or seed variance.
- [ ] Every number is in the verified column of the [open-issues table](#open-issues-verify-before-sending-anything).
- [ ] Each keyword survives three "why"s.
- [ ] Compute is stated where it matters ("on one 8 GB GPU"): it makes results credible and shows resourcefulness.
- [ ] Production experience is framed in terms core-AI teams care about: evaluation, data quality, latency and cost trade-offs, quantization, failure analysis.

**Form**
- [ ] One page, single column, standard section headings, text selectable in the PDF (ATS-safe; no text in images or tables).
- [ ] Links work (GitHub, write-ups, Hugging Face) and the linked repos have READMEs with results and one-command reproduction.
- [ ] Consistent tense, date format and name spelling (for example, the model name in issue 2).
- [ ] File named `Firstname_Lastname_Resume.pdf`; no photo, age or marital status.

**Tailoring**
- [ ] Ordered for the target team using its [company guide](../../companies/README.md): kernels and inference first for an inference team, post-training and evals first for a post-training team.
- [ ] The summary line matches the team, in one sentence, and says nothing you cannot link.

## Section order by target

| target | order |
|---|---|
| frontier-lab research engineer | Selected technical work → Experience → Projects → Skills → Education |
| applied ML / LLM product roles | Experience → Selected technical work → Projects → Skills → Education |
| founder-track credibility (investors, customers) | not a résumé: a one-paragraph bio plus links to shipped systems and write-ups |

## Rules and versioning

One page; PDF; single column. Version every change and log which version went where:

```text
version · date · change summary · sent to (company, role, date) · outcome
v2.1   · 2027-05-04 · added GRPO result, cut skills line · <company>, RE post-training, 2027-05-06 · screen
```

Re-read the log before each interview so you know which claims that interviewer has seen.
