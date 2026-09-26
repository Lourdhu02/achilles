# Roadmap: Oct 2026 → Sep 2027 (~25 h/week ≈ 1,250 h)

Milestone-based: **move on when the exit criteria are met, not when the calendar says so.** Adjust in the monthly review. Every week also includes **DSA 3 h** ([12-week plan](tracks/research-engineer/dsa-patterns.md#a-12-week-plan-for-people-with-full-time-jobs)), **one journal entry**, and, from stage 4, **one customer conversation** (founder track). Read each stage's topics in the [library](library/README.md) before its labs.

> [!NOTE]
> **No GPU?** Every exit criterion below can be met with a CPU laptop plus free Colab or Kaggle T4 sessions for the scale-up runs (S1–S5). [SETUP.md](SETUP.md#what-you-can-do-on-each-tier) lists the minimum hardware for each experiment.

| stage | when (target) | hours | do | exit criteria |
|---|---|---|---|---|
| 0 Setup | week 1 | 10 | SETUP; `measure_gpu.py`; self-rate modules 0–5; read [00](curriculum/00-learning-os.md) | GPU numbers in the journal; baseline ratings |
| 1 Foundations | Oct–Nov | 150 | [01](curriculum/01-math.md)–[03](curriculum/03-deep-learning.md); labs 01–03 | labs pass; derive the softmax-CE and LayerNorm backward on paper; napkin drills ≥ 80% |
| 2 Transformers | Nov–Dec | 150 | [04](curriculum/04-transformers.md); labs 04–07; **S1** pretraining run | your GPT trains on TinyStories; Triton kernel correct on GPU; **blog post 1** |
| 3 Scale | Jan 2027 | 100 | [05](curriculum/05-pretraining.md); labs 08–09; **S2** scaling sweep | your own fitted scaling law; a 7B run planned on paper and defended |
| 4 Post-training | Feb–Mar | 175 | [06](curriculum/06-post-training.md); labs 10–12; **S4/S5** | GRPO improves a verifiable task (with CIs); **write-up 2**; first OSS PR |
| 5 Inference | Mar–Apr | 125 | [07](curriculum/07-inference.md); labs 13–14; serving benchmark | measured tokens/s within 30% of your prediction; **write-up 3** |
| **Gate A** | end Mar | — | research engineer vs founder evidence ([founder README](tracks/founder/README.md#decision-gate-put-it-in-roadmapmd)) | decide the emphasis for the next 3 months against thresholds written down in advance |
| 6 Eval & research | Apr–May | 100 | [08](curriculum/08-evaluation-and-research.md); lab 15; pick a research project | project proposal plus first results with error bars |
| 7 Interp & safety | May | 75 | [09](curriculum/09-interpretability-and-safety.md); lab 16; values answers | SAE study; the values section of [stories](career/stories.md) written |
| 8 Applied & founder | Jun | 75 | [10](curriculum/10-applied-llm-systems.md); lab 17; founder 01–03; upgrade FinSentinelAI evals | measured RAG; 10+ problem interviews; 3 scored ideas |
| **Gate B** | end Jun | — | offers pipeline vs paid pilots or LOIs, against pre-registered thresholds | commit: apply broadly, found, or both in sequence |
| 9 Launch | Jul–Sep | 250 | research write-up (workshop-ready); 2–3 merged PRs; applications ([career](career/README.md)) **or** paid pilots | onsites or signed pilots |

## Gates: decide on evidence, not mood

Two weeks before each gate, write your thresholds into the journal and date them, using the decision sheet in the [founder track](tracks/founder/README.md#decision-gate-put-it-in-roadmapmd). For example: "founder if 2 paid pilots or 5 LOIs with a price; research engineer if 3 onsites or 1 offer". The numbers are yours to choose; what matters is that you choose them before you see the evidence. At the gate, compare the evidence with the thresholds and move your hours towards the stronger side.

## Research-engineer applications (from May)
Résumé v2 with a "Selected technical work" section → mocks every 2 weeks → waves: practice targets → core targets → frontier labs. Use [interview-loops](tracks/research-engineer/interview-loops.md), the [question bank](tracks/research-engineer/question-bank.md), and the [company guides](companies/README.md) for your two targets.

## If you fall behind
Cut breadth, never depth: skip stretch goals and spec-lab extras before skipping a core lab or a write-up. Two weeks behind: extend the stage. Six weeks behind: drop stage 7 to reading-only and keep 4, 5 and 6.
