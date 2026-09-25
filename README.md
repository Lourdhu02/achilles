# Core AI, from first principles

A training ground for becoming a **core AI engineer**, someone who can build, train, post-train, serve and evaluate large models from scratch, and who can use that depth to **land a frontier-lab role or build a company**.

This replaces an earlier interview-prep playbook. That version was 17,000 lines of reading and zero lines of code, and it contained numerical errors in exactly the places interviews test. This version is built around one belief: **you understand what you can build, measure and explain, and nothing else.**

## What's here
| | |
|---|---|
| [`curriculum/`](curriculum/README.md) | 11 modules, math → frontier: derivations, napkin math, traps, papers |
| [`labs/`](labs/README.md) | 17 from-scratch labs with automated tests: autograd → GPT → FlashAttention/Triton → KV cache → LoRA/DPO/GRPO → quantization → speculative decoding → evals → interpretability |
| [`tracks/research-engineer/`](tracks/research-engineer/README.md) | frontier-lab teams, interview loops, question bank, system design, portfolio |
| [`tracks/founder/`](tracks/founder/README.md) | problem selection, eval-driven product, unit economics, moats, GTM, fundraising |
| [`career/`](career/README.md) | résumé, stories, outreach, negotiation, visa: condensed and honest |
| [`journal/`](journal/README.md) | your experiments, paper notes and reviews: the proof of work |
| [`ROADMAP.md`](ROADMAP.md) | milestone-based plan, Oct 2026 → Sep 2027, with decision gates |
| [`SETUP.md`](SETUP.md) | environment for the RTX 5060 (Blackwell) and CPU |

## Start (today)
```bash
uv sync                                   # or: pip install numpy regex torch pytest  (see SETUP.md for CUDA)
python tools/measure_gpu.py               # measure your GPU's roofline
pytest labs/01_autograd                   # 37 failing tests: your first job is to make them pass
python tools/progress.py                  # scoreboard across all labs
```

## How it works
Each lab has a handout (`README.md`), a stub you implement (`exercise.py`), tests, and a reference (`solution.py`), which you read only after an honest attempt. Labs 01–07 ship complete; labs 08–17 are **spec labs**, where writing the tests is part of the exercise. CI verifies every reference solution on CPU.

The loop, every week: **Read → Derive → Build → Measure → Write** ([how to learn](curriculum/00-learning-os.md)).

## The one rule
Predict the number before you measure it. The gap between your prediction and reality is where you learn.
