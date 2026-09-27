# Track: research engineer at a frontier lab

This track turns the labs and curriculum into offers for core-ML roles: research engineer or Member of Technical Staff on a pretraining, post-training, inference, evals, interpretability or applied team at an AI lab, or on the equivalent core-ML teams at Google DeepMind, Meta, NVIDIA, Microsoft AI and similar. It covers what each team does, the evidence that gets you hired, three preparation plans, and a weekly schedule that fits around a full-time job.

## Contents

- [Who this track is for](#who-this-track-is-for)
- [The teams: what they do and what proves you can do it](#the-teams-what-they-do-and-what-proves-you-can-do-it)
- [Signal ladder (climb it in order)](#signal-ladder-climb-it-in-order)
- [Preparation plans](#preparation-plans)
- [A weekly schedule with a full-time job](#a-weekly-schedule-with-a-full-time-job)
- [Ready to apply when](#ready-to-apply-when)
- [Files in this track](#files-in-this-track)

## Who this track is for

- **ML or software engineers with one to five years of experience** who use models through APIs or fine-tune them, and want to work below the API: training, RL, kernels, evals. The plans assume about two years of experience and a laptop GPU with 8 GB of memory; adjust from there.
- **Researchers** (MS, PhD or self-taught) with ideas or papers who need engineering evidence: fast, correct code and a feel for systems.
- **CPU-only learners.** Every lab runs on a CPU. GPU scale-ups can wait, shrink, or move to a free notebook GPU; the interview preparation is identical.
- **Candidates outside the US.** Almost everything here is location-independent. Time-zone logistics are in [interview-loops.md](interview-loops.md#logistics-interviewing-across-time-zones-from-india), and each company guide has a "From India" section.

The job in one line: turn research ideas into correct, measured experiments and systems at scale. You write the training, inference and eval code, run the experiments, debug them, and report what the numbers mean. Where the line sits between research engineer and research scientist differs by company.

> [!NOTE]
> Titles vary (as of September 2026). "Research Engineer", "Member of Technical Staff", "Software Engineer, ML" and "Research Scientist" can describe overlapping work, and one title can mean different work at different companies. Read the job description and the team, not the title.

If you target research scientist roles that screen on publications, the track still applies, but rung 5 of the ladder has to become papers. If you want product engineering on top of models, start with the applied team below and [module 10](../../curriculum/10-applied-llm-systems.md).

## The teams: what they do and what proves you can do it

As of September 2026, six kinds of core team exist in some form at most frontier labs. Names, boundaries and which teams are hiring change often; each company guide maps its own: [Anthropic](../../companies/anthropic/README.md#where-core-ai-people-work), [OpenAI](../../companies/openai/README.md#where-core-ai-people-work), [Google](../../companies/google/README.md#where-core-ai-people-work), [Meta](../../companies/meta/README.md#where-core-ai-people-work), [NVIDIA](../../companies/nvidia/README.md#where-core-ai-people-work), [more labs](../../companies/more-labs.md).

### Pretraining and data

- **Day to day:** run small ablations (data mixtures, filtering thresholds, architecture and optimizer changes) and judge which will hold at scale; fit scaling laws to choose model size and token budget; keep large runs healthy (throughput, loss spikes, restarts, failing hardware); build pipelines for text extraction, deduplication, quality filtering and decontamination.
- **Interviews lean on:** napkin math ($6ND$, bytes per parameter, communication volume), parallelism strategies, training stability, data quality.
- **What proves it:** your own IsoFLOP sweep with a fitted exponent and bootstrap CIs; a data-filtering ablation measured in bits per byte; a 7B run planned on paper and defended ([module 05 §8](../../curriculum/05-pretraining.md#8-exercise-plan-a-7b-run-on-paper)).
- **Prepare with:** labs [04](../../labs/04_tokenizer/README.md), [05](../../labs/05_transformer/README.md) (with `train.py`), [08](../../labs/08_scaling_laws/README.md), [09](../../labs/09_parallelism/README.md) and [18](../../labs/18_moe/README.md) (mixture of experts); modules [02](../../curriculum/02-compute-and-hardware.md) and [05](../../curriculum/05-pretraining.md); designs [C](ml-system-design.md#c-pretraining-run-7b-on-2t-tokens) and [D](ml-system-design.md#d-pretraining-data-pipeline); question bank [§5](question-bank.md#5-pretraining-data-and-scaling) and [§6](question-bank.md#6-distributed-training).

### Post-training and RL

- **Day to day:** build SFT and preference data; train reward models and write verifiers; run preference optimization and RL with verifiable rewards; build rollout and trainer infrastructure; watch for reward hacking, length drift, entropy collapse and regressions on other skills.
- **Interviews lean on:** the policy-gradient derivation, KL control, deriving the DPO loss from the RLHF objective, verifier design, RL infrastructure.
- **What proves it:** GRPO improving a verifiable task on a held-out set, with CIs and an honest note on what got worse ([module 06 §8](../../curriculum/06-post-training.md#8-an-8-gb-practical-path)); a [GRPO ablation study](../../curriculum/08-evaluation-and-research.md#3-grpo-ablations-on-small-models); a merged TRL PR.
- **Prepare with:** labs [10](../../labs/10_lora/README.md), [11](../../labs/11_dpo/README.md) and [12](../../labs/12_grpo/README.md); [module 06](../../curriculum/06-post-training.md); design [E](ml-system-design.md#e-rl-post-training-infrastructure); question bank [§7](question-bank.md#7-post-training-and-rl).

### Inference and performance

- **Day to day:** profile and speed up serving: kernels in Triton or CUDA, batching and scheduling, KV-cache memory, quantization, speculative decoding; capacity planning and cost per token; proving that quality did not move when the numerics changed.
- **Interviews lean on:** roofline reasoning, prefill vs decode, memory-bandwidth math, tiling and online softmax, capacity planning.
- **What proves it:** a Triton kernel benchmarked against PyTorch SDPA, with a roofline explanation of the gap; decode tokens/s predicted and then measured within 30% (the roadmap's stage 5 exit criterion); a [KV-cache quantization study](../../curriculum/08-evaluation-and-research.md#4-kv-cache-quantization-quality-vs-speed-on-consumer-blackwell); a merged vLLM, SGLang or llama.cpp PR.
- **Prepare with:** labs [06](../../labs/06_attention_kernels/README.md), [07](../../labs/07_kv_cache_sampling/README.md), [13](../../labs/13_quantization/README.md) and [14](../../labs/14_speculative_decoding/README.md); modules [02](../../curriculum/02-compute-and-hardware.md) and [07](../../curriculum/07-inference.md); design [A](ml-system-design.md#a-llm-serving-platform); question bank [§8](question-bank.md#8-inference-and-serving) and [§10](question-bank.md#10-gpu-kernels-and-performance).

### Evals

- **Day to day:** design benchmarks and graders; build harnesses that score many checkpoints reliably; decide whether a change is real (CIs, paired tests, contamination checks); run capability and safety evals before releases.
- **Interviews lean on:** the bootstrap and paired tests, clustered standard errors, metric validity, contamination, the pitfalls of model graders, eval platform design.
- **What proves it:** a [public leaderboard re-analysed](../../curriculum/08-evaluation-and-research.md#7-error-bars-on-a-public-leaderboard) with paired CIs; a task added to lm-evaluation-harness; per-item results published so anyone can recompute your intervals.
- **Prepare with:** [lab 15](../../labs/15_eval_stats/README.md); [module 08](../../curriculum/08-evaluation-and-research.md); design [G](ml-system-design.md#g-eval-platform); question bank [§11](question-bank.md#11-evals-and-statistics).

### Interpretability and safety

- **Day to day:** train sparse autoencoders and study their features; circuit analysis, probing and steering; build safety evals and red-team models; study failure modes such as reward hacking; write carefully about what the evidence does and does not show.
- **Interviews lean on:** the methods and their limits, concrete alignment failure modes, and your own considered position on safety.
- **What proves it:** an [SAE study](../../curriculum/08-evaluation-and-research.md#6-sae-features-of-a-tinystories-model) on your own model; a written safety position ([module 09 §5](../../curriculum/09-interpretability-and-safety.md#5-your-position)); a small-scale replication of a published interpretability result.
- **Prepare with:** [lab 16](../../labs/16_interpretability/README.md); [module 09](../../curriculum/09-interpretability-and-safety.md); question bank [§12](question-bank.md#12-interpretability-and-safety); [values prep](../../career/stories.md#values-and-mission-prep).

### Applied and forward-deployed

- **Day to day:** build systems on frontier models with customers: RAG, agents and tool use, evals for the customer's task, latency and cost; turn a vague request into a measurable eval; carry what customers run into back to research teams.
- **Interviews lean on:** practical coding, system design for RAG and agents, evals, clear communication with non-specialists.
- **What proves it:** a RAG or agent system with a held-out eval set and improvements measured with CIs; your own projects upgraded with evals ([module 10 §6](../../curriculum/10-applied-llm-systems.md#6-upgrading-your-own-projects)).
- **Prepare with:** [lab 17](../../labs/17_retrieval/README.md); [module 10](../../curriculum/10-applied-llm-systems.md); designs [B](ml-system-design.md#b-enterprise-rag-with-access-control-and-citations) and [F](ml-system-design.md#f-agent-platform-with-tool-sandboxing-and-evals); question bank [§13](question-bank.md#13-applied-llm-systems).

> [!TIP]
> Pick one primary team and one adjacent team, and let the artifacts overlap. A decode-speed study serves inference and pretraining (throughput is a pretraining problem too); an eval study with honest statistics serves every team.

## Signal ladder (climb it in order)

Each rung is a kind of evidence. Interviewers probe the highest rung you claim, and a rung counts only when you can link to it. Times assume the roadmap's ~25 h/week and are planning estimates.

| rung | what it means | concrete example | time it takes |
|---|---|---|---|
| 1. Uses APIs | you compose hosted models and libraries | a chatbot over PDFs with a hosted model and a vector database | days to weeks |
| 2. **Built it from scratch** | you implemented the components and the tests pass | your BPE tokenizer and GPT (RoPE, GQA, SwiGLU) trained on TinyStories; FlashAttention in Triton; the DPO and GRPO losses | labs 01–14 are ~700 h (roadmap stages 1–5, October to April) |
| 3. **Measured it and predicted the numbers** | you wrote a number down before running, then explained the gap | "decode at batch 1 is memory-bound, so tokens/s ≈ bandwidth ÷ weight bytes", then measured within 30%; your own fitted scaling law | no extra months if you log a prediction before every measurement ([prediction log](../../curriculum/00-learning-os.md#5-the-prediction-log)) |
| 4. **Reproduced a paper** | you re-ran a published result at small scale and accounted for the differences | DPO's reward–KL trade-off on a 0.5B model; Chinchilla-style IsoFLOP curves at 1M–30M parameters | 3–6 weeks each |
| 5. **New result or merged OSS contribution** | something that did not exist before, reviewed by others | a GRPO ablation with a finding at 0.5B scale; a merged vLLM fix with a regression test; a Telugu task merged into lm-evaluation-harness | 1–3 months per result; PR review alone often takes weeks |
| 6. Others use or cite your work | outside adoption | your tokenizer or model downloaded and built on; your write-up cited; maintainers asking you to review PRs | months after release; you control quality and distribution, not adoption |

This repo gets you to rung 3 by construction. The [portfolio plan](portfolio.md) gets you to 4–5.

Climbing in order matters. A paper reproduction without rung 2 and 3 skills breaks at the first deep-dive question ("why that learning rate?"), and rung 5 results often start as a rung 4 reproduction that did not match.

## Preparation plans

Three plans for three starting points. All share the [roadmap](../../ROADMAP.md)'s rule: move on when the exit criteria are met, not when the calendar says so.

### Plan A: 12 months from scratch

This is the roadmap (October 2026 to September 2027, about 25 h/week, ~1,250 h) seen from the research-engineer side. The roadmap owns the learning milestones; the last column is what this track adds. Flagship milestones M0–M5 are defined in [portfolio.md](portfolio.md#flagship-telugu-first-small-lm).

| months | roadmap stages | build | interview preparation on top |
|---|---|---|---|
| Oct–Dec 2026 | 0 Setup, 1 Foundations, 2 Transformers | labs 01–07; S1 pretraining run; blog post 1; flagship M0–M1 | DSA 3 h/week; the matching question-bank section after each module; start the [story index](../../career/stories.md) |
| Jan–Apr 2027 | 3 Scale, 4 Post-training, 5 Inference; Gate A at the end of March | labs 08–14; write-ups 2 and 3; first OSS PR; flagship M2–M4 | one ML coding drill a week; napkin-math drills under 2 minutes; system designs A, C and E once each; a mock every two weeks from March (month 6) |
| Apr–Jun 2027 | 6 Eval and research, 7 Interp and safety, 8 Applied; Gate B at the end of June | labs 15–17; a research project with error bars; SAE study; FinSentinelAI evals; flagship M5 | from May: résumé v2 and practice-wave applications; mocks continue every two weeks |
| Jul–Sep 2027 | 9 Launch | a workshop-ready research write-up; 2–3 merged PRs | core and target waves; Plan C before each loop |

If you fall behind, follow the roadmap's [rule](../../ROADMAP.md#if-you-fall-behind): cut breadth, never depth.

### Plan B: 3-month sprint for an experienced ML engineer

For someone with several years of shipped ML who is fluent in PyTorch and knows transformers at the API level: about 20–25 h/week for 12 weeks (250–300 h), aimed at one primary team.

| weeks | focus | output |
|---|---|---|
| 1–2 | Diagnose. Read two company guides and pick the primary team. Do [lab 03](../../labs/03_napkin_math/README.md); self-grade one question-bank section a day; attempt the core of labs 05 and 07 from memory under a timer. | a ranked gap list for the target team |
| 3–6 | The labs your team leans on (team map above), plus labs 03 and 15 whatever the team. Three [ML coding drills](coding-interviews.md#ml-coding-drills) a week. | labs passing; a bug log |
| 5–10 | One [research project](../../curriculum/08-evaluation-and-research.md#research-projects-that-fit-an-8-gb-gpu) matched to the team, with a prediction and CIs. One OSS PR in a project that team uses. | a published write-up; a PR open or merged |
| 7–8 | Practice-wave applications ([apply in waves](../../career/applications.md#apply-in-waves)). | real interview practice |
| 9–12 | A mock every week, rotating round types; system designs A–H once each; 12–15 stories and the values answers; core and target waves. | loops on the calendar |

Skim modules 01–03 and labs 01–02 if their tests pass on your first attempt and you can derive the softmax cross-entropy backward on paper. Do not skip napkin math or eval statistics: both come up in several round types.

### Plan C: the final four weeks before a loop

Start when a loop is on the calendar. For coding, follow the [four-week drill plan](coding-interviews.md#a-four-week-drill-plan); this plan wraps the other rounds around it.

| when | do |
|---|---|
| 4 weeks out | Get the format from the recruiter ([what to ask](interview-loops.md#logistics-interviewing-across-time-zones-from-india)). One diagnostic mock per round type in the loop. Audit every number on your résumé ([numbers you must verify](../../career/stories.md#numbers-you-must-verify)). Read the company guide and its reading list. |
| 3 weeks out | Daily: two timed ML coding drills, 15 minutes of napkin math, and 45 minutes of DSA if the loop has it. Two system designs, one matching the team. Rehearse both deep-dive projects at 1, 5 and 20 minutes, recorded. |
| 2 weeks out | A full mock loop in one day with peers. Values and "why this company" answers said out loud. Two questions for each interviewer ([questions to ask](../../career/stories.md#questions-to-ask-them)). |
| final week | Taper: re-read your bug log and debriefs; no new topics. Check logistics (time zone, equipment, backup internet and power). If the loop runs late in your time zone, shift your sleep toward it over the last few days. |

## A weekly schedule with a full-time job

About 20–21 hours a week, close to the roadmap's 25-hour assumption. [Module 00](../../curriculum/00-learning-os.md#3-a-concrete-weekly-template) has the general learning template; this version adds interview preparation and keeps the roadmap's 3 hours of DSA.

| day | block | hours |
|---|---|---|
| Mon | module reading and derivations on paper; 15 minutes of flashcards | 2 |
| Tue | lab implementation: one function, tests first | 2 |
| Wed | DSA (1.5 h); napkin-math drills (0.5 h) | 2 |
| Thu | lab or experiment setup; launch an overnight GPU run | 2 |
| Fri | paper reading, or rest | 0–1 |
| Sat | deep block: lab, experiment or write-up (5 h); DSA (1.5 h) | 6.5 |
| Sun | OSS or flagship (3 h); question bank or a mock (1.5 h); weekly review and journal entry (1 h) | 5.5 |
| **total** | | **20–21** |

- **Let the GPU work while you sleep or work.** Queue long runs at night and read the logs in the morning.
- **Protect the two weekend blocks.** Weekday slots are for work that fits in 60–90 minutes: one drill, one derivation, one function.
- **At this pace** the roadmap's ~1,250 hours take about 14 months, and at 12–15 h/week about 19–24 months. Keep the order and cut breadth.
- **In application months,** swap Thursday's lab block for a mock or a system design, and spend the weekend block on write-ups and PRs.
- **Take one day fully off.** See [sustainability](../../career/sustainability.md).

## Ready to apply when

Start the practice wave when the first four boxes are ticked; apply to frontier labs when all of them are.

- [ ] Labs 01–17 pass (`python tools/progress.py`), and you can rewrite the core of labs 05, 07 and 11 from memory in 30 minutes each.
- [ ] Napkin-math drills: at least 80% right, each under 2 minutes ([lab 03](../../labs/03_napkin_math/README.md)).
- [ ] Two public write-ups with predictions and CIs, one on your target team's topic ([portfolio](portfolio.md#artifact-standards)).
- [ ] Two projects rehearsed at 1, 5 and 20 minutes, with every résumé number checked against your own records.
- [ ] One merged PR in a project your target team uses, and another in review.
- [ ] Mocks: three ML coding mocks in a row scored 3 or 4 on the [scoring sheet](interview-loops.md#mock-interview-protocol); one system design and one deep dive passed with a peer.
- [ ] If the loop has DSA: two mediums in 45 minutes, consistently ([dsa-patterns.md](dsa-patterns.md#how-to-use-this-list)).
- [ ] 12–15 stories and your values answers written ([stories](../../career/stories.md#values-and-mission-prep)), and a specific "why this company" for each target.
- [ ] Résumé v2 passes the [core-AI checklist](../../career/resume/guide.md#checklist-for-core-ai-roles).
- [ ] Logistics ready: notice period, visa and relocation questions ([visa and relocation](../../career/visa-and-relocation.md#questions-to-ask-recruiters-early)).

## Files in this track

| file | what it is | use it when |
|---|---|---|
| [interview-loops.md](interview-loops.md) | loop stages, every round type with a rubric and example prompts, the mock protocol, time-zone logistics | you want to know what a round scores; before every loop |
| [question-bank.md](question-bank.md) | hard questions with answers, by topic | after each module (its matching section); a full pass before loops |
| [ml-system-design.md](ml-system-design.md) | the framework, reference numbers and worked LLM-era designs | from stage 5; one timed design a week before loops |
| [coding-interviews.md](coding-interviews.md) | ML coding and practical engineering drills; a four-week plan | weekly once the matching labs pass; daily before loops |
| [dsa-patterns.md](dsa-patterns.md) | 18 DSA patterns with templates and problems | 3 h/week throughout; more before Big Tech loops |
| [portfolio.md](portfolio.md) | artifact standards, templates, open-source contributions, the flagship | month 1 to plan; before every write-up or release |

Outside this track: [companies](../../companies/README.md) for team-specific processes and projects, [career](../../career/README.md) for applications, outreach, the résumé and offers, [ROADMAP.md](../../ROADMAP.md) for the calendar, and the [journal](../../journal/README.md) for predictions and debriefs.
