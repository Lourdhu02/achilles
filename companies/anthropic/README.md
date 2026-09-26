# How to get into Anthropic

A guide to joining Anthropic as a core AI engineer: a research engineer or scientist working on pretraining, RL, inference, interpretability, alignment, safeguards or research infrastructure.
It covers where these people work, what Anthropic says it values, how its hiring works (its own guidance first, candidate reports second), and a 90-day plan for a 2-year ML engineer in India built on this repo's labs.

> [!IMPORTANT]
> Team names, open roles, offices, headcount and hiring steps change often. Every volatile fact below is marked **as of September 2026** and linked to its source. Re-check them before you apply; [refresh-brief.md](refresh-brief.md) is a quarterly checklist for doing that.

Companion files: [reading-list.md](reading-list.md) (what to read, in order) · [projects.md](projects.md) (portfolio projects that fit an 8 GB GPU) · [refresh-brief.md](refresh-brief.md) (refresh the facts).

## Contents
- [At a glance](#at-a-glance)
- [Where core-AI people work](#where-core-ai-people-work)
- [What they value](#what-they-value)
- [The hiring process](#the-hiring-process)
- [How to prepare with this repo](#how-to-prepare-with-this-repo)
- [Signals that get you noticed](#signals-that-get-you-noticed)
- [A 90-day plan](#a-90-day-plan)
- [Tips](#tips)
- [Common mistakes](#common-mistakes)
- [From India](#from-india)
- [Sources](#sources)

## At a glance

As of September 2026.

| | |
|---|---|
| **What it is** | An AI safety and research company, incorporated as a public benefit corporation. It builds the Claude family of models and sells them through claude.ai, Claude Code, the Claude Developer Platform (API) and the major clouds. |
| **What it ships now** | Its newsroom lists Claude Fable 5.1 and Claude Mythos 5.1 (1 September 2026) and Claude Opus 5.5 (22 September 2026) as the latest model launches ([news](https://www.anthropic.com/news)). |
| **What it researches now** | Its research page names five research teams: Alignment, Economics, Interpretability, Societal Impacts and the Frontier Red Team ([research](https://www.anthropic.com/research)). Recent posts cover AI for science, measuring the pace of AI development inside frontier labs, formal mathematics and an alignment assessment of cybersecurity incidents. |
| **Safety framework** | Responsible Scaling Policy **version 3.4, effective 8 July 2026** ([RSP](https://www.anthropic.com/responsible-scaling-policy)). |
| **Offices** | Most staff are in the Bay Area ([careers](https://www.anthropic.com/careers)); job titles also list London, and the jobs board lists India roles in Bangalore and Mumbai ([jobs](https://www.anthropic.com/jobs)). Anthropic announced a Bengaluru office on 8 October 2025 ([CNBC](https://www.cnbc.com/2025/10/08/anthropic-to-open-first-india-office-in-2026.html)) and opened it on 16 February 2026, its second Indo-Pacific office after Tokyo ([Anthropic](https://www.anthropic.com/news/bengaluru-office-partnerships-across-india)). |
| **India** | "India is the second-largest market for Claude.ai", and nearly half of Claude usage in India is computer and mathematical tasks (same post). Irina Ghose is Managing Director of India ([Anthropic](https://anthropic.com/news/anthropic-appoints-irina-ghose-as-managing-director-of-india)). |
| **Size** | Anthropic does not publish headcount. Third-party estimates for 2026 disagree by about 2x; Revelio Labs estimated about 3,950 employees in March 2026 ([Revelio Labs](https://www.reveliolabs.com/companies/anthropic-pbc/employees)). Treat any single number as rough. |
| **Internships** | "We don't currently offer internships" ([careers](https://www.anthropic.com/careers)). The closest equivalent is the [Anthropic Fellows Program](#signals-that-get-you-noticed). |

## Where core-AI people work

Anthropic groups open roles by department. The ones that hold core-AI work are **AI Research & Engineering**, **Safeguards (Trust & Safety)**, **Software Engineering - Infrastructure**, **Compute**, **Applied AI** and **Security** ([jobs](https://www.anthropic.com/jobs), as of September 2026). The table below maps the work to example job titles seen on the jobs board in September 2026, and to the evidence from this repo that fits each one.

| Area | What the work is | Example titles (September 2026) | Evidence that fits |
|---|---|---|---|
| **Pretraining** | Data, architecture, scaling-law science, and keeping large runs stable and efficient | Research Engineer/Scientist, Pre-training; Research Engineer, Pretraining Scaling (also a London variant) | [Lab 05](../../labs/05_transformer/README.md), [lab 08](../../labs/08_scaling_laws/README.md), [lab 09](../../labs/09_parallelism/README.md), [module 05](../../curriculum/05-pretraining.md); a scaling-law sweep with honest fits |
| **RL and post-training** | RL environments, reward signals, RL infrastructure and the science of scaling RL | Research Engineer, Machine Learning (Reinforcement Learning); Research Engineer, RL Engineering; Research Engineer, RL Scaling Science; Research Engineer, Machine Learning (RL Velocity) | [Lab 11](../../labs/11_dpo/README.md), [lab 12](../../labs/12_grpo/README.md), [module 06](../../curriculum/06-post-training.md); a GRPO run where you found and fixed reward hacking |
| **Inference** | Serving Claude fast and correctly across its own and cloud hardware | Staff Software Engineer, Inference; Staff + Senior Software Engineer, Inference Deployment / Inference Infrastructure; Cloud Inference | [Lab 06](../../labs/06_attention_kernels/README.md), [lab 07](../../labs/07_kv_cache_sampling/README.md), [lab 13](../../labs/13_quantization/README.md), [lab 14](../../labs/14_speculative_decoding/README.md), [module 07](../../curriculum/07-inference.md) |
| **Performance engineering** | Low-level optimization of kernels for accelerators | Its take-home is public (see [the hiring process](#the-hiring-process)) | [Lab 03](../../labs/03_napkin_math/README.md), [lab 06](../../labs/06_attention_kernels/README.md), [module 02](../../curriculum/02-compute-and-hardware.md) |
| **Interpretability** | Reverse-engineering what models compute: features, circuits, attribution graphs | Research Engineer, Interpretability; Research Scientist, Interpretability | [Lab 16](../../labs/16_interpretability/README.md), [module 09](../../curriculum/09-interpretability-and-safety.md); an SAE or circuit study |
| **Alignment (alignment science)** | Model organisms of misalignment, scalable oversight, alignment evaluations and audits, AI control | Research Engineer / Scientist, Alignment (also London) | [Lab 12](../../labs/12_grpo/README.md), [lab 15](../../labs/15_eval_stats/README.md), [lab 16](../../labs/16_interpretability/README.md); a small model-organism study |
| **Safeguards** | Classifiers, misuse detection, jailbreak robustness, enforcement at scale | ML/Research Engineer, Safeguards; Staff+ Software Engineer, Safeguards | [Lab 15](../../labs/15_eval_stats/README.md), [lab 17](../../labs/17_retrieval/README.md), [module 09 §3](../../curriculum/09-interpretability-and-safety.md#3-misuse-and-security-agent-builders-must-master-this) |
| **Frontier Red Team** | What frontier models imply for cyber, bio and autonomy risk; capability evaluations that feed the RSP | Research team listed on [research](https://www.anthropic.com/research) | Rigorous evals with CIs ([lab 15](../../labs/15_eval_stats/README.md)); security or bio background helps |
| **Societal Impacts and Economics** | How Claude is used in the real world and what it does to work and the economy | Research teams listed on [research](https://www.anthropic.com/research) | Careful measurement on real usage data; statistics |
| **Research infrastructure and compute** | Clusters, schedulers, data and training platforms that researchers run on | Software Engineer, Research Infrastructure; Staff+ Infrastructure Engineer, Cluster Infrastructure | [Lab 09](../../labs/09_parallelism/README.md), [module 02](../../curriculum/02-compute-and-hardware.md); strong distributed-systems work |
| **Applied AI and forward-deployed** | Building with customers on Claude: agents, retrieval, evals and deployment | Applied AI, Research Engineer; Applied AI Architect (Bangalore and Mumbai listings); Forward Deployed Engineer | [Module 10](../../curriculum/10-applied-llm-systems.md), [lab 17](../../labs/17_retrieval/README.md); measured agent systems |

> [!NOTE]
> Evaluation work is not one team. It shows up in Frontier Red Team capability evals, alignment audits, RL environments and engineering posts about eval noise (see [reading-list.md](reading-list.md#evaluation-and-engineering)). If you are strong at evals, say which of these you mean.

## What they value

Read the primary sources, not summaries. Interviewers can tell.

**Core Views on AI Safety** (8 March 2023, [link](https://www.anthropic.com/news/core-views-on-ai-safety)). The argument in short: we do not know how to make powerful systems reliably safe ("We do not know how to train systems to robustly behave well"), and some safety problems only appear in frontier-scale systems, so Anthropic builds frontier models to do safety research on them. It frames its bets as a portfolio across optimistic, intermediate and pessimistic scenarios. It lists research directions: mechanistic interpretability, scalable oversight, process-oriented learning, understanding generalization, testing for dangerous failure modes, and societal impacts and evaluations.

**Responsible Scaling Policy** (v3.4, effective 8 July 2026, [link](https://www.anthropic.com/responsible-scaling-policy)). The mechanism: capability thresholds (for example automating AI research, or meaningful help with chemical or biological weapons) are tied to required safeguards, organized as AI Safety Levels. The policy requires regular capability assessments, risk reports on deployed models, and external review. Version 3.4 changed the automated-R&D thresholds and the internal distribution of risk reports; read the redline linked from the policy page to see how a policy evolves in practice.

**Claude's constitution** ([link](https://www.anthropic.com/constitution)). The document that describes how Claude is meant to think and behave. It is the clearest statement of what "helpful, honest and harmless" means to Anthropic in practice.

**Company values** ([careers](https://www.anthropic.com/careers), as of September 2026): act for the global good; hold light and shade; be good to our users; ignite a race to the top on safety; do the simple thing that works; be helpful, honest and harmless; put the mission first. The page also describes a high-trust, low-ego organization.

What these mean when you are being interviewed:

| Value | What it looks like in a candidate |
|---|---|
| Do the simple thing that works | You start with the simplest baseline, measure it, and only add complexity the data asks for. Your write-ups report the boring baseline. |
| Hold light and shade | You can talk about both the benefits and the risks of AI without cheerleading or doom. |
| Helpful, honest, harmless (low ego) | You say "I don't know" quickly, correct your own mistakes out loud, and credit collaborators. |
| Race to the top on safety | You have a specific view on a safety problem, based on reading and ideally on something you built. |
| Put the mission first | You can explain why you want *this* work, not just a frontier-lab brand. |

> [!TIP]
> Before any values conversation, write the five answers in [career/stories.md](../../career/stories.md#values-and-mission-prep) and ground at least two of them in Anthropic's own documents: one in the Core Views or RSP, one in a paper from [reading-list.md](reading-list.md). Then have someone who disagrees with you argue against them.

## The hiring process

### What Anthropic says (primary sources)

As of September 2026:

- **Format.** Interviews run over Google Meet. "For technical roles, we use live coding tools like Colab and CodeSignal." In technical interviews "you can look things up; just be comfortable with basic syntax." Non-technical interviews are described as conversational ([careers](https://www.anthropic.com/careers)).
- **Reapplying.** You can reapply "after 12 months, or sooner if something materially changes" (same page).
- **Visas.** "We sponsor visas and green cards for eligible roles" (same page).
- **Using AI in your application** ([candidate AI guidance](https://www.anthropic.com/candidate-ai-guidance), last updated 10 July 2025). The short version, in their words: "Use Claude to refine your ideas, not replace them."
  - Applying: write your own draft first, then use AI to refine how you say it. Never to invent experience.
  - Take-home assessments: complete them **without AI unless the instructions explicitly allow it**.
  - Interview preparation: AI use is encouraged, to research, practise answers and prepare questions.
  - Live interviews: **no AI assistance**. They want to "see how you think through problems in real time."
- **The performance-engineering take-home is public.** On 21 January 2026 the performance optimization lead described the team's take-home ([Designing AI-resistant technical evaluations](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations)): optimize a program for a simulated accelerator resembling a TPU (manually managed memory, VLIW instruction packing, SIMD, multiple cores). It began as a 4-hour test and later became 2 hours, and this particular take-home **explicitly allows AI tools**, as an exception to the general guidance. The original version is released as an open challenge at `github.com/anthropics/original_performance_takehome`. The post's lessons on what makes a good evaluation (longer horizons, realistic environments, a wide scoring distribution, no single-insight puzzles) also tell you what they are looking for in you.

### What candidates report (secondary, treat as indicative)

Candidate write-ups and interview-prep sites ([Exponent (now Aced)](https://www.tryexponent.com/blog/anthropic-interview-process), [IGotAnOffer](https://igotanoffer.com/en/advice/anthropic-interview-process), [Glassdoor](https://www.glassdoor.com/Interview/Anthropic-Research-Engineer-Interview-Questions-EI_IE8109027.0,9_KO10,27.htm)) describe a loop along these lines for research engineering. It varies by team and year, so confirm the exact steps with your recruiter.

1. **Recruiter screen**: background, motivation, logistics; mission fit is already being assessed.
2. **Online coding assessment** (CodeSignal is the tool named on the careers page): a timed, multi-part problem where requirements grow as you go, judged on working, clean code.
3. **Hiring-manager or technical screen**: your past work and a technical conversation.
4. **Virtual onsite**, often over more than one day: practical coding, ML or system design, a deep dive into a past project, and a **values or culture interview**. Several guides report that strong engineers underestimate this round.
5. **References and team matching**, then an offer. Reported end-to-end times run from about one to four months.

> [!WARNING]
> Specific "Anthropic interview questions" posted online are mostly unverifiable and often stale. Prepare for the round *types*, not for leaked questions.

## How to prepare with this repo

Map each round to material you already have. The general versions of these rounds are in [interview-loops.md](../../tracks/research-engineer/interview-loops.md).

| Round | What to drill | Where |
|---|---|---|
| Online assessment / practical coding | Multi-part builds that grow in scope (a class you extend three times), timed, in a plain editor, no AI | [coding-interviews.md](../../tracks/research-engineer/coding-interviews.md#practical-engineering-drills); rewrite labs [04](../../labs/04_tokenizer/README.md) and [07](../../labs/07_kv_cache_sampling/README.md) from memory |
| ML coding | Attention, sampling, a training loop, DPO and GRPO losses, an SAE, in 15–30 minutes each | [coding-interviews.md](../../tracks/research-engineer/coding-interviews.md#ml-coding-drills); labs [05](../../labs/05_transformer/README.md), [11](../../labs/11_dpo/README.md), [12](../../labs/12_grpo/README.md), [16](../../labs/16_interpretability/README.md) |
| Debugging | A broken training run, a silent eval bug | [Module 03 §7](../../curriculum/03-deep-learning.md#7-debugging-a-training-run-the-playbook) |
| Systems and napkin math | "How many GPUs to serve this?", "Why is decode memory-bound?" | [Lab 03](../../labs/03_napkin_math/README.md), [module 02](../../curriculum/02-compute-and-hardware.md) |
| ML system design | An RL training pipeline, an eval platform, a jailbreak-classifier service, an inference fleet | [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md) |
| Research deep dive | Defend every number; say what surprised you and what you would do next | [portfolio.md](../../tracks/research-engineer/portfolio.md), [projects.md](projects.md), [module 08](../../curriculum/08-evaluation-and-research.md) |
| Interpretability and safety depth | Superposition, SAEs, attribution graphs, reward hacking, alignment faking, classifiers | [Module 09](../../curriculum/09-interpretability-and-safety.md), [lab 16](../../labs/16_interpretability/README.md), [reading-list.md](reading-list.md) |
| Values | Your five written positions, argued against | [career/stories.md](../../career/stories.md#values-and-mission-prep) |
| Performance take-home (if applying there) | Attempt the public challenge; write up your approach | [Lab 06](../../labs/06_attention_kernels/README.md), [projects.md](projects.md#9-the-public-performance-take-home-cpu-only) |
| Behavioral | 12–15 STAR stories with numbers | [career/stories.md](../../career/stories.md) |

Question practice: [question-bank.md](../../tracks/research-engineer/question-bank.md), especially the post-training, evaluation and "behavioral probes" sections.

## Signals that get you noticed

Ranked by how much they move a reviewer who does not know you, strongest first. All as of September 2026.

1. **Careful public work on a problem Anthropic works on.** A reproduction or extension of one of their papers (superposition, SAEs, attribution graphs, sycophancy, reward hacking, CoT faithfulness) with error bars, a negative result reported honestly, and code others can run. [projects.md](projects.md) lists eight that fit an 8 GB GPU. This is the signal every other item amplifies.
2. **The Anthropic Fellows Program.** A four-month, full-time, mentored research program. The jobs board lists fellows tracks in AI Safety & Security, ML Systems & Reinforcement Learning, and Economics & Policy, located in London, Ontario, San Francisco and remote in the US. The 2026 call describes a weekly stipend (3,850 USD, 2,310 GBP or 4,300 CAD), a compute budget of roughly 15,000 USD a month, mentorship from Anthropic researchers, cohorts starting several times a year, and reports that over 40% of earlier fellows went on to join Anthropic full time ([2024 announcement](https://alignment.anthropic.com/2024/anthropic-fellows-program/), [2026 call](https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/)). **Eligibility:** you need existing work authorization in the US, UK or Canada and must be located there; Anthropic does not sponsor visas for fellows. From India this is usually closed unless you already hold such authorization. The ML Systems & Reinforcement Learning track matters for core-AI candidates: the program is not only safety research. Fellows' work is public; for example, the open-source circuit-tracing library was led by two Fellows ([announcement](https://www.anthropic.com/research/open-source-circuit-tracing)).
3. **A strong result on the public performance take-home.** The January 2026 post invites people who beat the stated benchmark to contact recruiting. Read the post for the current threshold.
4. **Merged contributions to tools the field uses for this work**: Anthropic's open-sourced circuit-tracing tools ([announcement](https://www.anthropic.com/research/open-source-circuit-tracing)), and widely used interpretability and eval libraries. Small, correct PRs count.
5. **A referral from someone who has seen your work.** A referral without work behind it helps little. See [career/outreach.md](../../career/outreach.md).
6. **Clear public writing** about your experiments: blog posts, the Alignment Forum or LessWrong, a well-kept Hugging Face model card. See [career/public-presence.md](../../career/public-presence.md).
7. **Structured safety programs** such as ARENA (free curriculum) or mentored research programs like MATS. They produce the evidence in item 1 and a network of people who review your work.
8. **Conventional credentials** (brand-name employers, degrees). They help you pass the first filter and do not substitute for items 1 to 4.

## A 90-day plan

For a 2-year ML engineer in Bengaluru with a full-time job, an RTX-class 8 GB GPU (or free Colab/Kaggle), and about 12–15 hours a week. The goal after 90 days: two public artifacts aimed at one Anthropic team, a rehearsed loop, and an application.

**Pick one target team on day 1** (interpretability, alignment, RL, inference or safeguards). Everything below bends toward it.

| Weeks | Focus | Deliverable |
|---|---|---|
| **1–2** | Read the five "read first" items in [reading-list.md](reading-list.md). Write your five values answers in [career/stories.md](../../career/stories.md#values-and-mission-prep). Pass labs [03](../../labs/03_napkin_math/README.md) and [05](../../labs/05_transformer/README.md) if you have not. | A one-page note: "the problem I want to work on at Anthropic and why", with three citations. |
| **3–4** | Your team's core labs: [16](../../labs/16_interpretability/README.md) for interpretability; [12](../../labs/12_grpo/README.md) and [15](../../labs/15_eval_stats/README.md) for alignment, RL or safeguards; [06](../../labs/06_attention_kernels/README.md), [07](../../labs/07_kv_cache_sampling/README.md) and [13](../../labs/13_quantization/README.md) for inference. Start the matching project from [projects.md](projects.md): write its pre-registered prediction first. | Tests passing; a project plan with a predicted result and a stopping rule. |
| **5–8** | Run the project. One experiment a week, logged in `journal/`. Add error bars ([lab 15](../../labs/15_eval_stats/README.md)). Start ML-coding drills: three a week, timed. | Project 1 write-up published, with code, figures, CIs and a limitations section. |
| **9–10** | A smaller second artifact: a merged PR, the performance take-home, or a second project that extends the first. First mock interview. | Artifact 2 public. Mock debrief written. |
| **11–12** | Mocks every week: practical coding, system design, a 20-minute deep dive on project 1, a values conversation. Tailor your résumé to one role title from the jobs board. | Résumé that leads with the two artifacts; a 120-word note per referral request. |
| **13** | Apply (and to the Fellows Program if you are eligible). Ask for referrals from people who have read your write-ups. Plan the next 90 days whatever the outcome. | Application sent; next project chosen. |

> [!NOTE]
> Ninety days is enough for a credible first application, not a guaranteed offer. Most people who get in have a longer trail of public work. If you are rejected, you may reapply after 12 months or sooner with material changes ([careers](https://www.anthropic.com/careers)); your second artifact is that change.

## Tips

> [!TIP]
> **Aim at a team, not at "Anthropic".** A cover note that says "I reproduced your SAE feature-splitting result on a 70M model and found X" beats three paragraphs on the mission.

> [!TIP]
> **Do the simple thing, then say so.** Lead every write-up with the simplest baseline and its number. It matches a stated company value and it is what research reviewers check first.

> [!TIP]
> **Follow the AI-use rules to the letter.** No AI in live interviews; no AI in take-homes unless the instructions allow it; AI for preparation is fine. Say how you used AI tools in your projects if asked. Honesty is itself assessed.

> [!TIP]
> **Prepare for "you can look things up".** Live coding allows lookups, so they are testing reasoning and fluency, not recall. Practise in the tools named on the careers page (Colab and CodeSignal) until the environment is invisible.

> [!TIP]
> **Know the RSP well enough to criticise it.** "What would you change about the RSP?" is a fair question in a values conversation. A considered, specific answer is better than praise.

## Common mistakes

- **Mission cosplay.** Repeating safety vocabulary without a specific view. Interviewers probe past the first answer.
- **Treating the values round as a formality.** Candidate reports agree that strong engineers are rejected here.
- **Portfolio of API wrappers** for a core research role. Build and measure the mechanism instead.
- **Claims without error bars.** A 2-point gain on 200 examples is noise; say so before they do ([lab 15](../../labs/15_eval_stats/README.md)).
- **Over-scoped projects** that never finish. A finished small study beats an abandoned ambitious one.
- **Applying to many roles at once.** Pick the one or two titles that fit your evidence.
- **Assuming the Bengaluru office hires core researchers.** As of September 2026 its public listings are applied and customer-facing roles (see [From India](#from-india)).

## From India

As of September 2026; re-verify before relying on any of this.

- **Bengaluru office.** Opened 16 February 2026. Anthropic says it will hire "local talent across a wide array of roles" and offer "applied AI expertise to enterprise customers, digital natives, and startups", with partnerships across enterprises, education, agriculture and Indian-language evaluation ([Anthropic](https://www.anthropic.com/news/bengaluru-office-partnerships-across-india)). India listings seen on the jobs board in September 2026: Applied AI Architect (Bangalore and Mumbai) and Customer Success (Bangalore).
- **Core research roles** in the titles above are listed outside India, mainly in the San Francisco Bay Area and London. Getting one from India usually means relocating.
- **Remote.** The careers page says most staff are in the Bay Area and come to the office regularly, with more flexibility for people further away who visit periodically. Treat fully remote-from-India core research as the exception; ask the recruiter for the specific role.
- **Visas.** Anthropic says it sponsors visas and green cards for eligible roles. The Fellows Program does not sponsor visas. For US (H-1B, O-1) and UK (Skilled Worker, Global Talent) routes and their current rules, see [career/visa-and-relocation.md](../../career/visa-and-relocation.md) and an immigration lawyer.
- **The applied route.** An Applied AI role in Bengaluru is a real job in its own right: building agents and evals with Indian customers on Claude. Take it if you want that work. Do not take it only as a back door to research; internal moves are never guaranteed.
- **Your advantage.** Anthropic's India post names Indian-language evaluation work. Careful work on Indic tokenization, evals or safety behaviour in Indian languages is uncrowded and relevant (see [projects.md](projects.md)).
- **Time zones.** Interviews are on Google Meet; ask for slots that work in IST.

## Sources

Primary (Anthropic), accessed September 2026:
- Careers: https://www.anthropic.com/careers · Jobs board: https://www.anthropic.com/jobs
- Candidate AI guidance (updated 10 July 2025): https://www.anthropic.com/candidate-ai-guidance
- Research teams: https://www.anthropic.com/research · Newsroom: https://www.anthropic.com/news · Engineering blog: https://www.anthropic.com/engineering
- Core Views on AI Safety (March 2023): https://www.anthropic.com/news/core-views-on-ai-safety
- Responsible Scaling Policy (v3.4, July 2026): https://www.anthropic.com/responsible-scaling-policy
- Claude's constitution: https://www.anthropic.com/constitution
- Designing AI-resistant technical evaluations (January 2026): https://www.anthropic.com/engineering/AI-resistant-technical-evaluations
- Bengaluru office and India partnerships (February 2026): https://www.anthropic.com/news/bengaluru-office-partnerships-across-india
- India managing director: https://anthropic.com/news/anthropic-appoints-irina-ghose-as-managing-director-of-india
- Anthropic Fellows Program: https://alignment.anthropic.com/2024/anthropic-fellows-program/ and https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/
- Open-source circuit tracing: https://www.anthropic.com/research/open-source-circuit-tracing

Secondary (labelled where used):
- CNBC, "Anthropic to open first India office in 2026" (October 2025): https://www.cnbc.com/2025/10/08/anthropic-to-open-first-india-office-in-2026.html
- Revelio Labs headcount estimate: https://www.reveliolabs.com/companies/anthropic-pbc/employees
- Candidate-reported process: [Exponent (now Aced)](https://www.tryexponent.com/blog/anthropic-interview-process), [IGotAnOffer](https://igotanoffer.com/en/advice/anthropic-interview-process), [Glassdoor](https://www.glassdoor.com/Interview/Anthropic-Research-Engineer-Interview-Questions-EI_IE8109027.0,9_KO10,27.htm)
