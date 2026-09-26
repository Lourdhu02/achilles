# How to get into OpenAI

A working guide to getting hired as a core AI engineer at OpenAI: research engineering on reasoning and post-training, pretraining and scaling, safety, inference, and the applied teams that ship Codex, ChatGPT and the API. It separates what OpenAI says in its own words from what others report, maps every interview round to a lab in this repo, and ends with a 90-day plan for a 2-year ML engineer in India.

> [!IMPORTANT]
> Everything about teams, offices, headcount, programs and process here is **as of September 2026** and changes fast. OpenAI reorganized its safety teams twice in under two years. Re-verify on [openai.com/careers](https://openai.com/careers/) and the [interview guide](https://openai.com/interview-guide/) before acting, or run [agent-brief.md](agent-brief.md).

**Contents:** [At a glance](#at-a-glance) · [Where core-AI people work](#where-core-ai-people-work) · [What they value](#what-they-value) · [The hiring process](#the-hiring-process) · [How to prepare with this repo](#how-to-prepare-with-this-repo) · [Signals that get you noticed](#signals-that-get-you-noticed-ranked) · [90-day plan](#90-day-plan-2-yoe-ml-engineer-in-india) · [Tips and common mistakes](#tips-and-common-mistakes) · [From India](#from-india) · [Sources](#sources)

Also in this folder: [reading-list.md](reading-list.md) (27 papers, 5 marked "read first") · [projects.md](projects.md) (9 portfolio projects sized for 8 GB or CPU) · [agent-brief.md](agent-brief.md) (quarterly refresh).

---

## At a glance

| | As of September 2026 |
|---|---|
| **Mission** | "to ensure that artificial general intelligence ... benefits all of humanity" ([Charter](https://openai.com/charter/)) |
| **Structure** | Nonprofit OpenAI Foundation controlling a for-profit public benefit corporation after the October 2025 recapitalization. Check [openai.com/our-structure](https://openai.com/our-structure/) for the current form. |
| **Products** | ChatGPT; the API platform; Codex (coding agent across CLI, IDE and cloud); Sora (video); open-weight gpt-oss models ([model card, Aug 2025](https://arxiv.org/abs/2508.10925)). |
| **Frontier models** | Reasoning models trained with large-scale RL (the o-series from 2024, the GPT-5 family from Aug 2025). CNBC reported the rollout of a model called **GPT-6 "Astra"** on 3 September 2026 ([CNBC](https://www.cnbc.com/2026/09/03/open-ai-astra-gpt-6-cyber.html)). Check the [model release notes](https://help.openai.com/en/articles/9624314-model-release-notes) for the current lineup. |
| **Size** | About 4,500 employees in early 2026, with a reported plan to reach about 8,000 by the end of 2026 (Financial Times via [CNBC, 21 March 2026](https://www.cnbc.com/2026/03/21/openai-to-nearly-double-workforce-to-8000-by-end-2026-ft-reports.html)). Treat both numbers as reported, not official. |
| **Headquarters** | San Francisco. Core research roles are listed almost entirely in San Francisco; check each listing's location. |
| **India** | New Delhi office announced August 2025 ([TechCrunch](https://techcrunch.com/2025/08/21/openai-announces-new-delhi-office-as-it-expands-footprint-in-india)). In February 2026, at the India AI Impact Summit, OpenAI announced an "OpenAI for India" initiative and reported plans for Mumbai and Bengaluru offices in late 2026 ([Outlook Business](https://www.outlookbusiness.com/news/openai-for-india-sam-altman-to-open-new-offices-in-mumbai-bengaluru-in-2026)). Listings seen in India are go-to-market and applied roles, not research. See [From India](#from-india). |

What that means for you: OpenAI's frontier work is RL on reasoning models, agents that write and run code, and the evaluation and safety machinery around them. The public research record (below and in [reading-list.md](reading-list.md)) tells you which skills they reward: **careful evaluation, RL with verifiable rewards, process-level supervision, and monitoring what models think**.

## Where core-AI people work

OpenAI does not publish a stable org chart, and team names change. The areas below come from its research publications, system cards and recurring job titles. Search the [careers page](https://openai.com/careers/) by these keywords, not by team names you read in old posts.

| Area | What the work is | Evidence in the public record | Labs and modules that prepare you |
|---|---|---|---|
| **Reasoning and post-training / RL** | Large-scale RL on chains of thought, reward design, RL infrastructure, instruction following, model behavior. The core of every model since o1. | [Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) (2024), [o1 system card](https://arxiv.org/abs/2412.16720), [competitive programming with reasoning models](https://arxiv.org/abs/2502.06807), [deliberative alignment](https://arxiv.org/abs/2412.16339) | [Lab 12 GRPO](../../labs/12_grpo/README.md), [lab 11 DPO](../../labs/11_dpo/README.md), [module 06](../../curriculum/06-post-training.md) |
| **Pretraining and scaling** | Data, architecture, optimization and the scaling-law science that decides what to train next; training-run reliability on very large clusters. | [Kaplan scaling laws](https://arxiv.org/abs/2001.08361), [GPT-4 report](https://arxiv.org/abs/2303.08774) (predictable scaling section) | [Labs 02](../../labs/02_training_core/README.md), [05](../../labs/05_transformer/README.md), [08](../../labs/08_scaling_laws/README.md), [09](../../labs/09_parallelism/README.md), [module 05](../../curriculum/05-pretraining.md) |
| **Safety research and safety systems** | Alignment research (weak-to-strong, CoT monitoring, interpretability), safeguards and classifiers in production, misuse and abuse detection, model behavior policy. | [Weak-to-strong](https://arxiv.org/abs/2312.09390), [CoT monitoring](https://arxiv.org/abs/2503.11926), [Monitoring monitorability](https://arxiv.org/abs/2512.18311), [SAE scaling](https://arxiv.org/abs/2406.04093), [persona features](https://arxiv.org/abs/2506.19823) | [Lab 16](../../labs/16_interpretability/README.md), [lab 15](../../labs/15_eval_stats/README.md), [module 09](../../curriculum/09-interpretability-and-safety.md) |
| **Preparedness and frontier evals** | Measuring dangerous capabilities (bio/chem, cyber, AI self-improvement) and building the benchmarks that gate launches. | Preparedness Framework (v2, April 2025; see [openai.com/safety](https://openai.com/safety/)), [MLE-bench](https://arxiv.org/abs/2410.07095), [PaperBench](https://arxiv.org/abs/2504.01848), [SWE-Lancer](https://arxiv.org/abs/2502.12115) | [Lab 15](../../labs/15_eval_stats/README.md), [module 08](../../curriculum/08-evaluation-and-research.md) |
| **Inference and infrastructure** | Serving at very large scale: batching, KV-cache management, kernels, speculative decoding, fleet efficiency; the compute buildout (Stargate). | Job listings for inference, kernels, and supercomputing; gpt-oss release engineering (MXFP4, harmony format) | [Labs 03](../../labs/03_napkin_math/README.md), [06](../../labs/06_attention_kernels/README.md), [07](../../labs/07_kv_cache_sampling/README.md), [13](../../labs/13_quantization/README.md), [14](../../labs/14_speculative_decoding/README.md), [module 07](../../curriculum/07-inference.md) |
| **Applied engineering: ChatGPT, API, agents, Codex** | Product engineering on top of frontier models: agent loops, tool use, retrieval, evals-in-the-loop, latency and cost. Codex and agent teams sit closest to research. | [Codex paper](https://arxiv.org/abs/2107.03374) (2021), [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) (2024), the open-source [Codex CLI](https://github.com/openai/codex) and [Agents SDK](https://github.com/openai/openai-agents-python) | [Lab 17](../../labs/17_retrieval/README.md), [module 10](../../curriculum/10-applied-llm-systems.md), [lab 15](../../labs/15_eval_stats/README.md) |
| **Forward-deployed and solutions** | Engineers embedded with enterprise customers to design, build and deploy systems on OpenAI models. The one core-adjacent family currently hired in India (Applied AI Engineer, Solutions Engineer). | Listings such as [Applied AI Engineer, Delhi](https://openai.com/careers/applied-ai-engineer-delhi-india/) and [Solutions Engineer, Delhi](https://openai.com/careers/solutions-engineer-delhi-india/) | [Module 10](../../curriculum/10-applied-llm-systems.md), [founder track](../../tracks/founder/README.md) for customer-facing judgment |

> [!NOTE]
> **Safety organization, as reported in July 2026.** Several outlets reported that OpenAI folded its safety teams into the research organization, that the head of Safety Systems left, and that Preparedness risk areas were distributed to specialist teams ([The Next Web](https://thenextweb.com/news/openai-heidecke-safety-head-leaving-research-merger), [Engadget](https://www.engadget.com/2212941/openai-head-of-safety-leaving-company-reorganization/)). This is secondary reporting; do not quote internal structure in an interview as fact. Ask your recruiter where a role sits.

## What they value

Read the primary sources yourself, then form an opinion. Interviewers can tell whether you have.

**The Charter (2018).** The mission sentence in full: "OpenAI's mission is to ensure that artificial general intelligence (AGI), by which we mean highly autonomous systems that outperform humans at most economically valuable work, benefits all of humanity." The Charter commits to four principles:

1. **Broadly distributed benefits.** Use influence over AGI for everyone's benefit; avoid uses that harm humanity or unduly concentrate power.
2. **Long-term safety.** It includes the clause most candidates have never read: if "a value-aligned, safety-conscious project comes close to building AGI before we do, we commit to stop competing with and start assisting this project."
3. **Technical leadership.** Being at the frontier is treated as necessary to influence where AI goes.
4. **Cooperative orientation.** Work with other research and policy institutions.

**How they deploy.** OpenAI's stated approach is iterative deployment: ship models progressively so society and the company learn from real use. The artifacts that encode it, and that you should have read before an interview:

- [Model Spec](https://model-spec.openai.com/): how models should behave, the chain of command between platform, developer and user instructions, and the default behaviors. Useful for post-training and model-behavior roles.
- Preparedness Framework (linked from [openai.com/safety](https://openai.com/safety/)): tracked risk categories, capability thresholds, and the safeguards required before deployment. The April 2025 version (v2) reduced risk levels to two thresholds that matter, High and Critical, dropped persuasion as a tracked category, added research categories such as self-replication and hiding capabilities, and said requirements may be adjusted if a rival releases a high-risk system without comparable safeguards ([Axios, 15 April 2025](https://www.axios.com/2025/04/15/openai-risks-frameworks-changes)). A further revision was reported in August 2026; read the current version.
- System cards ([o1](https://arxiv.org/abs/2412.16720), [GPT-4o](https://arxiv.org/abs/2410.21276), [GPT-5](https://arxiv.org/abs/2601.03267), [gpt-oss](https://arxiv.org/abs/2508.10925)): what they measure before a launch, and how.

**Culture signals.** OpenAI's careers page has listed company values that emphasized AGI focus, intensity, scale and shipping things people love (the list was changed in 2023 and may have changed again; read the current page). In practice this reads as: bias to action, ownership end to end, and comfort with ambiguity. Mission alignment is probed explicitly in some loops.

> [!TIP]
> Prepare one honest paragraph on where you *disagree* with OpenAI, for example on iterative deployment, the pace of releases, or the trade-off between legible chains of thought and capability. A reasoned disagreement grounded in their own documents lands better than agreement recited from their website. Use [career/stories.md](../../career/stories.md#values-and-mission-prep) to draft it.

## The hiring process

### What OpenAI says (official)

From OpenAI's [interview guide](https://openai.com/interview-guide/) (read the live page; this is a summary as of September 2026):

| Stage | What happens |
|---|---|
| Application and résumé review | Apply to specific roles. The recruiting team reviews for fit. |
| Introductory call | About 30 minutes with a recruiter on your background, your motivation and the role. Sometimes a hiring-manager call follows. |
| Skills-based assessment | Format varies by team: pair coding, a take-home project, or other technical tests. Some roles have more than one. The guide says the recruiting team aims to update you within about a week. |
| Final interviews | Typically 4–6 hours with 4–6 people over one or two days, focused on your area of expertise and designed to stretch you. Virtual by default; you may choose to interview onsite in San Francisco. |
| Decision | Recruiter follow-up, references for some roles, offer. |

### What others report (secondary, unverified)

Candidate reports on third-party sites ([IGotAnOffer](https://igotanoffer.com/en/advice/openai-interview-process), [Interview Query](https://www.interviewquery.com/interview-guides/openai)) are consistent on a few points. Weigh them as anecdotes:

- **Practical coding over puzzles.** Multi-part problems that grow in scope (build a component, then extend it under new requirements), judged on working, readable code at speed, often in your own editor. Fewer classic LeetCode rounds than Big Tech, but not zero.
- **ML depth for research roles.** Implementing or debugging model code, reasoning about training dynamics, and a deep dive into your past project where every number gets probed.
- **System design** for infrastructure and applied roles, often LLM-specific (serving, eval pipelines, agent systems).
- **Behavioral and mission** conversations about ownership, speed, collaboration and why OpenAI.
- **Timelines** of roughly two to six weeks are reported; team matching is usually decided before the final loop, because you apply to a specific role.

## How to prepare with this repo

Map each round to something you will have built and measured.

| Round | Prepare with | Target before you apply |
|---|---|---|
| Practical coding | [coding-interviews.md](../../tracks/research-engineer/coding-interviews.md) practical drills; rebuild [lab 07](../../labs/07_kv_cache_sampling/README.md) (a batcher with a KV cache) and [lab 04](../../labs/04_tokenizer/README.md) (BPE) from a blank file | Multi-part build in 60 minutes with tests, talking as you go |
| ML coding | Every lab's core from memory: attention, sampling, the DPO loss, GRPO advantages and clipped loss ([lab 12](../../labs/12_grpo/README.md)), unbiased pass@k ([lab 15](../../labs/15_eval_stats/README.md)) | 15–30 minutes each, no references |
| ML debugging | [Module 03 debugging playbook](../../curriculum/03-deep-learning.md#7-debugging-a-training-run-the-playbook); plant bugs in lab 02 and lab 12 and swap with a friend | Find three planted bugs in 30 minutes |
| Napkin math | [Lab 03](../../labs/03_napkin_math/README.md); [lab 08](../../labs/08_scaling_laws/README.md) | Training FLOPs, KV-cache bytes and decode tokens/s, each in under 2 minutes |
| System design | [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md): an RL training pipeline with rollout workers, an eval platform, an inference fleet | One end-to-end design per week for 6 weeks |
| Research deep dive | Two of your write-ups at 1-, 5- and 20-minute depth ([interview-loops.md](../../tracks/research-engineer/interview-loops.md)) | Defend every number on your résumé |
| Research knowledge | [question-bank.md](../../tracks/research-engineer/question-bank.md) post-training, eval and scaling sections; [reading-list.md](reading-list.md) "read first" items | Explain process vs outcome supervision, reward hacking and pass@k bias cold |
| Mission and behavioral | [career/stories.md](../../career/stories.md); the Charter, Model Spec and Preparedness Framework | 12 stories; a written view on CoT monitoring and iterative deployment |

Curriculum order for OpenAI specifically: [06 post-training](../../curriculum/06-post-training.md) and [08 evaluation](../../curriculum/08-evaluation-and-research.md) first, then [07 inference](../../curriculum/07-inference.md) or [05 pretraining](../../curriculum/05-pretraining.md) depending on the team, and [09 safety](../../curriculum/09-interpretability-and-safety.md) for everyone.

## Signals that get you noticed (ranked)

Ranked by how directly each one maps to the work, and by what an engineer in India can realistically produce.

1. **OpenAI-run open challenges.** In March–April 2026 OpenAI ran [Parameter Golf](https://github.com/openai/parameter-golf): train the language model with the lowest bits per byte on FineWeb validation that fits in a 16,000,000-byte artifact and trains in under 10 minutes on 8×H100. Reporting described it as a talent search for early-career research hiring ([The Decoder](https://the-decoder.com/openai-turns-model-compression-into-a-talent-hunt-with-its-16-mb-parameter-golf-challenge/)). The official window has closed, but the repository and its submissions remain a public benchmark; a well-documented technique that beats a published entry is still a strong, directly legible signal. Watch for the next one. See [projects.md](projects.md) project 7.
2. **A result on OpenAI's own open artifacts.** A careful study using gpt-oss, the [Codex CLI](https://github.com/openai/codex), [simple-evals](https://github.com/openai/simple-evals), [PRM800K](https://github.com/openai/prm800k), [MLE-bench](https://github.com/openai/mle-bench), [frontier-evals](https://github.com/openai/frontier-evals) or the (archived) [weak-to-strong](https://github.com/openai/weak-to-strong) code. It shows you read their work closely and it gives an interviewer something concrete to discuss. See [projects.md](projects.md).
3. **An eval with honest statistics.** OpenAI publishes many benchmarks (SimpleQA, BrowseComp, HealthBench, SWE-bench Verified, GDPval). A re-analysis with confidence intervals, contamination checks or a grader-reliability study is small, cheap and exactly the skill their evals and preparedness work needs.
4. **RL on reasoning at small scale, written up well.** GRPO on a 0.5B model with a reward-hacking finding and a monitor that catches it. This is the closest a single GPU gets to their core research agenda.
5. **Merged pull requests in the open-source stack OpenAI touches** (Codex CLI, openai-agents-python, tiktoken; or vLLM, SGLang, llama.cpp, Triton for inference roles). Small, correct fixes count.
6. **The OpenAI Residency.** A six-month, full-time, paid program for people with strong engineering and math who want to move into research, embedded in research teams, in San Francisco. The 2026 cohort's applications closed around the start of 2026 ([residency page](https://openai.com/residency/)). Whether and when it reopens is not announced; check the page each quarter.
7. **Referrals from people who have seen your work.** Cold outreach works when it links one specific artifact. See [career/outreach.md](../../career/outreach.md).
8. **Kaggle-level competitive results or competitive programming.** Useful evidence of speed, weaker evidence of research judgment.

> [!WARNING]
> Building "a ChatGPT wrapper" is not a signal for core roles. Neither is a list of courses. Artifacts with measured numbers are.

## 90-day plan: 2-YOE ML engineer in India

Assumes about 12 hours a week alongside a job, an RTX 5060 8 GB or free Colab/Kaggle, and that you have finished labs 01–05. Each block ends in something public.

| Weeks | Focus | Output |
|---|---|---|
| 1–2 | Read the five "read first" papers in [reading-list.md](reading-list.md) with the extract-questions. Finish [lab 15](../../labs/15_eval_stats/README.md). | Paper notes in `journal/`; pass@k and paired-bootstrap code passing tests |
| 3–4 | [Lab 12](../../labs/12_grpo/README.md) core, then its S5 stretch: GRPO with LoRA on Qwen2.5-0.5B-Instruct on an arithmetic task | Reward, length and entropy curves; one failure mode reproduced |
| 5–7 | Flagship project: [projects.md](projects.md) project 3 (reward hacking and CoT monitoring) or project 1 (pass@k done right) | Repo with a one-command reproduction; results with 95% CIs |
| 8 | Write it up: question, setup, prediction, result, what surprised you, limitations | A blog post with one clear figure; posted to your network and relevant communities |
| 9–10 | Second, smaller project (weak-to-strong or SAE scaling) **or** one merged PR to the Codex CLI, Agents SDK or an inference engine | Second artifact |
| 11 | Interview preparation: two mock coding rounds, one system design (RL training pipeline), one deep dive on your flagship | Debriefs written within 24 hours |
| 12 | Apply to 3–5 matching roles (research engineer, Codex/agents, evals, or Applied AI Engineer in India), with outreach to one engineer per team linking your write-up | Applications tracked in [career/applications.md](../../career/applications.md) format |

> [!TIP]
> If your first goal is to join OpenAI from India soon, the realistic door in September 2026 is the **Applied AI Engineer / forward-deployed** family in Delhi, Mumbai or Bengaluru. Prepare that loop with module 10, a measured RAG or agent system, and customer stories. Keep the research artifacts going in parallel for an internal or later move.

## Tips and common mistakes

> [!TIP]
> **Lead with the number.** "GRPO on 0.5B raised exact-match from 31% to 58% (95% CI ±3) and the reward hack appeared at step 400; a CoT monitor caught 9 of 10 cases" is a résumé line. "Worked on RL for LLMs" is not. Use your own numbers, never these.

> [!TIP]
> **Read the system card before the interview.** For the team you are interviewing with, know the evaluations in the latest system card and what they would change. It is the fastest way to talk like a colleague.

> [!TIP]
> **Practice coding in your own editor, fast, with tests.** Practical rounds reward working code that is extended cleanly under new requirements. Rehearse with a timer and narrate trade-offs.

> [!TIP]
> **Know pass@k, process supervision and reward hacking cold.** They come straight from OpenAI's own papers and connect evaluation, RL and safety in one conversation.

Common mistakes:

- **Applying everywhere at once.** You apply to specific roles; ten unrelated applications look unfocused. Pick two or three that fit your artifacts.
- **Quoting internal org structure from news articles** as fact in interviews.
- **Claiming numbers you cannot rederive.** Every result gets probed three levels deep.
- **Treating mission questions as a formality.** Have a view on safety, deployment and what you would not build.
- **Only preparing LeetCode.** Practical and ML coding matter more for most core roles.
- **Ignoring statistics.** An eval without error bars is a weak signal at a company that publishes benchmarks.

## From India

As of September 2026; verify each point before relying on it.

- **Offices.** New Delhi office announced in August 2025. In February 2026 OpenAI announced "OpenAI for India" and, as reported by Indian outlets, plans for Mumbai and Bengaluru offices in late 2026 and a data-centre partnership with the Tata Group ([Outlook Business](https://www.outlookbusiness.com/news/openai-for-india-sam-altman-to-open-new-offices-in-mumbai-bengaluru-in-2026), [Free Press Journal](https://www.freepressjournal.in/tech/ai-summit-2026-delhi-openai-sam-altman-new-mumbai-bengaluru-offices)). One outlet noted the related OpenAI blog post was briefly unavailable ([Republic World](https://www.republicworld.com/tech/openai-expands-in-india-offices-in-bengaluru-and-mumbai-announced-but-blog-post-goes-missing)); confirm on openai.com.
- **Roles in India.** Listings seen in 2025–26 are go-to-market and applied engineering (Applied AI Engineer, Solutions Engineer, partnerships, policy). No core research roles were listed in India as of this writing.
- **Remote.** Core research and engineering roles are overwhelmingly on-site or hybrid in San Francisco. Do not assume remote from India; ask the recruiter.
- **Visa.** Moving to San Francisco usually means an H-1B (lottery-based, with 2025 fee and selection changes) or an O-1 (extraordinary ability, built on publications, open-source impact and awards). The Residency historically included relocation to San Francisco. See [career/visa-and-relocation.md](../../career/visa-and-relocation.md) and ask early: "Does this role sponsor visas, and what is the typical timeline?"
- **A route that works.** Join the India applied team, or a strong core-ML team at a multinational in India, build research artifacts in public, and move to a research-adjacent role later.

## Sources

Primary (OpenAI):

- Charter: https://openai.com/charter/
- Interview guide: https://openai.com/interview-guide/
- Careers: https://openai.com/careers/
- Residency: https://openai.com/residency/
- Model Spec: https://model-spec.openai.com/
- Safety hub (Preparedness Framework, system cards): https://openai.com/safety/
- Model release notes: https://help.openai.com/en/articles/9624314-model-release-notes
- gpt-oss model card: https://arxiv.org/abs/2508.10925
- OpenAI on GitHub: https://github.com/openai (Parameter Golf: https://github.com/openai/parameter-golf)

Secondary (reported; dated):

- TechCrunch, 21 Aug 2025, New Delhi office: https://techcrunch.com/2025/08/21/openai-announces-new-delhi-office-as-it-expands-footprint-in-india
- CNBC, 21 Mar 2026, workforce plan: https://www.cnbc.com/2026/03/21/openai-to-nearly-double-workforce-to-8000-by-end-2026-ft-reports.html
- CNBC, 3 Sep 2026, GPT-6 Astra: https://www.cnbc.com/2026/09/03/open-ai-astra-gpt-6-cyber.html
- Outlook Business, Feb 2026, OpenAI for India: https://www.outlookbusiness.com/news/openai-for-india-sam-altman-to-open-new-offices-in-mumbai-bengaluru-in-2026
- The Next Web and Engadget, July 2026, safety reorganization (links above)
- Axios, 15 Apr 2025, Preparedness Framework v2: https://www.axios.com/2025/04/15/openai-risks-frameworks-changes
- Interview reports: IGotAnOffer and Interview Query (links above)
- The Decoder, March 2026, Parameter Golf as a talent search (link above)
