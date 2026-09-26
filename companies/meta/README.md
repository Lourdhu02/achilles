# How to get into Meta AI

A practical guide to core-AI engineering roles in Meta's AI organizations: Meta Superintelligence Labs (MSL) and its groups, FAIR, the model teams that built Llama and now the Muse models, and PyTorch.
It covers where the work happens, what the interview loop tests, which signals count, and a 90-day plan for an ML engineer in India with about two years of experience.

> [!IMPORTANT]
> Meta reorganized its AI work at least four times between April 2025 and March 2026. Team names, leaders, open-weights policy, interview formats and India roles below are dated **as of September 2026** and many come from press reports, which are labelled. Re-verify on [metacareers.com](https://www.metacareers.com/) and Meta's own blogs before acting, and refresh this page each quarter with [refresh-brief.md](refresh-brief.md).

## Contents

- [At a glance](#at-a-glance)
- [Where core-AI people work](#where-core-ai-people-work)
- [What they value](#what-they-value)
- [The hiring process](#the-hiring-process)
- [How to prepare with this repo](#how-to-prepare-with-this-repo)
- [Signals, ranked](#signals-ranked)
- [90-day plan](#90-day-plan)
- [Tips and common mistakes](#tips-and-common-mistakes)
- [From India](#from-india)
- [Sources](#sources)

Companion files: [reading-list.md](reading-list.md) · [projects.md](projects.md) · [refresh-brief.md](refresh-brief.md)

## At a glance

### Timeline of the reorganizations (2025–2026)

| When | What happened | Source type |
|---|---|---|
| Apr 2025 | Llama 4 Scout and Maverick released as open-weight mixture-of-experts models; the larger Behemoth was previewed, not released | Meta blog |
| Apr–May 2025 | FAIR head Joelle Pineau departed; Rob Fergus returned from Google DeepMind to lead FAIR | Press reports |
| Jun 2025 | Meta invested in Scale AI; Alexandr Wang joined as Chief AI Officer. Meta Superintelligence Labs formed, with Nat Friedman leading products and applied research | Meta memo, press |
| Aug 19, 2025 | MSL split into four groups: **TBD Lab** (frontier models, led by Wang), **FAIR** (long-term research, Fergus), **Products and Applied Research** (Friedman), **MSL Infra** (Aparna Ramani) | Press reports of an internal memo |
| Oct 2025 | About 600 roles cut across FAIR, product AI and MSL Infra; TBD Lab was spared | Press reports |
| Nov 2025 | Yann LeCun confirmed he was leaving to found a world-models startup (AMI Labs, Paris) | Public statement, press |
| Mar 2026 | A new **Applied AI** engineering group formed to support MSL researchers, reported at about 6,500 engineers and PMs | Press reports |
| Apr 8, 2026 | **Muse Spark** released: a natively multimodal reasoning model, the first model from MSL, and **proprietary** (no weights) | Meta announcement, press |
| Aug 2026 | **Muse Glimmer 30B** released with open weights under Apache 2.0, "distilled from Muse Spark", authored by Meta Superintelligence Lab | [Hugging Face model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) (primary) |

> [!NOTE]
> Leaders, group names and headcounts in this table come from press coverage of internal memos. Treat them as reported facts. The model releases are primary-source facts. Meta's Muse Glimmer model card refers to an "Advanced AI Scaling Framework" that decides which models count as frontier, which shapes what gets open-weighted.

### What they build (as of September 2026)

| Area | Examples | Where it runs |
|---|---|---|
| Frontier models | Muse Spark (closed), Muse Glimmer (open, 30B dense, agentic, runs on a 24–32 GB GPU when 4-bit quantized); earlier Llama 1–4 | Meta AI assistant, apps, API partners |
| Research | FAIR: self-supervised vision (DINO, V-JEPA), segmentation (SAM), new architectures (Byte Latent Transformer, Large Concept Models, latent reasoning) | Papers and open code |
| Frameworks | PyTorch core, torch.compile / Inductor, FSDP2 and DTensor, torchtitan, torchao, ExecuTorch | Open source under the PyTorch Foundation |
| Ranking and ads ML | Recommendation models (DLRM, generative recommenders such as HSTU) at very large scale | Feed, Reels, ads |
| Hardware and infra | MTIA custom accelerators, very large GPU clusters, training reliability | Internal |

### Where they hire

Most MSL and FAIR roles are in the US (Menlo Park / Bay Area, New York, Seattle area), with FAIR and other research sites in Europe (Paris, London and others; check current listings). Meta opened a Bengaluru engineering office in 2025 and posted ML, software and hardware roles there; press coverage said it was being set up by enterprise engineering, not by MSL. See [From India](#from-india).

## Where core-AI people work

Titles you will see on metacareers: **Software Engineer, Machine Learning**; **Research Engineer**; **Research Scientist**; **Software Engineer, Systems / Infrastructure**; **AI kernels and performance** roles (for example under MTIA). Level is in the posting or set during the interview.

| Area | The work | What makes someone strong here | Repo prep |
|---|---|---|---|
| Pretraining (frontier models) | Data mixtures, architecture ablations, scaling-law fits, run stability, MoE, long context | Can plan a run on paper, predict loss vs. compute, debug a spike from curves | Labs [05](../../labs/05_transformer/README.md), [08](../../labs/08_scaling_laws/README.md), [09](../../labs/09_parallelism/README.md); [module 05](../../curriculum/05-pretraining.md) |
| Post-training and RL | SFT, preference optimization, RL with verifiable rewards, reward models, agentic RL, safety tuning | Implements DPO/GRPO from the objective, knows their failure modes, evaluates honestly | Labs [10](../../labs/10_lora/README.md), [11](../../labs/11_dpo/README.md), [12](../../labs/12_grpo/README.md); [module 06](../../curriculum/06-post-training.md) |
| Inference and efficiency | Quantization, speculative decoding, KV-cache layout, on-device models (ExecuTorch), serving cost | Predicts memory and tokens/s before measuring; ships a benchmarked speedup | Labs [07](../../labs/07_kv_cache_sampling/README.md), [13](../../labs/13_quantization/README.md), [14](../../labs/14_speculative_decoding/README.md); [module 07](../../curriculum/07-inference.md) |
| PyTorch and compilers | torch.compile (Dynamo captures Python bytecode into FX graphs; Inductor lowers them to Triton on GPU and C++/OpenMP on CPU), autograd, distributed (FSDP2, DTensor, pipeline and context parallel), torchao low-precision | Reads PyTorch internals, writes Triton, lands reviewed PRs with tests and benchmarks | Labs [01](../../labs/01_autograd/README.md), [06](../../labs/06_attention_kernels/README.md), [09](../../labs/09_parallelism/README.md); `train.py --compile` in lab 05 |
| Training infrastructure | Cluster scheduling, checkpointing, fault tolerance, collective communication, silent-data-corruption detection | Thinks in failure rates and recovery time, not just throughput | [Lab 03](../../labs/03_napkin_math/README.md), [lab 09](../../labs/09_parallelism/README.md) |
| FAIR research | Long-horizon research: self-supervised learning, world models, new architectures, reasoning | Publications, a clear research agenda, clean open code | [Module 08](../../curriculum/08-evaluation-and-research.md), [lab 15](../../labs/15_eval_stats/README.md) |
| Applied and ranking ML | Recommendation and ranking models, integrating models into products, the Applied AI group | Strong ML fundamentals plus production judgment; offline/online metric gaps | [Module 10](../../curriculum/10-applied-llm-systems.md), [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md) |

### A worked example of the thinking these teams expect

**Pretraining compute.** The Llama 3 paper reports a 405B-parameter model trained on about 15.6T tokens. With the $6ND$ rule:

$$C \approx 6 \times 405\times10^{9} \times 15.6\times10^{12} \approx 3.8\times10^{25}\ \text{FLOPs}$$

On 16,384 H100s at about 989 dense BF16 TFLOP/s each and 40% MFU (the paper reports 38–43%), that is $3.8\times10^{25} / (16384 \times 989\times10^{12} \times 0.4) \approx 5.9\times10^{6}$ s, about 68 days of pure compute. The paper also reports 419 unexpected interruptions over a 54-day snapshot, mostly hardware. At that scale checkpointing and fast restart are part of the model's cost, which is why infra roles matter. Practice this in [lab 03](../../labs/03_napkin_math/README.md).

**Inference memory.** Muse Glimmer's card lists 52 layers, a repeating [local, local, local, global] attention pattern with a 2,048-token window, 2 KV heads, and head dimension 128. In BF16 one layer stores $2 \times 2 \times 128 \times 2 = 1024$ bytes of K and V per token. At 131,072 tokens:

| Layout | KV cache |
|---|---|
| All 52 layers global | $52 \times 131072 \times 1024 \approx 7.0$ GB |
| 13 global + 39 local (window 2,048) | $13 \times 131072 \times 1024 + 39 \times 2048 \times 1024 \approx 1.75 + 0.08 = 1.83$ GB |

The local/global mix cuts long-context KV memory by about 3.8x. Add 4-bit weights ($29.6\text{B} \times 0.5$ bytes $\approx 15$ GB) and you can see why Meta targets a 24 GB card. If you can derive this in two minutes you are ready for the inference questions; [lab 07](../../labs/07_kv_cache_sampling/README.md) builds the cache.

## What they value

**Company values.** Meta's published values are: move fast; focus on long-term impact; build awesome things; live in the future; be direct and respect your colleagues; Meta, Metamates, me. In interviews this shows up as a bias for shipping measurable impact, and a performance culture with explicit, frequent reviews.

**Engineering culture.** Open source is central to how Meta builds and hires: PyTorch, FAIR's research code, and many Meta engineers review external pull requests in public. Someone who has landed PRs in `pytorch/pytorch`, `pytorch/ao` or `pytorch/torchtitan` has already been reviewed by the people who might interview them.

**Open-weights stance (as of September 2026).** It has changed and is now mixed:

- 2023–2025: LLaMA 1 released to researchers under a non-commercial license; Llama 2 through 4 released with weights under Meta's own community licenses (usable commercially with conditions, but not OSI open source).
- April 2026: the frontier model, Muse Spark, is proprietary. Press coverage attributes the change to concerns about competitors building on Meta's models and about safety.
- August 2026: Muse Glimmer 30B released with open weights under Apache 2.0, a more permissive license than any Llama release.
- PyTorch and most FAIR research code remain open source.

In practice: expect the frontier model to stay closed and smaller models to be open. Don't assume "Meta = open weights" in an interview; discuss the trade-off (safety, competitive distillation, ecosystem benefit) with specifics.

**Research and rigor.** The Llama 3 paper is a model of the evaluation discipline Meta's model teams publish: contamination analysis, confidence intervals on benchmarks, and detailed ablations. Evaluation claims without error bars read as weak here, as elsewhere.

## The hiring process

What Meta states publicly on metacareers, then what candidate reports add. Reports are labelled.

### Stages (official, metacareers)

1. **Recruiter conversation**: background, level, interests.
2. **Technical screen**: about 45 minutes; roughly 35 minutes coding in a shared online editor, with questions meant to be solved in 10–30 minutes each. Meta's prep pages say to talk through your approach, test your code and state complexity.
3. **Full loop**: coding, design (systems or product depending on background), and behavioral interviews. For machine-learning roles, Meta's ML full-loop prep page describes up to six 45-minute conversations with engineers, including ML design.
4. **Offer and team selection.**

### Rounds in detail

| Round | What it tests | Notes |
|---|---|---|
| Coding (1–2 rounds) | Data structures and algorithms; usually two problems per 45 minutes; clean, bug-free code under time pressure | **Reported:** since October 2025 Meta has run an **AI-enabled coding** round at some onsites, replacing one of the two classic coding rounds: about 60 minutes in CoderPad with a multi-file codebase and an AI assistant, in phases (fix bugs, implement, optimize). Early reports say it was piloted mainly at E4–E5. Ask your recruiter which format you will get. |
| ML system design | Design an ML system end to end: problem framing, data and labels, features or model inputs, model choice, training, offline and online evaluation, serving, monitoring | For ranking-heavy teams expect recommendation or integrity problems; for LLM teams expect training or serving pipelines. See [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md). |
| Systems design | Distributed systems for SWE-infra roles | For infra and PyTorch-distributed roles. |
| Behavioral | 45 minutes; ownership, conflict, ambiguity, growth, impact | Meta suggests STAR. **Reported:** interviewers probe for scope and measurable results that match the target level. |
| Research / project deep dive | For Research Engineer / Scientist roles: your papers or projects, and what you'd do next | **Reported;** the format varies by team. |

### Levels

Meta's engineering ladder uses IC levels commonly written E3 (entry) to E9. **Reported:** E4 is the usual target for about 2–4 years of experience, E5 is senior (expected to lead multi-quarter projects without supervision), and E6 is staff. Level is decided from interview performance, especially in design and behavioral rounds, not only from years. A strong 2-YOE candidate should prepare for E4 and have one story that shows E5-style scope.

### Team matching

**Reported:** for many E3–E5 software roles Meta hires into a general pool and matches you to a team after the loop, over a few weeks of conversations with managers. Specialized MSL, FAIR and PyTorch roles are more often hired directly by the team. If you want a specific team (for example PyTorch compiler), say so to the recruiter early and name it in team-matching calls.

## How to prepare with this repo

| Round | Do this | Target |
|---|---|---|
| Coding (classic) | [dsa-patterns.md](../../tracks/research-engineer/dsa-patterns.md): all 18 patterns; then mixed timed sets | Two mediums in 40 minutes, talking throughout, with tests |
| Coding (AI-enabled) | Practice on a multi-file repo: fix a planted bug in a lab, then extend it, using an assistant only for boilerplate. Explain every accepted suggestion. | You stay the one deciding; you catch the assistant's mistakes |
| ML coding and depth | [coding-interviews.md](../../tracks/research-engineer/coding-interviews.md); rewrite labs 05, 07 and 13 cores from memory | Attention, KV cache, INT8 quantization each in under 20 minutes |
| ML system design | [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md); design a feed ranker, an LLM serving stack, a post-training pipeline | A 45-minute answer with numbers (QPS, latency, memory, cost) |
| Napkin math | [Lab 03](../../labs/03_napkin_math/README.md) drills; the two worked examples above | Each estimate under 2 minutes |
| Behavioral | [career/stories.md](../../career/stories.md): 12–15 STAR stories; tag each with a Meta value | Two stories with quantified impact, one conflict, one failure |
| Deep dive | One project from [projects.md](projects.md) written up with error bars | Defend every number at 1, 5 and 20 minutes |
| Question bank | [question-bank.md](../../tracks/research-engineer/question-bank.md), inference and pretraining sections first | Answer without notes |

The [interview-loops.md](../../tracks/research-engineer/interview-loops.md) mock protocol applies: one mock every two weeks from month 2 of your preparation.

## Signals, ranked

Ranked by how directly each one predicts success in Meta's AI orgs and how achievable it is from India with an 8 GB GPU.

1. **Merged PRs in the PyTorch ecosystem**, reviewed by Meta engineers. As of September 2026 the active repos are `pytorch/pytorch` (compiler, distributed, core), `pytorch/torchtitan` (pretraining platform; LLM training at PyTorch is being consolidated there, including the new TitanRL stack announced in August 2026) and `pytorch/ao` (quantization and low-precision training). **torchtune is no longer maintained** (development stopped in July 2025) and **torchforge's development is paused** (its README points to torchtitan). Don't spend time on those two.
2. **A reproduction of a Meta model detail with rigorous evaluation**: for example, a small-scale ablation of an idea from the Llama 3 or multi-token-prediction papers, with confidence intervals and a clear negative result if that's what you found.
3. **Performance work with benchmarks**: a Triton kernel or torch.compile improvement with before/after numbers and a roofline explanation ([lab 06](../../labs/06_attention_kernels/README.md)).
4. **Publications** in main venues or strong workshops (matters most for FAIR and Research Scientist roles).
5. **Production ML impact at your current job**, quantified (latency, cost, revenue or quality metrics). This is what the behavioral and design rounds reward at E4/E5.
6. **Referrals** from Meta engineers who know your work, often a result of 1–3.
7. **Internships and residencies**: Meta's research internships mainly target PhD students. No current AI residency program for non-PhD engineers could be verified as of September 2026; check metacareers before planning around one.

## 90-day plan

For an ML engineer in Bengaluru with about two years of experience, an RTX 5060 (8 GB) and a full-time job: roughly 12–15 hours a week. DSA runs throughout, because the classic coding round is where strong ML candidates most often fail at Meta.

| Weeks | ML depth | PyTorch / open source | DSA (4–5 h/week) | Output by end |
|---|---|---|---|---|
| 1–2 | [Lab 03](../../labs/03_napkin_math/README.md); redo the Llama 3 compute and Muse Glimmer KV numbers above | Build PyTorch from source or set up an editable install; run the torchtitan debug model on CPU or GPU | Patterns 1–4 (two pointers, sliding window, binary search) | Napkin-math sheet in your journal |
| 3–4 | [Lab 05](../../labs/05_transformer/README.md) and `train.py`, with and without `--compile`; profile with `torch.profiler` | Read torch.compile docs; run `TORCH_LOGS=graph_breaks` on your model and fix a graph break | Patterns 5–8 (linked lists, trees) | A short note: compile speedup, graph breaks found, why |
| 5–6 | [Lab 13](../../labs/13_quantization/README.md), then torchao int8/int4 weight-only on a small model | Pick a torchao or torchtitan `good first issue`; comment before starting | Patterns 9–12 (graphs, backtracking, greedy, heaps) | First PR opened |
| 7–8 | [Lab 09](../../labs/09_parallelism/README.md) and [lab 14](../../labs/14_speculative_decoding/README.md) | Iterate on review; start a second, more substantial PR | Patterns 13–18 (DP, tries, union-find) | First PR merged or in final review |
| 9–10 | One project from [projects.md](projects.md), with error bars ([lab 15](../../labs/15_eval_stats/README.md)) | Share the write-up where PyTorch contributors discuss (forums, GPU MODE) | Mixed timed sets: two mediums in 40 minutes, 3x per week | Public write-up |
| 11–12 | ML system design: 4 practice designs (feed ranking, LLM serving, post-training pipeline, eval platform) | Keep one PR moving | Mock interviews (2 DSA, 1 design, 1 behavioral) | Résumé updated; referrals requested; applications sent |

> [!TIP]
> Treat day 90 as the start of applying, not the end of preparing. Your first PR, first write-up and 12 STAR stories are what a recruiter and a referrer need to see.

## Tips and common mistakes

**Tips**

- **Ask the recruiter for the exact loop.** Classic vs. AI-enabled coding, whether ML system design or product design, whether there is a research talk. Formats changed in 2025 and vary by role.
- **Talk and test in coding rounds.** State the approach and complexity before coding, then run through an example and edge cases. Silent correct code scores worse than narrated correct code.
- **Use numbers in design rounds.** Estimate QPS, latency budgets, memory and GPU counts. The worked examples above are the kind of reasoning that stands out.
- **Show level-appropriate scope.** For E4: owned a component end to end. For E5: led a cross-team project through ambiguity. Map each story to the level you're targeting.
- **Name the team in team matching.** If you want PyTorch compiler or torchtitan, your open-source work there is the best argument.

**Common mistakes**

- Preparing only ML depth and failing the DSA rounds. Meta's classic coding bar is high and weighted heavily.
- Contributing to archived or paused projects (torchtune, torchforge) instead of torchtitan, torchao or core PyTorch.
- Claiming Meta is "open source AI" without knowing that Muse Spark is closed and Muse Glimmer is open under Apache 2.0.
- Using the AI assistant in the AI-enabled round to write everything. Reports say interviewers evaluate your judgment: whether you verify, test and understand the code.
- Opening large unsolicited PRs. Start with a small fix, follow the contributing guide, and discuss bigger changes in an issue first.

## From India

**Offices (as of September 2026, verify).** Meta has offices in several Indian cities, including Gurugram, Mumbai, New Delhi, Hyderabad and Bengaluru. In February 2025 Meta posted about 40 roles for a new Bengaluru engineering office, including software, ML and hardware engineers and an engineering director; press coverage said enterprise engineering was setting it up. There is no public evidence of MSL, TBD Lab or FAIR research teams based in India. Check metacareers with the location filter set to India and the keywords "machine learning", "PyTorch" and "AI".

**Realistic routes**

| Route | How | Trade-off |
|---|---|---|
| Meta India role, then internal transfer | Join an ML or infra role in Bengaluru or Hyderabad; after about a year, look for internal transfer to a US or UK team | Slow, but internal transfers avoid the H-1B lottery (typically via L-1). Confirm transfer policy in your offer conversations. |
| Direct hire to US | Apply to US postings; Meta sponsors visas for many roles | H-1B is lottery-based, and a US$100,000 fee on certain new H-1B petitions was introduced in September 2025. Check the current rules. |
| Direct hire to UK / Europe | London, Paris and other sites | UK Skilled Worker visas have no lottery; Paris and London host research teams. |
| Open-source first | PyTorch contributions from India, then a referral | Needs no visa to start and builds O-1 style evidence over time. |

See [career/visa-and-relocation.md](../../career/visa-and-relocation.md) for the general rules, and confirm every visa detail with the recruiter and an immigration lawyer.

> [!WARNING]
> Compensation and headcount at MSL were widely reported to be unusual in 2025 (very large packages for a small number of senior researchers, followed by cuts elsewhere). Don't plan your career around press stories about individual offers. Plan around the roles actually posted for your level and location.

## Sources

Primary (verified September 2026):

- Muse Glimmer 30B model card (architecture, license, release date, "distilled from Muse Spark"): <https://huggingface.co/meta-models/Muse-Glimmer-30B>
- torchtitan README (active; TitanRL, Aug 2026; models; features): <https://github.com/pytorch/torchtitan>
- torchao README (active; float8/MXFP8 training, QAT): <https://github.com/pytorch/ao>
- torchtune "The future of torchtune" (development stopped, July 15, 2025): <https://github.com/meta-pytorch/torchtune/issues/2883>
- torchforge README (development paused; consolidated in torchtitan): <https://github.com/meta-pytorch/torchforge>
- The Llama 3 Herd of Models (compute, MFU, interruptions): <https://arxiv.org/abs/2407.21783>
- Meta Careers, preparing for your software engineering interview: <https://www.metacareers.com/blog/preparing-for-your-software-engineering-interview-at-meta/>
- Meta Careers, ML full-loop prep: <https://www.metacareers.com/ML-prep-onsite/>
- Meta Careers, technical screen prep: <https://www.metacareers.com/swe-prep-techscreen/>
- Meta Careers, hiring process: <https://www.metacareers.com/hiring-process/>

Press and reports (secondary; labelled "reported" above):

- MSL reorganization and October 2025 cuts: [TechTarget](https://www.techtarget.com/searchenterpriseai/news/366629774/Meta-restructures-AI-division-aiming-for-superintelligence), [Built In](https://builtin.com/artificial-intelligence/meta-superintelligence-reorg)
- Muse Spark as a closed model: [The Batch](https://www.deeplearning.ai/the-batch/with-muse-spark-meta-pivots-away-from-its-open-weights-llama-strategy), [AI News](https://www.artificialintelligence-news.com/news/meta-muse-spark-ai-model-open-source/)
- Muse Glimmer launch coverage: [CNBC](https://www.cnbc.com/2026/08/10/meta-muse-glimmer-open-weight-ai.html)
- AI-enabled coding interview: [interviewing.io](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples), [Hello Interview](https://www.hellointerview.com/blog/meta-ai-enabled-coding)
- Bengaluru office and hiring (Feb 2025): [Inc42](https://inc42.com/buzz/meta-eyes-india-expansion-to-hire-engineers-ai-talents-for-new-bengaluru-hub/)
