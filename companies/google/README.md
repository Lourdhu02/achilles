# How to get into Google DeepMind and Google's AI teams

A practical guide for becoming a core AI engineer (research engineer, research scientist or ML software engineer) at Google DeepMind, Google Research or Google Cloud AI, written for an ML engineer in India with about two years of experience.
Google differs from the other frontier labs in three ways that change how you prepare: its stack is JAX/XLA on TPUs, its hiring process is structured and committee-based, and **DSA rounds are real here**.
Facts about teams, processes, offices and visas are marked "as of September 2026". They change often, so re-verify them before you act.

## Contents

- [At a glance](#at-a-glance)
- [Where core-AI people work](#where-core-ai-people-work)
- [What they value](#what-they-value)
- [The hiring process](#the-hiring-process)
- [How to prepare with this repo](#how-to-prepare-with-this-repo)
- [Signals ranked](#signals-ranked)
- [90-day plan](#90-day-plan)
- [Tips and common mistakes](#tips-and-common-mistakes)
- [From India](#from-india)
- [Sources](#sources)

Companion files: [reading-list.md](reading-list.md) · [projects.md](projects.md) · [refresh-brief.md](refresh-brief.md) (a quarterly re-research prompt).

---

## At a glance

*As of September 2026. Org names and boundaries at Google move; check the pages linked in [Sources](#sources).*

| Org | What it is | Typical core-AI work |
|---|---|---|
| **Google DeepMind (GDM)** | Formed in April 2023 by merging DeepMind and the Brain team from Google Research. Builds the Gemini models and the open-weights Gemma family, plus science and generative-media work (for example AlphaFold, Veo, Imagen, Genie). Headquarters in London and Mountain View. | Pretraining, post-training and RL, multimodality, evals, serving, safety, science |
| **Google Research** | Google's research org outside GDM: algorithms and theory, systems, health, climate, applied ML across Google products. | Efficient ML, algorithms (e.g. speculative decoding came from here), applied research |
| **Google Cloud AI** | Serves Gemini and open models to enterprises through Vertex AI and Cloud TPU; builds agent and developer tooling. | Serving and inference at scale, fine-tuning services, evals for customers, applied/forward-deployed engineering |
| **ML infrastructure and TPU teams** | The teams behind TPUs, XLA, JAX, Pathways and the production training and serving stack. | Compilers, kernels (Pallas), sharding, large-scale training infra |

**What the models look like (as of September 2026).** Gemini is Google's frontier model line; its technical reports and model cards are published by GDM. The open-weights line is Gemma: Gemma 4 (technical report July 2026, [arXiv 2607.02770](https://arxiv.org/abs/2607.02770)) ships under Apache 2.0 in E2B, E4B, 12B, 26B-A4B (mixture-of-experts, 3.8B active) and 31B sizes, with hybrid sliding-window/global attention, a 262K vocabulary and up to 256K context ([model card](https://huggingface.co/google/gemma-4-E2B-it)). Gemma matters for you because it is the only Google model you can open, fine-tune and inspect, which makes it the natural substrate for a Google-facing portfolio.

**Where they hire.** GDM lists offices in London, the Bay Area, Bangalore, Cambridge (US), Montreal, New York City, Paris, Tokyo, Toronto and Zurich ([GDM careers](https://deepmind.google/careers/)). GDM roles are posted on its own careers page (Greenhouse-hosted job boards) and some also appear on the main Google Careers site, for example "Research Engineer, World Models, DeepMind". Google Research and Cloud AI roles are on [Google Careers](https://www.google.com/about/careers/applications/).

**Bengaluru research presence (as of September 2026).** Google's Bengaluru research work spans GDM and Google Research. Manish Gupta, a Senior Director at Google DeepMind based in Bengaluru, leads GDM research teams across India and Japan; Google Research also maintains an [India research lab page](https://research.google/teams/india-research-lab/). Naming has shifted since the 2023 merger (you will see "Google Research India" and "Google DeepMind India" in older and newer material), so search both names and read the lab pages before writing to anyone. Bengaluru researchers contribute to Gemini, and the lab has long worked on Indic languages and societal-impact problems. In July 2026 GDM announced its free [AI Research Foundations](https://www.skills.google/collections/deepmind) curriculum with NASSCOM and IISc Bengaluru at Google I/O Connect India.

> [!NOTE]
> Headcount in Bengaluru for core research roles is small compared with London and the Bay Area, and many India-based Google ML roles are SWE roles in product and Cloud teams. Plan for both routes: a direct research role in Bengaluru, and a Google SWE/ML role in India followed by an internal move.

## Where core-AI people work

The same person can be a "Research Engineer" at GDM, a "Software Engineer, Machine Learning" in Cloud, or a "Research Scientist" in Google Research. Pick targets by the work, not the title.

| Area | The work | Google-specific flavour | Repo labs and modules |
|---|---|---|---|
| **Pretraining** | Data mixtures, architecture, scaling laws, stable large runs | JAX on TPU pods; sharding with `jax.jit` + named shardings; DeepMind's Chinchilla paper set the compute-optimal recipe | [04](../../labs/04_tokenizer/README.md), [05](../../labs/05_transformer/README.md), [08](../../labs/08_scaling_laws/README.md), [09](../../labs/09_parallelism/README.md); [module 05](../../curriculum/05-pretraining.md) |
| **Post-training and RL** | SFT, preference learning, RL from verifiable rewards, reward models, reasoning ("thinking") models | Tunix is Google's open JAX post-training library (SFT, LoRA, DPO, PPO, GRPO) | [10](../../labs/10_lora/README.md), [11](../../labs/11_dpo/README.md), [12](../../labs/12_grpo/README.md); [module 06](../../curriculum/06-post-training.md) |
| **Serving and inference** | Latency/throughput, KV cache, batching, quantization, speculative decoding | Speculative decoding (Leviathan et al.) and MQA both came from Google; TPU serving economics differ from GPUs | [06](../../labs/06_attention_kernels/README.md), [07](../../labs/07_kv_cache_sampling/README.md), [13](../../labs/13_quantization/README.md), [14](../../labs/14_speculative_decoding/README.md); [module 07](../../curriculum/07-inference.md) |
| **TPU / infra** | Compilers, kernels, sharding, profiling, input pipelines, checkpointing | JAX, XLA, Pathways, Pallas (the JAX kernel language), Orbax, Grain, MaxText | [03](../../labs/03_napkin_math/README.md), [06](../../labs/06_attention_kernels/README.md), [09](../../labs/09_parallelism/README.md); [module 02](../../curriculum/02-compute-and-hardware.md) |
| **Evals** | Benchmarks, harnesses, statistics, long-context and multimodal evals | Heavy use of held-out and internal evals; error bars matter | [15](../../labs/15_eval_stats/README.md); [module 08](../../curriculum/08-evaluation-and-research.md) |
| **Safety and responsibility** | Frontier-risk evals, interpretability, red teaming, policy | GDM publishes a Frontier Safety Framework; Gemma Scope is its open SAE suite | [16](../../labs/16_interpretability/README.md); [module 09](../../curriculum/09-interpretability-and-safety.md) |
| **Applied** | Gemini in products, Cloud customers, agents, retrieval | Vertex AI, Gemini API, on-device Gemma | [17](../../labs/17_retrieval/README.md); [module 10](../../curriculum/10-applied-llm-systems.md) |

### Role families

| Role | Who gets it | What the interviews lean on |
|---|---|---|
| **Research Engineer (RE)** | Strong engineers with real ML depth; a PhD is not required | Coding (including DSA), ML fundamentals, maths/stats, research discussion |
| **Research Scientist (RS)** | Usually a PhD and a publication record in the area | Research talk and deep dive, ML theory, coding |
| **Software Engineer (SWE), incl. "SWE, Machine Learning"** | The largest intake; Google's standard SWE ladder | DSA-heavy coding, system design at senior levels, an ML domain round for ML roles |
| **Student Researcher** | BS/MS/PhD students (and some pre-academic researchers about to start a graduate programme) | Project fit with a host team; see [Signals ranked](#signals-ranked) |

> [!TIP]
> Without a PhD, the realistic research-side door is **Research Engineer at GDM** or **SWE (ML) on a Gemini, Cloud AI or ML-infra team**. RS roles without a PhD exist but need a publication record that already looks like one.

## What they value

From primary sources, as of September 2026:

- **Mission.** GDM's stated mission is to build AI responsibly to benefit humanity. Expect "why GDM?" to be probed; have a specific answer that names work you have read (a Gemma report section, a Gemini eval, a safety paper).
- **Google's AI Principles.** Revised in February 2025 and now organised under three headings: bold innovation, responsible development and deployment, and collaborative progress. The revision removed the earlier list of "applications we will not pursue". Know what changed and have a considered view; interviewers value a thoughtful position over a slogan. Prep with [career/stories.md](../../career/stories.md#values-and-mission-prep).
- **Frontier safety.** GDM publishes a Frontier Safety Framework (critical capability levels, evaluations, mitigations). Read the current version before a safety-adjacent interview.
- **Google's hiring attributes.** Google's hiring pages describe structured interviews that assess **general cognitive ability** (how you reason through a new problem), **role-related knowledge**, **leadership** (including influence without authority) and **"Googleyness"** (intellectual humility, comfort with ambiguity, collaboration, bias to action).
- **Engineering culture.** Google values code that others can read and maintain (it is known for code review and readability culture), design documents before large changes, and measurable impact. A portfolio with clean, tested code and short design docs fits this culture better than notebooks.

## The hiring process

> [!IMPORTANT]
> Everything below is either **publicly described by Google** or **widely reported by candidates** (labelled). Formats vary by team, level and year. Ask your recruiter for the exact loop and whether coding runs in an executable environment.

### Google (SWE, SWE-ML, Research Scientist on Google Careers)

**Publicly described by Google** ([How we hire](https://www.google.com/about/careers/applications/how-we-hire/)): apply online (Google limits how many roles you can apply to within a short window; check the current limit on the careers site), recruiter conversation, one or more short technical assessments or screens, then a set of **structured interviews** in which each interviewer uses consistent questions and a rubric, scored against the four attributes above. Google has also described, in *Work Rules!* (Laszlo Bock, 2015) and elsewhere, that hiring decisions are made by a **hiring committee** of people who did not interview you, based on the written interview feedback, rather than by a single hiring manager.

**Widely reported (as of September 2026):**

| Stage | Typical shape |
|---|---|
| Recruiter screen | Background, level, location, timelines |
| Technical phone screen(s) | One or two 45-minute DSA problems; for ML roles sometimes an ML fundamentals screen |
| Onsite loop (usually virtual) | About 4–5 interviews of ~45 minutes: 2–3 coding/DSA; one "Googleyness and leadership" behavioural; system design at senior levels (roughly L5+); for ML roles an ML design or ML knowledge round |
| Hiring committee | Reviews the packet (feedback, résumé, screens) and recommends hire/no-hire and level |
| Team matching | For generalist SWE hiring, you talk with teams after committee approval; it can take weeks. For team-specific roles it is folded in earlier |
| Offer review | Compensation and a final senior review |

A two-year engineer is typically interviewed at L3 or L4 (reported). Coding is often done in a shared editor where you may not be able to run the code, so you must be able to reason about correctness without a debugger.

### Google DeepMind (Research Engineer / Research Scientist)

**Widely reported** (Glassdoor, blogs from successful candidates such as [Aleksa Gordić's account](https://gordicaleksa.medium.com/how-i-got-a-job-at-deepmind-as-a-research-engineer-without-a-machine-learning-degree-1a45f2a781de) and [David Stutz's 2019 internship prep](https://davidstutz.de/how-i-prepared-for-deepmind-and-google-ai-research-internship-interviews-in-2019/); older accounts may be out of date):

1. **Recruiter call.**
2. **"Quiz" round.** Rapid, broad questions across maths (linear algebra, calculus, probability), statistics, CS fundamentals and ML. The point is breadth and fluency: you are expected to answer quickly and correctly, not to derive for twenty minutes.
3. **Coding rounds.** Often two, Google-style DSA problems at medium-to-hard level. Several reports say the code is expected to **run** (e.g. in CoderPad) by the end.
4. **ML rounds.** ML depth: derivations, debugging, design of experiments, discussion of your past work. Reports often call these the hardest part; they are usually gated on passing coding.
5. **Final/team rounds.** Conversations with team leads and the potential manager; a two-way assessment of fit with the team's plans and culture.

### What DSA looks like here, concretely

Google is the one company in this set where you should budget serious DSA time. Treat it as a skill with maintenance, not a cram:

- **Coverage:** all 18 patterns in [dsa-patterns.md](../../tracks/research-engineer/dsa-patterns.md). Reports skew toward graphs (BFS/DFS/topological sort), trees, dynamic programming, heaps, intervals and strings, but do not skip any pattern.
- **Standard:** a LeetCode medium in about 20–25 minutes, narrated, with complexity analysis and edge cases, written cleanly enough that you could submit it for code review.
- **Both modes:** practise in a plain editor without running code (Google SWE style) and in an executable pad with tests (GDM style).
- **Follow-ups:** Google interviewers commonly extend a problem ("now the input doesn't fit in memory", "now it's a stream", "now make it concurrent"). Practise the second and third part of every problem.

## How to prepare with this repo

Map every round to a file. Do the work in this order: labs first (they feed everything), DSA in parallel from day one.

| Round | Prepare with | Target |
|---|---|---|
| DSA coding | [dsa-patterns.md](../../tracks/research-engineer/dsa-patterns.md), [coding-interviews.md § DSA](../../tracks/research-engineer/coding-interviews.md#dsa) | ~150 problems understood, mediums in ≤ 25 min |
| GDM quiz | [module 01 math](../../curriculum/01-math.md), [module 03](../../curriculum/03-deep-learning.md), [question-bank.md](../../tracks/research-engineer/question-bank.md) | answer 30 rapid questions in 30 min with ≥ 80% correct |
| ML coding | [coding-interviews.md drills](../../tracks/research-engineer/coding-interviews.md#ml-coding-drills); rewrite each drill in **JAX** as well as PyTorch | attention, RoPE, sampling, AdamW, DPO loss from memory |
| ML knowledge / deep dive | [question-bank.md](../../tracks/research-engineer/question-bank.md), your journal write-ups | three "why"s deep on every résumé line |
| Napkin math / systems | [lab 03](../../labs/03_napkin_math/README.md) redone with TPU v5e numbers (below), [module 02](../../curriculum/02-compute-and-hardware.md) | each estimate in < 2 min |
| ML system design | [ml-system-design.md](../../tracks/research-engineer/ml-system-design.md) | serving on TPUs, an RL pipeline, an eval platform |
| Googleyness and leadership | [career/stories.md](../../career/stories.md), [values prep](../../career/stories.md#values-and-mission-prep) | 12–15 STAR stories with numbers |
| Round overview and mocks | [interview-loops.md](../../tracks/research-engineer/interview-loops.md) | one mock every two weeks |

### The TPU napkin-math you should be able to do cold

Redo lab 03's roofline with TPU numbers. From [How to Scale Your Model](https://jax-ml.github.io/scaling-book/roofline/) (verify against the [Cloud TPU docs](https://cloud.google.com/tpu/docs/v5e) before quoting):

| Chip | HBM | HBM bandwidth | bf16 FLOP/s | Critical intensity |
|---|---|---|---|---|
| TPU v5e | 16 GB | 8.2e11 B/s | 1.97e14 | ≈ 240 FLOPs/byte |
| TPU v5p | 96 GB | 2.8e12 B/s | 4.59e14 | ≈ 164 FLOPs/byte |
| TPU v6e (Trillium) | 32 GB | 1.6e12 B/s | 9.20e14 | ≈ 575 FLOPs/byte |

For a bf16 matmul of activations $[B, D]$ by weights $[D, F]$ with $B \ll D, F$:

$$
\text{intensity} = \frac{2BDF}{2BD + 2DF + 2BF} \approx B
$$

So the matmul is compute-bound only when the per-replica **token** batch $B$ exceeds the chip's critical intensity: about 240 tokens on v5e. Consequences you should be able to state in an interview:

- **Decode is memory-bound** at small batch because each step reads every weight for only $B$ tokens. On v5e, batching to roughly 240 concurrent sequences is what it takes to approach the compute roof (ignoring KV-cache reads, which make it worse).
- **Model sharding does not change the threshold**: sharding across $n$ chips scales both FLOPs and bandwidth by $n$, so the critical batch is per replica.
- **Memory check:** a 7B model in bf16 is 14 GB of weights, which almost fills a 16 GB v5e chip before any KV cache, so you serve it across several chips or quantize it.

### JAX: the one skill gap most PyTorch engineers have

Most candidates from industry know PyTorch only. Google's research stack is JAX. You do not need to be a JAX expert to get hired, but fluency is a strong signal and makes the first months much easier.

| Concept | PyTorch habit | JAX equivalent | Where the repo helps |
|---|---|---|---|
| Autodiff | `loss.backward()` mutates `.grad` | `jax.grad(loss_fn)(params, batch)` returns gradients as a new pytree | [lab 01](../../labs/01_autograd/README.md): you already built reverse mode |
| Compilation | eager by default, `torch.compile` optional | `jax.jit` traces once per input shape/dtype; Python side effects run only at trace time | recompiles on every new sequence length are the first bug you will hit |
| Batching | write the batch dimension by hand | `jax.vmap` adds it for you | rewrite lab 15's bootstrap with `vmap` |
| Randomness | global RNG state | explicit keys: `jax.random.split(key)` | makes runs reproducible by construction |
| Parallelism | DDP/FSDP wrappers | a `Mesh`, `NamedSharding` and `jit` (GSPMD), or `shard_map` for manual collectives | [lab 09](../../labs/09_parallelism/README.md): ring all-reduce and tensor parallelism are the same maths |
| Kernels | Triton | Pallas (`jax.experimental.pallas`), which targets TPU and GPU | [lab 06](../../labs/06_attention_kernels/README.md): online softmax and tiling carry over |
| Models / optimizers | `nn.Module`, `torch.optim` | Flax (NNX API) and Optax | [lab 02](../../labs/02_training_core/README.md): your AdamW is a few lines of Optax |

> [!WARNING]
> JAX's GPU builds are Linux-only. On Windows, use WSL2 for local JAX on the RTX 5060, and confirm your `jaxlib` CUDA build supports Blackwell consumer GPUs before debugging anything else. If it does not, use JAX on CPU for correctness tests and on Kaggle/Colab or TPU Research Cloud for real runs.

## Signals ranked

From strongest to weakest for someone without a PhD, as of September 2026. The generic ladder is in the [research-engineer track](../../tracks/research-engineer/README.md#signal-ladder-climb-it-in-order); these are the Google-specific versions.

1. **Merged contributions to Google's open ML stack.** JAX, Flax, Optax, Orbax, Grain, MaxText, Tunix, the Gemma libraries, or Gemma support in vLLM/SGLang/llama.cpp/Hugging Face. A substantive merged PR (a bug fix with a test, a performance fix with a benchmark, a missing feature) is reviewed by Google engineers, which makes it both a signal and a referral path. Docs-only PRs count for little.
2. **A reproduced or extended Google/GDM result with a clean write-up.** For example, a small-scale Chinchilla reproduction in JAX, a Gemma fine-tune with proper error bars, or a speculative-decoding speedup measured against the paper's analysis. See [projects.md](projects.md).
3. **Publications or strong workshop papers** in the area of the team you target. For RS roles this is close to required; for RE roles a single solid first-author workshop paper or arXiv preprint with released code is a real differentiator.
4. **Kaggle results.** Google owns Kaggle, and Gemma-themed competitions run there. A gold medal or a top finish in an LLM competition, with a public write-up of what worked and what did not, is recognised. A bronze medal from a tabular competition is not a core-AI signal.
5. **Compute you earned: TPU Research Cloud (TRC).** TRC gives accepted researchers free Cloud TPU quota for a limited period. You express interest at [sites.research.google/trc](https://sites.research.google/trc/about/), invitations go out on a rolling basis, and you are expected to share the resulting work publicly (paper, code or blog) and give feedback. The TPUs are free; other Google Cloud services you use alongside them (VMs, storage) are not, so set a budget alert. Using TRC well is also practice in exactly the stack GDM uses.
6. **Programmes.** The [Student Researcher Program](https://deepmind.google/student-researcher-program/) places BS/MS/PhD students (and some pre-academic researchers about to start a graduate programme) with GDM, Google Research and Cloud teams for roughly 12–24 weeks; openings are posted per cycle on Google Careers. Google PhD Fellowships support PhD students. If you are working rather than studying, these matter only if you plan to go back to school; do not plan around a residency programme unless you see one currently advertised.
7. **Referrals.** A referral from a Google engineer gets your application read; it does not lower the bar. The best referrals come from people who have reviewed your code (signal 1) or read your work (signal 2).
8. **Certificates and course completions.** Low signal on their own. The [AI Research Foundations](https://github.com/google-deepmind/ai-foundations) curriculum is worth doing for its labs, but list the project you built, not the certificate.

## 90-day plan

For a 2-YOE ML engineer in Bengaluru with a full-time job, ~20–25 hours a week, an 8 GB GPU and free Kaggle/Colab access. Adjust the pace; do not drop the DSA line.

| Weeks | Labs and ML | JAX / Google stack | DSA (every week) | Output |
|---|---|---|---|---|
| 1–2 | Labs [01](../../labs/01_autograd/README.md), [02](../../labs/02_training_core/README.md), [03](../../labs/03_napkin_math/README.md) | JAX basics: `grad`, `jit`, `vmap`, PRNG keys; redo lab 01's MLP in JAX | Patterns 1–4 (two pointers, sliding window, binary search): ~12 problems | Lab 03 table redone for TPU v5e |
| 3–4 | Labs [04](../../labs/04_tokenizer/README.md), [05](../../labs/05_transformer/README.md) | Flax NNX + Optax; start [project 1](projects.md#1-lab-05-gpt-in-jax-on-a-free-tpu) | Patterns 5–9 (linked lists, trees, graphs): ~15 problems | GPT in PyTorch passing tests; JAX port started |
| 5–6 | Labs [06](../../labs/06_attention_kernels/README.md), [07](../../labs/07_kv_cache_sampling/README.md) | Finish project 1; read How to Scale Your Model ch. 1–4 | Patterns 10–12 (backtracking, greedy, heaps): ~15 problems | JAX GPT matches PyTorch loss curve; apply to TRC |
| 7–8 | Labs [08](../../labs/08_scaling_laws/README.md), [09](../../labs/09_parallelism/README.md) | `jax.sharding` on a TPU slice or 8 fake CPU devices; How to Scale ch. 5–7 | Patterns 13–16 (subsets, k-way merge, DP): ~18 problems | Short scaling-law note with fits and CIs |
| 9–10 | Labs [10](../../labs/10_lora/README.md), [15](../../labs/15_eval_stats/README.md) | [Project 2](projects.md#2-a-rigorous-gemma-fine-tune): Gemma fine-tune with error bars | Patterns 17–18 + mixed sets; first timed mock | Project 2 write-up and repo |
| 11–12 | Labs [11](../../labs/11_dpo/README.md) or [14](../../labs/14_speculative_decoding/README.md) (pick by target team) | Open a first small PR to a JAX-stack or Gemma-tooling repo | Mixed sets, 2 timed mocks, Googleyness stories drafted | Résumé updated; applications and referral asks out |
| 13 | Review | GDM quiz drill: 100 rapid questions from [question-bank.md](../../tracks/research-engineer/question-bank.md) and modules 01–03 | 2 more mocks | Interview-ready checklist below |

DSA totals over the 90 days: about 80–100 problems, pattern-first. Continue to ~150 at 3–4 hours a week while you interview.

**Ready-to-apply checklist.** You can: solve a medium DSA problem in 25 minutes without running code; write causal attention with GQA and RoPE from memory in both PyTorch and JAX; compute a TPU v5e roofline threshold and a 7B model's serving memory in under two minutes; explain one project at 1, 5 and 20 minutes; tell 12 stories with numbers.

## Tips and common mistakes

**Tips**

- **Apply to specific roles, not "Google".** For GDM, read the job description and name the team's work in your cover note. For Google SWE, apply to at most a handful of roles that fit your level; recruiters see the whole list.
- **Write your résumé in impact form.** "Cut p95 latency from 800 ms to 310 ms by X" beats "worked on latency". Google's committee reads written feedback against your résumé, so every claim must survive probing.
- **Narrate in DSA rounds.** Interviewers write down your reasoning for the committee. State the brute force, its complexity, the better approach, then code. Test with a small example by hand at the end.
- **For the GDM quiz, speed matters.** Practise saying the answer first and the one-line reason second (e.g. "the variance of the sample mean is $\sigma^2/n$ because the samples are independent").
- **Know Google's papers as primary sources.** When you discuss MQA, speculative decoding or Chinchilla, you are talking to people whose colleagues wrote them. Be exact; see [reading-list.md](reading-list.md).
- **Use Gemma as your research substrate.** It is open, well documented, supported by major inference engines, and the thing GDM engineers look at daily.
- **Re-apply.** Reports consistently say candidates can re-interview after a cooling-off period (often cited as 6–12 months; confirm with your recruiter). Many Google engineers were rejected at least once.

**Common mistakes**

| Mistake | Why it hurts at Google | Fix |
|---|---|---|
| Skipping DSA because "labs don't do LeetCode" | Coding rounds are gated; strong ML depth does not rescue a failed coding screen | 3–4 h/week from week 1 |
| Only ever running code | Many Google rounds do not let you execute | Half your practice in a plain editor |
| Portfolio in notebooks only | Google's culture is code review and readability | Packaged repo, tests, a README with results and error bars |
| Treating JAX as optional for GDM | Not strictly required, but its absence shows in ML coding and infra discussions | Port one lab to JAX (project 1) |
| Vague "why Google" | Behavioural and team rounds probe mission fit | Cite a specific report section or result and what you would do next |
| Quoting TPU or model numbers from memory | Interviewers know the real ones | Quote with sources, or derive from first principles |
| Waiting passively for team matching | It can stall for weeks | Ask the recruiter which teams are hiring; reach out to teams whose work you know |

## From India

*Volatile: as of September 2026. Verify every rule with official sources before deciding.* General visa guidance is in [career/visa-and-relocation.md](../../career/visa-and-relocation.md).

- **Offices.** Google has engineering offices in several Indian cities, including Bengaluru and Hyderabad. Core research (GDM, Google Research) is concentrated in Bengaluru; Cloud AI and product ML roles are more widely spread. Use the location filter on Google Careers and the GDM careers page.
- **Route A: research role in Bengaluru.** Small intake, high bar, usually RE/RS. Your strongest levers are signals 1–3 above and a clear match to the Bengaluru team's published work.
- **Route B: Google SWE (ML) in India, then move internally.** Larger intake; DSA-heavy loop. Once inside, internal transfers to GDM or Research teams are possible but competitive and typically require time in role and a team willing to take you (reported; internal policies are not public). Build visibility by contributing to shared ML infrastructure and reading internal research.
- **Moving abroad from Google India.** The standard US route after an internal transfer is the L-1 visa, which by US law requires at least one continuous year of employment with the company abroad within the preceding three years. UK and Swiss moves use employer-sponsored work permits. Timelines and approvals change; ask your recruiter and the company's mobility team, and check [career/visa-and-relocation.md](../../career/visa-and-relocation.md).
- **Applying directly to London or the Bay Area from India.** Possible; GDM hires internationally and sponsors visas for many roles, but the bar is global and US hiring depends on H-1B rules that changed substantially in 2025. Confirm sponsorship for the specific role in the first recruiter call.
- **Local community.** Google for Developers and Kaggle events in Bengaluru, and university collaborations (e.g. with IISc), are real ways to meet Google researchers. Show up with work to discuss, not a résumé to hand over.

## Sources

Checked in September 2026. Where a primary page could not be fetched directly, the claim was cross-checked against official snippets or multiple independent reports.

**Google and Google DeepMind (primary)**
- Google DeepMind careers: https://deepmind.google/careers/
- GDM Student Researcher Program: https://deepmind.google/student-researcher-program/
- Google Careers and How we hire: https://www.google.com/about/careers/applications/ · https://www.google.com/about/careers/applications/how-we-hire/
- Google Research careers and India research lab: https://research.google/careers/ · https://research.google/teams/india-research-lab/
- TPU Research Cloud: https://sites.research.google/trc/about/ · FAQ https://sites.research.google/trc/faq/
- Google AI Principles: https://ai.google/principles/
- AI Research Foundations: https://www.skills.google/collections/deepmind · labs https://github.com/google-deepmind/ai-foundations
- How to Scale Your Model (Austin et al., Google DeepMind, 2025): https://jax-ml.github.io/scaling-book/
- Gemma 4 model card and report: https://huggingface.co/google/gemma-4-E2B-it · https://arxiv.org/abs/2607.02770

**Open-source stack**
- JAX https://github.com/jax-ml/jax · Flax https://github.com/google/flax · Optax https://github.com/google-deepmind/optax · Orbax https://github.com/google/orbax · Grain https://github.com/google/grain · MaxText https://github.com/AI-Hypercomputer/maxtext · Tunix https://github.com/google/tunix · Gemma (JAX) https://github.com/google-deepmind/gemma · Gemma (PyTorch) https://github.com/google/gemma_pytorch · Penzai https://github.com/google-deepmind/penzai

**Reports (secondary; label as such when you repeat them)**
- Laszlo Bock, *Work Rules!* (2015), on Google's hiring committees and structured interviews.
- Aleksa Gordić, "How I got a job at DeepMind as a research engineer": https://gordicaleksa.medium.com/how-i-got-a-job-at-deepmind-as-a-research-engineer-without-a-machine-learning-degree-1a45f2a781de
- David Stutz, "How I prepared for DeepMind and Google AI research internship interviews in 2019": https://davidstutz.de/how-i-prepared-for-deepmind-and-google-ai-research-internship-interviews-in-2019/
- Glassdoor interview reports for GDM Research Engineer: https://www.glassdoor.com/Interview/Google-DeepMind-Research-Engineer-Interview-Questions-EI_IE1596815.0,15_KO16,33.htm
- Business Standard (August 2026) on Bengaluru researchers' contributions to Gemini: https://www.business-standard.com/technology/artificial-intelligence/indians-involved-in-solving-world-s-toughest-problems-google-deepmind-126080200459_1.html

See also: [companies/more-labs.md](../more-labs.md) for other labs, and the sibling guides for [Anthropic](../anthropic/README.md), [OpenAI](../openai/README.md), [Meta](../meta/README.md) and [NVIDIA](../nvidia/README.md).
