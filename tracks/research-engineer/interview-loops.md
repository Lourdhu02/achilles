# Interview loops (the durable parts)

What research-engineer loops look like stage by stage, what each round type scores, and how to practise for it. Company-specific formats change every year: read the [company guides](../../companies/README.md) and [market-intel.md](../../career/market-intel.md#interview-formats-durable-patterns), and ask the recruiter. The round *types* below have been stable for years.

## Contents

- [Loop anatomy](#loop-anatomy)
- [Round types](#round-types): [practical coding](#1-practical-coding) · [ML coding](#2-ml-coding) · [ML debugging](#3-ml-debugging) · [napkin math and systems](#4-napkin-math-and-systems) · [ML system design](#5-ml-system-design) · [deep dive](#6-research-or-project-deep-dive) · [values or mission](#7-values-or-mission) · [behavioral](#8-behavioral) · [DSA](#9-dsa)
- [Rounds that allow AI coding tools](#rounds-that-allow-ai-coding-tools)
- [Mock-interview protocol](#mock-interview-protocol)
- [Logistics: interviewing across time zones from India](#logistics-interviewing-across-time-zones-from-india)
- [The two failure modes to avoid](#the-two-failure-modes-to-avoid)

## Loop anatomy

Typical stages as of September 2026. The order, the number of rounds and the timing vary by company, level and season; each company guide describes its current process: [Anthropic](../../companies/anthropic/README.md#the-hiring-process), [OpenAI](../../companies/openai/README.md#the-hiring-process), [Google](../../companies/google/README.md#the-hiring-process), [Meta](../../companies/meta/README.md#the-hiring-process), [NVIDIA](../../companies/nvidia/README.md#the-hiring-process), [more labs](../../companies/more-labs.md).

| stage | typical format | what gets decided | your job |
|---|---|---|---|
| Recruiter screen | a 20–30 minute call | fit with the role, logistics, rough level | a crisp story; the loop format in writing |
| Technical screen | 1–2 rounds of 45–90 minutes of practical or ML coding in a shared editor; some companies use an automated assessment | whether you code at the bar | clean, tested, narrated code |
| Take-home or work trial (some companies) | a time-boxed task of a few hours, or a paid trial of a day or more | how you work with real ambiguity | scope, tests, a short README of decisions |
| Onsite or virtual loop | 4–6 rounds of 45–60 minutes, in one day or spread over several | hire or not, and often the level | steady signal in every round |
| Debrief or hiring committee | interviewers' written feedback reviewed together | the decision | nothing; your answers are already on paper |
| Team matching (some companies) | calls with hiring managers | which team | interview them too: ask about the work |
| References | 2–3 calls or forms | confirmation | brief your referees |
| Offer | a call, then a written offer | the terms | negotiate ([offers](../../career/offers-and-negotiation.md)) |

**Timelines (as of September 2026; planning figures, not promises):** plan for roughly 4–8 weeks from the recruiter screen to an offer, and longer when team matching or committee review is a separate step. Ask the recruiter for the expected timeline at each stage, and run several processes in parallel ([apply in waves](../../career/applications.md#apply-in-waves)) so that offers overlap.

- **Recruiter screen.** Prepare a 60-second introduction that ends with your strongest link, a reason for this team, and your logistics (location, notice period, visa). Defer compensation numbers until the level is clear ([principles](../../career/offers-and-negotiation.md#principles)). Use the [recruiter-call script](../../career/applications.md#recruiter-call-script) and ask the [format questions](#logistics-interviewing-across-time-zones-from-india).
- **Take-home.** Ask how it is evaluated and whether AI coding tools are allowed. Keep to the time box and write down what you would do with more time. Submit tests and a README that explains your decisions and trade-offs; reviewers read both.
- **Team matching.** Ask what the team shipped last quarter, how success is measured, and what a new hire's first project would be. [Meta](../../companies/meta/README.md#team-matching) and [Google](../../companies/google/README.md#the-hiring-process) describe two versions of this step.
- **References.** Choose people who saw your technical work closely, and send them the job description and the two or three projects you would like them to mention.

## Round types

Each round has its format, a rubric, generic example prompts (not taken from any company), a strong-answer outline for one prompt, a preparation plan and the common failure modes. Every round scores both what you produce and how you get there; [coding-interviews.md](coding-interviews.md#how-these-rounds-are-scored) explains the coding rubric in more detail.

### 1. Practical coding

**Format:** 60–90 minutes; one problem that grows in stages (build a class, extend it, handle edge cases or concurrency), usually in a real IDE where you run the code.

| scored | strong | weak |
|---|---|---|
| Correctness | runs; edge cases raised and tested | the happy path only |
| Structure | small functions; stage 1 code survives stage 3 | a rewrite for every new requirement |
| Progress | core done early, most extensions reached | stuck in stage 1 |
| Communication | interface and plan stated first; trade-offs narrated | silent typing |

**Example prompts**
- An in-memory key-value store with per-key TTL; then transactions with nested rollback; then snapshots.
- A request micro-batcher for a model server: flush when the batch is full or the oldest request has waited too long; then per-request deadlines; then priorities.
- A parser for a stream of server-sent events from a model API; then reconnection with exponential backoff and jitter.

**Strong answer outline (micro-batcher)**
1. Pin down the interface: `submit(request) -> future`, `max_batch`, `max_wait_ms`, and what happens at shutdown.
2. Write the simplest correct version: a queue and a worker that waits on a condition variable (or an asyncio event), flushes on size or timeout, and resolves the futures.
3. Test before extending: a full batch flushes at once; a lone request flushes at the timeout; an empty queue does nothing.
4. Extend: expired requests fail before batching; priorities replace the FIFO with a heap keyed on (priority, arrival time).
5. Close with what changes on a real GPU server: padding waste (bucket by length), backpressure (a bounded queue), and why continuous batching replaces this for decode.

**Prep:** the [practical engineering drills](coding-interviews.md#practical-engineering-drills), each rewritten from memory in 60 minutes. Once a week, have a friend add a new requirement every 15 minutes.

**Failure modes:** over-engineering stage 1; no tests until the end; not running the code; going silent when stuck; ignoring a hint.

### 2. ML coding

**Format:** 45–60 minutes; implement a core ML component from scratch in Python with NumPy or PyTorch, usually without references, sometimes checked against a library implementation.

| scored | strong | weak |
|---|---|---|
| Correctness | shapes, masks and numerics right; checked against a reference | an off-by-one mask; softmax over the wrong axis |
| Numerical care | log-sum-exp, fp32 accumulation, `-inf` masking | `log(softmax(x))`; overflow in fp16 |
| Clarity | shapes written as comments; small functions | one 60-line function |
| Depth | complexity, memory, what changes on a GPU | stops at "it runs" |

**Example prompts**
- Causal multi-head self-attention with an optional KV cache for incremental decoding.
- Top-p sampling with temperature for a batch of logits.
- The DPO loss from policy and reference log-probabilities, with response masks.

**Strong answer outline (attention with a KV cache)**
1. State the shapes: `x: (B, T, C)`; q, k and v reshaped to `(B, H, T, hd)` with `hd = C // H`.
2. Scores are `q @ k.transpose(-2, -1) / sqrt(hd)`; the causal mask accounts for the cache offset (query $i$ sees keys up to `past_len + i`); softmax in fp32; output projection.
3. The cache: concatenate the new k and v along time and return the updated cache.
4. The test: decoding T tokens one at a time must match one full forward pass within tolerance.
5. The cost: $O(T^2 d)$ for the full pass against $O(Td)$ per cached step; the cache holds $2 \times \text{layers} \times \text{KV heads} \times hd \times T$ elements per sequence, which is why GQA's shared KV heads matter.

**Prep:** the [18 ML coding drills](coding-interviews.md#ml-coding-drills), each in 15–30 minutes from memory, with a bug log; labs [05](../../labs/05_transformer/README.md), [07](../../labs/07_kv_cache_sampling/README.md) and [11](../../labs/11_dpo/README.md).

**Failure modes:** shape bugs; masking the wrong axis; forgetting the $1/\sqrt{d}$ scale; no test; the unstable `log(softmax(x))`.

### 3. ML debugging

**Format:** 45–60 minutes; a training script, notebook or set of loss curves with several planted bugs, or a description of a failing run. Find the bugs, fix them and explain each symptom.

| scored | strong | weak |
|---|---|---|
| Method | hypotheses ranked by likelihood and cost to check | reading line by line, hoping |
| Symptom knowledge | maps curves to causes (flat at $\ln V$, suspiciously low loss, NaN after a few thousand steps) | guesses |
| Verification | confirms each fix, for example by overfitting one batch | stops at the first bug |
| Communication | says what each check rules out | silent edits |

**Example prompts**
- A small language model's loss sits at about 10.8 from the first step and never moves. The vocabulary has 50,257 tokens.
- Validation loss is far below training loss from step one.
- A bf16 run trains well for a few thousand steps, then the loss spikes and becomes NaN.

**Strong answer outline (loss flat at 10.8)**
1. Recognise the number: $\ln 50{,}257 \approx 10.82$ is the loss of a uniform prediction, so nothing is learning.
2. Check the cheap causes first: does the optimizer hold this model's parameters (or was the model rebuilt after the optimizer was created)? Is the learning rate non-zero after warmup starts? Are gradients non-zero (per-layer norms)? Are `zero_grad`, `backward` and `step` in the right order? Is the loss detached?
3. Fix it, then prove it: overfit a single batch to near-zero loss.
4. Keep going: these rounds often hide more than one bug, so check label shifting, the causal mask and padding in the loss next.

**Prep:** the [debugging playbook](../../curriculum/03-deep-learning.md#7-debugging-a-training-run-the-playbook); plant three bugs in one of your labs for a friend and swap; [question bank §2](question-bank.md#2-deep-learning-and-optimization).

**Failure modes:** reading without hypotheses; fixing the first bug and stopping; changing several things at once; not verifying.

### 4. Napkin math and systems

**Format:** 20–45 minutes on its own, or woven into design and deep-dive rounds. Estimate compute, memory, time or cost out loud.

| scored | strong | weak |
|---|---|---|
| Formulas | $6ND$ for training, $2N$ FLOPs per generated token, bytes per parameter, KV-cache size | numbers recalled without derivation |
| Assumptions | stated (utilization, bandwidth, precision) and sanity-checked | hidden, or 100% utilization |
| Bottleneck | names compute, memory bandwidth or communication as the limit | treats everything as FLOPs |
| Speed | the right order of magnitude in about two minutes | precise and slow, or fast and wrong |

**Example prompts**
- At batch size 1, how many tokens/s can one GPU generate for a 7B model in bf16? When does batching stop being free?
- How much GPU memory does full fine-tuning of a 7B model with AdamW need?
- How many GPUs does it take to serve a 70B model to a given number of concurrent users?

**Strong answer outline (7B decode)**
1. Each decode step reads every weight once: $7\times10^9 \times 2$ bytes = 14 GB.
2. At 3.35 TB/s of memory bandwidth (an H100 SXM), that is about 4.2 ms per step: at most ~240 tokens/s at batch 1, before counting KV-cache reads.
3. A batch of $B$ sequences costs $2NB$ FLOPs for the same weight read, so arithmetic intensity is about $B$ FLOPs per byte. With ~989 TFLOP/s of dense bf16, the ridge point is $989/3.35 \approx 300$: below a batch of roughly 300, decode is memory-bound and batching is nearly free.
4. The caveat: KV-cache reads grow with batch size and context, so long-context decode can stay memory-bound at any batch size. That is why GQA and KV-cache quantization matter.

**Prep:** [lab 03](../../labs/03_napkin_math/README.md) until each drill takes under 2 minutes; [reference numbers](../../curriculum/02-compute-and-hardware.md#8-reference-numbers); [capacity planning](../../curriculum/07-inference.md#6-capacity-planning-the-interview-question); [question bank §14](question-bank.md#14-napkin-math).

**Failure modes:** no units; forgetting that training costs three forward passes' worth of FLOPs; assuming 100% utilization; false precision.

### 5. ML system design

**Format:** 45–60 minutes, open-ended: a serving platform, a training or RL pipeline, a RAG system, an eval platform. The interviewer pushes deeper at points of their choosing.

| scored | strong | weak |
|---|---|---|
| Numbers first | users, SLOs, sizes and costs before any boxes | boxes first |
| Architecture | components and data flow that meet the numbers | a list of product names |
| Trade-offs | alternatives compared, with reasons | one option, no reason |
| Failure modes and evaluation | how it breaks, how you would know, how quality is measured | not mentioned |

**Example prompts**
- Serve a 70B chat model to a stated number of concurrent users with a p95 time-to-first-token target.
- Design the infrastructure for RL post-training with verifiable rewards: rollouts, verifiers, trainer and weight sync.
- Design an eval platform that lets a research team compare checkpoints every day.

**Strong answer outline (eval platform)**
1. Users and decisions: which checkpoint to promote, and which regressions block a release. Numbers: checkpoints per day, tasks, items per task, turnaround time.
2. Components: a versioned task registry (prompts, graders, contamination checks); a scheduler that batches generation on an inference engine; a store of per-item outputs and scores; a statistics layer (paired bootstrap, multiple-comparison control); dashboards and alerts.
3. Trade-offs: full runs against sampled subsets; exact-match graders against model graders (calibrated on human labels); cost against frequency.
4. Failure modes: prompt-template drift, nondeterministic decoding, eval items leaking into training data, grader bias.
5. Evaluate the platform itself: rerun one fixed checkpoint to measure the noise floor, which sets the smallest difference you can report.

**Prep:** [the framework](ml-system-design.md#the-framework) and the [one-page template](ml-system-design.md#one-page-answer-template); designs A–H once each, then one a week, timed at 45 minutes with a peer; compare your answer with the worked [eval platform](ml-system-design.md#g-eval-platform).

**Failure modes:** boxes before numbers; naming tools instead of reasoning; no evaluation plan; shallow everywhere and deep nowhere.

### 6. Research or project deep dive

**Format:** 45–60 minutes on one project from your résumé (sometimes with slides), or on a paper you choose. The interviewer drills into decisions, numbers and alternatives.

| scored | strong | weak |
|---|---|---|
| Ownership and rigor | clear about what you did; baselines, CIs, ablations | "we" throughout; one number, no baseline |
| Depth | survives three "why"s on any decision | stops at the second |
| Taste | what to try next and what to stop, with reasons | "more data, bigger model" |
| Clarity | works at 1, 5 and 20 minutes | one length, too long |

**Example prompts**
- Walk me through the project you are proudest of. What was the hardest decision?
- Your change improved accuracy by three points. How do you know it is not noise?
- Pick a recent paper you liked. What is the weakest part of its evidence, and what experiment would fix it?

**Strong answer outline (proudest project)**
1. One minute: the problem, your role, the result with its uncertainty, why it mattered.
2. Five minutes: setup, baseline, the key decision and the alternatives you rejected, the main result and one ablation, one surprise.
3. Hand over: "Where would you like to go deeper?"
4. For every number, know how it was measured, on how many items, against which baseline, and what result would have changed your mind.

**Prep:** rehearse two projects at 1, 5 and 20 minutes, recorded; answer the [probes about your own work](question-bank.md#probes-about-your-own-work); verify [every number](../../career/stories.md#numbers-you-must-verify); write each project up with the [write-up template](portfolio.md#experiment-write-up-template) first.

**Failure modes:** numbers you cannot reproduce; no baseline; defensiveness when challenged; claiming more than the evidence shows.

### 7. Values or mission

**Format:** 30–60 minutes of conversation about the risks and benefits of AI, trade-offs, and times you acted on your principles. As of September 2026, some companies run a dedicated round and others fold it into behavioral rounds; see what each company says it values: [Anthropic](../../companies/anthropic/README.md#what-they-value), [OpenAI](../../companies/openai/README.md#what-they-value), [Google](../../companies/google/README.md#what-they-value), [Meta](../../companies/meta/README.md#what-they-value), [NVIDIA](../../companies/nvidia/README.md#what-they-value).

| scored | strong | weak |
|---|---|---|
| Specificity | concrete risks, with mechanisms you understand | slogans |
| Reasoning | weighs both sides; says what would change your mind | certainty without argument |
| Consistency | views that match your stories and choices | views invented for the day |
| Knowledge | knows the company's published positions and engages with them | flattery |

**Example prompts**
- Which risks from advanced AI concern you most, and what would you work on to reduce them?
- Tell me about a time you pushed back on something you thought was wrong.
- A customer wants a capability that could be misused. How do you decide?

**Strong answer outline (risks)**
1. Choose one or two risks you understand mechanically: for example, reward hacking in RL post-training, or misuse through agents with tool access.
2. Say where you are uncertain and what evidence would change your view.
3. Connect them to work you would do: evals that detect the problem, interpretability that explains it, safeguards at deployment.
4. Engage honestly with the company's published positions, including any point where you see things differently.

**Prep:** [values and mission prep](../../career/stories.md#values-and-mission-prep); [your position](../../curriculum/09-interpretability-and-safety.md#5-your-position) in module 09; [question bank §12](question-bank.md#12-interpretability-and-safety).

**Failure modes:** generic answers; flattery; never having thought about it; strong positions with no reasoning; views that contradict your own stories.

### 8. Behavioral

**Format:** 30–60 minutes of "tell me about a time…" on ownership, conflict, failure, ambiguity and prioritization, sometimes inside other rounds.

| scored | strong | weak |
|---|---|---|
| Specificity | one real situation, with stakes | a general habit |
| Your actions | what *you* did, and why | what the team did |
| Result and learning | a measured outcome, and what you changed afterwards | no outcome |
| Scope | matches the level you are applying for | too small, or inflated |

**Example prompts**
- Tell me about delivering something with unclear requirements.
- Tell me about a project that failed. What would you do differently?
- Tell me about a technical disagreement with a teammate.

**Strong answer outline (failure):** the situation and stakes in two sentences; the decision you made and why it was reasonable at the time; what went wrong, owned without blame; what you changed afterwards and the evidence that the change worked; about two minutes, leaving room for follow-ups.

**Prep:** the [method](../../career/stories.md#method-star--learning), [template](../../career/stories.md#story-template) and [follow-up drill](../../career/stories.md#follow-up-drill) in stories.md; 12–15 stories that each cover several themes.

**Failure modes:** a disguised success offered as the failure; blaming others; rambling past three minutes; one story for every question.

### 9. DSA

**Format:** 45 minutes, one or two LeetCode-style problems (usually medium) in a shared editor that may not run code. As of September 2026, common in Big Tech loops and lighter at many frontier labs; see [market-intel.md](../../career/market-intel.md#interview-formats-durable-patterns) and, for one company's version, the [Google guide](../../companies/google/README.md#what-dsa-looks-like-here-concretely).

| scored | strong | weak |
|---|---|---|
| Problem solving | names the pattern; brute force first, then improves it | jumps into code |
| Correctness | clean code, dry-run on small cases | bugs found by the interviewer |
| Complexity | time and space stated and justified | not mentioned |
| Communication | thinks out loud; uses hints | silent |

**Example prompts**
- Return the running median of a stream after each insertion.
- Find the shortest substring that contains every character of a pattern.
- Given course prerequisites, return a valid order or report a cycle.

**Strong answer outline (running median):** clarify input size and duplicates; brute force (keep a sorted list) costs $O(n)$ per insertion; two heaps (a max-heap for the lower half, a min-heap for the upper) with sizes differing by at most one give $O(\log n)$ insertion and $O(1)$ median; dry-run odd and even counts, duplicates and negatives; mention an order-statistics tree, or an approximate sketch for streams too large for memory.

**Prep:** [dsa-patterns.md](dsa-patterns.md#how-to-use-this-list) at 3–4 h/week of maintenance; before a Big Tech loop, the [DSA plan](coding-interviews.md#dsa) in coding-interviews.md.

**Failure modes:** coding before a plan; no complexity analysis; missed edge cases; freezing silently.

## Rounds that allow AI coding tools

> [!IMPORTANT]
> As of September 2026, some companies allow or expect candidates to use AI coding assistants in some coding rounds or take-homes. The rules differ by company and by round, and they change often: check with your recruiter which tools are allowed, whether you share your screen, and how the round is scored.

When the tool is allowed, the interviewer is scoring your judgment: how you break the problem down, how you verify, and whether you can explain and change every line.

- **Tests first.** Turn the spec into tests before generating any code. They are your contract and your fastest check.
- **Own the design.** Decide the interfaces and data structures yourself, then ask for small, well-specified pieces.
- **Verify everything.** Read every generated line, run it, and probe edge cases, complexity and numerical stability. Never submit code you cannot explain.
- **Narrate.** Say what you are asking for, why, and what you are checking in the result.
- **Type it yourself when that is faster.** Short functions and tricky invariants are often quicker and safer by hand.
- **Be ready to continue without it** if the tool fails or turns out not to be allowed.

**Practice:** do the [practical drills](coding-interviews.md#practical-engineering-drills) with and without the tool under the same timer, log the time and the bugs you caught in each, then explain the tool-assisted solution line by line to a peer.

## Mock-interview protocol

From month 6 of the roadmap (March 2027 on the default calendar): one mock every two weeks, and one a week in the four weeks before a loop. Rotate: ML coding → system design → deep dive → practical coding → debugging or napkin math → behavioral and values.

1. **Set up.** A peer interviews you with a prompt you have not seen (swap prompts from the question bank and drills), a timer, a shared editor, cameras on, and a recording if you both agree.
2. **Run it like the real thing:** no pausing, no looking things up.
3. **Score it** with the sheet below, then take 10 minutes of spoken feedback.
4. **Debrief** in writing within 24 hours, and drill the gap that week.

Find partners in the GPU MODE, EleutherAI and Hugging Face communities, among former colleagues, and through [outreach](../../career/outreach.md); consider a paid mock service before final rounds.

**Scoring sheet** (1 = clear no, 2 = leaning no, 3 = leaning yes, 4 = clear yes)

| dimension | what a 4 looks like | what a 1 looks like | score |
|---|---|---|---|
| Understanding | clarifying questions asked; assumptions stated | solved the wrong problem | |
| Approach | plan stated before code; alternatives weighed | trial and error | |
| Correctness | works, with tests the candidate chose | broken or untested | |
| Depth | complexity, numerics and behavior at scale | nothing beyond "it works" | |
| Communication | narrated; used hints well | silent or defensive | |
| Pace | core finished with time for extensions | core unfinished | |
| **Overall** | one sentence: what would move this up one level? | | |

**Debrief template**

```text
Date · round type · prompt · interviewer
Scores: understanding / approach / correctness / depth / communication / pace
Where I stalled (minute mark), and why: knowledge, practice, nerves or communication
The right answer (looked up afterwards)
One drill for this week, with a link
Same stall as in one of the last three debriefs? If yes, it becomes next month's focus
```

After real rounds, use the [interview debrief](../../career/applications.md#interview-debrief-within-24-h-of-every-round) in career/applications.md.

## Logistics: interviewing across time zones from India

India Standard Time is UTC+5:30 all year, so the gap to the US and the UK changes when their clocks change, and they change on different dates (in 2026, the UK on 25 October and the US on 1 November). Re-check every invite in those weeks.

| interviewer's time zone | behind IST by | their 9:00–12:00 in IST |
|---|---|---|
| US Pacific, summer (PDT) | 12 h 30 min | 21:30–00:30 |
| US Pacific, winter (PST) | 13 h 30 min | 22:30–01:30 |
| US Eastern, summer (EDT) | 9 h 30 min | 18:30–21:30 |
| US Eastern, winter (EST) | 10 h 30 min | 19:30–22:30 |
| UK, summer (BST) | 4 h 30 min | 13:30–16:30 |
| UK, winter (GMT) | 5 h 30 min | 14:30–17:30 |

**Scheduling**
- Ask for slots in the interviewer's morning, which is your evening. A full loop starting at 9:00 Pacific can run past 2:00 IST; ask to split it over two or three days instead. Recruiters can often arrange this, but only if you ask.
- Confirm every invite in writing with both time zones.
- For late slots, shift your sleep toward the interview time over the preceding days, and keep the next morning free if you can.

**Ask the recruiter for the format**
- Which round types, in what order, how long each, and with whom (team, seniority)?
- The environment: your own IDE or a shared editor? Can you run code? Are AI coding tools allowed?
- Is there a DSA round, a take-home, or a presentation for the deep dive?
- Is the system design round ML-specific or general?
- If there is a values or mission round, what does it cover?
- The expected timeline, whether team matching is a separate step, and whether the role can be based in India or needs relocation ([visa questions](../../career/visa-and-relocation.md#questions-to-ask-recruiters-early)).

**Set up:** a wired connection with a mobile hotspot as backup; power backup (a UPS or inverter) for the router and laptop; a headset; a quiet room for the whole slot; the video and coding platforms tested the day before. Each company guide has a "From India" section: [Anthropic](../../companies/anthropic/README.md#from-india), [OpenAI](../../companies/openai/README.md#from-india), [Google](../../companies/google/README.md#from-india), [Meta](../../companies/meta/README.md#from-india), [NVIDIA](../../companies/nvidia/README.md#from-india).

## The two failure modes to avoid

1. **Breadth without depth:** naming ten techniques and deriving none. Interviewers go three "why"s deep on one. A typical chain:
   - "Why does GRPO not need a value network?" Because the baseline is the mean reward of a group of samples for the same prompt.
   - "What does dividing by the group's standard deviation do?" It rescales each prompt's advantages, which up-weights prompts whose rewards barely vary: those the model almost always solves or almost always fails.
   - "Is that a problem?" It biases which prompts drive learning. Dr. GRPO argues for removing it; you should be able to say what you would measure to decide.

   The fix: for every technique on your résumé, be able to derive it, name its failure mode, and say what you measured. Practise with the [question bank](question-bank.md#how-to-practice-this-bank), three "why"s per answer.
2. **Unverifiable claims:** every number on your résumé will be probed. If you cannot reproduce the reasoning behind it on a whiteboard, remove it. For each number, keep where it came from (a log, a dashboard, a commit), the sample size, the baseline, the uncertainty and your part in it. A smaller number you can defend beats a bigger one you cannot. See [numbers you must verify](../../career/stories.md#numbers-you-must-verify) and the [résumé guide](../../career/resume/guide.md#open-issues-verify-before-sending-anything).
