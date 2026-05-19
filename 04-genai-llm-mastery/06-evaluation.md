# LLM Evaluation — Done Right

Most candidates can't talk about LLM eval. That makes it a strong differentiator. Master this.

---

## Why LLM eval is hard

Traditional ML: clear labels, fixed metrics (accuracy, F1, AUC).

LLMs:
- Outputs are free-form text
- Multiple correct answers
- Quality is multi-dimensional (helpful, accurate, safe, coherent, formatted)
- Benchmarks get gamed quickly
- Human eval is gold but expensive

You need a portfolio of eval techniques. No single metric suffices.

---

## Public benchmarks (know these)

### Knowledge / reasoning
- **MMLU** (Massive Multitask Language Understanding) — 57 subjects, multiple choice. The leaderboard king. Gets gamed.
- **MMLU-Pro** — harder MMLU follow-up
- **HellaSwag** — common-sense completion
- **WinoGrande** — coreference resolution
- **ARC (AI2 Reasoning Challenge)** — science questions
- **BIG-Bench** — 200+ diverse tasks

### Math
- **GSM8K** — grade-school math word problems
- **MATH** — competition math
- **AIME** — American Invitational Math Exam

### Coding
- **HumanEval** — Python code generation from docstrings
- **MBPP** — entry-level Python tasks
- **SWE-Bench** — real GitHub issues
- **LiveCodeBench** — fresh problems to avoid contamination

### Truthfulness / factuality
- **TruthfulQA** — common misconceptions
- **HaluEval** — hallucination detection

### Instruction following
- **IFEval** — verifiable instruction following
- **MT-Bench** — multi-turn quality (Vicuna-style)

### Open-ended quality
- **LMSys Arena** — human-judged pairwise battles. Elo-style ranking.
- **Alpaca Eval** — automated head-to-head with GPT-4 judge

### Safety
- **HarmBench** — adversarial prompts
- **BeaverTails** — harmlessness eval

### Multimodal
- **MMMU** — multimodal understanding
- **MMBench, MME** — vision-language eval

---

## Why benchmarks fail

### Contamination
- Test sets leak into training data
- Numbers go up; real capability doesn't
- Mitigations: held-out evals, fresh benchmarks, time-stamped questions

### Saturation
- Top models hit 90%+ on MMLU; few percentage points of headroom mean little
- New harder benchmarks needed (e.g., MMLU-Pro)

### Gaming
- Optimizing for benchmark = better on benchmark, not necessarily better in practice
- e.g., chain-of-thought prompting + best-of-N tricks specifically for evals

### Multiple-choice tells you little about generation quality
- Picking A/B/C/D is easier than generating good free-form text
- Doesn't capture: coherence, style, format compliance, multi-turn reasoning

---

## Custom evaluation (most important for production)

### Why custom evals matter
- Public benchmarks rarely measure what you specifically care about
- Your users have specific needs
- Production performance ≠ benchmark performance

### Building a custom eval set

**Step 1: Define what you care about**
- Helpfulness, accuracy, format compliance, safety, latency, cost
- Pick 3-5 axes; don't try to measure everything

**Step 2: Collect a golden set**
- 100-500 representative queries
- For each: ideal answer (if applicable), eval criteria
- Sources: production logs, user-reported issues, edge cases you predict

**Step 3: Define scoring**
- Binary: yes/no for each criterion
- Ordinal: 1-5 scale (cleaner inter-rater agreement)
- Continuous: rare for free-form text

**Step 4: Score**
- Human: gold standard, 50-200 examples
- LLM-as-judge: scalable, ~80% agreement with human (after validation)
- Programmatic: regex / parser-based (e.g., "must include valid JSON")

**Step 5: Track over time**
- Run eval on every model change
- Catch regressions
- Build dashboards

---

## LLM-as-judge

### How it works
Use a stronger LLM to evaluate a weaker one's outputs.

```
Prompt to judge LLM:
"Evaluate the following answer to the user query. Score 1-10 on:
- Accuracy
- Helpfulness  
- Clarity

Query: [user query]
Answer: [model output]

Provide JSON: {"accuracy": int, "helpfulness": int, "clarity": int, "reasoning": str}"
```

### Pros
- Scalable (1000s of evals per hour)
- Cheaper than humans
- Reproducible (with temp=0)

### Cons
- Judge LLM has biases
- Can be gamed (verbosity bias, position bias)
- Cost: each eval costs a Claude/GPT call
- Trust calibration needed

### Best practices
1. **Validate against human ratings first** — 100 samples, both judged by humans and LLM, compute correlation
2. **Use stronger judge than the model being eval'd** (Claude Opus judges Sonnet outputs)
3. **Multiple criteria, structured output** — JSON with reasoning
4. **Randomize position** — in pairwise, swap which response is "A" vs "B"
5. **Multiple judges + average** — reduces individual bias
6. **Refresh judges periodically** — judge model drifts too

### Pairwise vs absolute
- **Pairwise:** "Which answer is better, A or B?" → produces Elo ratings
- **Absolute:** "Rate this answer 1-10" → easier to aggregate, scale-dependent

---

## RAGAS — specific RAG eval

Four metrics:
- **Faithfulness:** does answer follow from context? (LLM-as-judge)
- **Answer Relevance:** does answer address the query? (semantic similarity)
- **Context Precision:** are retrieved chunks relevant? (LLM-as-judge)
- **Context Recall:** does context contain the ground truth?

Use it as a starting point; customize for your domain.

---

## Agent evaluation

Agents are harder to eval because of:
- Long trajectories (many decisions)
- Non-deterministic paths
- Tool call success/failure
- Sub-goal completion

### Trajectory-level metrics
- Task completion (binary)
- Number of steps (efficiency)
- Tool call success rate
- Cost (USD per task)
- Latency (seconds per task)

### Sub-task metrics
- For each intermediate step: was the right tool called? Did it get the expected output?
- Useful for debugging

### Robustness
- Run multiple seeds → variance is informative
- Adversarial inputs: does the agent break?

---

## Designing an LLM eval pipeline (interview question)

When asked "How would you evaluate an LLM?" in an interview, structure your answer:

### Step 1: Clarify
- What's the use case? Chatbot? Code gen? Agent? RAG?
- What outcomes does the business care about?

### Step 2: Pick metrics
- Quality dimensions (accuracy, helpfulness, safety)
- System dimensions (latency, cost, throughput)

### Step 3: Gather eval data
- Golden set from production
- Synthetic adversarial
- Public benchmarks for breadth

### Step 4: Score
- Human (gold but limited)
- LLM-as-judge (scalable)
- Programmatic checks (cheap)

### Step 5: Aggregate + track
- Dashboards
- Alerts on regressions
- Slice by user segment, query type, etc.

### Step 6: Iterate
- A/B test new models / prompts
- Production-shadow new versions
- Continuous improvement

This 6-step framework handles 80% of "how would you eval X" questions.

---

## Hallucination detection

### Types of hallucination
- **Fabrication:** made-up facts (e.g., wrong dates, fake citations)
- **Distortion:** real facts but wrong context
- **Inconsistency:** contradicts earlier in same response or known truth

### Detection methods

**Self-consistency:**
- Generate multiple answers (temp > 0)
- Check agreement
- Disagreement → low confidence

**Faithfulness (for RAG):**
- Does each claim trace to retrieved context?
- LLM-as-judge or NLI model

**Fact-checking:**
- Extract claims, query knowledge base
- Compare

**Confidence scoring:**
- Get token-level probabilities
- Low-prob tokens → potential hallucination

---

## Production monitoring (live eval)

For deployed LLMs:
- Sample N% of queries for offline eval
- Track quality drift over time
- Capture user feedback (👍/👎)
- Watch for: latency spikes, cost increases, refusal rate

Tools:
- **LangSmith** (LangChain) — tracing + eval
- **Helicone** — observability
- **Arize Phoenix, Langfuse** — open source observability
- **TruLens** — eval-focused

---

## Common eval pitfalls

1. **Optimizing for a single metric** — model gets great at that metric, terrible at others
2. **Test set leak** — train data contains eval examples
3. **Stale eval set** — model improved on it; need fresh problems
4. **No human validation** — LLM-as-judge results not grounded
5. **Eval ≠ production distribution** — synthetic queries don't match real users
6. **Sample size too small** — 10 examples isn't an eval
7. **Ignoring variance** — same model, multiple runs, different scores

---

## Interview questions

1. "How do you evaluate an LLM?" — 6-step framework above
2. "What's the difference between MMLU and HumanEval?" — knowledge vs code
3. "How do you handle benchmark contamination?" — fresh evals, held-out sets, time-stamping
4. "What's LLM-as-judge? Pitfalls?" — pros/cons covered
5. "How would you eval a RAG system?" — RAGAS or similar
6. "How would you eval an agent?" — task completion + efficiency + robustness
7. "What's the role of human eval in 2026?" — gold standard for calibrating automated metrics, still essential for ambiguous quality dimensions
8. "How do you detect hallucinations?" — multiple methods

---

## Build this exercise (Month 7)

Build a complete eval pipeline for an LLM (yours or an API):

1. **Eval set:** 100 questions in a domain (e.g., your FinSentinelAI queries)
2. **Multiple LLMs:** GPT-4o-mini, Claude Haiku, Llama 3 70B
3. **Scoring:** human + LLM-as-judge
4. **Dashboard:** track scores, latency, cost per model
5. **Report:** which model is best for which query type?

This is a flagship-worthy public project. Write it up. Open-source it. It signals depth at every interview.

---

## Resources

- "Holistic Evaluation of Language Models (HELM)" — Stanford CRFM
- **RAGAS** — github.com/explodinggradients/ragas
- **lm-evaluation-harness** — EleutherAI (the standard for running benchmarks)
- **LMSys Chatbot Arena** — lmsys.org
- **OpenAI Evals** — github.com/openai/evals
- **TruLens** — trulens.org

---

Next: [`07-must-know-papers.md`](./07-must-know-papers.md)
