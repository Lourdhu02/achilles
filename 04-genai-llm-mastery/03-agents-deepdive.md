# Agents — Beyond ReAct

ECHOME gives you a head start. This file gets you to "design any agent system" depth.

---

## What an agent actually is

A loop:
```
WHILE not done:
    observe environment
    think (plan/reason)
    act (call a tool / generate output)
    update state
```

The LLM is the "brain"; tools, memory, and an orchestration layer are the rest.

---

## The major paradigms

### 1. ReAct (Reason + Act)
- LLM interleaves "Thought" → "Action" → "Observation" → "Thought" → ...
- Originally from Yao et al. 2022
- Most common today

Example trace:
```
Thought: I need to know today's temperature in Bangalore. I should use the weather tool.
Action: weather_tool(location="Bangalore")
Observation: Temperature: 32°C, Humidity: 65%
Thought: 32°C is high. The user asked if it's "hot today." I should answer yes.
Action: respond(text="Yes, it's hot — 32°C in Bangalore today.")
```

### 2. Plan-and-Execute
- First, LLM generates a full plan (list of steps)
- Then executes step by step
- More expensive (more LLM calls) but more reliable for complex tasks

### 3. Reflexion
- After acting, LLM critiques its own output
- Self-improves over iterations
- Slow but high quality

### 4. Tree-of-Thoughts
- LLM explores multiple reasoning paths
- Backtracking on failure
- Good for math/puzzles, overkill for simple tasks

### 5. AutoGPT-style (autonomous agents)
- Long-running, multi-step, self-directed
- Common failure: loops, drift, unbounded cost
- More research than production

### 6. Multi-agent orchestration
- Multiple specialized agents communicate
- E.g., "planner agent" + "coder agent" + "tester agent"
- Frameworks: CrewAI, AutoGen, LangGraph (which you use!)

---

## Tool calling

### How modern LLMs do it

LLMs (GPT-4, Claude, etc.) have NATIVE tool-calling:
```
Provide tool schemas (JSON) → LLM emits structured tool call → System executes → Pass result back
```

Schema example:
```json
{
  "name": "search_web",
  "description": "Search the web for current information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"}
    },
    "required": ["query"]
  }
}
```

### Tool design tips
- **Keep tool count <20** — LLM gets confused with too many
- **Clear, specific descriptions** — half the battle
- **Idempotent tools when possible** — safer for retries
- **Return structured data** — JSON, not free text
- **Error handling in tool** — return error messages the LLM can parse and recover from
- **Streaming for long-running tools** — UX matters

### Tool failure modes
- LLM calls wrong tool
- LLM hallucinates tool that doesn't exist
- LLM passes invalid arguments
- Tool itself fails (network, etc.)
- LLM enters loop calling same tool repeatedly

### Mitigations
- Tool-use traces are short → harder to hallucinate
- Validate args against schema before calling
- Tool-use telemetry: rate of each tool, error rate

---

## Memory

### Memory types

#### Short-term (working memory)
- The chat history in the LLM context
- Bounded by context window

#### Long-term (episodic / semantic)
- Stored externally (vector DB, key-value store, graph)
- Retrieved via similarity, recency, or explicit lookup

#### Procedural memory
- "How to do things" — patterns, code snippets, learned behaviors
- Often: a curated library the agent can search

### Memory architecture (ECHOME-style)

Your three-tier architecture is articulable in interviews as:

| Tier | Purpose | Storage | Retrieval |
|---|---|---|---|
| Episodic | Specific past events ("user said X yesterday") | Vector DB with timestamps | Similarity + recency |
| Semantic | Generalized facts ("user is a doctor") | Key-value or graph | Direct lookup |
| Procedural | Action templates / playbooks | Versioned templates | Match on task type |

**When asked about agent memory in interviews, lead with this. Then discuss tradeoffs (cost, latency, consolidation logic).**

### Memory consolidation
- Periodically: summarize old episodic memories into semantic (compress)
- Forget low-importance memories (decay)
- Update facts when contradicted

This is where ECHOME's depth shines — most candidates haven't designed memory consolidation.

---

## Planning

### Why planning hard for LLMs
- LLMs are next-token predictors; they don't naturally "plan"
- Plans drift / contradict themselves
- Complex plans → many steps → compound error

### Approaches

#### Decompose first, execute later (Plan-and-Execute)
- Single LLM call to generate plan
- Iterate through plan, executing each step
- Pros: clear, debuggable
- Cons: plan is brittle if environment changes

#### Iterative planning (ReAct)
- One step at a time, with reasoning each turn
- More adaptive
- More LLM calls

#### Hierarchical planning
- High-level plan (3-5 steps)
- Each step → sub-plan with more LLM calls
- Used in long-horizon tasks (browser agents, code agents)

#### Trees / search at inference time
- LLM generates multiple plan candidates
- Evaluate each (LLM-as-judge or simulation)
- Pick best
- Expensive but high quality (e.g., AlphaCode-style)

---

## Multi-agent systems

### When multi-agent makes sense
- Different agents have genuinely different capabilities / specializations
- Task is highly modular
- Want explicit roles / responsibilities for debuggability

### When NOT
- One LLM call with good prompt would do
- Latency-sensitive (each agent = more LLM calls)
- Cost-sensitive

### Patterns

#### Hierarchical (manager + workers)
- Manager assigns tasks, workers execute
- Manager checks/aggregates results

#### Pipeline (sequential specialists)
- Agent A → Agent B → Agent C
- Each transforms data for the next

#### Cooperative (all agents talk to all)
- Conference-style; agents exchange messages until consensus
- High coordination cost; can deadlock

#### Competitive (debate)
- Multiple agents propose; judge picks
- Used to improve reasoning quality

---

## State management

State across turns: the trickiest part.

### Options
- **Pass everything in context** (works only for short sessions)
- **Memory store + retrieval** (your ECHOME approach)
- **State machine** (LangGraph: explicit nodes + edges)
- **Database** (durability across sessions)

### LangGraph in particular
- State is a typed dict (or Pydantic model)
- Nodes are functions; edges are routing decisions
- Conditional edges = control flow
- Persistence: checkpointing every step
- Why LangGraph (vs vanilla LangChain): explicit control, durable, debuggable

ECHOME uses LangGraph — you should be able to articulate this as a deliberate design choice.

---

## Evaluation of agents (hard)

### What to evaluate
- **Task completion:** did the agent succeed end-to-end?
- **Efficiency:** how many tool calls? How long? Cost?
- **Robustness:** does it handle adversarial inputs, edge cases?
- **Faithfulness:** does it stay on-task? Hallucinate?
- **Safety:** doesn't do harmful actions?

### Benchmarks (current as of 2026)
- **AgentBench** (Liu et al.) — broad agent eval
- **GAIA** (Mialon et al.) — multi-modal real-world tasks
- **SWE-Bench** — software engineering tasks
- **WebArena** — browser agent tasks
- **ToolEval** — tool use
- **AppWorld** — sequential multi-app agent tasks

### Custom evaluation
For your specific application:
1. Define success criteria
2. Build a golden set (50-200 queries with known good outputs)
3. Run agent, score each
4. Track over time (regressions on changes)

---

## Failure modes (interview gold — show production thinking)

### Cost explosion
- Agent loops calling expensive tools or LLMs
- Mitigation: cost budget per session, kill switch

### Drift / topic shift
- Agent forgets original goal after many turns
- Mitigation: re-state goal in system prompt every N turns

### Tool misuse
- Agent calls tools incorrectly or in wrong order
- Mitigation: tool descriptions, examples in prompt, post-call validation

### Halt-and-loop
- Agent gets stuck repeating same action
- Mitigation: track recent actions, penalize repetition, max iterations

### Action without observation
- Agent acts but doesn't process the result
- Mitigation: explicit "Thought" or "Reflection" after every "Action"

### Prompt injection via tool outputs
- Tool returns text that overrides agent's system prompt
- Mitigation: sanitize tool outputs, separate tool output from system prompt clearly

---

## Modern agent stacks (know these in interviews)

### Production frameworks
- **LangGraph** (your stack) — explicit state machines
- **CrewAI** — multi-agent with role-playing
- **AutoGen** (Microsoft) — multi-agent
- **Vercel AI SDK** — Next.js-native, lightweight
- **Mastra** (TypeScript) — emerging

### Research / experimental
- **AutoGPT, BabyAGI, MetaGPT** — first wave, mostly research
- **OpenHands (formerly OpenDevin)** — open-source coder agent
- **Aider** — terminal-based code agent

### Hosted agent platforms
- **Anthropic Claude with Computer Use** — desktop/browser control
- **OpenAI Assistants API** (deprecated for Realtime/Responses APIs)
- **Vertex AI Agents** (Google)
- **Bedrock Agents** (Amazon)

---

## Designing an agent in an interview (the framework)

When asked "design an X agent" in interview:

### Step 1: Clarify
- What does the user want to do?
- What tools should the agent have access to?
- What's the success criterion?
- What's the cost / latency budget?

### Step 2: Architecture
- Single LLM with tool calling? Or multi-agent?
- Memory needs? (short-term only? long-term?)
- State management strategy?

### Step 3: Tools
- List 5-10 tools the agent might need
- Define schemas for 2-3
- Error handling per tool

### Step 4: Reasoning loop
- ReAct? Plan-execute?
- Max iterations?
- Cost cap?

### Step 5: Memory
- What to remember
- How long
- How to retrieve

### Step 6: Eval
- Success metrics
- Telemetry
- Continuous improvement

### Step 7: Failure modes + safety
- Cost cap, action audit, escape hatches

### Step 8: Scale
- Concurrent users
- Caching strategies
- Cost optimization

You should be able to do all 8 in 50 minutes. Practice 3-4 design problems.

---

## ECHOME-specific talking points (refresh before any GenAI interview)

Be ready to articulate:
1. **Why LangGraph specifically?** State machines + persistence + debuggability over vanilla LangChain
2. **The three-tier memory architecture details:** what goes where, how consolidation works
3. **CAT/IRT + Fisher Information:** adaptive testing, why this gives 70% efficiency gain
4. **Privacy-first / local deployment:** why and what tradeoffs
5. **XTTSv2 voice cloning:** zero-shot capabilities, latency/quality tradeoff
6. **What you'd do differently:** show critical thinking, not just defense

---

## Papers to read (in priority order)

1. **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao 2022)
2. **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick 2023)
3. **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn 2023)
4. **Voyager: An Open-Ended Embodied Agent with LLMs** (Wang 2023)
5. **MemGPT: Towards LLMs as Operating Systems** (Packer 2023) — memory architecture
6. **Generative Agents: Interactive Simulacra of Human Behavior** (Park 2023) — the Stanford Smallville paper
7. **AgentBench: Evaluating LLMs as Agents** (Liu 2023)
8. **Tree of Thoughts: Deliberate Problem Solving with LLMs** (Yao 2023)
9. **OpenAI o1 system card** — reasoning models
10. **Claude Computer Use technical report** — Anthropic

---

Next: [`04-fine-tuning.md`](./04-fine-tuning.md)
