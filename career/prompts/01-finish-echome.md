# Prompt: finish ECHOME

Paste everything below the line into a new Claude Code session with `Lourdhu02/echome` attached.

---

You are finishing **ECHOME** (https://github.com/Lourdhu02/echome), a local-first agent with three-tier memory (LangGraph, Qdrant, Ollama), an adaptive personality assessment (IRT graded response model, Fisher-information item selection, MAP estimation) and a voice pipeline (Whisper, XTTSv2). The goal is a project an ML hiring manager can clone, run in one command, and trust every number in.

## Where it stands

- Implemented and unit-tested: the CAT/IRT engine, the assessment API, the memory system, the orchestrator and the specialist agents. The voice engine is **Partial**, a thin wrapper.
- Measured on 2026-10-02:
  - `python -m tests.eval.run_eval` recalls **11 of 12** planted facts in the top 5, up to 52 turns later, at about 1 ms per retrieval. That run used the hashing-embedding fallback, not MiniLM.
  - A Monte Carlo run of the CAT with 500 simulated respondents × 8 traits: the 5-item cap always triggers first, so tests are **50% shorter** (40 of 80 items). Scores correlate **r = 0.97** with full-test scores, and **0.91 vs 0.935** with true trait values.
  - Script: `career/resume/evidence/echome_cat_sim.py` in `Lourdhu02/achilles`.
- **Known false claim:** the README says "reduces assessment length by 70% while maintaining SE < 0.32". With the current 10-items-per-trait bank, SE ≤ 0.32 is never reached within 5 items. Correct it.

## What to do

1. **Fix the README claim.** Add `tests/eval/cat_simulation.py`, which reproduces the Monte Carlo run (seeded and CLI-driven), and report items used, r with full-test scores, r with true θ and RMSE. Then explore the stopping rule honestly: try SE thresholds 0.30–0.45 and item caps 3–8, plot test length against r, and pick a default with a stated trade-off.
2. **Re-run the memory benchmark with real MiniLM embeddings** (not the hashing fallback). Grow it from 12 to at least 50 scenarios, including distractor facts, contradictions (a newer fact overrides an older one) and longer horizons (100+ turns). Report recall@1/5 with a 95% bootstrap CI, for full memory vs episodic-only vs no-memory.
3. **End-to-end answer quality:** with a small local Ollama model (e.g. `qwen2.5:3b`), score whether final answers contain the planted fact, with and without memory. Report the exact model and quantization.
4. **Finish the voice engine, or cut it.** Either make Whisper → LLM → XTTSv2 run end to end with a measured latency per stage on the target machine, or move it out of the README's feature list into "planned".
5. **One-command demo:** `docker compose up`, or a `make demo` that runs the assessment and a memory conversation without a GPU. Add a 60-second GIF or asciinema recording to the README.
6. **CI:** GitHub Actions running unit tests, the CAT simulation and the memory benchmark (hashing fallback allowed in CI, but labelled), and publishing the report to the job summary.
7. **README:** a "Results" section with one table per benchmark. Each table gives the command that produced it, the date, the hardware and the seed.

## Rules

- No invented or projected numbers. Every number comes from a script in the repo, with its dataset, baseline and seed. If a result is bad, report it and say what you'd try next.
- Commit as `Lourdhu Raju <b.lourdhuraju1234@gmail.com>` (`git config user.name/user.email` in the repo before the first commit), with no Co-Authored-By or session trailers.
- Work on a branch, keep commits small, open a PR when done, and make sure CI is green.

## Done when

CI is green, and the README's Results tables reproduce from a clean clone. The README contains no claim that a script doesn't back. Then give me two resume bullets, each with one measured number and its context.
