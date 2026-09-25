# 00 — How to learn at this level

## The loop: Read → Derive → Build → Measure → Write
1. **Read** the module, then one primary paper. Skim first, then read the method section with a pen.
2. **Derive** the core equation on paper *before* looking at code: the backward pass, the loss, the FLOP count.
3. **Build** it from scratch in the lab until the tests pass. Look at the reference only after an honest attempt (≥ 2 hours stuck on one thing), and then close it and rewrite from memory.
4. **Measure**: predict a number (memory, tokens/s, loss after N steps), then measure it, then explain the gap. Where prediction and measurement disagree, you learn something.
5. **Write** 300–1,500 words in `journal/`. If you cannot explain it in writing, you do not understand it yet. The best entries become blog posts.

## Rules that separate people who get good from people who collect tutorials
- **Outputs over inputs.** Track labs passed, experiments written up, PRs merged. Hours are an input.
- **Predict before you measure,** every time. Keep a running log of predictions and errors; your calibration is the skill.
- **One primary source beats five explainers.** Explainers are for when you are stuck.
- **Spaced repetition for facts and numbers** (Anki): the ~200 numbers and identities in modules 02–07. Ten minutes a day.
- **Implement the hard 20%.** Use libraries for plumbing; hand-write the core (loss, attention, sampler, optimizer step).
- **Ship small, weekly.** A result every week beats a big result in six months that never ships.

## Reading papers
Pass 1 (10 min): abstract, figures, conclusion. What is the claim, and what evidence would falsify it?
Pass 2 (1 h): method and experiments. What is compared against what, at what compute, with how many seeds?
Pass 3 (a day, for the few that matter): reproduce one figure at small scale. Note every detail the paper left out.
Write down for every paper: *claim · evidence · compute · what I'd test next · does it still hold?*

## Using LLMs as a tutor (without outsourcing your brain)
- Ask for Socratic questioning, critiques of *your* derivation, or counterexamples. Don't ask for solutions to labs you haven't attempted.
- Verify every factual claim that matters against the paper or the code. Models confidently misstate numbers (this repo's previous version stated a KV-cache size 8× too large).
- Use them to generate drill questions from a module, then answer out loud, timed.

## Weekly shape (~25 h: 5×2.5 h weekdays + 2×6 h weekend)
- Weekdays: one module section plus lab progress, and 30 min of DSA on 3 days.
- Saturday: deep lab work or a GPU run. Sunday: write-up, review, plan the next week ([template](../journal/templates/weekly-review.md)).
- Protect sleep and one full rest day in two weeks. See [career/sustainability.md](../career/sustainability.md).
