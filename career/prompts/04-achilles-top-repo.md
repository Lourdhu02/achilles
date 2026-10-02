# Prompt: make Achilles a top-rated open-source repo

Paste everything below the line into a new Claude Code session with `Lourdhu02/achilles` attached.

---

You are making **Achilles** (https://github.com/Lourdhu02/achilles, docs at https://lourdhu02.github.io/achilles/) a widely starred, genuinely useful open-source project. It is "core AI from first principles": 18 test-driven labs, each with a handout, a stub to implement, a test suite and a reference solution. The labs cover autograd, a Llama-style transformer, FlashAttention forward/backward in Triton, KV cache, LoRA, DPO, GRPO, quantization, speculative decoding, interpretability, retrieval and MoE. There are 229 reference tests, run in CI on Linux, macOS and Windows.

Stars come from **usefulness and discoverability**, never from gaming. No star-for-star schemes, bought stars, bot accounts or spam posting.

## What to do

1. **Audit, as a first-time visitor.** Clone fresh on CPU-only Linux, macOS and Colab, and time `README → first failing test → first passing lab`. Fix every friction point. Target: under 5 minutes to the first green test.
2. **README above the fold:** a one-sentence pitch, a 20-second GIF of a test going red → green, a "Why this and not nanoGPT / minitorch / CS336?" comparison table, and badges for CI, labs, tests and license. Cut anything that doesn't help a visitor decide in 10 seconds.
3. **Quality bar for every lab:**
   - the handout states learning goals and prerequisites, and estimates hours;
   - tests catch the classic bug (with a message explaining it);
   - the solution is idiomatic;
   - one "going further" exercise.

   Add a difficulty rating and a dependency graph between labs.
4. **Make it citable and trustworthy:** CITATION.cff (already present, keep it current), a CHANGELOG, semantic-versioned releases with notes, and a Zenodo DOI per release.
5. **Contributor funnel:** CONTRIBUTING with a 10-minute first-PR path, 15+ well-scoped `good first issue`s (a new test, a docs fix, a lab exercise), issue and PR templates, Discussions enabled, and a roadmap issue. Respond to every issue within 48 hours.
6. **Discoverability:** GitHub topics, a social preview image, a docs site with search, an SEO-friendly title and description, and a "Star History" badge.
7. **Launch plan:** write a draft (for me to post myself, not auto-posted) for each of:
   - Show HN
   - r/MachineLearning
   - r/LocalLLaMA
   - X/Twitter
   - LinkedIn
   - one deep blog post walking through the hardest lab (FlashAttention backward or GRPO) with benchmarks.

   Each draft leads with what the reader learns. Schedule: launch day, plus one new-lab post every 2 weeks.
8. **Metrics:** a monthly `tools/metrics.py` report of stars, clones, issues, PRs and lab completion from opt-in progress files. Use it to decide which labs to improve.

## Rules

- No fabricated testimonials, star counts or user numbers. Claims in the README must be checkable: tests pass, timings are measured.
- Commit as `Lourdhu Raju <b.lourdhuraju1234@gmail.com>`, with no Co-Authored-By or session trailers.
- Small PRs, CI green on all three OSes before merging.

## Done when

A fresh clone reaches a green test in under 5 minutes on all three platforms, every lab meets the quality bar, the contributor funnel and release v1.0 (with a DOI) exist, and the launch drafts are ready for me to review.
