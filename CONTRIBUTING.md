# Contributing to Achilles

Achilles is only useful if every number is right, every test is fair and every link works. The contributions that help most are the ones that catch where it is wrong.

**Contents**
- [What helps most](#what-helps-most)
- [Set up](#set-up)
- [How a lab is built](#how-a-lab-is-built)
- [Writing style](#writing-style)
- [Company guides and the library](#company-guides-and-the-library)
- [Pull request checklist](#pull-request-checklist)

## What helps most

| Contribution | Why it matters | How |
|---|---|---|
| **A wrong number, fact or derivation** | Readers memorize these and repeat them in interviews | Open a [content error issue](https://github.com/Lourdhu02/achilles/issues/new?template=content_error.yml) with the source, or send a PR with the fix and the source in the description |
| **A test that lets a wrong implementation pass** | The tests are the teacher; a weak test teaches the wrong thing | Add a failing case to the lab's `test_*.py` and show which bug it catches |
| **A clearer explanation** | Most confusion comes from one missing step | Improve the handout (`labs/NN_*/README.md`) or the curriculum module; keep headings stable |
| **Platform fixes** | Everyone should be able to run the labs on CPU, CUDA or Apple Silicon | Include OS, Python, torch version and device in the PR |
| **Outdated company or hiring facts** | Hiring processes change every quarter | Update the guide, date the fact ("as of Month Year") and link the official source |
| **A library item** | One great explainer saves readers days | Follow [library/README.md → Adding an item](library/README.md#adding-an-item) |

New labs are welcome too. Open an issue first with the mechanism, what the tests would check, and how it runs on a CPU in seconds.

## Set up

Follow [SETUP.md](SETUP.md) for your hardware, then check that everything passes before you change anything:

```bash
python -m pytest --impl=solution      # every reference solution passes (CI runs this on Linux, Windows and macOS)
python tools/check_links.py           # every relative link and #anchor resolves
```

## How a lab is built

```
labs/NN_name/
  README.md        handout: concepts, worked numbers, what to implement, tips, check yourself
  solution.py      reference implementation with # BEGIN SOLUTION / # END SOLUTION markers
  exercise.py      generated stub: solution blocks replaced by raise NotImplementedError
  test_*.py        the spec: loads exercise.py or solution.py through labs/_impl.py
```

- **Edit `solution.py`, never `exercise.py` by hand.** Regenerate the stub with `python tools/make_exercises.py --force NN_name`. Lines starting with `# HINT:` inside a solution block survive into the stub.
- **Tests must fail on the stub and pass on the solution.** Check both: `pytest labs/NN_name` should fail with `NotImplementedError`, and `pytest labs/NN_name --impl=solution` should pass.
- **Test properties, not implementations.** Compare against PyTorch or a closed form (bit-exact where possible), check invariants (for example, speculative decoding must be lossless), and include the edge case that catches the common bug.
- **Keep every test fast and CPU-only.** Seed everything. A test that takes longer than about 10 seconds on a laptop CPU gets `@pytest.mark.slow`; one that needs CUDA gets `@pytest.mark.gpu` and must skip cleanly without it.
- **Load the implementation through `labs/_impl.load`,** so `--impl` and `LABS_IMPL` keep working.

## Writing style

- Plain, direct English. Sentence-case headings. No emoji.
- Mechanism before jargon: say what happens, then name it.
- Compute every number you state, and show the arithmetic once so readers can check it.
- Use GitHub alerts for asides: `> [!TIP]`, `> [!NOTE]`, `> [!IMPORTANT]`, `> [!WARNING]`.
- **Don't rename headings casually.** Other files link to them by anchor, and `tools/check_links.py` will fail if an anchor disappears.
- Date anything volatile (teams, hiring loops, prices, visa rules) as "as of Month Year", and link the primary source.
- Link papers by their arXiv abstract page (`https://arxiv.org/abs/<id>`) or the authors' page. Never cite a paper or a number you have not checked.

## Company guides and the library

- Each folder in [companies/](companies/README.md) has a `refresh-brief.md` describing how to refresh it. Prefer official careers pages, engineering blogs and papers over forum posts; mark anything second-hand as such.
- The [library](library/README.md) keeps two to five items per topic. Add an item only if it beats one already there, and replace rather than grow. PDFs are never committed: `library/pdfs/` is git-ignored and `tools/fetch_library.py` downloads them for personal study.

## Pull request checklist

- [ ] `python -m pytest --impl=solution` passes
- [ ] For lab changes: the stub was regenerated with `tools/make_exercises.py` and still fails cleanly
- [ ] `python tools/check_links.py` reports no broken links
- [ ] New facts and numbers have a source or a shown calculation
- [ ] No large files: data, checkpoints and PDFs stay out of git

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE), and that you will follow the [code of conduct](CODE_OF_CONDUCT.md).
