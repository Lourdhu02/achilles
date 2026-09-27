# Journal: your proof of work

Everything you produce goes here: GPU measurements, experiment write-ups, paper notes, and weekly and monthly reviews. Writing forces the prediction that makes an experiment worth running, and the entries become your blog posts, résumé bullets and interview deep dives. Commit it.

```
journal/
  templates/        copy these; never edit them in place
  examples/         a worked example of a good entry
  experiments/      YYYY-MM-DD-short-name.md
  papers/           one file per paper: first-author-year-short-title.md
  reviews/          YYYY-Www.md (weekly) and YYYY-MM.md (monthly)
  private/          git-ignored: salaries, interview feedback, anything under NDA
```

Templates: [experiment](templates/experiment.md) · [paper notes](templates/paper-notes.md) · [weekly review](templates/weekly-review.md) · [monthly review](templates/monthly-review.md)

**Start with the worked example:** [does lab 16's model really learn an induction circuit?](examples/2026-09-26-induction-shortcut.md). It shows a prediction written first, one change at a time, three seeds, and a result that overturned a passing test.

## The rhythm

| When | What | Time |
|---|---|---|
| Before every experiment | Write the question and a numeric prediction in a new experiment file | 5 min |
| After every experiment | Fill in results, prediction vs result, threats to validity, next | 20–30 min |
| After each paper | Paper notes: the claim, the evidence, the equation you derived yourself, what you would test | 15 min |
| Every Sunday | Weekly review: shipped, predictions and calibration, stuck on, next week's top three | 15 min |
| End of each month | Monthly review: exit criteria, mastery ratings, the evidence for the [roadmap gates](../ROADMAP.md) | 45 min |

Log every numeric prediction in one place as well (see the [prediction log](../curriculum/00-learning-os.md#5-the-prediction-log)). After a few months, how far your predictions miss is the most honest measure of your understanding.

## A strong entry and a weak one

| | Weak | Strong |
|---|---|---|
| Question | "Try MoE" | "Does top-2 MoE at matched active parameters beat the dense baseline on TinyStories after 50M tokens?" |
| Prediction | none, or written afterwards | "Lower validation loss by 0.03–0.05 nats, because capacity doubles at equal FLOPs" |
| Evidence | one run, one number | three seeds, a confidence interval, and the baseline tuned as carefully as the new method |
| Result | "it worked" | the table, the plot, and the gap from the prediction, explained |
| Next | "more experiments" | the one experiment that would change your mind |

## From journal to portfolio

Every four to six weeks, turn your best experiment into a public write-up using the [portfolio template](../tracks/research-engineer/portfolio.md#experiment-write-up-template). Mine the surprises for interview stories ([career/stories.md](../career/stories.md)): "I predicted X, measured Y, and here is why" is the answer interviewers remember.

> [!WARNING]
> The journal is public if your fork is. Keep compensation, interview feedback, other people's names and anything covered by an NDA in `journal/private/`, which git ignores.
