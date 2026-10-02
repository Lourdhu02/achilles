# Résumé evidence

Where every number on the résumé comes from, so each one survives "how did you measure that?".

| claim | source |
|---|---|
| Sujanix figures (79% → 91% live, 9× p50, 181 img/s, 94× TensorRT, CTC bug, release accuracy) | [GitHub profile README](https://github.com/Lourdhu02/Lourdhu02), "How these numbers were measured"; employer benchmark reports (private) |
| spacedrift: INR 12 lakh gross revenue, 36 clients, 6 repeat, team of 5 | your own records (FY24–25 to FY26–27) |
| ECHOME memory: 11 of 12 facts recalled in the top 5, up to 52 turns later, ~1 ms retrieval | `python -m tests.eval.run_eval` in the echome repo (12 scenarios; hashing-embedding fallback, run 2026-10-02) |
| ECHOME assessment: half the items, r = 0.97 with full-test scores | [`echome_cat_sim.py`](echome_cat_sim.py): 500 simulated respondents × 8 traits, seed 0; CAT r = 0.91 and full test r = 0.935 against true scores |
| Paper: 23 pages, 51 references | [no-final-save](https://github.com/Lourdhu02/no-final-save) README |
| Achilles: 18 labs, 229 reference tests | this repository's README and CI |
| BrainOvision: +15% forecast accuracy | your own claim; know the metric (e.g. WMAE or MAPE) and the baseline before an interview |

> [!NOTE]
> The ECHOME README says the assessment is "70% shorter at SE < 0.32". With the current 10-item-per-trait bank, the stopping rule
> (5 items or SE ≤ 0.32) always stops at the 5-item cap, because SE ≤ 0.32 is never reached in 5 items. The defensible
> claim is 50% shorter, as on the résumé. Update the ECHOME README to match.
