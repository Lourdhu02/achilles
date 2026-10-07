# Cheat sheet

Two printable PDFs that condense the curriculum, the labs and the interview guides for revision.

| file | what it is |
|---|---|
| [llm-genai-cheatsheet.pdf](llm-genai-cheatsheet.pdf) | 24 pages, two columns: interview Q&A, formula sheet, napkin problems, from-scratch code, debugging scenarios, paper cards, glossary, references |
| [life-of-a-token.pdf](life-of-a-token.pdf) | 5 pages: the same material told as one story in nine chapters, with a one-page recap |

Use the story first to see how the pieces connect, then drill the cheat sheet. Numbers from lab reports and the October 2026 tooling snapshot should be re-checked before you quote them.

## Build

Needs a TeX Live install with `tcolorbox`, `pgfplots`, `listings` and `mathpazo`. Run `pdflatex` twice so the contents page fills in.

```bash
cd cheatsheet/src   && pdflatex main.tex && pdflatex main.tex
cd ../story         && pdflatex life-of-a-token.tex && pdflatex life-of-a-token.tex
```

Each section of the cheat sheet is its own file in `src/sec/`, included from `src/main.tex` in reading order.
