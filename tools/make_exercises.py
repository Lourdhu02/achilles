"""Generate exercise stubs from the reference solutions.

Inside ``labs/*/solution.py``, code between the markers

    # BEGIN SOLUTION
    ...
    # END SOLUTION

is stripped from ``exercise.py``:

* an indented block (inside a function) becomes ``raise NotImplementedError(...)``;
* a top-level block disappears entirely;
* lines inside a block that start with ``# HINT:`` are kept, so the stub
  carries its hints.

Existing ``exercise.py`` files hold your work and are never overwritten
unless you pass ``--force`` (optionally with a lab name).

    python tools/make_exercises.py                 # create missing stubs
    python tools/make_exercises.py --force 05_transformer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"
BEGIN = "# BEGIN SOLUTION"
END = "# END SOLUTION"
DEF_RE = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)")


def strip_solution(source: str, lab: str) -> str:
    out: list[str] = []
    defs: list[tuple[int, str]] = []  # (indent, name) of enclosing functions
    in_block = False
    block_indent = ""
    hints: list[str] = []

    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not in_block:
            m = DEF_RE.match(line)
            if m:
                indent = len(m.group(1))
                defs = [d for d in defs if d[0] < indent] + [(indent, m.group(2))]
            if stripped == BEGIN:
                in_block = True
                block_indent = line[: len(line) - len(line.lstrip())]
                hints = []
                continue
            if stripped == END:
                raise SyntaxError(f"{lab}: line {lineno}: END without BEGIN")
            out.append(line)
            continue

        if stripped == BEGIN:
            raise SyntaxError(f"{lab}: line {lineno}: nested BEGIN")
        if stripped == END:
            in_block = False
            out.extend(block_indent + h for h in hints)
            if block_indent:
                enclosing = [name for ind, name in defs if ind < len(block_indent)]
                what = enclosing[-1] if enclosing else "this block"
                out.append(f'{block_indent}raise NotImplementedError("{lab}: implement {what}")')
            continue
        if stripped.startswith("# HINT:"):
            hints.append(stripped)

    if in_block:
        raise SyntaxError(f"{lab}: unterminated {BEGIN}")
    header = (
        f"# Exercise stub generated from solution.py by tools/make_exercises.py.\n"
        f"# Replace every NotImplementedError, then run: pytest labs/{lab}\n"
    )
    return header + "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labs", nargs="*", help="lab directory names (default: all)")
    parser.add_argument("--force", action="store_true", help="overwrite existing exercise.py (destroys your work)")
    args = parser.parse_args(argv)

    names = args.labs or sorted(p.name for p in LABS.iterdir() if (p / "solution.py").exists())
    for name in names:
        solution = LABS / name / "solution.py"
        exercise = LABS / name / "exercise.py"
        if not solution.exists():
            print(f"skip {name}: no solution.py", file=sys.stderr)
            continue
        if exercise.exists() and not args.force:
            print(f"keep {exercise.relative_to(ROOT)} (exists; use --force to regenerate)")
            continue
        exercise.write_text(strip_solution(solution.read_text(), name))
        print(f"wrote {exercise.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
