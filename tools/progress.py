"""Scoreboard: how many tests YOUR implementations pass, per lab.

    python tools/progress.py            # all labs, your exercise.py files
    python tools/progress.py --impl solution
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Collector:
    def __init__(self) -> None:
        self.results: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})

    def pytest_runtest_logreport(self, report) -> None:
        lab = report.nodeid.split("/")[1] if report.nodeid.startswith("labs/") else "?"
        if report.when == "call" or (report.when == "setup" and not report.passed):
            key = "passed" if report.passed else "skipped" if report.skipped else "failed"
            self.results[lab][key] += 1

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            lab = report.nodeid.split("/")[1] if report.nodeid.startswith("labs/") else "?"
            self.results[lab]["failed"] += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="exercise")
    args = parser.parse_args()
    os.environ["LABS_IMPL"] = args.impl
    os.chdir(ROOT)

    collector = Collector()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main(["-q", "--tb=no", "-m", "not slow", "labs"], plugins=[collector])

    total_pass = total = 0
    print(f"\n{'lab':28} {'passed':>7} {'failed':>7}  progress")
    for lab in sorted(collector.results):
        r = collector.results[lab]
        n = r["passed"] + r["failed"]
        total_pass += r["passed"]
        total += n
        bar = "#" * round(20 * r["passed"] / n) if n else ""
        print(f"{lab:28} {r['passed']:>7} {r['failed']:>7}  [{bar:<20}]")
    if total:
        print(f"\n{total_pass}/{total} tests passing ({100 * total_pass / total:.0f}%) for impl={args.impl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
