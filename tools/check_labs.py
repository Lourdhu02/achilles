import sys
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"

sys.path.insert(0, str(ROOT))
from tools.make_exercises import strip_solution

def check_stubs():
    print("Checking stubs against solution.py...")
    drift = False
    for p in sorted(LABS.iterdir()):
        if not p.is_dir() or not (p / "solution.py").exists():
            continue
        expected = strip_solution((p / "solution.py").read_text(encoding="utf-8"), p.name)
        actual = (p / "exercise.py").read_text(encoding="utf-8")
        if expected != actual:
            print(f"ERROR: {p.name}/exercise.py has drifted from solution.py.")
            print(f"  Fix: python tools/make_exercises.py --force {p.name}")
            drift = True
    return drift

def get_readme_counts():
    text = (LABS / "README.md").read_text(encoding="utf-8")
    counts = {}
    for line in text.splitlines():
        if m := re.match(r"^\|\s*(\d{2})\s*\|.*?\|\s*(\d+)\s*\|", line):
            counts[m.group(1)] = int(m.group(2))
    return counts

def check_tests():
    print("Checking test failures and counts...")
    readme_counts = get_readme_counts()
    failed = False

    for p in sorted(LABS.iterdir()):
        if not p.is_dir() or not (p / "solution.py").exists():
            continue
        
        lab_num = p.name.split("_")[0]
        
        # 1. Count tests
        res = subprocess.run([sys.executable, "-m", "pytest", str(p), "--collect-only", "-q"], capture_output=True, text=True, cwd=str(ROOT))
        collected = sum(1 for line in res.stdout.splitlines() if "::" in line)
        
        expected = readme_counts.get(lab_num)
        if expected is not None and collected != expected:
            print(f"ERROR: {p.name} has {collected} tests collected, but README.md claims {expected}.")
            failed = True
        
        # 2. Check that every stub failure is a NotImplementedError. Some tests
        # deliberately validate fixed formulas or fixtures and do not exercise
        # learner code, so a passing test on a stub is valid. Terminal output is
        # intentionally abbreviated by pytest, so parsing its FAILED lines can
        # hide the exception type and produce false failures. JUnit XML
        # preserves each result's full message.
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as report:
            report_path = Path(report.name)
        try:
            res2 = subprocess.run(
                [sys.executable, "-m", "pytest", str(p), "--impl=exercise", "-q", f"--junitxml={report_path}"],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            try:
                cases = ET.parse(report_path).getroot().findall(".//testcase")
            except (ET.ParseError, FileNotFoundError) as exc:
                print(f"ERROR: {p.name} did not produce a readable JUnit report: {exc}")
                failed = True
                continue

            if res2.returncode not in (0, 1):
                print(f"ERROR: {p.name} pytest exited with {res2.returncode}: {res2.stderr.strip()}")
                failed = True

            for case in cases:
                outcome = case.find("failure")
                if outcome is None:
                    outcome = case.find("error")
                if outcome is None:
                    continue

                detail = outcome.attrib.get("message", "") + (outcome.text or "")
                if "NotImplementedError" not in detail:
                    print(f"ERROR: {p.name} -> {case.attrib['name']}: {outcome.attrib.get('message', '')}")
                    failed = True
        finally:
            report_path.unlink(missing_ok=True)

    return failed

def main():
    drift = check_stubs()
    tests_failed = check_tests()
    if drift or tests_failed:
        sys.exit(1)
    print("All labs checked successfully.")

if __name__ == "__main__":
    main()
