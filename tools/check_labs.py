import sys
import re
import subprocess
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
        
        # 2. Check that all failures are NotImplementedError
        res2 = subprocess.run([sys.executable, "-m", "pytest", str(p), "--impl=exercise", "-q", "--tb=line"], capture_output=True, text=True, cwd=str(ROOT))
        
        passed_count = 0
        non_not_impl = 0
        
        # In -q, pytest prints a line like "1 passed, 36 failed in 0.12s" at the end.
        for line in res2.stdout.splitlines():
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                if "NotImplementedError" not in line:
                    print(f"ERROR: {p.name} -> {line}")
                    failed = True
                    non_not_impl += 1
            if line.startswith("PASSED "):
                print(f"ERROR: {p.name} test passes on stub -> {line}")
                failed = True
                passed_count += 1
                
        # Also check the summary line
        if res2.stdout:
            last_line = res2.stdout.splitlines()[-1]
            if " passed" in last_line and " warnings" not in last_line.split(" passed")[0]: 
                # Be careful, it could be "1 passed, 1 warning"
                # Let's just check if there is a number before " passed"
                if re.search(r'\b\d+\s+passed\b', last_line):
                    print(f"ERROR: {p.name} has passing tests on the stub: {last_line}")
                    failed = True

    return failed

def main():
    drift = check_stubs()
    tests_failed = check_tests()
    if drift or tests_failed:
        sys.exit(1)
    print("All labs checked successfully.")

if __name__ == "__main__":
    main()
