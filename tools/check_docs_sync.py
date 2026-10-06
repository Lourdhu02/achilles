import os
import re
import sys
import subprocess
from pathlib import Path

def check_docs_sync():
    """Validates that labs/README.md test counts match pytest --collect-only."""
    root = Path(__file__).resolve().parent.parent
    labs_readme = root / "labs" / "README.md"
    if not labs_readme.exists():
        print("labs/README.md not found.")
        return 1

    print("Checking doc sync...")
    content = labs_readme.read_text(encoding="utf-8")
    table_pattern = re.compile(r"\| (\d+) \| \[.*?\]\((.*?)/README\.md\) \| .*? \| (\d+) \|")
    
    mismatches = []
    
    for match in table_pattern.finditer(content):
        lab_num, lab_dir, expected_tests = match.groups()
        expected_tests = int(expected_tests)
        
        lab_path = root / "labs" / lab_dir
        if not lab_path.exists():
            continue
            
        # Run pytest --collect-only
        cmd = [sys.executable, "-m", "pytest", str(lab_path), "--impl=solution", "--collect-only", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output for 'X tests collected'
        collect_match = re.search(r"(\d+) tests? collected", result.stdout)
        if collect_match:
            actual_tests = int(collect_match.group(1))
            if actual_tests != expected_tests:
                mismatches.append(f"{lab_dir}: expected {expected_tests}, got {actual_tests}")
        else:
            mismatches.append(f"{lab_dir}: could not parse pytest output")

    if mismatches:
        print("Test count mismatches found in labs/README.md:")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print("All test counts match documentation!")
    return 0

if __name__ == "__main__":
    sys.exit(check_docs_sync())
