"""Fail if a relative markdown link points at a missing file or a missing #anchor.

Anchors are checked against GitHub's heading slugs (lowercase, punctuation
dropped, spaces to hyphens, duplicates suffixed -1, -2, ...).
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", flags=re.S)
EXTERNAL = ("http://", "https://", "mailto:")
SKIP_DIRS = {"site", "site_src", "data", "runs", "node_modules", "venv"}  # build output, local data


def slugify(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip())  # [text](url) -> text
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


@lru_cache(maxsize=None)
def anchors(path: Path) -> frozenset[str]:
    seen: dict[str, int] = {}
    out = set()
    for line in FENCE.sub("", path.read_text(encoding="utf-8")).splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        slug = slugify(m.group(1))
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return frozenset(out)


def main() -> int:
    broken = []
    for md in sorted(ROOT.rglob("*.md")):
        rel = md.relative_to(ROOT)
        if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        for target in LINK.findall(FENCE.sub("", md.read_text(encoding="utf-8"))):
            if target.startswith(EXTERNAL):
                continue
            file_part, _, anchor = target.partition("#")
            path = (md.parent / file_part).resolve() if file_part else md
            if not path.exists():
                broken.append(f"{rel} -> {target} (missing file)")
            elif anchor and path.suffix == ".md" and anchor not in anchors(path):
                broken.append(f"{rel} -> {target} (missing anchor)")
    for b in broken:
        print(f"broken link: {b}")
    print(f"{len(broken)} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
