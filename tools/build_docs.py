"""Build the Achilles docs site (MkDocs Material) from the repo's markdown.

    pip install -r requirements-docs.txt
    python tools/build_docs.py            # build into site/
    python tools/build_docs.py --serve    # live preview at http://127.0.0.1:8000

MkDocs cannot use the repository root as its docs folder, so this script stages every
markdown page and assets/ into site_src/ (git-ignored) and adapts GitHub-only syntax on
the way:

* GitHub alerts (> [!TIP]) become MkDocs admonitions;
* <details>, <div> and <table> blocks get markdown="1" so markdown inside them renders;
* links to files that are not pages (solution.py, LICENSE, manifest.json, ...) point
  to the file on GitHub instead of a missing page.

Headings keep GitHub-style anchors (see tools/docs_hooks.py), so every #anchor link that
works on GitHub works on the site too.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "site_src"
REPO_URL = "https://github.com/Lourdhu02/achilles"
SKIP_DIRS = {".git", ".github", ".opencode", ".claude", ".venv", "venv", "site", "site_src", "data", "runs",
             "checkpoints", "wandb", "node_modules", "__pycache__", ".pytest_cache", "pdfs"}
FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
ALERT = re.compile(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\n((?:>.*(?:\n|$))*)", re.M)
KIND = {"NOTE": "note", "TIP": "tip", "IMPORTANT": "info", "WARNING": "warning", "CAUTION": "danger"}
LINK = re.compile(r"(\]\()([^)\s]+)(\))")
OPEN_TAG = re.compile(r"<(details|div|table|tr|td)((?:\s[^>]*)?)>")
MATHJAX_CONFIG = """window.MathJax = {
  tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]], processEscapes: true, processEnvironments: true },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" }
};
document$.subscribe(() => { MathJax.startup.output.clearCache(); MathJax.typesetClear(); MathJax.texReset(); MathJax.typesetPromise(); });
"""


def pages() -> list[Path]:
    found = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if not any(part in SKIP_DIRS for part in rel.parts[:-1]):
            found.append(rel)
    return sorted(found)


def alert_to_admonition(match: re.Match) -> str:
    body = [re.sub(r"^> ?", "", line) for line in match.group(2).splitlines()]
    return f"!!! {KIND[match.group(1)]}\n" + "".join(f"    {line}\n" if line.strip() else "\n" for line in body)


def rewrite_link(target: str, src: Path, staged: set[Path]) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "/")):
        return target
    path, _, anchor = target.partition("#")
    resolved = (ROOT / src.parent / path).resolve()
    try:
        rel = resolved.relative_to(ROOT)
    except ValueError:
        return target
    if rel in staged or rel.parts[:1] == ("assets",) or not resolved.exists():
        return target
    kind = "tree" if resolved.is_dir() else "blob"
    return f"{REPO_URL}/{kind}/main/{rel.as_posix()}" + (f"#{anchor}" if anchor else "")


def adapt(text: str, src: Path, staged: set[Path]) -> str:
    out, last = [], 0
    for fence in FENCE.finditer(text):
        out.append(_adapt_prose(text[last:fence.start()], src, staged))
        out.append(fence.group(0))
        last = fence.end()
    out.append(_adapt_prose(text[last:], src, staged))
    return "".join(out)


def _adapt_prose(text: str, src: Path, staged: set[Path]) -> str:
    text = ALERT.sub(alert_to_admonition, text)
    text = OPEN_TAG.sub(lambda m: m.group(0) if "markdown=" in m.group(2) else f'<{m.group(1)}{m.group(2)} markdown="1">', text)
    return LINK.sub(lambda m: m.group(1) + rewrite_link(m.group(2), src, staged) + m.group(3), text)


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    staged = set(pages())
    for rel in staged:
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(adapt((ROOT / rel).read_text(encoding="utf-8"), rel, staged), encoding="utf-8")
    shutil.copytree(ROOT / "assets", STAGE / "assets")
    (STAGE / "javascripts").mkdir()
    (STAGE / "javascripts" / "mathjax.js").write_text(MATHJAX_CONFIG, encoding="utf-8")
    print(f"staged {len(staged)} pages into {STAGE.relative_to(ROOT)}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="live preview instead of a one-off build")
    parser.add_argument("--strict", action="store_true", help="fail on MkDocs warnings")
    args = parser.parse_args()
    stage()
    cmd = [sys.executable, "-m", "mkdocs", "serve" if args.serve else "build", "-f", str(ROOT / "mkdocs.yml")]
    if args.strict:
        cmd.append("--strict")
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
