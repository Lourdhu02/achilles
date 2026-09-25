"""Download the PDFs listed in library/manifest.json for local, personal study.

Standard library only. Files land in library/pdfs/<topic>/<id>.pdf, which is
git-ignored: never commit downloaded papers.

Examples:
    python tools/fetch_library.py --list
    python tools/fetch_library.py --list --topic flashattention
    python tools/fetch_library.py --dry-run
    python tools/fetch_library.py --topic flashattention --topic kv-cache-serving
    python tools/fetch_library.py --id flashattention-1 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "library" / "manifest.json"
OUT_DIR = ROOT / "library" / "pdfs"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
TIMEOUT_S = 60
RETRIES = 3
BACKOFF_S = 2.0  # waits 2 s, then 4 s between attempts
DELAY_S = 1.0  # pause between downloads; arXiv asks for polite crawling

LICENSE_REMINDER = (
    "Reminder: these PDFs are for personal study only. They stay in library/pdfs/ "
    "(git-ignored). Do not commit or redistribute them; each keeps its own license "
    "(see the 'license' field in library/manifest.json and the landing page)."
)


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: expected a JSON array")
    ids = [e["id"] for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"{path}: duplicate ids: {sorted(dupes)}")
    return entries


def select(entries: list[dict], topics: list[str] | None, ids: list[str] | None) -> list[dict]:
    known = {e["topic"] for e in entries}
    for t in topics or []:
        if t not in known:
            raise SystemExit(f"unknown topic {t!r}; known topics: {', '.join(sorted(known))}")
    out = entries
    if topics:
        out = [e for e in out if e["topic"] in topics]
    if ids:
        out = [e for e in out if e["id"] in ids]
    return out


def target_path(entry: dict) -> Path:
    return OUT_DIR / entry["topic"] / f"{entry['id']}.pdf"


def fetch(url: str) -> bytes:
    """GET url with retries and exponential backoff; return the body."""
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if 400 <= e.code < 500 and e.code not in (408, 429):
                break  # client errors other than timeout/rate-limit will not fix themselves
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        if attempt < RETRIES:
            time.sleep(BACKOFF_S * 2 ** (attempt - 1))
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


def download(entry: dict, force: bool, dry_run: bool) -> tuple[str, str]:
    """Return (status, detail) for one entry."""
    dest = target_path(entry)
    rel = dest.relative_to(ROOT).as_posix()
    if dest.exists() and not force:
        return "skipped", f"exists: {rel}"
    if dry_run:
        return "would-get", f"{entry['pdf_url']} -> {rel}"
    try:
        body = fetch(entry["pdf_url"])
    except RuntimeError as e:
        return "failed", str(e)
    if not body.startswith(b"%PDF"):
        head = body[:60].decode("latin-1", errors="replace").replace("\n", " ")
        return "failed", f"not a PDF (body starts {head!r})"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    time.sleep(DELAY_S)
    return "ok", f"{len(body) / 1e6:.1f} MB -> {rel}"


def table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    widths[-1] = min(widths[-1], 90)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for r in rows:
        r = (*r[:-1], r[-1] if len(r[-1]) <= 90 else r[-1][:87] + "...")
        lines.append(fmt.format(*r))
    return "\n".join(lines)


def cmd_list(entries: list[dict]) -> None:
    rows = []
    for e in entries:
        pdf = "pdf" if e.get("pdf_url") else "-"
        have = "yes" if e.get("pdf_url") and target_path(e).exists() else ""
        rows.append((e["topic"], e["id"], e["kind"], e["level"], pdf, have, e["title"]))
    print(table(rows, ("topic", "id", "kind", "level", "pdf", "local", "title")))
    n_pdf = sum(1 for e in entries if e.get("pdf_url"))
    print(f"\n{len(entries)} entries, {n_pdf} with a PDF, "
          f"{len({e['topic'] for e in entries})} topics.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic", action="append", help="only this topic (repeatable)")
    p.add_argument("--id", action="append", dest="ids", help="only this entry id (repeatable)")
    p.add_argument("--list", action="store_true", help="list entries and exit")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    p.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    entries = select(load_manifest(args.manifest), args.topic, args.ids)
    if args.list:
        cmd_list(entries)
        return 0

    todo = [e for e in entries if e.get("pdf_url")]
    if not todo:
        print("Nothing to download for this selection.")
        return 0
    print(LICENSE_REMINDER + "\n")
    results = []
    for i, e in enumerate(todo, 1):
        status, detail = download(e, args.force, args.dry_run)
        print(f"[{i}/{len(todo)}] {status:<9} {e['id']}", flush=True)
        results.append((e["topic"], e["id"], status, detail))

    print("\n" + table(results, ("topic", "id", "status", "detail")))
    counts = {s: sum(1 for r in results if r[2] == s) for s in ("ok", "skipped", "would-get", "failed")}
    print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items() if v))
    if counts["failed"]:
        print("Some downloads failed. Re-run later, or open the landing_url in a browser "
              "(some hosts block scripts or need a network without a proxy).")
    print("\n" + LICENSE_REMINDER)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
