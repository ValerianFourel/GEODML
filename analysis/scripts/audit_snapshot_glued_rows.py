#!/usr/bin/env python3
"""Count search-snapshot rows whose snippet glues together other results' titles.

A row is flagged when its snippet contains at least ``--min-titles`` distinct titles of
other rows from the same keyword (exact, case-insensitive substring, titles of at least
``--min-title-chars`` characters). Read-only. Optionally joins an extract's pages file
(title + newline + snippet, as shown to generators) to count how often flagged rows were
actually shown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read_snapshot(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path, columns=["keyword", "position", "title", "url", "snippet"]).to_pylist()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def glued_titles(rows: list[dict], *, min_title_chars: int = 20) -> list[int]:
    """Per row, the number of other same-keyword titles found inside its snippet."""
    by_keyword = defaultdict(list)
    for index, row in enumerate(rows):
        by_keyword[row["keyword"]].append(index)
    counts = [0] * len(rows)
    for members in by_keyword.values():
        titles = {(rows[i]["title"] or "").strip().casefold() for i in members}
        titles = {t for t in titles if len(t) >= min_title_chars}
        for i in members:
            own = (rows[i]["title"] or "").strip().casefold()
            snippet = (rows[i]["snippet"] or "").casefold()
            counts[i] = sum(1 for t in titles if t != own and t in snippet)
    return counts


def page_id(title: str, snippet: str) -> str:
    """Same identity as page_readiness_ordering.page_text/page_id."""
    return hashlib.sha256(f"{title.strip()}\n{snippet.strip()}".encode()).hexdigest()


def audit(path: Path, *, min_titles: int, min_title_chars: int, pages: dict | None) -> dict:
    rows = read_snapshot(path)
    counts = glued_titles(rows, min_title_chars=min_title_chars)
    flagged = [i for i, c in enumerate(counts) if c >= min_titles]
    keywords = {rows[i]["keyword"] for i in flagged}
    report = {"snapshot": str(path), "rows": len(rows), "keywords": len({r["keyword"] for r in rows}),
              "flagged_rows": len(flagged), "flagged_share": len(flagged) / max(1, len(rows)),
              "keywords_with_a_flagged_row": len(keywords),
              "glued_title_count_distribution": dict(sorted(Counter(counts[i] for i in flagged).items())),
              "examples": [{"keyword": rows[i]["keyword"], "url": rows[i]["url"], "glued_titles": counts[i],
                            "snippet": (rows[i]["snippet"] or "")[:200]}
                           for i in sorted(flagged, key=lambda i: -counts[i])[:5]]}
    if pages is not None:
        ids = {page_id(rows[i]["title"] or "", rows[i]["snippet"] or "") for i in flagged}
        shown = [p for p in ids if p in pages]
        report["flagged_snippets_shown"] = len(shown)
        report["displays_of_flagged_snippets"] = sum(pages[p] for p in shown)
        report["displays_total"] = sum(pages.values())
        report["display_share"] = report["displays_of_flagged_snippets"] / max(1, report["displays_total"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, action="append", required=True, metavar="ENGINE=PATH")
    parser.add_argument("--pages", type=Path, help="extract pages.jsonl.gz to count displays of flagged rows")
    parser.add_argument("--min-titles", type=int, default=2)
    parser.add_argument("--min-title-chars", type=int, default=20)
    args = parser.parse_args(argv)
    pages = None
    if args.pages:
        with gzip.open(args.pages, "rt", encoding="utf-8") as stream:
            pages = {json.loads(line)["page_id"]: json.loads(line)["occurrences"] for line in stream if line.strip()}
    for spec in args.snapshot:
        engine, _, path = str(spec).partition("=")
        report = audit(Path(path), min_titles=args.min_titles, min_title_chars=args.min_title_chars, pages=pages)
        print(json.dumps({"engine": engine, **report}, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
