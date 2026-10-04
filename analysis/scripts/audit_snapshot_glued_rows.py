#!/usr/bin/env python3
"""Count search-snapshot rows that glue together other results' titles.

A row is flagged when its title or snippet contains at least ``--min-titles`` distinct
titles of other rows from the same keyword (exact, case-insensitive substring, titles of
at least ``--min-title-chars`` characters, its own title excluded). A real page title names
one page; it does not contain other results' full titles. Read-only. A seeded random sample
of flagged rows is printed for manual precision review. Optionally joins an extract's pages file
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
    """Per row, the number of other same-keyword titles found inside its title or snippet."""
    by_keyword = defaultdict(list)
    for index, row in enumerate(rows):
        by_keyword[row["keyword"]].append(index)
    counts = [0] * len(rows)
    for members in by_keyword.values():
        titles = {(rows[i]["title"] or "").strip().casefold() for i in members}
        titles = {t for t in titles if len(t) >= min_title_chars}
        for i in members:
            own = (rows[i]["title"] or "").strip().casefold()
            text = own + "\n" + (rows[i]["snippet"] or "").casefold()
            counts[i] = sum(1 for t in titles if t != own and t in text)
    return counts


def snapshots_from_traces(traces) -> dict[str, dict]:
    """Snapshot path and SHA-256 per engine, as recorded in the search events' raw payloads."""
    found: dict[str, dict] = {}
    for trace in traces:
        for event in trace.get("events", []):
            if not isinstance(event, dict) or event.get("event_type") != "search":
                continue
            payload = event.get("payload", {})
            raw = payload.get("raw_payload") or {}
            if payload.get("engine") and raw.get("snapshot") and raw.get("snapshot_sha256"):
                record = {"path": raw["snapshot"], "sha256": raw["snapshot_sha256"]}
                if found.setdefault(payload["engine"], record) != record:
                    raise ValueError(f"traces use more than one {payload['engine']} snapshot")
        if {"duckduckgo", "searxng"} <= found.keys():
            break
    return found


def resolve_snapshot(record: dict, search_root: Path) -> Path:
    """The recorded file if it still has the recorded hash, else a same-named copy under search_root."""
    recorded = Path(record["path"])
    candidates = [recorded] if recorded.is_file() else []
    candidates += sorted(p for p in Path(search_root).rglob(recorded.name)
                         if p.is_file() and "checkouts" not in p.parts and p != recorded)
    for path in candidates:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() == record["sha256"]:
            return path
    raise FileNotFoundError(f"no file with the recorded hash for {recorded} under {search_root}")


def page_id(title: str, snippet: str) -> str:
    """Same identity as page_readiness_ordering.page_text/page_id."""
    return hashlib.sha256(f"{title.strip()}\n{snippet.strip()}".encode()).hexdigest()


def _quantiles(values: list[int]) -> dict:
    ordered = sorted(values)
    return {q: ordered[min(len(ordered) - 1, int(float(q) * len(ordered)))] for q in ("0.5", "0.9", "0.99", "1.0")} if ordered else {}


def audit(path: Path, *, min_titles: int, min_title_chars: int, pages: dict | None, sample: int = 10) -> dict:
    import random
    rows = read_snapshot(path)
    counts = glued_titles(rows, min_title_chars=min_title_chars)
    flagged = [i for i, c in enumerate(counts) if c >= min_titles]
    keywords = {rows[i]["keyword"] for i in flagged}
    report = {"snapshot": str(path), "rows": len(rows), "keywords": len({r["keyword"] for r in rows}),
              "flagged_rows": len(flagged), "flagged_share": len(flagged) / max(1, len(rows)),
              "keywords_with_a_flagged_row": len(keywords),
              "glued_title_count_distribution": dict(sorted(Counter(counts[i] for i in flagged).items())),
              "title_chars_all": _quantiles([len(r["title"] or "") for r in rows]),
              "title_chars_flagged": _quantiles([len(rows[i]["title"] or "") for i in flagged]),
              "review_sample": [{"url": rows[i]["url"], "glued_titles": counts[i], "title": (rows[i]["title"] or "")[:240]}
                                for i in random.Random(20261004).sample(flagged, min(sample, len(flagged)))],
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
    parser.add_argument("--snapshot", type=Path, action="append", default=[], metavar="ENGINE=PATH")
    parser.add_argument("--locate-from", type=Path, action="append", default=[], metavar="DATASET_ROOT",
                        help="find each engine's snapshot from the paths and hashes recorded in this dataset's traces")
    parser.add_argument("--search-root", type=Path, help="where to look for a same-named copy (default: dataset parent)")
    parser.add_argument("--pages", type=Path, help="extract pages.jsonl.gz to count displays of flagged rows")
    parser.add_argument("--min-titles", type=int, default=2)
    parser.add_argument("--min-title-chars", type=int, default=20)
    args = parser.parse_args(argv)
    snapshots = [str(spec) for spec in args.snapshot]
    if args.locate_from:
        from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
        records = {}
        for root in args.locate_from:
            found = snapshots_from_traces(iter_sealed_rows(root, "traces", required=True))
            print(json.dumps({"dataset": str(root), "recorded_snapshots": found}, indent=1), flush=True)
            for engine, record in found.items():
                if records.setdefault(engine, record)["sha256"] != record["sha256"]:
                    print(json.dumps({"warning": f"datasets used different {engine} snapshots"}), flush=True)
        search_root = args.search_root or args.locate_from[0].parent
        snapshots += [f"{engine}={resolve_snapshot(record, search_root)}" for engine, record in sorted(records.items())]
    if not snapshots:
        parser.error("give --snapshot ENGINE=PATH or --locate-from DATASET_ROOT")
    pages = None
    if args.pages:
        with gzip.open(args.pages, "rt", encoding="utf-8") as stream:
            pages = {json.loads(line)["page_id"]: json.loads(line)["occurrences"] for line in stream if line.strip()}
    for spec in snapshots:
        engine, _, path = spec.partition("=")
        report = audit(Path(path), min_titles=args.min_titles, min_title_chars=args.min_title_chars, pages=pages)
        print(json.dumps({"engine": engine, **report}, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
