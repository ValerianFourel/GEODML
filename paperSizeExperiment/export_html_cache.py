#!/usr/bin/env python3
"""Export all cached HTML pages into structured JSONL files, deduplicated.

Creates consolidated_results/html_archive/ with:
  - pages_NNNN.jsonl  — batches of ~1000 pages, one JSON object per line
  - index.json        — master index mapping url -> {hash, domain, keyword, runs, batch_file}

Each JSONL line is:
{
  "url": "https://...",
  "domain": "example.com",
  "hash": "dcc91a7e0f7ea64f",
  "keywords": ["abandoned cart recovery", ...],
  "runs": ["duckduckgo_Llama-3.3-70B-Instruct_serp20_top10", ...],
  "html": "<html>..."
}

Usage:
  python paperSizeExperiment/export_html_cache.py
  python paperSizeExperiment/export_html_cache.py --batch-size 500
  python paperSizeExperiment/export_html_cache.py --dest /path/to/archive
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_DEST = SCRIPT_DIR / "consolidated_results" / "html_archive"
BATCH_SIZE = 1000


def url_to_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def build_url_index(output_root: Path) -> dict:
    """Scan all rankings.csv files to build url -> metadata mapping."""
    url_meta = {}  # url -> {domain, keywords: set, runs: set}

    skip_dirs = {"cross_model_analysis"}
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name in skip_dirs:
            continue
        rankings = run_dir / "rankings.csv"
        if not rankings.exists():
            continue

        run_name = run_dir.name
        with open(rankings, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("url", "").strip()
                if not url:
                    continue
                if url not in url_meta:
                    url_meta[url] = {
                        "domain": row.get("domain", ""),
                        "keywords": set(),
                        "runs": set(),
                    }
                kw = row.get("keyword", "").strip()
                if kw:
                    url_meta[url]["keywords"].add(kw)
                url_meta[url]["runs"].add(run_name)

    return url_meta


def find_html_file(url_hash: str, output_root: Path) -> Path | None:
    """Find the first available HTML cache file for a given hash."""
    for run_dir in output_root.iterdir():
        if not run_dir.is_dir():
            continue
        cache_file = run_dir / "html_cache" / f"{url_hash}.html"
        if cache_file.exists():
            return cache_file
    return None


def export(dest: Path, batch_size: int):
    if not OUTPUT_ROOT.exists():
        print(f"ERROR: Output root not found: {OUTPUT_ROOT}")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    # Step 1: Build URL index from rankings
    print("Building URL index from rankings.csv files...")
    url_meta = build_url_index(OUTPUT_ROOT)
    print(f"  Found {len(url_meta)} unique URLs across {len(set(r for m in url_meta.values() for r in m['runs']))} runs")

    # Step 2: Export to JSONL batches
    index = {}  # url -> {hash, domain, keywords, runs, batch_file}
    batch_num = 0
    batch_count = 0
    batch_file = None
    total_exported = 0
    total_missing = 0
    total_bytes = 0

    urls_sorted = sorted(url_meta.keys())

    for url in urls_sorted:
        meta = url_meta[url]
        h = url_to_hash(url)
        html_path = find_html_file(h, OUTPUT_ROOT)

        if html_path is None:
            total_missing += 1
            continue

        # Open new batch if needed
        if batch_file is None or batch_count >= batch_size:
            if batch_file:
                batch_file.close()
            batch_num += 1
            batch_fname = f"pages_{batch_num:04d}.jsonl"
            batch_file = open(dest / batch_fname, "w", encoding="utf-8")
            batch_count = 0
            print(f"  Writing {batch_fname}...")

        # Read HTML
        try:
            html = html_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  WARN: could not read {html_path}: {e}")
            total_missing += 1
            continue

        # Write JSONL line
        record = {
            "url": url,
            "domain": meta["domain"],
            "hash": h,
            "keywords": sorted(meta["keywords"]),
            "runs": sorted(meta["runs"]),
            "html": html,
        }
        line = json.dumps(record, ensure_ascii=False)
        batch_file.write(line + "\n")
        total_bytes += len(line.encode("utf-8"))

        # Index entry (without HTML)
        index[url] = {
            "hash": h,
            "domain": meta["domain"],
            "keywords": sorted(meta["keywords"]),
            "runs": sorted(meta["runs"]),
            "batch_file": f"pages_{batch_num:04d}.jsonl",
            "html_size_bytes": len(html.encode("utf-8")),
        }

        batch_count += 1
        total_exported += 1

        if total_exported % 2000 == 0:
            print(f"    ...exported {total_exported} pages ({total_bytes / (1024**3):.1f} GB)")

    if batch_file:
        batch_file.close()

    # Step 3: Write index
    index_path = dest / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_pages": total_exported,
            "total_missing": total_missing,
            "total_batches": batch_num,
            "batch_size": batch_size,
            "unique_domains": len(set(v["domain"] for v in index.values())),
            "unique_keywords": len(set(kw for v in index.values() for kw in v["keywords"])),
            "pages": index,
        }, f, indent=2, ensure_ascii=False)

    # Summary
    total_size_gb = total_bytes / (1024 ** 3)
    print(f"\n{'='*60}")
    print(f"HTML ARCHIVE EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Destination:    {dest}/")
    print(f"  Pages exported: {total_exported}")
    print(f"  Pages missing:  {total_missing} (URL in rankings but no cached HTML)")
    print(f"  JSONL batches:  {batch_num} files (up to {batch_size} pages each)")
    print(f"  Total size:     {total_size_gb:.2f} GB")
    print(f"  Unique domains: {len(set(v['domain'] for v in index.values()))}")
    print(f"  Unique keywords:{len(set(kw for v in index.values() for kw in v['keywords']))}")
    print(f"  Index:          {index_path}")
    print(f"\nStructure:")
    print(f"  html_archive/")
    print(f"    index.json           — master index (url -> metadata + batch pointer)")
    print(f"    pages_0001.jsonl     — first {batch_size} pages")
    print(f"    pages_0002.jsonl     — next {batch_size} pages")
    print(f"    ...                  — {batch_num} files total")


def main():
    parser = argparse.ArgumentParser(
        description="Export all cached HTML into structured JSONL archive")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST),
                        help=f"Destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Pages per JSONL file (default: {BATCH_SIZE})")
    args = parser.parse_args()

    export(Path(args.dest), args.batch_size)


if __name__ == "__main__":
    main()
