"""Consolidate raw serp_google_organic_chunk_*.json into serp_google_organic.csv.

Safe to run on partial data (e.g. after stopping the runner early). Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
RAW_DIR = OUTPUT_DIR / "raw"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from dataforseo.run_bundle_a import _write_csv  # noqa: E402


def main() -> int:
    chunks = sorted(RAW_DIR.glob("serp_google_organic_chunk_*.json"))
    if not chunks:
        print("no chunks found")
        return 1

    rows: list[dict] = []
    bad_tasks = 0
    keywords_seen: set[str] = set()
    for path in chunks:
        body = json.loads(path.read_text(encoding="utf-8"))
        for task in body.get("tasks") or []:
            if task.get("status_code") != 20000:
                bad_tasks += 1
                continue
            for result in task.get("result") or []:
                kw = result.get("keyword")
                if kw:
                    keywords_seen.add(kw)
                se_results_count = result.get("se_results_count")
                check_url = result.get("check_url")
                for item in result.get("items") or []:
                    if item.get("type") != "organic":
                        continue
                    rows.append(
                        {
                            "keyword": kw,
                            "rank_group": item.get("rank_group"),
                            "rank_absolute": item.get("rank_absolute"),
                            "domain": item.get("domain"),
                            "title": item.get("title"),
                            "description": item.get("description"),
                            "url": item.get("url"),
                            "breadcrumb": item.get("breadcrumb"),
                            "is_featured_snippet": item.get("is_featured_snippet"),
                            "se_results_count": se_results_count,
                            "check_url": check_url,
                        }
                    )

    csv_path = OUTPUT_DIR / "serp_google_organic.csv"
    _write_csv(rows, csv_path)
    print(
        f"chunks: {len(chunks)}, keywords: {len(keywords_seen)}, "
        f"organic rows: {len(rows)}, bad tasks: {bad_tasks}"
    )
    print(f"wrote {csv_path.relative_to(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
