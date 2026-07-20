"""
Normalize phase0_*.json SERP dumps into a long-format Parquet.

Input JSON shape:
    { "metadata": {...},
      "serp_results": {
          "<keyword>": {
              "query": str,
              "query_timestamp_utc": str,
              "response_timestamp_utc": str,
              "search_backend": str,
              "num_requested": int,
              "raw_results": [{"position","title","url","snippet",...}, ...],
              "error": str | null,
          }, ...
      }
    }

Output: one Parquet per JSON with columns
    keyword, position, title, url, snippet, engines, score,
    search_backend, query_timestamp_utc, response_timestamp_utc,
    num_requested, error, source_file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
SERP_DIR = BUNDLE_ROOT / "data" / "serp"


def json_to_rows(json_path: Path):
    with json_path.open() as f:
        doc = json.load(f)
    source = json_path.name
    for keyword, entry in doc.get("serp_results", {}).items():
        if not isinstance(entry, dict):
            continue
        search_backend = entry.get("search_backend")
        q_ts = entry.get("query_timestamp_utc")
        r_ts = entry.get("response_timestamp_utc")
        num_req = entry.get("num_requested")
        err = entry.get("error")
        raw = entry.get("raw_results") or []
        if not raw and err:
            yield {
                "keyword": keyword,
                "position": None,
                "title": None,
                "url": None,
                "snippet": None,
                "engines": None,
                "score": None,
                "search_backend": search_backend,
                "query_timestamp_utc": q_ts,
                "response_timestamp_utc": r_ts,
                "num_requested": num_req,
                "error": err,
                "source_file": source,
            }
            continue
        for row in raw:
            engines = row.get("engines")
            if isinstance(engines, list):
                engines = "|".join(str(e) for e in engines)
            yield {
                "keyword": keyword,
                "position": row.get("position"),
                "title": row.get("title"),
                "url": row.get("url"),
                "snippet": row.get("snippet"),
                "engines": engines,
                "score": row.get("score"),
                "search_backend": search_backend,
                "query_timestamp_utc": q_ts,
                "response_timestamp_utc": r_ts,
                "num_requested": num_req,
                "error": err,
                "source_file": source,
            }


def main() -> int:
    jsons = sorted(SERP_DIR.glob("phase0_top*.json"))
    if not jsons:
        print("no phase0 SERP JSONs found under", SERP_DIR)
        return 1
    for jp in jsons:
        rows = list(json_to_rows(jp))
        if not rows:
            print(f"  {jp.name}: 0 rows (skipping)")
            continue
        df = pd.DataFrame(rows)
        out = jp.with_suffix(".parquet")
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        print(f"  {jp.name}  ({len(df):,} rows)  →  {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
