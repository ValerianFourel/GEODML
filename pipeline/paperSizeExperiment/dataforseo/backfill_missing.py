"""Backfill DataForSEO data for keywords missing from keyword_overview.

For the 161 keywords not in Labs keyword_overview:
  - Google Ads search_volume/live  -> search_volume + cpc (ground truth, ~$0.05)
  - Labs search_intent/live         -> main_intent + foreign_intent (~$0.02)

Writes raw responses to output/raw/ and flat CSVs to output/ so the existing
merge_into_regression.py can pick them up (via keyword_overview if we fold
them in, or via new dfs_* columns).

Usage:
  python -m paperSizeExperiment.dataforseo.backfill_missing --dry-run
  python -m paperSizeExperiment.dataforseo.backfill_missing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
load_dotenv(EXPERIMENT_DIR / ".env")

from dataforseo.client import DataForSEOClient, polite_sleep  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "output"
RAW_DIR = OUTPUT_DIR / "raw"
KEYWORDS_FILE = EXPERIMENT_DIR / "keywords.txt"
KEYWORD_OVERVIEW_CSV = OUTPUT_DIR / "keyword_overview.csv"

PRICE_GOOGLE_ADS_SV_PER_TASK = 0.05
PRICE_SEARCH_INTENT_PER_TASK = 0.01
PRICE_SEARCH_INTENT_PER_KEYWORD = 0.0001


def load_all_keywords() -> list[str]:
    lines = KEYWORDS_FILE.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        kw = line.strip()
        if not kw or kw.startswith("#"):
            continue
        if kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


def load_covered_keywords() -> set[str]:
    if not KEYWORD_OVERVIEW_CSV.exists() or KEYWORD_OVERVIEW_CSV.stat().st_size == 0:
        return set()
    ko = pd.read_csv(KEYWORD_OVERVIEW_CSV, low_memory=False)
    return set(ko["keyword"].dropna().astype(str))


def run_google_ads_sv(
    client: DataForSEOClient, keywords: list[str], location_code: int, language_code: str
) -> list[dict]:
    print(f"  [google_ads_sv] calling with {len(keywords)} keywords")
    t0 = time.time()
    body = client.google_ads_search_volume(
        keywords, location_code=location_code, language_code=language_code
    )
    elapsed = time.time() - t0
    raw_path = RAW_DIR / "google_ads_search_volume_chunk_000.json"
    raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    rows: list[dict] = []
    for task in body.get("tasks") or []:
        if task.get("status_code") != 20000:
            print(
                f"  [google_ads_sv] ERROR status={task.get('status_code')} "
                f"msg={task.get('status_message')}"
            )
            continue
        for item in task.get("result") or []:
            rows.append(
                {
                    "keyword": item.get("keyword"),
                    "ga_search_volume": item.get("search_volume"),
                    "ga_cpc": item.get("cpc"),
                    "ga_competition": item.get("competition"),
                    "ga_low_top_of_page_bid": item.get("low_top_of_page_bid"),
                    "ga_high_top_of_page_bid": item.get("high_top_of_page_bid"),
                }
            )
    print(
        f"  [google_ads_sv] {len(rows)} items, {elapsed:.1f}s, "
        f"cost=${body.get('cost', 0.0):.4f}"
    )
    return rows


def run_search_intent(
    client: DataForSEOClient, keywords: list[str], language_code: str
) -> list[dict]:
    print(f"  [search_intent] calling with {len(keywords)} keywords")
    t0 = time.time()
    body = client.search_intent(keywords, language_code=language_code)
    elapsed = time.time() - t0
    raw_path = RAW_DIR / "search_intent_chunk_000.json"
    raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    rows: list[dict] = []
    for task in body.get("tasks") or []:
        if task.get("status_code") != 20000:
            print(
                f"  [search_intent] ERROR status={task.get('status_code')} "
                f"msg={task.get('status_message')}"
            )
            continue
        for result in task.get("result") or []:
            for item in result.get("items") or []:
                ki = item.get("keyword_intent") or {}
                si = item.get("secondary_keyword_intents") or []
                rows.append(
                    {
                        "keyword": item.get("keyword"),
                        "si_main_intent": ki.get("label"),
                        "si_main_intent_prob": ki.get("probability"),
                        "si_secondary_intents": json.dumps(si) if si else None,
                    }
                )
    print(
        f"  [search_intent] {len(rows)} items, {elapsed:.1f}s, "
        f"cost=${body.get('cost', 0.0):.4f}"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--location-code", type=int, default=2840)
    parser.add_argument("--language-code", type=str, default="en")
    parser.add_argument("--max-cost", type=float, default=0.5)
    parser.add_argument("--skip-google-ads", action="store_true")
    parser.add_argument("--skip-search-intent", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_kw = load_all_keywords()
    covered = load_covered_keywords()
    missing = [k for k in all_kw if k not in covered]

    print(f"Total keywords: {len(all_kw)}")
    print(f"Already in keyword_overview.csv: {len(covered)}")
    print(f"Missing: {len(missing)}")

    if not missing:
        print("Nothing to backfill.")
        return 0

    scheduled = 0.0
    if not args.skip_google_ads:
        scheduled += PRICE_GOOGLE_ADS_SV_PER_TASK
    if not args.skip_search_intent:
        scheduled += PRICE_SEARCH_INTENT_PER_TASK + len(missing) * PRICE_SEARCH_INTENT_PER_KEYWORD
    scheduled = round(scheduled, 4)

    print(f"\nEstimated cost: ${scheduled}")
    if scheduled > args.max_cost:
        print(f"ABORT: scheduled ${scheduled} > ceiling ${args.max_cost}")
        return 2

    if args.dry_run:
        print("Dry run — no API calls.")
        print(f"First 10 missing keywords: {missing[:10]}")
        return 0

    client = DataForSEOClient()

    if not args.skip_google_ads:
        print("\n-> google_ads/search_volume/live")
        sv_rows = run_google_ads_sv(
            client, missing, args.location_code, args.language_code
        )
        sv_df = pd.DataFrame(sv_rows)
        sv_csv = OUTPUT_DIR / "google_ads_search_volume.csv"
        sv_df.to_csv(sv_csv, index=False)
        print(
            f"  wrote {len(sv_df)} rows -> {sv_csv.relative_to(OUTPUT_DIR)} "
            f"(ga_search_volume non-null: {int(sv_df['ga_search_volume'].notna().sum()) if 'ga_search_volume' in sv_df.columns else 0})"
        )
        polite_sleep(0.5)

    if not args.skip_search_intent:
        print("\n-> labs/search_intent/live")
        si_rows = run_search_intent(client, missing, args.language_code)
        si_df = pd.DataFrame(si_rows)
        si_csv = OUTPUT_DIR / "search_intent.csv"
        si_df.to_csv(si_csv, index=False)
        print(
            f"  wrote {len(si_df)} rows -> {si_csv.relative_to(OUTPUT_DIR)} "
            f"(si_main_intent non-null: {int(si_df['si_main_intent'].notna().sum()) if 'si_main_intent' in si_df.columns else 0})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
