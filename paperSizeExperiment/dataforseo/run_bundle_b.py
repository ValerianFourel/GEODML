"""Bundle B — keyword overview + Google SERP + retry backlinks when subscription on.

Runs three things against the corpus:

  1. Backlinks block (Bulk Ranks / Backlinks / Referring Domains / Spam Score)
     — skipped automatically with a warning on 40204.
  2. Labs keyword_overview/live for 1,011 keywords
     (returns search_volume, cpc, competition, KD, search_intent, etc.) — ~$0.12
  3. SERP Google organic/live/regular for 1,011 keywords, depth=20 — ~$2.02

Total if backlinks are active: ~$4.86. If still pending: ~$2.14.

Outputs under paperSizeExperiment/dataforseo/output/:
  keyword_overview.csv
  serp_google_organic.csv          — one row per (keyword, rank)
  bulk_*.csv                        — retried here too; empty if still 40204
  raw/keyword_overview_chunk_*.json
  raw/serp_google_organic_chunk_*.json
  run_manifest_bundle_b.json

Usage:
  source venv312/bin/activate
  python -m paperSizeExperiment.dataforseo.run_bundle_b --dry-run
  python -m paperSizeExperiment.dataforseo.run_bundle_b --max-cost 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))
load_dotenv(EXPERIMENT_DIR / ".env")

from dataforseo.client import DataForSEOClient, batched, polite_sleep  # noqa: E402
from dataforseo.run_bundle_a import (  # noqa: E402
    DOMAIN_ENDPOINTS,
    OUTPUT_DIR,
    RAW_DIR,
    _extract_items,
    _flatten,
    _write_csv,
    load_domains,
    load_keywords,
    run_domain_endpoint,
)

SERP_TASKS_PER_REQUEST = 1  # live/regular accepts only 1 task per POST (async task_post allows more)
KEYWORDS_PER_OVERVIEW_TASK = 700  # keyword_overview supports up to ~1000 per task; use 700 for safety

PRICE_KEYWORD_OVERVIEW_PER_TASK = 0.01
PRICE_KEYWORD_OVERVIEW_PER_RESULT = 0.0001
PRICE_SERP_LIVE_REGULAR_PER_TASK = 0.002
PRICE_BACKLINKS_PER_TASK = 0.02
PRICE_BACKLINKS_PER_RESULT = 0.00003


def estimate_cost(n_domains: int, n_keywords: int) -> dict[str, float]:
    n_backlinks_tasks = (n_domains + 999) // 1000
    backlinks_cost = len(DOMAIN_ENDPOINTS) * (
        n_backlinks_tasks * PRICE_BACKLINKS_PER_TASK
        + n_domains * PRICE_BACKLINKS_PER_RESULT
    )
    n_overview_tasks = (n_keywords + KEYWORDS_PER_OVERVIEW_TASK - 1) // KEYWORDS_PER_OVERVIEW_TASK
    overview_cost = (
        n_overview_tasks * PRICE_KEYWORD_OVERVIEW_PER_TASK
        + n_keywords * PRICE_KEYWORD_OVERVIEW_PER_RESULT
    )
    serp_cost = n_keywords * PRICE_SERP_LIVE_REGULAR_PER_TASK
    return {
        "n_domains": n_domains,
        "n_keywords": n_keywords,
        "backlinks_cost_if_active": round(backlinks_cost, 4),
        "keyword_overview_cost": round(overview_cost, 4),
        "serp_google_cost": round(serp_cost, 4),
        "cost_total_if_backlinks_on": round(backlinks_cost + overview_cost + serp_cost, 4),
        "cost_total_if_backlinks_off": round(overview_cost + serp_cost, 4),
    }


def run_keyword_overview(
    client: DataForSEOClient, keywords: list[str], location_code: int, language_code: str
) -> tuple[list[dict], list[str], list[tuple[int, str]]]:
    all_items: list[dict] = []
    chunk_files: list[str] = []
    all_errors: list[tuple[int, str]] = []
    for i, chunk in enumerate(batched(keywords, n=KEYWORDS_PER_OVERVIEW_TASK)):
        t0 = time.time()
        body = client.keyword_overview(
            chunk, location_code=location_code, language_code=language_code
        )
        elapsed = time.time() - t0
        raw_path = RAW_DIR / f"keyword_overview_chunk_{i:03d}.json"
        raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        chunk_files.append(str(raw_path.relative_to(OUTPUT_DIR)))
        raw_items, errors = _extract_items(body)
        all_errors.extend(errors)
        for it in raw_items:
            flat = _flatten(it)
            ki = it.get("keyword_info") or {}
            kp = it.get("keyword_properties") or {}
            si = it.get("search_intent_info") or {}
            flat.update(
                {
                    "keyword": it.get("keyword"),
                    "ko.search_volume": ki.get("search_volume"),
                    "ko.cpc": ki.get("cpc"),
                    "ko.competition": ki.get("competition"),
                    "ko.competition_level": ki.get("competition_level"),
                    "ko.keyword_difficulty": kp.get("keyword_difficulty"),
                    "ko.detected_language": kp.get("detected_language"),
                    "ko.main_intent": si.get("main_intent"),
                    "ko.foreign_intent": json.dumps(si.get("foreign_intent"))
                    if si.get("foreign_intent")
                    else None,
                }
            )
            all_items.append(flat)
        print(
            f"  [keyword_overview] chunk {i + 1}: {len(chunk)} keywords, "
            f"{len(raw_items)} items, {elapsed:.1f}s"
            + (f", errors={errors}" if errors else "")
        )
        polite_sleep(0.5)
    return all_items, chunk_files, all_errors


SERP_CHUNK_PREFIX = "serp_google_organic_chunk_"


def _rows_from_serp_body(body: dict) -> tuple[list[dict], list[tuple[int, str]]]:
    rows: list[dict] = []
    errors: list[tuple[int, str]] = []
    for task in body.get("tasks") or []:
        tcode = task.get("status_code")
        if tcode != 20000:
            errors.append((tcode, task.get("status_message", "")))
            continue
        for result in task.get("result") or []:
            kw = result.get("keyword")
            check_url = result.get("check_url")
            se_results_count = result.get("se_results_count")
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
    return rows, errors


def _scan_existing_serp_chunks() -> tuple[set[str], int, list[Path]]:
    """Return (done_keywords, next_chunk_index, existing_chunk_paths).

    A keyword counts as done iff a chunk file contains a task with status_code
    20000 and data.keyword set. Failed tasks leave the keyword eligible for
    retry and the chunk file is left alone (later runs simply overwrite no
    existing index — new chunks start after the max on disk).
    """
    done: set[str] = set()
    paths = sorted(RAW_DIR.glob(f"{SERP_CHUNK_PREFIX}*.json"))
    max_idx = -1
    for p in paths:
        stem = p.stem  # serp_google_organic_chunk_NNN
        try:
            idx = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for task in body.get("tasks") or []:
            if task.get("status_code") != 20000:
                continue
            kw = (task.get("data") or {}).get("keyword")
            if kw:
                done.add(kw)
    return done, max_idx + 1, paths


def run_serp_google(
    client: DataForSEOClient,
    keywords: list[str],
    location_code: int,
    language_code: str,
    depth: int,
    resume: bool = True,
) -> tuple[list[dict], list[str], list[tuple[int, str]]]:
    existing_chunks: list[Path] = []
    next_idx = 0
    if resume:
        done, next_idx, existing_chunks = _scan_existing_serp_chunks()
        before = len(keywords)
        keywords = [k for k in keywords if k not in done]
        print(
            f"  [serp_google] resume: {len(done)} keywords already done "
            f"({before - len(keywords)} skipped), {len(keywords)} remaining; "
            f"next chunk index = {next_idx:03d}"
        )

    rows: list[dict] = []
    new_chunk_files: list[str] = []
    all_errors: list[tuple[int, str]] = []
    for offset, chunk in enumerate(batched(keywords, n=SERP_TASKS_PER_REQUEST)):
        i = next_idx + offset
        t0 = time.time()
        body = client.google_organic_live(
            chunk,
            location_code=location_code,
            language_code=language_code,
            depth=depth,
        )
        elapsed = time.time() - t0
        raw_path = RAW_DIR / f"{SERP_CHUNK_PREFIX}{i:03d}.json"
        raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        new_chunk_files.append(str(raw_path.relative_to(OUTPUT_DIR)))
        chunk_rows, chunk_errors = _rows_from_serp_body(body)
        rows.extend(chunk_rows)
        all_errors.extend(chunk_errors)
        print(
            f"  [serp_google] chunk {i:03d}: {len(chunk)} keywords, "
            f"{len(chunk_rows)} rows, {elapsed:.1f}s"
            + (f", errors={chunk_errors}" if chunk_errors else "")
        )
        polite_sleep(0.5)

    # Rebuild the full row set from ALL chunks on disk so the CSV is complete.
    all_rows: list[dict] = []
    all_paths = sorted(RAW_DIR.glob(f"{SERP_CHUNK_PREFIX}*.json"))
    for p in all_paths:
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        body_rows, _ = _rows_from_serp_body(body)
        all_rows.extend(body_rows)
    all_chunk_files = [str(p.relative_to(OUTPUT_DIR)) for p in all_paths]
    print(
        f"  [serp_google] merged {len(all_paths)} chunk files "
        f"({len(existing_chunks)} existing + {len(new_chunk_files)} new) "
        f"-> {len(all_rows)} rows total"
    )
    return all_rows, all_chunk_files, all_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cost", type=float, default=6.0)
    parser.add_argument("--location-code", type=int, default=2840)
    parser.add_argument("--language-code", type=str, default="en")
    parser.add_argument("--serp-depth", type=int, default=20)
    parser.add_argument("--skip-backlinks", action="store_true")
    parser.add_argument("--skip-keyword-overview", action="store_true")
    parser.add_argument("--skip-serp", action="store_true")
    parser.add_argument(
        "--no-resume-serp",
        action="store_true",
        help="Disable SERP resume (default: skip keywords already present "
        "in raw/serp_google_organic_chunk_*.json).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading domains / keywords...")
    domains = load_domains()
    keywords = load_keywords()
    print(f"  {len(domains)} unique domains, {len(keywords)} unique keywords")

    cost = estimate_cost(len(domains), len(keywords))
    print("\nCost estimate:")
    for k, v in cost.items():
        print(f"  {k}: {v}")

    scheduled = 0.0
    if not args.skip_backlinks:
        scheduled += cost["backlinks_cost_if_active"]
    if not args.skip_keyword_overview:
        scheduled += cost["keyword_overview_cost"]
    if not args.skip_serp:
        scheduled += cost["serp_google_cost"]
    scheduled = round(scheduled, 4)
    print(f"  scheduled (after skip flags): {scheduled}")

    if scheduled > args.max_cost:
        print(f"\nABORT: scheduled ${scheduled} > ceiling ${args.max_cost}")
        return 2

    if args.dry_run:
        print("\nDry run — no API calls.")
        return 0

    print("\nStarting live run...")
    t_start = datetime.now(timezone.utc)
    client = DataForSEOClient()
    manifest: dict = {
        "started_utc": t_start.isoformat(),
        "cost_estimate": cost,
        "endpoints": {},
    }

    if args.skip_backlinks:
        print("\nSkipping backlinks (per flag).")
    else:
        for _, endpoint in DOMAIN_ENDPOINTS:
            print(f"\n-> {endpoint}")
            items, chunk_files, errors = run_domain_endpoint(client, endpoint, domains)
            csv_path = OUTPUT_DIR / f"{endpoint}.csv"
            _write_csv(items, csv_path)
            print(
                f"  wrote {len(items)} rows -> {csv_path.relative_to(OUTPUT_DIR)}"
            )
            manifest["endpoints"][endpoint] = {
                "n_items": len(items),
                "n_chunks": len(chunk_files),
                "raw_files": chunk_files,
                "csv": str(csv_path.relative_to(OUTPUT_DIR)),
                "task_errors": errors,
            }

    if not args.skip_keyword_overview:
        print(f"\n-> keyword_overview")
        ko_items, ko_chunks, ko_errors = run_keyword_overview(
            client, keywords, args.location_code, args.language_code
        )
        ko_csv = OUTPUT_DIR / "keyword_overview.csv"
        _write_csv(ko_items, ko_csv)
        print(f"  wrote {len(ko_items)} rows -> {ko_csv.relative_to(OUTPUT_DIR)}")
        manifest["endpoints"]["keyword_overview"] = {
            "n_items": len(ko_items),
            "n_chunks": len(ko_chunks),
            "raw_files": ko_chunks,
            "csv": str(ko_csv.relative_to(OUTPUT_DIR)),
            "location_code": args.location_code,
            "language_code": args.language_code,
            "task_errors": ko_errors,
        }

    if not args.skip_serp:
        print(f"\n-> serp_google_organic (depth={args.serp_depth})")
        serp_rows, serp_chunks, serp_errors = run_serp_google(
            client,
            keywords,
            args.location_code,
            args.language_code,
            args.serp_depth,
            resume=not args.no_resume_serp,
        )
        serp_csv = OUTPUT_DIR / "serp_google_organic.csv"
        _write_csv(serp_rows, serp_csv)
        print(f"  wrote {len(serp_rows)} rows -> {serp_csv.relative_to(OUTPUT_DIR)}")
        manifest["endpoints"]["serp_google_organic"] = {
            "n_items": len(serp_rows),
            "n_chunks": len(serp_chunks),
            "raw_files": serp_chunks,
            "csv": str(serp_csv.relative_to(OUTPUT_DIR)),
            "location_code": args.location_code,
            "language_code": args.language_code,
            "depth": args.serp_depth,
            "task_errors": serp_errors,
        }

    t_end = datetime.now(timezone.utc)
    manifest["finished_utc"] = t_end.isoformat()
    manifest["elapsed_seconds"] = (t_end - t_start).total_seconds()
    (OUTPUT_DIR / "run_manifest_bundle_b.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"\nDone in {manifest['elapsed_seconds']:.1f}s. "
        f"Manifest: run_manifest_bundle_b.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
