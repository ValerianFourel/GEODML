"""Bundle A' — rescue broken confounders via DataForSEO.

Pulls, for the full corpus:
  - Bulk Ranks            ($0.00006/domain  = ~$0.81)
  - Bulk Backlinks        ($0.00006/domain  = ~$0.81)
  - Bulk Referring Domains($0.00006/domain  = ~$0.81)
  - Bulk Spam Score       ($0.00006/domain  = ~$0.81)
  - Labs Bulk Keyword Difficulty ($0.0005/keyword = ~$0.51)

Total target cost ≈ $3.75. Hard cost ceiling is enforced by --max-cost.

Output files (under paperSizeExperiment/dataforseo/output/):
  raw/bulk_ranks_chunk_<i>.json           — full API responses
  raw/bulk_backlinks_chunk_<i>.json
  raw/bulk_referring_domains_chunk_<i>.json
  raw/bulk_spam_score_chunk_<i>.json
  raw/bulk_keyword_difficulty_chunk_<i>.json
  bulk_ranks.csv                          — flat: target, rank, backlinks_spam_score, ...
  bulk_backlinks.csv                      — flat: target, backlinks, ...
  bulk_referring_domains.csv              — flat: target, referring_domains, ...
  bulk_spam_score.csv                     — flat: target, backlinks_spam_score
  bulk_keyword_difficulty.csv             — flat: keyword, keyword_difficulty
  run_manifest.json                       — inputs, timings, costs, chunk counts

Usage:
  python -m paperSizeExperiment.dataforseo.run_bundle_a --dry-run
  python -m paperSizeExperiment.dataforseo.run_bundle_a --max-cost 5
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

OUTPUT_DIR = SCRIPT_DIR / "output"
RAW_DIR = OUTPUT_DIR / "raw"

DOMAINS_SOURCE = EXPERIMENT_DIR / "consolidated_results" / "regression_dataset.csv"
KEYWORDS_SOURCE = EXPERIMENT_DIR / "keywords.txt"

PRICE_PER_DOMAIN_CALL = 0.00006
PRICE_PER_KEYWORD_CALL = 0.0005

DOMAIN_ENDPOINTS = [
    ("bulk_ranks", "bulk_ranks"),
    ("bulk_backlinks", "bulk_backlinks"),
    ("bulk_referring_domains", "bulk_referring_domains"),
    ("bulk_spam_score", "bulk_spam_score"),
]


def load_domains() -> list[str]:
    df = pd.read_csv(DOMAINS_SOURCE, usecols=["domain"], low_memory=False)
    domains = (
        df["domain"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    domains.sort()
    return domains


def load_keywords() -> list[str]:
    lines = KEYWORDS_SOURCE.read_text(encoding="utf-8").splitlines()
    keywords = []
    seen = set()
    for line in lines:
        kw = line.strip()
        if not kw or kw.startswith("#"):
            continue
        if kw not in seen:
            seen.add(kw)
            keywords.append(kw)
    return keywords


def estimate_cost(n_domains: int, n_keywords: int) -> dict[str, float]:
    per_endpoint = n_domains * PRICE_PER_DOMAIN_CALL
    total_domain_cost = per_endpoint * len(DOMAIN_ENDPOINTS)
    kd_cost = n_keywords * PRICE_PER_KEYWORD_CALL
    return {
        "n_domains": n_domains,
        "n_keywords": n_keywords,
        "cost_per_domain_endpoint": round(per_endpoint, 4),
        "cost_all_domain_endpoints": round(total_domain_cost, 4),
        "cost_keyword_difficulty": round(kd_cost, 4),
        "cost_total": round(total_domain_cost + kd_cost, 4),
    }


def _extract_items(body: dict) -> tuple[list[dict], list[tuple[int, str]]]:
    tasks = body.get("tasks") or []
    items: list[dict] = []
    errors: list[tuple[int, str]] = []
    for task in tasks:
        code = task.get("status_code")
        if code != 20000:
            errors.append((code, task.get("status_message", "")))
            continue
        for result in task.get("result") or []:
            for item in result.get("items") or []:
                items.append(item)
    return items, errors


def _flatten(item: dict) -> dict:
    """Flatten nested backlinks_info / ranked_serp_element etc. to single-level."""
    out: dict = {}
    for k, v in item.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_domain_endpoint(
    client: DataForSEOClient,
    endpoint: str,
    domains: list[str],
) -> tuple[list[dict], list[str], list[tuple[int, str]]]:
    method = getattr(client, endpoint)
    all_items: list[dict] = []
    chunk_files: list[str] = []
    all_errors: list[tuple[int, str]] = []
    for i, chunk in enumerate(batched(domains)):
        t0 = time.time()
        body = method(chunk)
        elapsed = time.time() - t0
        raw_path = RAW_DIR / f"{endpoint}_chunk_{i:03d}.json"
        raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        chunk_files.append(str(raw_path.relative_to(OUTPUT_DIR)))
        raw_items, errors = _extract_items(body)
        all_errors.extend(errors)
        items = [_flatten(it) for it in raw_items]
        all_items.extend(items)
        print(
            f"  [{endpoint}] chunk {i + 1}: {len(chunk)} targets, "
            f"{len(items)} items, {elapsed:.1f}s"
            + (f", errors={errors}" if errors else "")
        )
        if errors and any(code == 40204 for code, _ in errors):
            print(
                f"  [{endpoint}] ABORT: 40204 Access denied. "
                f"Skipping remaining chunks for this endpoint. "
                f"Activate the required subscription and re-run."
            )
            break
        polite_sleep(0.5)
    return all_items, chunk_files, all_errors


def run_keyword_difficulty(
    client: DataForSEOClient,
    keywords: list[str],
    location_code: int,
    language_code: str,
) -> tuple[list[dict], list[str], list[tuple[int, str]]]:
    all_items: list[dict] = []
    chunk_files: list[str] = []
    all_errors: list[tuple[int, str]] = []
    for i, chunk in enumerate(batched(keywords)):
        t0 = time.time()
        body = client.bulk_keyword_difficulty(
            chunk, location_code=location_code, language_code=language_code
        )
        elapsed = time.time() - t0
        raw_path = RAW_DIR / f"bulk_keyword_difficulty_chunk_{i:03d}.json"
        raw_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        chunk_files.append(str(raw_path.relative_to(OUTPUT_DIR)))
        raw_items, errors = _extract_items(body)
        all_errors.extend(errors)
        items = [_flatten(it) for it in raw_items]
        all_items.extend(items)
        print(
            f"  [bulk_keyword_difficulty] chunk {i + 1}: {len(chunk)} keywords, "
            f"{len(items)} items, {elapsed:.1f}s"
            + (f", errors={errors}" if errors else "")
        )
        if errors and any(code == 40204 for code, _ in errors):
            print(
                f"  [bulk_keyword_difficulty] ABORT: 40204 Access denied."
            )
            break
        polite_sleep(0.5)
    return all_items, chunk_files, all_errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost and print the run plan; do not call the API.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Hard ceiling (USD). Run aborts if estimated cost exceeds this.",
    )
    parser.add_argument(
        "--location-code",
        type=int,
        default=2840,
        help="DataForSEO location code for KD (2840 = USA). "
        "Labs does not offer English for Germany (2276); use 2840 for en.",
    )
    parser.add_argument(
        "--language-code",
        type=str,
        default="en",
        help="DataForSEO language code for KD.",
    )
    parser.add_argument(
        "--skip-backlinks",
        action="store_true",
        help="Skip the 4 Backlinks-API endpoints (they need a paid subscription).",
    )
    parser.add_argument(
        "--skip-kd",
        action="store_true",
        help="Skip bulk keyword difficulty.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading domains from {DOMAINS_SOURCE}...")
    domains = load_domains()
    print(f"  {len(domains)} unique domains")

    print(f"Loading keywords from {KEYWORDS_SOURCE}...")
    keywords = load_keywords()
    print(f"  {len(keywords)} unique keywords")

    cost = estimate_cost(len(domains), len(keywords))
    print("\nCost estimate:")
    for k, v in cost.items():
        print(f"  {k}: {v}")

    if cost["cost_total"] > args.max_cost:
        print(
            f"\nABORT: estimated cost ${cost['cost_total']} exceeds ceiling ${args.max_cost}"
        )
        return 2

    if args.dry_run:
        print("\nDry run — no API calls made.")
        return 0

    print("\nStarting live run...")
    t_start = datetime.now(timezone.utc)
    client = DataForSEOClient()

    manifest: dict = {
        "started_utc": t_start.isoformat(),
        "n_domains": len(domains),
        "n_keywords": len(keywords),
        "cost_estimate": cost,
        "endpoints": {},
    }

    if args.skip_backlinks:
        print("\nSkipping backlinks endpoints (per --skip-backlinks).")
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

    if args.skip_kd:
        print("\nSkipping bulk keyword difficulty (per --skip-kd).")
    else:
        print(f"\n-> bulk_keyword_difficulty")
        kd_items, kd_chunks, kd_errors = run_keyword_difficulty(
            client, keywords, args.location_code, args.language_code
        )
        kd_csv = OUTPUT_DIR / "bulk_keyword_difficulty.csv"
        _write_csv(kd_items, kd_csv)
        print(f"  wrote {len(kd_items)} rows -> {kd_csv.relative_to(OUTPUT_DIR)}")
        manifest["endpoints"]["bulk_keyword_difficulty"] = {
            "n_items": len(kd_items),
            "n_chunks": len(kd_chunks),
            "raw_files": kd_chunks,
            "csv": str(kd_csv.relative_to(OUTPUT_DIR)),
            "location_code": args.location_code,
            "language_code": args.language_code,
            "task_errors": kd_errors,
        }

    t_end = datetime.now(timezone.utc)
    manifest["finished_utc"] = t_end.isoformat()
    manifest["elapsed_seconds"] = (t_end - t_start).total_seconds()
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"\nDone in {manifest['elapsed_seconds']:.1f}s. "
        f"Manifest: {(OUTPUT_DIR / 'run_manifest.json').relative_to(OUTPUT_DIR)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
