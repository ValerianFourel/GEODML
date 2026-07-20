#!/usr/bin/env python3
"""Bulk-fetch domain-authority signals from DataForSEO Domain Analytics
(Whois Overview) for every domain in our dataset. Replaces both the
(low-coverage) Moz `conf_domain_authority` and the (heuristic) `conf_brand_recog`
with richer, ~99 %-coverage signals.

NOTE: We tried Backlinks API first but the account doesn't have a Backlinks
subscription. Whois Overview is available and gives us a richer signal set:
domain age, organic-search visibility (positions across Google), paid-search
metrics, and estimated traffic value. We use these to derive a brand-authority
composite.

Endpoint:
  POST /v3/domain_analytics/whois/overview/live   (~ $0.002 per domain)

Cost: ~$48 for all ~24k domains. Per-chunk JSON checkpoints in
.checkpoints/whois/ make the run resumable after rate-limits or interrupts.

Credentials (one of these env-var pairs):
  DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD
  DFS_LOGIN        / DFS_PASSWORD
  DFS_USER         / DFS_PASS

Outputs:
  ~/geodml_data/data/dataforseo/domain_authority_dfs.parquet
    Columns (one row per domain):
      domain                      str
      dfs_organic_count           int   total organic-ranking positions in Google
      dfs_organic_pos_1           int   # of #1 organic positions
      dfs_organic_etv             float estimated traffic value (USD)
      dfs_paid_count              int   total paid-ranking positions
      dfs_created_datetime        str   ISO date domain was registered
      dfs_domain_age_years        float years since creation (computed)
      dfs_authority_log           float log10(organic_count + 1) — clean DA proxy
      dfs_brand_proxy             int   binary: organic_count ≥ 100k OR pos_1 ≥ 500

Run:
  python scripts/fetch_dfs_domain_authority.py
  python scripts/fetch_dfs_domain_authority.py --dry-run     # print plan only
  python scripts/fetch_dfs_domain_authority.py --max 100     # test on 100 domains
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path.home() / "geodml_data"
MAIN = ROOT / "data" / "main"
DFS_DIR = ROOT / "data" / "dataforseo"
OUT_PARQUET = DFS_DIR / "domain_authority_dfs.parquet"
CHECKPOINT_DIR = DFS_DIR / ".checkpoints"

WHOIS_URL = "https://api.dataforseo.com/v3/domain_analytics/whois/overview/live"
BULK_LIMIT = 1000   # max filter-list size per request


def get_credentials() -> tuple[str, str]:
    """Return (login, password) from any of the recognised env-var pairs."""
    for u, p in [("DATAFORSEO_LOGIN", "DATAFORSEO_PASSWORD"),
                 ("DFS_LOGIN", "DFS_PASSWORD"),
                 ("DFS_USER", "DFS_PASS")]:
        login, pw = os.environ.get(u), os.environ.get(p)
        if login and pw:
            return login, pw
    print("ERROR: DataForSEO credentials not found in env.\n"
          "Set one of these pairs in your shell before running:\n"
          "  DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD\n"
          "  DFS_LOGIN        / DFS_PASSWORD\n"
          "  DFS_USER         / DFS_PASS\n", file=sys.stderr)
    sys.exit(2)


def auth_header(login: str, pw: str) -> dict:
    raw = f"{login}:{pw}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii"),
            "Content-Type":  "application/json"}


def collect_domains(max_n: int | None = None) -> list[str]:
    """All unique domains across the four per-variant + regression_dataset files."""
    files = [
        MAIN / "regression_dataset.parquet",
        MAIN / "full_experiment_data_biased.parquet",
        MAIN / "full_experiment_data_neutral.parquet",
        MAIN / "full_experiment_data_biased_rag.parquet",
        MAIN / "full_experiment_data_neutral_rag.parquet",
    ]
    doms = set()
    for f in files:
        if f.exists():
            d = pd.read_parquet(f, columns=["domain"])["domain"].dropna().unique()
            doms.update(d.tolist())
    doms = sorted(d for d in doms if isinstance(d, str) and "." in d)
    if max_n is not None:
        doms = doms[:max_n]
    return doms


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def fetch_whois_chunks(domains, headers, dry_run=False,
                       max_retries=4, base_sleep=0.5):
    """POST domains in chunks of BULK_LIMIT via the Whois Overview endpoint,
    with per-chunk JSON checkpoints so rate-limit / network failure mid-run
    can be resumed without re-billing completed chunks. Returns
    {domain: {organic_count, organic_pos_1, organic_etv, paid_count,
              created_datetime}}.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINT_DIR / "whois"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(domains) + BULK_LIMIT - 1) // BULK_LIMIT
    out = {}
    total_cost = 0.0

    for ci, chunk in enumerate(chunks(domains, BULK_LIMIT), 1):
        ckpt_path = ckpt_dir / f"chunk_{ci:04d}.json"

        # Resume: load existing checkpoint
        if ckpt_path.exists():
            try:
                with ckpt_path.open() as f:
                    cached = json.load(f)
                out.update(cached)
                print(f"  chunk {ci}/{n_chunks}: loaded from checkpoint "
                      f"({len(cached)} items)", flush=True)
                continue
            except Exception:
                pass

        if dry_run:
            print(f"  [DRY] whois chunk {ci}/{n_chunks}  "
                  f"({len(chunk)} domains)", flush=True)
            for d in chunk:
                out[d] = None
            continue

        # The whois/overview endpoint uses filters + limit (not targets)
        body = [{
            "limit": min(BULK_LIMIT, len(chunk)),
            "filters": [["domain", "in", chunk]],
        }]

        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(WHOIS_URL, headers=headers, json=body,
                                  timeout=180)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    wait = base_sleep * (2 ** attempt) + 2.0
                    print(f"    chunk {ci}: {last_err}, retry in {wait:.1f}s "
                          f"(attempt {attempt + 1}/{max_retries})",
                          file=sys.stderr, flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                j = r.json()
                break
            except Exception as ex:
                last_err = str(ex)
                wait = base_sleep * (2 ** attempt) + 2.0
                print(f"    chunk {ci}: {last_err}, retry in {wait:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        else:
            print(f"  chunk {ci}: GIVING UP after {max_retries} attempts "
                  f"({last_err}). Resume by re-running the script.",
                  file=sys.stderr, flush=True)
            continue

        chunk_result = {}
        tasks = j.get("tasks", [])
        if not tasks:
            print(f"  chunk {ci}: empty tasks array", file=sys.stderr, flush=True)
            continue
        t0 = tasks[0]
        if t0.get("status_code") != 20000:
            print(f"  chunk {ci}: task error {t0.get('status_code')}: "
                  f"{t0.get('status_message')}", file=sys.stderr, flush=True)
            # save empty checkpoint so we don't retry it
            with ckpt_path.open("w") as f:
                json.dump({}, f)
            continue
        result = (t0.get("result") or [{}])[0] or {}
        items = result.get("items", []) or []
        for item in items:
            d = item.get("domain")
            if not d:
                continue
            org = (item.get("metrics") or {}).get("organic") or {}
            paid = (item.get("metrics") or {}).get("paid") or {}
            chunk_result[d] = {
                "organic_count":     org.get("count"),
                "organic_pos_1":     org.get("pos_1"),
                "organic_etv":       org.get("etv"),
                "paid_count":        paid.get("count"),
                "created_datetime":  item.get("created_datetime"),
            }

        with ckpt_path.open("w") as f:
            json.dump(chunk_result, f)
        out.update(chunk_result)
        cost = t0.get("cost") or 0
        total_cost += cost
        print(f"  chunk {ci}/{n_chunks}: {len(items)} items returned "
              f"(cost ${cost:.4f}, cum ${total_cost:.2f})  [saved]",
              flush=True)
        time.sleep(base_sleep)
    return out, total_cost


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and exit; no API calls.")
    ap.add_argument("--max", type=int, default=None,
                    help="Limit to first N domains (for testing).")
    args = ap.parse_args()

    DFS_DIR.mkdir(parents=True, exist_ok=True)

    domains = collect_domains(max_n=args.max)
    n_chunks = (len(domains) + BULK_LIMIT - 1) // BULK_LIMIT
    print(f"[plan] {len(domains):,} unique domains across the dataset")
    print(f"[plan] {n_chunks} chunks via /domain_analytics/whois/overview/live")
    print(f"[plan] expected cost: ~${len(domains) * 0.002:.2f}")

    if args.dry_run:
        return 0

    login, pw = get_credentials()
    headers = auth_header(login, pw)

    print(f"\n[fetch] starting Whois Overview pull (saves per-chunk checkpoints to "
          f"{CHECKPOINT_DIR.relative_to(Path.home())}/whois/)\n")
    results, total_cost = fetch_whois_chunks(domains, headers, dry_run=False)
    print(f"\n[fetch] complete. total cost = ${total_cost:.2f}")

    # Build dataframe
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    rows = []
    for d in domains:
        r = results.get(d) or {}
        created = r.get("created_datetime")
        age_yrs = None
        if created:
            try:
                # Format like "2001-01-12 22:12:14 +00:00"
                dt = datetime.strptime(created[:19], "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                age_yrs = (now - dt).total_seconds() / (365.25 * 86400)
            except Exception:
                age_yrs = None
        oc = r.get("organic_count") or 0
        rows.append({
            "domain":                d,
            "dfs_organic_count":     r.get("organic_count"),
            "dfs_organic_pos_1":     r.get("organic_pos_1"),
            "dfs_organic_etv":       r.get("organic_etv"),
            "dfs_paid_count":        r.get("paid_count"),
            "dfs_created_datetime":  created,
            "dfs_domain_age_years":  round(age_yrs, 2) if age_yrs else None,
            "dfs_authority_log":     (math_log10(oc + 1) if oc is not None else None),
        })
    df = pd.DataFrame(rows)

    # Brand proxy: domain has a substantial Google footprint (replaces conf_brand_recog)
    df["dfs_brand_proxy"] = (
        (df["dfs_organic_count"].fillna(0) >= 100_000)
        | (df["dfs_organic_pos_1"].fillna(0) >= 500)
    ).astype(int)

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[done] saved → {OUT_PARQUET}")
    print(f"\nCoverage summary:")
    for c in ["dfs_organic_count", "dfs_organic_pos_1", "dfs_organic_etv",
              "dfs_paid_count", "dfs_created_datetime",
              "dfs_domain_age_years", "dfs_authority_log"]:
        cov = df[c].notna().mean() * 100
        med = df[c].median() if df[c].dtype != "O" and df[c].notna().any() else "n/a"
        print(f"  {c:25s} {cov:5.1f}%   (median: {med})")
    print(f"  dfs_brand_proxy (= 1):    {df['dfs_brand_proxy'].mean()*100:5.1f}%   "
          f"of domains")

    print("\nNext: python scripts/merge_dfs_domain_authority.py")


def math_log10(x):
    import math
    return math.log10(x) if x is not None and x > 0 else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
