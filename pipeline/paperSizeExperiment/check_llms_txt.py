"""Check each unique domain in consolidated_results/ for /llms.txt.

Produces consolidated_results/domains_llms_txt.csv and adds a `has_llms_txt`
column to every geodml_dataset.csv and to the merged CSVs.
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent / "consolidated_results"
RUNS = sorted((ROOT / "runs").glob("*/geodml_dataset.csv"))
MERGED = [
    ROOT / "merged" / "merged_all_runs.csv",
    ROOT / "merged" / "merged_all_8runs.csv",
]
AUDIT_CSV = ROOT / "domains_llms_txt.csv"

TIMEOUT = (5, 8)  # connect, read
WORKERS = 80
USER_AGENT = "Mozilla/5.0 (compatible; llms-txt-checker/1.0)"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=WORKERS, pool_maxsize=WORKERS)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/plain, */*;q=0.1"})
    return s


def collect_domains() -> list[str]:
    domains: set[str] = set()
    for f in RUNS:
        for chunk in pd.read_csv(f, usecols=["domain"], chunksize=50_000, dtype=str):
            for d in chunk["domain"].dropna().astype(str).str.strip().str.lower():
                if d:
                    domains.add(d)
    return sorted(domains)


def check_domain(session: requests.Session, domain: str) -> tuple[str, int, int, str]:
    url = f"https://{domain}/llms.txt"
    try:
        r = session.get(url, timeout=TIMEOUT, allow_redirects=True, stream=True)
        status = r.status_code
        ctype = r.headers.get("Content-Type", "")
        # Require 200 and non-HTML body (many sites serve a 200 SPA/404 page).
        has = 0
        if status == 200:
            head = next(r.iter_content(chunk_size=512, decode_unicode=False), b"") or b""
            r.close()
            sample = head[:512].lower()
            looks_html = b"<html" in sample or b"<!doctype html" in sample or "html" in ctype.lower()
            has = 0 if looks_html else 1
        else:
            r.close()
        return (domain, has, status, ctype)
    except requests.RequestException as e:
        return (domain, 0, -1, type(e).__name__)
    except Exception as e:  # pragma: no cover
        return (domain, 0, -2, type(e).__name__)


def run_checks(domains: list[str]) -> dict[str, int]:
    session = make_session()
    rows: list[tuple[str, int, int, str]] = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(check_domain, session, d): d for d in domains}
        with AUDIT_CSV.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["domain", "has_llms_txt", "status_code", "note"])
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                w.writerow(row)
                done += 1
                if done % 250 == 0 or done == len(domains):
                    elapsed = time.time() - t0
                    hits = sum(1 for r in rows if r[1] == 1)
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"  {done}/{len(domains)}  hits={hits}  {rate:.1f} req/s  elapsed={elapsed:.0f}s",
                        flush=True,
                    )
    return {r[0]: r[1] for r in rows}


def annotate_csv(path: Path, mapping: dict[str, int]) -> None:
    if not path.exists():
        print(f"  skip missing: {path}")
        return
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "domain" not in df.columns:
        print(f"  skip (no domain col): {path}")
        return
    df["has_llms_txt"] = (
        df["domain"].fillna("").astype(str).str.strip().str.lower().map(mapping).fillna(0).astype(int)
    )
    df.to_csv(path, index=False)
    print(f"  wrote {path}  rows={len(df)}  hits={int(df['has_llms_txt'].sum())}")


def main() -> None:
    print(f"collecting unique domains from {len(RUNS)} runs…", flush=True)
    domains = collect_domains()
    print(f"  {len(domains)} unique domains", flush=True)

    print("checking llms.txt in parallel…", flush=True)
    mapping = run_checks(domains)
    hits = sum(mapping.values())
    print(f"done: {hits}/{len(domains)} domains serve llms.txt", flush=True)

    print("annotating CSVs…", flush=True)
    for f in RUNS:
        annotate_csv(f, mapping)
    for f in MERGED:
        annotate_csv(f, mapping)


if __name__ == "__main__":
    sys.exit(main())
