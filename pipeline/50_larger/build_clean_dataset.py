#!/usr/bin/env python3
"""
Build the clean DML-ready dataset in 50_larger/data/geodml_dataset.csv.

Merges:
  - Rank data (pre_rank, post_rank, rank_delta) from the experiment JSON
  - Treatments (T1-T4, code-based + LLM-based) from the features CSV
  - Confounders (X1-X10) from the features CSV
  - Metadata (keyword, domain, url, search engine, LLM model)

Output: 50_larger/data/geodml_dataset.csv — one clean, self-contained file ready for DML.

Usage:
  python 50_larger/build_clean_dataset.py
  python 50_larger/build_clean_dataset.py --csv 50_larger/results/dml_dataset_searxng.csv --json 50_larger/results/searxng_Llama-3.3-70B-Instruct_YYYY-MM-DD_HHMM.json
"""

import argparse
import json
import glob
import sys
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUT_DIR = SCRIPT_DIR / "data"


def find_latest_json(results_dir: Path) -> Path | None:
    """Find the latest searxng experiment JSON in results dir."""
    # Try duckduckgo first, then searxng
    candidates = sorted(results_dir.glob("duckduckgo_*_20*.json"), reverse=True)
    if not candidates:
        candidates = sorted(results_dir.glob("searxng_*_20*.json"), reverse=True)
    return candidates[0] if candidates else None


def find_latest_csv(results_dir: Path) -> Path | None:
    """Find the latest dml_dataset CSV in results dir."""
    candidates = sorted(results_dir.glob("dml_dataset_*.csv"), reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Build clean DML dataset (50_larger)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to dml_dataset CSV (default: auto-detect)")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to experiment JSON (default: auto-detect)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Find input files
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = find_latest_csv(RESULTS_DIR)
        if not csv_path:
            print(f"No dml_dataset_*.csv found in {RESULTS_DIR}")
            print("Run run_page_scraper.py first, or specify --csv")
            return

    if args.json:
        json_path = Path(args.json)
    else:
        json_path = find_latest_json(RESULTS_DIR)
        if not json_path:
            print(f"No searxng_*.json found in {RESULTS_DIR}")
            print("Run run_ai_search.py first, or specify --json")
            return

    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    # ── Load raw features CSV ────────────────────────────────────────────
    raw = pd.read_csv(csv_path)
    print(f"Loaded {len(raw)} rows from {csv_path.name}")

    # ── Load rank_changes from experiment JSON ───────────────────────────
    with open(json_path) as f:
        experiment = json.load(f)

    rc_lookup = {}
    for kw_result in experiment["per_keyword_results"]:
        query = kw_result["query"]
        for rc in kw_result["rank_changes"]:
            rc_lookup[(query, rc["domain"])] = {
                "pre_rank": rc["pre_rank"],
                "post_rank": rc["post_rank"],
                "rank_delta": rc["rank_delta"],
            }

    # ── Build clean dataframe ────────────────────────────────────────────
    rows = []
    for _, r in raw.iterrows():
        key = (r["keyword"], r["domain"])
        rc = rc_lookup.get(key, {})

        rows.append({
            # ── Identifiers / metadata ──
            "keyword": r["keyword"],
            "domain": r["domain"],
            "url": r["url"],
            "search_engine": r["engine"],
            "llm_model": r["model"],

            # ── Outcome (Y) ──
            "pre_rank": rc.get("pre_rank"),
            "post_rank": rc.get("post_rank"),
            "rank_delta": rc.get("rank_delta"),

            # ── Treatments: code-based (T1-T4) ──
            "T1_statistical_density_code": r.get("T1_statistical_density"),
            "T2_question_heading_code": r.get("T2_question_heading_match"),
            "T3_structured_data_code": r.get("T3_structured_data"),
            "T4_citation_authority_code": r.get("T4_citation_authority"),

            # ── Treatments: LLM-based (T1-T4) ──
            "T1_statistical_density_llm": r.get("T1_llm_statistical_density"),
            "T2_question_heading_llm": r.get("T2_llm_question_heading"),
            "T3_structured_data_llm": r.get("T3_llm_structured_data"),
            "T4_citation_authority_llm": r.get("T4_llm_citation_authority"),

            # ── Confounders (X1-X10) ──
            "X1_domain_authority": r.get("X1_domain_authority"),
            "X1_global_rank": r.get("X1_global_rank"),
            "X2_domain_age_years": r.get("X2_domain_age_years"),
            "X3_word_count": r.get("X3_word_count"),
            "X4_lcp_ms": r.get("X4_lcp_ms"),
            "X6_readability": r.get("X6_readability"),
            "X7_internal_links": r.get("X7_internal_links"),
            "X7B_outbound_links": r.get("X7B_outbound_links"),
            "X8_keyword_difficulty": r.get("X8_keyword_difficulty"),
            "X9_images_with_alt": r.get("X9_images_with_alt"),
            "X10_https": r.get("X10_https"),
        })

    df = pd.DataFrame(rows)

    # ── Summary stats ────────────────────────────────────────────────────
    n_total = len(df)
    n_with_delta = df["rank_delta"].notna().sum()
    n_missing_delta = df["rank_delta"].isna().sum()

    print(f"\nClean dataset: {n_total} rows")
    print(f"  With rank_delta:    {n_with_delta} ({n_with_delta/n_total*100:.1f}%)")
    print(f"  Missing rank_delta: {n_missing_delta} ({n_missing_delta/n_total*100:.1f}%)")
    print(f"  Keywords:           {df['keyword'].nunique()}")
    print(f"  Unique domains:     {df['domain'].nunique()}")
    if len(df) > 0:
        print(f"  Search engine:      {df['search_engine'].unique()[0]}")
        print(f"  LLM model:          {df['llm_model'].unique()[0]}")

    # Coverage report
    print("\nColumn coverage:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / n_total * 100
        if pct < 100:
            print(f"  {col:<35} {non_null:>3}/{n_total} ({pct:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────
    out_csv = OUT_DIR / "geodml_dataset.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
