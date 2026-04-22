#!/usr/bin/env python3
"""Clean data: merge rankings + features into a single DML-ready CSV.

Self-contained script — imports nothing from ../src/.

Reads the outputs of gather_data.py (experiment.json, rankings.csv, features.csv)
and produces a single clean CSV ready for DML analysis.

Usage:
  python pipeline/clean_data.py
  python pipeline/clean_data.py --input-dir output/small_pool/ --output output/small_pool/geodml_dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Clean data: merge rankings + features into DML-ready CSV")
    parser.add_argument("--input-dir", type=str, default="output/",
                        help="Directory containing experiment.json, rankings.csv, features.csv (default: output/)")
    parser.add_argument("--output", type=str, default="output/geodml_dataset.csv",
                        help="Output CSV path (default: output/geodml_dataset.csv)")
    parser.add_argument("--new-features", type=str, default=None,
                        help="Path to new features CSV (default: pipeline/intermediate/features_new.csv)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────
    rankings_path = input_dir / "rankings.csv"
    features_path = input_dir / "features.csv"
    json_path = input_dir / "experiment.json"

    for p in [rankings_path, features_path, json_path]:
        if not p.exists():
            print(f"Missing required file: {p}")
            sys.exit(1)

    rankings = pd.read_csv(rankings_path)
    features = pd.read_csv(features_path)
    print(f"Loaded rankings: {len(rankings)} rows from {rankings_path}")
    print(f"Loaded features: {len(features)} rows from {features_path}")

    with open(json_path) as f:
        experiment = json.load(f)

    # ── Build rank_changes lookup from experiment JSON ────────────────────
    rc_lookup = {}
    for kw_result in experiment.get("per_keyword_results", []):
        query = kw_result["query"]
        for rc in kw_result.get("rank_changes", []):
            rc_lookup[(query, rc["domain"])] = {
                "pre_rank": rc["pre_rank"],
                "post_rank": rc["post_rank"],
                "rank_delta": rc["rank_delta"],
            }

    # Fallback: if experiment.json has no per_keyword_results, read keywords.jsonl
    if not rc_lookup:
        jsonl_path = input_dir / "keywords.jsonl"
        if jsonl_path.exists():
            print(f"  experiment.json has no per_keyword_results, falling back to {jsonl_path}")
            with open(jsonl_path) as jf:
                for line in jf:
                    kw_result = json.loads(line)
                    query = kw_result.get("query", "")
                    for rc in kw_result.get("rank_changes", []):
                        rc_lookup[(query, rc["domain"])] = {
                            "pre_rank": rc["pre_rank"],
                            "post_rank": rc["post_rank"],
                            "rank_delta": rc["rank_delta"],
                        }
            print(f"  Loaded {len(rc_lookup)} rank_changes from keywords.jsonl")

    # ── Merge rankings with features ──────────────────────────────────────
    # Features are keyed by URL; rankings have url+domain
    feat_by_url = {}
    feat_by_domain = {}
    for _, row in features.iterrows():
        url = str(row.get("url", "")).strip()
        domain = str(row.get("domain", "")).strip()
        row_dict = row.to_dict()
        if url:
            feat_by_url[url] = row_dict
        if domain:
            feat_by_domain[domain] = row_dict

    rows = []
    for _, r in rankings.iterrows():
        keyword = r.get("keyword", "")
        domain = str(r.get("domain", "")).strip()
        url = str(r.get("url", "")).strip()

        # Look up rank changes
        rc = rc_lookup.get((keyword, domain), {})

        # Look up features (try URL first, then domain fallback)
        feat = feat_by_url.get(url, {})
        if not feat:
            fallback_url = f"https://{domain}/"
            feat = feat_by_url.get(fallback_url, {})
        if not feat:
            feat = feat_by_domain.get(domain, {})

        rows.append({
            # Identifiers / metadata
            "keyword": keyword,
            "domain": domain,
            "url": url,
            "search_engine": r.get("engine", ""),
            "llm_model": r.get("model", ""),

            # Outcome (Y)
            "pre_rank": rc.get("pre_rank"),
            "post_rank": rc.get("post_rank"),
            "rank_delta": rc.get("rank_delta"),

            # Treatments: code-based (T1-T4)
            "T1_statistical_density_code": feat.get("T1_statistical_density"),
            "T2_question_heading_code": feat.get("T2_question_heading_match"),
            "T3_structured_data_code": feat.get("T3_structured_data"),
            "T4_citation_authority_code": feat.get("T4_citation_authority"),

            # Treatments: LLM-based (T1-T4)
            "T1_statistical_density_llm": feat.get("T1_llm_statistical_density"),
            "T2_question_heading_llm": feat.get("T2_llm_question_heading"),
            "T3_structured_data_llm": feat.get("T3_llm_structured_data"),
            "T4_citation_authority_llm": feat.get("T4_llm_citation_authority"),

            # Confounders (X1-X10)
            "X1_domain_authority": feat.get("X1_domain_authority"),
            "X1_global_rank": feat.get("X1_global_rank"),
            "X2_domain_age_years": feat.get("X2_domain_age_years"),
            "X3_word_count": feat.get("X3_word_count"),
            "X4_lcp_ms": feat.get("X4_lcp_ms"),
            "X6_readability": feat.get("X6_readability"),
            "X7_internal_links": feat.get("X7_internal_links"),
            "X7B_outbound_links": feat.get("X7B_outbound_links"),
            "X8_keyword_difficulty": feat.get("X8_keyword_difficulty"),
            "X9_images_with_alt": feat.get("X9_images_with_alt"),
            "X10_https": feat.get("X10_https"),
        })

    df = pd.DataFrame(rows)

    # ── Merge new features from extract_features.py ──────────────────────
    if args.new_features:
        new_features_path = Path(args.new_features)
    else:
        new_features_path = Path(__file__).resolve().parent / "intermediate" / "features_new.csv"
    if new_features_path.exists():
        new_feat = pd.read_csv(new_features_path)
        print(f"Loaded new features: {len(new_feat)} rows from {new_features_path}")

        # Merge on (keyword, url)
        new_feat_cols = [c for c in new_feat.columns if c not in ("keyword", "domain", "url")]
        merge_cols = ["keyword", "url"] + new_feat_cols
        df = df.merge(new_feat[merge_cols], on=["keyword", "url"], how="left")
        print(f"  Merged {len(new_feat_cols)} new columns: {new_feat_cols[:5]}...")
    else:
        print(f"No new features file found at {new_features_path} — skipping merge")

    # ── Summary stats ─────────────────────────────────────────────────────
    n_total = len(df)
    if n_total == 0:
        print("\nNo rows produced. Check input files.")
        sys.exit(1)

    n_with_delta = df["rank_delta"].notna().sum()
    n_missing_delta = df["rank_delta"].isna().sum()

    print(f"\nClean dataset: {n_total} rows")
    print(f"  With rank_delta:    {n_with_delta} ({n_with_delta/n_total*100:.1f}%)")
    print(f"  Missing rank_delta: {n_missing_delta} ({n_missing_delta/n_total*100:.1f}%)")
    print(f"  Keywords:           {df['keyword'].nunique()}")
    print(f"  Unique domains:     {df['domain'].nunique()}")
    if not df["search_engine"].empty:
        print(f"  Search engine:      {df['search_engine'].iloc[0]}")
    if not df["llm_model"].empty:
        print(f"  LLM model:          {df['llm_model'].iloc[0]}")

    # Coverage report
    print("\nColumn coverage:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / n_total * 100
        if pct < 100:
            print(f"  {col:<35} {non_null:>3}/{n_total} ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
