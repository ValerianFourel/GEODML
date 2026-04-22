#!/usr/bin/env python3
"""
Build the clean DML-ready dataset in data/geodml_dataset.csv.

Merges:
  - Rank data (pre_rank, post_rank, rank_delta) from the experiment JSON
  - Treatments (T1-T4, code-based + LLM-based) from the features CSV
  - Confounders (X1-X10) from the features CSV
  - Metadata (keyword, domain, url, search engine, LLM model)

Output: data/geodml_dataset.csv — one clean, self-contained file ready for DML.
"""

import json
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = Path("results/dml_dataset_searxng.csv")
JSON_PATH = Path("results/searxng_Llama-3.3-70B-Instruct_2026-02-16_1012.json")
OUT_DIR = Path("data")
OUT_CSV = OUT_DIR / "geodml_dataset.csv"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load raw features CSV ────────────────────────────────────────────
    raw = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(raw)} rows from {CSV_PATH}")

    # ── Load rank_changes from experiment JSON ───────────────────────────
    with open(JSON_PATH) as f:
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
            "pre_rank": rc.get("pre_rank"),     # SERP position before LLM
            "post_rank": rc.get("post_rank"),   # LLM re-ranked position
            "rank_delta": rc.get("rank_delta"), # pre_rank - post_rank (+ = promoted)

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
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
