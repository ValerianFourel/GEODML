#!/usr/bin/env python3
"""Bridge the data-coverage gaps in the published ValerianFourel/geodml-emnlp-2026 mirror.

Two issues addressed:

1. `data/main/full_experiment_unified.parquet` is missing ~62 k RAG-output rows.
   Only 2,938 of the ~64,909 actual RAG rows made it in (a publish-pipeline bug).
   This script rebuilds unified correctly as the union of the 4 per-variant
   `full_experiment_data_{variant}.parquet` files, normalizing column names so the
   union schema is consistent.

2. RAG-coverage is partial: 267 keywords have no biased_rag output, 396 keywords
   have no neutral_rag output (RAG retrieval failed). Downstream selection-style
   analyses confuse these with LLM rejections. This script writes
   `data/coverage/rag_coverage.parquet` mapping (keyword, variant) → has_rag_output.

Outputs:
  ~/geodml_data/data/main/full_experiment_unified_FIXED.parquet
  ~/geodml_data/data/coverage/rag_coverage.parquet
  ~/geodml_data/data/coverage/missing_rag_keywords.parquet
  docs/dataset_gap_bridge_2026-05-23.md     short note explaining the fix
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.home() / "geodml_data"
MAIN = ROOT / "data" / "main"
COV = ROOT / "data" / "coverage"
DOCS = Path(__file__).resolve().parent.parent / "docs"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
VARIANT_AXIS = {"biased": "biased",
                "neutral": "neutral",
                "biased_rag": "biased_passage",
                "neutral_rag": "neutral_passage"}


def main():
    COV.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    # ── load 4 per-variant files ────────────────────────────────────────────
    print("Loading per-variant LLM-output parquets …")
    parts = {}
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        df = pq.read_table(p).to_pandas()
        # normalize engine name
        df["engine_norm"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        # normalize pool col name
        if "serp_pool_size" in df.columns:
            df = df.rename(columns={"serp_pool_size": "pool"})
        # add axis_* columns matching the original unified schema
        df["axis_engine"] = df["engine_norm"]
        df["axis_model"] = df["llm_model"].str.split("/").str[-1] if df["llm_model"].dtype == object else df["llm_model"]
        df["axis_pool"] = df["pool"]
        df["axis_top_n"] = 10
        df["axis_prompt"] = "biased" if "biased" in v else "neutral"
        df["axis_passage_mode"] = "passage" if v.endswith("_rag") else "snippet"
        df["axis_variant"] = VARIANT_AXIS[v]
        parts[v] = df
        print(f"  {v:12s}: rows={len(df):>6}  kw={df['keyword'].nunique():>4}")

    # ── 1. Rebuild unified ──────────────────────────────────────────────────
    print("\nBuilding fixed unified table …")
    cols_common = sorted(set.intersection(*(set(d.columns) for d in parts.values())))
    print(f"  {len(cols_common)} columns common to all 4 variants")
    union = pd.concat([d[cols_common] for d in parts.values()], ignore_index=True)
    out_unified = MAIN / "full_experiment_unified_FIXED.parquet"
    union.to_parquet(out_unified)
    print(f"  wrote {out_unified}  rows={len(union):,}")
    # original schema diff
    orig = pq.read_table(MAIN / "full_experiment_unified.parquet").to_pandas()
    print(f"  original unified: rows={len(orig):,}  (missing {len(union)-len(orig):,} from RAG arms)")
    print(f"  axis_variant breakdown (fixed):")
    print(union["axis_variant"].value_counts().to_string())

    # ── 2. RAG-coverage table ───────────────────────────────────────────────
    print("\nBuilding RAG-coverage table …")
    all_kw = sorted(set.union(*(set(d["keyword"]) for d in parts.values())))
    print(f"  union of keywords across all variants: {len(all_kw)}")
    cov_rows = []
    for kw in all_kw:
        rec = {"keyword": kw}
        for v in VARIANTS:
            rec[f"has_output_{v}"] = int(kw in set(parts[v]["keyword"]))
            rec[f"n_rows_{v}"] = int((parts[v]["keyword"] == kw).sum())
        cov_rows.append(rec)
    cov = pd.DataFrame(cov_rows)

    cov["rag_coverage"] = (cov["has_output_biased_rag"] +
                           cov["has_output_neutral_rag"]).map(
                               {0: "no_rag", 1: "partial_rag", 2: "full_rag"})
    out_cov = COV / "rag_coverage.parquet"
    cov.to_parquet(out_cov)
    print(f"  wrote {out_cov}  rows={len(cov)}")
    print("  rag_coverage breakdown:")
    print(cov["rag_coverage"].value_counts().to_string())

    # ── 3. Missing-keyword inventory ────────────────────────────────────────
    missing = cov[(cov["has_output_biased_rag"] == 0) |
                  (cov["has_output_neutral_rag"] == 0)].copy()
    out_missing = COV / "missing_rag_keywords.parquet"
    missing.to_parquet(out_missing)
    print(f"\n  {len(missing)} keywords are missing under at least one RAG variant.")
    print(f"  wrote {out_missing}")

    # ── 4. Write the explainer markdown ─────────────────────────────────────
    note = f"""# Dataset gap bridge — `ValerianFourel/geodml-emnlp-2026`

*Generated by `scripts/bridge_dataset_gaps.py` on 2026-05-23.*

## Gap 1 — `full_experiment_unified.parquet` is missing ~62 k RAG rows

The published unified table contains only **2,938 of the ~64,909 actual RAG-output rows** in
the dataset (`axis_passage_mode=passage`). This is a publish-pipeline bug — the per-variant
files (`full_experiment_data_biased_rag.parquet` and `full_experiment_data_neutral_rag.parquet`)
have the complete data.

**Fix**: this script rebuilds `full_experiment_unified_FIXED.parquet` as the row-union of the
4 per-variant files, with consistent `axis_*` columns. Use the FIXED file for any analysis
that pools across variants (selection-style outcomes especially).

Row counts:
- Original unified: {len(orig):,} rows
- Fixed unified: {len(union):,} rows (+{len(union)-len(orig):,})

Per-variant breakdown of the fixed file:
```
{union["axis_variant"].value_counts().to_string()}
```

## Gap 2 — RAG retrieval failed for some keywords

The RAG pipeline only produces LLM-output for keywords where retrieval returned at least one
passage. Coverage is partial:

| Variant | Keywords with output | Keywords missing | Missing % |
|---|---|---|---|
| biased | {(cov['has_output_biased']==1).sum()} | {(cov['has_output_biased']==0).sum()} | {100*(cov['has_output_biased']==0).mean():.1f}% |
| neutral | {(cov['has_output_neutral']==1).sum()} | {(cov['has_output_neutral']==0).sum()} | {100*(cov['has_output_neutral']==0).mean():.1f}% |
| biased_rag | {(cov['has_output_biased_rag']==1).sum()} | {(cov['has_output_biased_rag']==0).sum()} | {100*(cov['has_output_biased_rag']==0).mean():.1f}% |
| neutral_rag | {(cov['has_output_neutral_rag']==1).sum()} | {(cov['has_output_neutral_rag']==0).sum()} | {100*(cov['has_output_neutral_rag']==0).mean():.1f}% |

For any "did the LLM admit this candidate?" style analysis on the RAG variants, the pool
must be restricted to keywords with `has_output_{{variant}}=1`. Otherwise unselected pool rows
from RAG-failed keywords get misread as LLM rejections.

**Fix**: this script writes `data/coverage/rag_coverage.parquet` with per-keyword indicators
and row counts so downstream code can filter cleanly.

## What this means for the paper

The published HF dataset is correct *at the per-variant-file level* — every actual LLM-output
row is present in `full_experiment_data_{{variant}}.parquet`. The bug is only in the
convenience union table. Replace `full_experiment_unified.parquet` with
`full_experiment_unified_FIXED.parquet` in any pooled analysis.

The RAG-coverage gap is a real selection-on-success issue that needs a Limitations paragraph,
not a code fix. It means RAG-arm conclusions generalize only to the keywords where retrieval
worked (~70% of the keyword set).
"""
    out_md = DOCS / "dataset_gap_bridge_2026-05-23.md"
    out_md.write_text(note)
    print(f"\nWrote explainer → {out_md}")
    print("\nDone.")


if __name__ == "__main__":
    main()
