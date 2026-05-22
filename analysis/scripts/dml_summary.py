#!/usr/bin/env python3
"""Paper-ready DML summary — pivots Stage D long parquets into the headline tables.

Reads: $GEODML_DATA_ROOT/data/dml_results/dml_results_long_{variant}.parquet
Writes:
  docs/dml_summary_long.csv    — all rows from all variants, concatenated
  docs/dml_summary_wide.csv    — one row per (treatment, method, learner), variants as columns
  docs/dml_headline.md         — paper-ready markdown table of the canonical estimates

Also prints the headline table to stdout so you can paste it back.

Usage:
  .venv/bin/python scripts/dml_summary.py
  .venv/bin/python scripts/dml_summary.py --estimator partialling_out --learner xgboost
  .venv/bin/python scripts/dml_summary.py --treatments T7_source_earned T2a_question_headings
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path(os.environ.get("GEODML_DATA_ROOT", "/e/scratch/scifi/fourel1"))
OUT_DIR = REPO_ROOT / "docs"

ACTIVE_VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]


def load_all(variants):
    import pyarrow.parquet as pq
    import pandas as pd
    frames = []
    for v in variants:
        p = DATA_ROOT / "data" / "dml_results" / f"dml_results_long_{v}.parquet"
        if not p.exists():
            print(f"WARN: missing {p}", file=sys.stderr)
            continue
        df = pq.read_table(p).to_pandas()
        if "variant" not in df.columns:
            df["variant"] = v
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no DML parquets found under data/dml_results")
    return pd.concat(frames, ignore_index=True)


def pick_canonical_rows(df, est_choice=None, lrn_choice=None):
    """Reduce 280 rows/variant to one row per (variant, treatment) using either:
    - explicit --estimator/--learner filters, or
    - the (estimator, learner) pair that appears for ALL treatments (most complete)."""
    method_col = next((c for c in ("method", "estimator") if c in df.columns), None)
    learner_col = next((c for c in ("learner", "ml_l", "ml_g") if c in df.columns), None)

    if est_choice and method_col:
        df = df[df[method_col] == est_choice]
    if lrn_choice and learner_col:
        df = df[df[learner_col] == lrn_choice]

    # If still multiple rows per (variant, treatment), pick the most-complete
    # (method, learner) combo across treatments
    if method_col and learner_col:
        # coverage = # treatments this combo appears in
        cov = (df.groupby([method_col, learner_col])["treatment"]
                 .nunique().sort_values(ascending=False))
        top = cov.index[0]
        df = df[(df[method_col] == top[0]) & (df[learner_col] == top[1])]
        print(f"  using canonical pair: {method_col}={top[0]}, {learner_col}={top[1]} "
              f"(covers {cov.iloc[0]} treatments)", file=sys.stderr)
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=ACTIVE_VARIANTS)
    ap.add_argument("--treatments", nargs="*", default=None,
                    help="restrict to specific treatments (e.g. T7_source_earned)")
    ap.add_argument("--estimator", default=None,
                    help="filter to one estimator (method/dml variant)")
    ap.add_argument("--learner", default=None,
                    help="filter to one ML learner (xgboost, ridge, ...)")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required", file=sys.stderr)
        return 1

    df = load_all(args.variants)
    print(f"loaded {len(df)} rows across {df['variant'].nunique()} variants", file=sys.stderr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. long: every estimate, concatenated ─────────────────────────────────
    long_path = OUT_DIR / "dml_summary_long.csv"
    df.to_csv(long_path, index=False)
    print(f"wrote {long_path}  ({len(df)} rows)", file=sys.stderr)

    if args.treatments:
        df = df[df["treatment"].isin(args.treatments)]

    # ── 2. canonical headline: one estimator/learner ──────────────────────────
    head = pick_canonical_rows(df.copy(), args.estimator, args.learner)
    keep = [c for c in ("variant", "treatment", "coef", "se", "ci_lower",
                        "ci_upper", "p_val", "sig_stars", "n_obs")
            if c in head.columns]
    head = head[keep].sort_values(["treatment", "variant"]).round(4)

    print("\n## HEADLINE — canonical DML estimates (one estimator/learner per row)\n")
    print(head.to_string(index=False))

    head_md = OUT_DIR / "dml_headline.md"
    with head_md.open("w") as f:
        f.write("# GEODML headline DML estimates\n\n")
        f.write("One row per (variant × treatment) under the canonical DML "
                "(estimator, learner) pair.\n\n")
        f.write(head.to_markdown(index=False) if hasattr(head, "to_markdown")
                else head.to_string(index=False))
        f.write("\n")
    print(f"\nwrote {head_md}", file=sys.stderr)

    # ── 3. wide: treatment × variant pivot of coef ────────────────────────────
    if {"variant", "treatment", "coef"}.issubset(head.columns):
        wide_coef = head.pivot(index="treatment", columns="variant", values="coef")
        wide_se = head.pivot(index="treatment", columns="variant", values="se") if "se" in head.columns else None
        print("\n## WIDE — point estimate (coef) by treatment × variant\n")
        print(wide_coef.round(4).to_string())
        if wide_se is not None:
            print("\n## WIDE — standard error by treatment × variant\n")
            print(wide_se.round(4).to_string())
        wide_path = OUT_DIR / "dml_summary_wide.csv"
        wide_coef.to_csv(wide_path)
        print(f"\nwrote {wide_path}", file=sys.stderr)

    # ── 4. RAG impact: rag - non_rag deltas ───────────────────────────────────
    pairs = [("biased", "biased_rag"), ("neutral", "neutral_rag")]
    if {"variant", "treatment", "coef"}.issubset(head.columns):
        print("\n## RAG IMPACT — coef(rag) − coef(non-rag)  per treatment\n")
        wide_coef = head.pivot(index="treatment", columns="variant", values="coef")
        delta_rows = []
        for nonrag, rag in pairs:
            if nonrag in wide_coef.columns and rag in wide_coef.columns:
                delta_rows.append((f"{rag} − {nonrag}",
                                  (wide_coef[rag] - wide_coef[nonrag]).round(4)))
        if delta_rows:
            delta_df = pd.DataFrame(dict(delta_rows))
            print(delta_df.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
