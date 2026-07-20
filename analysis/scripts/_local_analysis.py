#!/usr/bin/env python3
"""Local Mac re-analysis: examine treatment-vs-confounder structure + existing
multi-treatment DML, then run new specifications.

Reads from ~/geodml_data/ (snapshotted from HF).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 100)

ROOT = Path.home() / "geodml_data"


def section(t):
    print("\n" + "=" * 86)
    print(t)
    print("=" * 86)


def load_main(variant):
    return pq.read_table(ROOT / "data" / "main" / f"full_experiment_data_{variant}.parquet").to_pandas()


def main():
    # ── 1. Existing multi-treatment DML: per-study breakdown ──────────────────
    section("Existing multi-treatment DML — per-study breakdown")
    df = pq.read_table(ROOT / "data" / "dml_results" / "dml_multi_treatment.parquet").to_pandas()
    print(f"studies: {sorted(df['study'].unique())}")
    print(f"outcomes: {sorted(df['outcome'].unique())}")
    print(f"treatments: {sorted(df['treatment'].unique())}")
    for s in sorted(df["study"].unique()):
        print(f"\n--- {s} ---")
        cols = ["outcome", "treatment", "n", "coef", "se", "p_val", "p_val_bonferroni",
                "n_other_treats_in_X", "n_confounders_in_X"]
        cols = [c for c in cols if c in df.columns]
        sub = df[df["study"] == s][cols].round(4)
        print(sub.to_string(index=False))

    # ── 2. Treatment vs confounder reconsideration ────────────────────────────
    section("Treatment vs confounder — column inventory")
    biased = load_main("biased")
    treat_cols = [c for c in biased.columns if c.startswith("treat_")]
    conf_cols = [c for c in biased.columns if c.startswith("conf_")]
    print(f"# treat_*: {len(treat_cols)}")
    for c in treat_cols:
        dt = biased[c].dtype
        nu = biased[c].nunique()
        ni = biased[c].notna().sum()
        sample = biased[c].dropna().head(3).tolist()
        print(f"  {c:32s}  dtype={str(dt):10s}  n_unique={nu:>6}  n_complete={ni:>6}  sample={sample}")
    print(f"\n# conf_*: {len(conf_cols)}")
    for c in conf_cols:
        dt = biased[c].dtype
        nu = biased[c].nunique()
        ni = biased[c].notna().sum()
        print(f"  {c:28s}  dtype={str(dt):10s}  n_unique={nu:>6}  n_complete={ni:>6}")

    # ── 3. Domain-level vs content-level analysis ─────────────────────────────
    section("Domain identity — treat_source_earned + treat_source_brand + treat_source_type")
    # cross-tab of source_earned by source_type
    df = biased.dropna(subset=["treat_source_earned"])
    print(f"\ntreat_source_earned distribution (all rows, biased variant):")
    print(df["treat_source_earned"].value_counts().to_string())
    print(f"\ntreat_source_type distribution:")
    print(df["treat_source_type"].value_counts().to_string() if "treat_source_type" in df.columns else "—")
    print(f"\ntreat_source_brand distribution:")
    print(df["treat_source_brand"].value_counts().to_string())
    print(f"\nCross-tab treat_source_earned × treat_source_type:")
    if "treat_source_type" in df.columns:
        print(pd.crosstab(df["treat_source_earned"], df["treat_source_type"]))
    print(f"\nIs treat_source_earned domain-deterministic?")
    print("  i.e. does each domain map to the same earned-flag in every row?")
    per_domain = df.groupby("domain")["treat_source_earned"].nunique()
    print(f"  unique earned-flag values per domain: max={per_domain.max()}, "
          f"# domains with >1 distinct value = {(per_domain > 1).sum()}")
    print(f"  total unique domains: {df['domain'].nunique()}, "
          f"# flagged as earned (any row): {df.groupby('domain')['treat_source_earned'].max().sum():.0f}")

    # ── 4. Correlations: how much do treatments overlap with confounders? ─────
    section("Treatment ↔ Confounder correlations (Pearson, biased variant)")
    # restrict to rows with full data
    needed = [c for c in treat_cols + conf_cols if biased[c].dtype != "O"]
    sub = biased.dropna(subset=needed)
    print(f"complete-case n = {len(sub)}")
    treat_num = [c for c in treat_cols if biased[c].dtype != "O"]
    corr = sub[treat_num + conf_cols].corr().loc[treat_num, conf_cols]
    # show abs max correlation per treatment
    print("\nFor each treatment, the |corr| with each confounder (top 3 per treatment):")
    for t in treat_num:
        s = corr.loc[t].abs().sort_values(ascending=False).head(3)
        print(f"  {t:30s}  top: " + " ".join(f"{c}={corr.loc[t,c]:+.3f}" for c in s.index))

    # ── 5. R² of confounders predicting each treatment ────────────────────────
    section("R² of confounders predicting each treatment (LightGBM, simple holdout)")
    try:
        from sklearn.model_selection import train_test_split
        from lightgbm import LGBMRegressor, LGBMClassifier
        from sklearn.metrics import r2_score, roc_auc_score
        Xall = sub[conf_cols].fillna(sub[conf_cols].median()).values
        for t in treat_num + ["treat_source_earned", "treat_source_brand"]:
            if t not in sub.columns:
                continue
            y = sub[t].values
            if pd.api.types.is_bool_dtype(y) or set(np.unique(y[~pd.isna(y)])) <= {0.0, 1.0}:
                # binary
                mask = ~pd.isna(y)
                yb = y[mask].astype(int)
                Xt = Xall[mask]
                if len(np.unique(yb)) < 2:
                    print(f"  {t:30s}  SKIP (only one class)")
                    continue
                Xtr, Xte, ytr, yte = train_test_split(Xt, yb, random_state=42, test_size=0.3, stratify=yb)
                m = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                   verbose=-1, n_jobs=-1)
                m.fit(Xtr, ytr)
                p = m.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, p)
                base = yb.mean()
                print(f"  {t:30s}  AUC={auc:.3f}  base_rate={base:.3f}  (binary)")
            else:
                mask = ~pd.isna(y)
                Xt = Xall[mask]; yt = y[mask]
                Xtr, Xte, ytr, yte = train_test_split(Xt, yt, random_state=42, test_size=0.3)
                m = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                  verbose=-1, n_jobs=-1)
                m.fit(Xtr, ytr)
                yp = m.predict(Xte)
                r2 = r2_score(yte, yp)
                print(f"  {t:30s}  R²={r2:+.3f}  std(y)={yt.std():.3f}")
    except Exception as e:
        print(f"  (skipped — {e})")


if __name__ == "__main__":
    main()
