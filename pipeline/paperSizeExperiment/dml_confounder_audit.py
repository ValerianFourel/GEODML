"""Audit how the confounders performed in the DML study.

Three diagnostics:

  1. Coverage            — how many rows each confounder is non-null on
  2. Outcome importance  — LightGBM gain-importance fitting Y ~ confounders
                           (the g_0(X) nuisance model in DML PLR)
  3. Treatment importance — LightGBM gain-importance fitting D ~ confounders
                           for the top treatments (the m_0(X) nuisance model)

A confounder matters in DML iff it predicts BOTH Y and D. Large importance
on Y only means it helps residualise the outcome but doesn't bias treatment;
large importance on D only means it's irrelevant to outcome. The product
rank-importance is the closest proxy to the confounder's role in the
partialling-out identification.

Run on the POOLED subset (65,203 rows) for the global picture.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import CONFOUNDERS  # noqa: E402

INPUT_CSV = SCRIPT_DIR / "consolidated_results" / "regression_dataset.csv"
OUT_DIR = SCRIPT_DIR / "consolidated_results" / "dml_study"

OUTCOMES = ["rank_delta", "post_rank"]
TOP_TREATMENTS = [
    "treat_source_earned",       # T7, strongest effect
    "has_llms_txt",              # T_llms_txt
    "treat_topical_comp",        # T5
    "treat_freshness",           # T6
    "treat_structured_data",     # T3_new
    "T3_structured_data_code",   # T3_code
    "treat_question_headings",   # T2a
    "T1_statistical_density_code",  # T1_code
]


def _learner():
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, verbose=-1, random_state=42,
    )


def _prep(df: pd.DataFrame, target: str, confounders: list[str]):
    cols = confounders + [target]
    sub = df[cols].dropna(subset=[target])
    if len(sub) < 100:
        return None
    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imp.fit_transform(sub[confounders]),
        columns=confounders, index=sub.index,
    )
    Xs = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=confounders, index=sub.index,
    )
    return Xs, sub[target].values


def fit_importance(df: pd.DataFrame, target: str, confounders: list[str]) -> pd.Series:
    prep = _prep(df, target, confounders)
    if prep is None:
        return pd.Series(dtype=float)
    X, y = prep
    m = _learner()
    m.fit(X, y)
    return pd.Series(
        m.booster_.feature_importance(importance_type="gain"),
        index=confounders,
    )


def main() -> int:
    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["pre_rank"].notna()].copy()
    print(f"  {len(df):,} rows")

    confounders = [c for c in CONFOUNDERS if c in df.columns]
    print(f"  {len(confounders)} confounders in scope: {confounders}")

    print("\n=== 1. Coverage ===")
    cov = {
        c: int(df[c].notna().sum()) / len(df)
        for c in confounders
    }
    cov_s = pd.Series(cov).sort_values(ascending=False)
    for c, v in cov_s.items():
        print(f"  {c:32s}  {v*100:5.1f}%")

    print("\n=== 2. Outcome importance (g_0(X): Y ~ confounders, LGBM gain) ===")
    out_imp = {}
    for y in OUTCOMES:
        imp = fit_importance(df, y, confounders)
        if imp.empty:
            continue
        total = imp.sum()
        norm = imp / total * 100 if total > 0 else imp
        out_imp[y] = norm.sort_values(ascending=False)
        print(f"\n  outcome={y} (importance %):")
        for c, v in out_imp[y].items():
            print(f"    {c:32s}  {v:6.2f}")

    print("\n=== 3. Treatment importance (m_0(X): D ~ confounders, LGBM gain) ===")
    treat_imp = {}
    for t in TOP_TREATMENTS:
        if t not in df.columns:
            continue
        imp = fit_importance(df, t, confounders)
        if imp.empty:
            continue
        total = imp.sum()
        norm = imp / total * 100 if total > 0 else imp
        treat_imp[t] = norm
        print(f"\n  treatment={t}  (top 8 by importance %):")
        for c, v in norm.sort_values(ascending=False).head(8).items():
            print(f"    {c:32s}  {v:6.2f}")

    # Build combined summary DataFrame
    print("\n=== 4. Combined: average confounder rank across (Y, D) models ===")
    frames = list(out_imp.values()) + list(treat_imp.values())
    if frames:
        stacked = pd.concat(
            {f"out_{y}": s for y, s in out_imp.items()}
            | {f"treat_{t}": s for t, s in treat_imp.items()},
            axis=1,
        ).fillna(0.0)
        # mean rank across columns (lower = more important)
        ranks = stacked.rank(axis=0, ascending=False)
        combined = pd.DataFrame({
            "mean_rank": ranks.mean(axis=1),
            "mean_importance_pct": stacked.mean(axis=1),
            "coverage": [cov.get(c, 0.0) for c in stacked.index],
        }).sort_values("mean_rank")
        for c, row in combined.iterrows():
            flag = "★ dfs" if c.startswith("dfs_") else "  "
            print(
                f"  {flag}  {c:32s}  mean_rank={row['mean_rank']:5.1f}  "
                f"mean_imp%={row['mean_importance_pct']:6.2f}  cov={row['coverage']*100:5.1f}%"
            )
        combined.to_csv(OUT_DIR / "confounder_audit.csv")
        print(f"\n  wrote {OUT_DIR/'confounder_audit.csv'}")

    # Specifically: did the dfs_* confounders contribute?
    dfs_cols = [c for c in confounders if c.startswith("dfs_")]
    orig_cols = [c for c in confounders if not c.startswith("dfs_")]
    print("\n=== 5. Original vs DataForSEO contribution share ===")
    for label, series in (list(out_imp.items()) + list(treat_imp.items())):
        if not isinstance(series, pd.Series):
            continue
        orig_sum = series.reindex(orig_cols).sum()
        dfs_sum = series.reindex(dfs_cols).sum()
        total = orig_sum + dfs_sum
        if total == 0:
            continue
        print(
            f"  {label:35s}  orig={orig_sum:5.1f}%  dfs={dfs_sum:5.1f}%  "
            f"dfs_share={100*dfs_sum/total:4.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
