"""DML on the 'robust winners' subset.

A robust winner is a (keyword, url) pair that the LLM picked into the top-10
under BOTH the serp20 and serp50 candidate pools, within a given
(search_engine, llm_model) category.

For each of the 4 categories we:
  1. Identify robust winners.
  2. Build a per-category dataset by stacking the serp20 and serp50 rows
     for those (keyword, url) pairs (with pool_size as an extra confounder).
  3. Fit a DoubleML PLR for each treatment in ALL_TREATMENTS, controlling for
     CONFOUNDERS + pool indicator, with outcomes rank_delta and post_rank.

Output: consolidated_results/dml_robust_winners.csv (one row per
(category, treatment, outcome) fit).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import doubleml as dml
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import ALL_TREATMENTS, CONFOUNDERS

INPUT_CSV = SCRIPT_DIR / "consolidated_results" / "regression_dataset.csv"
OUT_CSV   = SCRIPT_DIR / "consolidated_results" / "dml_robust_winners.csv"
OUT_PIVOT = SCRIPT_DIR / "consolidated_results" / "dml_robust_winners_pivot.csv"

OUTCOMES = ["rank_delta", "post_rank"]
TREATMENTS = {**ALL_TREATMENTS, "T_llms_txt": "has_llms_txt"}


def lgbm():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05,
                         num_leaves=31, min_child_samples=20,
                         random_state=42, verbose=-1)


def fit_plr(sub: pd.DataFrame, t_col: str, y_col: str, conf_cols: list[str]) -> dict:
    cols = [t_col, y_col] + conf_cols
    sub = sub[cols].dropna(subset=[t_col, y_col]).copy()
    if len(sub) < 200 or sub[t_col].nunique() < 2:
        return {"n": len(sub), "coef": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_val": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "skip": True}
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(sub[conf_cols]),
                         columns=conf_cols, index=sub.index)
    X = pd.DataFrame(StandardScaler().fit_transform(X_imp),
                     columns=conf_cols, index=sub.index)
    data = dml.DoubleMLData.from_arrays(
        x=X.values, y=sub[y_col].values, d=sub[t_col].values)
    model = dml.DoubleMLPLR(data, ml_l=lgbm(), ml_m=lgbm(),
                            n_folds=5, score="partialling out")
    model.fit()
    ci = model.confint(level=0.95)
    return {
        "n": len(sub),
        "coef": float(model.coef[0]),
        "se": float(model.se[0]),
        "t_stat": float(model.t_stat[0]),
        "p_val": float(model.pval[0]),
        "ci_low": float(ci.iloc[0, 0]),
        "ci_high": float(ci.iloc[0, 1]),
        "skip": False,
    }


def model_short(name: str) -> str:
    if "Llama" in name: return "Llama-3.3-70B"
    if "Qwen" in name:  return "Qwen2.5-72B"
    return name


def stars(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""


def main():
    print(f"Loading {INPUT_CSV} …", flush=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["model_short"] = df["llm_model"].map(model_short)
    df["category"] = df["search_engine"] + "+" + df["model_short"]

    base_conf = [c for c in CONFOUNDERS if c in df.columns]
    conf_cols = base_conf + ["serp_pool_size"]

    print(f"  rows: {len(df):,}  treatments: {len(TREATMENTS)}  outcomes: {len(OUTCOMES)}")
    print(f"  base confounders ({len(base_conf)}): {base_conf}")
    print()

    # Build robust-winners subset per category
    robust = {}
    for cat, grp in df.groupby("category"):
        p20 = grp[grp.serp_pool_size == 20].set_index(["keyword", "url"])
        p50 = grp[grp.serp_pool_size == 50].set_index(["keyword", "url"])
        common_idx = p20.index.intersection(p50.index)
        sub = pd.concat([p20.loc[common_idx], p50.loc[common_idx]]).reset_index()
        # de-dup if any
        sub = sub.drop_duplicates(subset=["keyword","url","serp_pool_size"])
        robust[cat] = sub
        n_kw = sub["keyword"].nunique()
        n_pairs = len(common_idx)
        print(f"  {cat:<35} robust pairs: {n_pairs:>5}  rows: {len(sub):>5}  keywords: {n_kw}")
    print()

    rows = []
    n_fits = len(robust) * len(TREATMENTS) * len(OUTCOMES)
    i = 0
    t0 = time.time()
    for cat, sub in robust.items():
        for t_key, t_col in TREATMENTS.items():
            if t_col not in sub.columns:
                continue
            for y_col in OUTCOMES:
                i += 1
                try:
                    res = fit_plr(sub, t_col, y_col, conf_cols)
                    res.update({"category": cat, "treatment": t_key,
                                "treatment_col": t_col, "outcome": y_col})
                    rows.append(res)
                    sig = stars(res["p_val"])
                    print(f"  [{i:>3}/{n_fits}] {cat:<35} {t_key:<26} y={y_col:<11} "
                          f"n={res['n']:>5}  coef={res['coef']:+.4f}  p={res['p_val']:.3g}{sig}",
                          flush=True)
                except Exception as e:
                    print(f"  [{i:>3}/{n_fits}] {cat:<35} {t_key:<26} y={y_col:<11} ERROR: {e}",
                          flush=True)
                    rows.append({"category": cat, "treatment": t_key,
                                 "treatment_col": t_col, "outcome": y_col,
                                 "n": np.nan, "coef": np.nan, "se": np.nan,
                                 "t_stat": np.nan, "p_val": np.nan,
                                 "ci_low": np.nan, "ci_high": np.nan, "skip": True})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(out)} rows, {time.time()-t0:.0f}s)")

    # Pivot per outcome: treatment × category → "coef±se sig"
    def cell(r):
        if pd.isna(r["coef"]):
            return ""
        return f"{r['coef']:+.4f}{stars(r['p_val'])}"

    out["cell"] = out.apply(cell, axis=1)
    pivots = []
    for y in OUTCOMES:
        piv = (out[out.outcome == y]
               .pivot(index="treatment", columns="category", values="cell")
               .fillna(""))
        piv.columns = [f"{y}::{c}" for c in piv.columns]
        pivots.append(piv)
    pd.concat(pivots, axis=1).to_csv(OUT_PIVOT)
    print(f"Wrote {OUT_PIVOT}")


if __name__ == "__main__":
    main()
