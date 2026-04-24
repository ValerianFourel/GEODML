"""How much variance do the confounders explain?

For each nuisance model (g_0 for outcome, m_0 for treatment), report
5-fold cross-validated R² (same folds DML uses internally). We compute:

  1. R² with ALL confounders             (what DML saw)
  2. R² with ONLY original conf_* cols   (baseline, no DataForSEO)
  3. R² with ONLY dfs_* cols             (DataForSEO alone)
  4. Incremental ΔR² from adding DataForSEO

CV R² is the right yardstick: it tells you the genuine signal DML could
extract, free of overfit. Low Y-side R² means treatment effects live in
a noisy residual; low D-side R² is actually healthy for DML (strong
residual treatment variation to identify the effect from).

Runs on POOLED (65k rows).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import CONFOUNDERS  # noqa: E402

INPUT_CSV = SCRIPT_DIR / "consolidated_results" / "regression_dataset.csv"
OUT_DIR = SCRIPT_DIR / "consolidated_results" / "dml_study"

OUTCOMES = ["rank_delta", "post_rank"]
TREATMENTS = [
    "treat_source_earned",
    "has_llms_txt",
    "treat_topical_comp",
    "treat_freshness",
    "treat_structured_data",
    "T3_structured_data_code",
    "treat_question_headings",
    "T1_statistical_density_code",
]
N_FOLDS = 5
RANDOM_STATE = 42


def _learner():
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
    )


def cv_r2(df: pd.DataFrame, target: str, confounders: list[str]) -> tuple[float, int]:
    sub = df[confounders + [target]].dropna(subset=[target])
    if len(sub) < 100 or not confounders:
        return float("nan"), len(sub)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(sub[confounders])
    X = StandardScaler().fit_transform(X)
    y = sub[target].values.astype(float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y)
    for tr, te in kf.split(X):
        m = _learner()
        m.fit(X[tr], y[tr])
        y_pred[te] = m.predict(X[te])
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, len(sub)


def main() -> int:
    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["pre_rank"].notna()].copy()
    print(f"  {len(df):,} rows")

    confounders = [c for c in CONFOUNDERS if c in df.columns]
    orig = [c for c in confounders if not c.startswith("dfs_")]
    dfs = [c for c in confounders if c.startswith("dfs_")]
    print(f"  {len(orig)} original + {len(dfs)} DataForSEO = {len(confounders)} confounders")

    rows: list[dict] = []
    print(f"\n=== 5-fold CV R² (higher = more variance explained) ===")
    print(f"{'target':32s}  {'n':>6s}  {'R²_all':>8s}  {'R²_orig':>8s}  {'R²_dfs':>8s}  {'ΔR²_dfs':>8s}")
    print("-" * 80)

    targets = [("OUT " + y, y) for y in OUTCOMES] + [("TRT " + t, t) for t in TREATMENTS]
    for label, tgt in targets:
        if tgt not in df.columns:
            continue
        r2_all, n = cv_r2(df, tgt, confounders)
        r2_orig, _ = cv_r2(df, tgt, orig)
        r2_dfs, _ = cv_r2(df, tgt, dfs)
        delta = r2_all - r2_orig if not (np.isnan(r2_all) or np.isnan(r2_orig)) else float("nan")
        rows.append({
            "target": tgt, "label": label, "n": n,
            "r2_all": r2_all, "r2_orig": r2_orig, "r2_dfs": r2_dfs, "delta_r2_dfs": delta,
        })
        print(
            f"{label:32s}  {n:>6,d}  {r2_all:>8.4f}  {r2_orig:>8.4f}  {r2_dfs:>8.4f}  {delta:>+8.4f}"
        )

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "variance_explained.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
