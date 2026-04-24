"""Which confounders drive R²(Y|X)? Which ones are significant on rank?

Two diagnostics on the POOLED subset:

  A. Leave-one-out ΔR² — for each confounder c, refit g(X\c) → Y and report
     how much 5-fold CV R² drops when c is removed. This is the confounder's
     *unique* contribution to outcome prediction (partial R²), after the
     other confounders have had their say. Large ΔR² means "nothing else
     in X substitutes for this column".

  B. OLS significance — standardised OLS of outcome ~ all confounders,
     median-imputed, with HC3 robust SEs. Coef, t-stat, p-value per
     confounder. Tells you which individual confounders carry detectable
     signal about rank after controlling for all others.

Run for both outcomes (rank_delta, post_rank).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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
N_FOLDS = 5
RANDOM_STATE = 42


def _learner():
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
    )


def cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pred = np.empty_like(y, dtype=float)
    for tr, te in kf.split(X):
        m = _learner()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def prep_X(df: pd.DataFrame, y_col: str, cols: list[str]):
    sub = df[cols + [y_col]].dropna(subset=[y_col])
    if not cols:
        return np.empty((len(sub), 0)), sub[y_col].values.astype(float)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(sub[cols])
    X = StandardScaler().fit_transform(X)
    return X, sub[y_col].values.astype(float)


def leave_one_out(df: pd.DataFrame, y_col: str, confounders: list[str]) -> pd.DataFrame:
    X_full, y = prep_X(df, y_col, confounders)
    r2_full = cv_r2(X_full, y)
    print(f"  full R²(Y={y_col}|X) = {r2_full:.4f}")
    rows = []
    for i, c in enumerate(confounders):
        remaining = [x for x in confounders if x != c]
        X_minus, _ = prep_X(df, y_col, remaining)
        r2_minus = cv_r2(X_minus, y)
        delta = r2_full - r2_minus
        print(f"    drop {c:32s}  R²={r2_minus:.4f}  ΔR²={delta:+.5f}")
        rows.append({
            "outcome": y_col, "confounder": c,
            "r2_full": r2_full, "r2_without": r2_minus,
            "delta_r2": delta,
        })
    return pd.DataFrame(rows)


def ols_significance(df: pd.DataFrame, y_col: str, confounders: list[str]) -> pd.DataFrame:
    X, y = prep_X(df, y_col, confounders)
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit(cov_type="HC3")
    names = ["intercept"] + confounders
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "outcome": y_col, "confounder": name,
            "coef": float(model.params[i]),
            "se": float(model.bse[i]),
            "t_stat": float(model.tvalues[i]),
            "p_val": float(model.pvalues[i]),
            "ci_low": float(model.conf_int()[i][0]),
            "ci_high": float(model.conf_int()[i][1]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["pre_rank"].notna()].copy()
    print(f"  {len(df):,} rows")

    confounders = [c for c in CONFOUNDERS if c in df.columns]
    print(f"  {len(confounders)} confounders")

    # A. Leave-one-out ΔR²
    print(f"\n=== A. Leave-one-out ΔR² per confounder ===")
    loo_frames = []
    for y_col in OUTCOMES:
        print(f"\noutcome = {y_col}")
        loo_frames.append(leave_one_out(df, y_col, confounders))
    loo = pd.concat(loo_frames, ignore_index=True)
    loo_path = OUT_DIR / "confounder_loo_r2.csv"
    loo.to_csv(loo_path, index=False)
    print(f"\nwrote {loo_path}")

    # B. OLS significance
    print(f"\n=== B. OLS significance (standardised, HC3 SEs) ===")
    ols_frames = []
    for y_col in OUTCOMES:
        print(f"\noutcome = {y_col}")
        ols_df = ols_significance(df, y_col, confounders)
        # Flag significance
        def stars(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            if p < 0.10:  return "·"
            return ""
        ols_df["stars"] = ols_df["p_val"].apply(stars)
        # Sort by |t|
        ols_df["abs_t"] = ols_df["t_stat"].abs()
        print(ols_df.sort_values("abs_t", ascending=False)
              [["confounder", "coef", "se", "t_stat", "p_val", "stars"]]
              .to_string(index=False))
        ols_df = ols_df.drop(columns=["abs_t"])
        ols_frames.append(ols_df)
    ols = pd.concat(ols_frames, ignore_index=True)
    ols_path = OUT_DIR / "confounder_ols_significance.csv"
    ols.to_csv(ols_path, index=False)
    print(f"\nwrote {ols_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
