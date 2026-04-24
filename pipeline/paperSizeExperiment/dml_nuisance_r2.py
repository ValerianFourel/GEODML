"""Per-fit predictive performance of DML's internal nuisance models.

DML PLR estimates θ from the two nuisance regressions:

    Y = g₀(X) + ε_y     →  ε̂_y = Y - ĝ(X)
    D = m₀(X) + ε_d     →  ε̂_d = D - m̂(X)
    ε̂_y = θ · ε̂_d + noise   (final orthogonal regression)

This script reports, per (subset, treatment, outcome), the 5-fold CV R² of:

  g: X → Y   (how well confounders predict the outcome)
  m: X → D   (how well confounders predict the treatment)
  θ: D̃ → Ỹ  (how much of the *residualised* outcome variance the
             residualised treatment explains — a proxy for the effect size
             relative to post-confounder noise)

Same folds, learner, and preprocessing as `dml_study.py` so numbers are
directly comparable to the DML coefficients.
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
from config import ALL_TREATMENTS, CONFOUNDERS  # noqa: E402

INPUT_CSV = SCRIPT_DIR / "consolidated_results" / "regression_dataset.csv"
OUT_DIR = SCRIPT_DIR / "consolidated_results" / "dml_study"

OUTCOMES = ["rank_delta", "post_rank"]
TREATMENTS = {**ALL_TREATMENTS, "T_llms_txt": "has_llms_txt"}
N_FOLDS = 5
RANDOM_STATE = 42


def _learner():
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=31, verbose=-1, random_state=RANDOM_STATE,
    )


def cv_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pred = np.empty_like(y, dtype=float)
    for tr, te in kf.split(X):
        m = _learner()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def preprocess(df: pd.DataFrame, t_col: str, y_col: str, confounders: list[str]):
    sub = df[confounders + [t_col, y_col]].dropna(subset=[t_col, y_col])
    if len(sub) < 100:
        return None
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(sub[confounders])
    X = StandardScaler().fit_transform(X)
    y = sub[y_col].values.astype(float)
    d = sub[t_col].values.astype(float)
    return X, y, d


def main() -> int:
    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["pre_rank"].notna()].copy()
    print(f"  {len(df):,} rows (POOLED)")

    confounders = [c for c in CONFOUNDERS if c in df.columns]
    print(f"  {len(confounders)} confounders")

    # Cache Y predictions per outcome (they don't depend on the treatment)
    y_preds: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for y_col in OUTCOMES:
        sub = df[confounders + [y_col]].dropna(subset=[y_col])
        imp = SimpleImputer(strategy="median")
        X = StandardScaler().fit_transform(imp.fit_transform(sub[confounders]))
        y = sub[y_col].values.astype(float)
        yhat = cv_predict(X, y)
        r2_y = r2(y, yhat)
        y_preds[y_col] = (sub.index.values, yhat, r2_y)
        print(f"  g: X → Y={y_col:12s}  R² = {r2_y:+.4f}  (n={len(y):,})")

    rows = []
    print(f"\n=== Per-treatment nuisance R² (POOLED) ===")
    print(
        f"{'treatment':30s} {'outcome':11s} {'n':>6s}  "
        f"{'R²(Y|X)':>8s} {'R²(D|X)':>8s} {'R²(Ỹ|D̃)':>9s}  {'coef':>8s}"
    )
    print("-" * 90)

    for t_key, t_col in TREATMENTS.items():
        if t_col not in df.columns:
            continue
        for y_col in OUTCOMES:
            prep = preprocess(df, t_col, y_col, confounders)
            if prep is None:
                continue
            X, y, d = prep

            # Nuisance m: X → D
            dhat = cv_predict(X, d)
            r2_d = r2(d, dhat)

            # Nuisance g: X → Y (recompute on this treatment's subset since
            # dropna may remove rows)
            yhat = cv_predict(X, y)
            r2_y = r2(y, yhat)

            # Residuals
            y_tilde = y - yhat
            d_tilde = d - dhat

            # Structural regression: Ỹ = θ · D̃ + noise (no intercept)
            denom = float((d_tilde * d_tilde).sum())
            theta = float((d_tilde * y_tilde).sum() / denom) if denom > 0 else float("nan")
            y_tilde_pred = theta * d_tilde
            r2_struct = r2(y_tilde, y_tilde_pred)

            print(
                f"{t_key:30s} {y_col:11s} {len(y):>6,d}  "
                f"{r2_y:>+8.4f} {r2_d:>+8.4f} {r2_struct:>+9.4f}  {theta:>+8.4f}"
            )
            rows.append({
                "subset": "POOLED", "treatment": t_key, "treatment_col": t_col,
                "outcome": y_col, "n": len(y),
                "r2_g_Y_given_X": r2_y,
                "r2_m_D_given_X": r2_d,
                "r2_struct_Ytilde_given_Dtilde": r2_struct,
                "theta": theta,
            })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "nuisance_r2.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
