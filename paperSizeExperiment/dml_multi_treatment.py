"""Multi-treatment DML studies on POOLED.

Two analyses based on DoubleML's multi-treatment API
(https://docs.doubleml.org/stable/guide/se_confint.html):

  STUDY 1 — Joint inference on marginal effects.
      Pass all 19 treatments as d_cols in a single DoubleMLData. Each θ_j
      is estimated with only X as confounders (same as our main study),
      BUT the multiplier-bootstrap gives:
        - simultaneous 95% confidence bands (confint(joint=True))
        - Romano-Wolf stepdown adjusted p-values  (p_adjust("romano-wolf"))
        - Bonferroni-adjusted p-values             (p_adjust("bonferroni"))
      This controls the family-wise error rate across the 19 tests.

  STUDY 2 — Mutually-controlled partial effects.
      For each treatment j, put the OTHER 18 treatments into X. Each θ_j
      is now the effect of treatment j holding the other treatments fixed
      — a stronger identification claim. Answers: "does each treatment
      still matter once we control for the other content-quality signals?"

Both runs on POOLED (65,203 rows) for both outcomes (rank_delta, post_rank).
"""

from __future__ import annotations

import sys
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
from config import ALL_TREATMENTS, CONFOUNDERS  # noqa: E402

INPUT_CSV = SCRIPT_DIR / "consolidated_results" / "regression_dataset.csv"
OUT_DIR = SCRIPT_DIR / "consolidated_results" / "dml_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREATMENTS = {**ALL_TREATMENTS, "T_llms_txt": "has_llms_txt"}
OUTCOMES = ["rank_delta", "post_rank"]
N_FOLDS = 5
N_BOOTSTRAP = 500
RANDOM_STATE = 42


def _learners():
    mk = lambda: LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31,
        verbose=-1, random_state=RANDOM_STATE,
    )
    return mk(), mk()


def _preprocess(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Median-impute + standardise selected columns; drops rows with NaN in cols."""
    sub = df[cols].copy()
    imp = SimpleImputer(strategy="median")
    mat = imp.fit_transform(sub)
    mat = StandardScaler().fit_transform(mat)
    return pd.DataFrame(mat, columns=cols, index=sub.index)


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "·"
    return ""


def direction(coef: float, outcome: str) -> str:
    if np.isnan(coef):
        return ""
    if outcome == "rank_delta":
        return "promotes" if coef > 0 else "demotes"
    return "promotes" if coef < 0 else "demotes"


# ── STUDY 1 ─────────────────────────────────────────────────────────────────

def study_1_joint_inference(df: pd.DataFrame, confounders: list[str]) -> pd.DataFrame:
    """All 19 treatments jointly, multiplier-bootstrap for family-wise CIs."""
    t_keys = list(TREATMENTS.keys())
    t_cols = [TREATMENTS[k] for k in t_keys if TREATMENTS[k] in df.columns]
    valid_keys = [k for k in t_keys if TREATMENTS[k] in t_cols]

    print(f"\n=== STUDY 1 — Joint inference on {len(t_cols)} treatments ===")
    rows = []
    for y_col in OUTCOMES:
        print(f"\noutcome = {y_col}")

        # Drop rows with NaN on Y or any treatment (keeps sample consistent)
        needed = [y_col] + t_cols
        sub = df[needed + confounders].dropna(subset=needed).copy()
        print(f"  n = {len(sub):,}")
        X = _preprocess(sub, confounders)
        Y = sub[y_col].values.astype(float)
        D = sub[t_cols].values.astype(float)

        data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)
        ml_l, ml_m = _learners()
        model = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m,
                                n_folds=N_FOLDS, score="partialling out")
        model.fit()
        # Multiplier bootstrap for joint inference (Gaussian by default)
        model.bootstrap(method="normal", n_rep_boot=N_BOOTSTRAP)

        # Per-treatment marginal CIs and joint CIs
        ci_ind = model.confint(level=0.95, joint=False)
        ci_joint = model.confint(level=0.95, joint=True)
        # Adjusted p-values
        pa_rw = model.p_adjust(method="romano-wolf")
        pa_bon = model.p_adjust(method="bonferroni")

        for i, (k, c) in enumerate(zip(valid_keys, t_cols)):
            rows.append({
                "study": "joint_inference",
                "outcome": y_col,
                "treatment": k,
                "treatment_col": c,
                "n": len(sub),
                "coef": float(model.coef[i]),
                "se": float(model.se[i]),
                "t_stat": float(model.t_stat[i]),
                "p_val": float(model.pval[i]),
                "ci_lower_marg": float(ci_ind.iloc[i, 0]),
                "ci_upper_marg": float(ci_ind.iloc[i, 1]),
                "ci_lower_joint": float(ci_joint.iloc[i, 0]),
                "ci_upper_joint": float(ci_joint.iloc[i, 1]),
                "p_val_romano_wolf": float(pa_rw.iloc[i, 1]),
                "p_val_bonferroni": float(pa_bon.iloc[i, 1]),
            })

        # Print a compact summary now
        print(f"  {'treatment':30s} {'coef':>9s} {'p':>8s} {'p_RW':>8s} {'p_Bonf':>8s}  sig")
        for r in rows[-len(t_cols):]:
            marks = (
                f"marg={stars(r['p_val'])} "
                f"RW={stars(r['p_val_romano_wolf'])} "
                f"Bonf={stars(r['p_val_bonferroni'])}"
            )
            print(
                f"  {r['treatment']:30s} {r['coef']:>+9.4f} "
                f"{r['p_val']:>8.4g} {r['p_val_romano_wolf']:>8.4g} "
                f"{r['p_val_bonferroni']:>8.4g}  {marks}"
            )

    return pd.DataFrame(rows)


# ── STUDY 2 ─────────────────────────────────────────────────────────────────

def study_2_mutually_controlled(df: pd.DataFrame, confounders: list[str]) -> pd.DataFrame:
    """For each treatment j, put the other 18 treatments into X as confounders."""
    t_keys = list(TREATMENTS.keys())
    t_cols = [TREATMENTS[k] for k in t_keys if TREATMENTS[k] in df.columns]
    valid_keys = [k for k in t_keys if TREATMENTS[k] in t_cols]

    print(f"\n=== STUDY 2 — Mutually-controlled partial effects ===")
    print(f"  each treatment estimated with ({len(confounders)} confounders + "
          f"{len(t_cols) - 1} other treatments) as X")

    rows = []
    for y_col in OUTCOMES:
        print(f"\noutcome = {y_col}")
        for i, (k, c) in enumerate(zip(valid_keys, t_cols)):
            other = [tc for tc in t_cols if tc != c]
            X_cols = confounders + other
            needed = [y_col, c]
            sub = df[needed + X_cols].dropna(subset=[y_col, c]).copy()
            if len(sub) < 100:
                continue

            X = _preprocess(sub, X_cols)
            Y = sub[y_col].values.astype(float)
            D = sub[c].values.astype(float)

            data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)
            ml_l, ml_m = _learners()
            model = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m,
                                    n_folds=N_FOLDS, score="partialling out")
            model.fit()
            ci = model.confint(level=0.95)
            rows.append({
                "study": "mutually_controlled",
                "outcome": y_col,
                "treatment": k,
                "treatment_col": c,
                "n": len(sub),
                "coef": float(model.coef[0]),
                "se": float(model.se[0]),
                "t_stat": float(model.t_stat[0]),
                "p_val": float(model.pval[0]),
                "ci_lower": float(ci.iloc[0, 0]),
                "ci_upper": float(ci.iloc[0, 1]),
                "n_other_treats_in_X": len(other),
                "n_confounders_in_X": len(confounders),
            })
            print(
                f"  {k:30s} coef={model.coef[0]:+.4f}  p={model.pval[0]:.4g}  "
                f"{stars(float(model.pval[0])):3s}  n={len(sub):,}"
            )
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["pre_rank"].notna()].copy()
    print(f"  {len(df):,} rows")

    confounders = [c for c in CONFOUNDERS if c in df.columns]
    print(f"  {len(confounders)} confounders: {confounders}")

    s1 = study_1_joint_inference(df, confounders)
    s2 = study_2_mutually_controlled(df, confounders)

    out = pd.concat([s1, s2], ignore_index=True, sort=False)
    out_path = OUT_DIR / "dml_multi_treatment.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(out)} rows)")

    # Also write separate CSVs for convenience
    s1.to_csv(OUT_DIR / "dml_multi_treatment_study1_joint.csv", index=False)
    s2.to_csv(OUT_DIR / "dml_multi_treatment_study2_partial.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
