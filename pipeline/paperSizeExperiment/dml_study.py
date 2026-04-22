"""Expanded DML study.

Ranking convention (GEODML):
    rank 1 = best position, the goal
    pre_rank / post_rank:  LOWER is BETTER
    rank_delta = pre_rank - post_rank  →  POSITIVE means LLM promoted the page

Direction tagging for the coefficient:
    outcome = rank_delta  →  POSITIVE coef means treatment helps (GOOD)
    outcome = post_rank   →  NEGATIVE coef means treatment helps (GOOD)

Subsets (15):
    8 individual runs
    per-engine      : duckduckgo, searxng
    per-model       : Llama-3.3-70B, Qwen2.5-72B
    per-pool        : serp20, serp50
    POOLED (all)

Treatments (19): 18 pre-declared in config + has_llms_txt
Outcomes (2):    rank_delta (primary), post_rank (secondary)
Estimator:       DoubleML PLR with LightGBM (5-fold cross-fitting)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import doubleml as dml
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import ALL_TREATMENTS, TREATMENT_LABELS, CONFOUNDERS

INPUT_CSV = SCRIPT_DIR / "consolidated_results" / "regression_dataset.csv"
OUT_DIR = SCRIPT_DIR / "consolidated_results" / "dml_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["rank_delta", "post_rank"]
N_FOLDS = 5
RANDOM_STATE = 42

TREATMENTS = {**ALL_TREATMENTS, "T_llms_txt": "has_llms_txt"}
TREATMENT_LABELS = {**TREATMENT_LABELS, "T_llms_txt": "llms.txt present (binary)"}

# ── Direction & significance helpers ─────────────────────────────────────────

def interpret(coef: float, outcome: str) -> str:
    """Label the causal direction given GEODML's ranking convention."""
    if coef is None or np.isnan(coef):
        return ""
    if outcome == "rank_delta":
        return "promotes" if coef > 0 else "demotes"
    # post_rank: lower = better
    return "promotes" if coef < 0 else "demotes"


def stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _learners():
    mk = lambda: LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31,
        verbose=-1, random_state=RANDOM_STATE,
    )
    return mk(), mk()


# ── DML ──────────────────────────────────────────────────────────────────────

def preprocess(df, t_col, y_col, confounders):
    cols = confounders + [t_col, y_col]
    sub = df[cols].dropna(subset=[t_col, y_col])
    if len(sub) < 50:
        return None
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(sub[confounders]),
                         columns=confounders, index=sub.index)
    X = pd.DataFrame(StandardScaler().fit_transform(X_imp),
                     columns=confounders, index=sub.index)
    return X, sub[y_col].values, sub[t_col].values


def run_dml(X, Y, D):
    ml_l, ml_m = _learners()
    data = dml.DoubleMLData.from_arrays(x=X.values, y=Y, d=D)
    model = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m,
                            n_folds=N_FOLDS, score="partialling out")
    model.fit()
    ci = model.confint(level=0.95)
    return dict(
        coef=float(model.coef[0]),
        se=float(model.se[0]),
        t_stat=float(model.t_stat[0]),
        p_val=float(model.pval[0]),
        ci_lower=float(ci.iloc[0, 0]),
        ci_upper=float(ci.iloc[0, 1]),
    )


# ── Subset iterator ──────────────────────────────────────────────────────────

def run_model_short(run_id: str) -> str:
    if "Llama" in run_id: return "Llama-3.3-70B"
    if "Qwen"  in run_id: return "Qwen2.5-72B"
    return "?"


def run_engine(run_id: str) -> str:
    return run_id.split("_", 1)[0]


def run_pool(run_id: str) -> int:
    return 20 if "serp20" in run_id else 50


def iter_subsets(df):
    base_conf = [c for c in CONFOUNDERS if c in df.columns]

    # 1. Individual runs
    for r in sorted(df["run_id"].unique()):
        yield r, df[df["run_id"] == r].copy(), list(base_conf), "run"

    # 2. Per-engine
    df = df.copy()
    df["_engine"] = df["run_id"].map(run_engine)
    for eng in sorted(df["_engine"].unique()):
        sub = df[df["_engine"] == eng].copy()
        # Add run-within-engine dummies as extra confounders
        dummies = pd.get_dummies(sub["run_id"], prefix="run", drop_first=True).astype(float)
        sub = pd.concat([sub.reset_index(drop=True),
                         dummies.reset_index(drop=True)], axis=1)
        yield f"ENG:{eng}", sub, base_conf + list(dummies.columns), "engine"

    # 3. Per-model
    df["_model"] = df["run_id"].map(run_model_short)
    for m in sorted(df["_model"].unique()):
        sub = df[df["_model"] == m].copy()
        dummies = pd.get_dummies(sub["run_id"], prefix="run", drop_first=True).astype(float)
        sub = pd.concat([sub.reset_index(drop=True),
                         dummies.reset_index(drop=True)], axis=1)
        yield f"MOD:{m}", sub, base_conf + list(dummies.columns), "model"

    # 4. Per-pool
    df["_pool"] = df["run_id"].map(run_pool)
    for p in sorted(df["_pool"].unique()):
        sub = df[df["_pool"] == p].copy()
        dummies = pd.get_dummies(sub["run_id"], prefix="run", drop_first=True).astype(float)
        sub = pd.concat([sub.reset_index(drop=True),
                         dummies.reset_index(drop=True)], axis=1)
        yield f"POOL:{p}", sub, base_conf + list(dummies.columns), "pool"

    # 5. POOLED (all)
    dummies = pd.get_dummies(df["run_id"], prefix="run", drop_first=True).astype(float)
    pooled = pd.concat([df.reset_index(drop=True),
                        dummies.reset_index(drop=True)], axis=1)
    yield "POOLED", pooled, base_conf + list(dummies.columns), "pooled"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_CSV} …", flush=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df["pre_rank"].notna()].copy()
    print(f"  rows: {len(df):,}  runs: {df['run_id'].nunique()}", flush=True)

    subsets = list(iter_subsets(df))
    total = len(subsets) * len(TREATMENTS) * len(OUTCOMES)
    print(f"  subsets: {len(subsets)} · treatments: {len(TREATMENTS)} · "
          f"outcomes: {len(OUTCOMES)} · planned fits: {total}", flush=True)

    rows, t0, i = [], time.time(), 0
    for subset_label, sub_df, conf_list, subset_type in subsets:
        for t_key, t_col in TREATMENTS.items():
            if t_col not in sub_df.columns:
                continue
            conf = [c for c in conf_list if c != t_col and c in sub_df.columns]
            for y_col in OUTCOMES:
                i += 1
                prep = preprocess(sub_df, t_col, y_col, conf)
                if prep is None:
                    rows.append(dict(
                        subset=subset_label, subset_type=subset_type,
                        treatment=t_key, treatment_col=t_col,
                        treatment_label=TREATMENT_LABELS.get(t_key, t_key),
                        outcome=y_col, n=0,
                        coef=np.nan, se=np.nan, t_stat=np.nan,
                        p_val=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                        stars="", direction="", note="insufficient",
                    ))
                    continue
                X, Y, D = prep
                try:
                    res = run_dml(X, Y, D); note = ""
                except Exception as e:
                    res = dict(coef=np.nan, se=np.nan, t_stat=np.nan, p_val=np.nan,
                               ci_lower=np.nan, ci_upper=np.nan)
                    note = f"ERR:{type(e).__name__}"[:60]
                rows.append(dict(
                    subset=subset_label, subset_type=subset_type,
                    treatment=t_key, treatment_col=t_col,
                    treatment_label=TREATMENT_LABELS.get(t_key, t_key),
                    outcome=y_col, n=len(Y),
                    **res,
                    stars=stars(res["p_val"]),
                    direction=interpret(res["coef"], y_col),
                    note=note,
                ))
                if i % 20 == 0 or i == total:
                    el = time.time() - t0
                    eta = (el / i) * (total - i)
                    print(f"  [{i}/{total}] {subset_label:22s} {t_key:28s} {y_col:10s}  "
                          f"coef={res['coef']:+.4f} p={res['p_val']:.3f} n={len(Y):,}  "
                          f"ETA={eta/60:.1f}min", flush=True)

    out = pd.DataFrame(rows)
    long_path = OUT_DIR / "dml_results_long.csv"
    out.to_csv(long_path, index=False)
    print(f"\nwrote {long_path}  ({len(out)} rows)")

    # Pivots (coef + stars)
    for y in OUTCOMES:
        sub = out[out.outcome == y].copy()
        coef_piv = sub.pivot_table(index="treatment", columns="subset",
                                   values="coef", aggfunc="first")
        star_piv = sub.pivot_table(index="treatment", columns="subset",
                                   values="stars", aggfunc="first")
        disp = coef_piv.copy().astype(object)
        for col in disp.columns:
            for ix in disp.index:
                v = coef_piv.loc[ix, col]
                s = star_piv.loc[ix, col] if pd.notna(star_piv.loc[ix, col]) else ""
                disp.loc[ix, col] = f"{v:+.4f}{s}" if pd.notna(v) else "—"
        piv_path = OUT_DIR / f"dml_pivot_{y}.csv"
        disp.to_csv(piv_path)
        print(f"wrote {piv_path}")

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
