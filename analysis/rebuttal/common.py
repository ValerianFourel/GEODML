"""Shared harness for ARR 9568 rebuttal experiments.

Reuses the EXACT builders/estimator from scripts/dml_canonical.py (imported,
not copied) so the sample frames, filters, seed, folds, and LightGBM
hyperparameters are identical to the headline run. Sample frames are cached
to rebuttal/out/cache/ so every experiment operates on the same rows.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "rebuttal" / "out"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# ── import scripts/dml_canonical.py without running main() ────────────────
_spec = importlib.util.spec_from_file_location(
    "dml_canonical", REPO / "scripts" / "dml_canonical.py")
dc = importlib.util.module_from_spec(_spec)
sys.modules["dml_canonical"] = dc
_spec.loader.exec_module(dc)
sys.stdout = sys.__stdout__  # undo the module-level tee installed on import

TREATMENTS = dc.TREATMENTS            # 6 canonical treatments
CONFOUNDERS = dc.CONFOUNDERS          # 29 (28 + has_llms_txt)
plr_estimate = dc.plr_estimate        # Robinson PLR, LGBM, KFold(5,seed=42), IF SEs
cross_fit_resid = dc.cross_fit_resid
LGBM_KW = dc._LGBM_KW

RANK_PRE = "conf_serp_position"       # the paper's `rank_pre` confounder
SERP_BLOCK = ["conf_title_has_kw", "conf_title_len",
              "conf_snippet_len", "conf_serp_position"]

# paper label -> column
PAPER_LABEL = {
    "T1": "treat_stats_density",
    "T2": "treat_question_headings",
    "T3": "treat_structured_data",
    "T4": "treat_freshness",            # paper T4 = freshness
    "T5": "treat_topical_comp",
    "T6": "T4_citation_authority_code", # citation authority (code name is historical)
}
COL_TO_PAPER = {v: k for k, v in PAPER_LABEL.items()}

OUTCOMES = [("selected", True), ("rank_delta", False), ("post_rank", False)]

ALPHA6 = 0.05 / 6  # 0.008333


def get_pool() -> pd.DataFrame:
    f = CACHE / "pool_admission.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = dc.build_pool_admission()
    df, _ = dc.add_cell_dummies(df)
    sys.stdout = sys.__stdout__
    df.to_parquet(f)
    return df


def get_admitted() -> pd.DataFrame:
    f = CACHE / "admitted.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = dc.build_admitted_sample()
    df, _ = dc.add_cell_dummies(df)
    sys.stdout = sys.__stdout__
    df.to_parquet(f)
    return df


def cell_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("cell_")]


def specB_fit(df, treatment, outcome, is_clf, confounders=None, extra_note=""):
    """One headline-spec (Spec B mutually-controlled) fit."""
    conf = CONFOUNDERS if confounders is None else confounders
    other = [t for t in TREATMENTS if t != treatment]
    t0 = time.time()
    r = plr_estimate(df, treatment, other, conf + cell_cols(df), outcome,
                     is_clf=is_clf)
    r.update(treatment=treatment, outcome=outcome,
             seconds=round(time.time() - t0, 1), note=extra_note)
    return r


def fmt_row(r):
    ci_lo = r["coef"] - 1.96 * r["se"]
    ci_hi = r["coef"] + 1.96 * r["se"]
    sig = "SIG" if r["p_val"] < ALPHA6 else ""
    return (f"{r['coef']:+.4f} ({r['se']:.4f}) [{ci_lo:+.4f}, {ci_hi:+.4f}]"
            f" p={r['p_val']:.2e} {sig}")


def published_specB() -> pd.DataFrame:
    p = (Path.home() / "geodml_data/data/dml_results/"
         "dml_canonical_2026-05-25_llms_as_confounder.parquet")
    df = pd.read_parquet(p)
    return df[(df.spec == "B") & (df["slice"] == "POOLED")].copy()


def crossfit_r2(y, X, seed=42, is_clf=False):
    """Cross-fitted R^2 of the outcome nuisance E[Y|X], same folds as headline."""
    resid = cross_fit_resid(y, X, seed=seed, is_clf=is_clf)
    return 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
