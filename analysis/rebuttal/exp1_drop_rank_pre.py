#!/usr/bin/env python3
"""Experiment 1 — total effects when dropping rank_pre / the SERP block.

Spec A' = headline confounders minus conf_serp_position (rank_pre).
Spec B' = headline confounders minus the 4-column SERP-derived block.
All fits are headline-spec (mutually controlled, Spec B in paper terms).
Also computes cross-fitted outcome-nuisance R^2 per spec per outcome.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, TREATMENTS, CONFOUNDERS, OUTCOMES, RANK_PRE,
                    SERP_BLOCK, get_pool, get_admitted, specB_fit, cell_cols,
                    crossfit_r2, ALPHA6)

t_start = time.time()
pool, admitted = get_pool(), get_admitted()

SPECS = {
    "headline": CONFOUNDERS,
    "minus_rank_pre": [c for c in CONFOUNDERS if c != RANK_PRE],
    "minus_serp_block": [c for c in CONFOUNDERS if c not in SERP_BLOCK],
}

rows = []
for outcome, is_clf in OUTCOMES:
    df = pool if outcome == "selected" else admitted
    for spec_name, conf in SPECS.items():
        if spec_name == "headline":
            continue  # published + step0 already cover it
        for tr in TREATMENTS:
            r = specB_fit(df, tr, outcome, is_clf, confounders=conf)
            r["conf_spec"] = spec_name
            rows.append(r)
            print(f"  [{outcome:10s}|{spec_name:16s}] {tr:30s} "
                  f"coef={r['coef']:+.4f} se={r['se']:.4f} p={r['p_val']:.2e} "
                  f"({r['seconds']}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "exp1_drop_rank_pre_fits.csv", index=False)

# ── outcome-nuisance cross-fitted R^2 per spec per outcome ────────────────
print("\n[exp1] outcome-nuisance R^2 (Y ~ confounders + cell dummies, "
      "5-fold cross-fitted, seed 42)", flush=True)
r2_rows = []
for outcome, is_clf in OUTCOMES:
    df0 = pool if outcome == "selected" else admitted
    for spec_name, conf in SPECS.items():
        df = df0.dropna(subset=[outcome]).copy()
        X_cols = conf + cell_cols(df)
        for c in X_cols:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = 0.0
        if len(df) > 200_000:
            df = df.sample(n=200_000, random_state=42).reset_index(drop=True)
        y = df[outcome].astype(float).values
        X = df[X_cols].astype(float).values
        r2 = crossfit_r2(y, X, is_clf=is_clf)
        r2_rows.append({"outcome": outcome, "conf_spec": spec_name, "r2": r2,
                        "n": len(df), "n_x_cols": len(X_cols)})
        print(f"  R2[{outcome:10s}|{spec_name:16s}] = {r2:.4f}", flush=True)

pd.DataFrame(r2_rows).to_csv(OUT / "exp1_nuisance_r2.csv", index=False)
print(f"[exp1] total {time.time()-t_start:.0f}s")
