#!/usr/bin/env python3
"""Experiment 2 — DoubleML omitted-variable-bias sensitivity analysis.

The headline estimator is a hand-rolled Robinson PLR (scripts/dml_canonical.py),
so each pair is refit here with doubleml.DoubleMLPLR on the IDENTICAL frame
(same dropna -> median-fill -> 200k seed-42 subsample), identical LightGBM
hyperparameters, n_folds=5, score='partialling out'. The DoubleML theta is
reported next to the headline theta as a bridge check.

Deviation note: the headline code uses LGBMClassifier.predict_proba for the
outcome nuisance when Y=selected; DoubleMLPLR requires a regressor for ml_l,
so LGBMRegressor is used there (bridge check quantifies the difference).

RV/RVa at level 0.95 (alpha=0.05) for all 18 pairs; sensitivity_benchmark
with benchmarking sets ['conf_domain_authority'] and ['conf_serp_position']
for the 8 required pairs + any Bonferroni-significant pair.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import doubleml as dml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, TREATMENTS, CONFOUNDERS, OUTCOMES, get_pool,
                    get_admitted, cell_cols, LGBM_KW, COL_TO_PAPER, ALPHA6)

print(f"doubleml version: {dml.__version__}", flush=True)

REQUIRED_PAIRS = {  # paper labels from the rebuttal prompt
    ("treat_topical_comp", "selected"),      # T5 x admission
    ("treat_topical_comp", "post_rank"),     # T5 x post-rank
    ("treat_topical_comp", "rank_delta"),    # T5 x delta_rank
    ("treat_question_headings", "post_rank"),# T2 x post-rank
    ("treat_structured_data", "selected"),   # T3 x admission
    ("treat_structured_data", "post_rank"),  # T3 x post-rank
    ("treat_freshness", "selected"),         # T4(paper)=freshness x admission
    ("treat_freshness", "post_rank"),        # T4(paper)=freshness x post-rank
}


def prep_frame(df, focal, outcome):
    """EXACT mirror of plr_estimate() data prep (scripts/dml_canonical.py:159-176)."""
    ctrl_T = [t for t in TREATMENTS if t != focal]
    X_cols = CONFOUNDERS + cell_cols(df)
    df = df.dropna(subset=[outcome, focal]).copy()
    for c in ctrl_T + X_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = 0.0
    if len(df) > 200_000:
        df = df.sample(n=200_000, random_state=42).reset_index(drop=True)
    return df, ctrl_T + X_cols


t_start = time.time()
pool, admitted = get_pool(), get_admitted()

rows, sentences = [], []
for outcome, is_clf in OUTCOMES:
    base = pool if outcome == "selected" else admitted
    for tr in TREATMENTS:
        t0 = time.time()
        df, x_cols = prep_frame(base, tr, outcome)
        data = dml.DoubleMLData(
            df[[outcome, tr] + x_cols].astype(float),
            y_col=outcome, d_cols=tr, x_cols=x_cols)
        np.random.seed(42)
        m = dml.DoubleMLPLR(
            data,
            ml_l=LGBMRegressor(random_state=42, **LGBM_KW),
            ml_m=LGBMRegressor(random_state=42, **LGBM_KW),
            n_folds=5)
        m.fit(n_jobs_cv=1)
        theta, se, pval = float(m.coef[0]), float(m.se[0]), float(m.pval[0])

        m.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95)
        sp = m.sensitivity_params
        rv = float(np.asarray(sp["rv"]).ravel()[0])
        rva = float(np.asarray(sp["rva"]).ravel()[0])

        row = dict(treatment=tr, paper_label=COL_TO_PAPER.get(tr, tr),
                   outcome=outcome, n=len(df), theta_dml=theta, se_dml=se,
                   p_dml=pval, rv=rv, rva=rva)

        required = (tr, outcome) in REQUIRED_PAIRS
        if required:
            for bench_name, bench_set in [
                    ("domain_authority", ["conf_domain_authority"]),
                    ("rank_pre", ["conf_serp_position"])]:
                b = m.sensitivity_benchmark(benchmarking_set=bench_set)
                cf_y = float(b["cf_y"].iloc[0])
                cf_d = float(b["cf_d"].iloc[0])
                rho_b = float(b["rho"].iloc[0])
                # sign-robust if a confounder of benchmark strength cannot
                # push the estimate across zero
                m.sensitivity_analysis(cf_y=max(cf_y, 0.0), cf_d=max(cf_d, 0.0),
                                       rho=1.0, level=0.95)
                lo = float(np.asarray(m.sensitivity_params["theta"]["lower"]).ravel()[0])
                hi = float(np.asarray(m.sensitivity_params["theta"]["upper"]).ravel()[0])
                sign_robust = (lo > 0 and hi > 0) if theta > 0 else (lo < 0 and hi < 0)
                row[f"bench_{bench_name}_cf_y"] = cf_y
                row[f"bench_{bench_name}_cf_d"] = cf_d
                row[f"bench_{bench_name}_rho"] = rho_b
                row[f"bench_{bench_name}_theta_lo"] = lo
                row[f"bench_{bench_name}_theta_hi"] = hi
                row[f"bench_{bench_name}_sign_robust"] = sign_robust

        rows.append(row)
        print(f"  [{outcome:10s}] {tr:30s} theta={theta:+.4f} se={se:.4f} "
              f"RV={rv:.3f} RVa={rva:.3f} "
              f"{'(benchmarked)' if required else ''} "
              f"({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "exp2_sensitivity.csv", index=False)
print(f"[exp2] total {time.time()-t_start:.0f}s")
