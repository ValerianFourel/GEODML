#!/usr/bin/env python3
"""Compare code-only vs LLM-only DML specifications.

For each treatment family (T1 stats, T2 Q-headings, T3 schema, T4 citations),
two parallel codings exist:
  - "code" version  (T1_code, T2_code, T3_code, T4_code)
  - "LLM"  version  (T1_llm,  T2_llm,  T3_llm,  T4_llm)

We run joint mutually-controlled DML twice:
  1. CODE spec: {T1_code, T2_code, T3_code, T4_code, T5, T6, T_llms_txt} as the
     treatment set; the OTHER 6 treatments + 25 confounders go into the X-set.
  2. LLM  spec: {T1_llm,  T2_llm,  T3_llm,  T4_llm,  T5, T6, T_llms_txt} same idea.

Same outcomes (rank_delta and post_rank), same sample frame, same nuisance
learners — the only thing that differs is which coding lens you use for T1-T4.

T7_source_earned and T_llms_txt are kept in both specs as shared controls;
T7 is excluded from main figures elsewhere but reported alongside for context.

Writes:
  ~/geodml_data/data/dml_results/dml_code_vs_llm.parquet
  docs/dml_code_vs_llm_2026-05-24.md

Run:
  python scripts/dml_code_vs_llm.py
"""
from __future__ import annotations

import sys
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path.home() / "geodml_data"
MAIN = ROOT / "data" / "main"
DML = ROOT / "data" / "dml_results"
OUT_REPORT = REPO_ROOT / "docs" / "dml_code_vs_llm_2026-05-24.md"
OUT_PARQUET = DML / "dml_code_vs_llm.parquet"

# Shared treatments (no code/LLM distinction)
SHARED = ["T5_topical_comp", "T6_freshness", "T_llms_txt"]

CODE_FOCAL = ["T1_code", "T2_code", "T3_code", "T4_code"]
LLM_FOCAL = ["T1_llm", "T2_llm", "T3_llm", "T4_llm"]

# Mapping from treatment ID → column in regression_dataset.parquet
COL_MAP = {
    "T1_code": "T1_statistical_density_code",
    "T2_code": "T2_question_heading_code",
    "T3_code": "T3_structured_data_code",
    "T4_code": "T4_citation_authority_code",
    "T1_llm": "T1_statistical_density_llm",
    "T2_llm": "T2_question_heading_llm",
    "T3_llm": "T3_structured_data_llm",
    "T4_llm": "T4_citation_authority_llm",
    "T5_topical_comp": "treat_topical_comp",
    "T6_freshness": "treat_freshness",
    "T_llms_txt": "has_llms_txt",
    "T7_source_earned": "treat_source_earned",
}

CONFOUNDERS = [
    "conf_title_kw_sim", "conf_snippet_kw_sim", "conf_title_len",
    "conf_snippet_len", "conf_brand_recog", "conf_title_has_kw",
    "conf_word_count", "conf_readability", "conf_internal_links",
    "conf_outbound_links", "conf_images_alt", "conf_bm25",
    "conf_https", "conf_domain_authority", "conf_backlinks",
    "conf_referring_domains", "conf_serp_position",
]


# ── tee output to a markdown report ────────────────────────────────────────


class _Tee:
    def __init__(self):
        self.buf = io.StringIO()
    def write(self, s):
        sys.__stdout__.write(s); self.buf.write(s)
    def flush(self):
        sys.__stdout__.flush()


_tee = _Tee()
sys.stdout = _tee


def out(s=""):
    print(s, flush=True)


def section(s, char="="):
    out("\n" + char * 88)
    out(s)
    out(char * 88)


def fmt_df(df, **kw):
    out(df.to_string(index=False, **kw))


# ── load ───────────────────────────────────────────────────────────────────


def load_regression_dataset():
    df = pq.read_table(MAIN / "regression_dataset.parquet").to_pandas()
    out(f"  loaded regression_dataset.parquet  rows={len(df):,}  cols={df.shape[1]}")
    return df


# ── DML primitives (Robinson PLR with cross-fit GBM) ───────────────────────


def cross_fit_resid(y, X, n_splits=5, seed=0, is_clf=False):
    """Return out-of-fold residuals y - g_hat(X)."""
    yhat = np.full_like(y, np.nan, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        gbm = GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=seed, subsample=0.8
        ).fit(X[tr], y[tr])
        yhat[te] = gbm.predict(X[te])
    return y - yhat


def plr_estimate(df_in, focal_T, Tcols, Xcols, outcome_col):
    """Robinson PLR for treatment `focal_T` with the other `Tcols` + `Xcols`
    as controls. Returns dict with coef/se/p/etc."""
    df = df_in.dropna(subset=[outcome_col, focal_T]).copy()
    other = [t for t in Tcols if t != focal_T]
    X_full = df[other + Xcols].fillna(0).values.astype(float)

    T = df[focal_T].astype(float).values
    Y = df[outcome_col].astype(float).values

    # Residualise T and Y on X via 5-fold cross-fit
    T_resid = cross_fit_resid(T, X_full, n_splits=5, seed=42)
    Y_resid = cross_fit_resid(Y, X_full, n_splits=5, seed=42)

    # OLS on residuals
    denom = float(np.dot(T_resid, T_resid))
    if denom == 0 or not np.isfinite(denom):
        return {"coef": np.nan, "se": np.nan, "p_val": np.nan,
                "n": int(len(df))}
    theta = float(np.dot(T_resid, Y_resid) / denom)
    eps = Y_resid - theta * T_resid
    # Heteroskedasticity-robust SE
    se = float(np.sqrt(np.sum((T_resid * eps) ** 2)) / denom)
    p = float(2 * (1 - stats.norm.cdf(abs(theta / se))))
    return {"coef": theta, "se": se, "p_val": p, "n": int(len(df))}


def run_spec(df, spec_name, focal_set, outcomes):
    """Joint mutually-controlled DML for `focal_set ∪ SHARED` on each outcome."""
    Tids = focal_set + SHARED
    Tcols = [COL_MAP[t] for t in Tids]
    Xcols = [c for c in CONFOUNDERS if c in df.columns]
    out(f"\n  [{spec_name}] treatments = {Tids}  ({len(Tcols)} cols)")
    out(f"  [{spec_name}] confounders = {len(Xcols)} cols")

    # Pre-fill any NaN treatment cells with median so all rows used
    for c in Tcols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    rows = []
    for outcome in outcomes:
        outcome_col = "rank_delta" if outcome == "rank_delta" else "post_rank"
        for tid, tcol in zip(Tids, Tcols):
            t0 = time.time()
            r = plr_estimate(df, tcol, Tcols, Xcols, outcome_col)
            r["spec"] = spec_name
            r["treatment"] = tid
            r["treatment_col"] = tcol
            r["outcome"] = outcome
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            out(f"    [{spec_name}/{outcome}] {tid:25s} n={r['n']:>5d} "
                f"coef={r['coef']:+.4f} se={r['se']:.4f} p={r['p_val']:.4f}  "
                f"({r['seconds']}s)")
    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────


def main():
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    out("# Code-only vs LLM-only DML comparison\n")
    out(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    section("1. Load")
    df = load_regression_dataset()

    section("2. Run CODE-only DML")
    res_code = run_spec(df, "CODE", CODE_FOCAL,
                        outcomes=["rank_delta", "post_rank"])

    section("3. Run LLM-only DML")
    res_llm = run_spec(df, "LLM", LLM_FOCAL,
                       outcomes=["rank_delta", "post_rank"])

    res = pd.concat([res_code, res_llm], ignore_index=True)
    res["sig"] = res["p_val"].apply(
        lambda p: "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2
                  else "·" if p < 1e-1 else "")
    res.to_parquet(OUT_PARQUET)
    out(f"\n  → saved {OUT_PARQUET.relative_to(Path.home())}  rows={len(res)}")

    section("4. Side-by-side comparison (rank_delta)")
    rd = res[res["outcome"] == "rank_delta"].copy()
    # Pair code/llm versions of T1-T4
    pair_rows = []
    for fam in ["T1", "T2", "T3", "T4"]:
        c = rd[(rd["treatment"] == f"{fam}_code") & (rd["spec"] == "CODE")]
        l = rd[(rd["treatment"] == f"{fam}_llm")  & (rd["spec"] == "LLM")]
        if len(c) and len(l):
            pair_rows.append({
                "family": fam,
                "code_coef": c["coef"].iloc[0],
                "code_se": c["se"].iloc[0],
                "code_p": c["p_val"].iloc[0],
                "code_sig": c["sig"].iloc[0],
                "llm_coef": l["coef"].iloc[0],
                "llm_se": l["se"].iloc[0],
                "llm_p": l["p_val"].iloc[0],
                "llm_sig": l["sig"].iloc[0],
                "delta_coef": l["coef"].iloc[0] - c["coef"].iloc[0],
                "agree_sign": "yes" if c["coef"].iloc[0] * l["coef"].iloc[0] > 0 else "no",
            })
    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        out("\n  T1–T4 code vs LLM head-to-head on rank_delta:")
        fmt_df(pairs)

    section("5. Shared treatments — should give same number across specs")
    shared_rows = res[res["treatment"].isin(SHARED)].pivot_table(
        index=["treatment", "outcome"], columns="spec",
        values="coef", aggfunc="first"
    )
    out(shared_rows.to_string())

    # Save report
    OUT_REPORT.write_text(_tee.buf.getvalue())
    sys.stdout = sys.__stdout__
    print(f"\n  → report saved to {OUT_REPORT.relative_to(REPO_ROOT)}")
    print(f"  → parquet saved to {OUT_PARQUET.relative_to(Path.home())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
