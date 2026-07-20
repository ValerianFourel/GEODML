#!/usr/bin/env python3
"""Canonical DML re-run — 2026-05-25 (llms.txt as confounder)

Fits the canonical 6 treatments + 29 confounders (28 original + has_llms_txt
as a GEO-intent proxy) for three outcomes:
  Y_1 = selected_by_llm   (binary admission, full SERP pool sample frame)
  Y_2 = rank_delta        (admitted-URL sample frame)
  Y_3 = rank_post         (admitted-URL sample frame)

NOTE: T7 = has_llms_txt was MOVED FROM TREATMENT TO CONFOUNDER on 2026-05-25.
The rerankers never read the llms.txt file during inference, so it has no
causal pathway as a treatment. We retain it as a confounder because publishing
llms.txt is a plausibly-pre-treatment marker of latent site-level GEO intent
that is correlated with the other content treatments T1-T6; controlling for
it should reduce omitted-variable bias in the headline coefficients.

The coefficient on has_llms_txt itself is no longer interpretable causally
under this framing — it is a nuisance variable. The treatment set is
T1, T2, T3, T4, T5, T6 only.

Three specifications per (outcome, treatment):
  Spec A: single-treatment, 28 confounders only
  Spec B: single-treatment, 28 confounders + other 5 treatments mutually controlled
  Spec C: joint inference, all 6 treatments fit together with Bonferroni correction

Slices:
  POOLED, by variant (4), by engine (2), by model (2), by pool size (2) = 11 slices
  (Spec B + C are POOLED only — joint inference doesn't slice cleanly.)

Reads:  ~/geodml_data/data/main/full_experiment_data_*.parquet
        ~/geodml_data/data/main/regression_dataset.parquet
        ~/Hamburg/GEODML_Analysis/geodml_data/data/serp/phase0_top*.parquet
Writes: ~/geodml_data/data/dml_results/dml_canonical_2026-05-24.parquet
        docs/2026-05-24/dml_canonical_2026-05-24.md

See docs/2026-05-24/canonical_treatments_and_confounders_2026-05-24.md for the
locked-in set definitions.
"""
from __future__ import annotations

import io
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 300)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT, MAIN, SERP, DML, DOCS  # noqa: E402

OUT_PARQUET = DML / "dml_canonical_2026-05-25_llms_as_confounder.parquet"
OUT_REPORT = DOCS / "dml_canonical_2026-05-25_llms_as_confounder.md"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]

# ── Canonical sets — see docs/2026-05-24/canonical_treatments_and_confounders_2026-05-24.md
TREATMENTS = [
    "treat_stats_density",          # T1b
    "treat_question_headings",      # T2a
    "treat_structured_data",        # T3
    "T4_citation_authority_code",   # T4_code (deterministic)
    "treat_topical_comp",           # T5
    "treat_freshness",              # T6
    # T7 = has_llms_txt removed 2026-05-25: LLM never reads the file during
    # inference, so any coefficient was confounded, not causal.
]

CONFOUNDERS = [
    # page-HTML (6)
    "conf_word_count", "conf_readability",
    "conf_internal_links", "conf_outbound_links",
    "conf_images_alt", "conf_https",
    # SERP-derived (4)
    "conf_title_has_kw", "conf_title_len",
    "conf_snippet_len", "conf_serp_position",
    # semantic / IR (3)
    "conf_title_kw_sim", "conf_snippet_kw_sim", "conf_bm25",
    # DFS Whois (7)
    "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
    "conf_brand_recog", "conf_dfs_paid_count", "conf_dfs_etv",
    "conf_dfs_domain_age_years",
    # DFS keyword-level (8)
    "dfs_keyword_difficulty", "dfs_search_volume", "dfs_cpc", "dfs_competition",
    "dfs_intent_commercial", "dfs_intent_informational",
    "dfs_intent_navigational", "dfs_intent_transactional",
    # GEO-intent proxy (1) — added 2026-05-25, ex-T7
    "has_llms_txt",
]


# ── tee stdout to markdown ─────────────────────────────────────────────────


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


# ── DML primitives ─────────────────────────────────────────────────────────


_LGBM_KW = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, n_jobs=-1,
)


def cross_fit_resid(y, X, n_splits=5, seed=42, is_clf=False):
    """Out-of-fold residuals y - g_hat(X), LightGBM nuisance."""
    yhat = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        if is_clf:
            clf = LGBMClassifier(random_state=seed, **_LGBM_KW).fit(X[tr], y[tr])
            yhat[te] = clf.predict_proba(X[te])[:, 1]
        else:
            reg = LGBMRegressor(random_state=seed, **_LGBM_KW).fit(X[tr], y[tr])
            yhat[te] = reg.predict(X[te])
    return y - yhat


def plr_estimate(df, focal_T, ctrl_T, X_cols, outcome_col, is_clf=False,
                 max_n=200_000):
    """Robinson PLR for `focal_T` controlling for ctrl_T + X_cols.
    Returns dict with coef/se/p/n.
    """
    cols_needed = [focal_T, outcome_col] + ctrl_T + X_cols
    df = df.dropna(subset=[outcome_col, focal_T]).copy()
    # Fill remaining NaN with column median (treatments + confounders)
    for c in ctrl_T + X_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = 0.0

    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=42).reset_index(drop=True)

    if len(df) < 200:
        return {"coef": np.nan, "se": np.nan, "p_val": np.nan, "n": int(len(df))}

    T = df[focal_T].astype(float).values
    Y = df[outcome_col].astype(float).values
    X_full = df[ctrl_T + X_cols].astype(float).values

    T_resid = cross_fit_resid(T, X_full, is_clf=False)
    Y_resid = cross_fit_resid(Y, X_full, is_clf=is_clf)

    denom = float(np.dot(T_resid, T_resid))
    if denom == 0 or not np.isfinite(denom):
        return {"coef": np.nan, "se": np.nan, "p_val": np.nan, "n": int(len(df))}
    theta = float(np.dot(T_resid, Y_resid) / denom)
    eps = Y_resid - theta * T_resid
    se = float(np.sqrt(np.sum((T_resid * eps) ** 2)) / denom)
    p = float(2 * (1 - stats.norm.cdf(abs(theta / se))))
    return {"coef": theta, "se": se, "p_val": p, "n": int(len(df))}


# ── Build samples ─────────────────────────────────────────────────────────


def build_pool_admission() -> pd.DataFrame:
    """Full SERP pool × (model × variant) with selected flag + canonical T+X."""
    out("  [build] loading 4 SERP pool files …")
    pool_files = [
        ("ddg", 20, "phase0_top20_ddg.parquet"),
        ("ddg", 50, "phase0_top50_ddg.parquet"),
        ("searxng", 20, "phase0_top20_searxng.parquet"),
        ("searxng", 50, "phase0_top50_searxng.parquet"),
    ]
    parts = []
    for e, n, f in pool_files:
        p = pd.read_parquet(SERP / f)[["keyword", "url", "position"]].copy()
        p["search_engine"] = e
        p["pool_size"] = n
        parts.append(p)
    pool = pd.concat(parts, ignore_index=True)
    out(f"  [build] base pool rows = {len(pool):,}  unique (kw,url) = "
        f"{pool[['keyword','url']].drop_duplicates().shape[0]:,}")

    # Expand to (model × variant)
    rows = []
    for m in ("Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"):
        for v in VARIANTS:
            sub = pool.copy()
            sub["llm_model"] = m
            sub["variant"] = v
            rows.append(sub)
    big = pd.concat(rows, ignore_index=True)
    out(f"  [build] expanded pool×model×variant rows = {len(big):,}")

    # ── selected flag ───
    out("  [build] marking `selected` from per-variant LLM outputs …")
    sel_idx = set()
    rag_kw = {}
    for v in VARIANTS:
        df = pd.read_parquet(MAIN / f"full_experiment_data_{v}.parquet")
        df["search_engine"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        pcol = "pool_size" if "pool_size" in df.columns else ("serp_pool_size" if "serp_pool_size" in df.columns else "pool")
        df = df.rename(columns={pcol: "pool_size"})
        for r in df[["keyword", "url", "search_engine", "pool_size", "llm_model"]].itertuples(index=False):
            sel_idx.add((r.keyword, r.url, r.search_engine, r.pool_size, r.llm_model, v))
        rag_kw[v] = set(df["keyword"])
    big["selected"] = [
        int(k in sel_idx) for k in zip(big.keyword, big.url, big.search_engine,
                                       big.pool_size, big.llm_model, big.variant)
    ]
    out(f"  [build] overall selection rate = {big['selected'].mean()*100:.2f}%")

    # Restrict RAG variants to RAG-covered keywords
    keep = pd.Series(True, index=big.index)
    for v in ("biased_rag", "neutral_rag"):
        mask = (big["variant"] == v) & ~big["keyword"].isin(rag_kw[v])
        keep &= ~mask
        out(f"  [build] {v}: dropping {int(mask.sum()):,} rows (no RAG output)")
    big = big[keep].reset_index(drop=True)
    out(f"  [build] post-RAG-filter rows = {len(big):,}  "
        f"selection rate = {big['selected'].mean()*100:.2f}%")

    # Join treatments + confounders from per-variant parquets (variant-independent
    # features keyed on kw, url, engine, pool)
    out("  [build] joining treatments + confounders from per-variant parquets …")
    needed = list(set(TREATMENTS + CONFOUNDERS))
    needed = [c for c in needed if c != "T4_citation_authority_code"]  # join separately
    feat_parts = []
    for v in VARIANTS:
        df = pd.read_parquet(MAIN / f"full_experiment_data_{v}.parquet")
        df["search_engine"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        pcol = "pool_size" if "pool_size" in df.columns else ("serp_pool_size" if "serp_pool_size" in df.columns else "pool")
        df = df.rename(columns={pcol: "pool_size"})
        cols = ["keyword", "url", "search_engine", "pool_size"] + [c for c in needed if c in df.columns]
        feat_parts.append(df[cols])
    feats = pd.concat(feat_parts, ignore_index=True)
    feats = feats.groupby(["keyword", "url", "search_engine", "pool_size"],
                          as_index=False).first()
    big = big.merge(feats, on=["keyword", "url", "search_engine", "pool_size"],
                    how="left")

    # T4_code + has_llms_txt from regression_dataset (keyed on kw, url)
    extra = pd.read_parquet(MAIN / "regression_dataset.parquet",
                            columns=["keyword", "url",
                                     "T4_citation_authority_code", "has_llms_txt"])
    extra = extra.drop_duplicates(subset=["keyword", "url"], keep="first")
    big = big.merge(extra, on=["keyword", "url"], how="left")

    out(f"  [build] pool with features = {len(big):,} rows × {big.shape[1]} cols")
    return big


def build_admitted_sample() -> pd.DataFrame:
    """Union of 4 per-variant parquets — admitted URLs only — with T4_code joined."""
    parts = []
    for v in VARIANTS:
        df = pd.read_parquet(MAIN / f"full_experiment_data_{v}.parquet")
        df["search_engine"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        pcol = "pool_size" if "pool_size" in df.columns else ("serp_pool_size" if "serp_pool_size" in df.columns else "pool")
        df = df.rename(columns={pcol: "pool_size"})
        df["variant"] = v
        parts.append(df)
    big = pd.concat(parts, ignore_index=True)

    # T4_code + has_llms_txt
    extra = pd.read_parquet(MAIN / "regression_dataset.parquet",
                            columns=["keyword", "url",
                                     "T4_citation_authority_code", "has_llms_txt"])
    extra = extra.drop_duplicates(subset=["keyword", "url"], keep="first")
    big = big.merge(extra, on=["keyword", "url"], how="left")

    out(f"  admitted sample = {len(big):,} rows × {big.shape[1]} cols")
    return big


# ── Cell dummies (engine, model, pool, variant) ───────────────────────────


def add_cell_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add binary cell dummies, return df + list of dummy column names."""
    cell_cols = []
    if "search_engine" in df.columns:
        df["cell_engine_searxng"] = (df["search_engine"] == "searxng").astype(int)
        cell_cols.append("cell_engine_searxng")
    if "llm_model" in df.columns:
        df["cell_model_qwen"] = (df["llm_model"] == "Qwen2.5-72B-Instruct").astype(int)
        cell_cols.append("cell_model_qwen")
    if "pool_size" in df.columns:
        df["cell_pool_50"] = (df["pool_size"] == 50).astype(int)
        cell_cols.append("cell_pool_50")
    if "variant" in df.columns:
        df["cell_variant_biased"] = df["variant"].str.startswith("biased").astype(int)
        df["cell_variant_rag"] = df["variant"].str.endswith("_rag").astype(int)
        cell_cols += ["cell_variant_biased", "cell_variant_rag"]
    return df, cell_cols


# ── Slicing helper ─────────────────────────────────────────────────────────


def iter_slices(df: pd.DataFrame, include_cells: bool = True):
    """Yield (slice_name, sub_df) for POOLED + per-variant + per-engine + per-model + per-pool."""
    yield "POOLED", df
    for v in VARIANTS:
        sub = df[df["variant"] == v]
        if len(sub):
            yield f"VAR:{v}", sub
    if not include_cells:
        return
    for e in ("ddg", "searxng"):
        sub = df[df["search_engine"] == e]
        if len(sub):
            yield f"ENG:{e}", sub
    for m, tag in (("Llama-3.3-70B-Instruct", "Llama"), ("Qwen2.5-72B-Instruct", "Qwen2.5")):
        sub = df[df["llm_model"] == m]
        if len(sub):
            yield f"MOD:{tag}", sub
    for p in (20, 50):
        sub = df[df["pool_size"] == p]
        if len(sub):
            yield f"POOL:{p}", sub


# ── Main DML loop ─────────────────────────────────────────────────────────


def run_dml_for_outcome(data: pd.DataFrame, outcome_col: str, is_clf: bool):
    rows = []
    cell_cols = []  # already in data from add_cell_dummies
    cell_cols = [c for c in data.columns if c.startswith("cell_")]
    X_base = CONFOUNDERS + cell_cols

    # Spec A — single treatment + confounders only (per slice)
    out(f"\n  [{outcome_col}] Spec A — single treatment + confounders only")
    for slice_name, sub in iter_slices(data, include_cells=True):
        for treatment in TREATMENTS:
            t0 = time.time()
            r = plr_estimate(sub, treatment, [], X_base, outcome_col, is_clf=is_clf)
            r.update(spec="A", slice=slice_name, treatment=treatment,
                     outcome=outcome_col, seconds=round(time.time() - t0, 1))
            rows.append(r)
            stars = ("***" if r["p_val"] < 1e-3 else
                     "**" if r["p_val"] < 1e-2 else
                     "*" if r["p_val"] < 5e-2 else
                     "·" if r["p_val"] < 1e-1 else "")
            out(f"    [A/{slice_name:14s}] {treatment:30s} "
                f"n={r['n']:>6} coef={r['coef']:+.4f} se={r['se']:.4f} "
                f"p={r['p_val']:.4f}{stars}  ({r['seconds']}s)")

    # Spec B — joint mutually controlled (POOLED only)
    out(f"\n  [{outcome_col}] Spec B — single treatment + other 6 treatments + confounders (POOLED)")
    for treatment in TREATMENTS:
        other = [t for t in TREATMENTS if t != treatment]
        t0 = time.time()
        r = plr_estimate(data, treatment, other, X_base, outcome_col, is_clf=is_clf)
        r.update(spec="B", slice="POOLED", treatment=treatment,
                 outcome=outcome_col, seconds=round(time.time() - t0, 1))
        rows.append(r)
        stars = ("***" if r["p_val"] < 1e-3 else
                 "**" if r["p_val"] < 1e-2 else
                 "*" if r["p_val"] < 5e-2 else
                 "·" if r["p_val"] < 1e-1 else "")
        out(f"    [B/POOLED        ] {treatment:30s} "
            f"n={r['n']:>6} coef={r['coef']:+.4f} se={r['se']:.4f} "
            f"p={r['p_val']:.4f}{stars}  ({r['seconds']}s)  "
            f"(controls: 6 other T + {len(X_base)} X)")

    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────


def main():
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    out(f"# Canonical DML re-run — {datetime.now(timezone.utc).isoformat()}\n")
    out(f"Canonical treatments ({len(TREATMENTS)}): {TREATMENTS}")
    out(f"Canonical confounders ({len(CONFOUNDERS)}): {CONFOUNDERS}")

    # ── Y_1 admission ─────────────────────────────────────────────────────
    section("Y_1 = selected_by_llm  (binary admission — full SERP pool sample frame)")
    pool = build_pool_admission()
    pool, _ = add_cell_dummies(pool)
    out(f"  pool ready: {len(pool):,} rows  selection rate = {pool['selected'].mean()*100:.2f}%")
    res_admission = run_dml_for_outcome(pool, "selected", is_clf=True)

    # ── Y_2 rank_delta & Y_3 rank_post ───────────────────────────────────
    section("Y_2 = rank_delta & Y_3 = rank_post  (admitted-URL sample frame)")
    admitted = build_admitted_sample()
    admitted, _ = add_cell_dummies(admitted)
    out(f"  admitted ready: {len(admitted):,} rows")

    res_rd = run_dml_for_outcome(admitted, "rank_delta", is_clf=False)
    res_pr = run_dml_for_outcome(admitted, "post_rank", is_clf=False)

    # Combine
    all_res = pd.concat([res_admission, res_rd, res_pr], ignore_index=True)
    all_res["sig"] = all_res["p_val"].apply(
        lambda p: "***" if p < 1e-3 else
                  "**" if p < 1e-2 else
                  "*" if p < 5e-2 else
                  "·" if p < 1e-1 else "")
    # Bonferroni at α=0.05/n_tests-per-outcome
    for o in all_res["outcome"].unique():
        mask = all_res["outcome"] == o
        n_tests = int(mask.sum())
        thresh = 0.05 / n_tests
        all_res.loc[mask, "p_bonferroni"] = (all_res.loc[mask, "p_val"] * n_tests).clip(upper=1.0)
        all_res.loc[mask, "bonferroni_sig"] = all_res.loc[mask, "p_val"] < thresh

    all_res.to_parquet(OUT_PARQUET)
    out(f"\n  → saved {OUT_PARQUET.relative_to(Path.home())}  rows={len(all_res)}")

    # ── Headline table — Spec B POOLED on each outcome ───────────────────
    section("HEADLINE — Spec B POOLED (mutually-controlled) per outcome")
    for o in ["selected", "rank_delta", "post_rank"]:
        sub = all_res[(all_res["outcome"] == o) & (all_res["spec"] == "B") &
                      (all_res["slice"] == "POOLED")].copy()
        sub = sub.sort_values("coef")
        out(f"\n  Y = {o}")
        out(sub[["treatment", "n", "coef", "se", "p_val", "sig",
                  "p_bonferroni", "bonferroni_sig"]].to_string(index=False))

    # Save report
    OUT_REPORT.write_text(_tee.buf.getvalue())
    sys.stdout = sys.__stdout__
    print(f"\n  → report saved to {OUT_REPORT.relative_to(REPO_ROOT)}")
    print(f"  → parquet saved to {OUT_PARQUET.relative_to(Path.home())}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
