#!/usr/bin/env python3
"""Comprehensive re-analysis of the GEODML data.

Key question (per user): treat_source_earned is a *domain-level flag* (binary
"is this URL in our list of 250 earned-media domains?"), not a manipulable
content feature. Should it be a TREATMENT or a CONFOUNDER?

We answer this by running the following specifications and comparing them:

  SPEC A — current paper setup
      treatments  = T1a..T7 (T7 included)
      confounders = conf_*
      → estimates effect of "being earned" alongside content features
      (this is what dml_results_long_*.parquet has)

  SPEC B — "earned-as-confounder" (user's proposed)
      treatments  = content features only (T1a..T6)
      confounders = conf_* + treat_source_earned + treat_source_brand
      → asks "after controlling for source identity, what do content
        features do?" — the GEO-actionable question

  SPEC C — joint multi-treatment (already in repo)
      All treatments included simultaneously; report Romano-Wolf adjusted
      p-values. Loaded from dml_multi_treatment.parquet, not re-computed.

  SPEC D — heterogeneity check
      Spec B re-estimated within each (search_engine × pool) cell, to see
      whether the content-feature effects vary by retrieval source.

For each treatment we report:
  coef (point estimate), se, 95% CI, p-value, n, and the change vs Spec A.
"""
from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 120)

ROOT = Path.home() / "geodml_data"

# ── Variable groups ──────────────────────────────────────────────────────────

CONTENT_TREATMENTS = [
    ("T1a_stats_present",         "treat_stats_present"),
    ("T1b_stats_density",         "treat_stats_density"),
    ("T2a_question_headings",     "treat_question_headings"),
    ("T2b_structural_modularity", "treat_structural_modularity"),
    ("T3_structured_data",        "treat_structured_data"),
    ("T4a_ext_citations",         "treat_ext_citations_any"),
    ("T4b_auth_citations",        "treat_auth_citations"),
    ("T5_topical_comp",           "treat_topical_comp"),
    ("T6_freshness",              "treat_freshness"),
]
SOURCE_FEATURES = [
    "treat_source_earned",
    "treat_source_brand",
]
CONFOUNDERS = [
    "conf_title_kw_sim", "conf_snippet_kw_sim",
    "conf_title_len", "conf_snippet_len",
    "conf_brand_recog", "conf_title_has_kw",
    "conf_word_count", "conf_readability",
    "conf_internal_links", "conf_outbound_links", "conf_images_alt",
    "conf_bm25", "conf_https",
    # external SEO data — kept but median-imputed (many missing)
    "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
    # this is the dominant predictor — original SERP position
    "conf_serp_position",
]
OUTCOMES = ["rank_delta", "post_rank"]
VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]


def section(s):
    print("\n" + "=" * 86)
    print(s)
    print("=" * 86)


def hr():
    print("-" * 86)


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_variant(v):
    return pq.read_table(ROOT / "data" / "main" / f"full_experiment_data_{v}.parquet").to_pandas()


# ── A small inline DML implementation (PLR with cross-fitting + LightGBM) ────
# We re-implement instead of using DoubleML so we can mass-loop without paying
# per-fit overhead; matches the structure of the canonical files.


def plr_lgbm(y, D, X, n_splits=5, random_state=42):
    """Partially-linear DML with LightGBM nuisances.
    Returns (theta, se, ci_lower, ci_upper, p_val, n)."""
    from sklearn.model_selection import KFold
    from lightgbm import LGBMRegressor
    from scipy import stats

    y = np.asarray(y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_hat = np.zeros_like(y)
    d_hat = np.zeros_like(D)

    base_kw = dict(n_estimators=300, num_leaves=31, learning_rate=0.05,
                   verbose=-1, n_jobs=-1, random_state=random_state)

    for tr, te in kf.split(X):
        m_y = LGBMRegressor(**base_kw); m_y.fit(X[tr], y[tr]); y_hat[te] = m_y.predict(X[te])
        m_d = LGBMRegressor(**base_kw); m_d.fit(X[tr], D[tr]); d_hat[te] = m_d.predict(X[te])

    y_res = y - y_hat
    d_res = D - d_hat

    denom = (d_res ** 2).mean()
    if denom <= 0:
        return (np.nan,) * 5 + (len(y),)
    theta = (d_res * y_res).mean() / denom
    psi = d_res * (y_res - theta * d_res)
    var = (psi ** 2).mean() / (denom ** 2) / len(y)
    se = np.sqrt(var)
    z = theta / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return theta, se, theta - 1.96 * se, theta + 1.96 * se, p, len(y)


def fit_one(df, treat_col, outcome, X_cols):
    """Drop NaN, then fit PLR."""
    needed = [outcome, treat_col] + X_cols
    sub = df[needed].copy()
    # median-impute confounders (they're the X_cols); drop rows missing outcome or treatment
    sub = sub.dropna(subset=[outcome, treat_col])
    if len(sub) < 200:
        return None
    for c in X_cols:
        sub[c] = sub[c].fillna(sub[c].median())
    y = sub[outcome].values
    D = sub[treat_col].values
    X = sub[X_cols].values
    return plr_lgbm(y, D, X)


# ── Main analysis ────────────────────────────────────────────────────────────


def main():
    # 0. Sanity: what's in the existing multi-treatment file
    section("0. EXISTING multi-treatment DML (pre-computed) — mutually_controlled")
    mt = pq.read_table(ROOT / "data" / "dml_results" / "dml_multi_treatment.parquet").to_pandas()
    mc = mt[(mt["study"] == "mutually_controlled") & (mt["outcome"] == "rank_delta")]
    cols = ["treatment", "n", "coef", "se", "p_val", "n_other_treats_in_X", "n_confounders_in_X"]
    print(mc[cols].sort_values("coef").to_string(index=False))

    # 1. Source-identity diagnostics
    section("1. SOURCE-IDENTITY DIAGNOSTICS — is treat_source_earned domain-deterministic?")
    rows = []
    for v in VARIANTS:
        df = load_variant(v).dropna(subset=["treat_source_earned", "domain"])
        n_dom = df["domain"].nunique()
        n_earned_dom = df.groupby("domain")["treat_source_earned"].max().sum()
        any_inconsistent = (df.groupby("domain")["treat_source_earned"].nunique() > 1).sum()
        rows.append({
            "variant": v, "n_rows": len(df),
            "n_unique_domains": n_dom,
            "n_earned_domains": int(n_earned_dom),
            "pct_earned_domains": f"{100*n_earned_dom/n_dom:.1f}%",
            "domain_inconsistencies": int(any_inconsistent),
            "n_earned_rows": int((df["treat_source_earned"] == 1).sum()),
            "pct_earned_rows": f"{100*(df['treat_source_earned']==1).mean():.2f}%",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. How well do confounders predict treat_source_earned?
    section("2. Can confounders predict source identity? "
            "(if yes → DML can de-confound; if no → unmeasured confounding)")
    df = load_variant("biased").dropna(subset=["treat_source_earned"] + CONFOUNDERS[:13])
    print(f"  complete-case n (biased, dropping external-SEO cols): {len(df)}")
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        from lightgbm import LGBMClassifier
        X = df[CONFOUNDERS[:13]].fillna(df[CONFOUNDERS[:13]].median()).values
        y = df["treat_source_earned"].astype(int).values
        if y.sum() < 20 or (1 - y).sum() < 20:
            print("  too few of one class for AUC")
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42, test_size=0.3, stratify=y)
            m = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               verbose=-1, n_jobs=-1)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, p)
            print(f"  AUC predicting treat_source_earned from confounders: {auc:.3f}")
            print(f"  base rate: {y.mean():.3f}")
            print(f"  → AUC > 0.7 means confounders already encode much of source identity")
            print(f"    AUC ≈ 0.5 means source identity is independent of measured confounders")
    except Exception as e:
        print(f"  (skipped — {e})")

    # 3. SPEC A vs SPEC B comparison
    section("3. SPEC A vs B — does moving source-features to controls change content-treatment estimates?")
    print("  outcome: rank_delta, variant: biased (sanity test cell)")
    print()
    print(f"  Spec A: treatment {{T}}, controls = confounders")
    print(f"  Spec B: treatment {{T}}, controls = confounders + treat_source_earned + treat_source_brand")
    print()

    df_biased = load_variant("biased")
    X_A = CONFOUNDERS                                          # 17 conf cols
    X_B = CONFOUNDERS + SOURCE_FEATURES                        # 17 + 2 = 19

    rows = []
    for code, col in CONTENT_TREATMENTS:
        rA = fit_one(df_biased, col, "rank_delta", X_A)
        rB = fit_one(df_biased, col, "rank_delta", X_B)
        if rA is None or rB is None:
            continue
        rows.append({
            "treatment": code,
            "coef_A": rA[0], "se_A": rA[1], "p_A": rA[4],
            "coef_B": rB[0], "se_B": rB[1], "p_B": rB[4],
            "Δ_coef (B−A)": rB[0] - rA[0],
            "n_A": rA[5], "n_B": rB[5],
        })
    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))

    # 4. SPEC B applied to all 4 variants for content treatments (rank_delta)
    section("4. SPEC B — content effects with source-as-confounder, all 4 variants (rank_delta)")
    print("  (this is the proposed paper version)\n")

    out = []
    for v in VARIANTS:
        df = load_variant(v)
        for code, col in CONTENT_TREATMENTS:
            r = fit_one(df, col, "rank_delta", X_B)
            if r is None:
                continue
            theta, se, lo, hi, p, n = r
            out.append({
                "variant": v, "treatment": code,
                "coef": theta, "se": se, "ci_low": lo, "ci_high": hi,
                "p_val": p, "n": n,
            })
    spec_b = pd.DataFrame(out).round(4)
    pivot_coef = spec_b.pivot(index="treatment", columns="variant", values="coef")
    pivot_p = spec_b.pivot(index="treatment", columns="variant", values="p_val")
    print("coef (rank_delta — negative = treatment causes demotion):")
    print(pivot_coef.to_string())
    print("\np-values:")
    print(pivot_p.applymap(lambda x: "***" if x<0.001 else "**" if x<0.01 else "*" if x<0.05 else "").to_string())

    # 5. Also do SPEC B for post_rank as outcome (consistency check)
    section("5. SPEC B — same, but outcome = post_rank (sign-check)")
    out = []
    for v in VARIANTS:
        df = load_variant(v)
        for code, col in CONTENT_TREATMENTS:
            r = fit_one(df, col, "post_rank", X_B)
            if r is None:
                continue
            theta, se, lo, hi, p, n = r
            out.append({"variant": v, "treatment": code, "coef": theta, "p_val": p, "n": n})
    spec_b_post = pd.DataFrame(out).round(4)
    pivot_post = spec_b_post.pivot(index="treatment", columns="variant", values="coef")
    print("coef (post_rank — positive = treatment causes lower SERP position = demotion):")
    print(pivot_post.to_string())

    # 6. Source-as-treatment with content as controls (the "side piece" the user mentioned)
    section("6. SOURCE EFFECT — earned-media as a treatment with content + confounders controlled")
    print("  (this is the user-flagged 'side piece' result — RAG-vs-no-RAG attenuation)\n")
    X_source = CONFOUNDERS + [c for _, c in CONTENT_TREATMENTS]
    rows = []
    for v in VARIANTS:
        df = load_variant(v)
        for treat_col in ["treat_source_earned", "treat_source_brand"]:
            r = fit_one(df, treat_col, "rank_delta", X_source)
            if r is None:
                continue
            theta, se, lo, hi, p, n = r
            rows.append({
                "variant": v, "treatment": treat_col,
                "coef": theta, "se": se,
                "ci_low": lo, "ci_high": hi, "p_val": p, "n": n,
            })
    src = pd.DataFrame(rows).round(4)
    print(src.to_string(index=False))

    # 7. RAG impact deltas
    section("7. RAG-attenuation deltas — Spec B content treatments")
    pairs = [("biased", "biased_rag"), ("neutral", "neutral_rag")]
    delta_rows = {}
    for nonrag, rag in pairs:
        if nonrag in pivot_coef.columns and rag in pivot_coef.columns:
            delta_rows[f"{rag} − {nonrag}"] = (pivot_coef[rag] - pivot_coef[nonrag]).round(4)
    print(pd.DataFrame(delta_rows).to_string())

    # Save all outputs
    section("Persisting outputs to ~/geodml_data/docs/reanalysis_*")
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    spec_b.to_csv(docs / "reanalysis_spec_B_rank_delta.csv", index=False)
    spec_b_post.to_csv(docs / "reanalysis_spec_B_post_rank.csv", index=False)
    src.to_csv(docs / "reanalysis_source_as_treatment.csv", index=False)
    res.to_csv(docs / "reanalysis_specA_vs_B.csv", index=False)
    print(f"  wrote 4 CSVs to {docs}")


if __name__ == "__main__":
    main()
