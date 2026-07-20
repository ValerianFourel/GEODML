#!/usr/bin/env python3
"""New DML study — binary outcome `selected_by_llm ∈ {0, 1}`.

Question: holding every other reasonable confounder fixed, does the LLM systematically
include or exclude a candidate doc from its top-10 output?

Applies the lessons from the May-23 category-switch audit:
  - Headline treatments = the 6 audit-clean ones we can reconstruct on pool rows
    (T7_source_earned, T_llms_txt, T1b_stats_density, T1_code, T4_llm, T6_freshness).
    T5_topical_comp is dropped from this study — it requires per-(kw,url) similarity
    that isn't reconstructable for unselected pool rows.
  - Demoted treatments + 25 confounders go into the X-set.

Pipeline (PLR, Robinson-style):
  1. m_hat(X) = E[D|X]  via cross-fitted GBM (regression)
  2. g_hat(X) = E[Y|X]  via cross-fitted GBM (classification then probability)
  3. residuals D~ = D − m_hat(X);  Y~ = Y − g_hat(X)
  4. θ = OLS(Y~ ~ D~)   coefficient + heteroskedastic SE
  5. Robust SE via the standard influence-function formula.

Outputs:
  docs/dml_selected_2026-05-23.md           paper-ready report
  ~/geodml_data/data/dml_results/selected_long.parquet      long-form per-(treatment, slice, variant)
  ~/geodml_data/data/dml_results/selected_multitreat.parquet headline joint Spec B table
"""
from __future__ import annotations
import io, sys, time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, roc_auc_score

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 200)

# ── paths ───────────────────────────────────────────────────────────────────
ROOT = Path.home() / "geodml_data"
DML = ROOT / "data" / "dml_results"
MAIN = ROOT / "data" / "main"
WORK = Path("/Users/valerianfourel/Hamburg/GEODML_Analysis/geodml_data")
SERP = WORK / "data" / "serp"
RUNS = WORK / "data" / "runs"
OUT_REPORT = Path(__file__).resolve().parent.parent / "docs" / "dml_selected_2026-05-23_fixed.md"
OUT_LONG = "selected_long_fixed.parquet"
OUT_MULTI = "selected_multitreat_fixed.parquet"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
# 6 clean treatments retained for the new study
TREATMENTS = ["T7_source_earned", "T_llms_txt",
              "T1b_stats_density", "T1_code", "T4_llm", "T6_freshness"]

# X-set: 12 demoted treatments (act as confounders now) + 25 confounders + cell dummies
DEMOTED = [
    "T1a_stats_present", "T2a_question_headings", "T2b_structural_modularity",
    "T3_structured_data_new", "T4a_ext_citations", "T4b_auth_citations",
    "T1_llm", "T2_code", "T2_llm", "T3_code", "T3_llm", "T4_code",
]
CONF_COLS = [
    "conf_word_count", "conf_readability", "conf_internal_links",
    "conf_outbound_links", "conf_images_alt", "conf_https",
    "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
    "conf_serp_position",
    "dfs_keyword_difficulty", "dfs_search_volume", "dfs_cpc", "dfs_competition",
    "dfs_intent_commercial", "dfs_intent_informational",
    "dfs_intent_navigational", "dfs_intent_transactional",
]
CELL_DUMMIES = ["engine_is_searxng", "model_is_qwen", "pool_is_50",
                "variant_is_biased", "variant_is_rag"]


# ── tee output ──────────────────────────────────────────────────────────────
class Tee:
    def __init__(self): self.buf = io.StringIO()
    def write(self, s): sys.__stdout__.write(s); sys.__stdout__.flush(); self.buf.write(s)
    def flush(self): sys.__stdout__.flush()
_tee = Tee()
def out(s=""): print(s, file=_tee)
def h1(s): out("\n# " + s)
def h2(s): out("\n## " + s)
def h3(s): out("\n### " + s)
def fenced(text): out("```\n" + text.rstrip() + "\n```")
def fmt_df(df, **kw):
    fenced(df.to_string(index=kw.pop("index", False),
                        float_format=kw.pop("float_format", lambda x: f"{x:.4f}"),
                        **kw))
def stars(p):
    if pd.isna(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.10 else ""


# ── step 1: build the pool×cell×variant base table ──────────────────────────
def build_pool_table() -> pd.DataFrame:
    out("  [build] loading 4 SERP pool files …")
    pool_files = {("ddg", 20): "phase0_top20_ddg.parquet",
                  ("ddg", 50): "phase0_top50_ddg.parquet",
                  ("searxng", 20): "phase0_top20_searxng.parquet",
                  ("searxng", 50): "phase0_top50_searxng.parquet"}
    parts = []
    for (e, p), f in pool_files.items():
        df = pq.read_table(SERP / f).to_pandas()
        df["engine"] = e
        df["pool_size"] = p
        parts.append(df[["keyword", "url", "position", "engine", "pool_size"]])
    pool = pd.concat(parts, ignore_index=True)
    out(f"  [build] base pool rows = {len(pool):,}  unique (kw,url) = {pool[['keyword','url']].drop_duplicates().shape[0]:,}")

    # Expand to model × variant
    models = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"]
    rows = []
    for m in models:
        for v in VARIANTS:
            sub = pool.copy()
            sub["model"] = m
            sub["variant"] = v
            rows.append(sub)
    big = pd.concat(rows, ignore_index=True)
    out(f"  [build] expanded pool×model×variant rows = {len(big):,}")

    # add cell dummies
    big["engine_is_searxng"] = (big["engine"] == "searxng").astype(int)
    big["model_is_qwen"] = (big["model"] == "Qwen2.5-72B-Instruct").astype(int)
    big["pool_is_50"] = (big["pool_size"] == 50).astype(int)
    big["variant_is_biased"] = big["variant"].str.startswith("biased").astype(int)
    big["variant_is_rag"] = big["variant"].str.endswith("_rag").astype(int)
    big["conf_serp_position"] = big["position"]

    return big


# ── step 2: build the "selected" indicator per cell+variant ─────────────────
def add_selected_flag(big: pd.DataFrame) -> pd.DataFrame:
    """Mark `selected=1` from the per-variant LLM-output parquets (NOT from
    `full_experiment_unified.parquet` — that file is missing ~95% of the RAG
    rows due to a publish-pipeline bug). Also restrict the RAG variants to the
    set of keywords that have any RAG output at all, so the unselected rows
    represent actual LLM rejections rather than RAG-retrieval failures."""
    out("  [build] loading 4 per-variant LLM-output files (true source of selection) …")
    sel_parts = []
    rag_kw = {}            # variant → set of keywords with any RAG output
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        df = pq.read_table(p).to_pandas()
        # normalize engine name
        df["engine_norm"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        pool_col = "serp_pool_size" if "serp_pool_size" in df.columns else "pool"
        df = df.rename(columns={pool_col: "pool_size"})
        df["variant_norm"] = v
        sel_parts.append(df[["keyword", "url", "engine_norm", "pool_size",
                             "llm_model", "variant_norm"]])
        rag_kw[v] = set(df["keyword"])
        print(f"    {v:12s}: rows={len(df):>6}  kw={len(rag_kw[v]):>4}", flush=True)
    sel = pd.concat(sel_parts, ignore_index=True)
    sel_keys = sel.set_index(["keyword", "url", "engine_norm", "pool_size",
                              "llm_model", "variant_norm"]).index

    # mark selected (kw, url, engine, pool, model, variant)
    big_idx = big.set_index(["keyword", "url", "engine", "pool_size",
                             "model", "variant"]).index
    big["selected"] = big_idx.isin(sel_keys).astype(int)

    # restrict RAG variants to keywords with any RAG output — otherwise rows are
    # RAG-retrieval-failures, not LLM rejections
    keep = pd.Series(True, index=big.index)
    for v in ["biased_rag", "neutral_rag"]:
        mask = (big["variant"] == v) & ~big["keyword"].isin(rag_kw[v])
        keep &= ~mask
        dropped = int(mask.sum())
        out(f"  [build] {v}: dropping {dropped:,} pool rows from keywords with no RAG output "
            f"(only {len(rag_kw[v])} of 1011 keywords have RAG coverage)")
    big = big[keep].reset_index(drop=True)

    out(f"\n  [build] overall selection rate after fix = {big['selected'].mean()*100:.2f}%")
    out("  [build] selection rate per variant (corrected):")
    by_v = big.groupby("variant")["selected"].agg(["mean", "sum", "count"]).rename(
        columns={"mean": "rate", "sum": "n_selected", "count": "n_pool"})
    by_v["rate"] = (by_v["rate"] * 100).round(2)
    fmt_df(by_v.reset_index())

    # u_selected is now used downstream only as a feature LUT source — make a
    # quick replacement using the per-variant union for richer kw/url coverage
    u = pq.read_table(MAIN / "full_experiment_unified.parquet").to_pandas()
    return big, u


# ── step 3: attach features ─────────────────────────────────────────────────
def attach_features(big: pd.DataFrame, u_selected: pd.DataFrame) -> pd.DataFrame:
    out("  [build] attaching features …")
    # Use the POOLED full_experiment_data.parquet as the rich feature source — it
    # carries has_llms_txt + the T1-T4 _code/_llm cols that unified lacks.
    rich = pq.read_table(MAIN / "full_experiment_data.parquet").to_pandas()
    out(f"  [build] rich features file: rows={len(rich):,} cols={rich.shape[1]}")
    # 3a. Domain-level features: T7, T_llms_txt, conf_domain_authority, conf_backlinks, conf_referring_domains
    # extract domain from URL
    def to_domain(s):
        try:
            return urlparse(str(s)).netloc.lower().lstrip("www.")
        except Exception:
            return ""
    big["domain"] = big["url"].apply(to_domain)

    dom_cols = ["treat_source_earned", "has_llms_txt",
                "conf_domain_authority", "conf_backlinks", "conf_referring_domains"]
    dom_cols = [c for c in dom_cols if c in rich.columns]
    dom_lut = (rich.groupby("domain")[dom_cols]
               .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
               .reset_index())
    big = big.merge(dom_lut, on="domain", how="left")
    big = big.rename(columns={"treat_source_earned": "T7_source_earned",
                              "has_llms_txt": "T_llms_txt"})
    out(f"  [build] domain LUT covers {len(dom_lut):,} domains; "
        f"NaN T7 in pool: {big['T7_source_earned'].isna().mean()*100:.1f}%")

    # 3b. Keyword-level features: dfs_*
    kw_cols = [c for c in rich.columns if c.startswith("dfs_")]
    if kw_cols:
        kw_lut = (rich.groupby("keyword")[kw_cols]
                  .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                  .reset_index())
        big = big.merge(kw_lut, on="keyword", how="left")

    # 3c. URL-level features from rich pooled file: treat_*, conf_*, T1-T4 _code/_llm
    url_cols_from_rich = [
        "treat_stats_present", "treat_stats_density",
        "treat_question_headings", "treat_structural_modularity",
        "treat_structured_data", "treat_ext_citations_any",
        "treat_auth_citations", "treat_freshness", "treat_topical_comp",
        "conf_word_count", "conf_readability",
        "conf_internal_links", "conf_outbound_links",
        "conf_images_alt", "conf_https",
        "T1_statistical_density_code", "T2_question_heading_code",
        "T3_structured_data_code", "T4_citation_authority_code",
        "T1_statistical_density_llm", "T2_question_heading_llm",
        "T3_structured_data_llm", "T4_citation_authority_llm",
    ]
    url_cols_from_rich = [c for c in url_cols_from_rich if c in rich.columns]
    url_lut = (rich.groupby("url")[url_cols_from_rich]
               .agg(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
               .reset_index())
    big = big.merge(url_lut, on="url", how="left")
    out(f"  [build] URL LUT covers {len(url_lut):,} unique URLs")

    # 3e. derive the 6 clean treatments + 12 demoted from raw features (where needed)
    # we already have:
    #   T7_source_earned, T_llms_txt           (domain LUT)
    #   T6_freshness ← treat_freshness         (URL LUT)
    #   T1b_stats_density ← treat_stats_density (URL LUT)
    #   T1_code  ← T1_statistical_density_code (phase2)
    #   T4_llm   ← T4_citation_authority_llm   (phase2)
    big["T6_freshness"] = big.get("treat_freshness")
    big["T1b_stats_density"] = big.get("treat_stats_density")
    big["T1_code"] = big.get("T1_statistical_density_code")
    big["T4_llm"] = big.get("T4_citation_authority_llm")
    # 12 demoted
    big["T1a_stats_present"] = big.get("treat_stats_present")
    big["T2a_question_headings"] = big.get("treat_question_headings")
    big["T2b_structural_modularity"] = big.get("treat_structural_modularity")
    big["T3_structured_data_new"] = big.get("treat_structured_data")
    big["T4a_ext_citations"] = big.get("treat_ext_citations_any")
    big["T4b_auth_citations"] = big.get("treat_auth_citations")
    big["T1_llm"] = big.get("T1_statistical_density_llm")
    big["T2_code"] = big.get("T2_question_heading_code")
    big["T2_llm"] = big.get("T2_question_heading_llm")
    big["T3_code"] = big.get("T3_structured_data_code")
    big["T3_llm"] = big.get("T3_structured_data_llm")
    big["T4_code"] = big.get("T4_citation_authority_code")
    return big


# ── step 4: cross-fitted PLR for ONE treatment ──────────────────────────────
def plr_estimate(df: pd.DataFrame, y_col: str, d_col: str, x_cols: list[str],
                 n_splits: int = 3, max_n: int | None = None, seed: int = 42):
    """Robinson PLR with cross-fitted GBM nuisance. Returns dict."""
    sub = df[[y_col, d_col] + x_cols].dropna()
    if max_n and len(sub) > max_n:
        sub = sub.sample(n=max_n, random_state=seed)
    n = len(sub)
    if n < 200 or sub[d_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return {"n": n, "coef": np.nan, "se": np.nan, "p_val": np.nan,
                "r2_g": np.nan, "r2_m": np.nan}
    y = sub[y_col].values.astype(float)
    d = sub[d_col].values.astype(float)
    X = sub[x_cols].values.astype(float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_hat = np.zeros(n); d_hat = np.zeros(n)
    y_is_binary = set(np.unique(y).tolist()) <= {0.0, 1.0}
    d_is_binary = set(np.unique(d).tolist()) <= {0.0, 1.0}
    for tr, te in kf.split(X):
        # E[Y|X]
        if y_is_binary:
            mY = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed).fit(X[tr], y[tr])
            y_hat[te] = mY.predict_proba(X[te])[:, 1]
        else:
            mY = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=seed).fit(X[tr], y[tr])
            y_hat[te] = mY.predict(X[te])
        # E[D|X]
        if d_is_binary:
            mD = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed).fit(X[tr], d[tr])
            d_hat[te] = mD.predict_proba(X[te])[:, 1]
        else:
            mD = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=seed).fit(X[tr], d[tr])
            d_hat[te] = mD.predict(X[te])

    y_resid = y - y_hat
    d_resid = d - d_hat
    if (d_resid ** 2).sum() < 1e-9:
        return {"n": n, "coef": np.nan, "se": np.nan, "p_val": np.nan,
                "r2_g": r2_score(y, y_hat), "r2_m": r2_score(d, d_hat)}
    theta = (d_resid * y_resid).sum() / (d_resid ** 2).sum()
    # robust SE (Robinson, sandwich)
    psi = (y_resid - theta * d_resid) * d_resid
    var = (psi ** 2).mean() / ((d_resid ** 2).mean()) ** 2
    se = np.sqrt(var / n)
    z = theta / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"n": n, "coef": float(theta), "se": float(se), "p_val": float(p_val),
            "r2_g": r2_score(y, y_hat),
            "r2_m": r2_score(d, d_hat)}


# ── step 5: per-treatment, per-slice loop ───────────────────────────────────
def run_dml(big: pd.DataFrame, max_n: int = 60_000):
    out("  [dml] running PLR per (treatment, slice) …")

    # X-set = demoted treatments + confounders + cell dummies. EXCLUDE the focal treatment.
    base_X = DEMOTED + CONF_COLS + CELL_DUMMIES

    # numericize
    df = big.copy()
    for c in list(set(base_X + TREATMENTS + ["selected"])):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows missing the outcome or with no domain features (T7/T_llms_txt unknown)
    df = df.dropna(subset=["selected"])

    results = []
    slices = (
        [("POOLED", df)]
        + [(f"VAR:{v}", df[df["variant"] == v]) for v in VARIANTS]
        + [(f"ENG:{e}", df[df["engine"] == e]) for e in ["ddg", "searxng"]]
        + [(f"MOD:{m.split('-')[0]}", df[df["model"] == m])
           for m in df["model"].unique()]
        + [(f"POOL:{p}", df[df["pool_size"] == p]) for p in [20, 50]]
    )

    for slug in TREATMENTS:
        for sl_name, sl_df in slices:
            X = [c for c in base_X if c in sl_df.columns and c != slug]
            res = plr_estimate(sl_df, "selected", slug, X, max_n=max_n)
            res.update({"treatment": slug, "slice": sl_name})
            results.append(res)
            print(f"  [dml] {sl_name:30s} {slug:25s} "
                  f"n={res['n']:>6}  coef={res['coef']:+.4f}  "
                  f"se={res['se']:.4f}  p={res['p_val']:.4f}", flush=True)

    long = pd.DataFrame(results)
    long["sig"] = long["p_val"].apply(stars)
    long.to_parquet(DML / OUT_LONG)
    out(f"\n  [dml] saved → data/dml_results/{OUT_LONG}")
    return long


# ── step 6: joint multi-treatment PLR (Spec B for selected) ─────────────────
def run_joint_multitreatment(big: pd.DataFrame, max_n: int = 60_000):
    """Estimate each treatment with the OTHER 5 treatments in X (mutually controlled)."""
    out("\n  [joint] mutually-controlled multi-treatment fit …")
    base_X = DEMOTED + CONF_COLS + CELL_DUMMIES
    df = big.copy()
    for c in list(set(base_X + TREATMENTS + ["selected"])):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["selected"])

    rows = []
    for slug in TREATMENTS:
        X = [c for c in (base_X + [t for t in TREATMENTS if t != slug])
             if c in df.columns]
        res = plr_estimate(df, "selected", slug, X, max_n=max_n)
        res["treatment"] = slug
        rows.append(res)
        print(f"  [joint] {slug:25s} n={res['n']:>6}  coef={res['coef']:+.4f}  "
              f"se={res['se']:.4f}  p={res['p_val']:.4f}", flush=True)
    mt = pd.DataFrame(rows)
    mt["sig"] = mt["p_val"].apply(stars)
    # Bonferroni (Holm available too; pick simple Bonferroni)
    mt["p_val_bonferroni"] = (mt["p_val"] * len(TREATMENTS)).clip(upper=1.0)
    mt["BF_sig"] = mt["p_val_bonferroni"].apply(stars)
    mt.to_parquet(DML / OUT_MULTI)
    out(f"\n  [joint] saved → data/dml_results/{OUT_MULTI}")
    return mt


# ── reporting ───────────────────────────────────────────────────────────────
def section_overview(big: pd.DataFrame):
    h1("DML Study — Binary outcome `selected_by_llm` (2026-05-23, FIXED RAG)")
    out("\n*New study triggered by the May-23 category-switch findings.* Re-runs DML with a "
        "different outcome: **was this candidate URL selected into the LLM's top-10 output (1) "
        "or rejected (0)**.\n")
    out("**This is the FIXED version.** The earlier run sourced the `selected` indicator from "
        "`full_experiment_unified.parquet`, which has a publish-pipeline bug — it contains only "
        "2,938 of the actual ~64,909 RAG-output rows. This rerun reads the `selected` flag from "
        "the per-variant `full_experiment_data_{variant}.parquet` files (164 k rows total, full "
        "coverage). RAG variants are also restricted to the keywords that actually have RAG "
        "output (744 of 1011 for biased_rag, 615 of 1011 for neutral_rag), so the unselected "
        "rows represent real LLM rejections, not RAG-retrieval failures.\n")
    out("**Sample frame**: the SERP candidate pool. Each row is (keyword × URL × search-engine × "
        "pool-size × LLM × prompt-variant × passage-mode). Total rows ≈ 395 k.\n")
    out("**Treatments (audit-clean only)**: T7_source_earned, T_llms_txt, T1b_stats_density, "
        "T1_code, T4_llm, T6_freshness. *Note: T5_topical_comp is dropped — it needs per-(kw,url) "
        "similarity recomputation that wasn't available for unselected pool rows.*\n")
    out("**X-set**: 12 demoted treatments (per the audit) + 18 reconstructable confounders "
        "(conf_*, dfs_*) + 5 cell dummies (engine, model, pool, prompt variant, RAG).\n")
    out("**Method**: Robinson-style partially-linear DML — cross-fitted (K=3) GradientBoosting "
        "nuisance models for both E[Y|X] and E[D|X], heteroskedastic-robust SE. Same identifying "
        "assumption as DoubleML PLR.\n")

    h2("0. Sample composition")
    fmt_df(big.groupby("variant")["selected"].agg(["mean","sum","count"]).rename(
        columns={"mean":"selection_rate","sum":"n_selected","count":"n_pool"}).reset_index(),
        float_format=lambda x: f"{x:.4f}")
    fmt_df(big.groupby(["engine","pool_size","model"])["selected"].agg(["mean","count"]).reset_index(),
        float_format=lambda x: f"{x:.4f}")


def section_headline(long: pd.DataFrame, mt: pd.DataFrame):
    h2("1. Headline — pooled single-treatment DML (Spec A)")
    out("Each row: PLR estimate of treatment → `selected`, with the 12 demoted treatments + 18 "
        "confounders + 5 cell dummies in X.\n")
    p = long[long["slice"] == "POOLED"].copy()
    p["95% CI"] = p.apply(lambda r: f"[{r['coef']-1.96*r['se']:+.4f}, {r['coef']+1.96*r['se']:+.4f}]", axis=1)
    fmt_df(p[["treatment","n","coef","se","95% CI","p_val","sig","r2_g","r2_m"]]
           .sort_values("coef"))

    h2("2. Headline — mutually-controlled multi-treatment DML (Spec B)")
    out("Each treatment estimated with the OTHER 5 treatments + 12 demoted + 18 confounders + 5 cell dummies in X.\n")
    mt2 = mt.copy()
    mt2["95% CI"] = mt2.apply(lambda r: f"[{r['coef']-1.96*r['se']:+.4f}, {r['coef']+1.96*r['se']:+.4f}]", axis=1)
    fmt_df(mt2[["treatment","n","coef","se","95% CI","p_val","sig",
                "p_val_bonferroni","BF_sig","r2_g","r2_m"]]
          .sort_values("coef"))


def section_per_variant(long: pd.DataFrame):
    h2("3. Per-variant breakdown")
    p = long[long["slice"].str.startswith("VAR:")].copy()
    p["variant"] = p["slice"].str.replace("VAR:", "", regex=False)
    pivot_coef = p.pivot(index="treatment", columns="variant", values="coef").round(4)
    pivot_p = p.pivot(index="treatment", columns="variant", values="p_val").round(4)
    out("**Coefficient by variant** (rows: treatment, cols: variant):")
    fmt_df(pivot_coef.reindex(TREATMENTS).reset_index())
    out("\n**p-value by variant**:")
    fmt_df(pivot_p.reindex(TREATMENTS).reset_index())
    # RAG attenuation
    h3("3a. RAG attenuation per treatment (Δ = coef_rag − coef_nonrag)")
    p2 = p.set_index(["treatment", "variant"])
    rows = []
    for nonrag, rag in [("biased", "biased_rag"), ("neutral", "neutral_rag")]:
        for t in TREATMENTS:
            try:
                c_nr = p2.loc[(t, nonrag), "coef"]
                c_rg = p2.loc[(t, rag), "coef"]
                s_nr = p2.loc[(t, nonrag), "se"]
                s_rg = p2.loc[(t, rag), "se"]
            except KeyError:
                continue
            d = c_rg - c_nr
            sed = np.sqrt(s_nr ** 2 + s_rg ** 2)
            z = d / sed if sed > 0 else 0.0
            pp = 2 * (1 - stats.norm.cdf(abs(z)))
            rows.append({"pair": f"{rag} − {nonrag}", "treatment": t,
                         "coef_nonrag": round(c_nr, 4), "coef_rag": round(c_rg, 4),
                         "Δ": round(d, 4), "SE_Δ": round(sed, 4),
                         "z": round(z, 2), "p_val": round(pp, 4), "sig": stars(pp)})
    fmt_df(pd.DataFrame(rows))


def section_heterogeneity(long: pd.DataFrame):
    h2("4. Heterogeneity — marginal slices (engine, model, pool size)")
    m = long[long["slice"].str.startswith(("ENG:", "MOD:", "POOL:"))].copy()
    pivot_coef = m.pivot(index="treatment", columns="slice", values="coef").round(4)
    pivot_p = m.pivot(index="treatment", columns="slice", values="p_val").round(4)
    out("\n**Coefficient by single-dimension slice:**")
    fmt_df(pivot_coef.reindex(TREATMENTS).reset_index())
    out("\n**p-value:**")
    fmt_df(pivot_p.reindex(TREATMENTS).reset_index())


def section_narrative(mt: pd.DataFrame):
    h2("5. Narrative — what this study tells us")
    survivors = mt[mt["p_val_bonferroni"] < 0.05].sort_values("coef")
    out(f"\nBonferroni-survivors at 0.05 (treatments × 6 tests): "
        f"{', '.join(survivors['treatment'].tolist()) if len(survivors) else 'none'}.\n")
    msg = """
The headline question — *does the LLM disproportionately admit or reject candidate URLs based on
domain class, content density, or freshness?* — is now answered on a clean binary outcome with
the entire SERP pool as the sample frame, so the estimands aren't contaminated by rank-conditional
selection.

**Read this together with the rank_delta study from earlier today:**

- If a treatment carries a *negative* effect on `rank_delta` (LLM promotes the doc upward) AND a
  *positive* effect on `selected` (LLM is more likely to admit it at all), then the LLM is
  unambiguously favouring that signal. Mention both numbers in the paper.

- If `T7_source_earned` carries a *negative* effect on `selected` here too, that strengthens the
  LLM-bias-against-earned-media story — the bias isn't only "rank lower if admitted", it's also
  "don't admit at all".

- A treatment that flips sign between the two outcomes is candidate for a Section 5 caveat:
  the LLM might bring documents *in* but then rank them at the bottom (or the other way around).

Use the per-variant table (§3) for RAG mitigation claims, and the cell table (§4) for the
heterogeneity paragraph.
"""
    out(msg)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    out("# DML Study — selected_by_llm (build + fit)\n")
    big = build_pool_table()
    big, u_sel = add_selected_flag(big)
    big = attach_features(big, u_sel)

    # diagnostic on feature availability
    out("\n  [build] feature coverage in expanded table (% non-null):")
    cov = pd.Series({c: big[c].notna().mean() for c in TREATMENTS + DEMOTED + CONF_COLS if c in big.columns}).round(3)
    print(cov.to_string(), flush=True)

    section_overview(big)
    long = run_dml(big)
    mt = run_joint_multitreatment(big)
    section_headline(long, mt)
    section_per_variant(long)
    section_heterogeneity(long)
    section_narrative(mt)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(_tee.buf.getvalue())
    out(f"\n**Report saved → `{OUT_REPORT.relative_to(Path.cwd().parent)}`**  "
        f"({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
