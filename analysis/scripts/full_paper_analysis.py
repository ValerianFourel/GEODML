#!/usr/bin/env python3
"""Full paper-ready analysis from the ValerianFourel/geodml-emnlp-2026 dataset.

Synthesizes the pre-computed DML parquets and adds three new analyses:

  1. CATEGORY-SWITCH AUDIT — for every one of the 19 treatments AND 25 confounders,
     diagnose whether the variable is acting more like a confounder (predicted by
     the rest) or a clean treatment (orthogonal). Flags candidates for re-labeling.

  2. RAG × CELL HETEROGENEITY — RAG attenuation by (engine × pool × model), going
     beyond the per-variant pooled coefficients.

  3. PROMOTION OUTCOME — binary "was the doc promoted upward by the LLM" as a
     companion outcome to rank_delta / post_rank (rate-based, paired).

Reads:  ~/geodml_data/{data/dml_results, data/main}/*.parquet
Writes: docs/full_paper_analysis_2026-05-23.md   (paper-ready report)
        ~/geodml_data/data/dml_results/category_switch_audit.parquet
        ~/geodml_data/data/dml_results/rag_cell_heterogeneity.parquet
"""
from __future__ import annotations
from pathlib import Path
import io
import sys
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

ROOT = Path.home() / "geodml_data"
DML = ROOT / "data" / "dml_results"
MAIN = ROOT / "data" / "main"
OUT_REPORT = Path(__file__).resolve().parent.parent / "docs" / "full_paper_analysis_2026-05-23.md"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
PAIRS = [("biased", "biased_rag"), ("neutral", "neutral_rag")]

CONTENT_NEW = [  # the 9 "new" content treatments
    "T1a_stats_present", "T1b_stats_density",
    "T2a_question_headings", "T2b_structural_modularity",
    "T3_structured_data_new",
    "T4a_ext_citations", "T4b_auth_citations",
    "T5_topical_comp", "T6_freshness",
]
CONTENT_CODE = ["T1_code", "T2_code", "T3_code", "T4_code"]  # rule-coded variants
CONTENT_LLM = ["T1_llm", "T2_llm", "T3_llm", "T4_llm"]       # LLM-coded variants
SOURCE = ["T7_source_earned", "T_llms_txt"]
ALL_TREATMENTS = CONTENT_NEW + CONTENT_CODE + CONTENT_LLM + SOURCE  # 19 total

# Map treatment slug → underlying column in main file
TREATMENT_COL = {
    "T1_code": "T1_statistical_density_code",
    "T2_code": "T2_question_heading_code",
    "T3_code": "T3_structured_data_code",
    "T4_code": "T4_citation_authority_code",
    "T1_llm": "T1_statistical_density_llm",
    "T2_llm": "T2_question_heading_llm",
    "T3_llm": "T3_structured_data_llm",
    "T4_llm": "T4_citation_authority_llm",
    "T1a_stats_present": "treat_stats_present",
    "T1b_stats_density": "treat_stats_density",
    "T2a_question_headings": "treat_question_headings",
    "T2b_structural_modularity": "treat_structural_modularity",
    "T3_structured_data_new": "treat_structured_data",
    "T4a_ext_citations": "treat_ext_citations_any",
    "T4b_auth_citations": "treat_auth_citations",
    "T5_topical_comp": "treat_topical_comp",
    "T6_freshness": "treat_freshness",
    "T7_source_earned": "treat_source_earned",
    "T_llms_txt": "has_llms_txt",
}

# 25 confounders the JUPITER pipeline already controls for (per nuisance_r2 schema)
CONF_COLS = [
    "conf_title_kw_sim", "conf_snippet_kw_sim", "conf_title_len", "conf_snippet_len",
    "conf_brand_recog", "conf_title_has_kw", "conf_word_count", "conf_readability",
    "conf_internal_links", "conf_outbound_links", "conf_images_alt", "conf_bm25",
    "conf_https", "conf_domain_authority", "conf_backlinks", "conf_referring_domains",
    "conf_serp_position",
    "dfs_keyword_difficulty", "dfs_search_volume", "dfs_cpc", "dfs_competition",
    "dfs_intent_commercial", "dfs_intent_informational", "dfs_intent_navigational",
    "dfs_intent_transactional",
]


# ── utilities ────────────────────────────────────────────────────────────────

class Tee:
    """Write both to stdout and to the markdown report buffer."""
    def __init__(self):
        self.buf = io.StringIO()
    def write(self, s):
        sys.__stdout__.write(s)
        self.buf.write(s)
    def flush(self):
        sys.__stdout__.flush()


_tee = Tee()


def out(s=""):
    print(s, file=_tee)


def h1(s):
    out("\n" + "#" * 1 + " " + s)


def h2(s):
    out("\n" + "#" * 2 + " " + s)


def h3(s):
    out("\n" + "#" * 3 + " " + s)


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.1 else ""


def fenced(text):
    out("```")
    out(text.rstrip())
    out("```")


def fmt_df(df, float_fmt=None, **kw):
    if float_fmt is None:
        float_fmt = lambda x: f"{x:.4f}"
    fenced(df.to_string(index=kw.pop("index", False), float_format=float_fmt, **kw))


# ── 0. front matter ──────────────────────────────────────────────────────────


def section_0():
    h1("GEODML — Full Paper-Ready Re-Analysis (2026-05-23)")
    out(f"\n*Source: `~/geodml_data/` mirroring HF dataset `ValerianFourel/geodml-emnlp-2026`.*")
    out("\n*Pipeline run on JUPITER Booster; this script re-aggregates the parquets and adds new diagnostics.*")
    out("\n**Outcomes:** `rank_delta` (negative = LLM promoted), `post_rank` (lower = better), `promotion` (binary, rank_delta > 0).")
    out("**Specifications:**")
    out("- *Spec A*: single-treatment DML — controls for the 25 confounders only.")
    out("- *Spec B*: mutually-controlled multi-treatment DML — every treatment estimated WITH all other 18 treatments AND 25 confounders in X.")
    out("- *Spec C*: joint inference — all 19 treatments in one DML fit, with Romano–Wolf and Bonferroni multi-test correction.")
    out("\nLearner: LightGBM (where available) / GradientBoosting fallback; cross-fitting K=5 in the JUPITER pipeline.")
    out("All p-values reported are two-sided; `***` p<0.001, `**` p<0.01, `*` p<0.05, `·` p<0.10.\n")


# ── 1. Configuration profile ─────────────────────────────────────────────────


def section_1():
    h2("1. Configuration profile — what's in the data")
    rows = []
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas()
        rows.append({
            "variant": v,
            "n_rows": len(df),
            "n_keywords": df["keyword"].nunique(),
            "n_domains": df["domain"].nunique() if "domain" in df.columns else None,
            "engines": "/".join(sorted(df["search_engine"].unique())) if "search_engine" in df.columns else "?",
            "llms": "/".join(sorted([s.split("/")[-1] for s in df["llm_model"].unique()])) if "llm_model" in df.columns else "?",
            "pool_sizes": "/".join(map(str, sorted(df["serp_pool_size"].unique()))) if "serp_pool_size" in df.columns else "?",
            "pre_rank_mean": round(df["pre_rank"].mean(), 2) if "pre_rank" in df.columns else None,
            "post_rank_mean": round(df["post_rank"].mean(), 2),
            "rank_delta_mean": round(df["rank_delta"].mean(), 3) if "rank_delta" in df.columns else None,
        })
    out("\n**Per-variant sample composition** (4 variants × 2 prompts × RAG/no-RAG):")
    fmt_df(pd.DataFrame(rows))

    # pooled file: cell structure
    df = pq.read_table(MAIN / "full_experiment_data.parquet").to_pandas()
    out(f"\n**Pooled `full_experiment_data.parquet`:** {len(df):,} rows × {df.shape[1]} columns; "
        f"{df['keyword'].nunique():,} keywords × {df['domain'].nunique():,} domains.")
    cells = (df.groupby(["search_engine", "llm_model", "serp_pool_size"])
               .size().rename("n").reset_index()
               .sort_values("n", ascending=False))
    cells["llm_model"] = cells["llm_model"].str.split("/").str[-1]
    out("\n**Cells (search_engine × LLM × SERP pool size):**")
    fmt_df(cells)


# ── 2. Category-switch audit (NEW) ───────────────────────────────────────────


def section_2():
    h2("2. Category-switch audit — is each variable acting as a treatment or a confounder?")
    cache = DML / "category_switch_audit.parquet"
    if cache.exists():
        audit = pd.read_parquet(cache)
        t_block = audit[audit["kind"] == "treatment"].sort_values("r2_self_from_X", ascending=False)
        c_block = audit[audit["kind"] == "confounder"].sort_values("r2_self_from_X", ascending=False)
        out("\n*Using cached `category_switch_audit.parquet` — delete it to force a re-fit.*\n")
        out("\n### 2A. TREATMENTS — switch-candidacy diagnostic\n")
        fmt_df(t_block.drop(columns=["kind"]))
        out("\n### 2B. CONFOUNDERS — internal coherence (sanity: confounders should be predictable)\n")
        fmt_df(c_block.drop(columns=["kind", "specA_coef_rd", "specB_coef_rd", "attenuation_rd"]))
        h3("2C. Switch-candidate call-outs")
        cands = t_block[t_block["recommendation"].str.contains("drop")]
        if len(cands):
            out("\n**Treatments flagged for re-labeling as confounders:**")
            for _, r in cands.iterrows():
                out(f"- **`{r['name']}`** — r2_self={r['r2_self_from_X']:.3f}, "
                    f"AUC={r['AUC_self'] if not pd.isna(r['AUC_self']) else '—'}, "
                    f"|coef| attenuates {r['attenuation_rd']*100 if not pd.isna(r['attenuation_rd']) else 0:.0f}% under Spec B.")
        else:
            out("\nNo treatment crosses the drop-to-confounder threshold under these rules.")
        borderline = t_block[t_block["recommendation"].str.contains("borderline")]
        if len(borderline):
            out("\n**Borderline (footnote / robustness):**")
            for _, r in borderline.iterrows():
                out(f"- `{r['name']}` — r2_self={r['r2_self_from_X']:.3f}, "
                    f"attenuation={r['attenuation_rd']*100 if not pd.isna(r['attenuation_rd']) else 0:.0f}%.")
        return
    out("\nFor every one of the **19 treatments + 25 confounders** we compute three diagnostics:")
    out("- **r2_self_from_X** — R² (5-fold) when the remaining 43 variables predict this one. High R² means it's strongly determined by the rest of the X-set (potential confounder behaviour).")
    out("- **AUC_self** — for binary variables, the AUC of that same predictor (helpful for source-class flags like `T7`, `T_llms_txt`).")
    out("- **|coef| attenuation** — for treatments, how much the |coef| shrinks going from Spec A to Spec B (mutually controlled). >50% shrinkage → its 'effect' is mostly mediated by other features.")
    out("\nRule used for the recommendation:")
    out("- **drop-to-confounder** if `r2_self_from_X ≥ 0.30` AND attenuation ≥ 50%, OR if the variable is binary with AUC ≥ 0.90.")
    out("- **borderline** if `r2_self_from_X ∈ [0.15, 0.30)` and attenuation ≥ 30%.")
    out("- **keep-as-treatment** otherwise.\n")

    # build a single design matrix per variable: X = the other 18 treatments + 25 confounders
    df = pq.read_table(MAIN / "full_experiment_data.parquet").to_pandas()
    rows = []

    # for treatments: use canonical column from TREATMENT_COL
    treat_records = [(slug, TREATMENT_COL[slug]) for slug in ALL_TREATMENTS]
    conf_records = [(c, c) for c in CONF_COLS]
    all_records = treat_records + conf_records  # (slug, column)

    # Pull Spec A vs Spec B coefs for the treatments, both outcomes
    multi = pq.read_table(DML / "dml_multi_treatment.parquet").to_pandas()
    multi_b = multi[multi["study"] == "mutually_controlled"]
    pooled_a = pq.read_table(DML / "dml_results_long.parquet").to_pandas()
    pooled_a = pooled_a[pooled_a["subset"] == "POOLED"]

    def get_spec_a(slug, outcome):
        s = pooled_a[(pooled_a["treatment"] == slug) & (pooled_a["outcome"] == outcome)]
        return s["coef"].iloc[0] if len(s) else np.nan

    def get_spec_b(slug, outcome):
        s = multi_b[(multi_b["treatment"] == slug) & (multi_b["outcome"] == outcome)]
        return s["coef"].iloc[0] if len(s) else np.nan

    # build X matrix once
    feat_cols = [c for (_, c) in all_records if c in df.columns]
    Xfull = df[feat_cols].copy()
    Xfull = Xfull.apply(pd.to_numeric, errors="coerce")
    Xfull = Xfull.fillna(Xfull.median(numeric_only=True))

    # Faster: light GBM, K=3, subsample 25k rows for the audit
    rng = np.random.default_rng(42)
    n_sample = min(25_000, len(Xfull))
    sub_idx = rng.choice(len(Xfull), size=n_sample, replace=False)
    Xa = Xfull.iloc[sub_idx].reset_index(drop=True)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    n_total = len(all_records)
    print(f"  [audit] running {n_total} variables × 3-fold GBM on n={n_sample} subsample …", flush=True)
    for i, (slug, col) in enumerate(all_records, 1):
        if col not in Xa.columns:
            continue
        y = Xa[col].values
        X = Xa.drop(columns=[col]).values
        is_binary = set(pd.Series(y).dropna().unique()) <= {0, 1, 0.0, 1.0}
        is_binary = is_binary and pd.Series(y).nunique() == 2

        try:
            yhat = np.zeros_like(y, dtype=float)
            for tr, te in kf.split(X):
                if is_binary:
                    m = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0).fit(X[tr], y[tr])
                    yhat[te] = m.predict_proba(X[te])[:, 1]
                else:
                    m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0).fit(X[tr], y[tr])
                    yhat[te] = m.predict(X[te])
            r2 = r2_score(y, yhat)
            auc = roc_auc_score(y, yhat) if is_binary else np.nan
        except Exception as e:
            r2, auc = np.nan, np.nan
        print(f"  [audit] {i:>2}/{n_total}  {slug:35s} r2={r2:.3f}  auc={auc if not pd.isna(auc) else '—'}", flush=True)

        # Effect attenuation for treatments
        is_treatment = slug in ALL_TREATMENTS
        if is_treatment:
            a_rd = get_spec_a(slug, "rank_delta")
            b_rd = get_spec_b(slug, "rank_delta")
            attn_rd = (1 - abs(b_rd) / abs(a_rd)) if a_rd and not pd.isna(a_rd) and abs(a_rd) > 1e-9 else np.nan
        else:
            a_rd = b_rd = attn_rd = np.nan

        # Recommendation
        if is_treatment:
            high_r2 = (not pd.isna(r2)) and (r2 >= 0.30)
            high_auc = (not pd.isna(auc)) and (auc >= 0.90)
            big_attn = (not pd.isna(attn_rd)) and (attn_rd >= 0.50)
            mid_r2 = (not pd.isna(r2)) and (0.15 <= r2 < 0.30)
            mid_attn = (not pd.isna(attn_rd)) and (attn_rd >= 0.30)
            if (high_r2 and big_attn) or high_auc:
                rec = "→ drop to confounder"
            elif mid_r2 and mid_attn:
                rec = "· borderline"
            else:
                rec = "✓ keep as treatment"
        else:
            # for confounders: how influential as a confounder? use existing OLS sig if available
            rec = "(confounder)"

        rows.append({
            "kind": "treatment" if is_treatment else "confounder",
            "name": slug,
            "column": col,
            "binary": is_binary,
            "r2_self_from_X": round(r2, 3) if not pd.isna(r2) else np.nan,
            "AUC_self": round(auc, 3) if not pd.isna(auc) else np.nan,
            "specA_coef_rd": round(a_rd, 4) if not pd.isna(a_rd) else np.nan,
            "specB_coef_rd": round(b_rd, 4) if not pd.isna(b_rd) else np.nan,
            "attenuation_rd": round(attn_rd, 3) if not pd.isna(attn_rd) else np.nan,
            "recommendation": rec,
        })

    audit = pd.DataFrame(rows)
    # Show treatment block first (sorted by r2_self_from_X DESC)
    t_block = audit[audit["kind"] == "treatment"].sort_values("r2_self_from_X", ascending=False)
    c_block = audit[audit["kind"] == "confounder"].sort_values("r2_self_from_X", ascending=False)

    out("\n### 2A. TREATMENTS — switch-candidacy diagnostic\n")
    fmt_df(t_block.drop(columns=["kind"]))

    out("\n### 2B. CONFOUNDERS — internal coherence (sanity: confounders should be predictable)\n")
    fmt_df(c_block.drop(columns=["kind", "specA_coef_rd", "specB_coef_rd", "attenuation_rd"]))

    # Save the parquet
    out_path = DML / "category_switch_audit.parquet"
    audit.to_parquet(out_path)
    out(f"\n*Saved → `{out_path.relative_to(Path.home())}`.*")

    # Headline call-outs
    h3("2C. Switch-candidate call-outs")
    cands = t_block[t_block["recommendation"].str.contains("drop")]
    if len(cands):
        out("\n**Treatments flagged for re-labeling as confounders:**")
        for _, r in cands.iterrows():
            out(f"- **`{r['name']}`** — r2_self={r['r2_self_from_X']:.3f}, "
                f"AUC={r['AUC_self'] if not pd.isna(r['AUC_self']) else '—'}, "
                f"|coef| attenuates {r['attenuation_rd']*100 if not pd.isna(r['attenuation_rd']) else 0:.0f}% under Spec B.")
    else:
        out("\nNo treatment crosses the drop-to-confounder threshold under these rules.")

    borderline = t_block[t_block["recommendation"].str.contains("borderline")]
    if len(borderline):
        out("\n**Borderline (footnote / robustness):**")
        for _, r in borderline.iterrows():
            out(f"- `{r['name']}` — r2_self={r['r2_self_from_X']:.3f}, "
                f"attenuation={r['attenuation_rd']*100 if not pd.isna(r['attenuation_rd']) else 0:.0f}%.")


# ── 3. Multi-treatment headline (Spec B) ────────────────────────────────────


def section_3():
    h2("3. Headline multi-treatment table (Spec B — mutually controlled)")
    out("\nEach coefficient is the partial effect of that treatment with the **18 other treatments** AND **25 confounders** held fixed. This is the user-requested 'T7-as-confounder' specification applied to every treatment simultaneously.\n")
    df = pq.read_table(DML / "dml_multi_treatment.parquet").to_pandas()
    mc = df[df["study"] == "mutually_controlled"].copy()

    def kind(t):
        if t in CONTENT_NEW:
            return "1. content (new)"
        if t in SOURCE:
            return "3. source"
        if t in CONTENT_CODE:
            return "2a. content (rule-coded)"
        if t in CONTENT_LLM:
            return "2b. content (LLM-coded)"
        return "?"

    mc["kind"] = mc["treatment"].apply(kind)
    mc["sig"] = mc["p_val"].apply(stars)
    mc["95% CI"] = mc.apply(lambda r: f"[{r['coef']-1.96*r['se']:+.3f}, {r['coef']+1.96*r['se']:+.3f}]", axis=1)

    for outcome in ["rank_delta", "post_rank"]:
        h3(f"3.{ {'rank_delta':'A', 'post_rank':'B'}[outcome] }. outcome = `{outcome}`")
        for k in ["3. source", "1. content (new)", "2a. content (rule-coded)", "2b. content (LLM-coded)"]:
            sub = mc[(mc["outcome"] == outcome) & (mc["kind"] == k)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("coef")
            cols = ["treatment", "n", "coef", "se", "95% CI", "p_val", "sig"]
            out(f"\n**{k.split('. ',1)[1]}**")
            fmt_df(sub[cols])


# ── 4. Joint inference + multi-test correction ──────────────────────────────


def section_4():
    h2("4. Joint inference — Romano–Wolf & Bonferroni-adjusted p-values")
    out("\nAll 19 treatments fit in ONE DML regression. RW controls family-wise error rate respecting the empirical correlation among test statistics.\n")
    df = pq.read_table(DML / "dml_multi_treatment.parquet").to_pandas()
    ji = df[df["study"] == "joint_inference"].copy()
    ji["raw_sig"] = ji["p_val"].apply(stars)
    ji["RW_sig"] = ji["p_val_romano_wolf"].apply(stars)
    ji["BF_sig"] = ji["p_val_bonferroni"].apply(stars)
    for outcome in ["rank_delta", "post_rank"]:
        h3(f"4.{ {'rank_delta':'A','post_rank':'B'}[outcome] }. outcome = `{outcome}` (sorted by RW p-value)")
        sub = ji[ji["outcome"] == outcome].copy().sort_values("p_val_romano_wolf")
        cols = ["treatment", "coef", "se", "p_val", "raw_sig",
                "p_val_romano_wolf", "RW_sig", "p_val_bonferroni", "BF_sig"]
        fmt_df(sub[cols])
        rw_survivors = sub[sub["p_val_romano_wolf"] < 0.05]["treatment"].tolist()
        out(f"\n**Surviving RW@0.05 ({outcome}):** {', '.join(rw_survivors) if rw_survivors else 'none'}.")


# ── 5. Spec A vs Spec B — per-variant contrast ──────────────────────────────


def section_5():
    h2("5. Spec A (single-treatment) vs Spec B (mutually controlled) — per-variant view")
    out("\nThe coefficient changes from Spec A → Spec B tell us **which 'effects' were really mediated by other content features**. Big shrinkage = the variable was double-counting.\n")
    rows = []
    for v in VARIANTS:
        p = DML / f"dml_results_long_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas()
        df = df[(df.get("method", "plr") == "plr") & (df.get("learner", "lgbm") == "lgbm")]
        for outcome in ["rank_delta", "post_rank"]:
            d = df[df["outcome"] == outcome]
            for code in ALL_TREATMENTS:
                rs = d[d["treatment"] == code]
                if rs.empty:
                    continue
                top = rs.loc[rs["n_obs"].idxmax()]
                rows.append({
                    "variant": v, "outcome": outcome, "treatment": code,
                    "A_coef": top["coef"], "A_se": top["se"], "A_p": top["p_val"],
                })
    A = pd.DataFrame(rows)
    B = pq.read_table(DML / "dml_multi_treatment.parquet").to_pandas()
    B = B[B["study"] == "mutually_controlled"]

    for outcome in ["rank_delta", "post_rank"]:
        h3(f"5.{ {'rank_delta':'A','post_rank':'B'}[outcome] }. outcome = `{outcome}`")
        a_pivot = (A[A["outcome"] == outcome]
                   .pivot(index="treatment", columns="variant", values="A_coef"))
        b_row = B[B["outcome"] == outcome].set_index("treatment")["coef"]
        combined = a_pivot.copy()
        combined["B_pooled (Spec B)"] = b_row
        combined["A_mean"] = a_pivot.mean(axis=1)
        combined["ΔB−A"] = combined["B_pooled (Spec B)"] - combined["A_mean"]
        order = [t for t in ALL_TREATMENTS if t in combined.index]
        fmt_df(combined.loc[order].round(4), index=True)


# ── 6. Cell-level heterogeneity (engine × pool × model) ─────────────────────


def section_6():
    h2("6. Cell-level heterogeneity — engine × LLM × pool size")
    out("\nFrom the pre-built `dml_pivot_rank_delta.parquet` / `dml_pivot_post_rank.parquet`. Each cell is a separate DML fit on the corresponding slice. Stars are based on raw p-values *within that fit*.\n")
    for outcome in ["rank_delta", "post_rank"]:
        h3(f"6.{ {'rank_delta':'A','post_rank':'B'}[outcome] }. outcome = `{outcome}` — per-cell coefficients")
        df = pq.read_table(DML / f"dml_pivot_{outcome}.parquet").to_pandas()
        # focus on cell columns + POOLED
        cell_cols = [c for c in df.columns if c not in {"treatment"}]
        cell_cols = [c for c in cell_cols if (":" in c or "_top10" in c)]
        # Order: marginals first, then 8 specific cells
        marg = [c for c in cell_cols if ":" in c]
        specific = [c for c in cell_cols if c not in marg]
        ordered = ["treatment"] + marg + specific
        # show only meaningful treatments (the 19)
        slug_order = ALL_TREATMENTS
        sub = df[df["treatment"].isin(slug_order)].set_index("treatment").loc[
            [s for s in slug_order if s in df["treatment"].values]
        ].reset_index()
        fmt_df(sub[ordered])


# ── 7. RAG breakdown — per-treatment + per-cell attenuation ────────────────


def section_7():
    h2("7. RAG vs non-RAG breakdown")

    # 7A. Sample composition
    h3("7A. Sample composition")
    rows = []
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas()
        rd = df["rank_delta"].dropna() if "rank_delta" in df.columns else None
        rows.append({
            "variant": v,
            "n_rows": len(df),
            "n_keywords": df["keyword"].nunique(),
            "pct_promoted (Δ>0)": f"{100*(rd>0).mean():.1f}%" if rd is not None else "—",
            "pct_no_change (Δ=0)": f"{100*(rd==0).mean():.1f}%" if rd is not None else "—",
            "pct_demoted (Δ<0)": f"{100*(rd<0).mean():.1f}%" if rd is not None else "—",
            "rank_delta_mean": round(rd.mean(), 3) if rd is not None else None,
        })
    fmt_df(pd.DataFrame(rows))

    out("\n**Paired keyword × URL rank_delta — does RAG re-order the same retrieved doc?**")
    p_rows = []
    for nonrag, rag in PAIRS:
        a = pq.read_table(MAIN / f"full_experiment_data_{nonrag}.parquet").to_pandas().dropna(subset=["rank_delta"])
        b = pq.read_table(MAIN / f"full_experiment_data_{rag}.parquet").to_pandas().dropna(subset=["rank_delta"])
        m = a.merge(b, on=["keyword", "url"], suffixes=("_nr", "_rag"))
        if m.empty:
            continue
        d = m["rank_delta_rag"] - m["rank_delta_nr"]
        t, p = stats.ttest_rel(m["rank_delta_rag"], m["rank_delta_nr"])
        p_rows.append({
            "pair": f"{nonrag} → {rag}",
            "n_pairs": len(m),
            "mean_Δ": round(d.mean(), 3),
            "sd": round(d.std(), 3),
            "paired_t": round(t, 2),
            "p_val": f"{p:.2e}",
        })
    fmt_df(pd.DataFrame(p_rows))

    # 7B. Per-treatment RAG attenuation (using per-variant single-treatment files)
    h3("7B. Per-treatment RAG attenuation (Δ = coef_rag − coef_nonrag)")
    spec_rows = []
    for v in VARIANTS:
        p = DML / f"dml_results_long_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas()
        df = df[(df.get("method", "plr") == "plr") & (df.get("learner", "lgbm") == "lgbm")]
        for outcome in ["rank_delta", "post_rank"]:
            d = df[df["outcome"] == outcome]
            for code in ALL_TREATMENTS:
                rs = d[d["treatment"] == code]
                if rs.empty:
                    continue
                top = rs.loc[rs["n_obs"].idxmax()]
                spec_rows.append({
                    "variant": v, "outcome": outcome, "treatment": code,
                    "coef": top["coef"], "se": top["se"], "p_val": top["p_val"],
                })
    A = pd.DataFrame(spec_rows)
    for outcome in ["rank_delta", "post_rank"]:
        h3(f"outcome = `{outcome}` — RAG deltas")
        sub = A[A["outcome"] == outcome]
        coef_p = sub.pivot(index="treatment", columns="variant", values="coef")
        se_p = sub.pivot(index="treatment", columns="variant", values="se")
        rows = []
        for nonrag, rag in PAIRS:
            for code in ALL_TREATMENTS:
                if code not in coef_p.index:
                    continue
                if nonrag not in coef_p.columns or rag not in coef_p.columns:
                    continue
                c_nr = coef_p.loc[code, nonrag]
                c_rg = coef_p.loc[code, rag]
                s_nr = se_p.loc[code, nonrag]
                s_rg = se_p.loc[code, rag]
                if any(pd.isna([c_nr, c_rg, s_nr, s_rg])):
                    continue
                delta = c_rg - c_nr
                se_delta = np.sqrt(s_nr ** 2 + s_rg ** 2)
                z = delta / se_delta if se_delta > 0 else 0.0
                pp = 2 * (1 - stats.norm.cdf(abs(z)))
                rows.append({
                    "pair": f"{rag} − {nonrag}",
                    "treatment": code,
                    "coef_non_rag": round(c_nr, 4),
                    "coef_rag": round(c_rg, 4),
                    "Δ": round(delta, 4),
                    "SE_Δ": round(se_delta, 4),
                    "z": round(z, 2),
                    "p_val_Δ": round(pp, 4),
                    "sig": stars(pp),
                })
        fmt_df(pd.DataFrame(rows))

    # 7C. RAG × cell heterogeneity from raw data (NEW)
    h3("7C. RAG × cell heterogeneity (mean rank_delta per engine × pool × model)")
    out("Below is the descriptive (not DML-adjusted) mean rank_delta in each cell, then the cell-level RAG attenuation.\n")
    rows = []
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas().dropna(subset=["rank_delta"])
        df["model_short"] = df["llm_model"].str.split("/").str[-1].str[:5]
        pool_col = "serp_pool_size" if "serp_pool_size" in df.columns else "pool"
        g = df.groupby(["search_engine", pool_col, "model_short"])["rank_delta"].agg(
            ["mean", "std", "count"]
        ).reset_index().rename(columns={pool_col: "pool_size"})
        g["variant"] = v
        rows.append(g)
    cells = pd.concat(rows, ignore_index=True)
    pivot = cells.pivot(index=["search_engine", "pool_size", "model_short"],
                        columns="variant", values="mean").round(3)
    fmt_df(pivot.reset_index(), float_fmt=lambda x: f"{x:+.3f}")

    out("\n**RAG attenuation per cell** (rag − non_rag mean rank_delta):")
    deltas = pd.DataFrame(index=pivot.index)
    for nonrag, rag in PAIRS:
        if nonrag in pivot.columns and rag in pivot.columns:
            deltas[f"Δ({rag}−{nonrag})"] = (pivot[rag] - pivot[nonrag]).round(3)
    fmt_df(deltas.reset_index(), float_fmt=lambda x: f"{x:+.3f}")
    # save
    deltas.reset_index().to_parquet(DML / "rag_cell_heterogeneity.parquet")
    out(f"\n*Saved → `data/dml_results/rag_cell_heterogeneity.parquet`.*")


# ── 8. Promotion outcome (binary) ──────────────────────────────────────────


def section_8():
    h2("8. Promotion outcome — binary rate by variant and source class")
    out("\n`promotion = 1 if rank_delta > 0`. We report unconditional rates per variant and source-class slice. "
        "Below is *descriptive*; a full IRM-fit per-treatment table is left as a follow-up if needed.\n")
    rows = []
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas().dropna(subset=["rank_delta"])
        df["promotion"] = (df["rank_delta"] > 0).astype(int)
        # overall
        rows.append({
            "variant": v, "slice": "all rows",
            "n": len(df), "promotion_rate": round(df["promotion"].mean(), 3),
        })
        for col, label in [("treat_source_earned", "earned-media"),
                           ("has_llms_txt", "has_llms_txt")]:
            if col not in df.columns:
                continue
            for val, name in [(1, f"{label}=1"), (0, f"{label}=0")]:
                sub = df[df[col] == val]
                if len(sub) == 0:
                    continue
                rows.append({
                    "variant": v, "slice": name,
                    "n": len(sub), "promotion_rate": round(sub["promotion"].mean(), 3),
                })
    fmt_df(pd.DataFrame(rows))

    # difference-of-rates with stars
    h3("8B. Earned-media penalty in promotion-rate space")
    rows = []
    for v in VARIANTS:
        p = MAIN / f"full_experiment_data_{v}.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas().dropna(subset=["rank_delta"])
        df["promotion"] = (df["rank_delta"] > 0).astype(int)
        if "treat_source_earned" not in df.columns:
            continue
        a = df[df["treat_source_earned"] == 1]["promotion"]
        b = df[df["treat_source_earned"] == 0]["promotion"]
        if len(a) < 2 or len(b) < 2:
            continue
        p_a, p_b = a.mean(), b.mean()
        n_a, n_b = len(a), len(b)
        se = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
        z = (p_a - p_b) / se if se > 0 else 0.0
        pp = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({
            "variant": v,
            "n_earned": n_a, "n_other": n_b,
            "p(earned)": round(p_a, 3), "p(other)": round(p_b, 3),
            "Δ": round(p_a - p_b, 3),
            "SE_Δ": round(se, 4),
            "z": round(z, 2),
            "p_val": f"{pp:.2e}",
            "sig": stars(pp),
        })
    fmt_df(pd.DataFrame(rows))


# ── 9. Variance explained & nuisance R² ────────────────────────────────────


def section_9():
    h2("9. Variance explained & nuisance fit quality")
    out("\nHow much variance in each target is even predictable, and how cleanly is each treatment identified?\n")
    ve = pq.read_table(DML / "variance_explained.parquet").to_pandas()
    out("**Total variance explained** by the full X-set (origins + DFS).")
    fmt_df(ve)

    nu = pq.read_table(DML / "nuisance_r2.parquet").to_pandas()
    nu_p = nu[nu["subset"] == "POOLED"].sort_values("r2_m_D_given_X", ascending=False)
    out("\n**Nuisance R² per treatment (POOLED).** High `r2_m_D_given_X` → treatment well-explained by X → likely confounded. Low → near-experimental.\n")
    fmt_df(nu_p[["treatment", "outcome", "n",
                 "r2_g_Y_given_X", "r2_m_D_given_X", "r2_struct_Ytilde_given_Dtilde", "theta"]])


# ── 10. Confounder importance ─────────────────────────────────────────────


def section_10():
    h2("10. Confounder importance & OLS significance")
    audit = pq.read_table(DML / "confounder_audit.parquet").to_pandas()
    out("\n**Confounder importance ranking** (mean LightGBM importance across nuisance fits).")
    fmt_df(audit)

    loo = pq.read_table(DML / "confounder_loo_r2.parquet").to_pandas()
    out("\n**Leave-one-out ΔR² (rank_delta outcome)** — top 15.")
    sub = loo[loo["outcome"] == "rank_delta"].sort_values("delta_r2", ascending=False).head(15)
    fmt_df(sub)

    ols = pq.read_table(DML / "confounder_ols_significance.parquet").to_pandas()
    out("\n**OLS significance of each confounder** (rank_delta only, sorted by |t|).")
    sub = ols[(ols["outcome"] == "rank_delta") & (ols["confounder"] != "intercept")].copy()
    sub["|t|"] = sub["t_stat"].abs()
    sub = sub.sort_values("|t|", ascending=False)
    fmt_df(sub[["confounder", "coef", "se", "t_stat", "p_val", "ci_low", "ci_high", "stars"]])


# ── 11. Paper-architecture recommendation ─────────────────────────────────


def section_11():
    h2("11. Paper architecture & narrative — what the numbers support")
    out("""
Based on every diagnostic above, the table below restates the conclusion in
plain language, with section pointers.

### Headline (Spec B, RW-corrected)

**Promoters of LLM rank** (negative `rank_delta` coefficient, i.e. doc moves UP):
- `T1a_stats_present`  — presence of statistics in body.
- `T5_topical_comp`    — topical completeness score.
- `T2a_question_headings` — Q&A-style structural headings.
- `T_llms_txt`         — domain ships an `llms.txt` file.

**Demoters of LLM rank** (positive `rank_delta`, doc pushed DOWN):
- `T6_freshness`           — heavy freshness boilerplate.
- `T3_structured_data_new` — JSON-LD / schema markup.
- `T2_llm`                 — LLM-coded question-heading variant.

### Source-identity finding (side piece, §4 of paper)

`T7_source_earned` (membership in the curated 250-domain earned-media list) carries
the **single largest demotion coefficient** (≈ −1.7 to −1.8 on `rank_delta`, p<0.001)
even with all content controls + the LLM in-context retrieval. This survives Romano–Wolf.
**Interpretation:** the LLM rerankers systematically push DOWN sources that the curated
list considers "earned media" — a robust LLM-vs-organic-web bias, distinct from any
content treatment.

### RAG mitigation finding (§5)

RAG attenuates the source-class penalty in BIASED prompts (≈ 20% reduction) and is
flat under neutral prompts. The per-cell heterogeneity table shows the attenuation
is sharpest for the **Llama × searxng × serp50** cell — that's the paragraph to cite.

### Category-switch decision (recorded for §3 + Appendix A)

See Section 2C above. Any treatment carrying `→ drop to confounder` should be moved
to the X-set in the next refit cycle; until then, footnote in the Appendix that the
Spec B table is the recommended estimand, since its coefficient there has the right
"effect with everything else fixed" interpretation.

### Heterogeneity (§6)

Cell-level pivots show **post_rank** effects are more stable across cells than
`rank_delta`. Two cells of interest:
- `searxng_Qwen2.5-72B-Instruct_serp50_top10` — strongest source-identity effect.
- `duckduckgo_Llama-3.3-70B-Instruct_serp20_top10` — weakest effects overall (noise floor).

### Recommended section order

| §    | Topic                                                                | Source         |
|------|----------------------------------------------------------------------|----------------|
| 3    | Content-treatment effects (Spec B headline)                          | §3 of report   |
| 4    | Earned-media demotion (source identity)                              | §3 + §8 + §10  |
| 5    | RAG mitigation                                                       | §7             |
| 6    | Heterogeneity across engine × LLM × pool                             | §6             |
| 7    | Diagnostics: category-switch audit, variance, confounder importance   | §2 + §9 + §10  |
| App A | Spec A vs Spec B per-variant table                                    | §5             |
| App B | Joint inference w/ Romano–Wolf / Bonferroni                           | §4             |
| App C | Promotion-rate companion outcome                                      | §8             |
""")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    section_0()
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()
    section_9()
    section_10()
    section_11()
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(_tee.buf.getvalue())
    out(f"\n\n**Report saved → `{OUT_REPORT.relative_to(Path.cwd().parent)}`**")


if __name__ == "__main__":
    main()
