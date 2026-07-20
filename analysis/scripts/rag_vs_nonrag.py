#!/usr/bin/env python3
"""Deep RAG vs non-RAG comparison.

Two comparisons are interesting:
  - biased     vs biased_rag      (effect of adding RAG to a BIASED prompt)
  - neutral    vs neutral_rag     (effect of adding RAG to a NEUTRAL prompt)

We look at:
  1. Sample composition: do RAG variants drop or add rows? per-cell coverage.
  2. Outcome distributions: rank_delta and post_rank histograms by variant.
  3. Per-treatment DML deltas: (rag − non_rag) for every treatment, per outcome.
  4. Which treatments FLIP significance with RAG? (paper-relevant)
  5. Source-effect attenuation: T7 + T_llms_txt deltas with CIs.
  6. Heterogeneity: RAG attenuation broken down by (engine × pool × model).
  7. Where the LLM actually changes ranks: per-domain win/loss with RAG.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 100)

ROOT = Path.home() / "geodml_data"
DML = ROOT / "data" / "dml_results"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
PAIRS = [("biased", "biased_rag"), ("neutral", "neutral_rag")]
CONTENT = [
    "T1a_stats_present", "T1b_stats_density",
    "T2a_question_headings", "T2b_structural_modularity",
    "T3_structured_data_new",
    "T4a_ext_citations", "T4b_auth_citations",
    "T5_topical_comp", "T6_freshness",
]
SOURCE = ["T7_source_earned"]


def section(s, char="="):
    print("\n" + char * 88)
    print(s)
    print(char * 88)


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.1 else ""


def load_main(v):
    return pq.read_table(ROOT / "data" / "main" / f"full_experiment_data_{v}.parquet").to_pandas()


# ── 1. Sample composition ────────────────────────────────────────────────────


def section_1():
    section("1. SAMPLE COMPOSITION — what RAG adds or drops")
    rows = []
    for v in VARIANTS:
        df = load_main(v)
        rows.append({
            "variant": v,
            "n_rows": len(df),
            "n_keywords": df["keyword"].nunique(),
            "n_domains": df["domain"].nunique() if "domain" in df.columns else None,
            "n_keywords_with_data": df.dropna(subset=["post_rank", "treat_stats_present"])["keyword"].nunique(),
            "rows_per_keyword (mean)": round(len(df) / df["keyword"].nunique(), 2),
            "post_rank_min": int(df["post_rank"].min()),
            "post_rank_max": int(df["post_rank"].max()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  Why are RAG row counts smaller?")
    print("  RAG variants only emit rerank outputs when the rag retrieval is non-empty.")
    print("  Keywords where retrieval failed are skipped, so RAG variants have fewer rows.")

    # how many keywords overlap between rag and non-rag?
    print("\n  Keyword overlap (rag vs non_rag):")
    for nonrag, rag in PAIRS:
        a = set(load_main(nonrag)["keyword"])
        b = set(load_main(rag)["keyword"])
        print(f"    {nonrag:12s} ∩ {rag:13s}: "
              f"|A|={len(a):>5}  |B|={len(b):>5}  |A∩B|={len(a & b):>5}  "
              f"|A−B|={len(a - b):>5}  |B−A|={len(b - a):>5}")


# ── 2. Outcome distributions ─────────────────────────────────────────────────


def section_2():
    section("2. OUTCOME DISTRIBUTIONS — does RAG change WHERE the LLM lands rows?")
    rows = []
    for v in VARIANTS:
        df = load_main(v).dropna(subset=["post_rank"])
        rd = df["rank_delta"].dropna() if "rank_delta" in df.columns else None
        pr = df["post_rank"]
        rows.append({
            "variant": v,
            "rank_delta_mean": round(rd.mean(), 3) if rd is not None else None,
            "rank_delta_std": round(rd.std(), 3) if rd is not None else None,
            "rank_delta_median": round(rd.median(), 2) if rd is not None else None,
            "post_rank_mean": round(pr.mean(), 2),
            "post_rank_std": round(pr.std(), 2),
            "post_rank_median": int(pr.median()),
            "pct_no_change (rank_delta=0)": (
                f"{100 * (rd == 0).mean():.1f}%" if rd is not None else "—"
            ),
            "pct_promoted (delta>0)": (
                f"{100 * (rd > 0).mean():.1f}%" if rd is not None else "—"
            ),
            "pct_demoted (delta<0)": (
                f"{100 * (rd < 0).mean():.1f}%" if rd is not None else "—"
            ),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # paired comparison: same keyword × same domain, rag vs non-rag rank_delta
    print("\n  Paired rank_delta comparison (same (keyword, domain) URL with vs without RAG):")
    for nonrag, rag in PAIRS:
        a = load_main(nonrag).dropna(subset=["rank_delta"])
        b = load_main(rag).dropna(subset=["rank_delta"])
        # merge on (keyword, url) — same retrieved document
        m = a.merge(b, on=["keyword", "url"], suffixes=("_nr", "_rag"))
        if m.empty:
            print(f"    {nonrag} vs {rag}: no overlapping (keyword, url) pairs found")
            continue
        d = m["rank_delta_rag"] - m["rank_delta_nr"]
        from scipy import stats
        t, p = stats.ttest_rel(m["rank_delta_rag"], m["rank_delta_nr"])
        print(f"    {nonrag:12s} → {rag:13s}: n_pairs={len(m):>6}  "
              f"mean(Δrank_delta)={d.mean():+.3f}  "
              f"sd={d.std():.3f}  paired t={t:+.2f}  p={p:.2e}")


# ── 3. Per-treatment DML deltas (RAG − non-RAG) ──────────────────────────────


def section_3():
    section("3. PER-TREATMENT DML DELTAS — does RAG amplify or dampen each effect?")
    print("  Source: pre-computed per-variant PLR+LightGBM canonical estimates")
    print("  (the per-variant single-treatment file, max-n row per cell).\n")

    # Load all 4 variants, pick canonical PLR-lgbm + max-n_obs row per (treatment, outcome)
    spec_a_rows = []
    for v in VARIANTS:
        df = pq.read_table(DML / f"dml_results_long_{v}.parquet").to_pandas()
        df = df[(df.get("method", "plr") == "plr") & (df.get("learner", "lgbm") == "lgbm")]
        for outcome in ["rank_delta", "post_rank"]:
            d = df[df["outcome"] == outcome]
            for code in CONTENT + SOURCE:
                rs = d[d["treatment"] == code]
                if rs.empty:
                    continue
                top = rs.loc[rs["n_obs"].idxmax()]
                spec_a_rows.append({
                    "variant": v, "outcome": outcome, "treatment": code,
                    "coef": top["coef"], "se": top["se"], "p_val": top["p_val"], "n": int(top["n_obs"]),
                })
    A = pd.DataFrame(spec_a_rows)

    for outcome in ["rank_delta", "post_rank"]:
        sub = A[A["outcome"] == outcome]
        coef_p = sub.pivot(index="treatment", columns="variant", values="coef")
        se_p = sub.pivot(index="treatment", columns="variant", values="se")
        p_p = sub.pivot(index="treatment", columns="variant", values="p_val")

        print(f"  ── outcome = {outcome} — per-variant coefs ──")
        print(coef_p.reindex(CONTENT + SOURCE).round(4).to_string())
        print()

        # RAG deltas with approximate combined SE
        print(f"  ── outcome = {outcome} — RAG attenuation (Δ = coef_rag − coef_non_rag) ──")
        rows = []
        for nonrag, rag in PAIRS:
            for code in CONTENT + SOURCE:
                if code not in coef_p.index:
                    continue
                c_nr = coef_p.loc[code, nonrag]
                c_rg = coef_p.loc[code, rag]
                s_nr = se_p.loc[code, nonrag]
                s_rg = se_p.loc[code, rag]
                delta = c_rg - c_nr
                # independent SEs (approx — same domains, different conditions, treat as indep)
                se_delta = np.sqrt(s_nr ** 2 + s_rg ** 2)
                z = delta / se_delta if se_delta > 0 else 0.0
                from scipy import stats
                pp = 2 * (1 - stats.norm.cdf(abs(z)))
                rows.append({
                    "pair": f"{rag} − {nonrag}",
                    "treatment": code,
                    "coef_non_rag": c_nr, "coef_rag": c_rg, "Δ": delta,
                    "SE_Δ": se_delta, "z": z, "p_val_Δ": pp, "sig": stars(pp),
                })
        delta_df = pd.DataFrame(rows).round(4)
        print(delta_df.to_string(index=False))
        print()


# ── 4. Which treatments FLIP significance with RAG ───────────────────────────


def section_4():
    section("4. WHICH TREATMENTS FLIP SIGNIFICANCE with RAG?")
    spec_a_rows = []
    for v in VARIANTS:
        df = pq.read_table(DML / f"dml_results_long_{v}.parquet").to_pandas()
        df = df[(df["method"] == "plr") & (df["learner"] == "lgbm")]
        for outcome in ["rank_delta", "post_rank"]:
            d = df[df["outcome"] == outcome]
            for code in CONTENT + SOURCE:
                rs = d[d["treatment"] == code]
                if rs.empty:
                    continue
                top = rs.loc[rs["n_obs"].idxmax()]
                spec_a_rows.append({
                    "variant": v, "outcome": outcome, "treatment": code,
                    "coef": top["coef"], "p_val": top["p_val"],
                })
    A = pd.DataFrame(spec_a_rows)
    for outcome in ["rank_delta", "post_rank"]:
        sub = A[A["outcome"] == outcome]
        coef_p = sub.pivot(index="treatment", columns="variant", values="coef")
        p_p = sub.pivot(index="treatment", columns="variant", values="p_val")
        print(f"\n  ── outcome = {outcome} ──")
        for nonrag, rag in PAIRS:
            print(f"\n  {nonrag} → {rag}:")
            for code in CONTENT + SOURCE:
                if code not in p_p.index:
                    continue
                p_nr = p_p.loc[code, nonrag]
                p_rg = p_p.loc[code, rag]
                s_nr = stars(p_nr)
                s_rg = stars(p_rg)
                if s_nr != s_rg or (s_nr in ("*", "**", "***") and s_rg in ("*", "**", "***")):
                    flag = ""
                    if (p_nr < 0.05) != (p_rg < 0.05):
                        flag = "  ← FLIP"
                    print(f"    {code:30s}  non-rag p={p_nr:.4f} {s_nr:3s}    "
                          f"rag p={p_rg:.4f} {s_rg:3s}    "
                          f"coef: {coef_p.loc[code,nonrag]:+.3f} → {coef_p.loc[code,rag]:+.3f}{flag}")


# ── 5. Heterogeneity: RAG effect per (engine × pool × model) ─────────────────


def section_5():
    section("5. HETEROGENEITY — RAG effect by (engine × pool × model)")
    print("  Computing per-cell mean rank_delta for each variant, then RAG attenuation.\n")

    rows = []
    for v in VARIANTS:
        df = load_main(v).dropna(subset=["rank_delta"])
        df["model_short"] = df["llm_model"].str[:5]  # 'Llama' / 'Qwen2'
        g = df.groupby(["search_engine", "pool", "model_short"])["rank_delta"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        g["variant"] = v
        rows.append(g)
    cells = pd.concat(rows, ignore_index=True)
    pivot = cells.pivot(index=["search_engine", "pool", "model_short"],
                        columns="variant", values="mean").round(3)
    print("  Mean rank_delta per cell, by variant:")
    print(pivot.to_string())

    print("\n  RAG attenuation = mean_rd(rag) − mean_rd(non_rag) per cell:")
    deltas = pd.DataFrame()
    for nonrag, rag in PAIRS:
        if nonrag in pivot.columns and rag in pivot.columns:
            deltas[f"Δ({rag}−{nonrag})"] = (pivot[rag] - pivot[nonrag]).round(3)
    print(deltas.to_string())


# ── 6. Source-bias attenuation in detail ─────────────────────────────────────


def section_6():
    section("6. SOURCE-BIAS ATTENUATION — T7 + T_llms_txt with full CIs")
    spec_a_rows = []
    for v in VARIANTS:
        df = pq.read_table(DML / f"dml_results_long_{v}.parquet").to_pandas()
        df = df[(df["method"] == "plr") & (df["learner"] == "lgbm")]
        for outcome in ["rank_delta", "post_rank"]:
            d = df[df["outcome"] == outcome]
            for code in SOURCE:
                rs = d[d["treatment"] == code]
                if rs.empty:
                    continue
                top = rs.loc[rs["n_obs"].idxmax()]
                spec_a_rows.append({
                    "variant": v, "outcome": outcome, "treatment": code,
                    "coef": top["coef"], "se": top["se"], "p_val": top["p_val"],
                    "ci_low": top["coef"] - 1.96 * top["se"],
                    "ci_high": top["coef"] + 1.96 * top["se"],
                })
    A = pd.DataFrame(spec_a_rows)
    print(A.round(4).to_string(index=False))

    print("\n  RAG attenuation (rag − non_rag) with combined SE:")
    from scipy import stats
    for outcome in ["rank_delta", "post_rank"]:
        sub = A[A["outcome"] == outcome]
        for nonrag, rag in PAIRS:
            for code in SOURCE:
                rs_nr = sub[(sub["variant"] == nonrag) & (sub["treatment"] == code)]
                rs_rg = sub[(sub["variant"] == rag) & (sub["treatment"] == code)]
                if rs_nr.empty or rs_rg.empty:
                    continue
                c_nr, c_rg = rs_nr["coef"].iloc[0], rs_rg["coef"].iloc[0]
                s_nr, s_rg = rs_nr["se"].iloc[0], rs_rg["se"].iloc[0]
                delta = c_rg - c_nr
                se_d = np.sqrt(s_nr ** 2 + s_rg ** 2)
                z = delta / se_d
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                attn = abs(delta) / abs(c_nr) * 100 if c_nr != 0 else 0.0
                print(f"  [{outcome:>10s}] {rag} − {nonrag}  {code}: "
                      f"Δ={delta:+.3f} (SE={se_d:.3f})  "
                      f"= {attn:.1f}% of non-RAG magnitude  "
                      f"p={p:.4f} {stars(p)}")


# ── 7. Where the LLM actually changes its mind: per-domain RAG effect ────────


def section_7():
    section("7. PER-DOMAIN RAG EFFECT — which sources benefit, which lose")
    print("  For each domain, mean(post_rank) under RAG vs non-RAG (lower = better).\n")

    for nonrag, rag in PAIRS:
        a = load_main(nonrag).dropna(subset=["post_rank"])
        b = load_main(rag).dropna(subset=["post_rank"])
        a_dom = a.groupby("domain")["post_rank"].agg(["mean", "count"]).rename(
            columns={"mean": "post_nr", "count": "n_nr"})
        b_dom = b.groupby("domain")["post_rank"].agg(["mean", "count"]).rename(
            columns={"mean": "post_rag", "count": "n_rag"})
        # also pull treat_source_earned (deterministic per domain)
        flag = a.groupby("domain")["treat_source_earned"].max()
        m = a_dom.join(b_dom, how="inner")
        m["earned"] = flag.reindex(m.index).fillna(0).astype(int)
        m = m[(m["n_nr"] >= 5) & (m["n_rag"] >= 5)]  # filter low-n
        m["Δ_post_rank (rag−nr)"] = (m["post_rag"] - m["post_nr"]).round(3)
        print(f"  ── {nonrag} vs {rag}  (domains with ≥5 rows in each) ──")
        print(f"  n_domains qualifying: {len(m)}")
        print(f"  earned-media domains in subset: {int(m['earned'].sum())}")
        print(f"\n  Mean Δ_post_rank by source class:")
        by_kind = m.groupby("earned")["Δ_post_rank (rag−nr)"].agg(["mean", "std", "count"])
        by_kind.index = ["non-earned", "earned"]
        print(by_kind.round(3).to_string())

        # show top 10 RAG-helped and top 10 RAG-hurt domains
        m_sorted = m.sort_values("Δ_post_rank (rag−nr)")
        print(f"\n  TOP 10 RAG-HELPED domains (lower post_rank = better):")
        print(m_sorted.head(10)[["post_nr", "post_rag", "Δ_post_rank (rag−nr)",
                                  "n_nr", "n_rag", "earned"]].round(2).to_string())
        print(f"\n  TOP 10 RAG-HURT domains:")
        print(m_sorted.tail(10)[["post_nr", "post_rag", "Δ_post_rank (rag−nr)",
                                  "n_nr", "n_rag", "earned"]].round(2).to_string())
        print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    print("\n" + "=" * 88)
    print("END")
    print("=" * 88)


if __name__ == "__main__":
    main()
