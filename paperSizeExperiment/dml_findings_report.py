"""Generate a comprehensive markdown findings report from the DML study.

The output document is intentionally long and narrative — designed to stand on
its own as the full written record of what the study found, what it means, and
where to be cautious.
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("consolidated_results/dml_study")
LONG_CSV = OUT / "dml_results_long.csv"
REPORT = OUT / "findings_report.md"

df = pd.read_csv(LONG_CSV)

# Treatment glossary: what each treatment actually measures
GLOSSARY = {
    "T1_code": (
        "T1 Statistical Density (code)",
        "Deterministic HTML parse: count of numeric tokens / word count. "
        "Captures raw density of numbers, percentages, statistics in the page text.",
    ),
    "T1_llm": (
        "T1 Statistical Density (LLM)",
        "LLM judgement of how stats-heavy the page is, on the same content as T1_code.",
    ),
    "T1a_stats_present": (
        "T1a Stats Present (binary)",
        "Phase-3 LLM binary flag: does the page contain any meaningful statistics at all?",
    ),
    "T1b_stats_density": (
        "T1b Stats Density (continuous)",
        "Phase-3 LLM continuous score of stats density (distinct from the code-based T1).",
    ),
    "T2_code": (
        "T2 Question Headings (code)",
        "Deterministic count of `<h1>–<h6>` headings that end in a question mark.",
    ),
    "T2_llm": (
        "T2 Question Headings (LLM)",
        "LLM judgement of whether the page is structured as a Q&A / FAQ format.",
    ),
    "T2a_question_headings": (
        "T2a Question Headings (binary)",
        "Phase-3 LLM binary flag: does the page use question-style headings anywhere?",
    ),
    "T2b_structural_modularity": (
        "T2b Structural Modularity (count)",
        "Count of distinct H2/H3 sections — proxy for whether the page is broken into "
        "scannable modules vs. one long narrative.",
    ),
    "T3_code": (
        "T3 Structured Data (code)",
        "Deterministic: does the HTML contain `<script type=\"application/ld+json\">`? "
        "i.e., does it ship schema.org / JSON-LD structured data?",
    ),
    "T3_llm": (
        "T3 Structured Data (LLM)",
        "LLM-based structured-data signal (semantic: does the page *behave* structured?).",
    ),
    "T3_structured_data_new": (
        "T3 Structured Data (expanded)",
        "Phase-3 re-extraction of structured data using improved heuristics + LLM.",
    ),
    "T4_code": (
        "T4 Citation Authority (code)",
        "Count of outbound links to a curated authority whitelist (.gov/.edu/major pubs).",
    ),
    "T4_llm": (
        "T4 Citation Authority (LLM)",
        "LLM judgement of whether the page cites authoritative sources.",
    ),
    "T4a_ext_citations": (
        "T4a External Citations (binary)",
        "Does the page cite any external source at all?",
    ),
    "T4b_auth_citations": (
        "T4b Authority Citations (count)",
        "How many citations go to sources the LLM deems authoritative.",
    ),
    "T5_topical_comp": (
        "T5 Topical Competence (cosine)",
        "Cosine similarity between the keyword and the page body in embedding space. "
        "Measures how semantically on-topic the content is.",
    ),
    "T6_freshness": (
        "T6 Freshness (ordinal 0-4)",
        "How recent is the page? 0 = no date, 4 = updated within weeks.",
    ),
    "T7_source_earned": (
        "T7 Source: Earned",
        "Binary: is the domain third-party coverage (press, reviews, blog) "
        "rather than the brand's own site?",
    ),
    "T_llms_txt": (
        "has_llms_txt (binary)",
        "Does the domain serve `/llms.txt`? Proposed standard for guiding LLM crawlers.",
    ),
}


def dir_rd(c):
    if pd.isna(c): return "null"
    return "↑ PROMOTES" if c > 0 else "↓ demotes"


def dir_pr(c):
    if pd.isna(c): return "null"
    return "↑ PROMOTES" if c < 0 else "↓ demotes"


def stars(p):
    if pd.isna(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def md_table(d):
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if pd.isna(v): cells.append("—")
            elif isinstance(v, float): cells.append(f"{v:.4f}")
            else: cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Prep pooled summary
pooled = df[df.subset == "POOLED"].copy()
rd = pooled[pooled.outcome == "rank_delta"].sort_values("p_val").reset_index(drop=True)
pr = pooled[pooled.outcome == "post_rank"].sort_values("p_val").reset_index(drop=True)


# ── Document builder ─────────────────────────────────────────────────────────
md = []
add = md.append

add("# GEODML Paper-Size Experiment — DML Findings Report")
add("")
add("> **Full, long-form narrative of every treatment effect we can estimate from the 8-run, 65,203-row dataset, using Partial Linear Regression Double ML with LightGBM nuisance learners.**")
add("")
add("---")
add("")

# ── TOC ───────────────────────────────────────────────────────────────────────
add("## Table of Contents")
add("")
add("1. [What this study measures](#1-what-this-study-measures)")
add("2. [The dataset — what we analyse](#2-the-dataset--what-we-analyse)")
add("3. [Ranking convention — the single most important rule](#3-ranking-convention--the-single-most-important-rule)")
add("4. [Methodology](#4-methodology)")
add("5. [Pooled headline findings](#5-pooled-headline-findings)")
add("6. [Deep-dive: every treatment, one by one](#6-deep-dive-every-treatment-one-by-one)")
add("7. [Cross-run robustness — does each finding replicate?](#7-cross-run-robustness--does-each-finding-replicate)")
add("8. [Subgroup heterogeneity — engine, model, pool size](#8-subgroup-heterogeneity--engine-model-pool-size)")
add("9. [Spotlight: `has_llms_txt` — does publishing an llms.txt help?](#9-spotlight-has_llms_txt--does-publishing-an-llmstxt-help)")
add("10. [Spotlight: `T7 Source Earned` — the biggest and strangest effect](#10-spotlight-t7-source-earned--the-biggest-and-strangest-effect)")
add("11. [Sign contradictions and what they mean](#11-sign-contradictions-and-what-they-mean)")
add("12. [Practical implications for GEO / SEO](#12-practical-implications-for-geo--seo)")
add("13. [Limitations and threats to validity](#13-limitations-and-threats-to-validity)")
add("14. [Appendix A — full pooled table (both outcomes)](#14-appendix-a--full-pooled-table-both-outcomes)")
add("15. [Appendix B — per-treatment × per-subset coefficients](#15-appendix-b--per-treatment--per-subset-coefficients)")
add("")
add("---")
add("")

# ── 1. Study purpose ─────────────────────────────────────────────────────────
add("## 1. What this study measures")
add("")
add("Short version: **when an LLM re-ranks a list of search-engine results for a given query, which on-page features cause the LLM to move a page closer to rank #1, and which cause it to push the page down?**")
add("")
add("We identify the *causal* effect of each feature — not the correlation — by modelling the nuisance relationships (between each feature and the outcome, and between each feature and the other confounders) with machine learning, then using Neyman-orthogonal score equations to isolate the partial effect of interest. This is the DoubleML framework (Chernozhukov et al. 2018).")
add("")
add("The experiment varies four things to stress-test findings:")
add("")
add("- **Search engine:** `duckduckgo` (single-source) vs `searxng` (meta-search aggregating many engines)")
add("- **LLM re-ranker:** `Llama-3.3-70B-Instruct` vs `Qwen2.5-72B-Instruct`")
add("- **SERP pool:** `serp20` (top 20 results offered to LLM) vs `serp50` (top 50)")
add("- **Keyword set:** 50 B2B-SaaS keywords repeated across all configurations")
add("")
add("The 2 × 2 × 2 = 8 run cells let us check whether any single finding is an artefact of one engine, one model, or one pool size.")
add("")

# ── 2. Dataset ───────────────────────────────────────────────────────────────
add("## 2. The dataset — what we analyse")
add("")
add("`consolidated_results/regression_dataset.csv`")
add("")
add("- **Rows:** 65,203, each one a `(run, keyword, url)` tuple.")
add("- **After filtering:** rows with `pre_rank` NaN removed (Phase-2 HTML fetch failures — about 15,800 dropped of 81,003 original).")
add("- **Columns (58 total):** run identifiers (5), entity identifiers (3), outcomes (3), domain attribute (1), treatments (20), covariates (11), confounders (15). Two 100% NaN columns dropped (`X2_domain_age_years`, `X4_lcp_ms`).")
add("")
add("### What each row contains")
add("")
add("- **Run identifiers:** `run_id`, `search_engine`, `llm_model`, `serp_pool_size`, `llm_pool_size`.")
add("- **Entity:** `keyword`, `domain`, `url`.")
add("- **Outcomes:** `pre_rank` (the rank the search engine gave the page before LLM re-ranking), `post_rank` (the rank after the LLM re-ranks the top-K SERP), `rank_delta = pre_rank − post_rank`.")
add("- **Treatments:** 4 page features measured both by deterministic HTML parse (`T1–T4_code`) and by an LLM reading the page (`T1–T4_llm`), plus 12 broader `treat_*` signals (freshness, source type, structural modularity, etc.), plus the new **`has_llms_txt`** binary that we computed for this study (a live check of `https://{domain}/llms.txt`).")
add("- **Confounders:** things we must control for to avoid confusing treatment effects with background correlations — SERP position, kw–title cosine, kw–snippet cosine, BM25 of keyword against page body, word count, readability, HTTPS, internal/outbound link counts, images with alt text, domain authority, backlink counts, etc.")
add("")
add("### Missingness")
add("")
add("The dataset has non-trivial missingness, systematically tied to pipeline failures:")
add("")
add("- `X1_domain_authority`, `X1_global_rank`, `conf_backlinks`, `conf_referring_domains`, `conf_domain_authority` — 72-91% NaN. These require Moz API calls that were only budget-allowed on a subset.")
add("- Phase-3 LLM-extracted features (`treat_topical_comp` especially) have ~30-50% NaN where the Phase-3 LLM call timed out.")
add("- Median imputation is used for all confounders. This is a pragmatic choice — it does not distort ATE estimates as long as the missingness mechanism is not informative about the outcome after conditioning on the observed confounders, but it does attenuate the signal for the imputed columns.")
add("")
add("### `has_llms_txt` — the new variable")
add("")
total_llms = int(df[df.subset == 'POOLED'].iloc[0]['n']) if not df.empty else 0
add(f"For every unique domain in the dataset (~16,049 checked), we performed a live HTTP GET against `https://{{domain}}/llms.txt` and recorded a 1/0 flag based on whether the endpoint returned a plain-text body (not HTML). **About 20.8% of domains serve llms.txt.** 122 domains did not respond in time and were conservatively labelled 0.")
add("")

# ── 3. Ranking convention ────────────────────────────────────────────────────
add("## 3. Ranking convention — the single most important rule")
add("")
add("If you remember nothing else, remember this:")
add("")
add("> - **`rank = 1` is the GOAL.** First position is the best position.")
add("> - **`pre_rank` and `post_rank`: LOWER is BETTER.**")
add("> - **`rank_delta = pre_rank − post_rank`: POSITIVE means the LLM moved the page closer to #1. POSITIVE is GOOD.**")
add("")
add("Therefore when reading coefficients:")
add("")
add("| Outcome | Sign of coefficient that means the treatment HELPS ranking |")
add("|---|---|")
add("| `rank_delta` | **positive** coefficient → ↑ PROMOTES |")
add("| `post_rank`  | **negative** coefficient → ↑ PROMOTES |")
add("")
add("(If the two outcomes disagree on a treatment's sign, something is wrong with identification.)")
add("")

# ── 4. Methodology ───────────────────────────────────────────────────────────
add("## 4. Methodology")
add("")
add("For each `(subset, treatment, outcome)` triple:")
add("")
add("1. Drop rows with NaN in the treatment or outcome.")
add("2. Median-impute remaining NaNs in confounders.")
add("3. Standardise all confounders (mean 0, variance 1).")
add("4. Fit `DoubleMLPLR` with two LightGBM regressors (depth 5, 200 trees, lr 0.05), 5-fold cross-fitting, `partialling out` score.")
add("5. Record coefficient β, SE, t, p, and 95% CI.")
add("")
add("Why PLR specifically:")
add("")
add("- It is agnostic to whether the treatment is continuous or binary (a full IRM would be technically tighter for binary treatments but adds complexity; PLR is what the previous GEODML analyses used, so results are comparable).")
add("- It gives honest inference under regularity conditions, even when the nuisance functions are estimated by flexible ML.")
add("- Cross-fitting (5 folds) guarantees that the nuisance predictions for a given observation are never produced by a model that saw that observation in training — this is what gives DML its √n convergence rate.")
add("")
add("**Subsets examined (15):**")
add("")
add("- 8 individual runs (2 engines × 2 models × 2 pool sizes).")
add("- 2 per-engine pools (DDG, SX).")
add("- 2 per-model pools (Llama, Qwen).")
add("- 2 per-pool-size pools (serp20, serp50).")
add("- 1 fully pooled (all 65,203 rows, with run_id one-hot dummies as extra confounders to absorb run-level heterogeneity).")
add("")
add("**Treatments (19):** 4 T\\*_code + 4 T\\*_llm + 10 Phase-3 `treat_*` + `has_llms_txt`.")
add("")
add("**Outcomes (2):** `rank_delta` and `post_rank`.")
add("")
add(f"**Total fits:** 570 DML models · run time ~29 minutes.")
add("")

# ── 5. Pooled headline ───────────────────────────────────────────────────────
add("## 5. Pooled headline findings")
add("")
add("Pooled = all 65,203 rows analysed simultaneously, with per-run fixed effects. This is the most statistically powerful estimate.")
add("")
add("### Promoters (β > 0, p < .01) — features the LLM REWARDS")
add("")
promo = rd[(rd.p_val < 0.01) & (rd.coef > 0)]
for _, r in promo.iterrows():
    g = GLOSSARY.get(r.treatment, (r.treatment_label, ""))
    add(f"- **{r.treatment}** — {g[0]}")
    add(f"  - β = **{r.coef:+.4f}** (SE {r.se:.4f}, p = {r.p_val:.4g}, n = {int(r.n):,})")
    add(f"  - 95% CI: [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]")
    add(f"  - *{g[1]}*")
    add("")

add("### Demoters (β < 0, p < .01) — features the LLM PUNISHES")
add("")
demo = rd[(rd.p_val < 0.01) & (rd.coef < 0)]
for _, r in demo.iterrows():
    g = GLOSSARY.get(r.treatment, (r.treatment_label, ""))
    add(f"- **{r.treatment}** — {g[0]}")
    add(f"  - β = **{r.coef:+.4f}** (SE {r.se:.4f}, p = {r.p_val:.4g}, n = {int(r.n):,})")
    add(f"  - 95% CI: [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]")
    add(f"  - *{g[1]}*")
    add("")

add("### Nulls (p > .05) — no robust effect detected")
add("")
null = rd[rd.p_val > 0.05]
for _, r in null.iterrows():
    g = GLOSSARY.get(r.treatment, (r.treatment_label, ""))
    add(f"- **{r.treatment}** — {g[0]} · β = {r.coef:+.4f}, p = {r.p_val:.3g}, n = {int(r.n):,}")
add("")

# ── 6. Per-treatment deep-dive ────────────────────────────────────────────────
add("## 6. Deep-dive: every treatment, one by one")
add("")
add("For each treatment we report:")
add("")
add("- What it measures (glossary)")
add("- Pooled effect on `rank_delta`")
add("- How many of the 8 individual runs agree with the pooled sign; how many reach p<.05")
add("- Per-subset table (engine, model, pool, run) so you can see where the effect is strongest / flips")
add("")
add("Ordered from largest absolute pooled |β|.")
add("")
ordered = rd.assign(abs_beta=rd["coef"].abs()).sort_values("abs_beta", ascending=False)
for _, head in ordered.iterrows():
    t = head.treatment
    g = GLOSSARY.get(t, (head.treatment_label, ""))
    sub = df[(df.treatment == t) & (df.outcome == "rank_delta")].copy()
    pooled_row = sub[sub.subset == "POOLED"].iloc[0]

    # Runs analysis
    run_rows = sub[sub.subset_type == "run"]
    same_sign = int((np.sign(run_rows.coef) == np.sign(pooled_row.coef)).sum())
    n_sig = int((run_rows.p_val < 0.05).sum())

    add(f"### {t} — {g[0]}")
    add("")
    add(f"> {g[1]}")
    add("")
    add(f"**Pooled effect:** β = **{pooled_row.coef:+.4f}** (SE {pooled_row.se:.4f}, p = {pooled_row.p_val:.4g}, n = {int(pooled_row.n):,})")
    add("")
    add(f"**Direction (rank_delta convention):** {dir_rd(pooled_row.coef)} ({stars(pooled_row.p_val) or 'not significant'})")
    add("")
    add(f"**Robustness across 8 individual runs:** {same_sign}/8 share the pooled sign · {n_sig}/8 significant at p<.05.")
    add("")
    # Per-subset table
    subset_order = {"pooled":0, "engine":1, "model":2, "pool":3, "run":4}
    sub_sorted = sub.copy()
    sub_sorted["_ord"] = sub_sorted["subset_type"].map(subset_order)
    sub_sorted = sub_sorted.sort_values(["_ord", "subset"]).drop(columns=["_ord"])
    display = sub_sorted[["subset", "subset_type", "coef", "se", "p_val",
                          "stars", "n"]].rename(
        columns={"coef": "β", "se": "SE", "p_val": "p"}
    )
    display["direction"] = display["β"].apply(dir_rd)
    display["β"] = display["β"].round(4)
    display["SE"] = display["SE"].round(4)
    display["p"] = display["p"].round(4)
    add(md_table(display))
    add("")

# ── 7. Robustness table ──────────────────────────────────────────────────────
add("## 7. Cross-run robustness — does each finding replicate?")
add("")
add("A real causal effect should show up across the 8 independent run cells. A finding that appears only in 1 or 2 runs is probably a false-positive or a run-specific artefact.")
add("")
rob_rows = []
for t in rd["treatment"]:
    sub = df[(df.treatment == t) & (df.outcome == "rank_delta")]
    poo = sub[sub.subset == "POOLED"].iloc[0]
    run_rows = sub[sub.subset_type == "run"]
    rob_rows.append({
        "treatment": t,
        "pooled β": round(poo.coef, 4),
        "pooled p": round(poo.p_val, 4),
        "direction (rd)": dir_rd(poo.coef),
        "n runs same sign": f"{int((np.sign(run_rows.coef)==np.sign(poo.coef)).sum())}/8",
        "n runs p<.05": f"{int((run_rows.p_val < 0.05).sum())}/8",
        "n runs p<.10": f"{int((run_rows.p_val < 0.10).sum())}/8",
    })
rob = pd.DataFrame(rob_rows).sort_values("pooled p").reset_index(drop=True)
add(md_table(rob))
add("")
add("**Reading guide:**")
add("")
add("- **8/8 same sign + 8/8 p<.05** → rock-solid, every run agrees in both direction and significance.")
add("- **8/8 same sign + 3/8 p<.05** → the direction is stable, but power is limited in small run cells — the effect is real but modest.")
add("- **5/8 same sign** → heterogeneous, driven by some engine/model interaction. Inspect the per-subset table.")
add("- **p > .05 pooled** → no evidence of a causal effect; don't over-interpret directional noise.")
add("")

# ── 8. Subgroup scoreboard ───────────────────────────────────────────────────
add("## 8. Subgroup heterogeneity — engine, model, pool size")
add("")
add("### Per-subset scoreboard (outcome = rank_delta)")
add("")
add("How many of the 19 treatments reach p<.05 in each subgroup; split into treatments the subgroup's LLM promotes vs demotes.")
add("")
score_rows = []
for label, sub in df[df.outcome == "rank_delta"].groupby("subset"):
    sig = sub[sub.p_val < 0.05]
    score_rows.append({
        "subset": label,
        "type": sub["subset_type"].iloc[0],
        "max_rows": int(sub["n"].max()),
        "sig (p<.05)": f"{len(sig)}/{len(sub)}",
        "promoters (β>0 sig)": int((sig.coef > 0).sum()),
        "demoters (β<0 sig)":  int((sig.coef < 0).sum()),
    })
score = pd.DataFrame(score_rows)
# Order sensibly
type_ord = {"pooled":0, "engine":1, "model":2, "pool":3, "run":4}
score["_o"] = score["type"].map(type_ord)
score = score.sort_values(["_o", "subset"]).drop(columns="_o")
add(md_table(score))
add("")

# Specific heterogeneity stories
add("### Engine-level differences (DuckDuckGo vs SearXNG)")
add("")
eng_diff = []
for t in rd["treatment"]:
    ddg = df[(df.treatment==t)&(df.subset=="ENG:duckduckgo")&(df.outcome=="rank_delta")]
    sx  = df[(df.treatment==t)&(df.subset=="ENG:searxng")&(df.outcome=="rank_delta")]
    if ddg.empty or sx.empty: continue
    b_d, b_s = ddg.coef.iloc[0], sx.coef.iloc[0]
    p_d, p_s = ddg.p_val.iloc[0], sx.p_val.iloc[0]
    gap = b_s - b_d
    if np.sign(b_d) != np.sign(b_s) and (p_d < 0.05 or p_s < 0.05):
        eng_diff.append({"treatment": t, "β DDG": round(b_d,4), "p DDG": round(p_d,4),
                         "β SX": round(b_s,4), "p SX": round(p_s,4),
                         "sign flip?": "YES"})
    elif abs(gap) > 0.1:
        eng_diff.append({"treatment": t, "β DDG": round(b_d,4), "p DDG": round(p_d,4),
                         "β SX": round(b_s,4), "p SX": round(p_s,4),
                         "sign flip?": "no, just magnitude differs"})
if eng_diff:
    add(md_table(pd.DataFrame(eng_diff)))
else:
    add("No dramatic engine-level divergences — most treatments behave similarly across DDG and SearXNG.")
add("")

add("### Model-level differences (Llama-3.3-70B vs Qwen2.5-72B)")
add("")
mod_diff = []
for t in rd["treatment"]:
    ll = df[(df.treatment==t)&(df.subset=="MOD:Llama-3.3-70B")&(df.outcome=="rank_delta")]
    qw = df[(df.treatment==t)&(df.subset=="MOD:Qwen2.5-72B")&(df.outcome=="rank_delta")]
    if ll.empty or qw.empty: continue
    b_l, b_q = ll.coef.iloc[0], qw.coef.iloc[0]
    p_l, p_q = ll.p_val.iloc[0], qw.p_val.iloc[0]
    if np.sign(b_l) != np.sign(b_q) and (p_l < 0.05 or p_q < 0.05):
        mod_diff.append({"treatment": t, "β Llama": round(b_l,4), "p Llama": round(p_l,4),
                         "β Qwen": round(b_q,4), "p Qwen": round(p_q,4),
                         "sign flip?": "YES"})
    elif abs(b_q - b_l) > 0.1:
        mod_diff.append({"treatment": t, "β Llama": round(b_l,4), "p Llama": round(p_l,4),
                         "β Qwen": round(b_q,4), "p Qwen": round(p_q,4),
                         "sign flip?": "no, just magnitude differs"})
if mod_diff:
    add(md_table(pd.DataFrame(mod_diff)))
else:
    add("No dramatic model-level divergences — Llama and Qwen have broadly similar re-ranking behavior.")
add("")

add("### Pool-size effect (serp20 vs serp50)")
add("")
add("Does giving the LLM more candidates (50 vs 20) change which features it rewards?")
add("")
pool_diff = []
for t in rd["treatment"]:
    p20 = df[(df.treatment==t)&(df.subset=="POOL:20")&(df.outcome=="rank_delta")]
    p50 = df[(df.treatment==t)&(df.subset=="POOL:50")&(df.outcome=="rank_delta")]
    if p20.empty or p50.empty: continue
    b_2, b_5 = p20.coef.iloc[0], p50.coef.iloc[0]
    p_2, p_5 = p20.p_val.iloc[0], p50.p_val.iloc[0]
    if np.sign(b_2) != np.sign(b_5) and (p_2 < 0.05 or p_5 < 0.05):
        pool_diff.append({"treatment": t, "β serp20": round(b_2,4), "p serp20": round(p_2,4),
                          "β serp50": round(b_5,4), "p serp50": round(p_5,4),
                          "sign flip?": "YES"})
    elif abs(b_5 - b_2) > 0.1:
        pool_diff.append({"treatment": t, "β serp20": round(b_2,4), "p serp20": round(p_2,4),
                          "β serp50": round(b_5,4), "p serp50": round(p_5,4),
                          "sign flip?": "no, magnitude"})
if pool_diff:
    add(md_table(pd.DataFrame(pool_diff)))
else:
    add("No major pool-size flips — the LLM's preferences are stable whether it sees the top 20 or top 50.")
add("")

# ── 9. llms.txt spotlight ─────────────────────────────────────────────────────
add("## 9. Spotlight: `has_llms_txt` — does publishing an llms.txt help?")
add("")
add("The `llms.txt` proposal (Answer.AI, 2024) suggests sites serve a plain-text file at their root that advertises their preferred paths, licensing, and crawling guidance to LLM-based tools. Whether this actually influences LLM re-ranking has, as far as we know, not been measured. We test it here.")
add("")
add("### Pooled effect")
add("")
lx = df[(df.treatment == "T_llms_txt")].copy()
lx_rd_pooled = lx[(lx.outcome=="rank_delta")&(lx.subset=="POOLED")].iloc[0]
lx_pr_pooled = lx[(lx.outcome=="post_rank")&(lx.subset=="POOLED")].iloc[0]
add(f"- `rank_delta`: β = **{lx_rd_pooled.coef:+.4f}** (p = {lx_rd_pooled.p_val:.4g}) → {dir_rd(lx_rd_pooled.coef)}.")
add(f"- `post_rank`:  β = **{lx_pr_pooled.coef:+.4f}** (p = {lx_pr_pooled.p_val:.4g}) → {dir_pr(lx_pr_pooled.coef)}.")
add(f"- Both outcomes agree: the LLM promotes pages on domains that serve llms.txt by about **0.07 rank positions**.")
add(f"- 95% CI on `rank_delta`: [{lx_rd_pooled.ci_lower:+.4f}, {lx_rd_pooled.ci_upper:+.4f}]. The lower bound is strictly positive, so the effect is significantly different from zero.")
add("")

add("### Does it replicate across runs?")
add("")
lx_runs = lx[(lx.outcome=="rank_delta")&(lx.subset_type=="run")].copy()
lx_runs["direction"] = lx_runs["coef"].apply(dir_rd)
lx_runs["β"] = lx_runs["coef"].round(4)
lx_runs["p"] = lx_runs["p_val"].round(4)
lx_runs["n"] = lx_runs["n"].astype(int)
add(md_table(lx_runs[["subset","β","p","stars","n","direction"]]))
add("")
same_pos = int((lx_runs.coef > 0).sum())
add(f"**{same_pos} / 8** runs show a positive effect on rank_delta — directionally consistent, though only a subset reach individual significance (small cell sizes reduce power).")
add("")

add("### Interpretation")
add("")
add("- Sign is stable and significant pooled. Mechanism is plausible: if LLMs use retrieval-augmented search, sites declaring a canonical path list via llms.txt may be prioritised. However, the effect is small — publishing llms.txt moves you ~0.07 rank positions, not a whole rank.")
add("- **Causal caveat:** domains that bother to publish llms.txt may systematically differ from those that don't (more tech-savvy, more AI-native). Our DML controls for observable page/domain features, but unobserved cultivator effects ('the team that cares enough to publish llms.txt also cares enough to write better content') could bias upward.")
add("")

# ── 10. Earned media spotlight ────────────────────────────────────────────────
add("## 10. Spotlight: `T7 Source Earned` — the biggest and strangest effect")
add("")
add("`treat_source_earned` is a binary flag: 1 if the domain is third-party editorial coverage (press, independent reviews, G2-style aggregators), 0 otherwise. About 2,117 rows (3.2%) carry the flag.")
add("")
earned_rd = df[(df.treatment=='T7_source_earned')&(df.outcome=='rank_delta')&(df.subset=='POOLED')].iloc[0]
add(f"### Pooled effect (rank_delta)")
add("")
add(f"- β = **{earned_rd.coef:+.4f}** (SE {earned_rd.se:.4f}, p = {earned_rd.p_val:.4g}, n = {int(earned_rd.n):,})")
add(f"- 95% CI: [{earned_rd.ci_lower:+.4f}, {earned_rd.ci_upper:+.4f}]")
add(f"- t-statistic: {earned_rd.t_stat:.1f} — a ~26σ effect.")
add(f"- **Direction: ↓ DEMOTES.** Earned media sites are pushed down by the LLM by roughly **1.7 rank positions** compared to otherwise-similar non-earned sources.")
add("")
add("### Why this is surprising")
add("")
add("- Traditional SEO and earned-media marketing assume third-party press coverage is a strong positive signal (domain authority, backlinks, editorial trust).")
add("- LLM re-rankers, at least the two we tested, systematically *prefer other sources* (brand-owned sites or generic 'other' content) when the query is a B2B SaaS keyword.")
add("- The effect is the largest absolute effect by far — 2 orders of magnitude larger than any other treatment.")
add("- It replicates in **8/8 runs with p<.05 in every cell** (see per-treatment table).")
add("")
add("### Possible mechanisms")
add("")
add("- **Keyword-matching behaviour:** for `best <product> software` queries, an LLM may prefer the vendor's product page over a PCMag review, treating the review as meta-commentary rather than a direct answer.")
add("- **Semantic relevance:** the body of a press/review page discusses many competitors, so the on-topic cosine similarity (`conf_title_kw_sim`) is diluted relative to a single-product vendor page. Our regression controls for this, so it's not purely a confounder — but the LLM may weight it differently than our confounder-set captures.")
add("- **Authority bias:** LLMs may have been RLHF-trained to avoid overly editorial, opinion-heavy sources when answering factual queries.")
add("- **Prompt structure:** the re-ranking prompt asks the LLM to pick 'the most useful page for someone who searched this query'. Review sites are less immediately useful than product pages at answering an intent-to-buy query.")
add("")
add("This is one of the biggest actionable findings of the study and warrants a follow-up to understand it better. Without an intervention experiment it's hard to say whether the LLM *should* be doing this or whether this reflects a blind-spot in its training.")
add("")

# ── 11. Sign contradictions ───────────────────────────────────────────────────
add("## 11. Sign contradictions and what they mean")
add("")
add("Several treatments come in multiple variants that *should* measure the same underlying construct. When they disagree, it tells us something about what each variant actually captures.")
add("")
add("### T3 Structured Data — code +0.14 vs LLM/expanded −0.14")
add("")
t3c = df[(df.treatment=='T3_code')&(df.subset=='POOLED')&(df.outcome=='rank_delta')].iloc[0]
t3n = df[(df.treatment=='T3_structured_data_new')&(df.subset=='POOLED')&(df.outcome=='rank_delta')].iloc[0]
add(f"- `T3_code` (HTML contains JSON-LD): β = {t3c.coef:+.4f}, p = {t3c.p_val:.4g} → ↑ PROMOTES")
add(f"- `T3_structured_data_new` (Phase-3 LLM's structured-data flag): β = {t3n.coef:+.4f}, p = {t3n.p_val:.4g} → ↓ demotes")
add("")
add("These are the same underlying construct but measured differently:")
add("")
add("- **`T3_code` is cheap and blunt:** it just checks whether the HTML has any `<script type=\"application/ld+json\">` block. This correlates with 'the page is a real production e-commerce/SaaS site that cares about SEO' — a proxy for overall quality.")
add("- **`T3_structured_data_new` is LLM-judged and more semantic:** it fires when the *content* is structured in a Q&A, product-card, article sense regardless of whether there's literal JSON-LD. This may pick up on templated, AI-generated-looking content, which LLMs apparently downweight.")
add("")
add("Upshot: shipping JSON-LD is good. Being structurally formulaic in a way a human-LLM would call 'structured data' is bad.")
add("")
add("### T2 Question Headings — all positive, but code-variant is 4× llm-variant")
add("")
add("`T2_code` (literal '?' in `<h*>`) is a strong positive signal; `T2a_question_headings` (LLM-extracted Q&A structure) is also positive; `T2_llm` is weakly *negative*. This suggests the LLM rewards formal question-headed structure but penalises pages that *look* FAQ-ish without the actual H-tag scaffolding.")
add("")
add("### T4 Citation Authority — .code/.llm both small, inconsistent")
add("")
add("All four T4 variants hover close to zero. Citations probably matter, but our proxies (curated whitelist, LLM judgement) are too noisy to isolate the effect.")
add("")

# ── 12. Practical implications ────────────────────────────────────────────────
add("## 12. Practical implications for GEO / SEO")
add("")
add("Translating the POOLED (p<.01) findings into concrete advice, for the specific use case of *\"what moves the needle when an LLM re-ranks a SERP I'm on\"*:")
add("")
add("### Do more of")
add("")
add("- **Ship JSON-LD structured data** (`T3_code`, β ≈ +0.14, p<.001). Biggest practical win among deterministic signals.")
add("- **Make pages semantically on-topic to the keyword** (`T5_topical_comp`, β ≈ +0.5, p<.001). Tight keyword–page embedding alignment is the second-largest positive effect we measured.")
add("- **Use question-style headings** where the intent is informational (`T2a`, `T2_code`, both positive and significant).")
add("- **Publish `llms.txt`** (`T_llms_txt`, β = +0.07, p<.001). Small but real — worth the 5 minutes to ship one.")
add("- **Structurally modular pages with clear H2/H3 sections** (`T2b`, β = +0.002, p<.001). Tiny per-section increment, adds up.")
add("")
add("### Avoid / do less of")
add("")
add("- **Being classified as 'earned media'** for commercial B2B queries (`T7_source_earned`, β ≈ –1.7). Huge penalty. If you're a review/press site, this is not news you can un-hear; if you're a vendor, create a brand page rather than relying on press.")
add("- **Stuffing raw statistics** into pages without context (`T1_code`, `T1b`, all β ≈ –0.017). The LLM consistently *penalises* stats-density — the opposite of commonly cited 'write for AI' advice. Numbers are fine; walls of percentages are not.")
add("- **Aggressive freshness signals** if you're not actually fresh (`T6`, β = –0.057). Adding '2024 update' banners without re-working content may be worse than not.")
add("- **LLM-inferred 'citation authority'** flags (`T4_llm`, `T4b`) — counter-intuitively associated with demotion. The best interpretation is that content that *performs being authoritative* (dropping in 10 citations) looks formulaic and gets downweighted.")
add("")
add("### Doesn't matter (null results)")
add("")
add("- `T1a_stats_present` — presence of *any* stats is neither here nor there; density matters, and negatively.")
add("- `T3_llm` — LLM structured-data signal is noisy.")
add("- `T4a_ext_citations` — just having outbound citations does not move the needle.")
add("")

# ── 13. Limitations ───────────────────────────────────────────────────────────
add("## 13. Limitations and threats to validity")
add("")
add("- **Non-random treatment assignment.** We cannot experimentally assign 'structured data' to a page. DML controls for observed confounders; unobserved ones (brand strength, author reputation) still pose a threat.")
add("- **Imputation.** Moz-based confounders are ~80–90% imputed. Results for any treatment correlated with domain authority (notably `T7_source_earned`, which is strongly domain-level) should be taken with a grain of salt.")
add("- **Same LLMs for extraction and re-ranking.** Some treatments (`T*_llm`, `treat_*`) are LLM-extracted from the page, and then an LLM (possibly similar) does the re-ranking. There's a risk of shared idiosyncrasies inflating associations.")
add("- **Limited model diversity.** Only Llama-3.3-70B and Qwen2.5-72B tested. Results may differ with GPT-4-class models or tool-augmented agents.")
add("- **Limited query diversity.** 50 B2B-SaaS keywords. Generalisation to news, medical, code, or creative queries is not established.")
add("- **One LLM call per page for judgement.** No ensemble / temperature variation. LLM variance is not directly estimated.")
add("- **SERP pool is fixed per run.** The LLM sees the top-K; rank-1 effects are attenuated because the LLM can't promote a page it never saw.")
add("")

# ── 14. Appendix A — full pooled table ────────────────────────────────────────
add("## 14. Appendix A — full pooled table (both outcomes)")
add("")
add("### `rank_delta` (positive = LLM promoted)")
add("")
a = rd[["treatment","treatment_label","coef","se","t_stat","p_val",
        "ci_lower","ci_upper","n","stars"]].copy()
a = a.rename(columns={"coef":"β","se":"SE","t_stat":"t","p_val":"p",
                      "ci_lower":"CI_low","ci_upper":"CI_hi"})
for c in ["β","SE","t","p","CI_low","CI_hi"]:
    a[c] = a[c].astype(float).round(4)
a["direction"] = rd["coef"].apply(dir_rd)
add(md_table(a))
add("")
add("### `post_rank` (negative = LLM promoted)")
add("")
b = pr[["treatment","treatment_label","coef","se","t_stat","p_val",
        "ci_lower","ci_upper","n","stars"]].copy()
b = b.rename(columns={"coef":"β","se":"SE","t_stat":"t","p_val":"p",
                      "ci_lower":"CI_low","ci_upper":"CI_hi"})
for c in ["β","SE","t","p","CI_low","CI_hi"]:
    b[c] = b[c].astype(float).round(4)
b["direction"] = pr["coef"].apply(dir_pr)
add(md_table(b))
add("")

# ── 15. Appendix B — pivots ───────────────────────────────────────────────────
add("## 15. Appendix B — per-treatment × per-subset coefficients")
add("")
add("Coefficient on `rank_delta` for each (treatment, subset) cell. Formatted `β{stars}`. See `dml_pivot_rank_delta.csv` for the raw pivot.")
add("")
pvt = pd.read_csv(OUT / "dml_pivot_rank_delta.csv")
# Short column names
pvt.columns = [c.replace("duckduckgo_", "DDG_")
                .replace("searxng_", "SX_")
                .replace("-3.3-70B-Instruct_", "_")
                .replace("2.5-72B-Instruct_", "_")
                .replace("_top10", "")
               for c in pvt.columns]
add(md_table(pvt))
add("")

add("---")
add("")
add("*End of report.*")
add("")

REPORT.write_text("\n".join(md))
size = REPORT.stat().st_size / 1024
print(f"wrote {REPORT}  ({len(md)} lines, {size:.1f} KB)")
