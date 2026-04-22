"""Build a high-resolution markdown report from the 570-fit DML long CSV.

Sections:
  1. Ranking convention + what signs mean
  2. POOLED headline table (rank_delta) with direction + robustness
  3. Per-treatment detail  — one section per treatment showing all 15 subsets
  4. Cross-outcome consistency (rank_delta vs post_rank)
  5. Per-subset-type scoreboard

Ranking convention (GEODML):
    rank 1 = best position, the goal.
    pre_rank, post_rank: LOWER = BETTER (closer to #1).
    rank_delta = pre_rank - post_rank: POSITIVE = LLM promoted the page.

Coefficient interpretation:
    outcome = rank_delta → POSITIVE coef = treatment PROMOTES (GOOD).
    outcome = post_rank  → NEGATIVE coef = treatment PROMOTES (GOOD).
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("consolidated_results/dml_study")
long_df = pd.read_csv(OUT / "dml_results_long.csv")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if pd.isna(v): cells.append("")
            elif isinstance(v, float): cells.append(f"{v:.4f}")
            else: cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dir_arrow(coef, outcome):
    if pd.isna(coef): return ""
    if outcome == "rank_delta":
        return "↑ promotes" if coef > 0 else "↓ demotes"
    return "↑ promotes" if coef < 0 else "↓ demotes"


md = [
    "# DML Study — Paper-Size Experiment (high-resolution)",
    "",
    "## Setup",
    "",
    "- **Input:** `consolidated_results/regression_dataset.csv` (65,203 rows after `pre_rank` filter)",
    "- **Method:** Partial Linear Regression DML (DoubleML `DoubleMLPLR`), LightGBM nuisance learners (200 trees · depth 5 · lr 0.05) · 5-fold cross-fitting",
    "- **Confounders (17):** kw-title/snippet cosine, title/snippet len, brand recog, kw-in-title, word count, readability, internal/outbound links, img alt, BM25, HTTPS, domain authority, backlinks, referring domains, SERP position. Median-imputed and standardised.",
    "- **Subsets (15):** 8 individual runs · per-engine (DDG, SX) · per-model (Llama, Qwen) · per-pool (20, 50) · POOLED. Non-run subsets add run_id one-hot dummies as extra confounders.",
    "- **Treatments (19):** 4×T\\*\\_code, 4×T\\*\\_llm, 10×treat\\_\\*, plus `has_llms_txt`.",
    "- **Outcomes:** `rank_delta` (primary), `post_rank` (secondary).",
    "- **570 DML fits in 29 min.**",
    "",
    "## Ranking convention (critical)",
    "",
    "> `rank = 1` is the best position — the goal.",
    "> `pre_rank`, `post_rank`: **lower = better**.",
    "> `rank_delta = pre_rank − post_rank`: **positive = LLM promoted the page (GOOD)**.",
    "",
    "| Outcome | Coefficient sign that means PROMOTES (GOOD) |",
    "|---|---|",
    "| `rank_delta` | **positive** |",
    "| `post_rank`  | **negative** |",
    "",
    "Significance: * p<.10 · ** p<.05 · *** p<.01.",
    "",
]

# ── 2. Headline: POOLED rank_delta, sorted by |effect| among significant ─────
pooled = long_df[long_df.subset == "POOLED"].copy()
rd = pooled[pooled.outcome == "rank_delta"].copy()
rd["direction"] = rd["coef"].apply(lambda c: dir_arrow(c, "rank_delta"))
rd["abs_coef"] = rd["coef"].abs()
rd_sorted = rd.sort_values(["stars","abs_coef"], ascending=[True, False]) \
              .sort_values("p_val").reset_index(drop=True)

md += [
    "## 1. POOLED — rank_delta (all 65,203 rows, run FE included)",
    "",
    "Ordered by p-value.  direction column: `↑ promotes` = LLM *helps* pages with this treatment (closer to rank 1).",
    "",
]
tbl = rd_sorted[["treatment","treatment_label","coef","se","p_val",
                 "ci_lower","ci_upper","n","stars","direction"]] \
        .round(4) \
        .rename(columns={"coef":"β","se":"SE","p_val":"p",
                         "ci_lower":"CI_low","ci_upper":"CI_hi"})
md.append(md_table(tbl))
md.append("")

# ── 3. Robustness: direction agreement across the 8 runs ─────────────────────
md += [
    "## 2. Cross-run robustness (outcome = rank_delta)",
    "",
    "For each treatment: pooled β, how many of the 8 individual runs share its sign, and how many are significant at p<.05.",
    "",
]
rob_rows = []
for t in long_df.treatment.unique():
    sub = long_df[(long_df.treatment == t) & (long_df.outcome == "rank_delta")]
    poo = sub[sub.subset == "POOLED"]
    if poo.empty: continue
    pcoef = poo.coef.iloc[0]; pp = poo.p_val.iloc[0]
    run_rows = sub[sub.subset_type == "run"]
    same = int((np.sign(run_rows.coef) == np.sign(pcoef)).sum())
    sig  = int((run_rows.p_val < 0.05).sum())
    dir_lbl = dir_arrow(pcoef, "rank_delta")
    rob_rows.append({
        "treatment": t,
        "pooled_β": round(pcoef, 4),
        "pooled_p": round(pp, 4),
        "direction": dir_lbl,
        "same-sign 8 runs": f"{same}/8",
        "p<.05 8 runs":   f"{sig}/8",
    })
rob = pd.DataFrame(rob_rows).sort_values("pooled_p").reset_index(drop=True)
md.append(md_table(rob))
md.append("")

# ── 4. Subset scoreboard ─────────────────────────────────────────────────────
md += [
    "## 3. Per-subset-type scoreboard (outcome = rank_delta)",
    "",
    "How many treatments reach p<.05 in each non-run subset. Signs: # promoters (+ve β significant) vs # demoters (-ve β significant).",
    "",
]
score_rows = []
for label, sub in long_df[long_df.outcome == "rank_delta"].groupby("subset"):
    sig = sub[sub.p_val < 0.05]
    promoters = int((sig.coef > 0).sum())
    demoters  = int((sig.coef < 0).sum())
    score_rows.append({
        "subset": label,
        "subset_type": sub["subset_type"].iloc[0],
        "n_rows_in_subset": int(sub["n"].max()),
        "significant (p<.05)": f"{len(sig)}/{len(sub)}",
        "promoters": promoters,
        "demoters":  demoters,
    })
md.append(md_table(pd.DataFrame(score_rows)))
md.append("")

# ── 5. Cross-outcome consistency ─────────────────────────────────────────────
md += [
    "## 4. Cross-outcome consistency check",
    "",
    "A well-identified treatment should produce opposite signs on `rank_delta` and `post_rank`.  Any mismatch flags instability.",
    "",
]
con_rows = []
for t in long_df.treatment.unique():
    sub = long_df[(long_df.treatment == t) & (long_df.subset == "POOLED")]
    rd_ = sub[sub.outcome == "rank_delta"]; pr_ = sub[sub.outcome == "post_rank"]
    if rd_.empty or pr_.empty: continue
    r_c = rd_.coef.iloc[0]; p_c = pr_.coef.iloc[0]
    agree = "✓" if (np.sign(r_c) == -np.sign(p_c) and np.sign(r_c) != 0) else "✗"
    con_rows.append({
        "treatment": t,
        "β (rank_delta)": round(r_c, 4),
        "β (post_rank)":  round(p_c, 4),
        "signs mirror": agree,
    })
con = pd.DataFrame(con_rows).sort_values("treatment")
md.append(md_table(con))
md.append("")

# ── 6. Per-treatment detail: all 15 subsets ─────────────────────────────────
md += [
    "## 5. Per-treatment detail — all 15 subsets (outcome = rank_delta)",
    "",
    "One section per treatment ordered by absolute pooled effect.  Helps spot effect heterogeneity across engine/model/pool-size.",
    "",
]
order = rd_sorted["treatment"].tolist()
for t in order:
    sub = long_df[(long_df.treatment == t) & (long_df.outcome == "rank_delta")].copy()
    label = sub["treatment_label"].iloc[0]
    sub = sub[["subset","subset_type","coef","se","p_val","stars","n"]]
    sub["direction"] = sub["coef"].apply(lambda c: dir_arrow(c, "rank_delta"))
    sub = sub.rename(columns={"coef":"β","se":"SE","p_val":"p"})
    subset_order = {"pooled":0, "engine":1, "model":2, "pool":3, "run":4}
    sub["_ord"] = sub["subset_type"].map(subset_order)
    sub = sub.sort_values(["_ord","subset"]).drop(columns="_ord")
    sub = sub.round(4)
    md.append(f"### `{t}` — {label}")
    md.append("")
    md.append(md_table(sub))
    md.append("")

# ── 7. Key findings narrative ───────────────────────────────────────────────
md += [
    "## 6. Narrative — what the full picture says",
    "",
]

strong_promo = rd_sorted[(rd_sorted.p_val < 0.01) & (rd_sorted.coef > 0)]
strong_demo  = rd_sorted[(rd_sorted.p_val < 0.01) & (rd_sorted.coef < 0)]

md.append("### Treatments the LLM reliably rewards (pooled β > 0, p<.01)")
md.append("")
for _, r in strong_promo.iterrows():
    md.append(f"- **{r['treatment']}** ({r['treatment_label']}): β = {r['coef']:+.4f}, p = {r['p_val']:.4g}")
md.append("")

md.append("### Treatments the LLM reliably penalises (pooled β < 0, p<.01)")
md.append("")
for _, r in strong_demo.iterrows():
    md.append(f"- **{r['treatment']}** ({r['treatment_label']}): β = {r['coef']:+.4f}, p = {r['p_val']:.4g}")
md.append("")

# llms.txt spotlight
llms = long_df[(long_df.treatment == "T_llms_txt") & (long_df.outcome == "rank_delta")]
pooled_llms = llms[llms.subset == "POOLED"].iloc[0]
md.append("### `has_llms_txt` spotlight (the added treatment)")
md.append("")
md.append(f"- Pooled β = {pooled_llms.coef:+.4f} (p = {pooled_llms.p_val:.4g}).")
md.append(f"- Interpretation: serving `/llms.txt` is associated with a `{dir_arrow(pooled_llms.coef,'rank_delta')}` effect on rank_delta.")
md.append("")
md.append("Per-subgroup:")
md.append("")
llm_sub = llms[["subset","subset_type","coef","se","p_val","stars","n"]].copy()
llm_sub["direction"] = llm_sub["coef"].apply(lambda c: dir_arrow(c, "rank_delta"))
llm_sub = llm_sub.rename(columns={"coef":"β","se":"SE","p_val":"p"})
subset_order = {"pooled":0, "engine":1, "model":2, "pool":3, "run":4}
llm_sub["_ord"] = llm_sub["subset_type"].map(subset_order)
llm_sub = llm_sub.sort_values(["_ord","subset"]).drop(columns="_ord").round(4)
md.append(md_table(llm_sub))
md.append("")

md_path = OUT / "dml_summary.md"
md_path.write_text("\n".join(md))
print(f"wrote {md_path}  ({len(md)} lines)")

# Console preview
print("\n── POOLED rank_delta (top 10 by p) ──")
print(tbl.head(10).to_string(index=False))
print("\n── Cross-run robustness (top 10) ──")
print(rob.head(10).to_string(index=False))
print("\n── Per-subset-type scoreboard ──")
print(pd.DataFrame(score_rows).to_string(index=False))
