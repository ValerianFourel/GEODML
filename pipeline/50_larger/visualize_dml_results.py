#!/usr/bin/env python3
"""
GEODML 50_larger — DML Results Visualization Suite

Generates publication-quality plots from all DML experiment results.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#999999"
ORANGE = "#e08214"
GREEN = "#1b7837"
PURPLE = "#762a83"

TREATMENT_LABELS = {
    "T1": "T1: Statistical\nDensity",
    "T2": "T2: Question\nHeadings",
    "T3": "T3: Structured\nData",
    "T4": "T4: Citation\nAuthority",
}
TREATMENT_SHORT = {"T1": "Stat. Density", "T2": "Question Hdgs", "T3": "Struct. Data", "T4": "Citation Auth."}


def sig_color(p):
    if p < 0.01: return BLUE
    if p < 0.05: return RED
    if p < 0.1:  return ORANGE
    return GREY


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "\u2020"
    return ""


# ── Load data ────────────────────────────────────────────────────────────────
def load_all():
    dml_main = pd.read_csv(SCRIPT_DIR / "results" / "dml_results.csv")
    test32 = pd.read_csv(SCRIPT_DIR / "test" / "results" / "all_experiments.csv")
    test16 = pd.read_csv(SCRIPT_DIR / "test_diff" / "results" / "all_experiments.csv")
    diag_lgbm = pd.read_csv(SCRIPT_DIR / "test_full" / "results" / "full_diagnostics.csv")
    diag_rf = pd.read_csv(SCRIPT_DIR / "test_full_rf" / "results" / "full_diagnostics.csv")
    return dml_main, test32, test16, diag_lgbm, diag_rf


# ── Figure 1: Forest plot — rank_delta, all treatments, PLR ──────────────────
def fig1_forest_rank_delta(test16):
    """Forest plot of causal effects on rank_delta (PLR only)."""
    plr = test16[(test16["method"] == "PLR") & test16["coef"].notna()].copy()
    plr = plr.sort_values(["treatment", "path"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    for _, r in plr.iterrows():
        t_short = TREATMENT_SHORT[r["treatment"]]
        labels.append(f"{t_short} ({r['path']})")

    y_pos = np.arange(len(labels))
    coefs = plr["coef"].values
    ci_lo = plr["ci_lower"].values
    ci_hi = plr["ci_upper"].values
    pvals = plr["p_val"].values

    colors = [sig_color(p) for p in pvals]

    for i in range(len(labels)):
        ax.plot([ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]],
                color=colors[i], linewidth=2, solid_capstyle="round")
        ax.plot(coefs[i], y_pos[i], 'o', color=colors[i], markersize=8,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        stars = sig_label(pvals[i])
        if stars:
            ax.annotate(stars, (ci_hi[i] + 0.3, y_pos[i]), fontsize=10,
                        fontweight="bold", color=colors[i], va="center")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Causal Effect on rank_delta (theta)\n"
                   "<-- LLM demotes more | LLM promotes more -->", fontsize=11)
    ax.set_title("DML Causal Estimates: Treatment Effects on LLM Re-Ranking\n"
                 "Y = rank_delta (pre_rank - post_rank), PLR method, 50 SERP / top-20 re-rank",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(color=BLUE, label="p < 0.01"),
        mpatches.Patch(color=RED, label="p < 0.05"),
        mpatches.Patch(color=ORANGE, label="p < 0.10"),
        mpatches.Patch(color=GREY, label="n.s."),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_forest_rank_delta.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig1_forest_rank_delta.png'}")


# ── Figure 2: Heatmap — all outcomes × treatments × methods ─────────────────
def fig2_heatmap_all(test32, test16):
    """P-value heatmap across all experiment configurations."""
    combined = pd.concat([test16, test32], ignore_index=True)
    valid = combined[combined["coef"].notna()].copy()

    outcomes = ["rank_delta", "pre_rank", "post_rank"]
    methods = ["PLR", "IRM"]
    treatments = ["T1", "T2", "T3", "T4"]
    paths = ["code", "llm"]

    col_labels = []
    for o in outcomes:
        for m in methods:
            col_labels.append(f"{o}\n({m})")

    row_labels = []
    for t in treatments:
        for p in paths:
            row_labels.append(f"{TREATMENT_SHORT[t]}\n({p})")

    pval_matrix = np.full((len(row_labels), len(col_labels)), np.nan)
    coef_matrix = np.full((len(row_labels), len(col_labels)), np.nan)

    for _, r in valid.iterrows():
        ri = treatments.index(r["treatment"]) * 2 + paths.index(r["path"])
        ci = outcomes.index(r["outcome"]) * 2 + methods.index(r["method"])
        pval_matrix[ri, ci] = r["p_val"]
        coef_matrix[ri, ci] = r["coef"]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pval_matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if not np.isnan(pval_matrix[i, j]):
                p = pval_matrix[i, j]
                c = coef_matrix[i, j]
                stars = sig_label(p)
                text = f"{c:+.2f}\n{stars}" if stars else f"{c:+.2f}"
                color = "white" if p < 0.15 or p > 0.85 else "black"
                weight = "bold" if p < 0.05 else "normal"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color=color, fontweight=weight)

    # Grid lines
    for i in range(1, len(row_labels)):
        if i % 2 == 0:
            ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(1, len(col_labels)):
        if j % 2 == 0:
            ax.axvline(j - 0.5, color="white", linewidth=2)

    ax.set_title("DML P-value Heatmap: All Experiments (50_larger)\n"
                 "Coefficient shown in each cell; *** p<0.001, ** p<0.01, * p<0.05, \u2020 p<0.1",
                 fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_heatmap_all.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig2_heatmap_all.png'}")


# ── Figure 3: Code vs LLM measurement comparison ────────────────────────────
def fig3_code_vs_llm(test16):
    """Side-by-side comparison of code-based vs LLM-based treatment measurement."""
    plr = test16[(test16["method"] == "PLR") & test16["coef"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    treatments = ["T1", "T2", "T3", "T4"]
    x = np.arange(len(treatments))
    width = 0.35

    code_coefs, code_ci_lo, code_ci_hi, code_pvals = [], [], [], []
    llm_coefs, llm_ci_lo, llm_ci_hi, llm_pvals = [], [], [], []

    for t in treatments:
        c = plr[(plr["treatment"] == t) & (plr["path"] == "code")]
        l = plr[(plr["treatment"] == t) & (plr["path"] == "llm")]
        if len(c) == 1:
            code_coefs.append(c.iloc[0]["coef"])
            code_ci_lo.append(c.iloc[0]["coef"] - c.iloc[0]["ci_lower"])
            code_ci_hi.append(c.iloc[0]["ci_upper"] - c.iloc[0]["coef"])
            code_pvals.append(c.iloc[0]["p_val"])
        else:
            code_coefs.append(0)
            code_ci_lo.append(0)
            code_ci_hi.append(0)
            code_pvals.append(1)
        if len(l) == 1:
            llm_coefs.append(l.iloc[0]["coef"])
            llm_ci_lo.append(l.iloc[0]["coef"] - l.iloc[0]["ci_lower"])
            llm_ci_hi.append(l.iloc[0]["ci_upper"] - l.iloc[0]["coef"])
            llm_pvals.append(l.iloc[0]["p_val"])
        else:
            llm_coefs.append(0)
            llm_ci_lo.append(0)
            llm_ci_hi.append(0)
            llm_pvals.append(1)

    code_colors = [sig_color(p) for p in code_pvals]
    llm_colors = [sig_color(p) for p in llm_pvals]

    bars1 = ax.bar(x - width/2, code_coefs, width,
                   yerr=[code_ci_lo, code_ci_hi],
                   color=code_colors, alpha=0.8,
                   edgecolor="black", linewidth=0.5, capsize=4, ecolor="black",
                   label="Code-based")
    bars2 = ax.bar(x + width/2, llm_coefs, width,
                   yerr=[llm_ci_lo, llm_ci_hi],
                   color=llm_colors, alpha=0.8,
                   edgecolor="black", linewidth=0.5, capsize=4, ecolor="black",
                   label="LLM-based", hatch="///")

    # Add significance stars
    for i, (cp, lp) in enumerate(zip(code_pvals, llm_pvals)):
        cs = sig_label(cp)
        ls = sig_label(lp)
        if cs:
            ax.annotate(cs, (x[i] - width/2, code_coefs[i]),
                        ha="center", va="bottom" if code_coefs[i] >= 0 else "top",
                        fontsize=11, fontweight="bold", color=sig_color(cp))
        if ls:
            ax.annotate(ls, (x[i] + width/2, llm_coefs[i]),
                        ha="center", va="bottom" if llm_coefs[i] >= 0 else "top",
                        fontsize=11, fontweight="bold", color=sig_color(lp))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([TREATMENT_SHORT[t] for t in treatments], fontsize=11)
    ax.set_ylabel("Causal Effect (theta)", fontsize=11)
    ax.set_title("Code-based vs LLM-based Treatment Measurement\n"
                 "Y = rank_delta, PLR method | Solid = code, Hatched = LLM",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Code-based"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="LLM-based"),
        mpatches.Patch(color=BLUE, label="p < 0.01"),
        mpatches.Patch(color=RED, label="p < 0.05"),
        mpatches.Patch(color=ORANGE, label="p < 0.10"),
        mpatches.Patch(color=GREY, label="n.s."),
    ]
    ax.legend(handles=patches, loc="lower left", framealpha=0.9, ncol=2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_code_vs_llm.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig3_code_vs_llm.png'}")


# ── Figure 4: DML vs OLS scatter ────────────────────────────────────────────
def fig4_dml_vs_ols(diag_lgbm):
    """Scatter: DML causal estimate vs naive OLS, by outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    outcomes = ["rank_delta", "pre_rank", "post_rank"]
    outcome_titles = {"rank_delta": "Y = rank_delta",
                      "pre_rank": "Y = pre_rank",
                      "post_rank": "Y = post_rank"}

    for idx, y_name in enumerate(outcomes):
        ax = axes[idx]
        sub = diag_lgbm[diag_lgbm["outcome"] == y_name].copy()

        colors = [sig_color(p) for p in sub["dml_pval"]]
        sizes = [120 if p < 0.05 else 80 for p in sub["dml_pval"]]

        ax.scatter(sub["ols_coef"], sub["dml_coef"], c=colors, s=sizes,
                   edgecolors="black", linewidth=0.5, zorder=3)

        for _, r in sub.iterrows():
            label = f"{r['treatment']}_{r['path']}"
            offset = (5, 5) if r["dml_coef"] >= 0 else (5, -10)
            ax.annotate(label, (r["ols_coef"], r["dml_coef"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=offset, textcoords="offset points")

        lim = max(abs(sub["ols_coef"]).max(), abs(sub["dml_coef"]).max()) * 1.4
        if lim > 0:
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7, alpha=0.4,
                    label="45-degree line")
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.axvline(0, color="gray", linewidth=0.4)
        ax.set_xlabel("OLS beta (naive)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("DML theta (causal)", fontsize=10)
        ax.set_title(outcome_titles[y_name], fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("DML Causal Estimates vs Naive OLS (LGBM nuisance)\n"
                 "Points on 45-degree line = DML agrees with OLS",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_dml_vs_ols.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig4_dml_vs_ols.png'}")


# ── Figure 5: PLR vs IRM sensitivity ────────────────────────────────────────
def fig5_plr_vs_irm(test16):
    """Scatter: PLR coef vs IRM coef for sensitivity analysis."""
    valid = test16[test16["coef"].notna()].copy()

    fig, ax = plt.subplots(figsize=(8, 7))

    for _, r in valid.iterrows():
        if r["method"] == "PLR":
            # Find matching IRM
            irm = valid[(valid["treatment"] == r["treatment"]) &
                        (valid["path"] == r["path"]) &
                        (valid["method"] == "IRM")]
            if len(irm) == 1:
                plr_c = r["coef"]
                irm_c = irm.iloc[0]["coef"]
                p_plr = r["p_val"]
                color = sig_color(p_plr)
                label_text = f"{r['treatment']}_{r['path']}"

                ax.scatter(plr_c, irm_c, c=[color], s=100,
                           edgecolors="black", linewidth=0.5, zorder=3)
                ax.annotate(label_text, (plr_c, irm_c),
                            fontsize=8, ha="left", va="bottom",
                            xytext=(5, 5), textcoords="offset points")

    lim_data = valid["coef"].abs().max() * 1.5
    ax.plot([-lim_data, lim_data], [-lim_data, lim_data], "k--",
            linewidth=0.7, alpha=0.4, label="Perfect agreement")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)

    ax.set_xlabel("PLR theta (Partially Linear Regression)", fontsize=11)
    ax.set_ylabel("IRM theta (Interactive Regression Model)", fontsize=11)
    ax.set_title("Method Sensitivity: PLR vs IRM\n"
                 "Y = rank_delta | Points near diagonal = methods agree",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(color=BLUE, label="PLR p < 0.01"),
        mpatches.Patch(color=RED, label="PLR p < 0.05"),
        mpatches.Patch(color=ORANGE, label="PLR p < 0.10"),
        mpatches.Patch(color=GREY, label="PLR n.s."),
    ]
    ax.legend(handles=patches, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_plr_vs_irm.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig5_plr_vs_irm.png'}")


# ── Figure 6: LGBM vs RF sensitivity ────────────────────────────────────────
def fig6_lgbm_vs_rf(diag_lgbm, diag_rf):
    """Scatter: LGBM DML coef vs RF DML coef for nuisance learner sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    outcomes = ["rank_delta", "pre_rank", "post_rank"]

    for idx, y_name in enumerate(outcomes):
        ax = axes[idx]
        lgbm = diag_lgbm[diag_lgbm["outcome"] == y_name].copy()
        rf = diag_rf[diag_rf["outcome"] == y_name].copy()

        merged = lgbm.merge(rf, on=["outcome", "treatment", "path"],
                            suffixes=("_lgbm", "_rf"))

        colors = [sig_color(p) for p in merged["dml_pval_lgbm"]]

        ax.scatter(merged["dml_coef_lgbm"], merged["dml_coef_rf"], c=colors,
                   s=80, edgecolors="black", linewidth=0.5, zorder=3)

        for _, r in merged.iterrows():
            ax.annotate(f"{r['treatment']}_{r['path']}",
                        (r["dml_coef_lgbm"], r["dml_coef_rf"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")

        lim = max(merged["dml_coef_lgbm"].abs().max(),
                  merged["dml_coef_rf"].abs().max()) * 1.3
        if lim > 0:
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7, alpha=0.4)
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.axvline(0, color="gray", linewidth=0.4)
        ax.set_xlabel("LGBM theta", fontsize=10)
        if idx == 0:
            ax.set_ylabel("RF theta", fontsize=10)
        ax.set_title(f"Y = {y_name}", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Nuisance Learner Sensitivity: LGBM vs Random Forest\n"
                 "Points near diagonal = robust to learner choice",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig6_lgbm_vs_rf.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig6_lgbm_vs_rf.png'}")


# ── Figure 7: Nuisance R2 diagnostic ────────────────────────────────────────
def fig7_nuisance_r2(diag_lgbm):
    """Nuisance model cross-validated R2 — diagnostic for DML validity."""
    rd = diag_lgbm[diag_lgbm["outcome"] == "rank_delta"].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"{r['treatment']}_{r['path']}" for _, r in rd.iterrows()]
    y_pos = np.arange(len(labels))
    w = 0.35

    ax.barh(y_pos - w/2, rd["r2_Y_X"], height=w, color="#4393c3", alpha=0.8,
            edgecolor="black", linewidth=0.3, label="R2(Y~X) outcome model")
    ax.barh(y_pos + w/2, rd["r2_D_X"], height=w, color="#d6604d", alpha=0.8,
            edgecolor="black", linewidth=0.3, label="R2(D~X) treatment model")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cross-validated R2", fontsize=11)
    ax.set_title("Nuisance Model Performance (LGBM, Y = rank_delta)\n"
                 "Negative R2 = confounders predict worse than mean",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_nuisance_r2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig7_nuisance_r2.png'}")


# ── Figure 8: Multi-outcome forest plot ──────────────────────────────────────
def fig8_multi_outcome_forest(test32, test16):
    """Three-panel forest plot: rank_delta, pre_rank, post_rank (PLR only)."""
    combined = pd.concat([test16, test32], ignore_index=True)
    plr = combined[(combined["method"] == "PLR") & combined["coef"].notna()].copy()

    outcomes = ["rank_delta", "pre_rank", "post_rank"]
    outcome_titles = {
        "rank_delta": "Y = rank_delta\n(pre - post, +ve = promoted)",
        "pre_rank": "Y = pre_rank\n(SERP position, lower = better)",
        "post_rank": "Y = post_rank\n(LLM position, lower = better)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, y_name in enumerate(outcomes):
        ax = axes[idx]
        sub = plr[plr["outcome"] == y_name].sort_values(["treatment", "path"],
                                                         ascending=[True, False])
        labels = [f"{TREATMENT_SHORT[r['treatment']]} ({r['path']})"
                  for _, r in sub.iterrows()]
        y_pos = np.arange(len(labels))
        coefs = sub["coef"].values
        ci_lo = sub["ci_lower"].values
        ci_hi = sub["ci_upper"].values
        pvals = sub["p_val"].values
        colors = [sig_color(p) for p in pvals]

        for i in range(len(labels)):
            ax.plot([ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]],
                    color=colors[i], linewidth=2, solid_capstyle="round")
            ax.plot(coefs[i], y_pos[i], 'o', color=colors[i], markersize=7,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)
            stars = sig_label(pvals[i])
            if stars:
                offset = max(ci_hi[i], coefs[i]) + 0.2
                ax.annotate(stars, (offset, y_pos[i]), fontsize=9,
                            fontweight="bold", color=colors[i], va="center")

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels if idx == 0 else [], fontsize=9)
        ax.set_xlabel("theta", fontsize=10)
        ax.set_title(outcome_titles[y_name], fontsize=11)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(color=BLUE, label="p < 0.01"),
        mpatches.Patch(color=RED, label="p < 0.05"),
        mpatches.Patch(color=ORANGE, label="p < 0.10"),
        mpatches.Patch(color=GREY, label="n.s."),
    ]
    axes[2].legend(handles=patches, loc="lower right", framealpha=0.9, fontsize=8)

    fig.suptitle("DML Causal Estimates Across All Outcomes (PLR, 50_larger)\n"
                 "95% CIs shown; *** p<0.001, ** p<0.01, * p<0.05, \u2020 p<0.1",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_multi_outcome_forest.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig8_multi_outcome_forest.png'}")


# ── Figure 9: Summary table as figure ───────────────────────────────────────
def fig9_summary_table(test16, diag_lgbm):
    """Publication-style summary table rendered as a figure."""
    plr = test16[(test16["method"] == "PLR") & test16["coef"].notna()].copy()
    plr = plr.sort_values(["treatment", "path"], ascending=[True, False])

    rd_diag = diag_lgbm[diag_lgbm["outcome"] == "rank_delta"].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    columns = ["Treatment", "Path", "N", "DML theta", "SE", "p-value", "Sig",
               "95% CI", "OLS beta", "OLS p"]
    rows_data = []

    for _, r in plr.iterrows():
        diag_match = rd_diag[(rd_diag["treatment"] == r["treatment"]) &
                             (rd_diag["path"] == r["path"])]
        ols_coef = f"{diag_match.iloc[0]['ols_coef']:+.3f}" if len(diag_match) > 0 else "-"
        ols_p = f"{diag_match.iloc[0]['ols_pval']:.3f}" if len(diag_match) > 0 else "-"

        ci_str = f"[{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]"
        stars = sig_label(r["p_val"])

        rows_data.append([
            TREATMENT_SHORT[r["treatment"]],
            r["path"],
            str(int(r["n_obs"])),
            f"{r['coef']:+.3f}",
            f"{r['se']:.3f}",
            f"{r['p_val']:.4f}",
            stars if stars else "",
            ci_str,
            ols_coef,
            ols_p,
        ])

    table = ax.table(cellText=rows_data, colLabels=columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#2166ac")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight significant rows
    for i, row in enumerate(rows_data):
        p_val = float(row[5])
        if p_val < 0.01:
            for j in range(len(columns)):
                table[i + 1, j].set_facecolor("#d1e5f0")
        elif p_val < 0.05:
            for j in range(len(columns)):
                table[i + 1, j].set_facecolor("#e0ecf4")
        elif p_val < 0.1:
            for j in range(len(columns)):
                table[i + 1, j].set_facecolor("#fff7bc")

    ax.set_title("Summary: DML Causal Estimates on rank_delta (PLR, 50_larger)\n"
                 "Highlighted: blue = p<0.01, light blue = p<0.05, yellow = p<0.10",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig9_summary_table.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'fig9_summary_table.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading results...")
    dml_main, test32, test16, diag_lgbm, diag_rf = load_all()
    print(f"  DML main:   {len(dml_main)} rows")
    print(f"  Test 32:    {len(test32)} rows")
    print(f"  Test 16:    {len(test16)} rows")
    print(f"  Diag LGBM:  {len(diag_lgbm)} rows")
    print(f"  Diag RF:    {len(diag_rf)} rows")
    print()

    fig1_forest_rank_delta(test16)
    fig2_heatmap_all(test32, test16)
    fig3_code_vs_llm(test16)
    fig4_dml_vs_ols(diag_lgbm)
    fig5_plr_vs_irm(test16)
    fig6_lgbm_vs_rf(diag_lgbm, diag_rf)
    fig7_nuisance_r2(diag_lgbm)
    fig8_multi_outcome_forest(test32, test16)
    fig9_summary_table(test16, diag_lgbm)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
