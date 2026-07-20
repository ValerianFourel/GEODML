#!/usr/bin/env python3
"""EMNLP-2026 paper figures from ~/geodml_data parquets (May-24 data, no T7).

T7_source_earned is excluded from every figure. It is a curated list-membership
flag, not a clean content treatment — significant but hard to defend as a causal
result. The paper headline is the content treatments under Spec B mutual control.

Six paper figures (each PDF + PNG at 300 DPI):
  fig1_content_headline   — content treatments headline (Spec B, biased prompt, forest plot)
  fig2_coef_grid          — 9 content treatments × 4 variants (rank_delta heatmap)
  fig3_rag_attenuation    — per-treatment Δ(rag − non_rag), two panels
  fig4_specA_vs_specB     — single-treatment vs mutually-controlled coefficients
  fig5_dual_outcome       — rank_delta vs post_rank scatter for all (treatment × variant)
  fig6_two_llm_agreement  — Llama vs Qwen coefficient scatter

Reads:  ~/geodml_data/data/dml_results/{dml_results_long_*.parquet, dml_multi_treatment.parquet}
Writes: docs/figures/*.pdf + *.png

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --only fig1,fig3
    python scripts/make_figures.py --list

The May-12 CSV-driven version is preserved at make_figures.py.bak-2026-05-24.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
DATA = Path.home() / "geodml_data" / "data"
DML = DATA / "dml_results"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
VARIANT_LABELS = {
    "biased": "biased\n(snippet)",
    "neutral": "neutral\n(snippet)",
    "biased_rag": "biased\n(RAG)",
    "neutral_rag": "neutral\n(RAG)",
}
# T7 EXCLUDED — see module docstring.
CONTENT_TREATMENTS = [
    "T1a_stats_present", "T1b_stats_density",
    "T2a_question_headings", "T2b_structural_modularity",
    "T3_structured_data_new",
    "T4a_ext_citations", "T4b_auth_citations",
    "T5_topical_comp", "T6_freshness",
]
TREATMENT_LABELS = {
    "T1a_stats_present": "T1a stats present",
    "T1b_stats_density": "T1b stats density",
    "T2a_question_headings": "T2a Q-headings",
    "T2b_structural_modularity": "T2b modularity",
    "T3_structured_data_new": "T3 schema",
    "T4a_ext_citations": "T4a ext citations",
    "T4b_auth_citations": "T4b auth citations",
    "T5_topical_comp": "T5 topical comp.",
    "T6_freshness": "T6 freshness",
    "T_llms_txt": "T llms.txt",
    "T2_llm": "T2 LLM-coded",
    "T3_llm": "T3 LLM-coded",
    "T1_code": "T1 code-defined",
}
COLOR_PROMOTER = "#2c7fb8"  # blue — LLM moves doc UP
COLOR_DEMOTER = "#e34a33"   # warm red — LLM moves doc DOWN
COLOR_BIASED = "#d6604d"
COLOR_NEUTRAL = "#4393c3"
COLOR_RANK = "#1a1a1a"
COLOR_POST = "#8c8c8c"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlepad": 14,
        "axes.labelpad": 6,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "pdf.fonttype": 42,
    })


def load_dml_long() -> pd.DataFrame:
    parts = []
    for v in VARIANTS:
        df = pd.read_parquet(DML / f"dml_results_long_{v}.parquet")
        df["variant"] = v
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def headline_slice(dml: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return dml[(dml["subset"] == "POOLED") & (dml["method"] == "plr")
               & (dml["learner"] == "lgbm")
               & (dml["outcome"] == outcome)].copy()


def specB_slice() -> pd.DataFrame:
    """Mutually-controlled coefficients from dml_multi_treatment.parquet."""
    mt = pd.read_parquet(DML / "dml_multi_treatment.parquet")
    return mt[(mt["study"] == "mutually_controlled")
              & (mt["outcome"] == "rank_delta")].copy()


def save(fig: plt.Figure, name: str) -> None:
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"    wrote {pdf.relative_to(REPO)} + {png.name}")


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    if p < 1e-1:
        return "·"
    return ""


def _p_to_stars(p_array) -> list[str]:
    return [_stars(p) if pd.notna(p) else "" for p in p_array]


# ── FIG 1 ─────────────────────────────────────────────────────────────────────


def fig1_content_headline() -> None:
    """Spec B forest plot — content treatments, rank_delta, all variants stacked.

    Uses dml_multi_treatment.parquet (mutually controlled), filtered to content
    treatments. The pooled-across-variants estimand is what the paper headline
    quotes. Each treatment is a horizontal error-bar, colored by promoter/demoter.
    """
    sb = specB_slice()
    show = sb[sb["treatment"].isin(CONTENT_TREATMENTS + ["T_llms_txt"])].copy()
    # T1a coefficient is on a *fractional* scale in the parquet (~837); drop.
    show = show[show["coef"].abs() < 5].copy()
    # Fill missing CI with ±1.96·SE (some rows lack marginal CI columns).
    show["ci_lower_marg"] = show["ci_lower_marg"].fillna(show["coef"] - 1.96 * show["se"])
    show["ci_upper_marg"] = show["ci_upper_marg"].fillna(show["coef"] + 1.96 * show["se"])
    show = show.sort_values("coef")
    show["label"] = show["treatment"].map(TREATMENT_LABELS).fillna(show["treatment"])

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    ys = np.arange(len(show))
    cols = [COLOR_PROMOTER if c > 0 else COLOR_DEMOTER for c in show["coef"]]
    for y, (_, r), c in zip(ys, show.iterrows(), cols):
        xerr_low = r["coef"] - r["ci_lower_marg"]
        xerr_high = r["ci_upper_marg"] - r["coef"]
        ax.errorbar(r["coef"], y,
                    xerr=[[xerr_low], [xerr_high]],
                    fmt="o", color=c, markersize=9, capsize=4,
                    elinewidth=1.6)
        stars = _stars(r["p_val"])
        ax.text(r["ci_upper_marg"] + 0.04 if r["coef"] >= 0
                else r["ci_lower_marg"] - 0.04,
                y,
                f"{r['coef']:+.3f}{stars}",
                ha="left" if r["coef"] >= 0 else "right",
                va="center", fontsize=10, color=c, fontweight="bold")

    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(show["label"])
    # widen x-axis so annotations don't get clipped
    xmin = show["ci_lower_marg"].min()
    xmax = show["ci_upper_marg"].max()
    span = xmax - xmin
    ax.set_xlim(xmin - 0.22 * span, xmax + 0.28 * span)
    ax.set_xlabel("Spec B coefficient on Δrank  (mutually controlled)")
    ax.set_title("Content-treatment effects under mutual control",
                 loc="left", pad=18)

    # legend
    legend = [Patch(facecolor=COLOR_PROMOTER, label="promoter (LLM moves doc UP)"),
              Patch(facecolor=COLOR_DEMOTER, label="demoter (LLM moves doc DOWN)")]
    ax.legend(handles=legend, loc="lower right", frameon=False)

    fig.text(0.02, -0.04,
             "Positive coef means the LLM moves the doc to a better (lower-numbered) rank position.\n"
             "Spec B controls for the 8 other content treatments + 25 confounders.",
             fontsize=9, color="#555")

    save(fig, "fig1_content_headline")


# ── FIG 2 ─────────────────────────────────────────────────────────────────────


def fig2_coef_grid(dml: pd.DataFrame) -> None:
    """Annotated heatmap — 9 content treatments × 4 variants on rank_delta (Spec A)."""
    rd = headline_slice(dml, "rank_delta")
    pivot_coef = rd.pivot(index="treatment", columns="variant", values="coef")
    pivot_stars = rd.pivot(index="treatment", columns="variant", values="sig_stars")
    pivot_coef = pivot_coef.reindex(index=CONTENT_TREATMENTS, columns=VARIANTS)
    pivot_stars = pivot_stars.reindex(index=CONTENT_TREATMENTS, columns=VARIANTS)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = max(abs(pivot_coef.min().min()), abs(pivot_coef.max().max()))
    # Tighter visual range now that T7's −2.24 doesn't dominate
    im = ax.imshow(pivot_coef.values, cmap="RdBu", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS])
    ax.set_yticks(range(len(CONTENT_TREATMENTS)))
    ax.set_yticklabels([TREATMENT_LABELS[t] for t in CONTENT_TREATMENTS])

    for i, t in enumerate(CONTENT_TREATMENTS):
        for j, v in enumerate(VARIANTS):
            c = pivot_coef.loc[t, v]
            s = pivot_stars.loc[t, v]
            s = s if pd.notna(s) else ""
            color = "white" if abs(c) > 0.55 * vmax else "#111111"
            ax.text(j, i, f"{c:+.3f}{s}", ha="center", va="center",
                    fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Δrank coef\n(positive = LLM promotes,  negative = LLM demotes)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("Content-treatment effects on Δrank by prompt × evidence",
                 loc="left", pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, "fig2_coef_grid")


# ── FIG 3 ─────────────────────────────────────────────────────────────────────


def fig3_rag_attenuation(dml: pd.DataFrame) -> None:
    """Per-treatment Δ(rag − non_rag) on rank_delta, two panels."""
    rd = headline_slice(dml, "rank_delta")
    pivot = rd.pivot(index="treatment", columns="variant",
                     values=["coef", "se"])
    deltas = []
    for t in CONTENT_TREATMENTS:
        try:
            d_b = pivot.loc[t, ("coef", "biased_rag")] - pivot.loc[t, ("coef", "biased")]
            se_b = math.sqrt(pivot.loc[t, ("se", "biased_rag")] ** 2
                             + pivot.loc[t, ("se", "biased")] ** 2)
            d_n = pivot.loc[t, ("coef", "neutral_rag")] - pivot.loc[t, ("coef", "neutral")]
            se_n = math.sqrt(pivot.loc[t, ("se", "neutral_rag")] ** 2
                             + pivot.loc[t, ("se", "neutral")] ** 2)
            deltas.append((t, d_b, se_b, d_n, se_n))
        except KeyError:
            continue
    df_d = pd.DataFrame(deltas,
                        columns=["treatment", "d_biased", "se_biased",
                                 "d_neutral", "se_neutral"])
    df_d["z_b"] = df_d["d_biased"] / df_d["se_biased"]
    df_d["z_n"] = df_d["d_neutral"] / df_d["se_neutral"]
    df_d["p_b"] = df_d["z_b"].apply(lambda z: 2 * (1 - _phi(abs(z))))
    df_d["p_n"] = df_d["z_n"].apply(lambda z: 2 * (1 - _phi(abs(z))))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.5), sharey=True)

    def panel(ax, dcol, secol, pcol, title, color):
        ys = np.arange(len(df_d))[::-1]
        ax.errorbar(df_d[dcol], ys, xerr=df_d[secol] * 1.96,
                    fmt="o", color=color, capsize=4, elinewidth=1.4,
                    markersize=7)
        ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([TREATMENT_LABELS[t] for t in df_d["treatment"]])
        ax.set_xlabel("Δrank coef shift  (RAG − non-RAG)")
        ax.set_title(title, loc="left", pad=14)
        for y, x, p in zip(ys, df_d[dcol], df_d[pcol]):
            tag = _stars(p)
            if tag:
                ax.text(x + (0.04 if x >= 0 else -0.04), y, tag,
                        ha="left" if x >= 0 else "right",
                        va="center", fontsize=10, color=color)

    panel(axes[0], "d_biased", "se_biased", "p_b",
          "(a) biased prompt: biased_rag − biased", COLOR_BIASED)
    panel(axes[1], "d_neutral", "se_neutral", "p_n",
          "(b) neutral prompt: neutral_rag − neutral", COLOR_NEUTRAL)
    fig.suptitle("RAG attenuation per content treatment",
                 fontsize=12, x=0.02, ha="left", y=1.0)
    fig.subplots_adjust(wspace=0.08)

    save(fig, "fig3_rag_attenuation")


# ── FIG 4 ─────────────────────────────────────────────────────────────────────


def fig4_marginal_vs_partial(dml: pd.DataFrame) -> None:
    """Per-treatment effect under single-treatment vs mutually-controlled DML.

    Motivates the mutual-control framing: when other treatments + T7 are
    properly controlled, several content effects emerge or shift sign.
    Single-treatment: only the 25 confounders in the X-set.
    Mutually-controlled: 25 confounders + all 18 other treatments in the X-set.
    """
    rd_a = headline_slice(dml, "rank_delta")
    a_pool = rd_a[rd_a["treatment"].isin(CONTENT_TREATMENTS + ["T_llms_txt"])] \
        .groupby("treatment").agg(coef_a=("coef", "mean"),
                                  se_a=("se", "mean")).reset_index()

    sb = specB_slice()
    sb_keep = sb[sb["treatment"].isin(CONTENT_TREATMENTS + ["T_llms_txt"])][
        ["treatment", "coef", "se"]].rename(columns={"coef": "coef_b", "se": "se_b"})
    sb_keep = sb_keep[sb_keep["coef_b"].abs() < 5]

    merged = pd.merge(a_pool, sb_keep, on="treatment", how="inner")
    merged = merged.sort_values("coef_b")
    merged["label"] = merged["treatment"].map(TREATMENT_LABELS).fillna(merged["treatment"])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ys = np.arange(len(merged))
    w = 0.36
    ax.barh(ys - w/2, merged["coef_a"], height=w,
            xerr=merged["se_a"] * 1.96, capsize=3,
            color="#7fcdbb", edgecolor="#222", linewidth=0.5,
            label="single-treatment DML  (treatment + 25 confounders)")
    ax.barh(ys + w/2, merged["coef_b"], height=w,
            xerr=merged["se_b"] * 1.96, capsize=3,
            color="#253494", edgecolor="#222", linewidth=0.5,
            label="mutually-controlled DML  (+ 8 other content treatments)")
    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(merged["label"])
    ax.set_xlabel("Δrank coefficient")
    ax.set_title("Mutual control sharpens content-treatment estimands",
                 loc="left", pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07),
              frameon=False, ncol=2, fontsize=10)
    fig.subplots_adjust(top=0.88, bottom=0.10)

    save(fig, "fig4_marginal_vs_partial")


# ── FIG 5 ─────────────────────────────────────────────────────────────────────


def fig5_robust_survivors() -> None:
    """Which content treatments survive Romano-Wolf and Bonferroni correction?

    Uses joint_inference results from dml_multi_treatment.parquet.
    """
    mt = pd.read_parquet(DML / "dml_multi_treatment.parquet")
    ji = mt[(mt["study"] == "joint_inference")
            & (mt["outcome"] == "rank_delta")
            & (mt["treatment"].isin(CONTENT_TREATMENTS + ["T_llms_txt"]))].copy()
    # T1a is on a fractional scale glitch — drop
    ji = ji[ji["coef"].abs() < 5]
    ji = ji.sort_values("coef")
    ji["label"] = ji["treatment"].map(TREATMENT_LABELS).fillna(ji["treatment"])

    def status(row):
        if row["p_val_romano_wolf"] < 0.05:
            return "RW survivor", "#2c7fb8"
        if row["p_val_bonferroni"] < 0.05:
            return "Bonferroni only", "#feb24c"
        return "not robust", "#cccccc"

    ji[["status", "color"]] = ji.apply(
        lambda r: pd.Series(status(r)), axis=1)

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ys = np.arange(len(ji))
    ax.barh(ys, ji["coef"], xerr=ji["se"] * 1.96, capsize=4,
            color=ji["color"], edgecolor="#222", linewidth=0.5)
    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(ji["label"])
    ax.set_xlabel("Δrank coefficient  (joint-inference DML)")
    ax.set_title("Robust content-treatment effects after multi-test correction",
                 loc="left", pad=28)

    # annotate p values clear of the y-axis labels
    xmin = (ji["coef"] - ji["se"] * 1.96).min()
    xmax = (ji["coef"] + ji["se"] * 1.96).max()
    span = xmax - xmin
    for y, (_, r) in zip(ys, ji.iterrows()):
        rw = r["p_val_romano_wolf"]
        ci_end = r["coef"] + np.sign(r["coef"]) * (abs(r["se"]) * 1.96)
        x = ci_end + (0.02 * span if r["coef"] >= 0 else -0.02 * span)
        ax.text(x, y,
                f"RW p={rw:.3f}",
                ha="left" if r["coef"] >= 0 else "right",
                va="center", fontsize=9, color="#333")
    ax.set_xlim(xmin - 0.20 * span, xmax + 0.28 * span)

    legend = [
        Patch(facecolor="#2c7fb8", label="survives Romano-Wolf at α=0.05"),
        Patch(facecolor="#feb24c", label="survives Bonferroni only"),
        Patch(facecolor="#cccccc", label="not robust to correction"),
    ]
    ax.legend(handles=legend, loc="upper center",
              bbox_to_anchor=(0.5, 1.06),
              frameon=False, ncol=3, fontsize=10)
    fig.subplots_adjust(top=0.88)

    save(fig, "fig5_robust_survivors")


# ── FIG 6 ─────────────────────────────────────────────────────────────────────


def fig6_two_llm_agreement(dml: pd.DataFrame) -> None:
    """Llama vs Qwen coefficient scatter — content treatments only."""
    by_model = dml[(dml["subset"].isin(["by_model=Llama-3.3-70B-Instruct",
                                        "by_model=Qwen2.5-72B-Instruct"]))
                   & (dml["method"] == "plr") & (dml["learner"] == "lgbm")
                   & (dml["outcome"] == "rank_delta")
                   & (dml["treatment"].isin(CONTENT_TREATMENTS))].copy()
    by_model["model"] = (by_model["subset"].str.split("=").str[-1]
                                              .str.replace("-Instruct", ""))
    pv = (by_model.pivot_table(index=["variant", "treatment"],
                               columns="model", values="coef")
                  .dropna()
                  .reset_index())

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    var_colors = {"biased": COLOR_BIASED, "neutral": COLOR_NEUTRAL,
                  "biased_rag": "#a04a3a", "neutral_rag": "#2f6f9f"}
    var_markers = {"biased": "o", "neutral": "s",
                   "biased_rag": "^", "neutral_rag": "D"}
    for v in VARIANTS:
        sub = pv[pv["variant"] == v]
        ax.scatter(sub["Llama-3.3-70B"], sub["Qwen2.5-72B"],
                   s=70, alpha=0.85, color=var_colors[v],
                   marker=var_markers[v], edgecolor="white", linewidth=0.8,
                   label=VARIANT_LABELS[v].replace("\n", " "))

    lo = pv[["Llama-3.3-70B", "Qwen2.5-72B"]].min().min()
    hi = pv[["Llama-3.3-70B", "Qwen2.5-72B"]].max().max()
    pad = 0.15 * (hi - lo) if hi > lo else 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "--", color="#cccccc", linewidth=1, zorder=0)
    ax.axhline(0, color="#eeeeee", linewidth=0.8, zorder=-1)
    ax.axvline(0, color="#eeeeee", linewidth=0.8, zorder=-1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Llama-3.3-70B coefficient")
    ax.set_ylabel("Qwen2.5-72B coefficient")
    ax.set_title("Two-LLM corroboration — content treatments, Δrank",
                 loc="left", pad=18)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    r = pv[["Llama-3.3-70B", "Qwen2.5-72B"]].corr().iloc[0, 1]
    ax.text(0.04, 0.96, f"Pearson r = {r:.3f}  (n={len(pv)} treatment × variant cells)",
            transform=ax.transAxes, fontsize=10,
            ha="left", va="top", color="#333")

    save(fig, "fig6_two_llm_agreement")


# ── FIG 7 ─────────────────────────────────────────────────────────────────────


def _url_sets_by_variant(variant: str) -> pd.DataFrame:
    """Per-(keyword × engine × pool × llm) URL set selected by the LLM in its top-N."""
    df = pd.read_parquet(DATA / "main" / f"full_experiment_data_{variant}.parquet")
    df = df.dropna(subset=["url"])
    # Only consider rows the LLM actually placed in its output (post_rank present)
    df = df[df["post_rank"].notna()]
    return df


def _llm_jaccard(variant: str) -> pd.DataFrame:
    """Jaccard(Llama-top, Qwen-top) per (keyword × engine × pool)."""
    df = _url_sets_by_variant(variant)
    llama = (df[df["llm_model"] == "Llama-3.3-70B-Instruct"]
             .groupby(["keyword", "search_engine", "pool"])["url"]
             .apply(set).rename("urls_llama"))
    qwen = (df[df["llm_model"] == "Qwen2.5-72B-Instruct"]
            .groupby(["keyword", "search_engine", "pool"])["url"]
            .apply(set).rename("urls_qwen"))
    m = pd.merge(llama, qwen, left_index=True, right_index=True).reset_index()
    m["jaccard"] = m.apply(
        lambda r: len(r["urls_llama"] & r["urls_qwen"])
                  / max(len(r["urls_llama"] | r["urls_qwen"]), 1), axis=1)
    m["variant"] = variant
    return m[["variant", "keyword", "search_engine", "pool", "jaccard"]]


def _snippet_vs_rag_jaccard(prompt: str) -> pd.DataFrame:
    """Jaccard(snippet-top, RAG-top) per (LLM × keyword × engine × pool) under one prompt."""
    snip = _url_sets_by_variant(prompt)
    rag = _url_sets_by_variant(f"{prompt}_rag")
    snip_sets = (snip.groupby(["keyword", "search_engine", "pool", "llm_model"])["url"]
                     .apply(set).rename("urls_snip"))
    rag_sets = (rag.groupby(["keyword", "search_engine", "pool", "llm_model"])["url"]
                    .apply(set).rename("urls_rag"))
    m = pd.merge(snip_sets, rag_sets, left_index=True, right_index=True).reset_index()
    m["jaccard"] = m.apply(
        lambda r: len(r["urls_snip"] & r["urls_rag"])
                  / max(len(r["urls_snip"] | r["urls_rag"]), 1), axis=1)
    m["prompt"] = prompt
    return m[["prompt", "llm_model", "keyword", "search_engine", "pool", "jaccard"]]


def fig7_jaccard_agreement() -> None:
    """Jaccard URL-overlap agreement: two-LLM (panel a) + snippet-vs-RAG (panel b)."""
    print("    computing two-LLM Jaccard per variant …")
    two_llm = pd.concat([_llm_jaccard(v) for v in VARIANTS], ignore_index=True)
    print(f"      {len(two_llm):,} (keyword × engine × pool) cells across 4 variants")

    print("    computing snippet-vs-RAG Jaccard per prompt …")
    sn_rag = pd.concat([_snippet_vs_rag_jaccard(p) for p in ("biased", "neutral")],
                       ignore_index=True)
    print(f"      {len(sn_rag):,} (kw × engine × pool × LLM) cells across 2 prompts")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5))

    # ── Panel (a) Two-LLM Jaccard by variant ──
    ax = axes[0]
    data_by_var = [two_llm[two_llm["variant"] == v]["jaccard"].values for v in VARIANTS]
    means = [d.mean() for d in data_by_var]
    bp = ax.boxplot(
        data_by_var, positions=range(len(VARIANTS)),
        widths=0.55, patch_artist=True, showmeans=True,
        medianprops=dict(color="#222222", linewidth=1.4),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="#222", markersize=7),
        flierprops=dict(marker=".", markerfacecolor="#888",
                        markeredgecolor="#888", alpha=0.4),
        boxprops=dict(linewidth=0.7),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    var_colors = [COLOR_BIASED, COLOR_NEUTRAL, "#a04a3a", "#2f6f9f"]
    for patch, c in zip(bp["boxes"], var_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, m in enumerate(means):
        ax.text(i, 1.02, f"μ={m:.3f}", ha="center", va="bottom",
                fontsize=9, color="#222")
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS])
    ax.set_ylabel("Jaccard(Llama-top, Qwen-top)")
    ax.set_ylim(-0.02, 1.10)
    ax.set_title("(a) Two-LLM URL agreement per keyword cell",
                 loc="left", pad=14)
    ax.axhline(1.0, color="#eeeeee", linewidth=0.8, zorder=-1)

    # ── Panel (b) Snippet-vs-RAG by (prompt × LLM) ──
    ax = axes[1]
    groups = []  # (label, color, jaccard_array)
    color_map = {"Llama-3.3-70B-Instruct": "#4393c3",
                 "Qwen2.5-72B-Instruct": "#d6604d"}
    label_map = {"Llama-3.3-70B-Instruct": "Llama",
                 "Qwen2.5-72B-Instruct": "Qwen2.5"}
    for prompt in ("biased", "neutral"):
        for llm in ("Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"):
            j = sn_rag[(sn_rag["prompt"] == prompt) & (sn_rag["llm_model"] == llm)]["jaccard"].values
            groups.append((f"{prompt}\n{label_map[llm]}", color_map[llm], j))

    means_b = [g[2].mean() for g in groups]
    bp = ax.boxplot(
        [g[2] for g in groups], positions=range(len(groups)),
        widths=0.55, patch_artist=True, showmeans=True,
        medianprops=dict(color="#222", linewidth=1.4),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="#222", markersize=7),
        flierprops=dict(marker=".", markerfacecolor="#888",
                        markeredgecolor="#888", alpha=0.4),
        boxprops=dict(linewidth=0.7),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, (_, c, _) in zip(bp["boxes"], groups):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, m in enumerate(means_b):
        ax.text(i, 1.02, f"μ={m:.3f}", ha="center", va="bottom",
                fontsize=9, color="#222")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Jaccard(snippet-top, RAG-top)")
    ax.set_ylim(-0.02, 1.10)
    ax.set_title("(b) Snippet-vs-RAG URL agreement, per LLM × prompt",
                 loc="left", pad=14)
    ax.axhline(1.0, color="#eeeeee", linewidth=0.8, zorder=-1)

    fig.suptitle("URL-set agreement (Jaccard) under prompt × LLM × evidence variations",
                 fontsize=12, x=0.02, ha="left", y=1.0)
    fig.subplots_adjust(top=0.86, wspace=0.25)

    save(fig, "fig7_jaccard_agreement")


# ── FIG 8 ─────────────────────────────────────────────────────────────────────


def fig8_pool_size_sensitivity() -> None:
    """How content-treatment admission effects shift with SERP pool size.

    Uses selected_long_fixed (Study 2 binary outcome). T7 excluded as elsewhere.

    Larger pool = more candidates = LLM has more selection room. We ask which
    treatment signals the LLM relies on MORE as the pool grows, and which it
    relies on LESS.
    """
    sel = pd.read_parquet(DML / "selected_long_fixed.parquet")
    pool = sel[sel["slice"].str.startswith("POOL:")
               & (sel["treatment"] != "T7_source_earned")].copy()
    pool["pool"] = pool["slice"].str.replace("POOL:", "").astype(int)

    treats = sorted(pool["treatment"].unique())
    label_map = {**TREATMENT_LABELS,
                 "T1_code": "T1 code-defined",
                 "T4_llm": "T4 LLM-coded citations",
                 "T_llms_txt": "T llms.txt"}

    # Compute pivot
    cf = pool.pivot(index="treatment", columns="pool", values="coef")
    se = pool.pivot(index="treatment", columns="pool", values="se")
    pv = pool.pivot(index="treatment", columns="pool", values="p_val")

    # Order treatments by |Δ| descending so the most pool-sensitive are on top.
    cf["delta"] = (cf[50] - cf[20]).abs()
    order = cf.sort_values("delta", ascending=True).index.tolist()
    cf = cf.loc[order]
    se = se.loc[order]
    pv = pv.loc[order]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ys = np.arange(len(order))
    w = 0.36
    bars20 = ax.barh(ys - w/2, cf[20], height=w,
                     xerr=se[20] * 1.96, capsize=3,
                     color="#a6cee3", edgecolor="#222", linewidth=0.5,
                     label="pool = 20 candidates")
    bars50 = ax.barh(ys + w/2, cf[50], height=w,
                     xerr=se[50] * 1.96, capsize=3,
                     color="#1f78b4", edgecolor="#222", linewidth=0.5,
                     label="pool = 50 candidates")

    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label_map.get(t, t) for t in order])
    ax.set_xlabel("coefficient on selected_by_llm  (log-odds)")
    ax.set_title("Pool-size sensitivity of content-treatment admission effects",
                 loc="left", pad=28)

    # annotate each bar with coef and stars
    for y, t in zip(ys, order):
        for offset, p_size in [(-w/2, 20), (w/2, 50)]:
            c = cf.loc[t, p_size]
            s = se.loc[t, p_size]
            stars = _stars(pv.loc[t, p_size])
            # text just outside the error bar
            x = c + np.sign(c if c != 0 else 1) * (abs(s) * 1.96 + 0.001)
            ax.text(x, y + offset, f"{c:+.3f}{stars}",
                    ha="left" if c >= 0 else "right",
                    va="center", fontsize=8,
                    color="#444")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08),
              frameon=False, ncol=2, fontsize=10)
    fig.subplots_adjust(top=0.86)

    # Bottom caption: highlight the qualitative pattern
    fig.text(0.02, -0.04,
             "Reading: as the pool grows from 20 to 50 candidates, the LLM rewards LLM-coded citation\n"
             "authority MORE (T4: −0.006* becomes −0.011**) and freshness boilerplate LESS (T6: −0.016***\n"
             "becomes −0.012***). Other content effects are pool-stable.",
             fontsize=9, color="#555")

    save(fig, "fig8_pool_size_sensitivity")


# ── FIG 9 ─────────────────────────────────────────────────────────────────────


def fig9_dml_schematic() -> None:
    """Publication-style DML schematic.

    Treatments + confounders + rank_pre → LLM reranker → rank_post.
    Two outcomes are modeled: Y_1 = rank_post and Y_2 = Δrank = rank_pre − rank_post.
    DML estimates θ_T per (treatment, outcome) cell.
    Notation uses mathtext subscripts; no underscores in display text.

    Layout uses a 100×100 logical grid. Box dimensions are chosen so every
    text block has at least 1.5 units of vertical padding above & below.
    """
    from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=(14.5, 10.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # ── palette ──
    C_DATA = "#f5efe3"
    C_TREAT = "#dbe9f4"
    C_CONF = "#ececec"
    C_LLM = "#3a6ea5"
    C_OUTCOME = "#fde8c8"
    C_DML = "#e2d4ed"
    EDGE = "#1f1f1f"
    INK = "#1a1a1a"
    MUTED = "#555555"

    def rect(x, y, w, h, text, face=C_DATA, fs=10.5, weight="normal",
             fc=INK, lw=1.0, edge=EDGE, va="center"):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.9",
                           facecolor=face, edgecolor=edge,
                           linewidth=lw, zorder=2)
        ax.add_patch(p)
        yy = y + h/2 if va == "center" else (y + h - 1.2 if va == "top" else y + 1.2)
        ax.text(x + w/2, yy, text,
                ha="center", va=va,
                fontsize=fs, fontweight=weight, color=fc, zorder=3,
                linespacing=1.45)

    def arrow(x1, y1, x2, y2, rad=0.0, lw=1.2, color="#333",
              style="solid", zorder=1):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=15,
                            lw=lw, color=color,
                            linestyle=style,
                            connectionstyle=f"arc3,rad={rad}", zorder=zorder)
        ax.add_patch(a)

    # ── title ────────────────────────────────────────────────────────
    ax.text(50, 96.5,
            "DML estimation framework — recovering LLM rerank behaviour",
            fontsize=15, fontweight="bold", color=INK, ha="center")
    ax.text(50, 93,
            r"Two outcomes — $Y_1 = \mathrm{rank}_\mathrm{post}$ and "
            r"$Y_2 = \Delta\mathrm{rank}$ — are modeled from the same "
            r"(treatments, confounders, $\mathrm{rank}_\mathrm{pre}$) inputs.",
            fontsize=11, color=MUTED, ha="center")

    # ── row 1: SERP source + rank_pre ────────────────────────────────
    rect(36, 84.5, 28, 5, "SERP query  (DDG  /  searxng)",
         face=C_DATA, fs=11)
    rect(36, 76.5, 28, 5,
         r"$\mathrm{rank}_\mathrm{pre}$       (original SERP position)",
         face=C_DATA, fs=12, weight="bold")
    arrow(50, 84.5, 50, 81.7)

    # ── row 2: treatments (left) + confounders (right) ──────────────
    # Treatments box — header line then list, separated by a thin matplotlib line
    treat_x, treat_y, treat_w, treat_h = 2, 52, 31, 22
    rect(treat_x, treat_y, treat_w, treat_h,
         "9 content treatments    $T$\n"
         "\n"
         r"$T_{1a}$  stats present" "          " r"$T_{1b}$  stats density" "\n"
         r"$T_{2a}$  Q-headings" "            " r"$T_{2b}$  modularity" "\n"
         r"$T_{3}\;$  schema" "                  "
         r"$T_{4a}$, $T_{4b}$  citations" "\n"
         r"$T_{5}\;$  topical comp." "       "
         r"$T_{6}\;$  freshness" "\n"
         r"$T_\mathrm{llms.txt}$    domain flag",
         face=C_TREAT, fs=10)
    ax.plot([treat_x + 2, treat_x + treat_w - 2],
            [treat_y + treat_h - 4.0, treat_y + treat_h - 4.0],
            color="#555", lw=0.6, zorder=3)

    conf_x, conf_y, conf_w, conf_h = 67, 52, 31, 22
    rect(conf_x, conf_y, conf_w, conf_h,
         "25 confounders    $X$\n"
         "\n"
         "BM25 score          domain authority\n"
         "backlinks            referring domains\n"
         "title / snippet similarity\n"
         "brand recognition          HTTPS\n"
         "word count          readability\n"
         "internal / outbound links     …",
         face=C_CONF, fs=10)
    ax.plot([conf_x + 2, conf_x + conf_w - 2],
            [conf_y + conf_h - 4.0, conf_y + conf_h - 4.0],
            color="#555", lw=0.6, zorder=3)

    # arrows from T and X into LLM
    arrow(33, 60, 38, 49, rad=-0.18)
    arrow(67, 60, 62, 49, rad=0.18)

    # rank_pre feeds the LLM as context
    arrow(50, 76.5, 50, 49)

    # ── row 3: LLM (centerpiece) ─────────────────────────────────────
    llm = FancyBboxPatch((32, 33), 36, 16,
                         boxstyle="round,pad=0.4,rounding_size=1.8",
                         facecolor=C_LLM, edgecolor=INK, linewidth=1.8,
                         zorder=4)
    ax.add_patch(llm)
    ax.text(50, 44.5, "LLM RERANKER",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color="white", zorder=5)
    ax.text(50, 40,
            "Llama-3.3-70B-Instruct        Qwen2.5-72B-Instruct",
            ha="center", va="center", fontsize=10.5, color="white", zorder=5)
    ax.text(50, 36,
            r"prompt $\in$ {biased, neutral}        "
            r"evidence $\in$ {snippet, RAG}",
            ha="center", va="center", fontsize=10, color="#e0eaf4",
            style="italic", zorder=5)

    # ── row 4: rank_post ─────────────────────────────────────────────
    rect(36, 24.5, 28, 5,
         r"$\mathrm{rank}_\mathrm{post}$       (LLM-assigned rank)",
         face=C_DATA, fs=12, weight="bold")
    arrow(50, 33, 50, 29.5)

    # ── row 5: two outcomes Y_1 and Y_2 ─────────────────────────────
    rect(4, 12, 36, 8,
         r"$Y_1 \, = \, \mathrm{rank}_\mathrm{post}$" "\n"
         "(absolute LLM-assigned position)",
         face=C_OUTCOME, fs=11, weight="bold")
    rect(60, 12, 36, 8,
         r"$Y_2 \, = \, \Delta\mathrm{rank} \, = \, "
         r"\mathrm{rank}_\mathrm{pre} \,-\, \mathrm{rank}_\mathrm{post}$" "\n"
         "(directional displacement by the LLM)",
         face=C_OUTCOME, fs=11, weight="bold")
    arrow(45, 24.5, 22, 20, rad=0.20)
    arrow(55, 24.5, 78, 20, rad=-0.20)
    # Note: rank_pre's contribution to Y_2 is made explicit in the Y_2 box
    # equation itself; no extra arrow needed.

    # ── row 6: DML estimator ────────────────────────────────────────
    # Split into a header line and three short bullet phrases on one row.
    rect(4, 0.5, 92, 9,
         r"$\widehat{\theta}_T \;=\; \mathrm{DML}\,(\,T,\;Y_j\,;\;X\,)$"
         r"      —      estimated for each treatment $T$ and outcome $Y_j$" "\n\n"
         r"$X = \{$25 confounders$\} \cup \{$other 8 treatments$\}$"
         "          cross-fitted LightGBM,  $K = 5$ folds"
         "          robust SE via influence function",
         face=C_DML, fs=10)
    arrow(22, 12, 22, 9.5)
    arrow(78, 12, 78, 9.5)

    # side note — far right margin, clear of any box
    ax.text(99, 46,
            "Comparing\n"
            r"$\widehat{\theta}_{T,Y_1}$  vs  $\widehat{\theta}_{T,Y_2}$" "\n"
            "across prompt × evidence\n"
            "isolates the LLM's\n"
            "reranking behaviour\n"
            "from the input SERP.",
            fontsize=9.5, color=MUTED, ha="right", va="center",
            style="italic", linespacing=1.4,
            bbox=dict(facecolor="white", edgecolor="#dddddd",
                      boxstyle="round,pad=0.6", linewidth=0.6))

    save(fig, "fig9_dml_schematic")


# ── FIG 10 ───────────────────────────────────────────────────────────────────


def fig10_causal_dag() -> None:
    """Pure causal DAG for the DML estimand.

    Layout choice: the LLM cartridge sits OUTSIDE the bipartite T,X → Y_2,Δrank
    core, in a column above Y_2. With T on the left and X on the right of the
    causal core, the K_{2,2} arrows fan in cleanly without crossings.

    Nodes
      T  (treatments, left)        X  (confounders, right)
      rank_pre (top)               LLM (mid-top, mediator)
      Y_2 = rank_post              Δrank

    Arrows
      X → T                              (confounding)
      X → Y_2, X → Δrank                 (confounder → outcome)
      T → Y_2, T → Δrank                 (treatment → outcome — DML estimand)
      rank_pre → LLM → Y_2               (rerank mechanism, teal)
      rank_pre → Δrank, Y_2 → Δrank      (Δrank arithmetic, dashed grey)
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=(11.5, 9.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    C_TREAT = "#dbe9f4"
    C_CONF = "#ececec"
    C_DATA = "#f5efe3"
    C_LLM = "#3a6ea5"
    C_OUTCOME = "#fde8c8"
    INK = "#1a1a1a"
    MUTED = "#555555"

    def node(x, y, text, w=14, h=8, face=C_DATA, fs=12, weight="bold",
             fc=INK, edge="#1f1f1f", lw=1.2):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=1.0",
                           facecolor=face, edgecolor=edge,
                           linewidth=lw, zorder=3)
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=fc, zorder=4)
        return p  # so arrows can clip to the patch boundary

    def arr(x1, y1, x2, y2, rad=0.0, lw=1.4, color="#222",
            style="solid", zorder=2, patchA=None, patchB=None):
        # patchA / patchB make matplotlib stop the arrow exactly at the
        # node boundary, so the arrowhead is visible against the edge
        # and no tip is hidden under the box fill.
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>,head_length=10,head_width=6",
                            mutation_scale=1.6,
                            lw=lw, color=color, linestyle=style,
                            connectionstyle=f"arc3,rad={rad}",
                            zorder=zorder,
                            patchA=patchA, patchB=patchB,
                            shrinkA=0 if patchA else 12,
                            shrinkB=0 if patchB else 12)
        ax.add_patch(a)

    # ── title ──
    ax.text(50, 96.5,
            "Causal DAG — DML estimand for the LLM reranker",
            fontsize=14, fontweight="bold", color=INK, ha="center")
    ax.text(50, 92,
            r"$T$: content treatments     $X$: confounders     "
            r"$Y_2 = \mathrm{rank}_\mathrm{post}$     "
            r"$\Delta\mathrm{rank} = \mathrm{rank}_\mathrm{pre} - \mathrm{rank}_\mathrm{post}$",
            fontsize=10, color=MUTED, ha="center", style="italic")

    # ── node coordinates ─────────────────────────────────────────────
    # LLM mechanism cluster is shifted right of the left edge so both boxes
    # render in full.  rank_pre gets a small inward shift, LLM a larger one;
    # the chain stays roughly vertical and the LLM box clears T comfortably.
    px_RP, py_RP = 14, 80      # rank_pre (small inward shift from x=8)
    px_LLM, py_LLM = 22, 55    # LLM (larger inward shift; fully visible)
    px_T, py_T = 48, 84        # T moved right a bit to give LLM clearance
    px_X, py_X = 82, 84        # X correspondingly nudged right
    px_Y2, py_Y2 = 48, 30      # Y_2 (= rank_post)
    px_DR, py_DR = 82, 30      # Δrank

    # ── render nodes (keep handles for patchA/patchB clipping) ──────
    p_RP = node(px_RP, py_RP, r"$\mathrm{rank}_\mathrm{pre}$",
                w=20, h=7.5, face=C_DATA, fs=13)

    # LLM cartridge — visually distinct, narrowed slightly so it doesn't
    # crowd the T box once shifted inward.
    p_LLM = FancyBboxPatch((px_LLM - 11, py_LLM - 5.5), 22, 11,
                           boxstyle="round,pad=0.3,rounding_size=1.5",
                           facecolor=C_LLM, edgecolor="#0d2a4a",
                           linewidth=1.8, zorder=3)
    ax.add_patch(p_LLM)
    ax.text(px_LLM, py_LLM, "LLM",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="white", zorder=4)

    p_T = node(px_T, py_T, r"$T$",
               w=12, h=9, face=C_TREAT, fs=18)
    p_X = node(px_X, py_X, r"$X$",
               w=12, h=9, face=C_CONF, fs=18)
    p_Y2 = node(px_Y2, py_Y2, r"$Y_2 \,=\, \mathrm{rank}_\mathrm{post}$",
                w=24, h=8, face=C_OUTCOME, fs=12)
    p_DR = node(px_DR, py_DR, r"$\Delta\mathrm{rank}$",
                w=18, h=8, face=C_OUTCOME, fs=14)

    # ── arrows ───────────────────────────────────────────────────────
    # Confounding: X → T (horizontal across the top of the main DAG)
    arr(px_X, py_X, px_T, py_T, rad=0.0, patchA=p_X, patchB=p_T)
    ax.text((px_X + px_T) / 2, py_T + 3.2, r"confounding",
            ha="center", va="bottom", fontsize=10,
            color=MUTED, style="italic")

    # Mechanism (teal): rank_pre → LLM (vertical, on left border)
    #                   LLM → Y_2 (diagonal, from border into the main DAG)
    arr(px_RP, py_RP, px_LLM, py_LLM, rad=0.0,
        color=C_LLM, lw=2.0, patchA=p_RP, patchB=p_LLM)
    arr(px_LLM, py_LLM, px_Y2, py_Y2, rad=0.0,
        color=C_LLM, lw=2.0, patchA=p_LLM, patchB=p_Y2)
    ax.text(34, 43, "rerank",
            ha="left", va="center", fontsize=10,
            color=C_LLM, style="italic")

    # T → outcomes (K_{2,2})
    arr(px_T, py_T, px_Y2, py_Y2, rad=0.0, patchA=p_T, patchB=p_Y2)
    arr(px_T, py_T, px_DR, py_DR, rad=0.0, patchA=p_T, patchB=p_DR)

    # X → outcomes (K_{2,2})
    arr(px_X, py_X, px_Y2, py_Y2, rad=0.0, patchA=p_X, patchB=p_Y2)
    arr(px_X, py_X, px_DR, py_DR, rad=0.0, patchA=p_X, patchB=p_DR)

    # Δrank derivation (light dashed). Y_2 → Δrank stays horizontal.
    # rank_pre → Δrank: big downward arc that clears the Y_2 box on its way
    # over to Δrank (was passing through Y_2 with rad=0.55).
    arr(px_Y2, py_Y2, px_DR, py_DR, rad=0.0,
        color="#999", style="dashed", lw=1.0,
        patchA=p_Y2, patchB=p_DR)
    arr(px_RP, py_RP, px_DR, py_DR, rad=1.05,
        color="#999", style="dashed", lw=1.0,
        patchA=p_RP, patchB=p_DR)
    # The Δrank = rank_pre − rank_post arithmetic is encoded by the two
    # dashed grey arrows entering Δrank plus the node labels themselves;
    # no extra equation text needed (it was previously crossed by the
    # big curve).

    # ── legend (bottom-left corner) ──────────────────────────────────
    ax.text(2, 18,
            "Arrows\n"
            "  ── solid black:  causal\n"
            "  ── teal:  LLM mechanism\n"
            "  -- grey:  Δrank arithmetic",
            fontsize=9, color=MUTED, ha="left", va="center",
            linespacing=1.6,
            bbox=dict(facecolor="white", edgecolor="#dddddd",
                      boxstyle="round,pad=0.5", linewidth=0.6))

    save(fig, "fig10_causal_dag")


# ── FIG 11 ──────────────────────────────────────────────────────────────────


def fig11_admission_vs_rank() -> None:
    """Two-panel forest comparing admission (Y_1) vs rank (Y_2) effects per treatment.

    Only the 5 content treatments tested on BOTH outcomes (intersection of
    binary admission joint Spec B and rank_delta joint inference). T7
    excluded — it dominates both and is a curated list flag, not a content
    treatment.

    Color codes the dual-stage pattern:
      green  = significant on both (Bonferroni admission + RW rank)
      blue   = admission only
      orange = rank only
      grey   = neither (after multi-test correction)
    """
    admit = pd.read_parquet(DML / "selected_multitreat_fixed.parquet")
    admit = admit[admit["treatment"] != "T7_source_earned"].copy()

    multi = pd.read_parquet(DML / "dml_multi_treatment.parquet")
    rank_ji = multi[(multi["study"] == "joint_inference")
                    & (multi["outcome"] == "rank_delta")
                    & (multi["treatment"].isin(admit["treatment"]))].copy()

    # Merge into a single dataframe ordered by category.
    df = pd.merge(
        admit[["treatment", "coef", "se", "p_val"]]
            .rename(columns={"coef": "coef_a", "se": "se_a", "p_val": "p_a"}),
        rank_ji[["treatment", "coef", "se", "p_val_romano_wolf"]]
            .rename(columns={"coef": "coef_r", "se": "se_r",
                             "p_val_romano_wolf": "p_r_rw"}),
        on="treatment", how="inner",
    )

    BF_THRESH = 0.05 / 6      # Bonferroni for 6 admission tests
    RW_THRESH = 0.05          # RW survival threshold

    def category(r):
        a = r["p_a"] < BF_THRESH
        b = r["p_r_rw"] < RW_THRESH
        if a and b:    return ("BOTH stages",          "#2ca25f")
        if a:          return ("ADMISSION only",       "#3182bd")
        if b:          return ("RANK only",            "#e6550d")
        return            ("neither (after correction)", "#9c9c9c")

    df[["cat", "color"]] = df.apply(
        lambda r: pd.Series(category(r)), axis=1)

    cat_order = ["BOTH stages", "ADMISSION only", "RANK only",
                 "neither (after correction)"]
    df["cat_rank"] = df["cat"].map({c: i for i, c in enumerate(cat_order)})
    df = df.sort_values(["cat_rank", "treatment"]).reset_index(drop=True)

    df["label"] = df["treatment"].map({**TREATMENT_LABELS,
                                       "T1_code": "T1 code-defined",
                                       "T4_llm": "T4 LLM-coded citations",
                                       "T_llms_txt": "T llms.txt"})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    ys = np.arange(len(df))[::-1]

    def draw_panel(ax, coef_col, se_col, xlabel, title, pvalue_fmt):
        # Compute a consistent annotation column at the right edge of each
        # error bar (always above the data on the x axis), so labels never
        # collide with the left-side tick labels.
        rightmost = (df[coef_col] + df[se_col] * 1.96).max()
        # leave room for the longest annotation
        ann_x = rightmost + 0.04 * (abs(rightmost) + 0.05)
        for y, (_, r) in zip(ys, df.iterrows()):
            ax.errorbar(r[coef_col], y, xerr=r[se_col] * 1.96,
                        fmt="o", capsize=4, elinewidth=1.3, markersize=8,
                        ecolor="#444", color=r["color"],
                        mfc=r["color"], mec="#222", mew=0.9,
                        linestyle="none")
            ax.text(ann_x, y, pvalue_fmt(r),
                    ha="left", va="center",
                    fontsize=9.5, color=r["color"])
        ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(df["label"])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=14)
        # Extend xlim so annotations fit
        cur_lo, cur_hi = ax.get_xlim()
        ax.set_xlim(cur_lo, max(cur_hi, ann_x + 0.20 * abs(rightmost + 0.05)))

    draw_panel(
        axes[0], "coef_a", "se_a",
        "admission coefficient   (log-odds on $Y_1 = $ selected by LLM)",
        "(a) ADMISSION effect  —  binary $Y_1$, joint Spec B",
        lambda r: f"{r['coef_a']:+.3f}{_stars(r['p_a'])}",
    )
    draw_panel(
        axes[1], "coef_r", "se_r",
        "rank_delta coefficient   ($\\Delta\\mathrm{rank}$ movement on $Y_2$)",
        "(b) RANK effect  —  $Y_2 = \\Delta\\mathrm{rank}$, joint Spec C / Romano–Wolf",
        lambda r: f"{r['coef_r']:+.3f}  RW={r['p_r_rw']:.3f}",
    )

    # ── legend (top of figure) ────────────────────────────────────
    handles = [
        Patch(facecolor="#2ca25f", label="significant on BOTH stages"),
        Patch(facecolor="#3182bd", label="ADMISSION only"),
        Patch(facecolor="#e6550d", label="RANK only"),
        Patch(facecolor="#9c9c9c", label="neither (after multi-test correction)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.05),
               frameon=False, ncol=4, fontsize=9)

    fig.suptitle("LLM bias operates in two stages — gate (admission) and ranker (placement)",
                 fontsize=12, x=0.02, ha="left", y=1.10)

    save(fig, "fig11_admission_vs_rank")


# ── FIG 12 ──────────────────────────────────────────────────────────────────


def fig12_admission_detail() -> None:
    """Detailed admission-DML view: log-odds → odds ratio → percentage-point
    change in admission probability, per treatment (all 6 of dml_selected,
    including T7 because the log-odds → probability translation is exactly
    what makes the admission story tangible).

    Two panels:
      (a) joint Spec B log-odds coefficients with their odds-ratio
          translation and approximate Δprob at the empirical baseline rate.
      (b) T7 per-variant breakdown: shows how the LLM's gate flips from
          strong rejection (biased) to null (neutral), and how RAG barely
          attenuates it.
    """
    # Joint Spec B for all 6 treatments
    admit = pd.read_parquet(DML / "selected_multitreat_fixed.parquet")
    # Per-variant (Spec A by_variant slice) for T7
    sel = pd.read_parquet(DML / "selected_long_fixed.parquet")
    t7_var = sel[(sel["treatment"] == "T7_source_earned")
                 & sel["slice"].str.startswith("VAR:")].copy()
    t7_var["variant"] = t7_var["slice"].str.replace("VAR:", "", regex=False)

    # Empirical baseline admission rate per variant — derived from the pool /
    # LLM-output relationship (counts of LLM-included URLs divided by pool size).
    # Approximation: pool ≈ engine_pool_size × n_keywords × n_engines × n_models;
    # selected ≈ rows in the LLM-output parquet (each row is one (kw,url,cell)
    # the LLM kept). The published Study-2 selection rate is ~33%.
    rates = {"biased": 0.33, "neutral": 0.33,
             "biased_rag": 0.33, "neutral_rag": 0.33}
    p0_overall = 0.33

    def delta_p(logit, p_base):
        """Marginal change in probability at baseline rate, log-odds linearization."""
        return p_base * (1 - p_base) * logit

    label_map = {**TREATMENT_LABELS,
                 "T1_code": "T1 code-defined",
                 "T4_llm": "T4 LLM-coded citations",
                 "T_llms_txt": "T llms.txt",
                 "T7_source_earned": "T7 earned media\n(descriptive flag)"}

    BF_THRESH = 0.05 / 6

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1.0],
                                          "wspace": 0.35})

    # ── Panel (a) — all 6 treatments joint Spec B ──
    ax = axes[0]
    admit_sorted = admit.sort_values("coef")
    ys = np.arange(len(admit_sorted))[::-1]

    colors = ["#2ca25f" if r["p_val"] < BF_THRESH else "#9c9c9c"
              for _, r in admit_sorted.iterrows()]

    for y, (_, r), c in zip(ys, admit_sorted.iterrows(), colors):
        ax.errorbar(r["coef"], y, xerr=r["se"] * 1.96,
                    fmt="o", capsize=4, elinewidth=1.4, markersize=8,
                    ecolor="#444", color=c, mfc=c, mec="#222", mew=0.9,
                    linestyle="none")
    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label_map.get(t, t) for t in admit_sorted["treatment"]])
    ax.set_xlabel(r"DML log-odds coefficient  $\beta$  on $Y_1 = $ selected by LLM")
    ax.set_title("(a) Joint Spec B — all 6 treatments  "
                 f"(baseline admission rate p$_0$ = {p0_overall*100:.1f} percent)",
                 loc="left", pad=14)
    # Place annotations at a fixed right-edge x for the panel (so they don't
    # overlap markers regardless of coef sign).
    rightmost_a = (admit_sorted["coef"] + admit_sorted["se"] * 1.96).max()
    ann_x_a = rightmost_a + 0.10
    for y, (_, r), c in zip(ys, admit_sorted.iterrows(), colors):
        odds_ratio = math.exp(r["coef"])
        dp = delta_p(r["coef"], p0_overall) * 100
        stars = _stars(r["p_val"])
        ax.text(ann_x_a, y,
                f"β = {r['coef']:+.3f}{stars}\n"
                f"OR = {odds_ratio:.3f}\n"
                f"Δp ≈ {dp:+.2f} pp",
                ha="left", va="center",
                fontsize=8.5, color=c, linespacing=1.3)
    cur = ax.get_xlim()
    ax.set_xlim(cur[0], ann_x_a + 0.45)

    # ── Panel (b) — T7 per variant ──
    ax = axes[1]
    t7_var = t7_var.set_index("variant").reindex(VARIANTS)
    ys = np.arange(len(VARIANTS))[::-1]
    cols = [COLOR_BIASED if "biased" in v else COLOR_NEUTRAL
            for v in VARIANTS]
    for y, v, c in zip(ys, VARIANTS, cols):
        r = t7_var.loc[v]
        ax.errorbar(r["coef"], y, xerr=r["se"] * 1.96,
                    fmt="o", capsize=4, elinewidth=1.4, markersize=8,
                    ecolor="#444", color=c, mfc=c, mec="#222", mew=0.9,
                    linestyle="none")
    ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS])
    ax.set_xlabel(r"DML log-odds  $\beta$  on $Y_1$, T7 only")
    ax.set_title("(b) T7 by variant — prompt-induced, RAG-resistant",
                 loc="left", pad=14)
    # Fixed annotation column
    rightmost_b = (t7_var["coef"] + t7_var["se"] * 1.96).max()
    ann_x_b = rightmost_b + 0.15
    for y, v, c in zip(ys, VARIANTS, cols):
        r = t7_var.loc[v]
        odds_ratio = math.exp(r["coef"])
        dp = delta_p(r["coef"], rates[v]) * 100
        stars = (r["sig"] if pd.notna(r["sig"])
                 and not (isinstance(r["sig"], float) and pd.isna(r["sig"]))
                 else "")
        ax.text(ann_x_b, y,
                f"β = {r['coef']:+.3f}{stars}\n"
                f"OR = {odds_ratio:.3f}\n"
                f"Δp ≈ {dp:+.2f} pp  (baseline {rates[v]:.0%})",
                ha="left", va="center",
                fontsize=8.5, color=c, linespacing=1.3)
    cur = ax.get_xlim()
    ax.set_xlim(cur[0] - 0.1, ann_x_b + 1.5)

    fig.suptitle(
        "Admission outcome — DML log-odds, odds ratio, and percentage-point change at baseline",
        fontsize=12, x=0.02, ha="left", y=1.04)

    save(fig, "fig12_admission_detail")


# ── FIG 13 ──────────────────────────────────────────────────────────────────


def _admission_confounder_significance() -> pd.DataFrame:
    """Quick OLS-on-logistic of `selected` ~ confounder + variant + cell FE.

    Mirrors confounder_ols_significance.parquet (rank_delta, post_rank) for
    the binary admission outcome. Builds the SERP pool from phase0_top*.parquet,
    expands to (model × variant), marks `selected=1` if the URL appears in the
    per-variant LLM-output parquet, attaches confounder features by merging
    on (keyword, url), then OLS per confounder.
    """
    import statsmodels.api as sm

    # SERP files live in a separate root from the HF snapshot.
    serp_candidates = [
        Path("/Users/valerianfourel/Hamburg/GEODML_Analysis/geodml_data/data/serp"),
        Path("/Users/valerianfourel/Hamburg/geodml-dataset/data/serp"),
        DATA / "serp",
    ]
    serp_root = next((p for p in serp_candidates if p.exists()), None)
    if serp_root is None:
        raise FileNotFoundError("No SERP pool dir with phase0_top*.parquet found")

    # 1. Build base pool from the 4 phase0 files.
    pool_parts = []
    for (e, n), fname in [(("ddg", 20),     "phase0_top20_ddg.parquet"),
                          (("ddg", 50),     "phase0_top50_ddg.parquet"),
                          (("searxng", 20), "phase0_top20_searxng.parquet"),
                          (("searxng", 50), "phase0_top50_searxng.parquet")]:
        p = pd.read_parquet(serp_root / fname)
        p["engine"] = e
        p["pool_size"] = n
        pool_parts.append(p[["keyword", "url", "position", "engine", "pool_size"]])
    pool = pd.concat(pool_parts, ignore_index=True)

    # 2. Expand to (model × variant)
    models = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"]
    rows = []
    for m in models:
        for v in VARIANTS:
            sub = pool.copy()
            sub["model"] = m
            sub["variant"] = v
            rows.append(sub)
    big = pd.concat(rows, ignore_index=True)

    # 3. Selection flag from per-variant LLM output
    sel_idx = set()
    var_kw = {v: set() for v in VARIANTS}
    for v in VARIANTS:
        df = pd.read_parquet(DATA / "main" / f"full_experiment_data_{v}.parquet")
        df["engine_norm"] = df["search_engine"].replace({"duckduckgo": "ddg"})
        pool_col = "serp_pool_size" if "serp_pool_size" in df.columns else "pool"
        df = df.rename(columns={pool_col: "pool_size"})
        keys = list(zip(df["keyword"], df["url"], df["engine_norm"],
                        df["pool_size"], df["llm_model"], [v]*len(df)))
        sel_idx.update(keys)
        var_kw[v] = set(df["keyword"])
    big_keys = list(zip(big["keyword"], big["url"], big["engine"],
                        big["pool_size"], big["model"], big["variant"]))
    big["selected"] = [int(k in sel_idx) for k in big_keys]

    # 4. Restrict RAG variants to RAG-covered keywords
    keep = pd.Series(True, index=big.index)
    for v in ["biased_rag", "neutral_rag"]:
        mask = (big["variant"] == v) & ~big["keyword"].isin(var_kw[v])
        keep &= ~mask
    big = big[keep].reset_index(drop=True)
    print(f"      pool size: {len(big):,}, admission rate: {big['selected'].mean():.3f}")

    # 5. Attach confounder features from the union of per-variant LLM outputs
    feat_parts = []
    for v in VARIANTS:
        df = pd.read_parquet(DATA / "main" / f"full_experiment_data_{v}.parquet")
        conf_cols = [c for c in df.columns if c.startswith("conf_")
                     and df[c].dtype != "O"]
        feat_parts.append(df[["keyword", "url"] + conf_cols])
    feats = pd.concat(feat_parts, ignore_index=True)
    feats = feats.groupby(["keyword", "url"], as_index=False).mean(numeric_only=True)
    big = big.merge(feats, on=["keyword", "url"], how="left")

    conf_cols = [c for c in big.columns if c.startswith("conf_")
                 and big[c].dtype != "O"]
    X_fe = pd.get_dummies(
        big[["variant", "engine", "model"]],
        drop_first=True).astype(float)
    y = big["selected"].values

    rows_out = []
    for c in conf_cols:
        x = big[c].fillna(big[c].median()).values.astype(float)
        if x.std() == 0:
            continue
        x = (x - x.mean()) / x.std()
        X = np.column_stack([x, X_fe.values, np.ones(len(x))])
        try:
            model = sm.OLS(y, X).fit(cov_type="HC1")
            rows_out.append({
                "confounder": c,
                "coef": float(model.params[0]),
                "se": float(model.bse[0]),
                "t_stat": float(model.tvalues[0]),
                "p_val": float(model.pvalues[0]),
            })
        except Exception:
            continue
    return pd.DataFrame(rows_out).sort_values("p_val")


def fig13_top_confounders() -> None:
    """Top-significant confounders for admission (Y_1) and rank_delta (Y_2).

    Panel (a): admission confounders — computed inline by OLS-on-logistic
                regression of selected ~ confounder + variant + engine + LLM
                fixed effects (since the published
                confounder_ols_significance.parquet only covers rank_delta
                and post_rank).
    Panel (b): rank_delta confounders — from
                confounder_ols_significance.parquet, with the mechanical
                pre-rank dominator (conf_serp_position) excluded so we see
                the next layer of significance.
    """
    print("    computing admission confounder significance …")
    adm_conf = _admission_confounder_significance().head(10)
    print(f"      top admission confounder: {adm_conf.iloc[0]['confounder']} "
          f"(t = {adm_conf.iloc[0]['t_stat']:.1f})")

    rd_conf = pd.read_parquet(DML / "confounder_ols_significance.parquet")
    rd_conf = rd_conf[(rd_conf["outcome"] == "rank_delta")
                      & (rd_conf["confounder"] != "intercept")
                      & (rd_conf["confounder"] != "conf_serp_position")
                      ].copy()
    rd_conf = rd_conf.reindex(rd_conf["t_stat"].abs().sort_values(
        ascending=False).index).head(10)

    def clean_name(s):
        return (s.replace("conf_", "")
                 .replace("dfs_", "dfs ")
                 .replace("_", " "))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0),
                             gridspec_kw={"wspace": 0.55})

    def draw_panel(ax, df_in, xlabel, title):
        df_sorted = df_in.sort_values("t_stat")
        ys = np.arange(len(df_sorted))
        cols = ["#3182bd" if r["coef"] > 0 else "#e6550d"
                for _, r in df_sorted.iterrows()]
        for y, (_, r), c in zip(ys, df_sorted.iterrows(), cols):
            ax.barh(y, r["t_stat"], xerr=0, color=c, edgecolor="#222",
                    linewidth=0.5, height=0.65)
        ax.axvline(0, color="#cccccc", linestyle="--", linewidth=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([clean_name(c) for c in df_sorted["confounder"]])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=14)

        # Compute a single right-margin column where ALL annotations live,
        # so they never overlap the bars or the y-axis tick labels.
        tmax = df_sorted["t_stat"].max()
        tmin = df_sorted["t_stat"].min()
        span = tmax - tmin
        ann_x = tmax + 0.08 * span
        for y, (_, r), c in zip(ys, df_sorted.iterrows(), cols):
            p = r["p_val"]
            p_str = ("p < 1e-3" if p < 1e-3
                     else f"p = {p:.2e}" if p < 1e-2
                     else f"p = {p:.3f}")
            ax.text(ann_x, y,
                    f"t = {r['t_stat']:+5.1f}    {p_str}",
                    ha="left", va="center",
                    fontsize=10, color=c)
        # widen so the annotations don't get clipped
        ax.set_xlim(tmin - 0.07 * span, ann_x + 0.50 * span)

    draw_panel(
        axes[0], adm_conf,
        "OLS t-statistic on selected (admission)",
        "(a) Top admission confounders\n"
        "(OLS-on-logistic, controls for variant × engine × LLM)",
    )
    draw_panel(
        axes[1], rd_conf,
        r"OLS t-statistic on $\Delta\mathrm{rank}$",
        "(b) Top rank_delta confounders\n"
        "(conf_serp_position excluded — mechanically dominates)",
    )

    legend = [
        Patch(facecolor="#3182bd", label="positive coef on outcome"),
        Patch(facecolor="#e6550d", label="negative coef on outcome"),
    ]
    fig.legend(handles=legend, loc="upper center",
               bbox_to_anchor=(0.5, 1.04),
               frameon=False, ncol=2, fontsize=9)

    fig.suptitle("Top significant confounders — admission (Y₁) vs ranking (Y₂)",
                 fontsize=12, x=0.02, ha="left", y=1.08)

    save(fig, "fig13_top_confounders")


# ── driver ───────────────────────────────────────────────────────────────────


FIGURES = {
    "fig1": ("fig1_content_headline", fig1_content_headline, False),
    "fig2": ("fig2_coef_grid", fig2_coef_grid, True),
    "fig3": ("fig3_rag_attenuation", fig3_rag_attenuation, True),
    "fig4": ("fig4_marginal_vs_partial", fig4_marginal_vs_partial, True),
    "fig5": ("fig5_robust_survivors", fig5_robust_survivors, False),
    "fig6": ("fig6_two_llm_agreement", fig6_two_llm_agreement, True),
    "fig7": ("fig7_jaccard_agreement", fig7_jaccard_agreement, False),
    "fig8": ("fig8_pool_size_sensitivity", fig8_pool_size_sensitivity, False),
    "fig9": ("fig9_dml_schematic", fig9_dml_schematic, False),
    "fig10": ("fig10_causal_dag", fig10_causal_dag, False),
    "fig11": ("fig11_admission_vs_rank", fig11_admission_vs_rank, False),
    "fig12": ("fig12_admission_detail", fig12_admission_detail, False),
    "fig13": ("fig13_top_confounders", fig13_top_confounders, False),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="Comma-separated keys (e.g. fig1,fig3)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, (name, _, _) in FIGURES.items():
            print(f"  {k}  → docs/figures/{name}.pdf")
        return 0

    setup_style()
    print(f"[make_figures] loading parquets …")
    dml = load_dml_long()
    print(f"  dml rows: {len(dml):,}  variants: {sorted(dml.variant.unique())}")
    print(f"  T7 excluded — content treatments only ({len(CONTENT_TREATMENTS)} treatments)")
    print(f"[make_figures] output → {OUT.relative_to(REPO)}/")

    keys = args.only.split(",") if args.only else list(FIGURES.keys())
    for k in keys:
        if k not in FIGURES:
            print(f"  unknown: {k}", file=sys.stderr)
            continue
        name, fn, needs_dml = FIGURES[k]
        print(f"\n  [{k}] {name}")
        try:
            fn(dml) if needs_dml else fn()
        except Exception:
            import traceback
            traceback.print_exc()

    print(f"\n[done] figures in {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
