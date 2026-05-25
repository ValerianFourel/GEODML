"""Figure: layer-wise admission pre-commitment probe.

Reads the Y1_admission_inctx rows from
`interpretability/output/probing_results_<variant>.csv` (all four
variants) and renders a 2-panel figure:

(a) ROC AUC per layer, both poolings, averaged across the 4 prompt
    variants and the 2 LLMs.
(b) Per-variant breakdown — does the SERP context (prompt variant)
    shift the layer-wise curve, or is the decision crystallisation
    the same regardless of prompt?

Headline numbers (layer 0 / peak layer / final layer) are printed and
embedded as figure annotations.

Usage:
    python scripts/make_fig_admission_probe.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO / "interpretability" / "output"
OUT_DIR = REPO / "docs" / "2026-05-24" / "figures_canonical" / "tmp"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]

VARIANT_COLOR = {
    "biased":      "#d6604d",
    "biased_rag":  "#b2182b",
    "neutral":     "#4393c3",
    "neutral_rag": "#2166ac",
}
VARIANT_LABEL = {
    "biased":      "biased",
    "biased_rag":  "biased + RAG",
    "neutral":     "neutral",
    "neutral_rag": "neutral + RAG",
}


def load_admission() -> pd.DataFrame:
    frames = []
    for v in VARIANTS:
        p = PROBING_DIR / f"probing_results_{v}.csv"
        if not p.exists():
            print(f"WARN: {p} missing")
            continue
        df = pd.read_csv(p)
        df = df[(df.treatment == "Y1_admission_inctx") & (df.frame == "full")].copy()
        df["variant"] = v
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit("No Y1_admission_inctx rows found in any variant CSV. "
                         "Did the merge step finish?")
    raw = pd.concat(frames, ignore_index=True)
    # multiple chain-retry duplicates collapsed by averaging
    return (raw.groupby(["variant", "layer", "pooling"])[
        ["accuracy", "roc_auc", "n_train", "n_test"]
    ].mean().reset_index())


def make_figure(df: pd.DataFrame):
    plt.rcParams.update({
        "font.family":         "serif",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.titlepad":       14,
        "axes.labelpad":       8,
        "xtick.labelsize":     11,
        "ytick.labelsize":     11,
        "pdf.fonttype":        42,
    })

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.5, 6.0),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.24},
    )

    # ===== Panel A: pooled across variants, both poolings =====
    pooled = (df.groupby(["layer", "pooling"])["roc_auc"]
                .agg(["mean", "min", "max"]).reset_index())

    POOL_STYLE = {"mean": ("-", 2.4, "#1d4d8a"),
                  "last_token": ("--", 1.6, "#5a5a5a")}
    POOL_LABEL = {"mean": "mean pooling",
                  "last_token": "last-token pooling"}

    for p in ("mean", "last_token"):
        sub = pooled[pooled.pooling == p].sort_values("layer")
        ls, lw, c = POOL_STYLE[p]
        axA.fill_between(sub["layer"], sub["min"], sub["max"],
                         color=c, alpha=0.11, linewidth=0)
        axA.plot(sub["layer"], sub["mean"],
                 color=c, lw=lw, linestyle=ls, label=POOL_LABEL[p])

    # headline numbers (mean pooling)
    mean_sub = pooled[pooled.pooling == "mean"]
    peak = mean_sub.loc[mean_sub["mean"].idxmax()]
    layer0 = mean_sub.loc[mean_sub.layer == 0, "mean"].iloc[0]
    last_layer = int(mean_sub.layer.max())
    layerN = mean_sub.loc[mean_sub.layer == last_layer, "mean"].iloc[0]

    # markers with arrows pointing INTO clear areas of the plot
    axA.plot(peak["layer"], peak["mean"], marker="o", markersize=11,
             color="#1d4d8a", markeredgecolor="white", markeredgewidth=1.8, zorder=6)
    axA.plot(0, layer0, marker="s", markersize=9, color="#666",
             markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    axA.plot(last_layer, layerN, marker="s", markersize=9, color="#666",
             markeredgecolor="white", markeredgewidth=1.2, zorder=6)

    # Annotation BOX in the wide top-left area (curves are in the middle/bottom)
    txt = (f"layer 0:   {layer0:.3f}\n"
           f"peak L{int(peak['layer'])}:  {peak['mean']:.3f}\n"
           f"layer {last_layer}:  {layerN:.3f}\n"
           f"L0 → peak: +{peak['mean'] - layer0:.3f}")
    axA.text(2.5, 0.985, txt,
             fontsize=10.5, color="#1d4d8a", family="monospace",
             ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#bbb", linewidth=0.8))

    # chance line + label well away from data
    axA.axhline(0.5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    axA.text(80.5, 0.512, "chance", color="#777", fontsize=9,
             ha="right", va="bottom", style="italic")

    axA.set_xlim(-3, 83)
    axA.set_ylim(0.45, 1.02)
    axA.set_xlabel("transformer layer", fontsize=12)
    axA.set_ylabel(r"probe ROC AUC  (is URL admitted?)", fontsize=12)
    axA.set_title("(a)  admission probe  —  pooled over 4 prompt variants",
                  loc="left", fontsize=13)
    axA.grid(axis="y", alpha=0.22)

    # legend in upper-right corner, away from peak marker
    axA.legend(loc="upper right", frameon=False, fontsize=11,
               handlelength=2.4, labelspacing=0.5, borderpad=0.3,
               bbox_to_anchor=(0.99, 0.85))

    # ===== Panel B: per-variant breakdown, mean pooling =====
    sub_mean = df[df.pooling == "mean"]
    for v in VARIANTS:
        s = sub_mean[sub_mean.variant == v].sort_values("layer")
        if s.empty:
            continue
        axB.plot(s["layer"], s["roc_auc"],
                 color=VARIANT_COLOR[v], lw=2.0,
                 linestyle="--" if "rag" in v else "-",
                 label=VARIANT_LABEL[v], alpha=0.92)

    axB.axhline(0.5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    axB.text(80.5, 0.512, "chance", color="#777", fontsize=9,
             ha="right", va="bottom", style="italic")
    axB.set_xlim(-3, 83)
    axB.set_ylim(0.45, 1.02)
    axB.set_xlabel("transformer layer", fontsize=12)
    axB.set_ylabel(r"probe ROC AUC", fontsize=12)
    axB.set_title("(b)  per prompt variant  —  mean pooling",
                  loc="left", fontsize=13)
    axB.grid(axis="y", alpha=0.22)
    axB.legend(loc="upper right", frameon=False, fontsize=10.5, ncol=2,
               handlelength=2.0, labelspacing=0.5, columnspacing=1.4,
               bbox_to_anchor=(0.99, 0.92))

    # caption with generous breathing room below the panels
    fig.text(0.50, 0.02,
             "Behavioural pre-commitment probe.  For each URL span in the rerank prompt, the label is 1 if the (model, variant) admitted that URL, else 0.\n"
             "Linear probe (logistic regression on frozen hidden states), one fit per (layer, pooling), 80/20 stratified split.  "
             "Pooled across Llama-3.3-70B and Qwen-2.5-72B.   "
             "Shaded band in (a):  min–max envelope across the 4 prompt variants.",
             ha="center", va="bottom", fontsize=10, color="#444", style="italic")
    fig.subplots_adjust(top=0.91, bottom=0.20, left=0.06, right=0.985)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_admission()
    print(f"Loaded {len(df)} admission-probe rows over "
          f"{df['variant'].nunique()} variants × "
          f"{df['layer'].nunique()} layers × "
          f"{df['pooling'].nunique()} poolings.")

    # headline numbers (mean pooling, averaged over variants)
    pooled = df[df.pooling == "mean"].groupby("layer")["roc_auc"].mean()
    print("\n=== Headline (mean pooling, averaged over 4 variants) ===")
    print(f"  layer 0  ROC AUC: {pooled.iloc[0]:.3f}")
    print(f"  peak     ROC AUC: {pooled.max():.3f}  at layer {int(pooled.idxmax())}")
    print(f"  layer 80 ROC AUC: {pooled.iloc[-1]:.3f}")
    print(f"  L0 → peak gain :  +{pooled.max() - pooled.iloc[0]:.3f}")

    fig = make_figure(df)
    png = OUT_DIR / "fig_admission_probe.png"
    pdf = OUT_DIR / "fig_admission_probe.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
