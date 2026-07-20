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


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as REPO, PROBING as PROBING_DIR, FIGURES as OUT_DIR  # noqa: E402

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


def _style():
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


def make_fig_admission_pooled(df: pd.DataFrame):
    """Single-panel: admission probe, pooled over variants, both poolings."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    pooled = (df.groupby(["layer", "pooling"])["roc_auc"]
                .agg(["mean", "min", "max"]).reset_index())

    POOL_STYLE = {"mean": ("-", 2.6, "#1d4d8a"),
                  "last_token": ("--", 1.7, "#5a5a5a")}
    POOL_LABEL = {"mean": "mean pooling",
                  "last_token": "last-token pooling"}

    for p in ("mean", "last_token"):
        sub = pooled[pooled.pooling == p].sort_values("layer")
        ls, lw, c = POOL_STYLE[p]
        ax.fill_between(sub["layer"], sub["min"], sub["max"],
                        color=c, alpha=0.11, linewidth=0)
        ax.plot(sub["layer"], sub["mean"],
                color=c, lw=lw, linestyle=ls, label=POOL_LABEL[p])

    # headline numbers (mean pooling)
    mean_sub = pooled[pooled.pooling == "mean"]
    peak = mean_sub.loc[mean_sub["mean"].idxmax()]
    layer0 = mean_sub.loc[mean_sub.layer == 0, "mean"].iloc[0]
    last_layer = int(mean_sub.layer.max())
    layerN = mean_sub.loc[mean_sub.layer == last_layer, "mean"].iloc[0]

    ax.plot(peak["layer"], peak["mean"], marker="o", markersize=12,
            color="#1d4d8a", markeredgecolor="white", markeredgewidth=2.0, zorder=6)
    ax.plot(0, layer0, marker="s", markersize=10, color="#666",
            markeredgecolor="white", markeredgewidth=1.4, zorder=6)
    ax.plot(last_layer, layerN, marker="s", markersize=10, color="#666",
            markeredgecolor="white", markeredgewidth=1.4, zorder=6)

    # Annotation box top-left (corner) — plenty of room there
    txt = (f"layer 0:     {layer0:.3f}\n"
           f"peak L{int(peak['layer'])}:    {peak['mean']:.3f}\n"
           f"layer {last_layer}:    {layerN:.3f}\n"
           f"L0 → peak:   +{peak['mean'] - layer0:.3f}")
    ax.text(0.015, 0.97, txt, transform=ax.transAxes,
            fontsize=11, color="#1d4d8a", family="monospace",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="#bbb", linewidth=0.8, alpha=0.97))

    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(80.5, 0.515, "chance", color="#777", fontsize=10,
            ha="right", va="bottom", style="italic")

    ax.set_xlim(-3, 83)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("transformer layer", fontsize=13)
    ax.set_ylabel(r"probe ROC AUC  (is URL admitted?)", fontsize=13)
    ax.set_title("Admission probe  —  pooled over 4 prompt variants",
                 loc="left", fontsize=13.5)
    ax.grid(axis="y", alpha=0.22)

    # legend BELOW the axes — never clashes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=False, fontsize=11.5,
              handlelength=2.6, columnspacing=2.4)

    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.97)
    return fig


def make_fig_admission_variants(df: pd.DataFrame):
    """Single-panel: per-variant admission probe curves, mean pooling."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    sub_mean = df[df.pooling == "mean"]
    for v in VARIANTS:
        s = sub_mean[sub_mean.variant == v].sort_values("layer")
        if s.empty:
            continue
        ax.plot(s["layer"], s["roc_auc"],
                color=VARIANT_COLOR[v], lw=2.2,
                linestyle="--" if "rag" in v else "-",
                label=VARIANT_LABEL[v], alpha=0.92)

    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(80.5, 0.515, "chance", color="#777", fontsize=10,
            ha="right", va="bottom", style="italic")
    ax.set_xlim(-3, 83)
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("transformer layer", fontsize=13)
    ax.set_ylabel(r"probe ROC AUC", fontsize=13)
    ax.set_title("Admission probe  —  per prompt variant  (mean pooling)",
                 loc="left", fontsize=13.5)
    ax.grid(axis="y", alpha=0.22)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=4, frameon=False, fontsize=11,
              handlelength=2.4, columnspacing=2.0)

    fig.text(0.50, 0.005,
             "Same probe as the pooled curve, split out by the four prompt variants.\n"
             "Solid = no-RAG prompt;  dashed = +RAG.  Red = biased system prompt;  blue = neutral.",
             ha="center", va="bottom", fontsize=10, color="#444", style="italic")
    fig.subplots_adjust(top=0.92, bottom=0.28, left=0.10, right=0.97)
    return fig


def _save(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


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

    _save(make_fig_admission_pooled(df),   "fig_admission_pooled")
    _save(make_fig_admission_variants(df), "fig_admission_variants")


if __name__ == "__main__":
    main()
