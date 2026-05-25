"""Saliency figure — grouped bar chart of attention to treatment tokens.

Reads `interpretability/output/saliency_summary_full.csv` (one row per
(model, treatment) pair) and renders a single-panel grouped bar chart:

  x-axis: 4 treatments
  y-axis: saliency_ratio  (mean treatment-token saliency  /  mean other-token saliency)
  one bar pair per treatment — Llama vs Qwen
  reference line at ratio = 1.0  (baseline)

Same single-panel style as the other Stage-F figures: legend below axes,
breathing room, no overlap with bars.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "2026-05-24" / "figures_canonical" / "tmp"

INPUT = REPO / "interpretability" / "output" / "saliency_summary_full.csv"

# Display order + pretty labels.
TREATMENT_DISPLAY = [
    ("T1b_stats_density",      r"$T_{1b}$  stats density"),
    ("T2a_question_headings",  r"$T_{2a}$  Q-headings"),
    ("T3_structured_data_new", r"$T_{3}$  schema (JSON-LD)"),
    ("T7_source_earned",       r"$T_{7}$  source_earned  (descriptive)"),
]

MODEL_COLOR = {
    "Llama-3.3-70B": "#2c7fb8",   # blue
    "Qwen-2.5-72B":  "#d6604d",   # red
}


def load_data() -> pd.DataFrame:
    """Saliency summary lacks a model column; rows are 4 treatments per model
    in order Llama, Qwen. Reconstruct it."""
    df = pd.read_csv(INPUT)
    n = len(df) // 2
    df["model"] = ["Llama-3.3-70B"] * n + ["Qwen-2.5-72B"] * n
    return df


def make_fig(df: pd.DataFrame):
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlepad":     14,
        "axes.labelpad":     10,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "pdf.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    treatments = [t for t, _ in TREATMENT_DISPLAY]
    labels = [lbl for _, lbl in TREATMENT_DISPLAY]
    x = np.arange(len(treatments))
    w = 0.36

    models = ["Llama-3.3-70B", "Qwen-2.5-72B"]
    for i, m in enumerate(models):
        ratios = []
        for t in treatments:
            row = df[(df.model == m) & (df.treatment == t)]
            ratios.append(float(row["saliency_ratio"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(
            x + (i - 0.5) * w, ratios, width=w,
            color=MODEL_COLOR[m], edgecolor="white", linewidth=1.0,
            label=m, alpha=0.92,
        )
        for bar, r in zip(bars, ratios):
            if not np.isnan(r):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        r + 0.05 if r >= 1.0 else r + 0.05,
                        f"{r:.2f}×", ha="center", va="bottom",
                        fontsize=10, color=MODEL_COLOR[m], fontweight="bold")

    # baseline reference line at ratio = 1.0
    ax.axhline(1.0, color="#444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(treatments) - 0.5, 1.02, "baseline (= mean of other tokens)",
            color="#444", fontsize=9.5, ha="right", va="bottom", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_xlim(-0.6, len(treatments) - 0.4)
    ax.set_ylim(0, 2.20)
    ax.set_ylabel("saliency ratio   (treatment / other tokens)", fontsize=13)
    ax.set_title(
        "Gradient saliency to treatment tokens, by reranker backbone",
        loc="left", fontsize=13.5,
    )
    ax.grid(axis="y", alpha=0.22)

    # legend BELOW axes (matches the other Stage-F figures)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=2, frameon=False, fontsize=11.5,
        handlelength=1.6, columnspacing=3.0, borderpad=0.4,
    )

    fig.text(
        0.50, 0.005,
        "For each token in the rerank prompt, gradient saliency is computed w.r.t. the model's chosen-URL log-probability.   "
        "Bars show the mean saliency of treatment-typed tokens divided by the mean saliency of all other tokens.\n"
        "Headline: Qwen attends 1.9× the baseline to numeric tokens ($T_{1b}$) while Llama attends only 0.9× — a sharp inter-model divergence.   "
        "Both models barely look at JSON-LD ($T_{3}$): saliency ratio ≤ 0.40×.",
        ha="center", va="bottom", fontsize=10, color="#444", style="italic",
    )
    fig.subplots_adjust(top=0.92, bottom=0.28, left=0.10, right=0.97)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print("Loaded saliency summary:")
    print(df[["model","treatment","saliency_ratio","n_treatment_tokens"]].to_string(index=False))

    fig = make_fig(df)
    png = OUT_DIR / "fig_saliency.png"
    pdf = OUT_DIR / "fig_saliency.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
