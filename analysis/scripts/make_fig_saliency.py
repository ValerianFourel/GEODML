"""Saliency figure — diverging horizontal bars: attended vs. ignored.

Reads `interpretability/output/saliency_summary_full.csv` (one row per
(model, treatment)) and renders a compact single-panel diverging
bar chart centred on the baseline (ratio = 1.0):

  right side  →  saliency > 1.0   (model attends MORE than baseline → "attended")
  left side   →  saliency < 1.0   (model attends LESS than baseline → "ignored")

3 canonical treatments only ($T_{1b}, T_{2a}, T_{3}$). T7_source_earned
is excluded per paper policy. Each treatment is labelled with its DML
direction (promoter / demoter / null) so the reader can read the
"attention vs. effect" relationship in one glance.

Designed to be small enough for a single-column figure in EMNLP layout
(figsize ~6.0 × 3.4 in).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "2026-05-24" / "figures_canonical" / "tmp"

INPUT = REPO / "interpretability" / "output" / "saliency_summary_full.csv"

# Display order (top-to-bottom in the figure) + DML direction annotation
# from the Spec B headline. Promote / Demote / Null.
TREATMENT_DISPLAY = [
    # (key, pretty label, DML direction)
    ("T1b_stats_density",      r"$T_{1b}$  stats density",      "null"),
    ("T2a_question_headings",  r"$T_{2a}$  Q-headings",         "promote"),
    ("T3_structured_data_new", r"$T_{3}$  schema (JSON-LD)",    "demote"),
]

MODEL_COLOR = {
    "Llama-3.3-70B": "#2c7fb8",
    "Qwen-2.5-72B":  "#d6604d",
}
DML_BADGE = {
    "promote": ("DML: promoter", "#2c7fb8"),
    "demote":  ("DML: demoter",  "#b2182b"),
    "null":    ("DML: null",     "#888888"),
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    n = len(df) // 2
    df["model"] = ["Llama-3.3-70B"] * n + ["Qwen-2.5-72B"] * n
    return df


def make_fig(df: pd.DataFrame):
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.titlepad":     10,
        "axes.labelpad":     6,
        "xtick.labelsize":   10,
        "ytick.labelsize":   11,
        "pdf.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    treatments = TREATMENT_DISPLAY
    n_t = len(treatments)
    bar_h = 0.38
    models = ["Llama-3.3-70B", "Qwen-2.5-72B"]

    # We plot (ratio - 1) so the bar grows right when > 1 and left when < 1.
    for i, (key, label, dml_dir) in enumerate(treatments):
        for j, m in enumerate(models):
            row = df[(df.model == m) & (df.treatment == key)]
            if not len(row):
                continue
            ratio = float(row["saliency_ratio"].iloc[0])
            y = i * 1.0 + (0.5 - j) * bar_h
            ax.barh(
                y, ratio - 1.0, height=bar_h * 0.92, left=1.0,
                color=MODEL_COLOR[m], edgecolor="white", linewidth=0.8,
                alpha=0.95, label=m if i == 0 else None,
            )
            # number annotation just outside the bar end
            xtxt = ratio + (0.06 if ratio >= 1.0 else -0.06)
            ax.text(xtxt, y, f"{ratio:.2f}×",
                    ha="left" if ratio >= 1.0 else "right",
                    va="center", fontsize=10, fontweight="bold",
                    color=MODEL_COLOR[m])

    # baseline reference line at 1.0
    ax.axvline(1.0, color="#333", linestyle="-", linewidth=1.0, alpha=0.55)

    # zone labels at the top of the plot area
    ax.text(0.45, -0.85, "← model IGNORES",
            ha="center", va="center", fontsize=10, color="#888",
            style="italic")
    ax.text(1.55, -0.85, "model ATTENDS →",
            ha="center", va="center", fontsize=10, color="#444",
            style="italic", fontweight="bold")

    # treatment labels include the DML direction inline
    ytick_positions = list(range(n_t))
    yticklabels = []
    for key, label, dml_dir in treatments:
        suffix = {
            "promote": "  (promoter)",
            "demote":  "  (demoter)",
            "null":    "  (null)",
        }[dml_dir]
        yticklabels.append(f"{label}{suffix}")
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(yticklabels, fontsize=11.5)
    # extra headroom at top for the IGNORES/ATTENDS labels
    ax.set_ylim(n_t - 0.45, -1.30)

    ax.set_xlim(0.0, 2.40)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("saliency ratio   (treatment-tokens / other-tokens)",
                  fontsize=11.5)
    ax.grid(axis="x", alpha=0.18)

    # legend below the axes
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, frameon=False, fontsize=10.5,
        handlelength=1.4, columnspacing=2.6, borderpad=0.2,
    )

    fig.text(0.5, 0.005,
             "Gradient saliency of each token w.r.t. the chosen-URL log-prob; "
             "ratio = mean(treatment tokens) / mean(other tokens).   "
             "Baseline = 1.0× (vertical line).",
             ha="center", va="bottom", fontsize=9.5, color="#444",
             style="italic")

    fig.subplots_adjust(top=0.93, bottom=0.30, left=0.30, right=0.97)
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
