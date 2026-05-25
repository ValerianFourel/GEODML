"""
Unified probing figure for the EMNLP paper.

Reads `interpretability/output/probing_results_<variant>.csv` for the 4 variants
and renders a single 2-panel layout:

  Panel A — layer-wise ROC AUC for the 4 treatments (mean pooling, averaged
            across variants since variant has no measurable effect on the probe).
  Panel B — pooling comparison: mean pooling vs last-token pooling, showing
            that content lives in the sequence average rather than at the
            decision token.

Outputs PNG (300 dpi) and PDF (vector) to docs/2026-05-24/figures_canonical/tmp/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
PROBING_DIR = REPO / "interpretability" / "output"
OUT_DIR = REPO / "docs" / "2026-05-24" / "figures_canonical" / "tmp"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
TREATMENTS = [
    "T1b_stats_density",
    "T2a_question_headings",
    "T3_structured_data",
    "T4_citation_authority",
    "T5_topical_comp",
    "T6_freshness",
    # T7 = has_llms_txt dropped 2026-05-25: LLM never reads the file at
    # inference time, so any probe accuracy on it reflects spurious
    # train-data correlations, not a causal internal representation.
]

# 6-class palette, colour-blind safe (Okabe-Ito subset).
TREAT_COLOR = {
    "T1b_stats_density":     "#000000",   # black
    "T2a_question_headings": "#0072B2",   # blue
    "T3_structured_data":    "#D55E00",   # vermilion
    "T4_citation_authority": "#CC79A7",   # reddish purple
    "T5_topical_comp":       "#009E73",   # green
    "T6_freshness":          "#E69F00",   # orange
}
TREAT_LABEL = {
    "T1b_stats_density":     r"$T_{1}$  stats density",
    "T2a_question_headings": r"$T_{2}$  question headings",
    "T3_structured_data":    r"$T_{3}$  schema (JSON-LD)",
    "T4_citation_authority": r"$T_{4}$  citation authority",
    "T5_topical_comp":       r"$T_{5}$  topical comprehensiveness",
    "T6_freshness":          r"$T_{6}$  freshness",
}


# ---------------------------------------------------------------------------
# Load + deduplicate
# ---------------------------------------------------------------------------

def load_probing() -> pd.DataFrame:
    """Concat all 4 variants, drop chain-retry duplicates, return tidy frame."""
    frames = []
    for v in VARIANTS:
        p = PROBING_DIR / f"probing_results_{v}.csv"
        if not p.exists():
            print(f"WARN: {p} missing — skipping")
            continue
        df = pd.read_csv(p)
        df["variant"] = v
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["frame"] == "full"].copy()      # robust_winners only on some variants
    raw = raw[raw["treatment"].isin(TREATMENTS)].copy()
    # Each (variant, treatment, layer, pooling) appears 2–4 times because
    # multiple chain-retry attempts wrote into the same per-cell CSV. Their
    # numerical contents are identical or near-identical (same probe on the
    # same data), so we average over the retries.
    return (
        raw.groupby(["variant", "treatment", "layer", "pooling"], as_index=False)[
            ["accuracy", "roc_auc", "n_train", "n_test"]
        ].mean()
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.family":       "serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlepad":     14,
        "axes.labelpad":     8,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "pdf.fonttype":      42,
    })


def make_fig_layerwise(df: pd.DataFrame):
    """Single-panel: layer-wise ROC AUC per treatment, mean pooling."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    main = (df[df["pooling"] == "mean"]
            .groupby(["treatment", "layer"])
            .agg(roc_mean=("roc_auc", "mean"),
                 roc_min=("roc_auc", "min"),
                 roc_max=("roc_auc", "max"))
            .reset_index())

    available = set(main["treatment"].unique())
    missing = [t for t in TREATMENTS if t not in available]
    if missing:
        print(f"NOTE: treatments not yet in data, will be skipped: {missing}")

    peaks = []
    for t in TREATMENTS:
        sub = main[main.treatment == t].sort_values("layer")
        if sub.empty:
            continue
        c = TREAT_COLOR[t]
        ax.fill_between(sub["layer"], sub["roc_min"], sub["roc_max"],
                        color=c, alpha=0.14, linewidth=0)
        ax.plot(sub["layer"], sub["roc_mean"],
                color=c, lw=2.4, label=TREAT_LABEL[t])
        peak = sub.loc[sub["roc_mean"].idxmax()]
        peaks.append((t, c, peak))
        ax.plot(peak["layer"], peak["roc_mean"],
                marker="o", markersize=11, color=c,
                markeredgecolor="white", markeredgewidth=1.8, zorder=6)

    ax.set_xlim(-3, 83)
    ax.set_ylim(0.94, 1.012)
    ax.set_xlabel("transformer layer", fontsize=13)
    ax.set_ylabel("probe  ROC AUC", fontsize=13)
    ax.set_title("Layer-wise probing accuracy by treatment  (mean pooling)",
                 loc="left", fontsize=13.5)
    ax.grid(axis="y", alpha=0.22)

    # legend BELOW-LEFT, peaks table BELOW-RIGHT — both outside the data area
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.14),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=2.0,
        labelspacing=0.55,
        columnspacing=1.4,
        borderpad=0.3,
    )

    if peaks:
        rows = ["peak ROC AUC  (mean pooling)", "─" * 30]
        for t, c, peak in sorted(peaks, key=lambda r: -r[2]["roc_mean"]):
            short = TREAT_LABEL[t].split(" ", 1)[0]
            rows.append(f"  {short:6s}  L{int(peak['layer']):>2}   {peak['roc_mean']:.3f}")
        ax.text(1.0, -0.14, "\n".join(rows),
                transform=ax.transAxes,
                family="monospace", fontsize=10.5, color="#333",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                          edgecolor="#bbb", linewidth=0.8))

    fig.text(0.50, 0.005,
             "Linear probe (logistic regression on frozen hidden states) per (treatment, layer); 80/20 stratified split.   "
             "Llama-3.3-70B + Qwen-2.5-72B; averaged across 4 prompt variants.\n"
             "Shaded band:  min–max envelope across variants  (invisibly narrow — variant has no measurable effect).",
             ha="center", va="bottom", fontsize=10, color="#444", style="italic")
    fig.subplots_adjust(top=0.92, bottom=0.34, left=0.10, right=0.97)
    return fig


def make_fig_pooling(df: pd.DataFrame):
    """Single-panel: mean vs last-token pooling comparison."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    pool = (df.groupby(["treatment", "layer", "pooling"])["roc_auc"]
              .mean().reset_index())
    available = set(pool["treatment"].unique())

    for t in TREATMENTS:
        if t not in available:
            continue
        c = TREAT_COLOR[t]
        sub_mean = pool[(pool.treatment == t) & (pool.pooling == "mean")].sort_values("layer")
        sub_last = pool[(pool.treatment == t) & (pool.pooling == "last_token")].sort_values("layer")
        ax.plot(sub_mean["layer"], sub_mean["roc_auc"], color=c, lw=2.2,
                alpha=0.95, label=TREAT_LABEL[t])
        ax.plot(sub_last["layer"], sub_last["roc_auc"], color=c, lw=1.5,
                linestyle="--", alpha=0.80)

    ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(2.5, 0.515, "chance", color="#777", fontsize=9.5,
            ha="left", va="bottom", style="italic")
    ax.set_xlim(-3, 83)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("transformer layer", fontsize=13)
    ax.set_ylabel("probe  ROC AUC", fontsize=13)
    ax.set_title("Pooling comparison  (mean vs. last-token, per treatment)",
                 loc="left", fontsize=13.5)
    ax.grid(axis="y", alpha=0.22)

    # two-row legend below the axes:
    # row 1 = treatments (same colors as fig A)
    # row 2 = pooling style (solid vs dashed, in neutral gray)
    treat_handles, treat_labels = ax.get_legend_handles_labels()
    pool_handles = [
        Line2D([0], [0], color="#444", lw=2.2, label="mean pooling"),
        Line2D([0], [0], color="#444", lw=1.5, linestyle="--",
               label="last-token pooling"),
    ]

    leg1 = ax.legend(
        treat_handles, treat_labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=3, frameon=False, fontsize=10.5,
        handlelength=2.0, labelspacing=0.6, columnspacing=1.6,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=pool_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.30),
        ncol=2, frameon=False, fontsize=10.5,
        handlelength=2.6, labelspacing=0.55, columnspacing=2.4,
    )

    fig.text(0.50, 0.005,
             "Mean pooling (solid) averages hidden states over all tokens of the page; last-token pooling (dashed) takes the final token only.\n"
             "Mean pooling near-saturates from layer 0;  last-token requires several layers to absorb context before reaching parity.",
             ha="center", va="bottom", fontsize=10, color="#444", style="italic")
    fig.subplots_adjust(top=0.92, bottom=0.36, left=0.10, right=0.97)
    return fig


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _save(fig, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_probing()
    print(f"Loaded {len(df)} probe rows over {df['variant'].nunique()} variants "
          f"× {df['treatment'].nunique()} treatments × {df['layer'].nunique()} layers "
          f"× {df['pooling'].nunique()} poolings.")
    _save(make_fig_layerwise(df), "fig_probing_layerwise")
    _save(make_fig_pooling(df),   "fig_probing_pooling")


if __name__ == "__main__":
    main()
