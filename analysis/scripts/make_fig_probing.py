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
    "T7_llms_txt",
]

# 7-class palette, colour-blind safe (Okabe-Ito with one substitution).
TREAT_COLOR = {
    "T1b_stats_density":     "#000000",   # black
    "T2a_question_headings": "#0072B2",   # blue
    "T3_structured_data":    "#D55E00",   # vermilion
    "T4_citation_authority": "#CC79A7",   # reddish purple
    "T5_topical_comp":       "#009E73",   # green
    "T6_freshness":          "#E69F00",   # orange
    "T7_llms_txt":           "#56B4E9",   # sky blue
}
TREAT_LABEL = {
    "T1b_stats_density":     r"$T_{1}$  stats density",
    "T2a_question_headings": r"$T_{2}$  question headings",
    "T3_structured_data":    r"$T_{3}$  schema (JSON-LD)",
    "T4_citation_authority": r"$T_{4}$  citation authority",
    "T5_topical_comp":       r"$T_{5}$  topical comprehensiveness",
    "T6_freshness":          r"$T_{6}$  freshness",
    "T7_llms_txt":           r"$T_{7}$  llms.txt",
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

def make_figure(df: pd.DataFrame):
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.0, 5.4),
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.26},
    )

    # ---- Panel A: layer-wise ROC AUC (mean pooling, averaged over variants)
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

    peaks_a = []
    for t in TREATMENTS:
        sub = main[main.treatment == t].sort_values("layer")
        if sub.empty:
            continue
        c = TREAT_COLOR[t]
        # Faint band = variant agreement envelope (min-max across the 4 variants)
        axA.fill_between(sub["layer"], sub["roc_min"], sub["roc_max"],
                         color=c, alpha=0.14, linewidth=0)
        axA.plot(sub["layer"], sub["roc_mean"],
                 color=c, lw=2.0, label=TREAT_LABEL[t])
        peak = sub.loc[sub["roc_mean"].idxmax()]
        peaks_a.append((t, c, peak))
        axA.plot(peak["layer"], peak["roc_mean"],
                 marker="o", markersize=8, color=c,
                 markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        # Layer number annotation next to each peak marker
        axA.annotate(f"L{int(peak['layer'])}",
                     xy=(peak["layer"], peak["roc_mean"]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8.5, color=c, fontweight="bold")

    axA.set_xlim(-2, 82)
    axA.set_ylim(0.85, 1.015)
    axA.set_xlabel("transformer layer", fontsize=11)
    axA.set_ylabel("probe  ROC AUC", fontsize=11)
    axA.set_title("(a)  layer-wise probing accuracy  ·  mean pooling",
                  loc="left", fontsize=11.5, pad=8)
    axA.grid(axis="y", alpha=0.25)
    axA.legend(loc="lower right", frameon=False, fontsize=10,
               handlelength=1.8, labelspacing=0.5, borderpad=0.5)

    # ---- Panel B: pooling comparison
    pool = (df.groupby(["treatment", "layer", "pooling"])["roc_auc"]
              .mean().reset_index())

    for t in TREATMENTS:
        if t not in available:
            continue
        c = TREAT_COLOR[t]
        sub_mean = pool[(pool.treatment == t) & (pool.pooling == "mean")].sort_values("layer")
        sub_last = pool[(pool.treatment == t) & (pool.pooling == "last_token")].sort_values("layer")
        axB.plot(sub_mean["layer"], sub_mean["roc_auc"], color=c, lw=2.0, alpha=0.95)
        axB.plot(sub_last["layer"], sub_last["roc_auc"], color=c, lw=1.3,
                 linestyle="--", alpha=0.80)

    axB.axhline(0.5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    axB.text(2, 0.518, "chance", color="#666",
             fontsize=8.5, ha="left", va="bottom", style="italic")
    axB.set_xlim(-2, 82)
    axB.set_ylim(0.48, 1.005)
    axB.set_xlabel("transformer layer", fontsize=11)
    axB.set_ylabel("probe  ROC AUC", fontsize=11)
    axB.set_title("(b)  pooling comparison",
                  loc="left", fontsize=11.5, pad=8)
    axB.grid(axis="y", alpha=0.25)

    pool_legend = [
        Line2D([0], [0], color="#444", lw=2.0, label="mean pooling"),
        Line2D([0], [0], color="#444", lw=1.3, linestyle="--", label="last-token pooling"),
    ]
    axB.legend(handles=pool_legend, loc="lower right", frameon=False, fontsize=10,
               handlelength=2.4, labelspacing=0.5, borderpad=0.5)

    # ---- caption text under both panels --------------------------------
    fig.text(0.50, 0.005,
             "Linear probes (logistic regression on frozen hidden states), per (treatment, layer, pooling), trained on Llama-3.3-70B and Qwen2.5-72B.\n"
             "Shaded bands in (a): min–max envelope across the 4 prompt variants — invisible because variant has no measurable effect on the probe.   "
             "Dots: peak-AUC layer per treatment.",
             ha="center", va="bottom", fontsize=9.5, color="#444", style="italic")

    # leave space at the bottom for the caption
    fig.subplots_adjust(bottom=0.18)
    return fig


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_probing()
    print(f"Loaded {len(df)} probe rows over {df['variant'].nunique()} variants "
          f"× {df['treatment'].nunique()} treatments × {df['layer'].nunique()} layers "
          f"× {df['pooling'].nunique()} poolings.")
    fig = make_figure(df)

    out_png = OUT_DIR / "fig_probing_unified.png"
    out_pdf = OUT_DIR / "fig_probing_unified.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.18)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
