#!/usr/bin/env python3
"""Publication-quality visualizations for DML causal inference results.

Reads from output/results/:
  - all_experiments.csv   (treatment effects from analyze.py)
  - confounder_importances.csv (confounder feature importances)

Generates 6 plots in the same directory.

Usage:
  python pipeline/visualize.py
  python pipeline/visualize.py --input-dir output/results/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Styling ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#999999"
LIGHT_BLUE = "#92c5de"
LIGHT_RED = "#f4a582"


def _sig_color(p):
    """Return color by significance level."""
    if p < 0.01:
        return BLUE
    if p < 0.05:
        return LIGHT_BLUE
    if p < 0.1:
        return LIGHT_RED
    return GREY


def _sig_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _robust_xlim(values, margin=0.15):
    """Compute axis limits using IQR-based outlier fencing.

    Returns (lo, hi) that exclude extreme outliers so most data is visible.
    Values beyond the fence are clipped (and annotated by the caller).
    """
    vals = np.array([v for v in values if np.isfinite(v)])
    if len(vals) == 0:
        return -1, 1
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    fence = max(iqr * 3, 1.0)  # at least 1.0 range
    lo = max(vals.min(), q1 - fence)
    hi = min(vals.max(), q3 + fence)
    span = hi - lo if hi > lo else 1.0
    return lo - span * margin, hi + span * margin


# ── Plot 1: Treatment Forest Plot ────────────────────────────────────────────

def plot_treatment_forest(df, output_dir):
    """Forest plot: horizontal CIs for each treatment, 3 columns for outcomes. LGBM only."""
    sub = df[(df["learner"] == "lgbm") & (df["method"] == "plr") & df["coef"].notna()].copy()
    if sub.empty:
        print("  Skipping treatment_forest.png: no LGBM PLR results")
        return

    outcomes = [o for o in ["rank_delta", "pre_rank", "post_rank"] if o in sub["outcome"].unique()]
    if not outcomes:
        return

    treatments = sub["treatment"].unique()
    n_treat = len(treatments)

    fig, axes = plt.subplots(1, len(outcomes), figsize=(5 * len(outcomes), max(4, n_treat * 0.45)),
                             sharey=True)
    if len(outcomes) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes):
        oc_data = sub[sub["outcome"] == outcome].set_index("treatment")
        y_pos = np.arange(n_treat)

        # Compute robust x-limits from all coefs in this panel
        all_coefs = []
        all_ci = []
        for t in treatments:
            if t not in oc_data.index:
                continue
            row = oc_data.loc[t]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            all_coefs.append(row["coef"])
            all_ci.extend([row["ci_lower"], row["ci_upper"]])
        xlo, xhi = _robust_xlim(all_coefs + all_ci)

        for i, t in enumerate(treatments):
            if t not in oc_data.index:
                continue
            row = oc_data.loc[t]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            coef = row["coef"]
            ci_lo = row["ci_lower"]
            ci_hi = row["ci_upper"]
            p = row["p_val"]
            color = _sig_color(p)

            # Clip CIs to visible range; annotate if clipped
            vis_lo = max(ci_lo, xlo)
            vis_hi = min(ci_hi, xhi)
            vis_coef = np.clip(coef, xlo, xhi)

            ax.plot([vis_lo, vis_hi], [i, i], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(vis_coef, i, "o", color=color, markersize=7, zorder=5)

            if coef > xhi or coef < xlo:
                ax.annotate(f"{coef:.1f}", xy=(vis_coef, i), fontsize=6,
                            color=color, ha="left", va="bottom")

        ax.set_xlim(xlo, xhi)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t for t in treatments], fontsize=8)
        ax.set_xlabel("Causal Effect (theta)")
        ax.set_title(outcome)
        ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(color=BLUE, label="p < 0.01"),
        mpatches.Patch(color=LIGHT_BLUE, label="p < 0.05"),
        mpatches.Patch(color=LIGHT_RED, label="p < 0.1"),
        mpatches.Patch(color=GREY, label="n.s."),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.suptitle("DML Treatment Effects (PLR + LightGBM)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = output_dir / "treatment_forest.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 2: Treatment Heatmap ────────────────────────────────────────────────

def plot_treatment_heatmap(df, output_dir):
    """Heatmap: treatments (rows) x outcomes (cols). Cell = coef + stars. LGBM PLR only."""
    sub = df[(df["learner"] == "lgbm") & (df["method"] == "plr") & df["coef"].notna()].copy()
    if sub.empty:
        print("  Skipping treatment_heatmap.png: no data")
        return

    pivot_coef = sub.pivot_table(index="treatment", columns="outcome", values="coef")
    pivot_pval = sub.pivot_table(index="treatment", columns="outcome", values="p_val")

    if pivot_coef.empty:
        return

    # Reorder columns
    col_order = [c for c in ["rank_delta", "pre_rank", "post_rank"] if c in pivot_coef.columns]
    pivot_coef = pivot_coef[col_order]
    pivot_pval = pivot_pval[col_order]

    fig, ax = plt.subplots(figsize=(max(6, len(col_order) * 2.5), max(5, len(pivot_coef) * 0.5)))

    # Robust vmax: use IQR fence so outliers don't wash out the colormap
    flat = pivot_coef.values[~np.isnan(pivot_coef.values)]
    q1, q3 = np.percentile(np.abs(flat), [25, 75])
    iqr = q3 - q1
    vmax = min(np.abs(flat).max(), q3 + 3 * iqr)
    vmax = max(vmax, 0.1)  # minimum range
    im = ax.imshow(pivot_coef.values, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(col_order)))
    ax.set_yticks(np.arange(len(pivot_coef.index)))
    ax.set_xticklabels(col_order, fontsize=10)
    ax.set_yticklabels(pivot_coef.index, fontsize=9)

    for i in range(len(pivot_coef.index)):
        for j in range(len(col_order)):
            val = pivot_coef.values[i, j]
            pv = pivot_pval.values[i, j]
            if np.isnan(val):
                continue
            stars = _sig_stars(pv) if not np.isnan(pv) else ""
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}{stars}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Causal Effect (theta)", shrink=0.8)
    ax.set_title("Treatment Effects Heatmap (PLR + LightGBM)\nBlue = positive, Red = negative", fontsize=12)
    fig.tight_layout()
    path = output_dir / "treatment_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 3: Confounder Importance ─────────────────────────────────────────────

def plot_confounder_importance(output_dir):
    """Horizontal bar chart: importance + significance of each confounder on Y and D."""
    conf_path = output_dir / "confounder_importances.csv"
    if not conf_path.exists():
        print("  Skipping confounder_importance.png: no confounder_importances.csv")
        return

    conf_df = pd.read_csv(conf_path)
    if conf_df.empty:
        print("  Skipping confounder_importance.png: empty data")
        return

    has_pvals = "pval_outcome" in conf_df.columns

    # Average importance and median p-value per confounder
    avg = conf_df.groupby("confounder")[["importance_outcome", "importance_treatment"]].mean()
    avg = avg.sort_values("importance_outcome", ascending=True)

    if has_pvals:
        med_pval = conf_df.groupby("confounder")[["pval_outcome", "pval_treatment"]].median()
        med_pval = med_pval.loc[avg.index]  # align order

    n_conf = len(avg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, n_conf * 0.4)), sharey=True)
    y_pos = np.arange(n_conf)

    # ── Outcome panel (Y ~ X) ────────────────────────────────────────────
    if has_pvals:
        colors_y = [_sig_color(med_pval["pval_outcome"].iloc[i]) for i in range(n_conf)]
    else:
        colors_y = [BLUE] * n_conf

    ax1.barh(y_pos, avg["importance_outcome"], color=colors_y, alpha=0.85, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(avg.index, fontsize=8)
    ax1.set_xlabel("Avg Feature Importance")
    ax1.set_title("Outcome Model (Y ~ X)")

    if has_pvals:
        for i in range(n_conf):
            imp = avg["importance_outcome"].iloc[i]
            pv = med_pval["pval_outcome"].iloc[i]
            stars = _sig_stars(pv)
            if stars:
                ax1.text(imp, i, f" {stars} p={pv:.3f}", va="center", ha="left", fontsize=7)

    # ── Treatment panel (D ~ X) ──────────────────────────────────────────
    if has_pvals:
        colors_d = [_sig_color(med_pval["pval_treatment"].iloc[i]) for i in range(n_conf)]
    else:
        colors_d = [RED] * n_conf

    ax2.barh(y_pos, avg["importance_treatment"], color=colors_d, alpha=0.85, edgecolor="white")
    ax2.set_xlabel("Avg Feature Importance")
    ax2.set_title("Treatment Model (D ~ X)")

    if has_pvals:
        for i in range(n_conf):
            imp = avg["importance_treatment"].iloc[i]
            pv = med_pval["pval_treatment"].iloc[i]
            stars = _sig_stars(pv)
            if stars:
                ax2.text(imp, i, f" {stars} p={pv:.3f}", va="center", ha="left", fontsize=7)

    # ── Legend ────────────────────────────────────────────────────────────
    if has_pvals:
        legend_elements = [
            mpatches.Patch(color=BLUE, label="p < 0.01"),
            mpatches.Patch(color=LIGHT_BLUE, label="p < 0.05"),
            mpatches.Patch(color=LIGHT_RED, label="p < 0.1"),
            mpatches.Patch(color=GREY, label="n.s."),
        ]
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=8,
                   title="OLS significance\n(median across experiments)", title_fontsize=7)

    fig.suptitle("Confounder Importance & Statistical Significance\n"
                 "(LightGBM importance + OLS p-values, averaged across treatments & outcomes)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    path = output_dir / "confounder_importance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 4: DML vs OLS ───────────────────────────────────────────────────────

def plot_dml_vs_ols(df, output_dir):
    """Scatter: DML theta (x) vs OLS beta (y). 45-degree line shows agreement."""
    sub = df[df["coef"].notna() & df["ols_coef"].notna() & (df["method"] == "plr")].copy()
    if sub.empty:
        print("  Skipping dml_vs_ols.png: no paired DML/OLS data")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # Robust axis limits
    all_vals = np.concatenate([sub["coef"].values, sub["ols_coef"].values])
    xlo, xhi = _robust_xlim(all_vals, margin=0.1)

    for outcome in sub["outcome"].unique():
        oc = sub[sub["outcome"] == outcome]
        ax.scatter(oc["coef"], oc["ols_coef"], label=outcome, alpha=0.7, s=50, edgecolors="black",
                   linewidths=0.5)

        # Annotate outliers clipped beyond visible range
        for _, row in oc.iterrows():
            if row["coef"] > xhi or row["coef"] < xlo or row["ols_coef"] > xhi or row["ols_coef"] < xlo:
                cx = np.clip(row["coef"], xlo, xhi)
                cy = np.clip(row["ols_coef"], xlo, xhi)
                ax.annotate(f'{row["treatment"]}', xy=(cx, cy), fontsize=5,
                            color="red", ha="left")

    # 45-degree line
    ax.plot([xlo, xhi], [xlo, xhi], "k--", linewidth=0.8, alpha=0.5, label="DML = OLS")
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(xlo, xhi)

    ax.set_xlabel("DML Causal Effect (theta)")
    ax.set_ylabel("OLS Coefficient (beta)")
    ax.set_title("DML vs OLS Estimates\nPoints far from diagonal = confounding bias corrected by DML")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    path = output_dir / "dml_vs_ols.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 5: P-value Heatmap ──────────────────────────────────────────────────

def plot_pvalue_heatmap(df, output_dir):
    """P-value heatmap: treatments x (outcome_learner). Cleaner version."""
    sub = df[df["p_val"].notna()].copy()
    if sub.empty:
        print("  Skipping pvalue_heatmap.png: no data")
        return

    sub["config"] = sub["outcome"] + "\n" + sub["learner"]
    pivot = sub.pivot_table(index="treatment", columns="config", values="p_val", aggfunc="first")

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), max(5, len(pivot) * 0.5)))

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.15)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha="center", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            stars = _sig_stars(val)
            text_color = "white" if val < 0.08 else "black"
            ax.text(j, i, f"{val:.3f}{stars}", ha="center", va="center",
                    fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    ax.set_title("P-value Heatmap (PLR): Treatments x (Outcome, Learner)", fontsize=12)
    fig.tight_layout()
    path = output_dir / "pvalue_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Plot 6: Rank Interpretation ───────────────────────────────────────────────

def plot_rank_interpretation(df, output_dir):
    """Annotated bar chart for rank_delta only. Plain-English annotations."""
    sub = df[(df["outcome"] == "rank_delta") & (df["learner"] == "lgbm") &
             (df["method"] == "plr") & df["coef"].notna()].copy()
    if sub.empty:
        print("  Skipping rank_interpretation.png: no rank_delta LGBM PLR data")
        return

    sub = sub.sort_values("coef", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.5)))

    # Robust x-limits
    xlo, xhi = _robust_xlim(sub["coef"].values, margin=0.25)

    y_pos = np.arange(len(sub))
    clipped_coefs = np.clip(sub["coef"].values, xlo, xhi)
    colors = [BLUE if c > 0 else RED for c in sub["coef"]]

    ax.barh(y_pos, clipped_coefs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["treatment"].values, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Causal Effect on rank_delta (positions)")

    # Annotations — place text outward from bar end, but flip inside if clipped
    span = xhi - xlo
    x_offset = span * 0.02

    for i, (_, row) in enumerate(sub.iterrows()):
        coef = row["coef"]
        pval = row["p_val"]
        stars = _sig_stars(pval)
        vis_coef = np.clip(coef, xlo, xhi)
        is_clipped = (coef > xhi or coef < xlo)

        if coef > 0:
            direction = "LLM promotes"
        else:
            direction = "LLM demotes"

        val_str = f"{abs(coef):.2f}" if not is_clipped else f"{abs(coef):.0f}"
        annotation = f"{direction} by {val_str} pos {stars}"

        if is_clipped:
            # Place annotation inside the bar (toward zero) so it stays visible
            if coef >= 0:
                ax.text(vis_coef - x_offset, i, annotation, va="center", ha="right",
                        fontsize=7, color="white", style="italic", fontweight="bold")
            else:
                ax.text(vis_coef + x_offset, i, annotation, va="center", ha="left",
                        fontsize=7, color="white", style="italic", fontweight="bold")
        else:
            if coef >= 0:
                ax.text(vis_coef + x_offset, i, annotation, va="center", ha="left",
                        fontsize=7, color="black", style="italic")
            else:
                ax.text(vis_coef - x_offset, i, annotation, va="center", ha="right",
                        fontsize=7, color="black", style="italic")

    # Legend
    legend_elements = [
        mpatches.Patch(color=BLUE, label="LLM promotes (positive delta)"),
        mpatches.Patch(color=RED, label="LLM demotes (negative delta)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_title(
        "Causal Effect on LLM Re-Ranking (rank_delta = pre_rank - post_rank)\n"
        "Positive = LLM moves page UP in ranking | PLR + LightGBM",
        fontsize=11,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    path = output_dir / "rank_interpretation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize DML causal inference results")
    parser.add_argument("--input-dir", type=str, default="pipeline/results_llama3.3-70b_plr_lgbm+rf_new-10treat_3out_5fold/",
                        help="Directory containing all_experiments.csv and confounder_importances.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    experiments_path = input_dir / "all_experiments.csv"

    if not experiments_path.exists():
        print(f"Not found: {experiments_path}")
        print("Run pipeline/analyze.py first.")
        return

    df = pd.read_csv(experiments_path)
    print(f"Loaded {len(df)} experiment rows from {experiments_path}")

    print("\nGenerating plots:")
    plot_treatment_forest(df, input_dir)
    plot_treatment_heatmap(df, input_dir)
    plot_confounder_importance(input_dir)
    plot_dml_vs_ols(df, input_dir)
    plot_pvalue_heatmap(df, input_dir)
    plot_rank_interpretation(df, input_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
