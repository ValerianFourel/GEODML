#!/usr/bin/env python3
"""Canonical-set figures (v4, paper-polish + crystal-clear sign conventions).

The core readability rule: EVERY coefficient axis is plotted in promoter
direction so that "positive = good for the URL" holds universally:

    Y_1 = selected_by_llm        coef > 0   →  URL more likely admitted   (good)
    Y_2 = pre_rank - post_rank   coef > 0   →  URL moved UP toward top    (good)
    Y_3 = post_rank              coef < 0   →  URL placed closer to top   (good)

For Y_3 the plotted value is -beta so that positive bars / right-side dots
mean "promoter" just like Y_1 / Y_2.  The ORIGINAL coef is preserved in the
right-hand annotation table on the forest plots, so the reader can recover
the raw model output.

Every figure with a coefficient axis carries a small two-tone "promoter
direction" indicator above the axis (red = demote | blue = promote → URL).

Reads:  ~/geodml_data/data/dml_results/dml_canonical_2026-05-24.parquet
Writes: docs/2026-05-24/figures_canonical/tmp/{fig??_*.pdf, .png}  (300 DPI)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import stats as scstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as REPO, DML, FIGURES as OUT  # noqa: E402

DML_PARQUET = DML / "dml_canonical_2026-05-25_llms_as_confounder.parquet"

# ── Treatments ── (T7 = has_llms_txt dropped 2026-05-25: LLM never reads
# the file at inference time, so any coefficient was confounded not causal.)
TREATMENTS = [
    "treat_stats_density",
    "treat_question_headings",
    "treat_structured_data",
    "T4_citation_authority_code",
    "treat_topical_comp",
    "treat_freshness",
]
TREATMENT_LABELS = {
    "treat_stats_density":         r"T1    stats density",
    "treat_question_headings":     r"T2    Q-headings",
    "treat_structured_data":       r"T3    schema (JSON-LD)",
    "T4_citation_authority_code":  r"T4    citation authority",
    "treat_topical_comp":          r"T5    topical comp.",
    "treat_freshness":             r"T6    freshness",
}
TREATMENT_SHORT = {
    "treat_stats_density":         "T1",
    "treat_question_headings":     "T2",
    "treat_structured_data":       "T3",
    "T4_citation_authority_code":  "T4",
    "treat_topical_comp":          "T5",
    "treat_freshness":             "T6",
}

# ── Variants ──
VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
VARIANT_LABELS = {
    "biased":      "biased  (snippet)",
    "neutral":     "neutral (snippet)",
    "biased_rag":  "biased  (RAG)",
    "neutral_rag": "neutral (RAG)",
}
VARIANT_COLORS = {
    "biased":      "#a50f15",
    "neutral":     "#f1894c",
    "biased_rag":  "#08306b",
    "neutral_rag": "#6baed6",
}

# ── Outcomes ──
OUTCOMES = ["selected", "rank_delta", "post_rank"]
OUTCOME_PROMOTER_SIGN = {"selected": +1, "rank_delta": +1, "post_rank": -1}

# Three short outcome titles (math-mode)
M_Y1 = r"$Y_1\!:\;$admission"
M_Y2 = r"$Y_2 \!=\! \Delta\,\mathrm{rank}$"
M_Y3 = r"$Y_3 \!=\! \mathrm{rank\_post}$"
OUTCOME_TITLE = {"selected": M_Y1, "rank_delta": M_Y2, "post_rank": M_Y3}

# What "positive in promoter direction" means in words, per outcome
OUTCOME_PROMOTER_WORDS = {
    "selected":   r"URL more likely to be admitted by the LLM",
    "rank_delta": r"LLM moves URL UP from its SERP position",
    "post_rank":  r"LLM places URL closer to the top (rank_post $\downarrow$)",
}

# ── Palette ──
C_PROMOTER = "#08519c"   # deep blue
C_DEMOTER  = "#a50f15"   # brick red
C_NULL     = "#969696"
C_BOTH     = "#005a32"   # forest green
C_ASYM     = "#cc4c02"   # dark amber


def setup_style() -> None:
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "font.size":         12.5,
        "axes.titlesize":    14.0,
        "axes.labelsize":    12.5,
        "xtick.labelsize":   11.0,
        "ytick.labelsize":   12.0,
        "legend.fontsize":   11.0,
        "legend.frameon":    False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "axes.titlepad":     22,
        "axes.titleweight":  "bold",
        "axes.linewidth":    1.0,
        "lines.linewidth":   1.6,
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.30,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    if p < 1e-1: return "·"
    return ""


def _sig_color(coef: float, p: float, threshold: float = 0.05) -> str:
    """Color by significance + promoter direction (positive plotted coef = promoter)."""
    if pd.isna(p) or p >= threshold:
        return C_NULL
    return C_PROMOTER if coef > 0 else C_DEMOTER


def save(fig: plt.Figure, name: str) -> None:
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf); fig.savefig(png); plt.close(fig)
    print(f"    wrote {pdf.relative_to(REPO)}  +  {png.name}")


def panel_label(ax, text: str, *, x: float = -0.10, y: float = 1.06):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=15, fontweight="bold", ha="left", va="bottom", color="#111")


# ── DIRECTION HINT — subtle text under the x-axis (no ugly arrows) ───────


def direction_hint(ax, *, y_offset_below_axis: float = -0.13):  # noqa: ARG001
    """No-op (user removed demoter/promoter axis hints — kept for call-site compat)."""
    return


def slice_df(df, outcome, spec, slc) -> pd.DataFrame:
    sub = df[(df.outcome == outcome) & (df.spec == spec) & (df.slice == slc)].copy()
    sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
    sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]
    return sub.set_index("treatment").reindex(TREATMENTS).reset_index()


# ── Forest with promoter-direction axis + right-hand table ───────────────


def forest_with_table(sub: pd.DataFrame, *, title: str, subtitle: str,
                      x_axis_label: str, original_coef_units: str,
                      promoter_sign: int = +1, decimals: int = 4,
                      show_plotted_in_table: bool = False) -> plt.Figure:
    """Forest, plotted in promoter direction. Right-hand table shows ORIGINAL
    beta (sign as fitted) by default, so the reader sees both the visualized
    direction and the raw model output.  Set show_plotted_in_table=True when
    the original sign would be visually inconsistent with the bar direction
    (use this for promoter_sign=-1, e.g. fig03 rank_post)."""
    sub = sub.copy()
    sub["coef_plot"]  = promoter_sign * sub["coef"]
    sub["ci_lo_plot"] = promoter_sign * sub["ci_lo"]
    sub["ci_hi_plot"] = promoter_sign * sub["ci_hi"]
    if promoter_sign < 0:
        sub[["ci_lo_plot", "ci_hi_plot"]] = sub[["ci_hi_plot", "ci_lo_plot"]].values
    sub = sub.sort_values("coef_plot", ascending=True).reset_index(drop=True)
    sub["label"] = sub["treatment"].map(TREATMENT_LABELS).fillna(sub["treatment"])
    sub["stars"] = sub["p_val"].apply(_stars)

    fig = plt.figure(figsize=(14.0, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 2.0], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    tx = fig.add_subplot(gs[1]); tx.set_axis_off()

    # subtle alternating row backgrounds for readability
    for y in range(len(sub)):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)

    ys = np.arange(len(sub))
    for y, r in zip(ys, sub.itertuples()):
        sig = r.p_val < 0.05
        col = (C_PROMOTER if r.coef_plot > 0 else C_DEMOTER) if sig else C_NULL
        face = col if sig else "white"
        xerr = [[max(0, r.coef_plot - r.ci_lo_plot)],
                [max(0, r.ci_hi_plot - r.coef_plot)]]
        ax.errorbar(r.coef_plot, y, xerr=xerr, fmt="o",
                    color=col, markerfacecolor=face, markeredgecolor=col,
                    markeredgewidth=2.0, markersize=12,
                    capsize=5, elinewidth=1.8, zorder=3)

    ax.axvline(0, color="#888", linestyle="--", linewidth=1, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels(sub["label"], family="monospace")
    xs_lo = sub[["coef_plot","ci_lo_plot","ci_hi_plot"]].min().min()
    xs_hi = sub[["coef_plot","ci_lo_plot","ci_hi_plot"]].max().max()
    span = xs_hi - xs_lo
    ax.set_xlim(xs_lo - 0.10 * span, xs_hi + 0.10 * span)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.set_xlabel(x_axis_label)

    # right-hand table
    tx.set_xlim(0, 1); tx.set_ylim(-0.6, len(sub) - 0.4)
    tx.text(0.02, len(sub) - 0.45,
            rf"$\hat\beta$  ({original_coef_units})",
            ha="left", va="bottom", fontsize=12.5, fontweight="bold", color="#111")
    tx.text(0.50, len(sub) - 0.45, "95 % CI",
            ha="left", va="bottom", fontsize=12.5, fontweight="bold", color="#111")
    fmt = f"{{:+.{decimals}f}}"
    for y, r in zip(ys, sub.itertuples()):
        sig = r.p_val < 0.05
        col = (C_PROMOTER if r.coef_plot > 0 else C_DEMOTER) if sig else "#555"
        if y % 2 == 0:
            tx.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)
        # When show_plotted_in_table is set, display the promoter-direction
        # value (matching the bar).  Otherwise display the raw fitted coef.
        if show_plotted_in_table:
            tbl_coef  = r.coef_plot
            tbl_ci_lo = r.ci_lo_plot
            tbl_ci_hi = r.ci_hi_plot
        else:
            tbl_coef  = r.coef
            tbl_ci_lo = r.ci_lo
            tbl_ci_hi = r.ci_hi
        ci_text = f"[{tbl_ci_lo:+.{decimals}f}, {tbl_ci_hi:+.{decimals}f}]"
        tx.text(0.02, y, f"{fmt.format(tbl_coef)} {r.stars}",
                ha="left", va="center", fontsize=12,
                color=col, family="monospace",
                fontweight="bold" if sig else "normal")
        tx.text(0.50, y, ci_text,
                ha="left", va="center", fontsize=11.5, color="#555",
                family="monospace")
    return fig


def fig01_admission_forest(df):
    sub = slice_df(df, "selected", "B", "POOLED")
    fig = forest_with_table(
        sub,
        title=r"Admission stage — $Y_1$ = does the LLM include the URL?",
        subtitle=r"positive $\hat\beta$ $\Rightarrow$ URL more likely to be admitted   (treatment promotes inclusion)",
        x_axis_label=r"$\hat\beta$  on  $\Pr(\mathrm{admit})$  —  mutually-controlled  (single $T$ + 6 other $T$ + 28 $X$)",
        original_coef_units=r"log-odds, original",
        promoter_sign=+1, decimals=4)
    save(fig, "fig01_admission_forest")


def fig02_rank_delta_forest(df):
    sub = slice_df(df, "rank_delta", "B", "POOLED")
    fig = forest_with_table(
        sub,
        title=r"Ranking stage — $Y_2 = \Delta\,\mathrm{rank} = \mathrm{rank\_pre} - \mathrm{rank\_post}$",
        subtitle=r"positive $\hat\beta$ $\Rightarrow$ LLM moves URL UP from SERP   "
                 r"(rank\_pre$=$5, rank\_post$=$2  $\Rightarrow$  $\Delta=+3$)",
        x_axis_label=r"$\hat\beta$  on  $\Delta\,\mathrm{rank}$  —  mutually-controlled",
        original_coef_units=r"rank positions",
        promoter_sign=+1, decimals=3)
    save(fig, "fig02_rank_delta_forest")


def fig03_post_rank_forest(df):
    sub = slice_df(df, "post_rank", "B", "POOLED")
    fig = forest_with_table(
        sub,
        title=r"Ranking stage — $Y_3 = \mathrm{rank\_post}$  (LLM's final position;  $\mathrm{rank\_post}=1$ best)",
        subtitle=r"axis flipped: positive bar = promoter   (raw $\hat\beta < 0$ on $\mathrm{rank\_post}$ = closer to top)",
        x_axis_label=r"$-\hat\beta$  on  $\mathrm{rank\_post}$  —  mutually-controlled",
        original_coef_units=r"$-\hat\beta$, promoter dir.",
        promoter_sign=-1, decimals=3,
        show_plotted_in_table=True)
    save(fig, "fig03_post_rank_forest")


# ── FIG 04 — three-outcome dashboard ─────────────────────────────────────


def fig04_three_outcome_grid(df):
    rows = []
    for t in TREATMENTS:
        row = {"treatment": t}
        for o in OUTCOMES:
            rec = df[(df.outcome == o) & (df.spec == "B") & (df.slice == "POOLED")
                     & (df.treatment == t)].iloc[0]
            z = rec.coef / rec.se if rec.se > 0 else 0
            row[o + "_z_promo"] = z * OUTCOME_PROMOTER_SIGN[o]  # promoter-direction z
            row[o + "_p"]       = rec.p_val
            row[o + "_coef"]    = rec.coef                       # ORIGINAL
        rows.append(row)
    G = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.0, 7.5))
    grid = G[[f"{o}_z_promo" for o in OUTCOMES]].values
    vmax = max(5.0, float(np.nanmax(np.abs(grid))))
    cmap = LinearSegmentedColormap.from_list(
        "promoter_demoter", [(0.0, C_DEMOTER), (0.5, "#fcfcfc"), (1.0, C_PROMOTER)])
    im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=+vmax, aspect="auto")

    for i, t in enumerate(TREATMENTS):
        for j, o in enumerate(OUTCOMES):
            stars = _stars(G.iloc[i][f"{o}_p"])
            coef = G.iloc[i][f"{o}_coef"]
            ax.text(j, i - 0.10, f"{coef:+.4f}",
                    ha="center", va="center", fontsize=12,
                    color="#111", fontweight="bold", family="monospace")
            if stars:
                ax.text(j, i + 0.27, stars,
                        ha="center", va="center", fontsize=13, color="#111")
    ax.set_xticks(range(3))
    ax.set_xticklabels([OUTCOME_TITLE[o] for o in OUTCOMES], fontsize=13)
    ax.set_yticks(range(len(TREATMENTS)))
    ax.set_yticklabels([TREATMENT_LABELS[t] for t in TREATMENTS], family="monospace")
    ax.tick_params(axis="x", which="both", pad=12)

    cbar = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85)
    cbar.set_label(r"promoter-direction  $z = \hat\beta\cdot\mathrm{sign}/\mathrm{SE}$",
                   fontsize=11.5)
    cbar.ax.tick_params(labelsize=10.5)
    save(fig, "fig04_three_outcome_grid")


# ── FIG 05 — admission vs rank scatter ───────────────────────────────────


def fig05_admission_vs_rank_scatter(df):
    """REDESIGN — side-by-side forest plots replacing the quadrant scatter.
    Same treatment ordering on both panels (sorted by admission β)."""
    a  = slice_df(df, "selected",   "B", "POOLED")
    rd = slice_df(df, "rank_delta", "B", "POOLED")

    a = a.sort_values("coef", ascending=True).reset_index(drop=True)
    order = a["treatment"].tolist()
    rd = rd.set_index("treatment").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.8), sharey=True)
    plt.subplots_adjust(wspace=0.06, left=0.18)

    for ax, sub, title, xlabel in [
        (axes[0], a,  r"$Y_1$ : admission",
                       r"$\hat\beta$  on  $\Pr(\mathrm{admit})$"),
        (axes[1], rd, r"$Y_2 = \Delta\,\mathrm{rank}$",
                       r"$\hat\beta$  on  $\Delta\,\mathrm{rank}$"),
    ]:
        ys = np.arange(len(sub))
        for y in ys:
            if y % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)

        for y, r in zip(ys, sub.itertuples()):
            sig = r.p_val < 0.05
            col = (C_PROMOTER if r.coef > 0 else C_DEMOTER) if sig else C_NULL
            face = col if sig else "white"
            ax.errorbar(r.coef, y, xerr=1.96 * r.se, fmt="o",
                        color=col, markerfacecolor=face,
                        markeredgecolor=col, markeredgewidth=2.0,
                        markersize=12, capsize=5, elinewidth=1.7, zorder=3)
            stars = _stars(r.p_val)
            ax.text(r.coef + 1.96 * r.se,  y + 0.30,
                    f"{r.coef:+.3f} {stars}",
                    ha="left", va="center", fontsize=10, color=col,
                    family="monospace", fontweight="bold" if sig else "normal")

        ax.axvline(0, color="#888", linestyle="--", linewidth=1, zorder=1)
        ax.set_yticks(ys)
        if ax is axes[0]:
            ax.set_yticklabels([TREATMENT_LABELS[t] for t in sub["treatment"]],
                               family="monospace", fontsize=12.5)
        ax.set_xlabel(xlabel, fontsize=13)
        xs_lo = (sub["coef"] - 1.96 * sub["se"]).min()
        xs_hi = (sub["coef"] + 1.96 * sub["se"]).max()
        span = xs_hi - xs_lo
        ax.set_xlim(xs_lo - 0.10 * span, xs_hi + 0.25 * span)

    legend = [
        Patch(facecolor=C_PROMOTER, label="promoter (sig)"),
        Patch(facecolor=C_DEMOTER,  label="demoter (sig)"),
        Patch(facecolor="white", edgecolor=C_NULL, label="not significant"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False, fontsize=12)
    save(fig, "fig05_admission_vs_rank_scatter")
    return  # short-circuit; everything below is the old (now-dead) scatter code

    # ─── dead code below — kept temporarily to preserve diff history ───
    M = a[["treatment","coef","se","p_val"]].rename(
        columns={"coef":"coef_a","se":"se_a","p_val":"p_a"}).merge(
        rd[["treatment","coef","se","p_val"]].rename(
            columns={"coef":"coef_r","se":"se_r","p_val":"p_r"}), on="treatment")
    M["label"] = M["treatment"].map(TREATMENT_SHORT)

    fig, ax = plt.subplots(figsize=(12.0, 10.5))

    # Pre-compute axis limits so we can quadrant-shade BEFORE plotting markers
    pad = 0.20
    x_lo = float(min(M.coef_a.min() - 1.96 * M.se_a.max(), 0))
    x_hi = float(max(M.coef_a.max() + 1.96 * M.se_a.max(), 0))
    y_lo = float(min(M.coef_r.min() - 1.96 * M.se_r.max(), 0))
    y_hi = float(max(M.coef_r.max() + 1.96 * M.se_r.max(), 0))
    x_span = x_hi - x_lo; y_span = y_hi - y_lo
    x_lo -= pad * x_span; x_hi += pad * x_span
    y_lo -= pad * y_span; y_hi += pad * y_span
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    # Quadrant shading
    ax.axhspan(0,    y_hi, xmin=0.5, xmax=1.0, color=C_BOTH,    alpha=0.07, zorder=0)
    ax.axhspan(y_lo, 0,    xmin=0.0, xmax=0.5, color=C_DEMOTER, alpha=0.07, zorder=0)
    ax.axhspan(0,    y_hi, xmin=0.0, xmax=0.5, color="#cccccc", alpha=0.06, zorder=0)
    ax.axhspan(y_lo, 0,    xmin=0.5, xmax=1.0, color=C_ASYM,    alpha=0.07, zorder=0)

    ax.axhline(0, color="#888", linewidth=1.1, linestyle="--", zorder=1)
    ax.axvline(0, color="#888", linewidth=1.1, linestyle="--", zorder=1)

    # Per-treatment offsets to avoid label clashes (chosen manually for the
    # canonical 7).
    label_offsets = {
        "T1":      (16,  -6),    # cluster near origin → push right & down
        "T2":      (18,  12),
        "T3":      (-22, -14),   # bottom-left, label SW
        "T4":      (18,  -16),   # near origin, push right & down
        "T5":      (18,  10),    # far bottom-right
        "T6":      (-20, -14),
        "T7":      (18,  10),
    }

    for _, row in M.iterrows():
        both_sig = (row.p_a < 0.05) and (row.p_r < 0.05)
        same_dir = np.sign(row.coef_a) == np.sign(row.coef_r)
        if both_sig and same_dir and row.coef_a > 0:
            col = C_BOTH
        elif both_sig and same_dir and row.coef_a < 0:
            col = C_DEMOTER
        elif both_sig and not same_dir:
            col = C_ASYM
        elif (row.p_a < 0.05) ^ (row.p_r < 0.05):
            col = "#3182bd"
        else:
            col = C_NULL
        ax.errorbar(row.coef_a, row.coef_r,
                    xerr=1.96*row.se_a, yerr=1.96*row.se_r,
                    fmt="o", color=col, markersize=20, capsize=4.5, elinewidth=1.5,
                    markeredgecolor="white", markeredgewidth=2.0, zorder=3)
        dx, dy = label_offsets.get(row.label, (16, 10))
        ha = "left" if dx >= 0 else "right"
        ax.annotate(row.label,
                    (row.coef_a, row.coef_r),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=15, fontweight="bold", color=col, ha=ha, zorder=4)

    # Quadrant tags moved to the EXTREME corners with bigger padding
    qpad_x = 0.04 * (x_hi - x_lo); qpad_y = 0.04 * (y_hi - y_lo)
    quadrant_kw = dict(fontsize=13, fontstyle="italic", alpha=0.95, fontweight="bold")
    ax.text(x_hi - qpad_x, y_hi - qpad_y, "promote\nboth stages",
            ha="right", va="top", color=C_BOTH, **quadrant_kw)
    ax.text(x_lo + qpad_x, y_lo + qpad_y, "demote\nboth stages",
            ha="left",  va="bottom", color=C_DEMOTER, **quadrant_kw)
    ax.text(x_hi - qpad_x, y_lo + qpad_y, "admit only\n(stage-asymmetric)",
            ha="right", va="bottom", color=C_ASYM, **quadrant_kw)
    ax.text(x_lo + qpad_x, y_hi - qpad_y, "rank only\n(stage-asymmetric)",
            ha="left",  va="top", color="#666", **quadrant_kw)

    ax.set_xlabel(r"$\hat\beta_{\mathrm{admit}}$    "
                  r"(mutually-controlled coef on $\Pr(\mathrm{admit})$;  "
                  r"positive = promoter)",
                  fontsize=14)
    ax.set_ylabel(r"$\hat\beta_{\Delta\mathrm{rank}}$    "
                  r"(mutually-controlled coef on $\Delta\,\mathrm{rank}$;  "
                  r"positive = promoter)",
                  fontsize=14)
    # title removed per user request — quadrant tags + legend carry meaning.

    legend = [Patch(facecolor=C_BOTH,    label="promote both (sig)"),
              Patch(facecolor=C_DEMOTER, label="demote both (sig)"),
              Patch(facecolor=C_ASYM,    label="stage-asymmetric"),
              Patch(facecolor="#3182bd", label="sig on one stage"),
              Patch(facecolor=C_NULL,    label="neither sig")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=5, frameon=False, fontsize=12)
    save(fig, "fig05_admission_vs_rank_scatter")


# ── FIG 06 — marginal vs partial trajectories ────────────────────────────


def fig06_marginal_vs_partial(df):
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.8))
    plt.subplots_adjust(wspace=0.30, top=0.86)
    panel_labels = ["(a)", "(b)", "(c)"]
    # Human-readable subtitles for each outcome, shown under the panel letter.
    panel_subs = {
        "selected":   r"admission  $Y_1$  —  was URL kept by LLM",
        "rank_delta": r"$\Delta\mathrm{rank}=\mathrm{rank}_{\mathrm{pre}}-\mathrm{rank}_{\mathrm{post}}$  —  how far LLM moved URL",
        "post_rank":  r"$-\mathrm{rank}_{\mathrm{post}}$  —  final LLM position (sign flipped)",
    }
    # Fix y-axis order ONCE (by selected/Spec-B promoter-direction coef) and
    # apply it to all 3 panels so the same row = the same treatment across
    # panels (essential for visual comparison).
    sel_b = slice_df(df, "selected", "B", "POOLED")
    sel_b["plot_b"] = OUTCOME_PROMOTER_SIGN["selected"] * sel_b["coef"]
    fixed_order = sel_b.sort_values("plot_b")["treatment"].tolist()
    order_index = {t: i for i, t in enumerate(fixed_order)}

    # See-through markers: face alpha 0.45 (sig) so overlapping circles+squares
    # both stay visible; NS markers are open (white) with a coloured ring.
    MFC_ALPHA = 0.45
    for ax, o, pl in zip(axes, OUTCOMES, panel_labels):
        a = slice_df(df, o, "A", "POOLED")
        b = slice_df(df, o, "B", "POOLED")
        sign = OUTCOME_PROMOTER_SIGN[o]
        m = a[["treatment","coef","se","p_val"]].rename(
            columns={"coef":"coef_a","se":"se_a","p_val":"p_a"}).merge(
            b[["treatment","coef","se","p_val"]].rename(
                columns={"coef":"coef_b","se":"se_b","p_val":"p_b"}), on="treatment")
        m["plot_a"] = sign * m["coef_a"]
        m["plot_b"] = sign * m["coef_b"]
        # use SHORT labels so they fit on the y-axis without bleeding into the
        # neighbouring panel.
        m["label"] = m["treatment"].map(TREATMENT_SHORT)

        m["__ord"] = m["treatment"].map(order_index)
        m = m.sort_values("__ord").reset_index(drop=True)
        ys = np.arange(len(m))
        for y in ys:
            if y % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)
        for y, row in zip(ys, m.itertuples()):
            sig_a = row.p_a < 0.05; sig_b = row.p_b < 0.05
            line_c = "#333" if (sig_a and sig_b) else "#cccccc"
            col_a = _sig_color(row.plot_a, row.p_a)
            col_b = _sig_color(row.plot_b, row.p_b)
            ax.plot([row.plot_a, row.plot_b], [y, y],
                    color=line_c, linewidth=1.7, alpha=0.7, zorder=2)
            # Marginal (circle). Face uses RGBA so it stays see-through even
            # when the square sits on top.
            face_a = (*to_rgb(col_a), MFC_ALPHA) if sig_a else "white"
            ax.errorbar(row.plot_a, y, xerr=1.96*row.se_a, fmt="o",
                        ecolor=col_a, markersize=13, capsize=3, elinewidth=1.2, zorder=3,
                        markerfacecolor=face_a,
                        markeredgecolor=col_a, markeredgewidth=1.8)
            # Mutually-controlled (square).
            face_b = (*to_rgb(col_b), MFC_ALPHA) if sig_b else "white"
            ax.errorbar(row.plot_b, y, xerr=1.96*row.se_b, fmt="s",
                        ecolor=col_b, markersize=13, capsize=3, elinewidth=1.2, zorder=4,
                        markerfacecolor=face_b,
                        markeredgecolor=col_b, markeredgewidth=1.8)
        ax.axvline(0, color="#888", linestyle="--", linewidth=1, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(m["label"].tolist(), family="monospace", fontsize=12.5)
        ax.set_xlabel(r"$\hat\beta$  in promoter direction")
        # Panel letter on the left, outcome description on the right of the title row
        panel_label(ax, pl)
        ax.set_title(panel_subs[o], loc="left", x=-0.02, pad=10,
                     fontsize=12.5, color="#222")

    legend = [
        plt.Line2D([0],[0], marker='o', color="#222", linestyle="",
                   markerfacecolor=(*to_rgb("#222"), MFC_ALPHA), markeredgecolor="#222",
                   markeredgewidth=1.6, markersize=12,
                   label="marginal  (single $T$ + 28 confounders)"),
        plt.Line2D([0],[0], marker='s', color="#222", linestyle="",
                   markerfacecolor=(*to_rgb("#222"), MFC_ALPHA), markeredgecolor="#222",
                   markeredgewidth=1.6, markersize=12,
                   label="mutually-controlled  (joint with other 5 $T$)"),
        Patch(facecolor=(*to_rgb(C_PROMOTER), MFC_ALPHA), edgecolor=C_PROMOTER,
              linewidth=1.4, label="promoter (sig)"),
        Patch(facecolor=(*to_rgb(C_DEMOTER), MFC_ALPHA), edgecolor=C_DEMOTER,
              linewidth=1.4, label="demoter (sig)"),
        Patch(facecolor="white", edgecolor=C_NULL, linewidth=1.4, label="not significant"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=5, fontsize=11.5, frameon=False)
    save(fig, "fig06_marginal_vs_partial")


# ── FIG 07/08 — variant grids ────────────────────────────────────────────


def _variant_grid(df, outcome, ax):
    sub = df[(df.outcome == outcome) & (df.spec == "A")
             & (df.slice.str.startswith("VAR:"))].copy()
    sub["variant"] = sub["slice"].str.replace("VAR:", "", regex=False)
    sub["sig"] = sub["p_val"] < 0.05
    pv = sub.pivot(index="treatment", columns="variant", values="coef").reindex(TREATMENTS)
    pv_sig = sub.pivot(index="treatment", columns="variant", values="sig").reindex(TREATMENTS)

    # plot in promoter direction
    sign = OUTCOME_PROMOTER_SIGN[outcome]
    pv = pv * sign

    bar_w = 0.20
    x_pos = np.arange(len(TREATMENTS))
    for j, v in enumerate(VARIANTS):
        offset = (j - 1.5) * bar_w
        heights = pv[v].values.astype(float)
        sigs = pv_sig[v].values.astype(bool)
        colors = [VARIANT_COLORS[v] if s else "#eeeeee" for s in sigs]
        edges  = [VARIANT_COLORS[v] if s else "#bbbbbb" for s in sigs]
        # No label= here — we build a static legend below to avoid matplotlib
        # picking up the colour of the FIRST bar (which is grey when T1 is NS
        # in every variant, e.g. rank_delta, breaking the legend swatches).
        ax.bar(x_pos + offset, heights, bar_w,
               color=colors, edgecolor=edges, linewidth=1.1)
    ax.axhline(0, color="#333", linewidth=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([TREATMENT_SHORT[t] for t in TREATMENTS], fontsize=12.5)
    ax.set_ylabel(r"$\hat\beta$  in promoter direction")


def _variant_legend_handles():
    handles = [Patch(facecolor=VARIANT_COLORS[v], edgecolor=VARIANT_COLORS[v],
                     label=VARIANT_LABELS[v]) for v in VARIANTS]
    handles.append(Patch(facecolor="#eeeeee", edgecolor="#bbbbbb",
                         label="NS (p ≥ 0.05)"))
    return handles


def fig07_variant_grid_admission(df):
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    _variant_grid(df, "selected", ax)
    fig.legend(handles=_variant_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=11)
    save(fig, "fig07_variant_grid_admission")


def fig08_variant_grid_rank(df):
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    _variant_grid(df, "rank_delta", ax)
    fig.legend(handles=_variant_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, fontsize=11)
    save(fig, "fig08_variant_grid_rank")


# ── NEW: Difference forest helper for figs 09–12 ─────────────────────────


def _diff_forest_panel(ax, df, outcome, slice_a, slice_b):
    """One panel: per-treatment Δβ = β(slice_b) − β(slice_a), plotted in
    promoter direction.  CI = ±1.96·sqrt(SE_a² + SE_b²)  (independent fits).
    Returns the DataFrame.  Does NOT set yticklabels (caller handles that
    on axes[0] only, to play nicely with sharey=True)."""
    sign = OUTCOME_PROMOTER_SIGN[outcome]
    rows = []
    for t in TREATMENTS:
        ra = df[(df.outcome == outcome) & (df.spec == "A") & (df.slice == slice_a)
                & (df.treatment == t)].iloc[0]
        rb = df[(df.outcome == outcome) & (df.spec == "A") & (df.slice == slice_b)
                & (df.treatment == t)].iloc[0]
        delta = sign * (rb.coef - ra.coef)
        delta_se = math.sqrt(ra.se ** 2 + rb.se ** 2)
        z = delta / delta_se if delta_se > 0 else 0.0
        p_delta = float(2 * (1 - scstats.norm.cdf(abs(z))))
        rows.append({"treatment": t, "delta": delta, "delta_se": delta_se,
                     "p_delta": p_delta})
    D = pd.DataFrame(rows)
    D = D.set_index("treatment").reindex(TREATMENTS).reset_index()
    ys = np.arange(len(D))[::-1]

    for y in ys:
        if (len(D) - 1 - y) % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)

    for y, r in zip(ys, D.itertuples()):
        sig = r.p_delta < 0.05
        col = (C_PROMOTER if r.delta > 0 else C_DEMOTER) if sig else C_NULL
        face = col if sig else "white"
        ax.errorbar(r.delta, y, xerr=1.96 * r.delta_se, fmt="o",
                    color=col, markerfacecolor=face,
                    markeredgecolor=col, markeredgewidth=2.0,
                    markersize=11, capsize=4.5, elinewidth=1.5, zorder=3)
        stars = _stars(r.p_delta)
        ax.text(r.delta + 1.96 * r.delta_se, y + 0.30,
                f"{r.delta:+.3f} {stars}",
                ha="left", va="center", fontsize=9.5, color=col,
                family="monospace", fontweight="bold" if sig else "normal")

    ax.axvline(0, color="#888", linestyle="--", linewidth=1, zorder=1)
    ax.set_yticks(ys)
    ax.set_ylim(-0.6, len(D) - 0.4)
    return D


# ── Comparison scatters in PROMOTER direction ────────────────────────────


def _comparison_scatter(df, *, slice_x: str, slice_y: str,
                         label_x: str, label_y: str,
                         supertitle: str, subtitle: str, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.5))
    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, o, pl in zip(axes, OUTCOMES, panel_labels):
        sign = OUTCOME_PROMOTER_SIGN[o]
        sx = slice_df(df, o, "A", slice_x).set_index("treatment").reindex(TREATMENTS)
        sy = slice_df(df, o, "A", slice_y).set_index("treatment").reindex(TREATMENTS)
        # promoter direction
        sx_plot = sign * sx["coef"]; sy_plot = sign * sy["coef"]

        cis_lo = pd.concat([sx_plot - 1.96*sx["se"], sy_plot - 1.96*sy["se"]])
        cis_hi = pd.concat([sx_plot + 1.96*sx["se"], sy_plot + 1.96*sy["se"]])
        lo = float(min(cis_lo.min(), 0)); hi = float(max(cis_hi.max(), 0))
        span = hi - lo
        lo -= 0.12 * span; hi += 0.12 * span

        # quadrant shading: top-right quadrant = both promote
        ax.axhspan(0, hi, xmin=0.5, xmax=1.0, color=C_PROMOTER, alpha=0.05, zorder=0)
        ax.axhspan(lo, 0, xmin=0.0, xmax=0.5, color=C_DEMOTER,  alpha=0.05, zorder=0)

        ax.plot([lo, hi], [lo, hi], color="#bbbbbb", linestyle="--", linewidth=1.3, zorder=1)
        ax.axhline(0, color="#999", linewidth=0.9, zorder=1)
        ax.axvline(0, color="#999", linewidth=0.9, zorder=1)

        for t in TREATMENTS:
            rx_p = sx.loc[t, "p_val"]; ry_p = sy.loc[t, "p_val"]
            xv = sign * sx.loc[t, "coef"]; yv = sign * sy.loc[t, "coef"]
            xs_se = sx.loc[t, "se"]; ys_se = sy.loc[t, "se"]
            both_sig = (rx_p < 0.05) and (ry_p < 0.05)
            either_sig = (rx_p < 0.05) or (ry_p < 0.05)
            if both_sig:    fc = "#1f1f1f"
            elif either_sig: fc = "#7f7f7f"
            else:            fc = "white"
            ax.errorbar(xv, yv,
                        xerr=1.96*xs_se, yerr=1.96*ys_se,
                        fmt="o", color="#1f1f1f", markerfacecolor=fc,
                        markeredgecolor="#1f1f1f", markeredgewidth=1.6,
                        markersize=14, capsize=4, elinewidth=1.2, zorder=3)
            on_above = yv > xv
            dx, dy = (13, 13) if on_above else (13, -16)
            ax.annotate(TREATMENT_SHORT[t],
                        (xv, yv),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=12, fontweight="bold", color="#111", zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(label_x, fontweight="bold")
        ax.set_ylabel(label_y, fontweight="bold")
        ax.set_title(OUTCOME_TITLE[o], loc="left", pad=22)
        panel_label(ax, pl)

    legend = [
        plt.Line2D([0],[0], marker='o', markerfacecolor="#1f1f1f", markeredgecolor="#1f1f1f",
                   markersize=12, linestyle="", label="sig in both"),
        plt.Line2D([0],[0], marker='o', markerfacecolor="#7f7f7f", markeredgecolor="#1f1f1f",
                   markersize=12, linestyle="", label="sig in one"),
        plt.Line2D([0],[0], marker='o', markerfacecolor="white", markeredgecolor="#1f1f1f",
                   markersize=12, linestyle="", label="neither sig"),
        plt.Line2D([0],[0], color="#bbb", linestyle="--", label=r"identity $y=x$"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.025),
               ncol=4, fontsize=11.5)
    fig.suptitle(supertitle, fontsize=15, y=1.10, x=0.04, ha="left", fontweight="bold")
    fig.text(0.04, 1.05, subtitle, fontsize=11.5, color="#444", fontstyle="italic",
             transform=fig.transFigure)
    save(fig, filename)


def fig09_compare_rag_vs_snippet(df):
    """REDESIGN — difference forest.  Per (outcome, treatment), shows
    Δβ = β_RAG − β_snippet (in promoter direction).  Two markers per
    treatment row: one for biased prompts, one for neutral prompts.
    Replaces the cross-style scatter."""
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.0), sharey=True)
    plt.subplots_adjust(wspace=0.06)
    panel_labels = ["(a)", "(b)", "(c)"]

    for ax, o, pl in zip(axes, OUTCOMES, panel_labels):
        sign = OUTCOME_PROMOTER_SIGN[o]
        rows = []
        for t in TREATMENTS:
            r = {"treatment": t}
            for prompt_tag, snip_slice, rag_slice in [
                ("biased",  "VAR:biased",  "VAR:biased_rag"),
                ("neutral", "VAR:neutral", "VAR:neutral_rag"),
            ]:
                sn = df[(df.outcome == o) & (df.spec == "A") & (df.slice == snip_slice)
                        & (df.treatment == t)].iloc[0]
                rg = df[(df.outcome == o) & (df.spec == "A") & (df.slice == rag_slice)
                        & (df.treatment == t)].iloc[0]
                delta = sign * (rg.coef - sn.coef)
                delta_se = math.sqrt(sn.se ** 2 + rg.se ** 2)
                z = delta / delta_se if delta_se > 0 else 0
                p_delta = float(2 * (1 - scstats.norm.cdf(abs(z))))
                r[f"{prompt_tag}_d"] = delta
                r[f"{prompt_tag}_se"] = delta_se
                r[f"{prompt_tag}_p"] = p_delta
            rows.append(r)
        D = pd.DataFrame(rows)
        D = D.set_index("treatment").reindex(TREATMENTS).reset_index()

        ys = np.arange(len(D))[::-1]
        for y in ys:
            if (len(D) - 1 - y) % 2 == 0:
                ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)

        for y, row in zip(ys, D.itertuples()):
            # biased on top of row, neutral below
            for offset, prompt_tag, prompt_color in [
                (+0.22, "biased",  "#a50f15"),
                (-0.22, "neutral", "#08306b"),
            ]:
                d  = getattr(row, f"{prompt_tag}_d")
                se = getattr(row, f"{prompt_tag}_se")
                p  = getattr(row, f"{prompt_tag}_p")
                sig = p < 0.05
                ax.errorbar(d, y + offset, xerr=1.96 * se, fmt="o",
                            color=prompt_color,
                            markerfacecolor=(prompt_color if sig else "white"),
                            markeredgecolor=prompt_color, markeredgewidth=1.8,
                            markersize=10, capsize=4, elinewidth=1.3, zorder=3)
        ax.axvline(0, color="#888", linestyle="--", linewidth=1, zorder=1)
        ax.set_yticks(ys)
        if ax is axes[0]:
            ax.set_yticklabels([TREATMENT_LABELS[t] for t in D["treatment"]],
                               family="monospace", fontsize=12)
        ax.set_xlabel(r"$\Delta\hat\beta$  =  $\hat\beta_{\mathrm{RAG}}$  −  "
                      r"$\hat\beta_{\mathrm{snippet}}$    (promoter direction)",
                      fontsize=12)
        panel_label(ax, pl)

    legend = [
        plt.Line2D([0],[0], marker='o', color="#a50f15", markerfacecolor="#a50f15",
                   markersize=11, linestyle="", label="biased prompt"),
        plt.Line2D([0],[0], marker='o', color="#08306b", markerfacecolor="#08306b",
                   markersize=11, linestyle="", label="neutral prompt"),
        plt.Line2D([0],[0], marker='o', color="#666", markerfacecolor="white",
                   markeredgecolor="#666", markersize=11, linestyle="",
                   label=r"$\Delta$ not sig"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, fontsize=11.5)
    save(fig, "fig09_compare_rag_vs_snippet")


def _three_panel_diff_forest(df, *, slice_a, slice_b,
                              label_a_plain, label_b_plain,
                              label_a_math, label_b_math,
                              supertitle, filename):
    """Common layout for figs 10/11/12 — one Δβ marker per treatment per
    outcome panel.  Labels use math-mode (label_*_math, sans $$ wrapper) for
    axis labels; plain text (label_*_plain) for the legend."""
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 7.0), sharey=True)
    plt.subplots_adjust(wspace=0.06, left=0.16)
    panel_labels = ["(a)", "(b)", "(c)"]
    for i, (ax, o, pl) in enumerate(zip(axes, OUTCOMES, panel_labels)):
        D = _diff_forest_panel(ax, df, o, slice_a, slice_b)
        if i == 0:
            ax.set_yticklabels([TREATMENT_LABELS[t] for t in D["treatment"]],
                               family="monospace", fontsize=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel(rf"$\Delta\hat\beta = \hat\beta_{{{label_b_math}}} - "
                      rf"\hat\beta_{{{label_a_math}}}$    (promoter direction)",
                      fontsize=12)
        panel_label(ax, pl)

    legend = [
        Patch(facecolor=C_PROMOTER,
              label=f"{label_b_plain} more promoter (sig)"),
        Patch(facecolor=C_DEMOTER,
              label=f"{label_a_plain} more promoter (sig)"),
        Patch(facecolor="white", edgecolor=C_NULL,
              label=r"$\Delta$ not sig"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, fontsize=11.5)
    save(fig, filename)


def fig10_compare_ddg_vs_searxng(df):
    _three_panel_diff_forest(df,
        slice_a="ENG:ddg", slice_b="ENG:searxng",
        label_a_plain="DDG", label_b_plain="SearXNG",
        label_a_math=r"\mathrm{DDG}", label_b_math=r"\mathrm{SearXNG}",
        supertitle=r"DDG vs SearXNG  —  SERP-engine-induced shift in each "
                   r"treatment effect  ($\Delta\hat\beta$ in promoter direction)",
        filename="fig10_compare_ddg_vs_searxng")


def fig11_compare_llama_vs_qwen(df):
    _three_panel_diff_forest(df,
        slice_a="MOD:Llama", slice_b="MOD:Qwen2.5",
        label_a_plain="Llama", label_b_plain="Qwen",
        label_a_math=r"\mathrm{Llama}", label_b_math=r"\mathrm{Qwen}",
        supertitle=r"Llama vs Qwen  —  model-induced shift in each treatment "
                   r"effect  ($\Delta\hat\beta$ in promoter direction)",
        filename="fig11_compare_llama_vs_qwen")


def fig12_compare_pool20_vs_pool50(df):
    _three_panel_diff_forest(df,
        slice_a="POOL:20", slice_b="POOL:50",
        label_a_plain="pool=20", label_b_plain="pool=50",
        label_a_math=r"\mathrm{pool}=20", label_b_math=r"\mathrm{pool}=50",
        supertitle=r"Pool=20 vs pool=50  —  pool-size-induced shift in each "
                   r"treatment effect  ($\Delta\hat\beta$ in promoter direction)",
        filename="fig12_compare_pool20_vs_pool50")


# ── FIG 13 — admission detail ────────────────────────────────────────────


def fig13_admission_detail(df):
    a = slice_df(df, "selected", "B", "POOLED").sort_values("coef", ascending=True).reset_index(drop=True)
    p0 = 0.5837

    def delta_p(beta, p0): return p0 * (1 - p0) * beta

    fig = plt.figure(figsize=(15.0, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.8], wspace=0.07)
    ax = fig.add_subplot(gs[0])
    tx = fig.add_subplot(gs[1]); tx.set_axis_off()

    ys = np.arange(len(a))
    for y in ys:
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)
            tx.axhspan(y - 0.5, y + 0.5, color="#f6f6f6", zorder=0)
    for y, r in zip(ys, a.itertuples()):
        sig = r.p_val < 0.05
        col = _sig_color(r.coef, r.p_val)
        face = col if sig else "white"
        ax.errorbar(r.coef, y, xerr=1.96*r.se, fmt="o", color=col,
                    markerfacecolor=face, markeredgecolor=col, markeredgewidth=1.9,
                    markersize=12, capsize=4.5, elinewidth=1.5, zorder=3)
    ax.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax.set_yticks(ys); ax.set_yticklabels(
        [TREATMENT_LABELS[t] for t in a.treatment], family="monospace")
    ax.set_xlabel(r"DML log-odds  $\hat\beta$  on  $\Pr(Y_1\!=\!1)$  —  mutually-controlled, POOLED")
    ax.set_ylim(-0.6, len(a) - 0.4)

    tx.set_xlim(0, 1); tx.set_ylim(-0.6, len(a) - 0.4)
    headers = [(0.03, r"$\hat\beta$  (log-odds)"),
               (0.40, r"OR $= e^{\hat\beta}$"),
               (0.70, r"$\Delta p$  (pp)")]
    for x, h in headers:
        tx.text(x, len(a) - 0.45, h, ha="left", va="bottom",
                fontsize=12.5, fontweight="bold", color="#111")
    for y, r in zip(ys, a.itertuples()):
        sig = r.p_val < 0.05
        col = _sig_color(r.coef, r.p_val) if sig else "#666"
        OR = math.exp(r.coef); dp = delta_p(r.coef, p0) * 100
        stars = _stars(r.p_val)
        tx.text(0.03, y, f"{r.coef:+.4f} {stars}",
                ha="left", va="center", fontsize=12, color=col,
                family="monospace", fontweight="bold" if sig else "normal")
        tx.text(0.40, y, f"{OR:.4f}",
                ha="left", va="center", fontsize=12, color="#444",
                family="monospace")
        tx.text(0.70, y, f"{dp:+.3f}",
                ha="left", va="center", fontsize=12, color="#444",
                family="monospace")
    save(fig, "fig13_admission_detail")


# ── FIG 14 — robust survivors ────────────────────────────────────────────


def fig14_robust_survivors(df):
    a = slice_df(df, "selected", "B", "POOLED")
    rd = slice_df(df, "rank_delta", "B", "POOLED")
    pr = slice_df(df, "post_rank", "B", "POOLED")
    BF = 0.05 / 7

    def classify(t):
        ra = a[a.treatment == t].iloc[0]
        rr = rd[rd.treatment == t].iloc[0]
        rp = pr[pr.treatment == t].iloc[0]
        prom_a = ra.coef > 0; prom_r = rr.coef > 0; prom_p = rp.coef < 0
        a_sig = ra.p_val < BF; r_sig = rr.p_val < BF; p_sig = rp.p_val < BF
        if a_sig and r_sig and p_sig:
            same = (prom_a == prom_r == prom_p)
            if same and prom_a:  return "promote all 3 (consistent)", C_BOTH, ra, rr, rp
            elif same:           return "demote all 3 (consistent)",  C_DEMOTER, ra, rr, rp
            else:                return "mixed direction",            C_ASYM, ra, rr, rp
        if a_sig and (r_sig or p_sig):  return "admit + 1 rank stage", "#3182bd", ra, rr, rp
        if r_sig and p_sig:             return "both rank stages",    "#7570b3", ra, rr, rp
        if a_sig:                       return "admit only",          "#9ecae1", ra, rr, rp
        if r_sig or p_sig:              return "rank only (1 stage)", "#bcbddc", ra, rr, rp
        return                                 "none (after Bonferroni)", C_NULL, ra, rr, rp

    rows = []
    for t in TREATMENTS:
        cat, col, ra, rr, rp = classify(t)
        rows.append({"treatment": t, "label": TREATMENT_LABELS[t],
                     "category": cat, "color": col,
                     "coef_a": ra.coef, "coef_r": rr.coef, "coef_p": rp.coef,
                     "p_a": ra.p_val, "p_r": rr.p_val, "p_p": rp.p_val})
    G = pd.DataFrame(rows)
    cat_order = ["promote all 3 (consistent)", "demote all 3 (consistent)",
                 "mixed direction",
                 "admit + 1 rank stage", "both rank stages",
                 "admit only", "rank only (1 stage)",
                 "none (after Bonferroni)"]
    G["cat_rank"] = G["category"].map({c: i for i, c in enumerate(cat_order)})
    G = G.sort_values(["cat_rank", "treatment"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(17.0, 7.0))
    ys = np.arange(len(G))[::-1]
    col_x_treatment = 0.02
    col_xs   = [0.34, 0.55, 0.76]
    col_x_cat = 0.98
    headers = [r"$Y_1$  admission", r"$Y_2 \!=\! \Delta\,\mathrm{rank}$",
               r"$Y_3 \!=\! \mathrm{rank\_post}$"]
    dir_hints = ["+ = promoter", "+ = promoter", "− = promoter"]

    for y, row in zip(ys, G.itertuples()):
        ax.barh(y, 1.0, height=0.86, color=row.color, alpha=0.13, zorder=0)
        ax.text(col_x_treatment, y, row.label, ha="left", va="center",
                fontsize=12.5, color="#111", family="monospace")
        for x, val, p in zip(col_xs,
                              [row.coef_a, row.coef_r, row.coef_p],
                              [row.p_a, row.p_r, row.p_p]):
            ax.text(x, y, f"{val:+.4f} {_stars(p)}",
                    ha="center", va="center", fontsize=12,
                    family="monospace", color="#111",
                    fontweight="bold" if p < BF else "normal")
        ax.text(col_x_cat, y, row.category, ha="right", va="center",
                fontsize=12, color=row.color, fontweight="bold")

    # column headers (inside the plot area, no overhead subtitle)
    ax.text(col_x_treatment, len(G) + 0.4, "treatment", ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color="#111")
    for x, h, d in zip(col_xs, headers, dir_hints):
        ax.text(x, len(G) + 0.7, h, ha="center", va="bottom", fontsize=12.5,
                fontweight="bold", color="#111")
        ax.text(x, len(G) + 0.2, d, ha="center", va="bottom", fontsize=10.5,
                color="#666", fontstyle="italic")
    ax.text(col_x_cat, len(G) + 0.4, "joint category", ha="right", va="bottom",
            fontsize=12.5, fontweight="bold", color="#111")

    ax.set_xlim(0, 1); ax.set_ylim(-0.55, len(G) + 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    # title removed per user request — table is self-explanatory with its
    # column headers and direction hints.
    save(fig, "fig14_robust_survivors")


# ── main ─────────────────────────────────────────────────────────────────


def main():
    if not DML_PARQUET.exists():
        print(f"ERROR: {DML_PARQUET} not found.", file=sys.stderr)
        return 1
    setup_style()
    df = pd.read_parquet(DML_PARQUET)
    print(f"Loaded {DML_PARQUET.relative_to(Path.home())}  rows={len(df)}\n")

    print("[01] admission forest");           fig01_admission_forest(df)
    print("[02] rank_delta forest");          fig02_rank_delta_forest(df)
    print("[03] post_rank forest");           fig03_post_rank_forest(df)
    print("[04] three-outcome grid");         fig04_three_outcome_grid(df)
    print("[05] admission vs rank scatter");  fig05_admission_vs_rank_scatter(df)
    print("[06] marginal vs partial");        fig06_marginal_vs_partial(df)
    print("[07] variant grid admission");     fig07_variant_grid_admission(df)
    print("[08] variant grid rank");          fig08_variant_grid_rank(df)
    print("[09] RAG vs snippet");             fig09_compare_rag_vs_snippet(df)
    print("[10] DDG vs SearXNG");             fig10_compare_ddg_vs_searxng(df)
    print("[11] Llama vs Qwen");              fig11_compare_llama_vs_qwen(df)
    print("[12] pool 20 vs 50");              fig12_compare_pool20_vs_pool50(df)
    print("[13] admission detail");           fig13_admission_detail(df)
    print("[14] robust survivors");           fig14_robust_survivors(df)

    print(f"\n  → 14 figures in {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
