"""
Programmatic causal DAG for the DML estimand of the LLM reranker.

Replicates `docs/2026-05-24/figures_canonical/tmp/preview (1).webp`.

Nodes
-----
- rank_pre  (pre-rerank position, the LLM input)
- T         (content treatments)
- X         (confounders)
- LLM       (the reranker mechanism)
- Y_2 = rank_post
- Delta rank = rank_pre - rank_post

Arrows
------
- solid black: causal edges  (X->T, T->Y_2, T->Delta, X->Y_2, X->Delta)
- teal: LLM mechanism        (rank_pre->LLM, LLM->Y_2)
- dashed grey: arithmetic    (Y_2->Delta and rank_pre->Delta)
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "2026-05-24" / "figures_canonical" / "tmp"


# --- styling ---------------------------------------------------------------

CAUSAL = "#1d1d1d"
TEAL   = "#2c7fb8"
GREY   = "#9a9a9a"

COL_T    = "#cfe2f3"
COL_X    = "#dcdcdc"
COL_RPRE = "#f1ece1"
COL_LLM  = "#2c5f8a"
COL_Y    = "#fde2c4"


def _box(ax, cx, cy, w, h, *, fc, ec="#5a5a5a", lw=0.9, zorder=2):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(p)


def _arrow(ax, p0, p1, *, color, lw=1.4, ls="-", mut=14, zorder=3):
    a = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>", mutation_scale=mut, linewidth=lw,
        color=color, linestyle=ls, zorder=zorder,
        shrinkA=0, shrinkB=0,
        joinstyle="round", capstyle="round",
    )
    ax.add_patch(a)


# --- node geometry (figure coords on a 0..1 canvas) -----------------------
# Boxes are described as (cx, cy, w, h) so we can anchor edges cleanly.

NODES = {
    # name      : (cx,   cy,   w,    h)
    "rank_pre" : (0.205, 0.770, 0.150, 0.072),
    "T"        : (0.545, 0.830, 0.105, 0.078),
    "X"        : (0.870, 0.830, 0.105, 0.078),
    "LLM"      : (0.270, 0.520, 0.200, 0.102),
    "Y2"       : (0.545, 0.240, 0.210, 0.078),
    "DR"       : (0.870, 0.240, 0.165, 0.078),
}


def edge(name, side, *, frac=0.0):
    """Return a point on the edge of node `name`.

    side: 'top' | 'bottom' | 'left' | 'right'
    frac: shift along the perpendicular axis (-0.5..0.5)
    """
    cx, cy, w, h = NODES[name]
    if side == "top":
        return (cx + frac * w, cy + h / 2)
    if side == "bottom":
        return (cx + frac * w, cy - h / 2)
    if side == "left":
        return (cx - w / 2, cy + frac * h)
    if side == "right":
        return (cx + w / 2, cy + frac * h)
    raise ValueError(side)


# --- figure ----------------------------------------------------------------

def make_figure():
    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.02)
    ax.axis("off")

    # ---- legend strip ----------------------------------------------------
    ax.text(0.50, 0.945,
            r"$T$: content treatments     "
            r"$X$: confounders     "
            r"$Y_{2}=\mathrm{rank}_{\mathrm{post}}$     "
            r"$\Delta\mathrm{rank}=\mathrm{rank}_{\mathrm{pre}}-\mathrm{rank}_{\mathrm{post}}$",
            ha="center", va="center", fontsize=11.5, color="#333",
            family="serif", style="italic")

    # ---- boxes -----------------------------------------------------------
    _box(ax, *NODES["rank_pre"], fc=COL_RPRE)
    ax.text(NODES["rank_pre"][0], NODES["rank_pre"][1],
            r"$\mathrm{rank}_{\mathrm{pre}}$",
            ha="center", va="center", fontsize=12)

    _box(ax, *NODES["T"], fc=COL_T)
    ax.text(NODES["T"][0], NODES["T"][1], r"$T$",
            ha="center", va="center", fontsize=16, family="serif", style="italic")

    _box(ax, *NODES["X"], fc=COL_X)
    ax.text(NODES["X"][0], NODES["X"][1], r"$X$",
            ha="center", va="center", fontsize=16, family="serif", style="italic")

    _box(ax, *NODES["LLM"], fc=COL_LLM, ec=COL_LLM)
    ax.text(NODES["LLM"][0], NODES["LLM"][1], "LLM",
            ha="center", va="center", fontsize=20, color="white", family="sans-serif")

    _box(ax, *NODES["Y2"], fc=COL_Y)
    ax.text(NODES["Y2"][0], NODES["Y2"][1],
            r"$Y_{2}\,=\,\mathrm{rank}_{\mathrm{post}}$",
            ha="center", va="center", fontsize=13)

    _box(ax, *NODES["DR"], fc=COL_Y)
    ax.text(NODES["DR"][0], NODES["DR"][1], r"$\Delta\mathrm{rank}$",
            ha="center", va="center", fontsize=14)

    # ---- X -> T  (confounding) ------------------------------------------
    _arrow(ax, edge("X", "left"), edge("T", "right"),
           color=CAUSAL, lw=1.6, mut=14)
    ax.text(0.710, 0.860, "confounding", ha="center", va="bottom",
            fontsize=11, color="#333", style="italic")

    # ---- T -> Y2 / Δrank,  X -> Y2 / Δrank  (4 black causal arrows) -----
    # offset slightly off-centre so the four arrowheads spread cleanly
    _arrow(ax, edge("T", "bottom", frac=-0.20), edge("Y2", "top", frac=-0.10),
           color=CAUSAL, lw=1.5)
    _arrow(ax, edge("T", "bottom", frac=+0.20), edge("DR", "top", frac=-0.20),
           color=CAUSAL, lw=1.5)
    _arrow(ax, edge("X", "bottom", frac=-0.30), edge("Y2", "top", frac=+0.18),
           color=CAUSAL, lw=1.5)
    _arrow(ax, edge("X", "bottom", frac=+0.10), edge("DR", "top", frac=+0.18),
           color=CAUSAL, lw=1.5)

    # ---- LLM mechanism (teal) -------------------------------------------
    # rank_pre -> LLM  (vertical drop, slight rightward lean)
    _arrow(ax, edge("rank_pre", "bottom", frac=+0.10),
               edge("LLM", "top", frac=-0.10),
           color=TEAL, lw=2.2, mut=18)
    # LLM -> Y2  (diagonal, labelled "rerank")
    llm_out = (NODES["LLM"][0] + NODES["LLM"][2] / 2 - 0.012,
               NODES["LLM"][1] - NODES["LLM"][3] / 2 + 0.012)
    y2_in   = (NODES["Y2"][0] - NODES["Y2"][2] / 2 + 0.012,
               NODES["Y2"][1] + NODES["Y2"][3] / 2 - 0.012)
    _arrow(ax, llm_out, y2_in, color=TEAL, lw=2.2, mut=18)
    ax.text(0.445, 0.395, "rerank", ha="center", va="center",
            fontsize=11, color=TEAL, style="italic")

    # ---- arithmetic edges (dashed grey) ---------------------------------
    # Y2 -> Δrank  (short horizontal)
    _arrow(ax, edge("Y2", "right"), edge("DR", "left"),
           color=GREY, lw=1.3, ls=(0, (4, 3)), mut=13)
    # rank_pre -> Δrank  (long curve sweeping BELOW the figure).
    # Start at the LEFT edge of rank_pre so the curve has room to swing wide,
    # end at the BOTTOM of Δrank so it enters from below.
    rp_anchor = edge("rank_pre", "left", frac=-0.30)
    dr_anchor = edge("DR", "bottom", frac=-0.15)
    curve = mpatches.FancyArrowPatch(
        rp_anchor, dr_anchor,
        connectionstyle="arc3,rad=0.55",
        arrowstyle="-|>", mutation_scale=13,
        color=GREY, lw=1.3, linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax.add_patch(curve)

    # ---- legend box (bottom-left), rendered on top so it masks the curve
    lx, ly = 0.280, -0.025
    lw, lh = 0.420, 0.150
    _box(ax, lx + lw / 2, ly + lh / 2, lw, lh, fc="white", ec="#bbb", lw=0.8, zorder=10)
    ax.text(lx + 0.014, ly + lh - 0.022, "Arrows",
            fontsize=11.5, ha="left", va="center", color="#222", zorder=11)

    def legend_row(y, color, ls, label, lw_line=1.7):
        ax.plot([lx + 0.020, lx + 0.078], [y, y],
                color=color, lw=lw_line, linestyle=ls, zorder=11,
                solid_capstyle="butt")
        ax.text(lx + 0.092, y, label, fontsize=10,
                ha="left", va="center", color="#222", zorder=11)

    legend_row(ly + lh - 0.058, CAUSAL, "-",          "solid black:  causal")
    legend_row(ly + lh - 0.090, TEAL,   "-",          "teal:  LLM mechanism")
    legend_row(ly + lh - 0.122, GREY,   (0, (4, 3)),  r"dashed grey:  $\Delta$rank arithmetic")

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    png = OUT_DIR / "dml_dag_estimand.png"
    pdf = OUT_DIR / "dml_dag_estimand.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
