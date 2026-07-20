#!/usr/bin/env python3
"""DML estimation framework diagram (paper figure, no data dependency).

Replaces  docs/2026-05-24/figures_canonical/tmp/dml_estimation_framework.webp
with a programmatically-drawn PDF + PNG that reflects the canonical study:

  - 7 content treatments  T  (was 9), with NO horizontal divider in the box
  - 28 confounders        X  (was 25), with NO horizontal divider in the box
  - 3 outcomes  Y_1 = selected,  Y_2 = Δrank,  Y_3 = post_rank
  - footer:  X = {28 confounders} ∪ {other 6 treatments}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO_ROOT as REPO, FIGURES as OUT_DIR  # noqa: E402

# Palette
CREAM_BG     = "#f5ecda"; CREAM_EDGE  = "#c9b48a"
BLUE_BG      = "#e2ecf6"; BLUE_EDGE   = "#7ea2c9"; BLUE_TEXT  = "#1f3b66"
GREY_BG      = "#ececec"; GREY_EDGE   = "#b8b8b8"; GREY_TEXT  = "#333333"
NAVY_BG      = "#3a5e8c"; NAVY_TEXT   = "#ffffff"
PEACH_BG     = "#fce7c1"; PEACH_EDGE  = "#d8a96a"; PEACH_TEXT = "#5b3b07"
PURPLE_BG    = "#dcd2eb"; PURPLE_EDGE = "#a59ac3"; PURPLE_TEXT= "#2a1c4a"
CALLOUT_BG   = "#fbfbfb"; CALLOUT_EDGE= "#cccccc"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.30,
        "pdf.fonttype":     42,
    })


def box(ax, xy, wh, *, fc, ec, linewidth=1.3, alpha=1.0, zorder=2,
        rounding=0.014):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.000,rounding_size={rounding}",
        linewidth=linewidth, edgecolor=ec, facecolor=fc, alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return (x, y, w, h)


def text(ax, xy, s, *, fontsize=12, color="#111", ha="center", va="center",
         fontweight="normal", fontstyle="normal", zorder=4, family=None):
    kw = dict(ha=ha, va=va, fontsize=fontsize, color=color,
              fontweight=fontweight, fontstyle=fontstyle, zorder=zorder)
    if family:
        kw["family"] = family
    ax.text(xy[0], xy[1], s, **kw)


def arrow(ax, p1, p2, *, color="#333", lw=1.4, mut=14, shrinkA=2, shrinkB=2,
          zorder=1):
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        color=color, linewidth=lw, mutation_scale=mut,
        shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder,
    )
    ax.add_patch(a)


def main():
    setup_style()
    fig, ax = plt.subplots(figsize=(16.0, 11.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ── SERP query + rank_{pre} at top center  (bigger boxes, larger text) ──
    sq_w, sq_h = 0.300, 0.066
    sq_x = 0.500 - sq_w / 2
    sq_y = 0.902
    box(ax, (sq_x, sq_y), (sq_w, sq_h), fc=CREAM_BG, ec=CREAM_EDGE)
    text(ax, (sq_x + sq_w / 2, sq_y + sq_h / 2),
         r"SERP query   (DDG / SearXNG)",
         fontsize=16, fontweight="bold")

    rp_w, rp_h = 0.300, 0.066
    rp_x = 0.500 - rp_w / 2
    rp_y = 0.812
    box(ax, (rp_x, rp_y), (rp_w, rp_h), fc=CREAM_BG, ec=CREAM_EDGE)
    text(ax, (rp_x + rp_w / 2, rp_y + rp_h / 2),
         r"$\mathrm{rank}_{\mathrm{pre}}$   (original SERP position)",
         fontsize=16, fontweight="bold")
    arrow(ax, (0.500, sq_y), (0.500, rp_y + rp_h))

    # ── Treatments box (LEFT, single column, NO divider line) ────────
    tx, ty, tw, th = 0.035, 0.495, 0.310, 0.305
    box(ax, (tx, ty), (tw, th), fc=BLUE_BG, ec=BLUE_EDGE, linewidth=1.5)

    text(ax, (tx + tw / 2, ty + th - 0.034),
         r"6 content treatments   $T$",
         fontsize=15, fontweight="bold", color=BLUE_TEXT)

    treatments = [
        (r"$T_{1}$",  "stats density"),
        (r"$T_{2}$",  "Q-headings"),
        (r"$T_{3}$",  "schema (JSON-LD)"),
        (r"$T_{4}$",  "citation authority"),
        (r"$T_{5}$",  "topical comp."),
        (r"$T_{6}$",  "freshness"),
    ]
    sym_x = tx + 0.050
    lbl_x = tx + 0.135
    row_top = ty + th - 0.085
    row_gap = 0.030
    for i, (sym, lbl) in enumerate(treatments):
        y = row_top - i * row_gap
        text(ax, (sym_x, y), sym, fontsize=13.5, fontweight="bold",
             color=BLUE_TEXT, ha="left")
        text(ax, (lbl_x, y), lbl, fontsize=12.5, color="#111", ha="left")

    # ── Confounders box (RIGHT, NO divider line) ─────────────────────
    cx, cy, cw, ch = 0.655, 0.495, 0.310, 0.305
    box(ax, (cx, cy), (cw, ch), fc=GREY_BG, ec=GREY_EDGE, linewidth=1.5)
    text(ax, (cx + cw / 2, cy + ch - 0.034),
         r"28 confounders   $X$",
         fontsize=15, fontweight="bold", color=GREY_TEXT)

    # All 28 confounders enumerated explicitly in 2 columns.
    conf_left = [
        "word count",
        "readability",
        "internal links",
        "outbound links",
        "images alt",
        "HTTPS",
        "title has kw",
        "title length",
        "snippet length",
        "title-kw sim",
        "snippet-kw sim",
        "BM25 score",
        "SERP position",
        "brand recognition",
    ]
    conf_right = [
        "domain authority",
        "backlinks",
        "referring domains",
        "DFS paid count",
        "DFS ETV",
        "domain age (yrs)",
        "kw difficulty",
        "search volume",
        "CPC",
        "competition",
        "intent: commercial",
        "intent: informational",
        "intent: navigational",
        "intent: transactional",
    ]
    assert len(conf_left) + len(conf_right) == 28
    col_l_x = cx + 0.018
    col_r_x = cx + cw / 2 + 0.005
    conf_top = cy + ch - 0.068
    conf_gap = 0.016
    conf_fs  = 10.5
    for i in range(len(conf_left)):
        y = conf_top - i * conf_gap
        text(ax, (col_l_x, y), conf_left[i], fontsize=conf_fs, color="#222",
             ha="left", family="monospace")
        text(ax, (col_r_x, y), conf_right[i], fontsize=conf_fs, color="#222",
             ha="left", family="monospace")

    # ── LLM RERANKER (center, filled navy) ───────────────────────────
    rx, ry, rw, rh = 0.305, 0.340, 0.390, 0.135
    box(ax, (rx, ry), (rw, rh), fc=NAVY_BG, ec=NAVY_BG, linewidth=0,
        rounding=0.018)
    text(ax, (rx + rw / 2, ry + rh - 0.040),
         "LLM   RERANKER",
         fontsize=17, fontweight="bold", color=NAVY_TEXT)
    text(ax, (rx + rw / 2, ry + rh - 0.075),
         r"Llama-3.3-70B-Instruct       Qwen2.5-72B-Instruct",
         fontsize=12, color=NAVY_TEXT)
    text(ax, (rx + rw / 2, ry + rh - 0.107),
         r"prompt $\in$ {biased, neutral}       "
         r"evidence $\in$ {snippet, RAG}",
         fontsize=11.5, color=NAVY_TEXT)

    # Arrows into the reranker — converge on the TOP edge (clean down-into-box).
    # Treatments enter top-left, confounders enter top-right, rank_pre enters
    # top-center.  All arrowheads land on (ry + rh).
    llm_top = ry + rh
    arrow(ax, (tx + tw - 0.060, ty),        # bottom-right of T box
              (rx + 0.070,       llm_top))
    arrow(ax, (cx + 0.060,       cy),        # bottom-left of C box
              (rx + rw - 0.070,  llm_top))
    arrow(ax, (0.500, rp_y),                 # bottom of rank_pre
              (0.500, llm_top))

    # ── LLM outputs box ──────────────────────────────────────────────
    out_x, out_y, out_w, out_h = 0.350, 0.262, 0.300, 0.056
    box(ax, (out_x, out_y), (out_w, out_h),
        fc=CREAM_BG, ec=CREAM_EDGE)
    text(ax, (out_x + out_w / 2, out_y + out_h / 2),
         r"(selected, $\mathrm{rank\_post}$)   —   LLM outputs",
         fontsize=13, fontweight="bold")
    arrow(ax, (0.500, ry), (0.500, out_y + out_h))

    # ── Three outcomes (peach) ───────────────────────────────────────
    y_box_y, y_box_h = 0.115, 0.110
    out_boxes = [
        (0.040, 0.155, r"$Y_1 \;=\; \mathrm{selected}_{\mathrm{by\;LLM}}$",
                       "binary admission"),
        (0.385, 0.500, r"$Y_2 \;=\; \Delta\,\mathrm{rank} \;=\; "
                       r"\mathrm{rank\_pre} - \mathrm{rank\_post}$",
                       "directional displacement"),
        (0.730, 0.845, r"$Y_3 \;=\; \mathrm{rank\_post}$",
                       "absolute position   (1 = best)"),
    ]
    for x0, x_center, top_t, sub_t in out_boxes:
        box(ax, (x0, y_box_y), (0.230, y_box_h),
            fc=PEACH_BG, ec=PEACH_EDGE, linewidth=1.4)
        text(ax, (x_center, y_box_y + y_box_h - 0.034), top_t,
             fontsize=13, fontweight="bold", color=PEACH_TEXT)
        text(ax, (x_center, y_box_y + y_box_h - 0.078), sub_t,
             fontsize=11.5, color="#7a5413", fontstyle="italic")

    # Outputs → Y outcomes: clean tree (vertical drop → horizontal manifold →
    # three vertical arrows down to each Y box top).
    y_top = y_box_y + y_box_h
    mani_y = (out_y + y_top) / 2  # midway between outputs box bottom and Y tops
    ax.plot([0.500, 0.500], [out_y, mani_y], color="#666", lw=1.2, zorder=1)
    ax.plot([0.155, 0.845], [mani_y, mani_y], color="#666", lw=1.2, zorder=1)
    arrow(ax, (0.155, mani_y), (0.155, y_top), color="#666", lw=1.2, mut=12)
    arrow(ax, (0.500, mani_y), (0.500, y_top), color="#666", lw=1.2, mut=12)
    arrow(ax, (0.845, mani_y), (0.845, y_top), color="#666", lw=1.2, mut=12)

    # ── DML equation footer ──────────────────────────────────────────
    fx, fy, fw, fh = 0.080, 0.010, 0.840, 0.090
    box(ax, (fx, fy), (fw, fh), fc=PURPLE_BG, ec=PURPLE_EDGE, linewidth=1.3)
    text(ax, (fx + fw / 2, fy + fh - 0.024),
         r"$\hat\theta_T \;=\; \mathrm{DML}(\,T,\;Y_j;\;X\,)$  "
         r"—  estimated separately for each treatment  $T$  and each outcome  "
         r"$Y_j \in \{Y_1, Y_2, Y_3\}$",
         fontsize=13, fontweight="bold", color=PURPLE_TEXT)
    text(ax, (fx + fw / 2, fy + fh - 0.050),
         r"$X \;=\; \{\,28\;\mathrm{confounders}\,\}\;\;\cup\;\;"
         r"\{\,6\;\mathrm{treatments}\,\}$",
         fontsize=12, color="#3c2c5e")
    text(ax, (fx + fw / 2, fy + fh - 0.076),
         r"cross-fitted LightGBM,   $K\!=\!5$  folds        "
         r"robust SE via influence function        "
         r"marginal $=$ single $T$,   mutually-controlled $=$ joint with other $T$",
         fontsize=11.5, color="#3c2c5e")

    # Save
    pdf = OUT_DIR / "dml_estimation_framework.pdf"
    png = OUT_DIR / "dml_estimation_framework.png"
    fig.savefig(pdf); fig.savefig(png); plt.close(fig)
    print(f"wrote {pdf.relative_to(REPO)}")
    print(f"wrote {png.relative_to(REPO)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
