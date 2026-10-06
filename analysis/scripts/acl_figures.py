#!/usr/bin/env python3
"""Figures for the ACL ARR paper (October 2026 draft).

  axis-examples  pick, for two keywords, one prompt near the start, the middle and the end of the
                 information-seeking -> action-readiness axis (seeded random choice inside each band).
                 Reads the frozen population and its final axis map; light enough for a login shell.
  render         draw the four figures as vector PDFs (plus PNG previews):
                   fig1-prompt-axis       the axis and the example prompts       (needs --axis-examples)
                   fig2-mini-internet     the closed corpus and its search        (constants below)
                   fig3-pipeline          every step from prompt to analysis      (constants below)
                   fig4-cited-vs-prompt   prompt position vs the position of the cited sources
                                          (needs --results: results.json of intent_stages_study.py analyze)

Every number drawn in fig2 and fig3 is a constant in TESTBED with its source; nothing is invented.
Fig4 draws only the descriptive curves stored by the analysis; no curve is drawn without data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Established counts (paper draft, evidence/claim-source-map.md; pasted cluster evidence of 2026-10-04/05).
TESTBED = {
    "keywords": 1011,
    "prompts": 26009,
    "ddg_rows": 10338,
    "searxng_rows": 13555,
    "rows": 23893,
    "pages": 21384,
    "glued_rows": 270,
    "hub_keywords": 530,
    "results_per_search": 20,
    "parallel_top_k": 7,
    "reactive_top_k": 3,
    "cells_per_prompt": 12,
}
BANDS = {"start": (0.0, 0.15), "middle": (0.425, 0.575), "end": (0.85, 1.0)}
ANCHORS = [(0.0, "understand /\nexplain"), (0.25, "investigate\nevidence"), (0.5, "evaluate\nalternatives"),
           (0.75, "prepare a\ndecision or plan"), (1.0, "act now:\nan enabling step")]
COLUMN, DOUBLE = 3.15, 6.3  # ACL single- and double-column widths in inches

INK, MUTED, LINE = "#1f2a30", "#5b6770", "#c9d1d6"
INFO, ACTION = "#2f6f9f", "#c0632b"  # information seeking (blue) -> action ready (orange)
MODEL = {"llama4": "#2f6f9f", "qwen38": "#c0632b"}
MODEL_LABEL = {"llama4": "Llama-4-Scout", "qwen38": "Qwen3.8"}
ENGINE_STYLE = {"duckduckgo": "-", "searxng": "--"}
ENGINE_LABEL = {"duckduckgo": "DuckDuckGo", "searxng": "SearXNG"}


# ---------------------------------------------------------------- axis examples

def axis_examples(final_axis_rows, population_rows, *, keywords=None, seed=20261006) -> dict:
    """One prompt per band for each chosen keyword. Without --keyword, the first keyword is
    'robinhood vs etrade' (the paper's running example) when it fills all bands, and the second is
    drawn at random among keywords that fill all three bands."""
    position = {str(r["candidate_id"]): float(r["axis_1_percentile_0_1"]) for r in final_axis_rows}
    by_keyword: dict[str, list] = {}
    for row in population_rows:
        cid = str(row["candidate_id"])
        if cid in position and row.get("keyword") and row.get("question"):
            by_keyword.setdefault(row["keyword"], []).append((position[cid], row["question"], cid))
    def banded(kw):
        return {b: sorted(p for p in by_keyword.get(kw, []) if lo <= p[0] <= hi) for b, (lo, hi) in BANDS.items()}
    eligible = sorted(k for k in by_keyword if all(banded(k).values()))
    rng = random.Random(seed)
    if keywords:
        chosen = list(keywords)
        missing = [k for k in chosen if k not in eligible]
        if missing:
            raise ValueError(f"keywords without a prompt in every band: {missing}")
    else:
        chosen = ["robinhood vs etrade"] if "robinhood vs etrade" in eligible else []
        pool = [k for k in eligible if k not in chosen]
        chosen += rng.sample(pool, 2 - len(chosen))
    out = []
    for kw in chosen:
        picks = {}
        for band, members in banded(kw).items():
            p, text, cid = rng.choice(members)
            picks[band] = {"position": p, "prompt": text, "candidate_id": cid}
        out.append({"keyword": kw, "prompts_in_keyword": len(by_keyword[kw]), "picks": picks})
    return {"seed": seed, "bands": BANDS, "eligible_keywords": len(eligible), "keywords": out}


# ---------------------------------------------------------------- figure 4 from generation rows only

def cited_from_generations(generation_rows, prompt_rows, page_rows, *, bootstrap=200, seed=20261006) -> dict:
    """Figure-4 curves and within-keyword slopes from the published generation rows alone (no traces).
    Each answer's cited position is the top-weighted (1/log2(r+1)) mean prompt-scale percentile of its
    ranked URLs; a URL seen with several snippet texts takes the mean over them. Rows are deduplicated
    on (model, prompt, engine, method, condition). Only the ranking stage K is available this way."""
    import numpy as np
    from analysis.interpretability.pipeline import geo_drivers as geo
    from analysis.interpretability.pipeline import intent_stages as stages
    by_url: dict[str, list] = {}
    for page in page_rows:
        for url in page["urls"]:
            by_url.setdefault(url, []).append(float(page["prompt_scale_percentile_0_1"]))
    url_u = {u: float(np.mean(v)) for u, v in by_url.items()}
    prompts = {r["candidate_id"]: (float(r["axis_1_percentile_0_1"]), r["keyword"]) for r in prompt_rows}
    seen, rows, counts = set(), [], {"rows": 0, "duplicates": 0, "unknown_prompt": 0, "empty_or_unmatched_ranking": 0,
                                      "ranked_urls": 0, "ranked_urls_matched": 0}
    for model, row in generation_rows:
        counts["rows"] += 1
        key = (model, row["prompt_id"], row["engine"], row["method"], row["condition"])
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        if row["prompt_id"] not in prompts:
            counts["unknown_prompt"] += 1
            continue
        num = den = 0.0
        for r, url in enumerate(row.get("ranking") or []):
            counts["ranked_urls"] += 1
            if url in url_u:
                counts["ranked_urls_matched"] += 1
                w = 1.0 / np.log2(r + 2.0)
                num += w * url_u[url]
                den += w
        if den == 0:
            counts["empty_or_unmatched_ranking"] += 1
            continue
        x, keyword = prompts[row["prompt_id"]]
        rows.append((model, row["engine"], row["method"], row["condition"], keyword, x, num / den))
    models = sorted({r[0] for r in rows})
    keyword_code = {k: i for i, k in enumerate(sorted({r[4] for r in rows}))}
    col = lambda i: np.asarray([r[i] for r in rows])
    model_a, engine_a, cond_a = col(0), col(1), col(3)
    x, k = col(5).astype(float), np.asarray([keyword_code[r[4]] for r in rows])
    cited = col(6).astype(float)
    draws = stages.keyword_draws(len(keyword_code), bootstrap, seed)
    groups, slopes = {}, {}
    natural = cond_a == "natural"
    for model in models:
        for engine in sorted(set(engine_a)) + ["both engines"]:
            mask = (model_a == model) & ((engine_a == engine) if engine != "both engines" else True)
            name = f"{model} · {engine}"
            groups[name] = {c: {"u": stages.stage_curves(x, k, {"K": cited}, mask & extra)}
                            for c, extra in (("natural", natural), ("all", np.ones(len(x), bool)))}
            m = mask & natural
            point = geo.within_keyword_slope(x[m], cited[m], k[m])
            boot = [geo.within_keyword_slope(x[m], cited[m], k[m], d[k[m]]) for d in draws]
            slopes[name] = {"answers_natural": int(m.sum()), "slope_K_on_x": point,
                            "ci95": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]}
    return {"source": "published generation rows (HF exchange), no traces", "counts": counts,
            "answers_by_model": {m: int((model_a == m).sum()) for m in models},
            "curves": {"bins": 20, "groups": groups}, "slopes_natural": slopes, "bootstrap": bootstrap,
            "note": "descriptive association; cited position = top-weighted mean percentile of ranked URLs"}


# ---------------------------------------------------------------- drawing helpers

def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK})
    return plt


def _save(fig, out_dir: Path, name: str) -> list[str]:
    paths = []
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        paths.append(str(path))
    return paths


def _box(ax, x, y, w, h, text, *, face="#ffffff", edge=MUTED, size=7, weight="normal", color=INK, align="center", lw=0.7):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=face, ec=edge, lw=lw, zorder=2))
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w / 2 if align == "center" else x + 0.008
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, fontweight=weight, color=color, linespacing=1.15, zorder=3)


def _arrow(ax, a, b, *, color=MUTED, lw=0.8, style="-|>", ls="-", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def _gradient(ax, x0, x1, y, h):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("axis", [INFO, "#d9d4c7", ACTION])
    ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(x0, x1, y, y + h), aspect="auto", cmap=cmap, zorder=1)


# ---------------------------------------------------------------- figure 1

def fig1(plt, examples: dict, out_dir: Path) -> list[str]:
    rows = examples["keywords"]
    fig = plt.figure(figsize=(DOUBLE, 0.95 + 0.72 * len(rows)))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-0.04, 1.04)
    top = 1.0
    ax.set_ylim(0, top)
    ax.axis("off")
    axis_y = 0.20
    _gradient(ax, 0.0, 1.0, axis_y, 0.035)
    ax.text(0.0, axis_y + 0.06, "information seeking", ha="left", va="bottom", fontsize=7.5, color=INFO, fontweight="bold")
    ax.text(1.0, axis_y + 0.06, "action ready", ha="right", va="bottom", fontsize=7.5, color=ACTION, fontweight="bold")
    for x, label in ANCHORS:
        ax.plot([x, x], [axis_y - 0.012, axis_y], color=MUTED, lw=0.6)
        ax.text(x, axis_y - 0.02, label, ha="center", va="top", fontsize=6.2, color=MUTED, linespacing=1.05)
        ax.text(x + (0.006 if x == 0 else -0.006 if x == 1 else 0), axis_y + 0.017, f"{x:g}",
                ha="left" if x == 0 else "right" if x == 1 else "center", va="center", fontsize=5.8, color="white")
    ax.text(0.5, 0.012, f"prompt position = percentile of the measured axis among {TESTBED['prompts']:,} prompts "
            "(two frozen LLM2Vec views; anchors are the verbal destinations used in generation)",
            ha="center", va="bottom", fontsize=6.2, color=MUTED, style="italic")
    band_h = (top - axis_y - 0.12) / len(rows)
    for i, row in enumerate(rows):
        y0 = top - 0.03 - (i + 1) * band_h
        ax.text(0.0, y0 + band_h * 0.97, f"keyword: “{row['keyword']}”", ha="left", va="top",
                fontsize=7, fontweight="bold")
        for band, slot in zip(("start", "middle", "end"), (0.0, 0.345, 0.69)):
            pick = row["picks"][band]
            p = float(pick["position"])
            text = textwrap.fill(pick["prompt"], 46)
            colour = INFO if p < 0.34 else (ACTION if p > 0.66 else "#7d7563")
            card_w, card_h = 0.305, band_h * 0.70
            _box(ax, slot, y0 + band_h * 0.10, card_w, card_h, text, size=6.1, edge=colour, align="left", lw=0.8)
            ax.text(slot + card_w - 0.006, y0 + band_h * 0.10 + card_h - 0.012, f"x = {p:.2f}", ha="right", va="top",
                    fontsize=6, color=colour, fontweight="bold", zorder=4)
            ax.plot([slot + card_w / 2, p], [y0 + band_h * 0.10, axis_y + 0.035], color=colour, lw=0.5, alpha=0.7, zorder=0)
            ax.plot([p], [axis_y + 0.0175], "o", ms=3.2, mfc="white", mec=colour, mew=0.9, zorder=3)
    return _save(fig, out_dir, "fig1-prompt-axis")


# ---------------------------------------------------------------- figure 2

def fig2(plt, out_dir: Path) -> list[str]:
    import numpy as np
    t = TESTBED
    fig = plt.figure(figsize=(DOUBLE, 2.55))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rng = np.random.default_rng(7)
    # the corpus: topic clusters, each with DuckDuckGo and SearXNG rows
    ax.text(0.02, 0.965, f"The closed “mini internet”: {t['keywords']:,} keyword topics, frozen once",
            fontsize=8, fontweight="bold", va="top")
    centers = [(0.06 + 0.083 * c, 0.80 - 0.175 * r) for r in range(4) for c in range(6)]
    names = ["robinhood vs etrade"]  # the paper's running example; other clusters are schematic and unnamed
    hub_center = centers[14]
    hits = [0, 2, 9, 14, 20]
    for i, (cx, cy) in enumerate(centers):
        ax.add_patch(plt.Circle((cx, cy), 0.036, fc="#f3f1ec", ec=LINE, lw=0.6))
        for j in range(7):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.006, 0.027)
            colour = "#7a8b99" if j % 2 else "#b7a993"
            size = 2.0
            if i in hits and j == 0:
                colour, size = ACTION, 3.4
            ax.plot(cx + r * np.cos(a), cy + r * np.sin(a) * 1.9, "o", ms=size, color=colour, mew=0)
        if i < len(names):
            ax.text(cx, cy - 0.072, names[i], ha="center", va="top", fontsize=5.0, color=MUTED)
    ax.text(0.27, 0.215, f"… {t['keywords'] - len(centers):,} more topics (schematic)", ha="center", fontsize=6, color=MUTED, style="italic")
    ax.plot([0.03], [0.045], "o", ms=3, color="#b7a993", mew=0)
    ax.text(0.04, 0.045, f"DuckDuckGo row ({t['ddg_rows']:,})", va="center", fontsize=6)
    ax.plot([0.20], [0.045], "o", ms=3, color="#7a8b99", mew=0)
    ax.text(0.21, 0.045, f"SearXNG row ({t['searxng_rows']:,})", va="center", fontsize=6)
    ax.text(0.37, 0.045, f"= {t['rows']:,} rows, {t['pages']:,} distinct pages", va="center", fontsize=6, color=MUTED)
    # glued-title hub row
    hx, hy = hub_center
    ax.plot([hx], [hy], "*", ms=8, color="#8c3b2f", zorder=4)
    for k, (cx, cy) in enumerate(centers):
        if k % 3 == 1 and (cx, cy) != hub_center:
            ax.plot([hx, cx], [hy, cy], color="#8c3b2f", lw=0.35, alpha=0.45, zorder=2)
    ax.plot([0.03], [0.135], "*", ms=6, color="#8c3b2f")
    ax.text(0.045, 0.135, f"glued-title DuckDuckGo rows ({t['glued_rows']}, 2.6%) contain many common words and match many "
            f"queries:\none such row was shown under {t['hub_keywords']} of {t['keywords']:,} topics", fontsize=5.8,
            color="#8c3b2f", va="center")
    # the query and the corpus-wide search
    _box(ax, 0.57, 0.80, 0.19, 0.10, "query written by the AI\n(illustrative: “how to move a\nbrokerage account today”)",
         size=6.2, edge=ACTION)
    for i in hits:
        cx, cy = centers[i]
        _arrow(ax, (0.57, 0.84), (cx + 0.02, cy + 0.01), color=ACTION, lw=0.55, rad=0.12)
    _box(ax, 0.57, 0.36, 0.19, 0.39,
         "frozen search scores the query\nagainst every row of every topic:\n\n1. exact keyword match first\n"
         "2. 4 × words shared with the\n    row’s keyword + words shared\n    with its title and snippet\n"
         "3. stored rank, then a hash\n\nthe prompt’s keyword is not passed", size=5.9, align="left")
    _arrow(ax, (0.665, 0.80), (0.665, 0.75))
    _box(ax, 0.80, 0.62, 0.18, 0.11, f"top {t['results_per_search']} rows per search\n(from any topic)", size=6.4)
    _arrow(ax, (0.76, 0.58), (0.80, 0.665))
    _box(ax, 0.80, 0.43, 0.18, 0.13, "cross-encoder reranker\n(bge-reranker-v2-m3)\nkeeps the best-scored", size=6.2)
    _arrow(ax, (0.89, 0.62), (0.89, 0.56))
    _box(ax, 0.80, 0.24, 0.18, 0.13, f"≤ {t['parallel_top_k']} snippets shown\nto the generator,\nwhich ranks what it used",
         size=6.2, edge=INFO)
    _arrow(ax, (0.89, 0.43), (0.89, 0.37))
    ax.text(0.57, 0.31, "Both generators searched byte-identical\nsnapshots; every query, row, score and\noutput is logged.",
            fontsize=6, color=MUTED, va="top")
    return _save(fig, out_dir, "fig2-mini-internet")


# ---------------------------------------------------------------- figure 3

def fig3(plt, out_dir: Path) -> list[str]:
    t = TESTBED
    fig = plt.figure(figsize=(DOUBLE, 2.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.12, 1)
    ax.axis("off")
    heads = [(0.005, "1  Prompts on the axis"), (0.255, "2  Generation grid"), (0.53, "3  Search in the mini internet"),
             (0.79, "4  Judges and analysis")]
    for x, h in heads:
        ax.text(x, 0.975, h, fontsize=7.6, fontweight="bold", va="top")
    # column 1
    _box(ax, 0.005, 0.80, 0.22, 0.10, f"{t['keywords']:,} keyword topics\n(B2B software, frozen)", size=6.4)
    _box(ax, 0.005, 0.62, 0.22, 0.13, "Qwen3-32B and Gemma-4-31B write\nprompts toward targets on the axis;\n"
         "model review + text checks", size=6.1)
    _box(ax, 0.005, 0.44, 0.22, 0.13, "measure each prompt with two frozen\nLLM2Vec views (Qwen3-8B, Mistral-7B)\n"
         "→ position $x \\in [0, 1]$", size=6.1)
    _box(ax, 0.005, 0.28, 0.22, 0.11, f"{t['prompts']:,} prompts, 18–30 per topic\nrebuild check: difference 0.0",
         size=6.3, edge=INFO, weight="bold")
    _gradient(ax, 0.02, 0.215, 0.215, 0.025)
    ax.text(0.02, 0.19, "information", fontsize=5.6, color=INFO, va="top")
    ax.text(0.215, 0.19, "action", fontsize=5.6, color=ACTION, va="top", ha="right")
    for a, b in ((0.80, 0.75), (0.62, 0.57), (0.44, 0.39)):
        _arrow(ax, (0.115, a), (0.115, b))
    # column 2
    _box(ax, 0.255, 0.74, 0.245, 0.16, "2 generators\nLlama-4-Scout-17B-16E · Qwen3.8-27B\n(temperature 0, pinned)",
         size=6.2)
    _box(ax, 0.255, 0.49, 0.245, 0.21, "2 search methods\nParallel Expansion: 3 queries at once\n→ pooled → top 7\n"
         "Reactive Loop: up to 3 searches,\neach → top 3, may stop early", size=6.0)
    _box(ax, 0.255, 0.33, 0.245, 0.12, "2 engines (frozen snapshots)\nDuckDuckGo · SearXNG", size=6.2)
    _box(ax, 0.255, 0.15, 0.245, 0.14, f"3 evidence conditions\n→ {t['cells_per_prompt']} cells per prompt,\n"
         "≈ 312k answers per generator", size=6.1, weight="bold")
    _arrow(ax, (0.227, 0.335), (0.255, 0.80))
    # column 3
    _box(ax, 0.53, 0.78, 0.235, 0.12, "the AI writes its own search\nqueries  [Q]", size=6.3, edge=ACTION)
    _box(ax, 0.53, 0.61, 0.235, 0.13, f"frozen corpus-wide search returns\n{t['results_per_search']} rows per query  [R]",
         size=6.2)
    _box(ax, 0.53, 0.44, 0.235, 0.13, "cross-encoder scores the candidates [C]\nand keeps a shortlist  [P]", size=6.2)
    _box(ax, 0.53, 0.27, 0.235, 0.13, "generator ranks the sources it used [K]\nand writes the answer  [A]", size=6.2,
         edge=INFO)
    for a, b in ((0.78, 0.74), (0.61, 0.57), (0.44, 0.40)):
        _arrow(ax, (0.6475, a), (0.6475, b))
    _arrow(ax, (0.50, 0.62), (0.53, 0.84))
    # column 4
    _box(ax, 0.79, 0.74, 0.205, 0.16, "LLM judges (ongoing)\nNemotron-3-Nano: relevance order\nGemma-4-31B SI-v4: which\n"
         "sources the answer rests on", size=5.9, face="#f6f3ec")
    _box(ax, 0.79, 0.46, 0.205, 0.24, "same frozen axis applied to\nqueries, pages and answers:\nQ → R → C → P → K → A\n\n"
         "within-keyword slope of each\nstage on x; keyword bootstrap,\nwithin-keyword permutations", size=5.9)
    _box(ax, 0.79, 0.27, 0.205, 0.15, "does intent enter the pool\n(retrieval, reranker) or the\nordering by the generator?",
         size=6.0, edge=ACTION, weight="bold")
    _arrow(ax, (0.765, 0.335), (0.79, 0.55))
    _arrow(ax, (0.765, 0.335), (0.79, 0.80), ls="--")
    _arrow(ax, (0.8925, 0.46), (0.8925, 0.42))
    ax.text(0.53, 0.20, "Q, R, C, P, K, A: the stages measured on the axis (Section 5).", fontsize=6, color=MUTED, va="top")
    return _save(fig, out_dir, "fig3-pipeline")


# ---------------------------------------------------------------- figure 4

def _centers(block):
    return [(a + b) / 2 for a, b in zip(block["bin_lower"], block["bin_upper"])]


def _series(ax, block, term, *, color, ls="-", label=None, band=True, lw=1.1, marker=None):
    import numpy as np
    entry = block["terms"].get(term)
    if not entry:
        return False
    x = np.asarray(_centers(block))
    m = np.asarray([np.nan if v is None else v for v in entry["mean"]], float)
    se = np.asarray([np.nan if v is None else v for v in entry["se_keyword_cluster"]], float)
    ok = np.isfinite(m)
    if not ok.any():
        return False
    ax.plot(x[ok], m[ok], ls=ls, color=color, lw=lw, label=label, marker=marker, ms=2.2)
    if band:
        good = ok & np.isfinite(se)
        ax.fill_between(x[good], (m - 1.96 * se)[good], (m + 1.96 * se)[good], color=color, alpha=0.14, lw=0)
    return True


def fig4(plt, results: dict, out_dir: Path, *, condition="natural") -> list[str]:
    curves = results.get("curves", {}).get("groups")
    if not curves:
        raise ValueError("results.json has no curves; rerun intent_stages_study.py analyze at this commit")
    has_pool = any("P" in g.get(condition, {}).get("u", {}).get("terms", {}) for g in curves.values())
    if has_pool:
        fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.45), gridspec_kw={"wspace": 0.28})
    else:
        fig, a = plt.subplots(1, 1, figsize=(COLUMN, 2.5))
        b = None
    for ax in [a] + ([b] if b is not None else []):
        ax.set_xlim(0, 1)
        ax.set_xlabel("prompt position $x$ (0 information, 1 action)")
        ax.grid(color="#eef1f3", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    drawn = 0
    for name, group in curves.items():
        parts = [s.strip() for s in name.split("·")]
        if len(parts) != 2 or parts[1] == "both engines":
            continue
        model, engine = parts
        drawn += _series(a, group[condition]["u"], "K", color=MODEL.get(model, INK), ls=ENGINE_STYLE.get(engine, "-"),
                         label=f"{MODEL_LABEL.get(model, model)} · {ENGINE_LABEL.get(engine, engine)}")
    a.set_ylabel("position of the cited sources\n(same 0–1 scale, top-weighted)")
    a.set_title("(a) cited sources follow the prompt", loc="left")
    a.legend(frameon=False, loc="upper left")
    for name, group in (curves.items() if b is not None else []):
        parts = [s.strip() for s in name.split("·")]
        if len(parts) == 2 and parts[1] == "both engines":
            colour = MODEL.get(parts[0], INK)
            block = group[condition]["u"]
            _series(b, block, "R", color=colour, ls=":", band=False, lw=0.9)
            _series(b, block, "P", color=colour, ls="--", band=False, lw=0.9)
            _series(b, block, "K", color=colour, ls="-", lw=1.2, label=MODEL_LABEL.get(parts[0], parts[0]))
            _series(b, block, "G", color=colour, ls="-.", band=False, lw=0.9)
    from matplotlib.lines import Line2D
    if b is None:
        return _finish_fig4(plt, fig, [a], drawn, condition, out_dir)
    handles, labels = b.get_legend_handles_labels()
    handles += [Line2D([], [], color=MUTED, ls=":", lw=0.9), Line2D([], [], color=MUTED, ls="--", lw=0.9),
                Line2D([], [], color=MUTED, ls="-", lw=1.2)]
    labels += ["retrieved by the AI’s searches [R]", "shortlist shown [P]", "cited / ranked [K]"]
    if any(l.get_linestyle() == "-." for l in b.get_lines()):
        handles.append(Line2D([], [], color=MUTED, ls="-.", lw=0.9))
        labels.append("answer rests on [G] (judge, development)")
    b.legend(handles, labels, frameon=False, loc="upper left")
    b.set_ylabel("position on the axis")
    b.set_title("(b) where the shift enters: pool vs. ordering", loc="left")
    return _finish_fig4(plt, fig, [a, b], drawn, condition, out_dir)


def _finish_fig4(plt, fig, axes, drawn, condition, out_dir):
    lows = []
    for ax in axes:
        lines = [l for l in ax.get_lines() if len(l.get_ydata()) > 2]
        values = [v for l in lines for v in l.get_ydata()]
        if values:
            lows.append((min(values), max(values)))
    if lows:
        lo, hi = min(l for l, _ in lows), max(h for _, h in lows)
        pad = 0.04
        for ax in axes:
            ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
    if not drawn:
        raise ValueError("no model x engine curves found in results.json")
    fig.text(0.5, -0.10 if len(axes) > 1 else -0.2, f"{condition} condition; 20 bins of x; bands: ±1.96 keyword-clustered standard errors; "
             "Descriptive, not causal.",
             ha="center", fontsize=6.2, color=MUTED)
    return _save(fig, out_dir, "fig4-cited-vs-prompt")


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("axis-examples")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--population-prompts", type=Path, required=True)
    p.add_argument("--keyword", action="append", help="fixed keyword (repeat for two); default: robinhood vs etrade + a seeded random one")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("cited-from-generations", help="figure-4 curves from published generation rows (no traces)")
    p.add_argument("--generations", type=Path, required=True, help="JSON map: generation object path -> model id")
    p.add_argument("--root", type=Path, required=True, help="directory the object paths are relative to")
    p.add_argument("--snippets", type=Path, required=True, help="snippet-embeddings-corpus-v1/snippets.parquet")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--population-prompts", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("render")
    p.add_argument("--axis-examples", type=Path)
    p.add_argument("--results", type=Path, help="results.json of intent_stages_study.py analyze")
    p.add_argument("--condition", choices=["natural", "all"], default="natural")
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "axis-examples":
        from analysis.scripts import page_readiness_ordering as readiness
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        value = axis_examples(readiness.read_jsonl(args.final_axis_map), readiness.read_jsonl(args.population_prompts),
                              keywords=args.keyword, seed=args.seed)
        args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"keywords": [k["keyword"] for k in value["keywords"]], "output": str(args.output)}))
        return 0
    if args.command == "cited-from-generations":
        import pyarrow.parquet as pq
        from analysis.scripts import page_readiness_ordering as readiness
        short = {"meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4", "Qwen/Qwen3.8-27B": "qwen38"}
        mapping = json.loads(args.generations.read_text())

        def generation_rows():
            for path, model in mapping.items():
                if model not in short or not (args.root / path).exists():
                    continue
                with (args.root / path).open(encoding="utf-8") as stream:
                    for line in stream:
                        yield short[model], json.loads(line)["row"]

        axis = {r["candidate_id"]: r for r in readiness.read_jsonl(args.final_axis_map)}
        prompts = [{**axis[r["candidate_id"]], "keyword": r["keyword"]} for r in readiness.read_jsonl(args.population_prompts)
                   if r["candidate_id"] in axis]
        pages = pq.read_table(args.snippets, columns=["urls", "prompt_scale_percentile_0_1"]).to_pylist()
        value = cited_from_generations(generation_rows(), prompts, pages, bootstrap=args.bootstrap)
        args.output.write_text(json.dumps(value, indent=1) + "\n")
        print(json.dumps({"counts": value["counts"], "answers_by_model": value["answers_by_model"],
                          "slopes_natural": value["slopes_natural"]}, indent=1))
        return 0
    plt = _setup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = {"fig2": fig2(plt, args.output_dir), "fig3": fig3(plt, args.output_dir)}
    if args.axis_examples:
        written["fig1"] = fig1(plt, json.loads(args.axis_examples.read_text(encoding="utf-8")), args.output_dir)
    if args.results:
        written["fig4"] = fig4(plt, json.loads(args.results.read_text(encoding="utf-8")), args.output_dir,
                               condition=args.condition)
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
