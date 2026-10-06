#!/usr/bin/env python3
"""Self-contained HTML report of the exploratory funnel review (Mac; addendum A1 of funnel_study.md).

Reads the outputs of funnel_review.py (all keywords), funnel_explore.py (exploration traces),
funnel_study.py report --split exploration, funnel_contrasts.py, funnel_keywords.py and the feature
table's coverage, plus an optional narrative file and the acl_figures.py PNGs, and writes one HTML page
(inline SVG charts drawn to scale, embedded figures, a source line under every section), a results
bundle (<name>-results.json) and CSV tables (<name>-tables/). Every number shown comes from those files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

STRATA = ("llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive")
ENGINE_STRATA = ("llama4 · duckduckgo", "llama4 · searxng", "qwen38 · duckduckgo", "qwen38 · searxng")
LABEL = {"llama4": "Llama-4-Scout", "qwen38": "Qwen3.8", "Parallel": "Parallel Expansion", "Reactive": "Reactive Loop",
         "duckduckgo": "DuckDuckGo", "searxng": "SearXNG"}
SERIES = ("--s1", "--s2", "--s3", "--s4")


def nice(stratum: str) -> str:
    return " · ".join(LABEL.get(p.strip(), p.strip()) for p in stratum.split("·"))


def fmt(v, d=3):
    if v is None:
        return "—"
    return f"{v:+.{d}f}" if isinstance(v, float) and abs(v) < 100 else f"{v:,.{d}f}" if isinstance(v, float) else f"{v:,}"


def ci_cell(entry, d=3, key="slope"):
    if not entry:
        return '<td class="num">—</td>'
    lo, hi = entry.get("ci95", [None, None])
    sig = lo is not None and hi is not None and (lo > 0 or hi < 0)
    ci = f'<span class="ci">[{lo:+.{d}f}, {hi:+.{d}f}]</span>' if lo is not None else ""
    return f'<td class="num{" sig" if sig else ""}">{entry[key]:+.{d}f} {ci}</td>'


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------- SVG charts

def line_chart(series: list[dict], *, ylabel: str, title: str, width=560, height=300, band=True) -> str:
    """series: [{name, color_var, x: [...], y: [...], se: [...]}]; one shared linear scale per axis."""
    pts = [(x, y) for s in series for x, y in zip(s["x"], s["y"]) if y is not None]
    if not pts:
        return ""
    lows = [y - 1.96 * (e or 0) for s in series for y, e in zip(s["y"], s.get("se") or [0] * len(s["y"])) if y is not None]
    highs = [y + 1.96 * (e or 0) for s in series for y, e in zip(s["y"], s.get("se") or [0] * len(s["y"])) if y is not None]
    ymin, ymax = min(lows), max(highs)
    pad = (ymax - ymin) * 0.08 or 0.01
    ymin, ymax = ymin - pad, ymax + pad
    left, right, top, bottom = 62, 16, 34, 46
    w, h = width - left - right, height - top - bottom
    sx = lambda x: left + x * w
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * h
    ticks = _ticks(ymin, ymax, 5)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" class="chart">',
             f'<text x="{left}" y="18" class="ctitle">{esc(title)}</text>']
    for t in ticks:
        parts.append(f'<line x1="{left}" x2="{left + w}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                     f'<text x="{left - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:.3g}</text>')
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{sx(t):.1f}" y="{top + h + 18}" class="tick" text-anchor="middle">{t:g}</text>')
    parts.append(f'<text x="{left + w / 2}" y="{height - 8}" class="axl" text-anchor="middle">prompt position x (0 information seeking → 1 action ready)</text>')
    parts.append(f'<text transform="translate(14 {top + h / 2}) rotate(-90)" class="axl" text-anchor="middle">{esc(ylabel)}</text>')
    for i, s in enumerate(series):
        color = f"var({s.get('color', SERIES[i % 4])})"
        xs, ys, se = s["x"], s["y"], s.get("se") or [None] * len(s["y"])
        ok = [(x, y, e) for x, y, e in zip(xs, ys, se) if y is not None]
        if band and all(e is not None for _, _, e in ok) and len(ok) > 1:
            upper = " ".join(f"{sx(x):.1f},{sy(y + 1.96 * e):.1f}" for x, y, e in ok)
            lower = " ".join(f"{sx(x):.1f},{sy(y - 1.96 * e):.1f}" for x, y, e in reversed(ok))
            parts.append(f'<polygon points="{upper} {lower}" fill="{color}" opacity="0.13"/>')
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in ok)
        dash = ' stroke-dasharray="5 4"' if s.get("dashed") else ""
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    legend_y = top + 4
    for i, s in enumerate(series):
        color = f"var({s.get('color', SERIES[i % 4])})"
        dash = ' stroke-dasharray="5 4"' if s.get("dashed") else ""
        y = legend_y + i * 16
        parts.append(f'<line x1="{left + 10}" x2="{left + 30}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"{dash}/>'
                     f'<text x="{left + 36}" y="{y + 4}" class="leg">{esc(s["name"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _ticks(lo, hi, n):
    import math
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n)) if span > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.ceil(lo / step) * step
    out = []
    t = start
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def curve_series(curves: dict, stratum: str, term: str, name: str, color: str, dashed=False):
    block = curves.get(stratum)
    if not block or term not in block["terms"]:
        return None
    centres = [(a + b) / 2 for a, b in zip(block["bin_lower"], block["bin_upper"])]
    entry = block["terms"][term]
    return {"name": name, "x": centres, "y": entry["mean"], "se": entry["se_keyword_cluster"], "color": color, "dashed": dashed}


def histogram(values, *, title, width=560, height=220, bins=30) -> str:
    import numpy as np
    v = np.asarray([x for x in values if x is not None and x == x], float)
    if not len(v):
        return ""
    counts, edges = np.histogram(v, bins=bins)
    left, right, top, bottom = 50, 16, 30, 40
    w, h = width - left - right, height - top - bottom
    sx = lambda x: left + (x - edges[0]) / (edges[-1] - edges[0]) * w
    sy = lambda c: top + h - c / counts.max() * h
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" class="chart">',
             f'<text x="{left}" y="18" class="ctitle">{esc(title)}</text>']
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        parts.append(f'<rect x="{sx(a) + 0.5:.1f}" y="{sy(c):.1f}" width="{max(sx(b) - sx(a) - 1, 0.5):.1f}" height="{top + h - sy(c):.1f}" fill="var(--s1)" opacity="0.8"/>')
    if edges[0] < 0 < edges[-1]:
        parts.append(f'<line x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{top}" y2="{top + h}" class="zero"/>')
    for t in _ticks(edges[0], edges[-1], 6):
        parts.append(f'<text x="{sx(t):.1f}" y="{top + h + 16}" class="tick" text-anchor="middle">{t:.3g}</text>')
    parts.append(f'<text x="{left + w / 2}" y="{height - 6}" class="axl" text-anchor="middle">shrunken slope of cited intent on x (per keyword)</text>')
    parts.append(f'<text x="{left - 8}" y="{top + 8}" class="tick" text-anchor="end">{int(counts.max())}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- page

CSS = """
/* Layout: one reading column (~74ch) with full-width tables that scroll inside their own frame. */
:root { --bg:#f5f6f4; --surface:#ffffff; --fg:#1d2524; --muted:#5d6967; --line:#d9dedb; --accent:#0f6b63;
  --info:#2f6f9f; --action:#c0632b; --s1:#2f6f9f; --s2:#c0632b; --s3:#4f8a3a; --s4:#8a4f9e; --sig:#0f6b63;
  --font-display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif; --font-body:"IBM Plex Sans",system-ui,sans-serif;
  --font-data:"IBM Plex Mono",ui-monospace,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121716; --surface:#1a2120; --fg:#e3e9e7; --muted:#9aa8a5;
  --line:#2c3634; --accent:#5cc3b5; --info:#7fb4dc; --action:#e39a6c; --s1:#7fb4dc; --s2:#e39a6c; --s3:#94c47f; --s4:#c49ad6; --sig:#5cc3b5; color-scheme:dark } }
:root[data-theme="dark"] { --bg:#121716; --surface:#1a2120; --fg:#e3e9e7; --muted:#9aa8a5; --line:#2c3634; --accent:#5cc3b5;
  --info:#7fb4dc; --action:#e39a6c; --s1:#7fb4dc; --s2:#e39a6c; --s3:#94c47f; --s4:#c49ad6; --sig:#5cc3b5; color-scheme:dark }
body { background:var(--bg); color:var(--fg); font:15px/1.6 var(--font-body); }
main { max-width:1040px; margin:0 auto; padding-inline:clamp(16px,4vw,40px); padding-block:28px 64px; display:grid; gap:28px; }
header { display:grid; gap:10px; border-bottom:1px solid var(--line); padding-block:8px 20px; }
h1 { font:600 clamp(28px,4.4vw,40px)/1.1 var(--font-display); letter-spacing:-0.01em; margin:0; text-wrap:balance; }
h2 { font:600 22px/1.25 var(--font-display); margin:0; text-wrap:balance; }
h3 { font:600 16px/1.3 var(--font-body); margin:0; }
p, li { max-width:74ch; margin:0; } section { display:grid; gap:14px; min-width:0; }
main > *, section > *, .grid2 > * { min-width:0; }
.lede { color:var(--muted); font-size:16px; max-width:74ch; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { font:500 12px/1 var(--font-data); padding:6px 9px; border:1px solid var(--line); border-radius:3px; background:var(--surface); color:var(--muted); }
.chip.explore { border-color:var(--action); color:var(--action); }
.axis { height:8px; border-radius:2px; background:linear-gradient(90deg,var(--info),var(--line) 50%,var(--action)); }
.axislabels { display:flex; justify-content:space-between; font:500 12px var(--font-data); color:var(--muted); }
.frame { overflow-x:auto; border:1px solid var(--line); border-radius:4px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { font:600 12px var(--font-body); color:var(--muted); background:var(--bg); position:sticky; top:0; }
td.num { font-family:var(--font-data); font-variant-numeric:tabular-nums; white-space:nowrap; }
td.sig { color:var(--sig); font-weight:600; } .ci { color:var(--muted); font-weight:400; font-size:12px; }
tr:last-child td { border-bottom:0; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr)); gap:16px; }
.chart { width:100%; height:auto; background:var(--surface); border:1px solid var(--line); border-radius:4px; }
.chart .grid { stroke:var(--line); stroke-width:1; } .chart .zero { stroke:var(--fg); stroke-width:1; stroke-dasharray:3 3; }
.chart .tick, .chart .leg { fill:var(--muted); font:11px var(--font-data); } .chart .axl { fill:var(--muted); font:11px var(--font-body); }
.chart .ctitle { fill:var(--fg); font:600 13px var(--font-body); }
.note { font-size:13px; color:var(--muted); max-width:80ch; }
.callout { border-left:3px solid var(--accent); padding:10px 14px; background:var(--surface); border-radius:0 4px 4px 0; display:grid; gap:6px; }
.findings { display:grid; gap:10px; padding-left:20px; margin:0; }
code { font:12px var(--font-data); background:var(--surface); border:1px solid var(--line); padding:1px 4px; border-radius:3px; overflow-wrap:anywhere; }
nav.toc { display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; } nav.toc a { color:var(--accent); }
a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
td.blockrow { font:600 12px var(--font-body); color:var(--accent); background:var(--bg); letter-spacing:0.02em; }
figure.fig { margin:0; display:grid; gap:6px; } figure.fig img { width:100%; height:auto; background:#ffffff; border:1px solid var(--line); border-radius:4px; }
figcaption { font-size:13px; color:var(--muted); max-width:80ch; } figcaption .src { font:11px var(--font-data); }
"""


def table(head, rows) -> str:
    return ('<div class="frame"><table><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(r) + "</tr>" for r in rows) + "</tbody></table></div>")


def slope_table(slopes: dict, metrics: list[tuple[str, str]], strata=STRATA, d=3) -> str:
    rows = []
    for metric, label in metrics:
        cells = [f"<td>{esc(label)}</td>"]
        for s in strata:
            cells.append(ci_cell((slopes.get(s) or {}).get(metric), d))
        rows.append(cells)
    return table(["measure (within-keyword slope on x, 95% keyword-bootstrap CI)"] + [esc(nice(s)) for s in strata], rows)


def means_table(slopes: dict, metrics, strata=STRATA, d=3) -> str:
    rows = []
    for metric, label in metrics:
        cells = [f"<td>{esc(label)}</td>"]
        for s in strata:
            e = (slopes.get(s) or {}).get(metric)
            cells.append(f'<td class="num">{e["low_x_mean"]:.{d}f} → {e["high_x_mean"]:.{d}f}</td>' if e and e.get("low_x_mean") is not None
                         and e.get("high_x_mean") is not None else '<td class="num">—</td>')
        rows.append(cells)
    return table(["measure (mean for x &lt; 0.2 → mean for x ≥ 0.8)"] + [esc(nice(s)) for s in strata], rows)


BLOCKS = [("A1", "A1 · intent (visible text)"), ("A2", "A2 · topic (visible text)"), ("A3", "A3 · snippet surface (visible text)"),
          ("A4", "A4 · URL string (visible to the generator)"), ("B", "B · search signals (retriever only)"),
          ("C1", "C1 · off-page SEO (no component sees it directly)"), ("C2", "C2 · page body (unseen by every component: negative controls)")]
STAGES = (("R|U", "retrieval R|U"), ("R0|U", "prompt-text search R0|U"), ("P|C", "shortlist P|C"), ("K|P", "ranking K|P"))
DECOMPOSITION = (("retrieval|U", "retrieval R|U"), ("reranker_candidate|R", "scored C|R"), ("shortlist|C", "shortlist P|C"),
                 ("ranking|P", "ranking K|P"))
FIGURES = {"fig4": ("fig4-cited-vs-prompt-by-method.png", "Prompt position against the position of the cited sources, by model and search "
                    "method (all keywords, natural condition, published generation rows)."),
           "fig5": ("fig5-funnel-odds-ratios.png", "Odds ratio per SD of selected features at each stage (exploration keywords, "
                    "main specification, 95% keyword-bootstrap intervals)."),
           "fig6": ("fig6-decomposition.png", "Exact cumulative log risk ratio from the keyword's own rows (U) to the ranked list (K)."),
           "fig7": ("fig7-offtopic.png", "Share of items from another keyword's rows at each stage."),
           "fig8": ("fig8-stage-intent.png", "Page intent of the items at each stage, against prompt position.")}


def figure(args, key: str) -> str:
    """One embedded PNG figure (data URI) with its caption; empty when the file is absent."""
    import base64
    if not args.figures:
        return ""
    name, caption = FIGURES[key]
    path = Path(args.figures) / name
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure class="fig"><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
            f"<figcaption>{esc(caption)} <span class='src'>{esc(name)}</span></figcaption></figure>")


def source(*paths) -> str:
    return '<p class="note">Source: ' + ", ".join(f"<code>{esc(p)}</code>" for p in paths if p) + "</p>"


def or_cell(e) -> str:
    import math
    if not e:
        return '<td class="num">—</td>'
    lo, hi = e.get("ci95") or [None, None]
    sig = lo is not None and hi is not None and (lo > 0 or hi < 0)
    p = f' p={e["permutation_p"]:.3f}' if e.get("permutation_p") is not None else ""
    ci = f' <span class="ci">[{math.exp(lo):.2f}, {math.exp(hi):.2f}]{p}</span>' if lo is not None and hi is not None else ""
    return f'<td class="num{" sig" if sig else ""}">{e["odds_ratio_per_sd"]:.2f}{ci}</td>'


def funnel_strata(models: dict) -> list:
    return [s for s in STRATA if s in models]


def write_tables(directory: Path, review, explore, funnel, contrasts_dir, keywords_dir) -> list:
    """CSV tables next to the report: every number of the page in long format."""
    import csv
    import shutil
    directory.mkdir(parents=True, exist_ok=False)
    written = []

    def dump(name, header, rows):
        with open(directory / name, "w", newline="", encoding="utf-8") as stream:
            w = csv.writer(stream)
            w.writerow(header)
            w.writerows(rows)
        written.append(name)

    def slope_rows(slopes):
        for stratum, metrics in slopes.items():
            for metric, e in metrics.items():
                yield [stratum, metric, e.get("n"), e.get("mean"), e.get("slope"), *(e.get("ci95") or [None, None]),
                       e.get("low_x_mean"), e.get("high_x_mean")]
    head = ["stratum", "measure", "n", "mean", "slope", "ci95_lo", "ci95_hi", "mean_x_below_0.2", "mean_x_at_least_0.8"]
    dump("review_slopes.csv", head, slope_rows(review["slopes"]))
    if explore:
        dump("stage_slopes.csv", head, slope_rows(explore["slopes"]))
    if funnel:
        rows, blocks = [], []
        for spec, strata in funnel["models"].items():
            for stratum, stages in strata.items():
                for stage, res in stages.items():
                    for feature, e in (res.get("features") or {}).items():
                        rows.append([spec, stratum, stage, feature, e["block"], e["beta_per_sd"], e["odds_ratio_per_sd"],
                                     *(e.get("ci95") or [None, None]), e.get("permutation_p")])
                    for block, b in (res.get("blocks") or {}).items():
                        blocks.append([spec, stratum, stage, block, b.get("fit_share")])
        dump("stage_models.csv", ["specification", "stratum", "stage", "feature", "block", "beta_per_sd", "odds_ratio_per_sd",
                                  "beta_ci95_lo", "beta_ci95_hi", "permutation_p"], rows)
        dump("block_fit_shares.csv", ["specification", "stratum", "stage", "block", "fit_share"], blocks)
        dec = []
        for stratum, features in funnel.get("decomposition", {}).items():
            for feature, d in features.items():
                for key, _ in DECOMPOSITION:
                    dec.append([stratum, feature, d.get("answers"), key, d["log_rr"].get(key), *((d.get("ci95") or {}).get(key) or [None, None]),
                                (d.get("share") or {}).get(key), d.get("total_log_rr_K_given_U"), *(d.get("ci95_total") or [None, None])])
        dump("decomposition.csv", ["stratum", "feature", "answers", "step", "log_rr", "ci95_lo", "ci95_hi", "share_of_total",
                                   "total_log_rr_K_given_U", "total_ci95_lo", "total_ci95_hi"], dec)
    if contrasts_dir and (Path(contrasts_dir) / "contrasts.csv").exists():
        shutil.copy(Path(contrasts_dir) / "contrasts.csv", directory / "contrasts.csv")
        written.append("contrasts.csv")
    if keywords_dir and (Path(keywords_dir) / "keywords.csv").exists():
        shutil.copy(Path(keywords_dir) / "keywords.csv", directory / "keywords.csv")
        written.append("keywords.csv")
    return written


def build(args) -> int:
    review = json.loads((Path(args.review) / "review.json").read_text())
    explore = json.loads((Path(args.explore) / "explore.json").read_text()) if args.explore else None
    funnel = json.loads((Path(args.funnel) / "results.json").read_text()) if args.funnel else None
    contrasts = json.loads((Path(args.contrasts) / "contrasts.json").read_text()) if args.contrasts else None
    keywords_summary = json.loads((Path(args.keywords) / "keywords-summary.json").read_text()) if args.keywords else None
    coverage = json.loads((Path(args.features) / "coverage.json").read_text())
    narrative = Path(args.narrative).read_text() if args.narrative and Path(args.narrative).exists() else ""
    rs, rc = review["slopes"], review["curves"]
    review_file = f"{Path(args.review).name}/review.json"
    sections = []

    # data and coverage
    cov_keys = [("html_usable", "HTML usable"), ("dfs_domain", "DataForSEO domain"), ("open_pagerank", "Open PageRank"),
                ("llms_txt", "llms.txt checked"), ("google_top20_url", "Google top-20 URL"), ("google_top20_domain", "Google top-20 domain")]
    cov_rows = [[f"<td>{esc(LABEL.get(e, e))}</td>", f'<td class="num">{c.get("rows", 0):,}</td>']
                + [f'<td class="num">{c[k]:.1%}</td>' if isinstance(c.get(k), float) else '<td class="num">—</td>' for k, _ in cov_keys]
                for e, c in coverage["by_engine"].items()]
    cov_head = ["engine", "usable rows"] + [label for _, label in cov_keys]
    val = coverage["validation_against_experiment1"]
    funnel_counts = (funnel or {}).get("assembled", {}).get("counts", {})
    data_html = (f"<p>Published generation rows: {review['answers']:,} answers ({', '.join(f'{LABEL.get(k, k)} {v:,}' for k, v in review['answers_by_model'].items())}); "
                 f"{review['natural_answers']:,} in the natural condition, 1,011 keywords. "
                 + (f"Exploration traces: {explore['answers']:,} answers ({explore['natural']:,} natural) from {explore['keywords']} exploration keywords, "
                    "Qwen3.8 only (the Llama traces could not be downloaded on this connection). " if explore else "")
                 + (f"Funnel tables: {funnel_counts.get('u_items', 0):,} keyword-row items, {funnel_counts.get('candidates', 0):,} scored candidates, "
                    f"{funnel_counts.get('presented', 0):,} presented items. " if funnel_counts else "")
                 + f"Re-extracted page features agree with experiment 1 on {val['body_word_count']['n']:,} shared URLs "
                 f"(Spearman {min(v['spearman'] for v in val.values()):.3f}–{max(v['spearman'] for v in val.values()):.3f}).</p>"
                 + "<h3>Feature coverage of the snapshot rows</h3>" + table(cov_head, cov_rows)
                 + source(review_file, explore and f"{Path(args.explore).name}/explore.json", f"{Path(args.features).name}/coverage.json",
                          funnel and f"{Path(args.funnel).name}/results.json"))
    sections.append(("data", "Data and coverage", data_html))

    # behaviour
    beh = [("ranking_len", "sources ranked"), ("search_count", "search actions"), ("shown", "snippets shown"),
           ("answer_chars", "stored answer length (characters)"), ("answer_steps", "answer contains numbered steps"),
           ("answer_imperatives", "imperative verbs in answer"), ("answer_currency", "answer mentions a price"),
           ("answer_urls", "answer mentions a URL")]
    charts = [line_chart([s for s in (curve_series(rc, st, "ranking_len", nice(st), SERIES[i]) for i, st in enumerate(STRATA)) if s],
                         ylabel="sources ranked", title="Sources ranked per answer"),
              line_chart([s for s in (curve_series(rc, st, "search_count", nice(st), SERIES[i]) for i, st in enumerate(STRATA) if "Reactive" in st) if s],
                         ylabel="search actions", title="Search actions (Reactive Loop)")]
    sections.append(("behaviour", "Generator behaviour along the axis",
                     "<p>Natural condition, all keywords, published generation rows. Slopes are the change from the most "
                     "informational to the most action-ready prompt of the same keyword.</p>" + slope_table(rs, beh, d=3)
                     + means_table(rs, beh[:4], d=2) + '<div class="grid2">' + "".join(charts) + "</div>" + source(review_file)))

    # cited sources
    cited = [("cited_u", "intent of cited sources (0–1 page scale)"), ("cited_gap", "|cited intent − x|"),
             ("null_gap", "|keyword's own rows − x| (reference)"), ("alignment_gain", "alignment gain over the keyword's rows"),
             ("top1_gap", "|top-1 intent − x|"), ("cited_on_topic", "cited sources from the prompt's keyword"),
             ("cited_glued", "cited glued-title rows"), ("r0_u", "intent of a literal search on the prompt text (R0)"),
             ("r0_overlap", "cited sources found by the literal prompt search")]
    sections.append(("cited", "Cited sources follow the prompt",
                     figure(args, "fig4") + slope_table(rs, cited) + means_table(rs, cited[:5] + cited[7:8]) + source(review_file)))

    # funnel stages
    stage_html = ""
    if explore:
        es = explore["slopes"]
        stage = [("R0_u", "literal prompt search (R0)"), ("R_u", "retrieved by the AI's searches (R)"), ("C_u", "scored by the reranker (C)"),
                 ("P_u", "shown to the generator (P)"), ("K_u", "ranked (K, top-weighted)"), ("R0_recovered", "share of R0 the AI's searches recover"),
                 ("R_from_R0", "share of R that R0 would also return"), ("R_offtopic", "off-topic share in R"), ("C_offtopic", "off-topic share in C"),
                 ("P_offtopic", "off-topic share in P"), ("K_offtopic", "off-topic share in K"), ("R_glued", "glued-title share in R"),
                 ("P_glued", "glued-title share in P"), ("K_glued", "glued-title share in K"), ("R_ad", "ad-redirect share in R")]
        queries = [("q_count", "queries written"), ("q_words", "words per query"), ("q_prompt_reuse", "query words taken from the prompt"),
                   ("q_keyword_inclusion", "queries containing the keyword"), ("q_action_share", "action vocabulary share"),
                   ("q_information_share", "information vocabulary share"), ("q_distinct_share", "distinct queries")]
        qstrata = [s for s in STRATA if s in es]
        stage_html += (f"<p>{explore['natural']:,} natural answers from {explore['keywords']} exploration keywords, traced to the exact "
                       "snapshot rows each answer retrieved (R), had scored (C), saw (P) and ranked (K).</p>"
                       + figure(args, "fig8") + slope_table(es, stage, strata=qstrata) + means_table(es, stage[:5], strata=qstrata)
                       + figure(args, "fig7") + "<h3>The AI's own search queries</h3>" + slope_table(es, queries, strata=qstrata)
                       + means_table(es, queries, strata=qstrata) + source(f"{Path(args.explore).name}/explore.json"))
    if funnel:
        main = funnel["models"].get("main", {})
        intent = [("intent_alignment", "intent alignment −|u − x|"), ("intent_x_prompt", "page intent × (x − ½)"),
                  ("page_intent_z", "page intent (z)"), ("topic_similarity", "topic similarity")]
        for st in funnel_strata(main):
            rows = [[f"<td>{esc(label)}</td>"] + [or_cell((main[st].get(stage, {}).get("features") or {}).get(f)) for stage, _ in STAGES]
                    for f, label in intent]
            stage_html += f"<h3>Intent at each stage · {esc(nice(st))}</h3>" + table(["feature (odds ratio per SD [95% CI], permutation p)"]
                                                                                     + [lab for _, lab in STAGES], rows)
        stage_html += source(f"{Path(args.funnel).name}/results.json")
    if stage_html:
        sections.append(("stages", "Funnel stages: where intent enters (exploration keywords)", stage_html))

    # SEO and page features by visibility block
    seo = [("cited_dfs_organic_count", "C1 · domain organic keywords (log)"), ("cited_dfs_organic_pos_1", "C1 · domain top-1 keywords (log)"),
           ("cited_opr", "C1 · Open PageRank"), ("cited_has_llms_txt", "C1 · llms.txt"), ("cited_brand_list", "C1 · SaaS brand list"),
           ("cited_earned_list", "C1 · review/press list"), ("cited_google_url", "C1 · URL in Google top 20 for the keyword"),
           ("cited_google_domain", "C1 · domain in Google top 20"), ("cited_url_user_content", "A4 · user-content platform"),
           ("cited_url_wikipedia", "A4 · Wikipedia"), ("cited_url_ad_redirect", "A4 · ad redirect"),
           ("cited_body_structured_data", "C2 · JSON-LD structured data"), ("cited_body_question_headings", "C2 · question headings"),
           ("cited_body_freshness", "C2 · freshness (0–4)"), ("cited_body_word_count", "C2 · page words (log)"),
           ("cited_html_usable", "C2 · page HTML available")]
    seo_html = ("<p>(a) How the cited sources' off-page (C1), URL (A4) and page-body (C2) properties change with prompt position: "
                "descriptions of what gets cited, all keywords.</p>" + slope_table(rs, seo) + means_table(rs, seo) + source(review_file))
    if funnel:
        main = funnel["models"].get("main", {})
        null = funnel.get("negative_control_null", {})
        seo_html += ("<p>(b) Stage models: each feature's odds ratio per SD at each stage, all blocks together (main specification). "
                     "Blocks follow what each component can see. C2 features are unseen by every component, so their associations measure "
                     "proxy confounding; their 95th percentile of |β| (the empirical null) is the bar a C1 or A4 association must clear.</p>"
                     + figure(args, "fig5"))
        for st in funnel_strata(main):
            rows = []
            feats = {}
            for stage, _ in STAGES:
                for f, e in (main[st].get(stage, {}).get("features") or {}).items():
                    feats.setdefault(f, e["block"])
            for block, label in BLOCKS:
                members = sorted(f for f, b in feats.items() if b == block)
                if not members:
                    continue
                rows.append([f'<td colspan="5" class="blockrow">{esc(label)}</td>'])
                for f in members:
                    rows.append([f"<td>{esc(f)}</td>"] + [or_cell((main[st].get(stage, {}).get("features") or {}).get(f)) for stage, _ in STAGES])
            share = [[f"<td>{esc(label)}</td>"] + [f'<td class="num">{((main[st].get(stage, {}).get("blocks") or {}).get(block) or {}).get("fit_share") or 0:.1%}</td>'
                                                   for stage, _ in STAGES] for block, label in BLOCKS]
            nullrow = [["<td>C2 empirical null: 95th percentile of |β|</td>"] + [f'<td class="num">{(null.get(st, {}).get(stage) or {}).get("quantile", float("nan")):.3f}</td>'
                                                                               for stage, _ in STAGES]]
            seo_html += (f"<h3>{esc(nice(st))}</h3>" + table(["feature (odds ratio per SD [95% CI], permutation p)"] + [lab for _, lab in STAGES], rows)
                         + table(["block: share of the fit lost when the block is dropped"] + [lab for _, lab in STAGES], share + nullrow))
        specs = [s for s in ("main", "visible", "complete") if s in funnel["models"]]
        robust = []
        for st in funnel_strata(main):
            for f in ("intent_alignment", "topic_similarity", "dfs_organic_count"):
                for stage, label in STAGES:
                    cells = [f"<td>{esc(nice(st))}</td><td>{esc(f)}</td><td>{esc(label)}</td>"]
                    cells += [or_cell((funnel["models"][sp].get(st, {}).get(stage, {}).get("features") or {}).get(f)) for sp in specs]
                    robust.append(cells)
        seo_html += ("<h3>Robustness across specifications</h3>"
                     + table(["stratum", "feature", "stage"] + [f"{sp} specification" for sp in specs], robust)
                     + source(f"{Path(args.funnel).name}/results.json"))
    sections.append(("seo", "SEO and page features by visibility block", seo_html))

    # exact decomposition
    if funnel:
        dec = funnel.get("decomposition", {})
        dec_rows = []
        for st in STRATA:
            for f, d in (dec.get(st) or {}).items():
                if not d.get("answers"):
                    continue
                cells = [f"<td>{esc(nice(st))}</td><td>{esc(f)}</td>"]
                for key, _ in DECOMPOSITION:
                    v = d["log_rr"].get(key)
                    ci = (d.get("ci95") or {}).get(key) or [None, None]
                    sig = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
                    cells.append(f'<td class="num{" sig" if sig else ""}">{v:+.3f} <span class="ci">[{ci[0]:+.3f}, {ci[1]:+.3f}]</span></td>'
                                 if v is not None and ci[0] is not None else '<td class="num">—</td>')
                lo, hi = d.get("ci95_total") or [None, None]
                cells.append(f'<td class="num">{d["total_log_rr_K_given_U"]:+.3f} <span class="ci">[{lo:+.3f}, {hi:+.3f}]</span></td>'
                             if lo is not None else f'<td class="num">{d["total_log_rr_K_given_U"]:+.3f}</td>')
                dec_rows.append(cells)
        sections.append(("decomposition", "Exact decomposition: admission into the pool and ranking within it",
                         "<p>For each feature, items in the top quartile are compared with items in the bottom quartile among the keyword's own "
                         "rows U of each answer (binary features: 1 against 0), with equal weight per answer. The log risk ratio of being ranked "
                         "splits exactly into retrieval, scoring, shortlist and ranking terms; admission into the pool is the first three, "
                         "ranking within it the last. 95% keyword-bootstrap intervals.</p>" + figure(args, "fig6")
                         + table(["stratum", "feature"] + [lab for _, lab in DECOMPOSITION] + ["total log RR(K|U)"], dec_rows)
                         + source(f"{Path(args.funnel).name}/results.json")))

    # contrasts
    con_html = ""
    if contrasts:
        measures = [("cited_u", "intent of cited sources"), ("alignment_gain", "alignment gain"), ("cited_on_topic", "on-topic share"),
                    ("cited_glued", "glued-title share"), ("r0_overlap", "overlap with the literal prompt search"), ("ranking_len", "sources ranked")]
        names = list(contrasts["contrasts"])
        rows = []
        for m, label in measures:
            cells = [f"<td>{esc(label)}</td>"]
            for n in names:
                e = contrasts["contrasts"][n].get(m)
                cells.append(ci_cell(e, 3, key="difference") if e else '<td class="num">—</td>')
            rows.append(cells)
        con_html += (f"<p>Differences between two strata's within-keyword slopes on x, natural condition, all keywords "
                     f"({contrasts['answers']:,} answers); 95% intervals from {contrasts['bootstrap']} keyword-bootstrap draws shared by both strata.</p>"
                     + table(["slope difference"] + [esc(n) for n in names], rows) + source(f"{Path(args.contrasts).name}/contrasts.json"))
    if funnel and funnel.get("contrasts"):
        crow = []
        for name, by in funnel["contrasts"].items():
            for st, e in by.items():
                lo, hi = e.get("ci95") or [None, None]
                sig = lo is not None and (lo > 0 or hi < 0)
                crow.append([f"<td>{esc(name)}</td><td>{esc(nice(st))}</td>",
                             f'<td class="num{" sig" if sig else ""}">{e["estimate"]:+.3f} <span class="ci">[{lo:+.3f}, {hi:+.3f}]</span></td>'])
        con_html += ("<h3>Stage contrasts (shared draws, β per SD)</h3>" + table(["contrast", "stratum", "difference [95% CI]"], crow)
                     + source(f"{Path(args.funnel).name}/results.json"))
    con_html += ("<h3>By search engine</h3>" + slope_table(rs, [("cited_u", "intent of cited sources"), ("alignment_gain", "alignment gain"),
                                                                 ("cited_on_topic", "on-topic share"), ("cited_glued", "glued-title share"),
                                                                 ("ranking_len", "sources ranked")], strata=ENGINE_STRATA) + source(review_file))
    sections.append(("contrasts", "Model × method and engine contrasts", con_html))

    # keywords
    if keywords_summary:
        ks = keywords_summary
        import csv
        rows_csv = list(csv.DictReader(open(Path(args.keywords) / "keywords.csv")))
        hist = histogram([float(r["slope_shrunk"]) for r in rows_csv if r["slope_shrunk"]], title="Keyword slopes after shrinkage")
        def kw_rows(items):
            return [[f"<td>{esc(r['keyword'])}</td>", f'<td class="num">{r["slope_shrunk"]:+.3f}</td>',
                     f'<td class="num">[{r["slope_shrunk_lo95"]:+.3f}, {r["slope_shrunk_hi95"]:+.3f}]</td>', f"<td>{esc(r.get('kw_main_intent') or '—')}</td>"]
                    for r in items]
        cls = [[f"<td>{esc(k)}</td>", ci_cell(v), f'<td class="num">{v["keywords"] if v else "—"}</td>'] for k, v in ks["by_intent_class"].items()]
        ter = [[f"<td>difficulty tercile {int(k) + 1} of 3</td>", ci_cell(v), f'<td class="num">{v["keywords"] if v else "—"}</td>'] for k, v in ks["by_difficulty_tercile"].items()]
        corr = [[f"<td>{esc(k)}</td>", f'<td class="num">{v:+.3f}</td>'] for k, v in ks["correlations_with_shrunk_slope"].items()]
        lo, hi = ks.get("pooled_mean_slope_ci95") or [None, None]
        pooled = f"{ks['pooled_mean_slope']:+.3f}" + (f" (95% CI {lo:+.3f} to {hi:+.3f})" if lo is not None else "")
        null = (f"; the shuffles' Q ranged up to {ks['null_q_max']:.0f} (mean {ks['null_q_mean']:.0f})" if ks.get("null_q_max") else "")
        gem = ks.get("gemma") or {}
        sections.append(("keywords", "By keyword",
                         f"<p>{ks['keywords_with_slope']:,} keywords with a slope of the cited sources' intent on x. Random-effects mean slope "
                         f"{pooled}; between-keyword variance τ² = {ks['between_keyword_variance_tau2']:.4f}, I² = {ks['i2']:.2f}. "
                         f"Heterogeneity Q = {ks['heterogeneity_q']:.0f} on {ks['heterogeneity_df']} df against {ks['permutations']} within-keyword "
                         f"shuffles of x{null}: permutation p = {ks['heterogeneity_permutation_p']:.3f}, the smallest value this test can give. "
                         f"After shrinkage, the 95% interval lies above 0 for {ks['share_keywords_shrunk_interval_above_0']:.0%} of keywords and "
                         f"below 0 for {ks['share_keywords_shrunk_interval_below_0']:.1%}.</p>"
                         + (f"<p class='note'>The fixed-effect mean ({ks['fixed_effect_mean_slope']:+.3f}) is lower because precise keywords have "
                            "smaller slopes (mean raw slope by SE quintile: " + ", ".join(f"{v:+.3f}" for v in ks.get("raw_slope_mean_by_se_quintile", []))
                            + "); the random-effects mean is the shrinkage centre.</p>" if ks.get("fixed_effect_mean_slope") is not None else "")
                         + hist + '<div class="grid2"><div>' + "<h3>By DataForSEO intent class and difficulty</h3>"
                         + table(["keyword group", "pooled within-keyword slope [95% CI]", "keywords"], cls + ter)
                         + "</div><div><h3>Spearman correlation with the shrunken slope</h3>" + table(["keyword descriptor", "ρ"], corr) + "</div></div>"
                         + '<div class="grid2"><div><h3>Steepest 20</h3>' + table(["keyword", "slope", "95% interval", "intent"], kw_rows(ks["top20"]))
                         + "</div><div><h3>Flattest or negative 20</h3>" + table(["keyword", "slope", "95% interval", "intent"], kw_rows(ks["bottom20"])) + "</div></div>"
                         + ("<p class='note'>Gemma mean support: " + (f"{gem.get('natural_answers_with_grades', 0):,} natural answers joined." if gem.get("supplied")
                            else "empty until the HoreKa gemma-extract is returned.") + "</p>")
                         + '<p class="note">Full table (one row per keyword: answers per model, raw and shrunken slope, DataForSEO intent, difficulty, '
                         "volume and CPC, off-topic, glued, Google top-20 shares, domain organic count, three example prompts): "
                         "<code>tables/keywords.csv</code>.</p>" + source(f"{Path(args.keywords).name}/keywords-summary.json", f"{Path(args.keywords).name}/keywords.csv")))

    gemma = ("<p>Gemma SI-v4 grades exist only on HoreKa. They are development judgments (the judge failed semantic review, "
             "<code>scientific_result: false</code>) and enter as stage G through <code>intent_stages_study.py gemma-extract</code> and "
             "<code>analyze --gemma</code>. No Gemma number appears in this report; the by-keyword Gemma column stays empty until that extract is "
             "returned and <code>funnel_keywords.py --gemma</code> is rerun.</p>")
    limits = ("<ul class='findings'><li>Closed testbed: a frozen word-overlap search over 23,893 snippet rows, not the live web.</li>"
              "<li>Everything here is exploratory (addendum A1). The stage models use the 30% exploration keywords; the confirmatory family "
              "(P1–P4) runs on the held-out 70% on HoreKa.</li>"
              "<li>The stage models cover Qwen3.8 only: the Llama trace bundles (51 GB) could not be downloaded on this connection. "
              "Llama's cited-source results come from the published generation rows.</li>"
              "<li>Glued-title DuckDuckGo rows act as hubs; engines are reported separately.</li>"
              "<li>Query and answer intent need GPU embeddings and are not measured here; query vocabulary is a lexical proxy.</li>"
              "<li>The Gemma judge is not validated; no claim rests on it. All results are associations; x is a measured property of the "
              "prompt text, not a randomised treatment.</li></ul>")
    sections.append(("gemma", "Gemma support (development)", gemma))
    sections.append(("limits", "Limitations", limits))

    out = Path(args.output)
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    tables = write_tables(out.with_name(out.stem + "-tables"), review, explore, funnel, args.contrasts, args.keywords)
    files = "".join(f"<li><code>tables/{esc(t)}</code></li>" for t in tables)
    sections.append(("files", "Files", "<p>Every number on this page is in these tables and in <code>results.json</code>, "
                     f"published next to the page.</p><ul class='findings'>{files}<li><code>results.json</code></li></ul>"))
    toc = "".join(f'<a href="#{sid}">{esc(t)}</a>' for sid, t, _ in sections)
    body = "".join(f'<section id="{sid}"><h2>{esc(t)}</h2>{content}</section>' for sid, t, content in sections)
    page = f"""<title>Prompt Intent Funnel</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
<header>
<div class="chips"><span class="chip explore">exploratory · addendum A1</span><span class="chip">no GPU · Mac</span>
<span class="chip">commit {esc(args.commit[:7])}</span><span class="chip">{esc(args.date)}</span></div>
<h1>How prompt intent moves generative search</h1>
<p class="lede">From the prompt's position on the information-seeking → action-ready axis to the AI's queries, the frozen search,
the reranker's shortlist and the generator's ranking, with SEO and page data from the first experiment. Associations only.</p>
<div class="axis" aria-hidden="true"></div><div class="axislabels"><span>0 · information seeking</span><span>1 · action ready</span></div>
<nav class="toc">{toc}</nav>
</header>
{('<section id="summary"><h2>What the data show</h2><div class="callout">' + narrative + "</div></section>") if narrative else ""}
{body}
</main>"""
    out.write_text(page, encoding="utf-8")
    bundle = out.with_name(out.stem + "-results.json")
    bundle.write_text(json.dumps({"review": review, "explore": explore, "funnel": funnel, "contrasts": contrasts,
                                  "keywords": keywords_summary, "coverage": coverage, "commit": args.commit, "date": args.date,
                                  "exploratory": True}, indent=1), encoding="utf-8")
    print(json.dumps({"report": str(out), "bundle": str(bundle), "tables": tables, "bytes": out.stat().st_size}), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--explore", type=Path)
    parser.add_argument("--funnel", type=Path, help="funnel_study.py report folder (results.json)")
    parser.add_argument("--keywords", type=Path)
    parser.add_argument("--contrasts", type=Path, help="funnel_contrasts.py output folder")
    parser.add_argument("--figures", type=Path, help="folder with the fig4-fig8 PNGs to embed")
    parser.add_argument("--features", type=Path, default=Path.home() / "Hamburg/geodml-inputs/funnel-features-v1")
    parser.add_argument("--narrative", type=Path, help="HTML fragment with the written findings")
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--date", default="")
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
