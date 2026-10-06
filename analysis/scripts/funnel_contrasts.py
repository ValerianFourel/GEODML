#!/usr/bin/env python3
"""Model, method and engine contrasts of the within-keyword slopes on x (Mac, CPU; exploratory).

Reads the per-answer table of funnel_review.py (all keywords, natural condition) and, for each measure,
estimates the difference between two strata's within-keyword slopes on the prompt position x, with a
95% interval from keyword-bootstrap draws shared by both strata (the strata share keywords and prompts,
so independent intervals would be wrong). Contrasts: Llama − Qwen within each search method,
Reactive − Parallel within each model, SearXNG − DuckDuckGo within each model. Associations only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

MEASURES = ("cited_u", "alignment_gain", "cited_gap", "cited_on_topic", "cited_glued", "r0_overlap", "ranking_len",
            "answer_imperatives", "answer_chars", "cited_google_url", "cited_dfs_organic_count", "cited_body_word_count")
METHOD = {"Parallel-Expansion-v1": "Parallel", "Reactive-Snippet-Loop-v1": "Reactive"}
CONTRASTS = [  # (name, column, value a, value b, fixed column, fixed value)
    ("Llama − Qwen · Parallel", "model", "llama4", "qwen38", "method", "Parallel"),
    ("Llama − Qwen · Reactive", "model", "llama4", "qwen38", "method", "Reactive"),
    ("Reactive − Parallel · Llama", "method", "Reactive", "Parallel", "model", "llama4"),
    ("Reactive − Parallel · Qwen", "method", "Reactive", "Parallel", "model", "qwen38"),
    ("SearXNG − DuckDuckGo · Llama", "engine", "searxng", "duckduckgo", "model", "llama4"),
    ("SearXNG − DuckDuckGo · Qwen", "engine", "searxng", "duckduckgo", "model", "qwen38"),
]


def slope_contrast(x, y, k, in_a, in_b, draws) -> dict:
    """Difference of within-keyword slopes (a − b) with shared keyword-bootstrap draws."""
    ok = np.isfinite(y) & np.isfinite(x)
    a, b = ok & in_a, ok & in_b
    if a.sum() < 50 or b.sum() < 50:
        return {}
    sa, sb = geo.within_keyword_slope(x[a], y[a], k[a]), geo.within_keyword_slope(x[b], y[b], k[b])
    boot = np.asarray([geo.within_keyword_slope(x[a], y[a], k[a], d[k[a]]) - geo.within_keyword_slope(x[b], y[b], k[b], d[k[b]])
                       for d in draws])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    return {"slope_a": float(sa), "slope_b": float(sb), "difference": float(sa - sb), "ci95": [float(lo), float(hi)],
            "answers_a": int(a.sum()), "answers_b": int(b.sum()), "draws": int(len(boot))}


def build(args) -> int:
    import pandas as pd
    frame = pd.read_parquet(Path(args.review) / "answers.parquet")
    frame = frame[frame["condition"] == "natural"].copy()
    frame["method"] = frame["method"].map(METHOD).fillna(frame["method"])
    keywords = sorted(frame["keyword"].unique())
    code = {kw: i for i, kw in enumerate(keywords)}
    k = frame["keyword"].map(code).to_numpy(np.int64)
    x = frame["x"].to_numpy(float)
    draws = stages.keyword_draws(len(keywords), args.bootstrap, args.seed)
    out = {"answers": int(len(frame)), "keywords": len(keywords), "bootstrap": args.bootstrap, "seed": args.seed,
           "condition": "natural", "exploratory": True, "contrasts": {}}
    rows = []
    for name, column, va, vb, fixed, fv in CONTRASTS:
        base = (frame[fixed] == fv).to_numpy()
        in_a, in_b = base & (frame[column] == va).to_numpy(), base & (frame[column] == vb).to_numpy()
        out["contrasts"][name] = {}
        for measure in MEASURES:
            if measure not in frame:
                continue
            c = slope_contrast(x, frame[measure].to_numpy(float), k, in_a, in_b, draws)
            if c:
                out["contrasts"][name][measure] = c
                rows.append({"contrast": name, "measure": measure, **{key: c[key] for key in ("slope_a", "slope_b", "difference")},
                             "ci95_lo": c["ci95"][0], "ci95_hi": c["ci95"][1], "answers_a": c["answers_a"], "answers_b": c["answers_b"]})
        print(json.dumps({"contrast": name, "cited_u": out["contrasts"][name].get("cited_u", {}).get("difference")}), flush=True)
    target = Path(args.output)
    for candidate in (target, target.with_name(target.name + ".partial")):
        if candidate.exists():
            raise ValueError(f"refusing to overwrite {candidate}")
    partial = readiness.new_directory(target)
    readiness.write_json(partial / "contrasts.json", out)
    with open(partial / "contrasts.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["contrast"])
        writer.writeheader()
        writer.writerows(rows)
    partial.rename(target.resolve())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", type=Path, required=True, help="funnel_review.py output folder")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
