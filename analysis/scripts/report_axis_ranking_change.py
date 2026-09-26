#!/usr/bin/env python3
"""Describe ranking differences BETWEEN prompts along their recorded axis.

Within-keyword comparisons, not a fixed-query/candidate causal experiment.
Uses verified local ledger completions; shared-HF completions may be absent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.report_latent_ranking_relationship import (
    COORDINATES, _load_coordinates, _load_observations, _correlation,
)
from analysis.interpretability.pipeline.axis_permutation_metrics import ranking_agreement
from analysis.scripts.summarize_axis_ranking_change import format_top_k_summary
import numpy as np

FACTORS = ("model", "method", "engine", "condition")
METRICS = ("top1_change", "top3_set_distance", "kendall_distance_common", "url_set_distance") + tuple(
    f"top{k}_full_{name}"
    for k in (3, 5)
    for name in ("set_change", "ordered_change", "set_distance", "kendall_distance_common")
)


def compare_prompts(observations, *, max_pairs=2000, seed=20260926):
    """Sample prompt pairs uniformly within keywords, separately by condition."""
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    groups = defaultdict(list)
    for row in observations:
        groups[tuple(row[k] for k in (*FACTORS, "keyword_id"))].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["prompt_id"])
        if len({r["prompt_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate prompt within comparison group")
        n = len(rows)
        total = n * (n - 1) // 2
        if total <= max_pairs:
            indices = combinations(range(n), 2)
        else:
            digest = hashlib.sha256(json.dumps([seed, key]).encode()).hexdigest()
            rng = random.Random(int(digest, 16))
            chosen = set()
            while len(chosen) < max_pairs:
                i, j = sorted(rng.sample(range(n), 2))
                chosen.add((i, j))
            indices = sorted(chosen)
        for i, j in indices:
            a, b = rows[i], rows[j]
            left, right = a["ranking"], b["ranking"]
            if not left or not right:
                continue
            agreement = ranking_agreement(left, right)
            union = set(left) | set(right)
            top_union = set(left[:3]) | set(right[:3])
            top_k = {}
            for k in (3, 5):
                prefix = f"top{k}_full_"
                values = dict.fromkeys(("set_change", "ordered_change", "set_distance", "kendall_distance_common"))
                if len(left) >= k and len(right) >= k:
                    a_top, b_top = left[:k], right[:k]
                    common = set(a_top) & set(b_top)
                    agreement_k = ranking_agreement(a_top, b_top)
                    tau = agreement_k["kendall_tau_common"]
                    values = {
                        "set_change": float(set(a_top) != set(b_top)),
                        "ordered_change": float(a_top != b_top),
                        "set_distance": 1 - len(common) / len(set(a_top) | set(b_top)),
                        "kendall_distance_common": None if tau is None else (1 - tau) / 2,
                    }
                top_k.update({prefix + name: value for name, value in values.items()})
            for field, role in COORDINATES:
                # Orient each comparison from low to high axis position.
                low, high = (a, b) if a[field] <= b[field] else (b, a)
                delta = high[field] - low[field]
                if delta == 0:
                    continue
                result.append({
                    **dict(zip((*FACTORS, "keyword_id"), key)),
                    "coordinate_role": role,
                    "low_prompt": low["prompt_id"], "high_prompt": high["prompt_id"],
                    "axis_gap": delta,
                    "low_top1": low["ranking"][0], "high_top1": high["ranking"][0],
                    "ranking_length_change": len(high["ranking"]) - len(low["ranking"]),
                    "common_urls": agreement["common_count"],
                    "top1_change": float(not agreement["top1_match"]),
                    "top3_set_distance": 1 - len(set(left[:3]) & set(right[:3])) / len(top_union),
                    "kendall_distance_common": (
                        None if agreement["kendall_tau_common"] is None
                        else (1 - agreement["kendall_tau_common"]) / 2
                    ),
                    "url_set_distance": 1 - len(set(left) & set(right)) / len(union),
                    **top_k,
                })
    return result


def summarize(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[tuple(row[k] for k in (*FACTORS, "coordinate_role"))].append(row)
    summaries, bins = [], []
    for key, rows in sorted(groups.items()):
        identity = dict(zip((*FACTORS, "coordinate_role"), key))
        for metric in METRICS:
            valid = [r for r in rows if r[metric] is not None]
            x = np.array([r["axis_gap"] for r in valid])
            y = np.array([r[metric] for r in valid])
            rho = _correlation(x, y) if len(valid) >= 3 else float("nan")
            # Also report per-keyword rhos: pooled distance association can
            # otherwise be driven by differences between keywords.
            by_keyword = defaultdict(list)
            for r in valid:
                by_keyword[r["keyword_id"]].append(r)
            within = []
            for group in by_keyword.values():
                if len(group) >= 3:
                    value = _correlation(np.array([r["axis_gap"] for r in group]),
                                         np.array([r[metric] for r in group]))
                    if math.isfinite(value):
                        within.append(value)
            summaries.append({
                **identity, "metric": metric, "pairs": len(valid),
                "candidate_pairs": len(rows), "excluded_pairs": len(rows) - len(valid),
                "keywords": len(by_keyword),
                "mean_distance": float(y.mean()) if len(y) else None,
                "rho_axis_gap_vs_ranking_distance": rho if math.isfinite(rho) else None,
                "median_within_keyword_rho": float(np.median(within)) if within else None,
                "keywords_with_defined_rho": len(within),
            })
            for b in range(5):
                selected = [r for r in valid if min(4, int(r["axis_gap"] * 5)) == b]
                bins.append({**identity, "metric": metric,
                             "gap_lower": b / 5, "gap_upper": (b + 1) / 5,
                             "pairs": len(selected),
                             "mean_distance": float(np.mean([r[metric] for r in selected])) if selected else None})
    return summaries, bins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs-per-keyword", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--summary-only", action="store_true",
                        help="Print the compact top-3/top-5 table instead of the full JSON.")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; use a new path")
    registration = args.dataset_root / "local-only/population-registration-v1"
    coords, provenance = _load_coordinates(registration / "population-selection-records.jsonl",
                                           registration / "manifest.json")
    observations, accounting = _load_observations(args.dataset_root, coords,
                                                 {"qwen38", "llama4"}, stripe_count=256)
    pairs = compare_prompts(observations, max_pairs=args.max_pairs_per_keyword, seed=args.seed)
    summaries, bins = summarize(pairs)
    args.output_dir.mkdir(parents=True)
    for name, rows in (("prompt-pairs.csv", pairs), ("summary.csv", summaries), ("gap-bins.csv", bins)):
        with (args.output_dir / name).open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    report = {"format_version": "geodml-direct-axis-ranking-v2", "scientific_result": False,
              "comparison": "different prompts within keyword/model/method/engine/condition",
              "limitation": "Exact query, input candidates, evidence and surface realization are not matched. Descriptive association only. Pairs share prompts; no independent-pair p-values.",
              "top_k_policy": "Full top-3 and top-5 metrics require at least k URLs in both rankings. Set distance is Jaccard distance. Order distance uses only URLs shared within both top-k lists, requiring at least two. Legacy metrics are preserved unchanged.",
              "seed": args.seed, "max_pairs_per_keyword": args.max_pairs_per_keyword,
              "coordinates": provenance, "accounting": accounting,
              "observations": len(observations), "summaries": summaries}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = format_top_k_summary(report)
    (args.output_dir / "summary.txt").write_text(summary + "\n")
    print(summary if args.summary_only else json.dumps(report, indent=2, allow_nan=False))
    print("RESULTS=" + str(args.output_dir))


if __name__ == "__main__":
    main()
