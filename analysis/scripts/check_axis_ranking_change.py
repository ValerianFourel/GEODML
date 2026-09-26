#!/usr/bin/env python3
"""Check the prompt-axis -> top-3/top-5 ranking-change analysis on saved data.

Read-only and CPU-only. Loads verified rankings exactly as
report_axis_ranking_change.py does, recomputes every full top-k measurement,
pair set and Spearman correlation from scratch, compares them with the report
code pair by pair, and describes how the dataset stores what it compares. Any
mismatch exits non-zero. This validates the computation, not a scientific effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.stats import spearmanr

from analysis.scripts.report_axis_ranking_change import FACTORS, compare_prompts, summarize
from analysis.scripts.report_latent_ranking_relationship import _load_coordinates, _load_observations

OBSERVED = "observed_axis_1_percentile_0_1"
ASSIGNED = "assigned_axis_1_0_1"
TOP_K = (3, 5)
MEASURES = ("set_change", "ordered_change", "set_distance", "kendall_distance_common")
TOLERANCE = 1e-9
SHOWN_FAILURES = 20


def top_k_measures(left, right, k):
    """Spec measurements, written independently of the report; None if a list is short."""
    if len(left) < k or len(right) < k:
        return None
    a, b = left[:k], right[:k]
    b_position = {url: index for index, url in enumerate(b)}
    shared = [url for url in a if url in b_position]  # in a's order
    kendall = None
    if len(shared) >= 2:
        pairs = list(combinations(shared, 2))
        kendall = sum(b_position[first] > b_position[second] for first, second in pairs) / len(pairs)
    return {"set_change": float(set(a) != set(b)), "ordered_change": float(a != b),
            "set_distance": 1 - len(set(a) & set(b)) / len(set(a) | set(b)),
            "kendall_distance_common": kendall}


def loose(url):
    """Diagnostic only: scheme, www., case of host, trailing slash and fragment ignored."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    return host + (parts.path.rstrip("/") or "/") + ("?" + parts.query if parts.query else "")


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def independent_summary(records, metric):
    valid = [r for r in records if r[metric] is not None]
    by_keyword = defaultdict(list)
    for r in valid:
        by_keyword[r["keyword_id"]].append(r)
    within = [rho for rows in by_keyword.values()
              if (rho := spearman([r["axis_gap"] for r in rows], [r[metric] for r in rows])) is not None]
    return {"pairs": len(valid), "excluded_pairs": len(records) - len(valid), "keywords": len(by_keyword),
            "mean_distance": statistics.fmean(r[metric] for r in valid) if valid else None,
            "rho_axis_gap_vs_ranking_distance": spearman([r["axis_gap"] for r in valid], [r[metric] for r in valid]),
            "median_within_keyword_rho": statistics.median(within) if within else None,
            "keywords_with_defined_rho": len(within)}


def same(left, right):
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= TOLERANCE


def spread(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "min": values[0], "median": statistics.median(values), "max": values[-1]}


def describe(root, coordinates, observations, accounting, examples):
    """How the saved data looks: the facts a reader needs before trusting the table."""
    tables = {directory.name: sum(1 for _ in directory.glob("part-*.manifest.json"))
              for directory in sorted((root / "data").glob("*")) if directory.is_dir()}
    observed = [row[OBSERVED] for row in coordinates.values()]
    prompts_per_keyword = defaultdict(set)
    cell_prompts = Counter()
    lengths, shapes = Counter(), Counter()
    for row in observations:
        prompts_per_keyword[row["keyword_id"]].add(row["prompt_id"])
        cell_prompts[tuple(row[k] for k in (*FACTORS, "keyword_id"))] += 1
        ranking = row["ranking"]
        lengths["10+" if len(ranking) >= 10 else str(len(ranking))] += 1
        shapes["urls"] += len(ranking)
        shapes["http_not_https"] += sum(url.startswith("http://") for url in ranking)
        shapes["with_query"] += sum("?" in url for url in ranking)
        shapes["with_fragment"] += sum("#" in url for url in ranking)
        shapes["trailing_slash"] += sum(url.endswith("/") for url in ranking)
        shapes["rankings_with_loose_duplicates"] += len({loose(u) for u in ranking}) != len(ranking)
    combos = Counter(" | ".join(str(row[k]) for k in FACTORS) for row in observations)
    shown = sorted(observations, key=lambda r: r["task_id"])[:examples]
    return {
        "dataset_root": str(root), "data_tables_shard_count": tables, "accounting": accounting,
        "coordinates": {"rows": len(coordinates), "observed_range": [min(observed), max(observed)],
                        "observed_distinct": len(set(observed)),
                        "assigned_distinct": len({row[ASSIGNED] for row in coordinates.values()}),
                        "axis_bins": dict(sorted(Counter(row["axis_bin"] for row in coordinates.values()).items()))},
        "observations_by_model_method_engine_condition": dict(sorted(combos.items())),
        "keywords_with_observations": len(prompts_per_keyword),
        "prompts_per_keyword_any_cell": spread([len(v) for v in prompts_per_keyword.values()]),
        "prompts_per_comparison_group": spread(list(cell_prompts.values())),
        "ranking_length_counts": dict(sorted(lengths.items(), key=lambda item: (len(item[0]), item[0]))),
        "url_shapes": dict(shapes),
        "examples": [{"task_id": r["task_id"], "prompt_id": r["prompt_id"], "keyword_id": r["keyword_id"],
                      **{k: r[k] for k in FACTORS}, OBSERVED: r[OBSERVED],
                      "ranking_length": len(r["ranking"]), "top5": r["ranking"][:5]} for r in shown],
    }


def check(root, *, models=("qwen38", "llama4"), stripes=256, keyword_limit=None,
          max_pairs=2000, seed=20260926, examples=2):
    started = time.monotonic()
    registration = root / "local-only/population-registration-v1"
    coordinates, provenance = _load_coordinates(registration / "population-selection-records.jsonl",
                                                registration / "manifest.json")
    observations, accounting = _load_observations(root, coordinates, set(models), stripe_count=stripes)
    loaded = time.monotonic()
    groups = defaultdict(list)
    for row in observations:
        groups[tuple(row[k] for k in (*FACTORS, "keyword_id"))].append(row)
    keywords = sorted({key[-1] for key in groups}, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    selected = set(keywords if keyword_limit is None else keywords[:keyword_limit])
    failures, combos = [], defaultdict(lambda: {"production": [], "independent": [], "counts": Counter(),
                                                 "keywords": set(), "usable_keywords": defaultdict(set)})

    def fail(kind, **detail):
        failures.append({"kind": kind, **detail})

    for key in sorted(groups):
        if key[-1] not in selected:
            continue
        combo = combos[key[:-1]]
        combo["keywords"].add(key[-1])
        rows = sorted(groups[key], key=lambda r: r["prompt_id"])
        try:
            produced = {(p["low_prompt"], p["high_prompt"]): p
                        for p in compare_prompts(rows, max_pairs=max_pairs, seed=seed)
                        if p["coordinate_role"] == "observed_latent"}
        except ValueError as error:
            fail("report_rejected_group", group=list(key), error=str(error))
            continue
        combo["production"].extend(produced.values())
        sampled = len(rows) * (len(rows) - 1) // 2 > max_pairs
        counts = combo["counts"]
        for a, b in combinations(rows, 2):
            counts["prompt_pairs"] += 1
            low, high = (a, b) if a[OBSERVED] <= b[OBSERVED] else (b, a)
            report = produced.pop((low["prompt_id"], high["prompt_id"]), None)
            gap = abs(a[OBSERVED] - b[OBSERVED])
            if not a["ranking"] or not b["ranking"] or gap == 0:
                counts["empty_ranking_pairs" if not a["ranking"] or not b["ranking"] else "zero_gap_pairs"] += 1
                if report is not None:
                    fail("report_kept_unusable_pair", group=list(key), pair=[low["prompt_id"], high["prompt_id"]])
                continue
            if report is None:
                if sampled:
                    counts["not_sampled_pairs"] += 1
                else:
                    fail("report_missing_pair", group=list(key), pair=[low["prompt_id"], high["prompt_id"]])
                continue
            record = {"keyword_id": key[-1], "axis_gap": gap}
            if not same(report["axis_gap"], gap):
                fail("axis_gap_mismatch", group=list(key), report=report["axis_gap"], expected=gap)
            for k in TOP_K:
                mine = top_k_measures(a["ranking"], b["ranking"], k)
                counts[f"top{k}_{'usable' if mine else 'short'}_pairs"] += 1
                for name in MEASURES:
                    expected = None if mine is None else mine[name]
                    field = f"top{k}_full_{name}"
                    record[field] = expected
                    if not same(report[field], expected):
                        fail("measure_mismatch", group=list(key), pair=[low["prompt_id"], high["prompt_id"]],
                             measure=field, report=report[field], expected=expected)
                if mine is None:
                    continue
                combo["usable_keywords"][k].add(key[-1])
                if mine["set_change"] > mine["ordered_change"] or mine["set_change"] != float(mine["set_distance"] > 0):
                    fail("measure_invariant", group=list(key), measure=f"top{k}", values=mine)
                if mine["set_change"] and {loose(u) for u in a["ranking"][:k]} == {loose(u) for u in b["ranking"][:k]}:
                    counts[f"top{k}_string_only_membership_changes"] += 1
            combo["independent"].append(record)
        for pair in produced:
            fail("report_extra_pair", group=list(key), pair=list(pair))
    groups_out = []
    for combo_key, combo in sorted(combos.items()):
        summaries, _ = summarize(combo["production"])
        reported = {row["metric"]: row for row in summaries if row["coordinate_role"] == "observed_latent"}
        row = {**dict(zip(FACTORS, combo_key)), "keywords": len(combo["keywords"]), **combo["counts"]}
        for k in TOP_K:
            row[f"top{k}_usable_keywords"] = len(combo["usable_keywords"][k])
            for name in MEASURES:
                metric = f"top{k}_full_{name}"
                mine = independent_summary(combo["independent"], metric)
                row[metric] = mine
                theirs = reported.get(metric)
                if theirs is None:
                    if combo["independent"]:
                        fail("summary_missing", group=list(combo_key), metric=metric)
                    continue
                for field, value in mine.items():
                    if not same(theirs[field], value):
                        fail("summary_mismatch", group=list(combo_key), metric=metric, field=field,
                             report=theirs[field], expected=value)
        groups_out.append(row)
    return {"format_version": "geodml-axis-ranking-check-v1", "scientific_result": False,
            "status": "PASS" if not failures else "FAIL", "failure_count": len(failures),
            "failures": failures[:SHOWN_FAILURES], "coordinates": provenance,
            "keywords_checked": len(selected), "keywords_available": len(keywords),
            "max_pairs_per_keyword": max_pairs, "seed": seed,
            "seconds": {"load": round(loaded - started, 1), "check": round(time.monotonic() - loaded, 1)},
            "data": describe(root, coordinates, observations, accounting, examples), "groups": groups_out}


def text(result):
    def pct(value):
        return "NA" if value is None else f"{100 * value:.1f}"

    def num(value):
        return "NA" if value is None else f"{value:+.3f}"

    data = result["data"]
    lines = [f"CHECK={result['status']} failures={result['failure_count']} "
             f"keywords={result['keywords_checked']}/{result['keywords_available']} "
             f"load_s={result['seconds']['load']} check_s={result['seconds']['check']}",
             "DATA " + json.dumps({k: data[k] for k in (
                 "accounting", "keywords_with_observations", "prompts_per_keyword_any_cell",
                 "prompts_per_comparison_group", "ranking_length_counts", "url_shapes")}),
             "COORDINATES " + json.dumps(data["coordinates"]),
             "TABLES " + json.dumps(data["data_tables_shard_count"])]
    lines += ["EXAMPLE " + json.dumps(example) for example in data["examples"]]
    lines.append(f"{'model':<7} {'method':<22} {'engine':<10} {'cond':<8} {'k':>1} {'KW':>4} {'useKW':>5} "
                 f"{'pairs':>7} {'short':>6} {'empty':>5} {'tie':>4} {'SetΔ%':>6} {'ListΔ%':>6} "
                 f"{'Jacc':>5} {'ρ':>6} {'ρ_kw':>6} {'strΔ':>5}")
    for row in result["groups"]:
        for k in TOP_K:
            summary = row[f"top{k}_full_set_distance"]
            lines.append(
                f"{row['model']:<7} {str(row['method'])[:22]:<22} {str(row['engine'])[:10]:<10} "
                f"{str(row['condition'])[:8]:<8} {k:>1} {row['keywords']:>4} {row[f'top{k}_usable_keywords']:>5} "
                f"{summary['pairs']:>7} {row.get(f'top{k}_short_pairs', 0):>6} {row.get('empty_ranking_pairs', 0):>5} "
                f"{row.get('zero_gap_pairs', 0):>4} {pct(row[f'top{k}_full_set_change']['mean_distance']):>6} "
                f"{pct(row[f'top{k}_full_ordered_change']['mean_distance']):>6} "
                f"{'NA' if summary['mean_distance'] is None else format(summary['mean_distance'], '.3f'):>5} "
                f"{num(summary['rho_axis_gap_vs_ranking_distance']):>6} {num(summary['median_within_keyword_rho']):>6} "
                f"{row.get(f'top{k}_string_only_membership_changes', 0):>5}")
    lines += ["FAILURE " + json.dumps(failure) for failure in result["failures"]]
    lines.append("KEY: KW keywords in group; useKW with >=1 full top-k pair; short = a list below k URLs; "
                 "empty/tie = pairs the report drops (empty list, equal coordinate); strΔ = membership "
                 "changes that vanish when scheme/www/trailing slash are ignored (diagnostic only).")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen38", "llama4"])
    parser.add_argument("--stripes", type=int, default=256)
    parser.add_argument("--keywords", type=int, help="check only this many keywords (stable hash order)")
    parser.add_argument("--max-pairs-per-keyword", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--output", type=Path, help="new JSON path for the full result")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("output already exists; use a new path")
    result = check(args.dataset_root.resolve(), models=tuple(args.models), stripes=args.stripes,
                   keyword_limit=args.keywords, max_pairs=args.max_pairs_per_keyword,
                   seed=args.seed, examples=args.examples)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(text(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
