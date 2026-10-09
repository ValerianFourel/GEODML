#!/usr/bin/env python3
"""Does the Gemma SI-v4 judge agree with the generator LLM's own source choices?

Per answered cell the generator saw `presented` sources and returned `generator_ranking`: the shown sources
it kept, most important first. Gemma graded every shown source 0-5 for how much the answer rests on it.
  keep  - do kept sources get higher grades than dropped ones? Cell AUC (ties count half), shares of kept
          sources graded 0 and of grade>=4 sources dropped, and pooled kappa of "kept" vs "graded >=1/>=4".
  order - among kept sources, does the generator's order follow Gemma's grades? Kendall tau-b, and whether
          the generator's first source is in Gemma's top grade group (against its chance share).
Baseline for both: the order the sources were shown. A generator that keeps or orders by shown position
alone scores the same against Gemma as the position does; the paired differences test agreement beyond it.

Cells count when every shown source has a scored grade (the judge's `complete`); the others are tallied by
reason. Duplicate cells across runs (partial snapshots, recovery runs) keep the complete copy, else the
first input. Intervals: percentile bootstrap over prompt_id clusters. Associations only: Gemma is a reference
model, not ground truth, and semantic acceptance of its judgments is not established.

Inputs: run roots (plan.json; each shard's latest report) or directories of downloaded cells.jsonl.gz.
Writes OUTPUT/{report.json,strata.csv,sources.csv.gz}; refuses an existing OUTPUT. Never changes judgments.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.source_importance import kendall_tau_b

STRATA = ("model", "method", "engine", "condition")
METRICS = ("keep_auc", "position_keep_auc", "keep_auc_minus_position", "kept_zero_share", "dropped_used_share",
           "important_dropped_share", "order_tau_b", "position_order_tau_b", "order_tau_minus_position",
           "generator_follows_shown_order_tau_b", "top_source_alignment", "top_source_chance",
           "top_alignment_minus_chance")


def report_files(path: Path):
    """cells.jsonl.gz of each shard's latest report (run root), or every one below a directory."""
    if (path / "plan.json").is_file():
        for shard in json.loads((path / "plan.json").read_text())["shards"]:
            reports = Path(shard["directory"]) / "results/reports"
            if not (reports / "latest.json").is_file():
                continue
            writer = json.loads((reports / "latest.json").read_text()).get("directory")
            if not isinstance(writer, str) or Path(writer).name != writer or writer in (".", ".."):
                raise ValueError(f"invalid report pointer in {reports}")
            if (reports / writer / "cells.jsonl.gz").is_file():
                yield reports / writer / "cells.jsonl.gz"
    else:
        yield from sorted(path.rglob("cells.jsonl.gz"))


def compact(cell: dict, origin: str) -> dict:
    sources = {s["url"]: s for s in cell.get("sources", [])}
    presented = list(cell.get("presented") or [])
    return {"fingerprint": cell.get("fingerprint") or cell.get("cell_id"), "origin": origin,
            **{k: cell.get(k) for k in ("cell_id", "prompt_id", *STRATA)},
            "status": cell.get("status"), "map_eligibility": cell.get("map_eligibility"),
            "presented": presented, "ranking": list(cell.get("generator_ranking") or []),
            "grades": [(cell.get("grades") or {}).get(u) for u in presented],
            "statuses": [sources.get(u, {}).get("status", "missing") for u in presented]}


def load(inputs, counts: Counter) -> tuple[dict, list]:
    chosen, files = {}, []
    for root in inputs:
        for path in report_files(Path(root)):
            raw = path.read_bytes()
            files.append({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            for line in io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="utf-8"):
                row = compact(json.loads(line), str(path))
                counts["cells_read"] += 1
                previous = chosen.get(row["fingerprint"])
                if previous is None:
                    chosen[row["fingerprint"]] = row
                    continue
                counts["duplicate_cells"] += 1
                if not complete(previous) and complete(row):
                    chosen[row["fingerprint"]] = row
                elif complete(previous) and complete(row) and previous["grades"] != row["grades"]:
                    counts["duplicate_complete_cells_with_different_grades"] += 1
    return chosen, files


def complete(row: dict) -> bool:
    return row["status"] == "ok" and bool(row["presented"]) and all(g is not None for g in row["grades"])


def auc(high, low):
    """P(a random `high` value exceeds a random `low` value), ties counting half."""
    if not high or not low:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in high for b in low)
    return wins / (len(high) * len(low))


def share(flags):
    return sum(flags) / len(flags) if flags else None


def minus(a, b):
    return None if a is None or b is None else a - b


def cell_metrics(row: dict) -> dict:
    """Keep and order agreement for one complete cell; undefined metrics are None."""
    presented, grades = row["presented"], dict(zip(row["presented"], row["grades"]))
    kept = [u for u in row["ranking"] if u in grades]
    if len(kept) != len(row["ranking"]) or len(set(kept)) != len(kept):
        raise ValueError(f"generator ranking outside the shown sources: {row['fingerprint']}")
    kept_set = set(kept)
    dropped = [u for u in presented if u not in kept_set]
    position = {u: i for i, u in enumerate(presented)}
    out = {"keep_auc": auc([grades[u] for u in kept], [grades[u] for u in dropped]),
           # Shown position as a predictor of keeping (earlier = higher score), same pairs as keep_auc.
           "position_keep_auc": auc([-position[u] for u in kept], [-position[u] for u in dropped]),
           "kept_zero_share": share([grades[u] == 0 for u in kept]),
           "dropped_used_share": share([grades[u] >= 1 for u in dropped]),
           "important_dropped_share": share([u not in kept_set for u in presented if grades[u] >= 4])}
    out["keep_auc_minus_position"] = minus(out["keep_auc"], out["position_keep_auc"])
    if len(kept) >= 2:
        generator_order = [-i for i in range(len(kept))]
        shown_order = [-position[u] for u in kept]
        kept_grades = [grades[u] for u in kept]
        out["order_tau_b"] = kendall_tau_b(generator_order, kept_grades)
        out["position_order_tau_b"] = kendall_tau_b(shown_order, kept_grades)
        out["generator_follows_shown_order_tau_b"] = kendall_tau_b(generator_order, shown_order)
    else:
        out["order_tau_b"] = out["position_order_tau_b"] = out["generator_follows_shown_order_tau_b"] = None
    out["order_tau_minus_position"] = minus(out["order_tau_b"], out["position_order_tau_b"])
    best = max(grades.values())
    if kept and best > 0:
        top = {u for u, g in grades.items() if g == best}
        out["top_source_alignment"] = float(kept[0] in top)
        out["top_source_chance"] = len(top & kept_set) / len(kept)
    else:
        out["top_source_alignment"] = out["top_source_chance"] = None
    out["top_alignment_minus_chance"] = minus(out["top_source_alignment"], out["top_source_chance"])
    return out


def bootstrap(values, clusters, *, replicates: int, seed: int, level: float = 0.95) -> dict:
    """Mean of defined values with a percentile interval from resampling whole clusters."""
    keep = [(v, c) for v, c in zip(values, clusters) if v is not None]
    if not keep:
        return {"estimate": None, "n": 0, "clusters": 0, "low": None, "high": None}
    index = {c: i for i, c in enumerate(sorted({c for _, c in keep}, key=str))}
    sums, counts = np.zeros(len(index)), np.zeros(len(index))
    for v, c in keep:
        sums[index[c]] += float(v)
        counts[index[c]] += 1
    out = {"estimate": sums.sum() / counts.sum(), "n": len(keep), "clusters": len(index), "low": None, "high": None}
    if replicates and len(index) > 1:
        rng = np.random.default_rng(seed)
        stats = np.empty(replicates)
        for r in range(replicates):
            weight = np.bincount(rng.integers(0, len(index), len(index)), minlength=len(index))
            stats[r] = weight @ sums / (weight @ counts)
        tail = (1 - level) / 2
        out["low"], out["high"] = (float(x) for x in np.quantile(stats, [tail, 1 - tail]))
    return out


def kappa(table: Counter) -> float | None:
    """Cohen's kappa for a 2x2 table keyed by (a, b) booleans."""
    n = sum(table.values())
    if not n:
        return None
    observed = (table[(True, True)] + table[(False, False)]) / n
    a = (table[(True, True)] + table[(True, False)]) / n
    b = (table[(True, True)] + table[(False, True)]) / n
    expected = a * b + (1 - a) * (1 - b)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def analyze(inputs, output: Path, *, replicates: int = 2000, seed: int = 20261009) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    cells, files = load(inputs, counts)
    rows, exclusions = [], Counter()
    by_grade = defaultdict(Counter)
    agreement = {"kept_vs_graded_1_or_more": Counter(), "kept_vs_graded_4_or_more": Counter()}
    with gzip.open(output / "sources.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fingerprint", "cell_id", "prompt_id", *STRATA, "url", "shown_position", "kept",
                         "generator_rank", "grade", "source_status", "map_eligibility", "cell_complete"])
        for row in cells.values():
            is_complete = complete(row)
            rank = {u: i + 1 for i, u in enumerate(row["ranking"])}
            for i, (url, grade, status) in enumerate(zip(row["presented"], row["grades"], row["statuses"])):
                writer.writerow([row["fingerprint"], row["cell_id"], row["prompt_id"], *(row[k] for k in STRATA),
                                 url, i + 1, int(url in rank), rank.get(url, ""), "" if grade is None else grade,
                                 status, row["map_eligibility"], int(is_complete)])
            if not is_complete:
                if row["status"] != "ok":
                    reason = f"cell_status:{row['status']}"
                elif not row["presented"]:
                    reason = "no_shown_sources"
                else:
                    reason = "source_status:" + "+".join(sorted({s for s, g in zip(row["statuses"], row["grades"])
                                                                 if g is None}))
                exclusions[reason] += 1
                continue
            metrics = cell_metrics(row)
            rows.append({**{k: row[k] for k in ("prompt_id", *STRATA)}, **metrics})
            kept = set(row["ranking"])
            for url, grade in zip(row["presented"], row["grades"]):
                by_grade[grade][url in kept] += 1
                agreement["kept_vs_graded_1_or_more"][(url in kept, grade >= 1)] += 1
                agreement["kept_vs_graded_4_or_more"][(url in kept, grade >= 4)] += 1

    def summarize(subset, seed_offset, with_intervals):
        clusters = [r["prompt_id"] for r in subset]
        return {"cells": len(subset), **{m: bootstrap([r[m] for r in subset], clusters,
                replicates=replicates if with_intervals else 0, seed=seed + seed_offset) for m in METRICS}}

    groups = defaultdict(list)
    for r in rows:
        groups[("all",)].append(r)
        groups[("model", r["model"])].append(r)
        for field in STRATA[1:]:
            groups[(field, r["model"], r[field])].append(r)
    strata = {}
    for i, key in enumerate(sorted(groups, key=str)):
        # Intervals for the overall and per-generator rows; finer strata report estimates only.
        strata[":".join(map(str, key))] = summarize(groups[key], i, len(key) <= 2)
    with (output / "strata.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stratum", "cells", "metric", "estimate", "n", "clusters", "low", "high"])
        for name, value in strata.items():
            for metric in METRICS:
                m = value[metric]
                writer.writerow([name, value["cells"], metric, m["estimate"], m["n"], m["clusters"], m["low"], m["high"]])
    report = {"format_version": "gemma-generator-agreement-v1", "scientific_result": False,
              "semantic_acceptance": "not_established", "reference_is_human_gold": False,
              "interpretation": "associations between the generator's kept/ordered sources and Gemma's grades",
              "inputs": [str(Path(p).resolve()) for p in inputs], "files": files,
              "counts": {**counts, "unique_cells": len(cells), "complete_cells": len(rows)},
              "excluded_cells_by_reason": dict(exclusions.most_common()),
              "bootstrap": {"cluster": "prompt_id", "replicates": replicates, "seed": seed, "level": 0.95},
              "kept_share_by_grade": {str(g): {"sources": t[True] + t[False], "kept": t[True],
                                               "kept_share": t[True] / (t[True] + t[False])}
                                      for g, t in sorted(by_grade.items())},
              "source_level_kappa": {name: {"kappa": kappa(t), "table": {f"kept={a},graded={b}": n
                                                                         for (a, b), n in sorted(t.items())}}
                                     for name, t in agreement.items()},
              "strata": strata}
    (output / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Gemma v4 run root (plan.json) or a directory of downloaded cells.jsonl.gz")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the results")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args(argv)
    report = analyze(args.input, args.output, replicates=args.replicates, seed=args.seed)
    overall = report["strata"]["all"]
    print(json.dumps({"complete_cells": report["counts"]["complete_cells"],
                      "unique_cells": report["counts"]["unique_cells"],
                      **{m: overall[m]["estimate"] for m in ("keep_auc", "position_keep_auc", "order_tau_b",
                                                              "position_order_tau_b", "top_source_alignment",
                                                              "top_source_chance")},
                      "output": str(args.output)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
