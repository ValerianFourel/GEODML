#!/usr/bin/env python3
"""Compare bounded SI review reports; reference-model agreement is not accuracy.

Reads report directories produced by run_source_importance_judge. Never changes
judgments, resolves reviewer disagreements, or promotes a protocol automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.run_source_importance_judge import rows, file_hash
from analysis.interpretability.pipeline.cluster_bootstrap import cluster_bootstrap, paired_difference


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def load(directory):
    directory = Path(directory)
    cells = {}
    for cell in rows(directory / "cells.jsonl.gz"):
        if cell["cell_id"] in cells:
            raise ValueError("duplicate cell in report")
        cells[cell["cell_id"]] = cell
    return json.loads((directory / "summary.json").read_text()), cells


def comparable(a, b):
    for field in ("judged_answer_sha256", "masked_answer_sha256", "presented", "generator_ranking"):
        if field not in a or field not in b:
            raise ValueError(f"missing comparison input: {field}")
        if a.get(field) != b.get(field):
            raise ValueError(f"comparison inputs differ: {field}")
    sa, sb = ({s["url"]: s for s in c["sources"]} for c in (a, b))
    if sa.keys() != sb.keys() or any(sa[u].get("source_sha256") != sb[u].get("source_sha256") for u in sa):
        raise ValueError("comparison source inputs differ")
    return sa, sb


def compare(baseline, candidate, *, references=None, repetitions=(), repeat_kind="end_to_end", replicates=2000):
    old_summary, old = load(baseline)
    new_summary, new = load(candidate)
    config = lambda s: {k: v for k, v in s["configuration"].items() if k not in ("repetition_id", "tokenizer_path")}
    if config(old_summary) != config(new_summary):
        raise ValueError("architecture comparison requires the same judge configuration")
    if old_summary.get("preprocessing") != new_summary.get("preprocessing") or old_summary.get("max_tokens") != new_summary.get("max_tokens"):
        raise ValueError("preprocessing or output budgets differ; report as a separate configuration comparison")
    if old.keys() != new.keys():
        raise ValueError("architecture comparison requires the same frozen cell cohort")
    refs = {}
    if references:
        for ref in rows(references):
            key = (ref["cell_id"], ref["url"])
            if key in refs:
                raise ValueError("duplicate reference; retain reviewer disagreements explicitly")
            if not ref.get("reviewer_models") or not ref.get("frozen_before_grade_review"):
                raise ValueError("reference provenance and independent range freeze required")
            refs[key] = ref
    counts, comparison_rows, subgroup, maps = Counter(), [], defaultdict(Counter), {}
    for cid in old:
        if old[cid].get("status") != "ok" or new[cid].get("status") != "ok":
            counts["input_ineligible_cells"] += 1
            continue
        sa, sb = comparable(old[cid], new[cid])
        counts["common_complete_cells"] += int(all(sa[u].get("importance") is not None and
                                                   sb[u].get("importance") is not None for u in sa))
        for url in sa:
            counts["observed_sources"] += 1
            a, b = sa[url].get("importance"), sb[url].get("importance")
            counts["baseline_scored"] += a is not None
            counts["candidate_scored"] += b is not None
            reference = refs.get((cid, url))
            if not reference:
                counts["missing_references"] += 1
                continue
            if reference["source_sha256"] != sb[url]["source_sha256"] or reference["masked_answer_sha256"] != new[cid].get("masked_answer_sha256"):
                raise ValueError("reference input hash mismatch")
            if "map_assessment" in reference:
                if reference.get("map_sha256") != new[cid].get("map_sha256") or not reference.get("map_sha256"):
                    raise ValueError("map assessment refers to a different map")
                assessment = reference["map_assessment"]
                if set(assessment) != {"faithful", "essential_omission", "meaning_reversal"} or any(type(v) is not bool for v in assessment.values()):
                    raise ValueError("invalid map assessment")
                previous = maps.setdefault(reference["map_sha256"], assessment)
                if previous != assessment:
                    raise ValueError("map references disagree; retain disagreement as unresolved")
            if "candidate_pair_support" in reference:
                if not reference.get("candidate_raw_output_sha256") or reference["candidate_raw_output_sha256"] != sb[url].get("raw_output_sha256"):
                    raise ValueError("witness assessment refers to a different output")
                for label in reference["candidate_pair_support"]:
                    if label not in ("supported", "unsupported", "ambiguous"):
                        raise ValueError("invalid pair support label")
                    counts["candidate_pairs_" + label] += 1
            interval = reference.get("acceptable_grade_range")
            if interval is None:
                counts["unresolved_ranges"] += 1
                continue
            if not (isinstance(interval, list) and len(interval) == 2 and
                    all(type(n) is int for n in interval) and 0 <= interval[0] <= interval[1] <= 5):
                raise ValueError("invalid independently specified grade range")
            a_ok = interval[0] <= a <= interval[1] if a is not None else None
            b_ok = interval[0] <= b <= interval[1] if b is not None else None
            if a_ok is not None and b_ok is not None:
                counts["paired_resolved_grades"] += 1
                counts["baseline_in_range"] += a_ok
                counts["candidate_in_range"] += b_ok
                comparison_rows.append({"baseline": a_ok, "candidate": b_ok,
                                        "keyword": reference.get("keyword_id")})
                fields = {k: new[cid].get(k) for k in ("model", "method", "engine", "condition")}
                fields.update({k: reference.get(k) for k in ("style", "readiness_stratum", "source_length_stratum")})
                for field, value in fields.items():
                    group = subgroup[f"{field}:{value}"]
                    group["sources"] += 1
                    group["candidate_outside_range"] += not b_ok
                    group["baseline_outside_range"] += not a_ok
    if refs.keys() - {(cid, s["url"]) for cid, c in new.items() for s in c.get("sources", [])}:
        raise ValueError("references contain cases outside this comparison")
    interval = (cluster_bootstrap(comparison_rows, cluster="keyword",
                statistic=paired_difference("candidate", "baseline"), replicates=replicates, seed=20260930)
                if comparison_rows and all(r["keyword"] is not None for r in comparison_rows) else None)
    repeats = Counter()
    seen_executions = {new_summary["execution_sha256"]}
    all_runs = [(new_summary, new)]
    for directory in repetitions:
        summary, cells = load(directory)
        if summary["execution_sha256"] in seen_executions:
            raise ValueError("repeat reused an execution identity; use separate uncached repetition IDs")
        seen_executions.add(summary["execution_sha256"])
        if config(summary) != config(new_summary) or summary["protocol"] != new_summary["protocol"]:
            raise ValueError("repeat configuration differs")
        if not cells.keys() <= new.keys():
            raise ValueError("repeat includes unknown cells")
        all_runs.append((summary, cells))
    # Pairwise comparisons across the three executions; comparisons are correlated.
    for i, (_, first) in enumerate(all_runs):
        for _, second in all_runs[i + 1:]:
            for cid in first.keys() & second.keys():
                if first[cid].get("status") != "ok" or second[cid].get("status") != "ok":
                    continue
                a, b = comparable(first[cid], second[cid])
                if repeat_kind == "fixed_map" and first[cid].get("map_sha256") != second[cid].get("map_sha256"):
                    raise ValueError("fixed-map repetition changed the map")
                for url in a:
                    x, y = a[url].get("importance"), b[url].get("importance")
                    if x is not None and y is not None:
                        repeats["grade_pairs"] += 1
                        repeats["exact"] += x == y
                        repeats["within_one"] += abs(x-y) <= 1
                        repeats["two_or_more"] += abs(x-y) >= 2
                        repeats["grade4_crossings"] += (x >= 4) != (y >= 4)
                    else:
                        repeats["missing_pairs"] += 1
                tops = []
                for sources in (a, b):
                    grades = [s.get("importance") for s in sources.values()]
                    if any(g is None for g in grades):
                        tops.append(None)
                    else:
                        maximum = max(grades, default=0)
                        tops.append({u for u, s in sources.items() if s["importance"] == maximum} if maximum > 0 else set())
                if all(t is not None for t in tops) and any(tops):
                    repeats["nonempty_top_comparisons"] += 1
                    repeats["same_top_group"] += tops[0] == tops[1]
    hours = [s.get("timing", {}).get("si", {}).get("gpu_hours") for s in (old_summary, new_summary)]
    ratio = (hours[1] / hours[0] if counts["common_complete_cells"] and hours[0] and hours[1] is not None and
             not old_summary.get("fixed_maps_sha256") and not new_summary.get("fixed_maps_sha256") else None)
    return {"scientific_result": False, "reference_is_human_gold": False,
            "baseline_report_sha256": file_hash(Path(baseline) / "cells.jsonl.gz"),
            "candidate_report_sha256": file_hash(Path(candidate) / "cells.jsonl.gz"),
            "reference_sha256": file_hash(references) if references else None, "counts": dict(counts),
            "baseline_range_agreement": rate(counts["baseline_in_range"], counts["paired_resolved_grades"]),
            "candidate_range_agreement": rate(counts["candidate_in_range"], counts["paired_resolved_grades"]),
            "paired_keyword_interval": interval, "subgroups": dict(subgroup),
            "map_fidelity": rate(sum(m["faithful"] for m in maps.values()), len(maps)),
            "essential_map_errors": sum(m["essential_omission"] or m["meaning_reversal"] for m in maps.values()),
            "resolved_pair_support": rate(counts["candidate_pairs_supported"],
                counts["candidate_pairs_supported"] + counts["candidate_pairs_unsupported"]),
            "ambiguous_pair_fraction": rate(counts["candidate_pairs_ambiguous"], sum(
                counts["candidate_pairs_" + label] for label in ("supported", "unsupported", "ambiguous"))),
            "repeat_kind": repeat_kind, "repetition_executions": len(all_runs), "repeats": dict(repeats),
            "warm_gpu_hour_ratio": ratio, "warm_gpu_hour_ratio_ceiling": 3.0,
            "promotion": "requires_semantic_review_provenance_map_fidelity_and_fresh_evaluation",
            "limitations": ["Reference-model agreement is not accuracy.",
                            "Source judgments and pairwise repetitions are clustered, not independent.",
                            "No interval is computed without keyword IDs.",
                            "Compute ratio requires matched cohorts and end-to-end runs including mapping."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--repeat", action="append", type=Path, default=[])
    parser.add_argument("--repeat-kind", choices=("fixed_map", "end_to_end"), default="end_to_end")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.baseline, args.candidate, references=args.references,
                     repetitions=args.repeat, repeat_kind=args.repeat_kind)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
