#!/usr/bin/env python3
"""Audit generated-question coverage and continuity on frozen readiness axis 1.

This is a read-only, post-projection audit.  It does not embed text itself: the
Qwen and aligned Mistral projection artifacts must already exist.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_hf_dataset import (  # noqa: E402
    atomic_json,
    atomic_text,
    read_json,
    read_jsonl,
    sha256_file,
)
from interpretability.pipeline.readiness_prompt_population import (  # noqa: E402
    audit_question_diversity,
)


FORMAT_VERSION = "readiness-axis-1-continuity-audit-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--aligned-projections", required=True)
    parser.add_argument(
        "--validations",
        nargs="*",
        default=(),
        help="optional validation JSONL files; candidates must pass every review",
    )
    parser.add_argument(
        "--tolerance-steps",
        nargs="+",
        type=float,
        default=(0.5, 1.0, 2.0, 3.0),
        help="axis-1 tolerances expressed in target-grid steps",
    )
    parser.add_argument(
        "--primary-tolerance-steps",
        type=float,
        default=0.5,
        help="tolerance used for detailed continuity diagnostics",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_sha() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _quantiles(values) -> dict[str, float | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {key: None for key in ("minimum", "p25", "median", "p75", "p90", "p95", "maximum")}
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _normalize(value: float, low: float, high: float) -> float:
    if not high > low:
        raise ValueError("axis-1 bounds must have positive width")
    return (float(value) - low) / (high - low)


def _histogram_summary(values, *, bin_count: int) -> dict[str, object]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "bin_count": bin_count,
            "occupied_bin_count": 0,
            "occupied_bin_fraction": 0.0,
            "histogram_total_variation_from_uniform": None,
            "axis_1_span": 0.0,
            "outside_unit_interval_count": 0,
            "counts": [0] * bin_count,
        }
    outside = int(np.sum((array < 0.0) | (array > 1.0)))
    clipped = np.clip(array, 0.0, np.nextafter(1.0, 0.0))
    indices = np.floor(clipped * bin_count).astype(int)
    counts = np.bincount(indices, minlength=bin_count)
    probability = counts / counts.sum()
    uniform = np.full(bin_count, 1.0 / bin_count)
    return {
        "count": len(array),
        "bin_count": bin_count,
        "occupied_bin_count": int(np.count_nonzero(counts)),
        "occupied_bin_fraction": float(np.count_nonzero(counts) / bin_count),
        "histogram_total_variation_from_uniform": float(
            0.5 * np.sum(np.abs(probability - uniform))
        ),
        "axis_1_span": float(np.ptp(array)),
        "outside_unit_interval_count": outside,
        "counts": counts.tolist(),
    }


def _accepted_candidate_ids(candidate_ids: set[str], validation_paths) -> set[str]:
    if not validation_paths:
        return set(candidate_ids)
    reviews: dict[str, list[bool]] = defaultdict(list)
    for path in validation_paths:
        for row in read_jsonl(path):
            reviews[str(row["candidate_id"])].append(bool(row["accepted"]))
    unknown = set(reviews) - candidate_ids
    missing = candidate_ids - set(reviews)
    if unknown or missing:
        raise ValueError(
            "validation identities differ from candidates: "
            f"unknown={len(unknown)} missing={len(missing)}"
        )
    return {candidate_id for candidate_id, values in reviews.items() if all(values)}


def _maximum_verified_assignment(pool, target_values, tolerance: float):
    if not pool:
        return ()
    reference = np.asarray([row["reference_axis_1"] for row in pool])
    candidate = np.asarray([row["candidate_axis_1"] for row in pool])
    consensus = np.asarray([row["consensus_axis_1"] for row in pool])
    target = np.asarray(target_values, dtype=np.float64)
    reference_error = np.abs(target[:, None] - reference[None, :])
    candidate_error = np.abs(target[:, None] - candidate[None, :])
    verified = (reference_error <= tolerance) & (candidate_error <= tolerance)
    disagreement = np.abs(reference - candidate)
    base_cost = np.abs(target[:, None] - consensus[None, :]) + 0.10 * disagreement[None, :]
    unmatched_penalty = (float(np.max(base_cost)) + 1.0) * (len(target) + 1)
    costs = np.full((len(target), len(pool) + len(target)), 2.0 * unmatched_penalty)
    costs[:, : len(pool)] = np.where(verified, base_cost, 2.0 * unmatched_penalty)
    for target_index in range(len(target)):
        costs[target_index, len(pool) + target_index] = unmatched_penalty
    target_indices, candidate_indices = linear_sum_assignment(costs)
    return tuple(
        (int(target_index), pool[int(candidate_index)])
        for target_index, candidate_index in zip(target_indices, candidate_indices)
        if candidate_index < len(pool) and verified[target_index, candidate_index]
    )


def audit_axis_1_continuity(
    *,
    plan_dir: str | Path,
    candidate_paths,
    aligned_projection_path: str | Path,
    validation_paths=(),
    tolerance_steps=(0.5, 1.0, 2.0, 3.0),
    primary_tolerance_steps: float = 0.5,
) -> dict[str, object]:
    plan = Path(plan_dir)
    manifest = read_json(plan / "plan_manifest.json")
    if manifest.get("target_design") != "axis-1-linear":
        raise ValueError("continuity audit requires an axis-1-linear plan")
    targets = sorted(read_jsonl(plan / "target_grid.jsonl"), key=lambda row: row["target_index"])
    if len(targets) < 2:
        raise ValueError("axis-1 continuity audit requires at least two targets")
    target_values = np.asarray([row["normalized_axis_1"] for row in targets])
    gaps = np.diff(target_values)
    target_step = float(np.median(gaps))
    if target_step <= 0 or not np.allclose(gaps, target_step, rtol=1e-9, atol=1e-12):
        raise ValueError("axis-1 targets must be strictly increasing and equally spaced")
    steps = sorted(set(float(value) for value in (*tolerance_steps, primary_tolerance_steps)))
    if not steps or any(value <= 0 for value in steps):
        raise ValueError("tolerance steps must be positive")

    candidates = [row for path in candidate_paths for row in read_jsonl(path)]
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    if not candidates or len(candidate_by_id) != len(candidates):
        raise ValueError("candidates must be nonempty and uniquely identified")
    projection_path = Path(aligned_projection_path)
    if not projection_path.is_file():
        raise FileNotFoundError(
            f"aligned projections do not exist: {projection_path}; run both LLM2Vec "
            "projections and compare-projections first"
        )
    aligned = {str(row["candidate_id"]): row for row in read_jsonl(projection_path)}
    if set(aligned) != set(candidate_by_id):
        raise ValueError(
            "aligned projection identities differ from candidates: "
            f"projected={len(aligned)} candidates={len(candidate_by_id)}"
        )
    accepted_ids = _accepted_candidate_ids(set(candidate_by_id), validation_paths)
    bounds = read_json(plan / "subspace_bounds.json")
    low = float(bounds["axis_1_low"])
    high = float(bounds["axis_1_high"])

    measured = []
    by_keyword: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate_id in sorted(accepted_ids):
        candidate_row = candidate_by_id[candidate_id]
        projection_row = aligned[candidate_id]
        reference_axis_1 = _normalize(projection_row["reference_raw_axis_1"], low, high)
        candidate_axis_1 = _normalize(
            projection_row["candidate_aligned_raw_axis_1"], low, high
        )
        consensus_axis_1 = (reference_axis_1 + candidate_axis_1) / 2.0
        target_axis_1 = float(candidate_row["target_normalized_axis_1"])
        row = {
            "candidate_id": candidate_id,
            "keyword_id": str(candidate_row["keyword_id"]),
            "target_id": str(candidate_row["target_id"]),
            "target_index": int(candidate_row["target_index"]),
            "target_axis_1": target_axis_1,
            "reference_axis_1": reference_axis_1,
            "candidate_axis_1": candidate_axis_1,
            "consensus_axis_1": consensus_axis_1,
            "reference_intended_error": abs(reference_axis_1 - target_axis_1),
            "candidate_intended_error": abs(candidate_axis_1 - target_axis_1),
            "consensus_intended_error": abs(consensus_axis_1 - target_axis_1),
            "cross_view_axis_1_disagreement": abs(reference_axis_1 - candidate_axis_1),
        }
        measured.append(row)
        by_keyword[row["keyword_id"]].append(row)

    planned_keyword_count = int(manifest["keyword_count"])
    planned_pair_count = planned_keyword_count * len(targets)
    threshold_results = []
    primary_assignments = {}
    for step_count in steps:
        tolerance = step_count * target_step
        reference_hits = sum(row["reference_intended_error"] <= tolerance for row in measured)
        candidate_hits = sum(row["candidate_intended_error"] <= tolerance for row in measured)
        consensus_hits = sum(row["consensus_intended_error"] <= tolerance for row in measured)
        dual_hits = sum(
            row["reference_intended_error"] <= tolerance
            and row["candidate_intended_error"] <= tolerance
            for row in measured
        )
        assignments = {
            keyword_id: _maximum_verified_assignment(pool, target_values, tolerance)
            for keyword_id, pool in by_keyword.items()
        }
        globally_matched = sum(len(rows) for rows in assignments.values())
        fully_covered = sum(len(rows) == len(targets) for rows in assignments.values())
        threshold_results.append(
            {
                "tolerance_steps": step_count,
                "normalized_tolerance": tolerance,
                "reference_intended_hit_count": reference_hits,
                "candidate_intended_hit_count": candidate_hits,
                "consensus_intended_hit_count": consensus_hits,
                "dual_view_intended_hit_count": dual_hits,
                "dual_view_intended_hit_fraction": dual_hits / len(measured) if measured else 0.0,
                "globally_matchable_target_count": globally_matched,
                "globally_matchable_planned_fraction": globally_matched / planned_pair_count,
                "fully_coverable_keyword_count": fully_covered,
                "fully_coverable_keyword_fraction": fully_covered / planned_keyword_count,
            }
        )
        if np.isclose(step_count, primary_tolerance_steps):
            primary_assignments = assignments

    matched_rows = [row for rows in primary_assignments.values() for _, row in rows]
    complete_keyword_spans = []
    complete_keyword_median_gaps = []
    complete_keyword_maximum_gaps = []
    for assignments in primary_assignments.values():
        if len(assignments) != len(targets):
            continue
        values = np.sort([row["consensus_axis_1"] for _, row in assignments])
        observed_gaps = np.diff(values)
        complete_keyword_spans.append(float(np.ptp(values)))
        complete_keyword_median_gaps.append(float(np.median(observed_gaps)))
        complete_keyword_maximum_gaps.append(float(np.max(observed_gaps)))

    diversity = audit_question_diversity(
        [candidate_by_id[candidate_id] for candidate_id in sorted(accepted_ids)]
    )
    return {
        "format_version": FORMAT_VERSION,
        "created_at": _now(),
        "git_commit_sha": _git_sha(),
        "slurm": {
            key.lower(): os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_NODELIST",
            )
            if key in os.environ
        },
        "inputs": {
            "plan_manifest": _file_identity(plan / "plan_manifest.json"),
            "subspace_bounds": _file_identity(plan / "subspace_bounds.json"),
            "target_grid": _file_identity(plan / "target_grid.jsonl"),
            "candidates": [_file_identity(path) for path in candidate_paths],
            "aligned_projections": _file_identity(projection_path),
            "validations": [_file_identity(path) for path in validation_paths],
        },
        "target_design": "axis-1-linear",
        "axis_1_target_count": len(targets),
        "axis_1_target_step": target_step,
        "primary_tolerance_steps": primary_tolerance_steps,
        "primary_normalized_tolerance": primary_tolerance_steps * target_step,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(measured),
        "validation_filter_applied": bool(validation_paths),
        "represented_keyword_count": len(by_keyword),
        "planned_keyword_count": planned_keyword_count,
        "planned_target_pair_count": planned_pair_count,
        "intended_target_error": {
            "reference_qwen": _quantiles(row["reference_intended_error"] for row in measured),
            "candidate_aligned_mistral": _quantiles(row["candidate_intended_error"] for row in measured),
            "consensus": _quantiles(row["consensus_intended_error"] for row in measured),
            "cross_view_axis_1_disagreement": _quantiles(
                row["cross_view_axis_1_disagreement"] for row in measured
            ),
        },
        "all_eligible_axis_1_distribution": {
            "reference_qwen": _histogram_summary(
                (row["reference_axis_1"] for row in measured), bin_count=len(targets)
            ),
            "candidate_aligned_mistral": _histogram_summary(
                (row["candidate_axis_1"] for row in measured), bin_count=len(targets)
            ),
            "consensus": _histogram_summary(
                (row["consensus_axis_1"] for row in measured), bin_count=len(targets)
            ),
        },
        "tolerance_sweep": threshold_results,
        "primary_global_assignment": {
            "matched_target_count": len(matched_rows),
            "matched_planned_fraction": len(matched_rows) / planned_pair_count,
            "fully_covered_keyword_count": len(complete_keyword_spans),
            "fully_covered_keyword_fraction": len(complete_keyword_spans)
            / planned_keyword_count,
            "matched_consensus_distribution": _histogram_summary(
                (row["consensus_axis_1"] for row in matched_rows),
                bin_count=len(targets),
            ),
            "fully_covered_keyword_axis_1_span": _quantiles(complete_keyword_spans),
            "fully_covered_keyword_median_gap": _quantiles(
                complete_keyword_median_gaps
            ),
            "fully_covered_keyword_maximum_gap": _quantiles(
                complete_keyword_maximum_gaps
            ),
        },
        "wording_diversity": diversity,
        "interpretation_guard": (
            "This audit describes generated prompt semantics and wording. It does not "
            "define the randomized experimental policy variable B."
        ),
    }


def _report(summary: dict[str, object]) -> str:
    primary = summary["primary_global_assignment"]
    lines = [
        "# Axis-1 continuity audit",
        "",
        f"- Candidates: {summary['candidate_count']}",
        f"- Eligible after validation filter: {summary['eligible_candidate_count']}",
        f"- Target step: {summary['axis_1_target_step']:.8f}",
        f"- Primary tolerance: {summary['primary_normalized_tolerance']:.8f} "
        f"({summary['primary_tolerance_steps']:.2f} target steps)",
        f"- Globally matchable targets: {primary['matched_target_count']} / "
        f"{summary['planned_target_pair_count']}",
        f"- Fully coverable keywords: {primary['fully_covered_keyword_count']} / "
        f"{summary['planned_keyword_count']}",
        f"- Delexicalized unique fraction: "
        f"{summary['wording_diversity']['delexicalized_unique_fraction']:.4f}",
        "",
        "## Tolerance sweep",
        "",
        "| Steps | Tolerance | Dual intended hits | Global matches | Full keywords |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tolerance_sweep"]:
        lines.append(
            f"| {row['tolerance_steps']:.2f} | {row['normalized_tolerance']:.8f} | "
            f"{row['dual_view_intended_hit_count']} | "
            f"{row['globally_matchable_target_count']} | "
            f"{row['fully_coverable_keyword_count']} |"
        )
    lines.extend(("", summary["interpretation_guard"], ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    summary = audit_axis_1_continuity(
        plan_dir=args.plan_dir,
        candidate_paths=args.candidates,
        aligned_projection_path=args.aligned_projections,
        validation_paths=args.validations,
        tolerance_steps=args.tolerance_steps,
        primary_tolerance_steps=args.primary_tolerance_steps,
    )
    output.mkdir(parents=True)
    atomic_json(output / "axis_1_continuity_audit.json", summary)
    atomic_text(output / "axis_1_continuity_report.md", _report(summary))
    primary = summary["primary_global_assignment"]
    print(
        f"candidates={summary['candidate_count']} "
        f"eligible={summary['eligible_candidate_count']} "
        f"half_step_matches={primary['matched_target_count']} "
        f"full_keywords={primary['fully_covered_keyword_count']}"
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
