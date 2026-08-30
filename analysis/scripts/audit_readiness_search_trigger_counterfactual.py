#!/usr/bin/env python3
"""Audit stored readiness candidates under a relaxed search-trigger contract.

This is a versioned counterfactual audit. It never mutates the historical Gold
selection. The experimental policy variable B remains defined by randomized
prompt generation; readiness embeddings only diagnose the generated text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_hf_dataset import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    atomic_text,
    read_json,
    read_jsonl,
)
from interpretability.pipeline.readiness_prompt_population import (  # noqa: E402
    ReadinessPromptTarget,
    ReadinessQuestionCandidate,
    ReadinessSubspaceBounds,
    search_review_passes_contract,
    select_spatially_matched_questions,
)
from scripts.build_readiness_prompt_population import (  # noqa: E402
    _aligned_projection_rows,
    _normalize_coordinate,
    _read_plan_keywords,
    _read_plan_targets,
)


FORMAT_VERSION = "readiness-search-trigger-counterfactual-v1"


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    accepted_candidate_ids: frozenset[str]
    distance_tolerance: float
    require_template_uniqueness: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        required=True,
        help=(
            "A merged checkpoint or completed round containing candidates, "
            "validation, and Qwen/Mistral projections. Repeat to audit the "
            "deduplicated union of independent section branches."
        ),
    )
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--robustness-battery", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--historical-distance-tolerance",
        type=float,
        default=None,
        help="Override the stored Gold tolerance (fallback: 0.017).",
    )
    parser.add_argument(
        "--relaxed-distance-tolerance",
        type=float,
        default=0.035,
    )
    parser.add_argument("--disagreement-weight", type=float, default=0.10)
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _checkpoint_inputs(root: Path) -> dict[str, object]:
    merged = root / "merged"
    if (merged / "candidates.jsonl").is_file():
        base = merged
        candidate_paths = (merged / "candidates.jsonl",)
    else:
        base = root
        candidate_list = root / "candidate-files.txt"
        if candidate_list.is_file():
            candidate_paths = []
            for line in candidate_list.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                path = Path(line.strip())
                candidate_paths.append(
                    path.resolve() if path.is_absolute() else (root / path).resolve()
                )
            candidate_paths = tuple(candidate_paths)
        elif (root / "candidates.jsonl").is_file():
            candidate_paths = (root / "candidates.jsonl",)
        else:
            raise ValueError(f"cannot locate checkpoint candidates under {root}")
    validation = base / "validation.jsonl"
    reference = base / "projections" / "qwen"
    candidate = base / "projections" / "mistral"
    required = (*candidate_paths, validation, reference, candidate)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"checkpoint inputs are missing: {missing}")
    return {
        "candidate_paths": candidate_paths,
        "validation": validation,
        "reference_projections": reference,
        "candidate_projections": candidate,
    }


def _historical_tolerance(root: Path, override: float | None) -> float:
    if override is not None:
        value = override
    else:
        diagnostics = root / "strict-selection" / "spatial_coverage_diagnostics.json"
        stored = read_json(diagnostics) if diagnostics.is_file() else {}
        value = float(stored.get("distance_tolerance", 0.017))
    if value < 0:
        raise ValueError("historical distance tolerance must be nonnegative")
    return float(value)


def _candidate_rows(paths: Sequence[Path]) -> tuple[ReadinessQuestionCandidate, ...]:
    rows = tuple(
        ReadinessQuestionCandidate(**row)
        for path in paths
        for row in read_jsonl(path)
    )
    if not rows or len({row.candidate_id for row in rows}) != len(rows):
        raise ValueError("candidate inputs must be nonempty and globally unique")
    return rows


def _review_index(
    path: Path,
    candidate_ids: set[str],
) -> dict[str, dict[str, object]]:
    rows = {str(row["candidate_id"]): row for row in read_jsonl(path)}
    if set(rows) != candidate_ids:
        raise ValueError("validation does not cover the exact candidate set")
    recomputed = {
        candidate_id
        for candidate_id, row in rows.items()
        if search_review_passes_contract(row, contract="question-v1")
    }
    stored = {
        candidate_id
        for candidate_id, row in rows.items()
        if bool(row["accepted"])
    }
    if stored != recomputed:
        raise ValueError(
            "stored Gold acceptance differs from the question-v1 contract"
        )
    return rows


def _coordinate_index(
    reference: Path,
    candidate: Path,
    battery: Path,
    bounds: ReadinessSubspaceBounds,
    candidate_ids: set[str],
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    rows, identities, agreement = _aligned_projection_rows(
        reference,
        candidate,
        battery,
    )
    if {str(row["candidate_id"]) for row in rows} != candidate_ids:
        raise ValueError("dual-view projections do not cover the exact candidate set")
    coordinates = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        reference_axis_1 = _normalize_coordinate(
            row["reference_raw_axis_1"], bounds.axis_1_low, bounds.axis_1_high
        )
        reference_axis_2 = _normalize_coordinate(
            row["reference_raw_axis_2"], bounds.axis_2_low, bounds.axis_2_high
        )
        aligned_axis_1 = _normalize_coordinate(
            row["candidate_aligned_raw_axis_1"],
            bounds.axis_1_low,
            bounds.axis_1_high,
        )
        aligned_axis_2 = _normalize_coordinate(
            row["candidate_aligned_raw_axis_2"],
            bounds.axis_2_low,
            bounds.axis_2_high,
        )
        coordinates[candidate_id] = {
            "reference_normalized_axis_1": reference_axis_1,
            "reference_normalized_axis_2": reference_axis_2,
            "candidate_aligned_normalized_axis_1": aligned_axis_1,
            "candidate_aligned_normalized_axis_2": aligned_axis_2,
            "consensus_normalized_axis_1": (reference_axis_1 + aligned_axis_1) / 2,
            "consensus_normalized_axis_2": (reference_axis_2 + aligned_axis_2) / 2,
            "cross_embedding_disagreement": float(
                np.hypot(
                    reference_axis_1 - aligned_axis_1,
                    reference_axis_2 - aligned_axis_2,
                )
            ),
        }
    return coordinates, {"identities": identities, "agreement": agreement}


def evaluate_scenarios(
    candidates: Sequence[ReadinessQuestionCandidate],
    targets: (
        Sequence[ReadinessPromptTarget]
        | Mapping[str, Sequence[ReadinessPromptTarget]]
    ),
    coordinates: Mapping[str, Mapping[str, float]],
    scenarios: Sequence[Scenario],
    *,
    target_design: str,
    planned_keywords: Sequence[tuple[str, str]],
    disagreement_weight: float,
) -> dict[str, dict[str, object]]:
    """Run deterministic strict dual-view selection for each counterfactual."""

    results = {}
    for scenario in scenarios:
        selected, diagnostics = select_spatially_matched_questions(
            candidates,
            targets,
            coordinates,
            accepted_candidate_ids=set(scenario.accepted_candidate_ids),
            disagreement_weight=disagreement_weight,
            distance_tolerance=scenario.distance_tolerance,
            target_design=target_design,
            require_both_views_within_tolerance=True,
            require_delexicalized_template_uniqueness=(
                scenario.require_template_uniqueness
            ),
            allow_missing_keyword_for_template=True,
            planned_keywords=planned_keywords,
        )
        results[scenario.name] = {
            "scenario": scenario,
            "selected": selected,
            "diagnostics": diagnostics,
        }
    return results


def _targets_by_keyword(
    targets: (
        Sequence[ReadinessPromptTarget]
        | Mapping[str, Sequence[ReadinessPromptTarget]]
    ),
    keywords: Sequence[tuple[str, str]],
) -> dict[str, tuple[ReadinessPromptTarget, ...]]:
    if isinstance(targets, Mapping):
        return {key: tuple(value) for key, value in targets.items()}
    return {keyword_id: tuple(targets) for keyword_id, _ in keywords}


def _npy_header(stream) -> tuple[tuple[int, ...], str]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported npy version inside npz: {version}")
    return tuple(int(value) for value in shape), str(dtype)


def _embedding_artifacts(
    projection_roots: Sequence[Path],
    candidate_count: int,
) -> list[dict[str, object]]:
    artifacts = []
    for root in projection_roots:
        paths = sorted(root.glob("*.npz"))
        if not paths:
            raise ValueError(f"no raw embedding npz found in {root}")
        for path in paths:
            arrays = {}
            with zipfile.ZipFile(path) as archive:
                for name in sorted(archive.namelist()):
                    if not name.endswith(".npy"):
                        continue
                    with archive.open(name) as stream:
                        shape, dtype = _npy_header(stream)
                    arrays[Path(name).stem] = {"shape": list(shape), "dtype": dtype}
            candidate_sized_arrays = [
                name
                for name, metadata in arrays.items()
                if metadata["shape"]
                and int(metadata["shape"][0]) == candidate_count
            ]
            if not candidate_sized_arrays:
                raise ValueError(
                    f"embedding archive does not expose {candidate_count} rows: {path}"
                )
            artifacts.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "arrays": arrays,
                    "candidate_sized_arrays": candidate_sized_arrays,
                }
            )
    return artifacts


def _merge_population(
    roots: Sequence[Path],
    battery: Path,
    bounds: ReadinessSubspaceBounds,
) -> tuple[
    tuple[ReadinessQuestionCandidate, ...],
    dict[str, dict[str, object]],
    dict[str, dict[str, float]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    candidate_index: dict[str, ReadinessQuestionCandidate] = {}
    review_index: dict[str, dict[str, object]] = {}
    coordinate_index: dict[str, dict[str, float]] = {}
    projection_summaries = []
    embedding_artifacts = []
    for root in roots:
        inputs = _checkpoint_inputs(root)
        local_candidates = _candidate_rows(inputs["candidate_paths"])
        local_ids = {row.candidate_id for row in local_candidates}
        local_reviews = _review_index(inputs["validation"], local_ids)
        local_coordinates, projection_summary = _coordinate_index(
            inputs["reference_projections"],
            inputs["candidate_projections"],
            battery,
            bounds,
            local_ids,
        )
        embedding_artifacts.extend(
            _embedding_artifacts(
                (
                    inputs["reference_projections"],
                    inputs["candidate_projections"],
                ),
                len(local_candidates),
            )
        )
        projection_summaries.append(
            {
                "checkpoint_root": str(root),
                "candidate_count": len(local_candidates),
                **projection_summary,
            }
        )
        for candidate in local_candidates:
            previous = candidate_index.setdefault(candidate.candidate_id, candidate)
            if previous != candidate:
                raise ValueError(
                    f"candidate identity differs across roots: {candidate.candidate_id}"
                )
        for candidate_id, review in local_reviews.items():
            previous = review_index.setdefault(candidate_id, review)
            if previous != review:
                raise ValueError(
                    f"validation differs across roots: {candidate_id}"
                )
        for candidate_id, coordinate in local_coordinates.items():
            previous = coordinate_index.setdefault(candidate_id, coordinate)
            if any(
                not np.isclose(
                    previous[name], value, rtol=0.0, atol=1e-12
                )
                for name, value in coordinate.items()
            ):
                raise ValueError(
                    f"aligned coordinates differ across roots: {candidate_id}"
                )
    candidate_ids = set(candidate_index)
    if set(review_index) != candidate_ids or set(coordinate_index) != candidate_ids:
        raise AssertionError("merged population identity coverage is inconsistent")
    return (
        tuple(candidate_index[key] for key in sorted(candidate_index)),
        review_index,
        coordinate_index,
        projection_summaries,
        embedding_artifacts,
    )


def _candidate_diagnostics(
    candidates: Sequence[ReadinessQuestionCandidate],
    reviews: Mapping[str, Mapping[str, object]],
    coordinates: Mapping[str, Mapping[str, float]],
    targets_by_keyword: Mapping[str, Sequence[ReadinessPromptTarget]],
    *,
    gold_ids: set[str],
    relaxed_ids: set[str],
    historical_tolerance: float,
    relaxed_tolerance: float,
    availability: dict[tuple[str, str], dict[str, int]],
    minimum_distances: list[float],
) -> Iterable[dict[str, object]]:
    grouped: dict[str, list[ReadinessQuestionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.keyword_id, []).append(candidate)
    for keyword_id in sorted(targets_by_keyword):
        keyword_candidates = sorted(
            grouped.get(keyword_id, ()), key=lambda row: row.candidate_id
        )
        keyword_targets = tuple(targets_by_keyword[keyword_id])
        target_values = np.asarray(
            [target.normalized_axis_1 for target in keyword_targets],
            dtype=np.float64,
        )
        if not keyword_candidates:
            for target in keyword_targets:
                availability[(keyword_id, target.target_id)] = {
                    "gold_at_relaxed_tolerance": 0,
                    "search_trigger_v2_at_historical_tolerance": 0,
                    "search_trigger_v2_at_relaxed_tolerance": 0,
                }
            continue
        reference = np.asarray(
            [
                coordinates[row.candidate_id]["reference_normalized_axis_1"]
                for row in keyword_candidates
            ]
        )
        aligned = np.asarray(
            [
                coordinates[row.candidate_id][
                    "candidate_aligned_normalized_axis_1"
                ]
                for row in keyword_candidates
            ]
        )
        robust_distances = np.maximum(
            np.abs(reference[:, None] - target_values[None, :]),
            np.abs(aligned[:, None] - target_values[None, :]),
        )
        gold_mask = np.asarray(
            [row.candidate_id in gold_ids for row in keyword_candidates]
        )
        relaxed_mask = np.asarray(
            [row.candidate_id in relaxed_ids for row in keyword_candidates]
        )
        for target_index, target in enumerate(keyword_targets):
            distances = robust_distances[:, target_index]
            availability[(keyword_id, target.target_id)] = {
                "gold_at_relaxed_tolerance": int(
                    np.count_nonzero(gold_mask & (distances <= relaxed_tolerance))
                ),
                "search_trigger_v2_at_historical_tolerance": int(
                    np.count_nonzero(
                        relaxed_mask & (distances <= historical_tolerance)
                    )
                ),
                "search_trigger_v2_at_relaxed_tolerance": int(
                    np.count_nonzero(relaxed_mask & (distances <= relaxed_tolerance))
                ),
            }
        nearest_indices = np.argmin(robust_distances, axis=1)
        nearest_distances = robust_distances[
            np.arange(len(keyword_candidates)), nearest_indices
        ]
        eligible_counts = np.count_nonzero(
            robust_distances <= relaxed_tolerance, axis=1
        )
        for row_index, candidate in enumerate(keyword_candidates):
            review = reviews[candidate.candidate_id]
            coordinate = coordinates[candidate.candidate_id]
            minimum_distance = float(nearest_distances[row_index])
            minimum_distances.append(minimum_distance)
            yield {
                "candidate_id": candidate.candidate_id,
                "task_id": candidate.task_id,
                "keyword_id": candidate.keyword_id,
                "keyword": candidate.keyword,
                "generated_target_id": candidate.target_id,
                "question": candidate.question,
                "generator_id": candidate.generator_id,
                "round_index": candidate.round_index,
                "gold_question_v1_accepted": candidate.candidate_id in gold_ids,
                "search_trigger_v2_accepted": candidate.candidate_id in relaxed_ids,
                "recovered_by_search_trigger_v2": (
                    candidate.candidate_id in relaxed_ids
                    and candidate.candidate_id not in gold_ids
                ),
                "exact_keyword_present": bool(review["exact_keyword_present"]),
                "single_question": bool(review["single_question"]),
                "topic_relevant": bool(review["topic_relevant"]),
                "search_intent": bool(review["search_intent"]),
                "web_answerable": bool(review["web_answerable"]),
                "standalone": bool(review["standalone"]),
                "natural_language": bool(review["natural_language"]),
                "relevance_score_1_5": int(review["relevance_score_1_5"]),
                "reference_normalized_axis_1": coordinate[
                    "reference_normalized_axis_1"
                ],
                "candidate_aligned_normalized_axis_1": coordinate[
                    "candidate_aligned_normalized_axis_1"
                ],
                "consensus_normalized_axis_1": coordinate[
                    "consensus_normalized_axis_1"
                ],
                "cross_embedding_disagreement": coordinate[
                    "cross_embedding_disagreement"
                ],
                "nearest_planned_target_id": keyword_targets[
                    int(nearest_indices[row_index])
                ].target_id,
                "minimum_dual_view_target_distance": minimum_distance,
                "eligible_target_count_at_relaxed_tolerance": int(
                    eligible_counts[row_index]
                ),
            }


def _gap_attribution(
    targets_by_keyword: Mapping[str, Sequence[ReadinessPromptTarget]],
    scenario_cells: Mapping[str, set[tuple[str, str]]],
    availability: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[dict[str, object]]:
    rows = []
    for keyword_id in sorted(targets_by_keyword):
        for target in targets_by_keyword[keyword_id]:
            cell = (keyword_id, target.target_id)
            flags = {name: cell in cells for name, cells in scenario_cells.items()}
            counts = availability[cell]
            if flags["question_v1_historical"]:
                stage = "historical_gold_selection"
            elif flags["search_trigger_v2_historical"]:
                stage = "recovered_by_validation_contract"
            elif flags["question_v1_relaxed_tolerance"]:
                stage = "recovered_by_tolerance"
            elif flags["search_trigger_v2_relaxed_tolerance"]:
                stage = "recovered_by_combined_contract_and_tolerance"
            elif flags["search_trigger_v2_relaxed_no_template"]:
                stage = "blocked_by_global_template_uniqueness"
            elif counts["search_trigger_v2_at_relaxed_tolerance"] == 0:
                stage = "no_dual_view_candidate_within_relaxed_tolerance"
            else:
                stage = "one_to_one_assignment_conflict"
            rows.append(
                {
                    "keyword_id": keyword_id,
                    "target_id": target.target_id,
                    "target_index": target.target_index,
                    "target_normalized_axis_1": target.normalized_axis_1,
                    "counterfactual_recovery_stage": stage,
                    **flags,
                    **counts,
                }
            )
    return rows


def _report(summary: Mapping[str, object]) -> str:
    scenarios = summary["scenarios"]
    lines = [
        "# Readiness search-trigger counterfactual",
        "",
        f"- Input checkpoint roots: {len(summary['checkpoint_roots'])}",
        f"- Stored candidates: {summary['candidate_count']}",
        f"- Gold question-v1 accepted: {summary['gold_accepted_count']}",
        f"- Search-trigger-v2 accepted: {summary['relaxed_accepted_count']}",
        f"- Reviews recovered by v2: {summary['validation_recovered_count']}",
        f"- Historical tolerance: {summary['historical_distance_tolerance']:.6f}",
        f"- Relaxed base tolerance: {summary['relaxed_distance_tolerance']:.6f}",
        "",
        "| Scenario | Selected | Missing |",
        "|---|---:|---:|",
    ]
    for name in (
        "question_v1_historical",
        "search_trigger_v2_historical",
        "question_v1_relaxed_tolerance",
        "search_trigger_v2_relaxed_tolerance",
        "search_trigger_v2_relaxed_no_template",
    ):
        row = scenarios[name]
        lines.append(f"| {name} | {row['selected_count']} | {row['missing_count']} |")
    lines.extend(
        [
            "",
            "## Remaining-gap attribution at search-trigger-v2 / relaxed tolerance",
            "",
        ]
    )
    for reason, count in sorted(summary["remaining_gap_reasons"].items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "The v2 contract ignores exact-keyword, single-question, and standalone "
            "review gates. It still requires topical relevance, search intent, web "
            "answerability, natural language, and relevance >= 4/5.",
            "",
            "Every scenario is recomputed over the deduplicated union of the supplied "
            "checkpoint roots. With one root, the question-v1 historical scenario is "
            "required to reproduce its durable selected count when a round summary exists.",
            "",
            "Existing candidate files were produced after the historical generator-side "
            "exact-keyword and question-form checks. Therefore this audit cannot recover "
            "paraphrased-keyword or non-question generations that were never persisted.",
            "",
            "Raw Qwen and Mistral embedding archives are reused in place and inventoried; "
            "they are not copied. Candidate projection identity is checked in both views.",
            "",
            "These diagnostics describe prompt-space coverage. Embeddings do not define "
            "the randomized experimental policy variable B.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    roots = tuple(Path(value).resolve() for value in args.checkpoint_root)
    if len(set(roots)) != len(roots):
        raise ValueError("checkpoint roots must be unique")
    plan = Path(args.plan_dir).resolve()
    battery = Path(args.robustness_battery).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite audit output: {output}")
    if args.relaxed_distance_tolerance < 0:
        raise ValueError("relaxed distance tolerance must be nonnegative")
    historical_tolerances = {
        _historical_tolerance(root, args.historical_distance_tolerance)
        for root in roots
    }
    if len(historical_tolerances) != 1:
        raise ValueError("checkpoint roots use different historical tolerances")
    historical_tolerance = historical_tolerances.pop()
    bounds = ReadinessSubspaceBounds(**read_json(plan / "subspace_bounds.json"))
    (
        candidates,
        reviews,
        coordinates,
        projection_summaries,
        embedding_artifacts,
    ) = _merge_population(
        roots,
        battery,
        bounds,
    )
    candidate_keywords = sorted(
        {(row.keyword_id, row.keyword) for row in candidates}
    )
    keywords = _read_plan_keywords(plan, candidate_keywords)
    targets, target_design = _read_plan_targets(plan, keywords)
    if target_design not in {"axis-1-linear", "axis-1-quantized-uniform"}:
        raise ValueError("this counterfactual currently requires an axis-1 target plan")
    targets_by_keyword = _targets_by_keyword(targets, keywords)
    target_count = sum(len(values) for values in targets_by_keyword.values())

    gold_ids = {
        candidate_id
        for candidate_id, review in reviews.items()
        if search_review_passes_contract(review, contract="question-v1")
    }
    relaxed_ids = {
        candidate_id
        for candidate_id, review in reviews.items()
        if search_review_passes_contract(review, contract="search-trigger-v2")
    }
    scenarios = (
        Scenario(
            "question_v1_historical",
            frozenset(gold_ids),
            historical_tolerance,
            True,
        ),
        Scenario(
            "search_trigger_v2_historical",
            frozenset(relaxed_ids),
            historical_tolerance,
            True,
        ),
        Scenario(
            "question_v1_relaxed_tolerance",
            frozenset(gold_ids),
            args.relaxed_distance_tolerance,
            True,
        ),
        Scenario(
            "search_trigger_v2_relaxed_tolerance",
            frozenset(relaxed_ids),
            args.relaxed_distance_tolerance,
            True,
        ),
        Scenario(
            "search_trigger_v2_relaxed_no_template",
            frozenset(relaxed_ids),
            args.relaxed_distance_tolerance,
            False,
        ),
    )
    results = evaluate_scenarios(
        candidates,
        targets,
        coordinates,
        scenarios,
        target_design=target_design,
        planned_keywords=keywords,
        disagreement_weight=args.disagreement_weight,
    )
    gold_reproduction = None
    if len(roots) == 1:
        historical_summary = roots[0] / "verified_round_summary.json"
        if historical_summary.is_file():
            expected = int(read_json(historical_summary)["selected_count"])
            observed = len(results["question_v1_historical"]["selected"])
            if observed != expected:
                raise ValueError(
                    "question-v1 historical selection was not reproduced: "
                    f"expected {expected}, observed {observed}"
                )
            gold_reproduction = {
                "expected_selected_count": expected,
                "observed_selected_count": observed,
                "passed": True,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        scenario_root = staging / "scenarios"
        scenario_root.mkdir()
        scenario_cells = {}
        scenario_summaries = {}
        for name, result in results.items():
            selected = result["selected"]
            scenario_dir = scenario_root / name
            scenario_dir.mkdir()
            atomic_jsonl(
                scenario_dir / "selected.jsonl",
                (asdict(row) for row in selected),
            )
            scenario_cells[name] = {
                (row.keyword_id, row.target_id) for row in selected
            }
            scenario_summaries[name] = {
                "selected_count": len(selected),
                "missing_count": target_count - len(selected),
                "accepted_candidate_count": len(
                    result["scenario"].accepted_candidate_ids
                ),
                "distance_tolerance": result["scenario"].distance_tolerance,
                "require_template_uniqueness": result[
                    "scenario"
                ].require_template_uniqueness,
                "selection_diagnostics": result["diagnostics"],
            }
            atomic_json(scenario_dir / "summary.json", scenario_summaries[name])

        availability: dict[tuple[str, str], dict[str, int]] = {}
        minimum_distances: list[float] = []
        atomic_jsonl(
            staging / "candidate_projection_diagnostics.jsonl",
            _candidate_diagnostics(
                candidates,
                reviews,
                coordinates,
                targets_by_keyword,
                gold_ids=gold_ids,
                relaxed_ids=relaxed_ids,
                historical_tolerance=historical_tolerance,
                relaxed_tolerance=args.relaxed_distance_tolerance,
                availability=availability,
                minimum_distances=minimum_distances,
            ),
        )
        gaps = _gap_attribution(targets_by_keyword, scenario_cells, availability)
        atomic_jsonl(staging / "target_gap_attribution.jsonl", gaps)
        remaining_reasons = Counter(
            row["counterfactual_recovery_stage"]
            for row in gaps
            if not row["search_trigger_v2_relaxed_tolerance"]
        )
        gate_failures = Counter()
        for review in reviews.values():
            for field in (
                "exact_keyword_present",
                "single_question",
                "topic_relevant",
                "search_intent",
                "web_answerable",
                "standalone",
                "natural_language",
            ):
                if not bool(review[field]):
                    gate_failures[field] += 1
            if int(review["relevance_score_1_5"]) < 4:
                gate_failures["relevance_score_below_4"] += 1
        distance_array = np.asarray(minimum_distances, dtype=np.float64)
        summary = {
            "format_version": FORMAT_VERSION,
            "created_at": _now(),
            "checkpoint_roots": [str(root) for root in roots],
            "plan_dir": str(plan),
            "robustness_battery": str(battery),
            "candidate_count": len(candidates),
            "target_count": target_count,
            "gold_accepted_count": len(gold_ids),
            "relaxed_accepted_count": len(relaxed_ids),
            "validation_recovered_count": len(relaxed_ids - gold_ids),
            "historical_distance_tolerance": historical_tolerance,
            "relaxed_distance_tolerance": args.relaxed_distance_tolerance,
            "disagreement_weight": args.disagreement_weight,
            "validation_gate_failure_counts": dict(sorted(gate_failures.items())),
            "persisted_generator_contract": {
                "all_candidates_contain_exact_keyword": (
                    gate_failures["exact_keyword_present"] == 0
                ),
                "all_candidates_are_single_questions": (
                    gate_failures["single_question"] == 0
                ),
            },
            "minimum_dual_view_target_distance_quantiles": {
                str(quantile): float(np.quantile(distance_array, quantile))
                for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
            },
            "scenarios": scenario_summaries,
            "remaining_gap_reasons": dict(sorted(remaining_reasons.items())),
            "projection_summaries": projection_summaries,
            "raw_embedding_artifacts": embedding_artifacts,
            "single_root_gold_reproduction": gold_reproduction,
            "contract_guard": (
                "search-trigger-v2 is a counterfactual diagnostic and does not "
                "replace the historical Gold question-v1 selection"
            ),
            "scientific_guard": (
                "Readiness embeddings diagnose generated prompt text; they do "
                "not define randomized policy B"
            ),
        }
        atomic_json(staging / "counterfactual_summary.json", summary)
        atomic_json(staging / "embedding_artifacts.json", embedding_artifacts)
        atomic_text(staging / "counterfactual_report.md", _report(summary))
        shutil.move(str(staging), str(output))

    print(f"candidates={len(candidates)}")
    print(f"gold_accepted={len(gold_ids)}")
    print(f"search_trigger_v2_accepted={len(relaxed_ids)}")
    print(
        "search_trigger_v2_selected_0.035="
        f"{scenario_summaries['search_trigger_v2_relaxed_tolerance']['selected_count']}"
    )
    print(
        "remaining_0.035="
        f"{scenario_summaries['search_trigger_v2_relaxed_tolerance']['missing_count']}"
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
