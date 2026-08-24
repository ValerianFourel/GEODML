#!/usr/bin/env python3
"""Independently count prompts that satisfy every strict readiness contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.readiness_prompt_population import (
    audit_question_diversity,
    delexicalize_question,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"required JSON artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read required JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"required JSONL artifact is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield value


def _matches_count(row: Mapping[str, object], key: str, expected: int) -> bool:
    try:
        return int(row.get(key, -1)) == expected
    except (TypeError, ValueError):
        return False


def _finite_distance_within(row: Mapping[str, object], key: str, limit: float) -> bool:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and 0.0 <= value <= limit


def _validation_acceptance_is_consistent(row: Mapping[str, object]) -> bool:
    required_true = (
        "accepted",
        "exact_keyword_present",
        "single_question",
        "topic_relevant",
        "search_intent",
        "web_answerable",
        "standalone",
        "natural_language",
    )
    try:
        score = int(row.get("relevance_score_1_5", 0))
    except (TypeError, ValueError):
        return False
    return all(row.get(key) is True for key in required_true) and score >= 4


def _candidate_matches_selection(
    candidate: Mapping[str, object], selected: Mapping[str, object]
) -> bool:
    stable_fields = (
        "candidate_id",
        "keyword_id",
        "keyword",
        "target_id",
        "question",
        "generator_id",
        "generator_model",
    )
    return all(candidate.get(key) == selected.get(key) for key in stable_fields)


def _question_hash_matches(
    candidate: Mapping[str, object], selected: Mapping[str, object]
) -> bool:
    expected = str(candidate.get("question_sha256", ""))
    question = str(selected.get("question", ""))
    observed = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return bool(expected) and observed == expected


def _diversity_thresholds(
    diversity_audit: Mapping[str, object],
) -> dict[str, float | int]:
    thresholds = diversity_audit.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("selected diversity audit lacks its thresholds")
    try:
        return {
            "minimum_delexicalized_unique_fraction": float(
                thresholds["minimum_delexicalized_unique_fraction"]
            ),
            "maximum_template_fraction": float(
                thresholds["maximum_template_fraction"]
            ),
            "minimum_median_keyword_unique_fraction": float(
                thresholds["minimum_median_keyword_unique_fraction"]
            ),
            "minimum_keyword_unique_fraction": float(
                thresholds["minimum_keyword_unique_fraction"]
            ),
            "maximum_opening_frame_fraction": float(
                thresholds["maximum_opening_frame_fraction"]
            ),
            "opening_frame_tokens": int(diversity_audit["opening_frame_tokens"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selected diversity audit has invalid thresholds") from exc


def audit_fully_compliant_prompts(
    final_root: str | Path,
    *,
    maximum_failure_examples: int = 20,
) -> dict[str, object]:
    """Audit final artifacts without modifying or re-selecting any prompt.

    ``fully_compliant_prompt_count`` counts rows that independently pass every
    attributable row contract. ``ready_to_export_count`` is nonzero only when
    those rows also form the exact, globally valid selected set recorded by all
    manifests and the recomputed diversity audit.
    """

    if maximum_failure_examples < 0:
        raise ValueError("maximum failure examples must be nonnegative")
    root = Path(final_root).resolve()
    summary_path = root / "verified_round_summary.json"
    selection_root = root / "strict-selection"
    selection_path = selection_root / "spatially_selected_questions.jsonl"
    selection_manifest_path = selection_root / "run_manifest.json"
    diagnostics_path = selection_root / "spatial_coverage_diagnostics.json"
    diversity_root = root / "selected-diversity"
    diversity_manifest_path = diversity_root / "run_manifest.json"
    diversity_audit_path = diversity_root / "question_diversity_audit.json"
    candidates_path = root / "merged" / "candidates.jsonl"
    candidates_manifest_path = candidates_path.with_suffix(
        candidates_path.suffix + ".manifest.json"
    )
    validation_path = root / "merged" / "validation.jsonl"
    validation_manifest_path = validation_path.with_suffix(
        validation_path.suffix + ".manifest.json"
    )

    summary = _read_json(summary_path)
    selection_manifest = _read_json(selection_manifest_path)
    diagnostics = _read_json(diagnostics_path)
    diversity_manifest = _read_json(diversity_manifest_path)
    stored_diversity = _read_json(diversity_audit_path)
    candidates_manifest = _read_json(candidates_manifest_path)
    validation_manifest = _read_json(validation_manifest_path)
    selected = list(_read_jsonl(selection_path))

    try:
        coordinate_contract = selection_manifest["coordinate_acceptance_contract"]
        tolerance = float(coordinate_contract["distance_tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("strict selection lacks a distance tolerance") from exc
    if (
        not isinstance(coordinate_contract, dict)
        or coordinate_contract.get("enabled") is not True
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError(
            "strict coordinate contract is not enabled with a positive tolerance"
        )

    failures: list[set[str]] = [set() for _ in selected]
    candidate_id_indices: dict[str, list[int]] = defaultdict(list)
    target_pair_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    question_indices: dict[str, list[int]] = defaultdict(list)
    template_indices: dict[str, list[int]] = defaultdict(list)

    for index, row in enumerate(selected):
        candidate_id = str(row.get("candidate_id", "")).strip()
        keyword_id = str(row.get("keyword_id", "")).strip()
        keyword = str(row.get("keyword", "")).strip()
        target_id = str(row.get("target_id", "")).strip()
        question = str(row.get("question", "")).strip()
        required = {
            "nonempty_candidate_id": candidate_id,
            "nonempty_keyword_id": keyword_id,
            "nonempty_keyword": keyword,
            "nonempty_target_id": target_id,
            "nonempty_question": question,
        }
        failures[index].update(name for name, value in required.items() if not value)
        candidate_id_indices[candidate_id].append(index)
        target_pair_indices[(keyword_id, target_id)].append(index)
        question_indices[" ".join(question.split()).casefold()].append(index)
        if keyword and question:
            try:
                template_indices[delexicalize_question(question, keyword)].append(index)
            except ValueError:
                failures[index].add("question_contains_exact_keyword")
        if keyword not in question:
            failures[index].add("question_contains_exact_keyword")
        if not question.endswith("?") or question.count("?") != 1:
            failures[index].add("single_question")
        if row.get("both_views_within_tolerance") is not True:
            failures[index].add("both_views_within_tolerance")
        for key in (
            "target_distance",
            "reference_target_distance",
            "candidate_aligned_target_distance",
        ):
            if not _finite_distance_within(row, key, tolerance):
                failures[index].add(f"{key}_within_tolerance")

    for groups, failure_name in (
        (candidate_id_indices, "unique_candidate_id"),
        (target_pair_indices, "unique_keyword_target_assignment"),
        (question_indices, "unique_exact_question"),
        (template_indices, "unique_delexicalized_template"),
    ):
        for indices in groups.values():
            if len(indices) > 1:
                for index in indices:
                    failures[index].add(failure_name)

    selected_ids = {key for key in candidate_id_indices if key}
    candidate_ids: set[str] = set()
    selected_candidates: dict[str, dict[str, object]] = {}
    candidate_duplicate_count = 0
    candidate_count = 0
    for candidate in _read_jsonl(candidates_path):
        candidate_count += 1
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in candidate_ids:
            candidate_duplicate_count += 1
        candidate_ids.add(candidate_id)
        if candidate_id in selected_ids:
            selected_candidates[candidate_id] = candidate

    validation_ids: set[str] = set()
    selected_validations: dict[str, dict[str, object]] = {}
    validation_duplicate_count = 0
    validation_count = 0
    accepted_count = 0
    for validation in _read_jsonl(validation_path):
        validation_count += 1
        candidate_id = str(validation.get("candidate_id", ""))
        if not candidate_id or candidate_id in validation_ids:
            validation_duplicate_count += 1
        validation_ids.add(candidate_id)
        if validation.get("accepted") is True:
            accepted_count += 1
        if candidate_id in selected_ids:
            selected_validations[candidate_id] = validation

    for candidate_id, indices in candidate_id_indices.items():
        candidate = selected_candidates.get(candidate_id)
        validation = selected_validations.get(candidate_id)
        for index in indices:
            row = selected[index]
            if candidate is None:
                failures[index].add("candidate_present_in_merged_source")
            else:
                if not _candidate_matches_selection(candidate, row):
                    failures[index].add("selected_content_matches_merged_candidate")
                if not _question_hash_matches(candidate, row):
                    failures[index].add("question_sha256_matches_merged_candidate")
            if validation is None:
                failures[index].add("independent_validation_present")
            elif not _validation_acceptance_is_consistent(validation):
                failures[index].add("independent_acceptance_contract")

    global_checks: dict[str, bool] = {}

    def check(name: str, value: object) -> None:
        global_checks[name] = value is True

    selected_count = len(selected)
    check("selection_is_nonempty", selected_count > 0)
    check("candidate_ids_are_unique", candidate_duplicate_count == 0)
    check("validation_ids_are_unique", validation_duplicate_count == 0)
    check("validation_covers_exact_candidate_set", validation_ids == candidate_ids)
    check(
        "candidate_manifest_count_matches",
        _matches_count(candidates_manifest, "candidate_count", candidate_count),
    )
    for name, row in (
        ("summary", summary),
        ("selection_manifest", selection_manifest),
        ("spatial_diagnostics", diagnostics),
    ):
        check(
            f"{name}_candidate_count_matches",
            _matches_count(row, "candidate_count", candidate_count),
        )
    check(
        "validation_manifest_candidate_count_matches",
        _matches_count(validation_manifest, "candidate_count", validation_count),
    )
    check(
        "validation_manifest_reviewed_count_matches",
        _matches_count(validation_manifest, "reviewed_count", validation_count),
    )
    check(
        "validation_manifest_accepted_count_matches",
        _matches_count(validation_manifest, "accepted_count", accepted_count),
    )
    check(
        "summary_accepted_count_matches",
        _matches_count(summary, "independently_accepted_count", accepted_count),
    )
    for name, row in (
        ("summary", summary),
        ("selection_manifest", selection_manifest),
        ("spatial_diagnostics", diagnostics),
    ):
        check(
            f"{name}_selected_count_matches",
            _matches_count(row, "selected_count", selected_count),
        )
    check(
        "spatial_diagnostics_verified_count_matches",
        _matches_count(diagnostics, "verified_selected_count", selected_count),
    )
    check(
        "diversity_manifest_count_matches",
        _matches_count(diversity_manifest, "row_count", selected_count),
    )
    check(
        "diversity_audit_count_matches",
        _matches_count(stored_diversity, "row_count", selected_count),
    )
    check(
        "strict_dual_view_summary_contract_enabled",
        summary.get("strict_dual_view_contract_enabled") is True,
    )
    check(
        "strict_dual_view_diagnostics_contract_enabled",
        diagnostics.get("require_both_views_within_tolerance") is True,
    )
    surface_contract = selection_manifest.get("surface_acceptance_contract")
    check(
        "delexicalized_uniqueness_selection_contract_enabled",
        isinstance(surface_contract, dict) and surface_contract.get("enabled") is True,
    )
    check(
        "delexicalized_uniqueness_summary_contract_enabled",
        summary.get("delexicalized_template_uniqueness_enabled") is True,
    )
    check(
        "delexicalized_uniqueness_diagnostics_contract_enabled",
        diagnostics.get("require_delexicalized_template_uniqueness") is True,
    )
    check(
        "diagnostics_reports_unique_templates",
        diagnostics.get("selected_delexicalized_templates_are_unique") is True,
    )
    check(
        "summary_reports_selected_diversity_passed",
        summary.get("selected_diversity_gate_passed") is True,
    )
    check(
        "diversity_manifest_reports_passed",
        diversity_manifest.get("all_checks_passed") is True,
    )
    check(
        "stored_diversity_audit_reports_passed",
        stored_diversity.get("all_checks_passed") is True,
    )

    recomputed_diversity: dict[str, object] | None = None
    recomputed_diversity_error: str | None = None
    if selected:
        try:
            recomputed_diversity = audit_question_diversity(
                selected,
                **_diversity_thresholds(stored_diversity),
            )
            check(
                "recomputed_diversity_passes",
                recomputed_diversity.get("all_checks_passed") is True,
            )
            check(
                "recomputed_diversity_matches_stored_checks",
                recomputed_diversity.get("checks") == stored_diversity.get("checks"),
            )
            for key in (
                "row_count",
                "keyword_count",
                "delexicalized_template_count",
                "opening_frame_tokens",
            ):
                check(
                    f"recomputed_diversity_{key}_matches",
                    recomputed_diversity.get(key) == stored_diversity.get(key),
                )
        except ValueError as exc:
            recomputed_diversity_error = str(exc)
            check("recomputed_diversity_passes", False)
            check("recomputed_diversity_matches_stored_checks", False)
    else:
        check("recomputed_diversity_passes", False)
        check("recomputed_diversity_matches_stored_checks", False)

    failure_counts = Counter(
        failure_name for row_failures in failures for failure_name in row_failures
    )
    fully_compliant_count = sum(not row_failures for row_failures in failures)
    failed_count = selected_count - fully_compliant_count
    global_contracts_passed = all(global_checks.values())
    audit_passed = (
        selected_count > 0
        and failed_count == 0
        and global_contracts_passed
    )
    failure_examples = []
    for index, row_failures in enumerate(failures):
        if not row_failures:
            continue
        failure_examples.append(
            {
                "candidate_id": str(selected[index].get("candidate_id", "")),
                "failed_checks": sorted(row_failures),
            }
        )
        if len(failure_examples) >= maximum_failure_examples:
            break

    return {
        "format_version": "fully-compliant-readiness-prompt-audit-v1",
        "created_at": _now(),
        "final_root": str(root),
        "audit_passed": audit_passed,
        "claimed_selected_count": selected_count,
        "fully_compliant_prompt_count": fully_compliant_count,
        "failed_prompt_count": failed_count,
        "ready_to_export_count": selected_count if audit_passed else 0,
        "candidate_count": candidate_count,
        "validation_count": validation_count,
        "independently_accepted_candidate_count": accepted_count,
        "distance_tolerance": tolerance,
        "global_contracts_passed": global_contracts_passed,
        "global_checks": global_checks,
        "failed_global_checks": sorted(
            name for name, passed in global_checks.items() if not passed
        ),
        "failed_prompt_checks": dict(sorted(failure_counts.items())),
        "failure_examples": failure_examples,
        "failure_examples_truncated": failed_count > len(failure_examples),
        "population_spacing_gate_passed": summary.get("spacing_gate_passed") is True,
        "complete_30330_population_passed": (
            summary.get("verified_population_passed") is True
        ),
        "recomputed_diversity": recomputed_diversity,
        "recomputed_diversity_error": recomputed_diversity_error,
        "definition": (
            "A fully compliant prompt exactly matches its immutable merged candidate, "
            "satisfies every independent-validator acceptance field, lies within the "
            "preregistered tolerance in both frozen embedding views, has unique IDs, "
            "keyword-target assignment, exact wording, and delexicalized template, "
            "and belongs to a selected set whose diversity and artifact contracts "
            "are independently recomputed and consistent."
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--report-file")
    parser.add_argument("--maximum-failure-examples", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="print the full JSON report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = audit_fully_compliant_prompts(
            args.final_root,
            maximum_failure_examples=args.maximum_failure_examples,
        )
        if args.report_file:
            _atomic_json(Path(args.report_file).resolve(), report)
    except (OSError, ValueError) as exc:
        print(f"AUDIT=ERROR\nerror={exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"AUDIT={'PASS' if report['audit_passed'] else 'FAIL'}")
        print(f"claimed_ready_prompts={report['claimed_selected_count']}")
        print(f"fully_compliant_prompts={report['fully_compliant_prompt_count']}")
        print(f"failed_prompts={report['failed_prompt_count']}")
        print(f"ready_to_export={report['ready_to_export_count']}")
        print(f"validated_candidates={report['validation_count']}")
        print(
            "independently_accepted_candidates="
            f"{report['independently_accepted_candidate_count']}"
        )
        print(
            "complete_30330_population="
            f"{'YES' if report['complete_30330_population_passed'] else 'NO'}"
        )
        if report["failed_global_checks"]:
            print("failed_global_checks=" + ",".join(report["failed_global_checks"]))
        if report["failed_prompt_checks"]:
            print(
                "failed_prompt_checks="
                + ",".join(
                    f"{key}:{value}"
                    for key, value in report["failed_prompt_checks"].items()
                )
            )
        if args.report_file:
            print(f"report={Path(args.report_file).resolve()}")
    return 0 if report["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
