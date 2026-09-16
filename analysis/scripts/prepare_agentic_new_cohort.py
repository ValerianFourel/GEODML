#!/usr/bin/env python3
"""Freeze new shared generator prompts from the original pilot's audited sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.inference_wave import (
    _atomic_json,
    _atomic_jsonl,
)
from analysis.scripts.select_readiness_axis_pilot import (
    FORMAT_VERSION,
    Candidate,
    _canonical,
    _normalize,
    select_candidates,
)


def _question_key(question: str) -> str:
    # Match the readiness population's exact-question diversity policy.
    return " ".join(question.split()).casefold()


def _verified_rows(
    root: Path, entry: Mapping[str, Any], label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (root / entry["path"]).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"{label} hash mismatch: {path}")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if (
        not rows
        or any(not isinstance(row, dict) for row in rows)
        or type(entry["rows"]) is not int
        or len(rows) != entry["rows"]
    ):
        raise ValueError(f"{label} row count or JSON object mismatch: {path}")
    return rows, {"path": str(path), "sha256": digest, "rows": len(rows)}


def _validate_excluded_rows(
    prompts: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    records: list[dict[str, Any]],
    by_id: Mapping[str, Candidate],
) -> set[str]:
    excluded_ids = {row["candidate_id"] for row in prompts}
    if len(excluded_ids) != len(prompts) or not excluded_ids.issubset(by_id):
        raise ValueError("excluded prompt IDs are duplicated or outside the population")
    if len(axes) != len(excluded_ids) or len(records) != len(excluded_ids):
        raise ValueError("excluded artifact lengths differ")
    if {row["candidate_id"] for row in axes} != excluded_ids or {
        row["candidate_id"] for row in records
    } != excluded_ids:
        raise ValueError("excluded artifact candidate ID sets differ")
    for row in prompts:
        if row != by_id[row["candidate_id"]].prompt:
            raise ValueError("excluded prompt differs from the frozen population")
    for row in axes:
        if row != by_id[row["candidate_id"]].axis:
            raise ValueError("excluded axis row differs from the frozen population")
    for row in records:
        if row.get("axis_bin") != by_id[row["candidate_id"]].axis_bin:
            raise ValueError("excluded axis bin differs from the frozen population")
    return excluded_ids


def prepare_new_cohort(
    selection_root: Path,
    output: Path,
    *,
    prompt_count: int = 120,
    axis_bins: int = 20,
    master_seed: int = 20260916,
    expected_population_count: int = 26009,
    expected_excluded_count: int = 500,
    source_git_commit: str,
    exclude_cohort_roots: Sequence[Path] = (),
) -> Path:
    """Validate original inputs, exclude previous text/IDs, and freeze one cohort."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cohort: {output}")
    if axis_bins < 2 or prompt_count < axis_bins or prompt_count % axis_bins:
        raise ValueError("prompt count must be a positive multiple of axis bins")
    if not re.fullmatch(r"[0-9a-f]{40}", source_git_commit):
        raise ValueError("source Git commit must be a full 40-character SHA")
    manifest_path = selection_root / "selection-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    original = json.loads(manifest_bytes)
    if original["format_version"] != FORMAT_VERSION:
        raise ValueError(
            "expected an original readiness-axis-balanced-pilot-v1 selection"
        )
    if original["diagnostics"]["axis_bins"] != axis_bins:
        raise ValueError("axis-bin count differs from the original selection")

    prompts, prompt_source = _verified_rows(
        selection_root, original["sources"]["prompts"], "population prompts"
    )
    axes, axis_source = _verified_rows(
        selection_root, original["sources"]["axis_map"], "population axis map"
    )
    old_prompts, old_prompt_source = _verified_rows(
        selection_root, original["artifacts"]["prompts"], "excluded prompts"
    )
    old_axes, _ = _verified_rows(
        selection_root, original["artifacts"]["axis_map"], "excluded axis map"
    )
    old_records, old_record_source = _verified_rows(
        selection_root, original["artifacts"]["selection_records"], "excluded records"
    )
    if (
        len(prompts) != expected_population_count
        or len(old_prompts) != expected_excluded_count
    ):
        raise ValueError("population or excluded prompt count differs from expected")
    # Compute bins from the frozen population percentiles, never rerank/rebin a subset.
    population = _normalize(prompts, axes, axis_bins)
    by_id = {row.candidate_id: row for row in population}
    excluded_ids = _validate_excluded_rows(old_prompts, old_axes, old_records, by_id)
    original_excluded_count = len(excluded_ids)
    excluded_text = {_question_key(row["question"]) for row in old_prompts}
    additional_exclusions = []
    for cohort_root in exclude_cohort_roots:
        cohort_manifest_path = cohort_root / "selection-manifest.json"
        cohort_bytes = cohort_manifest_path.read_bytes()
        cohort = json.loads(cohort_bytes)
        if cohort["format_version"] != "agentic-new-prompt-cohort-v1":
            raise ValueError("extra exclusion must be an agentic-new-prompt-cohort-v1")
        if cohort["diagnostics"]["axis_bins"] != axis_bins:
            raise ValueError("extra exclusion axis-bin count differs")
        if cohort["population_count"] != len(population):
            raise ValueError("extra exclusion population count differs")
        for key, source in (("prompts", prompt_source), ("axis_map", axis_source)):
            previous_source = cohort["sources"][key]
            if (
                previous_source["sha256"] != source["sha256"]
                or type(previous_source["rows"]) is not int
                or previous_source["rows"] != source["rows"]
            ):
                raise ValueError(f"extra exclusion population source differs: {key}")
        cohort_rows: dict[str, list[dict[str, Any]]] = {}
        provenance: dict[str, Any] = {
            "selection_manifest": {
                "path": str(cohort_manifest_path.resolve()),
                "sha256": hashlib.sha256(cohort_bytes).hexdigest(),
            },
        }
        for key in ("prompts", "axis_map", "selection_records"):
            cohort_rows[key], provenance[key] = _verified_rows(
                cohort_root, cohort["artifacts"][key], f"extra excluded {key}"
            )
        cohort_ids = _validate_excluded_rows(
            cohort_rows["prompts"], cohort_rows["axis_map"],
            cohort_rows["selection_records"], by_id,
        )
        if cohort["prompt_count"] != len(cohort_ids):
            raise ValueError("extra exclusion prompt count differs")
        provenance["prompt_count"] = len(cohort_ids)
        additional_exclusions.append(provenance)
        excluded_ids.update(cohort_ids)
        excluded_text.update(
            _question_key(row["question"]) for row in cohort_rows["prompts"]
        )

    seen_text: set[str] = set()
    eligible = []
    text_overlap_count = 0
    repeated_text_count = 0
    for row in sorted(population, key=lambda candidate: candidate.candidate_id):
        if row.candidate_id in excluded_ids:
            continue
        text = _question_key(row.prompt["question"])
        if text in excluded_text:
            text_overlap_count += 1
            continue
        if text in seen_text:
            repeated_text_count += 1
            continue
        seen_text.add(text)
        eligible.append(row)
    selected, diagnostics = select_candidates(
        [row.prompt for row in eligible],
        [row.axis for row in eligible],
        sample_size=prompt_count,
        axis_bins=axis_bins,
        master_seed=master_seed,
    )
    population_counts = Counter((row.keyword_id, row.axis_bin) for row in eligible)
    sample_counts = Counter((row.keyword_id, row.axis_bin) for row in selected)
    records = [
        {
            "candidate_id": row.candidate_id,
            "keyword_id": row.keyword_id,
            "axis_bin": row.axis_bin,
            "observed_axis_1_percentile_0_1": row.observed_percentile,
            "assigned_axis_1_0_1": row.assigned_coordinate,
            "stratum_population_count": population_counts[
                (row.keyword_id, row.axis_bin)
            ],
            "stratum_sample_count": sample_counts[(row.keyword_id, row.axis_bin)],
            "analysis_weight": population_counts[(row.keyword_id, row.axis_bin)]
            / sample_counts[(row.keyword_id, row.axis_bin)],
        }
        for row in selected
    ]
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for key, filename, rows in (
        ("prompts", "pilot-prompts.jsonl", [row.prompt for row in selected]),
        ("axis_map", "pilot-axis.jsonl", [row.axis for row in selected]),
        ("selection_records", "selection-records.jsonl", records),
    ):
        path = output / filename
        _atomic_jsonl(path, rows)
        artifacts[key] = {
            "path": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows),
        }
    final_manifest = output / "selection-manifest.json"
    _atomic_json(
        final_manifest,
        {
            "format_version": "agentic-new-prompt-cohort-v1",
            "status": "planned",
            "scientific_result": False,
            "source_git_commit": source_git_commit,
            "master_seed": master_seed,
            "selection_id": hashlib.sha256(
                _canonical(
                    {
                        "master_seed": master_seed,
                        "candidate_ids": [row.candidate_id for row in selected],
                    }
                )
            ).hexdigest(),
            "prompt_count": len(selected),
            "expected_cells_per_model": 12 * len(selected),
            "population_count": len(population),
            "excluded_prompt_count": len(excluded_ids),
            "excluded_prompt_ids_sha256": hashlib.sha256(
                _canonical(sorted(excluded_ids))
            ).hexdigest(),
            "additional_text_overlap_count": text_overlap_count,
            "deduplicated_eligible_text_count": repeated_text_count,
            "text_identity_policy": "collapse-whitespace-casefold-v1",
            "eligible_population_count": len(eligible),
            "sources": {"prompts": prompt_source, "axis_map": axis_source},
            "exclusion": {
                "selection_manifest": {
                    "path": str(manifest_path.resolve()),
                    "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                },
                "prompts": old_prompt_source,
                "selection_records": old_record_source,
            },
            **({
                "original_excluded_prompt_count": original_excluded_count,
                "additional_excluded_prompt_count": len(excluded_ids) - original_excluded_count,
                "additional_exclusions": additional_exclusions,
            } if additional_exclusions else {}),
            "axis_policy": "preserve-frozen-population-percentiles-and-bin-boundaries",
            "selection_design": "equal axis-bin quotas and near-equal keyword quotas over eligible prompts",
            "analysis_weight_population": "eligible prompts after exclusion and exact-text deduplication",
            "interpretation_guard": "Observed axis bins stratify selection; they do not redefine the assigned experimental variable.",
            "diagnostics": diagnostics,
            "artifacts": artifacts,
        },
    )
    return final_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=120)
    parser.add_argument("--axis-bins", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=20260916)
    parser.add_argument("--expected-population-count", type=int, default=26009)
    parser.add_argument(
        "--expected-excluded-count", type=int, default=500,
        help="expected prompt count in the original selection, before extra exclusions",
    )
    parser.add_argument(
        "--exclude-cohort-root", type=Path, action="append", default=[],
        help="also exclude a verified prior new-prompt cohort; may be repeated",
    )
    parser.add_argument("--source-git-commit", required=True)
    arguments = parser.parse_args()
    try:
        manifest = prepare_new_cohort(
            arguments.selection_root.resolve(),
            arguments.output_dir.resolve(),
            prompt_count=arguments.prompt_count,
            axis_bins=arguments.axis_bins,
            master_seed=arguments.master_seed,
            expected_population_count=arguments.expected_population_count,
            expected_excluded_count=arguments.expected_excluded_count,
            source_git_commit=arguments.source_git_commit,
            exclude_cohort_roots=[root.resolve() for root in arguments.exclude_cohort_root],
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(str(error)) from error
    print(f"NEW_COHORT_PREFLIGHT=PASS prompts={arguments.prompt_count} overlap=0")
    print(f"MANIFEST={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
