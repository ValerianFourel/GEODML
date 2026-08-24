#!/usr/bin/env python3
"""Verify and atomically export globally selected strict readiness prompts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"required JSON artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"required JSONL artifact is missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _require_true(row: Mapping[str, object], key: str, source: Path) -> None:
    if row.get(key) is not True:
        raise ValueError(f"strict export requires {key}=true in {source}")


def _require_count(
    row: Mapping[str, object], key: str, expected: int, source: Path
) -> None:
    if int(row.get(key, -1)) != expected:
        raise ValueError(
            f"{source} {key}={row.get(key)!r} does not match selected rows={expected}"
        )


def _atomic_copy(source: Path, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary export path already exists: {temporary}")
    try:
        with source.open("rb") as source_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(output: Path, payload: Mapping[str, object]) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"temporary manifest path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _accepted_candidate_ids(rows: Iterable[Mapping[str, object]]) -> set[str]:
    accepted = set()
    observed = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in observed:
            raise ValueError("merged validation candidate ids must be nonempty and unique")
        observed.add(candidate_id)
        if row.get("accepted") is True:
            accepted.add(candidate_id)
    return accepted


def export_fully_compliant_prompts(
    final_root: str | Path,
    output_file: str | Path,
) -> dict[str, object]:
    root = Path(final_root).resolve()
    output = Path(output_file).resolve()
    output_manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or output_manifest.exists():
        raise ValueError(f"refusing to overwrite strict prompt export: {output}")

    summary_path = root / "verified_round_summary.json"
    selection_root = root / "strict-selection"
    selection_path = selection_root / "spatially_selected_questions.jsonl"
    selection_manifest_path = selection_root / "run_manifest.json"
    diagnostics_path = selection_root / "spatial_coverage_diagnostics.json"
    diversity_root = root / "selected-diversity"
    diversity_manifest_path = diversity_root / "run_manifest.json"
    diversity_audit_path = diversity_root / "question_diversity_audit.json"
    validation_path = root / "merged" / "validation.jsonl"
    validation_manifest_path = validation_path.with_suffix(
        validation_path.suffix + ".manifest.json"
    )

    summary = _read_json(summary_path)
    selection_manifest = _read_json(selection_manifest_path)
    diagnostics = _read_json(diagnostics_path)
    diversity_manifest = _read_json(diversity_manifest_path)
    diversity_audit = _read_json(diversity_audit_path)
    validation_manifest = _read_json(validation_manifest_path)
    selected = _read_jsonl(selection_path)
    validation = _read_jsonl(validation_path)

    if not selected:
        raise ValueError("strict prompt selection is empty")
    selected_count = len(selected)
    _require_count(summary, "selected_count", selected_count, summary_path)
    _require_count(
        selection_manifest,
        "selected_count",
        selected_count,
        selection_manifest_path,
    )
    _require_count(diagnostics, "selected_count", selected_count, diagnostics_path)
    _require_count(
        diagnostics,
        "verified_selected_count",
        selected_count,
        diagnostics_path,
    )

    _require_true(summary, "strict_dual_view_contract_enabled", summary_path)
    _require_true(
        summary,
        "delexicalized_template_uniqueness_enabled",
        summary_path,
    )
    _require_true(summary, "selected_diversity_gate_passed", summary_path)
    _require_true(diversity_manifest, "all_checks_passed", diversity_manifest_path)
    _require_true(diversity_audit, "all_checks_passed", diversity_audit_path)
    _require_count(
        diversity_manifest,
        "row_count",
        selected_count,
        diversity_manifest_path,
    )
    _require_count(
        diversity_audit,
        "row_count",
        selected_count,
        diversity_audit_path,
    )
    coordinate_contract = selection_manifest.get("coordinate_acceptance_contract")
    surface_contract = selection_manifest.get("surface_acceptance_contract")
    if (
        not isinstance(coordinate_contract, dict)
        or coordinate_contract.get("enabled") is not True
    ):
        raise ValueError("strict dual-view coordinate acceptance contract is not enabled")
    if (
        not isinstance(surface_contract, dict)
        or surface_contract.get("enabled") is not True
    ):
        raise ValueError("delexicalized-template uniqueness contract is not enabled")
    _require_true(
        diagnostics,
        "require_both_views_within_tolerance",
        diagnostics_path,
    )
    _require_true(
        diagnostics,
        "require_delexicalized_template_uniqueness",
        diagnostics_path,
    )
    _require_true(
        diagnostics,
        "selected_delexicalized_templates_are_unique",
        diagnostics_path,
    )

    tolerance = float(coordinate_contract.get("distance_tolerance", -1.0))
    if tolerance <= 0:
        raise ValueError("strict coordinate contract has no positive distance tolerance")
    accepted_ids = _accepted_candidate_ids(validation)
    _require_count(
        validation_manifest,
        "reviewed_count",
        len(validation),
        validation_manifest_path,
    )
    _require_count(
        validation_manifest,
        "candidate_count",
        len(validation),
        validation_manifest_path,
    )
    _require_count(
        validation_manifest,
        "accepted_count",
        len(accepted_ids),
        validation_manifest_path,
    )

    candidate_ids = set()
    target_pairs = set()
    for row in selected:
        candidate_id = str(row.get("candidate_id", ""))
        keyword_id = str(row.get("keyword_id", ""))
        target_id = str(row.get("target_id", ""))
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("selected candidate ids must be nonempty and unique")
        candidate_ids.add(candidate_id)
        target_pair = (keyword_id, target_id)
        if not keyword_id or not target_id or target_pair in target_pairs:
            raise ValueError("selected keyword-target assignments must be nonempty and unique")
        target_pairs.add(target_pair)
        if candidate_id not in accepted_ids:
            raise ValueError(f"selected candidate lacks independent acceptance: {candidate_id}")
        if row.get("both_views_within_tolerance") is not True:
            raise ValueError(f"selected candidate fails strict dual-view tolerance: {candidate_id}")
        for key in (
            "target_distance",
            "reference_target_distance",
            "candidate_aligned_target_distance",
        ):
            if float(row.get(key, float("inf"))) > tolerance:
                raise ValueError(
                    f"selected candidate exceeds {key} tolerance: {candidate_id}"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_copy(selection_path, output)
    manifest = {
        "format_version": "fully-compliant-readiness-prompt-export-v1",
        "created_at": _now(),
        "final_root": str(root),
        "selected_count": selected_count,
        "definition": (
            "Globally selected prompts independently accepted by the frozen validator, "
            "within the preregistered target tolerance in both frozen embedding views, "
            "unique by immutable candidate id and keyword-target assignment, globally "
            "delexicalized-template unique, and passing selected-set diversity."
        ),
        "population_spacing_gate_passed": bool(summary.get("spacing_gate_passed")),
        "complete_30330_population_passed": bool(
            summary.get("verified_population_passed")
        ),
        "distance_tolerance": tolerance,
        "source_selection": _identity(selection_path),
        "source_final_summary": _identity(summary_path),
        "source_selection_manifest": _identity(selection_manifest_path),
        "source_spatial_diagnostics": _identity(diagnostics_path),
        "source_selected_diversity_manifest": _identity(diversity_manifest_path),
        "source_selected_diversity_audit": _identity(diversity_audit_path),
        "source_validation": _identity(validation_path),
        "source_validation_manifest": _identity(validation_manifest_path),
        "output": _identity(output),
    }
    _atomic_json(output_manifest, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--output-file", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = export_fully_compliant_prompts(args.final_root, args.output_file)
    print(f"fully_compliant_prompts={manifest['selected_count']}")
    print(f"output={manifest['output']['path']}")
    print(f"manifest={Path(args.output_file).resolve()}.manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
