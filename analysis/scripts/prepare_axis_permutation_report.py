"""Prepare a frozen CPU report configuration without scanning inference results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

COHORT_FORMATS = {
    "agentic-paired-new-cohort-trial-v1",
    "agentic-four-generator-backlog-v1",
    "agentic-generator-backlog-v2",
}
SELECTION_FORMATS = {"readiness-axis-balanced-pilot-v1", "agentic-new-prompt-cohort-v1"}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _object(path):
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected manifest object: {path}")  # noqa: TRY004 -- serialized input
    return value


def _verified(spec, base, provenance):
    path = (base / spec["path"]).resolve()
    raw = path.read_bytes()
    if _sha(raw) != spec["sha256"]:
        raise ValueError(f"frozen artifact hash mismatch: {path}")
    provenance[str(path)] = {"path": str(path), "sha256": _sha(raw)}
    return path, raw


def _rows(raw):
    values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = {}
    for row in values:
        key = row["candidate_id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("missing or duplicate candidate ID")
        result[key] = row
    if not result:
        raise ValueError("empty prompt or coordinate artifact")
    return result


def _immutable(path, raw):
    """Publish complete bytes with no replacement, including concurrent prepares."""
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"refusing to replace different report input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".report-prepare-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise FileExistsError(f"concurrent different report input: {path}") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def _coordinates(prompts_path, prompts_raw, selection_path, provenance, sidecars, output):
    prompts = _rows(prompts_raw)
    for prompt in prompts.values():
        if prompt.get("question_sha256") != _sha(prompt["question"].encode()):
            raise ValueError("prompt question hash mismatch")
    if not selection_path.is_file():
        return None, "missing_frozen_selection_manifest"
    selection_raw = selection_path.read_bytes()
    selection = json.loads(selection_raw)
    if selection["format_version"] not in SELECTION_FORMATS:
        raise ValueError("unsupported selection manifest format")
    provenance[str(selection_path)] = {"path": str(selection_path), "sha256": _sha(selection_raw)}
    selected_path, selected_raw = _verified(selection["artifacts"]["prompts"], selection_path.parent, provenance)
    if selected_path != prompts_path or selected_raw != prompts_raw:
        raise ValueError("selection manifest refers to different prompts")
    axis_path, axis_raw = _verified(selection["artifacts"]["axis_map"], selection_path.parent, provenance)
    axes = _rows(axis_raw)
    if set(axes) != set(prompts):
        raise ValueError("axis and prompt candidate sets differ")
    normalized = []
    for candidate_id, axis in sorted(axes.items()):
        question_hash = prompts[candidate_id]["question_sha256"]
        if axis.get("text_sha256") != question_hash:
            raise ValueError("axis text hash differs from exact prompt question")
        values = {key: value for key, value in axis.items()
                  if key.startswith("consensus_normalized_axis_") and value is not None}
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ValueError("observed coordinates must be finite numbers")
        prompt = prompts[candidate_id]
        recorded = {**prompt, **prompt.get("readiness_coordinates", {})}
        if any(recorded.get(key) is not None and recorded[key] != value for key, value in values.items()):
            raise ValueError("prompt and axis artifact disagree on observed coordinates")
        normalized.append({"candidate_id": candidate_id, "question_sha256": question_hash,
                           **values})
    raw = b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n"
                   for row in normalized)
    path = output.parent / "report-inputs" / ("coordinates-" + _sha(raw) + ".jsonl")
    sidecars[path] = raw
    provenance[str(path)] = {"path": str(path), "sha256": _sha(raw),
                             "source_axis_path": str(axis_path), "join": "candidate_id+exact-text-sha256"}
    status = ("verified_consensus_axis_1" if all("consensus_normalized_axis_1" in row for row in normalized)
              else "partial_or_missing_consensus_axis_1")
    return str(path), status


def prepare_report_config(*, output, study_id, legacy_generator_roots=(), cohort_run_roots=(),
                          judge_plan_manifest=None, judge_outcomes=None):
    """Freeze explicit plans and a verified coordinate join, never result counts."""
    output = Path(output).resolve()
    if not study_id.strip():
        raise ValueError("study_id must be non-empty")
    if bool(judge_plan_manifest) != bool(judge_outcomes):
        raise ValueError("judge plan and outcomes must be supplied together")
    if not legacy_generator_roots and not cohort_run_roots:
        raise ValueError("at least one generator source is required")
    provenance, sidecars, statuses, agentic = {}, {}, [], []
    for root in sorted({Path(p).resolve() for p in legacy_generator_roots}):
        manifest = _object(root / "run_manifest.json")
        if manifest["format_version"] != "agentic-search-execution-calibration-v2":
            raise ValueError("legacy source must be an execution-calibration-v2 shard")
        sources = manifest["prompt_sources"]
        prompts_path, prompts_raw = _verified(sources["prompts_jsonl"], root, provenance)
        _verified(sources["selection_records_jsonl"], root, provenance)
        axis, status = _coordinates(prompts_path, prompts_raw, prompts_path.parent / "selection-manifest.json",
                                    provenance, sidecars, output)
        spec = {"prompts_jsonl": str(prompts_path), "generator_roots": [str(root)]}
        if axis:
            spec["axis_jsonl"] = axis
        agentic.append(spec)
        statuses.append({"source": str(root), "observed_coordinates": status})
    for root in sorted({Path(p).resolve() for p in cohort_run_roots}):
        manifest_path = root / "run_manifest.json"
        manifest = _object(manifest_path)
        if manifest["format_version"] not in COHORT_FORMATS:
            raise ValueError("unsupported outer cohort run format")
        if not manifest.get("models"):
            raise ValueError("cohort manifest has no planned models")
        prompts_path = root / "cohort/pilot-prompts.jsonl"
        selection_path = root / "cohort/selection-manifest.json"
        if manifest["format_version"] == "agentic-paired-new-cohort-trial-v1":
            recorded_selection, _ = _verified(manifest["cohort_manifest"], root, provenance)
            if recorded_selection != selection_path:
                raise ValueError("cohort manifest path is outside the run cohort")
            _verified(manifest["tasks"], root, provenance)
        else:
            for relative in ("cohort/selection-manifest.json", "cohort/pilot-prompts.jsonl",
                             "cohort/pilot-axis.jsonl", "tasks.jsonl"):
                _verified({"path": relative, "sha256": manifest["frozen_files"][relative]}, root, provenance)
        prompts_raw = prompts_path.read_bytes()
        axis, status = _coordinates(prompts_path, prompts_raw, selection_path, provenance, sidecars, output)
        roots = []
        for slug in sorted(manifest["models"]):
            model_root = root / "models" / slug
            if model_root.resolve().parent != root / "models":
                raise ValueError("invalid model directory in cohort manifest")
            roots.append(str(model_root))
        spec = {"prompts_jsonl": str(prompts_path), "task_manifest": str(manifest_path),
                "generator_roots": roots}
        if axis:
            spec["axis_jsonl"] = axis
        agentic.append(spec)
        statuses.append({"source": str(root), "observed_coordinates": status})
    judges = []
    if judge_plan_manifest:
        path = Path(judge_plan_manifest).resolve()
        raw = path.read_bytes()
        plan = json.loads(raw)
        provenance[str(path)] = {"path": str(path), "sha256": _sha(raw)}
        if plan["format_version"] not in {"agentic-search-judge-v1", "agentic-search-judge-recorded-conversation-v2"}:
            raise ValueError("unsupported judge plan format")
        for name in ("bulk_tasks", "private_mapping"):
            _verified(plan["artifacts"][name], path.parent, provenance)
        judges.append({"plan_manifest": str(path), "outcome_files": [str(Path(judge_outcomes).resolve())],
                       "role": "bulk"})
    config = {
        "format_version": "axis-permutation-study-config-v1", "study_id": study_id,
        "axis_fields": {
            "target_normalized_axis_1": {"role": "assigned", "construct": "decision-readiness", "domain": [0, 1]},
            "consensus_normalized_axis_1": {"role": "observed", "construct": "decision-readiness", "domain": [0, 1]},
        },
        "fit": {"seed": 20260916, "holdout_modulus": 5, "ridge": 1.0, "min_train": 6, "min_test": 2},
        "sources": {"direct": [], "agentic": agentic, "judges": judges},
        "preparation": {"format_version": "axis-permutation-report-preparation-v1",
                        "direct_study": "not_configured_no_plan_found_in_supplied_inventory",
                        "coordinate_status": statuses,
                        "verified_artifacts": sorted(provenance.values(), key=lambda row: row["path"]),
                        "result_scan_performed": False},
    }
    raw = _bytes(config)
    if output.exists() and output.read_bytes() != raw:
        raise FileExistsError(f"refusing to replace different report configuration: {output}")
    for path, payload in sidecars.items():
        _immutable(path, payload)
    _immutable(output, raw)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-generator-root", type=Path, action="append", default=[])
    parser.add_argument("--cohort-run-root", type=Path, action="append", default=[])
    parser.add_argument("--judge-plan-manifest", type=Path)
    parser.add_argument("--judge-outcomes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    args = parser.parse_args(argv)
    config = prepare_report_config(output=args.output, study_id=args.study_id,
        legacy_generator_roots=args.legacy_generator_root, cohort_run_roots=args.cohort_run_root,
        judge_plan_manifest=args.judge_plan_manifest, judge_outcomes=args.judge_outcomes)
    print("REPORT_CONFIG=" + str(args.output.resolve()))
    print("PREPARATION=" + json.dumps(config["preparation"], sort_keys=True))
    print("RESULT_SCAN_PERFORMED=false")
    print("INFERENCE_OR_ALLOCATION_STARTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
