#!/usr/bin/env python3
"""Validate and freeze Phase-1 semantic-readiness corpus inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    DEFAULT_WEB_SPEC,
)
from interpretability.pipeline.semantic_readiness_phase1 import (  # noqa: E402
    PHASE1_FREEZE_VERSION,
    SurfaceSnapshotInput,
    TransferSnapshotInput,
    build_phase1_freeze_audit,
)
from interpretability.pipeline.semantic_readiness_transfer import (  # noqa: E402
    DEFAULT_TRANSFER_SPEC,
)


DEFAULT_OUTPUT = ANALYSIS_ROOT / "output" / "semantic_readiness_phase1_freeze_v1"
DEFAULT_SURFACE = ANALYSIS_ROOT / "output" / "query_free_surface_corpus_v2"
DEFAULT_WEB = ANALYSIS_ROOT / "output" / "semantic_readiness_web_v2"
DEFAULT_BASE = ANALYSIS_ROOT / "output" / "semantic_readiness_corpus_v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-corpus", default=DEFAULT_BASE / "semantic_readiness_corpus.jsonl")
    parser.add_argument("--base-manifest", default=DEFAULT_BASE / "run_manifest.json")
    parser.add_argument("--surface-corpus", default=DEFAULT_SURFACE / "surface_coverage_corpus.jsonl")
    parser.add_argument("--surface-manifest", default=DEFAULT_SURFACE / "run_manifest.json")
    parser.add_argument("--surface-provenance", default=DEFAULT_SURFACE / "source_provenance.json")
    parser.add_argument("--web-records", default=DEFAULT_WEB / "web_text_records.jsonl")
    parser.add_argument("--web-manifest", default=DEFAULT_WEB / "run_manifest.json")
    parser.add_argument("--web-raw-responses", default=DEFAULT_WEB / "raw_responses")
    parser.add_argument("--web-specification", default=DEFAULT_WEB_SPEC)
    parser.add_argument("--transfer-specification", default=DEFAULT_TRANSFER_SPEC)
    parser.add_argument(
        "--surface-source-input",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help=(
            "Portable raw Layer-1 source path override; repeat for provenance "
            "paths that do not exist on this machine."
        ),
    )
    parser.add_argument(
        "--transfer-source-input",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help="Optional acquired Layer-2 snapshot; repeat once per source.",
    )
    parser.add_argument(
        "--transfer-source-revision",
        action="append",
        default=[],
        metavar="SOURCE_ID=REVISION",
        help="Pinned upstream revision paired with --transfer-source-input.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Atomically replace only the two Phase-1 audit files in an existing output directory.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and not args.replace_output:
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = _transfer_inputs(
        args.transfer_source_input,
        args.transfer_source_revision,
    )
    audit = build_phase1_freeze_audit(
        base_corpus_path=args.base_corpus,
        base_manifest_path=args.base_manifest,
        surface_corpus_path=args.surface_corpus,
        surface_manifest_path=args.surface_manifest,
        surface_provenance_path=args.surface_provenance,
        web_records_path=args.web_records,
        web_manifest_path=args.web_manifest,
        web_raw_responses_dir=args.web_raw_responses,
        web_specification_path=args.web_specification,
        transfer_specification_path=args.transfer_specification,
        surface_inputs=_surface_inputs(args.surface_source_input),
        transfer_inputs=inputs,
    )
    audit["generated_at"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    audit["git_commit_sha"] = _git_sha()
    audit["environment"] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    audit["phase1_contract"] = {
        "phase": 1,
        "annotation_allowed": False,
        "embedding_allowed": False,
        "geo_outcomes_allowed": False,
        "production_inference_allowed": False,
    }
    manifest_path = output / "phase1_corpus_freeze_manifest.json"
    report_path = output / "phase1_corpus_freeze_report.md"
    _atomic_text(
        manifest_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(report_path, _report(audit))
    print(f"Layer 1 frozen: {audit['phase_gate']['layer1_frozen']}")
    print(
        "Layer 2 snapshots frozen: "
        f"{audit['phase_gate']['layer2_snapshots_frozen']}"
    )
    print(f"Phase gate: {audit['phase_gate']['status']}")
    print(f"output: {output}")
    return 0 if audit["phase_gate"]["layer1_frozen"] else 1


def _transfer_inputs(
    raw_paths: list[str],
    raw_revisions: list[str],
) -> tuple[TransferSnapshotInput, ...]:
    paths = _key_values(raw_paths, "transfer source input")
    revisions = _key_values(raw_revisions, "transfer source revision")
    if set(paths) != set(revisions):
        raise SystemExit(
            "--transfer-source-input and --transfer-source-revision must name "
            "the same source IDs"
        )
    return tuple(
        TransferSnapshotInput(
            source_id=source_id,
            path=Path(paths[source_id]),
            revision=revisions[source_id],
        )
        for source_id in sorted(paths)
    )


def _surface_inputs(raw_paths: list[str]) -> tuple[SurfaceSnapshotInput, ...]:
    paths = _key_values(raw_paths, "surface source input")
    return tuple(
        SurfaceSnapshotInput(source_id=source_id, path=Path(paths[source_id]))
        for source_id in sorted(paths)
    )


def _key_values(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, raw = str(value).partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator or not key or not raw:
            raise SystemExit(f"invalid {label}; expected SOURCE_ID=VALUE: {value!r}")
        if key in result:
            raise SystemExit(f"duplicate {label}: {key}")
        result[key] = raw
    return result


def _report(audit: dict[str, object]) -> str:
    base = audit["base_corpus"]
    web = audit["web_inputs"]
    transfer = audit["transfer_registry"]
    gate = audit["phase_gate"]
    assert isinstance(base, dict)
    assert isinstance(web, dict)
    assert isinstance(transfer, dict)
    assert isinstance(gate, dict)
    lines = [
        "# Phase 1 — semantic-readiness corpus freeze",
        "",
        f"Artifact contract: `{PHASE1_FREEZE_VERSION}`",
        "",
        "## Layer 1",
        "",
        f"- Frozen: `{base['frozen']}`",
        f"- Records: `{base['record_count']}`",
        f"- Development: `{base['development_count']}`",
        f"- Locked confirmation: `{base['confirmation_count']}`",
        f"- Exact-unique text hashes: `{base['unique_text_sha256_count']}`",
        f"- Exact reconstruction from recorded inputs: `{base['exact_rebuild_matches']}`",
        f"- Cross-split groups: `{base['cross_split_group_count']}`",
        f"- Unknown included licenses: `{base['unknown_license_count']}`",
        f"- Corpus SHA-256: `{base['sha256']}`",
        "",
        "## Stack Exchange acquisition",
        "",
        f"- Frozen probes: `{web['probe_count']}`",
        f"- Unique records acquired: `{web['record_count']}`",
        f"- Licensed records included: `{web['licensed_included_count']}`",
        f"- Unknown-license records excluded: `{web['unknown_license_excluded_count']}`",
        f"- Raw-response directory SHA-256: `{web['raw_response_directory_sha256']}`",
        "",
        "## Layer 2 transfer registry",
        "",
        f"- Registry sources: `{transfer['source_count']}`",
        f"- Development sources: `{transfer['development_source_count']}`",
        f"- Locked-confirmation sources: `{transfer['confirmation_source_count']}`",
        f"- All revision-pinned snapshots present: `{transfer['all_snapshots_present']}`",
        "",
        "| Split | Source | Access | Snapshot | Revision |",
        "|---|---|---|---:|---|",
    ]
    for row in transfer["sources"]:
        lines.append(
            f"| {row['split']} | `{row['source_id']}` | {row['access']} | "
            f"{row['snapshot_present']} | `{row['source_revision'] or 'pending'}` |"
        )
    lines.extend(
        [
            "",
            "## Phase gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- Safe to begin base-only annotation: `{gate['safe_to_begin_base_only_annotation']}`",
            f"- Safe to begin full Phase 2: `{gate['safe_to_begin_full_phase2']}`",
            "",
            "No annotation, embedding, GEO outcome inspection, or model inference was",
            "performed by this Phase-1 audit.",
            "",
        ]
    )
    blockers = [*base["blockers"], *transfer["blockers"]]
    if blockers:
        lines.extend(("## Outstanding blockers", ""))
        lines.extend(f"- {item}" for item in blockers)
        lines.append("")
    return "\n".join(lines)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
