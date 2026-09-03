#!/usr/bin/env python3
"""Strictly union a complete set of disjoint readiness partition checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.readiness_hf_dataset import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    read_json,
    read_jsonl,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _latest_verified_round(root: Path) -> Path:
    rounds = []
    for summary in root.glob("round-*/verified_round_summary.json"):
        name = summary.parent.name
        try:
            index = int(name.removeprefix("round-"))
        except ValueError:
            continue
        rounds.append((index, summary.parent))
    if not rounds:
        raise ValueError(f"partition has no verified round: {root}")
    return max(rounds)[1]


def _merge_rows(
    paths: Sequence[Path],
    *,
    key: str,
    expected_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row_id = str(row.get(key, ""))
            if not row_id:
                raise ValueError(f"row lacks {key}: {path}")
            existing = merged.get(row_id)
            if existing is not None and existing != row:
                raise ValueError(f"partition rows conflict for {key}={row_id}")
            merged[row_id] = row
    if expected_ids is not None and set(merged) != expected_ids:
        raise ValueError(f"merged {key} identities do not equal candidate identities")
    return [merged[row_id] for row_id in sorted(merged)]


def _candidate_paths(round_root: Path) -> list[Path]:
    listing = round_root / "candidate-files.txt"
    if not listing.is_file():
        raise ValueError(f"partition round lacks candidate file list: {listing}")
    paths = [
        Path(value).resolve()
        for value in listing.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(f"partition candidate file list is incomplete: {listing}")
    return paths


def _normalized_embedding_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    embedding = dict(manifest.get("embedding", {}))
    embedding.setdefault("attention_implementation", "eager")
    return embedding


def _archive_locator_candidates(
    manifest_path: Path,
    row: Mapping[str, object],
) -> tuple[Path, ...]:
    declared = Path(str(row.get("path", "")))
    if not declared.is_absolute():
        declared = manifest_path.resolve().parent / declared
    candidates = [declared]
    attempt_name = declared.parent.name
    for view in ("qwen", "mistral"):
        if attempt_name.startswith(f".{view}-attempt-"):
            candidates.append(declared.parent.parent / view / declared.name)
            break
    return tuple(candidates)


def _resolve_embedding_archive(
    manifest_path: Path,
    row: Mapping[str, object],
    *,
    sha256_cache: dict[Path, str],
) -> dict[str, object]:
    try:
        expected_size = int(row.get("size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding archive inventory is invalid") from exc
    expected_sha256 = str(row.get("sha256", ""))
    if expected_size <= 0 or (expected_sha256 and len(expected_sha256) != 64):
        raise ValueError("embedding archive inventory is invalid")
    for candidate in _archive_locator_candidates(manifest_path, row):
        path = candidate.resolve()
        if not path.is_file() or path.stat().st_size != expected_size:
            continue
        if expected_sha256:
            observed_sha256 = sha256_cache.get(path)
            if observed_sha256 is None:
                observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                sha256_cache[path] = observed_sha256
            if observed_sha256 != expected_sha256:
                continue
        normalized = {
            "path": str(path),
            "size_bytes": expected_size,
        }
        if expected_sha256:
            normalized["sha256"] = expected_sha256
        return normalized
    raise ValueError(
        f"embedding archive is missing or changed: "
        f"{_archive_locator_candidates(manifest_path, row)[0]}"
    )


def _source_embedding_archives(
    roots: Sequence[Path],
    manifests: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    archives: dict[tuple[object, ...], dict[str, object]] = {}
    sha256_cache: dict[Path, str] = {}
    for root, manifest in zip(roots, manifests):
        manifest_path = root / "projection_manifest.json"
        if bool(manifest.get("embedding_arrays_included", True)):
            consolidated = root / "question_embeddings.restricted-local.npz"
            rows: Sequence[Mapping[str, object]] = (
                {
                    "path": str(consolidated),
                    "size_bytes": consolidated.stat().st_size
                    if consolidated.is_file()
                    else -1,
                },
            )
        else:
            inventory = manifest.get("source_embedding_archives", [])
            if not isinstance(inventory, list) or not inventory or not all(
                isinstance(row, dict) for row in inventory
            ):
                raise ValueError(
                    "coordinate-only projection has invalid embedding archive inventory"
                )
            rows = inventory
        for row in rows:
            normalized = _resolve_embedding_archive(
                manifest_path,
                row,
                sha256_cache=sha256_cache,
            )
            identity = (
                ("sha256", normalized["sha256"], normalized["size_bytes"])
                if "sha256" in normalized
                else ("path", normalized["path"], normalized["size_bytes"])
            )
            archives.setdefault(identity, normalized)
    return list(archives.values())


def _merge_projection_view(
    round_roots: Sequence[Path],
    output: Path,
    *,
    view: str,
    candidate_ids: set[str],
    candidate_file: Path,
    include_embedding_arrays: bool,
) -> dict[str, object]:
    roots = [round_root / "projections" / view for round_root in round_roots]
    manifests = [read_json(root / "projection_manifest.json") for root in roots]
    stable = (
        "map_id",
        "map",
        "reference_coordinates",
    )
    for key in stable:
        if any(manifest.get(key) != manifests[0].get(key) for manifest in manifests[1:]):
            raise ValueError(f"partition {view} projection manifests differ on {key}")
    embeddings_config = [_normalized_embedding_manifest(row) for row in manifests]
    if any(row != embeddings_config[0] for row in embeddings_config[1:]):
        raise ValueError(f"partition {view} embedding stacks differ")

    projection_paths = [root / "question_projections.jsonl" for root in roots]
    projection_rows = _merge_rows(
        projection_paths,
        key="candidate_id",
        expected_ids=candidate_ids,
    )
    output.mkdir(parents=True)
    atomic_jsonl(output / "question_projections.jsonl", projection_rows)
    source_embedding_archives = _source_embedding_archives(roots, manifests)
    if include_embedding_arrays:
        embedding_by_id: dict[str, np.ndarray] = {}
        for archive in source_embedding_archives:
            with np.load(str(archive["path"]), allow_pickle=False) as payload:
                ids = [str(value) for value in payload["candidate_ids"]]
                values = np.asarray(payload["embeddings"], dtype=np.float32)
            if len(ids) != len(values) or len(set(ids)) != len(ids):
                raise ValueError(
                    f"partition {view} embedding identities are inconsistent"
                )
            for candidate_id, embedding in zip(ids, values):
                existing = embedding_by_id.get(candidate_id)
                if existing is not None and not np.array_equal(existing, embedding):
                    raise ValueError(
                        f"partition {view} embeddings conflict for {candidate_id}"
                    )
                embedding_by_id[candidate_id] = embedding
        if set(embedding_by_id) != candidate_ids:
            raise ValueError(
                f"merged {view} embeddings do not equal candidate identities"
            )
        ordered_ids = sorted(candidate_ids)
        atomic_npz(
            output / "question_embeddings.restricted-local.npz",
            candidate_ids=np.asarray(ordered_ids),
            embeddings=np.asarray(
                [embedding_by_id[candidate_id] for candidate_id in ordered_ids],
                dtype=np.float32,
            ),
        )
    manifest = {
        "format_version": manifests[0]["format_version"],
        "created_at": _now(),
        "git_commit_sha": manifests[0]["git_commit_sha"],
        "map_id": manifests[0]["map_id"],
        "map": manifests[0]["map"],
        "reference_coordinates": manifests[0]["reference_coordinates"],
        "candidate_files": [_identity(candidate_file)],
        "candidate_count": len(candidate_ids),
        "embedding": embeddings_config[0],
        "embedding_arrays_included": include_embedding_arrays,
        "source_embedding_archives": source_embedding_archives,
        "partition_projection_manifests": [
            _identity(root / "projection_manifest.json") for root in roots
        ],
        "merge_contract": (
            "exact-id-union-with-identical-overlap-v1"
            if include_embedding_arrays
            else "exact-coordinate-id-union-source-embedding-archives-retained-v1"
        ),
    }
    atomic_json(output / "projection_manifest.json", manifest)
    return manifest


def merge_partition_checkpoints(
    partition_roots: Sequence[str | Path],
    output_directory: str | Path,
    *,
    include_embedding_arrays: bool = True,
) -> dict[str, object]:
    roots = [Path(value).resolve() for value in partition_roots]
    output = Path(output_directory).resolve()
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("at least two distinct partition roots are required")
    if output.exists():
        raise ValueError(f"refusing to overwrite partition merge: {output}")
    pipeline_manifests = [read_json(root / "pipeline_manifest.json") for root in roots]
    partition_counts = {int(row.get("work_partition_count", 1)) for row in pipeline_manifests}
    partition_indices = {int(row.get("work_partition_index", 0)) for row in pipeline_manifests}
    partition_salts = {str(row.get("work_partition_salt", "")) for row in pipeline_manifests}
    partition_count = len(roots)
    if (
        partition_counts != {partition_count}
        or partition_indices != set(range(partition_count))
        or len(partition_salts) != 1
    ):
        raise ValueError("partition manifests do not form one complete indexed set")
    stable_keys = (
        "git_commit_sha",
        "plan_manifest_sha256",
        "generator_ids",
        "generator_models",
        "validator_id",
        "validator_model",
        "text_contract",
        "acceptance_contract_version",
        "generation_profile",
        "distance_tolerance",
        "disagreement_weight",
        "refinement_candidates_per_task",
        "refinement_minimum_target_axis_1",
        "refinement_task_priority",
        "master_seed",
        "work_partition_count",
        "work_partition_salt",
        "keyword_section_plan_sha256",
        "initial_candidate_file_list_sha256",
        "initial_logical_round_index",
    )
    for key in stable_keys:
        reference = pipeline_manifests[0].get(key)
        if any(row.get(key) != reference for row in pipeline_manifests[1:]):
            raise ValueError(f"partition pipeline manifests differ on {key}")
    round_roots = [_latest_verified_round(root) for root in roots]

    candidate_source_paths = [
        path for round_root in round_roots for path in _candidate_paths(round_root)
    ]
    candidate_rows = _merge_rows(candidate_source_paths, key="candidate_id")
    candidate_ids = {str(row["candidate_id"]) for row in candidate_rows}
    if not candidate_ids:
        raise ValueError("partition union has no candidates")

    output.mkdir(parents=True)
    candidate_file = output / "candidates.jsonl"
    atomic_jsonl(candidate_file, candidate_rows)
    atomic_json(
        output / "candidates.jsonl.manifest.json",
        {
            "format_version": "readiness-partition-candidate-union-v1",
            "created_at": _now(),
            "candidate_count": len(candidate_rows),
            "source_files": [_identity(path) for path in candidate_source_paths],
            "merge_contract": "exact-id-union-with-identical-overlap-v1",
        },
    )

    validation_paths = [round_root / "validation.jsonl" for round_root in round_roots]
    validation_rows = _merge_rows(
        validation_paths,
        key="candidate_id",
        expected_ids=candidate_ids,
    )
    validation_manifests = [
        read_json(path.with_suffix(path.suffix + ".manifest.json"))
        for path in validation_paths
    ]
    judge_keys = ("judge_id", "judge_model", "judge_backend", "judge_precision")
    for key in judge_keys:
        if any(
            row.get(key) != validation_manifests[0].get(key)
            for row in validation_manifests[1:]
        ):
            raise ValueError(f"partition validation manifests differ on {key}")
    acceptance_contracts = {
        str(row.get("acceptance_contract_version", "question-v1"))
        for row in validation_manifests
    }
    if len(acceptance_contracts) != 1:
        raise ValueError("partition validation manifests differ on acceptance contract")
    validation_file = output / "validation.jsonl"
    atomic_jsonl(validation_file, validation_rows)
    atomic_json(
        output / "validation.jsonl.manifest.json",
        {
            "format_version": validation_manifests[0]["format_version"],
            "completed_at": _now(),
            "git_commit_sha": pipeline_manifests[0]["git_commit_sha"],
            "candidate_files": [_identity(candidate_file)],
            "candidate_count": len(candidate_rows),
            "reviewed_count": len(validation_rows),
            "accepted_count": sum(bool(row["accepted"]) for row in validation_rows),
            **{key: validation_manifests[0][key] for key in judge_keys},
            "acceptance_contract_version": acceptance_contracts.pop(),
            "acceptance_contract": validation_manifests[0]["acceptance_contract"],
            "partition_validation_manifests": [
                _identity(path.with_suffix(path.suffix + ".manifest.json"))
                for path in validation_paths
            ],
            "merge_contract": "exact-id-union-with-identical-overlap-v1",
        },
    )

    projection_manifests = {}
    for view in ("qwen", "mistral"):
        projection_manifests[view] = _merge_projection_view(
            round_roots,
            output / "projections" / view,
            view=view,
            candidate_ids=candidate_ids,
            candidate_file=candidate_file,
            include_embedding_arrays=include_embedding_arrays,
        )

    manifest = {
        "format_version": "readiness-partition-checkpoint-union-v2",
        "created_at": _now(),
        "git_commit_sha": pipeline_manifests[0]["git_commit_sha"],
        "text_contract": pipeline_manifests[0].get(
            "text_contract", "question-v1"
        ),
        "acceptance_contract_version": pipeline_manifests[0].get(
            "acceptance_contract_version", "question-v1"
        ),
        "generation_profile": pipeline_manifests[0].get(
            "generation_profile", "balanced-v1"
        ),
        "refinement_minimum_target_axis_1": pipeline_manifests[0].get(
            "refinement_minimum_target_axis_1"
        ),
        "refinement_task_priority": pipeline_manifests[0].get(
            "refinement_task_priority", "stable-hash"
        ),
        "partition_count": partition_count,
        "partition_salt": partition_salts.pop(),
        "partition_roots": [str(root) for root in roots],
        "partition_pipeline_manifests": [
            _identity(root / "pipeline_manifest.json") for root in roots
        ],
        "partition_round_roots": [str(root) for root in round_roots],
        "candidate_count": len(candidate_rows),
        "maximum_candidate_round_index": max(
            int(row["round_index"]) for row in candidate_rows
        ),
        "accepted_count": sum(bool(row["accepted"]) for row in validation_rows),
        "qwen_map_id": projection_manifests["qwen"]["map_id"],
        "mistral_map_id": projection_manifests["mistral"]["map_id"],
        "embedding_arrays_included": include_embedding_arrays,
        "scientific_guard": (
            "Rows are unioned by immutable candidate id; overlapping candidate, "
            "validation, and projection records must be exactly equal. "
            + (
                "Embedding records are also exact-unioned into consolidated archives."
                if include_embedding_arrays
                else (
                    "Restricted-local embedding arrays are not reconstructed; their "
                    "source archives remain in the immutable partition checkpoints."
                )
            )
        ),
    }
    atomic_json(output / "merge_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--omit-embedding-arrays",
        action="store_true",
        help=(
            "Exact-merge candidates, validations, and both projection-coordinate "
            "views without reconstructing consolidated restricted-local embedding "
            "arrays. Source archives remain in the partition checkpoints."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = merge_partition_checkpoints(
        args.partition_root,
        args.output_dir,
        include_embedding_arrays=not args.omit_embedding_arrays,
    )
    print(
        f"partition_candidates={manifest['candidate_count']} "
        f"accepted={manifest['accepted_count']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
