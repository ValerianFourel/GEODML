#!/usr/bin/env python3
"""Verify that a readiness projection belongs to an immutable candidate set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def _file_content_identity(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def _manifest_content_identity(row: Mapping[str, object]) -> tuple[str, int]:
    sha256 = str(row.get("sha256", ""))
    size_bytes = row.get("size_bytes")
    if len(sha256) != 64 or size_bytes is None:
        raise ValueError("candidate file manifest lacks a content identity")
    return sha256, int(size_bytes)


def verify_projection_checkpoint(
    projection_manifest: str | Path,
    *,
    expected_count: int,
    candidate_file_list: str | Path,
    expected_attention: str,
    allow_coordinate_only: bool = False,
) -> None:
    """Fail unless a projection matches candidate content and embedding settings.

    Candidate paths are provenance locators, not content identity.  Atomic checkpoint
    finalization can move an otherwise immutable artifact tree, so relocation is
    accepted only when every candidate file retains its exact SHA-256 and byte size.
    """

    manifest_path = Path(projection_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_paths = [
        Path(value).resolve()
        for value in Path(candidate_file_list)
        .read_text(encoding="utf-8")
        .splitlines()
        if value.strip()
    ]
    if not candidate_paths:
        raise ValueError("candidate file list is empty")

    actual_rows = manifest.get("candidate_files")
    if not isinstance(actual_rows, list) or not all(
        isinstance(row, dict) for row in actual_rows
    ):
        raise ValueError("projection manifest has invalid candidate_files")

    expected_identities = [_file_content_identity(path) for path in candidate_paths]
    actual_identities = [_manifest_content_identity(row) for row in actual_rows]
    if actual_identities != expected_identities:
        raise ValueError("projection candidate content identity differs")
    if int(manifest.get("candidate_count", -1)) != expected_count:
        raise ValueError("projection candidate count differs")

    embedding = manifest.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("projection embedding manifest is invalid")
    attention = str(embedding.get("attention_implementation", "eager"))
    if attention != expected_attention:
        raise ValueError("projection attention implementation differs")
    coordinate_only = manifest.get("embedding_arrays_included") is False
    if coordinate_only and not allow_coordinate_only:
        raise ValueError("projection checkpoint omits consolidated embedding arrays")
    if coordinate_only:
        archives = manifest.get("source_embedding_archives")
        if not isinstance(archives, list) or not archives:
            raise ValueError(
                "coordinate-only projection lacks source embedding archives"
            )
        for row in archives:
            if not isinstance(row, dict):
                raise ValueError("source embedding archive inventory is invalid")
            path = Path(str(row.get("path", "")))
            try:
                expected_size = int(row.get("size_bytes", -1))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "source embedding archive inventory is invalid"
                ) from exc
            if (
                not path.is_file()
                or expected_size <= 0
                or path.stat().st_size != expected_size
            ):
                raise ValueError(
                    f"source embedding archive is missing or changed: {path}"
                )
        incremental = manifest.get("incremental_reuse", {})
        embedded_count = (
            int(incremental.get("embedded_candidate_count", 0))
            if isinstance(incremental, dict)
            else 0
        )
        new_archive = manifest.get("new_embedding_archive")
        if embedded_count > 0:
            if not isinstance(new_archive, dict) or new_archive not in archives:
                raise ValueError(
                    "coordinate-only projection lacks its new embedding archive"
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-manifest", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--candidate-file-list", required=True)
    parser.add_argument("--expected-attention", required=True)
    parser.add_argument("--allow-coordinate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verify_projection_checkpoint(
            args.projection_manifest,
            expected_count=args.expected_count,
            candidate_file_list=args.candidate_file_list,
            expected_attention=args.expected_attention,
            allow_coordinate_only=args.allow_coordinate_only,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
