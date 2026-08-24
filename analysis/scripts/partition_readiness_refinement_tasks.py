#!/usr/bin/env python3
"""Create one immutable, keyword-disjoint refinement batch for a work partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def target_partition(
    row: Mapping[str, object],
    *,
    partition_count: int,
    partition_salt: str,
) -> int:
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("refinement task lacks a target object")
    keyword_id = str(row.get("keyword_id", ""))
    target_id = str(target.get("target_id", ""))
    if not keyword_id or not target_id:
        raise ValueError("refinement task lacks keyword_id or target_id")
    # Keep every target intensity for one keyword on the same producer.  The
    # target id remains a required task-identity field, but must not influence
    # ownership or the two producers could generate prompts for one keyword.
    digest = hashlib.sha256(f"{partition_salt}\0{keyword_id}".encode()).hexdigest()
    return int(digest[:16], 16) % partition_count


def select_partition_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    source_sha256: str,
    limit: int,
    partition_count: int,
    partition_index: int,
    partition_salt: str,
) -> list[Mapping[str, object]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if partition_count <= 0 or not 0 <= partition_index < partition_count:
        raise ValueError("partition must satisfy 0 <= index < count")
    owned = [
        row
        for row in rows
        if target_partition(
            row,
            partition_count=partition_count,
            partition_salt=partition_salt,
        )
        == partition_index
    ]
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in owned:
        generator_id = str(row.get("generator_id", ""))
        if not generator_id:
            raise ValueError("refinement task lacks generator_id")
        grouped.setdefault(generator_id, []).append(row)
    for generator_rows in grouped.values():
        generator_rows.sort(
            key=lambda row: hashlib.sha256(
                f'{source_sha256}:{row["task_id"]}'.encode()
            ).hexdigest()
        )
    selected: list[Mapping[str, object]] = []
    generators = sorted(grouped)
    while len(selected) < min(limit, len(owned)):
        progressed = False
        for generator_id in generators:
            if grouped[generator_id] and len(selected) < limit:
                selected.append(grouped[generator_id].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def prepare_partition_batch(
    source: str | Path,
    output: str | Path,
    *,
    limit: int,
    partition_count: int,
    partition_index: int,
    partition_salt: str,
) -> dict[str, object]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("refinement task source contains missing or duplicate task ids")
    selected = select_partition_batch(
        rows,
        source_sha256=source_sha256,
        limit=limit,
        partition_count=partition_count,
        partition_index=partition_index,
        partition_salt=partition_salt,
    )
    owned_count = sum(
        target_partition(
            row,
            partition_count=partition_count,
            partition_salt=partition_salt,
        )
        == partition_index
        for row in rows
    )
    identity = {
        "format_version": "readiness-refinement-task-batch-v3",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_task_count": len(rows),
        "task_limit": limit,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "partition_salt": partition_salt,
        "owned_source_task_count": owned_count,
        "selected_task_count": len(selected),
        "selected_task_ids": [str(row["task_id"]) for row in selected],
        "selection_method": "stable-keyword-hash-partition-generator-round-robin-v3",
    }
    if output_path.exists() or manifest_path.exists():
        if not output_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"partial refinement task batch: {output_path}")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"immutable refinement task batch differs: {output_path}")
        actual_ids = [
            str(json.loads(line)["task_id"])
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if actual_ids != identity["selected_task_ids"]:
            raise ValueError(f"refinement task batch content differs: {output_path}")
        return identity

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-salt", default="readiness-target-partition-v1")
    return parser


def main() -> int:
    args = _parser().parse_args()
    identity = prepare_partition_batch(
        args.source_tasks,
        args.output,
        limit=args.limit,
        partition_count=args.partition_count,
        partition_index=args.partition_index,
        partition_salt=args.partition_salt,
    )
    print(
        "REFINEMENT BATCH: "
        f"partition={identity['partition_index']}/{identity['partition_count']} "
        f"selected={identity['selected_task_count']}/"
        f"{identity['owned_source_task_count']} "
        f"global_missing={identity['source_task_count']} "
        f"limit={identity['task_limit']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
