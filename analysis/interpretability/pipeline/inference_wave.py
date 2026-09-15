"""Deterministic missing-task waves for restartable parallel inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = "geodml-inference-wave-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class InferenceWave:
    wave_id: str
    format_version: str
    task_id_field: str
    master_seed: int
    source_task_count: int
    source_task_ids_sha256: str
    completed_task_ids_sha256: str
    workers: tuple[tuple[dict[str, Any], ...], ...]

    @property
    def pending_task_count(self) -> int:
        return sum(len(worker) for worker in self.workers)


@dataclass(frozen=True)
class InferenceWaveArtifacts:
    manifest_path: Path
    worker_task_paths: tuple[Path, ...]


def build_inference_wave(
    tasks: Sequence[Mapping[str, Any]],
    *,
    task_id_field: str,
    completed_task_ids: set[str],
    worker_count: int,
    master_seed: int = 20260915,
) -> InferenceWave:
    """Freeze missing tasks and assign them by stable modulo."""

    if not task_id_field.strip():
        raise ValueError("task ID field must be non-empty")
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker count must be a positive integer")
    indexed: dict[str, dict[str, Any]] = {}
    for number, raw in enumerate(tasks, 1):
        row = dict(raw)
        task_id = row.get(task_id_field)
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"task {number} lacks {task_id_field}")
        if task_id in indexed:
            raise ValueError(f"duplicate task ID: {task_id}")
        indexed[task_id] = row
    if not indexed:
        raise ValueError("task queue must not be empty")
    unknown_completed = completed_task_ids - set(indexed)
    if unknown_completed:
        raise ValueError("completed task IDs do not belong to the source queue")
    missing_ids = set(indexed) - completed_task_ids
    if not missing_ids:
        raise ValueError("source queue has no missing tasks")
    if worker_count > len(missing_ids):
        raise ValueError("cannot assign more workers than missing tasks")
    ordered_ids = sorted(
        missing_ids,
        key=lambda task_id: (
            hashlib.sha256(f"{master_seed}:wave:{task_id}".encode()).hexdigest(),
            task_id,
        ),
    )
    workers: list[list[dict[str, Any]]] = [[] for _ in range(worker_count)]
    for position, task_id in enumerate(ordered_ids):
        workers[position % worker_count].append(indexed[task_id])
    identity = {
        "format_version": FORMAT_VERSION,
        "task_id_field": task_id_field,
        "master_seed": master_seed,
        "source_task_ids": sorted(indexed),
        "completed_task_ids": sorted(completed_task_ids),
        "workers": [[row[task_id_field] for row in worker] for worker in workers],
    }
    return InferenceWave(
        wave_id="inference-wave-" + _hash(identity)[:24],
        format_version=FORMAT_VERSION,
        task_id_field=task_id_field,
        master_seed=master_seed,
        source_task_count=len(indexed),
        source_task_ids_sha256=_hash(sorted(indexed)),
        completed_task_ids_sha256=_hash(sorted(completed_task_ids)),
        workers=tuple(tuple(worker) for worker in workers),
    )


def _atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def write_inference_wave(
    output_directory: str | Path,
    *,
    wave: InferenceWave,
    source_tasks_path: Path | None = None,
) -> InferenceWaveArtifacts:
    """Write immutable per-worker queues and their manifest."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite inference wave: {output}")
    worker_root = output / "workers"
    worker_root.mkdir(parents=True)
    worker_paths: list[Path] = []
    worker_artifacts: list[dict[str, Any]] = []
    for index, rows in enumerate(wave.workers):
        path = worker_root / f"worker-{index:05d}.jsonl"
        _atomic_jsonl(path, rows)
        worker_paths.append(path)
        worker_artifacts.append(
            {
                "worker_index": index,
                "task_count": len(rows),
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest_path = output / "run_manifest.json"
    manifest = {
        "wave_id": wave.wave_id,
        "format_version": wave.format_version,
        "status": "planned",
        "scientific_result": False,
        "task_id_field": wave.task_id_field,
        "master_seed": wave.master_seed,
        "assignment_policy": "stable-hash-order-modulo-v1",
        "source_task_count": wave.source_task_count,
        "source_task_ids_sha256": wave.source_task_ids_sha256,
        "completed_task_ids_sha256": wave.completed_task_ids_sha256,
        "pending_task_count": wave.pending_task_count,
        "worker_count": len(wave.workers),
        "worker_task_counts": [len(worker) for worker in wave.workers],
        "workers": worker_artifacts,
    }
    if source_tasks_path is not None:
        source_path = source_tasks_path.resolve()
        manifest["source_tasks"] = {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    _atomic_json(
        manifest_path,
        manifest,
    )
    return InferenceWaveArtifacts(
        manifest_path=manifest_path,
        worker_task_paths=tuple(worker_paths),
    )
