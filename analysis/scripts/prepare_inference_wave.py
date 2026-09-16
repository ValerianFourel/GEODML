#!/usr/bin/env python3
"""Partition only missing inference tasks into a deterministic worker wave."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.inference_wave import (  # noqa: E402
    build_inference_wave,
    write_inference_wave,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            rows.append(value)
    return rows


def _completed_from_jsonl(paths: list[Path], id_field: str) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        for row in _read_jsonl(path):
            task_id = row.get(id_field)
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"completed row lacks {id_field}: {path}")
            completed.add(task_id)
    return completed


def _completed_from_roots(roots: list[Path], id_field: str) -> set[str]:
    completed: set[str] = set()
    for root in roots:
        outcome_paths = sorted(root.rglob("outcomes.jsonl"))
        rows = [row for path in outcome_paths for row in _read_jsonl(path)]
        result_roots = (
            [root / "results"]
            if (root / "results").is_dir()
            else sorted(path for path in root.rglob("results") if path.is_dir())
        )
        if not outcome_paths:
            for result_root in result_roots:
                for path in sorted(result_root.glob("*.json")):
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError(f"completed result must be an object: {path}")
                    rows.append(value)
        if not outcome_paths and not result_roots and root.is_dir():
            for path in sorted(root.glob("*.json")):
                if path.name in {"run_manifest.json", "config.json"}:
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"completed result must be an object: {path}")
                rows.append(value)
        for row in rows:
            task_id = row.get(id_field)
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"completed result lacks {id_field}: {root}")
            completed.add(task_id)
    return completed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id-field", required=True)
    parser.add_argument("--completed-jsonl", type=Path, action="append", default=[])
    parser.add_argument(
        "--completed-results-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument(
        "--dispatch-mode", choices=("backlog", "partition"), default="backlog",
        help="Share the frozen backlog with task claims, or preserve legacy static partitions",
    )
    parser.add_argument("--master-seed", type=int, default=20260915)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        tasks = _read_jsonl(arguments.tasks.resolve())
        completed = _completed_from_jsonl(
            [path.resolve() for path in arguments.completed_jsonl],
            arguments.task_id_field,
        )
        completed.update(
            _completed_from_roots(
                [path.resolve() for path in arguments.completed_results_root],
                arguments.task_id_field,
            )
        )
        wave = build_inference_wave(
            tasks,
            task_id_field=arguments.task_id_field,
            completed_task_ids=completed,
            worker_count=arguments.worker_count,
            master_seed=arguments.master_seed,
            dispatch_mode=arguments.dispatch_mode,
        )
        artifacts = write_inference_wave(
            arguments.output_dir,
            wave=wave,
            source_tasks_path=arguments.tasks,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(f"WAVE_ID={wave.wave_id}")
    print(f"MISSING_TASKS={wave.pending_task_count}")
    print(f"WORKERS={len(wave.workers)}")
    print(f"MANIFEST={artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
