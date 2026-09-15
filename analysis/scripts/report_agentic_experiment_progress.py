#!/usr/bin/env python3
"""Report artifact-backed progress for an agentic-search experiment."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _percent(completed: int, expected: int) -> float:
    return round(100.0 * completed / expected, 3) if expected else 0.0


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _judge_item_count(path: Path) -> int:
    if path.is_file() and path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if path.is_file() and path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return len(value) if isinstance(value, list) else 1
    if path.is_dir():
        result_root = path / "results"
        search_root = result_root if result_root.is_dir() else path
        return sum(
            1
            for item in search_root.rglob("*.json")
            if "manifest" not in item.name and "identity" not in item.name
        )
    return 0


def collect_progress(
    run_root: Path,
    *,
    models: Sequence[str],
    shard_count: int,
    cells_per_shard: int,
    recent_window_minutes: float,
    now: float | None = None,
    judge_results: Path | None = None,
    expected_judge_items: int | None = None,
) -> dict[str, Any]:
    """Collect progress from result artifacts and advisory manifests."""
    if shard_count <= 0 or cells_per_shard <= 0:
        raise ValueError("shard_count and cells_per_shard must be positive")
    if recent_window_minutes <= 0:
        raise ValueError("recent_window_minutes must be positive")
    if not models:
        raise ValueError("at least one model is required")
    if expected_judge_items is not None and expected_judge_items <= 0:
        raise ValueError("expected_judge_items must be positive")
    if judge_results is not None and expected_judge_items is None:
        raise ValueError("--judge-results requires --expected-judge-items")

    timestamp = time.time() if now is None else now
    recent_cutoff = timestamp - recent_window_minutes * 60.0
    shards: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    recent_result_count = 0

    for model in models:
        model_completed = 0
        model_failures = 0
        for shard_index in range(shard_count):
            shard_root = run_root / "models" / model / f"shard-{shard_index}"
            manifest = _load_manifest(shard_root / "run_manifest.json")
            result_paths = list((shard_root / "results").glob("*.json"))
            artifact_count = len(result_paths)
            completed = min(artifact_count, cells_per_shard)
            failed_ids = manifest.get("failed_cell_ids") or []
            if not isinstance(failed_ids, list):
                failed_ids = []
            recent = sum(
                1
                for path in result_paths
                if recent_cutoff <= path.stat().st_mtime <= timestamp
            )
            recent_result_count += recent
            model_completed += completed
            model_failures += len(failed_ids)
            shards.append({
                "model": model,
                "shard": shard_index,
                "status": manifest.get(
                    "status", "not_started" if not shard_root.exists() else "missing_manifest"
                ),
                "completed": completed,
                "expected": cells_per_shard,
                "remaining": max(0, cells_per_shard - completed),
                "percent": _percent(completed, cells_per_shard),
                "failed_cells": len(failed_ids),
                "manifest_completed": manifest.get("completed_count"),
                "manifest_remaining": manifest.get("remaining_count"),
                "artifact_overflow": max(0, artifact_count - cells_per_shard),
                "recent_results": recent,
            })

        model_expected = shard_count * cells_per_shard
        model_summaries.append({
            "model": model,
            "completed": model_completed,
            "expected": model_expected,
            "remaining": model_expected - model_completed,
            "percent": _percent(model_completed, model_expected),
            "failed_cells": model_failures,
        })

    generation_completed = sum(item["completed"] for item in model_summaries)
    generation_expected = len(models) * shard_count * cells_per_shard
    generation_failures = sum(item["failed_cells"] for item in model_summaries)
    recent_rate = recent_result_count / recent_window_minutes
    generation_remaining = generation_expected - generation_completed
    eta_minutes = generation_remaining / recent_rate if recent_rate > 0 else None

    generation = {
        "completed": generation_completed,
        "expected": generation_expected,
        "remaining": generation_remaining,
        "percent": _percent(generation_completed, generation_expected),
        "failed_cells": generation_failures,
    }
    rate = {
        "window_minutes": recent_window_minutes,
        "recent_results": recent_result_count,
        "cells_per_minute": round(recent_rate, 3),
        "eta_minutes_at_recent_rate": round(eta_minutes, 1) if eta_minutes is not None else None,
    }

    if expected_judge_items is None:
        judging: dict[str, Any] = {"status": "not_planned"}
        overall_percent = None
    else:
        judge_completed = min(
            _judge_item_count(judge_results) if judge_results is not None else 0,
            expected_judge_items,
        )
        judge_remaining = expected_judge_items - judge_completed
        judging = {
            "status": "complete" if judge_remaining == 0 else (
                "in_progress" if judge_completed else "not_started"
            ),
            "completed": judge_completed,
            "expected": expected_judge_items,
            "remaining": judge_remaining,
            "percent": _percent(judge_completed, expected_judge_items),
        }
        overall_percent = _percent(
            generation_completed + judge_completed,
            generation_expected + expected_judge_items,
        )

    return {
        "run_root": str(run_root),
        "generation": generation,
        "models": model_summaries,
        "shards": shards,
        "recent_rate": rate,
        "judging": judging,
        "overall_artifact_percent": overall_percent,
    }


def _format_count(value: int) -> str:
    return f"{value:,}"


def render_text(progress: dict[str, Any]) -> str:
    generation = progress["generation"]
    lines = [
        f"RUN_ROOT={progress['run_root']}",
        (
            "GENERATION="
            f"{_format_count(generation['completed'])}/"
            f"{_format_count(generation['expected'])} "
            f"({generation['percent']:.2f}%) "
            f"remaining={_format_count(generation['remaining'])} "
            f"failed={generation['failed_cells']}"
        ),
    ]
    for model in progress["models"]:
        lines.append(
            f"MODEL={model['model']} "
            f"{_format_count(model['completed'])}/{_format_count(model['expected'])} "
            f"({model['percent']:.2f}%) remaining={_format_count(model['remaining'])} "
            f"failed={model['failed_cells']}"
        )
        for shard in progress["shards"]:
            if shard["model"] != model["model"]:
                continue
            lines.append(
                f"  SHARD={shard['shard']} status={shard['status']} "
                f"{_format_count(shard['completed'])}/{_format_count(shard['expected'])} "
                f"({shard['percent']:.2f}%) remaining={_format_count(shard['remaining'])} "
                f"failed={shard['failed_cells']} recent={shard['recent_results']}"
            )
    rate = progress["recent_rate"]
    lines.append(
        f"RECENT_RATE={rate['cells_per_minute']:.3f} cells/min "
        f"window={rate['window_minutes']:g}min recent_results={rate['recent_results']}"
    )
    if rate["eta_minutes_at_recent_rate"] is not None:
        lines.append(
            "ROUGH_GENERATION_ETA="
            f"{rate['eta_minutes_at_recent_rate']:.1f} minutes at the recent aggregate rate"
        )
    judging = progress["judging"]
    if judging["status"] == "not_planned":
        lines.append("JUDGING=not_planned denominator=undefined")
        lines.append("OVERALL_ARTIFACT_PERCENT=undefined until judge tasks are materialized")
    else:
        lines.append(
            f"JUDGING={judging['completed']}/{judging['expected']} "
            f"({judging['percent']:.2f}%) status={judging['status']} "
            f"remaining={judging['remaining']}"
        )
        lines.append(
            f"OVERALL_ARTIFACT_PERCENT={progress['overall_artifact_percent']:.2f}%"
        )
    lines.append("PROGRESS_JSON=" + json.dumps(progress, sort_keys=True))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--cells-per-shard", type=int, default=1500)
    parser.add_argument("--recent-window-minutes", type=float, default=15.0)
    parser.add_argument("--judge-results", type=Path)
    parser.add_argument("--expected-judge-items", type=int)
    parser.add_argument("--json", action="store_true", dest="json_only")
    return parser


def main() -> int:
    args = _parser().parse_args()
    progress = collect_progress(
        args.run_root,
        models=tuple(args.models or ("qwen38", "llama4")),
        shard_count=args.shard_count,
        cells_per_shard=args.cells_per_shard,
        recent_window_minutes=args.recent_window_minutes,
        judge_results=args.judge_results,
        expected_judge_items=args.expected_judge_items,
    )
    print(json.dumps(progress, indent=2, sort_keys=True) if args.json_only else render_text(progress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
