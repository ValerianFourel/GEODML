from __future__ import annotations

import json
from pathlib import Path

from analysis.scripts.report_agentic_experiment_progress import collect_progress


def _write_shard(
    root: Path,
    model: str,
    shard: int,
    *,
    result_count: int,
    status: str,
    failed_ids: list[str] | None = None,
) -> None:
    shard_root = root / "models" / model / f"shard-{shard}"
    results = shard_root / "results"
    results.mkdir(parents=True)
    for index in range(result_count):
        (results / f"cell-{index}.json").write_text(
            json.dumps({"cell_id": f"{model}-{shard}-{index}"})
        )
    (shard_root / "run_manifest.json").write_text(json.dumps({
        "status": status,
        "completed_count": max(0, result_count - 1),
        "remaining_count": 5 - result_count,
        "failed_cell_ids": failed_ids or [],
    }))


def test_collect_progress_uses_artifacts_and_keeps_judging_separate(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path,
        "qwen38",
        0,
        result_count=5,
        status="complete",
    )
    _write_shard(
        tmp_path,
        "qwen38",
        1,
        result_count=3,
        status="checkpointed",
        failed_ids=["failed-cell"],
    )

    progress = collect_progress(
        tmp_path,
        models=("qwen38", "llama4"),
        shard_count=2,
        cells_per_shard=5,
        recent_window_minutes=15,
        now=10_000.0,
    )

    assert progress["generation"] == {
        "completed": 8,
        "expected": 20,
        "remaining": 12,
        "percent": 40.0,
        "failed_cells": 1,
    }
    assert progress["models"][0]["completed"] == 8
    assert progress["models"][0]["percent"] == 80.0
    assert progress["models"][1]["completed"] == 0
    assert progress["models"][1]["percent"] == 0.0
    assert progress["shards"][1]["manifest_completed"] == 2
    assert progress["shards"][1]["completed"] == 3
    assert progress["judging"]["status"] == "not_planned"
    assert progress["overall_artifact_percent"] is None


def test_collect_progress_includes_materialized_judge_items(tmp_path: Path) -> None:
    _write_shard(
        tmp_path,
        "qwen38",
        0,
        result_count=5,
        status="complete",
    )
    judge_results = tmp_path / "judge-results.jsonl"
    judge_results.write_text('{"task_id":"a"}\n{"task_id":"b"}\n')

    progress = collect_progress(
        tmp_path,
        models=("qwen38",),
        shard_count=1,
        cells_per_shard=5,
        recent_window_minutes=15,
        now=10_000.0,
        judge_results=judge_results,
        expected_judge_items=5,
    )

    assert progress["judging"] == {
        "status": "in_progress",
        "completed": 2,
        "expected": 5,
        "remaining": 3,
        "percent": 40.0,
    }
    assert progress["overall_artifact_percent"] == 70.0
