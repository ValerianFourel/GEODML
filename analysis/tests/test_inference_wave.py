"""Contracts for restartable, variable-width inference waves."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.inference_wave import (
    build_inference_wave,
    write_inference_wave,
)


def test_wave_is_disjoint_complete_balanced_and_deterministic(tmp_path: Path) -> None:
    tasks = [{"task_id": f"task-{index:02d}", "payload": index} for index in range(17)]
    completed = {"task-01", "task-07", "task-09"}
    first = build_inference_wave(
        tasks,
        task_id_field="task_id",
        completed_task_ids=completed,
        worker_count=3,
        master_seed=41,
    )
    repeated = build_inference_wave(
        list(reversed(tasks)),
        task_id_field="task_id",
        completed_task_ids=completed,
        worker_count=3,
        master_seed=41,
    )

    assert first == repeated
    assigned = [row["task_id"] for worker in first.workers for row in worker]
    assert len(assigned) == len(set(assigned)) == 14
    assert set(assigned) == {row["task_id"] for row in tasks} - completed
    counts = [len(worker) for worker in first.workers]
    assert max(counts) - min(counts) <= 1

    artifacts = write_inference_wave(tmp_path / "wave", wave=first)
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["status"] == "planned"
    assert manifest["assignment_policy"] == "stable-hash-order-modulo-v1"
    assert manifest["pending_task_count"] == 14
    assert manifest["worker_task_counts"] == counts
    assert len(artifacts.worker_task_paths) == 3


def test_next_wave_can_change_worker_count_without_repeating_completed() -> None:
    tasks = [{"task_id": f"task-{index:02d}"} for index in range(17)]
    completed = {f"task-{index:02d}" for index in range(11)}
    wave = build_inference_wave(
        tasks,
        task_id_field="task_id",
        completed_task_ids=completed,
        worker_count=5,
        master_seed=41,
    )
    assigned = [row["task_id"] for worker in wave.workers for row in worker]
    assert set(assigned) == {f"task-{index:02d}" for index in range(11, 17)}
    assert len(assigned) == 6


def test_wave_rejects_duplicate_tasks_or_excess_workers() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_inference_wave(
            [{"id": "a"}, {"id": "a"}],
            task_id_field="id",
            completed_task_ids=set(),
            worker_count=1,
        )
    with pytest.raises(ValueError, match="more workers"):
        build_inference_wave(
            [{"id": "a"}],
            task_id_field="id",
            completed_task_ids=set(),
            worker_count=2,
        )


def test_generation_queue_and_cli_wave_support_arbitrary_width(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    selections = tmp_path / "selections.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {
                    "candidate_id": f"prompt-{index}",
                    "question": f"Question {index}?",
                    "keyword": f"keyword-{index}",
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    selections.write_text(
        "".join(
            json.dumps({"candidate_id": f"prompt-{index}", "axis_bin": index}) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    repository = Path(__file__).resolve().parents[2]
    queue_root = tmp_path / "queue"
    prepared = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/prepare_agentic_generation_tasks.py",
            "--prompts-jsonl",
            str(prompts),
            "--selection-records-jsonl",
            str(selections),
            "--prompt-count",
            "2",
            "--output-dir",
            str(queue_root),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr
    assert "CELLS=24" in prepared.stdout

    wave_root = tmp_path / "wave"
    wave = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/prepare_inference_wave.py",
            "--tasks",
            str(queue_root / "tasks.jsonl"),
            "--task-id-field",
            "cell_id",
            "--worker-count",
            "5",
            "--output-dir",
            str(wave_root),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert wave.returncode == 0, wave.stderr
    manifest = json.loads((wave_root / "run_manifest.json").read_text())
    assert manifest["pending_task_count"] == 24
    assert manifest["worker_task_counts"] == [5, 5, 5, 5, 4]
    assert manifest["source_tasks"]["path"] == str(
        (queue_root / "tasks.jsonl").resolve()
    )


def test_wave_cli_reads_completed_judge_outcome_journals(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "".join(
            json.dumps({"judge_task_id": f"judge-{index}"}) + "\n" for index in range(3)
        ),
        encoding="utf-8",
    )
    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "outcomes.jsonl").write_text(
        json.dumps({"judge_task_id": "judge-1"}) + "\n", encoding="utf-8"
    )
    repository = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/prepare_inference_wave.py",
            "--tasks",
            str(tasks),
            "--task-id-field",
            "judge_task_id",
            "--completed-results-root",
            str(completed),
            "--worker-count",
            "2",
            "--output-dir",
            str(tmp_path / "wave"),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "MISSING_TASKS=2" in run.stdout
