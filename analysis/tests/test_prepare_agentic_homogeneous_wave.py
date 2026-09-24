"""Held Slurm preparation for one-model finite waves."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_segment_plan import build_segment_plan
from analysis.scripts.prepare_agentic_homogeneous_wave import build_submission


def _keywords():
    return [{
        "keyword_id": "keyword-1",
        "priority_rank": 0,
        "completed": 0,
        "active": 0,
        "blocked": 0,
        "eligible_remaining": 240,
        "estimated_remaining_seconds": 72000,
        "task_set_ref": "task-set-keyword-1",
    }]


def _plan(model="qwen38", interactive=0):
    return build_segment_plan(
        audit_id="audit-1",
        ledger_sequence=1,
        acceptance_policy_id="experiment-v2",
        model=model,
        keyword_rows=_keywords(),
        segment_count=20,
        interactive_segment_count=interactive,
        approval_status="approved",
    ).to_dict()


def _build(tmp_path: Path, plan, *, empty_segments=()):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    runner = tmp_path / "runner.sbatch"
    runner.write_text("#!/bin/bash\n")
    repository = tmp_path / "repository"
    repository.mkdir(exist_ok=True)
    materialization = tmp_path / "materialization"
    waves = []
    for segment in plan["segments"]:
        wave = materialization / "segments" / segment["segment_id"] / "wave"
        wave.mkdir(parents=True, exist_ok=True)
        (wave / "run_manifest.json").write_text("{}")
        task_count = int(segment["segment_id"] not in empty_segments)
        waves.append({
            "segment_id": segment["segment_id"],
            "wave_root": str(wave),
            "primary_keyword_id": "keyword-1" if task_count else None,
            "backlog": {"task_count": task_count},
        })
    materialization.mkdir(exist_ok=True)
    (materialization / "materialization.json").write_text(json.dumps({
        "plan_id": plan["plan_id"], "model": plan["model"], "waves": waves,
    }))
    runtime = {
        "ACL_ARR_VENV": "/cluster/venv",
        "GEODML_CACHE_ROOT": "/cluster/cache",
        "GEODML_WORKER_LAUNCHER": "/cluster/worker.sh",
    }
    if plan["model"] in {"qwen38", "llama4"}:
        runtime.update({
            "SEARCH_AGENTIC_PROFILE": "/cluster/profile.json",
            "SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT": "/cluster/cross-encoder",
            "SEARCH_AGENTIC_CROSS_ENCODER_REVISION": "revision",
            "SEARCH_AGENTIC_DDG_SNAPSHOT": "/cluster/ddg.parquet",
            "SEARCH_AGENTIC_SEARXNG_SNAPSHOT": "/cluster/searxng.parquet",
            "SEARCH_AGENTIC_PROMPTS_JSONL": "/cluster/prompts.jsonl",
            "SEARCH_AGENTIC_SELECTION_RECORDS_JSONL": "/cluster/records.jsonl",
            "SEARCH_AGENTIC_PROMPT_COUNT": "26009",
        })
    else:
        runtime.update({
            "GEODML_JUDGE_MANIFEST": "/cluster/judge.json",
            "GEODML_JUDGE_ROLE": "bulk",
            "GEODML_JUDGE_PROFILE": "/cluster/nemotron.json",
        })
    return build_submission(
        plan,
        plan_path=plan_path,
        execution_repository=repository,
        runner=runner,
        account="project_123",
        partition="booster",
        output_root=tmp_path / "wave-output",
        dataset_root=tmp_path / "dataset",
        materialization_root=materialization,
        runtime_environment=runtime,
        execution_commit="a" * 40,
        allocation_estimate="20 one-hour Qwen segments; maximum 80 GPU-hours",
    )


def test_twenty_qwen_jobs_are_independent_held_and_never_alternate(tmp_path):
    bundle = _build(tmp_path, _plan())
    assert bundle["batch_job_count"] == 20
    assert bundle["maximum_concurrent_allocations"] == 5
    assert bundle["minimum_observed_start_gap_seconds"] == 600
    assert {job["model"] for job in bundle["jobs"]} == {"qwen38"}
    assert all("--hold" in job["command"] for job in bundle["jobs"])
    assert all("--no-requeue" in job["command"] for job in bundle["jobs"])
    assert all("--time=01:00:00" in job["command"] for job in bundle["jobs"])
    assert all(not any(arg.startswith("--array") for arg in job["command"])
               for job in bundle["jobs"])
    assert len({job["segment_id"] for job in bundle["jobs"]}) == 20
    exports = [
        next(arg for arg in job["command"] if arg.startswith("--export="))
        for job in bundle["jobs"]
    ]
    assert all("GEODML_WORKER_INDEX=0" in value for value in exports)
    assert all("GEODML_WORKER_COUNT=1" in value for value in exports)


def test_interactive_segments_are_reserved_without_becoming_batch_jobs(tmp_path):
    bundle = _build(tmp_path, _plan(model="llama4", interactive=2))
    assert bundle["batch_job_count"] == 18
    assert bundle["interactive_segment_count"] == 2
    assert {job["model"] for job in bundle["jobs"]} == {"llama4"}
    assert {row["model"] for row in bundle["interactive_segments"]} == {"llama4"}
    assert all(
        row["allocation_command"][0] == "salloc"
        and "--time=01:00:00" in row["allocation_command"]
        and row["step_command"][0] == "srun"
        and row["environment"]["GEODML_WORKER_COUNT"] == "1"
        for row in bundle["interactive_segments"]
    )


def test_empty_keyword_segments_do_not_create_allocations(tmp_path):
    plan = _plan(model="llama4", interactive=2)
    empty = {
        plan["segments"][0]["segment_id"],
        plan["segments"][-1]["segment_id"],
    }
    bundle = _build(tmp_path, plan, empty_segments=empty)
    assert bundle["batch_job_count"] == 17
    assert bundle["interactive_segment_count"] == 1
    assert bundle["exhausted_segment_count"] == 2
    assert {row["segment_id"] for row in bundle["exhausted_segments"]} == empty


def test_mixed_or_unapproved_wave_fails_before_creating_commands(tmp_path):
    mixed = _plan()
    mixed["segments"][3]["model"] = "llama4"
    with pytest.raises(ValueError, match="exactly one model"):
        _build(tmp_path, mixed)
    unapproved = _plan()
    unapproved["segments"][0]["approval_status"] = "proposed"
    with pytest.raises(ValueError, match="explicitly approved"):
        _build(tmp_path, unapproved)


def test_cli_is_dry_run_and_writes_no_submission_state(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(model="nemotron")))
    runner = tmp_path / "runner.sbatch"
    runner.write_text("#!/bin/bash\n")
    repository = tmp_path / "repository"
    repository.mkdir()
    materialization = tmp_path / "materialization"
    waves = []
    plan = json.loads(plan_path.read_text())
    for segment in plan["segments"]:
        wave = materialization / "segments" / segment["segment_id"] / "wave"
        wave.mkdir(parents=True)
        (wave / "run_manifest.json").write_text("{}")
        waves.append({
            "segment_id": segment["segment_id"],
            "wave_root": str(wave),
            "primary_keyword_id": "keyword-1",
            "backlog": {"task_count": 1},
        })
    (materialization / "materialization.json").write_text(json.dumps({
        "plan_id": plan["plan_id"], "model": "nemotron", "waves": waves,
    }))
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({
        "ACL_ARR_VENV": "/cluster/venv",
        "GEODML_CACHE_ROOT": "/cluster/cache",
        "GEODML_WORKER_LAUNCHER": "/cluster/worker.sh",
        "GEODML_JUDGE_MANIFEST": "/cluster/judge.json",
        "GEODML_JUDGE_ROLE": "bulk",
        "GEODML_JUDGE_PROFILE": "/cluster/nemotron.json",
    }))
    output = tmp_path / "wave-output"
    command = [
        sys.executable,
        "analysis/scripts/prepare_agentic_homogeneous_wave.py",
        "--plan", str(plan_path),
        "--execution-repository", str(repository),
        "--runner", str(runner),
        "--account", "project_123",
        "--partition", "booster",
        "--output-root", str(output),
        "--dataset-root", str(tmp_path / "dataset"),
        "--materialization-root", str(materialization),
        "--runtime-environment", str(runtime),
        "--execution-commit", "a" * 40,
        "--allocation-estimate", "20 one-hour Nemotron segments",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["dry_run"] is True
    assert not output.exists()
