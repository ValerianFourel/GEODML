"""Finite homogeneous waves and dry-run incremental replanning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_segment_plan import (
    build_segment_plan,
)


def _keywords():
    return [
        {
            "keyword_id": "far",
            "priority_rank": 1,
            "completed": 2,
            "active": 0,
            "blocked": 0,
            "eligible_remaining": 100,
            "estimated_remaining_seconds": 9000,
            "task_set_ref": "tasks-far",
        },
        {
            "keyword_id": "blocked",
            "priority_rank": 2,
            "completed": 8,
            "active": 0,
            "blocked": 4,
            "eligible_remaining": 0,
            "estimated_remaining_seconds": None,
            "task_set_ref": "tasks-blocked",
        },
        {
            "keyword_id": "near",
            "priority_rank": 0,
            "completed": 98,
            "active": 0,
            "blocked": 0,
            "eligible_remaining": 2,
            "estimated_remaining_seconds": 300,
            "task_set_ref": "tasks-near",
        },
    ]


def test_twenty_segment_qwen_wave_is_homogeneous_and_opposite_direction():
    plan = build_segment_plan(
        audit_id="audit-1",
        ledger_sequence=42,
        acceptance_policy_id="pilot-v2",
        model="qwen38",
        keyword_rows=_keywords(),
        segment_count=20,
        interactive_segment_count=2,
    )

    assert plan.segment_count == 20
    assert plan.maximum_gpu_hours == 80
    assert plan.max_concurrent_allocations == 5
    assert plan.start_gap_seconds == 600
    assert plan.drain_seconds == 300
    assert {segment.model for segment in plan.segments} == {"qwen38"}
    assert {segment.stage for segment in plan.segments} == {"generation"}
    assert [row.keyword_id for row in plan.keyword_priority] == ["near", "far", "blocked"]
    assert all(segment.mode == "batch" for segment in plan.segments[:18])
    assert all(segment.direction == "forward" for segment in plan.segments[:18])
    assert all(segment.mode == "interactive" for segment in plan.segments[18:])
    assert plan.segments[18].keyword_ids == ("blocked", "far", "near")


@pytest.mark.parametrize("model", ["qwen38", "llama4", "nemotron"])
def test_each_supported_wave_has_exactly_one_model(model):
    plan = build_segment_plan(
        audit_id="audit",
        ledger_sequence=1,
        acceptance_policy_id="policy",
        model=model,
        keyword_rows=_keywords(),
        segment_count=3,
    )
    assert {segment.model for segment in plan.segments} == {model}


def test_plan_rejects_long_or_overwide_or_mixed_model_requests():
    arguments = {
        "audit_id": "audit",
        "ledger_sequence": 1,
        "acceptance_policy_id": "policy",
        "model": "qwen38,llama4",
        "keyword_rows": _keywords(),
        "segment_count": 20,
    }
    with pytest.raises(ValueError, match="unsupported model"):
        build_segment_plan(**arguments)
    arguments["model"] = "qwen38"
    arguments["walltime_seconds"] = 3601
    with pytest.raises(ValueError, match="one-hour"):
        build_segment_plan(**arguments)
    arguments["walltime_seconds"] = 3600
    arguments["max_concurrent_allocations"] = 6
    with pytest.raises(ValueError, match="five-allocation"):
        build_segment_plan(**arguments)
    arguments["max_concurrent_allocations"] = 5
    arguments["drain_seconds"] = 3600
    with pytest.raises(ValueError, match="drain_seconds"):
        build_segment_plan(**arguments)


def test_updater_defaults_to_dry_run_and_is_idempotent(tmp_path: Path):
    audit = tmp_path / "audit.json"
    wave = tmp_path / "wave.json"
    audit.write_text(json.dumps({
        "audit_id": "audit-1",
        "ledger_sequence": 42,
        "acceptance_policy_id": "pilot-v2",
        "keywords": _keywords(),
        "throughput": {
            "measurements": [],
            "aggregate": {
                "measurement_count": 2,
                "minimum_seconds_per_task": 10,
                "median_seconds_per_task": 15,
                "maximum_seconds_per_task": 20,
            },
            "groups": [],
        },
    }))
    wave.write_text(json.dumps({
        "model": "llama4",
        "segment_count": 20,
        "interactive_segment_count": 1,
    }))
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "analysis/scripts/update_agentic_segments.py",
        "--audit", str(audit),
        "--wave", str(wave),
    ]
    first = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False
    )
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert not list(tmp_path.glob("plan*"))
    payload = json.loads(first.stdout)
    assert payload["plan"]["model"] == "llama4"
    assert payload["plan"]["segment_count"] == 20
    assert payload["plan"]["segments"][0]["estimated_capacity_tasks"] == {
        "conservative": 102,
        "expected": 102,
        "optimistic": 102,
    }
    estimate = payload["plan"]["remaining_resource_estimate"]
    assert estimate["eligible_tasks"] == 102
    assert estimate["allocation_hours"]["expected_hours"] == pytest.approx(0.425)
    assert estimate["gpu_hours"]["expected_hours"] == pytest.approx(1.7)


def test_updater_writes_versioned_plan_without_overwriting(tmp_path: Path):
    audit = tmp_path / "audit.json"
    wave = tmp_path / "wave.json"
    output = tmp_path / "plan-output"
    audit.write_text(json.dumps({
        "audit_id": "audit-1",
        "ledger_sequence": 42,
        "acceptance_policy_id": "pilot-v2",
        "keywords": _keywords(),
    }))
    wave.write_text(json.dumps({"model": "nemotron", "segment_count": 2}))
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "analysis/scripts/update_agentic_segments.py",
        "--audit", str(audit),
        "--wave", str(wave),
        "--write-plan", str(output),
    ]
    first = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False
    )
    repeated = subprocess.run(
        command, cwd=repository, text=True, capture_output=True, check=False
    )
    assert first.returncode == 0
    assert (output / "plan.json").is_file()
    assert (output / "segments.csv").is_file()
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
