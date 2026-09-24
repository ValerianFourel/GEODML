"""Admission honors actual starts, capacity, and interactive reservations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_admission import choose_release
from analysis.interpretability.pipeline.agentic_segment_plan import build_segment_plan


def _plan(*, interactive=1, approved=True):
    return build_segment_plan(
        audit_id="audit",
        ledger_sequence=1,
        acceptance_policy_id="policy",
        model="qwen38",
        keyword_rows=[{
            "keyword_id": "topic",
            "priority_rank": 0,
            "completed": 0,
            "active": 0,
            "blocked": 0,
            "eligible_remaining": 10,
            "estimated_remaining_seconds": 100,
            "task_set_ref": "tasks",
        }],
        segment_count=6,
        interactive_segment_count=interactive,
        approval_status="approved" if approved else "proposed",
    ).to_dict()


def _held_jobs(plan):
    return [
        {
            "job_id": str(100 + row["ordinal"]),
            "segment_id": row["segment_id"],
            "model": plan["model"],
            "state": "PENDING",
            "held": True,
            "start_epoch": None,
        }
        for row in plan["segments"]
        if row["mode"] == "batch"
    ]


def test_delayed_start_restarts_gap_and_only_one_released_pending_is_allowed():
    plan = _plan(interactive=0)
    jobs = _held_jobs(plan)
    jobs[0].update(state="RUNNING", held=False, start_epoch=1500)
    assert choose_release(plan=plan, jobs=jobs, now_epoch=2099).reason == "observed_start_gap"
    release = choose_release(plan=plan, jobs=jobs, now_epoch=2100)
    assert release.action == "release"
    jobs[1].update(held=False)
    assert choose_release(plan=plan, jobs=jobs, now_epoch=3000).reason == "released_job_start_unconfirmed"


def test_controller_restart_preserves_latest_observed_start():
    plan = _plan(interactive=0)
    decision = choose_release(
        plan=plan,
        jobs=_held_jobs(plan),
        now_epoch=1599,
        controller_state={"latest_observed_start_epoch": 1000},
    )
    assert decision.action == "wait"
    assert decision.next_start_not_before_epoch == 1600


def test_controller_never_releases_twice_before_first_release_has_started():
    plan = _plan(interactive=0)
    jobs = _held_jobs(plan)
    state = {"unconfirmed_release_job_id": jobs[0]["job_id"]}
    assert choose_release(
        plan=plan, jobs=jobs, now_epoch=5000, controller_state=state
    ).reason == "released_job_start_unconfirmed"
    assert choose_release(
        plan=plan, jobs=jobs[1:], now_epoch=5000, controller_state=state
    ).reason == "released_job_missing_unconfirmed"
    jobs[0].update(state="RUNNING", held=False, start_epoch=4900)
    assert choose_release(
        plan=plan, jobs=jobs, now_epoch=5499, controller_state=state
    ).reason == "observed_start_gap"


def test_five_active_allocations_block_and_four_active_allow_interactive_reservation():
    plan = _plan(interactive=1)
    jobs = _held_jobs(plan)
    jobs.extend({
        "job_id": f"other-{index}",
        "state": "RUNNING",
        "held": False,
        "start_epoch": 100 + index,
    } for index in range(4))
    decision = choose_release(plan=plan, jobs=jobs, now_epoch=1000)
    interactive_id = plan["segments"][-1]["segment_id"]
    assert decision.action == "reserve_interactive"
    assert decision.segment_id == interactive_id
    jobs.append({
        "job_id": "other-4", "state": "RUNNING", "held": False, "start_epoch": 200,
    })
    assert choose_release(plan=plan, jobs=jobs, now_epoch=1000).reason == "concurrency_limit"


def test_interactive_reservation_blocks_batch_until_scheduler_observes_it():
    plan = _plan(interactive=1)
    jobs = _held_jobs(plan)[:-1]
    jobs.extend({
        "job_id": f"other-{index}",
        "state": "RUNNING",
        "held": False,
        "start_epoch": 100 + index,
    } for index in range(4))
    reserved = choose_release(plan=plan, jobs=jobs, now_epoch=1000)
    assert reserved.action == "reserve_interactive"
    interactive_id = plan["segments"][-1]["segment_id"]
    state = {"interactive_reservation_segment_id": interactive_id}
    waiting = choose_release(
        plan=plan, jobs=jobs, now_epoch=1000, controller_state=state
    )
    assert waiting.action == "wait"
    assert waiting.reason == "interactive_reservation_unconfirmed"
    jobs.append({
        "job_id": "interactive-1",
        "segment_id": interactive_id,
        "model": "qwen38",
        "state": "PENDING",
        "held": False,
        "start_epoch": None,
    })
    observed = choose_release(
        plan=plan, jobs=jobs, now_epoch=1000, controller_state=state
    )
    assert observed.action == "wait"
    assert observed.reason == "released_job_start_unconfirmed"


def test_completed_interactive_allows_next_batch_release():
    plan = _plan(interactive=1)
    jobs = _held_jobs(plan)
    interactive_id = plan["segments"][-1]["segment_id"]
    completed = [interactive_id]
    release = choose_release(
        plan=plan,
        jobs=jobs,
        now_epoch=1000,
        completed_segment_ids=completed,
    )
    assert release.action == "release"


def test_interactive_reservation_dry_run_does_not_create_controller_files(tmp_path):
    plan = _plan(interactive=1)
    plan_path = tmp_path / "plan.json"
    snapshot_path = tmp_path / "scheduler.json"
    control = tmp_path / "control"
    plan_path.write_text(json.dumps(plan))
    snapshot_path.write_text(json.dumps({
        "snapshot_id": "snapshot",
        "complete": True,
        "jobs": _held_jobs(plan),
    }))
    completed = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/manage_agentic_wave.py",
            "--plan", str(plan_path),
            "--scheduler-snapshot", str(snapshot_path),
            "--controller-root", str(control),
            "--now-epoch", "1000",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["decision"]["action"] == "reserve_interactive"
    assert not control.exists()


def test_apply_admission_persists_interactive_reservation_without_launching(tmp_path):
    plan = _plan(interactive=1)
    plan_path = tmp_path / "plan.json"
    snapshot_path = tmp_path / "scheduler.json"
    storage_path = tmp_path / "storage.json"
    control = tmp_path / "control"
    plan_path.write_text(json.dumps(plan))
    snapshot_path.write_text(json.dumps({
        "snapshot_id": "snapshot",
        "captured_at_epoch": 1000,
        "complete": True,
        "jobs": _held_jobs(plan),
        "completed_segment_ids": [],
    }))
    storage_path.write_text(json.dumps({
        "format_version": "geodml-agentic-storage-snapshot-v1",
        "captured_at_epoch": 1000,
        "safe_to_admit": True,
    }))
    completed = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/manage_agentic_wave.py",
            "--plan", str(plan_path),
            "--scheduler-snapshot", str(snapshot_path),
            "--controller-root", str(control),
            "--storage-snapshot", str(storage_path),
            "--now-epoch", "1000",
            "--apply-admission",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    interactive_id = plan["segments"][-1]["segment_id"]
    state = json.loads((control / "controller-state.json").read_text())
    assert state["interactive_reservation_segment_id"] == interactive_id
    reservation = control / f"interactive-reservation-{interactive_id}.json"
    assert json.loads(reservation.read_text())["status"] == "reserved"


def test_mixed_model_plan_and_duplicate_live_segment_fail_closed():
    plan = _plan()
    jobs = _held_jobs(plan)
    plan["segments"][0]["model"] = "llama4"
    with pytest.raises(ValueError, match="mixed"):
        choose_release(plan=plan, jobs=jobs, now_epoch=1000)
    plan = _plan()
    jobs = _held_jobs(plan)
    jobs.append(dict(jobs[0], job_id="duplicate"))
    with pytest.raises(ValueError, match="multiple live jobs"):
        choose_release(plan=plan, jobs=jobs, now_epoch=1000)


def test_cli_is_dry_run_and_does_not_create_controller_files(tmp_path: Path):
    plan = _plan(interactive=0)
    plan_path = tmp_path / "plan.json"
    snapshot_path = tmp_path / "scheduler.json"
    control = tmp_path / "control"
    plan_path.write_text(json.dumps(plan))
    snapshot_path.write_text(json.dumps({
        "snapshot_id": "snapshot",
        "complete": True,
        "jobs": _held_jobs(plan),
    }))
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/manage_agentic_wave.py",
            "--plan", str(plan_path),
            "--scheduler-snapshot", str(snapshot_path),
            "--controller-root", str(control),
            "--now-epoch", "1000",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["decision"]["action"] == "release"
    assert not control.exists()
