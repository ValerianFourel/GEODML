"""Scheduler snapshots map held, running, and terminal homogeneous segments."""

from __future__ import annotations

from analysis.scripts.capture_agentic_scheduler_snapshot import capture


def test_capture_counts_other_geodml_allocations_and_maps_plan_history():
    plan_id = "segment-plan-" + "a" * 24
    segment_id = "segment-" + "b" * 24
    comment = f"geodml-v2:{plan_id}:{segment_id}:qwen38:batch"
    outputs = iter([
        "\n".join([
            f"101|geodml-qwen38-000|PENDING|N/A|JobHeldUser|{comment}",
            "102|geodml-legacy|RUNNING|2026-09-23T10:00:00|None|",
            "103|unrelated|RUNNING|2026-09-23T10:00:00|None|",
        ]),
        f"100|geodml-qwen38-000|FAILED|2026-09-23T09:00:00|{comment}|\n",
    ])
    snapshot = capture(
        plan={"plan_id": plan_id},
        since="2026-09-23",
        runner=lambda command: next(outputs),
    )
    assert snapshot["complete"] is True
    assert [job["job_id"] for job in snapshot["jobs"]] == ["101", "102"]
    assert snapshot["jobs"][0]["held"] is True
    assert snapshot["completed_segment_ids"] == [segment_id]
    assert {row["owner_id"] for row in snapshot["owners"]} == {
        "job100-worker0", "judge-100-0"
    }


def test_live_qwen_owner_overrides_stale_terminal_accounting():
    outputs = iter([
        '101|geodml-qwen-bout-0001|RUNNING|2026-10-02T10:00:00|None|',
        '101|geodml-qwen-bout-0001|FAILED|2026-10-02T09:00:00||\n'
        '102|geodml-qwen-bout-0002|TIMEOUT|2026-10-02T09:00:00||',
    ])
    snapshot = capture(plan={'plan_id': 'recovery'}, since='2026-10-02', runner=lambda command: next(outputs))
    assert not any(row['job_id'] == '101' for row in snapshot['owners'])
    owner = next(row for row in snapshot['owners'] if row['owner_id'] == 'horeka-bout0002-job102')
    assert owner['job_id'] == '102' and owner['state'] == 'TIMEOUT'


def test_recovery_inventory_includes_ordinary_interactive_allocations():
    outputs = iter(['103|interactive|RUNNING|2026-10-02T10:00:00|None|', ''])
    snapshot = capture(plan={'plan_id': 'recovery'}, since='2026-10-02',
                       include_all_jobs=True, runner=lambda command: next(outputs))
    assert [(row['job_id'], row['state']) for row in snapshot['jobs']] == [('103', 'RUNNING')]
