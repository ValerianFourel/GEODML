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
