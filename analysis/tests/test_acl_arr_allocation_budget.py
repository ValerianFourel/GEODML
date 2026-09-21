"""Allocation deadlines checkpoint admitted work without inventing failures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    build_acl_arr_experiment_plan,
    write_acl_arr_experiment_plan,
)
from analysis.scripts import run_acl_arr_vllm as runner
from analysis.tests.test_acl_arr_document_experiment import (
    _axis_rows,
    _document_sets,
    _models,
    _prompts,
)
from analysis.tests.test_acl_arr_runner_reliability import prepared


class Budget:
    """Mutable clock boundary controlled by completed tasks, not wall-clock sleeps."""

    def __init__(self, *, admit=True, seconds=100.0, end=123456):
        self.admit = admit
        self.seconds = seconds
        self.end = end

    def can_start(self):
        return self.admit

    def admission_stop_reason(self):
        return None if self.admit else "allocation_deadline"

    def work_seconds_left(self):
        return self.seconds

    def record(self):
        return {"allocation_end_epoch": self.end}


def test_deadline_stops_preparation_and_drains_admitted_tasks():
    async def exercise():
        budget = Budget()
        prepared_ids = []
        dispatched = []
        release = asyncio.Event()

        def source():
            for index in range(5):
                prepared_ids.append(index)
                yield prepared(index)

        class Client:
            async def complete(self, **kwargs):
                dispatched.append(kwargs["prompt"])
                if kwargs["prompt"] == "0":
                    budget.admit = False
                else:
                    await release.wait()
                return "{}", {}

        stream = runner._iter_execute(source(), client=Client(), maximum_concurrency=2,
                                      fake=False, budget=budget)
        first = await anext(stream)
        assert first["base"]["task_id"] == "0"
        release.set()
        rest = [row async for row in stream]
        assert [row["base"]["task_id"] for row in rest] == ["1"]
        assert prepared_ids == [0, 1]
        assert dispatched == ["0", "1"]

    asyncio.run(exercise())


def test_work_deadline_cancels_straggler_without_failure_result():
    async def exercise():
        budget = Budget()
        cancelled = []

        class Client:
            async def complete(self, **kwargs):
                if kwargs["prompt"] == "0":
                    budget.admit = False
                    budget.seconds = 0.0
                    return "{}", {}
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(kwargs["prompt"])

        rows = [row async for row in runner._iter_execute(
            (prepared(index) for index in range(5)), client=Client(),
            maximum_concurrency=2, fake=False, budget=budget,
        )]
        assert len(rows) == 1
        assert rows[0]["ok"] is True
        assert cancelled == ["1"]

    asyncio.run(exercise())


def test_expired_budget_never_prepares_or_dispatches():
    async def exercise():
        def source():
            pytest.fail("expired allocation prepared a task")
            yield prepared(0)

        return [row async for row in runner._iter_execute(
            source(), client=None, maximum_concurrency=2, fake=True,
            budget=Budget(admit=False, seconds=0.0),
        )]

    assert asyncio.run(exercise()) == []


def _arguments(tmp_path: Path, *extra: str):
    model = _models()[0]
    plan = build_acl_arr_experiment_plan(
        _prompts(), _axis_rows(), _document_sets(), models=(model,), top_n=2,
    )
    artifacts = write_acl_arr_experiment_plan(tmp_path / "plan", plan=plan)
    return runner._parser().parse_args([
        "primary", "--tasks", str(artifacts.task_files[(model.configuration_id, "rerank")]),
        "--plan-manifest", str(artifacts.manifest_path), "--output-dir", str(tmp_path / "out"),
        "--fake", "--max-concurrency", "1", *extra,
    ])


@pytest.mark.parametrize("scheduler", ["rolling", "chunked"])
def test_cutoff_checkpoints_then_new_allocation_resumes_only_missing(
    tmp_path, monkeypatch, scheduler,
):
    args = _arguments(tmp_path, "--scheduler", scheduler)
    budget = Budget()
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)
    original = runner._execute_one
    dispatched = []

    async def one(item, **kwargs):
        dispatched.append(item["base"]["task_id"])
        result = await original(item, **kwargs)
        if len(dispatched) == 2:
            budget.admit = False
        return result

    monkeypatch.setattr(runner, "_execute_one", one)
    assert asyncio.run(runner._run(args)) == 0
    manifest_path = tmp_path / "out/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "checkpointed"
    assert manifest["stop_reason"] == "allocation_deadline"
    assert manifest["scheduler"] == "rolling"
    assert manifest["requested_scheduler"] == scheduler
    assert manifest["allocation_budget"] == budget.record()
    assert "allocation_budget" not in manifest["resume_identity"]
    assert manifest["tasks"]["attempted_this_invocation"] == 2
    assert manifest["tasks"]["interrupted_this_invocation"] == 0
    assert manifest["completed_count"] == 2
    assert manifest["remaining_count"] > 0
    outcome_path = tmp_path / "out/outcomes.jsonl"
    prefix = outcome_path.read_bytes()
    assert (tmp_path / "out/failures.jsonl").read_text() == ""
    first_ids = set(dispatched)
    dispatched.clear()
    budget = Budget(end=654321)
    args.resume = True
    monkeypatch.setattr(runner, "_execute_one", original)

    async def resumed(item, **kwargs):
        dispatched.append(item["base"]["task_id"])
        return await original(item, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", resumed)
    assert asyncio.run(runner._run(args)) == 0
    final = json.loads(manifest_path.read_text())
    assert final["status"] == "complete"
    assert final["remaining_count"] == 0
    assert not first_ids.intersection(dispatched)
    assert outcome_path.read_bytes().startswith(prefix)
    assert final["resume_identity"] == manifest["resume_identity"]
    assert final["allocation_budget"] == budget.record()


def test_cancelled_task_remains_pending_and_is_not_a_failure(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    args.max_concurrency = 2
    budget = Budget()
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)
    original = runner._execute_one
    dispatched = []
    cancelled = []

    async def one(item, **kwargs):
        dispatched.append(item["base"]["task_id"])
        if len(dispatched) == 1:
            result = await original(item, **kwargs)
            budget.admit = False
            budget.seconds = 0.0
            return result
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(item["base"]["task_id"])

    monkeypatch.setattr(runner, "_execute_one", one)
    assert asyncio.run(runner._run(args)) == 0
    manifest = json.loads((tmp_path / "out/run_manifest.json").read_text())
    assert manifest["completed_count"] == 1
    assert manifest["tasks"]["attempted_this_invocation"] == 2
    assert manifest["tasks"]["interrupted_this_invocation"] == 1
    assert manifest["failures_written_this_invocation"] == 0
    assert len(cancelled) == 1
    assert (tmp_path / "out/failures.jsonl").read_text() == ""


def test_client_startup_is_bounded_by_work_deadline(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    args.fake = False
    args.server_model_revision = _models()[0].model_revision
    budget = Budget(seconds=0.01)
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)
    cancelled = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            budget.admit = False
            budget.seconds = 0.0
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(runner, "VllmChatClient", Client)
    assert asyncio.run(runner._run(args)) == 0
    manifest = json.loads((tmp_path / "out/run_manifest.json").read_text())
    assert manifest["stop_reason"] == "allocation_deadline"
    assert manifest["completed_count"] == 0
    assert manifest["tasks"]["attempted_this_invocation"] == 0
    assert manifest["tasks"]["interrupted_this_invocation"] == 0
    assert cancelled == [True]


def test_expired_allocation_writes_empty_checkpoint_without_inference(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    budget = Budget(admit=False, seconds=0.0)
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)

    async def one(*args, **kwargs):
        pytest.fail("expired allocation executed inference")

    monkeypatch.setattr(runner, "_execute_one", one)
    assert asyncio.run(runner._run(args)) == 0
    manifest = json.loads((tmp_path / "out/run_manifest.json").read_text())
    assert manifest["status"] == "checkpointed"
    assert manifest["stop_reason"] == "allocation_deadline"
    assert manifest["remaining_count"] == manifest["tasks"]["total_count"]
    assert manifest["tasks"]["attempted_this_invocation"] == 0
    assert (tmp_path / "out/outcomes.jsonl").read_bytes() == b""


def test_real_failures_before_deadline_keep_failure_exit_status(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    budget = Budget()
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)
    original = runner._execute_one

    async def one(item, **kwargs):
        result = await original(item, **kwargs)
        return {**result, "ok": False, "error": "deterministic validation failure"}

    monkeypatch.setattr(runner, "_execute_one", one)
    assert asyncio.run(runner._run(args)) == 2
    manifest = json.loads((tmp_path / "out/run_manifest.json").read_text())
    assert manifest["status"] == "complete_with_failures"
    assert manifest["stop_reason"] == "bounded_failures"
    assert manifest["failures_written_this_invocation"] == manifest["tasks"]["total_count"]
    assert manifest["tasks"]["interrupted_this_invocation"] == 0
