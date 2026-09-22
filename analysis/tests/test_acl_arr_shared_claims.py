"""Shared judge claims prevent overlapping calls and reuse committed outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import run_acl_arr_vllm as runner


@pytest.fixture
def judge_run(tmp_path, monkeypatch):
    tasks = [SimpleNamespace(judge_task_id=f"judge-{index}") for index in range(5)]
    queue = tmp_path / "tasks.jsonl"
    queue.write_text("original queue\n")
    plan = tmp_path / "plan.json"
    plan.write_text("{}")

    def context(*args, judge_role):
        return tasks, judge_role, "fixture/model", "a" * 40

    def prepare(task, *, max_tokens):
        return {
            "base": {"judge_task_id": task.judge_task_id, "fake_backend": False},
            "prompt": task.judge_task_id, "schema_name": "fixture", "schema": {},
            "temperature": 0.0, "max_tokens": max_tokens, "seed": 7,
            "validator": json.loads, "fake_output": '{"value":1}',
        }

    monkeypatch.setattr(runner, "_agentic_judge_context", context)
    monkeypatch.setattr(runner, "_prepare_agentic_judge", prepare)

    def arguments(output, *extra):
        return runner._parser().parse_args([
            "agentic-judge", "--tasks", str(queue), "--judge-manifest", str(plan),
            "--judge-role", "bulk", "--output-dir", str(tmp_path / output),
            "--claim-root", str(tmp_path / "claims"), "--server-model-revision", "a" * 40,
            "--max-concurrency", "1", "--pilot-only", *extra,
        ])

    return tasks, arguments, queue


@pytest.mark.parametrize("count,index", [(0, 0), (-1, 0), (2, -1), (2, 2)])
def test_bad_worker_slot_is_rejected(judge_run, count, index):
    _, arguments, _ = judge_run
    args = arguments("bad", "--fake", "--worker-count", str(count), "--worker-index", str(index))
    with pytest.raises(ValueError, match="worker"):
        asyncio.run(runner._run(args))


def test_multiple_workers_require_shared_claim_root(judge_run):
    _, arguments, _ = judge_run
    args = arguments("bad", "--fake", "--worker-count", "2")
    args.claim_root = None
    with pytest.raises(ValueError, match="claim-root"):
        asyncio.run(runner._run(args))


def test_slot_formula_is_stable_and_covers_every_task_once():
    ids = [f"judge-{index}" for index in range(20)]
    slots = [{key for key in ids if runner.task_in_worker(key, index, 3)} for index in range(3)]
    assert set.union(*slots) == set(ids)
    assert sum(map(len, slots)) == len(ids)
    for key in ids:
        expected = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 3
        assert key in slots[expected]


def test_two_production_partitions_resume_without_duplicate_calls(judge_run, monkeypatch):
    tasks, arguments, _ = judge_run
    original = runner._execute_one
    calls = []

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        return await original(item, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", execute)

    async def run_attempt(label):
        workers = []
        for index in (0, 1):
            args = arguments(f"{label}-{index}", "--fake", "--worker-count", "2",
                             "--worker-index", str(index), "--dispatch-mode", "partition")
            args.pilot_only = False
            workers.append(runner._run(args))
        return await asyncio.gather(*workers)

    assert asyncio.run(run_attempt("first")) == [0, 0]
    assert sorted(calls) == sorted(t.judge_task_id for t in tasks)
    calls.clear()
    assert asyncio.run(run_attempt("new-allocation")) == [0, 0]
    assert calls == []


def test_committed_outcomes_reused_across_new_queue_and_slot_without_calls(judge_run, monkeypatch, tmp_path):
    tasks, arguments, queue = judge_run
    original = runner._execute_one
    calls = []

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        return await original(item, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", execute)
    assert asyncio.run(runner._run(arguments("first", "--fake"))) == 0
    assert len(calls) == len(tasks)
    queue.write_text("same stable tasks in another queue serialization\n")
    args = arguments("second", "--fake", "--worker-count", "2", "--worker-index", "0")
    assert asyncio.run(runner._run(args)) == 0
    manifest = json.loads((tmp_path / "second/run_manifest.json").read_text())
    expected = sum(runner.task_in_worker(task.judge_task_id, 0, 2) for task in tasks)
    assert manifest["tasks"]["total_count"] == expected
    assert manifest["tasks"]["source_total_count"] == len(tasks)
    assert manifest["completed_count"] == expected
    assert manifest["shared_claims"]["reused_this_invocation"] == expected
    assert manifest["shared_claims"]["inference_tasks_this_invocation"] == 0
    assert len(calls) == len(tasks)
    args.resume = True
    assert asyncio.run(runner._run(args)) == 0
    assert len(calls) == len(tasks)


@pytest.mark.parametrize("dispatch_mode", ["partition", "backlog"])
def test_live_owner_blocks_duplicate_http_and_busy_worker_checkpoints(judge_run, monkeypatch, tmp_path, dispatch_mode):
    tasks, arguments, _ = judge_run
    del tasks[1:]

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def complete(self, **kwargs):
                calls.append(kwargs["prompt"])
                started.set()
                await release.wait()
                return '{"value":1}', {}

        monkeypatch.setattr(runner, "VllmChatClient", Client)
        first = asyncio.create_task(runner._run(arguments("first", "--dispatch-mode", dispatch_mode)))
        await started.wait()
        try:
            assert await runner._run(arguments("second", "--scheduler", "chunked", "--dispatch-mode", dispatch_mode)) == 0
            manifest = json.loads((tmp_path / "second/run_manifest.json").read_text())
            assert manifest["status"] == "checkpointed"
            assert manifest["stop_reason"] == "shared_claims_busy"
            assert manifest["remaining_count"] == 1
            assert manifest["failures_written_this_invocation"] == 0
            assert manifest["tasks"]["interrupted_this_invocation"] == 0
            assert manifest["shared_claims"]["busy_this_invocation"] == 1
            assert manifest["scheduler"] == "rolling"
            assert manifest["requested_scheduler"] == "chunked"
            assert len(calls) == 1
        finally:
            release.set()
            await first
        assert await runner._run(arguments("third", "--dispatch-mode", dispatch_mode)) == 0
        assert len(calls) == 1

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


@pytest.mark.parametrize("dispatch_mode", ["partition", "backlog"])
def test_cancelled_claim_releases_lock_and_missing_work_can_retry(judge_run, monkeypatch, tmp_path, dispatch_mode):
    tasks, arguments, _ = judge_run
    del tasks[1:]

    async def exercise():
        started = asyncio.Event()
        calls = []

        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def complete(self, **kwargs):
                calls.append(kwargs["prompt"])
                started.set()
                if len(calls) == 1:
                    await asyncio.Event().wait()
                return '{"value":1}', {}

        monkeypatch.setattr(runner, "VllmChatClient", Client)
        first = asyncio.create_task(runner._run(arguments("first", "--dispatch-mode", dispatch_mode)))
        await started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (tmp_path / "first/failures.jsonl").read_text() == ""
        assert not list((tmp_path / "claims").glob("*/*.failed.json"))
        assert await runner._run(arguments("second", "--dispatch-mode", dispatch_mode)) == 0
        assert len(calls) == 2

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_shared_commit_precedes_local_journal_so_local_write_crash_does_not_reinfer(
    judge_run, monkeypatch, tmp_path,
):
    tasks, arguments, _ = judge_run
    del tasks[1:]
    original_append, original_execute = runner._append, runner._execute_one
    calls = []

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        return await original_execute(item, **kwargs)

    def fail_outcome(stream, value):
        if Path(stream.name).name == "outcomes.jsonl":
            raise OSError("local disk full")
        original_append(stream, value)

    monkeypatch.setattr(runner, "_execute_one", execute)
    monkeypatch.setattr(runner, "_append", fail_outcome)
    monkeypatch.setenv("GEODML_EXECUTION_COMMIT", "1" * 40)
    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    with pytest.raises(OSError, match="local disk full"):
        asyncio.run(runner._run(arguments("first", "--fake")))
    saved_path, = (tmp_path / "claims").glob("*/*.json")
    saved = json.loads(saved_path.read_text())["outcome"]
    producer = saved["producer"]
    assert producer["execution_git_commit"] == "1" * 40
    assert producer["slurm"] == {
        "SLURM_JOB_ID": "1001", "SLURM_ARRAY_JOB_ID": "1000", "SLURM_ARRAY_TASK_ID": "1",
    }
    assert producer["run_id"]
    assert producer["invocation_id"]
    monkeypatch.setattr(runner, "_append", original_append)
    monkeypatch.setenv("GEODML_EXECUTION_COMMIT", "2" * 40)
    monkeypatch.setenv("SLURM_JOB_ID", "2002")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "2000")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    assert asyncio.run(runner._run(arguments("second", "--fake"))) == 0
    assert len(calls) == 1
    reused = json.loads((tmp_path / "second/outcomes.jsonl").read_text())
    assert reused["producer"] == producer
    assert reused["invocation_id"] != producer["invocation_id"]
    assert reused["run_id"] != producer["run_id"]
    assert reused["shared_reused"] is True


def test_resume_refuses_changing_worker_assignment(judge_run):
    _, arguments, _ = judge_run
    args = arguments("same-output", "--fake")
    assert asyncio.run(runner._run(args)) == 0
    args.resume = True
    args.worker_count = 2
    with pytest.raises(ValueError, match="worker.*assignment"):
        asyncio.run(runner._run(args))


@pytest.mark.parametrize("field,value", [
    ("judge_role", "validation"), ("disable_thinking", True),
    ("fake", False), ("pilot_only", False),
    ("max_attempts", 4), ("request_timeout", 12.0),
])
def test_claim_identity_separates_execution_contract(judge_run, field, value):
    tasks, arguments, _ = judge_run
    args = arguments("first", "--fake")
    item = runner._prepare_agentic_judge(tasks[0], max_tokens=512)

    def identity():
        return runner._shared_claim_identity(
            item, args=args, model_id="fixture/model", model_revision="a" * 40,
        )

    original = identity()
    setattr(args, field, value)
    assert identity() != original


def test_claim_identity_tracks_request_revision_and_ignores_worker_layout(judge_run):
    tasks, arguments, _ = judge_run
    args = arguments("first", "--fake")
    item = runner._prepare_agentic_judge(tasks[0], max_tokens=512)

    def identity(model="fixture/model", revision="a" * 40):
        return runner._shared_claim_identity(
            item, args=args, model_id=model, model_revision=revision,
        )

    original = identity()
    args.output_dir = "another-output"
    args.tasks = "different-queue"
    args.worker_count = 50
    args.worker_index = 20
    assert identity() == original
    assert identity(revision="b" * 40) != original
    assert identity(model="different/model") != original
    item["prompt"] += " changed recorded conversation"
    assert identity() != original


def test_shared_outcome_semantics_are_checked_even_with_valid_content_hash(
    judge_run, monkeypatch, tmp_path,
):
    tasks, arguments, _ = judge_run
    del tasks[1:]
    assert asyncio.run(runner._run(arguments("first", "--fake"))) == 0
    path, = (tmp_path / "claims").glob("*/*.json")
    envelope = json.loads(path.read_text())
    envelope["outcome"]["parsed_output"] = {"value": 999}
    envelope["outcome_sha256"] = hashlib.sha256(json.dumps(
        envelope["outcome"], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    path.write_text(json.dumps(envelope))

    async def forbidden(*args, **kwargs):
        pytest.fail("a corrupt shared result must not trigger more inference")

    monkeypatch.setattr(runner, "_execute_one", forbidden)
    with pytest.raises(ValueError, match="shared outcome output validation mismatch"):
        asyncio.run(runner._run(arguments("second", "--fake")))


def test_validation_failure_is_not_a_completed_shared_claim(judge_run, monkeypatch, tmp_path):
    tasks, arguments, _ = judge_run
    del tasks[1:]
    original = runner._prepare_agentic_judge

    def bad_output(*args, **kwargs):
        return {**original(*args, **kwargs), "fake_output": "not JSON"}

    monkeypatch.setattr(runner, "_prepare_agentic_judge", bad_output)
    assert asyncio.run(runner._run(arguments("first", "--fake"))) == 2
    assert list((tmp_path / "claims").glob("*/*.json")) == []
    manifest = json.loads((tmp_path / "first/run_manifest.json").read_text())
    assert manifest["failures_written_this_invocation"] == 1
    assert manifest["stop_reason"] == "bounded_failures"
    monkeypatch.setattr(runner, "_prepare_agentic_judge", original)
    assert asyncio.run(runner._run(arguments("second", "--fake"))) == 0


def test_empty_assigned_slot_is_complete_without_calls(judge_run, monkeypatch, tmp_path):
    tasks, arguments, _ = judge_run
    del tasks[1:]
    empty_index = 1 - int(runner.task_in_worker(tasks[0].judge_task_id, 1, 2))

    async def forbidden(*args, **kwargs):
        pytest.fail("empty assigned slot made an inference call")

    monkeypatch.setattr(runner, "_execute_one", forbidden)
    assert asyncio.run(runner._run(arguments(
        "empty", "--fake", "--worker-count", "2", "--worker-index", str(empty_index),
    ))) == 0
    manifest = json.loads((tmp_path / "empty/run_manifest.json").read_text())
    assert manifest["tasks"]["total_count"] == 0
    assert manifest["tasks"]["source_total_count"] == 1
    assert manifest["stop_reason"] == "queue_exhausted"


@pytest.mark.parametrize("dispatch_mode", ["partition", "backlog"])
def test_deadline_checkpoint_reuses_saved_claims_in_next_worker_run(judge_run, monkeypatch, tmp_path, dispatch_mode):
    from analysis.tests.test_acl_arr_allocation_budget import Budget

    tasks, arguments, _ = judge_run
    budget = Budget()
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)
    original = runner._execute_one
    calls = []

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        result = await original(item, **kwargs)
        if len(calls) == 2:
            budget.admit = False
        return result

    monkeypatch.setattr(runner, "_execute_one", execute)
    assert asyncio.run(runner._run(arguments("first", "--fake", "--dispatch-mode", dispatch_mode))) == 0
    first = json.loads((tmp_path / "first/run_manifest.json").read_text())
    assert first["status"] == "checkpointed"
    assert first["stop_reason"] == "allocation_deadline"
    assert first["completed_count"] == 2
    budget = Budget(end=654321)
    extra = ("--worker-count", "8", "--worker-index", "7") if dispatch_mode == "backlog" else ()
    assert asyncio.run(runner._run(arguments("next-allocation", "--fake", "--dispatch-mode", dispatch_mode, *extra))) == 0
    final = json.loads((tmp_path / "next-allocation/run_manifest.json").read_text())
    assert final["completed_count"] == len(tasks)
    assert final["shared_claims"]["reused_this_invocation"] == 2
    assert len(calls) == len(set(calls)) == len(tasks)


def test_backlog_requires_shared_claim_root_even_with_one_worker(judge_run):
    _, arguments, _ = judge_run
    args = arguments("backlog", "--fake", "--dispatch-mode", "backlog")
    args.claim_root = None
    with pytest.raises(ValueError, match="claim-root"):
        asyncio.run(runner._run(args))


def test_backlog_visits_own_slot_first_then_fills_every_other_slot(judge_run, monkeypatch, tmp_path):
    tasks, arguments, _ = judge_run
    calls = []
    original = runner._execute_one

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        return await original(item, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", execute)
    args = arguments("backlog", "--fake", "--dispatch-mode", "backlog",
                     "--worker-count", "3", "--worker-index", "1")
    assert asyncio.run(runner._run(args)) == 0
    preferred = [task.judge_task_id for task in tasks if runner.task_in_worker(task.judge_task_id, 1, 3)]
    assert calls[:len(preferred)] == preferred
    assert set(calls) == {task.judge_task_id for task in tasks}
    manifest = json.loads((tmp_path / "backlog/run_manifest.json").read_text())
    assert manifest["completed_count"] == manifest["tasks"]["total_count"] == len(tasks)
    assert manifest["worker_dispatch"]["mode"] == "backlog"
    assert manifest["stop_reason"] == "queue_exhausted"


def test_backlog_rechecks_busy_tasks_after_other_local_work_progresses(judge_run, monkeypatch, tmp_path):
    from contextlib import ExitStack

    from analysis.interpretability.pipeline.inference_claims import InferenceClaimStore

    tasks, arguments, _ = judge_run
    args = arguments("backlog", "--fake", "--dispatch-mode", "backlog")
    item = runner._prepare_agentic_judge(tasks[0], max_tokens=args.max_output_tokens)
    identity = runner._shared_claim_identity(item, args=args, model_id="fixture/model", model_revision="a" * 40)
    owner = ExitStack()
    owner.enter_context(InferenceClaimStore(tmp_path / "claims").try_claim(identity))
    original = runner._execute_one
    calls = []

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        owner.close()
        return await original(item, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", execute)
    try:
        assert asyncio.run(runner._run(args)) == 0
    finally:
        owner.close()
    assert calls[-1] == tasks[0].judge_task_id
    assert len(calls) == len(set(calls)) == len(tasks)
    manifest = json.loads((tmp_path / "backlog/run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["shared_claims"]["busy_this_invocation"] == 1
    assert manifest["tasks"]["interrupted_this_invocation"] == 0


def test_backlog_terminal_failure_is_saved_and_not_called_by_next_job(judge_run, monkeypatch, tmp_path):
    tasks, arguments, _ = judge_run
    original_prepare, original_execute = runner._prepare_agentic_judge, runner._execute_one
    calls = []

    def prepare(task, **kwargs):
        item = original_prepare(task, **kwargs)
        if task.judge_task_id == tasks[0].judge_task_id:
            item["fake_output"] = "invalid JSON"
        return item

    async def execute(item, **kwargs):
        calls.append(item["base"]["judge_task_id"])
        return await original_execute(item, **kwargs)

    monkeypatch.setattr(runner, "_prepare_agentic_judge", prepare)
    monkeypatch.setattr(runner, "_execute_one", execute)
    assert asyncio.run(runner._run(arguments("first", "--fake", "--dispatch-mode", "backlog"))) == 2
    assert len(calls) == len(tasks)
    monkeypatch.setattr(runner, "_prepare_agentic_judge", original_prepare)
    assert asyncio.run(runner._run(arguments("next", "--fake", "--dispatch-mode", "backlog",
                                            "--worker-count", "7", "--worker-index", "6"))) == 2
    assert len(calls) == len(tasks)
    manifest = json.loads((tmp_path / "next/run_manifest.json").read_text())
    assert manifest["completed_count"] == len(tasks) - 1
    assert manifest["remaining_count"] == 1
    assert manifest["shared_claims"]["failed_reused_this_invocation"] == 1
    assert manifest["shared_claims"]["inference_tasks_this_invocation"] == 0
    assert manifest["stop_reason"] == "bounded_failures"


def test_backlog_mixed_process_counts_fill_once_then_next_job_reuses(judge_run, monkeypatch, tmp_path):
    import multiprocessing
    import os
    from collections import Counter

    tasks, arguments, _ = judge_run
    context = multiprocessing.get_context("fork")
    start = context.Event()
    calls_path = tmp_path / "calls.txt"
    original = runner._execute_one

    async def execute(item, **kwargs):
        descriptor = os.open(calls_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, (item["base"]["judge_task_id"] + "\n").encode())
        finally:
            os.close(descriptor)
        await asyncio.sleep(0.02)
        return await original(item, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", execute)

    def worker(name, count, index):
        assert start.wait(5)
        code = asyncio.run(runner._run(arguments(name, "--fake", "--dispatch-mode", "backlog",
            "--worker-count", str(count), "--worker-index", str(index))))
        raise SystemExit(code)

    processes = [context.Process(target=worker, args=("one", 3, 0)),
                 context.Process(target=worker, args=("two", 5, 4))]
    try:
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(10)
        assert [p.exitcode for p in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    before = calls_path.read_text()
    assert Counter(before.splitlines()) == {task.judge_task_id: 1 for task in tasks}
    assert asyncio.run(runner._run(arguments("followup", "--fake", "--dispatch-mode", "backlog",
                                            "--worker-count", "11", "--worker-index", "10"))) == 0
    assert calls_path.read_text() == before
