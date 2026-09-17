"""Shared generator backlogs preserve completed cells across worker layouts."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from analysis.interpretability.pipeline.agentic_search import (
    AgenticResult,
    ContextCompactor,
    LexicalOverlapScorer,
    ReactiveSnippetLoopV1,
)
from analysis.scripts.run_agentic_search_integration_smoke import run_smoke
from analysis.tests.test_run_agentic_search_integration_smoke import (
    _FakeAllocationBudget,
    _FakeClientContext,
    _smoke_inputs,
)


def inputs_for(tmp_path):
    return replace(_smoke_inputs(tmp_path), shared_claim_root=tmp_path / "shared")


async def run(inputs, client):
    return await run_smoke(inputs, client_context=client,
                           compactor=ContextCompactor(LexicalOverlapScorer()))


def test_new_output_reuses_every_shared_cell_without_inference(tmp_path):
    inputs = inputs_for(tmp_path)
    first = _FakeClientContext()
    assert asyncio.run(run(inputs, first))["completed_count"] == 12
    second = _FakeClientContext()
    resumed = asyncio.run(run(replace(inputs, output=tmp_path / "new", worker_index=2,
                                     worker_count=3), second))
    assert resumed["completed_count"] == 12
    assert second.call_count == 0
    assert resumed["shared_backlog"]["reused_count"] == 12


def test_overlapping_workers_and_later_layout_never_repeat_completed_calls(tmp_path):
    inputs = inputs_for(tmp_path)
    clients = [_FakeClientContext(delay_seconds=.001) for _ in range(2)]

    async def both():
        return await asyncio.gather(*[
            run(replace(inputs, output=tmp_path / f"worker-{i}", worker_index=i,
                        worker_count=2), clients[i]) for i in range(2)
        ])

    asyncio.run(both())
    assert sum(client.call_count for client in clients) == 24
    later = _FakeClientContext()
    result = asyncio.run(run(replace(inputs, output=tmp_path / "later", worker_index=0,
                                    worker_count=1), later))
    assert result["completed_count"] == 12
    assert later.call_count == 0


def test_preexisting_completed_cells_export_without_touching_files(tmp_path):
    original = _smoke_inputs(tmp_path)
    asyncio.run(run(original, _FakeClientContext()))
    preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for folder in ("results", "traces", "diagnostics")
                 for path in (original.output / folder).glob("*.json")}
    shared = replace(original, shared_claim_root=tmp_path / "shared")
    client = _FakeClientContext()
    asyncio.run(run(shared, client))
    assert client.call_count == 0
    assert preserved == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in preserved}
    asyncio.run(run(replace(shared, output=tmp_path / "next"), client))
    assert client.call_count == 0


def test_shared_record_corruption_fails_before_inference(tmp_path):
    inputs = inputs_for(tmp_path)
    asyncio.run(run(inputs, _FakeClientContext()))
    path = next(inputs.shared_claim_root.glob("*/*.json"))
    value = json.loads(path.read_text())
    value["outcome"]["result"]["answer"] = "tampered"
    path.write_text(json.dumps(value))
    client = _FakeClientContext()
    with pytest.raises(ValueError, match="shared task outcome"):
        asyncio.run(run(replace(inputs, output=tmp_path / "another"), client))
    assert client.call_count == 0


def test_one_worker_finishes_cells_outside_preferred_slot(tmp_path):
    inputs = replace(inputs_for(tmp_path), worker_index=1, worker_count=4)
    client = _FakeClientContext()
    result = asyncio.run(run(inputs, client))
    assert result["completed_count"] == 12
    assert client.call_count == 24


def test_known_failure_is_durable_and_next_job_does_not_retry_it(tmp_path):
    inputs = inputs_for(tmp_path)
    first = _FakeClientContext(failed_final_passes=100)
    with pytest.raises(RuntimeError, match="failed after bounded retry"):
        asyncio.run(run(inputs, first))
    failures = list(inputs.shared_claim_root.glob("*/*.failed.json"))
    assert len(failures) == 1
    assert first.failed_final_calls == 6
    second = _FakeClientContext()
    with pytest.raises(RuntimeError, match="failed after bounded retry"):
        asyncio.run(run(replace(inputs, output=tmp_path / "later"), second))
    assert second.call_count == 0
    manifest = json.loads((tmp_path / "later/run_manifest.json").read_text())
    assert manifest["completed_count"] == 11
    assert manifest["remaining_count"] == 1
    assert manifest["stop_reason"] == "bounded_failures"


def test_reactive_retrieval_contract_failure_retries_cell_without_killing_worker(tmp_path):
    inputs = inputs_for(tmp_path)
    client = _FakeClientContext()
    original = ReactiveSnippetLoopV1.run
    skipped = 0

    async def skip_retrieval_once(self, user_prompt, condition):
        nonlocal skipped
        if skipped == 0:
            skipped += 1
            return AgenticResult(
                method_id=self.method_id,
                condition=condition,
                ranking=(),
                answer="Premature finish",
                final_snippets=(),
                trace=self._trace(user_prompt, condition, {"synthetic_test": 1}),
            )
        return await original(self, user_prompt, condition)

    with patch.object(ReactiveSnippetLoopV1, "run", skip_retrieval_once):
        manifest = asyncio.run(run(inputs, client))

    assert skipped == 1
    assert manifest["status"] == "complete"
    assert manifest["completed_count"] == 12
    assert manifest["failed_cell_ids"] == []
    assert len(list((inputs.output / "failed_traces").glob("*/*.json"))) == 1


def test_deadline_checkpoint_then_peer_completion_reuses_in_original_output(tmp_path):
    inputs = replace(inputs_for(tmp_path), request_concurrency=1)
    budget = _FakeAllocationBudget(can_start=lambda: not list((inputs.output / "results").glob("*.json")))
    with patch("analysis.scripts.run_agentic_search_integration_smoke.AllocationBudget.from_environment",
               return_value=budget):
        first = asyncio.run(run(inputs, _FakeClientContext()))
    assert first["completed_count"] == 1
    assert first["stop_reason"] == "allocation_deadline"
    second_client = _FakeClientContext()
    second = asyncio.run(run(replace(inputs, output=tmp_path / "peer"), second_client))
    assert second["completed_count"] == 12
    assert second_client.call_count == 22
    third_client = _FakeClientContext()
    third = asyncio.run(run(replace(inputs, request_concurrency=2, cell_concurrency=3), third_client))
    assert third["completed_count"] == 12
    assert third_client.call_count == 0


def test_shared_commit_survives_crash_before_local_result(tmp_path):
    import analysis.scripts.run_agentic_search_integration_smoke as runner
    inputs = replace(inputs_for(tmp_path), request_concurrency=1)
    original = runner._write_json_atomic

    def fail_result(path, value):
        if path.parent.name == "results":
            raise OSError("synthetic disk failure after shared commit")
        original(path, value)

    first = _FakeClientContext()
    with patch.object(runner, "_write_json_atomic", side_effect=fail_result), pytest.raises(OSError):
        asyncio.run(run(inputs, first))
    assert first.call_count == 2
    second = _FakeClientContext()
    manifest = asyncio.run(run(inputs, second))
    assert manifest["completed_count"] == 12
    assert second.call_count == 22


def test_same_output_writer_is_rejected_while_owner_runs(tmp_path):
    inputs = inputs_for(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient(_FakeClientContext):
        async def complete(self, **kwargs):
            started.set()
            await release.wait()
            return await super().complete(**kwargs)

    async def overlap():
        owner = asyncio.create_task(run(inputs, BlockingClient()))
        await started.wait()
        try:
            with pytest.raises(RuntimeError, match="already has an active writer"):
                await run(inputs, _FakeClientContext())
        finally:
            release.set()
            await owner

    asyncio.run(overlap())


def test_shared_reuse_preserves_actual_producer_not_consumer_job(tmp_path):
    inputs = inputs_for(tmp_path)
    with patch.dict("os.environ", {"GEODML_EXECUTION_COMMIT": "first-commit", "SLURM_JOB_ID": "100"}):
        asyncio.run(run(inputs, _FakeClientContext()))
    original = {p.name: p.read_bytes() for p in (inputs.output / "provenance").glob("*.json")}
    second = replace(inputs, output=tmp_path / "consumer")
    with patch.dict("os.environ", {"GEODML_EXECUTION_COMMIT": "next-commit", "SLURM_JOB_ID": "200"}):
        asyncio.run(run(second, _FakeClientContext()))
    assert original == {p.name: p.read_bytes() for p in (second.output / "provenance").glob("*.json")}
    for payload in original.values():
        assert json.loads(payload)["git_commit"] == "first-commit"
        assert json.loads(payload)["slurm_job_id"] == "100"


def test_deadline_cancellation_releases_claim_without_recording_failure(tmp_path):
    inputs = replace(inputs_for(tmp_path), request_concurrency=1)

    class BlockingClient(_FakeClientContext):
        entered = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def complete(self, **kwargs):
            self.call_count += 1
            await asyncio.Event().wait()

    client = BlockingClient()
    budget = _FakeAllocationBudget(work_seconds_left=lambda: 0.0 if client.entered else None)
    with patch("analysis.scripts.run_agentic_search_integration_smoke.AllocationBudget.from_environment",
               return_value=budget):
        result = asyncio.run(run(inputs, client))
    assert result["stop_reason"] == "allocation_deadline"
    assert result["completed_count"] == 0
    assert result["failed_cell_ids"] == []
    assert not list(inputs.shared_claim_root.glob("*/*.json"))
    second = _FakeClientContext()
    assert asyncio.run(run(replace(inputs, output=tmp_path / "next"), second))["completed_count"] == 12
    assert second.call_count == 24


def test_new_paths_and_worker_layout_reuse_but_changed_model_revision_does_not(tmp_path):
    inputs = inputs_for(tmp_path)
    asyncio.run(run(inputs, _FakeClientContext()))
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    moved = replace(_smoke_inputs(relocated), shared_claim_root=inputs.shared_claim_root,
                    worker_index=4, worker_count=5, base_url="http://different-host/v1")
    client = _FakeClientContext()
    assert asyncio.run(run(moved, client))["completed_count"] == 12
    assert client.call_count == 0
    changed = replace(moved, output=tmp_path / "different-model", model_revision="b" * 40)
    assert asyncio.run(run(changed, client))["completed_count"] == 12
    assert client.call_count == 24
