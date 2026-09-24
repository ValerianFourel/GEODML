import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline.agentic_dataset import initialize_dataset
from analysis.interpretability.pipeline.inference_claims import InferenceClaimStore
from analysis.scripts import manage_agentic_pilot_judging as manager
from analysis.scripts import report_agentic_500_pilot_results as reporter
from analysis.scripts import run_acl_arr_vllm as runner
from analysis.tests import test_manage_agentic_pilot_judging as preparation
from analysis.tests.test_acl_arr_allocation_budget import Budget
from analysis.tests.test_manage_agentic_pilot_judging import make_one_hour
from analysis.tests.test_nemotron_judge_pilot_batch import (
    QUEUE_WRAPPER,
    _run,
    _throughput_environment,
)

source = preparation.source
completed_allocation = preparation.completed_allocation


def worker_arguments(root, output, claim_root, *, dispatch="partition"):
    return runner._parser().parse_args([
        "agentic-judge", "--tasks", str(root / "plan/bulk_tasks.jsonl"),
        "--judge-manifest", str(root / "plan/run_manifest.json"),
        "--judge-role", "bulk", "--output-dir", str(output),
        "--claim-root", str(claim_root), "--server-model-revision", manager.REVISION,
        "--max-concurrency", "1", "--max-output-tokens", "2048",
        "--request-timeout", "120", "--max-attempts", "3",
        "--disable-thinking", "--resume", "--dispatch-mode", dispatch,
    ])


def dataset_worker_arguments(root, output, dataset_root, *, writer_id):
    return runner._parser().parse_args([
        "agentic-judge", "--tasks", str(root / "plan/bulk_tasks.jsonl"),
        "--judge-manifest", str(root / "plan/run_manifest.json"),
        "--judge-role", "bulk", "--output-dir", str(output),
        "--dataset-root", str(dataset_root), "--dataset-writer-id", writer_id,
        "--max-concurrency", "4", "--max-output-tokens", "2048",
        "--request-timeout", "120", "--max-attempts", "3",
        "--disable-thinking", "--resume", "--dispatch-mode", "backlog",
        "--fake",
    ])


def test_worker_claims_report_and_deadline_continuation_agree(source, completed_allocation, monkeypatch):
    root, _ = completed_allocation
    receipt = manager.verify(root)
    tasks, _, model, revision = runner._agentic_judge_context(
        root / "plan/run_manifest.json", root / "plan/bulk_tasks.jsonl", judge_role="bulk",
    )
    items = {runner._prepare_agentic_judge(task, max_tokens=2048)["seed"]:
             runner._prepare_agentic_judge(task, max_tokens=2048) for task in tasks}
    budget = Budget()
    monkeypatch.setattr(runner.AllocationBudget, "from_environment", lambda: budget)
    calls = []

    class Transport:
        def __init__(self, **kwargs):
            assert kwargs["timeout_seconds"] == 120.0
            assert type(kwargs["timeout_seconds"]) is float
            assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def complete(self, **kwargs):
            item = items[kwargs["seed"]]
            calls.append(item["base"]["judge_task_id"])
            if len(calls) == 2:
                budget.admit = False
            return item["fake_output"], {}

    monkeypatch.setattr(runner, "VllmChatClient", Transport)
    args = worker_arguments(root, root / "attempts/first", receipt["claim_root"])
    assert asyncio.run(runner._run(args)) == 0
    first = json.loads((Path(args.output_dir) / "run_manifest.json").read_text())
    assert first["status"] == "checkpointed"
    assert first["stop_reason"] == "allocation_deadline"
    assert first["completed_count"] == 2
    claims = {p: p.read_bytes() for p in Path(receipt["claim_root"]).glob("*/*.json")}
    assert len(claims) == 2

    for path, raw in claims.items():
        envelope = json.loads(raw)
        item = next(item for item in items.values()
                    if item["base"]["judge_task_id"] == envelope["identity"]["task_id"])
        identity = runner._shared_claim_identity(
            item, args=reporter._judge_claim_arguments(), model_id=model, model_revision=revision,
        )
        assert asdict(identity) == envelope["identity"]
        assert envelope["identity_sha256"] == path.stem
        assert InferenceClaimStore(receipt["claim_root"]).inspect(identity)[0] == "completed"
        old_args = SimpleNamespace(**vars(reporter._judge_claim_arguments()))
        old_args.request_timeout = 120
        old_identity = runner._shared_claim_identity(
            item, args=old_args, model_id=model, model_revision=revision,
        )
        assert old_identity.protocol != identity.protocol
        assert InferenceClaimStore(receipt["claim_root"]).inspect(old_identity)[0] == "missing"

    progress = reporter.collect_report(adaptive_plan=source[0], judge_run=root, expected_prompt_count=1)
    assert progress["generation"]["overall"]["completed"] == 24
    assert progress["nemotron"]["overall"]["completed"] == 2
    assert progress["nemotron"]["overall"]["missing"] == 22
    assert manager.audit(root)["claims"] == {"completed": 2, "missing": 22}
    budget = Budget(end=654321)
    continuation = root.with_name("continuation")
    following = preparation.continue_one_hour(root, continuation)
    resumed = worker_arguments(
        continuation, continuation / "attempts/next", following["claim_root"],
        dispatch=following["dispatch_mode"],
    )
    assert asyncio.run(runner._run(resumed)) == 0
    assert len(calls) == len(set(calls)) == 24
    assert all(path.read_bytes() == raw for path, raw in claims.items())
    progress = reporter.collect_report(adaptive_plan=source[0], judge_run=continuation, expected_prompt_count=1)
    assert progress["experiment_complete"] is True
    assert progress["nemotron"]["prompts"] == {"complete": 1, "partial": 0, "untouched": 0}
    assert all(row["completed"] == row["expected"] for row in progress["nemotron"]["by_method"].values())
    assert all(row["completed"] == row["expected"] for row in progress["nemotron"]["by_engine"].values())
    assert all(row["completed"] == row["expected"] for row in progress["nemotron"]["by_condition"].values())


def test_report_rejects_a_different_explicit_adaptive_plan(source):
    make_one_hour(source)
    replacement = source[0].with_name("different-adaptive.json")
    value = json.loads(source[0].read_text())
    value["judge"]["claim_root"] += "-different"
    replacement.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="adaptive source"):
        reporter.collect_report(adaptive_plan=replacement, judge_run=source[1], expected_prompt_count=1)


def test_executed_sbatch_wrapper_and_reporter_have_identical_claim_fingerprints(source, tmp_path):
    make_one_hour(source)
    env = _throughput_environment(tmp_path / "wrapper")
    env["GEODML_JUDGE_PILOT_ONLY"] = "0"
    manifest_path = Path(env["GEODML_JUDGE_PILOT_ROOT"]) / "plan/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pilot_only"] = False
    manifest_path.write_text(json.dumps(manifest))
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in Path(env["TEST_CAPTURE"]).read_text().splitlines()]
    command = commands[-1]
    args = runner._parser().parse_args(command[command.index("agentic-judge"):])
    tasks, _, model, revision = runner._agentic_judge_context(
        source[1] / "plan/run_manifest.json", source[1] / "plan/bulk_tasks.jsonl", judge_role="bulk",
    )
    assert args.max_output_tokens == 2048
    assert args.max_concurrency == 4
    for task in tasks:
        item = runner._prepare_agentic_judge(task, max_tokens=args.max_output_tokens)
        assert runner._shared_claim_identity(
            item, args=args, model_id=model, model_revision=revision,
        ) == runner._shared_claim_identity(
            item, args=reporter._judge_claim_arguments(), model_id=model, model_revision=revision,
        )


def test_agentic_judge_writes_final_dataset_and_reuses_ledger(
    completed_allocation, monkeypatch, tmp_path
):
    root, _ = completed_allocation
    dataset_root = tmp_path / "dataset"
    initialize_dataset(
        dataset_root,
        population_id="test-population",
        acceptance_policy_id="experiment-v2",
    )
    calls = 0
    execute = runner._execute_one

    async def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await execute(*args, **kwargs)

    monkeypatch.setattr(runner, "_execute_one", counted)
    first = dataset_worker_arguments(
        root, tmp_path / "attempt-1", dataset_root, writer_id="judge-job-1"
    )
    assert asyncio.run(runner._run(first)) == 0
    assert calls == 24
    assert not (Path(first.output_dir) / "outcomes.jsonl").exists()
    assert not (Path(first.output_dir) / "failures.jsonl").exists()
    assert not (Path(first.output_dir) / "attempts.jsonl").exists()
    judgment_manifests = list(
        (dataset_root / "data/judgments").glob("*.manifest.json")
    )
    assert sum(json.loads(path.read_text())["rows"] for path in judgment_manifests) == 24

    second = dataset_worker_arguments(
        root, tmp_path / "attempt-2", dataset_root, writer_id="judge-job-2"
    )
    assert asyncio.run(runner._run(second)) == 0
    assert calls == 24
    manifest = json.loads((Path(second.output_dir) / "run_manifest.json").read_text())
    assert manifest["direct_dataset"]["reused_count"] == 24
    assert manifest["completed_count"] == 24
