"""Real persisted evaluation queues with external inference/Slurm boundaries replaced."""
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_si_v4 as pilot
from analysis.scripts import run_si_v4_cycle as cycle
from analysis.scripts import run_source_importance_judge as judge
from analysis.tests.test_horeka_si_v4 import gemma_runtime
from analysis.tests.test_source_importance_v4 import CONFIG, MAP, freeze, support


def prepared_cycle(tmp_path, queue=None):
    inputs, _ = freeze(tmp_path)
    root = tmp_path / "cycle"
    root.mkdir()
    shutil.copytree(inputs, root / "candidate-inputs")
    manifest = cycle.read(root / "candidate-inputs/manifest.json")
    queue = queue or [{"name": "candidate-e2e-1", "inputs": "candidate-inputs"},
                      {"name": "candidate-e2e-2", "inputs": "candidate-inputs"},
                      {"name": "candidate-fixed-1", "inputs": "candidate-inputs", "fixed_map_from": "candidate-e2e-1"}]
    plan = {"format_version": cycle.FORMAT, "phase": "development", "budget": cycle.BUDGET,
            "inventories": {"candidate-inputs": manifest}, "queue": queue}
    (root / "evaluation-plan.json").write_text(json.dumps(plan))
    settings = {**CONFIG, "source_importance_only": True,
                "chat_template_sha256": hashlib.sha256(b"fixture template").hexdigest()}
    (root / "judge-config.json").write_text(json.dumps(settings))
    config = {"workload_mode": "evaluation-v4", "budget": cycle.BUDGET, "evaluation_phase": "development",
              "cycle_budget_path": str(root / "budget.json"), "workspace": str(tmp_path), "account": "fixture",
              "git_commit": "a" * 40, "repository": str(tmp_path), "walltime": "01:00:00",
              "estimate": {"minutes": [30, 60]}, "judge_config_sha256": judge.file_hash(root / "judge-config.json"),
              "evaluation_plan_sha256": judge.file_hash(root / "evaluation-plan.json")}
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    cycle.register_phase(config_path)
    return root, config_path, config


@pytest.fixture
def external_boundary(monkeypatch):
    from analysis.scripts import capture_agentic_scheduler_snapshot as scheduler
    state = SimpleNamespace(calls=[], commands=[], fail_map=False, interrupt_source=False,
                            stop_after_map=None, scheduler_jobs=[], scheduler_owners=[])
    class Transport:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return None
        async def complete(self, **kwargs):
            state.calls.append(kwargs["schema_name"])
            if kwargs["schema_name"] == "answer_map_v4":
                value = {**MAP, "claims": MAP["claims"][:-1]} if state.fail_map else MAP
                if state.stop_after_map is not None:
                    state.stop_after_map.touch()
            else:
                if state.interrupt_source:
                    state.interrupt_source = False
                    raise asyncio.CancelledError()
                value = support()
            return json.dumps(value), {"finish_reason": "stop", "completion_tokens": 30, "prompt_tokens": 100}
    tokenizer = SimpleNamespace(get_chat_template=lambda: "fixture template",
                                apply_chat_template=lambda *a, **kw: [1] * 10)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: tokenizer)))
    monkeypatch.setattr(judge, "VllmChatClient", Transport)
    monkeypatch.setattr(pilot, "capture_quota", lambda *a: {"fixture": True})
    monkeypatch.setattr(judge, "check_storage", lambda *a: {"safe_to_admit": True, "quota_verified": True})
    monkeypatch.setattr(scheduler, "capture", lambda **kw: {
        "complete": True, "captured_at_epoch": int(time.time()),
        "jobs": state.scheduler_jobs, "owners": state.scheduler_owners})
    def check_output(command, **kwargs):
        state.commands.append(command)
        if command[0] == "git":
            return "a" * 40 + "\n" if "rev-parse" in command else ""
        if command[0] == "squeue":
            return ""
        raise AssertionError(command)
    def run(command, **kwargs):
        state.commands.append(command)
        assert command[0] in ("sbatch", "sinfo")
        return SimpleNamespace(stdout="6001\n", returncode=0)
    monkeypatch.setattr(cycle.subprocess, "check_output", check_output)
    monkeypatch.setattr(cycle.subprocess, "run", run)
    monkeypatch.setenv("SLURM_JOB_ID", "6001")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(time.time() + 3600))
    monkeypatch.setenv("GEODML_INFERENCE_AUTH_REQUIRED", "1")
    return state


def review(root, config_path, config, attempt):
    return asyncio.run(cycle.review(SimpleNamespace(config=config_path,
        output=root / "attempts" / attempt / "trial", base_url="http://fixture"), config))


@pytest.mark.parametrize("fail_map", [False, True])
def test_finite_queue_runs_real_coordinators_and_resumes_verified_outputs(tmp_path, external_boundary, fail_map):
    root, config_path, config = prepared_cycle(tmp_path)
    external_boundary.fail_map = fail_map
    assert review(root, config_path, config, "first") == 0
    recorded_calls = list(external_boundary.calls)
    expected = ["answer_map_v4"] * 4 if fail_map else [
        "answer_map_v4", "source_importance_v4", "answer_map_v4", "source_importance_v4", "source_importance_v4"]
    assert recorded_calls == expected
    fixed = cycle.latest_report(root / "evaluation-results/candidate-fixed-1")
    fixed_cell = next(judge.rows(fixed / "cells.jsonl.gz"))
    assert fixed_cell["map_result"]["fixed_map"] is True
    assert (fixed_cell["sources"][0]["importance"] is None) == fail_map
    assert review(root, config_path, config, "second") == 0
    assert external_boundary.calls == recorded_calls
    assert all(row["reused"] for row in cycle.read(root / "evaluation-results/queue-summary.json")["runs"])
    with pytest.raises(ValueError, match="queue already exhausted"):
        cycle.submit(SimpleNamespace(config=config_path))
    assert all(command[0] != "sbatch" for command in external_boundary.commands)


def test_interrupted_source_resumes_without_regenerating_the_completed_map(tmp_path, external_boundary):
    root, config_path, config = prepared_cycle(tmp_path, [{"name": "candidate-e2e-1", "inputs": "candidate-inputs"}])
    external_boundary.interrupt_source = True
    with pytest.raises(asyncio.CancelledError):
        review(root, config_path, config, "first")
    assert review(root, config_path, config, "second") == 0
    assert external_boundary.calls.count("answer_map_v4") == 1
    assert external_boundary.calls.count("source_importance_v4") == 2


def test_final_incomplete_run_keeps_queue_resumable(tmp_path, external_boundary, monkeypatch):
    root, config_path, config = prepared_cycle(tmp_path, [{"name": "candidate-e2e-1", "inputs": "candidate-inputs"}])
    stop_file = tmp_path / "stop-admission"
    monkeypatch.setenv("GEODML_ADMISSION_STOP_FILE", str(stop_file))
    external_boundary.stop_after_map = stop_file
    assert review(root, config_path, config, "first") == 2
    summary = cycle.read(root / "evaluation-results/queue-summary.json")
    assert summary["complete"] is False and summary["status"] == "incomplete"
    assert len(summary["runs"]) == summary["planned_runs"] == 1
    assert cycle.read(cycle.latest_report(root / "evaluation-results/candidate-e2e-1") / "summary.json")["status"] == "incomplete"
    stop_file.unlink()
    external_boundary.stop_after_map = None
    assert review(root, config_path, config, "second") == 0
    assert external_boundary.calls == ["answer_map_v4", "source_importance_v4"]
    assert cycle.read(root / "evaluation-results/queue-summary.json")["complete"] is True


def test_completed_queue_refuses_changed_settings_before_reusing_results(tmp_path, external_boundary):
    root, config_path, config = prepared_cycle(tmp_path, [{"name": "candidate-e2e-1", "inputs": "candidate-inputs"}])
    review(root, config_path, config, "first")
    settings = cycle.read(root / "judge-config.json")
    settings["concurrency"] += 1
    (root / "judge-config.json").write_text(json.dumps(settings))
    calls = list(external_boundary.calls)
    with pytest.raises(ValueError, match="configuration changed"):
        review(root, config_path, config, "second")
    assert external_boundary.calls == calls


@pytest.mark.parametrize("block", ["budget", "unresolved", "settings"])
def test_submission_guards_stop_before_scheduler_or_gpu_work(tmp_path, external_boundary, block):
    root, config_path, _ = prepared_cycle(tmp_path)
    ledger = root / "budget.json"
    budget = cycle.read(ledger)
    if block == "budget":
        budget["submissions"] = [{"phase": "development", "job_id": str(i)} for i in range(4)]
    elif block == "unresolved":
        budget["submissions"] = [{"phase": "development", "submission_id": "unresolved"}]
    else:
        settings = cycle.read(root / "judge-config.json")
        settings["temperature"] = 1
        (root / "judge-config.json").write_text(json.dumps(settings))
    ledger.write_text(json.dumps(budget))
    match = {"budget": "allocation budget exhausted", "unresolved": "unresolved scheduler ownership",
             "settings": "judge configuration changed"}[block]
    with pytest.raises(ValueError, match=match):
        cycle.submit(SimpleNamespace(config=config_path))
    assert external_boundary.commands == []


def test_submission_records_intent_then_accepts_only_its_identified_job(tmp_path, external_boundary, monkeypatch):
    from analysis.scripts import horeka_nemotron as stage
    root, config_path, _ = prepared_cycle(tmp_path)
    assert cycle.submit(SimpleNamespace(config=config_path)) == 0
    submission = cycle.read(root / "budget.json")["submissions"][0]
    assert submission["job_id"] == "6001" and submission["phase"] == "development"
    assert any(c[0] == "sbatch" and "--time=01:00:00" in c for c in external_boundary.commands)
    executed = []
    monkeypatch.setattr(stage, "execute", lambda path: executed.append(path) or 0)
    assert cycle.run_segment(SimpleNamespace(config=config_path)) == 0
    monkeypatch.setenv("SLURM_JOB_ID", "6002")
    with pytest.raises(ValueError, match="not an identified member"):
        cycle.run_segment(SimpleNamespace(config=config_path))
    assert executed == [config_path]


@pytest.mark.parametrize("prior_state", ["missing", "PENDING", "RUNNING", "COMPLETED"])
def test_serial_submission_requires_terminal_prior_job_before_writer_registration(tmp_path, external_boundary, prior_state):
    root, config_path, _ = prepared_cycle(tmp_path)
    assert cycle.submit(SimpleNamespace(config=config_path)) == 0
    assert not (root / "evaluation-results").exists()
    original_ledger = (root / "budget.json").read_bytes()
    external_boundary.commands.clear()
    if prior_state != "missing":
        # Requeued or renamed jobs can have historical terminal accounting; live state wins.
        external_boundary.scheduler_owners = [
            {"job_id": "6001", "state": "COMPLETED", "start_epoch": int(time.time()) - 3600}]
    if prior_state in ("PENDING", "RUNNING"):
        external_boundary.scheduler_jobs = [{"job_id": "6001", "job_name": "renamed-cycle-job",
            "state": prior_state, "start_epoch": int(time.time()) - 3600}]
    if prior_state == "COMPLETED":
        assert cycle.submit(SimpleNamespace(config=config_path)) == 0
        assert len(cycle.read(root / "budget.json")["submissions"]) == 2
    else:
        with pytest.raises(ValueError):
            cycle.submit(SimpleNamespace(config=config_path))
        assert (root / "budget.json").read_bytes() == original_ledger
        assert not any(command[0] == "sbatch" for command in external_boundary.commands)


def test_fixed_subset_refuses_missing_maps_instead_of_regenerating(tmp_path):
    inputs, _ = freeze(tmp_path)
    report = tmp_path / "report"
    report.mkdir()
    (report / "maps.jsonl").write_text("")
    with pytest.raises(ValueError, match="no regeneration permitted"):
        cycle.fixed_subset(report, inputs, tmp_path / "fixed.jsonl")
    assert not (tmp_path / "fixed.jsonl").exists()


@pytest.mark.parametrize("walltime", ["00:30:00", "03:00:00"])
def test_horeka_preparation_freezes_queue_and_uses_finite_cycle_entrypoint(tmp_path, gemma_runtime, walltime):
    root, _, _ = prepared_cycle(tmp_path)
    output = tmp_path / "prepared"
    budget = tmp_path / "shared-cycle-budget.json"
    args = ["prepare", "--workspace", str(tmp_path), "--output", str(output),
            "--evaluation-plan", str(root / "evaluation-plan.json"), "--cycle-budget", str(budget),
            "--account", "fixture", "--approval", "Requested finite development evaluation", "--walltime", walltime]
    if walltime == "03:00:00":
        with pytest.raises(ValueError, match="cannot exceed one hour"):
            pilot.main(args)
        assert not output.exists()
        return
    assert pilot.main(args) == 0
    config = cycle.read(output / "config.json")
    assert config["workload_mode"] == "evaluation-v4"
    assert config["evaluation_plan_sha256"] == judge.file_hash(root / "evaluation-plan.json")
    assert config["source_importance_only"] is True
    assert config["walltime"] == "00:30:00"
    assert cycle.read(output / "evaluation-plan.json") == cycle.read(root / "evaluation-plan.json")
    launcher = (output / "run.sh").read_text()
    assert "run_si_v4_cycle.py" in launcher and "run-segment" in launcher
    registration = cycle.read(budget)["phases"]["development"]
    assert registration == {"config": str((output / "config.json").resolve()),
                            "sha256": judge.file_hash(output / "config.json")}
    assert cycle.checked_config(output / "config.json", cycle.read(budget)) == config


def test_started_segment_waits_for_submission_receipt_before_checking_membership(tmp_path, external_boundary, monkeypatch):
    from analysis.scripts import horeka_nemotron as stage
    root, config_path, _ = prepared_cycle(tmp_path)
    ledger = root / "budget.json"
    budget = cycle.read(ledger)
    budget["submissions"] = [{"phase": "development", "submission_id": "pending-receipt",
                              "config_sha256": judge.file_hash(config_path)}]
    ledger.write_text(json.dumps(budget))
    script = """
import fcntl,json,sys
from pathlib import Path
path=Path(sys.argv[1])
with path.with_suffix(path.suffix+'.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    print('LOCKED',flush=True)
    sys.stdin.readline()
    value=json.loads(path.read_text())
    value['submissions'][0]['job_id']='6001'
    pending=path.with_suffix('.receipt')
    pending.write_text(json.dumps(value))
    pending.replace(path)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(ledger)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    executed = []
    monkeypatch.setattr(stage, "execute", lambda path: executed.append(path) or 0)
    started = threading.Event()
    def start():
        started.set()
        return cycle.run_segment(SimpleNamespace(config=config_path))
    try:
        assert process.stdout.readline().strip() == "LOCKED"
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(start)
            assert started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
            assert executed == []
            process.stdin.write("record receipt\n")
            process.stdin.flush()
            assert future.result(timeout=3) == 0
        assert executed == [config_path]
        assert process.wait(timeout=3) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
