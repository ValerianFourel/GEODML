"""Persisted Gemma bout continuation and cross-cluster reservation contracts."""
import asyncio
import gzip
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_gemma_v4 as bulk, horeka_si_v4 as pilot
from analysis.scripts import run_source_importance_judge as judge
from analysis.scripts.run_si_v4_cycle import latest_report
from analysis.tests.test_source_importance_v4 import CONFIG, freeze
from analysis.tests.test_si_v4_cycle import external_boundary  # shared external transport fixture
from analysis.tests.test_source_importance_pipeline import dataset


def prepared(tmp_path):
    source, _ = freeze(tmp_path)
    root = tmp_path / "shard"
    root.mkdir()
    shutil.copytree(source, root / "inputs")
    settings = {**CONFIG, "source_importance_only": True,
                "chat_template_sha256": bulk.hashlib.sha256(b"fixture template").hexdigest()}
    bulk.save(root / "judge-config.json", settings)
    config = {"workload_mode": "gemma-v4-bulk", "model_id": "fixture", "account": "test",
              "workspace": str(tmp_path), "git_commit": "a" * 40,
              "judge_config_sha256": judge.file_hash(root / "judge-config.json"),
              "inventories": {"inputs": bulk.read(root / "inputs/manifest.json")}}
    bulk.save(root / "config.json", config)
    return root


def run(root, name):
    return asyncio.run(pilot.review(SimpleNamespace(config=root / "config.json",
        output=root / "attempts" / name / "trial", server_model_name="fixture", base_url="http://fixture")))


@pytest.mark.parametrize("fail_map", [False, True])
def test_bulk_bouts_resume_saved_tasks_and_never_retry_terminal_failures(tmp_path, monkeypatch, external_boundary, fail_map):
    root = prepared(tmp_path)
    monkeypatch.setattr(bulk, "capture_quota", lambda *a: {})
    external_boundary.fail_map = fail_map
    if not fail_map:
        stop = tmp_path / "stop"
        external_boundary.stop_after_map = stop
        monkeypatch.setenv("GEODML_ADMISSION_STOP_FILE", str(stop))
        assert run(root, "first") == 2
        summary = bulk.read(latest_report(root / "results") / "summary.json")
        assert summary["status"] == "incomplete"
        assert external_boundary.calls == ["answer_map_v4"]
        stop.unlink()
        external_boundary.stop_after_map = None
    assert run(root, "second") == (2 if fail_map else 0)
    calls = list(external_boundary.calls)
    summary = bulk.read(latest_report(root / "results") / "summary.json")
    assert summary["counts"]["cells_complete"] == (0 if fail_map else 1)
    assert run(root, "third") == (2 if fail_map else 0)
    assert external_boundary.calls == calls
    assert summary["states"]["not_requested"] == 1
    reconciled = bulk.reconcile_output(root / "results")
    assert reconciled["remaining_estimate"]["cells_with_pending_tasks"] == 0
    assert reconciled["remaining_estimate"]["next_bout_expected_minutes"] == 0


def reservation(tmp_path, name, ids):
    root = tmp_path / name
    root.mkdir()
    with gzip.open(root / "claims.jsonl.gz", "wt") as stream:
        for tid in ids:
            stream.write(json.dumps({"map_task_id": tid}) + "\n")
    plan = {"plan_id": name, "repo_id": "fixture/private", "git_commit": "a" * 40,
            "claims_sha256": judge.file_hash(root / "claims.jsonl.gz")}
    bulk.save(root / "plan.json", plan)
    return root, plan


def test_private_reservation_conflicts_and_idempotent_restart(tmp_path, monkeypatch):
    from analysis.interpretability.pipeline import agentic_hour_sync as sync
    state = {"revision": "0", "files": {}, "writes": 0}

    class Store:
        def __init__(self, repo):
            assert repo == "fixture/private"

        def head(self):
            return state["revision"]

        def read(self, path, revision):
            assert revision == state["revision"]
            return state["files"].get(path)

        def commit(self, revision, files, message):
            if revision != state["revision"]:
                raise sync.ConflictError("revision changed")
            state["files"].update(files)
            state["writes"] += 1
            state["revision"] = str(state["writes"])
            return state["revision"]

    monkeypatch.setattr(sync, "HubStore", Store)
    root, plan = reservation(tmp_path, "one", ["map-a", "map-b"])
    bulk.reserve(plan, root)
    assert bulk.read(root / "reservation.json")["revision"] == "1"
    bulk.reserve(plan, root)
    assert state["writes"] == 1
    other_root, other = reservation(tmp_path, "two", ["map-b", "map-c"])
    with pytest.raises(ValueError, match="already reserved"):
        bulk.reserve(other, other_root)
    assert not (other_root / "reservation.json").exists()
    third_root, third = reservation(tmp_path, "three", ["map-d"])
    bulk.reserve(third, third_root)
    assert state["writes"] == 2
    registry = json.loads(state["files"][bulk.REGISTRY])
    assert set(registry["plans"]) == {"one", "three"}
    # A labelled recovery may re-claim exactly the maps of the plan it recovers, and only that plan's.
    rec_root, rec = reservation(tmp_path, "recovery", ["map-b"])
    rec["recovery_of"] = "one"
    bulk.reserve(rec, rec_root)
    assert set(json.loads(state["files"][bulk.REGISTRY])["plans"]) == {"one", "three", "recovery"}
    bad_root, bad = reservation(tmp_path, "bad-recovery", ["map-b", "map-d"])
    bad["recovery_of"] = "one"
    with pytest.raises(ValueError, match="already reserved by (three|recovery)"):
        bulk.reserve(bad, bad_root)
    assert all(row["state"] == "reserved" for row in registry["plans"].values())


def test_reconciliation_does_not_recover_a_live_task_owner(tmp_path, monkeypatch):
    root = prepared(tmp_path)
    coordinator = judge.Coordinator(root / "inputs", root / "results", CONFIG)
    coordinator.db.execute("UPDATE tasks SET state='running',writer_id='owner',job_id='123' WHERE kind='answer_map'")
    coordinator.db.commit()
    coordinator.close()
    monkeypatch.setattr(bulk.subprocess, "check_output", lambda *a, **k: "123\n")
    with pytest.raises(ValueError, match="live or unknown"):
        bulk.reconcile_output(root / "results")
    assert not (root / "results/recovery-evidence.json").exists()


@pytest.mark.parametrize("receipt_job,comment_matches", [(None, True), ("901", True), ("other", True), (None, False)])
def test_worker_binds_durable_intent_before_server_start(tmp_path, monkeypatch, receipt_job, comment_matches):
    root = prepared(tmp_path)
    config = bulk.read(root / "config.json")
    config["bulk_root"] = str(tmp_path)
    bulk.save(root / "config.json", config)
    settings = bulk.read(root / "judge-config.json")
    settings["runtime_versions"] = {"fixture": "1"}
    bulk.save(root / "judge-config.json", settings)
    shard = {"id": "shard", "directory": str(root), "max_allocations": 1,
             "config_sha256": judge.file_hash(root / "config.json")}
    plan = {"plan_id": "p", "git_commit": "a" * 40, "claims_sha256": "c" * 64,
            "account": "test", "shards": [shard]}
    bulk.save(tmp_path / "plan.json", plan)
    bulk.save(tmp_path / "reservation.json", {"plan_id": "p", "claims_sha256": "c" * 64})
    intent_dir = root / "submissions/attempt-0001"
    comment = "geodml-gemma-v4:" + "a" * 32
    bulk.save(intent_dir / "intent.json", {"shard_id": "shard", "comment": comment,
        "git_commit": "a" * 40, "config_sha256": shard["config_sha256"],
        "plan_sha256": judge.file_hash(tmp_path / "plan.json")})
    if receipt_job is not None:
        bulk.save(intent_dir / "receipt.json", {"job_id": receipt_job})
    monkeypatch.setenv("SLURM_JOB_ID", "901")
    monkeypatch.setattr(bulk, "checked_plan", lambda p: plan)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    monkeypatch.setattr(bulk.gemma, "runtime_check", lambda: {"fixture": "1", "vllm": "fixture"})
    actual_comment = comment if comment_matches else "geodml-gemma-v4:" + "b" * 32
    monkeypatch.setattr(bulk.subprocess, "check_output", lambda *a, **k:
        f"Account=test Partition=accelerated UserId=fixture({os.getuid()}) Comment={actual_comment}")
    loaded = []
    monkeypatch.setattr(bulk.stage, "execute", lambda p: loaded.append(p) or 0)
    args = SimpleNamespace(config=root / "config.json")
    if receipt_job == "other" or not comment_matches:
        with pytest.raises(ValueError, match="submission|allocation"):
            bulk.run_shard(args)
        assert loaded == []
    else:
        assert bulk.run_shard(args) == 0
        assert bulk.read(intent_dir / "worker-binding.json")["job_id"] == "901"
        with pytest.raises(FileExistsError):
            bulk.run_shard(args)
        assert len(loaded) == 1


@pytest.mark.parametrize("field", ["source", "exclude_inputs", "repo_id"])
def test_restart_rejects_changed_scope_before_submission(tmp_path, monkeypatch, field):
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:qwen38"], exclude_inputs=[])
    spec = {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test",
            "repo_id": args.repo_id, "sources": args.source, "exclude_inputs": []}
    bulk.save(root / "preparation.json", spec)
    setattr(args, field, {"source": [f"{tmp_path}/other:qwen38"], "exclude_inputs": [tmp_path / "prior"],
                          "repo_id": "fixture/different"}[field])
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    with pytest.raises(ValueError, match="saved preparation differs"):
        bulk.start(args)
    assert not (root / "PREPARATION_SUBMISSION_ATTEMPTED").exists()


@pytest.mark.parametrize("login", [False, True, "fresh", "cpu"])
def test_preparation_builds_verified_finite_shards_from_saved_cells(tmp_path, monkeypatch, login):
    from analysis.scripts import verify_inference_allocation as boundary
    source = dataset(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    spec = {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test",
            "repo_id": "fixture/private", "sources": [f"{source}:qwen38"], "exclude_inputs": [],
            "authorization": "five-hour finite fixture"}
    if login == "fresh":
        spec["preparation_execution"] = "login"  # chosen at start; no GPU takeover receipts exist
    cpu = login == "cpu"
    if cpu:
        spec["preparation_execution"] = "cpu"
        login = False
        scratch = tmp_path / "node-local"
        scratch.mkdir()
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setenv("TMPDIR", str(scratch))
    bulk.save(root / "preparation.json", spec)
    settings = bulk.read(bulk.REPO / "analysis/config/si_v4_gemma_full_pass.template.json")
    bulk.save(root / "judge-config.json", settings)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    def verify(cluster):
        assert not login and not cpu, 'login and CPU preparation must never request a GPU boundary'
        return {"cluster": cluster, "fixture_boundary": True}
    monkeypatch.setattr(boundary, "verify", verify)
    if login is True:
        bulk.save(root / 'prequeue/login-takeover.json', {'helper_pin': 'a'*40, 'preparation_job': '99'})
        bulk.save(root / 'prequeue/login-preparation-retired.json', {'job_id': '99', 'state': 'CANCELLED'})
    if login:
        original = bulk.subprocess.check_output
        def git(command, **kwargs):
            if command[:3] == ['git', '-C', str(bulk.REPO)]:
                return 'a' * 40 + '\n' if command[3] == 'rev-parse' else ''
            return original(command, **kwargs)
        monkeypatch.setattr(bulk.subprocess, 'check_output', git)
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    assert bulk.prepare(SimpleNamespace(output=root, command='prepare-login' if login else 'prepare',
                                        inference_repository=bulk.REPO)) == 0
    plan = bulk.checked_plan(root)
    assert plan["cells"] == 3
    assert len(plan["shards"]) == 1  # all three synthetic answers share their map
    assert plan["maximum_allocations"] == 1
    assert plan["maximum_node_hours"] == 6  # one prep hour, one five-hour inference allocation
    assert plan["maximum_gpu_hours"] == 24
    assert len(list(judge.rows(root / "claims.jsonl.gz"))) == 1
    shard = plan["shards"][0]
    directory = bulk.Path(shard["directory"])
    config = bulk.read(directory / "config.json")
    assert config["workload_mode"] == "gemma-v4-bulk"
    assert config["walltime"] == "05:00:00"
    assert bulk.read(directory / "judge-config.json") == settings
    if login:
        assert plan['preparation_execution']['execution'] == 'login'
        assert plan['preparation_execution']['gpu_used'] is False
        assert plan['git_commit'] == 'a' * 40
    if cpu:
        assert plan['preparation_execution']['execution'] == 'cpu_slurm'
        assert plan['preparation_execution']['slurm_job_id'] == '123'
        assert plan['preparation_execution']['gpu_used'] is False
        assert list(scratch.iterdir()) == []  # the node-local index is removed
    assert not list((root / "frozen").glob("task-index*"))
    (directory / "inputs/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        bulk.checked_plan(root)


def test_preparation_aged_out_of_squeue_uses_terminal_accounting(tmp_path, monkeypatch):
    from analysis.scripts import horeka_gemma_v4_sender as sender
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:qwen38"], exclude_inputs=[])
    bulk.save(root / "preparation.json", {"git_commit": "a" * 40, "workspace": str(tmp_path),
        "account": "test", "repo_id": args.repo_id, "sources": args.source, "exclude_inputs": [],
        "preparation_deadline_epoch": bulk.time.time() + 3600})
    bulk.save(root / "preparation-submission.json", {"returncode": 0, "stdout": "42\n"})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    handed_off = []
    monkeypatch.setattr(sender, "send", lambda path: handed_off.append(path) or 0)

    def scheduler(command, **kwargs):
        if command[0] == "squeue":
            if "-j" in command:
                raise bulk.subprocess.CalledProcessError(1, command, stderr="Invalid job id specified")
            bulk.save(root / "plan.json", {"fixture": "preparation completed between polls"})
            return "77|RUNNING\n"  # unrelated allocation is preserved
        if command[0] == "sacct":
            return "42|COMPLETED\n"
        raise AssertionError(command)

    monkeypatch.setattr(bulk.subprocess, "check_output", scheduler)
    assert bulk.start(args) == 0
    assert handed_off == [root]


def test_takeover_login_preparation_still_requires_retired_gpu_job(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    bulk.save(root / "preparation.json", {"git_commit": "a" * 40, "sources": []})
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(FileNotFoundError):
        bulk.prepare(SimpleNamespace(output=root, command="prepare-login", inference_repository=bulk.REPO))


def test_start_on_login_prepares_once_without_sbatch_then_sends(tmp_path, monkeypatch):
    from analysis.scripts import horeka_gemma_v4_sender as sender
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:llama4"], exclude_inputs=[], prepare_on_login=True)
    spec = {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test", "repo_id": args.repo_id,
            "sources": args.source, "exclude_inputs": [], "preparation_execution": "login"}
    bulk.save(root / "preparation.json", spec)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    monkeypatch.setattr(bulk, "checked_plan", lambda path: bulk.read(path / "plan.json"))
    calls, handed_off = [], []

    def run(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["nice", "-n", "10"] and "prepare-login" in command
        assert kwargs["timeout"] == 3600 and kwargs["env"]["OMP_NUM_THREADS"] == "1"
        bulk.save(root / "plan.json", {"fixture": "login preparation"})

    monkeypatch.setattr(bulk.subprocess, "run", run)
    monkeypatch.setattr(sender, "send", lambda path: handed_off.append(path) or 0)
    assert bulk.start(args) == 0
    assert len(calls) == 1 and handed_off == [root]
    assert not (root / "PREPARATION_SUBMISSION_ATTEMPTED").exists()
    # A restart after the plan exists goes straight to the sender.
    assert bulk.start(args) == 0 and len(calls) == 1
    # An interrupted login preparation is never repeated automatically.
    (root / "plan.json").unlink()
    with pytest.raises(ValueError, match="no automatic repeat"):
        bulk.start(args)
    # The preparation mode is part of the pinned scope.
    args.prepare_on_login = False
    with pytest.raises(ValueError, match="saved preparation differs"):
        bulk.start(args)


def test_cpu_preparation_index_location_does_not_change_frozen_inputs(tmp_path):
    from analysis.scripts import prepare_source_importance_tasks as freeze_tasks
    source = dataset(tmp_path)
    scratch = tmp_path / "node-local"
    scratch.mkdir()
    common = ["--source", f"{source}:qwen38", "--protocol", "si-v4"]
    assert freeze_tasks.main([*common, "--output", str(tmp_path / "shared")]) == 0
    assert freeze_tasks.main([*common, "--output", str(tmp_path / "local"), "--index-directory", str(scratch)]) == 0
    for name in ("tasks.jsonl.gz", "cells.jsonl.gz"):
        assert gzip.decompress((tmp_path / "shared" / name).read_bytes()) == \
            gzip.decompress((tmp_path / "local" / name).read_bytes())  # gzip headers carry a timestamp
    assert list(scratch.iterdir()) == []


def test_start_on_cpu_submits_one_four_hour_cpuonly_preparation(tmp_path, monkeypatch):
    from analysis.scripts import horeka_gemma_v4_sender as sender
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:llama4"], exclude_inputs=[], prepare_on_cpu=True)
    bulk.save(root / "preparation.json", {"git_commit": "a" * 40, "workspace": str(tmp_path),
        "account": "test", "repo_id": args.repo_id, "sources": args.source, "exclude_inputs": [],
        "preparation_execution": "cpu", "preparation_deadline_epoch": bulk.time.time() + 3600})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    submissions, handed_off = [], []

    def run(command, **kwargs):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout="42\n", stderr="")

    def scheduler(command, **kwargs):
        if command[0] == "squeue":
            bulk.save(root / "plan.json", {"fixture": "CPU preparation completed"})
            return ""
        if command[0] == "sacct":
            return "42|COMPLETED\n"
        raise AssertionError(command)

    monkeypatch.setattr(bulk.subprocess, "run", run)
    monkeypatch.setattr(bulk.subprocess, "check_output", scheduler)
    monkeypatch.setattr(sender, "send", lambda path: handed_off.append(path) or 0)
    assert bulk.start(args) == 0
    assert handed_off == [root] and len(submissions) == 1
    command = submissions[0]
    assert command[0] == "sbatch" and "--no-requeue" in command
    assert "--partition=cpuonly" in command and "--time=04:00:00" in command
    assert not any(part.startswith(("--gres", "--partition=accelerated")) or part == "--exclusive" for part in command)
    script = (root / "prepare.sh").read_text()
    assert "srun" not in script and "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY" not in script
    assert script.rstrip().endswith(f"prepare --output {root}")
    # A restart after the plan exists goes straight to the sender; no second allocation.
    assert bulk.start(args) == 0 and len(submissions) == 1
    # The preparation mode is part of the pinned scope.
    args.prepare_on_cpu = False
    with pytest.raises(ValueError, match="saved preparation differs"):
        bulk.start(args)
    args.prepare_on_cpu = args.prepare_on_login = True
    with pytest.raises(ValueError, match="one preparation mode"):
        bulk.start(args)


def _cpu_run(tmp_path, monkeypatch, name, source, reuse=None):
    root = tmp_path / name
    root.mkdir()
    spec = {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test",
            "repo_id": "fixture/private", "sources": [f"{source}:qwen38"], "exclude_inputs": [],
            "authorization": "five-hour finite fixture", "preparation_execution": "cpu"}
    if reuse:
        spec["reuse_frozen"] = reuse
    bulk.save(root / "preparation.json", spec)
    bulk.save(root / "judge-config.json", bulk.read(bulk.REPO / "analysis/config/si_v4_gemma_full_pass.template.json"))
    return root


def test_cpu_preparation_adopts_a_verified_freeze_without_freezing_again(tmp_path, monkeypatch):
    from analysis.scripts import prepare_source_importance_tasks as freeze_tasks
    source = dataset(tmp_path)
    scratch = tmp_path / "node-local"
    scratch.mkdir()
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("TMPDIR", str(scratch))
    first = _cpu_run(tmp_path, monkeypatch, "timed-out", source)
    assert bulk.prepare(SimpleNamespace(output=first, command="prepare")) == 0
    old = first / "frozen"
    reuse = {"path": str(old), "manifest_sha256": judge.file_hash(old / "manifest.json")}

    def no_freeze(argv):
        raise AssertionError("a verified freeze must not be rebuilt")

    monkeypatch.setattr(freeze_tasks, "main", no_freeze)
    second = _cpu_run(tmp_path, monkeypatch, "reuse", source, reuse)
    assert bulk.prepare(SimpleNamespace(output=second, command="prepare")) == 0
    plan = bulk.checked_plan(second)
    assert plan["cells"] == 3 and plan["reused_frozen"] == reuse
    assert judge.file_hash(second / "frozen/manifest.json") == reuse["manifest_sha256"]
    assert bulk.read(second / "frozen-reuse.json")["path"] == str(old)
    assert list(scratch.iterdir()) == []  # node-local partition index removed
    first_shard = bulk.read(first / "plan.json")["shards"][0]["files"]
    assert plan["shards"][0]["files"]["inputs/manifest.json"] == first_shard["inputs/manifest.json"]
    # A changed or mismatched freeze is refused, never silently adopted.
    changed = _cpu_run(tmp_path, monkeypatch, "changed", source, {**reuse, "manifest_sha256": "0" * 64})
    with pytest.raises(ValueError, match="manifest changed since start"):
        bulk.prepare(SimpleNamespace(output=changed, command="prepare"))
    other = dataset(tmp_path / "other")
    mismatched = _cpu_run(tmp_path, monkeypatch, "mismatched", other, reuse)
    with pytest.raises(ValueError, match="different generator datasets"):
        bulk.prepare(SimpleNamespace(output=mismatched, command="prepare"))


def test_start_reusing_a_freeze_submits_one_hour_cpu_preparation(tmp_path, monkeypatch):
    from analysis.scripts import horeka_gemma_v4_sender as sender
    frozen = tmp_path / "old/frozen"
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text("{}")
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:llama4"], exclude_inputs=[], prepare_on_cpu=True,
                           reuse_frozen=frozen)
    reuse = {"path": str(frozen.resolve()), "manifest_sha256": judge.file_hash(frozen / "manifest.json")}
    bulk.save(root / "preparation.json", {"git_commit": "a" * 40, "workspace": str(tmp_path),
        "account": "test", "repo_id": args.repo_id, "sources": args.source, "exclude_inputs": [],
        "preparation_execution": "cpu", "reuse_frozen": reuse,
        "preparation_deadline_epoch": bulk.time.time() + 3600})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    submissions = []

    def run(command, **kwargs):
        submissions.append(command)
        return SimpleNamespace(returncode=0, stdout="43\n", stderr="")

    def scheduler(command, **kwargs):
        if command[0] == "squeue":
            bulk.save(root / "plan.json", {"fixture": "partitioned from the reused freeze"})
            return ""
        return "43|COMPLETED\n"

    monkeypatch.setattr(bulk.subprocess, "run", run)
    monkeypatch.setattr(bulk.subprocess, "check_output", scheduler)
    monkeypatch.setattr(sender, "send", lambda path: 0)
    assert bulk.start(args) == 0
    assert "--time=01:00:00" in submissions[0] and "--partition=cpuonly" in submissions[0]
    args.reuse_frozen = None  # the reused freeze is part of the pinned scope
    with pytest.raises(ValueError, match="saved preparation differs"):
        bulk.start(args)
    args.reuse_frozen, args.prepare_on_cpu = frozen, False
    with pytest.raises(ValueError, match="requires --prepare-on-cpu"):
        bulk.start(args)


def test_partition_index_location_does_not_change_shards(tmp_path):
    from analysis.scripts import prepare_source_importance_tasks as freeze_tasks
    from analysis.scripts.partition_si_v4_tasks import partition
    source = dataset(tmp_path)
    frozen = tmp_path / "frozen"
    assert freeze_tasks.main(["--source", f"{source}:qwen38", "--output", str(frozen), "--protocol", "si-v4",
                              "--max-tokens", "4096", "--map-max-tokens", "4096",
                              "--truncation-sensitivity-fraction", "0"]) == 0
    scratch = tmp_path / "node-local"
    scratch.mkdir()
    shared = partition(frozen, tmp_path / "shards-shared")
    local = partition(frozen, tmp_path / "shards-local", index_directory=scratch)
    assert [s["cells"] for s in shared] == [s["cells"] for s in local]
    for a, b in zip(shared, local):
        for name in ("inputs/cells.jsonl.gz", "inputs/tasks.jsonl.gz"):
            assert (bulk.Path(a["directory"]) / name).read_bytes() == (bulk.Path(b["directory"]) / name).read_bytes()
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("dirty,head,ok", [("", "b" * 40, True), (" M x.py", "b" * 40, False), ("", "c" * 40, False)])
def test_newer_sender_checkout_verifies_the_pinned_bout_checkout(tmp_path, monkeypatch, dirty, head, ok):
    pinned = tmp_path / "checkouts/bout"
    root = tmp_path / "workspace/run"
    root.mkdir(parents=True)
    with gzip.GzipFile(filename=str(root / "claims.jsonl.gz"), mode="wb", mtime=0):
        pass
    bulk.save(root / "plan.json", {"format_version": bulk.FORMAT, "git_commit": "b" * 40,
        "repository": str(pinned), "root": str(root), "walltime": bulk.WALLTIME, "max_inflight": 200,
        "poll_seconds": 600, "job_name": bulk.JOB_NAME, "partition": "accelerated",
        "workspace": str(tmp_path / "workspace"), "shards": [], "maximum_allocations": 0,
        "claims_sha256": judge.file_hash(root / "claims.jsonl.gz")})
    def git(command, **kwargs):
        repo, action = command[2], command[3]
        if repo == str(bulk.REPO):  # the sender's own checkout: clean, on a newer commit
            return "a" * 40 + "\n" if action == "rev-parse" else ""
        assert repo == str(pinned)
        return head + "\n" if action == "rev-parse" else dirty
    monkeypatch.setattr(bulk.subprocess, "check_output", git)
    if ok:
        assert bulk.checked_plan(root)["git_commit"] == "b" * 40
    else:
        with pytest.raises(ValueError):
            bulk.checked_plan(root)


def test_start_waits_for_another_gemma_sender_then_sends(tmp_path, monkeypatch):
    from analysis.scripts import horeka_gemma_v4_sender as sender
    root = tmp_path / "run"
    root.mkdir()
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:llama4"], exclude_inputs=[], prepare_on_cpu=True)
    bulk.save(root / "preparation.json", {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test",
        "repo_id": args.repo_id, "sources": args.source, "exclude_inputs": [], "preparation_execution": "cpu"})
    bulk.save(root / "plan.json", {"deadline_epoch": 10_000})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    now = [1_000.0]
    sleeps = []
    monkeypatch.setattr(bulk.time, "time", lambda: now[0])
    monkeypatch.setattr(bulk.time, "sleep", lambda s: sleeps.append(s) or now.__setitem__(0, now[0] + s))
    calls = []
    def send(path):
        calls.append(path)
        if len(calls) < 3:
            raise sender.SenderBusy("another Gemma sender holds the lock")
        return 0
    monkeypatch.setattr(sender, "send", send)
    assert bulk.start(args) == 0
    assert len(calls) == 3 and sleeps == [600, 600]
    # The wait is finite: it stops at the plan deadline instead of looping forever.
    calls.clear()
    now[0] = 9_500.0
    monkeypatch.setattr(sender, "send", lambda path: (_ for _ in ()).throw(sender.SenderBusy("busy")))
    with pytest.raises(ValueError, match="deadline"):
        bulk.start(args)



def test_map_recovery_start_records_cells_attempts_and_the_recovered_plan(tmp_path, monkeypatch):
    from analysis.scripts import horeka_gemma_v4_sender as sender
    root = tmp_path / "rec"
    cells = tmp_path / "cells.txt"
    cells.write_text("cell-a\ncell-b\n")
    args = SimpleNamespace(workspace=tmp_path, output=root, account="test", repo_id="fixture/private",
                           source=[f"{tmp_path}/dataset:llama4"], exclude_inputs=[], prepare_on_cpu=True,
                           cells=cells, map_validation_attempts=4, recovery_of="gemma-v4-old")
    with pytest.raises(ValueError, match="--cells and --recovery-of"):
        bulk.start(SimpleNamespace(**{**vars(args), "recovery_of": None}))
    with pytest.raises(ValueError, match="from 2 to 6"):
        bulk.start(SimpleNamespace(**{**vars(args), "map_validation_attempts": 9}))
    spec = {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test", "repo_id": args.repo_id,
            "sources": [f"{tmp_path.resolve()}/dataset:llama4"], "exclude_inputs": [], "preparation_execution": "cpu",
            "cells": {"path": str(cells.resolve()), "sha256": bulk.judge.file_hash(cells)},
            "map_validation_attempts": 4, "recovery_of": "gemma-v4-old"}
    root.mkdir()
    bulk.save(root / "preparation.json", spec)
    bulk.save(root / "plan.json", {"deadline_epoch": 10**12})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(sender, "send", lambda path: 0)
    assert bulk.start(args) == 0
    # A restart with different recovery options is refused, never silently merged.
    with pytest.raises(ValueError, match="saved preparation differs"):
        bulk.start(SimpleNamespace(**{**vars(args), "map_validation_attempts": 3}))


def test_judge_config_accepts_a_bounded_map_attempt_limit(tmp_path):
    base = json.loads((Path(bulk.REPO) / "analysis/config/si_v4_gemma_full_pass.template.json").read_text())
    base["tokenizer_path"] = "/tmp/tokenizer"
    for value, ok in ((4, True), (2, True), (1, False), (7, False), ("4", False)):
        path = tmp_path / f"c{value}.json"
        path.write_text(json.dumps({**base, "map_validation_attempts": value}))
        if ok:
            assert judge.load_config(path)["map_validation_attempts"] == value
        else:
            with pytest.raises(ValueError, match="map_validation_attempts"):
                judge.load_config(path)
