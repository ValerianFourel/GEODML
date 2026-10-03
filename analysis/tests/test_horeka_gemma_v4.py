"""Persisted Gemma bout continuation and cross-cluster reservation contracts."""
import asyncio
import gzip
import json
import os
import shutil
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


def test_gpu_preparation_builds_verified_finite_shards_from_saved_cells(tmp_path, monkeypatch):
    from analysis.scripts import verify_inference_allocation as boundary
    source = dataset(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    spec = {"git_commit": "a" * 40, "workspace": str(tmp_path), "account": "test",
            "repo_id": "fixture/private", "sources": [f"{source}:qwen38"], "exclude_inputs": [],
            "authorization": "five-hour finite fixture"}
    bulk.save(root / "preparation.json", spec)
    settings = bulk.read(bulk.REPO / "analysis/config/si_v4_gemma_full_pass.template.json")
    bulk.save(root / "judge-config.json", settings)
    monkeypatch.setattr(bulk, "clean_pin", lambda: "a" * 40)
    monkeypatch.setattr(bulk, "storage", lambda *a: None)
    monkeypatch.setattr(boundary, "verify", lambda cluster: {"cluster": cluster, "fixture_boundary": True})
    assert bulk.prepare(SimpleNamespace(output=root)) == 0
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
