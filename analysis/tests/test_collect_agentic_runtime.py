"""Safety and accounting contracts for live, observational diagnostics."""

import json
from argparse import Namespace

import pytest

from analysis.scripts import collect_agentic_runtime as runtime
from analysis.scripts.collect_agentic_runtime import (
    ClaimReader,
    bounded_log,
    client_identity,
    job_fields,
    safe_output,
    step_command,
    summarize,
)


def test_expired_or_wrong_owner_is_not_live():
    assert not job_fields("JobId=12 JobState=COMPLETED UserId=u(7)", "12", 7)["live"]
    assert not job_fields("JobId=12 JobState=RUNNING UserId=u(8)", "12", 7)["live"]
    assert job_fields("JobId=12 JobState=RUNNING UserId=u(7)", "12", 7)["live"]


def test_step_only_reuses_explicit_allocation():
    cmd = step_command("12", 5, ["python3", "probe.py"])
    assert "--jobid=12" in cmd and "--overlap" in cmd
    assert "--nodes=5" in cmd and "--immediate=15" in cmd
    assert not any("exclusive" in s or "gpus" in s for s in cmd)


def test_identity_does_not_confuse_server_or_other_run(tmp_path):
    root = tmp_path / "run"
    args = [
        "python3",
        "run_agentic_search_integration_smoke.py",
        "--output",
        str(root / "workers/w/output"),
    ]
    assert client_identity(args, root) == "w"
    assert client_identity(["vllm", "--api-key", "secret"], root) is None
    assert (
        client_identity(["python3", "search_vllm_stage.py", "run", *args], root) is None
    )
    assert client_identity(args[:-1] + [str(tmp_path / "other")], root) is None


def test_output_cannot_modify_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(ValueError):
        safe_output(root / "diagnostics", root)
    out = tmp_path / "report"
    safe_output(out, root)
    with pytest.raises(FileExistsError):
        safe_output(out, root)


def test_claim_counts_deduplicate_and_report_bad_records(tmp_path):
    role = tmp_path / "claims/original-llama4/aa"
    role.mkdir(parents=True)
    record = {
        "identity_sha256": "a" * 64,
        "outcome": {
            "result": {"cell_id": "cell"},
            "diagnostics": {"elapsed_seconds": 2},
        },
    }
    (role / ("a" * 64 + ".json")).write_text(json.dumps(record))
    duplicate = {**record, "identity_sha256": "b" * 64}
    (role / ("b" * 64 + ".json")).write_text(json.dumps(duplicate))
    (role / ("c" * 64 + ".failed.json")).write_text(
        json.dumps({"identity_sha256": "c" * 64, "outcome": {}})
    )
    (role / ("d" * 64 + ".json")).write_text("{")
    (role / "ignored.lock").write_text("")
    (role / ".claim.tmp").write_text("{")
    result = ClaimReader().snapshot(tmp_path)["original-llama4"]
    assert result["completed"] == 1 and result["duplicates"] == 1
    assert result["failed"] == 1 and result["invalid"] == 1


def test_log_rotation_and_redaction(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "api_key=secret\nAvg generation throughput: 12.0 tokens/s, Running: 2 reqs, Waiting: 0 reqs\n"
    )
    state = {}
    first = bounded_log(log, state)
    assert first["metrics"][0]["generation_tokens_s"] == 12
    assert "secret" not in json.dumps(first)
    log.write_text("ERROR secret\n")
    assert bounded_log(log, state)["errors"] == 1


def test_summary_zero_rate_unknown_eta_and_worker_delta():
    snapshots = [
        {
            "time": 10,
            "claims": {"original-llama4": {"completed": 2}},
            "workers": {"w": {"materialized": 1}},
        },
        {
            "time": 70,
            "claims": {"original-llama4": {"completed": 2}},
            "workers": {"w": {"materialized": 1}},
        },
    ]
    report = summarize(snapshots)
    assert report["roles"]["original-llama4"]["cells_per_minute"] == 0
    assert report["roles"]["original-llama4"]["eta_minutes"] is None
    assert report["workers"]["w"]["materialized_delta"] == 0


def test_offline_collection_writes_real_report_without_srun(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({"source_git_commit": "frozen"}))
    commands = []

    def fake_command(argv, **kwargs):
        commands.append(argv)
        return {"returncode": 1, "stdout": ""}

    monkeypatch.setattr(runtime, "command", fake_command)
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *a, **kw: pytest.fail("No step allowed")
    )
    output = tmp_path / "report"
    runtime.collect(Namespace(output=output, run_root=root, job_id="12", duration=600))
    saved = json.loads((output / "summary.json").read_text())
    assert saved["live_allocation"] is False
    assert saved["baseline_seconds"] == 0
    assert saved["roles"]["original-llama4"]["completed"] == 0
    assert all(argv[0] != "srun" for argv in commands)


def test_profiler_absence_is_a_report_not_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "clients", lambda *a: [{"worker": "w", "pid": 123}])
    monkeypatch.setattr(runtime.shutil, "which", lambda *a: None)
    monkeypatch.setattr(
        runtime, "command", lambda *a, **kw: pytest.fail("No command allowed")
    )
    runtime.profile_client(
        Namespace(output=tmp_path, run_root=tmp_path, job_id="12", worker="w")
    )
    report = json.loads((tmp_path / "profile-status.json").read_text())
    assert report["reason"] == "py_spy_missing"


def test_profiler_denial_is_recorded_without_stderr_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "clients", lambda *a: [{"worker": "w", "pid": 123}])
    monkeypatch.setattr(runtime.shutil, "which", lambda *a: "/tools/py-spy")
    commands = []

    def denied(argv, **kwargs):
        commands.append(argv)
        return {"returncode": 1, "stdout": "secret"}

    monkeypatch.setattr(runtime, "command", denied)
    runtime.profile_client(
        Namespace(output=tmp_path, run_root=tmp_path, job_id="12", worker="w")
    )
    text = (tmp_path / "profile-status.json").read_text()
    assert "secret" not in text
    assert json.loads(text)["status"] == "attachment_failed"
    assert commands[0][0] == "/tools/py-spy" and "--locals" not in commands[0]


def test_summary_five_nodes_and_invalid_telemetry(tmp_path):
    for index in range(5):
        (tmp_path / f"node-host{index}.jsonl").write_text(
            json.dumps(
                {
                    "clients": [{"cpu_percent_interval": 1200}],
                    "gpu": {
                        "returncode": 0,
                        "stdout": "0, 0, 90000, 150\n1, 100, 90000, 300\n",
                    },
                }
            )
            + "\n{\n"
        )
    result = runtime.node_summary(tmp_path)
    assert len(result) == 5
    assert result["host0"]["mean_sampled_gpu_utilization_percent"] == 50
    assert result["host0"]["errors"] == 1


def test_step_environment_drops_inherited_one_cpu_step(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "step-only")
    monkeypatch.setenv("SLURM_CONF", "/config")
    result = runtime.step_environment()
    assert "SLURM_CPUS_PER_TASK" not in result and "SLURM_JOB_NODELIST" not in result
    assert result["SLURM_CONF"] == "/config"


def test_live_monitor_step_failure_still_produces_report(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text("{}")
    future = runtime.time.strftime(
        "%Y-%m-%dT%H:%M:%S", runtime.time.localtime(runtime.time.time() + 3600)
    )
    job = f"JobId=12 JobState=RUNNING UserId=u({runtime.os.getuid()}) NumNodes=5 EndTime={future}"
    monkeypatch.setattr(
        runtime, "command", lambda *a, **kw: {"returncode": 0, "stdout": job}
    )
    clock = iter([0, 2])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(clock))
    launched = []

    class FailedMonitor:
        def __init__(self, argv, **kwargs):
            launched.append(argv)

        def wait(self, **kwargs):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(runtime.subprocess, "Popen", FailedMonitor)
    output = tmp_path / "report"
    runtime.collect(Namespace(output=output, run_root=root, job_id="12", duration=1))
    report = json.loads((output / "summary.json").read_text())
    assert report["probe_exit_code"] == 1
    assert report["node_coverage_complete"] is False
    assert len(launched) == 1 and "--jobid=12" in launched[0]


def test_node_probe_writes_telemetry_without_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "node0.cluster")
    monkeypatch.setattr(
        runtime, "clients", lambda *a: [{"pid": 1, "start_ticks": 3, "cpu_seconds": 10}]
    )
    monkeypatch.setattr(
        runtime,
        "command",
        lambda *a, **kw: {"returncode": -1, "stdout": "", "error": "FileNotFoundError"},
    )
    runtime.node_probe(
        Namespace(output=tmp_path, run_root=tmp_path, job_id="12", duration=0)
    )
    record = json.loads((tmp_path / "node-node0.jsonl").read_text())
    assert record["gpu"]["returncode"] == -1
    assert record["clients"][0]["pid"] == 1
    assert (tmp_path / "hardware-node0.json").exists()
