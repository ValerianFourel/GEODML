"""Serving credentials are ephemeral child state, not experiment configuration."""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import pytest

from analysis.scripts import inference_endpoint_security as security
from analysis.scripts import search_vllm_stage as stage
from analysis.tests.test_inference_endpoint_security import endpoint
from analysis.tests.test_search_vllm_stage import gpu_inventory, profile


@pytest.fixture
def stage_run(tmp_path, monkeypatch):
    record = profile()
    path = tmp_path / "profile.json"
    stage.create_or_verify_profile(path, record)
    original = path.read_bytes()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("VLLM_API_KEY", "inherited-not-a-real-key")
    monkeypatch.setenv("VLLM_HOST_IP", "0.0.0.0")
    monkeypatch.setattr(stage, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(stage, "discover_visible_gpus", gpu_inventory)
    monkeypatch.setattr(stage, "ensure_port_available", lambda *args: None)
    monkeypatch.setattr(stage, "_terminate_group", lambda *args: None)
    monkeypatch.setattr(stage, "wait_for_controller", lambda *args: 0)
    monkeypatch.setattr(stage, "ensure_private_network_namespace", lambda command: isolation_receipt())
    monkeypatch.setattr(stage, "verify_private_network_namespace", isolation_receipt)
    launches = []

    def spawn(argv, **kwargs):
        assert kwargs["close_fds"] is True
        launches.append((list(argv), kwargs["env"]))
        return SimpleNamespace(pid=100 + len(launches), poll=lambda: None)

    monkeypatch.setattr(stage.subprocess, "Popen", spawn)

    def run():
        return stage.run_stage(path, tmp_path / "server.log", ["fixture-controller"],
                               cache_base=tmp_path / "cache", startup_timeout_seconds=1)

    return run, launches, path, original


def receipt():
    return {
        "mode": "per-run-bearer-v1", "status": "verified", "loopback_only": True,
        "anonymous_rejected": True, "wrong_key_rejected": True, "model_verified": True,
    }


def isolation_receipt():
    return {
        "format_version": "geodml-network-namespace-v1", "status": "verified",
        "outside_parent_pid": 99, "outside_namespace": {"device": 4, "inode": 50},
        "current_namespace": {"device": 4, "inode": 51}, "interfaces": ["lo"],
        "loopback_up": True, "private_bind_verified": True, "external_tcp_isolated": True,
    }


def test_stage_shares_fresh_key_only_through_child_environments(stage_run, monkeypatch, tmp_path):
    run, launches, path, original = stage_run
    probes = []

    def ready(*args, **kwargs):
        probes.append(kwargs)
        return receipt()

    monkeypatch.setattr(stage, "wait_until_ready", ready)
    assert run() == 0
    assert len(launches) == 2
    server_key = launches[0][1]["VLLM_API_KEY"]
    assert server_key != "inherited-not-a-real-key"
    assert len(server_key) >= 32
    assert probes[0]["api_key"] == server_key
    for argv, env in launches:
        assert env["VLLM_API_KEY"] == server_key
        assert env["GEODML_INFERENCE_AUTH_REQUIRED"] == "1"
        assert env["VLLM_HOST_IP"] == "127.0.0.1"
        assert server_key not in " ".join(argv)
        assert "--api-key" not in argv
    assert os.environ["VLLM_API_KEY"] == "inherited-not-a-real-key"
    assert path.read_bytes() == original
    runtime = (tmp_path / "server.log.runtime.json").read_text()
    assert server_key not in runtime
    assert json.loads(runtime)["access_control"] == receipt()
    assert json.loads(runtime)["network_isolation"] == isolation_receipt()
    assert stat.S_IMODE((tmp_path / "server.log").stat().st_mode) == 0o600
    launches.clear()
    assert run() == 0
    assert launches[0][1]["VLLM_API_KEY"] != server_key


def test_security_failure_never_starts_controller_and_redacts_echo(stage_run, monkeypatch, tmp_path):
    run, launches, _, _ = stage_run

    def rejected(*args, **kwargs):
        key = kwargs["api_key"]
        (tmp_path / "server.log").write_text("server echoed " + key)
        raise security.EndpointSecurityError("fixture echoed " + key)

    monkeypatch.setattr(stage, "wait_until_ready", rejected)
    with pytest.raises(RuntimeError) as caught:
        run()
    assert len(launches) == 1
    key = launches[0][1]["VLLM_API_KEY"]
    assert key not in str(caught.value)
    assert caught.value.__suppress_context__
    runtime = (tmp_path / "server.log.runtime.json").read_text()
    assert key not in runtime
    assert json.loads(runtime)["controller_status"] == "not_started"


def test_stage_rejects_multinode_before_server_launch(stage_run, monkeypatch):
    run, launches, _, _ = stage_run
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    with pytest.raises(ValueError, match="single.node|one node"):
        run()
    assert launches == []


def test_lost_isolation_prevents_controller_even_with_valid_http_auth(stage_run, monkeypatch, tmp_path):
    run, launches, _, _ = stage_run
    monkeypatch.setattr(stage, "wait_until_ready", lambda *args, **kwargs: receipt())

    def exposed(*args, **kwargs):
        raise security.EndpointSecurityError("network namespace has a non-loopback interface")

    monkeypatch.setattr(stage, "verify_private_network_namespace", exposed)
    with pytest.raises(security.EndpointSecurityError, match="non-loopback"):
        run()
    assert len(launches) == 1
    runtime = json.loads((tmp_path / "server.log.runtime.json").read_text())
    assert runtime["controller_status"] == "not_started"


def test_isolation_failure_prevents_server_and_runtime_probe(stage_run, monkeypatch, tmp_path):
    run, launches, _, _ = stage_run

    def unavailable(*args):
        raise security.EndpointSecurityError("network isolation unavailable")

    monkeypatch.setattr(stage, "ensure_private_network_namespace", unavailable)
    monkeypatch.setattr(stage, "verify_runtime", lambda *args: pytest.fail("runtime probed before isolation"))
    monkeypatch.setattr(stage, "discover_visible_gpus", lambda: pytest.fail("GPU discovery before isolation"))
    monkeypatch.setattr(stage, "ensure_port_available", lambda *args: pytest.fail("port bound before isolation"))
    with pytest.raises(security.EndpointSecurityError, match="isolation unavailable"):
        run()
    assert launches == []
    assert not (tmp_path / "server.log").exists()
    assert not (tmp_path / "server.log.runtime.json").exists()


@pytest.mark.parametrize("field,bad", [
    ("external_tcp_isolated", False), ("outside_parent_pid", 0),
    ("current_namespace", {"device": 4, "inode": 50}),
    ("interfaces", ["lo", "eth0"]), ("status", "unknown"),
])
def test_runtime_reader_rejects_invalid_isolation_receipts(stage_run, monkeypatch, tmp_path, field, bad):
    run, _, profile_path, _ = stage_run
    monkeypatch.setattr(stage, "wait_until_ready", lambda *args, **kwargs: receipt())
    assert run() == 0
    runtime_path = tmp_path / "server.log.runtime.json"
    value = json.loads(runtime_path.read_text())
    value["network_isolation"][field] = bad
    runtime_path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        stage.load_runtime_binding(
            runtime_path, expected_profile_sha256=stage.load_profile(profile_path)["profile_sha256"],
            require_approval=False,
        )


def test_private_log_refuses_symlink_and_restricts_existing_file(tmp_path):
    target = tmp_path / "target"
    target.write_text("preserve")
    target.chmod(0o644)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError), stage._private_server_log(link):
        pytest.fail("symlink accepted")
    assert target.read_text() == "preserve"
    with stage._private_server_log(target) as stream:
        stream.write(b" addition")
    assert target.read_text() == "preserve addition"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_live_readiness_rejects_unprotected_endpoint_before_controller():
    with endpoint(anonymous_status=200) as (base_url, key, _), \
            pytest.raises(security.EndpointSecurityError):
        stage.wait_until_ready(SimpleNamespace(poll=lambda: None), base_url=base_url,
                               model_id="fixture/model", api_key=key, timeout_seconds=1)
