"""Fail-closed namespace entry and kernel verification without model imports."""

from __future__ import annotations

import copy
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import inference_network_namespace as network
from analysis.scripts.inference_endpoint_security import EndpointSecurityError

OUTSIDE = {"device": 4, "inode": 100}
INSIDE = {"device": 4, "inode": 200}
RECORD = {"parent_pid": 10, "parent_namespace": OUTSIDE, "outside_fd": 7}


@pytest.mark.parametrize("failure,code", [
    (None, None), ("nodes", "job_node_list"), ("identity", "job_id"),
    ("query", "job_query_exit"), ("timeout", "job_query_exception"),
    ("sharing", "whole_node_exclusivity"),
])
def test_slurm_diagnostic_reports_specific_failure_without_launching(monkeypatch, capsys, failure, code):
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: "/usr/bin/scontrol")
    for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "SLURM_JOB_ID": "1927510", "SLURM_JOB_NUM_NODES": "1",
        "SLURM_STEP_NUM_NODES": "1", "SLURM_JOB_NODELIST": "node0",
        "HF_TOKEN": "fixture-secret", "SLURM_JWT": "fixture-secret",
    }.items():
        monkeypatch.setenv(key, value)

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10, stderr="fixture-secret")
        identity = "other" if failure == "identity" else "1927510"
        node = "node1" if failure == "nodes" else "node0"
        exclusive = "NO" if failure == "sharing" else "NODE"
        output = (f"JobId={identity} JobState=RUNNING NodeList={node} "
                  f"Exclusive={exclusive} Comment=fixture-secret")
        return subprocess.CompletedProcess(command, int(failure == "query"), output, "fixture-secret")

    monkeypatch.setattr(network.subprocess, "run", run)
    monkeypatch.setattr(network.os, "execvp", lambda *a: pytest.fail("executed a workload"))
    monkeypatch.setattr(network, "_enter_private_namespace", lambda *a: pytest.fail("entered namespace"))
    monkeypatch.setattr(network.socket, "socket", lambda *a: pytest.fail("opened socket"))
    assert network.main(["--diagnose-slurm"]) == (2 if failure else 0)
    captured = capsys.readouterr()
    assert "fixture-secret" not in captured.out + captured.err
    report = json.loads(captured.out)
    assert report["status"] == ("FAIL" if failure else "PASS")
    assert report["format_version"] == "geodml-slurm-boundary-diagnostic-v1"
    if failure:
        assert report["failed_check"] == code
        assert report["checks"][-1]["name"] == code
        assert report["checks"][-1]["passed"] is False
    else:
        assert report["receipt"]["slurm_job_id"] == "1927510"


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.os, "getppid", lambda: 10)
    monkeypatch.setattr(network, "_namespace", lambda pid: INSIDE.copy())
    monkeypatch.setattr(network, "_outside_namespace", lambda record: OUTSIDE.copy())
    monkeypatch.setattr(network.socket, "if_nameindex", lambda: [(1, "lo")])
    monkeypatch.setattr(network, "_loopback_flags", lambda **kwargs: network._UP | network._LOOPBACK)
    probes = []
    monkeypatch.setattr(network, "_private_bind_probe", lambda: probes.append(True))
    monkeypatch.setenv(network.MARKER, json.dumps(RECORD))
    for key, value in network.TRANSPORT_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    return probes


def test_verified_receipt_uses_kernel_facts_and_private_probe(isolated):
    receipt = network.verify_private_network_namespace()
    assert receipt == {
        "format_version": "geodml-network-namespace-v1", "status": "verified",
        "outside_parent_pid": 10, "outside_namespace": OUTSIDE,
        "current_namespace": INSIDE, "interfaces": ["lo"], "loopback_up": True,
        "private_bind_verified": True, "external_tcp_isolated": True,
    }
    assert isolated == [True]


@pytest.mark.parametrize("marker", [None, "1", "{}", "not-json", json.dumps({**RECORD, "outside_fd": True})])
def test_missing_or_forged_marker_never_creates_socket(isolated, monkeypatch, marker):
    if marker is None:
        monkeypatch.delenv(network.MARKER)
    else:
        monkeypatch.setenv(network.MARKER, marker)
    with pytest.raises(EndpointSecurityError):
        network.verify_private_network_namespace()
    assert isolated == []


@pytest.mark.parametrize("failure", ["same_namespace", "changed_parent", "outside_changed", "external_interface", "lo_down", "rdma_enabled"])
def test_invalid_kernel_or_transport_state_fails_before_bind(isolated, monkeypatch, failure):
    if failure == "same_namespace":
        monkeypatch.setattr(network, "_namespace", lambda pid: OUTSIDE.copy())
    elif failure == "changed_parent":
        monkeypatch.setattr(network.os, "getppid", lambda: 11)
    elif failure == "outside_changed":
        monkeypatch.setattr(network, "_outside_namespace", lambda record: {"device": 4, "inode": 101})
    elif failure == "external_interface":
        monkeypatch.setattr(network.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    elif failure == "lo_down":
        monkeypatch.setattr(network, "_loopback_flags", lambda: network._LOOPBACK)
    else:
        monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    with pytest.raises(EndpointSecurityError):
        network.verify_private_network_namespace()
    assert isolated == []


def test_unknown_kernel_state_redacts_exception_contents(isolated, monkeypatch):
    def inaccessible(*args):
        raise PermissionError("fixture-secret")

    monkeypatch.setattr(network, "_namespace", inaccessible)
    with pytest.raises(EndpointSecurityError) as caught:
        network.verify_private_network_namespace()
    assert "fixture-secret" not in str(caught.value)
    assert isolated == []


def test_descriptor_must_be_a_real_network_namespace(monkeypatch):
    monkeypatch.setattr(network.fcntl, "ioctl", lambda fd, command: network._CLONE_NEWNET)
    monkeypatch.setattr(network.os, "fstat", lambda fd: SimpleNamespace(st_dev=4, st_ino=100))
    assert network._outside_namespace(RECORD) == OUTSIDE
    monkeypatch.setattr(network.fcntl, "ioctl", lambda fd, command: 0x10000000)
    with pytest.raises(EndpointSecurityError, match="not a network namespace"):
        network._outside_namespace(RECORD)


def test_closed_outside_descriptor_fails_closed(isolated, monkeypatch):
    def closed(record):
        raise OSError("bad descriptor")

    monkeypatch.setattr(network, "_outside_namespace", closed)
    with pytest.raises(EndpointSecurityError, match="could not be verified"):
        network.verify_private_network_namespace()
    assert isolated == []


def test_entry_execs_unshare_without_fork_and_retains_only_namespace_fd(monkeypatch):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: "/usr/bin/unshare")
    monkeypatch.setattr(network.os, "getppid", lambda: 10)
    monkeypatch.setattr(network, "_namespace", lambda pid: OUTSIDE.copy())
    monkeypatch.setattr(network.os, "open", lambda path, flags: 7)
    monkeypatch.setattr(network.os, "fstat", lambda fd: SimpleNamespace(st_dev=4, st_ino=100))
    inherited = []
    monkeypatch.setattr(network.os, "set_inheritable", lambda fd, state: inherited.append((fd, state)))
    executed = []

    class Executed(BaseException):
        pass

    def execute(path, arguments):
        executed.append((path, arguments))
        raise Executed

    monkeypatch.setattr(network.os, "execv", execute)
    command = [sys.executable, "/repo/stage.py", "run", "--profile", "/data/profile.json"]
    with pytest.raises(Executed):
        network.ensure_private_network_namespace(command)
    assert inherited == [(7, True)]
    path, arguments = executed[0]
    assert path == "/usr/bin/unshare"
    assert arguments[:5] == [path, "--user", "--map-root-user", "--net", "--"]
    assert "--fork" not in arguments
    assert arguments[arguments.index("--outside-fd") + 1] == "7"
    assert arguments[arguments.index("--exec") + 1:] == command


def test_verified_exclusive_slurm_node_avoids_unavailable_user_namespace(monkeypatch):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1845304")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1845271")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "jpbo-108-47")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0,
            "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=RUNNING "
            "Exclusive=NODE NodeList=jpbo-108-47\n",
            "",
        )

    monkeypatch.setattr(network.subprocess, "run", run)
    monkeypatch.setattr(
        network, "_enter_private_namespace", lambda command: pytest.fail("called unshare")
    )
    receipt = network.ensure_private_network_namespace(["python", "stage.py"])
    assert receipt == {
        "format_version": "geodml-exclusive-slurm-boundary-v1",
        "status": "verified",
        "mode": "exclusive-slurm-node-authenticated-loopback",
        "slurm_job_id": "1845304",
        "slurm_array_job_id": "1845271",
        "slurm_array_task_id": "0",
        "node_list": "jpbo-108-47",
        "exclusive_disposition": "NODE",
        "other_jobs_excluded": True,
        "loopback_transport_required": True,
        "native_authentication_required": True,
    }
    assert calls == [(
        ["/usr/bin/scontrol", "show", "job", "--oneliner", "1845271_0"],
        {"capture_output": True, "text": True, "timeout": 10, "check": False},
    )]
    assert network.verify_private_network_namespace() == receipt


@pytest.mark.parametrize("failure", [None, "multinode_step", "wrong_host", "partial_cpus", "wrong_step", "shared"])
def test_exclusive_multinode_allocation_has_verified_single_node_step(monkeypatch, failure):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(network.socket, "gethostname", lambda: "node0")
    for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY": "1", "SLURM_JOB_ID": "1927510",
        "SLURM_JOB_NUM_NODES": "5", "SLURM_JOB_NODELIST": "node[0-4]",
        "SLURM_STEP_ID": "10", "SLURM_STEP_NUM_NODES": "1",
    }.items():
        monkeypatch.setenv(key, value)

    def run(command, **kwargs):
        if command[2] == "job":
            cpus = 160 if failure == "partial_cpus" else 1440
            sharing = "YES" if failure == "shared" else "NO"
            output = (f"JobId=1927510 JobState=RUNNING NodeList=node[0-4] NumNodes=5 "
                      f"NumCPUs={cpus} OverSubscribe={sharing} AllocTRES=cpu={cpus},node=5")
        elif command[2] == "node":
            output = "\n".join(f"NodeName=node{i} CPUTot=288" for i in range(5))
        elif command[2] == "step":
            nodes = "node[0-1]" if failure == "multinode_step" else "node0"
            if failure == "wrong_host":
                nodes = "node1"
            identity = "1927510.11" if failure == "wrong_step" else "1927510.10"
            output = f"StepId={identity} State=RUNNING Nodes={nodes}"
        else:
            pytest.fail(str(command))
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(network.subprocess, "run", run)
    monkeypatch.setattr(network, "_enter_private_namespace", lambda command: pytest.fail("called unshare"))
    if failure:
        with pytest.raises(EndpointSecurityError):
            network.ensure_private_network_namespace(["python", "stage.py"])
        report = network.diagnose_slurm_boundary()
        assert report["status"] == "FAIL"
        assert report["failed_check"] == {
            "multinode_step": "step_node", "wrong_host": "step_node",
            "partial_cpus": "whole_node_cpu_totals", "wrong_step": "step_id",
            "shared": "exclusivity_OverSubscribe",
        }[failure]
    else:
        receipt = network.ensure_private_network_namespace(["python", "stage.py"])
        assert receipt["node_list"] == "node0"
        assert receipt["slurm_job_id"] == "1927510"
        assert receipt["other_jobs_excluded"] is True
        report = network.diagnose_slurm_boundary()
        assert report["status"] == "PASS"
        assert report["checks"][-1]["name"] == "step_node"


def test_legacy_slurm_shared_zero_verifies_whole_node_exclusivity(monkeypatch):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1856708")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1856708")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "jpbo-003-33")
    output = (
        "JobId=1856708 ArrayJobId=1856708 ArrayTaskId=0 JobState=RUNNING "
        "Shared=0 NodeList=jpbo-003-33\n"
    )
    monkeypatch.setattr(
        network.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, output, ""),
    )
    receipt = network.ensure_private_network_namespace(["python", "stage.py"])
    assert receipt["exclusive_disposition"] == "NODE"
    assert receipt["slurm_job_id"] == "1856708"


def test_jupiter_slurm_full_node_record_verifies_whole_node_exclusivity(monkeypatch):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1856970")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1856970")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "jpbo-082-43")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "32")
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "288")
    output = (
        "JobId=1856970 ArrayJobId=1856970 ArrayTaskId=0 JobState=RUNNING "
        "NodeList=jpbo-082-43 NumNodes=1 NumCPUs=288 OverSubscribe=NO "
        "AllocTRES=cpu=288,node=1,billing=288,gres/gpu=4,gres/gpu:gh200=4\n"
    )
    monkeypatch.setattr(
        network.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, output, ""),
    )
    receipt = network.ensure_private_network_namespace(["python", "stage.py"])
    assert receipt["exclusive_disposition"] == "NODE"
    assert receipt["slurm_job_id"] == "1856970"


@pytest.mark.parametrize("scontrol_output", [
    "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=RUNNING Exclusive=NO NodeList=jpbo-108-47\n",
    "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=RUNNING Shared=1 NodeList=jpbo-108-47\n",
    "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=RUNNING OverSubscribe=NO NumCPUs=32 AllocTRES=cpu=32,node=1 NodeList=jpbo-108-47\n",
    "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=RUNNING OverSubscribe=YES NumCPUs=288 AllocTRES=cpu=288,node=1 NodeList=jpbo-108-47\n",
    "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=RUNNING OverSubscribe=NO NumCPUs=288 AllocTRES=cpu=32,node=1 NodeList=jpbo-108-47\n",
    "JobId=1845304 ArrayJobId=1845271 ArrayTaskId=0 JobState=PENDING Exclusive=NODE NodeList=(null)\n",
    "JobId=other JobState=RUNNING Exclusive=NODE NodeList=jpbo-108-47\n",
])
def test_unverified_slurm_exclusivity_fails_closed(monkeypatch, scontrol_output):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setenv("GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "1845304")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1845271")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "jpbo-108-47")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "288")
    monkeypatch.setattr(network.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        network.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, scontrol_output, ""),
    )
    monkeypatch.setattr(
        network, "_enter_private_namespace", lambda command: pytest.fail("called unshare")
    )
    with pytest.raises(EndpointSecurityError, match="exclusive Slurm node"):
        network.ensure_private_network_namespace(["python", "stage.py"])


def test_reentry_verifies_instead_of_recursively_unsharing(isolated, monkeypatch):
    monkeypatch.setattr(network, "_enter_private_namespace", lambda command: pytest.fail("recursive unshare"))
    assert network.ensure_private_network_namespace(["python", "stage.py"])["status"] == "verified"


def test_unavailable_unshare_has_no_fallback(monkeypatch):
    monkeypatch.delenv(network.MARKER, raising=False)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: None)
    with pytest.raises(EndpointSecurityError, match="unshare is required"):
        network.ensure_private_network_namespace(["python", "stage.py"])


def test_loopback_ioctl_enables_existing_loopback_without_ip_command(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(network.socket, "socket", lambda *args: nullcontext(SimpleNamespace(fileno=lambda: 8)))
    calls = []
    flags = network._LOOPBACK

    def ioctl(descriptor, command, request):
        nonlocal flags
        calls.append(command)
        assert descriptor == 8
        assert request[:16].rstrip(b"\0") == b"lo"
        if command == network._SET_FLAGS:
            flags = struct.unpack_from("H", request, 16)[0]
        return struct.pack("16sH22x", b"lo", flags)

    monkeypatch.setattr(network.fcntl, "ioctl", ioctl)
    assert network._loopback_flags(enable=True) == network._LOOPBACK | network._UP
    assert calls == [network._GET_FLAGS, network._SET_FLAGS, network._GET_FLAGS]


@pytest.mark.parametrize("field,value", [
    ("outside_parent_pid", True), ("outside_parent_pid", 0),
    ("outside_namespace", {"device": True, "inode": 100}),
    ("outside_namespace", {"device": 4, "inode": 0}),
    ("current_namespace", OUTSIDE), ("interfaces", ["lo", "eth0"]),
    ("loopback_up", 1), ("private_bind_verified", False),
    ("external_tcp_isolated", "true"), ("status", "pending"),
])
def test_persisted_receipt_validation_is_strict_and_kernel_independent(isolated, monkeypatch, field, value):
    receipt = copy.deepcopy(network.verify_private_network_namespace())
    monkeypatch.setattr(network, "_kernel_state", lambda record: pytest.fail("historical receipt read kernel"))
    assert network.validate_network_namespace_receipt(receipt) == receipt
    receipt[field] = value
    with pytest.raises(EndpointSecurityError, match="receipt"):
        network.validate_network_namespace_receipt(receipt)


def test_check_mode_uses_same_verification_and_prints_receipt(isolated, capsys):
    assert network.main(["--check"]) == 0
    assert "INFERENCE_NETWORK_NAMESPACE=" in capsys.readouterr().out
    assert isolated == [True]


def test_inside_mode_verifies_separation_before_any_interface_mutation(isolated, monkeypatch, capsys):
    monkeypatch.setattr(network, "_namespace", lambda pid: OUTSIDE.copy())
    monkeypatch.setattr(network, "_loopback_flags", lambda **kwargs: pytest.fail("changed host loopback"))
    assert network.main([
        "--inside", "--parent-pid", "10", "--parent-device", "4", "--parent-inode", "100",
        "--outside-fd", "7", "--check",
    ]) == 2
    assert "separation was not verified" in capsys.readouterr().err
    assert isolated == []


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux network namespaces")
def test_real_linux_cpu_only_private_network_check():
    if shutil.which("unshare") is None:
        pytest.skip("unshare is not installed")
    environment = {key: value for key, value in os.environ.items() if key != network.MARKER}
    completed = subprocess.run(
        [sys.executable, str(Path(network.__file__).resolve()), "--check"],
        env=environment, capture_output=True, text=True, timeout=15, check=False,
    )
    if completed.returncode and "Operation not permitted" in completed.stderr:
        pytest.skip("unprivileged user/network namespaces are disabled by this Linux host")
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout.split("INFERENCE_NETWORK_NAMESPACE=", 1)[1])
    assert network.validate_network_namespace_receipt(receipt)["external_tcp_isolated"] is True
