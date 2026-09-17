"""Start inference behind a verified network boundary.

No model is imported here. The default boundary is a Linux network namespace
containing only loopback. A generator job may instead opt into a whole-node
Slurm boundary, which must be verified from the controller before the first
listening socket. The same CPU-only namespace path is available through --check.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.inference_endpoint_security import EndpointSecurityError

MARKER = "GEODML_PRIVATE_NETWORK_NAMESPACE"
TRANSPORT_ENVIRONMENT = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "VLLM_HOST_IP": "127.0.0.1",
}
_GET_FLAGS = 0x8913
_SET_FLAGS = 0x8914
_UP = 0x1
_LOOPBACK = 0x8
_GET_NAMESPACE_TYPE = 0xB703
_CLONE_NEWNET = 0x40000000
_EXCLUSIVE_BOUNDARY_ENVIRONMENT = "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"


def _integer(value, *, minimum):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _namespace_record(value):
    return (
        isinstance(value, dict) and set(value) == {"device", "inode"}
        and _integer(value["device"], minimum=0)
        and _integer(value["inode"], minimum=1)
    )


def validate_network_namespace_receipt(value: Mapping) -> dict:
    """Validate saved evidence without making claims about the current kernel."""
    exclusive_keys = {
        "format_version", "status", "mode", "slurm_job_id", "slurm_array_job_id",
        "slurm_array_task_id", "node_list",
        "exclusive_disposition", "other_jobs_excluded", "loopback_transport_required",
        "native_authentication_required",
    }
    if isinstance(value, dict) and set(value) == exclusive_keys:
        if (
            value["format_version"] != "geodml-exclusive-slurm-boundary-v1"
            or value["status"] != "verified"
            or value["mode"] != "exclusive-slurm-node-authenticated-loopback"
            or not isinstance(value["slurm_job_id"], str)
            or not value["slurm_job_id"]
            or not (
                value["slurm_array_job_id"] is None
                and value["slurm_array_task_id"] is None
                or isinstance(value["slurm_array_job_id"], str)
                and value["slurm_array_job_id"]
                and isinstance(value["slurm_array_task_id"], str)
                and value["slurm_array_task_id"]
            )
            or not isinstance(value["node_list"], str)
            or not value["node_list"]
            or value["exclusive_disposition"] != "NODE"
            or any(value[key] is not True for key in (
                "other_jobs_excluded", "loopback_transport_required",
                "native_authentication_required",
            ))
        ):
            raise EndpointSecurityError("invalid exclusive Slurm boundary receipt")
        return dict(value)
    keys = {
        "format_version", "status", "outside_parent_pid", "outside_namespace",
        "current_namespace", "interfaces", "loopback_up", "private_bind_verified",
        "external_tcp_isolated",
    }
    if (
        not isinstance(value, dict) or set(value) != keys
        or value["format_version"] != "geodml-network-namespace-v1"
        or value["status"] != "verified"
        or not _integer(value["outside_parent_pid"], minimum=1)
        or not _namespace_record(value["outside_namespace"])
        or not _namespace_record(value["current_namespace"])
        or value["outside_namespace"] == value["current_namespace"]
        or value["interfaces"] != ["lo"]
        or any(value[key] is not True for key in (
            "loopback_up", "private_bind_verified", "external_tcp_isolated",
        ))
    ):
        raise EndpointSecurityError("invalid private network namespace receipt")
    return dict(value)


def _slurm_job_identity() -> tuple[str, str | None, str | None, str]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if not job_id.isdigit():
        raise EndpointSecurityError("exclusive Slurm node requires a numeric job ID")
    if bool(array_job) != bool(array_task):
        raise EndpointSecurityError("exclusive Slurm array identity is incomplete")
    if array_job:
        if not array_job.isdigit() or not array_task.isdigit():
            raise EndpointSecurityError("exclusive Slurm array identity is invalid")
        return job_id, array_job, array_task, f"{array_job}_{array_task}"
    return job_id, None, None, job_id


def _verify_exclusive_slurm_boundary() -> dict:
    if sys.platform != "linux":
        raise EndpointSecurityError("exclusive Slurm node verification requires Linux")
    if os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise EndpointSecurityError("exclusive Slurm boundary requires one allocated node")
    node_list = os.environ.get("SLURM_JOB_NODELIST", "")
    if not node_list:
        raise EndpointSecurityError("exclusive Slurm boundary requires an allocated node")
    job_id, array_job_id, array_task_id, reference = _slurm_job_identity()
    executable = shutil.which("scontrol")
    if executable is None:
        raise EndpointSecurityError("scontrol is required to verify the exclusive Slurm node")
    try:
        completed = subprocess.run(
            [executable, "show", "job", "--oneliner", reference],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise EndpointSecurityError("exclusive Slurm node could not be verified") from None
    fields = dict(
        item.split("=", 1) for item in completed.stdout.split()
        if "=" in item
    )
    if (
        completed.returncode != 0
        or fields.get("JobId") != job_id
        or fields.get("ArrayJobId") != array_job_id
        or fields.get("ArrayTaskId") != array_task_id
        or fields.get("JobState") != "RUNNING"
        or fields.get("Exclusive") != "NODE"
        or fields.get("NodeList") != node_list
    ):
        raise EndpointSecurityError("exclusive Slurm node could not be verified")
    os.environ.update(TRANSPORT_ENVIRONMENT)
    return validate_network_namespace_receipt({
        "format_version": "geodml-exclusive-slurm-boundary-v1",
        "status": "verified",
        "mode": "exclusive-slurm-node-authenticated-loopback",
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": array_task_id,
        "node_list": node_list,
        "exclusive_disposition": "NODE",
        "other_jobs_excluded": True,
        "loopback_transport_required": True,
        "native_authentication_required": True,
    })


def _namespace(process: int | str) -> dict[str, int]:
    info = Path(f"/proc/{process}/ns/net").stat()
    return {"device": info.st_dev, "inode": info.st_ino}


def _record(value) -> dict:
    if (
        not isinstance(value, dict) or set(value) != {"parent_pid", "parent_namespace", "outside_fd"}
        or not _integer(value["parent_pid"], minimum=1)
        or not _integer(value["outside_fd"], minimum=3)
        or not _namespace_record(value["parent_namespace"])
    ):
        raise EndpointSecurityError("invalid private network namespace launch record")
    return value


def _outside_namespace(record: dict) -> dict[str, int]:
    # Dereferencing the parent's proc namespace after creating a user namespace
    # can fail ptrace access checks. Retain a kernel namespace fd opened before
    # unshare instead. Child Popen calls must retain their default close_fds=True.
    descriptor = record["outside_fd"]
    if fcntl.ioctl(descriptor, _GET_NAMESPACE_TYPE) != _CLONE_NEWNET:
        raise EndpointSecurityError("outside descriptor is not a network namespace")
    info = os.fstat(descriptor)
    return {"device": info.st_dev, "inode": info.st_ino}


def _kernel_state(record: dict) -> tuple[dict, dict]:
    if sys.platform != "linux":
        raise EndpointSecurityError("private inference networking requires Linux")
    if os.getppid() != record["parent_pid"]:
        raise EndpointSecurityError("private network namespace parent changed")
    outside = _outside_namespace(record)
    current = _namespace("self")
    if outside != record["parent_namespace"] or current == outside:
        raise EndpointSecurityError("private network namespace separation was not verified")
    if sorted(name for _, name in socket.if_nameindex()) != ["lo"]:
        raise EndpointSecurityError("private network namespace has an unexpected interface")
    return outside, current


def _loopback_flags(*, enable=False) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        request = struct.pack("16sH22x", b"lo", 0)
        flags = struct.unpack_from("H", fcntl.ioctl(probe.fileno(), _GET_FLAGS, request), 16)[0]
        if not flags & _LOOPBACK:
            raise EndpointSecurityError("private lo interface is not loopback")
        if enable and not flags & _UP:
            fcntl.ioctl(probe.fileno(), _SET_FLAGS, struct.pack("16sH22x", b"lo", flags | _UP))
            flags = struct.unpack_from("H", fcntl.ioctl(probe.fileno(), _GET_FLAGS, request), 16)[0]
    return flags


def _private_bind_probe() -> None:
    # This wildcard bind happens only after kernel namespace separation and the
    # absence of external interfaces have been established.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.settimeout(2)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        with socket.create_connection(("127.0.0.1", listener.getsockname()[1]), timeout=2):
            connection, _ = listener.accept()
            connection.close()


def verify_private_network_namespace() -> dict:
    """Re-read kernel facts; an environment marker alone cannot pass this gate."""
    if os.environ.get(_EXCLUSIVE_BOUNDARY_ENVIRONMENT) == "1" and MARKER not in os.environ:
        return _verify_exclusive_slurm_boundary()
    try:
        record = _record(json.loads(os.environ.get(MARKER, "")))
        outside, current = _kernel_state(record)
        if any(os.environ.get(key) != value for key, value in TRANSPORT_ENVIRONMENT.items()):
            raise EndpointSecurityError("private inference transport settings changed")
        if not _loopback_flags() & _UP:
            raise EndpointSecurityError("private loopback interface is down")
        _private_bind_probe()
        if _kernel_state(record) != (outside, current):
            raise EndpointSecurityError("private network namespace changed during verification")
    except EndpointSecurityError:
        raise
    except (OSError, ValueError, TypeError, KeyError, struct.error):
        raise EndpointSecurityError("private network namespace could not be verified") from None
    return validate_network_namespace_receipt({
        "format_version": "geodml-network-namespace-v1", "status": "verified",
        "outside_parent_pid": record["parent_pid"], "outside_namespace": outside,
        "current_namespace": current, "interfaces": ["lo"], "loopback_up": True,
        "private_bind_verified": True, "external_tcp_isolated": True,
    })


def _enter_private_namespace(command: Sequence[str] | None) -> None:
    if sys.platform != "linux":
        raise EndpointSecurityError("private inference networking requires Linux")
    executable = shutil.which("unshare")
    if executable is None:
        raise EndpointSecurityError("unshare is required for private inference networking")
    descriptor = None
    try:
        parent_pid = os.getppid()
        outside = _namespace(parent_pid)
        if _namespace("self") != outside:
            raise EndpointSecurityError("initial network namespace differs from its parent")
        descriptor = os.open("/proc/self/ns/net", os.O_RDONLY)
        info = os.fstat(descriptor)
        if {"device": info.st_dev, "inode": info.st_ino} != outside:
            raise EndpointSecurityError("outside network namespace changed before isolation")
        os.set_inheritable(descriptor, True)
        arguments = [
            executable, "--user", "--map-root-user", "--net", "--",
            sys.executable, str(Path(__file__).resolve()), "--inside",
            "--parent-pid", str(parent_pid), "--parent-device", str(outside["device"]),
            "--parent-inode", str(outside["inode"]), "--outside-fd", str(descriptor),
        ]
        arguments.extend(["--check"] if command is None else ["--exec", *command])
        os.execv(executable, arguments)
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        raise EndpointSecurityError("could not enter private inference network namespace") from None
    except EndpointSecurityError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    if descriptor is not None:
        os.close(descriptor)
    raise EndpointSecurityError("namespace launcher unexpectedly returned")


def ensure_private_network_namespace(command: Sequence[str]) -> dict:
    """Enter a namespace or verify an explicitly requested whole-node boundary."""
    if isinstance(command, (str, bytes)) or not command or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise EndpointSecurityError("private namespace requires a nonempty command")
    if MARKER in os.environ:
        return verify_private_network_namespace()
    if os.environ.get(_EXCLUSIVE_BOUNDARY_ENVIRONMENT) == "1":
        return _verify_exclusive_slurm_boundary()
    _enter_private_namespace(command)
    raise EndpointSecurityError("private namespace entry unexpectedly returned")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-device", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-inode", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--outside-fd", type=int, help=argparse.SUPPRESS)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="CPU-only isolation and private bind check")
    action.add_argument("--exec", dest="command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.inside:
            record = _record({"parent_pid": args.parent_pid, "outside_fd": args.outside_fd, "parent_namespace": {
                "device": args.parent_device, "inode": args.parent_inode,
            }})
            _kernel_state(record)
            _loopback_flags(enable=True)
            os.environ[MARKER] = json.dumps(record, sort_keys=True)
            os.environ.update(TRANSPORT_ENVIRONMENT)
            receipt = verify_private_network_namespace()
        elif args.check:
            if MARKER in os.environ:
                receipt = verify_private_network_namespace()
            else:
                _enter_private_namespace(None)
                raise EndpointSecurityError("namespace check unexpectedly returned")
        else:
            receipt = ensure_private_network_namespace(args.command)
        if args.check:
            print("INFERENCE_NETWORK_NAMESPACE=" + json.dumps(receipt, sort_keys=True))
            return 0
        if not args.command:
            raise EndpointSecurityError("private namespace requires a nonempty command")
        os.execvp(args.command[0], args.command)
    except EndpointSecurityError as error:
        print(f"INFERENCE_NETWORK_NAMESPACE=FAIL {error}", file=sys.stderr)
        return 2
    except OSError:
        print("INFERENCE_NETWORK_NAMESPACE=FAIL operating-system isolation check failed", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
