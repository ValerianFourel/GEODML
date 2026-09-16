"""Security gates for the four maintained direct vLLM shell launchers."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "analysis/scripts/slurm/jupiter"
LAUNCHERS = (
    "run_acl_arr_document_pilot_4gpu.sh",
    "recover_acl_arr_llama_rerank_4gpu.sh",
    "run_acl_arr_llama_answer_4gpu.sh",
    "run_acl_arr_correction_smoke_4gpu.sh",
)


@pytest.mark.parametrize("name", LAUNCHERS)
def test_direct_launchers_require_shared_authentication_gate(name):
    source = (SCRIPTS / name).read_text()
    assert 'source "$REPOSITORY_ROOT/analysis/scripts/slurm/jupiter/inference_endpoint_security.sh"' in source
    assert "geodml_init_inference_endpoint" in source
    assert "geodml_private_server_log" in source
    assert "geodml_probe_inference_endpoint" in source
    assert "geodml_safe_server_tail" in source
    assert "curl " not in source
    assert "tail -n" not in source
    assert "--api-key" not in source
    assert "--host 127.0.0.1" in source
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


@pytest.fixture
def shell_environment(tmp_path):
    """Replace only external commands; execute the actual shell security gates."""
    executable = tmp_path / "bin/python3"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, secrets, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[0] == '-':\n"
        f"    os.execv({str(sys.executable)!r}, [{str(sys.executable)!r}, *args])\n"
        "if args[0].endswith('inference_endpoint_security.py') and args[1] == 'new-key':\n"
        "    print('test-only-' + secrets.token_hex(24))\n"
        "    raise SystemExit(0)\n"
        "event = 'probe' if args[0].endswith('inference_endpoint_security.py') else 'controller'\n"
        "with Path(os.environ['TEST_EVENTS']).open('a') as stream:\n"
        "    stream.write(json.dumps({'event': event, 'argv': args, 'key': os.environ.get('VLLM_API_KEY'),\n"
        "        'required': os.environ.get('GEODML_INFERENCE_AUTH_REQUIRED'),\n"
        "        'host': os.environ.get('VLLM_HOST_IP')}) + '\\n')\n"
        "if event == 'probe':\n"
        "    raise SystemExit(int(os.environ['TEST_PROBE_STATUS']))\n"
    )
    executable.chmod(0o700)
    server = executable.parent / "vllm"
    server.write_text("#!/usr/bin/env bash\n[[ -n ${VLLM_API_KEY:-} ]]\n")
    server.chmod(0o700)
    log = tmp_path / "server.log"
    log.touch(mode=0o644)
    events = tmp_path / "events.jsonl"
    environment = {
        **os.environ,
        "PATH": str(executable.parent) + os.pathsep + os.environ["PATH"],
        "REPOSITORY_ROOT": str(ROOT),
        "ACL_ARR_VENV": str(tmp_path),
        "SERVER_LOG": str(log),
        "LOG_ROOT": str(tmp_path),
        "TEST_EVENTS": str(events),
        "TEST_PROBE_STATUS": "0",
        "SLURM_JOB_NUM_NODES": "1",
    }
    return environment, events, log


def _run_shell(body, environment):
    return subprocess.run(
        ["bash", "-x", "-c", f'''set -euo pipefail
source {shlex.quote(str(SCRIPTS / "inference_endpoint_security.sh"))}
{body}
'''],
        env=environment,
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("name", LAUNCHERS)
@pytest.mark.parametrize("probe_status", [0, 2])
def test_runtime_auth_gate_precedes_controller(name, probe_status, shell_environment):
    environment, events, _ = shell_environment
    environment["TEST_PROBE_STATUS"] = str(probe_status)
    source = (SCRIPTS / name).read_text()
    if name == "run_acl_arr_document_pilot_4gpu.sh":
        body = source.split("start_server() {", 1)[1].split("\nmapfile -t", 1)[0]
        body = "start_server() {" + body
        body += '\nstart_server "$MODEL_ID" "$MODEL_REVISION" test\npython3 controller\n'
    else:
        body = 'echo "START_' + source.split('echo "START_', 1)[1]
    result = _run_shell(
        '''MODEL_ID=test-model
MODEL_REVISION=test-revision
SERVER_PORT=8001
SERVER_URL=http://127.0.0.1:8001/v1
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.8
stop_submit_epoch=$(( $(date +%s) + 60 ))
server_args=(serve test-model --host 127.0.0.1)
answer_args=(--base-url "$SERVER_URL")
recovery_args=(--base-url "$SERVER_URL")
stop_server() { :; }
kill() { return 0; }
sleep() { printf 'UNEXPECTED_WAIT\n'; return 2; }
setsid() { "$@"; }
''' + body,
        environment,
    )
    assert result.returncode == probe_status, result.stderr
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["event"] for row in rows] == (
        ["probe", "controller"] if probe_status == 0 else ["probe"]
    )
    assert "UNEXPECTED_WAIT" not in result.stdout
    for row in rows:
        assert row["key"].startswith("test-only-")
        assert row["required"] == "1"
        assert row["host"] == "127.0.0.1"
        assert row["key"] not in json.dumps(row["argv"])
        assert row["key"] not in result.stdout + result.stderr
    assert len({row["key"] for row in rows}) == 1


def test_keys_rotate_logs_are_private_and_failure_tail_is_redacted(shell_environment):
    environment, events, log = shell_environment
    result = _run_shell(
        '''geodml_init_inference_endpoint
geodml_private_server_log "$SERVER_LOG"
geodml_probe_inference_endpoint http://127.0.0.1:8001/v1 test-model
printf 'server accidentally printed %s\n' "$VLLM_API_KEY" > "$SERVER_LOG"
geodml_safe_server_tail "$SERVER_LOG"
geodml_init_inference_endpoint
geodml_probe_inference_endpoint http://127.0.0.1:8001/v1 test-model
''',
        environment,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert rows[0]["key"] != rows[1]["key"]
    assert "server accidentally printed [REDACTED]" in result.stdout
    for row in rows:
        assert row["key"] not in result.stdout + result.stderr
    assert log.stat().st_mode & 0o777 == 0o600


def test_multi_node_direct_endpoint_fails_before_generating_key(shell_environment):
    environment, events, _ = shell_environment
    environment["SLURM_JOB_NUM_NODES"] = "2"
    result = _run_shell("geodml_init_inference_endpoint\npython3 controller", environment)
    assert result.returncode == 2
    assert not events.exists()
    assert "require one local node" in result.stderr


def test_private_server_log_refuses_symlink_without_touching_target(shell_environment):
    environment, _, log = shell_environment
    target = log.with_name("original.log")
    target.write_text("unchanged")
    log.unlink()
    log.symlink_to(target)
    original_mode = target.stat().st_mode
    result = _run_shell('geodml_private_server_log "$SERVER_LOG"', environment)
    assert result.returncode == 2
    assert target.read_text() == "unchanged"
    assert target.stat().st_mode == original_mode


def test_failure_tail_refuses_unredacted_output_without_key(shell_environment):
    environment, _, log = shell_environment
    log.write_text("DO_NOT_PRINT_THIS_LOG")
    result = _run_shell(
        'unset VLLM_API_KEY\ngeodml_safe_server_tail "$SERVER_LOG"', environment,
    )
    assert result.returncode != 0
    assert "DO_NOT_PRINT_THIS_LOG" not in result.stdout + result.stderr
