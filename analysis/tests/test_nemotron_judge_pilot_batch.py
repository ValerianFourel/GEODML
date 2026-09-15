"""Execute the pilot wrapper with CPU-only Slurm, grammar, and server doubles."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
WRAPPER = REPOSITORY / "analysis/scripts/slurm/jupiter/run_nemotron_judge_pilot.sbatch"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"


def _environment(root: Path, *, stage_status: int = 0) -> dict[str, str]:
    repository = root / "repository"
    scripts = repository / "analysis/scripts"
    scripts.mkdir(parents=True)
    stage = scripts / "search_vllm_stage.py"
    stage.write_text(
        "import json, os, pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['TEST_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'prepare':\n"
        "    path = pathlib.Path(args[args.index('--profile') + 1])\n"
        "    path.write_text('{}')\n"
        "    sys.exit(int(os.environ.get('TEST_PREPARE_STATUS', '0')))\n"
        "if os.environ.get('TEST_HOLD'):\n"
        "    while not pathlib.Path(os.environ['TEST_HOLD']).exists():\n"
        "        time.sleep(0.02)\n"
        "sys.exit(int(os.environ['TEST_STAGE_STATUS']))\n"
    )
    commands = root / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "case \"${3:-}\" in\n"
        "  rev-parse) printf '%s\\n' \"${TEST_ACTUAL_COMMIT:-$GEODML_EXECUTION_COMMIT}\" ;;\n"
        "  status) printf '%s' \"${TEST_GIT_STATUS:-}\" ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n"
    )
    git.chmod(0o755)
    flock = commands / "flock"
    flock.write_text(
        f"#!{sys.executable}\n"
        "import fcntl, sys\n"
        "try:\n"
        "    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(1)\n"
    )
    flock.chmod(0o755)
    gpu = commands / "nvidia-smi"
    gpu.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['TEST_GPU_PID']).write_text(str(os.getpid()))\n"
        "print('2026/09/16, 0, NVIDIA GH200, 100, 97871, 1, 100', flush=True)\n"
        "time.sleep(120)\n"
    )
    gpu.chmod(0o755)
    (commands / "python3").symlink_to(sys.executable)
    bash_env = root / "bash-env"
    bash_env.write_text(
        "module() {\n"
        "  [[ $- != *u* ]] || return 92\n"
        "  test -n \"${DEBUGINFOD_URLS+x}\" || return 94\n"
        "  printf '%s\\n' \"$*\" >> \"$TEST_MODULE_CALLS\"\n"
        "}\n"
    )
    activation = root / "venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text('[[ $- != *u* ]] || return 93\n')
    fake_imports = root / "imports"
    fake_imports.mkdir()
    (fake_imports / "xgrammar.py").write_text(
        "import json, os\n"
        "class Grammar:\n"
        "    @staticmethod\n"
        "    def from_json_schema(schema):\n"
        "        if os.environ.get('TEST_BAD_SCHEMA'):\n"
        "            raise RuntimeError('unsupported schema')\n"
        "        with open(os.environ['TEST_SCHEMAS'], 'a') as stream:\n"
        "            stream.write(schema + '\\n')\n"
    )
    pilot = root / "pilot"
    plan = pilot / "plan"
    plan.mkdir(parents=True)
    queue = plan / "bulk_tasks.jsonl"
    tasks = [
        {
            "judge_task_id": f"task-{i}",
            "format_version": "agentic-search-judge-v1",
            "blind_case_id": f"blind-{i}",
            "prompt_text": "Compare these tools.",
            "evidence": [{"evidence_id": "E1", "url": "https://test", "title": "tool", "text": "snippet"}],
            "answer": "Example answer",
        }
        for i in range(24)
    ]
    queue.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    manifest = {
        "format_version": "agentic-search-judge-v1",
        "scientific_result": False,
        "pilot_only": True,
        "bulk_model": {"role": "bulk", "model_id": MODEL, "model_revision": REVISION},
        "artifacts": {
            "bulk_tasks": {
                "path": str(queue),
                "sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
            }
        },
    }
    (plan / "run_manifest.json").write_text(json.dumps(manifest))
    snapshot = root / "cache" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    weights = {f"weight-{i}": f"model-{i}.safetensors" for i in range(13)}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    (snapshot / "config.json").write_text(json.dumps({"architectures": ["NemotronHForCausalLM"]}))
    for name in [*weights.values(), "tokenizer.json", "tokenizer_config.json"]:
        (snapshot / name).write_text("{}")
    return {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(fake_imports) + os.pathsep + str(REPOSITORY),
        "BASH_ENV": str(bash_env),
        "SHELLOPTS": "braceexpand:hashall:interactive-comments:nounset",
        "ACL_ARR_VENV": str(activation.parent.parent),
        "GEODML_EXECUTION_REPOSITORY": str(repository),
        "GEODML_EXECUTION_COMMIT": "a" * 40,
        "GEODML_APPROVED_WALLTIME": "01:00:00",
        "GEODML_ALLOCATION_ESTIMATE": "10-30 minutes including model startup; maximum 4 GPU-hours",
        "GEODML_JUDGE_PILOT_ROOT": str(pilot),
        "GEODML_CACHE_ROOT": str(root / "cache-root"),
        "HF_HUB_CACHE": str(root / "cache"),
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "SLURM_JOB_ID": "987654",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "524288",
        "SLURM_GPUS_ON_NODE": "4",
        "TEST_CAPTURE": str(root / "capture"),
        "TEST_SCHEMAS": str(root / "schemas"),
        "TEST_GPU_PID": str(root / "gpu-pid"),
        "TEST_MODULE_CALLS": str(root / "modules"),
        "TEST_STAGE_STATUS": str(stage_status),
    }


def _run(env):
    return subprocess.run(
        ["bash", str(WRAPPER)], env=env, capture_output=True, text=True,
        check=False, timeout=15,
    )


@pytest.mark.parametrize("stage_status", [0, 2, 7])
def test_pilot_wrapper_checks_inputs_and_propagates_status(tmp_path, stage_status):
    env = _environment(tmp_path, stage_status=stage_status)
    result = _run(env)
    assert result.returncode == stage_status, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "capture").read_text().splitlines()]
    prepare, run = calls
    assert prepare[0] == "prepare"
    assert prepare[prepare.index("--tensor-parallel-size") + 1] == "4"
    assert prepare[prepare.index("--data-parallel-size") + 1] == "1"
    assert prepare[prepare.index("--model-id") + 1] == MODEL
    assert prepare[prepare.index("--model-revision") + 1] == REVISION
    assert prepare[prepare.index("--max-model-len") + 1] == "16384"
    assert "--enforce-eager" in prepare
    assert run[0] == "run"
    assert run[run.index("--startup-timeout-seconds") + 1] == "900"
    assert run[run.index("--") + 1:][:3] == [
        "python3", "analysis/scripts/run_acl_arr_vllm.py", "agentic-judge"
    ]
    assert {"--pilot-only", "--resume", "--disable-thinking"} <= set(run)
    assert run[run.index("--max-concurrency") + 1] == "4"
    assert run[run.index("--max-attempts") + 1] == "3"
    assert run[run.index("--max-output-tokens") + 1] == "2048"
    assert run[run.index("--server-model-revision") + 1] == REVISION
    assert len((tmp_path / "schemas").read_text().splitlines()) == 24
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["status"] == ("complete" if stage_status == 0 else "failed_or_interrupted")
    assert record["exit_code"] == stage_status
    assert record["finished_at_utc"]
    assert record["started_at_utc"]
    assert record["git_commit"] == "a" * 40
    assert record["model_id"] == MODEL
    assert record["model_revision"] == REVISION
    assert record["approved_walltime"] == "01:00:00"
    assert record["maximum_gpu_hours"] == 4
    assert record["scientific_result"] is False
    assert record["resources"]["SLURM_CPUS_PER_TASK"] == "32"
    assert record["task_count"] == 24
    assert (tmp_path / "modules").read_text().splitlines() == [
        "load Stages/2026 GCC Python CUDA", "load git"
    ]
    gpu_pid = tmp_path / "gpu-pid"
    if gpu_pid.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(gpu_pid.read_text()), 0)


@pytest.mark.parametrize("blocked", [
    "dirty", "commit", "approval", "resources", "logs", "model", "queue_hash",
    "task_count", "duplicate", "schema", "snapshot", "pilot_flag",
])
def test_pilot_refuses_invalid_input_before_starting_server(tmp_path, blocked):
    env = _environment(tmp_path)
    plan = tmp_path / "pilot/plan"
    manifest_path = plan / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if blocked == "dirty":
        env["TEST_GIT_STATUS"] = " M file.py"
    elif blocked == "commit":
        env["TEST_ACTUAL_COMMIT"] = "b" * 40
    elif blocked == "approval":
        env["GEODML_APPROVED_WALLTIME"] = "12:00:00"
    elif blocked == "resources":
        env["SLURM_JOB_NUM_NODES"] = "2"
    elif blocked == "logs":
        logs = tmp_path / "pilot/logs"
        logs.mkdir()
        (logs / "server.log").write_text("preserve")
    elif blocked == "model":
        manifest["bulk_model"]["model_revision"] = "b" * 40
    elif blocked == "pilot_flag":
        manifest["pilot_only"] = False
    elif blocked == "queue_hash":
        manifest["artifacts"]["bulk_tasks"]["sha256"] = "b" * 64
    elif blocked in {"task_count", "duplicate"}:
        queue = plan / "bulk_tasks.jsonl"
        lines = queue.read_text().splitlines()
        if blocked == "duplicate":
            lines[-1] = lines[0]
        else:
            lines = lines[:-1]
        queue.write_text("\n".join(lines) + "\n")
        manifest["artifacts"]["bulk_tasks"]["sha256"] = hashlib.sha256(queue.read_bytes()).hexdigest()
    elif blocked == "schema":
        env["TEST_BAD_SCHEMA"] = "1"
    elif blocked == "snapshot":
        cache = tmp_path / "cache" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
        (cache / "model-0.safetensors").unlink()
    manifest_path.write_text(json.dumps(manifest))
    result = _run(env)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "pilot/logs/allocation.json").exists()


def test_prepare_failure_is_recorded_without_starting_server(tmp_path):
    env = _environment(tmp_path)
    env["TEST_PREPARE_STATUS"] = "9"
    result = _run(env)
    assert result.returncode == 9, result.stderr
    assert len((tmp_path / "capture").read_text().splitlines()) == 1
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["exit_code"] == 9
    assert record["status"] == "failed_or_interrupted"


def test_wrapper_lock_rejects_duplicate_job(tmp_path):
    env = _environment(tmp_path)
    release = tmp_path / "release"
    env["TEST_HOLD"] = str(release)
    first = subprocess.Popen(
        ["bash", str(WRAPPER)], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        capture = tmp_path / "capture"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert first.poll() is None, "first worker exited before server startup"
            if capture.exists() and len(capture.read_text().splitlines()) == 2:
                break
            time.sleep(0.02)
        assert capture.exists() and len(capture.read_text().splitlines()) == 2
        second = _run({**env, "SLURM_JOB_ID": "987655"})
        assert second.returncode == 75, second.stderr
        assert "already has an active worker" in second.stderr
        assert len(capture.read_text().splitlines()) == 2
    finally:
        release.touch()
        _, stderr = first.communicate(timeout=10)
    assert first.returncode == 0, stderr


def test_wrapper_has_valid_shell_and_no_allocation_defaults():
    text = WRAPPER.read_text()
    assert "#SBATCH" not in text
    assert "sbatch " not in text
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
