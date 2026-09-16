"""Exercise the real batch wrapper without Slurm, model weights, or GPUs."""

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
WRAPPER = REPOSITORY / "analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch"


def _environment(root: Path, *, launcher_status: int = 0) -> dict[str, str]:
    repository = root / "repository"
    repository.mkdir()
    launcher = repository / "worker.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' launched >> \"$TEST_CAPTURE\"\n"
        "if [[ -n ${TEST_HOLD:-} ]]; then\n"
        "  while [[ ! -f $TEST_HOLD ]]; do sleep 0.02; done\n"
        "fi\n"
        "exit \"$TEST_LAUNCHER_STATUS\"\n"
    )
    launcher.chmod(0o755)
    commands = root / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "case \"${3:-}\" in\n"
        "rev-parse) printf '%s\\n' \"${TEST_ACTUAL_COMMIT:-$GEODML_EXECUTION_COMMIT}\" ;;\n"
        "status) printf '%s' \"${TEST_GIT_STATUS:-}\" ;;\n"
        "*) exit 91 ;;\n"
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
    wave = root / "wave"
    workers = wave / "workers"
    workers.mkdir(parents=True)
    queue = workers / "worker-00000.jsonl"
    queue.write_text('{"cell_id":"first"}\n')
    (wave / "run_manifest.json").write_text(json.dumps({
        "format_version": "geodml-inference-wave-v1",
        "status": "planned", "worker_count": 1,
        "workers": [{
            "worker_index": 0, "path": str(queue), "task_count": 1,
            "sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
        }],
    }))
    return {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "BASH_ENV": str(bash_env),
        "SHELLOPTS": "braceexpand:hashall:interactive-comments:nounset",
        "ACL_ARR_VENV": str(activation.parent.parent),
        "GEODML_EXECUTION_REPOSITORY": str(repository),
        "GEODML_EXECUTION_COMMIT": "a" * 40,
        "GEODML_APPROVED_WALLTIME": "01:00:00",
        "GEODML_ALLOCATION_ESTIMATE": "45 minutes plus startup; cap 4 GPU-hours",
        "GEODML_WAVE_ROOT": str(wave),
        "GEODML_WAVE_OUTPUT_ROOT": str(root / "output"),
        "GEODML_WORKER_LAUNCHER": str(launcher),
        "GEODML_MODEL_SLUG": "qwen38",
        "GEODML_WORKER_STDOUT": str(root / "slurm-%A_%a-%j.out"),
        "GEODML_WORKER_STDERR": str(root / "slurm-%A_%a-%j.err"),
        "SLURM_JOB_ID": "987654",
        "SLURM_JOB_END_TIME": str(int(time.time()) + 3600),
        "SLURM_ARRAY_JOB_ID": "987650",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "524288",
        "SLURM_GPUS_ON_NODE": "4",
        "TEST_CAPTURE": str(root / "capture"),
        "TEST_MODULE_CALLS": str(root / "modules"),
        "TEST_LAUNCHER_STATUS": str(launcher_status),
    }


def _run(env):
    return subprocess.run(
        ["bash", str(WRAPPER)], env=env, capture_output=True, text=True,
        check=False, timeout=10,
    )


@pytest.mark.parametrize("launcher_status", [0, 7, 143])
def test_bootstrap_records_provenance_and_preserves_launcher_exit(tmp_path, launcher_status):
    env = _environment(tmp_path, launcher_status=launcher_status)
    result = _run(env)
    assert result.returncode == launcher_status, result.stderr
    assert (tmp_path / "capture").read_text() == "launched\n"
    records = [json.loads(line) for line in (
        tmp_path / "output/worker-00000/allocation_attempts.jsonl"
    ).read_text().splitlines()]
    started, finished = records
    assert started["status"] == "running"
    assert started["finished_at_utc"] is None
    assert finished["status"] == ("complete" if launcher_status == 0 else "failed_or_interrupted")
    assert finished["exit_code"] == launcher_status
    assert started["attempt_id"] == finished["attempt_id"]
    assert started["started_at_utc"] == finished["started_at_utc"]
    assert finished["finished_at_utc"]
    assert finished["git_commit"] == "a" * 40
    assert finished["model_slug"] == "qwen38"
    assert finished["resources"]["SLURM_CPUS_PER_TASK"] == "32"
    assert finished["approved_walltime"] == "01:00:00"
    assert finished["allocation_estimate"] == env["GEODML_ALLOCATION_ESTIMATE"]
    assert finished["output"] == str(tmp_path / "output/worker-00000")
    assert finished["stdout"] == str(tmp_path / "slurm-987650_0-987654.out")
    assert finished["stderr"] == str(tmp_path / "slurm-987650_0-987654.err")
    assert finished["wave_manifest_sha256"] == hashlib.sha256(
        (tmp_path / "wave/run_manifest.json").read_bytes()
    ).hexdigest()
    assert finished["tasks_sha256"] == hashlib.sha256(
        (tmp_path / "wave/workers/worker-00000.jsonl").read_bytes()
    ).hexdigest()


def test_wave_refuses_duplicate_worker_before_appending_allocation(tmp_path):
    env = _environment(tmp_path)
    env["TEST_HOLD"] = str(tmp_path / "release")
    with subprocess.Popen(
        ["bash", str(WRAPPER)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ) as first:
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "capture").exists() and time.monotonic() < deadline:
                if first.poll() is not None:
                    break
                time.sleep(0.02)
            assert (tmp_path / "capture").exists()
            second_env = dict(env)
            second_env.pop("TEST_HOLD")
            result = _run(second_env)
            assert result.returncode == 75, result.stderr
            assert (tmp_path / "capture").read_text() == "launched\n"
            assert len((tmp_path / "output/worker-00000/allocation_attempts.jsonl").read_text().splitlines()) == 1
        finally:
            (tmp_path / "release").touch()
            first.communicate(timeout=5)
    assert first.returncode == 0


def test_wave_records_term_and_can_resume_with_new_attempt(tmp_path):
    env = _environment(tmp_path)
    env.pop("GEODML_MODEL_SLUG")
    env.pop("SLURM_ARRAY_JOB_ID")
    env["TEST_HOLD"] = str(tmp_path / "release")
    with subprocess.Popen(
        ["bash", str(WRAPPER)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ) as first:
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "capture").exists() and time.monotonic() < deadline:
                if first.poll() is not None:
                    break
                time.sleep(0.02)
            assert (tmp_path / "capture").exists()
            first.terminate()
            first.communicate(timeout=5)
        finally:
            (tmp_path / "release").touch()
            if first.poll() is None:
                first.terminate()
                first.communicate(timeout=5)
    assert first.returncode == 143
    env.pop("TEST_HOLD")
    assert _run(env).returncode == 0
    records = [json.loads(line) for line in (
        tmp_path / "output/worker-00000/allocation_attempts.jsonl"
    ).read_text().splitlines()]
    assert [row["status"] for row in records] == [
        "running", "failed_or_interrupted", "running", "complete",
    ]
    assert records[1]["exit_code"] == 143
    assert records[0]["attempt_id"] != records[2]["attempt_id"]
    assert all(row["model_slug"] is None for row in records)
    assert all(row["stdout"] == str(tmp_path / "slurm-987654_0-987654.out") for row in records)


@pytest.mark.parametrize("invalid", ["symlink", "dirty", "commit", "tasks"])
def test_wave_refuses_invalid_inputs_before_running(tmp_path, invalid):
    env = _environment(tmp_path)
    if invalid == "symlink":
        launcher = Path(env["GEODML_WORKER_LAUNCHER"])
        outside = tmp_path / "outside.sh"
        launcher.rename(outside)
        launcher.symlink_to(outside)
    elif invalid == "dirty":
        env["TEST_GIT_STATUS"] = " M file.py"
    elif invalid == "commit":
        env["TEST_ACTUAL_COMMIT"] = "b" * 40
    elif invalid == "tasks":
        (tmp_path / "wave/workers/worker-00000.jsonl").write_text('{}\n')
    result = _run(env)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "output/worker-00000/allocation_attempts.jsonl").exists()


def test_wave_records_actual_allocation_budget(tmp_path):
    env = _environment(tmp_path)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in (
        tmp_path / "output/worker-00000/allocation_attempts.jsonl"
    ).read_text().splitlines()]
    assert records[0]["allocation_budget"]["end_epoch"] == int(env["SLURM_JOB_END_TIME"])
    assert records[0]["allocation_budget"]["policy"] == "fill-approved-queue-v1"


@pytest.mark.parametrize("deadline", [None, "1"])
def test_wave_refuses_missing_or_elapsed_allocation_before_model_load(tmp_path, deadline):
    env = _environment(tmp_path)
    if deadline is None:
        env.pop("SLURM_JOB_END_TIME")
    else:
        env["SLURM_JOB_END_TIME"] = deadline
    result = _run(env)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()


def test_wave_does_not_label_checkpointed_result_complete(tmp_path):
    env = _environment(tmp_path)
    output = tmp_path / "output/worker-00000"
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text(json.dumps({
        "status": "checkpointed", "stop_reason": "allocation_deadline",
        "completed_count": 1, "remaining_count": 5,
    }))
    result = _run(env)
    assert result.returncode == 0, result.stderr
    terminal = json.loads((output / "allocation_attempts.jsonl").read_text().splitlines()[-1])
    assert terminal["status"] == "checkpointed"
    assert terminal["stop_reason"] == "allocation_deadline"
