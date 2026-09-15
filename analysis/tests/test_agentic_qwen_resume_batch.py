"""CPU-only shell contracts for the legacy Qwen shard resume wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "scripts/slurm/jupiter/run_agentic_search_qwen38_resume.sbatch"
)


def _environment(root: Path, *, launcher_status: int = 0) -> dict[str, str]:
    repository = root / "repository"
    launcher = repository / "analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test \"$GEODML_EXPECTED_JOB_ID\" = \"$SLURM_JOB_ID\"\n"
        "test -z \"${SEARCH_AGENTIC_CELL_IDS_JSONL+x}\"\n"
        "test \"$SEARCH_AGENTIC_PROMPT_COUNT\" = 500\n"
        "test \"$SEARCH_AGENTIC_PROMPT_SHARD_INDEX\" = 3\n"
        "test \"$SEARCH_AGENTIC_PROMPT_SHARD_COUNT\" = 4\n"
        "test \"$SEARCH_AGENTIC_EXPECTED_CELL_COUNT\" = 1500\n"
        "printf '%s\\n' \"$SEARCH_AGENTIC_SERVER_LOG\" \"$SEARCH_AGENTIC_GPU_TELEMETRY\" > \"$TEST_CAPTURE\"\n"
        "if [[ -n \"${TEST_HOLD_LAUNCHER_FILE:-}\" ]]; then\n"
        "  while [[ ! -e \"$TEST_HOLD_LAUNCHER_FILE\" ]]; do sleep 0.02; done\n"
        "fi\n"
        "exit \"$TEST_LAUNCHER_STATUS\"\n",
        encoding="utf-8",
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
        "esac\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    flock = commands / "flock"
    flock.write_text(
        f"#!{sys.executable}\n"
        "import fcntl, sys\n"
        "try:\n"
        "    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    flock.chmod(0o755)
    (commands / "python3").symlink_to(sys.executable)
    bash_env = root / "bash-env"
    bash_env.write_text(
        "module() {\n"
        "  [[ $- != *u* ]] || return 92\n"
        "  printf '%s\\n' \"$*\" >> \"$TEST_MODULE_CALLS\"\n"
        "}\n",
        encoding="utf-8",
    )
    activation = root / "venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text('[[ $- != *u* ]] || return 93\n', encoding="utf-8")
    output = root / "output"
    output.mkdir()
    (output / "config.json").write_text('{"config_sha256":"source-hash"}\n')
    profile = root / "profile.json"
    profile.write_text("{}\n", encoding="utf-8")
    return {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "BASH_ENV": str(bash_env),
        "SHELLOPTS": "braceexpand:hashall:interactive-comments:nounset",
        "ACL_ARR_VENV": str(activation.parent.parent),
        "GEODML_EXECUTION_REPOSITORY": str(repository),
        "GEODML_EXECUTION_COMMIT": "a" * 40,
        "GEODML_EXPECTED_JOB_ID": "old-job",
        "GEODML_APPROVED_WALLTIME": "01:00:00",
        "GEODML_ALLOCATION_ESTIMATE": "10-25 minutes for six missing cells",
        "GEODML_RESUME_ATTEMPT_DIR": str(root / "attempt"),
        "SLURM_JOB_ID": "987654",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "524288",
        "SLURM_GPUS_ON_NODE": "4",
        "SEARCH_AGENTIC_OUTPUT": str(output),
        "SEARCH_AGENTIC_PROFILE": str(profile),
        "SEARCH_AGENTIC_CELL_IDS_JSONL": "must-be-cleared.jsonl",
        "SEARCH_AGENTIC_PROMPT_COUNT": "500",
        "SEARCH_AGENTIC_PROMPT_SHARD_INDEX": "3",
        "SEARCH_AGENTIC_PROMPT_SHARD_COUNT": "4",
        "SEARCH_AGENTIC_EXPECTED_CELL_COUNT": "1500",
        "TEST_CAPTURE": str(root / "capture"),
        "TEST_MODULE_CALLS": str(root / "modules"),
        "TEST_LAUNCHER_STATUS": str(launcher_status),
    }


@pytest.mark.parametrize("launcher_status", [0, 7])
def test_resume_wrapper_binds_job_and_propagates_status(tmp_path, launcher_status):
    env = _environment(tmp_path, launcher_status=launcher_status)
    result = subprocess.run(
        ["bash", str(WRAPPER)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == launcher_status, result.stderr
    attempt = tmp_path / "attempt"
    assert (tmp_path / "capture").read_text().splitlines() == [
        str(attempt / "qwen38-job987654.server.log"),
        str(attempt / "qwen38-job987654.gpu.csv"),
    ]
    assert (tmp_path / "modules").read_text().splitlines() == [
        "load Stages/2026 GCC Python CUDA", "load git",
    ]
    record, = [json.loads(line) for line in (attempt / "allocation_attempts.jsonl").read_text().splitlines()]
    assert record["slurm_job_id"] == "987654"
    assert record["approved_walltime"] == "01:00:00"
    assert record["allocation_estimate"] == env["GEODML_ALLOCATION_ESTIMATE"]
    assert record["git_commit"] == "a" * 40
    assert record["source_config_sha256"] == "source-hash"
    assert record["output"] == str(tmp_path / "output")
    assert record["resources"]["SLURM_GPUS_ON_NODE"] == "4"
    assert record["resources"]["SLURM_CPUS_PER_TASK"] == "32"
    assert record["resources"]["SLURM_MEM_PER_NODE"] == "524288"
    assert record["started_at_utc"]
    assert record["stdout"] == str(attempt / "slurm-987654.out")
    assert record["stderr"] == str(attempt / "slurm-987654.err")


@pytest.mark.parametrize("blocked", ["dirty", "commit", "logs", "approval"])
def test_resume_wrapper_refuses_invalid_preflight(tmp_path, blocked):
    env = _environment(tmp_path)
    if blocked == "dirty":
        env["TEST_GIT_STATUS"] = " M file.py"
    elif blocked == "commit":
        env["TEST_ACTUAL_COMMIT"] = "b" * 40
    elif blocked == "logs":
        attempt = tmp_path / "attempt"
        attempt.mkdir()
        (attempt / "qwen38-job987654.server.log").write_text("preserved")
    else:
        del env["GEODML_APPROVED_WALLTIME"]
    result = subprocess.run(
        ["bash", str(WRAPPER)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "attempt/allocation_attempts.jsonl").exists()


def test_resume_wrapper_has_valid_shell_and_no_allocation_defaults():
    text = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH" not in text
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_resume_wrapper_rejects_concurrent_writer_until_launcher_exits(tmp_path):
    env = _environment(tmp_path)
    release = tmp_path / "release"
    env["TEST_HOLD_LAUNCHER_FILE"] = str(release)
    first = subprocess.Popen(
        ["bash", str(WRAPPER)], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "capture").exists() and time.monotonic() < deadline:
            assert first.poll() is None, "first worker exited before its launcher"
            time.sleep(0.02)
        assert (tmp_path / "capture").exists()
        second_env = {
            **env,
            "SLURM_JOB_ID": "987655",
            "GEODML_RESUME_ATTEMPT_DIR": str(tmp_path / "second-attempt"),
            "TEST_CAPTURE": str(tmp_path / "second-capture"),
        }
        second_env.pop("TEST_HOLD_LAUNCHER_FILE")
        second = subprocess.run(
            ["bash", str(WRAPPER)], env=second_env, capture_output=True,
            text=True, check=False, timeout=5,
        )
        assert second.returncode == 75, second.stderr
        assert "already being resumed" in second.stderr
        assert not (tmp_path / "second-capture").exists()
        assert not (tmp_path / "second-attempt/allocation_attempts.jsonl").exists()
    finally:
        release.touch()
        _, stderr = first.communicate(timeout=5)
    assert first.returncode == 0, stderr
    after = subprocess.run(
        ["bash", str(WRAPPER)], env=second_env, capture_output=True,
        text=True, check=False, timeout=5,
    )
    assert after.returncode == 0, after.stderr
    assert (tmp_path / "second-capture").exists()
