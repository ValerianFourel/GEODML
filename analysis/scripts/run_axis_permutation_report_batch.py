"""Run saved-artifact reporting in Slurm, without loading a model or serving an API."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.interpretability.pipeline.axis_permutation_report import analysis_settings


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _path(env, name):
    value = Path(env[name])
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return value.resolve()


def _probe_network(repository, attempt, env):
    enabled = env.get("GEODML_AXIS_REPORT_CHECK_NETWORK", "1")
    if enabled not in {"0", "1"}:
        raise ValueError("GEODML_AXIS_REPORT_CHECK_NETWORK must be 0 or 1")
    if enabled == "0":
        return {"status": "not_requested", "exit_code": None}
    log = attempt / "network-isolation.log"
    with log.open("x", encoding="utf-8") as stream:
        try:
            result = subprocess.run(
                [sys.executable, str(repository / "analysis/scripts/inference_network_namespace.py"), "--check"],
                cwd=repository, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=30, check=False,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            stream.write("Network-isolation check timed out after 30 seconds.\n")
            code = 124
        except OSError as error:
            stream.write(f"Network-isolation check could not start: {error}\n")
            code = 127
    return {"status": "passed" if code == 0 else "failed", "exit_code": code,
            "log": str(log), "timeout_seconds": 30,
            "scope": "server-free diagnostic; failure does not block CPU reporting"}


def run_batch(environment=None):
    env = dict(os.environ if environment is None else environment)
    required = (
        "SLURM_JOB_ID", "GEODML_EXECUTION_REPOSITORY", "GEODML_EXECUTION_COMMIT",
        "GEODML_AXIS_REPORT_CONFIG", "GEODML_AXIS_REPORT_OUTPUT", "GEODML_REPORT_ATTEMPT_DIR",
        "GEODML_APPROVED_WALLTIME", "GEODML_ALLOCATION_ESTIMATE",
    )
    for name in required:
        if not env.get(name, "").strip():
            raise ValueError(f"{name} is required")
    commit = env["GEODML_EXECUTION_COMMIT"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("GEODML_EXECUTION_COMMIT must be a full Git commit SHA")
    walltime = env["GEODML_APPROVED_WALLTIME"]
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", walltime)
    if not match or sum(int(value) for value in match.groups()) == 0:
        raise ValueError("GEODML_APPROVED_WALLTIME must be positive HH:MM:SS")
    repository = _path(env, "GEODML_EXECUTION_REPOSITORY")
    config_path = _path(env, "GEODML_AXIS_REPORT_CONFIG")
    output = _path(env, "GEODML_AXIS_REPORT_OUTPUT")
    attempt = _path(env, "GEODML_REPORT_ATTEMPT_DIR")
    attempt.mkdir(parents=True, exist_ok=True)
    with (attempt / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("report attempt is already running") from None
        allocation_path = attempt / "allocation.json"
        if allocation_path.exists():
            raise RuntimeError("report attempt already exists; use a fresh approved attempt directory")
        record = {
            "format_version": "axis-permutation-report-allocation-v1", "status": "starting",
            "slurm_job_id": env["SLURM_JOB_ID"], "git_commit": commit,
            "approved_walltime": walltime, "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
            "started_at_utc": _now(), "finished_at_utc": None, "exit_code": None,
            "repository": str(repository), "config": str(config_path), "output": str(output),
            "scientific_result": False, "inference_performed": False,
            "stdout": env.get("GEODML_REPORT_STDOUT", str(attempt / f"slurm-{env['SLURM_JOB_ID']}.out")),
            "stderr": env.get("GEODML_REPORT_STDERR", str(attempt / f"slurm-{env['SLURM_JOB_ID']}.err")),
            "python_executable": sys.executable, "python_version": sys.version,
            "virtual_environment": env.get("VIRTUAL_ENV"),
            "resources": {name: env.get(name) for name in (
                "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
                "SLURM_MEM_PER_NODE", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GRES",
                "SLURM_JOB_NODELIST", "SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME",
            )},
        }
        _write_json(allocation_path, record)
        try:
            actual = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                    env=env, text=True, capture_output=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
                                   env=env, text=True, capture_output=True, check=True).stdout.strip()
            if actual != commit or dirty:
                raise ValueError("report execution requires the exact clean pinned Git commit")
            config_bytes = config_path.read_bytes()
            record["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
            expected = env.get("GEODML_AXIS_REPORT_CONFIG_SHA256")
            if expected and expected != record["config_sha256"]:
                raise ValueError("report configuration does not match the submitted hash")
            config = json.loads(config_bytes)
            record["analysis_settings"] = analysis_settings(config)
            sources = config.get("sources")
            if (not isinstance(sources, dict) or not any(sources.values())
                    or set(sources) - {"direct", "agentic", "judges"}):
                raise ValueError("report configuration must name known saved-artifact sources")
            record["versions"] = {}
            for package in ("numpy", "scipy"):
                try:
                    record["versions"][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    record["versions"][package] = "not_installed"
            record["network_isolation_check"] = _probe_network(repository, attempt, env)
            record["status"] = "reporting"
            _write_json(allocation_path, record)
            print("AXIS_REPORT_ALLOCATION=" + json.dumps(record, sort_keys=True), flush=True)
            result = subprocess.run(
                [sys.executable, str(repository / "analysis/scripts/report_axis_permutation_study.py"),
                 "--config", str(config_path), "--output-dir", str(output)],
                cwd=repository, env=env, check=False,
            )
            code = result.returncode if result.returncode >= 0 else 128 - result.returncode
            if config_path.read_bytes() != config_bytes:
                raise ValueError("report configuration changed during execution")
            record["exit_code"] = code
            record["status"] = "complete" if code == 0 else "failed"
            return code
        except BaseException as error:
            record.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["finished_at_utc"] = _now()
            _write_json(allocation_path, record)


if __name__ == "__main__":
    raise SystemExit(run_batch())
