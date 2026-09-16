"""CPU-only batch contracts, with fake Git and report subprocesses."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.scripts import run_axis_permutation_report_batch as batch

REPOSITORY = Path(__file__).resolve().parents[2]
WRAPPER = REPOSITORY / "analysis/scripts/slurm/jupiter/run_axis_permutation_report.sbatch"


@pytest.fixture
def environment(tmp_path):
    repository = tmp_path / "repository"
    scripts = repository / "analysis/scripts"
    scripts.mkdir(parents=True)
    (scripts / "report_axis_permutation_study.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['TEST_REPORT_CALL']).write_text(json.dumps(sys.argv[1:]))\n"
        "if os.environ.get('TEST_CHANGE_CONFIG'):\n"
        "    pathlib.Path(sys.argv[sys.argv.index('--config') + 1]).write_text('{}')\n"
        "sys.exit(int(os.environ.get('TEST_REPORT_STATUS', '0')))\n"
    )
    (scripts / "inference_network_namespace.py").write_text(
        "import os, sys\n"
        "assert sys.argv[1:] == ['--check']\n"
        "print('SERVER_FREE_CHECK')\n"
        "sys.exit(int(os.environ.get('TEST_NETWORK_STATUS', '0')))\n"
    )
    commands = tmp_path / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[3] == 'rev-parse': print(os.environ.get('TEST_ACTUAL_COMMIT', os.environ['GEODML_EXECUTION_COMMIT']))\n"
        "elif sys.argv[3] == 'status': print(os.environ.get('TEST_GIT_DIRTY', ''))\n"
        "else: sys.exit(91)\n"
    )
    git.chmod(0o755)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format_version": "axis-permutation-study-config-v1", "study_id": "fixture",
        "axis_fields": {"target_normalized_axis_1": {"role": "assigned", "construct": "readiness"}},
        "sources": {"agentic": [{"generator_roots": ["unused-double-source"]}]},
    }))
    return {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "4", "SLURM_GPUS_ON_NODE": "4",
        "GEODML_EXECUTION_REPOSITORY": str(repository), "GEODML_EXECUTION_COMMIT": "a" * 40,
        "GEODML_AXIS_REPORT_CONFIG": str(config), "GEODML_AXIS_REPORT_OUTPUT": str(tmp_path / "output"),
        "GEODML_REPORT_ATTEMPT_DIR": str(tmp_path / "attempt"), "GEODML_APPROVED_WALLTIME": "00:20:00",
        "GEODML_ALLOCATION_ESTIMATE": "CPU reporting estimated 5-15 minutes; five-minute margin.",
        "TEST_REPORT_CALL": str(tmp_path / "report-call.json"),
    }


def _record(env):
    return json.loads((Path(env["GEODML_REPORT_ATTEMPT_DIR"]) / "allocation.json").read_text())


@pytest.mark.parametrize("report_status", [0, 7])
def test_report_subprocess_and_final_allocation(environment, report_status):
    environment["TEST_REPORT_STATUS"] = str(report_status)
    assert batch.run_batch(environment) == report_status
    record = _record(environment)
    assert record["status"] == ("complete" if report_status == 0 else "failed")
    assert record["exit_code"] == report_status
    assert record["started_at_utc"] <= record["finished_at_utc"]
    assert record["resources"]["SLURM_CPUS_PER_TASK"] == "4"
    assert record["approved_walltime"] == "00:20:00"
    assert record["inference_performed"] is False
    assert record["scientific_result"] is False
    assert record["network_isolation_check"]["status"] == "passed"
    assert len(record["config_sha256"]) == 64
    assert json.loads(Path(environment["TEST_REPORT_CALL"]).read_text()) == [
        "--config", environment["GEODML_AXIS_REPORT_CONFIG"],
        "--output-dir", environment["GEODML_AXIS_REPORT_OUTPUT"],
    ]


def test_failed_network_check_does_not_block_report(environment):
    environment["TEST_NETWORK_STATUS"] = "2"
    assert batch.run_batch(environment) == 0
    network = _record(environment)["network_isolation_check"]
    assert network["status"] == "failed"
    assert network["exit_code"] == 2
    assert Path(network["log"]).read_text() == "SERVER_FREE_CHECK\n"


def test_network_check_timeout_is_bounded_and_nonfatal(environment, monkeypatch):
    original_run = subprocess.run

    def run(arguments, **kwargs):
        if arguments[-1] == "--check":
            assert kwargs["timeout"] == 30
            raise subprocess.TimeoutExpired(arguments, 30)
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(batch.subprocess, "run", run)
    assert batch.run_batch(environment) == 0
    assert _record(environment)["network_isolation_check"]["exit_code"] == 124


def test_network_check_can_be_disabled(environment):
    environment["GEODML_AXIS_REPORT_CHECK_NETWORK"] = "0"
    assert batch.run_batch(environment) == 0
    assert _record(environment)["network_isolation_check"]["status"] == "not_requested"
    assert not (Path(environment["GEODML_REPORT_ATTEMPT_DIR"]) / "network-isolation.log").exists()


def test_duplicate_attempt_never_overwrites_record(environment):
    assert batch.run_batch(environment) == 0
    before = _record(environment)
    with pytest.raises(RuntimeError, match="already exists"):
        batch.run_batch(environment)
    assert _record(environment) == before


def test_parallel_attempt_lock(environment):
    attempt = Path(environment["GEODML_REPORT_ATTEMPT_DIR"])
    attempt.mkdir()
    with (attempt / ".batch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            batch.run_batch(environment)
    assert not (attempt / "allocation.json").exists()
    assert not Path(environment["TEST_REPORT_CALL"]).exists()


@pytest.mark.parametrize("bad_environment,match", [
    ({"TEST_GIT_DIRTY": "?? changed.py"}, "exact clean"),
    ({"TEST_ACTUAL_COMMIT": "b" * 40}, "exact clean"),
    ({"GEODML_AXIS_REPORT_CONFIG_SHA256": "0" * 64}, "submitted hash"),
    ({"TEST_CHANGE_CONFIG": "1"}, "changed during"),
])
def test_failure_is_recorded_and_propagates(environment, bad_environment, match):
    environment.update(bad_environment)
    with pytest.raises(ValueError, match=match):
        batch.run_batch(environment)
    record = _record(environment)
    assert record["status"] == "failed"
    assert record["exit_code"] == 1
    assert record["finished_at_utc"]


@pytest.mark.parametrize("name,value", [
    ("GEODML_APPROVED_WALLTIME", ""), ("GEODML_APPROVED_WALLTIME", "00:00:00"),
    ("GEODML_APPROVED_WALLTIME", "00:75:00"), ("GEODML_EXECUTION_COMMIT", "main"),
    ("GEODML_AXIS_REPORT_OUTPUT", "relative"), ("SLURM_JOB_ID", ""),
])
def test_invalid_allocation_inputs_refused(environment, name, value):
    environment[name] = value
    with pytest.raises(ValueError):
        batch.run_batch(environment)
    assert not Path(environment["TEST_REPORT_CALL"]).exists()


def test_real_report_generates_saved_snapshot_without_model(environment, tmp_path):
    from analysis.tests.test_axis_permutation_inputs import agentic

    source, _, _ = agentic(tmp_path / "input")
    config_path = Path(environment["GEODML_AXIS_REPORT_CONFIG"])
    config = json.loads(config_path.read_text())
    config["sources"] = {"agentic": [source]}
    config_path.write_text(json.dumps(config))
    environment.update(GEODML_EXECUTION_REPOSITORY=str(REPOSITORY), GEODML_AXIS_REPORT_CHECK_NETWORK="0")
    assert batch.run_batch(environment) == 0
    output = Path(environment["GEODML_AXIS_REPORT_OUTPUT"])
    latest = json.loads((output / "latest.json").read_text())
    snapshot = output / "snapshots" / latest["snapshot_id"]
    report = json.loads((snapshot / "report.json").read_text())
    assert report["generation"]["completed"] == 1
    assert report["generation"]["expected"] == 2
    assert report["scientific_result"] is False
    assert (snapshot / "report.md").is_file()
    assert len((snapshot / "questions.jsonl").read_text().splitlines()) == 1
    assert _record(environment)["status"] == "complete"


def test_shell_bootstrap_is_cpu_only_and_syntax_valid(environment, tmp_path):
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    activation = tmp_path / "venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text('[[ $- != *u* ]] || return 93\n')
    bash_env = tmp_path / "bash-env"
    calls = tmp_path / "modules"
    bash_env.write_text(
        "module() {\n"
        "  [[ $- != *u* ]] || return 92\n"
        "  test -n \"${DEBUGINFOD_URLS+x}\" || return 94\n"
        "  printf '%s\\n' \"$*\" >> \"$TEST_MODULE_CALLS\"\n"
        "}\n"
    )
    capture = Path(environment["GEODML_EXECUTION_REPOSITORY"]) / "analysis/scripts/run_axis_permutation_report_batch.py"
    capture.write_text(
        "import json, os, pathlib\n"
        "pathlib.Path(os.environ['TEST_REPORT_CALL']).write_text(json.dumps({\n"
        "k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}))\n"
    )
    commands = Path(environment["PATH"].split(os.pathsep)[0])
    (commands / "python3").symlink_to(sys.executable)
    environment.update(ACL_ARR_VENV=str(activation.parent.parent), BASH_ENV=str(bash_env),
                       TEST_MODULE_CALLS=str(calls))
    subprocess.run(["bash", "-u", str(WRAPPER)], env=environment, check=True)
    assert calls.read_text().splitlines() == ["load Stages/2026 GCC Python", "load git"]
    assert set(json.loads(Path(environment["TEST_REPORT_CALL"]).read_text()).values()) == {"4"}
    assert "#SBATCH" not in WRAPPER.read_text()
