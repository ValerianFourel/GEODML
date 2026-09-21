"""The approved allocation launcher keeps resource and scientific gates explicit."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOCATION = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_allocation.sh"
WORKER = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_worker.sh"
NEMOTRON = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_nemotron.sh"
OVERFLOW = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_overflow.sh"
DIAGNOSTIC = ROOT / "analysis/scripts/slurm/jupiter/diagnose_agentic_adaptive_allocation.sh"


def test_diagnostic_uses_one_existing_worker_step_and_no_inference(tmp_path):
    srun = tmp_path / "srun"
    srun.write_text(r'''#!/bin/bash
[[ -z "${SLURM_CPUS_PER_TASK+x}" && -z "${SLURM_STEP_ID+x}" ]] || exit 1
[[ -z "${SRUN_CPUS_PER_TASK+x}" && -z "${CUDA_VISIBLE_DEVICES+x}" ]] || exit 1
[[ "$SLURM_JOB_ID" == 1927510 && "$SLURM_JOB_NUM_NODES" == 5 ]] || exit 1
printf '%s\n' "$@"
exit 2
''')
    srun.chmod(0o755)
    prelude = r'''
scontrol() {
  if [[ "$2" == job ]]; then
    printf 'JobId=1927510 UserId=test(%s) JobState=%s NumNodes=5 NodeList=node[0-4]\n' "$(id -u)" "$TEST_JOB_STATE"
  else
    printf 'node0\nnode1\nnode2\nnode3\nnode4\n'
  fi
}
export -f scontrol
bash "$1" 1927510
'''
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                   "SLURM_CPUS_PER_TASK": "1", "SLURM_STEP_ID": "7",
                   "SRUN_CPUS_PER_TASK": "1", "CUDA_VISIBLE_DEVICES": "0"}
    for state in ("RUNNING", "COMPLETED"):
        result = subprocess.run(
            ["bash", "-c", prelude, "test", str(DIAGNOSTIC)],
            env={**environment, "TEST_JOB_STATE": state}, text=True, capture_output=True, check=False,
        )
        if state == "RUNNING":
            assert result.returncode == 2, result.stderr
            assert result.stdout.count("--jobid=1927510") == 1
            for argument in ("--nodes=1", "--ntasks=1", "--nodelist=node0", "--gpus-per-node=4",
                             "--cpus-per-task=32", "--mem=512G", "--immediate=15", "--diagnose-slurm"):
                assert argument in result.stdout
            assert "run_agentic_adaptive_worker" not in result.stdout
        else:
            assert result.returncode != 0
            assert "--jobid" not in result.stdout


def test_launcher_drops_inspection_step_resource_environment(tmp_path):
    plan = tmp_path / "run_manifest.json"
    plan.write_text("{}")
    srun = tmp_path / "srun"
    srun.write_text(r'''#!/bin/bash
[[ -z "${SLURM_CPUS_PER_TASK+x}" ]] || exit 1
[[ -z "${SLURM_JOB_CPUS_PER_NODE+x}" ]] || exit 1
[[ -z "${SLURM_STEP_ID+x}" ]] || exit 1
[[ -z "${SLURM_MEM_PER_NODE+x}" ]] || exit 1
[[ -z "${SRUN_CPUS_PER_TASK+x}" ]] || exit 1
[[ -z "${CUDA_VISIBLE_DEVICES+x}" ]] || exit 1
[[ "$SLURM_JOB_ID" == 1927510 ]] || exit 1
[[ -n "$SLURM_JOB_END_TIME" ]] || exit 1
''')
    srun.chmod(0o755)
    prelude = r'''
scontrol() {
  if [[ "$2" == job ]]; then
    echo 'StartTime=2026-09-21T00:00:00 EndTime=2026-09-21T07:00:00 NodeList=node[0-4] NumNodes=5'
  else
    printf 'node0\nnode1\nnode2\nnode3\nnode4\n'
  fi
}
date() { echo 1790000000; }
readarray() { IFS=$'\n' read -r -d '' -a nodes || true; }
export -f scontrol date readarray
bash "$1" "$2"
'''
    result = subprocess.run(
        ["bash", "-c", prelude, "test", str(ALLOCATION), str(plan)],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
             "SLURM_JOB_ID": "1927510", "SLURM_JOB_NUM_NODES": "5",
             "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_CPUS_PER_NODE": "1",
             "SLURM_MEM_PER_NODE": "1024", "SLURM_STEP_ID": "7",
             "SRUN_CPUS_PER_TASK": "1", "CUDA_VISIBLE_DEVICES": "0"},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shell_scripts_parse_and_never_submit_or_cancel_allocations():
    for path in (ALLOCATION, WORKER, NEMOTRON, OVERFLOW, DIAGNOSTIC):
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "sbatch " not in text
        assert "salloc " not in text
        assert "scancel" not in text


def test_launcher_uses_exact_approved_five_node_budget_and_child_steps():
    text = ALLOCATION.read_text()
    assert 'test "${SLURM_JOB_NUM_NODES:?}" = 5' in text
    assert "GEODML_APPROVED_WALLTIME=07:00:00" in text
    assert "for worker_index in 0 1 2 3 4" in text
    assert '--jobid="$job_id" --nodes=1 --ntasks=1' in text
    assert "--gpus-per-node=4 --cpus-per-task=32 --mem=512G" in text
    assert "allocation-owning shell remains open" in text


def test_worker_enforces_generation_barrier_before_judge_plan():
    text = WORKER.read_text()
    assert 'test "${SLURM_JOB_NUM_NODES:?}" = 5' in text
    assert 'test "${SLURM_STEP_NUM_NODES:?}" = 1' in text
    materialize = text.index("materialize_agentic_generator_claims.py")
    prepare = text.index("prepare_agentic_judge_tasks.py")
    nemotron = text.index("run_agentic_adaptive_nemotron.sh")
    assert materialize < prepare < nemotron
    assert "--recorded-conversation" in text
    assert "SEARCH_AGENTIC_WORKER_COUNT=5" in text
    assert "SEARCH_AGENTIC_EXPECTED_CELL_COUNT=6000" in text


def test_nemotron_profile_and_gpu_release_are_checked():
    text = NEMOTRON.read_text()
    assert 'test "$SLURM_JOB_NUM_NODES" = 5' in text
    assert 'test "$SLURM_STEP_NUM_NODES" = 1' in text
    assert 'assert p["serving"]["max_model_len"] == 73728' in text
    assert 'assert p["serving"]["dtype"] == "bfloat16"' in text
    assert 'assert "--enforce-eager" in p["server_argv"]' in text
    assert "GPU process remained after Nemotron stage shutdown" in text


def test_overflow_finishes_paired_qwen_before_partitioning_1200_backlogs():
    text = OVERFLOW.read_text()
    assert text.index("paired-qwen38-imported.json") < text.index("backlog-$slug")
    assert "--import-only" in text
    assert "if (( worker < 2 ))" in text
    assert "model_workers=2" in text
    assert "model_workers=3" in text
