"""The approved allocation launcher keeps resource and scientific gates explicit."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOCATION = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_allocation.sh"
WORKER = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_worker.sh"
NEMOTRON = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_nemotron.sh"
OVERFLOW = ROOT / "analysis/scripts/slurm/jupiter/run_agentic_adaptive_overflow.sh"


def test_shell_scripts_parse_and_never_submit_or_cancel_allocations():
    for path in (ALLOCATION, WORKER, NEMOTRON, OVERFLOW):
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
