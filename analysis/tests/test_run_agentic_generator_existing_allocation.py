from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "analysis/scripts/slurm/jupiter/run_agentic_generator_existing_allocation.sh"


def test_existing_allocation_helper_exposes_compact_interface() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "prepare|launch" in result.stdout
    assert "RESUME_RUN_ROOT RUN_ROOT" in result.stdout


def test_existing_allocation_helper_rejects_unknown_action() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "unknown"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 64
    assert "prepare or launch" in result.stderr


def test_existing_allocation_helper_initializes_external_srun_environment() -> None:
    script = SCRIPT.read_text()

    assert script.index("module load Stages/2026 GCC Python CUDA git") < script.index(
        'git -C "$repository" status'
    )
    assert 'scontrol show job --oneliner "$job_id"' in script
    assert 'export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-' in script
    assert 'export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE:-' in script
    assert 'export SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME:-' in script
