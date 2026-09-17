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
