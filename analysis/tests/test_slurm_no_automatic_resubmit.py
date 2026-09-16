"""A wall-time checkpoint must not silently purchase another allocation."""

import os
import subprocess
from pathlib import Path


def test_legacy_chain_helper_does_not_submit_another_allocation():
    common = Path(__file__).resolve().parents[2] / "analysis/scripts/slurm/_common.sh"
    function = "chain_resubmit() {" + common.read_text().split("chain_resubmit() {", 1)[1]
    source = "sbatch() { printf 'UNAUTHORIZED_SBATCH\\n'; }\n" + function + "\nchain_resubmit script.sbatch\n"
    result = subprocess.run(["bash", "-c", source], text=True, capture_output=True,
                            env={**os.environ, "SLURM_JOB_ID": "123", "ATTEMPT": "1",
                                 "MAX_ATTEMPTS": "6", "JUWELS_ACCOUNT": "test"}, check=False)
    assert "UNAUTHORIZED_SBATCH" not in result.stdout
    assert "approval" in result.stdout.lower()
    assert result.returncode != 0
