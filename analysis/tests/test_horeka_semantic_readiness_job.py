"""Static contracts for the four-A100 HoreKa readiness judge job."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SBATCH = (
    REPOSITORY_ROOT
    / "analysis/scripts/slurm/horeka/run_semantic_readiness_judge.sbatch"
)


class HorekaSemanticReadinessJobTests(unittest.TestCase):
    def test_wrapper_is_valid_bash(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SBATCH)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_job_requires_one_four_a100_node(self) -> None:
        job = SBATCH.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=accelerated", job)
        self.assertIn("#SBATCH --nodes=1", job)
        self.assertIn("#SBATCH --gres=gpu:4", job)
        self.assertIn("GEODML_REQUIRED_GPU_COUNT=4", job)
        self.assertIn("GEODML_DEVICE_MAP=balanced", job)
        self.assertIn('SLURM_JOB_PARTITION:-}\" != \"accelerated', job)
        self.assertIn('VISIBLE_GPU_COUNT\" != \"4', job)
        self.assertIn('gpu_name\" != *A100*', job)
        self.assertIn("four-A100 contract satisfied", job)


if __name__ == "__main__":
    unittest.main()
