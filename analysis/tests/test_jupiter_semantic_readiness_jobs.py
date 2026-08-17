"""Static contracts for JUPITER Phase-2 readiness judge jobs."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JUPITER_ROOT = REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter"
SBATCH = JUPITER_ROOT / "run_semantic_readiness_judge.sbatch"
SUBMIT = JUPITER_ROOT / "submit_semantic_readiness_panel.sh"


class JupiterSemanticReadinessJobTests(unittest.TestCase):
    def test_shell_scripts_are_valid_bash(self) -> None:
        for script in (SBATCH, SUBMIT):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_job_requires_one_complete_gh200_node(self) -> None:
        job = SBATCH.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=booster", job)
        self.assertIn("#SBATCH --nodes=1", job)
        self.assertIn("#SBATCH --gres=gpu:4", job)
        self.assertIn("module load Stages/2026", job)
        self.assertIn("module load PyTorch/2.9.1", job)
        self.assertIn("GEODML_REQUIRED_GPU_COUNT=4", job)
        self.assertIn('SLURM_JOB_PARTITION:-}\" != \"booster', job)
        self.assertIn('gpu_name\" != *GH200*', job)

    def test_job_pins_provenance_and_runs_offline(self) -> None:
        job = SBATCH.read_text(encoding="utf-8")

        self.assertIn("GEODML_EXPECTED_COMMIT", job)
        self.assertIn("READINESS_JUDGE_TASKS_SHA256", job)
        self.assertIn("READINESS_EXPECTED_TASKS_PER_SLOT", job)
        self.assertIn("JUDGE_MODEL_REVISION", job)
        self.assertIn("JUDGE_MODEL_FAMILY", job)
        self.assertIn("HF_HUB_OFFLINE=1", job)
        self.assertIn("TRANSFORMERS_OFFLINE=1", job)
        self.assertIn("--model-revision \"$JUDGE_MODEL_REVISION\"", job)
        self.assertIn("--model-family \"$JUDGE_MODEL_FAMILY\"", job)
        self.assertIn("--run-purpose production", job)
        self.assertIn("--resume", job)
        self.assertIn("artifact-sha256.txt", job)
        self.assertIn('manifest["task_count_for_slot"]', job)

    def test_submitter_requires_three_distinct_slots_and_snapshots(self) -> None:
        submitter = SUBMIT.read_text(encoding="utf-8")

        self.assertIn("PRIMARY_JUDGE_MODEL", submitter)
        self.assertIn("PRIMARY_JUDGE_FAMILY", submitter)
        self.assertIn("REPLICATE_A_JUDGE_MODEL", submitter)
        self.assertIn("REPLICATE_B_JUDGE_MODEL", submitter)
        self.assertIn("must use distinct model snapshots", submitter)
        self.assertIn("must use distinct model families", submitter)
        self.assertIn("Task-bank hash mismatch", submitter)
        self.assertIn("Refusing to change the frozen judge panel", submitter)
        self.assertIn("primary-frontier", submitter)
        self.assertIn("replicate-frontier-a", submitter)
        self.assertIn("replicate-frontier-b", submitter)
        self.assertEqual(submitter.count("\nsubmit_one "), 3)


if __name__ == "__main__":
    unittest.main()
