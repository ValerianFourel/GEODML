"""Static portability checks for the HoreKa latent-prompt Slurm wrapper."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SBATCH = REPOSITORY_ROOT / "analysis/scripts/slurm/horeka/run_latent_prompt_pilot.sbatch"
SUBMIT = REPOSITORY_ROOT / "analysis/scripts/slurm/horeka/submit_latent_prompt_pilot.sh"
MANIFEST = REPOSITORY_ROOT / "analysis/scripts/slurm/horeka/record_latent_run_manifest.py"
GPU_REQUIREMENTS = REPOSITORY_ROOT / "analysis/requirements-horeka-gpu.txt"
GENERATOR = REPOSITORY_ROOT / "analysis/scripts/generate_latent_prompt_pilot.py"


class HorekaLatentPromptJobTests(unittest.TestCase):
    def test_shell_wrappers_are_valid_bash(self) -> None:
        for path in (SBATCH, SUBMIT):
            with self.subTest(path=path.name):
                result = subprocess.run(
                    ["bash", "-n", str(path)], capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_horeka_resources_are_submission_time_configuration(self) -> None:
        submit = SUBMIT.read_text(encoding="utf-8")
        self.assertIn('HOREKA_PARTITION:=accelerated', submit)
        self.assertIn('HOREKA_PARTITION:=dev_accelerated', submit)
        self.assertIn('--account="$HOREKA_ACCOUNT"', submit)
        self.assertIn('--gres="gpu:$HOREKA_GPUS"', submit)
        self.assertIn("HOREKA_GPUS=4", submit)
        self.assertIn("--export=ALL", submit)

    def test_non_four_gpu_override_is_rejected(self) -> None:
        environment = {
            **os.environ,
            "HOREKA_ACCOUNT": "test-account",
            "PROMPT_GENERATOR_MODEL": "test-model",
            "HOREKA_GPUS": "2",
        }
        result = subprocess.run(
            ["bash", str(SUBMIT), "--dry-run"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 4", result.stderr)

    def test_job_has_validation_mode_and_scientific_provenance(self) -> None:
        job = SBATCH.read_text(encoding="utf-8")
        self.assertIn("LATENT_PROMPT_VALIDATE_ONLY", job)
        self.assertIn("#SBATCH --gres=gpu:4", job)
        self.assertIn("GEODML_REQUIRED_GPU_COUNT=4", job)
        self.assertIn("GEODML_DEVICE_MAP=balanced", job)
        self.assertIn('VISIBLE_GPU_COUNT" != "4"', job)
        self.assertIn("four-GPU visibility contract satisfied", job)
        self.assertIn("import accelerate, bitsandbytes", job)
        self.assertIn("run_manifest.json", job)
        self.assertIn("git rev-parse HEAD", job)
        self.assertIn('srun --ntasks=1 python3 "${arguments[@]}"', job)
        self.assertTrue(MANIFEST.is_file())

    def test_gpu_requirements_declare_sharding_dependencies(self) -> None:
        requirements = GPU_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("accelerate>=", requirements)
        self.assertIn("bitsandbytes>=", requirements)
        self.assertIn("transformers>=", requirements)

    def test_generator_persists_complete_provider_failure_diagnostics(self) -> None:
        generator = GENERATOR.read_text(encoding="utf-8")
        self.assertIn("PromptProviderValidationError", generator)
        self.assertIn("latent_prompt_failure.json", generator)
        self.assertIn('"serp_input"', generator)
        self.assertIn('"meta_prompt_request"', generator)
        self.assertIn('"generation_configuration"', generator)
        self.assertIn('"attempts"', generator)
        self.assertIn("asdict(attempt)", generator)

    def test_job_does_not_copy_juelich_specific_infrastructure(self) -> None:
        combined = SBATCH.read_text(encoding="utf-8") + SUBMIT.read_text(encoding="utf-8")
        for forbidden in ("jutil", "/e/scratch", "--partition=booster"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
