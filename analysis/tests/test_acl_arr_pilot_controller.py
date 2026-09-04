from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = (
    REPOSITORY_ROOT
    / "analysis/scripts/slurm/jupiter/run_acl_arr_document_pilot_4gpu.sh"
)


class AclArrPilotControllerTests(unittest.TestCase):
    def test_controller_is_valid_bash_and_freezes_approved_contract(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(CONTROLLER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn('ACL_ARR_APPROVED_WALLTIME" != "03:00:00"', source)
        self.assertIn('TENSOR_PARALLEL_SIZE="${ACL_ARR_TENSOR_PARALLEL_SIZE:-4}"', source)
        self.assertIn('judge_model_id="Qwen/Qwen2.5-72B-Instruct"', source)
        self.assertIn("--resume", source)
        self.assertIn('"scientific_result": False', source)


if __name__ == "__main__":
    unittest.main()
