from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = (
    REPOSITORY_ROOT
    / "analysis/scripts/slurm/jupiter/run_acl_arr_document_pilot_4gpu.sh"
)


class AclArrPilotControllerTests(unittest.TestCase):
    def test_serve_disables_request_logging_with_current_boolean_option(self) -> None:
        source = CONTROLLER.read_text(encoding="utf-8")
        command = source.split('    vllm serve "$model_id"', 1)[1].split(
            '    server_pid=$!', 1
        )[0]
        logging_options = [
            token for token in shlex.split(command.replace("\\\n", " "))
            if token.startswith("--") and "log-requests" in token
        ]
        self.assertEqual(len(logging_options), 1)
        # vLLM 0.28 exposes the positive boolean and its --no- negative form.
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--enable-log-requests", action=argparse.BooleanOptionalAction, default=True
        )
        args = parser.parse_args(logging_options)
        self.assertFalse(args.enable_log_requests)

    def _run_bootstrap(self, *, module_status: int = 0, provide_git: bool = True):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            empty_bin = root / "empty-bin"
            git_bin = root / "git-bin"
            empty_bin.mkdir()
            git_bin.mkdir()
            trace = root / "trace"
            git = git_bin / "git"
            git.write_text(
                '#!/bin/bash\nprintf "GIT %s\\n" "$*" >> "$TEST_TRACE"\nexit 73\n'
            )
            git.chmod(0o755)
            bootstrap = root / "bash-env"
            bootstrap.write_text(
                'module() {\n'
                '  printf "MODULE %s\\n" "$*" >> "$TEST_TRACE"\n'
                '  if [[ "$TEST_MODULE_STATUS" != 0 ]]; then\n'
                '    return "$TEST_MODULE_STATUS"\n'
                '  fi\n'
                '  if [[ "$*" == "load git" && "$TEST_PROVIDE_GIT" == 1 ]]; then\n'
                '    export PATH="$TEST_GIT_BIN:$PATH"\n'
                '  fi\n'
                '  return 0\n'
                '}\n'
            )
            environment = os.environ.copy()
            environment.update(
                PATH=str(empty_bin),
                BASH_ENV=str(bootstrap),
                TEST_TRACE=str(trace),
                TEST_GIT_BIN=str(git_bin),
                TEST_MODULE_STATUS=str(module_status),
                TEST_PROVIDE_GIT="1" if provide_git else "0",
                ACL_ARR_RUN_ROOT=str(root / "unused-run"),
                ACL_ARR_VENV=str(root / "unused-venv"),
                ACL_ARR_APPROVED_WALLTIME="03:00:00",
                ACL_ARR_ALLOCATION_ESTIMATE="local bootstrap fixture; no allocation",
                SLURM_JOB_ID="local-fixture",
            )
            result = subprocess.run(
                ["/bin/bash", str(CONTROLLER)],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            events = trace.read_text().splitlines() if trace.exists() else []
            self.assertFalse((root / "unused-run").exists())
            return result, events

    def test_bootstrap_loads_git_before_repository_detection(self) -> None:
        result, events = self._run_bootstrap()
        self.assertEqual(result.returncode, 73, result.stderr)
        self.assertEqual(events, [
            "MODULE load Stages/2026 GCC Python CUDA",
            "MODULE load git",
            "GIT rev-parse --show-toplevel",
        ])

    def test_bootstrap_does_not_hide_module_failure(self) -> None:
        result, events = self._run_bootstrap(module_status=42)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(events, ["MODULE load Stages/2026 GCC Python CUDA"])

    def test_bootstrap_reports_git_still_missing_after_module_load(self) -> None:
        result, events = self._run_bootstrap(provide_git=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("git is unavailable after loading the git module", result.stderr)
        self.assertEqual(events, [
            "MODULE load Stages/2026 GCC Python CUDA",
            "MODULE load git",
        ])

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
