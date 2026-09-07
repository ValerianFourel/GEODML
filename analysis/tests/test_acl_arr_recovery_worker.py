from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKER = (
    REPOSITORY_ROOT
    / "analysis/scripts/slurm/jupiter/recover_acl_arr_llama_rerank_4gpu.sh"
)


class RecoveryWorkerTests(unittest.TestCase):
    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            ACL_ARR_RUN_ROOT="/unused/original",
            ACL_ARR_VENV="/unused/venv",
            ACL_ARR_RECOVERY_ROOT="/unused/recovery",
            ACL_ARR_RECOVERY_JOB_ID="123",
            ACL_ARR_RECOVERY_APPROVED_WALLTIME="00:30:00",
            ACL_ARR_RECOVERY_ESTIMATE="Approved test estimate",
            GEODML_RECOVERY_COMMIT="a" * 40,
            SLURM_JOB_ID="123",
            SLURM_STEP_ID="0",
        )
        return environment

    def test_bash_syntax(self) -> None:
        result = subprocess.run(["bash", "-n", str(WORKER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_budget_stops_before_module_or_gpu_work(self) -> None:
        environment = self._environment()
        environment["ACL_ARR_RECOVERY_APPROVED_WALLTIME"] = "03:00:00"
        result = subprocess.run(
            ["bash", str(WORKER)], env=environment, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("approved only for 00:30:00", result.stderr)

    def test_wrong_allocation_stops_before_module_or_gpu_work(self) -> None:
        environment = self._environment()
        environment["SLURM_JOB_ID"] = "456"
        result = subprocess.run(
            ["bash", str(WORKER)], env=environment, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("specified existing allocation", result.stderr)

    def test_no_job_step_fails_without_closing_the_parent_shell(self) -> None:
        environment = self._environment()
        environment.pop("SLURM_STEP_ID")
        result = subprocess.run(
            ["bash", "-c", 'set +e; bash "$1"; printf "CHILD_STATUS=%s\\nPARENT_ALIVE\\n" "$?"',
             "_", str(WORKER)],
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CHILD_STATUS=2\nPARENT_ALIVE", result.stdout)

    def test_worker_only_cleans_its_private_server_group(self) -> None:
        source = WORKER.read_text()
        self.assertIn('setsid "$ACL_ARR_VENV/bin/vllm"', source)
        self.assertIn('kill -TERM -- "-$server_pid"', source)
        self.assertIn('kill -KILL -- "-$server_pid"', source)
        self.assertNotIn("scancel", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("killall", source)
        self.assertNotIn("salloc ", source)
        self.assertNotIn("sbatch ", source)
        self.assertNotIn("srun ", source)

    def test_cleanup_targets_only_launched_group_and_clears_ownership(self) -> None:
        source = WORKER.read_text()
        match = re.search(r"stop_server\(\) \{.*?\n\}", source, re.DOTALL)
        self.assertIsNotNone(match)
        shell = """
kill() { printf 'SIGNAL=%s\\n' "$*"; [[ "$1" != -0 ]]; }
sleep() { return 0; }
wait() { printf 'WAIT=%s\\n' "$*"; }
server_pid=98765
""" + match.group() + "\nstop_server\nprintf 'PID_AFTER=%s\\n' \"$server_pid\"\n"
        result = subprocess.run(["bash", "-c", shell], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIGNAL=-TERM -- -98765", result.stdout)
        self.assertIn("SIGNAL=-KILL -- -98765", result.stdout)
        self.assertIn("WAIT=98765", result.stdout)
        self.assertIn("PID_AFTER=\n", result.stdout)

    def test_embedded_python_preflights_compile(self) -> None:
        source = WORKER.read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            compile(block, str(WORKER), "exec")

    def test_grammar_preflight_constructs_initial_and_frozen_prefix_schemas(self) -> None:
        source = WORKER.read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
        grammar_blocks = [block for block in blocks if "RECOVERY_GRAMMAR=PASS" in block]
        self.assertEqual(len(grammar_blocks), 1)
        block = grammar_blocks[0]
        schemas = []

        def construct(schema):
            schemas.append(json.loads(schema))
            return object()

        module = SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=construct))
        with patch.dict(sys.modules, {"xgrammar": module}), redirect_stdout(io.StringIO()):
            exec(compile(block, str(WORKER), "exec"), {})
        self.assertEqual(len(schemas), 2)
        first = schemas[0]["properties"]["ranked_document_ids"]
        second = schemas[1]["properties"]["ranked_document_ids"]
        self.assertEqual(first["items"]["enum"], ["C001", "C002", "C003", "C004"])
        self.assertNotIn("prefixItems", first)
        self.assertEqual(second["prefixItems"], [{"const": "C002"}])
        self.assertEqual(second["items"]["enum"], ["C001", "C003", "C004"])
        self.assertEqual((second["minItems"], second["maxItems"]), (3, 3))
        self.assertLess(source.index("RECOVERY_GRAMMAR=PASS"), source.index('setsid "$ACL_ARR_VENV'))

    def test_grammar_preflight_does_not_hide_unsupported_schema_errors(self) -> None:
        source = WORKER.read_text()
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.DOTALL)
        grammar_blocks = [block for block in blocks if "RECOVERY_GRAMMAR=PASS" in block]
        self.assertEqual(len(grammar_blocks), 1)

        def unsupported(schema):
            raise RuntimeError("unsupported prefixItems")

        module = SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=unsupported))
        with patch.dict(sys.modules, {"xgrammar": module}):
            with self.assertRaisesRegex(RuntimeError, "unsupported prefixItems"):
                exec(compile(grammar_blocks[0], str(WORKER), "exec"), {})

    def test_preflight_and_module_order_precede_model_start(self) -> None:
        source = WORKER.read_text()
        self.assertLess(source.index("module load git"), source.index("$(git rev-parse"))
        self.assertLess(source.index("--preflight-only"), source.index('setsid "$ACL_ARR_VENV'))
        self.assertLess(source.index("parser.parse_args"), source.index('setsid "$ACL_ARR_VENV'))
        self.assertIn("--max-model-len 40960", source)
        self.assertIn("--max-concurrency 8", source)
        self.assertIn("--no-enable-log-requests", source)
        self.assertNotIn("--disable-log-requests", source)
        self.assertIn("--structured-outputs-config '{\"backend\":\"xgrammar\"}'", source)
        self.assertIn("--expected-plan-commit f9c35e499b708a4c812b49a3f550e097306ce25d", source)


if __name__ == "__main__":
    unittest.main()
