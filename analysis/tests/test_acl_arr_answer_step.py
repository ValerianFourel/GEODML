"""Budgeted answer execution checkpoints without changing its source artifacts."""
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import ast
import os
import subprocess

from analysis.scripts import run_acl_arr_pilot_answers as answers
from analysis.scripts.run_acl_arr_vllm import _answer_schema
from analysis.interpretability.pipeline.acl_arr_document_experiment import validate_answer_output


class AnswerStepTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.json"
        self.source.write_text("original data\n")
        self.hashes = {str(self.source): hashlib.sha256(self.source.read_bytes()).hexdigest()}
        self.items = [{"base": {"task_id": f"task-{i}", "pipeline": "answer", "fake_backend": False},
                       "prompt": f"original prompt {i}", "seed": i, "temperature": 0.0,
                       "max_tokens": 768, "schema": _answer_schema(), "schema_name": "acl_arr_answer",
                       "validator": lambda raw: validate_answer_output(raw, allowed_document_ids=["C001"])}
                      for i in range(3)]

    def run_step(self, client, output, **kwargs):
        return asyncio.run(answers.run_answers(self.items, output, client=client,
            source_hashes=self.hashes, metadata={"plan_source_git_commit": "a" * 40,
            "execution_git_commit": "b" * 40}, stop_at=10, clock=lambda: 0,
            max_concurrency=1, **kwargs))

    def test_valid_answers_checkpoint_and_keep_original_request(self):
        class Client:
            calls = []
            async def complete(self, **kwargs):
                self.calls.append(kwargs)
                return '{"answer":"Evidence [C001].","cited_document_ids":["C001"]}', {"completion_tokens": 20}
        client = Client()
        output = self.root / "answer"
        self.assertEqual(self.run_step(client, output), 0)
        manifest = json.loads((output / "answer_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_count"], 3)
        self.assertFalse(manifest["eligible_for_analysis"])
        self.assertFalse(manifest["scientific_result"])
        self.assertEqual(self.source.read_text(), "original data\n")
        for item, call in zip(self.items, client.calls):
            for key in ("prompt", "seed", "temperature", "max_tokens", "schema", "schema_name"):
                self.assertEqual(call[key], item[key])

    def test_deadline_stops_submission_and_does_not_claim_completion(self):
        now = [0]
        class Client:
            async def complete(self, **kwargs):
                now[0] = 11
                return '{"answer":"Evidence [C001].","cited_document_ids":["C001"]}', {}
        output = self.root / "partial"
        code = asyncio.run(answers.run_answers(self.items, output, client=Client(),
            source_hashes=self.hashes, metadata={}, stop_at=10, clock=lambda: now[0], max_concurrency=1))
        self.assertEqual(code, 3)
        manifest = json.loads((output / "answer_manifest.json").read_text())
        self.assertEqual(manifest["status"], "checkpointed")
        self.assertEqual(manifest["completed_count"], 1)
        self.assertEqual(manifest["unattempted_count"], 2)

    def test_invalid_answer_retains_raw_response_without_repair(self):
        class Client:
            async def complete(self, **kwargs):
                return '{"answer":"No citation","cited_document_ids":[]}', {}
        output = self.root / "invalid"
        self.assertEqual(self.run_step(Client(), output), 2)
        failures = [json.loads(s) for s in (output / "failures.jsonl").read_text().splitlines()]
        self.assertEqual(len(failures), 3)
        self.assertIn("No citation", failures[0]["raw_output"])
        self.assertEqual(len((output / "attempts.jsonl").read_text().splitlines()), 3)

    def test_resume_copies_successes_and_does_not_repeat_requests(self):
        class Client:
            calls = 0
            async def complete(self, **kwargs):
                self.calls += 1
                return '{"answer":"Evidence [C001].","cited_document_ids":["C001"]}', {}
        first = self.root / "first"
        self.run_step(Client(), first)
        second = self.root / "second"
        client = Client()
        self.assertEqual(self.run_step(client, second, resume_from=first), 0)
        self.assertEqual(client.calls, 0)
        self.assertEqual((first / "outcomes.jsonl").read_bytes(), (second / "outcomes.jsonl").read_bytes())
        with self.assertRaises(FileExistsError):
            self.run_step(client, second)

    def test_modified_source_rejected_before_output_creation(self):
        self.source.write_text("changed")
        output = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_step(None, output)
        self.assertFalse(output.exists())

    def test_resume_from_partial_runs_only_remaining_tasks(self):
        now = [0]
        class Client:
            calls = 0
            async def complete(self, **kwargs):
                self.calls += 1
                now[0] = 11
                return '{"answer":"Evidence [C001].","cited_document_ids":["C001"]}', {}
        first = self.root / "partial-first"
        asyncio.run(answers.run_answers(self.items, first, client=Client(),
            source_hashes=self.hashes, metadata={"plan_source_git_commit": "a" * 40,
            "execution_git_commit": "b" * 40}, stop_at=10, clock=lambda: now[0], max_concurrency=1))
        client = Client()
        self.assertEqual(self.run_step(client, self.root / "resumed", resume_from=first), 0)
        self.assertEqual(client.calls, 2)

    def test_source_loading_checks_recovery_and_renders_answer_pipeline(self):
        from analysis.tests.test_acl_arr_pilot_recovery import RecoveryTests
        from analysis.scripts.recover_acl_arr_llama_rerank import run_recovery
        fixture = RecoveryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source = fixture.source()
        recovered = self.root / "recovered-reranks"
        asyncio.run(run_recovery(source, recovered, client=None, fake=True,
            recovery_commit="b" * 40, max_concurrency=2, max_model_len=40960,
            approved_walltime="00:30:00", allocation_estimate="fixture"))
        with patch.object(answers, "load_source", return_value=source):
            with self.assertRaisesRegex(ValueError, "recovery does not match"):
                answers.load_answers(fixture.run, recovered, expected_count=6)
            # Simulate real-client provenance for the remaining loader checks.
            outcome_path = recovered / "outcomes.jsonl"
            rows = [json.loads(s) for s in outcome_path.read_text().splitlines()]
            for row in rows:
                row.update(source["prepared"][row["task_id"]]["base"])
            outcome_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest_path = recovered / "recovery_manifest.json"
            record = json.loads(manifest_path.read_text())
            record.update(fake_backend=False, outcomes_sha256=hashlib.sha256(outcome_path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(record))
            items, hashes, metadata = answers.load_answers(fixture.run, recovered, expected_count=6)
            self.assertEqual(len(items), 6)
            self.assertTrue(all(i["base"]["pipeline"] == "answer" for i in items))
            self.assertTrue(all(i["max_tokens"] == fixture.model.answer_max_tokens for i in items))
            self.assertIn(str((recovered / "outcomes.jsonl").resolve()), hashes)
            (recovered / "outcomes.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "recovery does not match"):
                answers.load_answers(fixture.run, recovered, expected_count=6)


class AnswerWorkerTests(unittest.TestCase):
    path = Path(__file__).resolve().parents[1] / "scripts/slurm/jupiter/run_acl_arr_llama_answer_4gpu.sh"

    def test_syntax_embedded_python_and_budget_starts_before_loading(self):
        code = self.path.read_text()
        subprocess.run(["bash", "-n", str(self.path)], check=True)
        for part in code.split("<<'PY'\n")[1:]:
            ast.parse(part.split("\nPY\n", 1)[0])
        self.assertLess(code.index("step_start_epoch="), code.index("module load"))
        self.assertIn("step_start_epoch + 3000", code)
        self.assertIn("--max-model-len 41984", code)
        self.assertIn("01:00:00", code)
        self.assertNotIn("scancel", code)
        self.assertNotIn("salloc ", code)

    def test_wrong_job_failure_does_not_exit_parent_shell(self):
        env = dict(os.environ)
        env.update({k: "fixture" for k in ("ACL_ARR_RUN_ROOT", "ACL_ARR_RERANK_RECOVERY_RESULTS",
            "ACL_ARR_VENV", "ACL_ARR_ANSWER_ROOT", "ACL_ARR_ANSWER_ESTIMATE", "GEODML_ANSWER_COMMIT")})
        env.update(ACL_ARR_ANSWER_JOB_ID="123", SLURM_JOB_ID="456", SLURM_STEP_ID="1",
                   ACL_ARR_ANSWER_APPROVED_WALLTIME="01:00:00")
        result = subprocess.run(["bash", "-c", 'set +e; bash "$1"; printf "PARENT_ALIVE status=%s\\n" "$?"',
                                 "check", str(self.path)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("PARENT_ALIVE status=2", result.stdout)

    def test_cleanup_signals_only_its_private_server_group(self):
        code = self.path.read_text().split("stop_server() {", 1)[1].split("\ntrap stop_server EXIT", 1)[0]
        script = 'kill() { printf "SIGNAL %s\\n" "$*"; return 1; }; wait() { :; }; server_pid=98765; '
        script += "stop_server() {" + code + '\nstop_server\nprintf "PID=%s\\n" "$server_pid"\n'
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        self.assertIn("SIGNAL -TERM -- -98765", result.stdout)
        self.assertIn("SIGNAL -KILL -- -98765", result.stdout)
        self.assertIn("PID=\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
