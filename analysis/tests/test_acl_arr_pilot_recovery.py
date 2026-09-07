"""A recovery must not mutate or rerun successful original pilot tasks."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts import recover_acl_arr_llama_rerank as recovery
from analysis.scripts.run_acl_arr_vllm import _prepare_primary, _sha256
from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    build_acl_arr_experiment_plan, write_acl_arr_experiment_plan,
)
from analysis.tests.test_acl_arr_document_experiment import (
    _prompts, _axis_rows, _document_sets, _models,
)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "original"
        self.model = replace(_models()[0], model_id=recovery.MODEL_ID,
                             model_revision=recovery.MODEL_REVISION)
        self.plan = build_acl_arr_experiment_plan(
            _prompts(), _axis_rows(), _document_sets(), models=(self.model,),
            top_n=2, source_git_commit="a" * 40,
        )
        self.artifacts = write_acl_arr_experiment_plan(self.run / "plan", plan=self.plan)
        self.shard = self.run / "results" / self.model.configuration_id / "rerank"
        self.shard.mkdir(parents=True)
        tasks = [json.loads(s) for s in self.artifacts.task_files[
            (self.model.configuration_id, "rerank")].read_text().splitlines()]
        from analysis.scripts.run_acl_arr_vllm import _primary_context
        _, task_objects, _, _ = _primary_context(self.artifacts.manifest_path,
            self.artifacts.task_files[(self.model.configuration_id, "rerank")])
        task = task_objects[0]
        assignment = next(a for a in self.plan.assignments if a.assignment_id == task.assignment_id)
        item = _prepare_primary(task,
            prompt=next(p for p in self.plan.prompts if p.prompt_id == task.prompt_id),
            assignment=assignment,
            document_set=next(d for d in self.plan.document_sets if d.candidate_set_id == assignment.candidate_set_id),
            model=self.model)
        outcome = {**item["base"], "raw_output": item["fake_output"],
                   "raw_output_sha256": __import__("hashlib").sha256(item["fake_output"].encode()).hexdigest(),
                   "parsed_output": item["validator"](item["fake_output"]), "usage": {}}
        (self.shard / "outcomes.jsonl").write_text(json.dumps(outcome) + "\n")
        (self.shard / "failures.jsonl").write_text("".join(json.dumps({"task_id": t["task_id"], "error": "fixture"}) + "\n" for t in tasks[1:]))
        manifest = {"model_id": self.model.model_id, "model_revision": self.model.model_revision,
                    "pipeline": "rerank", "fake_backend": False,
                    "tasks": {"sha256": _sha256(self.artifacts.task_files[(self.model.configuration_id, "rerank")])},
                    "source_manifest": str(self.artifacts.manifest_path.resolve()),
                    "outcomes_sha256": _sha256(self.shard / "outcomes.jsonl"),
                    "failures_sha256": _sha256(self.shard / "failures.jsonl")}
        (self.shard / "run_manifest.json").write_text(json.dumps(manifest))
        (self.run / "pilot-runtime-manifest.json").write_text(json.dumps({
            "scientific_result": False, "git_commit_sha": "a" * 40}))
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (self.run / "model-snapshots.json").write_text(json.dumps({"models": [{
            "model_id": self.model.model_id, "revision": self.model.model_revision,
            "snapshot": str(snapshot)}]}))

    def source(self):
        return recovery.load_source(self.run, "a" * 40, expected_counts=(6, 1, 5))

    def test_preflight_counts_and_never_writes(self):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        source = self.source()
        self.assertEqual(len(source["pending"]), 5)
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()})

    def test_fake_recovery_preserves_original_bytes_and_global_completion(self):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        source = self.source()
        out = self.root / "recovery"
        status = asyncio.run(recovery.run_recovery(source, out, client=None, fake=True,
            recovery_commit="b" * 40, max_concurrency=2, max_model_len=40960,
            approved_walltime="00:30:00", allocation_estimate="fixture"))
        self.assertEqual(status, 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()})
        rows = [json.loads(s) for s in (out / "outcomes.jsonl").read_text().splitlines()]
        self.assertEqual(len({r["task_id"] for r in rows}), 6)
        self.assertTrue((out / "outcomes.jsonl").read_bytes().startswith((self.shard / "outcomes.jsonl").read_bytes()))
        report = json.loads((out / "recovery_manifest.json").read_text())
        self.assertFalse(report["eligible_for_analysis"])
        self.assertFalse(report["scientific_result"])
        self.assertEqual(report["completed_count"], 6)
        self.assertEqual(report["remaining_count"], 0)
        self.assertEqual(report["original_success_count"], 1)
        self.assertEqual(report["plan_source_git_commit"], "a" * 40)
        self.assertEqual(report["recovery_git_commit"], "b" * 40)
        with self.assertRaises(FileExistsError):
            asyncio.run(recovery.run_recovery(source, out, client=None, fake=True,
                recovery_commit="b" * 40, max_concurrency=2, max_model_len=40960,
                approved_walltime="00:30:00", allocation_estimate="fixture"))

    def test_changed_outcome_hash_rejected(self):
        with (self.shard / "outcomes.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.source()

    def test_nonpilot_runtime_rejected(self):
        path = self.run / "pilot-runtime-manifest.json"
        record = json.loads(path.read_text())
        record["scientific_result"] = True
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "must be a pilot"):
            self.source()

    def test_unexpected_success_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts"):
            recovery.load_source(self.run, "a" * 40)

    def test_wrong_plan_commit_rejected(self):
        with self.assertRaisesRegex(ValueError, "commit"):
            recovery.load_source(self.run, "c" * 40, expected_counts=(6, 1, 5))

    def test_real_decode_path_calls_only_pending_tasks_and_audits_every_request(self):
        class Client:
            maximum_attempts = 1
            calls = []

            async def complete(self, **kwargs):
                self.calls.append(kwargs)
                array = kwargs["schema"]["properties"]["ranked_document_ids"]
                ids = array["items"]["enum"][:array["minItems"]]
                return json.dumps({"ranked_document_ids": ids}), {"prompt_tokens": 123}

        client = Client()
        output = self.root / "recovery-real-client-fixture"
        source = self.source()
        status = asyncio.run(recovery.run_recovery(source, output, client=client, fake=False,
            recovery_commit="b" * 40, max_concurrency=2, max_model_len=40960,
            approved_walltime="00:30:00", allocation_estimate="fixture"))
        self.assertEqual(status, 0)
        self.assertEqual(len(client.calls), 5)
        attempts = [json.loads(s) for s in (output / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual({r["task_id"] for r in attempts}, set(source["pending"]))
        self.assertTrue(all(r["protocol"] == recovery.PROTOCOL == r["decoding_protocol"] for r in attempts))
        self.assertTrue(all("seed" in r["request"] for r in attempts))

    def test_failed_recovery_remains_incomplete_and_preserves_attempt_errors(self):
        class Client:
            maximum_attempts = 1

            async def complete(self, **kwargs):
                raise RuntimeError("HTTP 400 fixture")

        output = self.root / "recovery-failed-client-fixture"
        status = asyncio.run(recovery.run_recovery(self.source(), output, client=Client(), fake=False,
            recovery_commit="b" * 40, max_concurrency=2, max_model_len=40960,
            approved_walltime="00:30:00", allocation_estimate="fixture"))
        self.assertEqual(status, 2)
        report = json.loads((output / "recovery_manifest.json").read_text())
        self.assertEqual(report["status"], "complete_with_failures")
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(report["remaining_count"], 5)
        self.assertFalse(report["eligible_for_analysis"])
        attempts = [json.loads(s) for s in (output / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual(len(attempts), 5)
        self.assertTrue(all("HTTP 400" in r["validator_error"] for r in attempts))


if __name__ == "__main__":
    unittest.main()
