"""Read-only context budgeting against the exact frozen primary requests."""

from __future__ import annotations

from collections import UserDict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    build_acl_arr_experiment_plan,
    render_primary_prompt,
    write_acl_arr_experiment_plan,
)
from analysis.scripts.check_acl_arr_context_budget import check_context_budget, main
from analysis.scripts import check_acl_arr_context_budget as context_budget
from analysis.scripts.run_acl_arr_vllm import _primary_context
from analysis.tests.test_acl_arr_document_experiment import (
    _axis_rows,
    _document_sets,
    _models,
    _prompts,
)


class RecordingTokenizer:
    def __init__(self, lengths=(1000, 1900, 1300, 1400, 1500, 1600)):
        self.lengths = lengths
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        length = self.lengths[len(self.calls)]
        self.calls.append((messages, kwargs))
        return [1] * length


class AclArrContextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model = _models()[0]
        plan = build_acl_arr_experiment_plan(
            _prompts(), _axis_rows(), _document_sets(),
            models=(self.model,), top_n=2, source_git_commit="0" * 40,
        )
        self.artifacts = write_acl_arr_experiment_plan(self.root / "plan", plan=plan)
        self.manifest = self.artifacts.manifest_path
        self.tasks = self.artifacts.task_files[(self.model.configuration_id, "rerank")]
        self.snapshot = self.root / "snapshots" / self.model.model_revision
        self.snapshot.mkdir(parents=True)
        self.config = self.snapshot / "config.json"
        self.config.write_text(json.dumps({"text_config": {"max_position_embeddings": 8192}}))
        self.locks = self.root / "model-snapshots.json"
        self.locks.write_text(json.dumps({"models": [{
            "model_id": self.model.model_id,
            "revision": self.model.model_revision,
            "snapshot": str(self.snapshot),
        }]}))
        self.tokenizer = RecordingTokenizer()
        self.load_calls = []

    def load(self, path, **kwargs):
        self.load_calls.append((path, kwargs))
        return self.tokenizer

    def check(self, tasks=None):
        return check_context_budget(
            self.manifest, tasks or self.tasks, self.locks,
            tokenizer_loader=self.load, tokenizer_version="fixture-transformers",
        )

    def test_counts_exact_rendered_chat_and_output_budget_without_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = self.check()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["max_prompt_tokens"], 1900)
        self.assertEqual(report["max_required_tokens"], 1900 + self.model.rerank_max_tokens)
        self.assertEqual(report["suggested_max_model_len"], 2048)
        self.assertEqual(report["task_count"], 6)
        self.assertEqual(report["tokenizer_version"], "fixture-transformers")
        self.assertEqual(report["model_id"], self.model.model_id)
        self.assertEqual(report["model_revision"], self.model.model_revision)
        self.assertEqual(report["snapshot"], str(self.snapshot))
        self.assertEqual(report["plan_manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(report["tasks_sha256"], hashlib.sha256(self.tasks.read_bytes()).hexdigest())
        self.assertEqual(self.load_calls, [(str(self.snapshot), {
            "local_files_only": True, "trust_remote_code": True,
        })])
        plan, tasks, _, _ = _primary_context(self.manifest, self.tasks)
        self.assertEqual(report["worst_task_id"], tasks[1].task_id)
        prompts = {p.prompt_id: p for p in plan.prompts}
        assignments = {a.assignment_id: a for a in plan.assignments}
        documents = {d.candidate_set_id: d for d in plan.document_sets}
        for task, (messages, kwargs) in zip(tasks, self.tokenizer.calls):
            assignment = assignments[task.assignment_id]
            expected = render_primary_prompt(
                task, prompts[task.prompt_id], assignment,
                documents[assignment.candidate_set_id],
            )
            self.assertEqual(messages, [{"role": "user", "content": expected}])
            self.assertEqual(kwargs, {
                "tokenize": True, "add_generation_prompt": True, "truncation": False,
                "return_dict": True,
            })
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_answer_uses_unchanged_answer_output_limit(self):
        tasks = self.artifacts.task_files[(self.model.configuration_id, "answer")]
        report = self.check(tasks)
        self.assertEqual(report["max_required_tokens"], 1900 + self.model.answer_max_tokens)
        self.assertEqual(report["suggested_max_model_len"], 3072)

    def test_mapping_output_counts_token_ids_instead_of_two_fields(self):
        class MappingTokenizer(RecordingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                return UserDict(input_ids=ids, attention_mask=[1] * len(ids))

        self.tokenizer = MappingTokenizer()
        report = self.check()
        self.assertEqual(report["max_prompt_tokens"], 1900)
        self.assertEqual(report["max_required_tokens"], 2028)

    def test_single_batch_mapping_counts_tokens_instead_of_batch_size(self):
        class BatchedTokenizer(RecordingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                return UserDict(input_ids=[ids], attention_mask=[[1] * len(ids)])

        self.tokenizer = BatchedTokenizer()
        self.assertEqual(self.check()["max_prompt_tokens"], 1900)

    def test_malformed_or_multiple_token_sequences_are_rejected(self):
        for value in ({"attention_mask": [1]}, {"input_ids": [[1], [2]]},
                      {"input_ids": []}, {"input_ids": [True]},
                      {"input_ids": ["input_ids", "attention_mask"]}):
            with self.subTest(value=value):
                self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: value)
                with self.assertRaisesRegex(ValueError, "input_ids"):
                    self.check()

    def test_counter_preserves_flat_sequence_compatibility(self):
        self.assertEqual(context_budget._input_token_count([3, 1, 9]), 3)
        self.assertEqual(context_budget._input_token_count((3, 1, 9)), 3)
        encoded = UserDict(input_ids=[1] * 32600, attention_mask=[1] * 32600)
        self.assertEqual(context_budget._input_token_count(encoded) + 256, 32856)

    def test_snapshot_identity_must_match_immutable_model_revision(self):
        payload = json.loads(self.locks.read_text())
        payload["models"][0]["revision"] = "f" * 40
        self.locks.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "exactly one snapshot"):
            self.check()
        self.assertEqual(self.load_calls, [])

    def test_duplicate_snapshot_identity_is_rejected(self):
        payload = json.loads(self.locks.read_text())
        payload["models"].append(dict(payload["models"][0]))
        self.locks.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "exactly one snapshot"):
            self.check()

    def test_task_hash_is_verified_before_tokenizer_loading(self):
        self.tasks.write_text(self.tasks.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.check()
        self.assertEqual(self.load_calls, [])

    def test_native_text_context_takes_precedence_and_cannot_be_exceeded(self):
        self.config.write_text(json.dumps({
            "max_position_embeddings": 99999,
            "text_config": {"max_position_embeddings": 2000},
        }))
        report = self.check()
        self.assertEqual(report["native_context_tokens"], 2000)
        self.assertEqual(report["native_context_source"], "text_config.max_position_embeddings")
        self.assertEqual(report["status"], "EXCEEDS_NATIVE_CONTEXT")
        self.assertIsNone(report["suggested_max_model_len"])

    def test_unknown_native_context_is_reported_without_unsafe_suggestion(self):
        self.config.write_text("{}")
        report = self.check()
        self.assertEqual(report["status"], "UNKNOWN_NATIVE_CONTEXT")
        self.assertIsNone(report["suggested_max_model_len"])

    def test_cli_loads_only_the_local_tokenizer_and_reports_json(self):
        output = io.StringIO()
        auto_tokenizer = SimpleNamespace(from_pretrained=self.load)
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=auto_tokenizer)}):
            with patch("analysis.scripts.check_acl_arr_context_budget.version", return_value="fixture"):
                with redirect_stdout(output):
                    status = main([
                        "--plan-manifest", str(self.manifest),
                        "--tasks", str(self.tasks),
                        "--model-snapshots", str(self.locks),
                    ])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["max_required_tokens"], 2028)

    def test_mistral_native_serving_is_not_misreported_as_hf_tokenization(self):
        with patch("analysis.scripts.check_acl_arr_context_budget._primary_context") as context:
            context.return_value = (None, [], SimpleNamespace(model_id="mistralai/Mistral-Small-4-119B-2603"), "rerank")
            with self.assertRaisesRegex(ValueError, "native Mistral tokenizer"):
                self.check()
        self.assertEqual(self.load_calls, [])


if __name__ == "__main__":
    unittest.main()
