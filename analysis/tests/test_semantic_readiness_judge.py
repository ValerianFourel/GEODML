"""Contracts for resumable semantic-readiness judge retries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from analysis.scripts.run_semantic_readiness_judge import (
    _load_rejected_attempts,
    _render_retry_prompt,
    _resume_attempt_limit,
    _run_local_batches,
    _validate_skipped_task_ids,
    _validate_run_contract,
)
from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    ABSTENTION_LABEL_RUBRIC_VERSION,
    ReadinessLabelTask,
)


class SemanticReadinessJudgeTests(unittest.TestCase):
    @staticmethod
    def _valid_response(overall: int = 50) -> str:
        return json.dumps(
            {
                "overall_readiness_0_100": overall,
                "information_seeking_1_7": 4,
                "evaluation_1_7": 4,
                "selection_commitment_1_7": 4,
                "action_implementation_1_7": 4,
                "category": "mixed",
                "not_applicable": False,
                "ambiguity_1_7": 2,
                "confidence_0_1": 0.8,
                "brief_reason": "The request mixes information and action.",
            }
        )

    def test_production_run_requires_model_revision(self) -> None:
        args = SimpleNamespace(
            start_index=0,
            limit=None,
            max_new_tokens=300,
            maximum_attempts=3,
            run_purpose="production",
            model_family="qwen",
            model_revision=None,
        )
        with self.assertRaisesRegex(SystemExit, "model-revision"):
            _validate_run_contract(args)

        args.model_revision = "a" * 40
        _validate_run_contract(args)

        args.model_family = None
        with self.assertRaisesRegex(SystemExit, "model-family"):
            _validate_run_contract(args)

    def test_debug_run_retains_backwards_compatible_optional_revision(self) -> None:
        _validate_run_contract(
            SimpleNamespace(
                start_index=0,
                limit=5,
                max_new_tokens=300,
                maximum_attempts=3,
                run_purpose="debug",
                model_family=None,
                model_revision=None,
            )
        )

    def test_nonlocal_batching_is_rejected(self) -> None:
        args = SimpleNamespace(
            start_index=0,
            limit=None,
            max_new_tokens=300,
            maximum_attempts=3,
            batch_size=2,
            backend="api",
            run_purpose="debug",
            model_family=None,
            model_revision=None,
        )
        with self.assertRaisesRegex(SystemExit, "backend local"):
            _validate_run_contract(args)

    def test_local_batches_retry_only_invalid_rows_and_restore_task_order(self) -> None:
        tasks = [
            ReadinessLabelTask(
                task_id=f"task:{index}",
                item_id=f"item:{index}",
                judge_slot="judge-a",
                presentation_variant="forward-anchors",
                rubric_version="test",
                prompt=f"PROMPT {index}",
            )
            for index in range(3)
        ]

        class FakeRanker:
            calls = 0

            def rank_batch(self, prompts, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return [SemanticReadinessJudgeTests._valid_response(10), "{}"]
                if self.calls == 2:
                    return [
                        SemanticReadinessJudgeTests._valid_response(30),
                        SemanticReadinessJudgeTests._valid_response(20),
                    ]
                raise AssertionError("unexpected extra generation batch")

        args = SimpleNamespace(
            batch_size=2,
            max_new_tokens=300,
            maximum_attempts=3,
            resume=False,
            model="llama",
            model_family="llama",
            model_revision="revision",
            backend="local",
            precision="full",
            judge_slot="judge-a",
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            rows = _run_local_batches(
                FakeRanker(),
                tasks,
                cache=cache,
                skipped_task_ids=frozenset(),
                args=args,
            )

            self.assertEqual([row["task_id"] for row in rows], [task.task_id for task in tasks])
            retried = json.loads((cache / "task_1.json").read_text(encoding="utf-8"))
            self.assertEqual(len(retried["rejected_attempts"]), 1)

    def test_retry_prompt_repeats_the_frozen_contract(self) -> None:
        prompt = _render_retry_prompt(
            "ORIGINAL PROMPT",
            "unknown readiness category",
            '{"category": "evaluation"}',
        )

        self.assertIn("ORIGINAL PROMPT", prompt)
        self.assertIn("unknown readiness category", prompt)
        self.assertIn('{"category": "evaluation"}', prompt)
        self.assertIn("<previous_invalid_response>", prompt)
        self.assertIn('"information_seeking_1_7"', prompt)
        self.assertIn('"selection_commitment_1_7"', prompt)
        self.assertIn('"brief_reason": <1 to 20 words>', prompt)
        self.assertIn('"information"|"criteria"|"comparison"', prompt)
        self.assertIn("Do not use category values such as evaluation or review", prompt)

    def test_retry_prompt_makes_observed_failure_rules_explicit(self) -> None:
        prompt = _render_retry_prompt(
            "ORIGINAL PROMPT",
            "category and not_applicable disagree",
            '{"category": "information", "not_applicable": true}',
        )

        self.assertIn("at most 20 whitespace-separated words", prompt)
        self.assertIn("silently\nre-read the original text", prompt)
        self.assertIn('category is "not_applicable" and not_applicable is true', prompt)
        self.assertIn('or "mixed", and not_applicable is false', prompt)
        self.assertIn("invalid pair from the previous response must not be repeated", prompt)
        self.assertIn("preserve every other semantic judgment", prompt)

    def test_retry_prompt_preserves_abstention_v2_contract(self) -> None:
        prompt = _render_retry_prompt(
            "ORIGINAL V2 PROMPT",
            "scores must be null",
            '{"answer_type":"dont_know","overall_readiness_0_100":50}',
            rubric_version=ABSTENTION_LABEL_RUBRIC_VERSION,
        )

        self.assertIn('"answer_type": <"rating"|"not_applicable"|"dont_know">', prompt)
        self.assertIn("all five scores and category must be null", prompt)
        self.assertIn("not_applicable means the construct is irrelevant", prompt)
        self.assertIn('The value "evaluation" is never a valid category', prompt)
        self.assertIn("never schema placeholders", prompt)
        self.assertNotIn('"not_applicable": <true|false>', prompt)

    def test_resume_attempt_limit_grants_only_failed_caches_extra_attempts(self) -> None:
        args = SimpleNamespace(
            maximum_attempts=5,
            resume=True,
            resume_extra_attempts=3,
        )

        self.assertEqual(_resume_attempt_limit(args, []), 5)
        self.assertEqual(_resume_attempt_limit(args, [{"attempt": 1}]), 5)
        self.assertEqual(
            _resume_attempt_limit(args, [{"attempt": index} for index in range(5)]),
            8,
        )

    def test_resume_repairs_only_exhausted_failed_cache(self) -> None:
        tasks = [
            ReadinessLabelTask(
                task_id=f"task:{index}",
                item_id=f"item:{index}",
                judge_slot="judge-a",
                presentation_variant="forward-anchors",
                rubric_version="test",
                prompt=f"PROMPT {index}",
            )
            for index in range(2)
        ]

        class FakeRanker:
            prompts = []

            def rank_batch(self, prompts, **kwargs):
                self.prompts.extend(prompts)
                return [SemanticReadinessJudgeTests._valid_response(70)]

        args = SimpleNamespace(
            batch_size=2,
            max_new_tokens=300,
            maximum_attempts=5,
            resume_extra_attempts=3,
            resume=True,
            model="qwen",
            model_family="qwen",
            model_revision="revision",
            backend="local",
            precision="full",
            judge_slot="judge-a",
        )
        rejected = [
            {"attempt": index, "error": "unknown readiness category", "raw": "{}"}
            for index in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            existing = {
                "task_id": tasks[0].task_id,
                "item_id": tasks[0].item_id,
                "judge_slot": tasks[0].judge_slot,
                "model": args.model,
                "model_family": args.model_family,
                "model_revision": args.model_revision,
                "backend": args.backend,
                "precision": args.precision,
                "raw_response": self._valid_response(30),
                "rejected_attempts": [],
            }
            (cache / "task_0.json").write_text(json.dumps(existing), encoding="utf-8")
            (cache / "task_1.failed.json").write_text(
                json.dumps({"attempts": rejected}),
                encoding="utf-8",
            )
            ranker = FakeRanker()

            rows = _run_local_batches(
                ranker,
                tasks,
                cache=cache,
                skipped_task_ids=frozenset(),
                args=args,
            )

            self.assertEqual([row["task_id"] for row in rows], [task.task_id for task in tasks])
            self.assertEqual(len(ranker.prompts), 1)
            repaired = json.loads((cache / "task_1.json").read_text(encoding="utf-8"))
            self.assertEqual(repaired["rejected_attempts"], rejected)
            self.assertFalse((cache / "task_1.failed.json").exists())

    def test_failed_attempts_are_reused_on_resume(self) -> None:
        attempts = [
            {
                "attempt": 1,
                "error": "brief_reason must contain 1 to 25 words",
                "raw": "{}",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.failed.json"
            path.write_text(json.dumps({"attempts": attempts}), encoding="utf-8")

            self.assertEqual(_load_rejected_attempts(path), attempts)

    def test_missing_failed_cache_has_no_prior_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.failed.json"
            self.assertEqual(_load_rejected_attempts(path), [])

    def test_only_explicit_tasks_in_the_selected_slice_may_be_skipped(self) -> None:
        task = ReadinessLabelTask(
            task_id="task:known",
            item_id="item:1",
            judge_slot="judge-a",
            presentation_variant="forward-anchors",
            rubric_version="test",
            prompt="PROMPT",
        )

        self.assertEqual(
            _validate_skipped_task_ids([task], ["task:known"]),
            frozenset({"task:known"}),
        )
        with self.assertRaisesRegex(SystemExit, "outside the selected judge slice"):
            _validate_skipped_task_ids([task], ["task:unknown"])


if __name__ == "__main__":
    unittest.main()
