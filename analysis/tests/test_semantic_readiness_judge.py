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
    _validate_skipped_task_ids,
    _validate_run_contract,
)
from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    ReadinessLabelTask,
)


class SemanticReadinessJudgeTests(unittest.TestCase):
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
