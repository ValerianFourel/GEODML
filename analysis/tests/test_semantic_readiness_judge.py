"""Contracts for resumable semantic-readiness judge retries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.scripts.run_semantic_readiness_judge import (
    _load_rejected_attempts,
    _render_retry_prompt,
)


class SemanticReadinessJudgeTests(unittest.TestCase):
    def test_retry_prompt_repeats_the_frozen_contract(self) -> None:
        prompt = _render_retry_prompt(
            "ORIGINAL PROMPT",
            "unknown readiness category",
        )

        self.assertIn("ORIGINAL PROMPT", prompt)
        self.assertIn("unknown readiness category", prompt)
        self.assertIn('"information_seeking_1_7"', prompt)
        self.assertIn('"selection_commitment_1_7"', prompt)
        self.assertIn('"brief_reason": <1 to 20 words>', prompt)
        self.assertIn('"information"|"criteria"|"comparison"', prompt)
        self.assertIn("Do not use category values such as evaluation or review", prompt)

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


if __name__ == "__main__":
    unittest.main()
