"""Tests for targeted high-axis generation-yield diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.scripts.audit_readiness_high_axis_generation_yield import audit


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _selected(
    candidate_id: str,
    keyword_id: str,
    target_id: str,
    target: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "keyword_id": keyword_id,
        "target_id": target_id,
        "target_normalized_axis_1": target,
        "reference_normalized_axis_1": target + 0.01,
        "candidate_aligned_normalized_axis_1": target - 0.02,
    }


class HighAxisGenerationYieldTests(unittest.TestCase):
    def test_reports_band_specific_recovery_from_generated_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            round_root = root / "round-01"
            output = root / "audit"
            _jsonl(baseline, [_selected("old", "k0", "t0", 0.75)])
            tasks = [
                {
                    "task_id": "task-1",
                    "keyword_id": "k1",
                    "generator_id": "a",
                    "target": {"target_id": "t1", "normalized_axis_1": 0.85},
                },
                {
                    "task_id": "task-2",
                    "keyword_id": "k2",
                    "generator_id": "b",
                    "target": {"target_id": "t2", "normalized_axis_1": 0.95},
                },
            ]
            _jsonl(round_root / "refinement-task-batch.jsonl", tasks)
            generated = [
                {"candidate_id": "new-1", "keyword_id": "k1", "target_id": "t1"},
                {"candidate_id": "new-2", "keyword_id": "k2", "target_id": "t2"},
            ]
            _jsonl(
                round_root / "generation" / "candidates" / "a.jsonl",
                generated,
            )
            _jsonl(
                round_root / "validation.jsonl",
                [
                    {"candidate_id": "new-1", "accepted": True},
                    {"candidate_id": "new-2", "accepted": False},
                ],
            )
            _jsonl(
                round_root
                / "strict-selection"
                / "spatially_selected_questions.jsonl",
                [
                    _selected("old", "k0", "t0", 0.75),
                    _selected("new-1", "k1", "t1", 0.85),
                ],
            )

            result = audit(
                baseline,
                round_root,
                output,
                minimum_target_axis_1=0.70,
            )

            self.assertEqual(result["overall"]["task_count"], 2)
            self.assertEqual(result["overall"]["generated_candidate_count"], 2)
            self.assertEqual(result["overall"]["accepted_generated_candidate_count"], 1)
            self.assertEqual(result["overall"]["recovered_target_cell_count"], 1)
            self.assertEqual(result["target_bands"]["0.80-0.90"]["recovered_target_cell_count"], 1)
            self.assertEqual(result["target_bands"]["0.90-1.00"]["recovered_target_cell_count"], 0)
            self.assertTrue((output / "high_axis_yield.md").is_file())

    def test_rejects_task_below_targeted_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.jsonl"
            round_root = root / "round-01"
            _jsonl(baseline, [_selected("old", "k0", "t0", 0.75)])
            _jsonl(
                round_root / "refinement-task-batch.jsonl",
                [
                    {
                        "task_id": "task-low",
                        "keyword_id": "k1",
                        "generator_id": "a",
                        "target": {
                            "target_id": "t1",
                            "normalized_axis_1": 0.69,
                        },
                    }
                ],
            )
            _jsonl(
                round_root / "generation" / "candidates" / "a.jsonl",
                [{"candidate_id": "new", "keyword_id": "k1", "target_id": "t1"}],
            )
            _jsonl(
                round_root / "validation.jsonl",
                [{"candidate_id": "new", "accepted": True}],
            )
            _jsonl(
                round_root
                / "strict-selection"
                / "spatially_selected_questions.jsonl",
                [_selected("old", "k0", "t0", 0.75)],
            )
            with self.assertRaisesRegex(ValueError, "below the high-axis threshold"):
                audit(
                    baseline,
                    round_root,
                    root / "audit",
                    minimum_target_axis_1=0.70,
                )


if __name__ == "__main__":
    unittest.main()
