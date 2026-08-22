"""Tests for the post-projection axis-1 continuity audit."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.scripts.audit_readiness_axis1_continuity import (
    audit_axis_1_continuity,
    main,
)


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class Axis1ContinuityAuditTests(unittest.TestCase):
    def _fixture(self, root: Path):
        plan = root / "plan"
        plan.mkdir()
        _json(
            plan / "plan_manifest.json",
            {
                "target_design": "axis-1-linear",
                "keyword_count": 2,
                "target_count_per_keyword": 3,
            },
        )
        _json(plan / "subspace_bounds.json", {"axis_1_low": 0.0, "axis_1_high": 1.0})
        targets = [
            {
                "target_id": f"target:{index}",
                "target_index": index,
                "normalized_axis_1": value,
            }
            for index, value in enumerate((0.0, 0.5, 1.0))
        ]
        _jsonl(plan / "target_grid.jsonl", targets)
        candidate_rows = []
        projection_rows = []
        actual_by_keyword = {
            "keyword:one": (0.0, 0.5, 1.0),
            "keyword:two": (0.0, 0.9, 1.0),
        }
        for keyword_id, actual_values in actual_by_keyword.items():
            keyword = keyword_id.replace(":", " ")
            for target, actual in zip(targets, actual_values):
                candidate_id = f"candidate:{keyword_id}:{target['target_index']}"
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "keyword_id": keyword_id,
                        "keyword": keyword,
                        "target_id": target["target_id"],
                        "target_index": target["target_index"],
                        "target_normalized_axis_1": target["normalized_axis_1"],
                        "question": (
                            f"How should a careful reader investigate {keyword} "
                            f"for scenario {target['target_index']}?"
                        ),
                    }
                )
                projection_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "reference_raw_axis_1": actual,
                        "candidate_aligned_raw_axis_1": actual,
                    }
                )
        candidates = root / "candidates.jsonl"
        projections = root / "aligned.jsonl"
        _jsonl(candidates, candidate_rows)
        _jsonl(projections, projection_rows)
        return plan, candidates, projections

    def test_half_step_and_one_step_coverage_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, candidates, projections = self._fixture(root)
            summary = audit_axis_1_continuity(
                plan_dir=plan,
                candidate_paths=(candidates,),
                aligned_projection_path=projections,
                tolerance_steps=(0.5, 1.0),
                primary_tolerance_steps=0.5,
            )
            self.assertAlmostEqual(summary["axis_1_target_step"], 0.5)
            self.assertAlmostEqual(summary["primary_normalized_tolerance"], 0.25)
            half_step, one_step = summary["tolerance_sweep"]
            self.assertEqual(half_step["dual_view_intended_hit_count"], 5)
            self.assertEqual(half_step["globally_matchable_target_count"], 5)
            self.assertEqual(half_step["fully_coverable_keyword_count"], 1)
            self.assertEqual(one_step["dual_view_intended_hit_count"], 6)
            self.assertEqual(one_step["globally_matchable_target_count"], 6)
            self.assertEqual(one_step["fully_coverable_keyword_count"], 2)
            self.assertEqual(
                summary["primary_global_assignment"]["fully_covered_keyword_count"],
                1,
            )

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, candidates, projections = self._fixture(root)
            output = root / "output"
            self.assertEqual(
                main(
                    (
                        "--plan-dir",
                        str(plan),
                        "--candidates",
                        str(candidates),
                        "--aligned-projections",
                        str(projections),
                        "--output-dir",
                        str(output),
                    )
                ),
                0,
            )
            payload = json.loads(
                (output / "axis_1_continuity_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["candidate_count"], 6)
            self.assertTrue((output / "axis_1_continuity_report.md").is_file())

    def test_missing_projections_fail_with_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan, candidates, _ = self._fixture(root)
            with self.assertRaisesRegex(FileNotFoundError, "run both LLM2Vec"):
                audit_axis_1_continuity(
                    plan_dir=plan,
                    candidate_paths=(candidates,),
                    aligned_projection_path=root / "missing.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
