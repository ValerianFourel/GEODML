"""Contracts for immutable ten-way keyword section plans."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.scripts.prepare_readiness_keyword_sections import (
    build_keyword_section_plan,
    verify_keyword_section_plan,
    write_keyword_section_plan,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class ReadinessKeywordSectionPlanTests(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        checkpoint = root / "pipeline/round-12"
        _write_json(
            checkpoint / "verified_round_summary.json",
            {"selected_count": 12799},
        )
        (checkpoint / "candidate-files.txt").write_text(
            "/immutable/candidates.jsonl\n", encoding="utf-8"
        )
        (checkpoint / "validation.jsonl").write_text(
            '{"candidate_id":"candidate-1","accepted":true}\n',
            encoding="utf-8",
        )
        _write_json(checkpoint / "validation.jsonl.manifest.json", {"rows": 1})
        _write_json(
            checkpoint / "projections/qwen/projection_manifest.json", {"rows": 1}
        )
        _write_json(
            checkpoint / "projections/mistral/projection_manifest.json", {"rows": 1}
        )
        _write_json(checkpoint / "strict-selection/run_manifest.json", {"rows": 1})
        tasks = []
        for keyword_index in range(100):
            for target_index in range(3):
                tasks.append(
                    {
                        "task_id": f"task-{keyword_index}-{target_index}",
                        "keyword_id": f"keyword-{keyword_index:03d}",
                        "generator_id": "qwen" if target_index % 2 else "gemma",
                        "target": {
                            "target_id": f"target-{keyword_index}-{target_index}"
                        },
                    }
                )
        task_file = (
            checkpoint
            / "strict-selection/generation_tasks_round_38.jsonl"
        )
        task_file.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in tasks),
            encoding="utf-8",
        )
        return checkpoint

    def test_plan_is_keyword_disjoint_exhaustive_and_records_resume_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary))
            plan = build_keyword_section_plan(
                checkpoint,
                section_count=10,
                partition_salt="ten-section-test",
            )
            self.assertEqual(plan["section_count"], 10)
            self.assertEqual(plan["remaining_task_count"], 300)
            self.assertEqual(plan["remaining_keyword_count"], 100)
            self.assertEqual(plan["next_round_index"], 38)
            self.assertEqual(plan["initial_logical_round_index"], 37)
            sections = plan["sections"]
            self.assertEqual(
                sum(int(section["remaining_task_count"]) for section in sections),
                300,
            )
            keyword_sets = [set(section["keyword_ids"]) for section in sections]
            self.assertEqual(len(set().union(*keyword_sets)), 100)
            for left in range(10):
                for right in range(left + 1, 10):
                    self.assertFalse(keyword_sets[left] & keyword_sets[right])

    def test_written_plan_verifies_and_detects_checkpoint_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            output = root / "ten-sections.json"
            written = write_keyword_section_plan(
                checkpoint,
                output,
                section_count=10,
                partition_salt="ten-section-test",
            )
            self.assertEqual(verify_keyword_section_plan(output), written)
            task_file = Path(str(written["source_task_file"]))
            task_file.write_text(
                task_file.read_text(encoding="utf-8")
                + '{"task_id":"new","keyword_id":"new","generator_id":"qwen",'
                + '"target":{"target_id":"new"}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_keyword_section_plan(output)


if __name__ == "__main__":
    unittest.main()
