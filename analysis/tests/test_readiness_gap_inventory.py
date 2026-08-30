from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.build_readiness_gap_inventory import build_gap_inventory


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _checkpoint(root: Path) -> tuple[Path, Path]:
    checkpoint = root / "checkpoint"
    selection = checkpoint / "strict-selection"
    selected = [
        {
            "keyword_id": "keyword:alpha",
            "target_id": "target:alpha:0",
            "both_views_within_tolerance": True,
        },
        {
            "keyword_id": "keyword:beta",
            "target_id": "target:beta:0",
            "both_views_within_tolerance": True,
        },
    ]
    tasks = []
    for keyword in ("alpha", "beta"):
        for axis_index, coordinate in ((1, 0.5), (2, 1.0)):
            tasks.append(
                {
                    "task_id": f"task:{keyword}:{axis_index}",
                    "keyword_id": f"keyword:{keyword}",
                    "keyword": keyword,
                    "target": {
                        "target_id": f"target:{keyword}:{axis_index}",
                        "target_index": axis_index,
                        "axis_1_index": axis_index,
                        "axis_2_index": 0,
                        "normalized_axis_1": coordinate,
                        "normalized_axis_2": 0.5,
                        "raw_axis_1": coordinate,
                        "raw_axis_2": 0.5,
                    },
                    "round_index": 7,
                    "generator_id": "generator-a",
                    "generation_seed": axis_index,
                    "requested_candidate_count": 4,
                    "feedback": (
                        "No independently validated candidate covered this cell."
                    ),
                }
            )
    task_path = selection / "generation_tasks_round_07.jsonl"
    _write_jsonl(selection / "spatially_selected_questions.jsonl", selected)
    _write_jsonl(task_path, tasks)
    _write_json(
        checkpoint / "verified_round_summary.json",
        {"selected_count": 2, "refinement_task_count": 4},
    )
    _write_json(
        selection / "spatial_coverage_diagnostics.json",
        {"keyword_count": 2, "target_count_per_keyword": 3},
    )
    _write_json(
        selection / "run_manifest.json",
        {"selected_count": 2, "next_round_task_count": 4},
    )
    return checkpoint, task_path


class ReadinessGapInventoryTests(unittest.TestCase):
    def test_builds_exhaustive_ranked_gap_inventory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, task_path = _checkpoint(root)
            output = root / "gap-inventory"

            manifest = build_gap_inventory(checkpoint, output, report_top=2)

            self.assertEqual(manifest["target_cell_count"], 6)
            self.assertEqual(manifest["selected_count"], 2)
            self.assertEqual(manifest["gap_count"], 4)
            self.assertEqual(manifest["completion_fraction"], 1 / 3)
            self.assertTrue(manifest["exhaustive_target_partition"])
            self.assertEqual(
                (output / "exact_gap_tasks.jsonl").read_bytes(),
                task_path.read_bytes(),
            )
            prioritized = [
                json.loads(line)
                for line in (
                    output / "prioritized_gap_inventory.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["priority_rank"] for row in prioritized],
                [1, 2, 3, 4],
            )
            self.assertTrue(
                all(row["axis_level_gap_count"] == 2 for row in prioritized)
            )
            self.assertIn(
                "same task objects reordered",
                (output / "gap_report.md").read_text(encoding="utf-8"),
            )

    def test_rejects_a_summary_gap_count_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _ = _checkpoint(root)
            _write_json(
                checkpoint / "verified_round_summary.json",
                {"selected_count": 2, "refinement_task_count": 3},
            )

            with self.assertRaisesRegex(ValueError, "gap-count mismatch"):
                build_gap_inventory(checkpoint, root / "gap-inventory")

    def test_rejects_overwriting_an_existing_inventory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _ = _checkpoint(root)
            output = root / "gap-inventory"
            output.mkdir()

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                build_gap_inventory(checkpoint, output)


if __name__ == "__main__":
    unittest.main()
