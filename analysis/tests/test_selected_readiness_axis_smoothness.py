from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.audit_selected_readiness_axis_smoothness import (
    audit_selected_population,
    main,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _selected_rows() -> list[dict[str, object]]:
    rows = []
    for keyword in ("alpha", "beta"):
        for index, coordinate in enumerate((0.0, 0.5, 1.0)):
            rows.append(
                {
                    "candidate_id": f"candidate:{keyword}:{index}",
                    "keyword_id": f"keyword:{keyword}",
                    "target_normalized_axis_1": coordinate,
                    "consensus_normalized_axis_1": coordinate,
                    "reference_normalized_axis_1": coordinate,
                    "candidate_aligned_normalized_axis_1": coordinate,
                }
            )
    return rows


class SelectedReadinessAxisSmoothnessTests(unittest.TestCase):
    def test_audits_axis_coverage_and_simulated_retrieval(self) -> None:
        with TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.jsonl"
            _write_jsonl(selected, _selected_rows())

            audit = audit_selected_population(
                selected,
                grid_points=101,
                histogram_bins=10,
                coverage_tolerances=(0.05,),
            )

            self.assertEqual(audit["prompt_count"], 6)
            self.assertEqual(audit["keyword_count"], 2)
            self.assertEqual(audit["global_axis_span"], 1.0)
            self.assertEqual(audit["target_consensus_error"]["maximum"], 0.0)
            self.assertEqual(audit["cross_view_disagreement"]["maximum"], 0.0)
            self.assertEqual(audit["target_observed_spearman"]["p50"], 1.0)
            self.assertEqual(audit["monotonic_adjacent_fraction"]["p50"], 1.0)
            self.assertEqual(
                audit["coverage"]["0.050000"]["consensus_fraction"],
                audit["coverage"]["0.050000"]["dual_view_robust_fraction"],
            )

    def test_cli_writes_machine_and_human_readable_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected.jsonl"
            output = root / "audit"
            _write_jsonl(selected, _selected_rows())

            status = main(
                [
                    "--population",
                    f"official={selected}",
                    "--output-dir",
                    str(output),
                    "--grid-points",
                    "101",
                    "--histogram-bins",
                    "10",
                ]
            )

            self.assertEqual(status, 0)
            payload = json.loads(
                (output / "axis_smoothness_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["populations"]["official"]["prompt_count"], 6)
            report = (output / "axis_smoothness_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Selected readiness axis-smoothness audit", report)
            self.assertIn("not search-ranking effects", report)

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                main(
                    [
                        "--population",
                        f"official={selected}",
                        "--output-dir",
                        str(output),
                    ]
                )

    def test_rejects_rows_without_dual_view_coordinates(self) -> None:
        with TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.jsonl"
            rows = _selected_rows()
            del rows[0]["reference_normalized_axis_1"]
            _write_jsonl(selected, rows)

            with self.assertRaisesRegex(ValueError, "lacks"):
                audit_selected_population(selected)


if __name__ == "__main__":
    unittest.main()
