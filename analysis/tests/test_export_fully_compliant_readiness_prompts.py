from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.export_fully_compliant_readiness_prompts import (
    export_fully_compliant_prompts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> list[dict[str, object]]:
    selected = [
        {
            "candidate_id": "candidate:one",
            "keyword_id": "keyword:one",
            "target_id": "target:one",
            "both_views_within_tolerance": True,
            "target_distance": 0.010,
            "reference_target_distance": 0.011,
            "candidate_aligned_target_distance": 0.012,
            "question": "What is alpha?",
        },
        {
            "candidate_id": "candidate:two",
            "keyword_id": "keyword:two",
            "target_id": "target:two",
            "both_views_within_tolerance": True,
            "target_distance": 0.013,
            "reference_target_distance": 0.014,
            "candidate_aligned_target_distance": 0.015,
            "question": "How does beta work?",
        },
    ]
    selection = root / "strict-selection"
    _write_jsonl(selection / "spatially_selected_questions.jsonl", selected)
    _write_json(
        selection / "run_manifest.json",
        {
            "selected_count": 2,
            "coordinate_acceptance_contract": {
                "enabled": True,
                "distance_tolerance": 0.017,
            },
            "surface_acceptance_contract": {"enabled": True},
        },
    )
    _write_json(
        selection / "spatial_coverage_diagnostics.json",
        {
            "selected_count": 2,
            "verified_selected_count": 2,
            "require_both_views_within_tolerance": True,
            "require_delexicalized_template_uniqueness": True,
            "selected_delexicalized_templates_are_unique": True,
        },
    )
    _write_json(
        root / "verified_round_summary.json",
        {
            "selected_count": 2,
            "strict_dual_view_contract_enabled": True,
            "delexicalized_template_uniqueness_enabled": True,
            "selected_diversity_gate_passed": True,
            "spacing_gate_passed": False,
            "verified_population_passed": False,
        },
    )
    _write_json(
        root / "selected-diversity/run_manifest.json",
        {"row_count": 2, "all_checks_passed": True},
    )
    _write_json(
        root / "selected-diversity/question_diversity_audit.json",
        {"row_count": 2, "all_checks_passed": True},
    )
    validation = [
        {"candidate_id": "candidate:one", "accepted": True},
        {"candidate_id": "candidate:two", "accepted": True},
        {"candidate_id": "candidate:three", "accepted": False},
    ]
    validation_path = root / "merged" / "validation.jsonl"
    _write_jsonl(validation_path, validation)
    _write_json(
        validation_path.with_suffix(".jsonl.manifest.json"),
        {"candidate_count": 3, "reviewed_count": 3, "accepted_count": 2},
    )
    return selected


class FullyCompliantPromptExportTests(unittest.TestCase):
    def test_exports_exact_selected_rows_with_audit_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            selected = _fixture(root)
            output = Path(directory) / "export" / "fully-compliant.jsonl"

            manifest = export_fully_compliant_prompts(root, output)

            self.assertEqual(manifest["selected_count"], 2)
            self.assertFalse(manifest["complete_30330_population_passed"])
            self.assertEqual(
                [json.loads(line) for line in output.read_text().splitlines()],
                selected,
            )
            self.assertTrue(output.with_suffix(".jsonl.manifest.json").is_file())

    def test_rejects_a_selected_prompt_that_fails_dual_view_tolerance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            selected = _fixture(root)
            selected[0]["both_views_within_tolerance"] = False
            _write_jsonl(
                root / "strict-selection/spatially_selected_questions.jsonl",
                selected,
            )

            with self.assertRaisesRegex(ValueError, "fails strict dual-view"):
                export_fully_compliant_prompts(
                    root,
                    Path(directory) / "fully-compliant.jsonl",
                )

    def test_rejects_a_selected_prompt_without_independent_acceptance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            _fixture(root)
            validation_path = root / "merged/validation.jsonl"
            validation = [
                {"candidate_id": "candidate:one", "accepted": True},
                {"candidate_id": "candidate:two", "accepted": False},
                {"candidate_id": "candidate:three", "accepted": True},
            ]
            _write_jsonl(validation_path, validation)

            with self.assertRaisesRegex(ValueError, "lacks independent acceptance"):
                export_fully_compliant_prompts(
                    root,
                    Path(directory) / "fully-compliant.jsonl",
                )

    def test_refuses_to_overwrite_an_existing_export(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            _fixture(root)
            output = Path(directory) / "fully-compliant.jsonl"
            output.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                export_fully_compliant_prompts(root, output)
