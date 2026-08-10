"""Tests for the prompt-only Milestone 2 calibration corpus."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.prompt_calibration import (
    DEFAULT_B_GRID,
    FORBIDDEN_CRITERIA,
    generate_calibration_records,
    load_calibration_manifest,
    write_calibration_corpus,
)


class PromptCalibrationTests(unittest.TestCase):
    def test_default_corpus_has_expected_number_of_records(self) -> None:
        self.assertEqual(len(generate_calibration_records()), 220)

    def test_every_b_by_s_combination_occurs_once(self) -> None:
        records = generate_calibration_records(number_style_seeds=4, first_style_seed=7)
        combinations = Counter(
            (record.assigned_bias, record.style_seed) for record in records
        )
        expected = {
            (bias, seed)
            for seed in range(7, 11)
            for bias in DEFAULT_B_GRID
        }
        self.assertEqual(set(combinations), expected)
        self.assertTrue(all(count == 1 for count in combinations.values()))

    def test_same_seed_has_same_style_plan_at_every_bias(self) -> None:
        records = generate_calibration_records(number_style_seeds=5)
        for seed in range(5):
            plans = {
                record.style_plan for record in records if record.style_seed == seed
            }
            self.assertEqual(len(plans), 1)

    def test_regeneration_differs_only_in_timestamp_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = write_calibration_corpus(
                root / "first", generated_at="2026-01-01T00:00:00Z"
            )
            second = write_calibration_corpus(
                root / "second", generated_at="2026-01-02T00:00:00Z"
            )
            first_rows = self._read_rows(first.manifest_path)
            second_rows = self._read_rows(second.manifest_path)
            for row in first_rows + second_rows:
                row.pop("generated_at")
            self.assertEqual(first_rows, second_rows)

    def test_jsonl_round_trip_preserves_scientific_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = write_calibration_corpus(
                temporary_directory,
                b_grid=(0.0, 0.5, 1.0),
                number_style_seeds=3,
                top_n=7,
            )
            loaded = load_calibration_manifest(artifacts.manifest_path)
            self.assertEqual(loaded, artifacts.records)
            self.assertTrue(
                all(row["top_n"] == 7 for row in self._read_rows(artifacts.manifest_path))
            )

    def test_corrupt_prompt_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = write_calibration_corpus(
                temporary_directory, b_grid=(0.0,), number_style_seeds=1
            )
            rows = self._read_rows(artifacts.manifest_path)
            rows[0]["prompt_hash"] = "0" * 64
            self._write_rows(artifacts.manifest_path, rows)
            with self.assertRaisesRegex(ValueError, "prompt_hash does not match"):
                load_calibration_manifest(artifacts.manifest_path)

    def test_duplicate_prompt_id_with_different_text_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = write_calibration_corpus(
                temporary_directory, b_grid=(0.0,), number_style_seeds=1
            )
            rows = self._read_rows(artifacts.manifest_path)
            conflicting = dict(rows[0])
            conflicting["prompt_template"] += "\nChanged content."
            conflicting["prompt_hash"] = hashlib.sha256(
                conflicting["prompt_template"].encode("utf-8")
            ).hexdigest()
            rows.append(conflicting)
            self._write_rows(artifacts.manifest_path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate prompt_id"):
                load_calibration_manifest(artifacts.manifest_path)

    def test_existing_artifacts_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = write_calibration_corpus(
                temporary_directory, b_grid=(0.0,), number_style_seeds=1
            )
            original = artifacts.manifest_path.read_text(encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_calibration_corpus(
                    temporary_directory, b_grid=(0.0,), number_style_seeds=1
                )
            self.assertEqual(
                artifacts.manifest_path.read_text(encoding="utf-8"), original
            )
            write_calibration_corpus(
                temporary_directory,
                b_grid=(0.0,),
                number_style_seeds=1,
                overwrite=True,
            )

    def test_every_record_has_required_placeholders(self) -> None:
        for record in generate_calibration_records():
            self.assertIn("{QUERY}", record.prompt_template)
            self.assertIn("{CANDIDATES}", record.prompt_template)
            self.assertIn("{TOP_N}", record.prompt_template)

    def test_forbidden_criteria_are_absent(self) -> None:
        for record in generate_calibration_records():
            lowered = record.prompt_template.lower()
            self.assertFalse(
                [criterion for criterion in FORBIDDEN_CRITERIA if criterion in lowered]
            )

    def test_report_identifies_finite_piecewise_constant_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = write_calibration_corpus(temporary_directory)
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("engineering scaffold", report)
            self.assertIn("finite monotonic phrase schedule", report)
            self.assertIn("piecewise-constant policy wording", report)
            self.assertIn(
                "Distinct policy realizations across B, including no preference: 5",
                report,
            )
            self.assertIn("Distinct non-empty preference phrases across B: 4", report)

    @staticmethod
    def _read_rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def _write_rows(path: Path, rows: list[dict]) -> None:
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
