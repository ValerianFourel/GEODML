"""Contracts for the immutable text-only readiness Hugging Face snapshot."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.scripts.build_readiness_text_hf_dataset import (
    _atomic_json,
    _dataset_checksums,
    _sanitize_model_fields,
    finalize_text_dataset,
    main,
    verify_text_dataset,
)
from analysis.tests.test_audit_fully_compliant_readiness_prompts import _fixture


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    assert pa is not None
    assert pq is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _likert_fixture(root: Path) -> None:
    prompts = [
        {
            "item_id": "item:one",
            "split": "development",
            "text": "Compare two software tools before choosing one.",
            "source_name": "databricks-dolly-15k",
            "license": "CC-BY-SA-3.0",
        },
        {
            "item_id": "item:two",
            "split": "confirmation",
            "text": "Which option should I implement next?",
            "source_name": "openassistant-oasst1",
            "license": "Apache-2.0",
        },
    ]
    annotations = [
        {
            "item_id": "item:one",
            "task_id": "task:one",
            "judge_slot": "primary-frontier",
            "split": "development",
            "overall_readiness_0_100": 65,
            "information_seeking_1_7": 4,
        },
        {
            "item_id": "item:two",
            "task_id": "task:two",
            "judge_slot": "primary-frontier",
            "split": "confirmation",
            "overall_readiness_0_100": 82,
            "information_seeking_1_7": 6,
        },
    ]
    _write_parquet(
        root / "data/prompts/development-00000.parquet",
        prompts[:1],
    )
    _write_parquet(
        root / "data/prompts/confirmation-00000.parquet",
        prompts[1:],
    )
    _write_parquet(
        root / "data/annotations/development-00000.parquet",
        annotations[:1],
    )
    _write_parquet(
        root / "data/annotations/confirmation-00000.parquet",
        annotations[1:],
    )
    (root / "README.md").write_text("# Verified Likert fixture\n", encoding="utf-8")
    _atomic_json(
        root / "dataset_manifest.json",
        {
            "publication_safe": True,
            "restricted_prompt_count_excluded": 1,
            "restricted_sources_excluded": ["allenai-wildchat-1m"],
            "table_counts": {"prompts": 2, "annotations": 2},
        },
    )
    _atomic_json(root / "checksums.json", _dataset_checksums(root))


@unittest.skipUnless(pa is not None, "pyarrow is required")
class ReadinessTextHfDatasetTests(unittest.TestCase):
    def test_finalizes_one_verified_text_only_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            likert = root / "likert"
            population = root / "population"
            output = root / "dataset"
            _likert_fixture(likert)
            _fixture(population)

            manifest = finalize_text_dataset(
                likert_dataset_root=likert,
                prompt_population_root=population,
                output_dir=output,
                rows_per_shard=1,
                git_commit_sha="1" * 40,
            )

            verify_text_dataset(output)
            self.assertEqual(manifest["table_counts"]["likert_prompts"], 2)
            self.assertEqual(manifest["table_counts"]["likert_annotations"], 2)
            self.assertEqual(manifest["table_counts"]["likert_graded_prompts"], 2)
            self.assertEqual(manifest["table_counts"]["generated_candidates"], 3)
            self.assertEqual(
                manifest["table_counts"]["candidate_compliance_annotations"],
                3,
            )
            self.assertEqual(
                manifest["table_counts"]["fully_compliant_prompts"],
                2,
            )
            self.assertFalse(manifest["generated_candidates_are_likert_graded"])
            self.assertFalse(any(output.rglob("*.jsonl")))
            self.assertFalse(any(output.rglob("*.npz")))
            self.assertIn(
                "generated candidates are not Likert graded",
                (output / "README.md").read_text(encoding="utf-8"),
            )

    def test_refuses_to_export_a_failed_prompt_population_audit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            likert = root / "likert"
            population = root / "population"
            _likert_fixture(likert)
            selected, _ = _fixture(population)
            selected[0]["both_views_within_tolerance"] = False
            path = population / "strict-selection/spatially_selected_questions.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in selected),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not independently ready"):
                finalize_text_dataset(
                    likert_dataset_root=likert,
                    prompt_population_root=population,
                    output_dir=root / "dataset",
                    git_commit_sha="2" * 40,
                )

    def test_refuses_to_overwrite_an_existing_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                finalize_text_dataset(
                    likert_dataset_root=Path(directory) / "unused-likert",
                    prompt_population_root=Path(directory) / "unused-population",
                    output_dir=output,
                )

    def test_publish_requires_exact_repository_confirmation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            likert = root / "likert"
            population = root / "population"
            output = root / "dataset"
            _likert_fixture(likert)
            _fixture(population)
            finalize_text_dataset(
                likert_dataset_root=likert,
                prompt_population_root=population,
                output_dir=output,
                git_commit_sha="3" * 40,
            )

            arguments = [
                "build_readiness_text_hf_dataset.py",
                "publish",
                "--dataset-dir",
                str(output),
                "--repo-id",
                "owner/expected",
                "--confirm-repo-id",
                "owner/wrong",
            ]
            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(SystemExit, "must exactly match"):
                    main()


class ReadinessTextSanitizationTests(unittest.TestCase):
    def test_replaces_an_absolute_model_cache_path_with_repository_id(self) -> None:
        revision = "a" * 40
        row = _sanitize_model_fields(
            {
                "candidate_id": "candidate:one",
                "generator_model": (
                    "/e/project/models/qwen/Qwen3-32B/" + revision
                ),
            }
        )

        self.assertEqual(row["generator_model"], "qwen/Qwen3-32B")
        self.assertEqual(row["split"], "data")
        self.assertNotIn("/e/project", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
