"""Contracts for the immutable text-only readiness Hugging Face snapshot."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.scripts.build_readiness_text_hf_dataset import (
    COUNTERFACTUAL_CONFIG,
    _atomic_json,
    _dataset_checksums,
    _sanitize_model_fields,
    annotate_counterfactual_variant,
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


def _counterfactual_fixture(
    root: Path,
    selected: list[dict[str, object]],
    *,
    candidate_count: int,
) -> None:
    scenario = "search_trigger_v2_relaxed_tolerance"
    selected_path = root / "scenarios" / scenario / "selected.jsonl"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selected),
        encoding="utf-8",
    )
    diagnostics = {
        "selected_count": len(selected),
        "verified_selected_count": len(selected),
        "require_both_views_within_tolerance": True,
        "require_delexicalized_template_uniqueness": True,
        "selected_delexicalized_templates_are_unique": True,
    }
    scenario_summary = {
        "selected_count": len(selected),
        "missing_count": 3 - len(selected),
        "accepted_candidate_count": len(selected),
        "distance_tolerance": 0.035,
        "require_template_uniqueness": True,
        "selection_diagnostics": diagnostics,
    }
    _atomic_json(selected_path.parent / "summary.json", scenario_summary)
    _atomic_json(
        root / "counterfactual_summary.json",
        {
            "format_version": "readiness-search-trigger-counterfactual-v1",
            "candidate_count": candidate_count,
            "validation_recovered_count": 1,
            "scenarios": {
                "question_v1_historical": {
                    "selected_count": 2,
                    "missing_count": 1,
                },
                scenario: scenario_summary,
            },
        },
    )


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
            candidate_path = population / "merged/candidates.jsonl"
            candidates = [
                json.loads(line)
                for line in candidate_path.read_text(encoding="utf-8").splitlines()
            ]
            for index, candidate in enumerate(candidates):
                candidate["generation_seed"] = 2**63 + index
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidates),
                encoding="utf-8",
            )

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
            generated_rows = []
            for path in sorted(
                (output / "data/generated_candidates").glob("*.parquet")
            ):
                generated_rows.extend(pq.read_table(path).to_pylist())
            self.assertEqual(
                [row["generation_seed"] for row in generated_rows],
                [str(2**63 + index) for index in range(3)],
            )
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

    def test_adds_an_existing_candidate_counterfactual_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            likert = root / "likert"
            population = root / "population"
            base = root / "base"
            output = root / "annotated"
            counterfactual = root / "counterfactual"
            _likert_fixture(likert)
            selected, validation = _fixture(population)
            candidate_path = population / "merged/candidates.jsonl"
            candidates = [
                json.loads(line)
                for line in candidate_path.read_text(encoding="utf-8").splitlines()
            ]
            candidates[2]["question"] = "Investigate likely causes and practical fixes"
            validation[2].update(
                {
                    "accepted": False,
                    "exact_keyword_present": False,
                    "single_question": False,
                    "topic_relevant": True,
                    "search_intent": True,
                    "web_answerable": True,
                    "standalone": False,
                    "natural_language": True,
                    "relevance_score_1_5": 5,
                }
            )
            candidate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in candidates),
                encoding="utf-8",
            )
            validation_path = population / "merged/validation.jsonl"
            validation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in validation),
                encoding="utf-8",
            )
            relaxed = list(selected)
            relaxed.append(
                {
                    "candidate_id": "candidate:three",
                    "keyword_id": "keyword:three",
                    "keyword": "gamma",
                    "target_id": "target:three",
                    "generator_id": "generator:three",
                    "generator_model": "model:three",
                    "question": "Investigate likely causes and practical fixes",
                    "both_views_within_tolerance": True,
                    "target_distance": 0.020,
                    "reference_target_distance": 0.025,
                    "candidate_aligned_target_distance": 0.030,
                }
            )
            finalize_text_dataset(
                likert_dataset_root=likert,
                prompt_population_root=population,
                output_dir=base,
                git_commit_sha="4" * 40,
            )
            _counterfactual_fixture(
                counterfactual,
                relaxed,
                candidate_count=3,
            )

            manifest = annotate_counterfactual_variant(
                dataset_dir=base,
                counterfactual_root=counterfactual,
                output_dir=output,
                rows_per_shard=2,
                git_commit_sha="5" * 40,
            )

            verify_text_dataset(output)
            self.assertEqual(manifest["table_counts"][COUNTERFACTUAL_CONFIG], 3)
            variant = manifest["counterfactual_prompt_variants"][
                COUNTERFACTUAL_CONFIG
            ]
            self.assertTrue(variant["existing_candidates_only"])
            self.assertFalse(variant["new_generation_performed"])
            self.assertEqual(variant["historical_selected_count"], 2)
            self.assertEqual(variant["incremental_selected_count"], 1)
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn(COUNTERFACTUAL_CONFIG, readme)
            self.assertIn("did not generate or embed new prompt text", readme)

    def test_rejects_counterfactual_row_that_fails_v2_review(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            likert = root / "likert"
            population = root / "population"
            base = root / "base"
            counterfactual = root / "counterfactual"
            _likert_fixture(likert)
            selected, _ = _fixture(population)
            finalize_text_dataset(
                likert_dataset_root=likert,
                prompt_population_root=population,
                output_dir=base,
                git_commit_sha="6" * 40,
            )
            rejected = dict(selected[0])
            rejected["candidate_id"] = "candidate:three"
            rejected["keyword_id"] = "keyword:three"
            rejected["keyword"] = "gamma"
            rejected["target_id"] = "target:three"
            rejected["question"] = "What is gamma?"
            _counterfactual_fixture(
                counterfactual,
                [rejected],
                candidate_count=3,
            )

            with self.assertRaisesRegex(ValueError, "fails search-trigger-v2"):
                annotate_counterfactual_variant(
                    dataset_dir=base,
                    counterfactual_root=counterfactual,
                    output_dir=root / "annotated",
                    git_commit_sha="7" * 40,
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

    def test_serializes_hash_derived_seeds_as_exact_decimal_strings(self) -> None:
        value = 2**64 + 1009

        row = _sanitize_model_fields(
            {
                "candidate_id": "candidate:one",
                "generation_seed": value,
            }
        )

        self.assertEqual(row["generation_seed"], str(value))


if __name__ == "__main__":
    unittest.main()
