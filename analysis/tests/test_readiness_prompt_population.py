"""Contracts for iterative question coverage of the readiness subspace."""

from __future__ import annotations

from dataclasses import replace
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_embedding_map import (
    READINESS_MAP_VERSION,
    ReadinessEmbeddingMap,
)
from analysis.interpretability.pipeline.readiness_prompt_population import (
    FakeReadinessQuestionGenerator,
    LocalReadinessQuestionGenerator,
    LocalSearchQuestionValidator,
    ReadinessQuestionCandidate,
    ReadinessSubspaceBounds,
    build_generation_tasks,
    build_refinement_tasks,
    build_target_grid,
    fit_reference_bounds,
    generate_question_candidates,
    parse_generated_question,
    parse_search_question_review,
    project_questions,
    select_diverse_questions,
    select_spatially_matched_questions,
    validate_generated_question,
)
from analysis.interpretability.pipeline.readiness_hf_dataset import (
    atomic_json,
    atomic_jsonl,
    read_json,
)
from analysis.scripts.build_readiness_prompt_population import (
    _compare_projections,
    _spatial_select,
)


class _StaticRanker:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0

    def rank(self, prompt, max_tokens=180, temperature=0.9, **kwargs):
        del prompt, max_tokens, temperature, kwargs
        self.call_count += 1
        return self.outputs.pop(0)


class ReadinessPromptPopulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = ReadinessSubspaceBounds(
            axis_1_low=-1.0,
            axis_1_high=1.0,
            axis_2_low=-1.0,
            axis_2_high=1.0,
            lower_quantile=0.05,
            upper_quantile=0.95,
            reference_split="development",
            reference_item_count=100,
        )

    def test_bounds_use_only_frozen_development_split(self) -> None:
        rows = [
            {
                "split": "development",
                "axis_1": index / 9,
                "axis_2": 1.0 - index / 9,
            }
            for index in range(10)
        ]
        rows.append({"split": "confirmation", "axis_1": 1e6, "axis_2": -1e6})
        bounds = fit_reference_bounds(rows, lower_quantile=0.0, upper_quantile=1.0)
        self.assertEqual(bounds.reference_item_count, 10)
        self.assertAlmostEqual(bounds.axis_1_low, 0.0)
        self.assertAlmostEqual(bounds.axis_1_high, 1.0)
        self.assertAlmostEqual(bounds.axis_2_low, 0.0)
        self.assertAlmostEqual(bounds.axis_2_high, 1.0)

    def test_default_grid_has_thirty_unique_serpentine_targets(self) -> None:
        targets = build_target_grid(self.bounds)
        self.assertEqual(len(targets), 30)
        self.assertEqual(len({target.target_id for target in targets}), 30)
        self.assertEqual(targets[0].target_id, "readiness-cell:00-00")
        self.assertEqual(targets[5].target_id, "readiness-cell:01-04")
        self.assertEqual(targets[9].target_id, "readiness-cell:01-00")
        self.assertEqual(
            {target.normalized_axis_1 for target in targets},
            set(np.linspace(0.0, 1.0, 6)),
        )

    def test_rounds_rotate_each_target_across_generator_models(self) -> None:
        targets = build_target_grid(self.bounds)[:2]
        keywords = (("keyword:one", "abandoned cart recovery"),)
        generators = ("gemma", "qwen", "mistral")
        round_zero = build_generation_tasks(keywords, targets, generators, round_index=0)
        round_one = build_generation_tasks(keywords, targets, generators, round_index=1)
        self.assertEqual(round_zero[0].generator_id, "gemma")
        self.assertEqual(round_one[0].generator_id, "qwen")
        self.assertNotEqual(round_zero[0].task_id, round_one[0].task_id)

    def test_fake_generation_retains_exact_keyword_and_task_identity(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            (target,),
            ("fake",),
            requested_candidate_count=2,
        )[0]
        rows = generate_question_candidates(
            (task,), FakeReadinessQuestionGenerator("fake")
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("abandoned cart recovery" in row.question for row in rows))
        self.assertEqual(len({row.candidate_id for row in rows}), 2)

    def test_local_generator_retries_invalid_json_and_reuses_cache(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            (target,),
            ("model-a",),
            requested_candidate_count=1,
        )[0]
        ranker = _StaticRanker(
            [
                "not json",
                '{"question":"How can a team understand abandoned cart recovery before choosing a practical approach?"}',
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            generator = LocalReadinessQuestionGenerator(
                ranker,
                generator_id="model-a",
                model_name="model/a",
                cache_directory=Path(temporary),
                maximum_attempts=2,
            )
            first = generator.generate(task)
            second = generator.generate(task)
        self.assertEqual(first, second)
        self.assertEqual(ranker.call_count, 2)

    def test_projection_matches_frozen_map_formula(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            (target,),
            ("fake",),
            requested_candidate_count=1,
        )[0]
        candidate = generate_question_candidates(
            (task,), FakeReadinessQuestionGenerator("fake")
        )[0]
        fitted = _map()
        embeddings = np.asarray([[3.0, 4.0, 0.0]])
        projected = project_questions(fitted, self.bounds, (candidate,), embeddings)[0]
        self.assertAlmostEqual(projected.raw_axis_1, 0.6)
        self.assertAlmostEqual(projected.raw_axis_2, 0.8)
        self.assertAlmostEqual(projected.normalized_axis_1, 0.8)
        self.assertAlmostEqual(projected.normalized_axis_2, 0.9)
        self.assertAlmostEqual(projected.predicted_scalar_readiness_0_1, 1.1)

    def test_selection_prefers_target_fit_then_builds_only_bad_cell_refinement(self) -> None:
        targets = build_target_grid(self.bounds)[:2]
        tasks = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            targets,
            ("fake-a", "fake-b"),
            requested_candidate_count=2,
        )
        candidates = generate_question_candidates(
            (tasks[0],), FakeReadinessQuestionGenerator(tasks[0].generator_id)
        ) + generate_question_candidates(
            (tasks[1],), FakeReadinessQuestionGenerator(tasks[1].generator_id)
        )
        embeddings = np.asarray(
            [
                [-1.0, -1.0, 0.1],
                [0.8, 0.8, 0.1],
                [-1.0, -0.5, 0.1],
                [0.8, 0.8, 0.1],
            ]
        )
        projections = project_questions(_map(), self.bounds, candidates, embeddings)
        selected, diagnostics = select_diverse_questions(
            candidates,
            projections,
            embeddings,
            novelty_weight=0.0,
            generator_balance_weight=0.0,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(diagnostics["selected_count"], 2)
        deliberately_bad = replace(selected[1], target_distance=0.9)
        refinement = build_refinement_tasks(
            (selected[0], deliberately_bad),
            targets,
            ("fake-a", "fake-b"),
            next_round_index=1,
            distance_tolerance=0.22,
        )
        self.assertEqual(len(refinement), 1)
        self.assertEqual(refinement[0].target.target_id, deliberately_bad.target_id)
        self.assertIn("closest earlier question", refinement[0].feedback)

    def test_external_decoded_text_cannot_bypass_question_contract(self) -> None:
        raw = 'prefix {"question":"What is abandoned cart recovery?"} suffix'
        question = parse_generated_question(raw)
        with self.assertRaisesRegex(ValueError, "8 to 60"):
            validate_generated_question(question, "abandoned cart recovery")
        with self.assertRaisesRegex(ValueError, "exact keyword"):
            validate_generated_question(
                "How can a team understand cart recovery before choosing a practical approach?",
                "abandoned cart recovery",
            )

    def test_independent_search_validator_enforces_semantic_contract_and_cache(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            (target,),
            ("fake",),
            requested_candidate_count=1,
        )[0]
        candidate = generate_question_candidates(
            (task,), FakeReadinessQuestionGenerator("fake")
        )[0]
        ranker = _StaticRanker(
            [
                '{"topic_relevant":true,"search_intent":true,'
                '"web_answerable":true,"standalone":true,'
                '"natural_language":true,"relevance_score_1_5":5,'
                '"concise_reason":"A direct, answerable search question."}'
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            validator = LocalSearchQuestionValidator(
                ranker,
                judge_id="independent-judge",
                model_name="model/judge",
                cache_directory=temporary,
            )
            first = validator.review(candidate)
            second = validator.review(candidate)
        self.assertTrue(first.accepted)
        self.assertEqual(first, second)
        self.assertEqual(ranker.call_count, 1)

        rejected = parse_search_question_review(
            '{"topic_relevant":true,"search_intent":false,'
            '"web_answerable":true,"standalone":true,'
            '"natural_language":true,"relevance_score_1_5":3,'
            '"concise_reason":"Not a genuine search request."}',
            candidate,
            judge_id="independent-judge",
            judge_model="model/judge",
        )
        self.assertFalse(rejected.accepted)

    def test_search_validator_normalizes_integer_equivalent_scores(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            (target,),
            ("fake",),
            requested_candidate_count=1,
        )[0]
        candidate = generate_question_candidates(
            (task,), FakeReadinessQuestionGenerator("fake")
        )[0]
        for score in ("5/5", 5.0):
            review = parse_search_question_review(
                '{"topic_relevant":true,"search_intent":true,'
                '"web_answerable":true,"standalone":true,'
                '"natural_language":true,'
                f'"relevance_score_1_5":{json.dumps(score)},'
                f'"concise_reason":{json.dumps("valid " * 100)}}}',
                candidate,
                judge_id="independent-judge",
                judge_model="model/judge",
            )
            self.assertTrue(review.accepted)
            self.assertEqual(review.relevance_score_1_5, 5)
            self.assertLessEqual(len(review.concise_reason), 240)

    def test_search_validator_caches_exhausted_parse_failure_as_rejection(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            (target,),
            ("fake",),
            requested_candidate_count=1,
        )[0]
        candidate = generate_question_candidates(
            (task,), FakeReadinessQuestionGenerator("fake")
        )[0]
        ranker = _StaticRanker(["not json", "still not json"])
        with tempfile.TemporaryDirectory() as temporary:
            validator = LocalSearchQuestionValidator(
                ranker,
                judge_id="independent-judge",
                model_name="model/judge",
                cache_directory=temporary,
                maximum_attempts=2,
            )
            first = validator.review(candidate)
            second = validator.review(candidate)
            cache_payload = json.loads(next(Path(temporary).glob("*.json")).read_text())
        self.assertFalse(first.accepted)
        self.assertEqual(first, second)
        self.assertEqual(ranker.call_count, 2)
        self.assertTrue(cache_payload["terminal_parse_failure"])
        self.assertEqual(len(cache_payload["failures"]), 2)

    def test_global_spatial_matching_can_reassign_candidates_to_better_cells(self) -> None:
        targets = build_target_grid(self.bounds)[:2]
        tasks = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            targets,
            ("fake-a", "fake-b"),
            requested_candidate_count=1,
        )
        candidates = tuple(
            generate_question_candidates(
                (task,), FakeReadinessQuestionGenerator(task.generator_id)
            )[0]
            for task in tasks
        )
        coordinates = {
            candidates[0].candidate_id: {
                "reference_normalized_axis_1": targets[1].normalized_axis_1,
                "reference_normalized_axis_2": targets[1].normalized_axis_2,
                "candidate_aligned_normalized_axis_1": targets[1].normalized_axis_1,
                "candidate_aligned_normalized_axis_2": targets[1].normalized_axis_2,
                "consensus_normalized_axis_1": targets[1].normalized_axis_1,
                "consensus_normalized_axis_2": targets[1].normalized_axis_2,
                "cross_embedding_disagreement": 0.0,
            },
            candidates[1].candidate_id: {
                "reference_normalized_axis_1": targets[0].normalized_axis_1,
                "reference_normalized_axis_2": targets[0].normalized_axis_2,
                "candidate_aligned_normalized_axis_1": targets[0].normalized_axis_1,
                "candidate_aligned_normalized_axis_2": targets[0].normalized_axis_2,
                "consensus_normalized_axis_1": targets[0].normalized_axis_1,
                "consensus_normalized_axis_2": targets[0].normalized_axis_2,
                "cross_embedding_disagreement": 0.0,
            },
        }
        selected, diagnostics = select_spatially_matched_questions(
            candidates,
            targets,
            coordinates,
            accepted_candidate_ids={row.candidate_id for row in candidates},
            disagreement_weight=0.0,
        )
        selected_by_target = {row.target_id: row for row in selected}
        self.assertEqual(
            selected_by_target[targets[0].target_id].candidate_id,
            candidates[1].candidate_id,
        )
        self.assertEqual(
            selected_by_target[targets[1].target_id].candidate_id,
            candidates[0].candidate_id,
        )
        self.assertAlmostEqual(diagnostics["mean_target_distance"], 0.0)

    def test_generated_question_projections_compare_with_frozen_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            battery = root / "battery"
            for directory in (reference, candidate, battery):
                directory.mkdir()
            atomic_json(reference / "projection_manifest.json", {"map_id": "qwen-map"})
            atomic_json(candidate / "projection_manifest.json", {"map_id": "mistral-map"})
            atomic_json(
                battery / "battery_manifest.json",
                {
                    "reference_map_id": "qwen-map",
                    "candidate_map_id": "mistral-map",
                },
            )
            atomic_json(
                battery / "readiness_robustness_battery.json",
                {
                    "cross_embedding_alignment": {
                        "reference_development_mean": [0.0, 0.0],
                        "reference_development_scale": [1.0, 1.0],
                        "candidate_development_mean": [0.0, 0.0],
                        "candidate_development_scale": [1.0, 1.0],
                        "orthogonal_rotation": [[1.0, 0.0], [0.0, 1.0]],
                    }
                },
            )
            rows = []
            for index in range(3):
                projection = {
                    "item_id": f"candidate:{index}",
                    "text_sha256": f"hash:{index}",
                    "raw_axis_1": float(index),
                    "raw_axis_2": float(2 - index),
                    "normalized_axis_1": float(index / 2),
                    "normalized_axis_2": float(1 - index / 2),
                    "predicted_scalar_readiness_0_1": float(index / 2),
                }
                rows.append(
                    {
                        "candidate_id": f"candidate:{index}",
                        "projection": projection,
                    }
                )
            atomic_jsonl(reference / "question_projections.jsonl", rows)
            atomic_jsonl(candidate / "question_projections.jsonl", rows)

            output = root / "comparison"
            _compare_projections(
                SimpleNamespace(
                    reference_projections=str(reference),
                    candidate_projections=str(candidate),
                    robustness_battery=str(battery),
                    output_dir=str(output),
                )
            )

            summary = read_json(output / "projection_comparison.json")
            self.assertAlmostEqual(summary["axis_1"]["spearman"], 1.0)
            self.assertAlmostEqual(summary["axis_2"]["spearman"], 1.0)
            self.assertTrue((output / "comparison_manifest.json").is_file())

    def test_spatial_select_stage_uses_validation_and_global_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = root / "plan"
            reference = root / "reference"
            candidate_root = root / "candidate"
            battery = root / "battery"
            for directory in (plan, reference, candidate_root, battery):
                directory.mkdir()
            targets = tuple(build_target_grid(self.bounds)[index] for index in (0, 5, 10))
            tasks = build_generation_tasks(
                (("keyword:one", "abandoned cart recovery"),),
                targets,
                ("fake",),
                requested_candidate_count=1,
            )
            candidates = tuple(
                generate_question_candidates(
                    (task,), FakeReadinessQuestionGenerator("fake")
                )[0]
                for task in tasks
            )
            candidate_file = root / "candidates.jsonl"
            atomic_jsonl(
                candidate_file,
                (
                    {
                        field: getattr(candidate, field)
                        for field in candidate.__dataclass_fields__
                    }
                    for candidate in candidates
                ),
            )
            atomic_json(plan / "subspace_bounds.json", {
                field: getattr(self.bounds, field)
                for field in self.bounds.__dataclass_fields__
            })
            atomic_jsonl(
                plan / "target_grid.jsonl",
                (
                    {field: getattr(target, field) for field in target.__dataclass_fields__}
                    for target in targets
                ),
            )
            atomic_json(plan / "plan_manifest.json", {"map_id": "qwen-map"})
            atomic_json(reference / "projection_manifest.json", {"map_id": "qwen-map"})
            atomic_json(candidate_root / "projection_manifest.json", {"map_id": "mistral-map"})
            atomic_json(
                battery / "battery_manifest.json",
                {"reference_map_id": "qwen-map", "candidate_map_id": "mistral-map"},
            )
            atomic_json(
                battery / "readiness_robustness_battery.json",
                {
                    "cross_embedding_alignment": {
                        "reference_development_mean": [0.0, 0.0],
                        "reference_development_scale": [1.0, 1.0],
                        "candidate_development_mean": [0.0, 0.0],
                        "candidate_development_scale": [1.0, 1.0],
                        "orthogonal_rotation": [[1.0, 0.0], [0.0, 1.0]],
                    }
                },
            )
            projected_rows = []
            for index, (candidate, target) in enumerate(
                zip(candidates, (targets[2], targets[0], targets[1]))
            ):
                projection = {
                    "item_id": candidate.candidate_id,
                    "text_sha256": candidate.question_sha256,
                    "raw_axis_1": target.raw_axis_1,
                    "raw_axis_2": target.raw_axis_2,
                    "normalized_axis_1": target.normalized_axis_1,
                    "normalized_axis_2": target.normalized_axis_2,
                    "predicted_scalar_readiness_0_1": index / 2,
                }
                projected_rows.append(
                    {"candidate_id": candidate.candidate_id, "projection": projection}
                )
            atomic_jsonl(reference / "question_projections.jsonl", projected_rows)
            atomic_jsonl(candidate_root / "question_projections.jsonl", projected_rows)
            validation = root / "validation.jsonl"
            atomic_jsonl(
                validation,
                (
                    {"candidate_id": candidate.candidate_id, "accepted": True}
                    for candidate in candidates
                ),
            )

            output = root / "spatial"
            _spatial_select(
                SimpleNamespace(
                    plan_dir=str(plan),
                    candidates=[str(candidate_file)],
                    reference_projections=str(reference),
                    candidate_projections=str(candidate_root),
                    robustness_battery=str(battery),
                    validations=[str(validation)],
                    generator_ids="fake",
                    next_round_index=1,
                    distance_tolerance=0.22,
                    disagreement_weight=0.10,
                    candidates_per_task=1,
                    master_seed=20260820,
                    output_dir=str(output),
                )
            )
            diagnostics = read_json(output / "spatial_coverage_diagnostics.json")
            self.assertEqual(diagnostics["selected_count"], 3)
            self.assertAlmostEqual(diagnostics["mean_target_distance"], 0.0)
            self.assertTrue((output / "run_manifest.json").is_file())


def _map() -> ReadinessEmbeddingMap:
    return ReadinessEmbeddingMap(
        map_id="readiness-map:test",
        map_version=READINESS_MAP_VERSION,
        embedding_model="synthetic",
        dimension=3,
        ridge_penalty=1.0,
        training_item_count=10,
        embedding_mean=(0.0, 0.0, 0.0),
        label_mean=0.5,
        scalar_direction=(1.0, 0.0, 0.0),
        scalar_unit_direction=(1.0, 0.0, 0.0),
        ordinal_boundaries_0_1=(0.125, 0.375, 0.625, 0.875),
        ordinal_plane_offsets=(0.125, 0.375, 0.625, 0.875),
        rubric_names=("a", "b", "c", "d"),
        rubric_coefficient_matrix=((1.0, 0.0, 0.0, 0.0),) * 3,
        rubric_singular_values=(1.0, 0.5, 0.2),
        rubric_first_component_share=0.8,
        supervised_subspace_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ordinal_rubric_names=("a", "b", "c", "d"),
        ordinal_direction=(1.0, 0.0, 0.0),
        ordinal_unit_direction=(1.0, 0.0, 0.0),
        ordinal_thresholds_by_rubric=((0.1,),) * 4,
        ridge_ordinal_cosine_similarity=1.0,
        pca_method="synthetic",
        pca_random_seed=1,
        pca_axes=((1.0, 0.0, 0.0),),
        pca_explained_variance_ratio=(1.0,),
        ridge_pca_absolute_cosine_similarity=(1.0,),
        ordinal_pca_absolute_cosine_similarity=(1.0,),
    )


if __name__ == "__main__":
    unittest.main()
