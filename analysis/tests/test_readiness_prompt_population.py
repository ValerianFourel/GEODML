"""Contracts for iterative question coverage of the readiness subspace."""

from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_embedding_map import (
    READINESS_MAP_VERSION,
    ReadinessEmbeddingMap,
)
from analysis.interpretability.pipeline.readiness_prompt_population import (
    FakeReadinessQuestionGenerator,
    LocalReadinessQuestionGenerator,
    ReadinessQuestionCandidate,
    ReadinessSubspaceBounds,
    build_generation_tasks,
    build_refinement_tasks,
    build_target_grid,
    fit_reference_bounds,
    generate_question_candidates,
    parse_generated_question,
    project_questions,
    select_diverse_questions,
    validate_generated_question,
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
