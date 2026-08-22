"""Contracts for iterative question coverage of the readiness subspace."""

from __future__ import annotations

from dataclasses import asdict, replace
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
    audit_question_diversity,
    build_axis_1_target_grid,
    build_generation_tasks,
    build_refinement_tasks,
    build_support_aware_keyword_targets,
    build_target_grid,
    delexicalize_question,
    fit_reference_bounds,
    generate_question_candidates,
    parse_generated_question,
    parse_search_question_review,
    project_questions,
    render_generation_request,
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
    _dual_view_refinement_feedback,
    _generate,
    _plan,
    _read_plan_targets,
    _spatial_select,
    _task_row,
    _validate_candidates,
    QuestionGenerationExhaustedError,
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

    def test_axis_1_grid_has_thirty_smooth_targets_and_fixed_nuisance_axis(self) -> None:
        targets = build_axis_1_target_grid(self.bounds)

        self.assertEqual(len(targets), 30)
        self.assertEqual(
            [target.normalized_axis_1 for target in targets],
            list(np.linspace(0.0, 1.0, 30)),
        )
        self.assertEqual({target.normalized_axis_2 for target in targets}, {0.5})
        self.assertEqual(targets[0].target_id, "readiness-axis-1-target:000")
        self.assertEqual(targets[-1].target_id, "readiness-axis-1-target:029")

    def test_rounds_rotate_each_target_across_generator_models(self) -> None:
        targets = build_target_grid(self.bounds)[:2]
        keywords = (("keyword:one", "abandoned cart recovery"),)
        generators = ("gemma", "qwen", "mistral")
        round_zero = build_generation_tasks(keywords, targets, generators, round_index=0)
        round_one = build_generation_tasks(keywords, targets, generators, round_index=1)
        self.assertEqual(round_zero[0].generator_id, "gemma")
        self.assertEqual(round_one[0].generator_id, "qwen")
        self.assertNotEqual(round_zero[0].task_id, round_one[0].task_id)

    def test_support_targets_are_seeded_balanced_and_keyword_specific(self) -> None:
        coordinate_rows = []
        for axis_1_index in range(5):
            for axis_2_index in range(5):
                for repeat in range(3):
                    coordinate_rows.append(
                        {
                            "split": "development",
                            "usable_for_axis": True,
                            "axis_1": -0.8 + 0.4 * axis_1_index + repeat * 0.001,
                            "axis_2": -0.8 + 0.4 * axis_2_index + repeat * 0.001,
                        }
                    )
        coordinate_rows.extend(
            [
                {
                    "split": "confirmation",
                    "usable_for_axis": True,
                    "axis_1": 1e6,
                    "axis_2": 1e6,
                },
                {
                    "split": "development",
                    "usable_for_axis": False,
                    "axis_1": 0.99,
                    "axis_2": 0.99,
                },
            ]
        )
        keywords = tuple(
            (f"keyword:{index}", f"topic phrase {index}") for index in range(4)
        )
        first, diagnostics = build_support_aware_keyword_targets(
            coordinate_rows,
            self.bounds,
            keywords,
            targets_per_keyword=6,
            support_grid_resolution=5,
            minimum_support_bin_count=3,
            master_seed=17,
        )
        second, repeated_diagnostics = build_support_aware_keyword_targets(
            coordinate_rows,
            self.bounds,
            keywords,
            targets_per_keyword=6,
            support_grid_resolution=5,
            minimum_support_bin_count=3,
            master_seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(diagnostics, repeated_diagnostics)
        self.assertEqual(diagnostics["pooled_target_count"], 24)
        self.assertLessEqual(diagnostics["target_bin_count_range"], 1)
        self.assertEqual({len(values) for values in first.values()}, {6})
        self.assertTrue(
            all(
                len({(target.axis_1_index, target.axis_2_index) for target in values})
                == 6
                for values in first.values()
            )
        )
        self.assertNotEqual(first["keyword:0"], first["keyword:1"])
        self.assertTrue(
            all(
                0.0 <= coordinate <= 1.0
                for values in first.values()
                for target in values
                for coordinate in (
                    target.normalized_axis_1,
                    target.normalized_axis_2,
                )
            )
        )
        tasks = build_generation_tasks(
            keywords,
            first,
            ("qwen", "gemma"),
            requested_candidate_count=2,
        )
        self.assertEqual(len(tasks), 24)
        self.assertEqual(
            tasks[0].target,
            first[tasks[0].keyword_id][tasks[0].target.target_index],
        )

    def test_keyword_target_plan_supports_an_exact_pilot_keyword_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan = Path(temporary_directory)
            target = build_target_grid(self.bounds)[0]
            atomic_json(
                plan / "plan_manifest.json",
                {
                    "target_design": "support-aware-random",
                    "target_count_per_keyword": 1,
                },
            )
            atomic_jsonl(
                plan / "keyword_target_grid.jsonl",
                (
                    {
                        "keyword_id": keyword_id,
                        "keyword": keyword,
                        "target": asdict(target),
                    }
                    for keyword_id, keyword in (
                        ("keyword:pilot", "pilot topic"),
                        ("keyword:outside", "outside topic"),
                    )
                ),
            )

            resolved, design = _read_plan_targets(
                plan,
                (("keyword:pilot", "pilot topic"),),
            )

            self.assertEqual(design, "support-aware-random")
            self.assertEqual(set(resolved), {"keyword:pilot"})
            self.assertEqual(
                tuple(asdict(value) for value in resolved["keyword:pilot"]),
                (asdict(target),),
            )

            with self.assertRaisesRegex(
                ValueError,
                "keyword target plan does not match candidate keywords",
            ):
                _read_plan_targets(
                    plan,
                    (("keyword:pilot", "wrong topic"),),
                )

    def test_support_generation_request_exposes_continuous_control(self) -> None:
        target = replace(
            build_target_grid(self.bounds)[0],
            target_id="readiness-support-target:000",
            normalized_axis_1=0.37,
            normalized_axis_2=0.62,
        )
        task = build_generation_tasks(
            (("keyword:one", "enterprise password manager"),),
            (target,),
            ("qwen",),
        )[0]
        first = render_generation_request(task, candidate_slot=0)
        second = render_generation_request(task, candidate_slot=1)
        self.assertIn("0.370 on a 0-to-1 continuum", first)
        self.assertIn("0.620 on a 0-to-1 continuum", first)
        self.assertIn("graded semantic mixture", first)
        self.assertNotEqual(first, second)

        shifted_task = build_generation_tasks(
            (("keyword:one", "enterprise password manager"),),
            (replace(target, normalized_axis_1=0.38),),
            ("qwen",),
        )[0]
        self.assertNotEqual(task.task_id, shifted_task.task_id)

        legacy_target = replace(target, target_id="readiness-cell:00-00")
        legacy_task = replace(task, target=legacy_target)
        legacy = render_generation_request(legacy_task, candidate_slot=0)
        self.assertNotIn("0.370 on a 0-to-1 continuum", legacy)

    def test_axis_1_generation_request_does_not_control_axis_2(self) -> None:
        target = build_axis_1_target_grid(self.bounds)[11]
        task = build_generation_tasks(
            (("keyword:one", "enterprise password manager"),),
            (target,),
            ("qwen",),
        )[0]

        request = render_generation_request(task, candidate_slot=0)

        self.assertIn(f"{target.normalized_axis_1:.3f} on a 0-to-1 continuum", request)
        self.assertIn("Control only readiness stage", request)
        self.assertIn("Decision mode: unconstrained", request)

    def test_diversity_audit_detects_keyword_substitution_templates(self) -> None:
        rows = [
            {
                "keyword_id": f"keyword:{index}",
                "keyword": f"topic phrase {index}",
                "question": (
                    f"What should I know about topic phrase {index} before "
                    "choosing a practical approach?"
                ),
            }
            for index in range(10)
        ]
        diagnostics = audit_question_diversity(
            rows,
            minimum_delexicalized_unique_fraction=0.90,
            maximum_template_fraction=0.20,
            minimum_median_keyword_unique_fraction=1.0,
            minimum_keyword_unique_fraction=1.0,
            maximum_opening_frame_fraction=0.20,
        )
        self.assertEqual(diagnostics["delexicalized_template_count"], 1)
        self.assertFalse(diagnostics["all_checks_passed"])
        self.assertEqual(
            delexicalize_question(rows[0]["question"], rows[0]["keyword"]),
            delexicalize_question(rows[1]["question"], rows[1]["keyword"]),
        )

    def test_diversity_audit_accepts_distinct_frames(self) -> None:
        rows = [
            {
                "keyword_id": "keyword:one",
                "keyword": "enterprise password manager",
                "question": question,
            }
            for question in (
                "Which evidence explains how an enterprise password manager affects access controls?",
                "Before deployment, what should a team verify about an enterprise password manager?",
                "How could an enterprise password manager be configured for a distributed workforce?",
                "When comparing options, which enterprise password manager criteria matter most?",
            )
        ]
        diagnostics = audit_question_diversity(
            rows,
            maximum_template_fraction=0.25,
            maximum_opening_frame_fraction=0.25,
            minimum_median_keyword_unique_fraction=1.0,
            minimum_keyword_unique_fraction=1.0,
        )
        self.assertTrue(diagnostics["all_checks_passed"])
        self.assertEqual(diagnostics["delexicalized_unique_fraction"], 1.0)

    def test_support_design_scales_to_thirty_thousand_uniform_targets(self) -> None:
        coordinate_rows = [
            {
                "split": "development",
                "usable_for_axis": True,
                "axis_1": -0.95 + 0.10 * axis_1_index + repeat * 0.0001,
                "axis_2": -0.95 + 0.10 * axis_2_index + repeat * 0.0001,
            }
            for axis_1_index in range(20)
            for axis_2_index in range(20)
            for repeat in range(3)
        ]
        keywords = tuple(
            (f"keyword:{index:04d}", f"topic phrase {index:04d}")
            for index in range(1_000)
        )
        targets, diagnostics = build_support_aware_keyword_targets(
            coordinate_rows,
            self.bounds,
            keywords,
            targets_per_keyword=30,
            support_grid_resolution=20,
            minimum_support_bin_count=3,
            master_seed=20260820,
        )
        self.assertEqual(len(targets), 1_000)
        self.assertEqual(sum(map(len, targets.values())), 30_000)
        self.assertEqual(diagnostics["pooled_target_count"], 30_000)
        self.assertEqual(diagnostics["eligible_support_bin_count"], 400)
        self.assertEqual(diagnostics["minimum_targets_per_eligible_bin"], 75)
        self.assertEqual(diagnostics["maximum_targets_per_eligible_bin"], 75)
        self.assertEqual(diagnostics["target_bin_count_range"], 0)

    def test_support_design_can_request_more_targets_than_support_bins(self) -> None:
        coordinate_rows = [
            {
                "split": "development",
                "usable_for_axis": True,
                "axis_1": -0.75 + 0.5 * axis_1_index + repeat * 0.001,
                "axis_2": -0.75 + 0.5 * axis_2_index + repeat * 0.001,
            }
            for axis_1_index in range(4)
            for axis_2_index in range(4)
            for repeat in range(3)
        ]
        targets, diagnostics = build_support_aware_keyword_targets(
            coordinate_rows,
            self.bounds,
            (("keyword:one", "topic phrase one"),),
            targets_per_keyword=40,
            support_grid_resolution=4,
            minimum_support_bin_count=3,
            master_seed=11,
        )
        cell_counts = {}
        for target in targets["keyword:one"]:
            cell = (target.axis_1_index, target.axis_2_index)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
        self.assertEqual(len(targets["keyword:one"]), 40)
        self.assertEqual(len(cell_counts), 16)
        self.assertLessEqual(max(cell_counts.values()) - min(cell_counts.values()), 1)
        self.assertLessEqual(diagnostics["target_bin_count_range"], 1)

    def test_support_aware_plan_writes_per_keyword_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            keywords_path = root / "keywords.txt"
            keywords_path.write_text("topic alpha\ntopic beta\n", encoding="utf-8")
            map_path = root / "map.json"
            atomic_json(
                map_path,
                {
                    field: getattr(_map(), field)
                    for field in _map().__dataclass_fields__
                },
            )
            coordinate_path = root / "coordinates.jsonl"
            atomic_jsonl(
                coordinate_path,
                (
                    {
                        "split": "development",
                        "usable_for_axis": True,
                        "axis_1": -0.9 + axis_1_index * 0.2 + repeat * 0.001,
                        "axis_2": -0.9 + axis_2_index * 0.2 + repeat * 0.001,
                    }
                    for axis_1_index in range(10)
                    for axis_2_index in range(10)
                    for repeat in range(3)
                ),
            )
            output = root / "plan"
            _plan(
                SimpleNamespace(
                    keywords=str(keywords_path),
                    map=str(map_path),
                    reference_coordinates=str(coordinate_path),
                    generator_ids="qwen,gemma",
                    output_dir=str(output),
                    axis_1_points=6,
                    axis_2_points=5,
                    target_design="support-aware-random",
                    targets_per_keyword=30,
                    support_grid_resolution=10,
                    minimum_support_bin_count=3,
                    support_include_unusable=False,
                    lower_quantile=0.0,
                    upper_quantile=1.0,
                    reference_split="development",
                    round_index=0,
                    candidates_per_task=2,
                    master_seed=20260820,
                )
            )
            manifest = read_json(output / "plan_manifest.json")
            diagnostics = read_json(output / "support_design.json")
            self.assertEqual(manifest["target_design"], "support-aware-random")
            self.assertEqual(manifest["task_count"], 60)
            self.assertEqual(manifest["requested_candidates_per_task"], 2)
            self.assertEqual(manifest["maximum_planned_candidate_count"], 120)
            self.assertEqual(
                len((output / "keyword_target_grid.jsonl").read_text().splitlines()),
                60,
            )
            self.assertEqual(
                len(
                    (output / "generation_tasks_round_00.jsonl")
                    .read_text()
                    .splitlines()
                ),
                60,
            )
            self.assertLessEqual(diagnostics["target_bin_count_range"], 1)
            self.assertFalse((output / "target_grid.jsonl").exists())

    def test_axis_1_plan_writes_30330_single_candidate_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            keywords_path = root / "keywords.txt"
            keywords_path.write_text(
                "".join(f"topic {index}\n" for index in range(1011)),
                encoding="utf-8",
            )
            map_path = root / "map.json"
            atomic_json(
                map_path,
                {
                    field: getattr(_map(), field)
                    for field in _map().__dataclass_fields__
                },
            )
            coordinate_path = root / "coordinates.jsonl"
            atomic_jsonl(
                coordinate_path,
                (
                    {
                        "split": "development",
                        "usable_for_axis": True,
                        "axis_1": -1.0 + index * 0.1,
                        "axis_2": 1.0 - index * 0.1,
                    }
                    for index in range(20)
                ),
            )
            output = root / "axis-1-plan"

            _plan(
                SimpleNamespace(
                    keywords=str(keywords_path),
                    map=str(map_path),
                    reference_coordinates=str(coordinate_path),
                    generator_ids="qwen,gemma",
                    output_dir=str(output),
                    axis_1_points=6,
                    axis_2_points=5,
                    target_design="axis-1-linear",
                    targets_per_keyword=30,
                    support_grid_resolution=10,
                    minimum_support_bin_count=3,
                    support_include_unusable=False,
                    lower_quantile=0.0,
                    upper_quantile=1.0,
                    reference_split="development",
                    round_index=0,
                    candidates_per_task=1,
                    master_seed=20260820,
                )
            )

            manifest = read_json(output / "plan_manifest.json")
            targets = [
                json.loads(line)
                for line in (output / "target_grid.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest["target_design"], "axis-1-linear")
            self.assertEqual(manifest["generation_control"], "continuous-axis-1-only-v1")
            self.assertEqual(manifest["task_count"], 30330)
            self.assertEqual(manifest["maximum_planned_candidate_count"], 30330)
            self.assertEqual(len(targets), 30)
            self.assertEqual({row["normalized_axis_2"] for row in targets}, {0.5})

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

    def test_spatial_matching_accepts_keyword_specific_targets(self) -> None:
        base_targets = build_target_grid(self.bounds)
        targets_by_keyword = {
            "keyword:one": (base_targets[0], base_targets[1]),
            "keyword:two": (
                replace(
                    base_targets[2],
                    target_id=base_targets[0].target_id,
                    target_index=0,
                ),
                replace(
                    base_targets[3],
                    target_id=base_targets[1].target_id,
                    target_index=1,
                ),
            ),
        }
        keywords = (
            ("keyword:one", "topic phrase one"),
            ("keyword:two", "topic phrase two"),
        )
        tasks = build_generation_tasks(
            keywords,
            targets_by_keyword,
            ("fake",),
            requested_candidate_count=1,
        )
        candidates = tuple(
            generate_question_candidates(
                (task,), FakeReadinessQuestionGenerator("fake")
            )[0]
            for task in tasks
        )
        coordinates = {}
        for candidate in candidates:
            target = targets_by_keyword[candidate.keyword_id][candidate.target_index]
            coordinates[candidate.candidate_id] = {
                "reference_normalized_axis_1": target.normalized_axis_1,
                "reference_normalized_axis_2": target.normalized_axis_2,
                "candidate_aligned_normalized_axis_1": target.normalized_axis_1,
                "candidate_aligned_normalized_axis_2": target.normalized_axis_2,
                "consensus_normalized_axis_1": target.normalized_axis_1,
                "consensus_normalized_axis_2": target.normalized_axis_2,
                "cross_embedding_disagreement": 0.0,
            }
        selected, diagnostics = select_spatially_matched_questions(
            candidates,
            targets_by_keyword,
            coordinates,
            accepted_candidate_ids={row.candidate_id for row in candidates},
            target_design="support-aware-random",
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(diagnostics["target_design"], "support-aware-random")
        self.assertEqual(diagnostics["target_count_per_keyword"], 2)
        self.assertTrue(all(row.target_distance == 0.0 for row in selected))
        self.assertTrue(
            diagnostics["pooled_support_coverage"]["spacing_gate_passed"]
        )
        self.assertTrue(diagnostics["overall_spacing_gate_passed"])

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

    def test_validation_shards_are_disjoint_and_cover_every_candidate(self) -> None:
        targets = build_target_grid(self.bounds)[:6]
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
        accepted_output = (
            '{"topic_relevant":true,"search_intent":true,'
            '"web_answerable":true,"standalone":true,'
            '"natural_language":true,"relevance_score_1_5":5,'
            '"concise_reason":"A direct, answerable search question."}'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_path = root / "candidates.jsonl"
            atomic_jsonl(candidate_path, (asdict(candidate) for candidate in candidates))
            outputs = []
            from unittest.mock import patch

            for shard_index in range(2):
                output = root / f"validation-{shard_index}.jsonl"
                outputs.append(output)
                validator = LocalSearchQuestionValidator(
                    _StaticRanker([accepted_output] * len(candidates)),
                    judge_id="independent-judge",
                    model_name="model/judge",
                    cache_directory=root / f"cache-{shard_index}",
                )
                args = SimpleNamespace(
                    output=str(output),
                    resume=False,
                    candidates=[str(candidate_path)],
                    shard_count=2,
                    shard_index=shard_index,
                    model="model/judge",
                    judge_id="independent-judge",
                    cache_dir=str(root / f"cache-{shard_index}"),
                    backend="local",
                    precision="full",
                    maximum_attempts=3,
                )
                with patch(
                    "analysis.scripts.build_readiness_prompt_population."
                    "LocalSearchQuestionValidator.from_model",
                    return_value=validator,
                ):
                    self.assertEqual(_validate_candidates(args), 0)

            shard_ids = [
                {
                    json.loads(line)["candidate_id"]
                    for line in output.read_text().splitlines()
                    if line.strip()
                }
                for output in outputs
            ]
            manifests = [
                json.loads(
                    output.with_suffix(".jsonl.manifest.json").read_text()
                )
                for output in outputs
            ]

        self.assertFalse(shard_ids[0] & shard_ids[1])
        self.assertEqual(
            shard_ids[0] | shard_ids[1],
            {candidate.candidate_id for candidate in candidates},
        )
        self.assertEqual([row["shard_index"] for row in manifests], [0, 1])
        self.assertTrue(
            all(row["total_candidate_count"] == len(candidates) for row in manifests)
        )

    def test_generator_atomically_resumes_after_each_accepted_question(self) -> None:
        target = build_target_grid(self.bounds)[0]
        task = build_generation_tasks(
            (("keyword:one", "enterprise password manager"),),
            (target,),
            ("qwen",),
            requested_candidate_count=2,
        )[0]
        first_question = (
            "What evidence should a team examine about an enterprise password manager "
            "before comparing approaches?"
        )
        second_question = (
            "How can an enterprise password manager support a practical access-control "
            "implementation plan?"
        )
        with tempfile.TemporaryDirectory() as temporary:
            interrupted = LocalReadinessQuestionGenerator(
                _StaticRanker(
                    [
                        json.dumps({"question": first_question}),
                        "invalid output",
                    ]
                ),
                generator_id="qwen",
                model_name="model/qwen",
                cache_directory=temporary,
                maximum_attempts=1,
            )
            with self.assertRaises(RuntimeError):
                interrupted.generate(task)
            partial = json.loads(next(Path(temporary).glob("*.json")).read_text())
            self.assertEqual(partial["questions"], [first_question])
            self.assertFalse(partial["complete"])
            self.assertTrue(partial["terminal_failure"])

            resumed_ranker = _StaticRanker([json.dumps({"question": second_question})])
            resumed = LocalReadinessQuestionGenerator(
                resumed_ranker,
                generator_id="qwen",
                model_name="model/qwen",
                cache_directory=temporary,
                maximum_attempts=1,
            )
            self.assertEqual(resumed.generate(task), (first_question, second_question))
            self.assertEqual(resumed_ranker.call_count, 1)
            complete = json.loads(next(Path(temporary).glob("*.json")).read_text())
            self.assertTrue(complete["complete"])
            self.assertFalse(complete["terminal_failure"])

    def test_generate_can_record_one_exhausted_task_and_finish_its_slice(self) -> None:
        targets = build_target_grid(self.bounds)[:2]
        tasks = build_generation_tasks(
            (("keyword:one", "abandoned cart recovery"),),
            targets,
            ("fake",),
            requested_candidate_count=1,
        )
        original_generate = generate_question_candidates

        def generate_or_fail(selected_tasks, generator):
            if selected_tasks[0].task_id == tasks[0].task_id:
                raise QuestionGenerationExhaustedError(
                    "question lost the exact keyword phrase"
                )
            return original_generate(selected_tasks, generator)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_path = root / "tasks.jsonl"
            output = root / "candidates.jsonl"
            atomic_jsonl(task_path, (_task_row(task) for task in tasks))
            args = SimpleNamespace(
                output=str(output),
                resume=False,
                tasks=str(task_path),
                generator_id="fake",
                start_index=0,
                limit=None,
                shard_count=1,
                shard_index=0,
                maximum_runtime_seconds=None,
                backend="fake",
                model="fake/model",
                cache_dir=str(root / "cache"),
                precision="full",
                max_new_tokens=180,
                temperature=0.9,
                maximum_attempts=1,
                allow_failed_tasks=True,
            )
            from unittest.mock import patch

            with patch(
                "analysis.scripts.build_readiness_prompt_population.generate_question_candidates",
                side_effect=generate_or_fail,
            ):
                self.assertEqual(_generate(args), 0)

            candidates = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failures = [
                json.loads(line)
                for line in output.with_suffix(".jsonl.failures.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            manifest = json.loads(
                output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual([row["task_id"] for row in failures], [tasks[0].task_id])
        self.assertEqual(manifest["completed_task_count"], 1)
        self.assertEqual(manifest["failed_task_count"], 1)
        self.assertFalse(manifest["slice_complete"])
        self.assertTrue(manifest["slice_terminal"])

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
            cache_path = next(Path(temporary).glob("*.json"))
            cache_payload = json.loads(cache_path.read_text())
        self.assertFalse(first.accepted)
        self.assertEqual(first, second)
        self.assertEqual(ranker.call_count, 2)
        self.assertTrue(cache_payload["terminal_parse_failure"])
        self.assertEqual(len(cache_payload["failures"]), 2)

    def test_search_validator_recovers_observed_byte_level_json_cache(self) -> None:
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
        observed = (
            '```jsonĊ{ĊĠĠ"topic_relevant":Ġtrue,Ċ'
            'ĠĠ"search_intent":Ġtrue,Ċ'
            'ĠĠ**"web_answerable":Ġtrue,Ċ'
            'ĠĠ"standalone":Ġtrue,Ċ'
            'ĠĠ"natural_language":Ġtrue,Ċ'
            'ĠĠ"relevance_score_1_5":Ġ5,Ċ'
            'ĠĠ"concise_reason":Ġ"DirectlyĠanswerableĠonline."Ċ}Ċ```'
        )
        ranker = _StaticRanker(["not json"])
        with tempfile.TemporaryDirectory() as temporary:
            validator = LocalSearchQuestionValidator(
                ranker,
                judge_id="independent-judge",
                model_name="model/judge",
                cache_directory=temporary,
                maximum_attempts=1,
            )
            rejected = validator.review(candidate)
            cache_path = next(Path(temporary).glob("*.json"))
            payload = json.loads(cache_path.read_text())
            payload["failures"][-1]["raw"] = observed
            cache_path.write_text(json.dumps(payload))

            recovered = validator.review(candidate)
            recovered_payload = json.loads(cache_path.read_text())

        self.assertFalse(rejected.accepted)
        self.assertTrue(recovered.accepted)
        self.assertTrue(recovered_payload["recovered_terminal_parse_failure"])
        self.assertEqual(ranker.call_count, 1)

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

    def test_strict_dual_view_gate_rejects_consensus_cancellation(self) -> None:
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
        coordinates = {
            candidate.candidate_id: {
                "reference_normalized_axis_1": target.normalized_axis_1 + 0.30,
                "reference_normalized_axis_2": target.normalized_axis_2,
                "candidate_aligned_normalized_axis_1": (
                    target.normalized_axis_1 - 0.30
                ),
                "candidate_aligned_normalized_axis_2": target.normalized_axis_2,
                "consensus_normalized_axis_1": target.normalized_axis_1,
                "consensus_normalized_axis_2": target.normalized_axis_2,
                "cross_embedding_disagreement": 0.60,
            }
        }

        loose, loose_diagnostics = select_spatially_matched_questions(
            (candidate,),
            (target,),
            coordinates,
            accepted_candidate_ids={candidate.candidate_id},
            disagreement_weight=0.0,
            distance_tolerance=0.22,
        )
        strict, strict_diagnostics = select_spatially_matched_questions(
            (candidate,),
            (target,),
            coordinates,
            accepted_candidate_ids={candidate.candidate_id},
            disagreement_weight=0.0,
            distance_tolerance=0.22,
            require_both_views_within_tolerance=True,
        )

        self.assertEqual(len(loose), 1)
        self.assertAlmostEqual(loose[0].target_distance, 0.0)
        self.assertFalse(loose[0].both_views_within_tolerance)
        self.assertEqual(loose_diagnostics["verified_selected_count"], 0)
        self.assertEqual(strict, ())
        self.assertEqual(strict_diagnostics["verified_selected_count"], 0)
        self.assertTrue(
            strict_diagnostics["require_both_views_within_tolerance"]
        )

    def test_strict_dual_view_gate_records_independent_passes(self) -> None:
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
        coordinates = {
            candidate.candidate_id: {
                "reference_normalized_axis_1": target.normalized_axis_1 + 0.10,
                "reference_normalized_axis_2": target.normalized_axis_2,
                "candidate_aligned_normalized_axis_1": (
                    target.normalized_axis_1 + 0.12
                ),
                "candidate_aligned_normalized_axis_2": target.normalized_axis_2,
                "consensus_normalized_axis_1": target.normalized_axis_1 + 0.11,
                "consensus_normalized_axis_2": target.normalized_axis_2,
                "cross_embedding_disagreement": 0.02,
            }
        }

        selected, diagnostics = select_spatially_matched_questions(
            (candidate,),
            (target,),
            coordinates,
            accepted_candidate_ids={candidate.candidate_id},
            disagreement_weight=0.0,
            distance_tolerance=0.22,
            require_both_views_within_tolerance=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0].both_views_within_tolerance)
        self.assertAlmostEqual(selected[0].reference_target_distance, 0.10)
        self.assertAlmostEqual(
            selected[0].candidate_aligned_target_distance, 0.12
        )
        self.assertEqual(diagnostics["verified_selected_count"], 1)
        self.assertEqual(diagnostics["verified_selected_fraction"], 1.0)

    def test_axis_1_matching_ignores_axis_2_but_requires_both_views(self) -> None:
        targets = build_axis_1_target_grid(self.bounds, axis_1_points=2)
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
        coordinates = {}
        for candidate, target, axis_2 in zip(candidates, targets, (20.0, -20.0)):
            coordinates[candidate.candidate_id] = {
                "reference_normalized_axis_1": target.normalized_axis_1,
                "reference_normalized_axis_2": axis_2,
                "candidate_aligned_normalized_axis_1": target.normalized_axis_1,
                "candidate_aligned_normalized_axis_2": -axis_2,
                "consensus_normalized_axis_1": target.normalized_axis_1,
                "consensus_normalized_axis_2": 0.0,
                "cross_embedding_disagreement": 40.0,
            }

        selected, diagnostics = select_spatially_matched_questions(
            candidates,
            targets,
            coordinates,
            accepted_candidate_ids={row.candidate_id for row in candidates},
            disagreement_weight=0.0,
            distance_tolerance=0.01,
            target_design="axis-1-linear",
            require_both_views_within_tolerance=True,
        )

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row.both_views_within_tolerance for row in selected))
        self.assertTrue(all(row.target_distance == 0.0 for row in selected))
        self.assertEqual(
            diagnostics["selection_method"],
            "global-axis-1-assignment-with-strict-dual-view-tolerance",
        )
        self.assertTrue(diagnostics["overall_spacing_gate_passed"])

    def test_template_uniqueness_rejects_keyword_substitution(self) -> None:
        target = build_target_grid(self.bounds)[0]
        tasks = build_generation_tasks(
            (
                ("keyword:one", "abandoned cart recovery"),
                ("keyword:two", "enterprise password manager"),
            ),
            (target,),
            ("fake",),
            requested_candidate_count=1,
        )
        candidates = tuple(
            generate_question_candidates(
                (task,), FakeReadinessQuestionGenerator("fake")
            )[0]
            for task in tasks
        )
        coordinates = {
            candidate.candidate_id: {
                "reference_normalized_axis_1": target.normalized_axis_1,
                "reference_normalized_axis_2": target.normalized_axis_2,
                "candidate_aligned_normalized_axis_1": target.normalized_axis_1,
                "candidate_aligned_normalized_axis_2": target.normalized_axis_2,
                "consensus_normalized_axis_1": target.normalized_axis_1,
                "consensus_normalized_axis_2": target.normalized_axis_2,
                "cross_embedding_disagreement": 0.0,
            }
            for candidate in candidates
        }

        selected, diagnostics = select_spatially_matched_questions(
            candidates,
            (target,),
            coordinates,
            accepted_candidate_ids={row.candidate_id for row in candidates},
            disagreement_weight=0.0,
            distance_tolerance=0.22,
            require_both_views_within_tolerance=True,
            require_delexicalized_template_uniqueness=True,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(diagnostics["template_duplicate_rejection_count"], 1)
        self.assertTrue(
            diagnostics["selected_delexicalized_templates_are_unique"]
        )
        self.assertTrue(
            diagnostics["require_delexicalized_template_uniqueness"]
        )

    def test_refinement_feedback_reports_both_measured_view_shifts(self) -> None:
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
        coordinates = {
            candidate.candidate_id: {
                "reference_normalized_axis_1": target.normalized_axis_1 + 0.30,
                "reference_normalized_axis_2": target.normalized_axis_2 + 0.10,
                "candidate_aligned_normalized_axis_1": (
                    target.normalized_axis_1 - 0.20
                ),
                "candidate_aligned_normalized_axis_2": (
                    target.normalized_axis_2 + 0.25
                ),
            }
        }

        feedback = _dual_view_refinement_feedback(
            target, (candidate,), coordinates
        )

        self.assertIn("frozen Qwen LLM2Vec", feedback)
        self.assertIn("development-aligned Mistral LLM2Vec", feedback)
        self.assertIn("Qwen-view shift needed", feedback)
        self.assertIn("aligned Mistral-view shift needed", feedback)
        self.assertIn("do not reuse this question frame", feedback)

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
                    require_both_views_within_tolerance=True,
                    require_delexicalized_template_uniqueness=True,
                    disagreement_weight=0.10,
                    candidates_per_task=1,
                    master_seed=20260820,
                    output_dir=str(output),
                )
            )
            diagnostics = read_json(output / "spatial_coverage_diagnostics.json")
            self.assertEqual(diagnostics["selected_count"], 3)
            self.assertAlmostEqual(diagnostics["mean_target_distance"], 0.0)
            self.assertEqual(diagnostics["verified_selected_count"], 3)
            self.assertTrue(
                diagnostics["selected_delexicalized_templates_are_unique"]
            )
            manifest = read_json(output / "run_manifest.json")
            self.assertTrue(manifest["coordinate_acceptance_contract"]["enabled"])
            self.assertTrue(manifest["surface_acceptance_contract"]["enabled"])


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
