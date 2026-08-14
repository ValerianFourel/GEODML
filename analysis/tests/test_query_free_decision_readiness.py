"""Contracts for the query-free LLM2Vec decision-readiness pilot."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from analysis.interpretability.pipeline.query_free_decision_readiness import (
    CONTENT_MARKER,
    REPRESENTATION_VIEWS,
    FakeQueryFreeObjectiveGenerator,
    build_generation_requests,
    build_ordinal_judge_tasks,
    build_pairwise_judge_tasks,
    fake_query_free_embeddings,
    fit_query_free_axis,
    generate_query_free_stimuli,
    load_query_free_specification,
    project_query_free_axis,
    query_free_contract_checks,
    representation_texts,
    stratified_random_a1_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "scripts" / "run_query_free_axis_pilot.py"


class QueryFreeDecisionReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contexts, cls.plans, cls.specification = load_query_free_specification()
        cls.requests = build_generation_requests(cls.contexts, cls.plans)
        cls.stimuli = generate_query_free_stimuli(
            cls.requests,
            generator=FakeQueryFreeObjectiveGenerator(),
        )

    def test_population_is_query_free_balanced_and_frozen(self) -> None:
        self.assertEqual(len(self.contexts), 64)
        self.assertEqual(len(self.plans), 4)
        self.assertEqual(len({item.macrodomain for item in self.contexts}), 8)
        self.assertEqual(len(self.requests), 896)
        self.assertEqual(
            sum(item.context.split == "development" for item in self.requests),
            560,
        )
        self.assertEqual(
            sum(item.context.split == "confirmation" for item in self.requests),
            336,
        )
        self.assertTrue(all("query" not in item.context.payload.casefold() for item in self.requests))
        self.assertEqual(self.specification["design"]["level_count"], 7)

    def test_a1_grids_are_deterministic_irregular_and_full_range(self) -> None:
        first = stratified_random_a1_grid(
            master_seed=20260817,
            context_id="health-01",
            plan_id="dev-direct",
        )
        repeat = stratified_random_a1_grid(
            master_seed=20260817,
            context_id="health-01",
            plan_id="dev-direct",
        )
        other_block = stratified_random_a1_grid(
            master_seed=20260817,
            context_id="health-02",
            plan_id="dev-direct",
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other_block)
        self.assertEqual((first[0], first[-1]), (0.0, 1.0))
        self.assertEqual(tuple(sorted(set(first))), first)
        self.assertGreater(min(b - a for a, b in zip(first, first[1:])), 0.04)

    def test_compilation_changes_only_objective_and_exposes_three_views(self) -> None:
        self.assertEqual(len(self.stimuli), 896)
        self.assertTrue(all(item.structural_valid for item in self.stimuli))
        self.assertEqual(
            len({item.stimulus_id for item in self.stimuli}),
            len(self.stimuli),
        )
        texts = representation_texts(self.stimuli)
        self.assertEqual(tuple(texts), REPRESENTATION_VIEWS)
        for item in self.stimuli:
            self.assertEqual(item.objective_clause.count(CONTENT_MARKER), 1)
            self.assertEqual(item.content_masked_text.count(CONTENT_MARKER), 2)
            self.assertNotIn(CONTENT_MARKER, item.full_content_text)
            self.assertIn(item.content_payload, item.full_content_text)
        blocks: dict[str, list] = {}
        for item in self.stimuli:
            blocks.setdefault(item.block_id, []).append(item)
        self.assertEqual(len(blocks), 128)
        for rows in blocks.values():
            self.assertEqual(len({item.content_payload for item in rows}), 1)
            self.assertEqual(len({item.compiler_signature for item in rows}), 1)
            self.assertEqual(len(rows), 7)

    def test_contract_flags_coordinate_leaks_and_off_axis_criteria(self) -> None:
        request = self.requests[3]
        failures = query_free_contract_checks(
            request,
            "Compare [CONTENT] at level 0.5 using price and first-party evidence.",
        )
        self.assertIn("numeric-coordinate-leak", failures)
        self.assertIn("off-axis-criterion", failures)
        missing_marker = query_free_contract_checks(request, "Compare alternatives.")
        self.assertIn("content-marker-count", missing_marker)

    def test_judge_exports_are_blinded_and_include_surface_controls(self) -> None:
        ordinal = build_ordinal_judge_tasks(self.stimuli)
        pairwise, codebook = build_pairwise_judge_tasks(self.stimuli)
        self.assertEqual(len(ordinal), 896)
        self.assertEqual(len(pairwise), 1280)
        self.assertEqual(set(codebook), {item.task_id for item in pairwise})
        self.assertTrue(
            any(item.comparison_kind == "same-a1-cross-plan" for item in pairwise)
        )
        for task in (*ordinal[:10], *pairwise[:10]):
            public = asdict(task)
            self.assertNotIn("assigned_a1", public)
            self.assertNotIn("expected_winner_stimulus_id", public)
        for task in pairwise:
            if task.comparison_kind == "same-a1-cross-plan":
                self.assertIsNone(codebook[task.task_id]["expected_winner_stimulus_id"])

    def test_blocked_fit_recovers_direction_without_reassigning_a1(self) -> None:
        development_indices = [
            index
            for index, item in enumerate(self.stimuli)
            if item.context_split == "development"
        ]
        development = tuple(self.stimuli[index] for index in development_indices)
        all_matrices = fake_query_free_embeddings(self.stimuli, noise=0.0)
        matrices = {
            view: matrix[development_indices] for view, matrix in all_matrices.items()
        }
        axis = fit_query_free_axis(
            development,
            matrices,
            embedding_model="fake-test-llm2vec",
        )
        self.assertGreater(abs(axis.shared_unit_direction[0]), 0.99)
        self.assertTrue(
            all(item.cosine_with_shared > 0.99 for item in axis.view_directions)
        )
        coordinates = project_query_free_axis(axis, development, matrices)
        assigned = {
            (item.stimulus_id, view): item.assigned_a1
            for item in development
            for view in REPRESENTATION_VIEWS
        }
        self.assertEqual(
            [item.assigned_a1 for item in coordinates],
            [assigned[(item.stimulus_id, item.representation_view)] for item in coordinates],
        )
        self.assertLess(
            np.mean([item.absolute_assigned_coordinate_error for item in coordinates]),
            0.002,
        )

    def test_incompatible_views_are_reported_instead_of_forced_to_agree(self) -> None:
        contexts = tuple(item for item in self.contexts if item.split == "development")[:2]
        plans = tuple(item for item in self.plans if item.split == "development")
        requests = build_generation_requests(contexts, plans)
        stimuli = generate_query_free_stimuli(
            requests,
            generator=FakeQueryFreeObjectiveGenerator(),
        )
        matrices = {}
        directions = {
            "intent-only": np.asarray([1.0, 0.0, 0.0, 0.0]),
            "content-masked": np.asarray([0.0, 1.0, 0.0, 0.0]),
            "full-content": np.asarray([1.0, 0.0, 0.0, 0.0]),
        }
        for view, direction in directions.items():
            rows = []
            for item in stimuli:
                baseline = np.asarray([0.0, 0.0, 4.0, 0.1])
                rows.append(baseline + item.assigned_a1 * direction)
            matrices[view] = np.asarray(rows)
        axis = fit_query_free_axis(
            stimuli,
            matrices,
            embedding_model="synthetic-disagreement",
        )
        cosines = {
            item.representation_view: item.cosine_with_shared
            for item in axis.view_directions
        }
        self.assertLess(cosines["content-masked"], 0.8)

    def test_prepare_cli_writes_non_scientific_reproducible_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "prepare",
                    "--output-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["scientific_result"])
            self.assertFalse(manifest["candidate_sets_bound"])
            self.assertFalse(manifest["reranking_outcomes_observed"])
            self.assertEqual(manifest["request_count"], 896)
            lines = (output / "generation_requests.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 896)


if __name__ == "__main__":
    unittest.main()
