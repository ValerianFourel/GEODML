"""Contracts for embedding-positioned query-bound A1 prompts."""

from __future__ import annotations

import math
import unittest

import numpy as np

from analysis.interpretability.pipeline.a1_embedding_axis import (
    A1EndpointProjection,
    QueryPriorA1Axis,
)
from analysis.interpretability.pipeline.a1_embedding_panel import (
    A1CandidateCoordinate,
    balanced_query_style_assignment,
    build_positioned_rows,
    measure_candidate_coordinates,
    randomize_positioned_schedule,
    render_candidate_for_measurement,
    select_embedding_trajectory,
)
from analysis.interpretability.pipeline.a1_prompt_manifold import A1Candidate


def _candidate(
    candidate_id: str,
    *,
    style_seed: int = 3,
    assigned_a1: float = 0.5,
    candidate_index: int = 0,
) -> A1Candidate:
    return A1Candidate(
        candidate_id=candidate_id,
        candidate_hash=f"hash-{candidate_id}",
        assigned_a1=assigned_a1,
        style_seed=style_seed,
        candidate_index=candidate_index,
        generation_seed=17,
        search_term="generation sentinel",
        search_objective_clause=f"objective {candidate_id}",
        prompt_template=(
            'Rank candidates for "{QUERY}". Objective: objective '
            f"{candidate_id}. Candidates: {{CANDIDATES}}. Return {{TOP_N}} IDs."
        ),
        generator_backend="fake",
        generator_model="fake",
        structural_valid=True,
        contract_failures=(),
    )


class A1EmbeddingPanelTests(unittest.TestCase):
    def test_query_level_style_assignment_is_seeded_and_balanced(self) -> None:
        queries = tuple(f"query {index}" for index in range(101))
        first = balanced_query_style_assignment(
            queries,
            (0, 1, 2, 3),
            master_seed=73,
        )
        repeat = balanced_query_style_assignment(
            queries,
            (0, 1, 2, 3),
            master_seed=73,
        )
        second = balanced_query_style_assignment(
            queries,
            (0, 1, 2, 3),
            master_seed=74,
        )

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, second)
        self.assertEqual({query for query, _style, _order in first}, set(queries))
        self.assertEqual([order for _query, _style, order in first], list(range(1, 102)))
        counts = [sum(style == value for _query, style, _order in first) for value in range(4)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_measurement_binds_query_and_fixed_measurement_inputs(self) -> None:
        rendered = render_candidate_for_measurement(
            _candidate("one"),
            "CRM for nonprofits",
        )

        self.assertIn("CRM for nonprofits", rendered)
        self.assertIn("[FROZEN CANDIDATE SET]", rendered)
        self.assertIn("Return 10 IDs", rendered)
        self.assertNotIn("{QUERY}", rendered)
        self.assertNotIn("{CANDIDATES}", rendered)
        self.assertNotIn("{TOP_N}", rendered)

    def test_matched_endpoint_projection_is_the_observed_coordinate(self) -> None:
        axis = QueryPriorA1Axis(
            axis_id="axis:test",
            axis_version="test",
            embedding_model="fake",
            dimension=2,
            direction=(1.0, 0.0),
            informational_anchor=-0.5,
            transactional_anchor=0.5,
            endpoint_pair_count=1,
            query_count=1,
            style_seeds=(3,),
        )
        endpoint = A1EndpointProjection(
            search_term="query",
            style_seed=3,
            informational_projection=-0.25,
            transactional_projection=0.75,
            projection_gap=1.0,
            informational_global_coordinate=0.25,
            transactional_global_coordinate=1.25,
        )
        raw_projections = (-0.25, 0.25, 0.75)
        embeddings = np.asarray(
            [
                (value, math.sqrt(1.0 - value**2))
                for value in raw_projections
            ]
        )
        candidates = tuple(
            _candidate(f"candidate-{index}", assigned_a1=1.0 - index / 2)
            for index in range(3)
        )

        coordinates = measure_candidate_coordinates(
            axis=axis,
            endpoint=endpoint,
            candidates=candidates,
            embeddings=embeddings,
        )

        np.testing.assert_allclose(
            [row.observed_a1 for row in coordinates],
            (0.0, 0.5, 1.0),
        )
        np.testing.assert_allclose(
            [row.global_a1 for row in coordinates],
            (0.25, 0.75, 1.25),
        )
        self.assertEqual(
            [row.generator_assigned_a1 for row in coordinates],
            [1.0, 0.5, 0.0],
        )

    def test_selection_uses_measured_coordinates_not_generator_labels(self) -> None:
        measured = (-0.2, 0.01, 0.24, 0.49, 0.76, 0.99, 1.2)
        coordinates = tuple(
            A1CandidateCoordinate(
                candidate_id=f"candidate-{index}",
                candidate_hash=f"hash-{index}",
                generator_assigned_a1=1.0 - index / 6,
                candidate_index=index,
                global_a1=value + 0.1,
                observed_a1=value,
            )
            for index, value in enumerate(measured)
        )

        selected = select_embedding_trajectory(coordinates, (0.0, 0.5, 1.0))

        self.assertEqual(
            [round(row.observed_a1, 2) for row in selected],
            [0.01, 0.49, 0.99],
        )
        self.assertEqual(
            [round(row.generator_assigned_a1, 3) for row in selected],
            [0.833, 0.5, 0.167],
        )

    def test_final_rows_retain_measured_and_proposal_coordinates_separately(self) -> None:
        candidates = (_candidate("low", assigned_a1=0.9), _candidate("high", assigned_a1=0.1))
        coordinates = (
            A1CandidateCoordinate("low", "hash-low", 0.9, 0, 0.1, 0.02),
            A1CandidateCoordinate("high", "hash-high", 0.1, 0, 1.1, 0.98),
        )

        rows = build_positioned_rows(
            search_term="API monitoring",
            style_seed=3,
            keyword_order=7,
            targets=(0.0, 1.0),
            selected_coordinates=coordinates,
            candidates_by_id={candidate.candidate_id: candidate for candidate in candidates},
            axis_id="axis:test",
        )

        self.assertEqual([row.observed_a1 for row in rows], [0.02, 0.98])
        self.assertEqual([row.source_generator_assigned_a1 for row in rows], [0.9, 0.1])
        self.assertTrue(
            all(row.query_bound_prompt_template.count("API monitoring") == 1 for row in rows)
        )
        self.assertTrue(all("{QUERY}" not in row.query_bound_prompt_template for row in rows))
        self.assertTrue(all("{CANDIDATES}" in row.query_bound_prompt_template for row in rows))

        scheduled = randomize_positioned_schedule(rows, master_seed=31)
        repeated = randomize_positioned_schedule(rows, master_seed=31)
        self.assertEqual(scheduled, repeated)
        self.assertEqual([row.schedule_order for row in scheduled], [1, 2])
        self.assertEqual(
            sorted(row.axis_order for row in scheduled),
            [1, 2],
        )
        self.assertEqual(
            sorted(row.within_keyword_order for row in scheduled),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
