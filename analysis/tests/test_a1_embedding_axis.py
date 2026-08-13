"""Contracts for the primary query-prior LLM2Vec A1 axis."""

from __future__ import annotations

import numpy as np
import unittest

from analysis.interpretability.pipeline.a1_embedding_axis import (
    INFORMATIONAL_OBJECTIVE,
    TRANSACTIONAL_OBJECTIVE,
    build_query_prior_endpoint_prompts,
    fit_query_prior_a1_axis,
    project_onto_query_prior_a1,
)


class A1EmbeddingAxisTests(unittest.TestCase):
    def test_endpoint_prompts_are_matched_full_query_bound_ranking_prompts(self) -> None:
        informational, transactional, keys = build_query_prior_endpoint_prompts(
            ("abandoned cart recovery", "CRM for nonprofits"),
            style_seeds=(2, 5),
        )

        self.assertEqual(len(informational), 4)
        self.assertEqual(len(transactional), 4)
        self.assertEqual(len(keys), 4)
        for left, right, (query, _style) in zip(informational, transactional, keys):
            self.assertEqual(left.count(query), 1)
            self.assertEqual(right.count(query), 1)
            self.assertIn("[FROZEN CANDIDATE SET]", left)
            self.assertIn("Return exactly 10 candidate identifiers", left)
            self.assertIn(INFORMATIONAL_OBJECTIVE, left)
            self.assertIn(TRANSACTIONAL_OBJECTIVE, right)
            self.assertEqual(
                left.replace(INFORMATIONAL_OBJECTIVE, "<INTENT>"),
                right.replace(TRANSACTIONAL_OBJECTIVE, "<INTENT>"),
            )

    def test_matched_query_nuisance_cancels_and_projection_defines_coordinate(self) -> None:
        keys = (
            ("query one", 0),
            ("query one", 1),
            ("query two", 0),
            ("query two", 1),
        )
        nuisance = np.asarray(
            [
                [0.2, 0.5, 0.0],
                [0.2, 0.0, 0.5],
                [-0.2, 0.5, 0.0],
                [-0.2, 0.0, 0.5],
            ]
        )
        informational = np.column_stack((-np.ones(4), nuisance))
        transactional = np.column_stack((np.ones(4), nuisance))

        axis, endpoints, diagnostics = fit_query_prior_a1_axis(
            informational,
            transactional,
            pair_keys=keys,
            embedding_model="fake-llm2vec",
        )

        self.assertEqual(axis.query_count, 2)
        self.assertEqual(axis.endpoint_pair_count, 4)
        self.assertEqual(axis.style_seeds, (0, 1))
        self.assertEqual(len(endpoints), 4)
        self.assertEqual(diagnostics.positive_pair_gap_rate, 1.0)
        self.assertEqual(diagnostics.positive_query_mean_gap_rate, 1.0)
        self.assertAlmostEqual(diagnostics.mean_informational_coordinate, 0.0)
        self.assertAlmostEqual(diagnostics.mean_transactional_coordinate, 1.0)

        midpoint = informational + transactional
        projected = project_onto_query_prior_a1(axis, midpoint)
        np.testing.assert_allclose(projected, np.full(4, 0.5), atol=1e-12)

    def test_axis_fit_rejects_unpaired_or_degenerate_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "pair_keys"):
            fit_query_prior_a1_axis(
                np.ones((2, 3)),
                np.ones((2, 3)),
                pair_keys=(("one", 0),),
                embedding_model="fake",
            )
        with self.assertRaisesRegex(ValueError, "nonzero semantic axis"):
            fit_query_prior_a1_axis(
                np.ones((2, 3)),
                np.ones((2, 3)),
                pair_keys=(("one", 0), ("two", 0)),
                embedding_model="fake",
            )


if __name__ == "__main__":
    unittest.main()
