"""CPU contracts for the query-specific informational-to-buy centroid path."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from analysis.interpretability.pipeline.llm2vec_gen_axis import (
    QUERY_CENTROID_AXIS_VERSION,
    anchor_query_to_decoded_text,
    build_decodable_axis,
    build_query_centroid_requests,
    projection_residual_diagnostics,
)
from analysis.scripts import validate_query_centroid_axis


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_BANK = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "query_conditioned_info_buy_centroid_v1.json"
)


class QueryCentroidConstructionTests(unittest.TestCase):
    def test_query_anchor_preserves_exact_query_and_raw_decode(self) -> None:
        anchored = anchor_query_to_decoded_text(
            "  abandoned   cart recovery ", "Choose a practical solution."
        )
        self.assertIn('Fixed query: "abandoned cart recovery"', anchored)
        self.assertTrue(anchored.endswith("Choose a practical solution."))

    def test_every_matched_endpoint_contains_the_exact_query(self) -> None:
        specification = json.loads(TEMPLATE_BANK.read_text(encoding="utf-8"))
        query = "abandoned cart recovery"
        informational, buy = build_query_centroid_requests(query, specification)
        self.assertEqual(len(informational), 6)
        self.assertEqual(
            [row["frame_id"] for row in informational],
            [row["frame_id"] for row in buy],
        )
        for row in [*informational, *buy]:
            self.assertIn(query, row["request"])
            self.assertNotIn("/QUERY/", row["request"])
            self.assertNotIn("/INTENT/", row["request"])
            self.assertNotIn("/CONSTRAINTS/", row["request"])
        self.assertTrue(
            all(
                "neutral explanatory information" in row["request"]
                for row in informational
            )
        )
        self.assertTrue(
            all("choose and buy or adopt" in row["request"] for row in buy)
        )

    def test_template_bank_rejects_missing_matched_placeholder(self) -> None:
        specification = json.loads(TEMPLATE_BANK.read_text(encoding="utf-8"))
        specification["surface_frames"][0]["template"] = "Missing placeholders"
        with self.assertRaisesRegex(ValueError, "must contain"):
            build_query_centroid_requests("query", specification)

    def test_projection_residual_is_zero_on_axis_and_positive_off_axis(self) -> None:
        informational = np.array([[[0.0, 0.0]], [[0.0, 1.0]]])
        buy = np.array([[[1.0, 0.0]], [[1.0, 1.0]]])
        axis = build_decodable_axis(
            informational, buy, axis_version=QUERY_CENTROID_AXIS_VERSION
        )
        on_axis = projection_residual_diagnostics(axis, np.array([[0.5, 0.5]]))
        off_axis = projection_residual_diagnostics(axis, np.array([[0.5, 2.0]]))
        self.assertAlmostEqual(on_axis["axis_coordinate"], 0.5)
        self.assertAlmostEqual(on_axis["off_axis_distance"], 0.0)
        self.assertGreater(off_axis["off_axis_distance"], 0.0)


class QueryCentroidCliTests(unittest.TestCase):
    def test_fake_run_writes_query_specific_centroids_and_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = [
                "validate_query_centroid_axis.py",
                "--backend",
                "fake",
                "--query",
                "abandoned cart recovery",
                "--target-grid",
                "0,0.25,0.5,0.75,1",
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(validate_query_centroid_axis.main(), 0)

            diagnostics = json.loads(
                (output / "query_centroid_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            rows = [
                json.loads(line)
                for line in (output / "decoded_query_centroid_grid.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(diagnostics["axis_version"], QUERY_CENTROID_AXIS_VERSION)
            self.assertEqual(diagnostics["surface_frame_count"], 6)
            self.assertTrue(diagnostics["centroids_recomputed_for_each_query"])
            self.assertTrue(
                diagnostics["query_included_before_encoding_in_every_endpoint"]
            )
            self.assertFalse(diagnostics["scientific_result"])
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["query_present_case_insensitive"] for row in rows))
            self.assertTrue(
                all(
                    "abandoned cart recovery" in row["query_anchored_text"]
                    for row in rows
                )
            )
            self.assertEqual(
                diagnostics["query_anchored_retention"]["retained_count"], 5
            )
            self.assertIn(
                "anchored_reconstruction_spearman", diagnostics["decode_cycle"]
            )
            self.assertTrue(
                all("reencoded_reconstruction_residual_ratio" in row for row in rows)
            )
            self.assertTrue(
                all(
                    abs(
                        row["assigned_state_projection"][
                            "off_axis_distance_over_centroid_distance"
                        ]
                    )
                    < 1e-12
                    for row in rows
                )
            )
            with np.load(output / "query_centroid_state.npz") as state:
                self.assertEqual(state["informational_endpoint_states"].shape[0], 6)
                self.assertEqual(state["buy_intent_endpoint_states"].shape[0], 6)
                self.assertEqual(state["assigned_grid_states"].shape[0], 5)
                self.assertEqual(state["raw_reencoded_grid_states"].shape[0], 5)
                self.assertEqual(state["reencoded_grid_states"].shape[0], 5)
                self.assertEqual(
                    state["query_anchored_reencoded_grid_states"].shape[0], 5
                )
            report = (output / "query_centroid_report.md").read_text(encoding="utf-8")
            self.assertIn("Mock output only", report)
            self.assertIn("multiple matched surface frames", report)


if __name__ == "__main__":
    unittest.main()
