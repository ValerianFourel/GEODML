"""CPU contracts for the randomized ownership-by-intent latent plane."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from analysis.interpretability.pipeline.llm2vec_gen_axis import (
    OWNERSHIP_INTENT_PLANE_VERSION,
    build_ownership_intent_requests,
)
from analysis.scripts import validate_ownership_intent_plane


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_BANK = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "query_conditioned_ownership_intent_plane_v1.json"
)


class OwnershipIntentConstructionTests(unittest.TestCase):
    def test_four_matched_corners_preserve_exact_query(self) -> None:
        specification = json.loads(TEMPLATE_BANK.read_text(encoding="utf-8"))
        corners = build_ownership_intent_requests(
            "abandoned cart recovery", specification
        )
        self.assertEqual(
            set(corners), {"o-1_i-1", "o-1_i+1", "o+1_i-1", "o+1_i+1"}
        )
        expected_ids = [row["frame_id"] for row in corners["o-1_i-1"]]
        self.assertEqual(len(expected_ids), 6)
        for rows in corners.values():
            self.assertEqual([row["frame_id"] for row in rows], expected_ids)
            for row in rows:
                self.assertIn("abandoned cart recovery", row["request"])
                self.assertIn("B2B software evaluator", row["request"])
                for token in (
                    "/QUERY/", "/OWNERSHIP/", "/INTENT/", "/SHARED_INVARIANTS/"
                ):
                    self.assertNotIn(token, row["request"])


class OwnershipIntentPlaneCliTests(unittest.TestCase):
    def test_fake_run_writes_orthogonal_5_by_5_by_2_design(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = [
                "validate_ownership_intent_plane.py",
                "--backend",
                "fake",
                "--query",
                "abandoned cart recovery",
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(validate_ownership_intent_plane.main(), 0)

            diagnostics = json.loads(
                (output / "ownership_intent_plane_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            rows = [
                json.loads(line)
                for line in (output / "decoded_ownership_intent_grid.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                diagnostics["plane_version"], OWNERSHIP_INTENT_PLANE_VERSION
            )
            self.assertEqual(diagnostics["ownership_grid"], [-2, -1, 0, 1, 2])
            self.assertEqual(diagnostics["intent_grid"], [-2, -1, 0, 1, 2])
            self.assertEqual(diagnostics["style_seeds"], [0, 1])
            self.assertEqual(diagnostics["latent_point_count"], 50)
            self.assertAlmostEqual(
                diagnostics["plane_geometry"]["orthogonal_basis_cosine"], 0.0
            )
            self.assertLess(
                diagnostics["plane_geometry"]["surface_projection_max_abs"], 1e-12
            )
            self.assertEqual(len(rows), 50)
            self.assertTrue(
                diagnostics["coordinates_outside_unit_square_are_feasibility_only"]
            )
            for row in rows:
                prompt = row["rendered_reranking_prompt"]
                self.assertIn("abandoned cart recovery", prompt)
                self.assertIn("{QUERY}", prompt)
                self.assertIn("{CANDIDATES}", prompt)
                self.assertIn("{TOP_N}", prompt)
                assigned = row["assigned_projection"]
                self.assertAlmostEqual(
                    assigned["ownership_coordinate"], row["assigned_ownership"]
                )
                self.assertAlmostEqual(
                    assigned["intent_coordinate"], row["assigned_intent"]
                )
            with np.load(output / "ownership_intent_plane_state.npz") as state:
                self.assertEqual(state["surface_residuals_orthogonal"].shape[0], 6)
                self.assertEqual(state["assigned_states"].shape[0], 50)
                self.assertEqual(state["matched_states"].shape[0], 50)
            report = (output / "ownership_intent_plane_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Mock output only", report)


if __name__ == "__main__":
    unittest.main()
