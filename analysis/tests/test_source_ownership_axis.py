"""CPU contracts for the seller-independent to seller-controlled field."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from analysis.interpretability.pipeline.llm2vec_gen_axis import (
    SOURCE_OWNERSHIP_AXIS_VERSION,
    build_source_ownership_requests,
)
from analysis.scripts import validate_source_ownership_axis


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_BANK = (
    ANALYSIS_ROOT
    / "interpretability"
    / "pipeline"
    / "specs"
    / "query_conditioned_source_ownership_axis_v1.json"
)


class SourceOwnershipConstructionTests(unittest.TestCase):
    def test_matched_regions_preserve_query_and_frames(self) -> None:
        specification = json.loads(TEMPLATE_BANK.read_text(encoding="utf-8"))
        regions = build_source_ownership_requests(
            "abandoned cart recovery", specification
        )
        self.assertEqual(
            set(regions), {"seller-independent", "neutral", "seller-controlled"}
        )
        frame_ids = [row["frame_id"] for row in regions["neutral"]]
        self.assertEqual(len(frame_ids), 6)
        for rows in regions.values():
            self.assertEqual([row["frame_id"] for row in rows], frame_ids)
            for row in rows:
                self.assertIn("abandoned cart recovery", row["request"])
                self.assertIn("B2B software evaluator", row["request"])
                self.assertNotIn("/QUERY/", row["request"])
                self.assertNotIn("/OWNERSHIP_POLICY/", row["request"])
                self.assertNotIn("/SHARED_INVARIANTS/", row["request"])
        independent = " ".join(
            row["request"] for row in regions["seller-independent"]
        ).casefold()
        neutral = " ".join(row["request"] for row in regions["neutral"]).casefold()
        controlled = " ".join(
            row["request"] for row in regions["seller-controlled"]
        ).casefold()
        self.assertIn("sources independent", independent)
        self.assertIn("do not use whether a publisher", neutral)
        self.assertIn("published by vendors", controlled)

    def test_bad_template_version_is_rejected(self) -> None:
        specification = json.loads(TEMPLATE_BANK.read_text(encoding="utf-8"))
        specification["template_bank_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_source_ownership_requests("crm software", specification)


class SourceOwnershipCliTests(unittest.TestCase):
    def test_fake_run_writes_centered_symmetric_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = [
                "validate_source_ownership_axis.py",
                "--backend",
                "fake",
                "--query",
                "abandoned cart recovery",
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(validate_source_ownership_axis.main(), 0)

            diagnostics = json.loads(
                (output / "source_ownership_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            rows = [
                json.loads(line)
                for line in (output / "decoded_source_ownership_grid.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                diagnostics["axis_version"], SOURCE_OWNERSHIP_AXIS_VERSION
            )
            self.assertEqual(diagnostics["surface_frame_count"], 6)
            self.assertEqual(diagnostics["latent_point_count"], 13)
            self.assertAlmostEqual(
                diagnostics["neutral_location_on_endpoint_axis"][
                    "lambda_coordinate"
                ],
                0.0,
            )
            self.assertTrue(diagnostics["all_nonownership_ranking_components_fixed"])
            self.assertEqual(len(rows), 13)
            self.assertAlmostEqual(rows[0]["assigned_lambda"], -1.0)
            self.assertAlmostEqual(rows[-1]["assigned_lambda"], 1.0)
            for row in rows:
                prompt = row["rendered_reranking_prompt"]
                self.assertIn("abandoned cart recovery", prompt)
                self.assertIn("{QUERY}", prompt)
                self.assertIn("{CANDIDATES}", prompt)
                self.assertIn("{TOP_N}", prompt)
                self.assertTrue(
                    row["ownership_policy_checks"][
                        "passes_lexical_invariant_screen"
                    ]
                )
            self.assertEqual(
                diagnostics["semantic_invariant_screen"]["passing_count"], 13
            )
            with np.load(output / "source_ownership_state.npz") as state:
                self.assertEqual(state["seller_independent_states"].shape[0], 6)
                self.assertEqual(state["neutral_states"].shape[0], 6)
                self.assertEqual(state["seller_controlled_states"].shape[0], 6)
                self.assertEqual(state["assigned_states"].shape[0], 13)
            report = (output / "source_ownership_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Mock output only", report)


if __name__ == "__main__":
    unittest.main()
