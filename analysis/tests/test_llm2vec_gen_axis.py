"""CPU-only contracts for the decodable LLM2Vec-Gen axis milestone."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from analysis.interpretability.pipeline.llm2vec_gen_axis import (
    ENCODING_INSTRUCTION_VERSION,
    axis_geometry_diagnostics,
    build_decodable_axis,
    build_encoding_text,
    build_query_conditioned_requests,
    decode_record_checks,
    inject_query_after_decode,
    interpolate_axis_centroids,
    interpolate_endpoint_pair,
    project_onto_axis,
    stable_array_hash,
)
from analysis.scripts import validate_llm2vec_gen_axis


def _states() -> tuple[np.ndarray, np.ndarray]:
    informational = np.array(
        [
            [[0.0, 0.1], [0.2, 0.0]],
            [[0.1, 0.4], [0.0, 0.2]],
            [[-0.1, 0.7], [0.1, 0.5]],
        ],
        dtype=np.float32,
    )
    transactional = informational.copy()
    transactional[:, :, 0] += np.array([1.0, 0.9, 1.1])[:, None]
    return informational, transactional


class DecodableAxisGeometryTests(unittest.TestCase):
    def test_centroid_path_is_calibrated_to_zero_and_one(self) -> None:
        informational, transactional = _states()
        axis = build_decodable_axis(informational, transactional)
        self.assertAlmostEqual(
            float(project_onto_axis(axis, interpolate_axis_centroids(axis, 0.0))),
            0.0,
            places=10,
        )
        self.assertAlmostEqual(
            float(project_onto_axis(axis, interpolate_axis_centroids(axis, 1.0))),
            1.0,
            places=10,
        )
        self.assertAlmostEqual(
            float(project_onto_axis(axis, interpolate_axis_centroids(axis, 0.35))),
            0.35,
            places=10,
        )

    def test_pair_interpolation_preserves_native_state_shape(self) -> None:
        informational, transactional = _states()
        midpoint = interpolate_endpoint_pair(
            informational[0], transactional[0], 0.5
        )
        self.assertEqual(midpoint.shape, (2, 2))
        np.testing.assert_allclose(
            midpoint, (informational[0] + transactional[0]) / 2.0
        )

    def test_leave_one_pair_out_direction_generalizes(self) -> None:
        informational, transactional = _states()
        diagnostics = axis_geometry_diagnostics(informational, transactional)
        self.assertEqual(diagnostics["pair_count"], 3)
        self.assertEqual(diagnostics["leave_one_pair_out_positive_rate"], 1.0)
        self.assertGreater(diagnostics["leave_one_pair_out_cosine_mean"], 0.99)

    def test_leave_one_pair_out_exposes_a_reversed_pair(self) -> None:
        informational, transactional = _states()
        transactional[2, :, 0] = informational[2, :, 0] - 1.0
        diagnostics = axis_geometry_diagnostics(informational, transactional)
        held_out = diagnostics["leave_one_pair_out"]
        self.assertFalse(held_out[2]["positive_direction"])
        self.assertLess(diagnostics["leave_one_pair_out_positive_rate"], 1.0)

    def test_invalid_state_shapes_and_coordinates_fail_closed(self) -> None:
        informational, transactional = _states()
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            build_decodable_axis(informational, transactional[:2])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            interpolate_endpoint_pair(informational[0], transactional[0], 1.1)

    def test_axis_and_array_hashes_are_reproducible(self) -> None:
        informational, transactional = _states()
        first = build_decodable_axis(informational, transactional)
        second = build_decodable_axis(informational.copy(), transactional.copy())
        self.assertEqual(first.axis_hash, second.axis_hash)
        self.assertEqual(stable_array_hash(informational), stable_array_hash(informational))


class ReconstructionTextContractTests(unittest.TestCase):
    def test_encoding_task_requests_generation_and_preserves_placeholders(self) -> None:
        template = "Rank {CANDIDATES} for {QUERY}; return {TOP_N} identifiers only."
        encoded = build_encoding_text(template)
        self.assertIn("Generate the reusable", encoded)
        self.assertTrue(encoded.endswith(template))
        self.assertEqual(
            ENCODING_INSTRUCTION_VERSION, "reranking-template-reconstruction-v1"
        )

    def test_decoded_template_checks_do_not_claim_semantic_validity(self) -> None:
        text = (
            "Rank {CANDIDATES} for {QUERY}. Return exactly {TOP_N} candidate "
            "identifiers only, with no explanation."
        )
        checks = decode_record_checks(text)
        self.assertTrue(checks["all_placeholders_preserved"])
        self.assertTrue(checks["mentions_identifier_only"])
        self.assertTrue(checks["prohibits_explanation"])
        self.assertEqual(checks["detected_off_axis_terms"], [])
        self.assertNotIn("semantic_valid", checks)

    def test_structural_checks_flag_off_axis_criteria(self) -> None:
        checks = decode_record_checks(
            "Rank {CANDIDATES} for {QUERY} using popularity and authority. "
            "Return {TOP_N} identifiers only with no explanation."
        )
        self.assertEqual(checks["detected_off_axis_terms"], ["authority", "popularity"])

    def test_query_is_injected_only_through_literal_placeholder(self) -> None:
        template = "Rank {CANDIDATES} for {QUERY}; return {TOP_N}."
        rendered = inject_query_after_decode(template, "abandoned cart recovery")
        self.assertIn("abandoned cart recovery", rendered)
        with self.assertRaisesRegex(ValueError, "does not preserve"):
            inject_query_after_decode("Rank the candidates", "query")

    def test_query_conditioned_endpoints_include_exact_query_before_encoding(self) -> None:
        query = "abandoned cart recovery"
        informational, transactional = build_query_conditioned_requests(query)
        self.assertIn(f'"{query}"', informational)
        self.assertIn(f'"{query}"', transactional)
        self.assertIn("learn and understand", informational)
        self.assertIn("begin implementing", transactional)
        self.assertNotIn("{QUERY}", informational)
        self.assertNotIn("{QUERY}", transactional)

    def test_query_conditioned_endpoints_reject_ambiguous_query_delimiters(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_query_conditioned_requests("   ")
        with self.assertRaisesRegex(ValueError, "double quotes"):
            build_query_conditioned_requests('abandoned "cart" recovery')


class FakeAxisCliTests(unittest.TestCase):
    def test_horeka_installer_preserves_existing_torch_stack(self) -> None:
        analysis_root = Path(__file__).resolve().parents[1]
        requirements = (
            analysis_root / "requirements-horeka-llm2vec-gen.txt"
        ).read_text(encoding="utf-8")
        installer = (
            analysis_root / "scripts" / "install_llm2vec_gen_runtime.sh"
        ).read_text(encoding="utf-8")
        active_requirements = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(
            any(line.startswith("llm2vec-gen") for line in active_requirements)
        )
        self.assertFalse(
            any(line.startswith("flash-attn") for line in active_requirements)
        )
        self.assertIn('LLM2VEC_GEN_VERSION="0.1.3"', installer)
        self.assertIn("pip install --no-deps", installer)
        self.assertNotIn("pip install flash-attn", installer)

    def test_fake_end_to_end_writes_explicitly_non_scientific_artifacts(self) -> None:
        endpoint_bank = (
            Path(__file__).resolve().parents[1]
            / "interpretability"
            / "pipeline"
            / "specs"
            / "search_purpose_endpoint_pairs_v1.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = [
                "validate_llm2vec_gen_axis.py",
                "--backend",
                "fake",
                "--endpoint-bank",
                str(endpoint_bank),
                "--target-grid",
                "0,0.5,1",
                "--decode-pairs",
                "1",
                "--probe-query",
                "abandoned cart recovery",
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(validate_llm2vec_gen_axis.main(), 0)

            diagnostics = json.loads(
                (output / "axis_diagnostics.json").read_text(encoding="utf-8")
            )
            rows = [
                json.loads(line)
                for line in (output / "decoded_latent_grid.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(diagnostics["status"], "feasibility-only")
            self.assertFalse(diagnostics["scientific_result"])
            self.assertFalse(
                diagnostics["query_strategy"][
                    "probe_query_used_during_axis_estimation"
                ]
            )
            self.assertTrue(
                diagnostics["query_strategy"][
                    "paired_endpoint_topics_used_during_axis_estimation"
                ]
            )
            self.assertFalse(diagnostics["query_strategy"]["vector_addition_tested"])
            self.assertEqual(
                diagnostics["reconstruction_geometry"][
                    "leave_one_pair_out_positive_rate"
                ],
                1.0,
            )
            self.assertTrue(
                all(
                    path["reconstruction_strictly_increasing"]
                    for path in diagnostics["decode_cycle"]["path_monotonicity"]
                )
            )
            self.assertTrue(diagnostics["decode_cycle"]["same_model_diagnostic_only"])
            self.assertEqual(len(rows), 16)
            self.assertEqual(
                sum(row["path_kind"] == "endpoint-reconstruction-control" for row in rows),
                12,
            )
            self.assertTrue(
                all(row["structural_checks"]["all_placeholders_preserved"] for row in rows)
            )
            self.assertTrue(
                all("abandoned cart recovery" in row["query_injected_after_decode"] for row in rows)
            )
            with np.load(output / "axis_state.npz") as state:
                self.assertEqual(state["informational_endpoint_states"].shape[0], 6)
            report = (output / "axis_feasibility_report.md").read_text(encoding="utf-8")
            self.assertIn("Mock output only", report)

    def test_fake_query_conditioned_axis_uses_query_in_both_endpoints(self) -> None:
        endpoint_bank = (
            Path(__file__).resolve().parents[1]
            / "interpretability"
            / "pipeline"
            / "specs"
            / "search_purpose_endpoint_pairs_v1.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            argv = [
                "validate_llm2vec_gen_axis.py",
                "--backend",
                "fake",
                "--endpoint-bank",
                str(endpoint_bank),
                "--target-grid",
                "0,0.5,1",
                "--decode-pairs",
                "0",
                "--probe-query",
                "abandoned cart recovery",
                "--query-conditioned-axis",
                "--output-dir",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(validate_llm2vec_gen_axis.main(), 0)

            diagnostics = json.loads(
                (output / "axis_diagnostics.json").read_text(encoding="utf-8")
            )
            rows = [
                json.loads(line)
                for line in (output / "decoded_latent_grid.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            query_axis = diagnostics["query_conditioned_axis"]
            self.assertTrue(
                diagnostics["query_strategy"][
                    "probe_query_used_during_query_conditioned_axis_estimation"
                ]
            )
            self.assertTrue(query_axis["query_is_informational_endpoint_substring"])
            self.assertTrue(query_axis["query_is_transactional_endpoint_substring"])
            self.assertEqual(
                sum(
                    row["path_kind"] == "query-conditioned-direct-request"
                    for row in rows
                ),
                3,
            )
            self.assertTrue(
                query_axis["decode_cycle"]["reconstruction_strictly_increasing"]
            )
            with np.load(output / "axis_state.npz") as state:
                self.assertIn("query_conditioned_informational_state", state.files)
                self.assertIn("query_conditioned_transactional_state", state.files)


if __name__ == "__main__":
    unittest.main()
