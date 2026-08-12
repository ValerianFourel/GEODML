"""Contracts for latent prompt validation and strict permutation linkage."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.prompt_calibration import generate_calibration_records
from analysis.interpretability.pipeline.prompt_policy_mapping import (
    FakePolicyPromptEmbedder,
    build_policy_latent_axis,
    locate_policy_prompts,
    map_policy_prompt_to_permutation,
    render_policy_prompt,
    validate_policy_prompt_locations,
    write_policy_mapping_artifacts,
)
from analysis.interpretability.pipeline.search_purpose_continuum import (
    PermutationValidationError,
    SearchCandidate,
)


def _records():
    return generate_calibration_records(
        b_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        number_style_seeds=2,
        top_n=2,
        prompt_space_version="policy-map-test-v1",
    )


def _axis_and_locations():
    records = _records()
    endpoints = tuple(record for record in records if record.assigned_bias in (0.0, 1.0))
    embedder = FakePolicyPromptEmbedder()
    axis = build_policy_latent_axis(embedder, endpoints)
    return records, axis, locate_policy_prompts(axis, embedder, records)


def _candidates():
    return (
        SearchCandidate(1, "Vendor product", "https://vendor.example/product", "First party"),
        SearchCandidate(2, "Independent review", "https://review.example/guide", "Third party"),
        SearchCandidate(3, "Directory", "https://directory.example/list", "Candidate list"),
    )


class PolicyAxisTests(unittest.TestCase):
    def test_matched_endpoints_define_reproducible_axis(self) -> None:
        records = _records()
        endpoints = tuple(record for record in records if record.assigned_bias in (0.0, 1.0))
        first = build_policy_latent_axis(FakePolicyPromptEmbedder(), endpoints)
        second = build_policy_latent_axis(FakePolicyPromptEmbedder(), endpoints)
        self.assertEqual(first, second)
        self.assertEqual(first.endpoint_style_seeds, (0, 1))
        self.assertEqual(first.leave_one_style_out_positive_rate, 1.0)
        self.assertTrue(all(abs(value - 1.0) < 1e-12 for value in first.pair_direction_cosines))

    def test_axis_requires_matched_surface_pairs(self) -> None:
        records = tuple(
            record
            for record in _records()
            if record.assigned_bias in (0.0, 1.0)
            and not (record.style_seed == 1 and record.assigned_bias == 1.0)
        )
        with self.assertRaisesRegex(ValueError, "lacks a matched"):
            build_policy_latent_axis(FakePolicyPromptEmbedder(), records)

    def test_locations_keep_assignment_separate_from_projection(self) -> None:
        records, axis, locations = _axis_and_locations()
        self.assertEqual([item.assigned_bias for item in locations], [item.assigned_bias for item in records])
        for location in locations:
            self.assertAlmostEqual(location.observed_axis_coordinate, location.assigned_bias)
            self.assertAlmostEqual(location.absolute_assigned_coordinate_error, 0.0)
            self.assertAlmostEqual(location.matched_off_axis_residual, 0.0)
            self.assertEqual(location.axis_id, axis.axis_id)
            self.assertTrue(
                location.prompt_assignment_id.startswith("policy-assignment:")
            )
            self.assertEqual(len(location.prompt_embedding), axis.dimension)
            self.assertEqual(len(location.embedding_hash), 64)

    def test_validation_detects_reversal_without_reassigning_B(self) -> None:
        _, _, locations = _axis_and_locations()
        changed = list(locations)
        index = next(i for i, item in enumerate(changed) if item.style_seed == 0 and item.assigned_bias == 0.75)
        changed[index] = replace(changed[index], observed_axis_coordinate=0.1)
        validation = validate_policy_prompt_locations(
            changed,
            max_absolute_coordinate_error=1.0,
            max_matched_off_axis_residual=0.1,
        )
        self.assertFalse(validation.passed)
        self.assertEqual(validation.nonmonotonic_style_seeds, (0,))
        self.assertEqual(changed[index].assigned_bias, 0.75)

    def test_validation_thresholds_are_explicit(self) -> None:
        _, _, locations = _axis_and_locations()
        validation = validate_policy_prompt_locations(
            locations,
            max_absolute_coordinate_error=0.05,
            max_matched_off_axis_residual=0.05,
        )
        self.assertTrue(validation.passed)
        with self.assertRaises(ValueError):
            validate_policy_prompt_locations(
                locations,
                max_absolute_coordinate_error=-1,
                max_matched_off_axis_residual=0.1,
            )


class PolicyPermutationMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        records, self.axis, locations = _axis_and_locations()
        self.record = next(item for item in records if item.style_seed == 0 and item.assigned_bias == 0.5)
        self.location = next(item for item in locations if item.prompt_id == self.record.prompt_id)
        self.rendered = render_policy_prompt(
            self.record,
            keyword="abandoned cart recovery",
            candidates=_candidates(),
            top_n=2,
        )

    def test_strict_permutation_is_linked_to_assignment_and_location(self) -> None:
        outcome = map_policy_prompt_to_permutation(
            self.rendered,
            self.location,
            "C002\nC001",
            reranker_run_id="smoke-run-1",
            reranker_model="fake-reranker",
        )
        self.assertEqual(outcome.assigned_bias, 0.5)
        self.assertEqual(outcome.observed_axis_coordinate, 0.5)
        self.assertEqual(outcome.ranking.candidate_ids, ("C002", "C001"))
        self.assertEqual(outcome.ranking.source_position_vector, (2, 1))
        self.assertEqual(outcome.prompt_instance_id, self.rendered.prompt_instance_id)
        self.assertEqual(
            outcome.prompt_assignment_id, self.rendered.prompt_assignment_id
        )
        self.assertEqual(outcome.prompt_embedding_hash, self.location.embedding_hash)

    def test_invalid_ranking_is_not_silently_repaired(self) -> None:
        with self.assertRaises(PermutationValidationError):
            map_policy_prompt_to_permutation(
                self.rendered,
                self.location,
                "I prefer C002 because it is independent.",
                reranker_run_id="smoke-run-1",
                reranker_model="fake-reranker",
            )

    def test_prompt_location_identity_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "identities do not match"):
            map_policy_prompt_to_permutation(
                self.rendered,
                replace(self.location, prompt_hash="0" * 64),
                "C002 C001",
                reranker_run_id="smoke-run-1",
                reranker_model="fake-reranker",
            )

    def test_artifacts_preserve_axis_locations_and_outcomes(self) -> None:
        records, axis, locations = _axis_and_locations()
        validation = validate_policy_prompt_locations(
            locations,
            max_absolute_coordinate_error=0.05,
            max_matched_off_axis_residual=0.05,
        )
        outcome = map_policy_prompt_to_permutation(
            self.rendered,
            self.location,
            "C002 C001",
            reranker_run_id="smoke-run-1",
            reranker_model="fake-reranker",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = write_policy_mapping_artifacts(
                temporary_directory,
                axis=axis,
                locations=locations,
                validation=validation,
                outcomes=(outcome,),
            )
            axis_payload = json.loads(artifacts.axis_path.read_text(encoding="utf-8"))
            validation_payload = json.loads(
                artifacts.validation_path.read_text(encoding="utf-8")
            )
            outcome_payload = json.loads(artifacts.outcomes_path.read_text(encoding="utf-8").strip())
            self.assertEqual(axis_payload["axis_id"], axis.axis_id)
            self.assertTrue(validation_payload["passed"])
            self.assertEqual(validation_payload["max_absolute_coordinate_error"], 0.05)
            self.assertEqual(outcome_payload["ranking"]["candidate_ids"], ["C002", "C001"])
            self.assertIn("assigned_bias", outcome_payload)
            self.assertIn("observed_axis_coordinate", outcome_payload)
            self.assertIn("prompt_embedding", json.loads(
                artifacts.locations_path.read_text(encoding="utf-8").splitlines()[0]
            ))
            self.assertEqual(
                outcome_payload["prompt_embedding_hash"], self.location.embedding_hash
            )
            self.assertIn("vectors are not decoded", artifacts.report_path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_policy_mapping_artifacts(
                    temporary_directory,
                    axis=axis,
                    locations=locations,
                    validation=validation,
                )

    def test_assignment_ids_distinguish_duplicate_prompt_text_across_B(self) -> None:
        records = generate_calibration_records(
            b_grid=(0.1, 0.2),
            number_style_seeds=1,
            top_n=2,
            prompt_space_version="duplicate-text-test-v1",
        )
        self.assertEqual(records[0].prompt_id, records[1].prompt_id)
        endpoints = generate_calibration_records(
            b_grid=(0.0, 1.0),
            number_style_seeds=2,
            top_n=2,
            prompt_space_version="duplicate-text-test-v1",
        )
        axis = build_policy_latent_axis(FakePolicyPromptEmbedder(), endpoints)
        locations = locate_policy_prompts(axis, FakePolicyPromptEmbedder(), records)
        self.assertNotEqual(
            locations[0].prompt_assignment_id,
            locations[1].prompt_assignment_id,
        )


if __name__ == "__main__":
    unittest.main()
