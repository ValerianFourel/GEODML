"""CPU-only contracts for query-conditioned latent prompt generation."""

from __future__ import annotations

import json
import unittest

import numpy as np

from analysis.interpretability.pipeline.prompt_latent_axis import (
    FakeLatentPromptProvider,
    FakePromptEmbedder,
    LatentPromptGenerationRequest,
    PromptProviderValidationError,
    build_latent_prompt_request,
    build_prompt_latent_axis,
    generate_prompt_at_coordinate,
    project_prompt_embeddings,
    render_selected_latent_prompt,
)
from analysis.interpretability.pipeline.search_purpose_continuum import (
    SearchCandidate,
    parse_ranking_permutation,
)


INFO_ENDPOINTS = (
    "Help the user learn and understand password managers. {QUERY} {CANDIDATES} "
    "Return exactly {TOP_N} candidate identifiers only, with no explanation.",
    "Provide an informational overview that explains CRM software. {QUERY} {CANDIDATES} "
    "Return exactly {TOP_N} candidate identifiers only, with no explanation.",
)
TRANSACTION_ENDPOINTS = (
    "Help the user select and start using a password manager now. {QUERY} {CANDIDATES} "
    "Return exactly {TOP_N} candidate identifiers only, with no explanation.",
    "Help the user select and deploy CRM software now. {QUERY} {CANDIDATES} "
    "Return exactly {TOP_N} candidate identifiers only, with no explanation.",
)


def _axis():
    return build_prompt_latent_axis(
        FakePromptEmbedder(),
        informational_endpoint_prompts=INFO_ENDPOINTS,
        transactional_endpoint_prompts=TRANSACTION_ENDPOINTS,
    )


def _request(target: float = 0.5, generation_seed: int = 11):
    return LatentPromptGenerationRequest(
        query="password manager for a small business",
        target_coordinate=target,
        style_seed=3,
        generation_seed=generation_seed,
        number_candidates=3,
        generator_model="fake-latent-generator-v1",
    )


class PromptLatentAxisTests(unittest.TestCase):
    def test_endpoint_pairs_define_reproducible_direction(self) -> None:
        self.assertEqual(_axis(), _axis())
        self.assertEqual(_axis().endpoint_pair_count, 2)
        self.assertEqual(_axis().dimension, 5)

    def test_endpoint_centroid_projections_are_zero_and_one(self) -> None:
        embedder = FakePromptEmbedder()
        axis = _axis()
        informational = project_prompt_embeddings(axis, embedder.embed(INFO_ENDPOINTS))
        transactional = project_prompt_embeddings(axis, embedder.embed(TRANSACTION_ENDPOINTS))
        self.assertAlmostEqual(float(np.mean(informational)), 0.0, places=10)
        self.assertAlmostEqual(float(np.mean(transactional)), 1.0, places=10)

    def test_endpoint_examples_must_be_paired(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be paired"):
            build_prompt_latent_axis(
                FakePromptEmbedder(),
                informational_endpoint_prompts=INFO_ENDPOINTS,
                transactional_endpoint_prompts=TRANSACTION_ENDPOINTS[:1],
            )

    def test_request_validates_target_coordinate(self) -> None:
        for target in (-0.01, 1.01):
            with self.subTest(target=target), self.assertRaises(ValueError):
                _request(target)

    def test_meta_prompt_is_conditioned_on_query(self) -> None:
        text = build_latent_prompt_request(_request(0.375))
        self.assertIn("password manager for a small business", text)
        self.assertIn("Target coordinate: 0.375000", text)
        self.assertIn("Number of candidates: 3", text)


class LatentPromptSelectionTests(unittest.TestCase):
    def test_generate_embed_project_select_is_reproducible(self) -> None:
        kwargs = {
            "axis": _axis(),
            "provider": FakeLatentPromptProvider(),
            "embedder": FakePromptEmbedder(),
        }
        first = generate_prompt_at_coordinate(_request(), **kwargs)
        second = generate_prompt_at_coordinate(_request(), **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first.candidate_projections), 3)
        self.assertEqual(first.validation_status, "latent-selected-unvalidated")
        self.assertEqual(
            first.absolute_target_error,
            min(item.absolute_target_error for item in first.candidate_projections),
        )

    def test_different_targets_select_different_prompts(self) -> None:
        kwargs = {
            "axis": _axis(),
            "provider": FakeLatentPromptProvider(),
            "embedder": FakePromptEmbedder(),
        }
        informational = generate_prompt_at_coordinate(_request(0.0), **kwargs)
        transactional = generate_prompt_at_coordinate(_request(1.0), **kwargs)
        self.assertNotEqual(informational.prompt_hash, transactional.prompt_hash)
        self.assertLess(
            informational.observed_axis_coordinate,
            transactional.observed_axis_coordinate,
        )

    def test_invalid_provider_json_is_rejected(self) -> None:
        class InvalidProvider:
            backend_name = "invalid"

            def generate(self, request_text, generation_config):
                return "not json"

        with self.assertRaisesRegex(
            PromptProviderValidationError, "after 3 deterministic attempts"
        ) as raised:
            generate_prompt_at_coordinate(
                _request(),
                axis=_axis(),
                provider=InvalidProvider(),
                embedder=FakePromptEmbedder(),
            )
        self.assertEqual(len(raised.exception.attempts), 3)
        self.assertEqual(
            [attempt.generation_seed for attempt in raised.exception.attempts],
            [11, 12, 13],
        )
        self.assertEqual(
            [attempt.raw_model_output for attempt in raised.exception.attempts],
            ["not json", "not json", "not json"],
        )
        self.assertTrue(
            all(
                "invalid JSON" in attempt.validation_error
                for attempt in raised.exception.attempts
            )
        )

    def test_invalid_first_attempt_is_retried_deterministically(self) -> None:
        class RetryProvider:
            backend_name = "retry"

            def __init__(self):
                self.seeds = []

            def generate(self, request_text, generation_config):
                self.seeds.append(generation_config["generation_seed"])
                if len(self.seeds) == 1:
                    templates = [
                        "Rerank {QUERY} using {CANDIDATES}. Return option IDs only."
                        for _ in range(3)
                    ]
                    return json.dumps({"prompt_templates": templates})
                return FakeLatentPromptProvider().generate(
                    request_text, generation_config
                )

        provider = RetryProvider()
        record = generate_prompt_at_coordinate(
            _request(generation_seed=41),
            axis=_axis(),
            provider=provider,
            embedder=FakePromptEmbedder(),
        )
        self.assertEqual(provider.seeds, [41, 42])
        self.assertEqual(record.generation_parameters["generation_seed"], 41)
        self.assertEqual(record.generation_parameters["validation_attempt_count"], 2)
        self.assertEqual(
            record.generation_parameters["attempted_generation_seeds"], [41, 42]
        )
        self.assertIn(
            "lacks {TOP_N}",
            record.generation_parameters["rejected_attempt_errors"][0],
        )

    def test_valid_json_in_markdown_fence_is_accepted(self) -> None:
        class FencedProvider:
            backend_name = "fenced"

            def generate(self, request_text, generation_config):
                raw = FakeLatentPromptProvider().generate(
                    request_text, generation_config
                )
                return f"```json\n{raw}\n```"

        record = generate_prompt_at_coordinate(
            _request(),
            axis=_axis(),
            provider=FencedProvider(),
            embedder=FakePromptEmbedder(),
        )
        self.assertTrue(record.raw_model_output.startswith("```json"))
        self.assertEqual(len(record.candidate_projections), 3)

    def test_valid_json_after_commentary_is_accepted(self) -> None:
        class CommentaryProvider:
            backend_name = "commentary"

            def generate(self, request_text, generation_config):
                raw = FakeLatentPromptProvider().generate(
                    request_text, generation_config
                )
                return f"Here is the requested JSON:\n{raw}\nDone."

        record = generate_prompt_at_coordinate(
            _request(),
            axis=_axis(),
            provider=CommentaryProvider(),
            embedder=FakePromptEmbedder(),
        )
        self.assertEqual(len(record.candidate_projections), 3)

    def test_candidate_ids_only_is_valid_output_contract_wording(self) -> None:
        class CandidateIdsProvider:
            backend_name = "candidate-ids"

            def generate(self, request_text, generation_config):
                templates = [
                    "Rerank {CANDIDATES} for {QUERY}. Return exactly {TOP_N} "
                    f"candidate IDs only, with no explanation. Variation {index}."
                    for index in range(3)
                ]
                return json.dumps({"prompt_templates": templates})

        record = generate_prompt_at_coordinate(
            _request(),
            axis=_axis(),
            provider=CandidateIdsProvider(),
            embedder=FakePromptEmbedder(),
        )
        self.assertEqual(len(record.candidate_projections), 3)

    def test_only_provide_option_ids_is_valid_output_contract_wording(self) -> None:
        class OptionIdsProvider:
            backend_name = "option-ids"

            def generate(self, request_text, generation_config):
                templates = [
                    "For {QUERY}, rerank {CANDIDATES}. Only provide the top "
                    f"{{TOP_N}} option IDs. Variation {index}."
                    for index in range(3)
                ]
                return json.dumps({"prompt_templates": templates})

        record = generate_prompt_at_coordinate(
            _request(),
            axis=_axis(),
            provider=OptionIdsProvider(),
            embedder=FakePromptEmbedder(),
        )
        self.assertEqual(len(record.candidate_projections), 3)

    def test_identifier_only_contract_cannot_request_explanations(self) -> None:
        class ContradictoryProvider:
            backend_name = "contradictory"

            def generate(self, request_text, generation_config):
                templates = [
                    "For {QUERY}, rerank {CANDIDATES}. Return exactly {TOP_N} "
                    f"candidate IDs only and explain the ranking. Variation {index}."
                    for index in range(3)
                ]
                return json.dumps({"prompt_templates": templates})

        with self.assertRaisesRegex(ValueError, "permits explanations"):
            generate_prompt_at_coordinate(
                _request(),
                axis=_axis(),
                provider=ContradictoryProvider(),
                embedder=FakePromptEmbedder(),
            )

    def test_off_axis_prompt_is_rejected(self) -> None:
        class OffAxisProvider:
            backend_name = "off-axis"

            def generate(self, request_text, generation_config):
                template = (
                    "Prefer first-party pages while ranking {QUERY}. {CANDIDATES} "
                    "Return exactly {TOP_N} candidate identifiers only, with no explanation."
                )
                return json.dumps({"prompt_templates": [template, template + " A", template + " B"]})

        with self.assertRaisesRegex(ValueError, "off-axis"):
            generate_prompt_at_coordinate(
                _request(),
                axis=_axis(),
                provider=OffAxisProvider(),
                embedder=FakePromptEmbedder(),
            )

    def test_selected_prompt_renders_and_accepts_strict_ranking(self) -> None:
        record = generate_prompt_at_coordinate(
            _request(),
            axis=_axis(),
            provider=FakeLatentPromptProvider(),
            embedder=FakePromptEmbedder(),
        )
        candidates = (
            SearchCandidate(1, "Guide", "https://guide.example", "Explanation"),
            SearchCandidate(2, "Setup", "https://setup.example", "Start now"),
            SearchCandidate(3, "Compare", "https://compare.example", "Comparison"),
        )
        rendered = render_selected_latent_prompt(record, candidates=candidates, top_n=2)
        self.assertIn("password manager for a small business", rendered.rendered_prompt)
        ranking = parse_ranking_permutation("C002 C001", rendered)
        self.assertEqual(ranking.source_position_vector, (2, 1))


if __name__ == "__main__":
    unittest.main()
