"""Milestone 3A policy-clause bank tests; fake provider only."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.policy_clause_bank import (
    HybridPromptComposer,
    PolicyClauseGenerationRequest,
    PolicyClauseGenerator,
    PolicyClauseStructuralError,
    FakePolicyClauseProvider,
    SPECIFICATION_VERSION,
    canonical_bias,
    default_generation_parameters,
)
from analysis.interpretability.pipeline.policy_clause_pilot import (
    build_pilot_requests,
    stratified_bias_schedule,
    write_policy_clause_pilot,
)


def _request(
    assigned_bias: float = 0.5,
    *,
    style_seed: int = 2,
    generation_seed: int = 7,
):
    return PolicyClauseGenerationRequest(
        assigned_bias=assigned_bias,
        style_seed=style_seed,
        generation_seed=generation_seed,
        specification_version=SPECIFICATION_VERSION,
        generator_model="fake-policy-provider-v1",
    )


class PolicyClauseBankTests(unittest.TestCase):
    def test_request_validates_bias(self) -> None:
        for value in (-0.001, 1.001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _request(value)

    def test_canonical_bias_is_stable(self) -> None:
        self.assertEqual(canonical_bias(0), "0.000000")
        self.assertEqual(canonical_bias(-0.0), "0.000000")
        self.assertEqual(canonical_bias(0.1234564), "0.123456")
        self.assertEqual(canonical_bias(1), "1.000000")

    def test_fake_provider_produces_reproducible_records(self) -> None:
        fixed_time = lambda: "2026-08-10T00:00:00Z"
        first = PolicyClauseGenerator(
            FakePolicyClauseProvider(), generated_at_factory=fixed_time
        ).generate(_request())
        second = PolicyClauseGenerator(
            FakePolicyClauseProvider(), generated_at_factory=fixed_time
        ).generate(_request())
        self.assertEqual(first, second)
        self.assertEqual(first.validation_status, "unvalidated")

    def test_same_request_uses_cached_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            provider = FakePolicyClauseProvider()
            generator = PolicyClauseGenerator(
                provider,
                cache_directory=temporary_directory,
                generated_at_factory=lambda: "2026-08-10T00:00:00Z",
            )
            first = generator.generate(_request())
            second = generator.generate(_request())
            self.assertEqual(first, second)
            self.assertEqual(provider.call_count, 1)

    def test_different_generation_seeds_can_change_candidate(self) -> None:
        generator = PolicyClauseGenerator(FakePolicyClauseProvider())
        first = generator.generate(_request(generation_seed=1))
        second = generator.generate(_request(generation_seed=2))
        self.assertNotEqual(first.policy_clause, second.policy_clause)

    def test_numeric_bias_is_not_exposed_in_composed_prompt(self) -> None:
        candidate = PolicyClauseGenerator(FakePolicyClauseProvider()).generate(
            _request(0.375, generation_seed=4)
        )
        prompt = HybridPromptComposer().compose(
            assigned_bias=0.375,
            style_seed=candidate.style_seed,
            policy_clause=candidate.policy_clause,
        )
        self.assertNotIn("0.375", prompt.prompt_template)
        self.assertNotIn("37.5%", prompt.prompt_template)

    def test_forbidden_criterion_is_rejected(self) -> None:
        class ForbiddenProvider:
            backend_name = "test-forbidden"

            def generate(self, request_text, generation_config):
                return json.dumps({
                    "policy_clause": "Prefer first-party sources for their freshness while retaining relevance."
                })

        with self.assertRaisesRegex(PolicyClauseStructuralError, "forbidden_criterion:freshness"):
            PolicyClauseGenerator(ForbiddenProvider()).generate(_request())

    def test_hard_exclusion_is_rejected(self) -> None:
        class ExclusionProvider:
            backend_name = "test-exclusion"

            def generate(self, request_text, generation_config):
                return json.dumps({
                    "policy_clause": "Only rank first-party software-product sources, regardless of relevance."
                })

        with self.assertRaisesRegex(PolicyClauseStructuralError, "hard_exclusion"):
            PolicyClauseGenerator(ExclusionProvider()).generate(_request(0.9))

    def test_required_placeholders_survive_composition(self) -> None:
        candidate = PolicyClauseGenerator(FakePolicyClauseProvider()).generate(_request())
        prompt = HybridPromptComposer().compose(
            assigned_bias=candidate.assigned_bias,
            style_seed=candidate.style_seed,
            policy_clause=candidate.policy_clause,
        )
        for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
            self.assertIn(placeholder, prompt.prompt_template)

    def test_invalid_provider_json_is_clear(self) -> None:
        class InvalidProvider:
            backend_name = "test-invalid-json"

            def generate(self, request_text, generation_config):
                return "not JSON"

        with self.assertRaisesRegex(PolicyClauseStructuralError, "invalid_json"):
            PolicyClauseGenerator(InvalidProvider()).generate(_request())

    def test_pilot_schedule_is_reused_across_style_seeds(self) -> None:
        schedule = stratified_bias_schedule(master_seed=12)
        requests = build_pilot_requests(
            generator_model="fake-policy-provider-v1",
            number_style_seeds=3,
            master_seed=12,
        )
        for style_seed in range(3):
            observed = tuple(
                request.assigned_bias
                for request in requests
                if request.style_seed == style_seed
            )
            self.assertEqual(observed, schedule)

    def test_dry_run_and_fake_generation_use_temporary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dry = write_policy_clause_pilot(
                root / "dry",
                mode="dry-run",
                provider=None,
                generator_model="fake-policy-provider-v1",
                number_style_seeds=2,
                number_bias_values=3,
            )
            self.assertTrue(dry.requests_path.exists())
            self.assertIsNone(dry.candidates_path)
            generated = write_policy_clause_pilot(
                root / "generated",
                mode="generate",
                provider=FakePolicyClauseProvider(),
                generator_model="fake-policy-provider-v1",
                number_style_seeds=2,
                number_bias_values=3,
                generation_parameters=default_generation_parameters(),
            )
            self.assertEqual(len(generated.candidates), 6)
            self.assertTrue(generated.candidates_path.exists())
            self.assertTrue(generated.full_prompts_path.exists())
            report = generated.report_path.read_text(encoding="utf-8")
            self.assertIn("These clauses are unvalidated candidates", report)
            self.assertIn("semantic monotonicity", report)


if __name__ == "__main__":
    unittest.main()
