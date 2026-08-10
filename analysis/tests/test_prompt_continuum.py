"""Focused contracts for the deterministic prompt-continuum scaffold."""

from __future__ import annotations

import hashlib
import random
import unittest

from analysis.interpretability.pipeline.prompt_continuum import (
    PromptGenerationRequest,
    TemplatePromptGenerator,
)
from analysis.interpretability.pipeline.prompts import build_rerank_prompt


VERSION = "template-v1"


def _generate(assigned_bias: float, style_seed: int = 42):
    return TemplatePromptGenerator().generate(
        PromptGenerationRequest(
            assigned_bias=assigned_bias,
            style_seed=style_seed,
            top_n=10,
            prompt_space_version=VERSION,
        )
    )


class TemplatePromptGeneratorTests(unittest.TestCase):
    def test_same_request_is_identical(self) -> None:
        first = _generate(0.5, 42)
        second = _generate(0.5, 42)
        self.assertEqual(first.prompt_template, second.prompt_template)
        self.assertEqual(first.prompt_hash, second.prompt_hash)
        self.assertEqual(first.prompt_id, second.prompt_id)

    def test_hash_and_id_are_derived_from_normalized_template(self) -> None:
        record = _generate(0.5, 42)
        expected_hash = hashlib.sha256(record.prompt_template.encode("utf-8")).hexdigest()
        self.assertEqual(record.prompt_hash, expected_hash)
        self.assertEqual(record.prompt_id, f"{VERSION}:{expected_hash[:16]}")

    def test_different_style_seeds_change_wording(self) -> None:
        self.assertNotEqual(
            _generate(0.5, 42).prompt_template,
            _generate(0.5, 43).prompt_template,
        )

    def test_style_plan_is_independent_of_assigned_bias(self) -> None:
        self.assertEqual(_generate(0.0, 42).style_plan, _generate(1.0, 42).style_plan)

    def test_generation_does_not_mutate_global_random_state(self) -> None:
        original_state = random.getstate()
        try:
            random.seed(2026)
            state_before_generation = random.getstate()
            _generate(0.5, 42)
            self.assertEqual(random.getstate(), state_before_generation)
        finally:
            random.setstate(original_state)

    def test_bias_endpoints(self) -> None:
        neutral = _generate(0.0).prompt_template.lower()
        strong = _generate(1.0).prompt_template.lower()
        self.assertNotIn("first-party", neutral)
        self.assertNotIn("software-product sources", neutral)
        self.assertIn("a strong preference", strong)
        self.assertIn("first-party software-product sources", strong)
        self.assertIn("relevance", strong)

    def test_placeholders_and_output_contract_are_preserved(self) -> None:
        for seed in range(50):
            template = _generate(0.5, seed).prompt_template
            self.assertIn("{QUERY}", template)
            self.assertIn("{CANDIDATES}", template)
            self.assertIn("{TOP_N}", template)
            self.assertIn("candidate identifiers", template)
            self.assertRegex(template.lower(), r"no explanation|do not (?:provide|include) an explanation")

    def test_invalid_assigned_bias_is_rejected(self) -> None:
        for assigned_bias in (-0.01, 1.01):
            with self.subTest(assigned_bias=assigned_bias), self.assertRaises(ValueError):
                PromptGenerationRequest(assigned_bias, 42, 10, VERSION)

    def test_invalid_top_n_is_rejected(self) -> None:
        for top_n in (0, -1):
            with self.subTest(top_n=top_n), self.assertRaises(ValueError):
                PromptGenerationRequest(0.5, 42, top_n, VERSION)


class LegacyPromptCompatibilityTests(unittest.TestCase):
    candidates = [
        {
            "position": 1,
            "url": "https://www.vendor.example/product",
            "title": "Vendor Product",
            "snippet": "Product page",
        },
        {
            "position": 2,
            "url": "https://review.example/comparison",
            "title": "Product Comparison",
            "snippet": "Comparison page",
        },
    ]

    def test_legacy_neutral_prompt_is_unchanged(self) -> None:
        expected = """Search keyword: crm software

Below are search engine results for the above keyword. Re-rank the results and return the top 2 URLs ordered by relevance to the keyword.

Return only root domains, one per line. No explanations.

Search results:
1. [vendor.example] Vendor Product — Product page
2. [review.example] Product Comparison — Comparison page

Re-ranked domains:"""
        self.assertEqual(
            build_rerank_prompt("crm software", self.candidates, top_n=2, variant="neutral"),
            expected,
        )

    def test_legacy_biased_prompt_is_unchanged(self) -> None:
        expected = """Search keyword: crm software

Below are search engine results for the above keyword. Re-rank the results and return the top 2 software product domains, ordered by relevance to the keyword.

Exclude non-product sites: review aggregators, directories, Wikipedia, news, blogs, forums, YouTube.

Return only root domains, one per line. No explanations.

Search results:
1. [vendor.example] Vendor Product — Product page
2. [review.example] Product Comparison — Comparison page

Re-ranked product domains:"""
        self.assertEqual(
            build_rerank_prompt("crm software", self.candidates, top_n=2, variant="biased"),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
