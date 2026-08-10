"""Contracts for the informational-to-action search-purpose foundation."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import unittest

from analysis.interpretability.pipeline.search_purpose_continuum import (
    PermutationValidationError,
    SearchCandidate,
    SearchPurposeGenerationRequest,
    SearchPurposeTemplateGenerator,
    load_search_purpose_specification,
    parse_ranking_permutation,
    render_search_purpose_prompt,
    write_search_purpose_pilot,
)


VERSION = "search-purpose-test-v1"


def _template(intensity: float = 0.5, style_seed: int = 42, top_n: int = 2):
    return SearchPurposeTemplateGenerator().generate(
        SearchPurposeGenerationRequest(
            assigned_action_intensity=intensity,
            style_seed=style_seed,
            top_n=top_n,
            prompt_space_version=VERSION,
        )
    )


def _candidates() -> tuple[SearchCandidate, ...]:
    return (
        SearchCandidate(1, "Explanatory guide", "https://guide.example/a", "Learn concepts"),
        SearchCandidate(2, "Action guide", "https://action.example/b", "Complete the task"),
        SearchCandidate(3, "Comparison", "https://compare.example/c", "Compare approaches"),
    )


class SearchPurposeTemplateTests(unittest.TestCase):
    def test_request_validates_assigned_intensity(self) -> None:
        for value in (-0.001, 1.001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SearchPurposeGenerationRequest(value, 0, 2, VERSION)

    def test_same_request_is_reproducible(self) -> None:
        self.assertEqual(_template(), _template())

    def test_style_plan_is_independent_of_intensity(self) -> None:
        self.assertEqual(_template(0.0).style_plan, _template(1.0).style_plan)

    def test_generation_does_not_mutate_global_random_state(self) -> None:
        original = random.getstate()
        try:
            random.seed(2026)
            before = random.getstate()
            _template()
            self.assertEqual(random.getstate(), before)
        finally:
            random.setstate(original)

    def test_endpoints_express_understanding_and_action(self) -> None:
        neutral = _template(0.0).purpose_clause.lower()
        action = _template(1.0).purpose_clause.lower()
        self.assertIn("learn and understand", neutral)
        self.assertIn("without selecting or carrying out", neutral)
        self.assertIn("complete the concrete action", action)
        self.assertIn("now", action)

    def test_axis_does_not_introduce_source_or_commercial_criteria(self) -> None:
        forbidden = (
            "first-party",
            "brand",
            "popular",
            "authority",
            "freshness",
            "citation",
            "purchase",
            "price",
        )
        for intensity in (0.0, 0.25, 0.5, 0.75, 1.0):
            text = _template(intensity).prompt_template.lower()
            for term in forbidden:
                with self.subTest(intensity=intensity, term=term):
                    self.assertNotIn(term, text)

    def test_required_placeholders_and_top_n_identity_are_preserved(self) -> None:
        first = _template(0.5, top_n=2)
        second = _template(0.5, top_n=3)
        for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
            self.assertIn(placeholder, first.prompt_template)
        self.assertNotEqual(first.prompt_id, second.prompt_id)
        self.assertEqual(first.prompt_hash, second.prompt_hash)

    def test_specification_is_version_checked(self) -> None:
        specification = load_search_purpose_specification()
        self.assertEqual(specification["specification_version"], "search-purpose-axis-v1")
        with self.assertRaises(ValueError):
            load_search_purpose_specification("unknown")


class SearchPurposeRenderingTests(unittest.TestCase):
    def test_keyword_candidates_and_output_size_are_rendered(self) -> None:
        rendered = render_search_purpose_prompt(
            _template(), keyword="password manager", candidates=_candidates(), top_n=2
        )
        self.assertIn("password manager", rendered.rendered_prompt)
        self.assertIn('"candidate_id":"C001"', rendered.rendered_prompt)
        self.assertIn("exactly 2 candidate identifiers", rendered.rendered_prompt)
        for placeholder in ("{QUERY}", "{CANDIDATES}", "{TOP_N}"):
            self.assertNotIn(placeholder, rendered.rendered_prompt)

    def test_candidate_set_is_invariant_across_intensity(self) -> None:
        neutral = render_search_purpose_prompt(
            _template(0.0), keyword="password manager", candidates=_candidates(), top_n=2
        )
        action = render_search_purpose_prompt(
            _template(1.0), keyword="password manager", candidates=_candidates(), top_n=2
        )
        self.assertEqual(neutral.candidate_set_id, action.candidate_set_id)
        self.assertEqual(neutral.candidates, action.candidates)
        self.assertNotEqual(neutral.rendered_prompt_hash, action.rendered_prompt_hash)

    def test_rendering_is_stable_and_normalizes_candidate_newlines(self) -> None:
        candidates = (
            SearchCandidate(1, "Title\ncontinued", "https://a.example", "line one\nline two"),
            SearchCandidate(2, "B", "https://b.example", "text"),
        )
        first = render_search_purpose_prompt(
            _template(), keyword="topic", candidates=candidates, top_n=2
        )
        second = render_search_purpose_prompt(
            _template(), keyword="topic", candidates=candidates, top_n=2
        )
        self.assertEqual(first, second)
        self.assertIn("Title continued", first.rendered_prompt)
        self.assertIn("line one line two", first.rendered_prompt)

    def test_duplicate_candidate_urls_are_rejected(self) -> None:
        duplicate = (
            SearchCandidate(1, "A", "https://same.example", "a"),
            SearchCandidate(2, "B", "https://same.example", "b"),
        )
        with self.assertRaisesRegex(ValueError, "URLs must be unique"):
            render_search_purpose_prompt(
                _template(), keyword="topic", candidates=duplicate, top_n=2
            )


class RankingPermutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rendered = render_search_purpose_prompt(
            _template(), keyword="password manager", candidates=_candidates(), top_n=2
        )

    def test_valid_output_becomes_ordered_top_k(self) -> None:
        result = parse_ranking_permutation("C002\nC001", self.rendered)
        self.assertEqual(result.candidate_ids, ("C002", "C001"))
        self.assertEqual(result.source_position_vector, (2, 1))
        self.assertFalse(result.is_full_permutation)

    def test_comma_separated_identifiers_are_accepted(self) -> None:
        result = parse_ranking_permutation("C003, C001", self.rendered)
        self.assertEqual(result.candidate_ids, ("C003", "C001"))

    def test_explanation_duplicate_unknown_and_wrong_count_are_rejected(self) -> None:
        invalid = (
            "Here are the results: C001 C002",
            "C001 C001",
            "C001 C999",
            "C001",
        )
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(PermutationValidationError):
                parse_ranking_permutation(output, self.rendered)


class SearchPurposePilotTests(unittest.TestCase):
    def test_manifest_crosses_keyword_intensity_and_style_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = write_search_purpose_pilot(
                root,
                keyword_candidates={"password manager": _candidates()},
                intent_grid=(0.0, 0.5, 1.0),
                style_seeds=(7, 8),
                top_n=2,
            )
            rows = [
                json.loads(line)
                for line in artifacts.manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["candidate_set_id"] for row in rows}), 1)
            observed = {
                (row["assigned_action_intensity"], row["style_seed"]) for row in rows
            }
            self.assertEqual(
                observed,
                {(intensity, seed) for intensity in (0.0, 0.5, 1.0) for seed in (7, 8)},
            )
            report = artifacts.report_path.read_text(encoding="utf-8")
            self.assertIn("finite phrase schedule", report)
            self.assertIn("No model inference", report)
            with self.assertRaises(FileExistsError):
                write_search_purpose_pilot(
                    root,
                    keyword_candidates={"password manager": _candidates()},
                    top_n=2,
                )


if __name__ == "__main__":
    unittest.main()
