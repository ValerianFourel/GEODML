"""Behavioral contracts for the calibrated two-axis prompt population."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.search_purpose_continuum import (
    PermutationValidationError,
    SearchCandidate,
)
from analysis.interpretability.pipeline.two_axis_prompt_population import (
    FakePairwiseJudge,
    FakeTwoAxisCandidateGenerator,
    FakeTwoAxisPromptEmbedder,
    LocalLLMPairwiseJudge,
    LocalLLMTwoAxisCandidateGenerator,
    PairwiseComparisonRequest,
    TwoAxisCandidateRequest,
    build_pairwise_comparison_requests,
    calibrate_candidates,
    generate_candidate_bank,
    judge_comparison_requests,
    load_population_specification,
    map_two_axis_prompt_to_permutation,
    measure_selected_latent_population,
    render_selected_two_axis_prompt,
    select_prompt_population,
)


GRID = (0.0, 0.5, 1.0)


class _StaticRanker:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0

    def rank(self, prompt, max_tokens=500, temperature=0.1):
        del prompt, max_tokens, temperature
        self.call_count += 1
        return self.outputs.pop(0)


class _RecordingRanker(_StaticRanker):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.prompts = []

    def rank(self, prompt, max_tokens=500, temperature=0.1):
        self.prompts.append(prompt)
        return super().rank(prompt, max_tokens=max_tokens, temperature=temperature)


def _pipeline():
    candidates = generate_candidate_bank(
        search_term="abandoned cart recovery",
        a1_grid=GRID,
        a2_grid=GRID,
        style_seeds=(3, 4),
        number_candidates=3,
        generator=FakeTwoAxisCandidateGenerator(),
    )
    requests = build_pairwise_comparison_requests(candidates)
    judgments = judge_comparison_requests(
        requests,
        candidates,
        (FakePairwiseJudge("judge-one"), FakePairwiseJudge("judge-two")),
    )
    calibrations = calibrate_candidates(candidates, requests, judgments)
    selected, diagnostics = select_prompt_population(
        candidates,
        calibrations,
        embedder=FakeTwoAxisPromptEmbedder(),
        monotonic_tolerance=0.02,
    )
    return candidates, requests, judgments, calibrations, selected, diagnostics


class CandidatePopulationTests(unittest.TestCase):
    def test_specification_is_versioned(self) -> None:
        self.assertEqual(
            load_population_specification()["specification_version"],
            "two-axis-prompt-population-v1",
        )

    def test_complete_grid_has_multiple_candidates_per_style(self) -> None:
        candidates, *_ = _pipeline()
        self.assertEqual(len(candidates), 2 * 3 * 3 * 3)
        self.assertTrue(all(candidate.structural_valid for candidate in candidates))
        self.assertTrue(all(candidate.search_term == "abandoned cart recovery" for candidate in candidates))
        self.assertTrue(all("abandoned cart recovery" not in candidate.prompt_template for candidate in candidates))
        self.assertTrue(all("{QUERY}" in candidate.prompt_template for candidate in candidates))

    def test_actor_task_and_output_contract_are_invariant(self) -> None:
        candidates, *_ = _pipeline()
        self.assertEqual(len({candidate.business_actor for candidate in candidates}), 1)
        self.assertEqual(len({candidate.output_contract for candidate in candidates}), 1)
        for candidate in candidates:
            lowered = candidate.prompt_template.lower()
            self.assertIn("business software evaluator", lowered)
            self.assertIn("candidate identifiers only", lowered)
            self.assertNotIn("price", lowered)

    def test_same_request_is_reproducible(self) -> None:
        first = generate_candidate_bank(
            search_term="crm software",
            a1_grid=GRID,
            a2_grid=GRID,
            style_seeds=(1,),
            number_candidates=2,
            generator=FakeTwoAxisCandidateGenerator(),
        )
        second = generate_candidate_bank(
            search_term="crm software",
            a1_grid=GRID,
            a2_grid=GRID,
            style_seeds=(1,),
            number_candidates=2,
            generator=FakeTwoAxisCandidateGenerator(),
        )
        self.assertEqual(first, second)

    def test_real_generator_parses_and_caches_structured_clauses(self) -> None:
        payload = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category without selecting a product.",
                    "source_preference_clause": "Conditional on equal topical relevance, prefer independent evidence.",
                },
                {
                    "search_objective_clause": "Learn the category mechanisms before evaluating products.",
                    "source_preference_clause": "Treat publisher ownership as neutral and rank by relevance.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker([json.dumps(payload)])
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.5,
                assigned_a2=0.5,
                style_seed=2,
                generation_seed=4,
                number_candidates=2,
                generator_model="test-generator",
            )
            self.assertEqual(generator.generate(request), generator.generate(request))
            self.assertEqual(ranker.call_count, 1)
            cached = list(Path(directory).glob("*.json"))
            self.assertEqual(len(cached), 1)

    def test_real_generator_retries_an_off_axis_candidate(self) -> None:
        invalid = {
            "candidates": [
                {
                    "search_objective_clause": "Find the most popular and recent solution.",
                    "source_preference_clause": "Rank by relevance.",
                },
                {
                    "search_objective_clause": "Understand the category.",
                    "source_preference_clause": "Prefer authoritative sources.",
                },
            ]
        }
        valid = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category without selecting a product.",
                    "source_preference_clause": "Conditional on equal topical relevance, prefer independent evidence.",
                },
                {
                    "search_objective_clause": "Learn the category mechanisms before evaluation.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker([json.dumps(invalid), json.dumps(valid)])
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.5,
                assigned_a2=0.5,
                style_seed=2,
                generation_seed=4,
                number_candidates=2,
                generator_model="test-generator",
            )
            self.assertEqual(len(generator.generate(request)), 2)
            self.assertEqual(ranker.call_count, 2)


class PairwiseCalibrationTests(unittest.TestCase):
    def test_every_comparison_is_presented_in_both_orders(self) -> None:
        _, requests, *_ = _pipeline()
        unordered = {}
        for request in requests:
            key = (
                request.axis,
                request.style_seed,
                request.fixed_coordinate,
                frozenset((request.left_candidate_id, request.right_candidate_id)),
                request.comparison_kind,
            )
            unordered.setdefault(key, set()).add(request.presentation_order)
        self.assertTrue(unordered)
        self.assertTrue(all(orders == {"forward", "reverse"} for orders in unordered.values()))

    def test_calibration_orders_frozen_endpoints(self) -> None:
        candidates, _, _, calibrations, *_ = _pipeline()
        calibration = {item.candidate_id: item for item in calibrations}
        for style in (3, 4):
            for fixed in GRID:
                low_a1 = [
                    calibration[candidate.candidate_id].realized_a1
                    for candidate in candidates
                    if candidate.style_seed == style
                    and candidate.assigned_a1 == 0
                    and candidate.assigned_a2 == fixed
                ]
                high_a1 = [
                    calibration[candidate.candidate_id].realized_a1
                    for candidate in candidates
                    if candidate.style_seed == style
                    and candidate.assigned_a1 == 1
                    and candidate.assigned_a2 == fixed
                ]
                self.assertLess(sum(low_a1) / len(low_a1), sum(high_a1) / len(high_a1))

    def test_real_judge_maps_blind_json_label_and_caches(self) -> None:
        candidates, *_ = _pipeline()
        left, right = candidates[0], candidates[1]
        request = PairwiseComparisonRequest(
            comparison_id="comparison-one",
            axis="A1",
            style_seed=left.style_seed,
            fixed_coordinate=left.assigned_a2,
            left_candidate_id=left.candidate_id,
            right_candidate_id=right.candidate_id,
            presentation_order="forward",
            comparison_kind="within-cell",
            question="Which is more decision-ready?",
        )
        with tempfile.TemporaryDirectory() as directory:
            ranker = _RecordingRanker(['{"winner":"right"}'])
            judge = LocalLLMPairwiseJudge(
                ranker,
                judge_id="judge-one",
                model_name="test-judge",
                cache_directory=directory,
            )
            mapping = {left.candidate_id: left, right.candidate_id: right}
            self.assertEqual(judge.compare(request, mapping), right.candidate_id)
            self.assertEqual(judge.compare(request, mapping), right.candidate_id)
            self.assertEqual(ranker.call_count, 1)
            self.assertIn('Search term: "abandoned cart recovery"', ranker.prompts[0])
            self.assertNotIn('Search term: "{QUERY}"', ranker.prompts[0])


class GlobalSelectionTests(unittest.TestCase):
    def test_selection_is_complete_unique_and_monotone(self) -> None:
        *_, selected, diagnostics = _pipeline()
        self.assertEqual(len(selected), 18)
        self.assertEqual(len({item.prompt_assignment_id for item in selected}), 18)
        self.assertEqual(len({item.candidate_hash for item in selected}), 18)
        self.assertEqual(diagnostics.a1_adjacent_reversal_rate, 0.0)
        self.assertEqual(diagnostics.a2_adjacent_reversal_rate, 0.0)
        self.assertEqual(diagnostics.fully_monotone_style_rate, 1.0)

    def test_latent_diagnostics_use_query_bound_selected_prompts(self) -> None:
        *_, selected, _ = _pipeline()
        diagnostics = measure_selected_latent_population(selected)
        self.assertEqual(diagnostics.selected_count, 18)
        self.assertEqual(diagnostics.exact_query_structural_retention_rate, 1.0)
        self.assertGreater(diagnostics.a1_endpoint_distance, 0.0)
        self.assertGreater(diagnostics.a2_endpoint_distance, 0.0)
        self.assertLess(diagnostics.adjacent_over_distant_distance_ratio, 1.0)

    def test_selected_hashes_are_not_reused_across_styles(self) -> None:
        candidates, _, _, calibrations, *_ = _pipeline()
        first_style_hashes = {
            (candidate.assigned_a1, candidate.assigned_a2): candidate.candidate_hash
            for candidate in candidates
            if candidate.style_seed == 3 and candidate.candidate_index == 0
        }
        candidates = tuple(
            replace(
                candidate,
                candidate_hash=first_style_hashes[
                    (candidate.assigned_a1, candidate.assigned_a2)
                ],
            )
            if candidate.style_seed == 4 and candidate.candidate_index == 0
            else candidate
            for candidate in candidates
        )
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        calibrations = tuple(
            replace(
                calibration,
                realized_a1=candidate_by_id[calibration.candidate_id].assigned_a1
                + (0.0 if candidate_by_id[calibration.candidate_id].candidate_index == 0 else 2.0),
                realized_a2=candidate_by_id[calibration.candidate_id].assigned_a2
                + (0.0 if candidate_by_id[calibration.candidate_id].candidate_index == 0 else 2.0),
            )
            for calibration in calibrations
        )
        selected, diagnostics = select_prompt_population(
            candidates,
            calibrations,
            embedder=FakeTwoAxisPromptEmbedder(),
            monotonic_tolerance=0.02,
        )
        self.assertEqual(len(selected), 18)
        self.assertEqual(len({item.candidate_hash for item in selected}), 18)
        self.assertEqual(diagnostics.duplicate_hash_count, 0)
        self.assertTrue(
            all(item.candidate_index == 0 for item in selected if item.style_seed == 3)
        )
        self.assertTrue(
            all(item.candidate_index != 0 for item in selected if item.style_seed == 4)
        )

    def test_impossible_neighbor_bound_fails_explicitly(self) -> None:
        candidates, _, _, calibrations, *_ = _pipeline()
        with self.assertRaisesRegex(ValueError, "no feasible globally selected"):
            select_prompt_population(
                candidates,
                calibrations,
                embedder=FakeTwoAxisPromptEmbedder(),
                monotonic_tolerance=0.02,
                maximum_neighbor_embedding_distance=1e-8,
            )

    def test_selected_prompt_maps_to_strict_permutation(self) -> None:
        *_, selected, _ = _pipeline()
        prompt = selected[0]
        candidates = (
            SearchCandidate(1, "Vendor guide", "https://vendor.example", "Guide"),
            SearchCandidate(2, "Independent review", "https://review.example", "Review"),
            SearchCandidate(3, "Industry study", "https://study.example", "Study"),
        )
        rendered = render_selected_two_axis_prompt(prompt, candidates=candidates, top_n=2)
        self.assertIn('Search term: "abandoned cart recovery"', rendered.rendered_prompt)
        outcome = map_two_axis_prompt_to_permutation(
            prompt,
            rendered,
            "C002 C001",
            reranker_run_id="test-run",
            reranker_model="test-reranker",
        )
        self.assertEqual(outcome.assigned_a1, prompt.assigned_a1)
        self.assertEqual(outcome.assigned_a2, prompt.assigned_a2)
        self.assertEqual(outcome.ranking.source_position_vector, (2, 1))
        with self.assertRaises(PermutationValidationError):
            map_two_axis_prompt_to_permutation(
                prompt,
                rendered,
                "C002 because it is better",
                reranker_run_id="test-run",
                reranker_model="test-reranker",
            )


if __name__ == "__main__":
    unittest.main()
