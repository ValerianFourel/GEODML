"""Contracts for the versioned search-trigger counterfactual audit."""

from __future__ import annotations

import unittest

from analysis.interpretability.pipeline.readiness_prompt_population import (
    ReadinessPromptTarget,
    ReadinessQuestionCandidate,
    search_review_passes_contract,
)
from analysis.scripts.audit_readiness_search_trigger_counterfactual import (
    Scenario,
    evaluate_scenarios,
)


def _review(**overrides):
    row = {
        "exact_keyword_present": True,
        "single_question": True,
        "topic_relevant": True,
        "search_intent": True,
        "web_answerable": True,
        "standalone": True,
        "natural_language": True,
        "relevance_score_1_5": 5,
    }
    row.update(overrides)
    return row


def _candidate(candidate_id: str, question: str) -> ReadinessQuestionCandidate:
    return ReadinessQuestionCandidate(
        candidate_id=candidate_id,
        task_id=f"task:{candidate_id}",
        keyword_id="keyword:one",
        keyword="password manager",
        target_id="target:source",
        target_index=0,
        target_normalized_axis_1=0.0,
        target_normalized_axis_2=0.5,
        target_raw_axis_1=0.0,
        target_raw_axis_2=0.5,
        round_index=0,
        generator_id="generator",
        generator_model="model",
        candidate_slot=0,
        generation_seed=1,
        question=question,
        question_sha256=f"sha:{candidate_id}",
        proposal_kind="test",
    )


def _target(target_id: str, index: int, axis_1: float) -> ReadinessPromptTarget:
    return ReadinessPromptTarget(
        target_id=target_id,
        target_index=index,
        axis_1_index=index,
        axis_2_index=0,
        normalized_axis_1=axis_1,
        normalized_axis_2=0.5,
        raw_axis_1=axis_1,
        raw_axis_2=0.5,
    )


def _coordinate(axis_1: float):
    return {
        "reference_normalized_axis_1": axis_1,
        "reference_normalized_axis_2": 0.5,
        "candidate_aligned_normalized_axis_1": axis_1,
        "candidate_aligned_normalized_axis_2": 0.5,
        "consensus_normalized_axis_1": axis_1,
        "consensus_normalized_axis_2": 0.5,
        "cross_embedding_disagreement": 0.0,
    }


class SearchTriggerCounterfactualTests(unittest.TestCase):
    def test_v2_ignores_keyword_question_and_standalone_gates(self) -> None:
        relaxed_only = _review(
            exact_keyword_present=False,
            single_question=False,
            standalone=False,
        )
        self.assertFalse(
            search_review_passes_contract(
                relaxed_only,
                contract="question-v1",
            )
        )
        self.assertTrue(
            search_review_passes_contract(
                relaxed_only,
                contract="search-trigger-v2",
            )
        )

    def test_v2_retains_search_and_quality_requirements(self) -> None:
        for field in (
            "topic_relevant",
            "search_intent",
            "web_answerable",
            "natural_language",
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    search_review_passes_contract(
                        _review(**{field: False}),
                        contract="search-trigger-v2",
                    )
                )
        self.assertFalse(
            search_review_passes_contract(
                _review(relevance_score_1_5=3),
                contract="search-trigger-v2",
            )
        )

    def test_scenarios_measure_relaxed_acceptance_without_mutating_gold(self) -> None:
        candidates = (
            _candidate("candidate:a", "How do I compare password manager options?"),
            _candidate("candidate:b", "Install one of these tools now"),
        )
        targets = (_target("target:low", 0, 0.2), _target("target:high", 1, 0.8))
        coordinates = {
            "candidate:a": _coordinate(0.2),
            "candidate:b": _coordinate(0.8),
        }
        scenarios = (
            Scenario(
                "gold",
                frozenset({"candidate:a"}),
                0.035,
                True,
            ),
            Scenario(
                "relaxed",
                frozenset({"candidate:a", "candidate:b"}),
                0.035,
                True,
            ),
        )
        results = evaluate_scenarios(
            candidates,
            targets,
            coordinates,
            scenarios,
            target_design="axis-1-linear",
            planned_keywords=(("keyword:one", "password manager"),),
            disagreement_weight=0.10,
        )
        self.assertEqual(len(results["gold"]["selected"]), 1)
        self.assertEqual(len(results["relaxed"]["selected"]), 2)
        self.assertEqual(
            {row.candidate_id for row in results["gold"]["selected"]},
            {"candidate:a"},
        )

    def test_contract_name_must_be_explicitly_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            search_review_passes_contract(_review(), contract="future-v3")


if __name__ == "__main__":
    unittest.main()
