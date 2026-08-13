"""Behavioral contracts for the A1-only prompt manifold pilot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from analysis.interpretability.pipeline.a1_prompt_manifold import (
    BUSINESS_ACTOR,
    NEUTRAL_SOURCE_CLAUSE,
    FakeA1CandidateGenerator,
    FakeA1Embedder,
    FakeA1PairwiseJudge,
    a1_contract_checks,
    build_a1_comparison_requests,
    calibrate_a1_candidates,
    embed_a1_candidates,
    generate_a1_candidate_bank,
    judge_a1_comparisons,
    select_a1_manifold,
)


def _pipeline():
    candidates = generate_a1_candidate_bank(
        search_term="abandoned cart recovery",
        style_seeds=(2, 3),
        number_candidates=4,
        generator=FakeA1CandidateGenerator(),
    )
    comparisons = build_a1_comparison_requests(candidates)
    judgments = judge_a1_comparisons(
        comparisons,
        candidates,
        (FakeA1PairwiseJudge("judge-one"), FakeA1PairwiseJudge("judge-two")),
    )
    calibrations = calibrate_a1_candidates(candidates, comparisons, judgments)
    input_rows = embed_a1_candidates(
        candidates,
        embedder=FakeA1Embedder("fake-llm2vec"),
        representation="input",
    )
    response_rows = embed_a1_candidates(
        candidates,
        embedder=FakeA1Embedder("fake-llm2vec-gen", response=True),
        representation="anticipated-response",
    )
    selected, diagnostics = select_a1_manifold(
        candidates, calibrations, input_rows, response_rows
    )
    return (
        candidates,
        comparisons,
        judgments,
        calibrations,
        input_rows,
        response_rows,
        selected,
        diagnostics,
    )


class A1PromptManifoldTests(unittest.TestCase):
    def test_candidate_bank_generates_only_the_objective_field(self) -> None:
        candidates, *_ = _pipeline()
        self.assertEqual(len(candidates), 2 * 7 * 4)
        self.assertTrue(all(item.structural_valid for item in candidates))
        self.assertTrue(
            all("abandoned cart recovery" not in item.prompt_template for item in candidates)
        )
        self.assertTrue(all(item.prompt_template.count("{QUERY}") == 1 for item in candidates))
        self.assertTrue(all(NEUTRAL_SOURCE_CLAUSE in item.prompt_template for item in candidates))
        self.assertTrue(all(BUSINESS_ACTOR in item.prompt_template for item in candidates))
        self.assertEqual(len({item.candidate_hash for item in candidates}), len(candidates))

    def test_contract_rejects_non_a1_semantics(self) -> None:
        template = (
            "Act as a business evaluator. Search objective: Prefer first-party publisher "
            "evidence at A1=0.5. Source policy: "
            + NEUTRAL_SOURCE_CLAUSE
            + '\nSearch term: "{QUERY}"\nCandidates:\n{CANDIDATES}\n{TOP_N}'
        )
        failures = a1_contract_checks(
            template,
            "Prefer first-party publisher evidence at A1=0.5.",
            assigned_a1=0.5,
            search_term="abandoned cart recovery",
        )
        self.assertIn("off-axis-criterion", failures)
        self.assertIn("numeric-coordinate-leak", failures)

    def test_contract_rejects_candidate_cardinality(self) -> None:
        objective = "Compare three candidates while preparing to shortlist a solution."
        template = (
            "Search objective: "
            + objective
            + "\nSource policy: "
            + NEUTRAL_SOURCE_CLAUSE
            + '\nSearch term: "{QUERY}"\nCandidates:\n{CANDIDATES}\n{TOP_N}'
        )
        failures = a1_contract_checks(
            template,
            objective,
            assigned_a1=0.75,
            search_term="abandoned cart recovery",
        )
        self.assertIn("candidate-cardinality-leak", failures)

    def test_pairwise_requests_are_blind_and_reversed(self) -> None:
        candidates, comparisons, *_ = _pipeline()
        self.assertTrue(comparisons)
        by_unordered_pair = {}
        for item in comparisons:
            key = tuple(sorted((item.left_candidate_id, item.right_candidate_id)))
            by_unordered_pair.setdefault(key, set()).add(item.presentation_order)
        self.assertTrue(all(value == {"forward", "reverse"} for value in by_unordered_pair.values()))
        self.assertEqual({item.question for item in comparisons}, {"decision-readiness"})
        self.assertTrue(all(item.candidate_id for item in candidates))

    def test_dual_representation_selection_is_complete_unique_and_strict(self) -> None:
        *_, input_rows, response_rows, selected, diagnostics = _pipeline()
        self.assertEqual(len(input_rows), 56)
        self.assertEqual(len(response_rows), 56)
        self.assertEqual(len(selected), 14)
        self.assertEqual(len({item.candidate_hash for item in selected}), 14)
        self.assertEqual(diagnostics.adjacent_reversal_rate, 0.0)
        self.assertEqual(diagnostics.fully_strict_monotone_style_rate, 1.0)
        self.assertEqual(diagnostics.exact_query_structural_retention_rate, 1.0)
        self.assertEqual(diagnostics.mean_style_spearman, 1.0)
        self.assertGreaterEqual(diagnostics.input_mean_tortuosity, 1.0)
        self.assertGreaterEqual(diagnostics.response_mean_tortuosity, 1.0)

    def test_fake_cli_writes_complete_contract_artifacts(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "run_a1_prompt_manifold_pilot.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            completed = subprocess.run(
                [
                    "python3",
                    str(script),
                    "fake-smoke",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("no scientific claim", completed.stdout)
            self.assertEqual(
                json.loads((output / "run_manifest.json").read_text())["scientific_result"],
                False,
            )
            with (output / "selected_a1_prompt_manifold.jsonl").open() as handle:
                self.assertEqual(sum(1 for _ in handle), 14)


if __name__ == "__main__":
    unittest.main()
