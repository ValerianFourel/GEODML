"""Behavioral contracts for the A1-only prompt manifold pilot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from analysis.interpretability.pipeline.a1_prompt_manifold import (
    BUSINESS_ACTOR,
    NEUTRAL_SOURCE_CLAUSE,
    A1CandidateRequest,
    FakeA1CandidateGenerator,
    FakeA1Embedder,
    FakeA1PairwiseJudge,
    LocalLLMA1CandidateGenerator,
    a1_contract_checks,
    build_a1_comparison_requests,
    calibrate_a1_candidates,
    embed_a1_candidates,
    generate_a1_candidate_bank,
    judge_a1_comparisons,
    select_a1_manifold,
    stratified_random_a1_grid,
)

SCRIPTS_ROOT = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_a1_prompt_manifold_pilot import _paths, _prepare_generation_root  # noqa: E402


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
    def test_stratified_random_grid_is_reproducible_smooth_and_seeded(self) -> None:
        first = stratified_random_a1_grid(30, master_seed=20260817)
        repeat = stratified_random_a1_grid(30, master_seed=20260817)
        other = stratified_random_a1_grid(30, master_seed=20260818)

        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 30)
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first[-1], 1.0)
        self.assertEqual(tuple(sorted(set(first))), first)
        nominal_step = 1 / 29
        steps = [right - left for left, right in zip(first, first[1:])]
        self.assertGreater(min(steps), 0.2 * nominal_step)
        self.assertLess(max(steps), 1.8 * nominal_step)

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

    def test_contract_accepts_assess_as_endpoint_evaluation_language(self) -> None:
        objective = (
            "Identify and assess B2B SaaS solutions that align with organizational "
            "needs and operational goals."
        )
        template = (
            "Search objective: "
            + objective
            + "\nSource policy: "
            + NEUTRAL_SOURCE_CLAUSE
            + '\nSearch term: "{QUERY}"\nCandidates:\n{CANDIDATES}\n{TOP_N}'
        )
        self.assertEqual(
            a1_contract_checks(
                template,
                objective,
                assigned_a1=1.0,
                search_term="abandoned cart recovery",
            ),
            (),
        )

    def test_contract_treats_pre_assessment_language_as_non_selection(self) -> None:
        objective = "Understand the category mechanisms before assessing any product."
        template = (
            "Search objective: "
            + objective
            + "\nSource policy: "
            + NEUTRAL_SOURCE_CLAUSE
            + '\nSearch term: "{QUERY}"\nCandidates:\n{CANDIDATES}\n{TOP_N}'
        )
        self.assertEqual(
            a1_contract_checks(
                template,
                objective,
                assigned_a1=0.0,
                search_term="abandoned cart recovery",
            ),
            (),
        )

    def test_cached_candidate_is_revalidated_before_resume(self) -> None:
        class Ranker:
            def __init__(self) -> None:
                self.calls = 0

            def rank(self, *_args, **_kwargs):
                self.calls += 1
                return json.dumps(
                    {
                        "search_objective_clause": (
                            "Identify and assess B2B SaaS solutions that align with "
                            "organizational needs."
                        )
                    }
                )

        request = A1CandidateRequest(
            assigned_a1=1.0,
            style_seed=0,
            generation_seed=3146858096,
            number_candidates=1,
            generator_model="test-generator",
        )
        with tempfile.TemporaryDirectory() as directory:
            first_ranker = Ranker()
            generator = LocalLLMA1CandidateGenerator(
                first_ranker,
                model_name="test-generator",
                cache_directory=directory,
            )
            expected = generator.generate(request)
            self.assertEqual(first_ranker.calls, 1)

            cached_ranker = Ranker()
            cached_generator = LocalLLMA1CandidateGenerator(
                cached_ranker,
                model_name="test-generator",
                cache_directory=directory,
            )
            cached_result = cached_generator.generate(request)
            self.assertEqual(cached_result, expected)
            self.assertEqual(cached_ranker.calls, 0)

            cache_path = next(Path(directory).glob("*.json"))
            payload = json.loads(cache_path.read_text())
            payload["identity"]["slot"] = 99
            cache_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cached_generator.generate(request)

    def test_generation_resume_accepts_only_a_partial_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "partial"
            cache = root / "cache" / "generation"
            cache.mkdir(parents=True)
            (cache / "one.json").write_text("{}")
            (cache / "two.json").write_text("{}")
            (cache / "three.failed.json").write_text("{}")
            paths = _paths(root)

            with self.assertRaisesRegex(ValueError, "already exists"):
                _prepare_generation_root(paths, resume=False)
            self.assertEqual(_prepare_generation_root(paths, resume=True), 2)

            paths["manifest"].write_text(
                json.dumps({"status": "generated-unjudged"})
            )
            with self.assertRaisesRegex(ValueError, "already complete"):
                _prepare_generation_root(paths, resume=True)

    def test_generation_resume_rejects_an_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "unrelated"
            root.mkdir()
            (root / "unrelated.txt").write_text("do not overwrite")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                _prepare_generation_root(_paths(root), resume=True)

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
