"""Behavioral contracts for the calibrated two-axis prompt population."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

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
    LLM2VecPromptEmbedder,
    PairwiseComparisonRequest,
    PairwiseJudgment,
    TwoAxisCandidateRequest,
    build_pairwise_comparison_requests,
    calibrate_candidates,
    diagnose_pairwise_judgments,
    generate_candidate_bank,
    judge_comparison_requests,
    load_population_specification,
    map_two_axis_prompt_to_permutation,
    measure_selected_latent_population,
    render_selected_two_axis_prompt,
    select_prompt_population,
    semantic_contract_checks,
)


GRID = (0.0, 0.5, 1.0)


class _StaticRanker:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0

    def rank(
        self,
        prompt,
        max_tokens=500,
        temperature=0.1,
        *,
        chat_template_kwargs=None,
    ):
        del prompt, max_tokens, temperature
        self.chat_template_kwargs = chat_template_kwargs
        self.call_count += 1
        return self.outputs.pop(0)


class _RecordingRanker(_StaticRanker):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.prompts = []

    def rank(
        self,
        prompt,
        max_tokens=500,
        temperature=0.1,
        *,
        chat_template_kwargs=None,
    ):
        self.prompts.append(prompt)
        return super().rank(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            chat_template_kwargs=chat_template_kwargs,
        )


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

    def test_llm2vec_installer_pins_qwen_capable_upstream_revision(self) -> None:
        script = (
            Path(__file__).parents[1] / "scripts" / "install_llm2vec_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("0fbcf3304139099bda75c3d6b5d8e835d4894563", script)
        self.assertIn('REGEX_VERSION="2025.11.3"', script)
        self.assertIn('TRANSFORMERS_VERSION="4.56.2"', script)
        self.assertIn('PEFT_VERSION="0.18.0"', script)
        self.assertIn('HUGGINGFACE_HUB_VERSION="0.36.2"', script)
        self.assertIn('TOKENIZERS_VERSION="0.22.2"', script)
        self.assertIn('SAFETENSORS_VERSION="0.8.0"', script)
        self.assertIn('ACCELERATE_VERSION="1.14.0"', script)
        self.assertIn('"regex==${REGEX_VERSION}"', script)
        self.assertIn('"transformers==${TRANSFORMERS_VERSION}"', script)
        self.assertIn('"peft==${PEFT_VERSION}"', script)
        self.assertIn('export PYTHONPATH="${VENV_SITE_PACKAGES}', script)
        self.assertGreaterEqual(script.count("--no-deps"), 2)
        self.assertIn("--no-deps", script)
        self.assertIn("--force-reinstall", script)
        self.assertIn("bidirectional_qwen2", script)
        self.assertIn("bidirectional_qwen3", script)
        self.assertNotIn("llm2vec==0.2.3", script)

    def test_llm2vec_loader_merges_mntp_before_loading_simcse(self) -> None:
        calls = []

        class FakeCUDA:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

        class FakeLLM2Vec:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append(("llm2vec", model_name, kwargs))
                return types.SimpleNamespace(
                    model=types.SimpleNamespace(config="merged-mntp-config"),
                    config="merged-mntp-config",
                )

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, model_name, **kwargs):
                calls.append(("peft", model_name, kwargs, model.config))
                return types.SimpleNamespace(config="simcse-config")

        modules = {
            "torch": types.SimpleNamespace(cuda=FakeCUDA(), bfloat16="bf16"),
            "llm2vec": types.SimpleNamespace(LLM2Vec=FakeLLM2Vec),
            "peft": types.SimpleNamespace(PeftModel=FakePeftModel),
        }
        with patch.dict(sys.modules, modules):
            embedder = LLM2VecPromptEmbedder(
                "qwen-base",
                mntp_model_name_or_path="mntp-adapter",
                peft_model_name_or_path="simcse-adapter",
                max_length=256,
            )

        self.assertEqual(calls[0][0:2], ("llm2vec", "qwen-base"))
        self.assertEqual(calls[0][2]["peft_model_name_or_path"], "mntp-adapter")
        self.assertTrue(calls[0][2]["merge_peft"])
        self.assertEqual(calls[0][2]["max_length"], 256)
        self.assertEqual(
            calls[1],
            (
                "peft",
                "simcse-adapter",
                {"local_files_only": True},
                "merged-mntp-config",
            ),
        )
        self.assertEqual(embedder._model.config, "simcse-config")
        self.assertEqual(
            embedder.model_name,
            "qwen-base+mntp:mntp-adapter+peft:simcse-adapter",
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
                    "search_objective_clause": "Understand the category and develop practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
                {
                    "search_objective_clause": "Learn the category mechanisms and practical evaluation criteria.",
                    "source_preference_clause": "Treat publisher ownership as neutral and rank by relevance.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(
                [json.dumps({"candidates": [candidate]}) for candidate in payload["candidates"]]
            )
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
            self.assertEqual(ranker.call_count, 2)
            self.assertEqual(ranker.chat_template_kwargs, {"enable_thinking": False})
            cached = list(Path(directory).glob("*.json"))
            self.assertEqual(len(cached), 1)

    def test_real_generator_extracts_fenced_json_after_thinking(self) -> None:
        payload = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category and develop practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
                {
                    "search_objective_clause": "Learn the category mechanisms and practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
            ]
        }
        wrapped = [
            "<think>check the constraints</think>\n```json\n"
            + json.dumps({"candidates": [candidate]})
            + "\n```"
            for candidate in payload["candidates"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(wrapped)
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

    def test_real_generator_preserves_raw_outputs_after_exhausted_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(["not json", "still not json"])
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
                maximum_attempts=2,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.5,
                assigned_a2=0.5,
                style_seed=2,
                generation_seed=4,
                number_candidates=2,
                generator_model="test-generator",
            )
            with self.assertRaisesRegex(ValueError, "rejected outputs"):
                generator.generate(request)
            failed = list(Path(directory).glob("*.failed.json"))
            self.assertEqual(len(failed), 1)
            payload = json.loads(failed[0].read_text(encoding="utf-8"))
            self.assertEqual(
                [item["raw"] for item in payload["rejected_attempts"]],
                ["not json", "still not json"],
            )

    def test_real_generator_rejects_multiple_schema_shaped_objects(self) -> None:
        payload = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category and develop practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
                {
                    "search_objective_clause": "Learn the category mechanisms and practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
            ]
        }
        ambiguous = json.dumps(payload) + "\n" + json.dumps(payload)
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker([ambiguous])
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
                maximum_attempts=1,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.5,
                assigned_a2=0.5,
                style_seed=2,
                generation_seed=4,
                number_candidates=2,
                generator_model="test-generator",
            )
            with self.assertRaisesRegex(ValueError, "multiple candidate JSON objects"):
                generator.generate(request)

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
                    "search_objective_clause": "Understand the category and develop practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
                {
                    "search_objective_clause": "Learn the category mechanisms and practical evaluation criteria.",
                    "source_preference_clause": "Apply no publisher-ownership preference and rank by relevance.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(
                [
                    json.dumps({"candidates": [invalid["candidates"][0]]}),
                    json.dumps({"candidates": [valid["candidates"][0]]}),
                    json.dumps({"candidates": [valid["candidates"][1]]}),
                ]
            )
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
            self.assertEqual(ranker.call_count, 3)

    def test_real_generator_retries_wrong_coordinate_directions(self) -> None:
        reversed_a2 = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category without selecting a product.",
                    "source_preference_clause": "Prefer seller-controlled evidence, conditional on equal topical relevance.",
                },
                {
                    "search_objective_clause": "Explore category mechanisms without choosing a product.",
                    "source_preference_clause": "Favor vendor-controlled content when it is equally relevant.",
                },
            ]
        }
        valid = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category without selecting a product.",
                    "source_preference_clause": "Prefer seller-independent evidence, conditional on equal topical relevance.",
                },
                {
                    "search_objective_clause": "Explore category mechanisms without choosing a product.",
                    "source_preference_clause": "Favor independent research when it is equally relevant.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(
                [
                    json.dumps({"candidates": [reversed_a2["candidates"][0]]}),
                    json.dumps({"candidates": [valid["candidates"][0]]}),
                    json.dumps({"candidates": [valid["candidates"][1]]}),
                ]
            )
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.0,
                assigned_a2=0.0,
                style_seed=2,
                generation_seed=4,
                number_candidates=2,
                generator_model="test-generator",
            )
            generated = generator.generate(request)
            self.assertEqual(generated[0][1], valid["candidates"][0]["source_preference_clause"])
            self.assertEqual(ranker.call_count, 3)
            cache = next(
                path
                for path in Path(directory).glob("*.json")
                if not path.name.endswith(".failed.json")
            )
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.assertIn("coordinate-mismatch:A2-independent", payload["rejected_attempts"][0]["error"])

    def test_real_generator_retries_a1_low_evaluation_contamination(self) -> None:
        contaminated = {
            "candidates": [
                {
                    "search_objective_clause": "Understand the category and develop evaluation criteria.",
                    "source_preference_clause": "Prefer seller-independent evidence, conditional on equal topical relevance.",
                },
                {
                    "search_objective_clause": "Learn the category and form practical evaluation approaches.",
                    "source_preference_clause": "Favor independent research when it is equally relevant.",
                },
            ]
        }
        valid = {
            "candidates": [
                {
                    "search_objective_clause": "Understand category mechanisms without selecting a product.",
                    "source_preference_clause": "Prefer seller-independent evidence, conditional on equal topical relevance.",
                },
                {
                    "search_objective_clause": "Explore relevant concepts without choosing a product.",
                    "source_preference_clause": "Favor independent research when it is equally relevant.",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(
                [
                    json.dumps({"candidates": [contaminated["candidates"][0]]}),
                    json.dumps({"candidates": [valid["candidates"][0]]}),
                    json.dumps({"candidates": [valid["candidates"][1]]}),
                ]
            )
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.0,
                assigned_a2=0.0,
                style_seed=2,
                generation_seed=4,
                number_candidates=2,
                generator_model="test-generator",
            )
            self.assertEqual(generator.generate(request)[0][0], valid["candidates"][0]["search_objective_clause"])
            self.assertEqual(ranker.call_count, 3)

    def test_real_generator_accepts_comparative_a2_and_midpoint_paraphrase(self) -> None:
        rows = [
            {
                "search_objective_clause": "Prioritize candidates that demonstrate a clear grasp of the category, accompanied by actionable evaluation methods and feasible implementation strategies.",
                "source_preference_clause": "Give preference to vendor-controlled content, such as product documentation or case studies, provided it is relevant to the topic.",
            },
            {
                "search_objective_clause": "Arrange the candidates based on their relevance to category understanding, practical evaluation criteria, and potential solution approaches.",
                "source_preference_clause": "Prioritize vendor-controlled evidence over seller-independent evidence, provided that topical relevance is equivalent.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(
                [json.dumps({"candidates": [candidate]}) for candidate in rows]
            )
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.5,
                assigned_a2=1.0,
                style_seed=0,
                generation_seed=10,
                number_candidates=2,
                generator_model="test-generator",
            )
            generated = generator.generate(request)
            self.assertEqual(len(generated), 2)
            self.assertEqual(ranker.call_count, 2)

    def test_real_generator_retries_candidate_cardinality_leak(self) -> None:
        invalid = {
            "search_objective_clause": "Arrange the three candidates based on category understanding and practical evaluation criteria and solution approaches.",
            "source_preference_clause": "Prioritize seller-independent evidence, provided it is topically relevant.",
        }
        valid = {
            "search_objective_clause": "Arrange the candidates based on category understanding and practical evaluation criteria and solution approaches.",
            "source_preference_clause": "Prioritize seller-independent evidence, provided it is topically relevant.",
        }
        with tempfile.TemporaryDirectory() as directory:
            ranker = _StaticRanker(
                [
                    json.dumps({"candidates": [invalid]}),
                    json.dumps({"candidates": [valid]}),
                ]
            )
            generator = LocalLLMTwoAxisCandidateGenerator(
                ranker,
                model_name="test-generator",
                cache_directory=directory,
                temperature=0.0,
            )
            request = TwoAxisCandidateRequest(
                assigned_a1=0.5,
                assigned_a2=0.0,
                style_seed=0,
                generation_seed=10,
                number_candidates=1,
                generator_model="test-generator",
            )
            self.assertEqual(generator.generate(request)[0][0], valid["search_objective_clause"])
            self.assertEqual(ranker.call_count, 2)
            cache = next(
                path
                for path in Path(directory).glob("*.json")
                if not path.name.endswith(".failed.json")
            )
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.assertIn(
                "candidate-cardinality-exposed",
                payload["rejected_attempts"][0]["error"],
            )

    def test_comparative_a2_direction_uses_first_preference_object(self) -> None:
        controlled = semantic_contract_checks(
            "{QUERY} {CANDIDATES} {TOP_N} business software evaluator candidate identifiers only no explanation",
            search_term="absent query",
            business_actor="business software evaluator",
            objective_clause="Understand the category and develop practical evaluation criteria.",
            source_preference_clause="Prioritize vendor-controlled evidence over seller-independent evidence, conditional on equal relevance.",
            assigned_a1=0.5,
            assigned_a2=1.0,
        )
        independent = semantic_contract_checks(
            "{QUERY} {CANDIDATES} {TOP_N} business software evaluator candidate identifiers only no explanation",
            search_term="absent query",
            business_actor="business software evaluator",
            objective_clause="Understand the category and develop practical evaluation criteria.",
            source_preference_clause="Prefer seller-independent research over vendor-controlled content, conditional on equal relevance.",
            assigned_a1=0.5,
            assigned_a2=0.0,
        )
        self.assertFalse(any(reason.startswith("coordinate-mismatch") for reason in controlled))
        self.assertFalse(any(reason.startswith("coordinate-mismatch") for reason in independent))


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

    def test_judgment_diagnostics_match_calibration_and_separate_judges(self) -> None:
        candidates, requests, judgments, *_ = _pipeline()
        diagnostics = diagnose_pairwise_judgments(candidates, requests, judgments)
        self.assertTrue(diagnostics["all_endpoint_slices_ordered"])
        self.assertEqual(diagnostics["failing_endpoint_slices"], [])
        self.assertEqual(diagnostics["judge_ids"], ["judge-one", "judge-two"])
        self.assertGreater(diagnostics["cross_judge_agreement_rate"], 0.9)
        self.assertEqual(len(diagnostics["slices"]), 12)
        for item in diagnostics["slices"]:
            self.assertGreater(item["pooled_endpoint_fit"]["upper_minus_lower"], 0)
            self.assertEqual(len(item["per_judge"]), 2)
            self.assertEqual(
                item["pooled_direct_endpoint_evidence"][
                    "presentation_order_consistency_rate"
                ],
                1.0,
            )

    def test_reversed_a2_judgments_name_the_failed_slice(self) -> None:
        candidates, requests, judgments, *_ = _pipeline()
        request_by_id = {request.comparison_id: request for request in requests}
        reversed_judgments = tuple(
            PairwiseJudgment(
                comparison_id=judgment.comparison_id,
                judge_id=judgment.judge_id,
                winner_candidate_id=(
                    None
                    if judgment.is_tie
                    else (
                        request_by_id[judgment.comparison_id].right_candidate_id
                        if judgment.winner_candidate_id
                        == request_by_id[judgment.comparison_id].left_candidate_id
                        else request_by_id[judgment.comparison_id].left_candidate_id
                    )
                ),
                is_tie=judgment.is_tie,
            )
            if request_by_id[judgment.comparison_id].axis == "A2"
            else judgment
            for judgment in judgments
        )
        diagnostics = diagnose_pairwise_judgments(
            candidates, requests, reversed_judgments
        )
        self.assertFalse(diagnostics["all_endpoint_slices_ordered"])
        self.assertTrue(diagnostics["failing_endpoint_slices"])
        self.assertEqual(
            {item["axis"] for item in diagnostics["failing_endpoint_slices"]},
            {"A2"},
        )
        with self.assertRaisesRegex(
            ValueError,
            r"A2 pairwise judgments do not order endpoint anchors for "
            r"style_seed=3, fixed_coordinate=0: upper_minus_lower=-",
        ):
            calibrate_candidates(candidates, requests, reversed_judgments)

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

    def test_real_judge_extracts_json_after_thinking(self) -> None:
        candidates, *_ = _pipeline()
        left, right = candidates[0], candidates[1]
        request = PairwiseComparisonRequest(
            comparison_id="comparison-wrapped",
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
            ranker = _RecordingRanker(
                ['<think>compare only A1</think>\n```json\n{"winner":"right"}\n```']
            )
            judge = LocalLLMPairwiseJudge(
                ranker,
                judge_id="judge-one",
                model_name="test-judge",
                cache_directory=directory,
            )
            mapping = {left.candidate_id: left, right.candidate_id: right}
            self.assertEqual(judge.compare(request, mapping), right.candidate_id)
            self.assertEqual(ranker.chat_template_kwargs, {"enable_thinking": False})


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
