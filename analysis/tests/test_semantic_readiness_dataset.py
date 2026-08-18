"""Contracts for natural-text readiness acquisition and blinded labeling."""

from __future__ import annotations

from dataclasses import asdict
import json
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_embedding_map import (
    evaluate_readiness_embedding_map,
    fit_readiness_embedding_map,
)

from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    ABSTENTION_LABEL_RUBRIC_VERSION,
    WebRetrievalProbe,
    build_readiness_label_tasks,
    build_semantic_readiness_corpus,
    aggregate_readiness_consensus,
    load_web_retrieval_specification,
    merge_web_records,
    parse_readiness_judgment,
    parse_stackexchange_items,
    summarize_readiness_judge_agreement,
)


class SemanticReadinessDatasetTests(unittest.TestCase):
    def test_web_spec_covers_sampling_regions_and_heldout_sites(self) -> None:
        probes = load_web_retrieval_specification()
        self.assertEqual(len(probes), 48)
        self.assertEqual(
            {item.sampling_region for item in probes},
            {"information", "comparison", "selection", "action"},
        )
        development_sites = {item.site for item in probes if item.split == "development"}
        confirmation_sites = {
            item.site for item in probes if item.split == "confirmation"
        }
        self.assertFalse(development_sites & confirmation_sites)

    def test_stackexchange_parser_retains_attribution_and_merges_routes(self) -> None:
        payload = {
            "items": [
                {
                    "question_id": 42,
                    "title": "How should I choose between these two tools?",
                    "link": "https://example.stackexchange.com/questions/42/x",
                    "tags": ["tools"],
                    "creation_date": 100,
                    "score": 3,
                    "content_license": "CC BY-SA 3.0",
                    "owner": {
                        "display_name": "Research User",
                        "link": "https://example.stackexchange.com/users/1/x",
                    },
                }
            ]
        }
        first_probe = WebRetrievalProbe(
            "probe-a", "superuser", "choose", "selection", "development"
        )
        second_probe = WebRetrievalProbe(
            "probe-b", "superuser", "tools", "comparison", "development"
        )
        records = merge_web_records(
            (
                *parse_stackexchange_items(payload, first_probe),
                *parse_stackexchange_items(payload, second_probe),
            )
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].license, "CC BY-SA 3.0")
        self.assertEqual(records[0].author_name, "Research User")
        self.assertEqual(records[0].retrieval_probe_ids, ("probe-a", "probe-b"))

    def test_corpus_and_label_tasks_hide_retrieval_and_source_metadata(self) -> None:
        payload = {
            "items": [
                {
                    "question_id": 42,
                    "title": "How should I choose between these two tools?",
                    "link": "https://superuser.com/questions/42/x",
                    "owner": {"display_name": "User"},
                    "content_license": "CC BY-SA 4.0",
                }
            ]
        }
        probe = WebRetrievalProbe(
            "probe-a", "superuser", "choose", "selection", "development"
        )
        web = parse_stackexchange_items(payload, probe)
        surface = (
            {
                "source_id": "databricks-dolly-15k",
                "source_record_id": "dolly:1",
                "text": "Explain the major parts of a bicycle pump.",
                "corpus_split": "development",
                "surface_family_id": "family:1",
            },
        )
        corpus = build_semantic_readiness_corpus(surface, web)
        tasks, codebook = build_readiness_label_tasks(
            corpus,
            judge_slots=("judge-a", "judge-b"),
        )
        self.assertEqual(len(corpus), 2)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            {item.presentation_variant for item in tasks},
            {"forward-anchors", "reverse-anchors"},
        )
        for task in tasks:
            public = asdict(task)
            self.assertNotIn("source_name", public)
            self.assertNotIn("sampling_region", task.prompt)
            self.assertNotIn("superuser.com", task.prompt)
            self.assertIn("overall_readiness_0_100", task.prompt)
            self.assertFalse(
                codebook[task.task_id]["retrieval_metadata_visible_to_judge"]
            )

    def test_unlicensed_web_records_remain_outside_label_corpus(self) -> None:
        payload = {
            "items": [
                {
                    "question_id": 9,
                    "title": "Which material should be selected for this repair?",
                    "link": "https://diy.stackexchange.com/questions/9/x",
                    "owner": {"display_name": "User"},
                }
            ]
        }
        probe = WebRetrievalProbe(
            "probe-a", "diy", "selected", "selection", "confirmation"
        )
        web = parse_stackexchange_items(payload, probe)
        self.assertEqual(web[0].license, "unknown")
        self.assertEqual(build_semantic_readiness_corpus((), web), ())

    def test_abstention_v2_separates_rating_not_applicable_and_dont_know(self) -> None:
        corpus = build_semantic_readiness_corpus(
            (
                {
                    "source_id": "databricks-dolly-15k",
                    "source_record_id": "dolly:1",
                    "text": "Can you help me decide whether this unclear option is suitable?",
                    "corpus_split": "development",
                    "surface_family_id": "family:1",
                },
            ),
            (),
        )
        legacy, _ = build_readiness_label_tasks(corpus, judge_slots=("judge-a",))
        tasks, _ = build_readiness_label_tasks(
            corpus,
            judge_slots=("judge-a",),
            rubric_version=ABSTENTION_LABEL_RUBRIC_VERSION,
        )
        task = tasks[0]
        self.assertNotEqual(task.task_id, legacy[0].task_id)
        self.assertEqual(task.rubric_version, ABSTENTION_LABEL_RUBRIC_VERSION)
        self.assertIn('"dont_know"', task.prompt)
        self.assertIn("all five readiness scores and category to null", task.prompt)

        dont_know = {
            "answer_type": "dont_know",
            "overall_readiness_0_100": None,
            "information_seeking_1_7": None,
            "evaluation_1_7": None,
            "selection_commitment_1_7": None,
            "action_implementation_1_7": None,
            "category": None,
            "ambiguity_1_7": 7,
            "confidence_0_1": 0.2,
            "brief_reason": "The intended decision cannot be determined from the text.",
        }
        parsed = parse_readiness_judgment(task, json.dumps(dont_know))
        self.assertEqual(parsed.answer_type, "dont_know")
        self.assertIsNone(parsed.overall_readiness_0_100)

        not_applicable = dict(dont_know, answer_type="not_applicable")
        parsed_not_applicable = parse_readiness_judgment(
            task, json.dumps(not_applicable)
        )
        self.assertEqual(parsed_not_applicable.answer_type, "not_applicable")

        invalid = dict(dont_know, overall_readiness_0_100=50)
        with self.assertRaisesRegex(ValueError, "must be null"):
            parse_readiness_judgment(task, json.dumps(invalid))

        rating = {
            **dont_know,
            "answer_type": "rating",
            "overall_readiness_0_100": 50,
            "information_seeking_1_7": 3,
            "evaluation_1_7": 6,
            "selection_commitment_1_7": 3,
            "action_implementation_1_7": 2,
            "category": "comparison",
        }
        self.assertEqual(
            parse_readiness_judgment(task, json.dumps(rating)).answer_type,
            "rating",
        )

    def test_label_parser_and_consensus_preserve_judge_disagreement(self) -> None:
        corpus = build_semantic_readiness_corpus(
            (
                {
                    "source_id": "databricks-dolly-15k",
                    "source_record_id": "dolly:1",
                    "text": "Compare the available materials before choosing one.",
                    "corpus_split": "development",
                    "surface_family_id": "family:1",
                },
            ),
            (),
        )
        tasks, _ = build_readiness_label_tasks(
            corpus, judge_slots=("judge-a", "judge-b", "judge-c")
        )
        scores = (52, 60, 56)
        judgments = []
        for task, score in zip(tasks, scores):
            raw = json.dumps(
                {
                    "overall_readiness_0_100": score,
                    "information_seeking_1_7": 3,
                    "evaluation_1_7": 6,
                    "selection_commitment_1_7": 3,
                    "action_implementation_1_7": 2,
                    "category": "comparison",
                    "not_applicable": False,
                    "ambiguity_1_7": 2,
                    "confidence_0_1": 0.9,
                    "brief_reason": "The request compares options before selection.",
                }
            )
            judgments.append(parse_readiness_judgment(task, raw))
        consensus = aggregate_readiness_consensus(judgments)
        self.assertEqual(consensus[0].overall_readiness_0_100, 56.0)
        self.assertEqual(consensus[0].overall_median_absolute_deviation, 4.0)
        self.assertTrue(consensus[0].usable_for_axis)
        agreement = summarize_readiness_judge_agreement(judgments)
        self.assertEqual(agreement["complete_panel_item_count"], 1)
        self.assertEqual(len(agreement["pairwise"]), 3)
        self.assertEqual(agreement["mean_pairwise_category_exact_agreement"], 1.0)
        self.assertGreater(agreement["mean_complete_item_overall_variance"], 0.0)

        invalid = raw.replace('"confidence_0_1": 0.9', '"confidence_0_1": 2.0')
        with self.assertRaisesRegex(ValueError, "confidence"):
            parse_readiness_judgment(tasks[0], invalid)

    def test_supervised_map_recovers_synthetic_direction_and_level_planes(self) -> None:
        from analysis.interpretability.pipeline.semantic_readiness_dataset import (
            ReadinessConsensus,
            SemanticReadinessItem,
        )

        items = []
        labels = []
        embeddings = []
        for index in range(40):
            readiness = index / 39
            split = "development" if index < 30 else "confirmation"
            item_id = f"item:{index}"
            items.append(
                SemanticReadinessItem(
                    item_id=item_id,
                    source_kind="synthetic-test",
                    source_name="source-a" if index % 2 else "source-b",
                    source_record_id=str(index),
                    text=f"Synthetic item {index}",
                    text_sha256=str(index),
                    split=split,
                    group_id=f"group:{index % 4}",
                    source_url=None,
                    author_name=None,
                    author_url=None,
                    license="test-only",
                )
            )
            labels.append(
                ReadinessConsensus(
                    item_id=item_id,
                    judge_count=3,
                    overall_readiness_0_100=readiness * 100,
                    information_seeking_1_7=7 - 6 * readiness,
                    evaluation_1_7=1 + 4 * readiness,
                    selection_commitment_1_7=1 + 5 * readiness,
                    action_implementation_1_7=1 + 6 * readiness,
                    not_applicable_vote_fraction=0.0,
                    ambiguity_mean=1.0,
                    confidence_mean=0.95,
                    overall_median_absolute_deviation=1.0,
                    usable_for_axis=True,
                )
            )
            embeddings.append([readiness - 0.5, 4.0, 0.01 * (index % 3), 0.0])
        matrix = np.asarray(embeddings)
        fitted = fit_readiness_embedding_map(
            items[:30],
            labels,
            matrix[:30],
            embedding_model="fake-llm2vec",
            ridge_penalty=0.01,
        )
        coordinates, diagnostics = evaluate_readiness_embedding_map(
            fitted,
            items[30:],
            labels,
            matrix[30:],
        )
        self.assertEqual(len(fitted.ordinal_plane_offsets), 4)
        self.assertEqual(len(fitted.supervised_subspace_axes), 2)
        self.assertEqual(len(fitted.ordinal_thresholds_by_rubric), 4)
        self.assertGreater(fitted.ridge_ordinal_cosine_similarity, 0.95)
        self.assertEqual(fitted.pca_method, "deterministic-randomized-svd-v1")
        self.assertEqual(fitted.pca_random_seed, 20260817)
        self.assertEqual(len(fitted.pca_axes), 4)
        self.assertEqual(len(fitted.pca_explained_variance_ratio), 4)
        self.assertTrue(
            all(value >= 0 for value in fitted.pca_explained_variance_ratio)
        )
        self.assertGreater(fitted.rubric_first_component_share, 0.95)
        self.assertGreater(diagnostics.spearman, 0.99)
        self.assertGreater(diagnostics.ordinal_spearman, 0.99)
        self.assertGreater(diagnostics.ordinal_pairwise_order_accuracy, 0.9)
        self.assertLess(
            np.mean([item.absolute_error for item in coordinates]),
            0.08,
        )
        from analysis.scripts.fit_semantic_readiness_map import _evaluate_by_source

        by_source = _evaluate_by_source(
            fitted,
            tuple(items[30:]),
            labels,
            matrix[30:],
        )
        self.assertEqual(set(by_source), {"source-a", "source-b"})
        self.assertTrue(all(row["status"] == "ok" for row in by_source.values()))


if __name__ == "__main__":
    unittest.main()
