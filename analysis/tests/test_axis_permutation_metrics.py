"""CPU contracts for prompt-axis predictive fits and ranking agreement."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from copy import deepcopy

from analysis.interpretability.pipeline.axis_permutation_metrics import (
    fit_axis_rankings,
    ranking_agreement,
)


def _rows(count=100):
    rows = []
    for i in range(count):
        axis = (i % 11) / 10
        rows.append(
            {
                "task_key": f"task-{i:04d}",
                "prompt_id": f"prompt-{i:04d}",
                "question_sha256": hashlib.sha256(f"question-{i}".encode()).hexdigest(),
                "candidate_ids": ["a", "b", "c"],
                "axes": {"B": axis},
                "ranking": ["a", "b", "c"] if axis >= 0.5 else ["c", "b", "a"],
            }
        )
    return rows


class RankingAgreementTests(unittest.TestCase):
    def test_equality_and_reversal(self):
        equal = ranking_agreement(["a", "b", "c"], ["a", "b", "c"])
        reverse = ranking_agreement(["a", "b", "c"], ["c", "b", "a"])
        self.assertTrue(equal["exact_match"])
        self.assertEqual(equal["kendall_tau_common"], 1)
        self.assertEqual(equal["concordant_pairs"], 3)
        self.assertFalse(reverse["exact_match"])
        self.assertFalse(reverse["top1_match"])
        self.assertEqual(reverse["kendall_tau_common"], -1)
        self.assertEqual(reverse["discordant_pairs"], 3)
        self.assertEqual(reverse["topk_overlap"], 1)

    def test_empty_and_disjoint_are_not_imputed(self):
        for left, right in (([], []), (["a"], []), ([], ["a"])):
            report = ranking_agreement(left, right)
            self.assertIsNone(report["exact_match"])
            self.assertIsNone(report["top1_match"])
            self.assertIsNone(report["kendall_tau_common"])
            self.assertIsNone(report["topk_overlap"])
            self.assertEqual(report["topk_k"], 0)
        disjoint = ranking_agreement(["a", "b"], ["c", "d"])
        self.assertEqual(disjoint["topk_overlap"], 0)
        self.assertIsNone(disjoint["kendall_tau_common"])
        self.assertFalse(disjoint["exact_match"])

    def test_partial_topk_uses_common_length_and_shared_pairs_only(self):
        report = ranking_agreement(["a", "b", "c", "d"], ["b", "e"])
        self.assertEqual(report["topk_k"], 2)
        self.assertEqual(report["topk_intersection_count"], 1)
        self.assertEqual(report["topk_overlap"], 0.5)
        self.assertEqual(report["common_count"], 1)
        self.assertIsNone(report["kendall_tau_common"])
        shared = ranking_agreement(["x", "a", "b"], ["b", "a", "z"])
        self.assertEqual(shared["kendall_tau_common"], -1)
        self.assertEqual(shared["discordant_pairs"], 1)

    def test_invalid_ids_fail(self):
        for value in (["a", "a"], [None], [1], [""], "ab"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ranking_agreement(value, ["a"])


class AxisRankingFitTests(unittest.TestCase):
    def test_axis_signal_improves_on_baseline_on_heldout_questions(self):
        report = fit_axis_rankings(_rows(), axis_field="B")
        self.assertEqual(report["status"], "fit")
        axis, baseline = report["axis_model"], report["baseline_model"]
        self.assertTrue(axis["convergence"]["success"])
        self.assertTrue(baseline["convergence"]["success"])
        self.assertGreater(axis["heldout"]["pairwise_accuracy"], 0.9)
        self.assertLess(
            axis["heldout"]["pairwise_log_loss"],
            baseline["heldout"]["pairwise_log_loss"],
        )
        self.assertGreater(report["heldout_improvement"]["brier_reduction"], 0)
        coefficients = {row["candidate_id"]: row for row in axis["coefficients"]}
        self.assertGreater(coefficients["a"]["axis_slope"], 0)
        self.assertLess(coefficients["c"]["axis_slope"], 0)
        self.assertEqual(report["support"]["all"]["min"], 0)
        self.assertEqual(report["support"]["all"]["max"], 1)
        json.dumps(report, allow_nan=False)

    def test_deterministic_row_and_candidate_order(self):
        rows = _rows()
        expected = fit_axis_rankings(rows, axis_field="B")
        random.Random(45).shuffle(rows)
        for row in rows:
            row["candidate_ids"].reverse()
        self.assertEqual(fit_axis_rankings(rows, axis_field="B"), expected)

    def test_heldout_axes_do_not_affect_training_standardization_or_coefficients(self):
        rows = _rows()
        original = fit_axis_rankings(rows, axis_field="B")
        for row in rows:
            if original["split_assignments"][row["task_key"]] == "test":
                row["axes"]["B"] += 100
        changed = fit_axis_rankings(rows, axis_field="B")
        self.assertEqual(changed["standardization"], original["standardization"])
        self.assertEqual(
            changed["axis_model"]["coefficients"],
            original["axis_model"]["coefficients"],
        )
        self.assertEqual(
            changed["axis_model"]["train"], original["axis_model"]["train"]
        )

    def test_coefficients_are_on_original_axis_units(self):
        rows = _rows()
        original = fit_axis_rankings(rows, axis_field="B")
        for row in rows:
            row["axes"]["B"] = 40 * row["axes"]["B"] + 7
        changed = fit_axis_rankings(rows, axis_field="B")
        for left, right in zip(
            original["axis_model"]["coefficients"],
            changed["axis_model"]["coefficients"],
        ):
            self.assertAlmostEqual(left["axis_slope"] / 40, right["axis_slope"])
            self.assertAlmostEqual(
                left["intercept"] - 7 * right["axis_slope"], right["intercept"]
            )
        self.assertAlmostEqual(
            original["axis_model"]["heldout"]["pairwise_log_loss"],
            changed["axis_model"]["heldout"]["pairwise_log_loss"],
        )

    def test_growth_and_duplicate_question_ids_cannot_change_split(self):
        rows = _rows()
        before = fit_axis_rankings(rows[:40], axis_field="B")
        alias = deepcopy(rows[0])
        alias.update(task_key="alternate-task", prompt_id="alternate-prompt-id")
        after = fit_axis_rankings(rows + [alias], axis_field="B")
        for task_key, split in before["split_assignments"].items():
            self.assertEqual(after["split_assignments"][task_key], split)
        self.assertEqual(
            after["split_assignments"]["alternate-task"],
            after["split_assignments"][rows[0]["task_key"]],
        )
        self.assertEqual(after["counts"]["questions"], len(rows))
        split_hashes = {
            split: {
                row["question_sha256"]
                for row in after["predictions"]
                if row["split"] == split
            }
            for split in ("train", "test")
        }
        self.assertFalse(split_hashes["train"] & split_hashes["test"])

    def test_insufficient_data_never_resplits(self):
        report = fit_axis_rankings(_rows(2), axis_field="B")
        self.assertEqual(report["status"], "insufficient_data")
        self.assertIsNone(report["axis_model"])
        aliases = []
        for i in range(25):
            row = deepcopy(_rows(1)[0])
            row["task_key"] = f"duplicate-content-{i}"
            aliases.append(row)
        report = fit_axis_rankings(aliases, axis_field="B")
        self.assertEqual(report["counts"]["questions"], 1)
        self.assertEqual(report["status"], "insufficient_data")

    def test_constant_and_missing_axes_are_explicit(self):
        rows = _rows()
        for row in rows:
            row["axes"]["B"] = 0.25
        report = fit_axis_rankings(rows, axis_field="B")
        self.assertEqual(report["status"], "constant_axis")
        rows[0]["axes"] = {}
        rows[1]["ranking"] = []
        report = fit_axis_rankings(rows, axis_field="B")
        self.assertEqual(report["counts"]["missing_axis"], 1)
        self.assertEqual(report["counts"]["unrankable"], 1)
        self.assertEqual(report["counts"]["eligible"], 98)
        self.assertEqual(
            fit_axis_rankings([], axis_field="B")["status"], "insufficient_data"
        )

    def test_partial_rankings_add_no_labels_for_omitted_candidates(self):
        rows = _rows()
        for row in rows:
            row["ranking"] = [
                candidate for candidate in row["ranking"] if candidate != "b"
            ]
        report = fit_axis_rankings(rows, axis_field="B")
        self.assertEqual(report["training_candidate_pair_counts"]["b"], 0)
        self.assertEqual(
            report["axis_model"]["train"]["observed_pair_count"],
            report["counts"]["train"],
        )
        coefficients = {
            row["candidate_id"]: row for row in report["axis_model"]["coefficients"]
        }
        self.assertEqual(coefficients["b"]["axis_slope"], 0)
        for row in report["predictions"]:
            self.assertEqual(len(row["axis_model"]["predicted_ranking"]), 2)
            self.assertEqual(len(row["axis_model"]["predicted_full_ranking"]), 3)

    def test_invalid_axes_and_pool_contracts_fail(self):
        mutations = [
            ("axes", {"B": True}),
            ("axes", {"B": float("nan")}),
            ("axes", {"B": float("inf")}),
            ("axes", {"B": "0.5"}),
            ("axes", {"B": None}),
            ("candidate_ids", ["a", "a"]),
            ("candidate_ids", ["a", "b"]),
            ("ranking", ["a", "unknown"]),
            ("ranking", ["a", "a"]),
            ("question_sha256", "not-a-hash"),
        ]
        for field, value in mutations:
            rows = _rows()
            rows[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fit_axis_rankings(rows, axis_field="B")
        with self.assertRaisesRegex(ValueError, "duplicate task_key"):
            fit_axis_rankings(_rows(1) * 2, axis_field="B")

    def test_invalid_fit_settings_fail(self):
        for setting, value in (
            ("ridge", 0),
            ("ridge", float("nan")),
            ("ridge", True),
            ("holdout_modulus", 1),
            ("min_train", 0),
            ("seed", True),
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                fit_axis_rankings(_rows(), axis_field="B", **{setting: value})


if __name__ == "__main__":
    unittest.main()
