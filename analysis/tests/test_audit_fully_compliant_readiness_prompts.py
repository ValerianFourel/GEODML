from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.interpretability.pipeline.readiness_prompt_population import (
    audit_question_diversity,
)
from analysis.scripts.audit_fully_compliant_readiness_prompts import (
    audit_fully_compliant_prompts,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = [
        {
            "candidate_id": "candidate:one",
            "keyword_id": "keyword:one",
            "keyword": "alpha",
            "target_id": "target:one",
            "generator_id": "generator:one",
            "generator_model": "model:one",
            "question": "How can alpha explain this?",
            "both_views_within_tolerance": True,
            "target_distance": 0.010,
            "reference_target_distance": 0.011,
            "candidate_aligned_target_distance": 0.012,
        },
        {
            "candidate_id": "candidate:two",
            "keyword_id": "keyword:two",
            "keyword": "beta",
            "target_id": "target:two",
            "generator_id": "generator:two",
            "generator_model": "model:two",
            "question": "Why does beta behave differently?",
            "both_views_within_tolerance": True,
            "target_distance": 0.013,
            "reference_target_distance": 0.014,
            "candidate_aligned_target_distance": 0.015,
        },
    ]
    candidates = [
        {
            **row,
            "question_sha256": hashlib.sha256(
                str(row["question"]).encode("utf-8")
            ).hexdigest(),
        }
        for row in selected
    ]
    candidates.append(
        {
            "candidate_id": "candidate:three",
            "keyword_id": "keyword:three",
            "keyword": "gamma",
            "target_id": "target:three",
            "generator_id": "generator:three",
            "generator_model": "model:three",
            "question": "What is gamma?",
            "question_sha256": hashlib.sha256(b"What is gamma?").hexdigest(),
        }
    )
    validation = [
        {
            "candidate_id": row["candidate_id"],
            "accepted": index < 2,
            "exact_keyword_present": True,
            "single_question": True,
            "topic_relevant": index < 2,
            "search_intent": index < 2,
            "web_answerable": index < 2,
            "standalone": index < 2,
            "natural_language": index < 2,
            "relevance_score_1_5": 5 if index < 2 else 1,
        }
        for index, row in enumerate(candidates)
    ]

    selection = root / "strict-selection"
    _write_jsonl(selection / "spatially_selected_questions.jsonl", selected)
    _write_json(
        selection / "run_manifest.json",
        {
            "candidate_count": 3,
            "selected_count": 2,
            "coordinate_acceptance_contract": {
                "enabled": True,
                "distance_tolerance": 0.017,
            },
            "surface_acceptance_contract": {"enabled": True},
        },
    )
    _write_json(
        selection / "spatial_coverage_diagnostics.json",
        {
            "candidate_count": 3,
            "selected_count": 2,
            "verified_selected_count": 2,
            "require_both_views_within_tolerance": True,
            "require_delexicalized_template_uniqueness": True,
            "selected_delexicalized_templates_are_unique": True,
        },
    )
    _write_json(
        root / "verified_round_summary.json",
        {
            "candidate_count": 3,
            "independently_accepted_count": 2,
            "selected_count": 2,
            "strict_dual_view_contract_enabled": True,
            "delexicalized_template_uniqueness_enabled": True,
            "selected_diversity_gate_passed": True,
            "spacing_gate_passed": False,
            "verified_population_passed": False,
        },
    )
    candidates_path = root / "merged/candidates.jsonl"
    _write_jsonl(candidates_path, candidates)
    _write_json(
        candidates_path.with_suffix(".jsonl.manifest.json"),
        {"candidate_count": 3},
    )
    validation_path = root / "merged/validation.jsonl"
    _write_jsonl(validation_path, validation)
    _write_json(
        validation_path.with_suffix(".jsonl.manifest.json"),
        {"candidate_count": 3, "reviewed_count": 3, "accepted_count": 2},
    )
    diversity = audit_question_diversity(
        selected,
        maximum_template_fraction=0.5,
        maximum_opening_frame_fraction=0.5,
    )
    assert diversity["all_checks_passed"] is True
    _write_json(root / "selected-diversity/question_diversity_audit.json", diversity)
    _write_json(
        root / "selected-diversity/run_manifest.json",
        {"row_count": 2, "all_checks_passed": True},
    )
    return selected, validation


def _convert_to_verified_round(root: Path) -> None:
    candidate_source = root / "source/candidates.jsonl"
    candidate_source.parent.mkdir(parents=True, exist_ok=True)
    (root / "merged/candidates.jsonl").replace(candidate_source)
    (root / "merged/candidates.jsonl.manifest.json").replace(
        candidate_source.with_suffix(".jsonl.manifest.json")
    )
    (root / "merged/validation.jsonl").replace(root / "validation.jsonl")
    (root / "merged/validation.jsonl.manifest.json").replace(
        root / "validation.jsonl.manifest.json"
    )
    (root / "candidate-files.txt").write_text(
        str(candidate_source.resolve()) + "\n",
        encoding="utf-8",
    )


class FullyCompliantPromptAuditTests(unittest.TestCase):
    def test_independently_counts_every_fully_compliant_prompt(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            _fixture(root)

            report = audit_fully_compliant_prompts(root)

            self.assertTrue(report["audit_passed"])
            self.assertEqual(report["claimed_selected_count"], 2)
            self.assertEqual(report["fully_compliant_prompt_count"], 2)
            self.assertEqual(report["failed_prompt_count"], 0)
            self.assertEqual(report["ready_to_export_count"], 2)
            self.assertFalse(report["complete_30330_population_passed"])

    def test_audits_an_immutable_verified_pipeline_round(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "round-07"
            _fixture(root)
            _convert_to_verified_round(root)

            report = audit_fully_compliant_prompts(root)

            self.assertTrue(report["audit_passed"])
            self.assertEqual(report["artifact_kind"], "verified-round")
            self.assertEqual(report["fully_compliant_prompt_count"], 2)
            self.assertEqual(report["ready_to_export_count"], 2)

    def test_audits_relaxed_search_triggers_without_keyword_or_question_form(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "round-00"
            selected, validation = _fixture(root)
            questions = (
                "Find causes and practical fixes for this issue",
                "Compare the evidence and choose a workable approach",
            )
            candidates_path = root / "merged/candidates.jsonl"
            candidates = [
                json.loads(line)
                for line in candidates_path.read_text(encoding="utf-8").splitlines()
            ]
            for index, question in enumerate(questions):
                selected[index]["question"] = question
                candidates[index]["question"] = question
                candidates[index]["question_sha256"] = hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest()
                validation[index]["exact_keyword_present"] = False
                validation[index]["single_question"] = False
                validation[index]["standalone"] = False
            _write_jsonl(
                root / "strict-selection/spatially_selected_questions.jsonl",
                selected,
            )
            _write_jsonl(candidates_path, candidates)
            _write_jsonl(root / "merged/validation.jsonl", validation)

            for path in (
                root / "strict-selection/run_manifest.json",
                root / "strict-selection/spatial_coverage_diagnostics.json",
                root / "verified_round_summary.json",
            ):
                row = json.loads(path.read_text(encoding="utf-8"))
                row["text_contract"] = "search-trigger-v2"
                row["acceptance_contract_version"] = "search-trigger-v2"
                if path.name == "run_manifest.json":
                    row["coordinate_acceptance_contract"][
                        "distance_tolerance"
                    ] = 0.035
                _write_json(path, row)
            validation_manifest_path = (
                root / "merged/validation.jsonl.manifest.json"
            )
            validation_manifest = json.loads(
                validation_manifest_path.read_text(encoding="utf-8")
            )
            validation_manifest["acceptance_contract_version"] = (
                "search-trigger-v2"
            )
            _write_json(validation_manifest_path, validation_manifest)

            diversity = audit_question_diversity(
                selected,
                maximum_template_fraction=0.5,
                maximum_opening_frame_fraction=0.5,
                allow_missing_keyword_for_template=True,
            )
            self.assertTrue(diversity["all_checks_passed"])
            _write_json(
                root / "selected-diversity/question_diversity_audit.json",
                diversity,
            )

            report = audit_fully_compliant_prompts(root)

            self.assertTrue(report["audit_passed"])
            self.assertEqual(report["text_contract"], "search-trigger-v2")
            self.assertEqual(report["fully_compliant_prompt_count"], 2)
            self.assertEqual(report["ready_to_export_count"], 2)

    def test_allows_spatial_selection_to_reassign_a_candidate_target(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            _fixture(root)
            candidates_path = root / "merged/candidates.jsonl"
            candidates = [
                json.loads(line)
                for line in candidates_path.read_text(encoding="utf-8").splitlines()
            ]
            candidates[0]["target_id"] = "original-generation-target"
            _write_jsonl(candidates_path, candidates)

            report = audit_fully_compliant_prompts(root)

            self.assertTrue(report["audit_passed"])
            self.assertEqual(report["fully_compliant_prompt_count"], 2)
            self.assertEqual(report["ready_to_export_count"], 2)

    def test_rejects_selected_text_that_does_not_match_immutable_candidate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            selected, _ = _fixture(root)
            selected[0]["question"] = "How should alpha explain this?"
            _write_jsonl(
                root / "strict-selection/spatially_selected_questions.jsonl",
                selected,
            )

            report = audit_fully_compliant_prompts(root)

            self.assertFalse(report["audit_passed"])
            self.assertEqual(report["fully_compliant_prompt_count"], 1)
            self.assertEqual(report["ready_to_export_count"], 0)
            self.assertEqual(
                report["failed_prompt_checks"][
                    "selected_content_matches_merged_candidate"
                ],
                1,
            )

    def test_rechecks_all_fields_behind_independent_acceptance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            _, validation = _fixture(root)
            validation[1]["web_answerable"] = False
            _write_jsonl(root / "merged/validation.jsonl", validation)

            report = audit_fully_compliant_prompts(root)

            self.assertFalse(report["audit_passed"])
            self.assertEqual(report["fully_compliant_prompt_count"], 1)
            self.assertEqual(
                report["failed_prompt_checks"]["independent_acceptance_contract"],
                1,
            )

    def test_global_manifest_failure_prevents_ready_export_count(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "final"
            _fixture(root)
            summary_path = root / "verified_round_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["selected_count"] = 99
            _write_json(summary_path, summary)

            report = audit_fully_compliant_prompts(root)

            self.assertFalse(report["audit_passed"])
            self.assertEqual(report["fully_compliant_prompt_count"], 2)
            self.assertEqual(report["ready_to_export_count"], 0)
            self.assertIn(
                "summary_selected_count_matches",
                report["failed_global_checks"],
            )


if __name__ == "__main__":
    unittest.main()
