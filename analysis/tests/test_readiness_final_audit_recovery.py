"""Recovery contracts for the four-GPU selected-prompt final audit."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.run_readiness_final_audit_4gpu import (
    audit_relaxed_selected_prompts,
    partition_candidate_rows,
    projection_shard_state,
)


def _candidate(candidate_id: str, question: str, keyword: str = "alpha software"):
    return {
        "candidate_id": candidate_id,
        "keyword_id": "keyword:alpha",
        "keyword": keyword,
        "question": question,
        "question_sha256": __import__("hashlib").sha256(question.encode()).hexdigest(),
    }


def _review(candidate_id: str):
    return {
        "candidate_id": candidate_id,
        "topic_relevant": True,
        "search_intent": True,
        "web_answerable": True,
        "natural_language": True,
        "relevance_score_1_5": 5,
        "accepted": True,
    }


class ReadinessFinalAuditRecoveryTests(unittest.TestCase):
    def test_relaxed_audit_accepts_metadata_bound_prompt_without_literal_keyword(self) -> None:
        question = "Compare the strongest options and their practical trade-offs."
        candidate = _candidate("candidate:one", question)
        selected = {
            **candidate,
            "target_id": "target:one",
            "target_index": 0,
            "both_views_within_tolerance": True,
        }

        passed, summary = audit_relaxed_selected_prompts(
            selected_rows=[selected],
            candidate_rows=[candidate],
            validation_rows=[_review("candidate:one")],
        )

        self.assertEqual([row["candidate_id"] for row in passed], ["candidate:one"])
        self.assertEqual(summary["relaxed_contract_pass_count"], 1)
        self.assertEqual(summary["gate_failures"], {})

    def test_relaxed_audit_rejects_conflicting_explicit_search_query(self) -> None:
        question = "Compare the strongest options and their practical trade-offs."
        candidate = {
            **_candidate("candidate:one", question),
            "search_query": "unrelated topic",
        }
        selected = {
            **candidate,
            "target_id": "target:one",
            "target_index": 0,
            "both_views_within_tolerance": True,
        }

        passed, summary = audit_relaxed_selected_prompts(
            selected_rows=[selected],
            candidate_rows=[candidate],
            validation_rows=[_review("candidate:one")],
        )

        self.assertEqual(passed, ())
        self.assertEqual(summary["gate_failures"], {"search_query_keyword_binding": 1})

    def test_partition_is_balanced_deterministic_and_exhaustive(self) -> None:
        rows = [_candidate(f"candidate:{index}", f"Question {index}") for index in range(17)]

        first = partition_candidate_rows(rows, shard_count=8)
        second = partition_candidate_rows(rows, shard_count=8)

        self.assertEqual(first, second)
        self.assertLessEqual(max(map(len, first)) - min(map(len, first)), 1)
        observed = [row["candidate_id"] for shard in first for row in shard]
        self.assertEqual(sorted(observed), sorted(row["candidate_id"] for row in rows))

    def test_incomplete_directory_is_not_reused_as_a_finished_shard(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "candidates.jsonl"
            input_path.write_text('{"candidate_id":"one"}\n{"candidate_id":"two"}\n')
            output = root / "projection"
            output.mkdir()

            self.assertEqual(projection_shard_state(input_path, output), "incomplete")

            (output / "question_projections.jsonl").write_text(
                json.dumps({"candidate_id": "one"})
                + "\n"
                + json.dumps({"candidate_id": "two"})
                + "\n"
            )
            (output / "projection_manifest.json").write_text("{}\n")
            self.assertEqual(projection_shard_state(input_path, output), "complete")

    def test_equal_row_count_with_different_candidate_ids_is_incomplete(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "candidates.jsonl"
            input_path.write_text('{"candidate_id":"one"}\n')
            output = root / "projection"
            output.mkdir()
            (output / "question_projections.jsonl").write_text(
                '{"candidate_id":"different"}\n'
            )
            (output / "projection_manifest.json").write_text("{}\n")

            self.assertEqual(projection_shard_state(input_path, output), "incomplete")


if __name__ == "__main__":
    unittest.main()
