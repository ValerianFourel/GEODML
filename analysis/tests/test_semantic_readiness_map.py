"""Contracts for compiling readiness labels with explicit missingness."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    build_readiness_label_tasks,
    build_semantic_readiness_corpus,
)
from analysis.scripts.fit_semantic_readiness_map import _compile_labels


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class SemanticReadinessMapTests(unittest.TestCase):
    def test_compile_labels_requires_exactly_declared_missing_tasks(self) -> None:
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
            corpus,
            judge_slots=("judge-a", "judge-b", "judge-c"),
        )
        raw_response = json.dumps(
            {
                "overall_readiness_0_100": 56,
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
        responses = [
            {"task_id": task.task_id, "raw_response": raw_response}
            for task in tasks[:2]
        ]
        missing_task_id = tasks[2].task_id

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / "tasks.jsonl"
            response_path = root / "responses.jsonl"
            _write_jsonl(task_path, map(asdict, tasks))
            _write_jsonl(response_path, responses)

            strict_output = root / "strict"
            strict_output.mkdir()
            strict_args = SimpleNamespace(
                tasks=str(task_path),
                responses=[str(response_path)],
                allow_missing_task_id=[],
            )
            with self.assertRaisesRegex(SystemExit, "missing=1"):
                _compile_labels(strict_args, strict_output)

            output = root / "allowed"
            output.mkdir()
            allowed_args = SimpleNamespace(
                tasks=str(task_path),
                responses=[str(response_path)],
                allow_missing_task_id=[missing_task_id],
            )
            _compile_labels(allowed_args, output)

            diagnostics = json.loads(
                (output / "label_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["task_count"], 3)
            self.assertEqual(diagnostics["judgment_count"], 2)
            self.assertEqual(diagnostics["missing_response_count"], 1)
            self.assertEqual(
                diagnostics["missing_response_task_ids"],
                [missing_task_id],
            )
            self.assertEqual(diagnostics["consensus_judge_count_counts"], {"2": 1})


if __name__ == "__main__":
    unittest.main()
