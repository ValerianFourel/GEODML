"""Contracts for provider-neutral semantic-readiness judge batches."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    ReadinessLabelTask,
)
from analysis.scripts.run_semantic_readiness_judge_batch import (
    BACKEND_NAME,
    _export_batch,
    _import_batch,
    _sha256_file,
)


def _valid_judgment(overall: int) -> str:
    return json.dumps(
        {
            "overall_readiness_0_100": overall,
            "information_seeking_1_7": 2,
            "evaluation_1_7": 4,
            "selection_commitment_1_7": 5,
            "action_implementation_1_7": 3,
            "category": "selection",
            "not_applicable": False,
            "ambiguity_1_7": 2,
            "confidence_0_1": 0.9,
            "brief_reason": "The request commits to selecting a product.",
        }
    )


def _provider_result(task_id: str, content: str) -> dict[str, object]:
    return {
        "id": f"batch-request-{task_id}",
        "custom_id": task_id,
        "response": {
            "status_code": 200,
            "request_id": f"request-{task_id}",
            "body": {
                "id": f"completion-{task_id}",
                "object": "chat.completion",
                "created": 1786924800,
                "model": "frontier-judge-2026-08-01",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 50,
                    "total_tokens": 250,
                },
            },
        },
        "error": None,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class SemanticReadinessJudgeBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tasks = (
            ReadinessLabelTask(
                task_id="readiness-label:task-a",
                item_id="item-a",
                judge_slot="primary-frontier",
                presentation_variant="forward-anchors",
                rubric_version="test-rubric",
                prompt="FROZEN RUBRIC\nPrompt A",
            ),
            ReadinessLabelTask(
                task_id="readiness-label:task-b",
                item_id="item-b",
                judge_slot="primary-frontier",
                presentation_variant="forward-anchors",
                rubric_version="test-rubric",
                prompt="FROZEN RUBRIC\nPrompt B",
            ),
        )
        self.task_path = self.root / "tasks.jsonl"
        _write_jsonl(self.task_path, [asdict(task) for task in self.tasks])
        self.judge_output = self.root / "judge-output"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _export(
        self,
        name: str,
        *,
        request_options: str | None = None,
        model_family: str = "independent-family-a",
    ) -> Path:
        export_directory = self.root / name
        _export_batch(
            SimpleNamespace(
                tasks=str(self.task_path),
                tasks_sha256=_sha256_file(self.task_path),
                expected_tasks=2,
                judge_slot="primary-frontier",
                provider="example-provider",
                model="frontier-judge",
                model_family=model_family,
                model_revision="frontier-judge-2026-08-01",
                expected_provider_model="frontier-judge-2026-08-01",
                output_dir=str(export_directory),
                judge_output_dir=str(self.judge_output),
                batch_endpoint="/v1/chat/completions",
                max_new_tokens=300,
                maximum_attempts=5,
                request_options=request_options,
            )
        )
        return export_directory

    def _import(self, export_directory: Path, batch_output: Path, batch_id: str) -> None:
        _import_batch(
            SimpleNamespace(
                tasks=str(self.task_path),
                export_manifest=str(export_directory / "batch_manifest.json"),
                batch_requests=None,
                batch_output=str(batch_output),
                provider_batch_id=batch_id,
                output_dir=str(self.judge_output),
            )
        )

    def test_invalid_response_is_preserved_and_retried_resumably(self) -> None:
        first_export = self._export("batch-1")
        requests = [
            json.loads(line)
            for line in (first_export / "batch_requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            [row["custom_id"] for row in requests],
            [task.task_id for task in self.tasks],
        )
        self.assertTrue(all(row["body"]["temperature"] == 0 for row in requests))

        first_output = self.root / "batch-1-output.jsonl"
        _write_jsonl(
            first_output,
            [
                _provider_result(self.tasks[0].task_id, _valid_judgment(72)),
                _provider_result(self.tasks[1].task_id, "{}"),
            ],
        )
        self._import(first_export, first_output, "provider-batch-1")
        self._import(first_export, first_output, "provider-batch-1")
        with self.assertRaisesRegex(SystemExit, "batch-import provenance"):
            self._import(first_export, first_output, "changed-provider-batch-id")

        run_manifest = json.loads(
            (self.judge_output / "run_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_manifest["completed_count"], 1)
        self.assertEqual(run_manifest["failed_count"], 1)
        self.assertEqual(run_manifest["missing_count"], 0)
        self.assertEqual(run_manifest["provider"], "example-provider")

        first_cache = json.loads(
            (
                self.judge_output
                / "task_cache"
                / "readiness-label_task-a.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(first_cache["backend"], BACKEND_NAME)
        self.assertEqual(first_cache["provider_batch_id"], "provider-batch-1")
        self.assertEqual(first_cache["provider_usage"]["total_tokens"], 250)
        self.assertEqual(
            len(list((self.judge_output / "batch_imports").glob("*.jsonl"))),
            1,
        )

        failed_cache = json.loads(
            (
                self.judge_output
                / "task_cache"
                / "readiness-label_task-b.failed.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(failed_cache["attempts"]), 1)
        self.assertEqual(failed_cache["attempts"][0]["raw"], "{}")

        second_export = self._export("batch-2")
        retry_requests = [
            json.loads(line)
            for line in (second_export / "batch_requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(retry_requests), 1)
        self.assertEqual(retry_requests[0]["custom_id"], self.tasks[1].task_id)
        retry_prompt = retry_requests[0]["body"]["messages"][0]["content"]
        self.assertIn("<previous_invalid_response>", retry_prompt)
        self.assertIn("{}", retry_prompt)

        second_output = self.root / "batch-2-output.jsonl"
        _write_jsonl(
            second_output,
            [_provider_result(self.tasks[1].task_id, _valid_judgment(84))],
        )
        self._import(second_export, second_output, "provider-batch-2")

        responses = [
            json.loads(line)
            for line in (self.judge_output / "judge_responses.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(responses), 2)
        second_cache = next(
            row for row in responses if row["task_id"] == self.tasks[1].task_id
        )
        self.assertEqual(second_cache["provider_batch_id"], "provider-batch-2")
        self.assertEqual(len(second_cache["rejected_attempts"]), 1)

    def test_request_options_cannot_override_frozen_generation_fields(self) -> None:
        options = self.root / "request-options.json"
        options.write_text('{"temperature": 0.7}\n', encoding="utf-8")

        with self.assertRaisesRegex(SystemExit, "reserved keys"):
            self._export("invalid-options", request_options=str(options))

    def test_export_freezes_judge_identity_before_provider_submission(self) -> None:
        self._export("frozen-panel")

        with self.assertRaisesRegex(SystemExit, "frozen judge identity"):
            self._export("conflicting-panel", model_family="changed-family")

    def test_import_rejects_an_altered_submitted_request_file(self) -> None:
        export_directory = self._export("altered-request")
        request_path = export_directory / "batch_requests.jsonl"
        request_path.write_text(
            request_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        batch_output = self.root / "altered-request-output.jsonl"
        _write_jsonl(
            batch_output,
            [_provider_result(self.tasks[0].task_id, _valid_judgment(50))],
        )

        with self.assertRaisesRegex(SystemExit, "does not match"):
            self._import(export_directory, batch_output, "provider-batch-altered")

    def test_import_rejects_result_for_an_unrequested_task(self) -> None:
        export_directory = self._export("batch-unknown")
        batch_output = self.root / "unknown-output.jsonl"
        _write_jsonl(batch_output, [_provider_result("unknown-task", _valid_judgment(50))])

        with self.assertRaisesRegex(SystemExit, "not declared"):
            self._import(export_directory, batch_output, "provider-batch-unknown")


if __name__ == "__main__":
    unittest.main()
