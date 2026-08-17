"""CPU contracts for the four-GPU readiness judge runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    ReadinessLabelTask,
)
from analysis.scripts.run_semantic_readiness_judge_4gpu import (
    _completed_resume_matches,
    _is_noop_completed_resume,
    _prompt_for_attempt,
    _render_chat_prompts,
    _shard_tasks,
    _validate_arguments,
)


def _task(index: int) -> ReadinessLabelTask:
    return ReadinessLabelTask(
        task_id=f"task:{index}",
        item_id=f"item:{index}",
        judge_slot="primary-frontier",
        presentation_variant="forward-anchors",
        rubric_version="test",
        prompt=f"PROMPT {index}",
    )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **options):
        self.calls.append((messages, options))
        return f"rendered:{messages[0]['content']}"


class SemanticReadinessJudge4GpuTests(unittest.TestCase):
    def test_strided_shards_are_complete_balanced_and_disjoint(self) -> None:
        tasks = tuple(_task(index) for index in range(11))
        shards = [_shard_tasks(tasks, rank, 4) for rank in range(4)]

        flattened = [task for shard in shards for task in shard]
        self.assertEqual({task.task_id for task in flattened}, {task.task_id for task in tasks})
        self.assertEqual(len(flattened), len(tasks))
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)
        self.assertEqual(
            [[task.task_id for task in shard] for shard in shards],
            [
                ["task:0", "task:4", "task:8"],
                ["task:1", "task:5", "task:9"],
                ["task:2", "task:6", "task:10"],
                ["task:3", "task:7"],
            ],
        )

    def test_qwen_thinking_is_disabled_at_chat_render_time(self) -> None:
        tokenizer = _FakeTokenizer()

        rendered = _render_chat_prompts(
            tokenizer,
            ["first", "second"],
            disable_thinking=True,
        )

        self.assertEqual(rendered, ["rendered:first", "rendered:second"])
        self.assertEqual(len(tokenizer.calls), 2)
        for messages, options in tokenizer.calls:
            self.assertEqual(messages[0]["role"], "user")
            self.assertFalse(options["enable_thinking"])
            self.assertFalse(options["tokenize"])
            self.assertTrue(options["add_generation_prompt"])

    def test_retry_prompt_preserves_the_original_frozen_rubric(self) -> None:
        task = _task(1)
        prompt = _prompt_for_attempt(
            task,
            [{"error": "bad category", "raw": '{"category":"review"}'}],
        )

        self.assertIn(task.prompt, prompt)
        self.assertIn("bad category", prompt)
        self.assertIn('{"category":"review"}', prompt)
        self.assertIn("<previous_invalid_response>", prompt)
        self.assertIn('"information_seeking_1_7"', prompt)

    def test_production_rejects_a_partial_task_slice(self) -> None:
        args = SimpleNamespace(
            judge_slot="primary-frontier",
            model="model",
            model_family="family",
            model_revision="revision",
            expected_tasks=5091,
            batch_size=8,
            max_input_tokens=2048,
            max_new_tokens=300,
            maximum_attempts=5,
            expected_world_size=4,
            start_index=0,
            limit=8,
            run_purpose="production",
        )

        with self.assertRaisesRegex(SystemExit, "may not set --limit"):
            _validate_arguments(args)

    def test_completed_cache_only_resume_preserves_existing_artifacts(self) -> None:
        args = SimpleNamespace(resume=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            worker_manifest = output / "workers" / "rank-0" / "worker_manifest.json"
            worker_manifest.parent.mkdir(parents=True)
            worker_manifest.write_text("{}\n", encoding="utf-8")
            (output / "run_manifest.json").write_text(
                json.dumps({"is_complete": True}) + "\n",
                encoding="utf-8",
            )

            self.assertTrue(
                _is_noop_completed_resume(
                    output,
                    args=args,
                    cached_count=3,
                    generated_count=0,
                    exhausted_task_ids=[],
                    worker_task_count=3,
                    worker_manifest_path=worker_manifest,
                )
            )

    def test_completed_resume_requires_identical_merged_responses(self) -> None:
        args = SimpleNamespace(resume=True)
        expected_manifest = {
            "format_version": "v1",
            "backend": "backend",
            "judge_slot": "slot",
            "model": "model",
            "model_family": "family",
            "model_revision": "revision",
            "task_file_sha256": "sha",
            "selected_task_count": 1,
            "start_index": 0,
            "limit": None,
            "world_size": 4,
            "batch_size_per_gpu": 32,
            "is_complete": True,
        }
        responses = [{"task_id": "task:0", "raw_response": "{}"}]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "run_manifest.json").write_text(
                json.dumps(expected_manifest) + "\n",
                encoding="utf-8",
            )
            (output / "judge_responses.jsonl").write_text(
                json.dumps(responses[0]) + "\n",
                encoding="utf-8",
            )

            self.assertTrue(
                _completed_resume_matches(
                    output,
                    args=args,
                    responses=responses,
                    expected_manifest=expected_manifest,
                )
            )
            responses[0]["raw_response"] = "changed"
            self.assertFalse(
                _completed_resume_matches(
                    output,
                    args=args,
                    responses=responses,
                    expected_manifest=expected_manifest,
                )
            )


if __name__ == "__main__":
    unittest.main()
