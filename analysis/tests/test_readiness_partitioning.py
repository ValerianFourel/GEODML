"""Contracts for disjoint readiness refinement and exact checkpoint union."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.scripts.merge_readiness_partition_checkpoints import (
    merge_partition_checkpoints,
)
from analysis.scripts.partition_readiness_refinement_tasks import (
    prepare_partition_batch,
    select_partition_batch,
    target_partition,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class ReadinessPartitioningTests(unittest.TestCase):
    def _tasks(self, count: int = 40) -> list[dict[str, object]]:
        return [
            {
                "task_id": f"task-{index}",
                "keyword_id": f"keyword-{index // 4}",
                "generator_id": "a" if index % 2 == 0 else "b",
                "target": {"target_id": f"target-{index}"},
            }
            for index in range(count)
        ]

    def test_target_partition_is_keyword_stable_disjoint_and_exhaustive(self) -> None:
        rows = self._tasks()
        owners = [
            target_partition(
                row,
                partition_count=2,
                partition_salt="test-salt",
            )
            for row in rows
        ]
        self.assertEqual(set(owners), {0, 1})
        owners_by_keyword: dict[str, set[int]] = {}
        for row, owner in zip(rows, owners):
            owners_by_keyword.setdefault(str(row["keyword_id"]), set()).add(owner)
        self.assertTrue(all(len(values) == 1 for values in owners_by_keyword.values()))
        changed_round = dict(rows[0], task_id="new-round-task")
        self.assertEqual(
            owners[0],
            target_partition(
                changed_round,
                partition_count=2,
                partition_salt="test-salt",
            ),
        )
        changed_target = dict(
            rows[0],
            task_id="new-target-task",
            target={"target_id": "another-intensity-for-the-same-keyword"},
        )
        self.assertEqual(
            owners[0],
            target_partition(
                changed_target,
                partition_count=2,
                partition_salt="test-salt",
            ),
        )
        selected = [
            select_partition_batch(
                rows,
                source_sha256="abc",
                limit=100,
                partition_count=2,
                partition_index=index,
                partition_salt="test-salt",
            )
            for index in (0, 1)
        ]
        ids = [{str(row["task_id"]) for row in group} for group in selected]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(ids[0] | ids[1], {str(row["task_id"]) for row in rows})

    def test_partition_batch_is_immutable_and_records_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tasks.jsonl"
            output = root / "batch.jsonl"
            _write_jsonl(source, self._tasks())
            first = prepare_partition_batch(
                source,
                output,
                limit=7,
                partition_count=2,
                partition_index=1,
                partition_salt="test-salt",
            )
            second = prepare_partition_batch(
                source,
                output,
                limit=7,
                partition_count=2,
                partition_index=1,
                partition_salt="test-salt",
            )
            self.assertEqual(first, second)
            self.assertEqual(first["selected_task_count"], 7)
            self.assertEqual(first["partition_index"], 1)
            self.assertEqual(
                first["selection_method"],
                "stable-keyword-hash-partition-generator-round-robin-v3",
            )
            self.assertLessEqual(
                first["selected_task_count"], first["owned_source_task_count"]
            )

    def _partition_fixture(
        self,
        root: Path,
        *,
        index: int,
        unique_id: str,
        partition_count: int = 2,
    ) -> Path:
        partition = root / f"partition-{index}"
        round_root = partition / "round-00"
        _write_json(
            partition / "pipeline_manifest.json",
            {
                "git_commit_sha": "abc123",
                "plan_manifest_sha256": "plan-sha",
                "generator_ids": ["a", "b"],
                "generator_models": ["model-a", "model-b"],
                "validator_id": "judge",
                "validator_model": "judge-model",
                "distance_tolerance": 0.017,
                "disagreement_weight": 0.1,
                "refinement_candidates_per_task": 4,
                "master_seed": 7,
                "work_partition_count": partition_count,
                "work_partition_index": index,
                "work_partition_salt": "test-salt",
            },
        )
        shared = {
            "candidate_id": "shared",
            "question": "Shared question?",
            "round_index": 0,
        }
        unique = {
            "candidate_id": unique_id,
            "question": f"Question {unique_id}?",
            "round_index": index + 1,
        }
        candidate_file = partition / "candidates.jsonl"
        _write_jsonl(candidate_file, [shared, unique])
        (round_root / "candidate-files.txt").parent.mkdir(parents=True, exist_ok=True)
        (round_root / "candidate-files.txt").write_text(str(candidate_file) + "\n")
        _write_json(round_root / "verified_round_summary.json", {"candidate_count": 2})
        validation = [
            {"candidate_id": "shared", "accepted": True},
            {"candidate_id": unique_id, "accepted": True},
        ]
        _write_jsonl(round_root / "validation.jsonl", validation)
        _write_json(
            round_root / "validation.jsonl.manifest.json",
            {
                "format_version": "test",
                "judge_id": "judge",
                "judge_model": "judge-model",
                "judge_backend": "local",
                "judge_precision": "full",
                "acceptance_contract": "test contract",
            },
        )
        for view, map_id in (("qwen", "qwen-map"), ("mistral", "mistral-map")):
            projection = round_root / "projections" / view
            projection_rows = [
                {"candidate_id": "shared", "projection": {"item_id": "shared"}},
                {"candidate_id": unique_id, "projection": {"item_id": unique_id}},
            ]
            _write_jsonl(projection / "question_projections.jsonl", projection_rows)
            projection.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                projection / "question_embeddings.restricted-local.npz",
                candidate_ids=np.asarray(["shared", unique_id]),
                embeddings=np.asarray(
                    [[1.0, 2.0], [3.0 + index, 4.0]], dtype=np.float32
                ),
            )
            _write_json(
                projection / "projection_manifest.json",
                {
                    "format_version": "test",
                    "git_commit_sha": "abc123",
                    "map_id": map_id,
                    "map": {"sha256": f"{view}-map"},
                    "reference_coordinates": {"sha256": f"{view}-reference"},
                    "embedding": {
                        "model": f"{view}-model",
                        "mntp_model": None,
                        "peft_model": None,
                        "batch_size": 8,
                        "max_length": 512,
                        "attention_implementation": "eager",
                    },
                },
            )
        return partition

    def test_checkpoint_merge_requires_identical_overlap_and_unions_unique_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self._partition_fixture(root, index=0, unique_id="left")
            right = self._partition_fixture(root, index=1, unique_id="right")
            output = root / "merged"
            manifest = merge_partition_checkpoints((left, right), output)
            self.assertEqual(manifest["candidate_count"], 3)
            self.assertEqual(manifest["accepted_count"], 3)
            self.assertEqual(manifest["maximum_candidate_round_index"], 2)
            ids = {
                json.loads(line)["candidate_id"]
                for line in (output / "candidates.jsonl").read_text().splitlines()
            }
            self.assertEqual(ids, {"shared", "left", "right"})
            with np.load(
                output / "projections/qwen/question_embeddings.restricted-local.npz",
                allow_pickle=False,
            ) as payload:
                self.assertEqual(set(payload["candidate_ids"]), ids)

    def test_checkpoint_merge_accepts_one_complete_ten_section_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sections = [
                self._partition_fixture(
                    root,
                    index=index,
                    unique_id=f"section-{index}",
                    partition_count=10,
                )
                for index in range(10)
            ]
            manifest = merge_partition_checkpoints(
                sections,
                root / "merged-ten",
            )
            self.assertEqual(manifest["partition_count"], 10)
            self.assertEqual(manifest["candidate_count"], 11)
            self.assertEqual(
                manifest["format_version"],
                "readiness-partition-checkpoint-union-v2",
            )

    def test_checkpoint_merge_rejects_an_incomplete_ten_section_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sections = [
                self._partition_fixture(
                    root,
                    index=index,
                    unique_id=f"section-{index}",
                    partition_count=10,
                )
                for index in range(2)
            ]
            with self.assertRaisesRegex(ValueError, "complete indexed set"):
                merge_partition_checkpoints(sections, root / "incomplete")


if __name__ == "__main__":
    unittest.main()
