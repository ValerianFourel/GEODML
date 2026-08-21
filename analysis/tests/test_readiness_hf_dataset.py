"""Contracts for the resumable semantic-readiness Hugging Face export."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_hf_dataset import (
    HUB_SCOPE,
    LOCAL_SCOPE,
    assemble_readiness_export,
    atomic_json,
    atomic_jsonl,
    embed_prompt_shards,
    read_jsonl,
)
from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    ABSTENTION_LABEL_RUBRIC_VERSION,
    SemanticReadinessItem,
    build_readiness_label_tasks,
)
from analysis.scripts.build_readiness_hf_dataset import (
    _public_embedding_manifest,
    _source_catalog,
)


def _response(answer_type: str = "rating") -> str:
    rating = answer_type == "rating"
    return json.dumps(
        {
            "answer_type": answer_type,
            "overall_readiness_0_100": 55 if rating else None,
            "information_seeking_1_7": 5 if rating else None,
            "evaluation_1_7": 6 if rating else None,
            "selection_commitment_1_7": 3 if rating else None,
            "action_implementation_1_7": 2 if rating else None,
            "category": "comparison" if rating else None,
            "ambiguity_1_7": 2,
            "confidence_0_1": 0.9,
            "brief_reason": "The text compares options before making a choice.",
        }
    )


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, texts):
        self.calls.append(tuple(texts))
        return np.asarray(
            [[len(text), index, 1.0] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


class ReadinessHfDatasetTests(unittest.TestCase):
    def test_assembly_keeps_complete_local_panel_and_excludes_restricted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            allowed = _item(
                "a",
                "Compare these two software packages before choosing one.",
                "databricks-dolly-15k",
                "CC-BY-SA-3.0",
            )
            restricted = _item(
                "b",
                "Can somebody tell me which option I should order?",
                "allenai-wildchat-1m",
                "ODC-BY-1.0-DATABASE-CONTENT-RIGHTS-NOT-GRANTED",
            )
            tasks, codebook = build_readiness_label_tasks(
                (allowed, restricted),
                judge_slots=("judge-a", "judge-b"),
                rubric_version=ABSTENTION_LABEL_RUBRIC_VERSION,
            )
            corpus_path = root / "corpus.jsonl"
            tasks_path = root / "tasks.jsonl"
            codebook_path = root / "codebook.jsonl"
            atomic_jsonl(corpus_path, map(asdict, (allowed, restricted)))
            atomic_jsonl(tasks_path, map(asdict, tasks))
            atomic_jsonl(
                codebook_path,
                ({"task_id": task_id, **row} for task_id, row in codebook.items()),
            )
            queue_root = root / "queue"
            for slot in ("judge-a", "judge-b"):
                output = queue_root / "full" / slot
                cache = output / "task_cache"
                cache.mkdir(parents=True)
                identity = {
                    "judge_slot": slot,
                    "model": f"model-{slot}",
                    "model_family": "test-family",
                    "model_revision": f"revision-{slot}",
                    "backend": "test",
                    "precision": "bfloat16",
                }
                atomic_json(output / "judge_identity.json", identity)
                for task in tasks:
                    if task.judge_slot != slot:
                        continue
                    atomic_json(
                        cache / f"{task.task_id.replace(':', '_')}.json",
                        {
                            **identity,
                            "task_id": task.task_id,
                            "item_id": task.item_id,
                            "raw_response": _response(),
                            "rejected_attempts": [],
                        },
                    )

            output = root / "bundle"
            manifest = assemble_readiness_export(
                corpus_path=corpus_path,
                tasks_path=tasks_path,
                codebook_path=codebook_path,
                queue_root=queue_root,
                output_dir=output,
                expected_judge_slots=("judge-a", "judge-b"),
                git_commit_sha="a" * 40,
            )

            self.assertEqual(manifest["restricted_sources"], ["allenai-wildchat-1m"])
            self.assertEqual(manifest["restricted_prompt_count"], 1)
            self.assertEqual(
                len(read_jsonl(output / LOCAL_SCOPE / "prompts.jsonl")), 2
            )
            safe_prompts = read_jsonl(output / HUB_SCOPE / "prompts.jsonl")
            self.assertEqual([row["item_id"] for row in safe_prompts], [allowed.item_id])
            safe_annotations = read_jsonl(output / HUB_SCOPE / "annotations.jsonl")
            self.assertEqual(len(safe_annotations), 2)
            self.assertEqual(
                {row["model"] for row in safe_annotations},
                {"model-judge-a", "model-judge-b"},
            )
            self.assertTrue(all(row["answer_type"] == "rating" for row in safe_annotations))
            self.assertNotIn(
                restricted.text,
                (output / HUB_SCOPE / "annotations.jsonl").read_text(),
            )

    def test_failed_and_missing_tasks_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            item = _item(
                "a",
                "Compare these two software packages before choosing one.",
                "databricks-dolly-15k",
                "CC-BY-SA-3.0",
            )
            tasks, codebook = build_readiness_label_tasks(
                (item,),
                judge_slots=("judge-a", "judge-b"),
                rubric_version=ABSTENTION_LABEL_RUBRIC_VERSION,
            )
            atomic_jsonl(root / "corpus.jsonl", [asdict(item)])
            atomic_jsonl(root / "tasks.jsonl", map(asdict, tasks))
            atomic_jsonl(
                root / "codebook.jsonl",
                ({"task_id": task_id, **row} for task_id, row in codebook.items()),
            )
            output = root / "queue" / "full" / "judge-a"
            output.joinpath("task_cache").mkdir(parents=True)
            identity = {
                "judge_slot": "judge-a",
                "model": "model-a",
                "model_family": "family-a",
                "model_revision": "revision-a",
                "backend": "test",
            }
            task = next(task for task in tasks if task.judge_slot == "judge-a")
            atomic_json(
                output / "task_cache" / f"{task.task_id.replace(':', '_')}.failed.json",
                {
                    **identity,
                    "task_id": task.task_id,
                    "item_id": task.item_id,
                    "attempts": [{"attempt": 1, "error": "invalid", "raw": "x"}],
                },
            )
            bundle = root / "bundle"
            assemble_readiness_export(
                corpus_path=root / "corpus.jsonl",
                tasks_path=root / "tasks.jsonl",
                codebook_path=root / "codebook.jsonl",
                queue_root=root / "queue",
                output_dir=bundle,
                expected_judge_slots=("judge-a", "judge-b"),
                git_commit_sha="b" * 40,
            )
            failures = read_jsonl(bundle / LOCAL_SCOPE / "failures.jsonl")
            missing = read_jsonl(bundle / LOCAL_SCOPE / "missing_tasks.jsonl")
            self.assertEqual(failures[0]["status"], "failed_validation")
            self.assertEqual(failures[0]["attempt_count"], 1)
            self.assertEqual(missing[0]["status"], "model_not_started")

    def test_embedding_shards_resume_without_recomputing_completed_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = [
                {
                    "item_id": f"item:{index}",
                    "text": f"prompt text {index}",
                    "text_sha256": str(index) * 64,
                }
                for index in range(3)
            ]
            prompt_path = root / "prompts.jsonl"
            atomic_jsonl(prompt_path, prompts)
            view = {
                "view_name": "test-view",
                "backend": "llm2vec",
                "embedding_model": "model",
                "embedding_model_revision": "revision",
            }
            first_embedder = _FakeEmbedder()
            first = embed_prompt_shards(
                prompts_path=prompt_path,
                output_dir=root / "embeddings",
                view=view,
                shard_size=2,
                embedder_factory=lambda _: first_embedder,
            )
            self.assertTrue(first["is_complete"])
            self.assertEqual(first["completed_shard_count"], 2)
            self.assertEqual(len(first_embedder.calls), 2)

            def fail_factory(_):
                raise AssertionError("completed resume must not reload the model")

            second = embed_prompt_shards(
                prompts_path=prompt_path,
                output_dir=root / "embeddings",
                view=view,
                shard_size=2,
                embedder_factory=fail_factory,
            )
            self.assertEqual(second, first)

            prompt_path.write_text(
                prompt_path.read_text(encoding="utf-8").replace("prompt text 0", "changed"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frozen embedding identity"):
                embed_prompt_shards(
                    prompts_path=prompt_path,
                    output_dir=root / "embeddings",
                    view=view,
                    shard_size=2,
                    embedder_factory=fail_factory,
                )

    def test_public_metadata_uses_repository_ids_and_source_attribution(self) -> None:
        manifest = _public_embedding_manifest(
            {
                "view": {
                    "view_name": "test-view",
                    "backend": "llm2vec",
                    "embedding_model": "/e/private/models/qwen/Qwen3-8B/revision-a",
                    "embedding_model_id": "Qwen/Qwen3-8B",
                    "embedding_model_revision": "revision-a",
                    "mntp_model": None,
                    "mntp_model_id": None,
                    "mntp_model_revision": None,
                    "peft_model": None,
                    "peft_model_id": None,
                    "peft_model_revision": None,
                },
                "embedding_dimension": 4096,
                "item_count": 2,
                "view_config_sha256": "c" * 64,
            }
        )
        self.assertEqual(manifest["embedding_model"], "Qwen/Qwen3-8B")
        self.assertNotIn("/e/private", json.dumps(manifest))

        catalog = _source_catalog(
            [
                {
                    "source_name": "databricks-dolly-15k",
                    "source_url": None,
                    "license": "CC-BY-SA-3.0",
                }
            ]
        )
        self.assertEqual(
            catalog[0]["source_url"],
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        )

    def test_source_catalog_preserves_multiple_licenses_per_source(self) -> None:
        catalog = _source_catalog(
            [
                {
                    "source_name": "stackexchange:askubuntu",
                    "source_url": "https://askubuntu.com/questions/1/example",
                    "license": "CC-BY-SA-3.0",
                },
                {
                    "source_name": "stackexchange:askubuntu",
                    "source_url": "https://askubuntu.com/questions/2/example",
                    "license": "CC-BY-SA-4.0",
                },
                {
                    "source_name": "stackexchange:askubuntu",
                    "source_url": "https://askubuntu.com/questions/3/example",
                    "license": "CC-BY-SA-4.0",
                },
            ]
        )

        self.assertEqual(
            catalog,
            [
                {
                    "split": "data",
                    "source_name": "stackexchange:askubuntu",
                    "license": "CC-BY-SA-3.0",
                    "source_url": "https://askubuntu.com",
                    "prompt_count": 1,
                },
                {
                    "split": "data",
                    "source_name": "stackexchange:askubuntu",
                    "license": "CC-BY-SA-4.0",
                    "source_url": "https://askubuntu.com",
                    "prompt_count": 2,
                },
            ],
        )


def _item(suffix: str, text: str, source_name: str, license_name: str) -> SemanticReadinessItem:
    import hashlib

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SemanticReadinessItem(
        item_id=f"semantic-item:{suffix * 24}",
        source_kind="test-source",
        source_name=source_name,
        source_record_id=f"record-{suffix}",
        text=text,
        text_sha256=text_hash,
        split="development",
        group_id=f"group-{suffix}",
        source_url=None,
        author_name=None,
        author_url=None,
        license=license_name,
    )


if __name__ == "__main__":
    unittest.main()
