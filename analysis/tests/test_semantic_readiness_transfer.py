"""Contracts for the versioned multi-dataset readiness transfer panel."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    build_readiness_label_tasks,
    build_semantic_readiness_corpus,
)
from analysis.interpretability.pipeline.semantic_readiness_transfer import (
    TransferPromptRecord,
    build_transfer_prompt_panel,
    extend_semantic_readiness_corpus,
    load_transfer_source_specification,
)
from analysis.scripts.build_semantic_readiness_dataset import _iter_source_rows


class SemanticReadinessTransferTests(unittest.TestCase):
    def test_specification_freezes_eight_sources_and_source_heldout_splits(self) -> None:
        sources = load_transfer_source_specification()
        self.assertEqual(len(sources), 8)
        self.assertEqual(
            {item.source_id for item in sources if item.split == "development"},
            {
                "openassistant-oasst1",
                "google-ccpe-m",
                "google-taskmaster-1",
                "microsoft-ms-marco-v1",
            },
        )
        self.assertEqual(
            {item.source_id for item in sources if item.split == "confirmation"},
            {
                "allenai-wildchat-1m",
                "google-schema-guided-dialogue",
                "amazon-shopping-queries",
                "lmsys-chat-1m",
            },
        )
        self.assertTrue(
            all(
                "local-only" in item.redistribution_policy
                for item in sources
                if item.source_id
                in {
                    "allenai-wildchat-1m",
                    "lmsys-chat-1m",
                    "microsoft-ms-marco-v1",
                }
            )
        )

    def test_local_snapshot_loader_orders_json_and_tsv_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.tsv").write_text(
                "query-2\tWhich account should I choose?\n",
                encoding="utf-8",
            )
            (root / "a.json").write_text(
                json.dumps(
                    [{"query_id": "query-1", "query": "How do accounts work?"}]
                ),
                encoding="utf-8",
            )
            rows = tuple(_iter_source_rows(root))
        self.assertEqual([row["query_id"] for row in rows], ["query-1", "query-2"])

    def test_all_source_adapters_extract_only_one_first_user_text(self) -> None:
        sources = load_transfer_source_specification()
        rows = {
            "openassistant-oasst1": (
                {
                    "message_id": "oasst-root",
                    "parent_id": None,
                    "role": "prompter",
                    "lang": "en",
                    "text": "How do solar panels generate electricity?",
                },
                {
                    "message_id": "oasst-reply",
                    "parent_id": "oasst-root",
                    "role": "prompter",
                    "lang": "en",
                    "text": "This contextual reply must not be sampled.",
                },
            ),
            "google-ccpe-m": (
                {
                    "conversationId": "ccpe-1",
                    "utterances": [
                        {"speaker": "ASSISTANT", "text": "What do you enjoy?"},
                        {"speaker": "USER", "text": "I prefer quiet historical films."},
                    ],
                },
            ),
            "google-taskmaster-1": (
                {
                    "conversation_id": "tm-1",
                    "utterances": [
                        {"speaker": "USER", "text": "Book a table for four tonight."},
                        {"speaker": "ASSISTANT", "text": "Where?"},
                    ],
                },
            ),
            "microsoft-ms-marco-v1": (
                {"query_id": "marco-1", "query": "how does mortgage refinancing work"},
            ),
            "allenai-wildchat-1m": (
                {
                    "conversation_hash": "wild-1",
                    "language": "English",
                    "conversation": [
                        {
                            "role": "user",
                            "content": "Compare these laptop options for me.",
                        },
                        {"role": "assistant", "content": "Certainly."},
                    ],
                },
            ),
            "google-schema-guided-dialogue": (
                {
                    "dialogue_id": "sgd-1",
                    "turns": [
                        {
                            "speaker": "USER",
                            "utterance": "Find a train leaving tomorrow morning.",
                        },
                        {"speaker": "SYSTEM", "utterance": "From where?"},
                    ],
                },
            ),
            "amazon-shopping-queries": (
                {
                    "query_id": "amazon-1",
                    "query": "waterproof trail shoes women",
                    "product_locale": "us",
                },
                {
                    "query_id": "amazon-es",
                    "query": "zapatos para lluvia mujer",
                    "product_locale": "es",
                },
            ),
            "lmsys-chat-1m": (
                {
                    "conversation_id": "lmsys-1",
                    "language": "English",
                    "conversation": [
                        {"role": "user", "content": "Help me choose a retirement account."},
                    ],
                },
            ),
        }
        revisions = {
            source_id: f"revision-{index}"
            for index, source_id in enumerate(rows)
        }
        records, diagnostics = build_transfer_prompt_panel(
            rows,
            source_revisions=revisions,
            sources=sources,
            maximum_per_source=10,
            master_seed=7,
        )
        self.assertEqual(len(records), 8)
        self.assertEqual({item.source_id for item in records}, set(rows))
        self.assertFalse(any("contextual reply" in item.text for item in records))
        self.assertFalse(any("zapatos" in item.text for item in records))
        self.assertEqual(sum(item.selected_prompt_count for item in diagnostics), 8)
        self.assertEqual(
            build_transfer_prompt_panel(
                rows,
                source_revisions=revisions,
                sources=sources,
                maximum_per_source=10,
                master_seed=7,
            )[0],
            records,
        )

    def test_extension_preserves_frozen_rows_tasks_and_exact_deduplication(self) -> None:
        sources = load_transfer_source_specification()
        base = build_semantic_readiness_corpus(
            (
                {
                    "source_id": "databricks-dolly-15k",
                    "source_record_id": "dolly:1",
                    "text": "Explain how solar panels generate electricity.",
                    "corpus_split": "development",
                    "surface_family_id": "family:1",
                },
            ),
            (),
        )
        old_tasks, _ = build_readiness_label_tasks(base, judge_slots=("judge-a",))
        duplicate_hash = base[0].text_sha256
        records = (
            TransferPromptRecord(
                transfer_record_id="transfer:base-duplicate",
                source_id="openassistant-oasst1",
                source_record_id="oasst-duplicate",
                text=base[0].text,
                text_sha256=duplicate_hash,
                split="development",
                group_id="oasst-duplicate",
                source_url=None,
                license="Apache-2.0",
                redistribution_policy="source-license-and-attribution",
                source_revision="revision-a",
            ),
            *build_transfer_prompt_panel(
                {
                    "google-taskmaster-1": (
                        {
                            "conversation_id": "tm-new",
                            "utterances": [
                                {
                                    "speaker": "USER",
                                    "text": "Reserve two seats for the evening show.",
                                },
                            ],
                        },
                    ),
                },
                source_revisions={"google-taskmaster-1": "revision-b"},
                sources=sources,
                maximum_per_source=10,
            )[0],
        )
        transfer, expanded, diagnostics = extend_semantic_readiness_corpus(
            base,
            records,
            sources=sources,
        )
        self.assertEqual(tuple(expanded[: len(base)]), base)
        self.assertEqual(len(transfer), 1)
        self.assertEqual(diagnostics.duplicate_of_base_count, 1)
        new_tasks, _ = build_readiness_label_tasks(expanded, judge_slots=("judge-a",))
        self.assertEqual(new_tasks[0].task_id, old_tasks[0].task_id)
        self.assertEqual(new_tasks[0].prompt, old_tasks[0].prompt)
        self.assertNotIn("google-taskmaster-1", new_tasks[-1].prompt)


if __name__ == "__main__":
    unittest.main()
