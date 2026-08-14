"""Behavioral contracts for the query-free surface-coverage corpus."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.query_free_surface_corpus import (
    RawSurfacePrompt,
    build_surface_coverage_corpus,
    read_dolly_surface_prompts,
    read_hh_surface_prompts,
)


class QueryFreeSurfaceCorpusTests(unittest.TestCase):
    def test_source_parsers_retain_only_human_instruction_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dolly = root / "dolly.jsonl"
            dolly.write_text(
                json.dumps(
                    {
                        "instruction": "Could you draft a concise project outline?",
                        "context": "Private source material",
                        "response": "This response must not enter the corpus.",
                        "category": "creative_writing",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            hh = root / "hh.jsonl.gz"
            payload = {
                "chosen": "\n\nHuman: How does this device work?\n\nAssistant: Answer A",
                "rejected": "\n\nHuman: How does this device work?\n\nAssistant: Answer B",
            }
            with gzip.open(hh, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

            dolly_rows = read_dolly_surface_prompts(dolly)
            hh_rows = read_hh_surface_prompts(hh)

        self.assertEqual(len(dolly_rows), 1)
        self.assertEqual(len(hh_rows), 1)
        self.assertNotIn("Private source material", dolly_rows[0].text)
        self.assertNotIn("response", dolly_rows[0].text.casefold())
        self.assertEqual(hh_rows[0].text, "How does this device work?")

    def test_corpus_is_deterministic_stratified_and_not_semantically_labeled(self) -> None:
        raw = []
        openings = (
            "How would you structure this request for a colleague?",
            "Please provide a short neutral outline for this task.",
            "Write a concise paragraph using a formal register.",
            "A detailed response would be useful for this situation.",
            "If time permits, organize the material into two clear sections.",
        )
        for source in ("source-a", "source-b"):
            for index in range(30):
                raw.append(
                    RawSurfacePrompt(
                        source_id=source,
                        source_record_id=f"{source}:{index}",
                        original_split="train",
                        source_category="test",
                        text=(
                            f"{openings[index % len(openings)]} "
                            f"Example from {source} number word {index}."
                        ),
                        has_attached_context=index % 2 == 0,
                    )
                )
        first, diagnostics = build_surface_coverage_corpus(
            raw,
            maximum_per_source=20,
            master_seed=17,
        )
        repeat, _ = build_surface_coverage_corpus(
            raw,
            maximum_per_source=20,
            master_seed=17,
        )

        self.assertEqual(first, repeat)
        self.assertEqual(len(first), 40)
        self.assertEqual(diagnostics.selected_count, 40)
        self.assertGreater(diagnostics.development_count, 0)
        self.assertGreater(diagnostics.confirmation_count, 0)
        self.assertEqual({item.source_id for item in first}, {"source-a", "source-b"})
        self.assertGreater(len({item.sentence_form for item in first}), 2)
        development_families = {
            item.surface_family_id
            for item in first
            if item.corpus_split == "development"
        }
        confirmation_families = {
            item.surface_family_id
            for item in first
            if item.corpus_split == "confirmation"
        }
        self.assertFalse(development_families & confirmation_families)
        self.assertTrue(all(not item.eligible_as_semantic_label for item in first))
        self.assertTrue(
            all(item.intended_use == "surface-style-coverage-only" for item in first)
        )

    def test_filter_removes_identifiers_long_text_and_exact_duplicates(self) -> None:
        valid = RawSurfacePrompt(
            source_id="source-a",
            source_record_id="valid",
            original_split="train",
            source_category="test",
            text="Could you organize these notes into a concise neutral response?",
            has_attached_context=False,
        )
        raw = (
            valid,
            replace(
                valid,
                source_record_id="duplicate",
                text=valid.text.upper(),
            ),
            replace(
                valid,
                source_record_id="url",
                text="Please inspect https://example.com and summarize its contents.",
            ),
            replace(
                valid,
                source_record_id="long",
                text="word " * 90,
            ),
        )
        records, diagnostics = build_surface_coverage_corpus(raw)
        rejections = dict(diagnostics.rejection_counts)

        self.assertEqual(len(records), 1)
        self.assertEqual(diagnostics.exact_duplicate_count, 1)
        self.assertEqual(rejections["external-identifier"], 1)
        self.assertEqual(rejections["too-long"], 1)


if __name__ == "__main__":
    unittest.main()
