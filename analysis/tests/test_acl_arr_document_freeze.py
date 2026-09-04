"""Contracts for freezing retrieved page text before ACL ARR inference."""

from __future__ import annotations

import unittest

from analysis.interpretability.pipeline.acl_arr_document_freeze import (
    build_frozen_document_sets,
)


class AclArrDocumentFreezeTests(unittest.TestCase):
    def test_freezes_extracted_text_without_live_fetch_or_silent_fallback(self) -> None:
        serp = [
            {
                "keyword": "alpha software",
                "position": index,
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": f"Snippet {index}",
            }
            for index in range(1, 4)
        ]
        pages = [
            {
                "url": f"https://example.com/{index}",
                "extracted_text": f"Full frozen page {index}." * 20,
            }
            for index in range(1, 4)
        ]
        frozen, summary = build_frozen_document_sets(
            serp,
            pages,
            minimum_documents=3,
            maximum_documents=3,
            max_document_characters=80,
            allow_snippet_fallback=False,
            search_snapshot_sha256="a" * 64,
        )
        self.assertEqual(summary["complete_keyword_count"], 1)
        self.assertEqual(summary["snippet_fallback_count"], 0)
        self.assertEqual(len(frozen), 1)
        documents = frozen[0]["documents"]
        self.assertEqual([row["document_id"] for row in documents], ["C001", "C002", "C003"])
        self.assertTrue(all(len(row["text"]) <= 80 for row in documents))
        self.assertTrue(all(row["text_source"] == "extracted_page" for row in documents))
        self.assertTrue(all(len(row["full_text_sha256"]) == 64 for row in documents))

        with self.assertRaisesRegex(ValueError, "incomplete frozen document pools"):
            build_frozen_document_sets(
                serp,
                pages[:2],
                minimum_documents=3,
                maximum_documents=3,
                max_document_characters=80,
                allow_snippet_fallback=False,
                search_snapshot_sha256="a" * 64,
            )

        fallback, fallback_summary = build_frozen_document_sets(
            serp,
            pages[:2],
            minimum_documents=3,
            maximum_documents=3,
            max_document_characters=80,
            allow_snippet_fallback=True,
            search_snapshot_sha256="a" * 64,
        )
        self.assertEqual(fallback_summary["snippet_fallback_count"], 1)
        self.assertEqual(fallback[0]["documents"][2]["text_source"], "serp_snippet")


if __name__ == "__main__":
    unittest.main()
