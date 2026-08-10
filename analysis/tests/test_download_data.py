from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from analysis.scripts.download_data import (
    main,
    patterns_for_component,
    select_repo_files,
)


class DownloadDataTests(unittest.TestCase):
    def test_full_component_preserves_complete_snapshot_behavior(self) -> None:
        self.assertIsNone(patterns_for_component("full"))

    def test_serp_component_selects_only_search_pools_and_readme(self) -> None:
        patterns = patterns_for_component("serp")
        self.assertIsNotNone(patterns)
        assert patterns is not None
        self.assertIn("data/serp/**", patterns)
        self.assertFalse(any("html_cache" in pattern for pattern in patterns))

    def test_dataforseo_component_excludes_raw_responses(self) -> None:
        repo_files = (
            "README.md",
            "data/dataforseo/domain_authority_dfs.parquet",
            "data/dataforseo/run_manifest.json",
            "data/dataforseo/raw/serp_google_organic_chunk_000.json",
            "data/dataforseo/.checkpoints/whois/chunk_0001.json",
        )

        selected = select_repo_files(
            repo_files,
            patterns_for_component("dataforseo") or (),
        )

        self.assertEqual(
            selected,
            (
                "README.md",
                "data/dataforseo/domain_authority_dfs.parquet",
                "data/dataforseo/run_manifest.json",
            ),
        )

    def test_dataforseo_full_component_includes_nested_raw_responses(self) -> None:
        repo_files = (
            "data/dataforseo/domain_authority_dfs.parquet",
            "data/dataforseo/raw/serp_google_organic_chunk_000.json",
            "data/dataforseo/.checkpoints/whois/chunk_0001.json",
        )

        selected = select_repo_files(
            repo_files,
            patterns_for_component("dataforseo-full") or (),
        )

        self.assertEqual(selected, tuple(sorted(repo_files)))

    def test_html_component_selects_tarballs_not_duplicate_loose_html(self) -> None:
        repo_files = (
            "README.md",
            "data/runs/run_a/phase2/html_cache.tar.gz",
            "data/runs/run_a/phase2/html_cache/page.html",
            "data/runs/run_a/phase2/keywords.jsonl",
        )

        selected = select_repo_files(
            repo_files,
            patterns_for_component("html") or (),
        )

        self.assertEqual(
            selected,
            (
                "README.md",
                "data/runs/run_a/phase2/html_cache.tar.gz",
            ),
        )

    def test_unknown_component_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown component"):
            patterns_for_component("unknown")

    def test_select_repo_files_matches_nested_component_paths(self) -> None:
        repo_files = (
            "README.md",
            "data/serp/phase0_top20_ddg.parquet",
            "data/serp/phase0_top50_searxng.json",
            "data/features/features_ddg_top20.parquet",
        )

        selected = select_repo_files(
            repo_files,
            patterns_for_component("serp") or (),
        )

        self.assertEqual(
            selected,
            (
                "README.md",
                "data/serp/phase0_top20_ddg.parquet",
                "data/serp/phase0_top50_searxng.json",
            ),
        )

    def test_dry_run_does_not_create_destination_or_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "not-created"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "--component",
                        "serp",
                        "--local-dir",
                        str(destination),
                        "--dry-run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertFalse(destination.exists())
            self.assertIn("data/serp/**", output.getvalue())


if __name__ == "__main__":
    unittest.main()
