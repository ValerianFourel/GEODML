"""Contracts for the query-conditioned randomized semantic A1 panel."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from analysis.interpretability.pipeline.a1_prompt_manifold import (
    FakeA1CandidateGenerator,
    FakeA1Embedder,
    FakeA1PairwiseJudge,
    build_a1_comparison_requests,
    calibrate_a1_candidates,
    embed_a1_candidates,
    generate_a1_candidate_bank,
    judge_a1_comparisons,
    select_a1_manifold,
)
from analysis.interpretability.pipeline.a1_query_panel import (
    build_query_conditioned_a1_panel,
)


def _selected_manifold():
    candidates = generate_a1_candidate_bank(
        search_term="calibration sentinel",
        style_seeds=(3, 7),
        number_candidates=3,
        generator=FakeA1CandidateGenerator(),
    )
    comparisons = build_a1_comparison_requests(candidates)
    judgments = judge_a1_comparisons(
        comparisons,
        candidates,
        (FakeA1PairwiseJudge("one"), FakeA1PairwiseJudge("two")),
    )
    calibrations = calibrate_a1_candidates(candidates, comparisons, judgments)
    input_embeddings = embed_a1_candidates(
        candidates,
        embedder=FakeA1Embedder("input"),
        representation="input",
    )
    response_embeddings = embed_a1_candidates(
        candidates,
        embedder=FakeA1Embedder("response", response=True),
        representation="anticipated-response",
    )
    selected, _ = select_a1_manifold(
        candidates,
        calibrations,
        input_embeddings,
        response_embeddings,
    )
    return selected


class A1QueryPanelTests(unittest.TestCase):
    def test_balanced_one_per_a1_keeps_every_level_and_randomizes_style(self) -> None:
        selected = _selected_manifold()
        queries = tuple(f"query {index}" for index in range(12))
        rows, diagnostics = build_query_conditioned_a1_panel(
            search_terms=queries,
            selected_prompts=selected,
            master_seed=73,
            style_assignment="balanced-one-per-a1",
        )

        expected_levels = {item.assigned_a1 for item in selected}
        self.assertEqual(len(expected_levels), 7)
        self.assertEqual(len(rows), 12 * 7)
        self.assertEqual(diagnostics.design, "randomized-complete-a1-block")
        self.assertEqual(diagnostics.style_assignment, "balanced-one-per-a1")
        self.assertEqual(diagnostics.prompts_per_query, 7)
        self.assertEqual(diagnostics.a1_level_coverage_rate, 1.0)
        self.assertEqual(diagnostics.complete_block_rate, 1.0)
        self.assertLessEqual(diagnostics.maximum_within_query_style_imbalance, 1)
        for query in queries:
            block = [row for row in rows if row.search_term == query]
            self.assertEqual({row.assigned_a1 for row in block}, expected_levels)
            counts = [
                sum(row.style_seed == style_seed for row in block)
                for style_seed in diagnostics.style_seeds
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_every_query_receives_the_complete_semantic_manifold(self) -> None:
        selected = _selected_manifold()
        queries = ("abandoned cart recovery", "CRM for nonprofits", "API monitoring")
        rows, diagnostics = build_query_conditioned_a1_panel(
            search_terms=queries,
            selected_prompts=selected,
            master_seed=41,
        )

        self.assertEqual(len(selected), 14)
        self.assertEqual(len(rows), 3 * 14)
        self.assertEqual(diagnostics.design, "randomized-complete-block")
        self.assertEqual(diagnostics.query_count, 3)
        self.assertEqual(diagnostics.prompts_per_query, 14)
        self.assertEqual(diagnostics.complete_block_rate, 1.0)
        self.assertEqual(diagnostics.exact_query_binding_rate, 1.0)
        self.assertEqual(diagnostics.duplicate_assignment_count, 0)
        self.assertEqual(diagnostics.duplicate_query_bound_prompt_count, 0)

        expected = {
            (item.prompt_assignment_id, item.assigned_a1, item.style_seed)
            for item in selected
        }
        for query in queries:
            block = [row for row in rows if row.search_term == query]
            observed = {
                (row.source_prompt_assignment_id, row.assigned_a1, row.style_seed)
                for row in block
            }
            self.assertEqual(observed, expected)
            self.assertEqual(
                sorted(row.within_keyword_order for row in block),
                list(range(1, 15)),
            )
            self.assertTrue(
                all(row.query_bound_prompt_template.count(query) == 1 for row in block)
            )
            self.assertTrue(
                all("{QUERY}" not in row.query_bound_prompt_template for row in block)
            )
            self.assertTrue(
                all(row.query_bound_prompt_template.count("{CANDIDATES}") == 1 for row in block)
            )

    def test_seed_changes_order_without_changing_membership(self) -> None:
        selected = _selected_manifold()
        arguments = {
            "search_terms": ("alpha", "beta", "gamma", "delta"),
            "selected_prompts": selected,
        }
        first, _ = build_query_conditioned_a1_panel(**arguments, master_seed=11)
        repeat, _ = build_query_conditioned_a1_panel(**arguments, master_seed=11)
        second, _ = build_query_conditioned_a1_panel(**arguments, master_seed=12)

        self.assertEqual(first, repeat)
        self.assertEqual(
            {row.panel_assignment_id for row in first},
            {row.panel_assignment_id for row in second},
        )
        self.assertNotEqual(
            [(row.search_term, row.source_prompt_assignment_id) for row in first],
            [(row.search_term, row.source_prompt_assignment_id) for row in second],
        )

    def test_duplicate_or_reserved_search_terms_are_rejected(self) -> None:
        selected = _selected_manifold()
        with self.assertRaisesRegex(ValueError, "duplicate search term"):
            build_query_conditioned_a1_panel(
                search_terms=("CRM", " crm "),
                selected_prompts=selected,
            )
        with self.assertRaisesRegex(ValueError, "reserved placeholder"):
            build_query_conditioned_a1_panel(
                search_terms=("bad {QUERY}",),
                selected_prompts=selected,
            )

    def test_cli_writes_a_reproducible_prompt_only_schedule(self) -> None:
        selected = _selected_manifold()
        script = Path(__file__).parents[1] / "scripts" / "build_a1_query_panel.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected_path = root / "selected.jsonl"
            selected_path.write_text(
                "".join(json.dumps(asdict(row)) + "\n" for row in selected),
                encoding="utf-8",
            )
            keywords_path = root / "keywords.txt"
            keywords_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            output = root / "panel"
            completed = subprocess.run(
                [
                    "python3",
                    str(script),
                    "--selected-manifold",
                    str(selected_path),
                    "--keywords-file",
                    str(keywords_path),
                    "--expected-keywords",
                    "3",
                    "--master-seed",
                    "91",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("14 prompts x 3 search terms", completed.stdout)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "scheduled-unrun")
            self.assertEqual(manifest["treatment"], "assigned_a1")
            self.assertEqual(manifest["blocking_variable"], "search_term")
            self.assertFalse(manifest["outcomes_observed"])
            self.assertEqual(
                len((output / "a1_query_prompt_panel.jsonl").read_text().splitlines()),
                42,
            )


if __name__ == "__main__":
    unittest.main()
