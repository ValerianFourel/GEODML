"""End-to-end contracts for the supervised readiness subspace bridge."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_hf_dataset import (
    atomic_jsonl,
    embed_prompt_shards,
    read_json,
    read_jsonl,
)
from analysis.interpretability.pipeline.readiness_hf_subspace import (
    fit_readiness_hf_subspace,
)
from analysis.interpretability.pipeline.semantic_readiness_dataset import (
    SemanticReadinessItem,
    normalize_semantic_readiness_text,
)


class _SyntheticEmbedder:
    def embed(self, texts):
        rows = []
        for text in texts:
            index = int(text.rsplit(" ", 1)[1])
            readiness = index / 39
            rows.append(
                [readiness - 0.5, 0.2 * (index % 3), 1.0, 0.01 * (index % 5)]
            )
        return np.asarray(rows, dtype=np.float32)


class ReadinessHfSubspaceTests(unittest.TestCase):
    def test_three_judge_bundle_fit_excludes_fourth_slot_and_holds_out_confirmation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            items = tuple(_item(index) for index in range(40))
            prompt_path = root / "prompts.jsonl"
            annotation_path = root / "annotations.jsonl"
            atomic_jsonl(
                prompt_path,
                (
                    {
                        "dataset_format_version": "semantic-readiness-hf-bundle-v1",
                        **asdict(item),
                    }
                    for item in items
                ),
            )
            rows = []
            selected_slots = ("judge-a", "judge-b", "judge-c")
            for index, item in enumerate(items):
                readiness = index / 39
                for slot_index, slot in enumerate(selected_slots):
                    answer_type = (
                        "dont_know" if index == 7 and slot == "judge-c" else "rating"
                    )
                    rows.append(
                        _annotation(
                            item.item_id,
                            slot,
                            readiness,
                            slot_index,
                            answer_type=answer_type,
                        )
                    )
                rows.append(
                    _annotation(
                        item.item_id,
                        "unfinished-judge-d",
                        1.0 - readiness,
                        0,
                    )
                )
            atomic_jsonl(annotation_path, rows)
            embedding_dir = root / "embedding-view"
            embed_prompt_shards(
                prompts_path=prompt_path,
                output_dir=embedding_dir,
                view={
                    "view_name": "synthetic-llm2vec",
                    "backend": "llm2vec",
                    "embedding_model_id": "test/model",
                    "embedding_model_revision": "r" * 40,
                },
                shard_size=11,
                embedder_factory=lambda _: _SyntheticEmbedder(),
            )

            output = root / "subspace"
            manifest = fit_readiness_hf_subspace(
                prompts_path=prompt_path,
                annotations_path=annotation_path,
                embedding_dir=embedding_dir,
                output_dir=output,
                judge_slots=selected_slots,
                git_commit_sha="a" * 40,
                ridge_penalty=0.01,
            )

            self.assertEqual(manifest["judge_slots"], list(selected_slots))
            self.assertEqual(
                manifest["excluded_judge_slots"], ["unfinished-judge-d"]
            )
            self.assertEqual(manifest["development_prompt_count"], 30)
            self.assertEqual(manifest["confirmation_prompt_count"], 10)
            self.assertEqual(manifest["training_item_count"], 30)
            self.assertEqual(manifest["embedding_dimension"], 4)
            self.assertEqual(manifest["compute_backend"], "numpy")
            self.assertEqual(
                manifest["evidence_assessment"]["status"], "inconclusive"
            )
            self.assertEqual(
                read_json(output / "subspace_manifest.json")["map_id"],
                manifest["map_id"],
            )
            consensus = read_jsonl(output / "readiness_consensus.jsonl")
            uncertain = next(row for row in consensus if row["item_id"] == "item:7")
            self.assertEqual(uncertain["judge_count"], 2)
            self.assertTrue(uncertain["usable_for_axis"])
            coordinates = read_jsonl(
                output / "readiness_embedding_coordinates.jsonl"
            )
            self.assertEqual(len(coordinates), 40)
            self.assertEqual(
                sum(row["evaluation_split"] == "confirmation" for row in coordinates),
                10,
            )
            diagnostics = read_json(
                output / "readiness_embedding_map_diagnostics.json"
            )
            self.assertGreater(diagnostics["confirmation"]["spearman"], 0.9)
            self.assertGreater(
                diagnostics["holdout_evidence"]["relative_mae_improvement"],
                0.0,
            )
            self.assertEqual(
                len(diagnostics["label_panel"]["pairwise_judge_agreement"]),
                3,
            )
            subspace_rows = read_jsonl(
                output / "readiness_supervised_subspace_coordinates.jsonl"
            )
            self.assertEqual(len(subspace_rows), 40)
            self.assertIn("axis_2", subspace_rows[0])
            exemplars = read_json(
                output / "readiness_axis_exemplars.restricted-local.json"
            )
            self.assertEqual(exemplars["scope"], "restricted-local")

    def test_unknown_compute_backend_is_rejected(self) -> None:
        from analysis.interpretability.pipeline.readiness_embedding_map import (
            fit_readiness_embedding_map,
        )

        with self.assertRaisesRegex(ValueError, "compute_backend"):
            fit_readiness_embedding_map(
                (),
                (),
                np.empty((0, 2)),
                embedding_model="test",
                compute_backend="unknown",
            )

    def test_spearman_ranks_average_ties(self) -> None:
        from analysis.interpretability.pipeline.readiness_embedding_map import _ranks

        ranks = _ranks(np.asarray([3.0, 1.0, 1.0, 2.0]))
        np.testing.assert_array_equal(ranks, np.asarray([3.0, 0.5, 0.5, 2.0]))

    def test_torch_linear_algebra_matches_numpy_on_cpu(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from analysis.interpretability.pipeline.readiness_embedding_map import (
            _randomized_pca,
            _ridge_coefficients,
            _torch_randomized_pca,
            _torch_ridge_coefficients,
        )

        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(30, 6))
        targets = rng.normal(size=(30, 2))
        expected = _ridge_coefficients(matrix, targets, 0.7)
        actual = _torch_ridge_coefficients(
            matrix,
            targets,
            0.7,
            device_name="cpu",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

        numpy_axes, _ = _randomized_pca(
            matrix,
            component_count=3,
            random_seed=11,
        )
        torch_axes, torch_variance = _torch_randomized_pca(
            matrix,
            component_count=3,
            random_seed=11,
            device_name="cpu",
        )
        self.assertEqual(torch_axes.shape, numpy_axes.shape)
        self.assertEqual(torch_variance.shape, (3,))
        np.testing.assert_allclose(torch_axes @ torch_axes.T, np.eye(3), atol=1e-10)


def _item(index: int) -> SemanticReadinessItem:
    text = f"Synthetic readiness request {index}"
    normalized = normalize_semantic_readiness_text(text)
    return SemanticReadinessItem(
        item_id=f"item:{index}",
        source_kind="synthetic-test",
        source_name="source-a" if index % 2 else "source-b",
        source_record_id=str(index),
        text=text,
        text_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        split="development" if index < 30 else "confirmation",
        group_id=f"group:{index % 4}",
        source_url=None,
        author_name=None,
        author_url=None,
        license="test-only",
    )


def _annotation(
    item_id: str,
    slot: str,
    readiness: float,
    offset: int,
    *,
    answer_type: str = "rating",
) -> dict[str, object]:
    rating = answer_type == "rating"
    overall = int(round(100 * readiness))
    likert = max(1, min(7, 1 + int(round(6 * readiness))))
    return {
        "task_id": f"task:{item_id}:{slot}",
        "item_id": item_id,
        "judge_slot": slot,
        "presentation_variant": f"variant-{offset}",
        "answer_type": answer_type,
        "overall_readiness_0_100": overall if rating else None,
        "information_seeking_1_7": 8 - likert if rating else None,
        "evaluation_1_7": likert if rating else None,
        "selection_commitment_1_7": likert if rating else None,
        "action_implementation_1_7": likert if rating else None,
        "category": "mixed" if rating else None,
        "ambiguity_1_7": 1,
        "confidence_0_1": 0.95,
        "brief_reason": "Synthetic test annotation.",
        "raw_response": json.dumps({"answer_type": answer_type}),
    }


if __name__ == "__main__":
    unittest.main()
