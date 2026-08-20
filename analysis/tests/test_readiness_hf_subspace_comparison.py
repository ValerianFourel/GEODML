"""Contracts for cross-embedding semantic-readiness confirmation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_hf_dataset import (
    atomic_json,
    atomic_jsonl,
    atomic_text,
    read_json,
    sha256_file,
)
from analysis.interpretability.pipeline.readiness_hf_subspace_comparison import (
    compare_readiness_hf_subspaces,
)


class ReadinessHfSubspaceComparisonTests(unittest.TestCase):
    def test_rotated_candidate_is_aligned_on_development_and_evaluated_on_confirmation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            angle = 0.7
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            _subspace(reference, "reference-map", np.eye(2), "reference/model")
            _subspace(candidate, "candidate-map", rotation, "candidate/model")

            output = root / "comparison"
            manifest = compare_readiness_hf_subspaces(
                reference_dir=reference,
                candidate_dir=candidate,
                output_dir=output,
                git_commit_sha="c" * 40,
            )

            self.assertEqual(manifest["reference_map_id"], "reference-map")
            comparison = read_json(output / "readiness_subspace_comparison.json")
            self.assertGreater(
                comparison["scalar_prediction_agreement"]["confirmation"]["spearman"],
                0.99,
            )
            self.assertGreater(
                comparison["supervised_subspace_alignment"][
                    "confirmation_flattened"
                ]["pearson"],
                0.99,
            )
            self.assertTrue(
                (output / "readiness_subspace_comparison_report.md").is_file()
            )

    def test_changed_prompt_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            _subspace(reference, "reference-map", np.eye(2), "reference/model")
            _subspace(candidate, "candidate-map", np.eye(2), "candidate/model")
            manifest_path = candidate / "subspace_manifest.json"
            manifest = read_json(manifest_path)
            manifest["inputs"]["prompts"]["sha256"] = "different"
            atomic_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "frozen design"):
                compare_readiness_hf_subspaces(
                    reference_dir=reference,
                    candidate_dir=candidate,
                    output_dir=root / "comparison",
                    git_commit_sha="c" * 40,
                )


def _subspace(
    root: Path,
    map_id: str,
    coordinate_rotation: np.ndarray,
    embedding_model: str,
) -> None:
    root.mkdir()
    scalar_rows = []
    subspace_rows = []
    for index in range(20):
        split = "development" if index < 12 else "confirmation"
        base = np.asarray([index / 19, ((index * 7) % 13) / 12])
        candidate = base @ coordinate_rotation
        scalar_rows.append(
            {
                "evaluation_split": split,
                "item_id": f"item:{index}",
                "observed_readiness_0_1": float(index / 19 + candidate[1] * 0.01),
            }
        )
        subspace_rows.append(
            {
                "split": split,
                "item_id": f"item:{index}",
                "axis_1": float(candidate[0]),
                "axis_2": float(candidate[1]),
            }
        )
    atomic_jsonl(root / "readiness_embedding_coordinates.jsonl", scalar_rows)
    atomic_jsonl(
        root / "readiness_supervised_subspace_coordinates.jsonl",
        subspace_rows,
    )
    evidence = {
        "assessment": {"status": "supportive"},
        "scalar_spearman": 0.7,
        "scalar_r_squared": 0.4,
        "relative_mae_improvement": 0.3,
    }
    atomic_json(
        root / "readiness_embedding_map_diagnostics.json",
        {"holdout_evidence": evidence},
    )
    atomic_text(root / "unused.txt", json.dumps({"map_id": map_id}))
    artifact_names = (
        "readiness_embedding_coordinates.jsonl",
        "readiness_supervised_subspace_coordinates.jsonl",
        "readiness_embedding_map_diagnostics.json",
    )
    atomic_json(
        root / "subspace_manifest.json",
        {
            "map_id": map_id,
            "embedding_model": embedding_model,
            "embedding_dimension": 4,
            "evidence_assessment": evidence["assessment"],
            "inputs": {
                "prompts": {"sha256": "p" * 64},
                "annotations": {"sha256": "a" * 64},
            },
            "judge_slots": ["judge-a", "judge-b", "judge-c"],
            "label_policy": {"minimum_rating_judges": 2},
            "split_policy": "development then confirmation",
            "prompt_count": 20,
            "artifacts": {
                name: {"sha256": sha256_file(root / name)}
                for name in artifact_names
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
