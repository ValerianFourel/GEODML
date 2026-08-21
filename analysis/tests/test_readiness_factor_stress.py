"""Tests for publication-safe readiness factor stress diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.interpretability.pipeline.readiness_factor_stress import (
    _publication_safe_item_ids,
    run_readiness_factor_stress,
)
from analysis.interpretability.pipeline.readiness_hf_dataset import (
    atomic_json,
    atomic_jsonl,
    read_json,
    sha256_file,
)


class ReadinessFactorStressTests(unittest.TestCase):
    def test_two_factor_structure_replicates_on_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset"
            dataset.mkdir()
            reference = root / "qwen"
            candidate = root / "mistral"
            rows = _synthetic_rows()
            _write_subspace(reference, rows, np.eye(2), "qwen-map")
            angle = 0.55
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            _write_subspace(candidate, rows, rotation, "mistral-map")
            safe_ids = frozenset(row["item_id"] for row in rows)
            dataset_manifest = {
                "included_prompt_count": len(safe_ids),
                "publication_safe": True,
            }
            atomic_json(dataset / "dataset_manifest.json", dataset_manifest)

            output = root / "factor-stress"
            with patch(
                "analysis.interpretability.pipeline.readiness_factor_stress."
                "_publication_safe_item_ids",
                return_value=(safe_ids, dataset_manifest),
            ):
                manifest = run_readiness_factor_stress(
                    dataset_dir=dataset,
                    reference_dir=reference,
                    candidate_dir=candidate,
                    output_dir=output,
                    git_commit_sha="c" * 40,
                    parallel_replicates=100,
                    bootstrap_replicates=100,
                    random_seed=37,
                )

            payload = read_json(output / "readiness_factor_stress.json")
            self.assertEqual(payload["parallel_analysis"]["retained_factor_count"], 2)
            self.assertLess(
                payload["factor_models"]["two_factor"][
                    "confirmation_off_diagonal_rmse"
                ],
                payload["factor_models"]["one_factor"][
                    "confirmation_off_diagonal_rmse"
                ],
            )
            self.assertGreater(
                min(payload["confirmation_replication"]["two_factor_congruence"]),
                0.90,
            )
            self.assertGreater(
                payload["frozen_axis_association"]["reference"]["two"][
                    "macro_r_squared"
                ],
                payload["frozen_axis_association"]["reference"]["one"][
                    "macro_r_squared"
                ],
            )
            self.assertEqual(
                manifest["assessment"]["status"], "descriptively-supportive"
            )
            self.assertTrue((output / "readiness_factor_stress_report.md").is_file())

    def test_dataset_must_be_marked_publication_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            atomic_json(
                root / "dataset_manifest.json",
                {"included_prompt_count": 1, "publication_safe": False},
            )
            with self.assertRaisesRegex(ValueError, "not marked publication-safe"):
                _publication_safe_item_ids(root)

    def test_changed_consensus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset"
            dataset.mkdir()
            rows = _synthetic_rows()[:200]
            reference = root / "qwen"
            candidate = root / "mistral"
            _write_subspace(reference, rows, np.eye(2), "qwen-map")
            _write_subspace(candidate, rows, np.eye(2), "mistral-map")
            consensus = [
                json.loads(line)
                for line in (candidate / "readiness_consensus.jsonl")
                .read_text()
                .splitlines()
            ]
            consensus[0]["evaluation_1_7"] += 0.1
            atomic_jsonl(candidate / "readiness_consensus.jsonl", consensus)
            safe_ids = frozenset(row["item_id"] for row in rows)
            dataset_manifest = {
                "included_prompt_count": len(safe_ids),
                "publication_safe": True,
            }
            atomic_json(dataset / "dataset_manifest.json", dataset_manifest)

            with patch(
                "analysis.interpretability.pipeline.readiness_factor_stress."
                "_publication_safe_item_ids",
                return_value=(safe_ids, dataset_manifest),
            ):
                with self.assertRaisesRegex(
                    ValueError, "consensus artifact hash mismatch"
                ):
                    run_readiness_factor_stress(
                        dataset_dir=dataset,
                        reference_dir=reference,
                        candidate_dir=candidate,
                        output_dir=root / "factor-stress",
                        git_commit_sha="c" * 40,
                        parallel_replicates=100,
                        bootstrap_replicates=100,
                    )


def _synthetic_rows():
    rng = np.random.default_rng(20260822)
    loadings = np.asarray(
        [
            [0.78, -0.32],
            [0.74, -0.55],
            [0.80, 0.42],
            [0.72, 0.61],
        ]
    )
    rows = []
    for index in range(850):
        factors = rng.normal(size=2)
        residual_scale = np.sqrt(1.0 - np.sum(loadings**2, axis=1))
        rubrics = factors @ loadings.T + rng.normal(size=4) * residual_scale
        rubrics = 1.0 / (1.0 + np.exp(-rubrics))
        overall = np.clip(
            0.5 + 0.18 * factors[0] + 0.04 * factors[1] + rng.normal(0.0, 0.03),
            0.0,
            1.0,
        )
        rows.append(
            {
                "item_id": f"item:{index:04d}",
                "split": "development" if index < 600 else "confirmation",
                "source_name": "source-a" if index % 2 else "source-b",
                "factors": factors,
                "rubrics": rubrics,
                "overall": overall,
            }
        )
    return rows


def _write_subspace(root, rows, rotation, map_id):
    root.mkdir()
    consensus = []
    coordinates = []
    scalar_coordinates = []
    for row in rows:
        rubrics = row["rubrics"]
        axes = row["factors"] @ rotation + np.asarray([0.01, -0.01])
        consensus.append(
            {
                "item_id": row["item_id"],
                "judge_count": 3,
                "overall_readiness_0_100": float(row["overall"] * 100.0),
                "information_seeking_1_7": float(7.0 - rubrics[0] * 6.0),
                "evaluation_1_7": float(1.0 + rubrics[1] * 6.0),
                "selection_commitment_1_7": float(1.0 + rubrics[2] * 6.0),
                "action_implementation_1_7": float(1.0 + rubrics[3] * 6.0),
                "not_applicable_vote_fraction": 0.0,
                "ambiguity_mean": 1.0,
                "confidence_mean": 0.9,
                "overall_median_absolute_deviation": 1.0,
                "usable_for_axis": True,
            }
        )
        coordinates.append(
            {
                "item_id": row["item_id"],
                "split": row["split"],
                "source_name": row["source_name"],
                "axis_1": float(axes[0]),
                "axis_2": float(axes[1]),
                "consensus_readiness_0_1": float(row["overall"]),
                "usable_for_axis": True,
            }
        )
        scalar_coordinates.append(
            {
                "evaluation_split": row["split"],
                "item_id": row["item_id"],
                "source_name": row["source_name"],
                "observed_readiness_0_1": float(row["overall"]),
                "consensus_readiness_0_1": float(row["overall"]),
                "absolute_error": 0.0,
            }
        )
    atomic_jsonl(root / "readiness_consensus.jsonl", consensus)
    atomic_jsonl(
        root / "readiness_supervised_subspace_coordinates.jsonl",
        coordinates,
    )
    atomic_jsonl(
        root / "readiness_embedding_coordinates.jsonl",
        scalar_coordinates,
    )
    atomic_json(
        root / "readiness_embedding_map_diagnostics.json",
        {"holdout_evidence": {"assessment": {"status": "supportive"}}},
    )
    artifacts = (
        "readiness_consensus.jsonl",
        "readiness_embedding_coordinates.jsonl",
        "readiness_supervised_subspace_coordinates.jsonl",
        "readiness_embedding_map_diagnostics.json",
    )
    atomic_json(
        root / "subspace_manifest.json",
        {
            "map_id": map_id,
            "embedding_model": f"synthetic/{map_id}",
            "embedding_dimension": 8,
            "evidence_assessment": {"status": "supportive"},
            "inputs": {
                "prompts": {"sha256": "p" * 64},
                "annotations": {"sha256": "a" * 64},
            },
            "judge_slots": ["judge-a", "judge-b", "judge-c"],
            "label_policy": {"minimum_rating_judges": 2},
            "split_policy": "fit development; freeze confirmation",
            "prompt_count": len(rows),
            "artifacts": {
                name: {"sha256": sha256_file(root / name)} for name in artifacts
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
