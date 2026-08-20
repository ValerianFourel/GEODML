"""Tests for the two-embedding nonlinear readiness robustness battery."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.interpretability.pipeline.readiness_hf_dataset import (
    atomic_json,
    atomic_jsonl,
    read_json,
    sha256_file,
)
from analysis.interpretability.pipeline.readiness_subspace_battery import (
    run_readiness_subspace_robustness_battery,
)


class ReadinessSubspaceBatteryTests(unittest.TestCase):
    def test_rotated_second_embedding_passes_nonlinear_holdout_battery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "qwen"
            candidate = root / "mistral"
            data = _synthetic_data()
            _write_subspace(reference, data, rotation=np.eye(2), map_id="qwen-map")
            angle = 0.65
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            _write_subspace(candidate, data, rotation=rotation, map_id="mistral-map")

            output = root / "battery"
            manifest = run_readiness_subspace_robustness_battery(
                reference_dir=reference,
                candidate_dir=candidate,
                output_dir=output,
                git_commit_sha="b" * 40,
                bootstrap_replicates=100,
                permutation_replicates=100,
                minimum_source_items_per_split=20,
            )

            self.assertEqual(manifest["reference_map_id"], "qwen-map")
            payload = read_json(output / "readiness_robustness_battery.json")
            self.assertIn(payload["assessment"]["status"], {"supportive", "strongly-supportive"})
            self.assertGreater(
                payload["representation_models"]["reference"]["incremental_tests"][
                    "axis_2_macro_r_squared_gain"
                ],
                0.05,
            )
            self.assertGreater(
                payload["cross_embedding_alignment"]["confirmation_axis_2"][
                    "spearman"
                ],
                0.99,
            )
            self.assertTrue(
                (output / "readiness_robustness_battery_report.md").is_file()
            )

    def test_changed_consensus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "qwen"
            candidate = root / "mistral"
            data = _synthetic_data()
            _write_subspace(reference, data, rotation=np.eye(2), map_id="qwen-map")
            _write_subspace(candidate, data, rotation=np.eye(2), map_id="mistral-map")
            rows = [json.loads(line) for line in (candidate / "readiness_consensus.jsonl").read_text().splitlines()]
            rows[0]["overall_readiness_0_100"] += 1
            atomic_jsonl(candidate / "readiness_consensus.jsonl", rows)

            with self.assertRaisesRegex(ValueError, "consensus artifact hash mismatch"):
                run_readiness_subspace_robustness_battery(
                    reference_dir=reference,
                    candidate_dir=candidate,
                    output_dir=root / "battery",
                    git_commit_sha="b" * 40,
                    bootstrap_replicates=100,
                    permutation_replicates=100,
                )


def _synthetic_data():
    rng = np.random.default_rng(21)
    rows = []
    for index in range(400):
        x1, x2 = rng.uniform(-1.0, 1.0, size=2)
        noise = rng.normal(0.0, 0.015, size=5)
        targets = np.clip(
            np.asarray(
                [
                    0.5 + 0.30 * x1 + 0.03 * x1**3,
                    0.5 + 0.28 * x1 - 0.15 * x2,
                    0.5 + 0.20 * x1 - 0.28 * x2,
                    0.5 + 0.25 * x1 - 0.22 * x2,
                    0.5 + 0.25 * x1 + 0.32 * x2,
                ]
            )
            + noise,
            0.0,
            1.0,
        )
        rows.append(
            {
                "item_id": f"item:{index:04d}",
                "split": "development" if index < 300 else "confirmation",
                "source_name": "source-a" if index % 2 else "source-b",
                "axes": np.asarray([x1, x2]),
                "targets": targets,
            }
        )
    return rows


def _write_subspace(root, data, *, rotation, map_id):
    root.mkdir()
    consensus = []
    scalar = []
    axes = []
    for row in data:
        target = row["targets"]
        transformed = row["axes"] @ rotation
        consensus.append(
            {
                "item_id": row["item_id"],
                "judge_count": 3,
                "overall_readiness_0_100": float(target[0] * 100),
                "information_seeking_1_7": float(7.0 - target[1] * 6.0),
                "evaluation_1_7": float(1.0 + target[2] * 6.0),
                "selection_commitment_1_7": float(1.0 + target[3] * 6.0),
                "action_implementation_1_7": float(1.0 + target[4] * 6.0),
                "not_applicable_vote_fraction": 0.0,
                "ambiguity_mean": 1.0,
                "confidence_mean": 0.9,
                "overall_median_absolute_deviation": 1.0,
                "usable_for_axis": True,
            }
        )
        scalar.append(
            {
                "evaluation_split": row["split"],
                "item_id": row["item_id"],
                "source_name": row["source_name"],
                "observed_readiness_0_1": float(target[0]),
                "consensus_readiness_0_1": float(target[0]),
                "absolute_error": 0.0,
            }
        )
        axes.append(
            {
                "item_id": row["item_id"],
                "split": row["split"],
                "source_name": row["source_name"],
                "axis_1": float(transformed[0]),
                "axis_2": float(transformed[1]),
                "consensus_readiness_0_1": float(target[0]),
                "usable_for_axis": True,
            }
        )
    atomic_jsonl(root / "readiness_consensus.jsonl", consensus)
    atomic_jsonl(root / "readiness_embedding_coordinates.jsonl", scalar)
    atomic_jsonl(root / "readiness_supervised_subspace_coordinates.jsonl", axes)
    evidence = {
        "assessment": {"status": "supportive"},
        "scalar_spearman": 0.9,
        "scalar_r_squared": 0.8,
        "relative_mae_improvement": 0.8,
    }
    atomic_json(
        root / "readiness_embedding_map_diagnostics.json",
        {"holdout_evidence": evidence},
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
            "evidence_assessment": evidence["assessment"],
            "inputs": {
                "prompts": {"sha256": "p" * 64},
                "annotations": {"sha256": "a" * 64},
            },
            "judge_slots": ["judge-a", "judge-b", "judge-c"],
            "label_policy": {"minimum_rating_judges": 2},
            "split_policy": "fit development; freeze confirmation",
            "prompt_count": len(data),
            "artifacts": {
                name: {"sha256": sha256_file(root / name)} for name in artifacts
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
