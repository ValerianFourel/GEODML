"""Tests for the agentic search cluster-readiness audit."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.verify_agentic_search_cluster_readiness import (
    EXPECTED_CONDITIONS,
    EXPECTED_ENGINES,
    EXPECTED_METHODS,
    EXPECTED_MODELS,
    ReadinessInputs,
    audit_readiness,
    build_experiment_matrix,
)


class AgenticSearchClusterReadinessTests(unittest.TestCase):
    def test_matrix_contains_all_48_unique_cells(self) -> None:
        matrix = build_experiment_matrix()

        self.assertEqual(len(matrix), 48)
        self.assertEqual(len({row["cell_id"] for row in matrix}), 48)
        self.assertEqual({row["method"] for row in matrix}, set(EXPECTED_METHODS))
        self.assertEqual({row["model_id"] for row in matrix}, {
            model.model_id for model in EXPECTED_MODELS
        })
        self.assertEqual({row["condition"] for row in matrix}, set(EXPECTED_CONDITIONS))
        self.assertEqual({row["search_engine"] for row in matrix}, set(EXPECTED_ENGINES))

    def test_complete_offline_fixture_passes_without_loading_weights(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            locks = []
            for index, model in enumerate(EXPECTED_MODELS):
                snapshot = root / f"model-{index}"
                _write_snapshot(snapshot)
                locks.append({
                    "model_id": model.model_id,
                    "revision": model.revision,
                    "snapshot": str(snapshot),
                })
            lock_path = root / "model-snapshots.json"
            lock_path.write_text(json.dumps({"models": locks}), encoding="utf-8")

            cross_encoder = root / ("e" * 40)
            _write_snapshot(cross_encoder)
            search_snapshots = {}
            for engine in EXPECTED_ENGINES:
                path = root / f"{engine}.jsonl"
                path.write_text(json.dumps({
                    "keyword": "test query",
                    "position": 1,
                    "title": "Result",
                    "url": "https://example.test/result",
                    "snippet": "Evidence",
                }) + "\n", encoding="utf-8")
                search_snapshots[engine] = path

            result = audit_readiness(ReadinessInputs(
                repository=root,
                model_snapshots=lock_path,
                cross_encoder_snapshot=cross_encoder,
                search_snapshots=search_snapshots,
                check_runtime_packages=False,
                require_clean_git=False,
            ))

        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["scientific_result"])
        self.assertEqual(result["matrix_cell_count"], 48)
        self.assertTrue(all(check["status"] == "PASS" for check in result["checks"]))

    def test_missing_engine_snapshot_fails_the_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "model-snapshots.json"
            locks = []
            for index, model in enumerate(EXPECTED_MODELS):
                snapshot = root / f"model-{index}"
                _write_snapshot(snapshot)
                locks.append({
                    "model_id": model.model_id,
                    "revision": model.revision,
                    "snapshot": str(snapshot),
                })
            lock_path.write_text(json.dumps({"models": locks}), encoding="utf-8")
            cross_encoder = root / ("e" * 40)
            _write_snapshot(cross_encoder)
            searxng = root / "searxng.jsonl"
            searxng.write_text("{}\n", encoding="utf-8")

            result = audit_readiness(ReadinessInputs(
                repository=root,
                model_snapshots=lock_path,
                cross_encoder_snapshot=cross_encoder,
                search_snapshots={"searxng": searxng},
                check_runtime_packages=False,
                require_clean_git=False,
            ))

        self.assertEqual(result["status"], "FAIL")
        engines = next(check for check in result["checks"] if check["name"] == "search_snapshots")
        self.assertIn("duckduckgo", engines["errors"][0])


def _write_snapshot(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["TestModel"]}) + "\n",
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


if __name__ == "__main__":
    unittest.main()
