"""Publication transfers sealed final files and leaves inference state alone."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    JsonlShardWriter,
    initialize_dataset,
)
from analysis.scripts.publish_agentic_dataset import build_manifest, enqueue


def _dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    writer = FinalDatasetWriter(root, writer_id="job")
    writer.append(
        "generations",
        {"task_id": "task-1", "answer": "saved"},
        transaction_id="task-1",
    )
    writer.seal()
    (root / "artifacts/raw").mkdir(parents=True)
    (root / "artifacts/raw/evidence.txt").write_text("captured evidence")
    (root / "control/private.txt").write_text("local control state")
    active = JsonlShardWriter(root, table="attempts", writer_id="running")
    active.append({"task_id": "task-2"}, transaction_id="task-2")
    active.close()
    return root


def test_manifest_includes_sealed_shards_and_artifacts_but_not_active_or_control(tmp_path):
    root = _dataset(tmp_path)
    manifest = build_manifest(root)
    files = set(manifest["files"])
    assert "README.md" in files
    assert "contract.json" in files
    assert "artifacts/raw/evidence.txt" in files
    assert any(name.startswith("data/generations/") and name.endswith(".jsonl")
               for name in files)
    assert not any(name.endswith(".inprogress") for name in files)
    assert not any(name.startswith("control/") for name in files)


def test_queue_is_idempotent_and_does_not_change_snapshot_identity(tmp_path):
    root = _dataset(tmp_path)
    first_path, first = enqueue(root, repo_id="ValerianFourel/private-geodml")
    second_path, second = enqueue(root, repo_id="ValerianFourel/private-geodml")
    assert first_path == second_path
    assert first == second
    assert first["required_visibility"] == "private"
    assert build_manifest(root)["snapshot_id"] == first["snapshot_id"]


def test_publication_rejects_credentials_in_allowlisted_files(tmp_path):
    root = _dataset(tmp_path)
    (root / "reports/leak.txt").write_text("hf_abcdefghijklmnopqrstuvwxyz123456")
    with pytest.raises(ValueError, match="credential-shaped"):
        build_manifest(root)


def test_cli_defaults_to_validation_only_without_queue_write(tmp_path):
    root = _dataset(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "analysis/scripts/publish_agentic_dataset.py",
            "--dataset-root", str(root),
            "--repo-id", "ValerianFourel/private-geodml",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["dry_run"] is True
    assert not list((root / "publication").glob("queue-*.json"))
