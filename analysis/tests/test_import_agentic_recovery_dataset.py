"""Recovery imports preserve bytes without promoting recovered slots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis.interpretability.pipeline.agentic_dataset import (
    initialize_dataset,
    iter_sealed_rows,
)
from analysis.scripts.import_agentic_recovery_dataset import import_recovery
from analysis.scripts.publish_agentic_dataset import build_manifest


def _recovery(tmp_path: Path) -> Path:
    recovery = tmp_path / "recovery"
    hub = recovery / "hub"
    data = hub / "data"
    data.mkdir(parents=True)
    summary = {
        "format_version": "geodml-recovery-dataset-v1",
        "status": "recovery_audit_complete_scientific_acceptance_pending",
        "stages": [{"model": "qwen38", "recovered_slots": 2}],
        "findings": {"unbound_or_invalid_judgment": 697},
        "capture": {"snapshot_sha256": "a" * 64},
        "limitations": ["Recovered payloads still need protocol acceptance."],
    }
    (hub / "summary.json").write_text(json.dumps(summary) + "\n")
    (data / "generations.jsonl.gz").write_bytes(b"normalized-generation-bytes")
    files = {
        str(path.relative_to(hub)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(hub.rglob("*"))
        if path.is_file()
    }
    (hub / "publication-manifest.json").write_text(json.dumps({
        "format_version": "geodml-recovery-publication-v1",
        "files": files,
        "state": "validated_recovery_snapshot",
        "git_commit": "b" * 40,
    }) + "\n")
    return recovery


def test_recovery_import_is_resumable_publishable_and_does_not_complete_tasks(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    recovery = _recovery(tmp_path)
    first = import_recovery(
        dataset_root=root, recovery_dataset=recovery, copy_mode="copy"
    )
    second = import_recovery(
        dataset_root=root, recovery_dataset=recovery, copy_mode="copy"
    )
    assert second == first
    records = list(iter_sealed_rows(root, "legacy_imports", required=True))
    assert len(records) == 1
    assert records[0]["findings"] == {"unbound_or_invalid_judgment": 697}
    assert records[0]["ledger_completions_created"] == 0
    assert records[0]["scientific_completion"] == "unverified"
    assert records[0]["local_forensics"]["included"] is False
    imported = root / records[0]["artifact_root"] / "data/generations.jsonl.gz"
    assert imported.read_bytes() == b"normalized-generation-bytes"
    publication = build_manifest(root)
    assert str(imported.relative_to(root)) in publication["files"]
