"""Storage failures block new allocation admission until explicitly cleared."""

from __future__ import annotations

import errno

from analysis.interpretability.pipeline.agentic_dataset import initialize_dataset
from analysis.interpretability.pipeline.agentic_storage import (
    acknowledge_incidents,
    record_storage_incident,
    storage_health,
)


def test_storage_incident_blocks_until_named_acknowledgment(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    path = record_storage_incident(
        root,
        OSError(errno.ENOSPC, "No space left on device"),
        operation="test_write",
        writer_id="writer",
        table="generations",
    )
    assert path is not None
    blocked = storage_health(root, minimum_free_bytes=0, minimum_free_inodes=0)
    assert blocked["safe_to_admit"] is False
    assert blocked["reasons"] == ["unacknowledged_storage_incident"]
    acknowledge_incidents(root, [path.stem])
    healthy = storage_health(root, minimum_free_bytes=0, minimum_free_inodes=0)
    assert healthy["safe_to_admit"] is True


def test_quota_evidence_must_be_fresh_and_within_limits_when_supplied(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    unknown = storage_health(
        root,
        minimum_free_bytes=0,
        minimum_free_inodes=0,
        quota_evidence={"fresh": False, "within_limits": True},
    )
    assert unknown["safe_to_admit"] is False
    healthy = storage_health(
        root,
        minimum_free_bytes=0,
        minimum_free_inodes=0,
        quota_evidence={"fresh": True, "within_limits": True},
    )
    assert healthy["safe_to_admit"] is True
