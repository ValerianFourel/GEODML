"""Nemotron sweep decisions require comparable, complete measurements."""

from __future__ import annotations

import json

import pytest

from analysis.scripts.manage_agentic_nemotron_sweep import select


def test_selection_waits_for_valid_results_and_is_immutable(tmp_path):
    root = tmp_path / "sweep"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({"sample_sha256": "a" * 64}))
    shared = {
        "sample_sha256": "a" * 64,
        "profile_sha256": "b" * 64,
        "status": "complete",
        "warmup_succeeded": 4,
        "succeeded": 64,
        "failed": 0,
    }
    for concurrency, seconds in ((4, 40), (8, 20), (16, 20), (32, 80)):
        path = root / f"concurrency-{concurrency}/result.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {**shared, "concurrency": concurrency, "measurement_seconds": seconds}
            )
        )
    decision = select(root, "b" * 64)
    assert decision["concurrency"] == 8
    assert decision["complete_candidate_set"] is True
    assert select(root, "b" * 64) == decision


def test_selection_rejects_mixed_serving_profile(tmp_path):
    root = tmp_path / "sweep"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({"sample_sha256": "a" * 64}))
    path = root / "concurrency-4/result.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "sample_sha256": "a" * 64,
                "profile_sha256": "wrong",
                "status": "complete",
                "warmup_succeeded": 4,
                "succeeded": 64,
                "failed": 0,
                "concurrency": 4,
                "measurement_seconds": 1,
            }
        )
    )
    with pytest.raises(ValueError, match="same sample and serving profile"):
        select(root, "b" * 64)
