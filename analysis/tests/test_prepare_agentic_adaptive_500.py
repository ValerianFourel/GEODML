"""The five-node plan freezes the original population before any allocation starts."""

from __future__ import annotations

import hashlib
import json

import pytest

from analysis.scripts import prepare_agentic_adaptive_500 as module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def frozen_study(tmp_path, monkeypatch):
    study = tmp_path / "study"
    prompts = write_jsonl(
        tmp_path / "inputs/prompts.jsonl",
        [
            {
                "candidate_id": f"p{i:03d}",
                "question": f"question {i}",
                "keyword": f"keyword {i}",
            }
            for i in range(500)
        ],
    )
    selection = write_jsonl(
        tmp_path / "inputs/selection.jsonl",
        [{"candidate_id": f"p{i:03d}", "axis_bin": i % 20} for i in range(500)],
    )
    snapshots = {}
    for engine in ("duckduckgo", "searxng"):
        path = tmp_path / f"inputs/{engine}.parquet"
        path.write_bytes(engine.encode())
        snapshots[engine] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    config = {
        "format_version": "agentic-search-execution-calibration-v2",
        "model_id": module.QWEN_MODEL["model_id"],
        "model_revision": module.QWEN_MODEL["model_revision"],
        "prompt_population_count": 500,
        "prompt_shard_count": 4,
        "condition_mode": "frozen-target-url-and-stable-shuffle-v1",
        "retrieval_mode": "frozen-snapshot-deterministic-lexical-v1",
        "prompt_sources": {
            "prompts_jsonl": prompts,
            "selection_records_jsonl": selection,
        },
        "search_snapshots": snapshots,
        "prompt_selection_seed": 20260912,
        "seed": 20260911,
        "cross_encoder_snapshot": str(tmp_path / "cross"),
        "cross_encoder_revision": "c" * 40,
    }
    path = study / "models/qwen38/shard-0/config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        module,
        "collect_progress",
        lambda *args, **kwargs: {
            "models": [
                {"model": "qwen38", "completed": 6000, "failed_cells": 0},
                {"model": "llama4", "completed": 0, "failed_cells": 0},
            ]
        },
    )
    return study


def test_prepare_freezes_exact_budget_barrier_and_6000_cells(frozen_study, tmp_path):
    run = tmp_path / "adaptive"
    plan = module.prepare(
        study_root=frozen_study, run_root=run, source_git_commit="d" * 40
    )
    assert plan["approved_allocation"]["maximum_gpu_hours"] == 140
    assert plan["approved_allocation"]["walltime"] == "07:00:00"
    assert plan["approved_allocation"]["allocation_count"] == 1
    assert plan["approved_allocation"]["node_count"] == 5
    assert plan["judge"]["barrier"].startswith("qwen38=6000/6000")
    assert plan["judge"]["sweep"]["hard_cap_seconds"] == 1800
    tasks = (run / "generation-tasks.jsonl").read_text().splitlines()
    assert len(tasks) == len({json.loads(row)["cell_id"] for row in tasks}) == 6000
    assert json.loads((run / "run_manifest.json").read_text()) == plan


def test_prepare_refuses_changed_frozen_input(frozen_study, tmp_path):
    (frozen_study.parent / "inputs/prompts.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        module.prepare(
            study_root=frozen_study,
            run_root=tmp_path / "adaptive",
            source_git_commit="d" * 40,
        )


def test_prepare_refuses_partial_unreviewed_llama_state(
    frozen_study, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        module,
        "collect_progress",
        lambda *args, **kwargs: {
            "models": [
                {"model": "qwen38", "completed": 6000, "failed_cells": 0},
                {"model": "llama4", "completed": 1, "failed_cells": 0},
            ]
        },
    )
    with pytest.raises(ValueError, match="explicit audit"):
        module.prepare(
            study_root=frozen_study,
            run_root=tmp_path / "adaptive",
            source_git_commit="d" * 40,
        )
