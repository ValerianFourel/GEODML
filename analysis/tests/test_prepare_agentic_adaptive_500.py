"""The five-node plan freezes the original population before any allocation starts."""

from __future__ import annotations

import hashlib
import json

import pytest

from analysis.scripts import prepare_agentic_adaptive_500 as module
from analysis.scripts.search_vllm_stage import build_profile


def nemotron_source(tmp_path):
    root = tmp_path / "nemotron-pilot"
    attempt = root / "attempts/job1861355-worker0"
    attempt.mkdir(parents=True)
    profile = build_profile(
        stage="nemotron-judge-pilot",
        model_id=module.NEMOTRON_MODEL["model_id"],
        model_revision=module.NEMOTRON_MODEL["model_revision"],
        vllm_executable="/environment/bin/vllm",
        vllm_version="0.28.0",
        vllm_help="--enforce-eager",
        visible_gpus=[
            {"index": i, "uuid": f"GPU-{i}", "name": "GH200", "memory_total_mib": 96000}
            for i in range(4)
        ],
        cuda_visible_devices="0,1,2,3",
        expected_gpu_name_pattern="GH200",
        max_model_len=73728,
        enforce_eager=True,
    )
    (attempt / "serving-profile.json").write_text(json.dumps(profile))
    # The actual pilot writer has a bulk model and no validation_model field.
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "pilot_only": True,
                "bulk_model": {"role": "bulk", **module.NEMOTRON_MODEL},
                "summary": {"bulk_task_count": 696, "validation_task_count": 0},
            }
        )
    )
    return root


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_backlog(tmp_path, slug, model_id):
    root = tmp_path / f"backlog-{slug}"
    claim_root = root / "claims"
    claim_root.mkdir(parents=True)
    (root / "run_manifest.json").write_text(
        json.dumps({"claim_root": str(claim_root.resolve())})
    )
    write_jsonl(root / "tasks.jsonl", [{"cell_id": "one"}])
    profile = root / f"profiles/{slug}.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{}")
    config = root / f"models/{slug}/outputs/attempts/job1/config.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "prompt_count": 1200,
                "cell_count": 14400,
                "prompt_sources": {},
                "search_snapshots": {},
                "cross_encoder_snapshot": "cross",
                "cross_encoder_revision": "c" * 40,
                "prompt_selection_seed": 20260912,
            }
        )
    )
    return root, profile


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


def test_prepare_uses_qwen_backlog_profile_for_older_paired_run(frozen_study, tmp_path):
    qwen, qwen_profile = write_backlog(
        tmp_path, "qwen38", module.QWEN_MODEL["model_id"]
    )
    llama, _ = write_backlog(tmp_path, "llama4", module.LLAMA_MODEL["model_id"])
    paired = tmp_path / "paired"
    write_jsonl(paired / "tasks.jsonl", [{"cell_id": "paired"}])
    paired_config = paired / "models/qwen38/outputs/worker-00000/config.json"
    paired_config.parent.mkdir(parents=True)
    paired_config.write_text(
        json.dumps(
            {
                "model_id": module.QWEN_MODEL["model_id"],
                "model_revision": module.QWEN_MODEL["model_revision"],
                "prompt_count": 120,
                "cell_count": 1440,
                "prompt_sources": {},
                "search_snapshots": {},
                "cross_encoder_snapshot": "cross",
                "cross_encoder_revision": "c" * 40,
                "prompt_selection_seed": 20260912,
            }
        )
    )

    plan = module.prepare(
        study_root=frozen_study,
        run_root=tmp_path / "adaptive",
        source_git_commit="d" * 40,
        paired_root=paired,
        backlog_roots=(llama, qwen),
    )

    assert plan["overflow"]["paired"]["serving_profile"] == str(qwen_profile.resolve())


def test_failed_paired_validation_does_not_create_partial_run(frozen_study, tmp_path):
    output = tmp_path / "adaptive"
    with pytest.raises(ValueError, match="lacks tasks or Qwen config"):
        module.prepare(
            study_root=frozen_study,
            run_root=output,
            source_git_commit="d" * 40,
            paired_root=tmp_path / "missing-paired",
        )
    assert not output.exists()


def test_bulk_only_nemotron_pilot_can_prepare_adaptive_plan(frozen_study, tmp_path):
    plan = module.prepare(
        study_root=frozen_study,
        run_root=tmp_path / "adaptive",
        source_git_commit="d" * 40,
        nemotron_source_root=nemotron_source(tmp_path),
    )
    assert plan["judge"]["validation_model"] is None
    assert plan["judge"]["validation_status"] == "not_configured"
    assert plan["judge"]["serving_profile"].endswith(
        "job1861355-worker0/serving-profile.json"
    )


def test_missing_nemotron_profile_reports_specific_error(frozen_study, tmp_path):
    with pytest.raises(ValueError, match="Nemotron serving profile not found"):
        module.prepare(
            study_root=frozen_study,
            run_root=tmp_path / "adaptive",
            source_git_commit="d" * 40,
            nemotron_source_root=tmp_path / "missing",
        )
    assert not (tmp_path / "adaptive").exists()
