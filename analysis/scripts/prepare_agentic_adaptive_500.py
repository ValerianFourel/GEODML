#!/usr/bin/env python3
"""Freeze the approved five-node adaptive continuation of the original 500 prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_generation_tasks import (
    build_cells,
    load_calibration_prompts,
)
from analysis.interpretability.pipeline.inference_wave import (
    _atomic_json,
    _atomic_jsonl,
)
from analysis.scripts.report_agentic_experiment_progress import (
    collect_progress,
)

FORMAT_VERSION = "agentic-adaptive-500-plan-v1"
LLAMA_MODEL = {
    "slug": "llama4",
    "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "model_revision": "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
    "disable_thinking": False,
    "query_max_tokens": 256,
    "final_max_tokens": 4096,
    "request_concurrency": 4,
    "cell_concurrency": 12,
}
QWEN_MODEL = {
    "slug": "qwen38",
    "model_id": "Qwen/Qwen3.8-27B",
    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
NEMOTRON_MODEL = {
    "slug": "nemotron",
    "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "model_revision": "bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
    "tensor_parallel_size": 4,
    "max_model_len": 73728,
    "max_output_tokens": 2048,
    "dtype": "bfloat16",
    "disable_thinking": True,
    "enforce_eager": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(entry: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(entry.get("path", ""))).resolve()
    expected = entry.get("sha256")
    if not path.is_file() or not isinstance(expected, str) or _sha256(path) != expected:
        raise ValueError(f"{label} is missing or its SHA-256 changed: {path}")
    return {"path": str(path), "sha256": expected}


def _source_config(study_root: Path) -> tuple[Path, dict[str, Any]]:
    path = study_root / "models/qwen38/shard-0/config.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format_version": "agentic-search-execution-calibration-v2",
        "model_id": QWEN_MODEL["model_id"],
        "model_revision": QWEN_MODEL["model_revision"],
        "prompt_population_count": 500,
        "prompt_shard_count": 4,
        "condition_mode": "frozen-target-url-and-stable-shuffle-v1",
        "retrieval_mode": "frozen-snapshot-deterministic-lexical-v1",
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(
            "original Qwen shard-0 config is not the approved 500-prompt design"
        )
    return path, value


def prepare(
    *,
    study_root: Path,
    run_root: Path,
    source_git_commit: str,
    paired_root: Path | None = None,
    backlog_roots: tuple[Path, ...] = (),
    nemotron_source_root: Path | None = None,
) -> dict[str, Any]:
    study_root, run_root = study_root.resolve(), run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite adaptive run: {run_root}")
    if re.fullmatch(r"[0-9a-f]{40}", source_git_commit) is None:
        raise ValueError("source Git commit must be a full SHA")
    progress = collect_progress(
        study_root,
        models=("qwen38", "llama4"),
        shard_count=4,
        cells_per_shard=1500,
        recent_window_minutes=15,
    )
    by_model = {row["model"]: row for row in progress["models"]}
    if by_model["qwen38"]["completed"] != 6000 or by_model["qwen38"]["failed_cells"]:
        raise ValueError(
            "original Qwen generation must be exactly 6000/6000 without failures"
        )
    if by_model["llama4"]["completed"] not in {0, 6000}:
        raise ValueError(
            "partially materialized original Llama output requires an explicit audit"
        )
    config_path, source = _source_config(study_root)
    prompt_source = _verified_file(
        source["prompt_sources"]["prompts_jsonl"], "prompt population"
    )
    selection_source = _verified_file(
        source["prompt_sources"]["selection_records_jsonl"], "selection records"
    )
    snapshots = {
        name: _verified_file(entry, f"{name} search snapshot")
        for name, entry in source["search_snapshots"].items()
    }
    prompts = load_calibration_prompts(
        Path(prompt_source["path"]),
        Path(selection_source["path"]),
        prompt_count=500,
        seed=int(source["prompt_selection_seed"]),
    )
    cells = build_cells(prompts)
    if len(cells) != 6000:
        raise AssertionError(
            "the original prompt population did not produce 6000 cells"
        )
    tasks = [{"cell_id": cell.cell_id, **cell.core} for cell in cells]
    task_path = run_root / "generation-tasks.jsonl"
    claim_root = run_root / "claims/original-llama4"
    judge_root = run_root / "judge-plan"
    llama_profiles = [
        path.resolve()
        for root in backlog_roots
        for path in (root / "profiles/llama4.json",)
        if path.is_file()
    ]
    qwen_profiles = [
        path.resolve()
        for root in backlog_roots
        for path in (root / "profiles/qwen38.json",)
        if path.is_file()
    ]
    nemotron_profiles = (
        sorted(
            {
                path.resolve()
                for pattern in (
                    "profiles/nemotron.json",
                    "profiles/nemotron-judge-pilot.json",
                    "**/serving-profile.json",
                )
                for path in nemotron_source_root.glob(pattern)
                if path.is_file()
            }
        )
        if nemotron_source_root
        else []
    )
    validation_model = None
    validation_source = None
    if nemotron_source_root:
        for candidate in sorted(nemotron_source_root.rglob("run_manifest.json")):
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            bulk, validation = value.get("bulk_model"), value.get("validation_model")
            if (
                isinstance(bulk, dict)
                and isinstance(validation, dict)
                and bulk.get("model_id") == NEMOTRON_MODEL["model_id"]
                and bulk.get("model_revision") == NEMOTRON_MODEL["model_revision"]
                and all(
                    isinstance(validation.get(key), str) and validation[key]
                    for key in ("model_id", "model_revision")
                )
            ):
                validation_model = validation
                validation_source = {
                    "path": str(candidate.resolve()),
                    "sha256": _sha256(candidate),
                }
                break
    paired = None
    if paired_root:
        paired_root = paired_root.resolve()
        paired_tasks = paired_root / "tasks.jsonl"
        paired_output = paired_root / "models/qwen38/outputs/worker-00000"
        paired_config_path = paired_output / "config.json"
        local_profile = paired_root / "profiles/qwen38.json"
        paired_profiles = (
            [local_profile.resolve()] if local_profile.is_file() else qwen_profiles
        )
        if not paired_tasks.is_file() or not paired_config_path.is_file():
            raise ValueError("paired overflow run lacks tasks or Qwen config")
        if len(paired_profiles) != 1:
            raise ValueError(
                "paired overflow requires one colocated or backlog Qwen serving profile"
            )
        paired_profile = paired_profiles[0]
        paired_config = json.loads(paired_config_path.read_text())
        if (
            paired_config.get("model_id"),
            paired_config.get("model_revision"),
            paired_config.get("prompt_count"),
            paired_config.get("cell_count"),
        ) != (
            QWEN_MODEL["model_id"],
            QWEN_MODEL["model_revision"],
            120,
            1440,
        ):
            raise ValueError(
                "paired overflow config differs from the frozen 120-prompt trial"
            )
        paired = {
            "root": str(paired_root),
            "tasks": {
                "path": str(paired_tasks),
                "sha256": _sha256(paired_tasks),
                "rows": sum(
                    bool(line.strip()) for line in paired_tasks.read_text().splitlines()
                ),
            },
            "source_output": str(paired_output),
            "source_config": {
                "path": str(paired_config_path),
                "sha256": _sha256(paired_config_path),
            },
            "serving_profile": str(paired_profile),
            "claim_root": str(run_root / "claims/paired-qwen38"),
            "prompt_sources": paired_config["prompt_sources"],
            "search_snapshots": paired_config["search_snapshots"],
            "cross_encoder_snapshot": paired_config["cross_encoder_snapshot"],
            "cross_encoder_revision": paired_config["cross_encoder_revision"],
            "prompt_count": 120,
            "expected_cells": 1440,
            "prompt_selection_seed": paired_config["prompt_selection_seed"],
        }
    backlogs = []
    for root in backlog_roots:
        root = root.resolve()
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        claim = manifest.get("claim_root")
        if not isinstance(claim, str) or not Path(claim).is_absolute():
            raise ValueError(f"backlog has no absolute recorded claim root: {root}")
        configs = sorted(root.glob("models/*/outputs/attempts/*/config.json"))
        if not configs:
            configs = sorted(root.glob("models/*/outputs/*/config.json"))
        if not configs:
            raise ValueError(f"backlog has no worker config: {root}")
        config_path = configs[-1]
        config = json.loads(config_path.read_text())
        slug = (
            "qwen38"
            if config.get("model_id") == QWEN_MODEL["model_id"]
            else (
                "llama4" if config.get("model_id") == LLAMA_MODEL["model_id"] else None
            )
        )
        profile = root / f"profiles/{slug}.json" if slug else Path()
        tasks_path = root / "tasks.jsonl"
        if (
            slug is None
            or config.get("prompt_count") != 1200
            or config.get("cell_count") != 14400
            or not profile.is_file()
            or not tasks_path.is_file()
        ):
            raise ValueError(
                f"backlog is not a complete supported 1200-prompt queue: {root}"
            )
        backlogs.append(
            {
                "slug": slug,
                "root": str(root),
                "claim_root": claim,
                "source_manifest": {
                    "path": str(manifest_path),
                    "sha256": _sha256(manifest_path),
                },
                "source_config": {
                    "path": str(config_path),
                    "sha256": _sha256(config_path),
                },
                "serving_profile": str(profile),
                "tasks": {
                    "path": str(tasks_path),
                    "sha256": _sha256(tasks_path),
                    "rows": sum(
                        bool(line.strip())
                        for line in tasks_path.read_text().splitlines()
                    ),
                },
                "prompt_sources": config["prompt_sources"],
                "search_snapshots": config["search_snapshots"],
                "cross_encoder_snapshot": config["cross_encoder_snapshot"],
                "cross_encoder_revision": config["cross_encoder_revision"],
                "prompt_selection_seed": config["prompt_selection_seed"],
                "expected_cells": 14400,
            }
        )
    if backlog_roots and [item["slug"] for item in backlogs].count("qwen38") != 1:
        raise ValueError("overflow requires exactly one explicit Qwen 1200 backlog")
    if backlog_roots and [item["slug"] for item in backlogs].count("llama4") != 1:
        raise ValueError("overflow requires exactly one explicit Llama 1200 backlog")
    if backlog_roots and len(llama_profiles) != 1:
        raise ValueError(
            "adaptive original generation requires one Llama serving profile"
        )
    if nemotron_source_root and (
        not nemotron_profiles
        or len({_sha256(path) for path in nemotron_profiles}) != 1
        or validation_model is None
    ):
        raise ValueError(
            "Nemotron source must contain one unique serving profile and a pinned validation model"
        )
    run_root.mkdir(parents=True)
    _atomic_jsonl(task_path, tasks)
    plan = {
        "format_version": FORMAT_VERSION,
        "status": "prepared",
        "scientific_result": False,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": source_git_commit,
        "approved_allocation": {
            "walltime": "07:00:00",
            "allocation_count": 1,
            "node_count": 5,
            "gpus_per_node": 4,
            "cpus_per_node": 32,
            "memory_per_node": "512G",
            "maximum_gpu_hours": 140,
            "estimate": (
                "One five-node interactive allocation for seven hours. Original Llama generation "
                "was estimated at 1-1.5 hours across five nodes; a concurrent Nemotron sweep is "
                "capped at 30 minutes, followed by resumable judging and compatible overflow."
            ),
            "admission_margin_seconds": 120,
            "cleanup_margin_seconds": 45,
        },
        "scheduler": {
            "policy": "adaptive-frozen-backlogs-v1",
            "priority_poll_seconds": 60,
            "minimum_model_residence_seconds": 300,
            "model_switch": "drain-checkpoint-stop-server-confirm-release-v1",
            "queue_exhaustion": "switch_without_repeating_or_inventing_tasks",
        },
        "original_study": {
            "root": str(study_root),
            "qwen_source_config": {
                "path": str(config_path.resolve()),
                "sha256": _sha256(config_path),
            },
            "expected_cells_per_model": 6000,
            "generation_tasks": {
                "path": str(task_path),
                "sha256": _sha256(task_path),
                "rows": len(tasks),
            },
            "prompt_sources": {
                "prompts_jsonl": prompt_source,
                "selection_records_jsonl": selection_source,
            },
            "search_snapshots": snapshots,
            "prompt_selection_seed": source["prompt_selection_seed"],
            "inference_seed": source["seed"],
            "cross_encoder_snapshot": source["cross_encoder_snapshot"],
            "cross_encoder_revision": source["cross_encoder_revision"],
            "qwen": QWEN_MODEL,
            "llama": {
                **LLAMA_MODEL,
                "claim_root": str(claim_root),
                "materialized_root": str(study_root / "models/llama4"),
                "serving_profile": str(llama_profiles[0]) if llama_profiles else None,
            },
        },
        "judge": {
            "barrier": "qwen38=6000/6000 AND llama4=6000/6000 AND failures=0",
            "plan_root": str(judge_root),
            "claim_root": str(run_root / "claims/nemotron"),
            "recorded_conversation": True,
            "bulk_model": NEMOTRON_MODEL,
            "validation_model": validation_model,
            "validation_model_source": validation_source,
            "serving_profile": str(nemotron_profiles[0]) if nemotron_profiles else None,
            "sweep": {
                "concurrencies": [4, 8, 16, 32],
                "measured_tasks": 64,
                "warmup_tasks": 4,
                "hard_cap_seconds": 1800,
                "tie_break": "lower_concurrency",
                "fallback_concurrency": 4,
            },
        },
        "overflow": {
            "paired": paired,
            "backlogs": backlogs,
            "policy": "exact-scientific-identity-only-no-registry-merge",
            "priority": ["paired-qwen-missing-179", "compatible-1200-backlogs"],
        },
    }
    _atomic_json(run_root / "run_manifest.json", plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument("--paired-root", type=Path)
    parser.add_argument("--backlog-root", type=Path, action="append", default=[])
    parser.add_argument("--nemotron-source-root", type=Path)
    args = parser.parse_args()
    plan = prepare(
        study_root=args.study_root,
        run_root=args.run_root,
        source_git_commit=args.source_git_commit,
        paired_root=args.paired_root,
        backlog_roots=tuple(args.backlog_root),
        nemotron_source_root=args.nemotron_source_root,
    )
    print(
        json.dumps(
            {
                "PLAN": str((args.run_root / "run_manifest.json").resolve()),
                "TASKS": plan["original_study"]["generation_tasks"]["rows"],
                "MAXIMUM_GPU_HOURS": plan["approved_allocation"]["maximum_gpu_hours"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
