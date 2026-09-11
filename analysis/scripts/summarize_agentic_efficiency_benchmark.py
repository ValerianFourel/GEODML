#!/usr/bin/env python3
"""Summarize paired real-GPU agentic integration efficiency trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


def _read_scalar(path: Path, cast: type = str) -> Any:
    return cast(path.read_text(encoding="utf-8").strip())


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return round(ordered[index], 6)


def _semantic_hash(output: Path) -> str | None:
    paths = sorted((output / "results").glob("*.json"))
    if not paths:
        return None
    records = []
    fields = (
        "cell_id",
        "method",
        "engine",
        "condition",
        "ranking",
        "answer",
        "final_snippet_count",
        "search_count",
    )
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        records.append({field: value.get(field) for field in fields})
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _diagnostics(output: Path) -> dict[str, Any]:
    cell_seconds: list[float] = []
    request_seconds: list[float] = []
    queue_seconds: list[float] = []
    total_tokens = 0
    llm_calls = 0
    failed_calls = 0
    completed_cells = 0
    for path in sorted((output / "diagnostics").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "complete":
            completed_cells += 1
        cell_seconds.append(float(value.get("elapsed_seconds", 0.0)))
        for call in value.get("llm_calls", []):
            llm_calls += 1
            request_seconds.append(float(call.get("request_seconds", 0.0)))
            queue_seconds.append(float(call.get("queue_seconds", 0.0)))
            if call.get("error") is not None:
                failed_calls += 1
            usage = call.get("usage") or {}
            total_tokens += int(usage.get("total_tokens") or 0)
    return {
        "completed_diagnostic_cells": completed_cells,
        "llm_calls": llm_calls,
        "failed_llm_calls": failed_calls,
        "total_tokens": total_tokens,
        "cell_seconds_sum": round(sum(cell_seconds), 6),
        "cell_seconds_p50": _percentile(cell_seconds, 0.50),
        "cell_seconds_p95": _percentile(cell_seconds, 0.95),
        "request_seconds_sum": round(sum(request_seconds), 6),
        "request_seconds_p50": _percentile(request_seconds, 0.50),
        "request_seconds_p95": _percentile(request_seconds, 0.95),
        "queue_seconds_sum": round(sum(queue_seconds), 6),
        "queue_seconds_p50": _percentile(queue_seconds, 0.50),
        "queue_seconds_p95": _percentile(queue_seconds, 0.95),
    }


def _telemetry(path: Path) -> dict[str, Any]:
    utilization: list[float] = []
    power: list[float] = []
    memory: list[float] = []
    gpu_indices: set[int] = set()
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) < 7:
                    continue
                try:
                    gpu_indices.add(int(row[1].strip()))
                    memory.append(float(row[3].strip()))
                    utilization.append(float(row[5].strip()))
                    power.append(float(row[6].strip()))
                except ValueError:
                    continue
    return {
        "telemetry_samples": len(utilization),
        "telemetry_gpu_count": len(gpu_indices),
        "gpu_utilization_mean_percent": (
            round(fmean(utilization), 3) if utilization else None
        ),
        "gpu_utilization_peak_percent": max(utilization, default=None),
        "gpu_memory_peak_mib": max(memory, default=None),
        "gpu_power_mean_w": round(fmean(power), 3) if power else None,
        "gpu_power_peak_w": max(power, default=None),
    }


def summarize_trial(path: Path) -> dict[str, Any]:
    output = path / "output"
    elapsed_seconds = _read_scalar(path / "elapsed_seconds.txt", float)
    step_status = _read_scalar(path / "step_status.txt", int)
    concurrency = _read_scalar(path / "concurrency.txt", int)
    model_id = _read_scalar(path / "model_id.txt")
    manifest_path = output / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    completed_count = int(manifest.get("completed_count", 0))
    allocated_gpus = _read_scalar(path / "allocated_gpus.txt", int)
    gpu_hours = elapsed_seconds * allocated_gpus / 3600.0
    return {
        "model_id": model_id,
        "request_concurrency": concurrency,
        "step_status": step_status,
        "run_status": manifest.get("status", "missing"),
        "completed_count": completed_count,
        "remaining_count": manifest.get("remaining_count"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "allocated_gpus": allocated_gpus,
        "gpu_hours": round(gpu_hours, 6),
        "valid_cells_per_second": round(
            completed_count / elapsed_seconds if elapsed_seconds else 0.0, 6
        ),
        "valid_cells_per_gpu_hour": round(
            completed_count / gpu_hours if gpu_hours else 0.0, 6
        ),
        "semantic_result_sha256": _semantic_hash(output),
        **_diagnostics(output),
        **_telemetry(path / "gpu.csv"),
    }


def compare(trials: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[int, dict[str, Any]]] = {}
    for trial in trials:
        by_model.setdefault(trial["model_id"], {})[
            trial["request_concurrency"]
        ] = trial
    comparisons = []
    for model_id, values in sorted(by_model.items()):
        serial = values.get(1)
        concurrent = values.get(4)
        verdict = "incomplete"
        speedup = None
        efficiency_change = None
        equivalent = False
        if serial and concurrent:
            equivalent = (
                serial["semantic_result_sha256"] is not None
                and serial["semantic_result_sha256"]
                == concurrent["semantic_result_sha256"]
            )
            complete = all(
                trial["step_status"] == 0
                and trial["run_status"] == "complete"
                and trial["completed_count"] == 12
                and trial["failed_llm_calls"] == 0
                for trial in (serial, concurrent)
            )
            if serial["elapsed_seconds"] and concurrent["elapsed_seconds"]:
                speedup = round(
                    serial["elapsed_seconds"] / concurrent["elapsed_seconds"], 3
                )
            base = serial["valid_cells_per_gpu_hour"]
            if base:
                efficiency_change = round(
                    concurrent["valid_cells_per_gpu_hour"] / base - 1.0, 4
                )
            if complete and equivalent:
                verdict = (
                    "promote-concurrency-4"
                    if efficiency_change is not None and efficiency_change > 0.05
                    else "keep-concurrency-1"
                )
            elif complete:
                verdict = "inconclusive-output-mismatch"
        comparisons.append({
            "model_id": model_id,
            "semantic_outputs_equal": equivalent,
            "end_to_end_speedup": speedup,
            "valid_cells_per_gpu_hour_change": efficiency_change,
            "verdict": verdict,
        })
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    trial_paths = sorted(arguments.benchmark_root.glob("*/concurrency-*"))
    trials = [
        summarize_trial(path)
        for path in trial_paths
        if (path / "step_status.txt").is_file()
    ]
    comparisons = compare(trials)
    result = {
        "format_version": "agentic-model-efficiency-benchmark-v1",
        "scientific_result": False,
        "benchmark_root": str(arguments.benchmark_root.resolve()),
        "trial_count": len(trials),
        "trials": trials,
        "comparisons": comparisons,
        "status": (
            "complete"
            if len(trials) == 8
            and len(comparisons) == 4
            and all(row["verdict"] not in {
                "incomplete", "inconclusive-output-mismatch"
            } for row in comparisons)
            else "incomplete"
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(arguments.output)
    for trial in trials:
        print("EFFICIENCY_TRIAL=" + json.dumps(trial, sort_keys=True))
    for comparison in comparisons:
        print("EFFICIENCY_COMPARISON=" + json.dumps(comparison, sort_keys=True))
    print("EFFICIENCY_BENCHMARK=" + json.dumps({
        "status": result["status"],
        "trial_count": result["trial_count"],
        "scientific_result": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
