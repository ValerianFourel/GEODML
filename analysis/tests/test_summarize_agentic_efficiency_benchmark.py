from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.scripts.summarize_agentic_efficiency_benchmark import (
    compare,
    summarize_trial,
)


def _trial(
    root: Path,
    *,
    concurrency: int,
    elapsed: int,
    answer: str = "same",
) -> Path:
    path = root / "model" / f"concurrency-{concurrency}"
    output = path / "output"
    (output / "results").mkdir(parents=True)
    (output / "diagnostics").mkdir()
    (path / "model_id.txt").write_text("test/model\n")
    (path / "concurrency.txt").write_text(f"{concurrency}\n")
    (path / "allocated_gpus.txt").write_text("4\n")
    (path / "elapsed_seconds.txt").write_text(f"{elapsed}\n")
    (path / "step_status.txt").write_text("0\n")
    (output / "run_manifest.json").write_text(json.dumps({
        "status": "complete",
        "completed_count": 12,
        "remaining_count": 0,
    }))
    for index in range(12):
        record = {
            "cell_id": str(index),
            "method": "method",
            "engine": "engine",
            "condition": "natural",
            "ranking": ["https://example.test"],
            "answer": answer,
            "final_snippet_count": 3,
            "search_count": 1,
        }
        (output / "results" / f"{index}.json").write_text(json.dumps(record))
        (output / "diagnostics" / f"{index}.json").write_text(json.dumps({
            "status": "complete",
            "elapsed_seconds": 2.0,
            "llm_calls": [{
                "request_seconds": 1.0,
                "queue_seconds": 0.25,
                "error": None,
                "usage": {"total_tokens": 10},
            }],
        }))
    with (path / "gpu.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for gpu in range(4):
            writer.writerow(["time", gpu, "GH200", 1000, 98304, 50, 200])
    return path


def test_summary_and_comparison_promote_faster_equivalent_trial(tmp_path: Path) -> None:
    serial = summarize_trial(_trial(tmp_path, concurrency=1, elapsed=120))
    concurrent = summarize_trial(_trial(tmp_path, concurrency=4, elapsed=60))

    assert serial["completed_count"] == 12
    assert serial["llm_calls"] == 12
    assert serial["total_tokens"] == 120
    assert serial["telemetry_gpu_count"] == 4
    assert serial["gpu_utilization_mean_percent"] == 50.0

    result = compare([serial, concurrent])
    assert result == [{
        "model_id": "test/model",
        "semantic_outputs_equal": True,
        "end_to_end_speedup": 2.0,
        "valid_cells_per_gpu_hour_change": 1.0,
        "verdict": "promote-concurrency-4",
    }]


def test_comparison_rejects_output_mismatch(tmp_path: Path) -> None:
    serial = summarize_trial(_trial(tmp_path, concurrency=1, elapsed=120))
    concurrent = summarize_trial(
        _trial(tmp_path, concurrency=4, elapsed=60, answer="different")
    )

    result = compare([serial, concurrent])
    assert result[0]["verdict"] == "inconclusive-output-mismatch"
