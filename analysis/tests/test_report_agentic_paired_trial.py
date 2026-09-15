from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from analysis.scripts.report_agentic_paired_trial import collect_progress


def _trial(root: Path):
    tasks = [{"cell_id": f"cell{i}", "prompt_id": f"prompt{i // 12}",
              "method": "method", "engine": "engine", "condition": str(i % 12)}
             for i in range(1440)]
    (root / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))
    return tasks


def test_counts_models_separately_and_keeps_old_study_out_of_total(tmp_path):
    tasks = _trial(tmp_path)
    output = tmp_path / "models/qwen38/outputs/worker-00000/results"
    output.mkdir(parents=True)
    for row in tasks[:13]:
        (output / (row["cell_id"] + ".json")).write_text(json.dumps(row))
    value = collect_progress(tmp_path, scheduler=False)
    assert value["completed"] == 13
    assert value["expected"] == 2880
    assert value["models"][0]["complete_prompt_groups"] == 1
    assert value["models"][0]["prompts_with_results"] == 2
    assert value["models"][1]["completed"] == 0
    assert value["models"][1]["remaining"] == 1440


def test_rejects_unknown_cells(tmp_path):
    _trial(tmp_path)
    output = tmp_path / "models/qwen38/outputs/worker-00000/results"
    output.mkdir(parents=True)
    (output / "bad.json").write_text('{"cell_id":"bad"}')
    with pytest.raises(ValueError, match="Unexpected result"):
        collect_progress(tmp_path, scheduler=False)


def test_reads_array_task_accounting_and_preserves_timeout(tmp_path, monkeypatch):
    _trial(tmp_path)
    output = tmp_path / "models/qwen38"
    output.mkdir(parents=True)
    (output / "job-id.txt").write_text("123\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(
        a, 0, "123_0|TIMEOUT|01:00:00|01:00:00|0:0\n", ""))
    value = collect_progress(tmp_path)
    assert value["jobs"][0]["state"] == "TIMEOUT"
    assert value["job_error"] is None
    assert value["models"][0]["saved_status"] == "no_manifest_yet"


def test_missing_root_is_not_zero_progress(tmp_path):
    with pytest.raises(ValueError, match="directory not found"):
        collect_progress(tmp_path / "missing", scheduler=False)
