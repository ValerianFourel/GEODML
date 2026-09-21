"""Task construction remains usable without importing the inference runner."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.interpretability.pipeline import agentic_generation_tasks as tasks
from analysis.scripts import run_agentic_search_integration_smoke as runner


@pytest.mark.parametrize("old_name,new_name", [
    ("CalibrationPrompt", "CalibrationPrompt"), ("SmokeCell", "SmokeCell"),
    ("METHODS", "METHODS"), ("CONDITIONS", "CONDITIONS"), ("ENGINES", "ENGINES"),
    ("_canonical", "_canonical"), ("_selection_key", "_selection_key"),
    ("_read_jsonl_objects", "_read_jsonl_objects"),
    ("_cells", "build_cells"), ("_load_calibration_prompts", "load_calibration_prompts"),
    ("_prompt_shard", "shard_prompts"), ("_select_cells", "select_cells"),
])
def test_runner_retains_identical_imports(old_name, new_name):
    assert getattr(runner, old_name) is getattr(tasks, new_name)


def test_task_module_import_does_not_load_the_runner_or_gpu_libraries():
    subprocess.run([
        sys.executable, "-c",
        (
            "import sys; from analysis.interpretability.pipeline import agentic_generation_tasks; "
            "assert len(agentic_generation_tasks.build_cells()) == 12; "
            "assert 'analysis.scripts.run_agentic_search_integration_smoke' not in sys.modules; "
            "assert 'analysis.scripts.run_acl_arr_vllm' not in sys.modules; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        ),
    ], cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True, text=True)


def test_empty_prompts_preserve_smoke_population_and_method_identity():
    assert tasks.build_cells([]) == tasks.build_cells(None)
    assert {cell.method_class for cell in tasks.build_cells()} == set(tasks.METHODS)


@pytest.mark.parametrize("rows,message", [
    ([{"cell_id": "unknown"}], "unknown cell IDs"),
    ([{}], "lacks cell_id"),
    ([{"cell_id": "repeat"}] * 2, "duplicate cell IDs"),
])
def test_cell_selection_rejects_invalid_ids(tmp_path, rows, message):
    path = tmp_path / "cells.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=message):
        tasks.select_cells(tasks.build_cells(), path)


@pytest.mark.parametrize("mutation,message", [
    ({"question_sha256": "altered"}, "question hash mismatch"),
    ({"axis_bin": True}, "invalid axis bin"),
    ({"keyword": ""}, "no keyword"),
    ({"question": ""}, "no question text"),
])
def test_prompt_selection_rejects_invalid_records(tmp_path, mutation, message):
    row = {"candidate_id": "prompt", "question": "Question?", "keyword": "topic", "axis_bin": 0}
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({**row, **mutation}) + "\n")
    with pytest.raises(ValueError, match=message):
        tasks.load_calibration_prompts(path, path, prompt_count=1, seed=7)


def test_prompt_selection_rejects_duplicate_ids(tmp_path):
    row = {"candidate_id": "prompt", "question": "Question?", "keyword": "topic", "axis_bin": 0}
    path = tmp_path / "prompts.jsonl"
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        tasks.load_calibration_prompts(path, path, prompt_count=1, seed=7)
