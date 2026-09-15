"""CPU contracts for a shared new-prompt cohort with no pilot overlap."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from analysis.scripts.run_agentic_search_integration_smoke import (
    _cells,
    _load_calibration_prompts,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "analysis/scripts/prepare_agentic_new_cohort.py"
COMMIT = "a" * 40


def _write_rows(path: Path, rows: list[dict]) -> dict:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
    }


def _fixture(root: Path) -> Path:
    prompts = []
    axes = []
    for keyword in range(5):
        for axis_bin in range(4):
            for copy in range(10):
                identity = f"p-{keyword}-{axis_bin}-{copy}"
                question = f"Question {identity}?"
                digest = hashlib.sha256(question.encode()).hexdigest()
                percentile = (axis_bin + (copy + 1) / 11) / 4
                prompts.append(
                    {
                        "candidate_id": identity,
                        "question": question,
                        "question_sha256": digest,
                        "keyword": f"topic {keyword}",
                        "keyword_id": str(keyword),
                        "target_normalized_axis_1": percentile,
                    }
                )
                axes.append(
                    {
                        "candidate_id": identity,
                        "text_sha256": digest,
                        "axis_1_percentile_0_1": percentile,
                    }
                )
    sources = {
        "prompts": _write_rows(root / "population.jsonl", prompts),
        "axis_map": _write_rows(root / "axes.jsonl", axes),
    }
    selection = root / "original"
    selection.mkdir()
    old_prompts = prompts[::10]
    old_axes = axes[::10]
    artifacts = {
        "prompts": _write_rows(selection / "pilot-prompts.jsonl", old_prompts),
        "axis_map": _write_rows(selection / "pilot-axis.jsonl", old_axes),
        "selection_records": _write_rows(
            selection / "selection-records.jsonl",
            [
                {"candidate_id": row["candidate_id"], "axis_bin": index % 4}
                for index, row in enumerate(old_prompts)
            ],
        ),
    }
    for artifact in artifacts.values():
        artifact["path"] = Path(artifact["path"]).name
    (selection / "selection-manifest.json").write_text(
        json.dumps(
            {
                "format_version": "readiness-axis-balanced-pilot-v1",
                "sources": sources,
                "artifacts": artifacts,
                "diagnostics": {"axis_bins": 4},
            }
        )
    )
    return selection


def _run(selection: Path, output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection-root",
            str(selection),
            "--prompt-count",
            "20",
            "--axis-bins",
            "4",
            "--expected-population-count",
            "200",
            "--expected-excluded-count",
            "20",
            "--master-seed",
            "20260916",
            "--source-git-commit",
            COMMIT,
            "--output-dir",
            str(output),
            *extra,
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shared_new_cohort_is_disjoint_reproducible_and_runner_compatible(tmp_path):
    selection = _fixture(tmp_path)
    preserved = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }
    output = tmp_path / "cohort"
    run = _run(selection, output)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "selection-manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (output / "pilot-prompts.jsonl").read_text().splitlines()
    ]
    old_ids = {
        json.loads(line)["candidate_id"]
        for line in (selection / "pilot-prompts.jsonl").read_text().splitlines()
    }
    assert len(rows) == 20
    assert not old_ids.intersection(row["candidate_id"] for row in rows)
    assert manifest["excluded_prompt_count"] == 20
    assert manifest["eligible_population_count"] == 180
    assert manifest["scientific_result"] is False
    assert manifest["source_git_commit"] == COMMIT
    assert manifest["expected_cells_per_model"] == 240
    loaded = _load_calibration_prompts(
        output / "pilot-prompts.jsonl",
        output / "selection-records.jsonl",
        prompt_count=20,
        seed=20260912,
    )
    assert Counter(row.axis_bin for row in loaded) == {index: 5 for index in range(4)}
    assert len(_cells(loaded)) == 240
    assert all(path.read_bytes() == content for path, content in preserved.items())
    repeated = tmp_path / "repeated"
    assert _run(selection, repeated).returncode == 0
    for filename in (
        "pilot-prompts.jsonl",
        "pilot-axis.jsonl",
        "selection-records.jsonl",
    ):
        assert (output / filename).read_bytes() == (repeated / filename).read_bytes()
    assert _run(selection, output).returncode != 0


@pytest.mark.parametrize(
    "relative",
    [
        "population.jsonl",
        "axes.jsonl",
        "original/pilot-prompts.jsonl",
        "original/pilot-axis.jsonl",
        "original/selection-records.jsonl",
    ],
)
def test_source_or_exclusion_hash_mismatch_fails_before_output(tmp_path, relative):
    selection = _fixture(tmp_path)
    path = tmp_path / relative
    path.write_text(path.read_text() + "\n")
    output = tmp_path / "cohort"
    run = _run(selection, output)
    assert run.returncode != 0
    assert "hash mismatch" in run.stderr
    assert not output.exists()


def test_non_divisible_cohort_is_rejected_before_output(tmp_path):
    selection = _fixture(tmp_path)
    output = tmp_path / "cohort"
    run = _run(selection, output, "--prompt-count", "21")
    assert run.returncode != 0
    assert "multiple of axis bins" in run.stderr
    assert not output.exists()


def test_new_ids_with_old_question_text_are_also_excluded(tmp_path):
    selection = _fixture(tmp_path)
    prompts = [
        json.loads(line)
        for line in (tmp_path / "population.jsonl").read_text().splitlines()
    ]
    axes = [
        json.loads(line) for line in (tmp_path / "axes.jsonl").read_text().splitlines()
    ]
    prompts[1]["question"] = "  " + prompts[0]["question"].upper() + "  \n"
    digest = hashlib.sha256(prompts[1]["question"].encode()).hexdigest()
    prompts[1]["question_sha256"] = digest
    axes[1]["text_sha256"] = digest
    original_path = selection / "selection-manifest.json"
    original = json.loads(original_path.read_text())
    original["sources"]["prompts"] = _write_rows(tmp_path / "population.jsonl", prompts)
    original["sources"]["axis_map"] = _write_rows(tmp_path / "axes.jsonl", axes)
    original_path.write_text(json.dumps(original))
    output = tmp_path / "cohort"
    run = _run(selection, output)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((output / "selection-manifest.json").read_text())
    assert manifest["additional_text_overlap_count"] == 1
    assert manifest["eligible_population_count"] == 179
    rows = [
        json.loads(line)
        for line in (output / "pilot-prompts.jsonl").read_text().splitlines()
    ]
    assert prompts[1]["candidate_id"] not in {row["candidate_id"] for row in rows}
    axis_by_id = {row["candidate_id"]: row for row in axes}
    selected_axes = [
        json.loads(line)
        for line in (output / "pilot-axis.jsonl").read_text().splitlines()
    ]
    assert all(row == axis_by_id[row["candidate_id"]] for row in selected_axes)


def test_wrong_expected_source_count_fails_before_output(tmp_path):
    selection = _fixture(tmp_path)
    output = tmp_path / "cohort"
    run = _run(selection, output, "--expected-population-count", "201")
    assert run.returncode != 0
    assert "count differs" in run.stderr
    assert not output.exists()
