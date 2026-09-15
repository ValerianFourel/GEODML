"""Read-only roadmap snapshots must not hide damaged or missing artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.scripts.report_agentic_roadmap_snapshot import collect_snapshot


def _shard(root: Path, *, count: int = 2) -> Path:
    output = root / "models/qwen38/shard-0"
    results = output / "results"
    results.mkdir(parents=True)
    for number in range(count):
        (results / f"cell-{number}.json").write_text(
            json.dumps({"cell_id": f"cell-{number}"})
        )
    (output / "run_manifest.json").write_text(json.dumps({
        "status": "checkpointed", "completed_count": count - 1,
        "failed_cell_ids": ["failed-cell"],
    }))
    return output


def _pilot(root: Path, rows: list[dict]) -> Path:
    pilot = root / "pilot"
    (pilot / "nemotron").mkdir(parents=True)
    (pilot / "nemotron/outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    return pilot


def test_artifacts_take_precedence_over_lagging_manifest(tmp_path):
    _shard(tmp_path)
    snapshot = collect_snapshot(tmp_path)
    assert snapshot["schema_version"] == 1
    assert snapshot["checked_at"].endswith("Z")
    assert snapshot["models"][0]["completed"] == 2
    assert snapshot["models"][0]["expected"] == 6000
    shard = snapshot["models"][0]["shards"][0]
    assert shard["manifest_completed"] == 1
    assert shard["status"] == "checkpointed"
    assert shard["failed_cells"] == 1
    assert snapshot["models"][1]["shards"][0]["status"] == "not_started"
    assert snapshot["pilot"]["status"] == "not_inspected"
    assert snapshot["jobs"] == []
    assert "quality" in snapshot["source_note"]


def test_missing_root_fails_instead_of_claiming_zero(tmp_path):
    with pytest.raises(ValueError, match="run root does not exist"):
        collect_snapshot(tmp_path / "typo")


@pytest.mark.parametrize("content", ["{", "[]", '{"cell_id": "wrong"}'])
def test_invalid_result_fails_with_its_path(tmp_path, content):
    output = _shard(tmp_path)
    path = output / "results/cell-0.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="cell-0.json"):
        collect_snapshot(tmp_path)


def test_unreadable_result_is_not_zero(tmp_path, monkeypatch):
    output = _shard(tmp_path)
    path = output / "results/cell-0.json"
    original = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == path:
            raise PermissionError("permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="cannot read.*cell-0.json"):
        collect_snapshot(tmp_path)


def test_duplicate_cells_across_shards_are_not_double_counted(tmp_path):
    _shard(tmp_path)
    results = tmp_path / "models/qwen38/shard-1/results"
    results.mkdir(parents=True)
    (results / "cell-0.json").write_text('{"cell_id": "cell-0"}')
    with pytest.raises(ValueError, match="duplicate cells across qwen38 shards"):
        collect_snapshot(tmp_path)


def test_unreadable_result_directory_is_not_zero(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    original = Path.iterdir

    def iterdir(self):
        if self == shard / "results":
            raise PermissionError("permission denied")
        return original(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    with pytest.raises(ValueError, match="cannot list.*results"):
        collect_snapshot(tmp_path)


@pytest.mark.parametrize("manifest", [[], {"completed_count": -1}, {"failed_cell_ids": "bad"}])
def test_invalid_manifest_is_not_zero(tmp_path, manifest):
    output = _shard(tmp_path)
    (output / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="run_manifest.json"):
        collect_snapshot(tmp_path)


def test_overflow_is_visible_not_clamped(tmp_path):
    _shard(tmp_path, count=1501)
    snapshot = collect_snapshot(tmp_path)
    assert snapshot["models"][0]["completed"] == 1501
    assert snapshot["models"][0]["shards"][0]["artifact_overflow"] == 1


def test_pilot_counts_unique_outcomes_not_manifest(tmp_path):
    row = {"judge_task_id": "judge-a", "parsed_output": {"score": 1}}
    pilot = _pilot(tmp_path, [row, row])
    (pilot / "nemotron/run_manifest.json").write_text(json.dumps({
        "status": "running", "completed_count": 0,
    }))
    snapshot = collect_snapshot(tmp_path, pilot_root=pilot)
    assert snapshot["pilot"]["completed"] == 1
    assert snapshot["pilot"]["manifest_completed"] == 0
    assert snapshot["pilot"]["duplicate_rows"] == 1


def test_conflicting_pilot_duplicate_is_an_error(tmp_path):
    pilot = _pilot(tmp_path, [
        {"judge_task_id": "judge-a", "parsed_output": {"score": 1}},
        {"judge_task_id": "judge-a", "parsed_output": {"score": 2}},
    ])
    with pytest.raises(ValueError, match="conflicting.*judge-a"):
        collect_snapshot(tmp_path, pilot_root=pilot)


@pytest.mark.parametrize("row", [{"task_id": "wrong", "parsed_output": {}},
                                  {"judge_task_id": "a", "parsed_output": None}])
def test_invalid_pilot_outcome_is_an_error(tmp_path, row):
    pilot = _pilot(tmp_path, [row])
    with pytest.raises(ValueError, match="outcomes.jsonl:1"):
        collect_snapshot(tmp_path, pilot_root=pilot)


def test_partial_jsonl_fails_with_retry_guidance(tmp_path):
    pilot = _pilot(tmp_path, [])
    (pilot / "nemotron/outcomes.jsonl").write_text('{"judge_task_id":')
    with pytest.raises(ValueError, match="Retry.*writer"):
        collect_snapshot(tmp_path, pilot_root=pilot)


def test_requested_pilot_missing_output_is_not_a_failure(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    snapshot = collect_snapshot(tmp_path, pilot_root=pilot)
    assert snapshot["pilot"]["completed"] == 0
    assert snapshot["pilot"]["status"] == "no_manifest"
    assert snapshot["pilot"]["inspected"] is True


def test_scheduler_call_is_read_only_and_bounded(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command == ["sacct", "--jobs=42,43", "--parsable2", "--noheader",
                           "--allocations", "--format=JobIDRaw,JobName,State,Elapsed,Timelimit,ExitCode"]
        assert kwargs["timeout"] == 15
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0,
            "42|qwen|COMPLETED|00:11:02|01:00:00|0:0\n"
            "43|llama|RUNNING|00:02:00|01:00:00|0:0\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    snapshot = collect_snapshot(tmp_path, jobs=["42", "43"])
    assert snapshot["jobs"][0]["state"] == "COMPLETED"
    assert snapshot["jobs"][1]["job_id"] == "43"
    assert "job_error" not in snapshot


@pytest.mark.parametrize("error", [FileNotFoundError("no sacct"),
                                   subprocess.TimeoutExpired("sacct", 15)])
def test_unavailable_scheduler_is_unknown(tmp_path, monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    snapshot = collect_snapshot(tmp_path, jobs=["42"])
    assert snapshot["jobs"] == []
    assert "job_error" in snapshot
    assert snapshot["requested_job_ids"] == ["42"]


def test_missing_scheduler_records_are_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, "", ""))
    snapshot = collect_snapshot(tmp_path, jobs=["42"])
    assert snapshot["jobs"] == []
    assert "no record for: 42" in snapshot["job_error"]


def test_bad_job_id_never_reaches_scheduler(tmp_path, monkeypatch):
    def run(*args, **kwargs):
        pytest.fail("invalid job ID reached subprocess")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="positive numeric allocation ID"):
        collect_snapshot(tmp_path, jobs=["--allusers"])


def test_cli_stdout_is_only_json_and_does_not_write_artifacts(tmp_path):
    _shard(tmp_path)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    script = Path(__file__).resolve().parents[1] / "scripts/report_agentic_roadmap_snapshot.py"
    run = subprocess.run([sys.executable, str(script), "--run-root", str(tmp_path)],
                         text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["models"][0]["completed"] == 2
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
