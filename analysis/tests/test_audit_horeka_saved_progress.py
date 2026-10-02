"""Operational audit preserves unknowns and existing cluster work."""
import json
import sqlite3
import subprocess
import sys

import pytest

from analysis.scripts import audit_horeka_saved_progress as audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("scheduler_failure", [False, True])
def test_saved_progress_cli_preserves_unknown_and_partial_failures(tmp_path, monkeypatch, capsys, scheduler_failure):
    workspace = tmp_path / "workspace"
    division = workspace / "qwen-bouts/division"
    put(division / "division.json", {"bouts": [
        {"number": number, "cell_count": 5, "primary_cells": 4, "spill_cells": 1}
        for number in range(1, 5)]})
    (workspace / "qwen-bouts/CURRENT").write_text(str(division))
    for number in range(1, 5):
        root = division / "bouts" / f"bout-{number:04d}"
        put(root / "submission.json", {"stdout": str(number), "returncode": 0})
        put(root / "config.json", {"dataset_root": str(workspace / "dataset"), "git_commit": "a" * 40})
    failed = division / "bouts/bout-0001"
    manifest = put(failed / "attempts/job1/results/run_manifest.json", {
        "completed_count": 3, "remaining_count": 2, "failed_cell_ids": ["c4"],
        "direct_dataset": {"committed_count": 2, "reused_count": 1}, "stop_reason": "bounded_failures"})
    put(failed / "attempts/job1/bout-result.json", {"status": "failed", "returncode": 1})
    (failed / "slurm-1.err").write_text("one agentic-search cell failed after bounded retry\n")
    corrupt = division / "bouts/bout-0002/attempts/job2/results/run_manifest.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text('{"completed_count":')
    put(division / "bouts/bout-0003/attempts/job3/bout-result.json", {
        "status": "failed", "completed_count": 5, "remaining_count": 0,
        "direct_dataset": {"committed_count": 4, "reused_count": 1}})
    gemma = workspace / "reviews/gemma-si-v4-fixture"
    put(gemma / "config.json", {"walltime": "03:00:00", "inventories": {
        "v4-inputs": {"unique_tasks": {"map": 2, "source": 3}},
        "v3-inputs": {"unique_tasks": {"source": 3}},
        "constructed-inputs": {"unique_tasks": 2}}})
    index = gemma / "attempts/job10/trial/constructed/control/index.sqlite"
    index.parent.mkdir(parents=True)
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE tasks (state TEXT, result TEXT)")
        db.executemany("INSERT INTO tasks VALUES (?, ?)", [("done", '{"ok":true}'), ("done", '{"ok":false}')])
    before = {p: p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "sacct" and "-X" in argv:
            if scheduler_failure:
                return subprocess.CompletedProcess(argv, 1, "", "accounting unavailable")
            raw = "".join(f"{n}|geodml-qwen-bout-{n:04d}|FAILED|100|05:00:00|1:0|start|end|\n" for n in range(1, 5))
            return subprocess.CompletedProcess(argv, 0, raw, "")
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0,
                "4|geodml-qwen-bout-0004|RUNNING|1:00|05:00:00|start|node\n"
                "20|geodml-gemma-si-v4|PENDING|0:00|03:00:00|future|Priority\n", "")
        return subprocess.CompletedProcess(argv, 0, "fixture command evidence\n", "")

    monkeypatch.setattr(audit.subprocess, "run", scheduler)
    monkeypatch.setattr(sys, "argv", ["audit", str(workspace)])
    audit.main()
    output = next((workspace / "reviews").glob("horeka-audit-*"))
    report = json.loads((output / "audit.json").read_text())
    rows = {row["job"]: row for row in report["qwen"]}
    assert rows["1"]["completed"] == 3 and rows["1"]["remaining"] == 2
    assert rows["1"]["explicit_failed"] == 1
    assert rows["1"]["newly_committed"] == 2 and rows["1"]["reused"] == 1
    assert rows["2"]["completed"] is None and rows["2"]["explicit_failed"] is None
    assert rows["3"]["completed"] == 5 and rows["3"]["progress_source"] == "bout_result"
    assert rows["4"]["state"] == "RUNNING"
    assert report["scheduler_complete"] is not scheduler_failure
    assert report["jobs"]["20"]["state"] == "PENDING"
    passes = report["gemma"][0]["passes"]
    constructed = next(row for row in passes if row["pass"] == "constructed")
    assert constructed["expected"] == 2 and constructed["counts"] == {"done_ok": 1, "done_failed": 1}
    assert next(row for row in passes if row["pass"] == "v4-pass1")["expected"] == 5
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert {p for p in workspace.rglob("*") if p.is_file() and output not in p.parents} == set(before)
    assert (output / "audit.tar.gz").is_file()
    assert "AUDIT_ARCHIVE" in capsys.readouterr().out
    assert {argv[0] for argv in calls} <= {"sacct", "squeue", "scontrol", "df", "/usr/lpp/mmfs/bin/mmlsquota"}
