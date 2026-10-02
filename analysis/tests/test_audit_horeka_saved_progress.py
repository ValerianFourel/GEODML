"""Operational audit preserves unknowns and existing cluster work."""
import gzip
import hashlib
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
    selected = workspace / "reviews/gemma-selected-fixture/run"
    put(selected / "config.json", {"workload_mode": "selected-v4-cells", "inventories": {
        "v4-inputs": {"unique_tasks": {"answer_map": 3, "source_dependency": 3}}}})
    inputs = selected / "v4-inputs"
    inputs.mkdir()
    cells = [{"cell_id": f"cell{i}", "map_task_id": f"map{i}", "sources": [{"dependency_id": f"dep{i}"}]} for i in range(1, 4)]
    with gzip.open(inputs / "cells.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(c) + "\n" for c in cells))
    put(inputs / "manifest.json", {"files": {"cells.jsonl.gz": hashlib.sha256((inputs / "cells.jsonl.gz").read_bytes()).hexdigest()}})
    selected_index = selected / "attempts/job21/trial/v4-pass1/control/index.sqlite"
    selected_index.parent.mkdir(parents=True)
    with sqlite3.connect(selected_index) as db:
        db.execute("CREATE TABLE tasks (id TEXT, kind TEXT, state TEXT, result TEXT)")
        db.executemany("INSERT INTO tasks VALUES (?,?,?,?)", [
            ("map1", "answer_map", "done", json.dumps({"ok": True, "parsed_output": {"eligibility": "eligible"}})),
            ("dep1", "source_dependency", "done", json.dumps({"ok": True, "parsed_output": {"status": "scored", "importance": 2}})),
            ("map2", "answer_map", "done", json.dumps({"ok": False, "error": "invalid span range"})),
            ("dep2", "source_dependency", "blocked", json.dumps({"ok": False, "status": "map_failed"})),
            ("map3", "answer_map", "pending", None), ("dep3", "source_dependency", "waiting", None)])
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
    selected_audit = next(r for r in report["gemma"] if r["root"] == str(selected))
    assert len(selected_audit["passes"]) == 1
    selected_pass = selected_audit["passes"][0]
    assert selected_pass["cell_counts"] == {"finished": 1, "failed": 1, "pending": 1}
    assert [r["cell_id"] for r in selected_pass["unfinished_cells"]] == ["cell2", "cell3"]
    assert selected_pass["unfinished_cells"][0]["errors"] == ["invalid span range"]
    assert report["qwen_unique_cells"].get("unavailable")
    passes = next(r for r in report["gemma"] if r["root"] == str(gemma))["passes"]
    constructed = next(row for row in passes if row["pass"] == "constructed")
    assert constructed["expected"] == 2 and constructed["counts"] == {"done_ok": 1, "done_failed": 1}
    assert next(row for row in passes if row["pass"] == "v4-pass1")["expected"] == 5
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert {p for p in workspace.rglob("*") if p.is_file() and output not in p.parents} == set(before)
    assert (output / "audit.tar.gz").is_file()
    assert "AUDIT_ARCHIVE" in capsys.readouterr().out
    assert {argv[0] for argv in calls} <= {"sacct", "squeue", "scontrol", "df", "/usr/lpp/mmfs/bin/mmlsquota"}


@pytest.mark.parametrize("unavailable", [False, True])
def test_unique_qwen_counts_deduplicate_spill_and_keep_unreadable_stripes_unknown(tmp_path, unavailable):
    division = tmp_path / "division"
    dataset = tmp_path / "dataset"
    ids = [f"{i:016x}" + "a" * 48 for i in range(4)]
    put(division / "bouts/bout-0001/bout.json", {"primary_fingerprints": ids[:2], "spill_fingerprints": ids[2:]})
    put(division / "bouts/bout-0002/bout.json", {"primary_fingerprints": ids[2:], "spill_fingerprints": ids[:1]})
    ledger = dataset / "control/task-ledger"
    (ledger / "events").mkdir(parents=True)
    (ledger / "locks").mkdir()
    for stripe in range(2):
        (ledger / "locks" / f"stripe-{stripe:04d}.lock").touch()
    rows = [{"format_version": "geodml-agentic-task-event-v1", "fingerprint": ids[0], "state": "running"},
            {"format_version": "geodml-agentic-task-event-v1", "fingerprint": ids[0], "state": "completed"}]
    (ledger / "events/stripe-0000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    row = {"format_version": "geodml-agentic-task-event-v1", "fingerprint": ids[1], "state": "terminal_failed"}
    (ledger / "events/stripe-0001.jsonl").write_text("{" if unavailable else json.dumps(row) + "\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = audit.unique_qwen_cells(division, {"bouts": [{"number": 1}, {"number": 2}], "dataset_root": str(dataset)}, [{"ledger_stripes": 2}])
    assert result["total"] == 4 and result["recorded_finished"] == 1
    assert result["known_unfinished"] == (1 if unavailable else 3)
    assert result["unknown"] == (2 if unavailable else 0)
    assert result["states"].get("terminal_failed", 0) == (0 if unavailable else 1)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
