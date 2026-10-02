"""Saved-input preservation, scheduler admission and finite development execution."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_si_v4 as pilot
from analysis.scripts import replay_source_importance_judge as replay
from analysis.tests.test_horeka_gemma_si import baseline


@pytest.mark.parametrize("legacy_missing_hash", [False, True])
def test_saved_bundle_bridge_preserves_semantic_inputs_and_joins(baseline, tmp_path, legacy_missing_hash):
    bundle = replay.freeze_baseline(baseline)
    if legacy_missing_hash:
        # Pre-v4 SI-v3 cells omitted this metadata; frozen tasks retain exact text.
        for cell in bundle["cells"]:
            for source in cell["sources"]:
                source.pop("source_sha256")
    unchanged = copy.deepcopy(bundle)
    for protocol, name in ((pilot.v3.PROTOCOL, "v3"), (pilot.v4.PROTOCOL, "v4")):
        out = tmp_path / name
        manifest = pilot.freeze_saved([bundle], out, protocol)
        assert manifest["cells"] == 20
        coordinator = pilot.judge.Coordinator(out, tmp_path / (name + "-results"), {"gpu_count": 4})
        try:
            cells = list(pilot.judge.rows(out / "cells.jsonl.gz"))
            tasks = {t["judge_task_id"]: t for t in pilot.judge.rows(out / "tasks.jsonl.gz")}
            original = {t["judge_task_id"]: t for t in bundle["tasks"]}
            for cell, old in zip(cells, bundle["cells"]):
                assert cell["presented"] == old["presented"]
                assert cell["generator_ranking"] == old["generator_ranking"]
                assert cell["provenance"]["status"] == "provenance_unresolved"
                assert tasks[cell["j1_task_id"]] == original[old["j1_task_id"]]
                for source, old_source in zip(cell["sources"], old["sources"]):
                    task = tasks[source.get("dependency_id", source.get("judge_task_id"))]
                    old_task = original[old_source["judge_task_id"]]
                    assert task["source_text"] == old_task["source_text"]
                    assert task["source_title"] == old_task["source_title"]
                    assert task["max_tokens"] == 4096
        finally:
            coordinator.close()
    assert bundle == unchanged


def test_saved_bundle_bridge_rejects_duplicate_cells(baseline, tmp_path):
    bundle = replay.freeze_baseline(baseline)
    with pytest.raises(ValueError, match="duplicate development"):
        pilot.freeze_saved([bundle, bundle], tmp_path / "duplicate", pilot.v4.PROTOCOL)


@pytest.mark.parametrize("change,reason", [
    ({"complete": False}, "complete"),
    ({"captured_at_epoch": 1}, "fresh"),
    ({"jobs": [{"state": "RUNNING", "job_name": "qwen"}] * 5}, "five"),
    ({"jobs": [{"state": "PENDING", "job_name": "qwen"}]}, "pending"),
    ({"jobs": [{"state": "RUNNING", "job_name": pilot.JOB_NAME}]}, "already exists"),
    ({"owners": [{"start_epoch": 9500}]}, "wait at least"),
])
def test_admission_preserves_limits_without_historical_exception(change, reason):
    snapshot = {"complete": True, "captured_at_epoch": 10000, "jobs": [], "owners": []}
    pilot.admission(snapshot, 10000)
    with pytest.raises(ValueError, match=reason):
        pilot.admission({**snapshot, **change}, 10000)


@pytest.mark.parametrize("three_hour", [False, True])
@pytest.mark.parametrize("blocked", [None, "scope", "duration", "recent", "queue", "storage", "existing", "attempted"])
def test_scoped_queue_exception_keeps_other_admission_guards(tmp_path, monkeypatch, blocked, three_hour):
    from analysis.scripts import capture_agentic_scheduler_snapshot as scheduler
    now = 10000
    name = "gemma-si-v4-development-20261002-3h" if three_hour else "gemma-si-v4-development-20261001"
    out = tmp_path / "reviews" / ("different-run" if blocked == "scope" else name)
    out.mkdir(parents=True)
    config = {"workspace": str(tmp_path), "job_name": pilot.JOB_NAME, "walltime": "01:00:00",
              "approval": "one hour", "account": "fixture", "trial": "si-v4-development",
              "git_commit": "9aa5d90e0b310f3eea587a60787727787ab92757",
              "model_id": pilot.gemma.MODEL, "model_revision": pilot.gemma.REVISION}
    config["walltime"] = "03:00:00" if three_hour else "01:00:00"
    if three_hour:
        config["git_commit"] = "a" * 40
    if blocked == "duration":
        config["walltime"] = "01:00:00" if three_hour else "03:00:00"
    raw = json.dumps(config)
    (out / "config.json").write_text(raw)
    if blocked == "attempted":
        (out / "attempts/job123").mkdir(parents=True)
    jobs = [{"state": "RUNNING", "job_name": "qwen", "start_epoch": 8000}] * 28
    jobs += [{"state": "PENDING", "job_name": "qwen", "start_epoch": None}]
    if blocked == "existing":
        jobs += [{"state": "PENDING", "job_name": pilot.JOB_NAME, "start_epoch": None}]
    snapshot = {"complete": True, "captured_at_epoch": now, "jobs": jobs,
                "owners": [{"start_epoch": 9950}] if blocked == "recent" else []}
    monkeypatch.setattr(scheduler, "capture", lambda **kw: snapshot)
    monkeypatch.setattr(pilot.time, "time", lambda: now)
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda command, **kw:
                        "\n".join(map(str, range(295 if blocked == "queue" else 294)))
                        if "--array" in command else "123\n")
    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(pilot, "capture_quota", lambda *a: {"fixture": True})
    monkeypatch.setattr(pilot.judge, "check_storage", lambda *a:
                        {"safe_to_admit": blocked != "storage", "quota_verified": True})
    argv = ["check", "--output", str(out), "--approved-existing-queue-exception"]
    if blocked:
        with pytest.raises(ValueError):
            pilot.main(argv)
        assert not (out / "admission.json").exists()
    else:
        assert pilot.main(argv) == 0
        evidence = json.loads((out / "admission.json").read_text())
        assert "waive five-active and no-pending" in evidence["approved_exception"]
        assert evidence["config_sha256"] == pilot.judge.file_hash(out / "config.json")
    assert (out / "config.json").read_text() == raw


def test_new_stage_uses_existing_gemma_boundary_and_v4_driver(tmp_path):
    config = {"repository": str(tmp_path), "trial": "si-v4-development",
              "model_id": pilot.gemma.MODEL, "model_revision": pilot.gemma.REVISION,
              "development_config": str(tmp_path / "config.json")}
    prepare, run = pilot.stage.stage_commands(config, python="/runtime/bin/python", attempt=tmp_path, cache=tmp_path / "cache")
    assert prepare[prepare.index("--model-revision") + 1] == pilot.gemma.REVISION
    assert "--language-model-only" in prepare and "--expected-gpu-name-pattern" in prepare
    assert run[run.index("--") + 1:run.index("--") + 4] == ["/runtime/bin/python", str(tmp_path / "analysis/scripts/horeka_si_v4.py"), "review"]
    assert run[run.index("--config") + 1] == str(tmp_path / "config.json")


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("status,expected_runs", [("finished", 5), ("finished_with_failures", 5), ("incomplete", 1)])
def test_review_runs_finite_queue_and_stops_on_incomplete_work(tmp_path, monkeypatch, status, expected_runs, selected):
    settings = tmp_path / "judge-config.json"
    settings.write_text('{}')
    config = {"model_id": pilot.gemma.MODEL, "workspace": str(tmp_path), "account": "fixture",
              "inventories": {}, "judge_config_sha256": pilot.judge.file_hash(settings)}
    if selected:
        config["workload_mode"] = "selected-v4-cells"
        expected_runs = 1
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(pilot, "capture_quota", lambda *a: {"fresh": True})
    monkeypatch.setattr(pilot.AllocationBudget, "from_environment", lambda **kw: SimpleNamespace(can_start=lambda: True))
    calls = []
    async def run(args):
        calls.append(args.output.name)
        report = args.output / "reports/fixture"
        report.mkdir(parents=True)
        (report / "summary.json").write_text(json.dumps({"status": status}))
        return 0 if status == "finished" else 2
    monkeypatch.setattr(pilot.judge, "run", run)
    out = tmp_path / "trial"
    result = asyncio.run(pilot.review(SimpleNamespace(config=config_path, output=out,
                        server_model_name=pilot.gemma.MODEL, base_url="http://fixture")))
    assert len(calls) == expected_runs
    assert calls[0] == "v4-pass1"
    assert result == (0 if status == "finished" else 2)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == ("finished" if status == "finished" else "partial_or_failed")
    assert summary["planned_runs"] == (1 if selected else 5)


def test_saved_bundle_bridge_rejects_conflicting_source_hash(baseline, tmp_path):
    bundle = replay.freeze_baseline(baseline)
    bundle["cells"][0]["sources"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="saved source hash"):
        pilot.freeze_saved([bundle], tmp_path / "conflict", pilot.v4.PROTOCOL)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("walltime,hours", [("01:00:00", 1), ("03:00:00", 3)])
def test_prepare_records_approved_walltime_and_resource_budget(baseline, tmp_path, monkeypatch, walltime, hours, selected):
    import hashlib
    import sys
    from pathlib import Path
    bundle = replay.freeze_baseline(baseline)
    extra = copy.deepcopy(bundle["cells"])
    for cell in extra:
        cell["cell_id"] += "-second"
    bundle["cells"] += extra
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    inputs = ["--bundle", str(bundle_path)]
    if selected:
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.json").write_text(json.dumps({"source_files": {
            str(bundle_path): pilot.judge.file_hash(bundle_path)}}))
        picked = tmp_path / "picked.json"
        assert pilot.main(["select", "--source-run", str(source), "--output", str(picked),
                           "--take", "2,5", "--seed", "17"]) == 0
        inputs = ["--selected-inputs", str(picked)]
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    receipt = pilot.gemma.prep(tmp_path) / "models-verified.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "verified", "models": [
        {"repo_id": repo, "revision": revision, "snapshot": str(snapshot)}
        for repo, revision in pilot.gemma.MODELS]}))
    template_path = Path(pilot.__file__).resolve().parents[2] / "analysis/config/si_v4_gemma.template.json"
    template = json.loads(template_path.read_text())
    template["chat_template_sha256"] = hashlib.sha256(b"fixture template").hexdigest()
    read = pilot.stage.read
    monkeypatch.setattr(pilot.stage, "read", lambda path: copy.deepcopy(template) if Path(path) == template_path else read(path))
    versions = {**template["runtime_versions"], "vllm": template["serving_version"].removeprefix("vllm-")}
    monkeypatch.setattr(pilot.gemma, "runtime_check", lambda: versions)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(get_chat_template=lambda: "fixture template"))))
    monkeypatch.setattr(pilot, "schema_check", lambda inputs: None)
    def check_output(cmd, **kw):
        if cmd[0] == "scontrol":
            end = pilot.datetime.datetime.now() + pilot.datetime.timedelta(minutes=40)
            return (f"JobName={pilot.JOB_NAME} JobState=RUNNING TimeLimit={walltime} "
                    f"Account=fixture UserId=user({pilot.os.getuid()}) EndTime={end.isoformat()}")
        return "" if "status" in cmd else "a" * 40
    monkeypatch.setattr(pilot.subprocess, "check_output", check_output)
    out = tmp_path / "prepared"
    argv = ["prepare", "--workspace", str(tmp_path), "--output", str(out), *inputs,
            "--account", "fixture", "--approval", "User approved " + walltime, "--walltime", walltime]
    if selected:
        with pytest.raises(ValueError, match="existing Gemma allocation"):
            pilot.main(argv)
        assert not out.exists()
        argv += ["--existing-job-id", "123"]
    assert pilot.main(argv) == 0
    config = json.loads((out / "config.json").read_text())
    assert config["walltime"] == walltime
    assert config["estimate"]["node_hours_max"] == hours
    assert config["estimate"]["gpu_hours_max"] == 4 * hours
    assert config["inventories"]["v4-inputs"]["cells"] == (2 if selected else 40)
    if selected:
        assert config["existing_job_id"] == "123"
        assert config["workload_mode"] == "selected-v4-cells"
        assert set(config["inventories"]) == {"v4-inputs"}
        assert config["estimate"]["additional_allocation_hours"] == 0
        return
    from analysis.scripts import capture_agentic_scheduler_snapshot as scheduler
    monkeypatch.setattr(scheduler, "capture", lambda **kw: {
        "complete": True, "captured_at_epoch": int(pilot.time.time()), "jobs": [], "owners": []})
    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(pilot, "capture_quota", lambda *a: {"fixture": True})
    monkeypatch.setattr(pilot.judge, "check_storage", lambda *a: {"safe_to_admit": True, "quota_verified": True})
    assert pilot.main(["check", "--output", str(out)]) == 0
    assert json.loads((out / "admission.json").read_text())["approved_exception"] is None


def test_manual_selection_preserves_chosen_inputs_and_detects_changes(baseline, tmp_path, capsys):
    bundle = replay.freeze_baseline(baseline)
    for cell in bundle["cells"]:
        for source in cell["sources"]:
            source.pop("source_sha256")
    source_bundle = tmp_path / "saved.json"
    source_bundle.write_text(json.dumps(bundle))
    source = tmp_path / "failed-run"
    source.mkdir()
    config = source / "config.json"
    config.write_text(json.dumps({"source_files": {str(source_bundle): pilot.judge.file_hash(source_bundle)}}))
    marker = source / "ALLOCATION_ATTEMPTED"
    marker.write_text("preserve original attempt")
    before = {p: p.read_bytes() for p in (source_bundle, config, marker)}
    picked = tmp_path / "picked.json"
    argv = ["select", "--source-run", str(source), "--output", str(picked),
            "--count", "8", "--seed", "17", "--take", "2,5"]
    assert pilot.main(argv) == 0
    shown = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"number"')]
    saved = pilot.verified_selection(picked)
    assert [cell["cell_id"] for cell in saved["cells"]] == [shown[1]["cell_id"], shown[4]["cell_id"]]
    assert len(saved["selection"]["sampled_cell_ids"]) == 8
    original = {cell["cell_id"]: (cell, tasks) for cell, tasks in replay.frozen_cells(bundle)}
    for cell, tasks in replay.frozen_cells(saved):
        assert (cell, tasks) == original[cell["cell_id"]]
    second = tmp_path / "second.json"
    assert pilot.main([*argv[:4], str(second), *argv[5:]]) == 0
    assert second.read_bytes() == picked.read_bytes()
    frozen = tmp_path / "v4"
    pilot.freeze_saved([saved], frozen, pilot.v4.PROTOCOL)
    coordinator = pilot.judge.Coordinator(frozen, tmp_path / "result", {"gpu_count": 4})
    coordinator.close()
    with pytest.raises(FileExistsError):
        pilot.main(argv)
    for bad in ("0", "9", "1,1", "", "not-a-number"):
        with pytest.raises(ValueError):
            pilot.main([*argv[:4], str(tmp_path / "invalid.json"), *argv[5:-1], bad])
        assert not (tmp_path / "invalid.json").exists()
    assert {p: p.read_bytes() for p in before} == before
    changed = copy.deepcopy(saved)
    changed["cells"][0]["prompt_id"] = "changed-after-selection"
    picked.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="content changed"):
        pilot.verified_selection(picked)
    source_bundle.write_text(source_bundle.read_text() + " ")
    with pytest.raises(ValueError, match="source bundle changed"):
        pilot.verified_selection(second)


def test_manual_selection_reads_choices_from_a_nonseekable_terminal(baseline, tmp_path, monkeypatch):
    import builtins
    import os
    import pty
    bundle = tmp_path / "saved.json"
    bundle.write_text(json.dumps(replay.freeze_baseline(baseline)))
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"source_files": {str(bundle): pilot.judge.file_hash(bundle)}}))
    master, slave = pty.openpty()
    terminal_path = os.ttyname(slave)
    open_file = builtins.open
    def tty_open(path, mode="r", *a, **kw):
        if path != "/dev/tty":
            return open_file(path, mode, *a, **kw)
        flags = os.O_RDWR if "+" in mode else os.O_WRONLY if "w" in mode else os.O_RDONLY
        return open_file(os.open(terminal_path, flags | os.O_NOCTTY), mode, *a, **kw)
    monkeypatch.setattr(builtins, "open", tty_open)
    try:
        os.write(master, b"2,3\n")
        output = tmp_path / "picked.json"
        assert pilot.main(["select", "--source-run", str(source), "--output", str(output),
                           "--count", "3", "--seed", "17"]) == 0
        selected = pilot.verified_selection(output)
        assert selected["selection"]["selected_cell_ids"] == selected["selection"]["sampled_cell_ids"][1:]
    finally:
        os.close(master)
        os.close(slave)
