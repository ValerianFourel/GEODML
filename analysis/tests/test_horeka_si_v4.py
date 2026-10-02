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


@pytest.mark.parametrize("compact", [False, True])
def test_new_stage_uses_existing_gemma_boundary_and_v4_driver(tmp_path, compact):
    config = {"repository": str(tmp_path), "trial": "si-v4-development",
              "model_id": pilot.gemma.MODEL, "model_revision": pilot.gemma.REVISION,
              "development_config": str(tmp_path / "config.json")}
    if compact:
        config["structured_outputs_config"] = {"backend": "xgrammar", "disable_any_whitespace": True}
    prepare, run = pilot.stage.stage_commands(config, python="/runtime/bin/python", attempt=tmp_path, cache=tmp_path / "cache")
    assert prepare[prepare.index("--model-revision") + 1] == pilot.gemma.REVISION
    assert "--language-model-only" in prepare and "--expected-gpu-name-pattern" in prepare
    assert run[run.index("--") + 1:run.index("--") + 4] == ["/runtime/bin/python", str(tmp_path / "analysis/scripts/horeka_si_v4.py"), "review"]
    assert run[run.index("--config") + 1] == str(tmp_path / "config.json")
    if compact:
        assert json.loads(prepare[prepare.index("--structured-outputs-config") + 1]) == {
            "backend": "xgrammar", "disable_any_whitespace": True}
    else:
        assert "--structured-outputs-config" not in prepare


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


@pytest.fixture
def gemma_runtime(tmp_path, monkeypatch):
    import hashlib
    import sys
    from pathlib import Path
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
    scheduler = {"walltime": "01:00:00", "remaining_minutes": 40,
                 "queue": "123|geodml-gemma-si-v4|RUNNING\n"}
    def check_output(cmd, **kw):
        if cmd[0] == "squeue":
            return scheduler["queue"]
        if cmd[0] == "scontrol":
            end = pilot.datetime.datetime.now() + pilot.datetime.timedelta(minutes=scheduler["remaining_minutes"])
            return (f"JobName={pilot.JOB_NAME} JobState=RUNNING TimeLimit={scheduler['walltime']} "
                    f"Account=fixture UserId=user({pilot.os.getuid()}) EndTime={end.isoformat()}")
        if cmd[0] == "git":
            return "" if "status" in cmd else "a" * 40
        if cmd[0] == "sacct":
            return ""
        raise AssertionError(cmd)
    monkeypatch.setattr(pilot.subprocess, "check_output", check_output)
    return scheduler


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("walltime,hours", [("01:00:00", 1), ("03:00:00", 3)])
def test_prepare_records_approved_walltime_and_resource_budget(baseline, tmp_path, monkeypatch, gemma_runtime, walltime, hours, selected):
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
    gemma_runtime["walltime"] = walltime
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
        settings = pilot.judge.load_config(out / "judge-config.json")
        assert settings["structured_outputs_config"] == config["structured_outputs_config"] == {
            "backend": "xgrammar", "disable_any_whitespace": True}
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


@pytest.mark.parametrize("condition", ["ready", "none", "pending", "multiple", "short", "wrong-job"])
def test_fresh_keeps_selection_and_prepares_only_for_a_running_job(
        baseline, tmp_path, monkeypatch, gemma_runtime, capsys, condition):
    import shlex
    import subprocess
    bundle = tmp_path / "saved.json"
    bundle.write_text(json.dumps(replay.freeze_baseline(baseline)))
    source = tmp_path / "reviews/gemma-old"
    old = source / "attempts/job123"
    old.mkdir(parents=True)
    (old / "trial-result.json").write_text('{"status":"failed"}')
    (source / "config.json").write_text(json.dumps({"source_files": {str(bundle): pilot.judge.file_hash(bundle)}}))
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    if condition in ("none", "pending"):
        gemma_runtime["queue"] = "" if condition == "none" else "123|geodml-gemma-si-v4|PENDING\n"
    elif condition == "multiple":
        gemma_runtime["queue"] += "456|geodml-gemma-si-v4|RUNNING\n"
        monkeypatch.setattr(pilot, "terminal_choice", lambda prompt: "456")
    elif condition == "short":
        gemma_runtime["remaining_minutes"] = 10
    argv = ["fresh", "--workspace", str(tmp_path), "--source-run", str(source),
            "--count", "3", "--take", "2,3"]
    if condition == "wrong-job":
        argv += ["--existing-job-id", "999"]
    prepared = condition in ("ready", "multiple")
    assert pilot.main(argv) == (0 if prepared else 2)
    session, = (tmp_path / "reviews").glob("gemma-selected-*")
    picked = pilot.verified_selection(session / "selected-inputs.json")
    assert len(picked["cells"]) == 2
    assert {p: p.read_bytes() for p in before} == before
    assert not list(session.rglob("attempts"))
    if prepared:
        job = "456" if condition == "multiple" else "123"
        config = json.loads((session / "run/config.json").read_text())
        assert config["existing_job_id"] == job
        assert config["inventories"]["v4-inputs"]["cells"] == 2
        copied = (session / "compute-command.sh").read_text()
        assert copied in capsys.readouterr().out
        subprocess.run(["bash", "-n"], input=copied, text=True, check=True)
        launcher = (session / "run/run.sh").read_text()
        subprocess.run(["bash", "-n"], input=launcher, text=True, check=True)
        invocation = shlex.split(launcher.split("exec ", 1)[1])
        assert invocation[-3:] == ["run-picked", "--config", str(session / "run/config.json")]
    else:
        assert not (session / "compute-command.sh").exists()
        if condition in ("none", "pending"):
            assert "SELECTION_SAVED_NO_RUNNING_ALLOCATION" in capsys.readouterr().out
        else:
            assert (session / "preparation-error.json").is_file()


@pytest.mark.parametrize("condition,reason", [
    ("ready", None), ("wrong-job", "recorded existing"), ("short", "20 minutes"),
    ("unresolved", "no terminal receipt"), ("duplicate", "already attempted"),
    ("overlap", "work to reconcile"), ("storage", "storage/quota"),
    ("locked", "another selected Gemma run"),
])
def test_run_picked_preserves_prior_work_and_checks_before_execution(
        baseline, tmp_path, monkeypatch, gemma_runtime, condition, reason):
    import fcntl
    import sqlite3
    root = tmp_path / "reviews/gemma-selected-fixture/run"
    root.mkdir(parents=True)
    pilot.freeze_saved([replay.freeze_baseline(baseline)], root / "v4-inputs", pilot.v4.PROTOCOL)
    config = {"workload_mode": "selected-v4-cells", "existing_job_id": "123",
              "minimum_remaining_seconds": 1200, "workspace": str(tmp_path), "account": "fixture"}
    path = root / "config.json"
    path.write_text(json.dumps(config))
    # Exercise a symlinked user path as well as preserved real attempt paths.
    alias = tmp_path / "run-alias"
    alias.symlink_to(root, target_is_directory=True)
    old = tmp_path / "reviews/gemma-old/attempts/job123"
    old.mkdir(parents=True)
    if condition != "unresolved":
        (old / "trial-result.json").write_text('{"status":"failed"}')
    if condition == "duplicate":
        (root / "attempts/job123").mkdir(parents=True)
    if condition == "short":
        gemma_runtime["remaining_minutes"] = 10
    index = old / "trial/v4-pass1/control/index.sqlite"
    index.parent.mkdir(parents=True)
    with sqlite3.connect(index) as db:
        db.execute("CREATE TABLE tasks (id TEXT, state TEXT)")
        if condition == "overlap":
            task = next(pilot.judge.rows(root / "v4-inputs/tasks.jsonl.gz"))
            db.execute("INSERT INTO tasks VALUES (?, 'done')", (task["judge_task_id"],))
    before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    monkeypatch.setenv("SLURM_JOB_ID", "999" if condition == "wrong-job" else "123")
    monkeypatch.setattr(pilot, "capture_quota", lambda *a: {"fixture": True})
    monkeypatch.setattr(pilot.judge, "check_storage", lambda *a: {
        "safe_to_admit": condition != "storage", "quota_verified": True})
    calls = []
    def execute(config_path):
        calls.append(config_path)
        with (tmp_path / "reviews/.gemma-job123.lock").open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return 0
    monkeypatch.setattr(pilot.stage, "execute", execute)
    with (tmp_path / "reviews/.gemma-job123.lock").open("a") as other:
        if condition == "locked":
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if reason:
            with pytest.raises(ValueError, match=reason):
                pilot.main(["run-picked", "--config", str(alias / "config.json")])
            assert calls == []
        else:
            assert pilot.main(["run-picked", "--config", str(alias / "config.json")]) == 0
            assert calls == [path.resolve()]
        assert {p: p.read_bytes() for p in before} == before


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


@pytest.mark.parametrize("selected", [False, True])
def test_html_task_monitor_handles_both_inventory_shapes_without_crediting_missing_work(tmp_path, selected):
    import html
    from pathlib import Path
    import re
    import sqlite3
    import subprocess
    import sys

    page = (Path(__file__).resolve().parents[1] / "docs/horeka-si-v4.html").read_text()
    section = re.search(r'<section\b[^>]*id="task-counts"[^>]*>(.*?)</section>', page, re.S)
    command = html.unescape(re.search(r'<pre><code>(.*?)</code></pre>', section[1], re.S)[1])
    script = re.search(r"<<'PY'\n(.*?)\nPY\n", command, re.S)[1]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"workload_mode": "selected-v4-cells" if selected else "legacy-development", "inventories": {
        "v4-inputs": {"unique_tasks": {"map": 2, "source_dependency": 3, "j1": 2}},
        "v3-inputs": {"unique_tasks": {"source": 4, "j1": 2}},
        "constructed-inputs": {"unique_tasks": 4},
    }}))
    files = [config]
    mixed = [("done", '{"ok": true}'), ("done", '{"ok": false}'),
             ("saved", '{"ok": true}'), ("running", None)]
    for name, rows in (
        ("v4-pass1", mixed if selected else []),
        ("constructed", mixed),
    ):
        path = tmp_path / "attempts/job123/trial" / name / "control/index.sqlite"
        path.parent.mkdir(parents=True)
        db = sqlite3.connect(path)
        try:
            db.execute("CREATE TABLE tasks (state TEXT, result TEXT)")
            db.executemany("INSERT INTO tasks VALUES (?, ?)", rows)
            db.commit()
        finally:
            db.close()
        files.append(path)
    before = {path: path.read_bytes() for path in files}

    result = subprocess.run([sys.executable, "-", str(tmp_path)], input=script,
                            text=True, capture_output=True, timeout=10, check=False)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "INVENTORY_MISMATCH" in result.stdout
    counted = next(line for line in lines if line.startswith("v4-pass1 " if selected else "constructed "))
    for field in ("total=4", "expected=7" if selected else "expected=4", "completed=1", "failed=1", "saved_ok=1", "running=1"):
        assert field in counted.split()
    if selected:
        assert "PLANNED_PASSES 1" in result.stdout
        assert not any(line.startswith(("v3-bridge ", "constructed ", "v4-pass2 ", "v4-pass3 ")) for line in lines)
    else:
        assert any(line.startswith("v4-pass1 total=0 expected=7 completed=0") for line in lines)
        assert any(line.startswith("v3-bridge NO_INDEX:") and "expected=6" in line for line in lines)
        assert any(line.startswith("v4-pass3 NO_INDEX:") and "expected=7" in line for line in lines)
    assert {path: path.read_bytes() for path in files} == before
