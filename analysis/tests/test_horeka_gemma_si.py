"""Frozen comparison identities, honest failures and narrowly scoped cache removal."""
import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_gemma_si as gemma
from analysis.scripts import replay_source_importance_judge as replay
from analysis.scripts import run_acl_arr_vllm
from analysis.tests.test_source_importance_pipeline import A, B, ANSWER, ScriptedServer, cell_for, prepare, trace


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "baseline"
    for pass_id in (1, 2):
        for model in ("qwen38", "llama4"):
            path = root / "runs" / f"pass{pass_id}-{model}" / "attempts/job123/trial"
            path.mkdir(parents=True)
            cells, tasks = [], {}
            for i in range(10):
                cell = cell_for(ANSWER, trace([A, B], [A, B]), cell_id=f"{model}-{i}")
                cell["model"] = model
                record, frozen = prepare.build_cell(cell, max_tokens=640, j1_max_tokens=64)
                cells.append(record)
                tasks.update({t["judge_task_id"]: t for t in frozen})
            results = [{"judge_task_id": t["judge_task_id"], "ok": True,
                        "parsed_output": {"importance": 2} if t["task"] == "source_importance"
                        else {"request_fulfillment": 5}} for t in tasks.values()]
            for name, values in (("cells", cells), ("tasks", tasks.values()), ("results", reversed(results))):
                (path / f"{name}.jsonl").write_text("".join(json.dumps(v) + "\n" for v in values))
            (path / "summary.json").write_text('{"status":"passed"}')
    return root


def test_replay_preserves_inputs_and_joins_shuffled_results_by_id(baseline, tmp_path, monkeypatch):
    bundle = replay.freeze_baseline(baseline)
    original = copy.deepcopy(bundle)
    monkeypatch.setattr(run_acl_arr_vllm, "VllmChatClient", ScriptedServer)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(bundle))
    args = SimpleNamespace(inputs=inputs, inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
                           output=tmp_path / "gemma", base_url="http://x/v1", server_model_name=gemma.MODEL)
    assert asyncio.run(replay.run(args)) == 0
    for pass_id in (1, 2):
        assert replay.indexed(replay.rows(args.output / f"pass{pass_id}/tasks.jsonl")) == replay.indexed(bundle["tasks"])
        cells = replay.rows(args.output / f"pass{pass_id}/cells.jsonl")
        assert len(cells) == 20
        assert all(c["grades"] == {A["url"]: 4, B["url"]: 0} and c["j1"] == 4 for c in cells)
    scores = json.loads((args.output / "comparison.json").read_text())["scores"]
    assert all(row["gemma"]["pass1"] == row["gemma"]["pass2"] for row in scores)
    assert all(row["nemotron"]["pass1"] == [2, 2] for row in scores if row["task"] == "source_importance")
    assert bundle == original
    with pytest.raises(FileExistsError):
        asyncio.run(replay.run(args))
    inputs.write_text(inputs.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        asyncio.run(replay.run(args))


def test_baseline_refuses_mapping_loss_and_task_drift(baseline):
    bundle = replay.freeze_baseline(baseline)
    bundle["cells"][0]["sources"][0]["judge_task_id"] = "wrong-id"
    with pytest.raises(ValueError, match="missing frozen"):
        replay.frozen_cells(bundle)
    path = baseline / "runs/pass2-qwen38/attempts/job123/trial/tasks.jsonl"
    tasks = replay.rows(path)
    next(t for t in tasks if t["task"] == "source_importance")["source_text"] = "changed"
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    with pytest.raises(ValueError):
        replay.freeze_baseline(baseline)


def test_missing_or_failed_scores_are_not_zero(baseline):
    bundle = replay.freeze_baseline(baseline)
    tid = bundle["tasks"][0]["judge_task_id"]
    scores = replay.compare(bundle, {"pass1": [{"judge_task_id": tid, "ok": False,
                                               "parsed_output": {"importance": 5}}], "pass2": []})
    assert all(s["gemma"] == {"pass1": None, "pass2": None} for s in scores)


def test_preparation_pins_gemma_replay_without_allocating(baseline, tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    prep_dir = gemma.prep(ws)
    prep_dir.mkdir(parents=True)
    bundle = replay.freeze_baseline(baseline)
    (prep_dir / "nemotron-baseline.json").write_text(json.dumps(bundle))
    (prep_dir / "models-verified.json").write_text(json.dumps({"status": "verified", "models": [
        {"repo_id": gemma.MODEL, "revision": gemma.REVISION, "snapshot": str(prep_dir)}]}))
    monkeypatch.setattr(gemma.subprocess, "check_output", lambda cmd, **kw: "a" * 40 if "rev-parse" in cmd else "")
    monkeypatch.setattr(gemma, "runtime_check", lambda: {"vllm": "fixture"})
    calls = []
    monkeypatch.setattr(gemma.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    out = tmp_path / "run"
    args = ["prepare", "--workspace", str(ws), "--output", str(out), "--walltime", "00:45:00", "--approval"]
    with pytest.raises(ValueError, match="approval"):
        gemma.main(args + [" "])
    assert gemma.main(args + ["fixture approval"]) == 0
    config = json.loads((out / "config.json").read_text())
    assert config["git_commit"] == "a" * 40 and config["walltime"] == "00:45:00"
    assert config["replay_inputs_sha256"] == hashlib.sha256((out / "inputs.json").read_bytes()).hexdigest()
    prepare_cmd, run_cmd = gemma.runner.stage_commands(config, python="/runtime/bin/python", attempt=out, cache=out / "cache")
    assert prepare_cmd[prepare_cmd.index("--model-id") + 1] == gemma.MODEL
    assert prepare_cmd[prepare_cmd.index("--model-revision") + 1] == gemma.REVISION
    assert "--language-model-only" in prepare_cmd
    assert run_cmd[run_cmd.index("--inputs") + 1] == str(out / "inputs.json")
    assert run_cmd[run_cmd.index("--server-model-name") + 1] == gemma.MODEL
    assert len(calls) == 1 and "--check-schema" in calls[0]
    assert (out / "run.sh").stat().st_mode & 0o111


def test_removal_keeps_results_and_other_models_and_refuses_active_jobs(tmp_path, monkeypatch):
    cache = tmp_path / "models" / ("models--" + gemma.runner.MODEL_ID.replace("/", "--"))
    cache.mkdir(parents=True)
    (cache / "weights").write_text("weights")
    other = tmp_path / "models/other-model"
    other.mkdir()
    results = tmp_path / "reviews/results.json"
    results.parent.mkdir()
    results.write_text("keep")
    receipt = gemma.runner.preparation_dir(tmp_path) / "models-verified.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("old receipt")
    monkeypatch.setattr(gemma.subprocess, "check_output", lambda *a, **k: "geodml-nemotron-horeka-trial\n")
    with pytest.raises(ValueError, match="queued/running"):
        gemma.remove_nemotron(tmp_path)
    assert (cache / "weights").exists()
    monkeypatch.setattr(gemma.subprocess, "check_output", lambda *a, **k: "qwen-bout\n")
    gemma.remove_nemotron(tmp_path)
    assert not cache.exists() and not receipt.exists()
    assert other.is_dir() and results.read_text() == "keep"
    cache.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="redirected"):
        gemma.remove_nemotron(tmp_path)
