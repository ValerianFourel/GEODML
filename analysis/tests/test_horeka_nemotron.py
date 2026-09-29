"""Nemotron on HoreKa keeps the pilot's scientific settings and only one approved hour."""

import json
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_nemotron as nemo
from analysis.scripts.prepare_horeka_qwen import model_inventory


def test_stage_commands_keep_the_frozen_pilot_settings_and_require_a100(tmp_path):
    config = {"repository": "/repo", "dataset_root": "/ds", "count": 12, "max_tokens": 4096}  # pre-SI config
    prepare, run = nemo.stage_commands(config, python="/rt/bin/python", attempt=tmp_path, cache=tmp_path / "c")
    si_prepare, si_run = nemo.stage_commands({**config, "trial": "source-importance", "max_tokens": 640},
                                             python="/rt/bin/python", attempt=tmp_path, cache=tmp_path / "c")
    assert si_run[si_run.index("--") + 2].endswith("try_source_importance_judge.py")
    assert si_run[si_run.index("--max-tokens") + 1] == "640"
    assert "nemotron-source-importance-horeka-trial" in si_prepare
    assert [a for a in si_prepare if a.startswith("--") and a != "--stage"] == [
        a for a in prepare if a.startswith("--") and a != "--stage"]  # same frozen serving flags
    joined = " ".join(prepare)
    for expected in ("--model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                     "--model-revision bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
                     "--expected-gpu-name-pattern A100", "--tensor-parallel-size 4", "--dtype bfloat16",
                     "--max-model-len 73728", "--gpu-memory-utilization 0.85", "--request-concurrency 4",
                     "--enforce-eager", "--vllm-executable /rt/bin/vllm"):
        assert expected in joined
    assert run[run.index("--") + 2].endswith("try_claims_v3_judge.py")
    assert run[run.index("--model") + 1] == "qwen38" and run[run.index("--dataset-root") + 1] == "/ds"


def workspace(tmp_path, verified=True):
    ws = tmp_path / "ws"
    (ws / "environment/qwen-runtime/bin").mkdir(parents=True)
    (ws / "environment/qwen-runtime/bin/python").write_text("")
    prep = nemo.preparation_dir(ws)
    prep.mkdir(parents=True)
    if verified:
        (prep / "models-verified.json").write_text(json.dumps({"status": "verified"}))
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "contract.json").write_text("{}")
    return ws, ds


def submit_args(tmp_path, ws, ds, **overrides):
    values = dict(workspace=ws, dataset=ds, output=tmp_path / "run", account="acct", partition="accelerated",
                  reservation="casualnet", walltime="01:00:00", approval="approved: 1 node, 1 h",
                  count=12, max_tokens=None, trial="source-importance", dry_run=True, no_submit=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def clean_git(monkeypatch):
    def fake(command, text=True):
        return "a" * 40 + "\n" if "rev-parse" in command else ""
    monkeypatch.setattr(nemo.subprocess, "check_output", fake)


def test_dry_run_prints_one_exclusive_four_gpu_hour(tmp_path, clean_git, capsys):
    ws, ds = workspace(tmp_path)
    assert nemo.submit(submit_args(tmp_path, ws, ds)) == 0
    out = json.loads(capsys.readouterr().out)
    for flag in ("--nodes=1", "--gres=gpu:4", "--exclusive", "--time=01:00:00", "--reservation=casualnet",
                 "--job-name=geodml-nemotron-horeka-trial", "--account=acct"):
        assert flag in out["command"]
    assert out["config"]["serving"]["max_model_len"] == 73728 and out["config"]["scientific_result"] is False
    assert out["config"]["trial"] == "source-importance" and out["config"]["max_tokens"] == 640
    assert not (tmp_path / "run").exists()


def test_replay_selection_and_llama_model_reach_the_frozen_command(tmp_path, clean_git):
    import hashlib
    ws, ds = workspace(tmp_path)
    cells = tmp_path / "cells.jsonl"
    cells.write_text('{"fingerprint":"example"}\n')
    args = submit_args(tmp_path, ws, ds, model="llama4", cells_from=cells, dry_run=False, no_submit=True)
    assert nemo.submit(args) == 0
    config = json.loads((args.output / "config.json").read_text())
    assert config["judge_protocol"] == "agentic-source-importance-v2"
    assert config["cells_from_sha256"] == hashlib.sha256(cells.read_bytes()).hexdigest()
    _, command = nemo.stage_commands(config, python="/rt/bin/python", attempt=tmp_path, cache=tmp_path / "cache")
    assert command[command.index("--model") + 1] == "llama4"
    assert command[command.index("--cells-from") + 1] == str(cells)
    assert command[command.index("--cells-from-sha256") + 1] == config["cells_from_sha256"]


def test_submit_refuses_without_verified_weights_or_approval(tmp_path, clean_git):
    ws, ds = workspace(tmp_path, verified=False)
    with pytest.raises(ValueError, match="download and verify"):
        nemo.submit(submit_args(tmp_path, ws, ds))
    ws2, ds2 = workspace(tmp_path / "second")
    with pytest.raises(ValueError, match="approval"):
        nemo.submit(submit_args(tmp_path / "second", ws2, ds2, approval=" "))
    (tmp_path / "second/run").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        nemo.submit(submit_args(tmp_path / "second", ws2, ds2))


def test_rejected_native_schema_stops_execution_before_model_start(tmp_path, clean_git, monkeypatch):
    from analysis.scripts import horeka_qwen_bouts, verify_inference_allocation

    ws, ds = workspace(tmp_path)
    args = submit_args(tmp_path, ws, ds, dry_run=False, no_submit=True)
    assert nemo.submit(args) == 0
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(verify_inference_allocation, "verify", lambda cluster: {"verified": True})
    monkeypatch.setattr(horeka_qwen_bouts, "compiler_environment", lambda: {})

    def command_output(command, **kwargs):
        if command[0] == "scontrol":
            return f"JobName={nemo.JOB_NAME} TimeLimit=01:00:00 EndTime=2026-09-29T10:00:00"
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        if "status" in command:
            return ""
        pytest.fail(f"unexpected command before schema acceptance: {command}")

    def reject_schema(command, **kwargs):
        if "--check-schema" not in command:
            pytest.fail(f"model preparation started despite rejected schema: {command}")
        return SimpleNamespace(returncode=1, stdout="", stderr="unsupported schema")

    monkeypatch.setattr(nemo.subprocess, "check_output", command_output)
    monkeypatch.setattr(nemo.subprocess, "run", reject_schema)
    with pytest.raises(RuntimeError, match="SI schema check failed"):
        nemo.execute(args.output / "config.json")
    diagnostic = json.loads((args.output / "attempts/job123/schema-check.json").read_text())
    assert diagnostic == {"returncode": 1, "stdout": "", "stderr": "unsupported schema"}


def test_walltime_is_limited_to_the_approved_hour():
    with pytest.raises(SystemExit):
        nemo.main(["submit", "--workspace", "w", "--dataset", "d", "--output", "o", "--account", "a",
                   "--walltime", "05:00:00", "--approval", "x"])


def test_nemotron_inventory_uses_only_the_pinned_revision():
    siblings = [SimpleNamespace(rfilename=name, size=10, lfs=SimpleNamespace(sha256="b" * 64) if name.endswith("safetensors") else None,
                                blob_id="c" * 40)
                for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model-00001.safetensors",
                             "modeling_nemotron_h.py", "original/x.bin")]
    api = SimpleNamespace(model_info=lambda repo, revision, files_metadata: SimpleNamespace(sha=revision, siblings=siblings))
    manifest = model_inventory(api, nemo.NEMOTRON)
    assert [(m["repo_id"], m["revision"]) for m in manifest["models"]] == list(nemo.NEMOTRON)
    assert "original/x.bin" not in manifest["models"][0]["files"]
    bad = SimpleNamespace(model_info=lambda repo, revision, files_metadata: SimpleNamespace(sha="0" * 40, siblings=siblings))
    with pytest.raises(ValueError, match="different pinned revision"):
        model_inventory(bad, nemo.NEMOTRON)


def test_interactive_mode_prepares_the_same_run_and_prints_salloc(tmp_path, clean_git, capsys):
    ws, ds = workspace(tmp_path)
    assert nemo.submit(submit_args(tmp_path, ws, ds, dry_run=False, no_submit=True, reservation=None,
                                   partition="dev_accelerated")) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["salloc"].startswith("salloc --nodes=1 --ntasks=1 --gres=gpu:4 --exclusive")
    assert "--no-requeue" not in out["salloc"]  # salloc rejects this sbatch-only option
    for flag in ("--time=01:00:00", "--partition=dev_accelerated", "--job-name=geodml-nemotron-horeka-trial"):
        assert flag in out["salloc"]
    assert "--output=" not in out["salloc"] and "run.sh" not in out["salloc"]
    run = tmp_path / "run"
    assert (run / "config.json").is_file() and (run / "run.sh").is_file() and (run / "interactive.json").is_file()
    assert not (run / "SUBMISSION_ATTEMPTED").exists() and not (run / "submission.json").exists()
