"""Nemotron on HoreKa keeps the pilot's scientific settings and only one approved hour."""

import json
from types import SimpleNamespace

import pytest

from analysis.scripts import horeka_nemotron as nemo
from analysis.scripts.prepare_horeka_qwen import model_inventory


def test_stage_commands_keep_the_frozen_pilot_settings_and_require_a100(tmp_path):
    config = {"repository": "/repo", "dataset_root": "/ds", "count": 12, "max_tokens": 4096}
    prepare, run = nemo.stage_commands(config, python="/rt/bin/python", attempt=tmp_path, cache=tmp_path / "c")
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
                  count=12, max_tokens=4096, dry_run=True)
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
    assert not (tmp_path / "run").exists()


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
