"""Submission safety for the fixed two-job generator trial, without Slurm."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from analysis.scripts import submit_agentic_paired_trial as module


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def trial(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    for name in ("run_agentic_generation_worker.sh", "run_inference_wave_worker.sbatch"):
        path = repository / "analysis/scripts/slurm/jupiter" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n")
        path.chmod(0o755)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", repository)
    cache = tmp_path / "cache"
    cache.mkdir()
    profile_root = tmp_path / "profiles"
    for specification in module.MODELS.values():
        profile = {
            "model": {key: specification[key] for key in ("model_id", "model_revision")},
            "serving": {"tensor_parallel_size": 4, "data_parallel_size": 1,
                        "host": "127.0.0.1", "port": 8010},
        }
        _json(profile_root / specification["profile"], profile)
        snapshot = cache / ("models--" + specification["model_id"].replace("/", "--")) / "snapshots" / specification["model_revision"]
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            _json(snapshot / filename, {})
        _json(snapshot / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
        _json(snapshot / "model.safetensors", {"fixture": "not real weights"})
    monkeypatch.setattr(module, "load_profile", lambda path: json.loads(path.read_text()))
    bge = cache / "models--BAAI--bge-reranker-v2-m3/snapshots" / module.BGE_REVISION
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        _json(bge / filename, {})
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("true\n")
    snapshots = {}
    for engine in ("DDG", "SEARXNG"):
        snapshot = tmp_path / f"{engine}.jsonl"
        snapshot.write_text(json.dumps({
            "keyword": "topic", "title": "a source", "position": 1,
            "url": "https://example.org/source", "snippet": "source evidence",
        }) + "\n")
        snapshots[f"SEARCH_AGENTIC_{engine}_SNAPSHOT"] = str(snapshot)
    environment = {
        **os.environ, **snapshots,
        "GEODML_APPROVED_WALLTIME": "01:00:00",
        "GEODML_ALLOCATION_ESTIMATE": "Two one-hour jobs, 8 GPU-hour cap; throughput trial.",
        "GEODML_EXECUTION_COMMIT": "a" * 40,
        "GEODML_EXECUTION_REPOSITORY": str(repository),
        "HF_HUB_CACHE": str(cache), "GEODML_CACHE_ROOT": str(cache),
        "ACL_ARR_VENV": str(venv),
    }
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "6.0.1")
    monkeypatch.setattr(module.shutil, "which", lambda *args, **kwargs: "/usr/bin/sbatch")
    calls = []
    responses = [(0, "12345\n", ""), (0, "12346;testcluster\n", "")]

    def run(command, **kwargs):
        if command[0] == "git":
            stdout = environment["GEODML_EXECUTION_COMMIT"] if command[3] == "rev-parse" else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")
        assert command[0] == "sbatch"
        calls.append((command, kwargs))
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        status, stdout, stderr = response
        return subprocess.CompletedProcess(command, status, stdout, stderr)

    monkeypatch.setattr(module.subprocess, "run", run)
    selections = []

    def prepare(selection_root, output, **kwargs):
        selections.append((selection_root, kwargs))
        output.mkdir(parents=True)
        prompts = []
        records = []
        for index in range(120):
            question = f"New question {index}?"
            prompts.append({
                "candidate_id": f"new-{index:03d}", "question": question,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "keyword": "topic",
            })
            records.append({"candidate_id": f"new-{index:03d}", "axis_bin": index % 20})
        for filename, rows in (("pilot-prompts.jsonl", prompts), ("selection-records.jsonl", records)):
            (output / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = output / "selection-manifest.json"
        _json(manifest, {"scientific_result": False, "prompt_count": 120})
        return manifest

    monkeypatch.setattr(module, "prepare_new_cohort", prepare)
    return {
        "arguments": {"selection_root": tmp_path / "old-selection", "output": tmp_path / "trial",
                      "profile_root": profile_root, "account": "test-account", "partition": "test-partition",
                      "submit": True, "environment": environment},
        "environment": environment, "calls": calls, "responses": responses,
        "selections": selections, "bge": bge,
    }


def test_submits_exactly_two_capped_model_isolated_jobs(trial):
    result = module.submit_trial(**trial["arguments"])
    assert result["status"] == "submitted"
    assert result["maximum_total_gpu_hours"] == 8
    assert len(trial["calls"]) == 2
    assert trial["selections"][0][1] == {
        "source_git_commit": "a" * 40, "prompt_count": 120,
        "axis_bins": 20, "master_seed": 20260916,
    }
    root = trial["arguments"]["output"]
    assert result["target_selection"]["target_url_count"] == 240
    assert result["target_selection"]["counts"] == {
        "duckduckgo": {"exact_keyword": 120}, "searxng": {"exact_keyword": 120},
    }
    assert len((root / "tasks.jsonl").read_text().splitlines()) == 1440
    for (slug, job), (command, kwargs) in zip(
        (("qwen38", "12345"), ("llama4", "12346")), trial["calls"], strict=True,
    ):
        assert {"--array=0-0", "--nodes=1", "--ntasks=1", "--cpus-per-task=32",
                "--mem=512G", "--gres=gpu:4", "--time=01:00:00", "--export=ALL",
                "--parsable", "--no-requeue", "--account=test-account",
                "--partition=test-partition"} <= set(command)
        env = kwargs["env"]
        assert kwargs["timeout"] == 60
        assert env["GEODML_MODEL_SLUG"] == slug
        assert env["SEARCH_AGENTIC_REQUEST_CONCURRENCY"] == "4"
        assert env["SEARCH_AGENTIC_CELL_CONCURRENCY"] == "12"
        assert env["SEARCH_AGENTIC_PROMPT_COUNT"] == "120"
        assert env["SEARCH_AGENTIC_PROMPT_SHARD_INDEX"] == "0"
        assert env["SEARCH_AGENTIC_PROMPT_SHARD_COUNT"] == "1"
        assert env["SEARCH_AGENTIC_PRODUCTION_CONDITIONS"] == "1"
        assert env["SEARCH_AGENTIC_PROMPT_SELECTION_SEED"] == "20260912"
        assert env["SEARCH_AGENTIC_PROFILE"].endswith(module.MODELS[slug]["profile"])
        model_root = root / "models" / slug
        assert env["GEODML_WAVE_OUTPUT_ROOT"] == str(model_root / "outputs")
        assert env["SEARCH_AGENTIC_PROMPTS_JSONL"] == str(root / "cohort/pilot-prompts.jsonl")
        assert (model_root / "job-id.txt").read_text().strip() == job
        assert result["models"][slug]["job_id"] == job
        wave = json.loads((model_root / "wave/run_manifest.json").read_text())
        assert wave["worker_count"] == 1 and wave["pending_task_count"] == 1440
    qwen = root / "models/qwen38/wave/workers/worker-00000.jsonl"
    llama = root / "models/llama4/wave/workers/worker-00000.jsonl"
    assert qwen.read_bytes() == llama.read_bytes()
    assert result["models"]["llama4"]["slurm_cluster"] == "testcluster"


@pytest.mark.parametrize("change", ["no_submit", "wrong_time", "empty_estimate", "missing_weight", "wrong_profile", "missing_bge"])
def test_invalid_preflight_never_submits(trial, change):
    arguments = trial["arguments"]
    if change == "no_submit":
        arguments["submit"] = False
    elif change == "wrong_time":
        trial["environment"]["GEODML_APPROVED_WALLTIME"] = "12:00:00"
    elif change == "empty_estimate":
        trial["environment"]["GEODML_ALLOCATION_ESTIMATE"] = ""
    elif change == "missing_weight":
        next(Path(trial["environment"]["HF_HUB_CACHE"]).glob("models--Qwen*/snapshots/*/model.safetensors")).unlink()
    elif change == "missing_bge":
        (trial["bge"] / "config.json").unlink()
    else:
        path = arguments["profile_root"] / module.MODELS["llama4"]["profile"]
        value = json.loads(path.read_text())
        value["model"]["model_revision"] = "b" * 40
        _json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.submit_trial(**arguments)
    assert not trial["calls"]
    assert not arguments["output"].exists()


def test_repeated_trial_root_is_rejected_without_more_allocations(trial):
    module.submit_trial(**trial["arguments"])
    with pytest.raises(FileExistsError, match="already exists"):
        module.submit_trial(**trial["arguments"])
    assert len(trial["calls"]) == 2


def test_second_submission_failure_preserves_first_job_and_both_intents(trial):
    trial["responses"][1] = (1, "", "temporary communication error")
    with pytest.raises(RuntimeError, match="inspect Slurm"):
        module.submit_trial(**trial["arguments"])
    root = trial["arguments"]["output"]
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "submission_incomplete"
    assert manifest["models"]["qwen38"]["status"] == "submitted"
    assert manifest["models"]["qwen38"]["job_id"] == "12345"
    assert (root / "models/qwen38/job-id.txt").read_text() == "12345\n"
    assert manifest["models"]["llama4"]["status"] == "submission_uncertain"
    assert not (root / "models/llama4/job-id.txt").exists()
    intent = json.loads((root / "models/llama4/submission-intent.json").read_text())
    assert intent["stderr"] == "temporary communication error"
    assert len(trial["calls"]) == 2


def test_ambiguous_first_response_never_retries_or_submits_second(trial):
    trial["responses"][0] = (0, "unexpected output without job ID", "")
    with pytest.raises(RuntimeError, match="unambiguous"):
        module.submit_trial(**trial["arguments"])
    assert len(trial["calls"]) == 1
    root = trial["arguments"]["output"]
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["models"]["qwen38"]["status"] == "submission_uncertain"
    assert manifest["models"]["llama4"]["status"] == "not_submitted"


def test_snapshot_preflight_fails_before_any_submission(trial):
    path = Path(trial["environment"]["SEARCH_AGENTIC_SEARXNG_SNAPSHOT"])
    row = json.loads(path.read_text())
    row["keyword"] = "different keyword"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="no shared keyword"):
        module.submit_trial(**trial["arguments"])
    assert not trial["calls"]
    manifest = json.loads((trial["arguments"]["output"] / "run_manifest.json").read_text())
    assert manifest["status"] == "preparation_failed"


def test_inherited_slurm_and_sbatch_settings_do_not_reach_submission(trial):
    trial["environment"].update({
        "SLURM_JOB_ID": "old-allocation", "SLURM_EXPORT_ENV": "NONE",
        "SBATCH_CLUSTERS": "wrong-cluster", "SBATCH_TIMELIMIT": "12:00:00",
    })
    module.submit_trial(**trial["arguments"])
    for _, kwargs in trial["calls"]:
        assert not any(key.startswith(("SLURM_", "SBATCH_")) for key in kwargs["env"])


def test_output_inside_repository_is_rejected_before_creating_or_submitting(trial):
    trial["arguments"]["output"] = module.REPOSITORY_ROOT / "generated-trial"
    with pytest.raises(ValueError, match="outside the pinned source"):
        module.submit_trial(**trial["arguments"])
    assert not trial["calls"]
    assert not trial["arguments"]["output"].exists()


def test_submission_timeout_is_uncertain_without_retry(trial):
    trial["responses"][0] = subprocess.TimeoutExpired(["sbatch"], 60)
    with pytest.raises(subprocess.TimeoutExpired):
        module.submit_trial(**trial["arguments"])
    assert len(trial["calls"]) == 1
    root = trial["arguments"]["output"]
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["models"]["qwen38"]["status"] == "submission_uncertain"
    assert manifest["models"]["llama4"]["status"] == "not_submitted"
