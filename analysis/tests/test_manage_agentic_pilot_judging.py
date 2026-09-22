"""CPU-only original-pilot preparation, ownership, and allocation contracts."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import manage_agentic_pilot_judging as manager
from analysis.tests.test_agentic_judge_transcript import _pilot_source


class Tokenizer:
    length = 100

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["enable_thinking"] is False
        return {"input_ids": list(range(self.length))}


@pytest.fixture
def source(tmp_path, monkeypatch):
    original = _pilot_source(tmp_path)
    study = tmp_path / "study"
    for alias, model in manager.GENERATORS.items():
        shard = study / "models" / alias / "shard-0"
        shutil.copytree(original["generator_root"], shard)
        manifest = json.loads((shard / "run_manifest.json").read_text())
        manifest["model_id"] = model
        (shard / "run_manifest.json").write_text(json.dumps(manifest))
    tasks = tmp_path / "generation.jsonl"
    task_rows = []
    for path in sorted((original["generator_root"] / "results").glob("*.json")):
        value = json.loads(path.read_text())
        task_rows.append(
            {
                k: value[k]
                for k in (
                    "cell_id",
                    "prompt_id",
                    "prompt_sha256",
                    "method",
                    "engine",
                    "condition",
                )
            }
        )
    tasks.write_text("".join(json.dumps(row) + "\n" for row in task_rows))
    adaptive = tmp_path / "adaptive.json"
    adaptive.write_text(
        json.dumps(
            {
                "original_study": {
                    "root": str(study),
                    "generation_tasks": manager._file_identity(tasks),
                    "prompt_sources": {
                        "prompts_jsonl": manager._file_identity(
                            original["prompts_jsonl"]
                        ),
                        "selection_records_jsonl": manager._file_identity(
                            original["selection_records_jsonl"]
                        ),
                    },
                },
                "judge": {
                    "bulk_model": {
                        "model_id": manager.MODEL,
                        "model_revision": manager.REVISION,
                    },
                    "claim_root": str(tmp_path / "claims"),
                    "validation_model": {
                        "model_id": "validator",
                        "model_revision": "b" * 40,
                    },
                },
            }
        )
    )
    snapshot = (
        tmp_path
        / "hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots"
        / manager.REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["NemotronHForCausalLM"]})
    )
    weights = {f"layer-{i}": f"model-{i}.safetensors" for i in range(13)}
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    for name in [*weights.values(), "tokenizer.json", "tokenizer_config.json"]:
        (snapshot / name).write_text("{}")
    monkeypatch.setattr(manager, "EXPECTED_CELLS", 12)
    monkeypatch.setattr(manager, "EXPECTED_PROMPTS", 1)
    monkeypatch.setattr(manager, "load_tokenizer", lambda _: Tokenizer())
    import sys

    monkeypatch.setitem(
        sys.modules,
        "xgrammar",
        SimpleNamespace(Grammar=SimpleNamespace(from_json_schema=json.loads)),
    )
    output = tmp_path / "judge"
    return adaptive, output, snapshot


def make(source):
    return manager.prepare(*source, "08:00:00", "a" * 40)


def test_context_only_needs_no_allocation_approval_and_writes_nothing(
    source, monkeypatch
):
    (source[2] / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "max_position_embeddings": 131072,
            }
        )
    )
    monkeypatch.setattr(Tokenizer, "length", 79823)
    before = {p: p.read_bytes() for p in source[0].parent.rglob("*") if p.is_file()}

    def no_launch(*args, **kwargs):
        pytest.fail("Context audit must not launch subprocesses")

    monkeypatch.setattr(manager.subprocess, "run", no_launch)
    report = manager.prepare(
        *source, None, "a" * 40, max_model_len=None, context_only=True
    )
    assert report["status"] == "context_measured_not_launch_approved"
    assert report["task_count"] == 24
    assert report["max_required_tokens"] == 81871
    assert report["max_model_len"] == 81920
    assert report["requires_walltime_approval"] is True
    assert not source[1].exists()
    assert before == {
        p: p.read_bytes() for p in source[0].parent.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="approved"):
        manager.prepare(*source, None, "a" * 40)


def test_full_prepare_retains_validation_and_original_sources(source):
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source[0].parent.rglob("*.json")
    }
    receipt = make(source)
    assert receipt["task_count"] == 24
    assert receipt["approved_allocation"]["maximum_gpu_hours"] == 64
    assert receipt["validation_task_count"] > 0
    assert receipt["validation_model"]["model_id"] == "validator"
    assert receipt["partition_counts"][0] > 0
    assert sum(receipt["partition_counts"]) == 24
    assert manager.verify(source[1]) == receipt
    for path, digest in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    manifest = json.loads((source[1] / "plan/run_manifest.json").read_text())
    assert manifest["pilot_only"] is False
    assert manifest["validation_fraction"] == 0.02
    assert manifest["format_version"] == "agentic-search-judge-recorded-conversation-v2"
    report = manager.audit(source[1])
    assert report["claims"] == {"missing": 24}
    assert not report["bulk_complete"]
    with pytest.raises(FileExistsError):
        make(source)


@pytest.mark.parametrize(
    "damage", ["missing", "identity", "snapshot", "hash", "trace", "context"]
)
def test_bad_sources_fail_before_creating_queue(source, monkeypatch, damage):
    adaptive, output, snapshot = source
    plan = json.loads(adaptive.read_text())
    shard = Path(plan["original_study"]["root"]) / "models/llama4/shard-0"
    result = next((shard / "results").glob("*.json"))
    if damage == "missing":
        result.unlink()
    elif damage in {"identity", "trace"}:
        value = json.loads(result.read_text())
        value["prompt_id" if damage == "identity" else "trace_sha256"] = "changed"
        result.write_text(json.dumps(value))
    elif damage == "snapshot":
        (snapshot / "model-0.safetensors").unlink()
    elif damage == "hash":
        Path(plan["original_study"]["generation_tasks"]["path"]).write_text("changed")
    else:
        monkeypatch.setattr(Tokenizer, "length", 73728)
    with pytest.raises((ValueError, FileNotFoundError)):
        make(source)
    assert not output.exists()


def test_changed_queue_and_cache_fail_verification(source):
    make(source)
    task_path = source[1] / "plan/bulk_tasks.jsonl"
    original = task_path.read_bytes()
    task_path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        manager.verify(source[1])
    task_path.write_bytes(original)
    (source[2] / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="snapshot changed"):
        manager.verify(source[1])


def test_submission_is_two_one_node_jobs_and_never_automatically_repeated(
    source, monkeypatch
):
    make(source)
    monkeypatch.setattr(
        manager.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40 + "\n",
    )
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr(manager.subprocess, "run", run)
    manager.submit(source[1], "scifi", "08:00:00")
    command, kwargs = calls[0]
    for flag in (
        "--array=0-1%2",
        "--nodes=1",
        "--time=08:00:00",
        "--gpus-per-node=4",
        "--no-requeue",
    ):
        assert flag in command
    assert kwargs["env"]["GEODML_EXECUTION_REPOSITORY"] == str(manager.REPOSITORY)
    with pytest.raises(FileExistsError):
        manager.submit(source[1], "scifi", "08:00:00")
    assert len(calls) == 1


def test_failed_sbatch_preserves_intent_and_needs_manual_review(source, monkeypatch):
    make(source)
    monkeypatch.setattr(
        manager.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40,
    )
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="scheduler unavailable"
        ),
    )
    with pytest.raises(RuntimeError, match="no automatic retry"):
        manager.submit(source[1], "scifi", "08:00:00")
    assert (source[1] / "submission-intent.json").exists()
    assert (
        json.loads((source[1] / "submission-result.json").read_text())["returncode"]
        == 1
    )


def test_approval_is_required_and_complete_queue_never_submits(source, monkeypatch):
    with pytest.raises(ValueError, match="approved"):
        manager.prepare(*source, "07:00:00", "a" * 40)
    make(source)
    with pytest.raises(ValueError, match="approved"):
        manager.submit(source[1], "scifi", "07:00:00")
    monkeypatch.setattr(
        manager.subprocess,
        "check_output",
        lambda args, **kw: "" if "status" in args else "a" * 40,
    )
    monkeypatch.setattr(manager, "audit", lambda root: {"bulk_complete": True})
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("complete queue submitted"),
    )
    manager.submit(source[1], "scifi", "08:00:00")
    assert not (source[1] / "submission-intent.json").exists()


@pytest.mark.parametrize("max_model_len", [73728, 90112])
def test_worker_uses_controller_deadline_and_offline_partition_environment(
    source, monkeypatch, max_model_len
):
    (source[2] / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "max_position_embeddings": 131072,
            }
        )
    )
    manager.prepare(*source, "08:00:00", "a" * 40, max_model_len=max_model_len)
    if max_model_len != 73728:
        (source[1] / "submission-intent.json").write_text(
            json.dumps(
                {
                    "max_model_len": max_model_len,
                    "approval": manager.APPROVAL,
                    "estimate": "Synthetic test approval, not a production estimate",
                }
            )
        )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "101")
    monkeypatch.setenv("SLURM_JOB_ID", "102")
    monkeypatch.setenv("GEODML_ROLE_END_TIME", "stale")
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    calls = []

    def check(args, **kw):
        if args[0] == "scontrol":
            assert args[-1] == "101_0"
            return (
                f"JobId=102 JobState=RUNNING NumNodes=1 TimeLimit=08:00:00 "
                f"UserId={os.environ['USER']}(123) OverSubscribe=NO ArrayJobId=101 ArrayTaskId=0 "
                "AllocTRES=cpu=288,node=1,gres/gpu=4 StartTime=start EndTime=end"
            )
        if args[0] == "date":
            return "1000\n" if args[2] == "start" else "29800\n"
        return "uuid0\nuuid1\nuuid2\nuuid3\n"

    monkeypatch.setattr(manager.subprocess, "check_output", check)
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda args, **kw: calls.append(kw) or SimpleNamespace(returncode=0),
    )
    assert manager.worker(source[1]) == 0
    env = calls[0]["env"]
    assert env["SLURM_JOB_END_TIME"] == "29800"
    assert env["SLURM_JOB_START_TIME"] == "1000"
    assert env["SLURM_GPUS_ON_NODE"] == "4"
    assert env["GEODML_JUDGE_DISPATCH_MODE"] == "partition"
    assert env["GEODML_JUDGE_PILOT_ONLY"] == "0"
    assert env["GEODML_JUDGE_WORKER_COUNT"] == "2"
    assert env["GEODML_JUDGE_MAX_MODEL_LEN"] == str(max_model_len)
    if max_model_len != 73728:
        assert env["GEODML_ALLOCATION_ESTIMATE"].startswith("Synthetic test")
    assert "GEODML_ROLE_END_TIME" not in env
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(env["HF_HUB_CACHE"]) == source[2].parents[2]
    assert calls[0]["pass_fds"]


def test_slurm_spool_wrapper_uses_pinned_checkout():
    wrapper = (
        manager.REPOSITORY
        / "analysis/scripts/slurm/jupiter/run_agentic_pilot_judging.sbatch"
    )
    text = wrapper.read_text()
    assert "BASH_SOURCE" not in text
    assert (
        '"$GEODML_EXECUTION_REPOSITORY/analysis/scripts/manage_agentic_pilot_judging.py"'
        in text
    )


def test_long_context_is_frozen_and_verified_without_truncation(source, monkeypatch):
    config = source[2] / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "max_position_embeddings": 131072,
            }
        )
    )
    monkeypatch.setattr(Tokenizer, "length", 79823)
    receipt = manager.prepare(*source, "08:00:00", "a" * 40, max_model_len=90112)
    assert receipt["max_model_len"] == 90112
    assert receipt["requires_context_reapproval"] is True
    context = json.loads((source[1] / "context-budget.json").read_text())
    assert context["max_required_tokens"] == 81871
    assert context["max_model_len"] == 90112
    assert manager.verify(source[1]) == receipt
    with pytest.raises(ValueError, match="context.*approval"):
        manager.submit(source[1], "scifi", "08:00:00")
    assert not (source[1] / "submission-intent.json").exists()


@pytest.mark.parametrize("native", [None, 80000])
def test_context_extension_requires_cached_native_capacity(source, native):
    config = {"architectures": ["NemotronHForCausalLM"]}
    if native is not None:
        config["max_position_embeddings"] = native
    (source[2] / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="context"):
        manager.prepare(*source, "08:00:00", "a" * 40, max_model_len=90112)
    assert not source[1].exists()


def test_auto_context_measures_all_tasks_and_keeps_task_bytes(source, monkeypatch):
    config = source[2] / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "max_position_embeddings": 131072,
            }
        )
    )
    original = make(source)
    task_bytes = (source[1] / "plan/bulk_tasks.jsonl").read_bytes()
    calls = []

    class VaryingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            calls.append(messages)
            self.length = 79823 if len(calls) < 24 else 97000
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(manager, "load_tokenizer", lambda _: VaryingTokenizer())
    output = source[1].with_name("long-judge")
    receipt = manager.prepare(
        source[0], output, source[2], "08:00:00", "a" * 40, max_model_len=None
    )
    assert len(calls) == original["task_count"] == 24
    assert receipt["max_model_len"] == 102400
    assert (output / "plan/bulk_tasks.jsonl").read_bytes() == task_bytes


def test_fixed_context_reports_full_queue_not_first_overflow(source, monkeypatch):
    calls = []

    class OversizedTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            calls.append(messages)
            self.length = 79823
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(manager, "load_tokenizer", lambda _: OversizedTokenizer())
    with pytest.raises(ValueError, match="Measured all 24.*maximum required=81871"):
        make(source)
    assert len(calls) == 24
    assert not source[1].exists()


def test_wrapper_executes_from_slurm_spool(tmp_path):
    wrapper = (
        manager.REPOSITORY
        / "analysis/scripts/slurm/jupiter/run_agentic_pilot_judging.sbatch"
    )
    spool = tmp_path / "slurm_script"
    spool.write_text(wrapper.read_text())
    environment = tmp_path / "environment"
    environment.write_text('export ACL_ARR_VENV="$TEST_VENV"\n')
    activation = tmp_path / "venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(":\n")
    commands = tmp_path / "bash-env"
    capture = tmp_path / "capture"
    commands.write_text(
        'module() { :; }\npython3() { printf "%s\\n" "$@" > "$TEST_CAPTURE"; }\nexport -f python3\n'
    )
    # exec does not invoke shell functions. Install a small executable double.
    python = activation.parent / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TEST_CAPTURE"\n')
    python.chmod(0o755)
    env = dict(
        os.environ,
        BASH_ENV=str(commands),
        TEST_VENV=str(activation.parent.parent),
        ACL_ARR_ENVIRONMENT_FILE=str(environment),
        GEODML_EXECUTION_REPOSITORY="/pinned/checkout",
        GEODML_CACHE_ROOT=str(tmp_path / "cache"),
        TEST_CAPTURE=str(capture),
        PATH=str(activation.parent) + os.pathsep + os.environ["PATH"],
    )
    result = subprocess.run(
        ["bash", str(spool), "/frozen/queue"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        "/pinned/checkout/analysis/scripts/manage_agentic_pilot_judging.py",
        "worker",
        "--run-root",
        "/frozen/queue",
    ]
