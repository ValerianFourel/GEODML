"""Execute the pilot wrapper with CPU-only Slurm, grammar, and server doubles."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
WRAPPER = REPOSITORY / "analysis/scripts/slurm/jupiter/run_nemotron_judge_pilot.sbatch"
QUEUE_WRAPPER = REPOSITORY / "analysis/scripts/slurm/jupiter/run_nemotron_judge_queue.sbatch"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"


def test_production_queue_preserves_non_pilot_claim_identity(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_PILOT_ONLY"] = "0"
    path = Path(env["GEODML_JUDGE_PILOT_ROOT"]) / "plan/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["pilot_only"] = False
    path.write_text(json.dumps(manifest))
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["TEST_CAPTURE"]).read_text().splitlines()]
    assert "--pilot-only" not in calls[-1]


def test_historical_fixed_pilot_cannot_disable_pilot_identity(tmp_path):
    env = _environment(tmp_path)
    env["GEODML_JUDGE_PILOT_ONLY"] = "0"
    result = _run(env)
    assert result.returncode != 0


def _environment(root: Path, *, stage_status: int = 0) -> dict[str, str]:
    repository = root / "repository"
    scripts = repository / "analysis/scripts"
    scripts.mkdir(parents=True)
    companion = scripts / "slurm/jupiter/run_nemotron_judge_pilot.sbatch"
    companion.parent.mkdir(parents=True)
    companion.write_text(WRAPPER.read_text())
    stage = scripts / "search_vllm_stage.py"
    stage.write_text(
        "import json, os, pathlib, sys, time\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['TEST_CAPTURE'], 'a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if os.environ.get('TEST_BOUNDARY_CAPTURE'):\n"
        "    pathlib.Path(os.environ['TEST_BOUNDARY_CAPTURE']).write_text(\n"
        "        os.environ.get('GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY', ''))\n"
        "if args[0] == 'prepare':\n"
        "    path = pathlib.Path(args[args.index('--profile') + 1])\n"
        "    path.write_text('{}')\n"
        "    sys.exit(int(os.environ.get('TEST_PREPARE_STATUS', '0')))\n"
        "if os.environ.get('TEST_HOLD'):\n"
        "    while not pathlib.Path(os.environ['TEST_HOLD']).exists():\n"
        "        time.sleep(0.02)\n"
        "if os.environ['TEST_STAGE_STATUS'] == '0' and not os.environ.get('TEST_MISSING_MANIFEST'):\n"
        "    output = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "    output.mkdir(parents=True, exist_ok=True)\n"
        "    queue = pathlib.Path(args[args.index('--tasks') + 1])\n"
        "    rows = [json.loads(line) for line in queue.read_text().splitlines()]\n"
        "    mode = args[args.index('--dispatch-mode') + 1] if '--dispatch-mode' in args else 'partition'\n"
        "    if '--worker-count' in args and mode == 'partition':\n"
        "        import hashlib\n"
        "        count = int(args[args.index('--worker-count') + 1])\n"
        "        index = int(args[args.index('--worker-index') + 1])\n"
        "        rows = [row for row in rows if int(hashlib.sha256(row['judge_task_id'].encode()).hexdigest(), 16) % count == index]\n"
        "    total = len(rows)\n"
        "    completed = int(os.environ.get('TEST_COMPLETED_COUNT', str(total)))\n"
        "    (output / 'run_manifest.json').write_text(json.dumps({\n"
        "        'status': os.environ.get('TEST_RUNTIME_STATUS', 'complete'),\n"
        "        'completed_count': completed, 'remaining_count': total - completed}))\n"
        "    if os.environ.get('TEST_RUNTIME_CORRUPT'):\n"
        "        (output / 'run_manifest.json').write_text('broken')\n"
        "sys.exit(int(os.environ['TEST_STAGE_STATUS']))\n"
    )
    commands = root / "bin"
    commands.mkdir()
    git = commands / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        "case \"${3:-}\" in\n"
        "  rev-parse) printf '%s\\n' \"${TEST_ACTUAL_COMMIT:-$GEODML_EXECUTION_COMMIT}\" ;;\n"
        "  status) printf '%s' \"${TEST_GIT_STATUS:-}\" ;;\n"
        "  *) exit 91 ;;\n"
        "esac\n"
    )
    git.chmod(0o755)
    flock = commands / "flock"
    flock.write_text(
        f"#!{sys.executable}\n"
        "import fcntl, sys\n"
        "try:\n"
        "    fcntl.flock(int(sys.argv[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        "    sys.exit(1)\n"
    )
    flock.chmod(0o755)
    gpu = commands / "nvidia-smi"
    gpu.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, time\n"
        "pathlib.Path(os.environ['TEST_GPU_PID']).write_text(str(os.getpid()))\n"
        "print('2026/09/16, 0, NVIDIA GH200, 100, 97871, 1, 100', flush=True)\n"
        "time.sleep(120)\n"
    )
    gpu.chmod(0o755)
    (commands / "python3").symlink_to(sys.executable)
    bash_env = root / "bash-env"
    bash_env.write_text(
        "module() {\n"
        "  [[ $- != *u* ]] || return 92\n"
        "  test -n \"${DEBUGINFOD_URLS+x}\" || return 94\n"
        "  printf '%s\\n' \"$*\" >> \"$TEST_MODULE_CALLS\"\n"
        "}\n"
    )
    activation = root / "venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text('[[ $- != *u* ]] || return 93\n')
    fake_imports = root / "imports"
    fake_imports.mkdir()
    (fake_imports / "xgrammar.py").write_text(
        "import json, os\n"
        "class Grammar:\n"
        "    @staticmethod\n"
        "    def from_json_schema(schema):\n"
        "        if os.environ.get('TEST_BAD_SCHEMA'):\n"
        "            raise RuntimeError('unsupported schema')\n"
        "        with open(os.environ['TEST_SCHEMAS'], 'a') as stream:\n"
        "            stream.write(schema + '\\n')\n"
    )
    pilot = root / "pilot"
    plan = pilot / "plan"
    plan.mkdir(parents=True)
    queue = plan / "bulk_tasks.jsonl"
    tasks = [
        {
            "judge_task_id": f"task-{i}",
            "format_version": "agentic-search-judge-v1",
            "blind_case_id": f"blind-{i}",
            "prompt_text": "Compare these tools.",
            "evidence": [{"evidence_id": "E1", "url": "https://test", "title": "tool", "text": "snippet"}],
            "answer": "Example answer",
        }
        for i in range(24)
    ]
    queue.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    manifest = {
        "format_version": "agentic-search-judge-v1",
        "scientific_result": False,
        "pilot_only": True,
        "bulk_model": {"role": "bulk", "model_id": MODEL, "model_revision": REVISION},
        "artifacts": {
            "bulk_tasks": {
                "path": str(queue),
                "sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
            }
        },
    }
    (plan / "run_manifest.json").write_text(json.dumps(manifest))
    snapshot = root / "cache" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    weights = {f"weight-{i}": f"model-{i}.safetensors" for i in range(13)}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    (snapshot / "config.json").write_text(json.dumps({"architectures": ["NemotronHForCausalLM"]}))
    for name in [*weights.values(), "tokenizer.json", "tokenizer_config.json"]:
        (snapshot / name).write_text("{}")
    return {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(fake_imports) + os.pathsep + str(REPOSITORY),
        "BASH_ENV": str(bash_env),
        "SHELLOPTS": "braceexpand:hashall:interactive-comments:nounset",
        "ACL_ARR_VENV": str(activation.parent.parent),
        "GEODML_EXECUTION_REPOSITORY": str(repository),
        "GEODML_EXECUTION_COMMIT": "a" * 40,
        "GEODML_APPROVED_WALLTIME": "01:00:00",
        "GEODML_ALLOCATION_ESTIMATE": "10-30 minutes including model startup; maximum 4 GPU-hours",
        "GEODML_JUDGE_PILOT_ROOT": str(pilot),
        "GEODML_CACHE_ROOT": str(root / "cache-root"),
        "HF_HUB_CACHE": str(root / "cache"),
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
        "SLURM_JOB_ID": "987654",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_CPUS_PER_TASK": "32",
        "SLURM_MEM_PER_NODE": "524288",
        "SLURM_GPUS_ON_NODE": "4",
        "TEST_CAPTURE": str(root / "capture"),
        "TEST_SCHEMAS": str(root / "schemas"),
        "TEST_GPU_PID": str(root / "gpu-pid"),
        "TEST_MODULE_CALLS": str(root / "modules"),
        "TEST_STAGE_STATUS": str(stage_status),
    }


def _run(env, wrapper=WRAPPER):
    return subprocess.run(
        ["bash", str(wrapper)], env=env, capture_output=True, text=True,
        check=False, timeout=15,
    )


@pytest.mark.parametrize("stage_status", [0, 2, 7])
def test_pilot_wrapper_checks_inputs_and_propagates_status(tmp_path, stage_status):
    env = _environment(tmp_path, stage_status=stage_status)
    result = _run(env)
    assert result.returncode == stage_status, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "capture").read_text().splitlines()]
    prepare, run = calls
    assert prepare[0] == "prepare"
    assert prepare[prepare.index("--tensor-parallel-size") + 1] == "4"
    assert prepare[prepare.index("--data-parallel-size") + 1] == "1"
    assert prepare[prepare.index("--model-id") + 1] == MODEL
    assert prepare[prepare.index("--model-revision") + 1] == REVISION
    assert prepare[prepare.index("--max-model-len") + 1] == "73728"
    assert "--enforce-eager" in prepare
    assert run[0] == "run"
    assert run[run.index("--startup-timeout-seconds") + 1] == "900"
    assert run[run.index("--") + 1:][:3] == [
        "python3", "analysis/scripts/run_acl_arr_vllm.py", "agentic-judge"
    ]
    assert {"--pilot-only", "--resume", "--disable-thinking"} <= set(run)
    assert run[run.index("--max-concurrency") + 1] == "4"
    assert run[run.index("--max-attempts") + 1] == "3"
    assert run[run.index("--max-output-tokens") + 1] == "2048"
    assert run[run.index("--server-model-revision") + 1] == REVISION
    assert len((tmp_path / "schemas").read_text().splitlines()) == 24
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["status"] == ("complete" if stage_status == 0 else "failed_or_interrupted")
    assert record["exit_code"] == stage_status
    assert record["finished_at_utc"]
    assert record["started_at_utc"]
    assert record["git_commit"] == "a" * 40
    assert record["model_id"] == MODEL
    assert record["model_revision"] == REVISION
    assert record["approved_walltime"] == "01:00:00"
    assert record["maximum_gpu_hours"] == 4
    assert record["scientific_result"] is False
    assert record["resources"]["SLURM_CPUS_PER_TASK"] == "32"
    assert record["task_count"] == 24
    assert (tmp_path / "modules").read_text().splitlines() == [
        "load Stages/2026 GCC Python CUDA", "load git"
    ]
    gpu_pid = tmp_path / "gpu-pid"
    if gpu_pid.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(gpu_pid.read_text()), 0)


@pytest.mark.parametrize("blocked", [
    "dirty", "commit", "approval", "resources", "logs", "model", "queue_hash",
    "task_count", "duplicate", "schema", "snapshot", "pilot_flag",
])
def test_pilot_refuses_invalid_input_before_starting_server(tmp_path, blocked):
    env = _environment(tmp_path)
    plan = tmp_path / "pilot/plan"
    manifest_path = plan / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if blocked == "dirty":
        env["TEST_GIT_STATUS"] = " M file.py"
    elif blocked == "commit":
        env["TEST_ACTUAL_COMMIT"] = "b" * 40
    elif blocked == "approval":
        env["GEODML_APPROVED_WALLTIME"] = "12:00:00"
    elif blocked == "resources":
        env["SLURM_JOB_NUM_NODES"] = "2"
    elif blocked == "logs":
        logs = tmp_path / "pilot/logs"
        logs.mkdir()
        (logs / "server.log").write_text("preserve")
    elif blocked == "model":
        manifest["bulk_model"]["model_revision"] = "b" * 40
    elif blocked == "pilot_flag":
        manifest["pilot_only"] = False
    elif blocked == "queue_hash":
        manifest["artifacts"]["bulk_tasks"]["sha256"] = "b" * 64
    elif blocked in {"task_count", "duplicate"}:
        queue = plan / "bulk_tasks.jsonl"
        lines = queue.read_text().splitlines()
        if blocked == "duplicate":
            lines[-1] = lines[0]
        else:
            lines = lines[:-1]
        queue.write_text("\n".join(lines) + "\n")
        manifest["artifacts"]["bulk_tasks"]["sha256"] = hashlib.sha256(queue.read_bytes()).hexdigest()
    elif blocked == "schema":
        env["TEST_BAD_SCHEMA"] = "1"
    elif blocked == "snapshot":
        cache = tmp_path / "cache" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
        (cache / "model-0.safetensors").unlink()
    manifest_path.write_text(json.dumps(manifest))
    result = _run(env)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "pilot/logs/allocation.json").exists()


def test_prepare_failure_is_recorded_without_starting_server(tmp_path):
    env = _environment(tmp_path)
    env["TEST_PREPARE_STATUS"] = "9"
    result = _run(env)
    assert result.returncode == 9, result.stderr
    assert len((tmp_path / "capture").read_text().splitlines()) == 1
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["exit_code"] == 9
    assert record["status"] == "failed_or_interrupted"


def test_wrapper_lock_rejects_duplicate_job(tmp_path):
    env = _environment(tmp_path)
    release = tmp_path / "release"
    env["TEST_HOLD"] = str(release)
    first = subprocess.Popen(
        ["bash", str(WRAPPER)], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        capture = tmp_path / "capture"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert first.poll() is None, "first worker exited before server startup"
            if capture.exists() and len(capture.read_text().splitlines()) == 2:
                break
            time.sleep(0.02)
        assert capture.exists() and len(capture.read_text().splitlines()) == 2
        second = _run({**env, "SLURM_JOB_ID": "987655"})
        assert second.returncode == 75, second.stderr
        assert "already has an active worker" in second.stderr
        assert len(capture.read_text().splitlines()) == 2
    finally:
        release.touch()
        _, stderr = first.communicate(timeout=10)
    assert first.returncode == 0, stderr


def test_wrapper_has_valid_shell_and_no_allocation_defaults():
    for wrapper in (WRAPPER, QUEUE_WRAPPER):
        text = wrapper.read_text()
        assert "#SBATCH" not in text
        assert "sbatch " not in text
        result = subprocess.run(
            ["bash", "-n", str(wrapper)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


def _throughput_environment(root):
    env = _environment(root)
    env["GEODML_JUDGE_CLAIM_ROOT"] = str(root / "shared-judgments")
    env["GEODML_APPROVED_WALLTIME"] = "00:30:00"
    env["SLURM_JOB_START_TIME"] = str(int(time.time()))
    env["SLURM_JOB_END_TIME"] = str(int(time.time()) + 1800)
    plan = root / "pilot/plan"
    queue = plan / "bulk_tasks.jsonl"
    rows = [json.loads(line) for line in queue.read_text().splitlines()]
    for index in range(24, 36):
        rows.append({**rows[0], "judge_task_id": f"task-{index}", "blind_case_id": f"blind-{index}"})
    queue.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = plan / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["bulk_tasks"]["sha256"] = hashlib.sha256(queue.read_bytes()).hexdigest()
    manifest["pilot"] = {"execution_mode": "throughput"}
    manifest["summary"] = {
        "pending_task_count": 36, "bulk_task_count": 36, "available_task_count": 60,
        "excluded_task_count": 24,
    }
    manifest_path.write_text(json.dumps(manifest))
    return env


def _recorded_queue(root, env, *, prompt_tokens):
    from analysis.tests.test_agentic_judge_transcript import _plan, _recorded_result

    source = root / "source"
    result, prompt, _ = _recorded_result(source)
    task, = _plan(source, result, prompt, recorded_conversation=True).bulk_tasks
    plan = root / "pilot/plan"
    queue = plan / "bulk_tasks.jsonl"
    queue.write_text(json.dumps(task.to_dict()) + "\n")
    manifest_path = plan / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = task.format_version
    manifest["artifacts"]["bulk_tasks"]["sha256"] = hashlib.sha256(queue.read_bytes()).hexdigest()
    manifest["summary"] = {"pending_task_count": 1, "bulk_task_count": 1,
                           "available_task_count": 1, "excluded_task_count": 0}
    manifest_path.write_text(json.dumps(manifest))
    (root / "imports/transformers.py").write_text(
        "import json, os\n"
        "class AutoTokenizer:\n"
        "    @staticmethod\n"
        "    def from_pretrained(path, **kwargs):\n"
        "        assert kwargs == {'local_files_only': True, 'trust_remote_code': True}\n"
        "        return AutoTokenizer()\n"
        "    def apply_chat_template(self, messages, **kwargs):\n"
        "        assert kwargs['truncation'] is False\n"
        "        assert kwargs['enable_thinking'] is False\n"
        "        assert 'RECORDED CONVERSATION' in messages[0]['content']\n"
        f"        return {{'input_ids': [1] * {prompt_tokens}}}\n"
    )


def test_full_conversation_context_overflow_stops_before_gpu_start(tmp_path):
    env = _throughput_environment(tmp_path)
    _recorded_queue(tmp_path, env, prompt_tokens=72000)
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode != 0
    assert "context" in result.stderr.lower(), result.stderr
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "gpu-pid").exists()


def test_full_conversation_context_observed_maximum_fits_without_truncation(tmp_path):
    env = _throughput_environment(tmp_path)
    _recorded_queue(tmp_path, env, prompt_tokens=64841)
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "capture").read_text().splitlines()]
    prepare, _ = calls
    assert prepare[prepare.index("--max-model-len") + 1] == "73728"
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["context_budget"]["max_prompt_tokens"] == 64841
    assert record["context_budget"]["max_required_tokens"] == 66889
    assert record["context_budget"]["max_model_len"] == 73728


def test_full_conversation_context_budget_is_recorded(tmp_path):
    env = _throughput_environment(tmp_path)
    _recorded_queue(tmp_path, env, prompt_tokens=1000)
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["context_budget"]["max_prompt_tokens"] == 1000
    assert record["context_budget"]["max_required_tokens"] == 3048


def test_queue_uses_variable_approved_time_and_reports_checkpoint(tmp_path):
    env = _throughput_environment(tmp_path)
    env["TEST_RUNTIME_STATUS"] = "checkpointed"
    env["TEST_COMPLETED_COUNT"] = "17"
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["status"] == "checkpointed"
    assert record["completed_count"] == 17
    assert record["remaining_count"] == 19
    assert record["task_count"] == 36
    assert record["available_task_count"] == 60
    assert record["excluded_task_count"] == 24
    assert record["execution_mode"] == "throughput"
    assert record["approved_walltime"] == "00:30:00"
    assert record["maximum_gpu_hours"] == 2
    assert record["allocation_budget"]["policy"] == "fill-approved-queue-v1"
    assert record["allocation_budget"]["end_epoch"] == int(env["SLURM_JOB_END_TIME"])
    assert len((tmp_path / "schemas").read_text().splitlines()) == 36


def test_queue_stops_after_exhausting_tasks(tmp_path):
    env = _throughput_environment(tmp_path)
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "capture").read_text().splitlines()
    assert len(calls) == 2
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["status"] == "complete"
    assert record["remaining_count"] == 0


def test_queue_passes_shared_claim_root_and_slot_to_runner(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_CLAIM_ROOT"] = str(tmp_path / "shared-judgments")
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    run = json.loads((tmp_path / "capture").read_text().splitlines()[1])
    assert run[run.index("--claim-root") + 1] == env["GEODML_JUDGE_CLAIM_ROOT"]
    assert run[run.index("--worker-index") + 1] == "0"
    assert run[run.index("--worker-count") + 1] == "1"


def test_queue_uses_verified_exclusive_slurm_boundary(tmp_path):
    env = _throughput_environment(tmp_path)
    env["TEST_BOUNDARY_CAPTURE"] = str(tmp_path / "boundary")
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "boundary").read_text() == "1"


def test_parallel_queue_attempts_have_separate_logs_and_same_plan(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_CLAIM_ROOT"] = str(tmp_path / "shared-judgments")
    env["GEODML_JUDGE_QUEUE_ROOT"] = str(tmp_path / "pilot")
    for job_id in ("987654", "987655"):
        result = _run({**env, "SLURM_JOB_ID": job_id}, QUEUE_WRAPPER)
        assert result.returncode == 0, result.stderr
        attempt = tmp_path / "pilot/attempts" / ("job" + job_id + "-worker0")
        record = json.loads((attempt / "logs/allocation.json").read_text())
        assert record["output"] == str(attempt / "nemotron")
        assert record["claim_root"] == env["GEODML_JUDGE_CLAIM_ROOT"]
    calls = [json.loads(line) for line in (tmp_path / "capture").read_text().splitlines()]
    for run in calls[1::2]:
        assert run[run.index("--tasks") + 1] == str(tmp_path / "pilot/plan/bulk_tasks.jsonl")


def test_queue_requires_shared_claim_root_before_model_load(tmp_path):
    env = _throughput_environment(tmp_path)
    env.pop("GEODML_JUDGE_CLAIM_ROOT", None)
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()


def test_queue_rejects_relative_claim_root(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_CLAIM_ROOT"] = "relative-claims"
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()


def test_queue_records_expanded_slurm_log_paths(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_STDOUT"] = str(tmp_path / "logs/slurm-%A_%a-%j.out")
    env["GEODML_JUDGE_STDERR"] = str(tmp_path / "logs/slurm-%j.err")
    env["SLURM_ARRAY_JOB_ID"] = "987650"
    env["SLURM_ARRAY_TASK_ID"] = "0"
    env["SLURM_ARRAY_TASK_COUNT"] = "1"
    env["SLURM_ARRAY_TASK_MIN"] = "0"
    env["SLURM_ARRAY_TASK_MAX"] = "0"
    env["SLURM_ARRAY_TASK_STEP"] = "1"
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["stdout"] == str(tmp_path / "logs/slurm-987650_0-987654.out")
    assert record["stderr"] == str(tmp_path / "logs/slurm-987654.err")


def test_queue_rejects_nonzero_based_automatic_array(tmp_path):
    env = _throughput_environment(tmp_path)
    env.update(SLURM_ARRAY_TASK_ID="1", SLURM_ARRAY_TASK_MIN="1",
               SLURM_ARRAY_TASK_MAX="3", SLURM_ARRAY_TASK_COUNT="3",
               SLURM_ARRAY_TASK_STEP="1")
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()


def test_queue_slot_count_matches_runner_and_stops_invalid_slots(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_WORKER_COUNT"] = "3"
    env["GEODML_JUDGE_WORKER_INDEX"] = "1"
    env["GEODML_JUDGE_DISPATCH_MODE"] = "partition"
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    expected = sum(int(hashlib.sha256(f"task-{i}".encode()).hexdigest(), 16) % 3 == 1 for i in range(36))
    assert record["task_count"] == record["completed_count"] == expected
    assert record["source_task_count"] == 36
    assert record["worker_index"] == 1 and record["worker_count"] == 3
    assert record["remaining_count"] == 0


def test_queue_default_backlog_does_not_strand_tasks_in_other_slots(tmp_path):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_WORKER_COUNT"] = "3"
    env["GEODML_JUDGE_WORKER_INDEX"] = "1"
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["task_count"] == record["completed_count"] == 36
    assert record["dispatch_mode"] == "backlog"
    run = json.loads((tmp_path / "capture").read_text().splitlines()[-1])
    assert run[run.index("--dispatch-mode") + 1] == "backlog"


def test_queue_empty_preferred_slot_can_process_other_slots(tmp_path):
    env = _throughput_environment(tmp_path)
    count = 1000
    occupied = {int(hashlib.sha256(f"task-{i}".encode()).hexdigest(), 16) % count for i in range(36)}
    env["GEODML_JUDGE_WORKER_COUNT"] = str(count)
    env["GEODML_JUDGE_WORKER_INDEX"] = str(next(index for index in range(count) if index not in occupied))
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["completed_count"] == 36


@pytest.mark.parametrize("worker_index,worker_count", [("2", "2"), ("-1", "2"), ("0", "0")])
def test_queue_refuses_invalid_slot_before_server(tmp_path, worker_index, worker_count):
    env = _throughput_environment(tmp_path)
    env["GEODML_JUDGE_WORKER_COUNT"] = worker_count
    env["GEODML_JUDGE_WORKER_INDEX"] = worker_index
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()


def test_queue_wrapper_runs_from_slurm_spool_copy(tmp_path):
    env = _throughput_environment(tmp_path)
    spool = tmp_path / "slurm-spool/job987654/slurm_script"
    spool.parent.mkdir(parents=True)
    spool.write_text(QUEUE_WRAPPER.read_text())
    assert not (spool.parent / WRAPPER.name).exists()
    result = _run(env, spool)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["status"] == "complete"
    assert record["execution_mode"] == "throughput"


@pytest.mark.parametrize("problem", ["missing_end", "expired", "too_late", "fixed_plan", "counts"])
def test_queue_refuses_invalid_budget_or_plan_before_server(tmp_path, problem):
    env = _throughput_environment(tmp_path)
    if problem == "missing_end":
        env.pop("SLURM_JOB_END_TIME")
    elif problem == "expired":
        env["SLURM_JOB_END_TIME"] = str(int(time.time()) - 1)
    elif problem == "too_late":
        env["SLURM_JOB_END_TIME"] = str(int(time.time()) + 60)
    else:
        path = tmp_path / "pilot/plan/run_manifest.json"
        manifest = json.loads(path.read_text())
        if problem == "fixed_plan":
            manifest["pilot"]["execution_mode"] = "fixed"
        else:
            manifest["summary"]["excluded_task_count"] = 25
        path.write_text(json.dumps(manifest))
    result = _run(env, QUEUE_WRAPPER)
    assert result.returncode != 0
    assert not (tmp_path / "capture").exists()
    assert not (tmp_path / "gpu-pid").exists()


@pytest.mark.parametrize("problem", ["missing", "incomplete_complete", "corrupt"])
def test_success_without_complete_or_checkpointed_manifest_is_failure(tmp_path, problem):
    env = _environment(tmp_path)
    if problem == "missing":
        env["TEST_MISSING_MANIFEST"] = "1"
    elif problem == "corrupt":
        env["TEST_RUNTIME_CORRUPT"] = "1"
    else:
        env["TEST_COMPLETED_COUNT"] = "2"
    result = _run(env)
    assert result.returncode == 2, result.stderr
    record = json.loads((tmp_path / "pilot/logs/allocation.json").read_text())
    assert record["status"] == "failed_or_interrupted"
    assert record["exit_code"] == 2
