"""Prepare, submit once, and audit the original 500-prompt Nemotron queue.

Preparation is CPU-only. Submission is explicit and requires the separately
approved two one-node, eight-hour allocations. Existing generation is read-only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline.agentic_judging import (
    AgenticJudgeModel,
    _atomic_json,
    _file_identity,
    build_agentic_judge_plan,
    write_agentic_judge_plan,
)
from analysis.interpretability.pipeline.inference_claims import InferenceClaimStore
from analysis.scripts.check_agentic_judge_context import (
    check_agentic_judge_context_budget,
)
from analysis.scripts.prepare_agentic_judge_tasks import (
    _completed_results,
    _merged_prompts,
)
from analysis.scripts.run_acl_arr_vllm import (
    _agentic_judge_context,
    _prepare_agentic_judge,
    _shared_claim_identity,
    _validate_shared_failure,
    _validate_shared_outcome,
    task_in_worker,
)

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
GENERATORS = {
    "qwen38": "Qwen/Qwen3.8-27B",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}
EXPECTED_CELLS = 6000
EXPECTED_PROMPTS = 500
APPROVAL = {
    "allocation_count": 2,
    "nodes_per_allocation": 1,
    "gpus_per_node": 4,
    "cpus_per_node": 32,
    "memory_per_node": "512G",
    "walltime": "08:00:00",
    "maximum_gpu_hours": 64,
    "admission_margin_seconds": 120,
    "cleanup_margin_seconds": 45,
}
ESTIMATE = (
    "Approved two independent one-node eight-hour Nemotron workers. "
    "Estimated 5-7 hours assuming comparable lengths to 696 judgments in "
    "34m39s; 8 hours includes startup and variability. Four GH200 GPUs, "
    "32 CPUs, 512G per node; maximum 64 aggregate GPU-hours. No requeue, "
    "extension or replacement. Frozen original-500 judging only."
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_file(entry):
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Frozen input hash mismatch: {path}")
    return path


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def snapshot_files(snapshot):
    if snapshot.name != REVISION:
        raise ValueError(
            "Snapshot directory must identify the pinned Nemotron revision"
        )
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("architectures") != ["NemotronHForCausalLM"]:
        raise ValueError("Unexpected cached Nemotron architecture")
    weights = sorted(
        set(
            json.loads((snapshot / "model.safetensors.index.json").read_text())[
                "weight_map"
            ].values()
        )
    )
    if len(weights) != 13:
        raise ValueError("Expected all 13 Nemotron weight shards")
    sizes = {}
    for name in [
        *weights,
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        path = snapshot / name
        if Path(name).name != name or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty cached model file: {name}")
        sizes[name] = path.stat().st_size
    # Weights are not read/hashed on the login node. Pin revision, shard coverage
    # and sizes; hash the small configuration/tokenizer files used by preflight.
    return {
        "path": str(snapshot),
        "file_sizes": sizes,
        "metadata": [
            _file_identity(snapshot / name)
            for name in sizes
            if not name.endswith(".safetensors")
        ],
    }


def load_tokenizer(snapshot):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )


def prepare(adaptive_path, output, snapshot, approved_walltime, commit):
    if approved_walltime != APPROVAL["walltime"]:
        raise ValueError(
            "This preparation requires the explicitly approved 08:00:00 budget"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Execution commit must be an immutable Git SHA")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite queue: {output}; use verify/status"
        )
    adaptive = json.loads(adaptive_path.read_text())
    study, judge = adaptive["original_study"], adaptive["judge"]
    if {key: judge["bulk_model"][key] for key in ("model_id", "model_revision")} != {
        "model_id": MODEL,
        "model_revision": REVISION,
    }:
        raise ValueError("Adaptive plan uses a different bulk judge")
    task_path = checked_file(study["generation_tasks"])
    prompt_path = checked_file(study["prompt_sources"]["prompts_jsonl"])
    selection_path = checked_file(study["prompt_sources"]["selection_records_jsonl"])
    task_rows = rows(task_path)
    expected = {row["cell_id"]: row for row in task_rows}
    prompt_rows = _merged_prompts(prompt_path, selection_path)
    if len(expected) != EXPECTED_CELLS or len(task_rows) != EXPECTED_CELLS:
        raise ValueError(
            "Original pilot must have exactly 6000 unique generation cells"
        )
    if (
        len(prompt_rows) != EXPECTED_PROMPTS
        or len({p["candidate_id"] for p in prompt_rows}) != EXPECTED_PROMPTS
    ):
        raise ValueError("Original pilot must have exactly 500 unique prompts")
    for row in prompt_rows:
        row["axis_bin"] = str(row["axis_bin"])
    roots, results = {}, []
    for alias, model in GENERATORS.items():
        root = Path(study["root"]) / "models" / alias
        paths = _completed_results(root, expected_model_id=model)
        seen = set()
        for path in paths:
            result = json.loads(path.read_text())
            cell = result["cell_id"]
            if cell in seen or cell not in expected or path.stem != cell:
                raise ValueError(f"Duplicate or out-of-pilot result: {path}")
            if any(
                result.get(k) != expected[cell].get(k)
                for k in (
                    "cell_id",
                    "prompt_id",
                    "prompt_sha256",
                    "method",
                    "engine",
                    "condition",
                )
            ):
                raise ValueError(f"Result identity differs from frozen cell: {path}")
            seen.add(cell)
        if seen != set(expected):
            raise ValueError(f"Incomplete generator: {alias}")
        roots[str(root)] = model
        results.extend(paths)
        print(f"GENERATOR_VERIFIED={alias} CELLS={len(seen)}", flush=True)
    validation = judge.get("validation_model")
    plan = build_agentic_judge_plan(
        results,
        prompt_rows=prompt_rows,
        generator_model_by_root=roots,
        bulk_model=AgenticJudgeModel("bulk", MODEL, REVISION),
        validation_model=None
        if validation is None
        else AgenticJudgeModel(
            "validation", validation["model_id"], validation["model_revision"]
        ),
        validation_fraction=0 if validation is None else 0.02,
        master_seed=20260915,
        recorded_conversation=True,
    )
    if len(plan.bulk_tasks) != 2 * EXPECTED_CELLS:
        raise ValueError("Judge coverage differs from both complete generators")
    cache = snapshot_files(snapshot)
    tokenizer = load_tokenizer(snapshot)
    # Bounded batches give visible progress without changing any request.
    context = []
    for start in range(0, len(plan.bulk_tasks), 100):
        batch = check_agentic_judge_context_budget(
            plan.bulk_tasks[start : start + 100],
            tokenizer=tokenizer,
            max_model_len=73728,
            max_output_tokens=2048,
            disable_thinking=True,
        )
        context.extend(batch["tasks"])
        print(f"CONTEXT_VERIFIED={len(context)}/{len(plan.bulk_tasks)}", flush=True)
    import xgrammar

    from analysis.interpretability.pipeline.agentic_judging import agentic_judge_schema

    for task in plan.bulk_tasks:
        xgrammar.Grammar.from_json_schema(
            json.dumps(agentic_judge_schema([e.evidence_id for e in task.evidence]))
        )
    # No output directory is created until all source/context/schema checks pass.
    artifacts = write_agentic_judge_plan(
        output / "plan", plan=plan, generation_tasks_path=task_path
    )
    manifest = json.loads(artifacts.manifest_path.read_text())
    manifest.update(pilot_only=False, pilot={"execution_mode": "throughput"})
    manifest["summary"].update(
        available_task_count=len(plan.bulk_tasks),
        pending_task_count=len(plan.bulk_tasks),
        excluded_task_count=0,
    )
    _atomic_json(artifacts.manifest_path, manifest)
    _atomic_json(
        output / "context-budget.json",
        {
            "status": "PASS",
            "max_model_len": 73728,
            "max_output_tokens": 2048,
            "disable_thinking": True,
            "tasks": context,
        },
    )
    receipt = {
        "format_version": "agentic-original-pilot-judging-v1",
        "status": "prepared",
        "scientific_result": False,
        "execution_commit": commit,
        "adaptive_source": _file_identity(adaptive_path),
        "approved_allocation": APPROVAL,
        "estimate": ESTIMATE,
        "snapshot": cache,
        "task_count": len(plan.bulk_tasks),
        "claim_root": judge["claim_root"],
        "dispatch_mode": "partition",
        "worker_count": 2,
        "partition_counts": [
            sum(task_in_worker(t.judge_task_id, i, 2) for t in plan.bulk_tasks)
            for i in range(2)
        ],
        "validation_model": None
        if plan.validation_model is None
        else asdict(plan.validation_model),
        "validation_task_count": len(plan.validation_tasks),
        "files": [
            _file_identity(p)
            for p in [
                artifacts.manifest_path,
                output / "plan/bulk_tasks.jsonl",
                output / "plan/validation_tasks.jsonl",
                output / "plan/private_mapping.jsonl",
                output / "context-budget.json",
            ]
        ],
    }
    _atomic_json(output / "launch.json", receipt)
    return receipt


def verify(root):
    receipt = json.loads((root / "launch.json").read_text())
    if (
        receipt["approved_allocation"] != APPROVAL
        or receipt["worker_count"] != 2
        or receipt["dispatch_mode"] != "partition"
        or receipt["task_count"] != 2 * EXPECTED_CELLS
    ):
        raise ValueError("Launch approval or task coverage mismatch")
    for entry in receipt["files"]:
        checked_file(entry)
    cache = snapshot_files(Path(receipt["snapshot"]["path"]))
    if cache != receipt["snapshot"]:
        raise ValueError("Cached snapshot changed after preflight")
    return receipt


def audit(root):
    receipt = verify(root)
    tasks, _, model, revision = _agentic_judge_context(
        root / "plan/run_manifest.json",
        root / "plan/bulk_tasks.jsonl",
        judge_role="bulk",
    )
    store = InferenceClaimStore(receipt["claim_root"])
    args = SimpleNamespace(
        judge_role="bulk",
        disable_thinking=True,
        fake=False,
        pilot_only=False,
        max_attempts=3,
        request_timeout=120,
    )
    counts = Counter()
    slots = [Counter(), Counter()]
    for n, task in enumerate(tasks, 1):
        item = _prepare_agentic_judge(task, max_tokens=2048)
        identity = _shared_claim_identity(
            item, args=args, model_id=model, model_revision=revision
        )
        state, _ = store.inspect(
            identity,
            validate=lambda outcome, item=item: _validate_shared_outcome(
                outcome, item=item, fake=False, pilot_only=False
            ),
            validate_failure=lambda outcome, item=item: _validate_shared_failure(
                outcome, item=item, fake=False, pilot_only=False
            ),
        )
        counts[state] += 1
        slots[0 if task_in_worker(task.judge_task_id, 0, 2) else 1][state] += 1
        if n % 500 == 0:
            print(f"CLAIMS_CHECKED={n}/{len(tasks)}", flush=True)
    return {
        "expected": len(tasks),
        "claims": dict(counts),
        "partitions": [dict(s) for s in slots],
        "bulk_complete": counts["completed"] == len(tasks),
        "validation_tasks_planned": receipt["validation_task_count"],
        "validation_status": "not_audited"
        if receipt["validation_model"]
        else "not_configured",
        "scientific_result": False,
    }


def submission_command(root, account):
    return [
        "sbatch",
        "--parsable",
        "--export=ALL",
        f"--account={account}",
        "--partition=booster",
        "--array=0-1%2",
        "--nodes=1",
        "--ntasks=1",
        "--gpus-per-node=4",
        "--cpus-per-task=32",
        "--mem=512G",
        "--exclusive",
        "--time=08:00:00",
        "--no-requeue",
        "--job-name=geodml-pilot-nemotron",
        f"--output={root}/logs/slurm-%A_%a.out",
        f"--error={root}/logs/slurm-%A_%a.err",
        str(
            REPOSITORY
            / "analysis/scripts/slurm/jupiter/run_agentic_pilot_judging.sbatch"
        ),
        str(root),
    ]


def submit(root, account, approved_walltime):
    if approved_walltime != APPROVAL["walltime"]:
        raise ValueError("Submission requires explicit approved walltime 08:00:00")
    receipt = verify(root)
    actual = subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    ).strip()
    if actual != receipt["execution_commit"] or dirty:
        raise ValueError("Use the clean pinned execution checkout")
    report = audit(root)
    if report["bulk_complete"]:
        print("QUEUE_ALREADY_COMPLETE: no allocation submitted", flush=True)
        return
    if any(s.get("busy", 0) for s in report["partitions"]):
        raise ValueError("An existing judge owns tasks; refusing another launch")
    if any(s.get("missing", 0) == 0 for s in report["partitions"]):
        raise ValueError(
            "One partition has no eligible work; obtain a smaller resume budget"
        )
    (root / "logs").mkdir(exist_ok=True)
    command = submission_command(root, account)
    # Exclusive create, not a filesystem lock: an interrupted/uncertain sbatch
    # never permits blind automatic resubmission. Preserve the intent for audit.
    with (root / "submission-intent.json").open("x") as stream:
        json.dump({"command": command, "approval": APPROVAL}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    env = dict(os.environ, GEODML_EXECUTION_REPOSITORY=str(REPOSITORY))
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, check=False
    )
    _atomic_json(
        root / "submission-result.json",
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        raise RuntimeError(
            "sbatch failed; inspect submission-result.json; no automatic retry"
        )
    print("SUBMITTED=" + result.stdout.strip(), flush=True)


def worker(root):
    receipt = verify(root)
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if index not in (0, 1):
        raise ValueError("Unexpected worker slot")
    job_id = os.environ["SLURM_JOB_ID"]
    reference = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{index}"
    record = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", reference], text=True
    )
    fields = dict(w.split("=", 1) for w in record.split() if "=" in w)
    if (
        fields.get("JobId") != job_id
        or fields.get("JobState") != "RUNNING"
        or fields.get("NumNodes") != "1"
        or fields.get("TimeLimit") != APPROVAL["walltime"]
        or fields.get("UserId", "").split("(")[0] != os.environ["USER"]
        or fields.get("OverSubscribe") != "NO"
        or fields.get("ArrayJobId") != os.environ["SLURM_ARRAY_JOB_ID"]
        or fields.get("ArrayTaskId") != str(index)
    ):
        raise ValueError("Controller allocation differs from approved one-node worker")
    tres = dict(
        x.split("=", 1) for x in fields.get("AllocTRES", "").split(",") if "=" in x
    )
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    )
    if tres.get("gres/gpu") != "4" or len(set(devices.split())) != 4:
        raise ValueError("Controller and visible devices must verify four GPUs")
    env = dict(os.environ)
    for name, field in (
        ("SLURM_JOB_START_TIME", "StartTime"),
        ("SLURM_JOB_END_TIME", "EndTime"),
    ):
        env[name] = subprocess.check_output(
            ["date", "-d", fields[field], "+%s"], text=True
        ).strip()
    for name in (
        "GEODML_ROLE_END_TIME",
        "GEODML_ADMISSION_STOP_FILE",
        "GEODML_JUDGE_PILOT_ROOT",
    ):
        env.pop(name, None)
    env.update(
        SLURM_GPUS_ON_NODE="4",
        GEODML_EXECUTION_REPOSITORY=str(REPOSITORY),
        GEODML_EXECUTION_COMMIT=receipt["execution_commit"],
        GEODML_JUDGE_QUEUE_ROOT=str(root),
        GEODML_JUDGE_PLAN_ROOT=str(root / "plan"),
        GEODML_JUDGE_CLAIM_ROOT=receipt["claim_root"],
        GEODML_JUDGE_WORKER_INDEX=str(index),
        GEODML_JUDGE_WORKER_COUNT="2",
        GEODML_JUDGE_DISPATCH_MODE="partition",
        GEODML_JUDGE_PILOT_ONLY="0",
        GEODML_JUDGE_STDOUT=str(root / "logs" / f"slurm-{reference}.out"),
        GEODML_JUDGE_STDERR=str(root / "logs" / f"slurm-{reference}.err"),
        GEODML_APPROVED_WALLTIME=APPROVAL["walltime"],
        GEODML_ALLOCATION_ESTIMATE=ESTIMATE,
        GEODML_START_MARGIN_SECONDS="120",
        GEODML_CLEANUP_MARGIN_SECONDS="45",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_CACHE=str(Path(receipt["snapshot"]["path"]).parents[2]),
    )
    # A second attempt for the same partition must not run alongside the first.
    # A failed lock check stops this step; no existing lock or outcome is deleted.
    with (root / f"worker-{index}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                "bash",
                str(
                    REPOSITORY
                    / "analysis/scripts/slurm/jupiter/run_nemotron_judge_queue.sbatch"
                ),
            ],
            env=env,
            cwd=REPOSITORY,
            pass_fds=(lock.fileno(),),
            check=False,
        )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--adaptive-plan", type=Path, required=True)
    create.add_argument("--snapshot", type=Path, required=True)
    create.add_argument("--execution-commit", required=True)
    create.add_argument("--approved-walltime", required=True)
    for name in ("verify", "status", "submit", "worker"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        if name == "submit":
            sub.add_argument("--account", required=True)
            sub.add_argument("--approved-walltime", required=True)
    create.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.command == "prepare":
        result = prepare(
            args.adaptive_plan.resolve(),
            root,
            args.snapshot.resolve(),
            args.approved_walltime,
            args.execution_commit,
        )
    elif args.command == "submit":
        submit(root, args.account, args.approved_walltime)
        return 0
    elif args.command == "worker":
        return worker(root)
    else:
        result = audit(root) if args.command == "status" else verify(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
