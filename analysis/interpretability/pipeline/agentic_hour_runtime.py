"""Shared-hour admission and execution using the existing model workers."""

from __future__ import annotations

import fcntl
import getpass
import json
import os
import re
import shutil
import signal
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from analysis.scripts.reconcile_agentic_dataset import TERMINAL_SCHEDULER_STATES
from analysis.scripts.verify_inference_allocation import verify as verify_boundary

from .agentic_admission import RESOURCE_HOLDING_STATES
from .agentic_hour_sync import atomic
from .agentic_hours import CLUSTERS, canonical, digest, identifier
from .inference_budget import AllocationBudget


def validate_request(request: dict, profile: dict) -> None:
    identifier(request["attempt_id"])
    import re
    if len(request["attempt_id"]) > 128 or not re.fullmatch(r"[0-9a-f]{40}", request.get("git_commit", "")):
        raise ValueError("a short attempt ID and full pinned Git SHA are required")
    cluster = request["cluster"]
    if cluster not in CLUSTERS or profile["cluster"] != cluster:
        raise ValueError("cluster/profile mismatch")
    if request["mode"] not in {"batch", "interactive"}:
        raise ValueError("mode must be batch or interactive")
    approval = request["approval"]
    seconds = approval.get("walltime_seconds")
    if (approval.get("status") != "approved" or not approval.get("evidence")
            or not approval.get("estimate") or type(seconds) is not int or not 0 < seconds <= 3600):
        raise ValueError("a specific approved allocation of at most one hour and its estimate are required")
    resources = approval["resources"]
    if resources.get("nodes") != 1 or resources.get("gpus") != 4:
        raise ValueError("shared-hour workers currently require one four-GPU node")
    if type(resources.get("cpus")) is not int or resources["cpus"] < 1 or not resources.get("memory"):
        raise ValueError("approved CPU and memory resources are required")
    if approval.get("maximum_gpu_hours") != seconds * 4 / 3600:
        raise ValueError("approved GPU-hour budget does not match resources and walltime")
    if not request.get("hour_ids") or not request.get("git_commit"):
        raise ValueError("a finite hour list and pinned code commit are required")


def admission(registry: dict, request: dict, snapshot: dict, storage: dict,
              *, now: int, existing_job_id: str | None = None) -> dict:
    cluster = request["cluster"]
    for value, age, label in ((snapshot, 120, "scheduler"), (storage, 300, "storage")):
        captured = value.get("captured_at_epoch")
        if value.get("cluster") != cluster or type(captured) is not int or not 0 <= now - captured <= age:
            raise ValueError(f"missing, stale, or wrong-cluster {label} evidence")
    if snapshot.get("complete") is not True:
        raise ValueError("complete scheduler inventory required")
    if storage.get("safe_to_admit") is not True or storage.get("quota_verified") is not True:
        raise ValueError("storage admission blocked; fresh quota evidence required")
    result = deepcopy(registry)
    state = result["admission"].setdefault(cluster, {})
    jobs = snapshot["jobs"]
    history = snapshot.get("owners", [])
    wave = state.get("finite_wave")
    if wave:
        terminal = {row.get("attempt_id") for row in [*jobs, *history]
                    if row["state"] in TERMINAL_SCHEDULER_STATES}
        if not set(wave["attempt_ids"]) <= terminal:
            raise ValueError("approved concurrent wave is outstanding; reconcile every allocation first")
        state.pop("finite_wave")
    starts = [row["start_epoch"] for row in [*jobs, *history]
              if isinstance(row.get("start_epoch"), int) and row["start_epoch"] > 0]
    last = max([state.get("last_start", 0), *starts])
    pending = state.get("pending_attempt")
    if pending:
        observed = [row for row in [*jobs, *history] if row.get("attempt_id") == pending]
        if not observed:
            raise ValueError("previous admission unconfirmed; reconcile Slurm before admitting more work")
        if not any(row.get("start_epoch") or row["state"] in TERMINAL_SCHEDULER_STATES for row in observed):
            raise ValueError("previous allocation start is unconfirmed")
        state.pop("pending_attempt", None)
    active = [row for row in jobs if row["state"] in RESOURCE_HOLDING_STATES]
    if existing_job_id is not None:
        matches = [row for row in active if str(row["job_id"]) == existing_job_id]
        if len(matches) != 1 or len(active) > 5:
            raise ValueError("existing allocation is missing or cluster concurrency exceeds five")
    else:
        if len(active) >= 5:
            raise ValueError("cluster concurrency limit reached")
        if any(row["state"] == "PENDING" and not row.get("held", False) for row in jobs):
            raise ValueError("another released allocation has not started")
        if last and now < last + 600:
            raise ValueError("ten-minute observed-start gap has not elapsed")
        state["pending_attempt"] = request["attempt_id"]
    state["last_start"] = last
    tickets = state.setdefault("tickets", {})
    attempt = request["attempt_id"]
    if attempt in tickets:
        raise ValueError("attempt already admitted; reconcile rather than allocating twice")
    tickets[attempt] = {"request_sha256": digest(request), "existing_job_id": existing_job_id,
                        "admitted_at": now, "storage_sha256": digest(storage),
                        "scheduler_sha256": digest(snapshot)}
    return result


def validate_reservation(request: dict, profile: dict) -> None:
    name = profile.get("reservation")
    if not name:
        return
    if profile["cluster"] != "horeka" or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("invalid explicit HoreKa reservation")
    result = subprocess.run(["scontrol", "-o", "show", "reservation", name],
                            check=True, capture_output=True, text=True, timeout=30)
    rows = [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in result.stdout.splitlines()]
    rows = [row for row in rows if row.get("ReservationName") == name]
    if len(rows) != 1:
        raise ValueError("reservation cannot be verified")
    row = rows[0]
    if (row.get("State") != "ACTIVE" or "MAINT" in row.get("Flags", "").split(",")
            or row.get("PartitionName") not in {"(null)", profile["partition"]}):
        raise ValueError("reservation is not active for this partition")
    accounts, users = row.get("Accounts", "(null)"), row.get("Users", "(null)")
    if ((accounts != "(null)" and profile["account"] not in accounts.split(","))
            or (users != "(null)" and getpass.getuser() not in users.split(","))
            or accounts == users == "(null)"):
        raise ValueError("reservation access is not verified for this account/user")
    end = datetime.fromisoformat(row["EndTime"]).timestamp()
    start = datetime.fromisoformat(row["StartTime"]).timestamp()
    if not start <= time.time() < end - request["approval"]["walltime_seconds"]:
        raise ValueError("reservation validity does not cover the requested duration")


def allocation_command(request: dict, profile: dict, repository: Path) -> list[str]:
    validate_request(request, profile)
    validate_reservation(request, profile)
    approval = request["approval"]
    resources = approval["resources"]
    seconds = approval["walltime_seconds"]
    command = ["sbatch", "--parsable", "--no-requeue"] if request["mode"] == "batch" else ["salloc"]
    command.extend([
        f"--account={profile['account']}", f"--partition={profile['partition']}",
        "--nodes=1", "--ntasks=1", "--gres=gpu:4", "--exclusive",
        f"--cpus-per-task={resources['cpus']}",
        "--mem=0" if resources["memory"] == "all" else f"--mem={resources['memory']}",
        f"--time={seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}",
        f"--job-name=geodml-hours-{request['attempt_id']}",
        f"--comment=geodml-hours:{request['cluster']}:{request['attempt_id']}",
    ])
    if profile.get("reservation"):
        command.append(f"--reservation={profile['reservation']}")
    if request["mode"] == "batch":
        directory = Path(request["attempt_dir"]).resolve()
        command.extend([f"--chdir={directory}", f"--output={directory}/slurm-%j.out",
                        f"--error={directory}/slurm-%j.err",
                        str(repository / "analysis/scripts/slurm/run_agentic_hours.sh"),
                        str(Path(request["attempt_dir"]).resolve() / "attempt.json")])
    return command


def verify_serving(attempt: dict) -> dict:
    import hashlib

    from analysis.scripts.search_vllm_stage import load_profile
    env = attempt["runtime_environment"]
    key = "GEODML_JUDGE_PROFILE" if attempt["model"] == "nemotron" else "SEARCH_AGENTIC_PROFILE"
    local = load_profile(env[key])
    if hashlib.sha256(Path(attempt["reference_profile"]).read_bytes()).hexdigest() != attempt["reference_profile_sha256"]:
        raise ValueError("frozen reference profile file changed")
    frozen = load_profile(attempt["reference_profile"])
    if local["model"] != frozen["model"]:
        raise ValueError("HoreKa must preserve the pinned model and revision")
    for key in ("dtype", "max_model_len", "tensor_parallel_size", "data_parallel_size", "request_concurrency"):
        if local["serving"][key] != frozen["serving"][key]:
            raise ValueError(f"scientific serving setting changed: {key}")
    if ({key: value for key, value in local["features"].items() if key != "attention_backend"}
            != {key: value for key, value in frozen["features"].items() if key != "attention_backend"}):
        raise ValueError("serving features differ from frozen JUPITER profile")
    expected = "GH200" if attempt["cluster"] == "jupiter" else "A100"
    if local["visible_gpu_assignment"]["expected_gpu_name_pattern"] != expected:
        raise ValueError("serving profile is for the wrong accelerator")
    return local


def execute(attempt_path: Path, *, expected_job_id: str) -> int:
    """Run within an allocation only. Never call salloc/sbatch or release it."""
    attempt = json.loads(attempt_path.read_bytes())
    profile = attempt["cluster_profile"]
    boundary_receipt = verify_boundary(attempt["cluster"], profile.get("execution_boundary"), profile=profile)
    validate_request(attempt["request"], profile)
    if os.environ.get("SLURM_JOB_ID") != expected_job_id:
        raise ValueError("explicit job ID differs from the current allocation")
    if not attempt.get("admission_ticket"):
        raise ValueError("attempt has no approved admission ticket")
    if attempt["admission_ticket"]["request_sha256"] != digest(attempt["request"]):
        raise ValueError("approved request changed after admission")
    if attempt["admission_ticket"].get("attempt_sha256") != digest({**attempt, "admission_ticket": None}):
        raise ValueError("staged attempt changed after admission")
    bound = attempt["admission_ticket"].get("existing_job_id")
    if bound and bound != expected_job_id:
        raise ValueError("ticket belongs to a different existing allocation")
    info = subprocess.check_output(["scontrol", "show", "job", expected_job_id, "-o"], text=True)
    fields = dict(token.split("=", 1) for token in info.split() if "=" in token)
    if not bound:
        required = f"Comment=geodml-hours:{attempt['cluster']}:{attempt['attempt_id']}"
        if required not in info.split():
            raise ValueError("allocation does not carry this attempt's admission identity")
    repo = Path(attempt["repository"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != attempt["request"]["git_commit"]:
        raise ValueError("execution checkout differs from pinned commit")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True).strip():
        raise ValueError("execution requires a clean checkout")
    if (attempt_path.parent / "STOP_ADMISSION").exists():
        raise ValueError("attempt has a stop-admission marker; review storage before loading a model")
    verify_serving(attempt)
    root = attempt_path.parent
    tasks = json.loads((root / "tasks.json").read_bytes())
    if digest(tasks) != attempt["tasks_sha256"]:
        raise ValueError("reserved task file changed")
    queue = (root / "backlog.jsonl").read_bytes()
    import hashlib
    if hashlib.sha256(queue).hexdigest() != attempt["backlog_sha256"]:
        raise ValueError("reserved backlog changed")
    env = dict(os.environ)
    # An inherited legacy queue must never enlarge or redirect an hour attempt.
    for key in list(env):
        if key.startswith(("SEARCH_AGENTIC_", "GEODML_JUDGE_", "GEODML_WORKER_")):
            env.pop(key)
    env.pop("GEODML_INFERENCE_CLAIM_ROOT", None)
    env.update(attempt["runtime_environment"])
    for field, variable in (("StartTime", "SLURM_JOB_START_TIME"), ("EndTime", "SLURM_JOB_END_TIME")):
        raw = fields.get(field, "")
        if raw in {"", "Unknown", "N/A", "None"}:
            raise ValueError("Slurm did not report the allocation's actual start/end")
        env[variable] = str(int(datetime.fromisoformat(raw).timestamp()))
    approval = attempt["request"]["approval"]
    seconds = approval["walltime_seconds"]
    cache_root = Path(os.path.expandvars(profile["cache_root"]))
    if "$" in str(cache_root) or not cache_root.is_absolute():
        raise ValueError("cluster cache root must resolve to an absolute path")
    cache_dir = cache_root / attempt["writer_id"]
    if any(cache_dir.resolve().is_relative_to(path.resolve()) for path in
           (repo, Path(attempt["dataset_root"]), root)):
        raise ValueError("disposable caches must be separate from code and scientific outputs")
    env.update({
        "GEODML_APPROVED_WALLTIME": f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}",
        "GEODML_ALLOCATION_ESTIMATE": approval["estimate"],
        "GEODML_WORKER_TASKS": str(root / "backlog.jsonl"),
        "GEODML_HOUR_TASKS": str(root / "tasks.json"),
        "GEODML_WORKER_OUTPUT": str(root / "output"),
        "GEODML_DATASET_ROOT": attempt["dataset_root"],
        "GEODML_DATASET_WRITER_ID": attempt["writer_id"],
        "GEODML_DATASET_LEDGER_STRIPES": str(attempt["ledger_stripes"]),
        "GEODML_DATASET_SEAL_INTERVAL_SECONDS": "300",
        "GEODML_DISPATCH_MODE": "backlog", "GEODML_WORKER_INDEX": "0", "GEODML_WORKER_COUNT": "1",
        "GEODML_MODEL_SLUG": attempt["model"], "GEODML_EXPECTED_JOB_ID": expected_job_id,
        "GEODML_EXECUTION_COMMIT": attempt["request"]["git_commit"],
        "GEODML_WAVE_LOG_ROOT": str(root / "logs"),
        "GEODML_CACHE_ROOT": str(cache_dir),
        "GEODML_ADMISSION_STOP_FILE": str(root / "STOP_ADMISSION"),
        "GEODML_START_MARGIN_SECONDS": str(attempt.get("start_margin_seconds", 300)),
        "GEODML_CLEANUP_MARGIN_SECONDS": str(attempt.get("cleanup_margin_seconds", 120)),
    })
    # Bind approval to actual allocation startup even when the parent omitted it.
    if fields.get("NumNodes") != "1":
        raise ValueError("shared-hour execution requires exactly one node")
    gpu_names = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    expected_gpu = "GH200" if attempt["cluster"] == "jupiter" else "A100"
    if len(gpu_names) != 4 or any(expected_gpu not in name for name in gpu_names):
        raise ValueError("allocation does not match the four-GPU cluster profile")
    budget = AllocationBudget.from_environment(env, require=True)
    if not budget.can_start():
        raise ValueError("allocation has no admission time left")
    root.mkdir(parents=True, exist_ok=True)
    allocation_lock = Path(attempt["dataset_root"]) / "control/hour-allocation-locks" / f"{attempt['cluster']}-{expected_job_id}.lock"
    allocation_lock.parent.mkdir(parents=True, exist_ok=True)
    with (root / ".execute.lock").open("a") as lock, allocation_lock.open("a") as job_lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = root / "execution.json"
        if receipt.exists():
            raise ValueError("attempt was already started; reconcile and create a fresh approved attempt")
        if cache_dir.exists():
            raise ValueError("attempt cache already exists; refusing to reuse or remove unknown files")
        cache_root.mkdir(parents=True, exist_ok=True)
        stats = os.statvfs(cache_root)
        if (stats.f_bavail * stats.f_frsize < profile.get("minimum_cache_free_bytes", 20 * 1024**3)
                or stats.f_favail < profile.get("minimum_cache_free_inodes", 100_000)):
            raise ValueError("insufficient job-cache bytes or inodes before model startup")
        cache_dir.mkdir()
        cache_marker = {"cluster": attempt["cluster"], "attempt_id": attempt["attempt_id"]}
        atomic(cache_dir / ".geodml-hour-cache.json", canonical(cache_marker))
        record = {"job_id": expected_job_id, "cluster": attempt["cluster"],
                  "attempt_id": attempt["attempt_id"], "writer_id": attempt["writer_id"],
                  "started_at_epoch": time.time(), "status": "running",
                  "allocation_budget": budget.record(), "approval": approval,
                  "scheduler_allocation": fields, "gpu_inventory": gpu_names,
                  "execution_boundary": boundary_receipt,
                  "stdout": str(root / "worker.log"), "stderr": str(root / "worker.log"),
                  "git_commit": attempt["request"]["git_commit"], "profile_sha256": digest(profile)}
        atomic(receipt, canonical(record))
        worker = "run_agentic_judge_worker.sh" if attempt["model"] == "nemotron" else "run_agentic_generation_worker.sh"
        # These leaf workers are portable; cluster module setup lives in the outer wrapper.
        log = root / "worker.log"
        with log.open("xb") as stream:
            child = subprocess.Popen(["bash", str(repo / "analysis/scripts/slurm/jupiter" / worker)],
                                     cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            previous = {}
            def stop(signum, frame):
                try:
                    atomic(root / "STOP_ADMISSION", b"interrupted\n")
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
            try:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    previous[sig] = signal.signal(sig, stop)
                code = child.wait()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        record.update(status="worker_stopped", exit_code=code, finished_at_epoch=time.time())
        try:
            os.killpg(child.pid, 0)
            record["cache_cleanup"] = "retained_live_process_group"
        except ProcessLookupError:
            if (not cache_dir.is_symlink()
                    and json.loads((cache_dir / ".geodml-hour-cache.json").read_bytes()) == cache_marker):
                shutil.rmtree(cache_dir)
                record["cache_cleanup"] = "removed_attempt_owned_cache"
        atomic(receipt, canonical(record))
        return code
