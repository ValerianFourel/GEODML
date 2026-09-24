#!/usr/bin/env python3
"""Plan, reserve, transfer and execute portable GEODML hour packages.

No command submits Slurm jobs. Admission prints a command only after recording
an explicitly approved attempt. The finite sync helper runs on a networked host.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline.agentic_hour_runtime import (
    admission,
    allocation_command,
    execute,
    validate_request,
    verify_serving,
)
from analysis.interpretability.pipeline.agentic_hour_sync import (
    Exchange,
    HubStore,
    atomic,
    checkpoint_files,
)
from analysis.interpretability.pipeline.agentic_hours import (
    build_plan,
    canonical,
    check_owner,
    claim_hours,
    digest,
    finish_hours,
    install_plan,
    inventory,
    verify_plan,
)
from analysis.interpretability.pipeline.agentic_storage import storage_health
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.scripts.capture_agentic_scheduler_snapshot import capture
from analysis.scripts.publish_agentic_dataset import build_manifest
from analysis.scripts.reconcile_agentic_dataset import (
    TERMINAL_SCHEDULER_STATES,
    reconcile,
)


def read(path) -> dict:
    return json.loads(Path(path).read_bytes())


def load_plan(exchange: Exchange, plan_id: str) -> dict:
    raw = exchange.store.read(f"coordination/plans/{plan_id}.json", exchange.store.head())
    if raw is None:
        raise ValueError("published plan is missing")
    plan = json.loads(raw)
    if plan.get("plan_id") != plan_id:
        raise ValueError("plan identity mismatch")
    verify_plan(plan)
    return plan


def stage(exchange: Exchange, *, request: dict, profile: dict, runtime: dict,
          reference_profile: Path, dataset: Path, repository: Path,
          operation_id: str, stripes: int = 256) -> dict:
    from analysis.scripts.verify_inference_allocation import cluster_profile
    profile = {**cluster_profile(request["cluster"]), **profile}
    validate_request(request, profile)
    directory = Path(request["attempt_dir"]).resolve()
    _, registry = exchange.snapshot()
    selected = [registry["hours"][key] for key in request["hour_ids"]]
    selected = selected[:1] + sorted(selected[1:], key=lambda row: (row["priority_rank"], row["hour_id"]),
                                    reverse=request["mode"] == "interactive")
    if len({row["model"] for row in selected}) != 1:
        raise ValueError("one attempt must use one model")
    model = selected[0]["model"]
    evidence = profile.get("validated_models", {}).get(model)
    if not evidence or not evidence.get("evidence") or not evidence.get("reference_profile_sha256"):
        raise ValueError("model has no validated cluster compatibility evidence")
    if hashlib.sha256(reference_profile.read_bytes()).hexdigest() != evidence["reference_profile_sha256"]:
        raise ValueError("reference profile differs from validated cluster profile")
    # Reject inherited scientific overrides and arbitrary runtime shell configuration.
    permitted = {"SEARCH_AGENTIC_PROFILE", "SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT",
                 "SEARCH_AGENTIC_CROSS_ENCODER_REVISION", "SEARCH_AGENTIC_DDG_SNAPSHOT",
                 "SEARCH_AGENTIC_SEARXNG_SNAPSHOT", "SEARCH_AGENTIC_PROMPTS_JSONL",
                 "SEARCH_AGENTIC_SELECTION_RECORDS_JSONL", "SEARCH_AGENTIC_PROMPT_COUNT",
                 "SEARCH_AGENTIC_PROMPT_SELECTION_SEED", "SEARCH_AGENTIC_REQUEST_CONCURRENCY",
                 "SEARCH_AGENTIC_CELL_CONCURRENCY", "GEODML_JUDGE_MANIFEST", "GEODML_JUDGE_ROLE",
                 "GEODML_JUDGE_PROFILE", "GEODML_JUDGE_DISABLE_THINKING", "GEODML_JUDGE_CONCURRENCY",
                 "GEODML_JUDGE_MAX_OUTPUT_TOKENS"}
    if set(runtime) - permitted or any(not isinstance(v, str) for v in runtime.values()):
        raise ValueError("runtime environment contains unsupported keys or non-string values")
    if profile.get("execution_boundary") != "exclusive-slurm-node":
        raise ValueError("Maintained shared hours require an exclusive-slurm-node cluster profile")
    runtime = {**runtime, "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY": "1"}
    tasks, plans = {}, {}
    for hour in selected:
        if hour["plan_id"] not in plans:
            plan = load_plan(exchange, hour["plan_id"])
            if plan["calibration"][model]["reference_profile_sha256"] != evidence["reference_profile_sha256"]:
                raise ValueError("cluster profile does not match the hour's frozen reference profile")
            exchange.download(plan["input_bundle"], dataset, stripes=stripes)
            plans[hour["plan_id"]] = plan
        for bundle in hour["checkpoints"]:
            exchange.download(bundle, dataset, stripes=stripes)
        plan = plans[hour["plan_id"]]
        for fp in hour["task_fingerprints"]:
            if fp not in hour["completed"] and fp not in hour["failed"]:
                tasks[fp] = plan["tasks"][fp]
    if not tasks:
        raise ValueError("selected hours have no eligible work")
    attempt = {"format_version": "geodml-hour-attempt-v1", "request": request,
               "attempt_id": request["attempt_id"], "cluster": request["cluster"],
               "model": model, "cluster_profile": profile, "runtime_environment": runtime,
               "reference_profile": str(reference_profile.resolve()),
               "reference_profile_sha256": evidence["reference_profile_sha256"],
               "dataset_root": str(dataset.resolve()), "repository": str(repository.resolve()),
               "writer_id": f"{request['cluster']}-{request['attempt_id']}",
               "ledger_stripes": stripes, "tasks_sha256": digest(tasks),
               "admission_ticket": None}
    verify_serving(attempt)
    if directory.exists() and (directory / "attempt.json").exists():
        prior = read(directory / "attempt.json")
        if prior["request"] != request or prior["runtime_environment"] != runtime:
            raise ValueError("attempt directory belongs to a different request")
    exchange.transact(operation_id, {"action": "reserve", "request": request,
                                    "profile_sha256": digest(profile)},
                      lambda state: claim_hours(state, hour_ids=request["hour_ids"],
                                                cluster=request["cluster"], attempt_id=request["attempt_id"],
                                                supported_models=list(profile["validated_models"])))
    _, registry = exchange.snapshot()
    owners = {}
    for key in request["hour_ids"]:
        owner = registry["hours"][key]["owner"]
        if not owner or owner["attempt_id"] != request["attempt_id"] or owner["cluster"] != request["cluster"]:
            raise ValueError("reservation is no longer owned by this attempt")
        owners[key] = owner
    queue = b"".join(canonical({**row["runnable_task"], "geodml_keyword_id": row["keyword_id"],
                                "geodml_task_fingerprint": fp}) + b"\n" for fp, row in tasks.items())
    attempt.update(owners=owners, backlog_sha256=hashlib.sha256(queue).hexdigest())
    if (directory / "attempt.json").exists():
        prior = read(directory / "attempt.json")
        if {**prior, "admission_ticket": None} != attempt:
            raise ValueError("staged attempt changed; use its saved manifest")
        return prior
    atomic(directory / "tasks.json", canonical(tasks))
    atomic(directory / "backlog.jsonl", queue)
    atomic(directory / "attempt.json", canonical(attempt))
    return attempt


def scheduler(attempt: dict, additional_job_ids: list[str] | None = None) -> dict:
    existing = attempt.get("admission_ticket", {}).get("existing_job_id") if attempt.get("admission_ticket") else None
    receipt = Path(attempt["request"]["attempt_dir"]) / "execution.json"
    if receipt.exists():
        existing = read(receipt)["job_id"]
    result = capture(plan={"plan_id": "shared-hours"}, since=attempt["request"]["since"],
                     include_job_ids=[*(additional_job_ids or []), *([existing] if existing else [])])
    result["cluster"] = attempt["cluster"]
    # An explicitly attached allocation can have a historical name/comment.
    if existing:
        for owner in list(result["owners"]):
            if str(owner["job_id"]) == existing:
                result["owners"].append({**owner, "owner_id": attempt["writer_id"],
                                          "cluster": attempt["cluster"], "attempt_id": attempt["attempt_id"]})
                break
    return result


def health(attempt: dict, quota_path: Path) -> dict:
    quota = read(quota_path)
    captured = quota.get("captured_at_epoch")
    quota["fresh"] = type(captured) is int and 0 <= time.time() - captured <= 300
    if quota.get("cluster") != attempt["cluster"]:
        quota["fresh"] = False
    result = storage_health(Path(attempt["dataset_root"]), quota_evidence=quota)
    result["cluster"] = attempt["cluster"]
    return result


def transient_network_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if getattr(getattr(error, "response", None), "status_code", None) in {429, 500, 502, 503, 504}:
        return True
    # HF versions use either requests or httpx. Avoid a new dependency for either.
    return any(cls.__name__ in {"TransportError", "ConnectError", "ConnectionError", "TimeoutException"}
               for cls in type(error).__mro__)


def sync_once(exchange: Exchange, attempt: dict, snapshot: dict, storage: dict) -> dict:
    root = Path(attempt["dataset_root"])
    directory = Path(attempt["request"]["attempt_dir"])
    stripes = attempt["ledger_stripes"]
    if snapshot.get("cluster") != attempt["cluster"] or snapshot.get("complete") is not True:
        raise ValueError("wrong-cluster or incomplete scheduler snapshot")
    if not 0 <= time.time() - snapshot.get("captured_at_epoch", 0) <= 120:
        raise ValueError("stale scheduler snapshot")
    if (storage.get("safe_to_admit") is not True or storage.get("quota_verified") is not True
            or not 0 <= time.time() - storage.get("captured_at_epoch", 0) <= 300):
        atomic(directory / "STOP_ADMISSION", b"storage health requires review\n")
    terminal = [row for row in snapshot.get("owners", []) if row["owner_id"] == attempt["writer_id"]
                and row["state"] in TERMINAL_SCHEDULER_STATES
                and row.get("cluster") == attempt["cluster"]]
    # Recovery itself validates terminal scheduler states and repairs only stopped writers.
    if terminal:
        reconciliation = reconcile(root, scheduler_snapshot={**snapshot, "owners": terminal},
                                   stripe_count=stripes, apply=True)
        if reconciliation["blocked"]:
            return {"status": "blocked", "reconciliation": reconciliation}
    tasks = read(directory / "tasks.json")
    names, outcomes = checkpoint_files(root, tasks, stripes=stripes, writer_id=attempt["writer_id"])
    if terminal:
        latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
        unpublished = [fp for fp in tasks if latest.get(fp, {}).get("state") in {"completed", "result_saved", "terminal_failed"}
                       and fp not in outcomes]
        if unpublished:
            return {"status": "blocked", "reason": "terminal_results_not_publishable", "fingerprints": unpublished}
    metadata = {"attempt_id": attempt["attempt_id"], "cluster": attempt["cluster"],
                "owners": attempt["owners"], "terminal": bool(terminal),
                "request": attempt["request"], "writer_id": attempt["writer_id"],
                "cluster_profile": attempt.get("cluster_profile"),
                "runtime_environment": attempt.get("runtime_environment"),
                "reference_profile_sha256": attempt.get("reference_profile_sha256")}
    if (directory / "execution.json").exists():
        metadata["execution"] = read(directory / "execution.json")
    if terminal:
        metadata["scheduler_confirmation"] = terminal
    signature = digest({"names": names, "outcomes": outcomes, "metadata": metadata})
    prior_sync = directory / "sync.json"
    if prior_sync.exists() and read(prior_sync).get("signature") == signature:
        return read(prior_sync)
    bundle = exchange.upload(root, names, outcomes=outcomes, metadata=metadata)
    exchange.download(bundle, root, stripes=stripes, import_outcomes=False, verify_remote=True)
    operation_id = "checkpoint-" + digest({"bundle": bundle, "owners": attempt["owners"]})[:32]
    exchange.transact(operation_id, {"action": "checkpoint", "bundle": bundle},
                      lambda state: finish_hours(state, owners=attempt["owners"], checkpoint=bundle,
                                                 outcomes=outcomes, terminal=bool(terminal)))
    result = {"status": "released" if terminal else "owned", "bundle": bundle,
              "signature": signature,
              "verified_tasks_in_checkpoint": len(outcomes)}
    atomic(directory / "sync.json", canonical(result))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-id", default="ValerianFourel/geodml-experiment-v2-paper-private")
    result.add_argument("--journal", type=Path)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    inspect = commands.add_parser("inspect-inputs")
    inspect.add_argument("--dataset-root", type=Path, required=True)
    inspect.add_argument("--stripes", type=int, default=256)
    inputs = commands.add_parser("upload-inputs")
    inputs.add_argument("--dataset-root", type=Path, required=True)
    inputs.add_argument("--scheduler-snapshot", type=Path, required=True)
    inputs.add_argument("--stripes", type=int, default=256)
    planning = commands.add_parser("plan")
    planning.add_argument("--dataset-root", type=Path, required=True)
    planning.add_argument("--calibration", type=Path, required=True)
    planning.add_argument("--input-bundle", required=True)
    planning.add_argument("--source-commit", required=True)
    planning.add_argument("--output", type=Path)
    planning.add_argument("--publish", action="store_true")
    planning.add_argument("--operation-id")
    planning.add_argument("--stripes", type=int, default=256)
    planning.add_argument("--cluster", choices=["jupiter", "horeka"], required=True)
    publish = commands.add_parser("publish-plan")
    publish.add_argument("--plan", type=Path, required=True)
    publish.add_argument("--cluster", choices=["jupiter", "horeka"], required=True)
    publish.add_argument("--operation-id", required=True)
    preparing = commands.add_parser("reserve")
    for name in ("request", "profile", "runtime", "reference-profile", "dataset-root", "repository"):
        preparing.add_argument("--" + name, type=Path, required=True)
    preparing.add_argument("--operation-id", required=True)
    preparing.add_argument("--stripes", type=int, default=256)
    gate = commands.add_parser("admit")
    gate.add_argument("--attempt", type=Path, required=True)
    gate.add_argument("--quota-evidence", type=Path, required=True)
    gate.add_argument("--existing-job-id")
    gate.add_argument("--operation-id", required=True)
    gate.add_argument("--apply", action="store_true")
    worker = commands.add_parser("execute")
    worker.add_argument("--attempt", type=Path, required=True)
    worker.add_argument("--job-id", required=True)
    release = commands.add_parser("release-unstarted")
    release.add_argument("--attempt", type=Path, required=True)
    release.add_argument("--operation-id", required=True)
    release.add_argument("--apply", action="store_true")
    sync = commands.add_parser("sync")
    sync.add_argument("--attempt", type=Path, required=True)
    sync.add_argument("--quota-evidence", type=Path, required=True)
    sync.add_argument("--until-epoch", type=int, required=True)
    sync.add_argument("--poll-seconds", type=int, default=30)
    sync.add_argument("--once", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "execute":
        return execute(args.attempt.resolve(), expected_job_id=args.job_id)
    if args.command == "inspect-inputs":
        tasks, completed, blocked = inventory(args.dataset_root, stripes=args.stripes)
        groups = {}
        for row in tasks:
            key = row["model"] + ":" + row["configuration_sha256"]
            groups[key] = groups.get(key, 0) + 1
        print(json.dumps({"configuration_task_counts": groups, "verified_completed": len(completed),
                          "blocked": len(blocked)}, indent=2))
        return 0
    if args.journal is None:
        raise ValueError("network commands require a durable --journal outside the checkout")
    exchange = Exchange(HubStore(args.repo_id), args.journal)
    if args.command == "list":
        _, state = exchange.snapshot()
        print(json.dumps(state, indent=2))
    elif args.command == "upload-inputs":
        snapshot = read(args.scheduler_snapshot)
        if (snapshot.get("cluster") != "jupiter" or snapshot.get("complete") is not True
                or snapshot.get("jobs") or not 0 <= time.time() - snapshot.get("captured_at_epoch", 0) <= 120):
            raise ValueError("bootstrap requires fresh proof that legacy allocations are stopped")
        tasks, _, _ = inventory(args.dataset_root, stripes=args.stripes)
        _, outcomes = checkpoint_files(args.dataset_root, {r["fingerprint"]: r for r in tasks}, stripes=args.stripes)
        manifest = build_manifest(args.dataset_root)
        names = [name for name in manifest["files"] if name.split("/")[0] in
                 {"README.md", "contract.json", "schemas", "data", "artifacts"}]
        bundle = exchange.upload(args.dataset_root, names, outcomes=outcomes,
                                 metadata={"kind": "frozen-inputs", "contract_sha256": digest(read(args.dataset_root / "contract.json"))})
        print(json.dumps({"input_bundle": bundle}))
    elif args.command == "plan":
        if args.cluster != "jupiter":
            raise ValueError("only the JUPITER chief may replan")
        if args.publish and (not args.operation_id or not args.output):
            raise ValueError("publishing requires a durable --output and --operation-id")
        _, state = exchange.snapshot()
        exchange.download(args.input_bundle, args.dataset_root, stripes=args.stripes)
        for bundle in sorted({b for hour in state["hours"].values() for b in hour["checkpoints"]}):
            exchange.download(bundle, args.dataset_root, stripes=args.stripes)
        tasks, completed, blocked = inventory(args.dataset_root, stripes=args.stripes)
        plan = build_plan(tasks=tasks, calibration=read(args.calibration), registry=state,
                          contract=read(args.dataset_root / "contract.json"), completed=completed,
                          blocked=blocked, source_commit=args.source_commit, input_bundle=args.input_bundle)
        if args.output:
            if args.output.exists():
                raise FileExistsError("refusing to overwrite a historical plan")
            atomic(args.output, canonical(plan))
        if args.publish:
            if not args.operation_id:
                raise ValueError("publishing requires --operation-id")
            exchange.transact(args.operation_id, {"action": "plan", "plan_sha256": digest(plan)},
                              lambda current: install_plan(current, plan, cluster=args.cluster),
                              {f"coordination/plans/{plan['plan_id']}.json": canonical(plan)})
        print(json.dumps(plan, indent=2))
    elif args.command == "publish-plan":
        value = read(args.plan)
        verify_plan(value)
        receipt = exchange.transact(args.operation_id, {"action": "plan", "plan_sha256": digest(value)},
                                    lambda current: install_plan(current, value, cluster=args.cluster),
                                    {f"coordination/plans/{value['plan_id']}.json": canonical(value)})
        print(json.dumps(receipt, indent=2))
    elif args.command == "reserve":
        value = stage(exchange, request=read(args.request), profile=read(args.profile),
                      runtime=read(args.runtime), reference_profile=args.reference_profile,
                      dataset=args.dataset_root, repository=args.repository,
                      operation_id=args.operation_id, stripes=args.stripes)
        print(json.dumps(value, indent=2))
    elif args.command == "admit":
        attempt = read(args.attempt)
        request = attempt["request"]
        validate_request(request, attempt["cluster_profile"])
        _, current = exchange.snapshot()
        tracked = [ticket["existing_job_id"] for ticket in current["admission"].get(attempt["cluster"], {}).get("tickets", {}).values()
                   if ticket.get("existing_job_id")]
        snapshot = scheduler(attempt, [*tracked, *([args.existing_job_id] if args.existing_job_id else [])])
        storage = health(attempt, args.quota_evidence)
        def change(state):
            for key, owner in attempt["owners"].items():
                check_owner(state["hours"][key], owner)
            updated = admission(state, request, snapshot, storage, now=int(time.time()),
                                existing_job_id=args.existing_job_id)
            updated["admission"][attempt["cluster"]]["tickets"][attempt["attempt_id"]]["attempt_sha256"] = digest({**attempt, "admission_ticket": None})
            return updated
        if args.apply:
            exchange.transact(args.operation_id, {"action": "admit", "request": request,
                                                 "existing_job_id": args.existing_job_id}, change)
            _, state = exchange.snapshot()
            attempt["admission_ticket"] = state["admission"][attempt["cluster"]]["tickets"][attempt["attempt_id"]]
            atomic(args.attempt, canonical(attempt))
        else:
            _, state = exchange.snapshot()
            change(state)
        command = None if args.existing_job_id else allocation_command(request, attempt["cluster_profile"], Path(attempt["repository"]))
        print(json.dumps({"applied": args.apply, "allocation_command": shlex.join(command) if args.apply and command else None,
                          "step_command": "Run the shared-hour wrapper with an explicit srun --jobid in the existing allocation."}, indent=2))
    elif args.command == "sync":
        if not 1 <= args.poll_seconds <= 60 or args.until_epoch <= time.time():
            raise ValueError("helper needs a finite future deadline and a 1–60 second poll interval")
        attempt = read(args.attempt)
        with (args.attempt.parent / ".sync.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            while time.time() < args.until_epoch:
                try:
                    current_health = health(attempt, args.quota_evidence)
                except (OSError, ValueError):
                    atomic(args.attempt.parent / "STOP_ADMISSION", b"storage evidence unavailable\n")
                    raise
                try:
                    result = sync_once(exchange, attempt, scheduler(attempt), current_health)
                except Exception as error:
                    if not transient_network_error(error):
                        raise
                    result = {"status": "offline_owned", "error_type": type(error).__name__}
                print(json.dumps(result), flush=True)
                if args.once or result["status"] in {"released", "blocked"}:
                    break
                time.sleep(min(args.poll_seconds, max(0, args.until_epoch - time.time())))
    elif args.command == "release-unstarted":
        attempt = read(args.attempt)
        snapshot = scheduler(attempt)
        def change(state):
            tickets = state["admission"].get(attempt["cluster"], {}).get("tickets", {})
            if (attempt["attempt_id"] in tickets or attempt.get("admission_ticket")
                    or (args.attempt.parent / "execution.json").exists()
                    or any(row.get("attempt_id") == attempt["attempt_id"]
                           for row in [*snapshot["jobs"], *snapshot["owners"]])):
                raise ValueError("attempt may have been allocated; require terminal-owner reconciliation")
            return finish_hours(state, owners=attempt["owners"], checkpoint=bundle,
                                outcomes={}, terminal=True)
        # Dry run never creates a remote bundle or releases ownership.
        bundle = "dry-run"
        _, state = exchange.snapshot()
        change(state)
        if args.apply:
            bundle = exchange.upload(Path(attempt["dataset_root"]), [], outcomes={},
                                     metadata={"kind": "unstarted-release", "attempt_id": attempt["attempt_id"]})
            exchange.transact(args.operation_id, {"action": "release-unstarted", "attempt_id": attempt["attempt_id"]}, change)
        print(json.dumps({"released": args.apply}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
