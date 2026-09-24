"""Portable JUPITER-equivalent work packages. No scheduler or network effects."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .agentic_audit_progress import audit_progress, audit_stage
from .agentic_dataset import iter_sealed_rows, verify_record_reference
from .agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from .inference_claims import ClaimIdentity

PLAN_VERSION = "geodml-hour-plan-v1"
REGISTRY_VERSION = "geodml-hours-v1"
CLUSTERS = {"jupiter", "horeka"}
MODELS = {"qwen38", "llama4", "nemotron"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ValueError("invalid portable identifier")
    return value


def reference_timing(calibration: dict, model: str, configuration: str) -> dict:
    """Read legacy per-model or new per-configuration reference measurements."""
    value = calibration.get(model, {})
    if "configurations" in value:
        return value["configurations"].get(configuration, {})
    return value


def empty_registry() -> dict:
    return {"format_version": REGISTRY_VERSION, "current_plan": None,
            "hours": {}, "admission": {}}


@audit_stage("inventory")
def inventory(root: Path, *, stripes: int = 256) -> tuple[list[dict], set[str], set[str]]:
    """Read registered tasks and accept only locally verified completions."""
    audit_progress(phase="keyword_memberships")
    members = {r["prompt_id"]: r for r in iter_sealed_rows(root, "keyword_memberships", required=True)}
    audit_progress(phase="ledger", prompts=len(members))
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
    audit_progress(phase="verify_tasks", tasks_checked=0, verified_completed=0, blocked=0)
    completed, blocked, tasks = set(), set(), []
    for row in iter_sealed_rows(root, "task_definitions", required=True):
        fingerprint = identity_fingerprint(ClaimIdentity(**row["claim_identity"]))
        member = members[row["prompt_id"]]
        tasks.append({**row, "fingerprint": fingerprint,
                      "configuration_sha256": row.get("configuration_sha256") or digest({
                          "model_id": row["claim_identity"]["model_id"],
                          "model_revision": row["claim_identity"]["model_revision"],
                          "protocol": row["claim_identity"]["protocol"],
                          "judge_plan_id": row.get("judge_plan_id"),
                      }),
                      "keyword_id": member["primary_keyword_id"],
                      "priority_rank": member["primary_priority_rank"]})
        event = latest.get(fingerprint, {})
        refs = event.get("record_references", [])
        if event.get("state") == "completed" and refs and all(verify_record_reference(root, r) for r in refs):
            completed.add(fingerprint)
        elif event.get("state") in {"completed", "claimed", "running", "result_saved", "terminal_failed"}:
            blocked.add(fingerprint)
        audit_progress(tasks_checked=len(tasks), verified_completed=len(completed), blocked=len(blocked))
    return tasks, completed, blocked


def build_plan(*, tasks: list[dict], calibration: dict, registry: dict,
               contract: dict, completed: set[str], blocked: set[str],
               source_commit: str, input_bundle: str, allow_uncalibrated: bool = False) -> dict:
    """Repack released work only. Reference costs are allocation wall seconds."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full Git SHA")
    if not input_bundle:
        raise ValueError("a published frozen input bundle is required")
    owned, known_done, known_failed = set(), set(completed), set(blocked)
    for hour in registry["hours"].values():
        known_done.update(hour["completed"])
        known_failed.update(hour["failed"])
        if hour.get("owner"):
            owned.update(hour["task_fingerprints"])
    unique, groups = {}, defaultdict(list)
    deferred = {}
    for row in tasks:
        fp = identity_fingerprint(ClaimIdentity(**row["claim_identity"]))
        if fp != row["fingerprint"] or fp in unique:
            raise ValueError("duplicate or mismatched task fingerprint")
        unique[fp] = row
        if row["model"] not in MODELS:
            raise ValueError("unsupported model")
        if fp in known_done or fp in owned:
            continue
        if fp in known_failed or row.get("blocked_reason"):
            deferred[fp] = "blocked_or_failed"
            continue
        if not set(row.get("dependency_fingerprints", [])) <= known_done:
            deferred[fp] = "generation_dependency"
            continue
        timing = reference_timing(calibration, row["model"], row["configuration_sha256"])
        if not timing and allow_uncalibrated:
            deferred[fp] = "awaiting_calibration"
            continue
        groups[(row["model"], row["priority_rank"], row["keyword_id"],
                row["configuration_sha256"], row["prompt_id"])].append(row)
    packages, current, current_key = [], [], None
    timings = {}
    for model, configuration in {(key[0], key[3]) for key in groups}:
        timing = reference_timing(calibration, model, configuration)
        if (timing.get("cluster") != "jupiter" or timing.get("gpus") != 4
                or timing.get("gpu_type") != "GH200" or not timing.get("evidence")
                or not timing.get("scientific_config_sha256") or not timing.get("reference_profile_sha256")):
            raise ValueError(f"{model}: validated four-GH200 reference calibration required")
        for key in ("seconds_per_task", "startup_seconds", "drain_seconds"):
            value = timing.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{model}: invalid {key}")
        if timing["seconds_per_task"] <= 0 or timing["startup_seconds"] + timing["drain_seconds"] >= 3600:
            raise ValueError(f"{model}: reference timing leaves no useful work")
        timings[(model, configuration)] = timing
    for rows in groups.values():
        for row in rows:
            if row.get("configuration_sha256") != timings[(row["model"], row["configuration_sha256"])]["scientific_config_sha256"]:
                raise ValueError("reference calibration does not match the registered scientific configuration")

    def flush():
        if not current:
            return
        model = current[0]["model"]
        timing = timings[(model, current[0]["configuration_sha256"])]
        seconds = timing["startup_seconds"] + timing["drain_seconds"] + len(current) * timing["seconds_per_task"]
        packages.append({"model": model, "keyword_id": current[0]["keyword_id"],
                         "configuration_sha256": current[0]["configuration_sha256"],
                         "priority_rank": current[0]["priority_rank"],
                         "task_fingerprints": [r["fingerprint"] for r in current],
                         "reference_seconds": seconds, "oversized_prompt_group": seconds > 3600})

    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r["task_id"], r["fingerprint"]))
        model = key[0]
        timing = timings[(model, key[3])]
        size = len(current) + len(rows)
        if current and (current_key != key[:4] or
                        size * timing["seconds_per_task"] + timing["startup_seconds"] + timing["drain_seconds"] > 3600):
            flush()
            current = []
        current_key = key[:4]
        current.extend(rows)
    flush()
    body = {"format_version": PLAN_VERSION, "base_registry_sha256": digest(registry),
            "contract_sha256": digest(contract), "source_commit": source_commit,
            "input_bundle": input_bundle, "calibration": calibration,
            "completed_before_plan": sorted(known_done), "deferred": deferred,
            "tasks": {fp: unique[fp] for p in packages for fp in p["task_fingerprints"]},
            "packages": packages}
    plan_id = "plan-" + digest(body)[:24]
    for ordinal, package in enumerate(packages, 1):
        package["hour_id"] = f"{plan_id}-{package['model']}-{ordinal:06d}"
    return {**body, "plan_id": plan_id}


def install_plan(registry: dict, plan: dict, *, cluster: str) -> dict:
    if cluster != "jupiter":
        raise ValueError("only the JUPITER chief may publish a plan")
    if plan["format_version"] != PLAN_VERSION or plan["base_registry_sha256"] != digest(registry):
        raise ValueError("stale plan: synchronize and rebuild before publishing")
    verify_plan(plan)
    result = deepcopy(registry)
    new_tasks = {fp for row in plan["packages"] for fp in row["task_fingerprints"]}
    if sum(len(row["task_fingerprints"]) for row in plan["packages"]) != len(new_tasks):
        raise ValueError("packages overlap")
    for old in result["hours"].values():
        if new_tasks.intersection(old["completed"]):
            raise ValueError("new plan would repeat verified work")
        if old.get("owner"):
            if new_tasks.intersection(old["task_fingerprints"]):
                raise ValueError("new plan overlaps owned work")
        elif old["status"] not in {"complete", "superseded"}:
            remaining = set(old["task_fingerprints"]) - set(old["completed"]) - set(old["failed"])
            if not remaining <= new_tasks | set(plan["deferred"]):
                raise ValueError("replan would lose unfinished work")
            old["status"] = "superseded"
            old["superseded_by"] = [r["hour_id"] for r in plan["packages"]
                                    if remaining.intersection(r["task_fingerprints"])]
    for package in plan["packages"]:
        result["hours"][package["hour_id"]] = {
            **package, "plan_id": plan["plan_id"], "status": "available",
            "owner": None, "generation": 0, "completed": [], "failed": [],
            "checkpoints": [], "attempts": [],
        }
    result["current_plan"] = plan["plan_id"]
    return result


def verify_plan(plan: dict) -> None:
    body = deepcopy(plan)
    plan_id = body.pop("plan_id")
    for ordinal, row in enumerate(body["packages"], 1):
        expected = f"{plan_id}-{row['model']}-{ordinal:06d}"
        if row.pop("hour_id") != expected:
            raise ValueError("hour ID differs from its plan")
    if "plan-" + digest(body)[:24] != plan_id:
        raise ValueError("plan content hash mismatch")


def claim_hours(registry: dict, *, hour_ids: list[str], cluster: str, attempt_id: str,
                supported_models: list[str]) -> dict:
    if cluster not in CLUSTERS or not hour_ids or len(set(hour_ids)) != len(hour_ids):
        raise ValueError("invalid cluster or hour selection")
    identifier(attempt_id)
    result = deepcopy(registry)
    selected = [result["hours"][key] for key in hour_ids]
    if any(owner.get("attempt_id") == attempt_id and owner.get("cluster") == cluster
           for row in result["hours"].values() for owner in row["attempts"]):
        raise ValueError("attempt ID already used; resume its original operation or create a new attempt")
    if len({row["model"] for row in selected}) != 1:
        raise ValueError("an allocation must keep one model loaded")
    for row in selected:
        if row["model"] not in supported_models:
            raise ValueError("model is not validated for this cluster")
        if row["owner"] or row["status"] not in {"available", "partial"}:
            raise ValueError(f"hour unavailable: {row['hour_id']}")
        row["generation"] += 1
        row["owner"] = {"cluster": cluster, "attempt_id": attempt_id,
                        "generation": row["generation"]}
        row["attempts"].append(dict(row["owner"]))
        row["status"] = "reserved"
    return result


def check_owner(hour: dict, owner: dict) -> None:
    if hour["owner"] != owner:
        raise ValueError("hour ownership changed; refusing stale writer")


def finish_hours(registry: dict, *, owners: dict, checkpoint: str,
                 outcomes: dict, terminal: bool) -> dict:
    """Called only after outcome references are downloaded and verified."""
    result = deepcopy(registry)
    for hour_id, owner in owners.items():
        row = result["hours"][hour_id]
        check_owner(row, owner)
        selected = set(row["task_fingerprints"])
        done = {fp for fp, event in outcomes.items() if fp in selected and event["state"] == "completed"}
        failed = {fp for fp, event in outcomes.items() if fp in selected and event["state"] == "terminal_failed"}
        if done.intersection(row["failed"]) or failed.intersection(row["completed"]):
            raise ValueError("conflicting terminal task outcomes")
        row["completed"] = sorted(set(row["completed"]) | done)
        row["failed"] = sorted(set(row["failed"]) | failed)
        if checkpoint not in row["checkpoints"]:
            row["checkpoints"].append(checkpoint)
        if terminal:
            row["owner"] = None
            row["status"] = ("complete" if set(row["completed"]) == selected else
                             "blocked" if selected <= set(row["completed"]) | set(row["failed"]) else "partial")
        else:
            row["status"] = "awaiting_sync" if set(row["completed"]) == selected else "running"
    return result
