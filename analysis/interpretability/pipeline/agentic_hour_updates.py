"""Explicit, resumable update workflows over the existing hour registry."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .agentic_dataset import iter_sealed_rows
from .agentic_hour_sync import ConflictError, atomic
from .agentic_hours import (
    build_plan,
    canonical,
    digest,
    install_plan,
    inventory,
    verify_plan,
)

PREFERRED_CLUSTER = {"qwen38": "horeka", "llama4": "jupiter", "nemotron": "jupiter"}
ALLOWED_MODELS = {"horeka": {"qwen38"}, "jupiter": set(PREFERRED_CLUSTER)}


def read(path):
    return json.loads(Path(path).read_bytes())


def load_plan(exchange, plan_id, revision):
    raw = exchange.store.read(f"coordination/plans/{plan_id}.json", revision)
    if raw is None:
        raise ValueError("published hour plan is missing")
    value = json.loads(raw)
    verify_plan(value)
    return value


def select_hours(state, *, model, cluster=None, mode="batch", count=1, first=None, configuration=None):
    if model not in PREFERRED_CLUSTER or mode not in {"batch", "interactive"} or count < 1:
        raise ValueError("explicit supported model, mode and positive count required")
    cluster = cluster or PREFERRED_CLUSTER[model]
    if model not in ALLOWED_MODELS.get(cluster, set()):
        raise ValueError("model is not enabled on the selected cluster")
    rows = [r for r in state["hours"].values() if r["model"] == model
            and not r.get("owner") and r["status"] in {"available", "partial"}
            and set(r["task_fingerprints"]) - set(r["completed"]) - set(r["failed"])]
    rows.sort(key=lambda r: (r["priority_rank"], r["keyword_id"], r["hour_id"]),
              reverse=mode == "interactive")
    if first:
        chosen = [r for r in rows if r["hour_id"] == first]
        if not chosen:
            raise ValueError("requested first hour is unavailable or belongs to another model")
        rows = chosen + [r for r in rows if r["hour_id"] != first]
    if rows:
        configuration = configuration or rows[0].get("configuration_sha256")
        if first and rows[0].get("configuration_sha256") != configuration:
            raise ValueError("requested first hour belongs to another configuration")
        rows = [r for r in rows if r.get("configuration_sha256") == configuration]
    return {"cluster": cluster, "model": model, "mode": mode, "reserved": False,
            "configuration_sha256": configuration,
            "allocation_approved": False, "hour_ids": [r["hour_id"] for r in rows[:count]]}


def progress(root, state, *, revision, deferred=None, prior_completed=(), stripes=256):
    tasks, local_done, local_blocked = inventory(root, stripes=stripes)
    prompts = {r["prompt_id"]: r for r in iter_sealed_rows(root, "prompts")}
    done, failed, owned, assignments = set(prior_completed), set(), {}, {}
    for hour in state["hours"].values():
        done.update(hour["completed"])
        failed.update(hour["failed"])
        if hour["status"] == "superseded":
            continue
        for fp in hour["task_fingerprints"]:
            assignments.setdefault(fp, []).append(hour["hour_id"])
            if hour.get("owner"):
                owned[fp] = hour["owner"]["cluster"]
    groups, totals = {}, Counter()
    for task in tasks:
        fp = task["fingerprint"]
        if fp in done:
            status = "completed"
        elif fp in failed:
            status = "failed"
        elif fp in local_done:
            status = "awaiting_sync"
        elif fp in owned:
            status = "owned"
        elif fp in local_blocked or task.get("blocked_reason"):
            status = "blocked"
        elif (deferred or {}).get(fp):
            status = deferred[fp]
        elif not set(task.get("dependency_fingerprints", [])) <= done:
            status = "generation_dependency"
        else:
            status = "eligible"
        axis_bin = prompts.get(task["prompt_id"], {}).get("axis_bin")
        key = (task["model"], task["priority_rank"], task["keyword_id"], axis_bin if type(axis_bin) is int else -1)
        group = groups.setdefault(key, {"model": task["model"], "keyword_id": task["keyword_id"],
                                       "priority_rank": task["priority_rank"], "axis_bin": axis_bin,
                                       "cells": Counter(), "hour_ids": set(), "owners": Counter()})
        group["cells"][status] += 1
        totals[status] += 1
        group["hour_ids"].update(assignments.get(fp, []))
        if fp in owned:
            group["owners"][owned[fp]] += 1
    rows = [{**groups[k], "hour_ids": sorted(groups[k]["hour_ids"])} for k in sorted(groups)]
    return {"format_version": "geodml-hour-progress-v1", "registry_revision": revision,
            "registry_sha256": digest(state), "current_plan": state["current_plan"],
            "task_count": len(tasks), "cells": dict(totals), "groups": rows,
            "missing_models": sorted(set(PREFERRED_CLUSTER) - {t["model"] for t in tasks}),
            "unknown_bin_tasks": sum(sum(r["cells"].values()) for r in rows if r["axis_bin"] is None)}


def markdown(report):
    lines = ["# Shared-hour progress", "", f"Registry revision: `{report['registry_revision']}`",
             f"Registry checksum: `{report['registry_sha256']}`", "",
             "| Model | Keyword | Axis bin | Cells by state | Hours | Cluster ownership |",
             "|---|---|---|---|---|---|"]
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for row in report["groups"]:
        lines.append("| " + " | ".join(cell(v) for v in (
            row["model"], row["keyword_id"], row["axis_bin"], dict(row["cells"]),
            ", ".join(row["hour_ids"]), dict(row["owners"]))) + " |")
    return ("\n".join(lines) + "\n").encode()


def publish_progress(exchange, report):
    # Reports are not ownership state. Refuse to label a raced snapshot as current.
    revision, state = exchange.snapshot()
    if digest(state) != report["registry_sha256"]:
        raise ConflictError("registry changed during reporting; rerun the update")
    prior = exchange.store.read("coordination/progress.json", revision)
    comparison = dict(report)
    comparison.pop("registry_revision")
    if prior:
        old = json.loads(prior)
        old.pop("registry_revision", None)
        if old == comparison:
            return
    exchange.store.commit(revision, {"coordination/progress.json": canonical(report),
                                    "coordination/progress.md": markdown(report)}, "GEODML progress view")


def bundles_for(exchange, state, revision, model=None):
    plan_ids = {h["plan_id"] for h in state["hours"].values()
                if h["status"] != "superseded" and (model is None or h["model"] == model)}
    if state["current_plan"]:
        plan_ids.add(state["current_plan"])
    bundles, plans = set(), {}
    for plan_id in sorted(plan_ids):
        value = load_plan(exchange, plan_id, revision)
        plans[plan_id] = value
        bundles.add(value["input_bundle"])
    # Completed generator records can be dependencies of a different model.
    bundles.update(b for h in state["hours"].values() for b in h["checkpoints"])
    return bundles, plans


def pull(exchange, site, *, model=None):
    from analysis.scripts.prepare_horeka_qwen import check_storage
    root = Path(site["dataset_root"])
    revision, state = exchange.snapshot()
    bundles, plans = bundles_for(exchange, state, revision, model)
    if site.get("input_bundle"):
        bundles.add(site["input_bundle"])
    if not bundles:
        raise ValueError("no published input bundle; reconcile and publish from JUPITER first")
    manifests = {b: exchange.manifest(b, revision) for b in bundles}
    expected = {}
    for manifest in manifests.values():
        for name, entry in manifest["files"].items():
            if name in expected and expected[name] != entry:
                raise ValueError("published bundles contain conflicting paths")
            expected[name] = entry
    missing = [e["bytes"] for n, e in expected.items() if not (root / n).exists()]
    quota = read(site["quota_evidence"])
    if site["cluster"] == "horeka":
        check_storage(root, quota, 2 * sum(missing) + max(missing, default=0), len(missing) * 2)
    else:
        from analysis.scripts.manage_agentic_hours import health
        root.mkdir(parents=True, exist_ok=True)
        if not health({"dataset_root": str(root), "cluster": site["cluster"]},
                      Path(site["quota_evidence"]))["safe_to_admit"]:
            raise ValueError("fresh safe storage evidence required before retrieval")
        import os
        import shutil
        root.mkdir(parents=True, exist_ok=True)
        stats = os.statvfs(root)
        if shutil.disk_usage(root).free < 2 * sum(missing) + max(missing, default=0) + 5 * 1024**3 or stats.f_favail < len(missing) * 2 + 1000:
            raise ValueError("insufficient transfer bytes or inodes")
    for bundle in sorted(bundles):
        exchange.download(bundle, root, stripes=site.get("stripes", 256), revision=revision)
    for plan_id, value in plans.items():
        atomic(Path(site["plan_dir"]) / f"{plan_id}.json", canonical(value))
    return state, plans, revision


def replan(exchange, site, state, plans):
    if site["cluster"] != "jupiter":
        raise ValueError("only JUPITER may publish replacement plans")
    root = Path(site["dataset_root"])
    tasks, done, blocked = inventory(root, stripes=site.get("stripes", 256))
    previous = plans.get(state["current_plan"])
    bundle = site.get("input_bundle") or (previous or {}).get("input_bundle")
    published_done = {fp for hour in state["hours"].values() for fp in hour["completed"]}
    if bundle:
        published_done.update(fp for fp, event in exchange.manifest(bundle)["outcomes"].items()
                              if event["state"] == "completed")
    if done - published_done:
        raise ValueError("local completions await publication; sync results before replanning")
    proposed = build_plan(tasks=tasks, calibration=read(site["calibration"]), registry=state,
                          contract=read(root / "contract.json"), completed=done, blocked=blocked,
                          source_commit=site["source_commit"], input_bundle=bundle,
                          allow_uncalibrated=True)
    existing = [h for h in state["hours"].values() if not h.get("owner")
                and h["status"] in {"available", "partial"}]
    def packages(rows):
        return sorted((r["model"], r["keyword_id"], r["priority_rank"],
                       tuple(r["task_fingerprints"]), r["reference_seconds"]) for r in rows)
    if (previous and packages(existing) == packages(proposed["packages"])
            and all(previous[k] == proposed[k] for k in (
                "calibration", "input_bundle", "contract_sha256", "source_commit", "deferred"))):
        return {"status": "unchanged", "plan_id": previous["plan_id"]}
    path = Path(site["plan_dir"]) / f"{proposed['plan_id']}.json"
    if path.exists() and read(path) != proposed:
        raise ValueError("historical plan conflicts")
    atomic(path, canonical(proposed))
    operation = "refresh-" + proposed["plan_id"]
    exchange.transact(operation, {"action": "plan", "plan_sha256": digest(proposed)},
                      lambda current: install_plan(current, proposed, cluster="jupiter"),
                      {f"coordination/plans/{proposed['plan_id']}.json": canonical(proposed)})
    return {"status": "published", "plan_id": proposed["plan_id"], "plan_file": str(path),
            "operation_id": operation}


def run(exchange, args):
    site = read(args.site)
    if site.get("cluster") not in ALLOWED_MODELS:
        raise ValueError("site must specify jupiter or horeka")
    if args.command == "select":
        _, state = exchange.snapshot()
        return select_hours(state, model=args.model, cluster=args.cluster,
                            mode=args.mode, count=args.count, first=args.first, configuration=args.configuration)
    scope = getattr(args, "scope", "results")
    if args.command == "update" and scope in {"plan", "both"} and site["cluster"] != "jupiter":
        raise ValueError("only JUPITER may refresh plans")
    if args.command != "status" and site["cluster"] == "horeka" and site.get("quota_project"):
        from analysis.scripts.prepare_horeka_qwen import capture_quota
        atomic(Path(site["quota_evidence"]), canonical(capture_quota(
            Path(site["workspace"]), site["quota_project"], site.get("quota_cluster", "hkn.scc.kit.edu"))))
    synced = []
    if args.command == "update" and scope in {"results", "both"}:
        from analysis.scripts.manage_agentic_hours import health, scheduler, sync_once
        for path in site.get("attempts", []):
            attempt = read(path)
            if attempt["cluster"] != site["cluster"]:
                raise ValueError("attempt belongs to another cluster")
            if Path(attempt["dataset_root"]).resolve() != Path(site["dataset_root"]).resolve():
                raise ValueError("attempt uses a different dataset root; use its own site configuration")
            synced.append(sync_once(exchange, attempt, scheduler(attempt),
                                    health(attempt, Path(site["quota_evidence"]))))
    if args.command == "status":
        revision, state = exchange.snapshot()
        if not (Path(site["dataset_root"]) / "contract.json").is_file():
            return {"status": "local_inputs_missing", "registry_revision": revision,
                    "current_plan": state["current_plan"], "published_hours": len(state["hours"])}
        plans = ({state["current_plan"]: load_plan(exchange, state["current_plan"], revision)}
                 if state["current_plan"] else {})
    else:
        state, plans, revision = pull(exchange, site, model=getattr(args, "model", None))
    result = {"sync": synced}
    if args.command == "update" and scope in {"plan", "both"}:
        if any(s.get("status") == "blocked" for s in synced):
            raise ValueError("local reconciliation is blocked; refusing to replan")
        result["plan"] = replan(exchange, site, state, plans)
        revision, state = exchange.snapshot()
        plans[state["current_plan"]] = load_plan(exchange, state["current_plan"], revision)
    current_plan = plans.get(state["current_plan"], {})
    if current_plan:
        bundle = exchange.manifest(current_plan['input_bundle'], revision)
        result['input_manifests'] = [str(Path(site['dataset_root']) / name) for name in bundle['files']
                                     if name.startswith('artifacts/shared-preparations/')]
    deferred = current_plan.get("deferred", {})
    if args.command == "update":
        local_tasks = {t["fingerprint"] for t in inventory(Path(site["dataset_root"]),
                                                         stripes=site.get("stripes", 256))[0]}
        if {fp for h in state["hours"].values() for fp in h["task_fingerprints"]} - local_tasks:
            raise ValueError("local mirror is incomplete; cannot replace global progress")
    report = progress(Path(site["dataset_root"]), state, revision=revision,
                      deferred=deferred, prior_completed=current_plan.get("completed_before_plan", []),
                      stripes=site.get("stripes", 256))
    if args.command == "update":
        publish_progress(exchange, report)
    if args.command != "status":
        atomic(Path(site["plan_dir"]) / 'progress.json', canonical(report))
        atomic(Path(site["plan_dir"]) / 'progress.md', markdown(report))
        result['local_progress'] = str(Path(site["plan_dir"]) / 'progress.json')
    result["progress"] = report
    return result
