#!/usr/bin/env python3
"""Publish verified Qwen cells from the shared JUPITER ledger to the private Hub.

Qwen waves write only to the shared on-disk ledger; nothing uploaded them. This
finite helper collects every completed or terminally failed qwen38 cell whose
record references are sealed, skips cells already listed in
coordination/qwen-results.json, and uploads the rest as sealed bundles (one per
writer) in the same verified format as Llama checkpoints. It never changes the
hour registry, claims, or running work. Dry run unless --apply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows, verify_record_reference
from analysis.interpretability.pipeline.agentic_hour_sync import ConflictError, Exchange, HubStore
from analysis.interpretability.pipeline.agentic_hours import canonical
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity

INDEX_PATH = "coordination/qwen-results.json"
INDEX_VERSION = "geodml-qwen-results-v1"
MODEL = "qwen38"


def collect(root: Path, *, stripes: int = 256) -> dict[str, dict]:
    """Verified terminal qwen38 outcomes by fingerprint, in bundle outcome format."""
    tasks = {}
    for row in iter_sealed_rows(root, "task_definitions", required=True):
        if row.get("model") == MODEL:
            tasks[identity_fingerprint(ClaimIdentity(**row["claim_identity"]))] = row
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
    outcomes = {}
    for fp, task in tasks.items():
        event = latest.get(fp, {})
        if event.get("state") not in {"completed", "terminal_failed"}:
            continue
        refs = event.get("record_references", [])
        if (event["state"] == "completed" and not refs) or not all(verify_record_reference(root, ref) for ref in refs):
            continue
        outcomes[fp] = {**event, "identity": task["claim_identity"]}
    return outcomes


def shard_names(outcomes: dict) -> list[str]:
    names = set()
    for event in outcomes.values():
        for ref in event.get("record_references", []):
            stem = f"data/{ref['table']}/part-{ref['writer_id']}-{ref['shard_sequence']:06d}"
            names.update((stem + ".jsonl", stem + ".manifest.json"))
    return sorted(names)


def read_index(store, revision) -> dict:
    raw = store.read(INDEX_PATH, revision)
    return {"format_version": INDEX_VERSION, "model": MODEL, "bundles": []} if raw is None else json.loads(raw)


def published_fingerprints(exchange: Exchange, index: dict) -> set[str]:
    done = set()
    for entry in index["bundles"]:
        done.update(exchange.manifest(entry["bundle"])["outcomes"])
    return done


def publish(exchange: Exchange, root: Path, *, stripes: int = 256, apply: bool = False,
            source_commit: str = "unknown") -> dict:
    outcomes = collect(root, stripes=stripes)
    index = read_index(exchange.store, exchange.store.head())
    already = published_fingerprints(exchange, index)
    new = {fp: event for fp, event in outcomes.items() if fp not in already}
    by_writer = defaultdict(dict)
    for fp, event in new.items():
        by_writer[str(event.get("owner_id"))][fp] = event
    plan = []
    for writer, events in sorted(by_writer.items()):
        states = [event["state"] for event in events.values()]
        plan.append({"writer_id": writer, "completed": states.count("completed"),
                     "failed": states.count("terminal_failed"), "files": shard_names(events), "outcomes": events})
    report = {"dataset_root": str(root), "verified_on_disk": len(outcomes), "already_published": len(already),
              "new_cells": len(new), "writers": len(plan), "files": len({n for p in plan for n in p["files"]}),
              "applied": False, "bundles": [], "blocked": []}
    if not apply or not plan:
        return report
    bundles, errors = exchange.upload_many(
        [(root, item["files"], item["outcomes"],
          {"kind": "qwen-results", "model": MODEL, "writer_id": item["writer_id"],
           "dataset_root": root.name, "source_commit": source_commit}) for item in plan],
        isolate_errors=True)
    entries = []
    for position, item in enumerate(plan):
        if position in errors:
            report["blocked"].append({"writer_id": item["writer_id"], "error": errors[position]})
            continue
        # Mark published only after the remote copy verifies against local files.
        exchange.download(bundles[position], root, stripes=stripes, import_outcomes=False, verify_remote=True)
        entries.append({"bundle": bundles[position], "writer_id": item["writer_id"],
                        "completed": item["completed"], "failed": item["failed"],
                        "published_at": int(time.time()), "source_commit": source_commit})
    for _ in range(8):
        revision = exchange.store.head()
        index = read_index(exchange.store, revision)
        known = {entry["bundle"] for entry in index["bundles"]}
        index["bundles"] += [entry for entry in entries if entry["bundle"] not in known]
        index["completed_total"] = sum(entry["completed"] for entry in index["bundles"])
        index["failed_total"] = sum(entry["failed"] for entry in index["bundles"])
        try:
            exchange.store.commit(revision, {INDEX_PATH: canonical(index)}, "GEODML qwen results index")
            break
        except ConflictError:
            continue
    else:
        raise ConflictError("results index busy; rerun (uploads are reused)")
    report.update(applied=True, bundles=entries, completed_total=index["completed_total"],
                  failed_total=index["failed_total"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="ValerianFourel/geodml-experiment-v2-paper-private")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--stripes", type=int, default=256)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--apply", action="store_true", help="upload; without it only counts are printed")
    args = parser.parse_args(argv)
    exchange = Exchange(HubStore(args.repo_id), args.journal)
    report = publish(exchange, args.dataset_root.resolve(), stripes=args.stripes, apply=args.apply,
                     source_commit=args.source_commit)
    print(json.dumps({k: v for k, v in report.items() if k != "bundles"}, indent=2))
    for entry in report["bundles"]:
        print("PUBLISHED " + json.dumps(entry))
    return 1 if report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
