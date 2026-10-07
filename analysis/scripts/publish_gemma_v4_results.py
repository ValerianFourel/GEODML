#!/usr/bin/env python3
"""Publish the saved Gemma SI-v4 judgments of finished shards to the private Hub (login host, CPU/I/O).

For each run root (a five-hour-bout plan) it takes, per shard, the latest judge report written at the end of
a bout: cells.jsonl.gz (every cell with its source grades, raw outputs and metrics), maps.jsonl (every answer
map, failed ones marked) and summary.json. Only shards whose report says finished or finished_with_failures
are published; others are listed as pending and can be published by a later rerun. Run-level files: plan.json,
judge-config.json and frozen/manifest.json.

Hub layout: reviews/gemma-si-v4/<plan_id>/{plan.json,...}, .../shards/<shard>/<file>, and
.../export-manifest.json (sha256 and bytes of every file, shard status and counts), written last.
Files already on the Hub with identical content are skipped, so a rerun only adds what changed.
After uploading, every file is checked against the server's own hash. Dry run unless --apply.
Never allocates, infers, edits runs or touches claims. Judgments are diagnostic: semantic acceptance is
not established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PREFIX = "reviews/gemma-si-v4"
SHARD_FILES = ("cells.jsonl.gz", "maps.jsonl", "summary.json")
RUN_FILES = ("plan.json", "judge-config.json", "frozen/manifest.json")
DONE = ("finished", "finished_with_failures")


def git_blob_id(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def collect(root: Path) -> dict:
    """Local files to publish for one run, keyed by Hub path, plus the shard table."""
    root = root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    base = f"{PREFIX}/{plan['plan_id']}"
    files, shards = {}, []
    for name in RUN_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing run file {path}")
        files[f"{base}/{name.replace('/', '-')}"] = path
    for shard in plan["shards"]:
        results = Path(shard["directory"]) / "results"
        latest = results / "reports/latest.json"
        row = {"shard": shard["id"], "cells": shard["cells"], "status": "not_started"}
        if latest.is_file():
            writer = json.loads(latest.read_text()).get("directory")
            if not isinstance(writer, str) or Path(writer).name != writer or writer in (".", ".."):
                raise ValueError(f"invalid report pointer in {latest}")
            report = results / "reports" / writer
            summary = json.loads((report / "summary.json").read_text())
            row.update(status=summary.get("status"), report=writer, states=summary.get("states", {}),
                       inference_failures=summary.get("inference_failures"))
            if summary.get("status") in DONE:
                for name in SHARD_FILES:
                    path = report / name
                    if not path.is_file():
                        raise ValueError(f"finished report lacks {path}")
                    files[f"{base}/shards/{shard['id']}/{name}"] = path
        shards.append(row)
    return {"root": str(root), "plan_id": plan["plan_id"], "base": base, "files": files, "shards": shards}


def publish(store, runs: list[dict], *, apply: bool, batch_bytes: int = 400_000_000) -> dict:
    """Upload missing or changed files in bounded commits, verify by server hashes, then write manifests."""
    from analysis.interpretability.pipeline.agentic_hour_sync import ConflictError
    local = {}
    for run in runs:
        for name, path in run["files"].items():
            raw = path.read_bytes()
            local[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "blob_id": git_blob_id(raw), "bytes": len(raw),
                           "path": path}

    def matches(name, info):
        return bool(info) and (info.get("sha256") == local[name]["sha256"] or info.get("blob_id") == local[name]["blob_id"])

    revision = store.head()
    remote = store.hashes(sorted(local), revision)
    todo = [n for n in sorted(local) if not matches(n, remote.get(n))]
    report = {"runs": [{"plan_id": r["plan_id"], "root": r["root"], "files": len(r["files"]),
                        "shards_published": sum(s["status"] in DONE for s in r["shards"]),
                        "shards_pending": [s["shard"] for s in r["shards"] if s["status"] not in DONE]} for r in runs],
              "files_total": len(local), "files_to_upload": len(todo),
              "bytes_to_upload": sum(local[n]["bytes"] for n in todo), "applied": False}
    if not apply:
        return report
    batches, current, size = [], [], 0
    for name in todo:
        if current and size + local[name]["bytes"] > batch_bytes:
            batches.append(current)
            current, size = [], 0
        current.append(name)
        size += local[name]["bytes"]
    if current:
        batches.append(current)
    for number, batch in enumerate(batches, 1):
        for _ in range(8):
            revision = store.head()
            try:
                revision = store.commit(revision, {n: local[n]["path"].read_bytes() for n in batch},
                                        f"Gemma SI-v4 judgments: batch {number}/{len(batches)}")
                break
            except ConflictError:
                continue
        else:
            raise ConflictError("repository kept advancing; rerun (finished batches are skipped)")
        print(json.dumps({"committed_batch": number, "of": len(batches), "files": len(batch), "revision": revision}),
              flush=True)
    revision = store.head()
    remote = store.hashes(sorted(local), revision)
    bad = [n for n in sorted(local) if not matches(n, remote.get(n))]
    if bad:
        raise ValueError(f"server hashes differ for {len(bad)} files, e.g. {bad[:3]}; nothing marked published")
    manifests = {}
    for run in runs:
        manifests[f"{run['base']}/export-manifest.json"] = json.dumps({
            "format_version": "gemma-si-v4-export-v1", "plan_id": run["plan_id"], "source_root": run["root"],
            "semantic_acceptance": "not_established", "shards": run["shards"],
            "files": {n: {k: local[n][k] for k in ("sha256", "bytes")} for n in sorted(run["files"])}},
            indent=1, sort_keys=True).encode()
    for _ in range(8):
        revision = store.head()
        try:
            revision = store.commit(revision, manifests, "Gemma SI-v4 export manifests")
            break
        except ConflictError:
            continue
    else:
        raise ConflictError("repository kept advancing while writing manifests; rerun")
    report.update(applied=True, revision=revision, verified_files=len(local),
                  manifests=sorted(manifests))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True, help="Gemma v4 run root with plan.json")
    parser.add_argument("--repo-id", default="ValerianFourel/geodml-experiment-v2-paper-private")
    parser.add_argument("--apply", action="store_true", help="upload; without it only counts are printed")
    args = parser.parse_args(argv)
    runs = [collect(root) for root in args.run]
    from analysis.interpretability.pipeline.agentic_hour_sync import HubStore
    report = publish(HubStore(args.repo_id), runs, apply=args.apply)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
