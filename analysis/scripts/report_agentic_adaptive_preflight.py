#!/usr/bin/env python3
"""Read the frozen inputs needed before wiring an adaptive 500-prompt run.

This performs no inference, Slurm submission, claim import or output mutation.
It reports incompatibilities rather than choosing a checkout or dataset by mtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.adaptive_inference import (
    file_sha256,
)
from analysis.interpretability.pipeline.inference_claims import (
    ClaimIdentity,
    InferenceClaimStore,
)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "value": json.loads(path.read_text(encoding="utf-8")),
    }


def run_record(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"run root does not exist: {root}")
    configs = set()
    for pattern in (
        "models/*/shard-*/config.json",
        "models/*/outputs/*/config.json",
        "models/*/outputs/**/config.json",
        "models/*/config.json",
    ):
        configs.update(root.glob(pattern))
    result: dict[str, Any] = {
        "run_root": str(root.resolve()),
        "configs": [],
        "issues": [],
    }
    for path in sorted(configs):
        record = file_record(path)
        config = record["value"]
        record["inputs"] = []
        for entry in [
            *config.get("prompt_sources", {}).values(),
            *config.get("search_snapshots", {}).values(),
        ]:
            input_path = Path(entry["path"])
            actual = file_sha256(input_path) if input_path.is_file() else None
            record["inputs"].append(
                {
                    "path": str(input_path),
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": actual,
                }
            )
            if actual is None or actual != entry.get("sha256"):
                result["issues"].append(
                    f"missing or changed frozen input: {input_path}"
                )
        result["configs"].append(record)
    if not configs:
        result["issues"].append("no worker configurations found in supported layouts")
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        result["manifest"] = file_record(manifest_path)
    tasks_path = root / "tasks.jsonl"
    if tasks_path.is_file():
        ids = [
            json.loads(line)["cell_id"]
            for line in tasks_path.read_text().splitlines()
            if line.strip()
        ]
        result["tasks"] = {
            "path": str(tasks_path.resolve()),
            "sha256": file_sha256(tasks_path),
            "rows": len(ids),
            "unique_cells": len(set(ids)),
        }
        if len(ids) != len(set(ids)):
            result["issues"].append("duplicate canonical task IDs")
    return result


def registry_record(roots: list[Path]) -> dict[str, Any]:
    """Report exact-identity overlap, not an unqualified sum of saved files."""
    locations: dict[ClaimIdentity, dict[str, str]] = defaultdict(dict)
    variants: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    per_root = {}
    errors = []
    for root in sorted(set(roots)):
        counts: Counter[str] = Counter()
        store = InferenceClaimStore(root)
        if not root.is_dir():
            errors.append({"path": str(root), "error": "claim directory missing"})
        seen = set()
        for path in sorted(root.glob("*/*.json")):
            try:
                identity = ClaimIdentity(**json.loads(path.read_text())["identity"])
                if identity in seen:
                    continue
                seen.add(identity)
                # inspect also verifies the filename-derived scientific identity,
                # envelope, payload checksum and mutually exclusive terminal states.
                state, _ = store.inspect(identity)
                if state == "missing":
                    raise ValueError(
                        "record does not match its identity-derived filename"
                    )
                counts[state] += 1
                locations[identity][str(root)] = state
                variants[
                    (
                        identity.task_id,
                        identity.model_id,
                        identity.model_revision,
                        identity.protocol,
                    )
                ].add(identity.request_sha256)
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append({"path": str(path), "error": str(error)})
        per_root[str(root)] = dict(counts)
    overlapping = sum(
        sum(state == "completed" for state in states.values()) > 1
        for states in locations.values()
    )
    mixed = sum(len(hashes) > 1 for hashes in variants.values())
    conflicts = []
    for identity, states in locations.items():
        if len(states) > 1 and "completed" in states.values():
            records = [
                InferenceClaimStore(Path(root)).inspect(identity) for root in states
            ]
            successes = [value for status, value in records if status == "completed"]
            if any(status == "failed" for status, _ in records) or any(
                value != successes[0] for value in successes[1:]
            ):
                conflicts.append(asdict(identity))
    return {
        "per_root": per_root,
        "unique_completed_identities": sum(
            "completed" in states.values() for states in locations.values()
        ),
        "overlapping_success_identities": overlapping,
        "tasks_with_multiple_request_hashes": mixed,
        "conflicts": conflicts,
        "errors": errors,
        "note": "Identity overlap only. This is not proof of compatibility with the new checkout.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path)
    parser.add_argument("--backlog-run-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    runs = [
        run_record(path.resolve())
        for path in [
            args.study_root,
            *([args.paired_root] if args.paired_root else []),
            *args.backlog_run_root,
        ]
    ]
    claim_roots = []
    for run in runs:
        manifest = run.get("manifest", {}).get("value", {})
        raw = manifest.get("claim_root")
        if raw is not None:
            if not isinstance(raw, str) or not Path(raw).is_absolute():
                raise ValueError("recorded shared claim roots must be absolute")
            claim_roots.append(Path(raw).resolve())
    report = {
        "format_version": "agentic-adaptive-input-audit-v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "registries": registry_record(claim_roots),
        "launch_ready": False,
        "note": "Read-only input audit; no adaptive plan has been frozen or launched.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return (
        0
        if not report["registries"]["errors"] and not any(run["issues"] for run in runs)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
