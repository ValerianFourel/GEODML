#!/usr/bin/env python3
"""Reconcile terminal dataset writers and stale task claims without inference."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.interpretability.pipeline.agentic_audit_progress import (
    audit_progress,
    audit_stage,
)
from analysis.interpretability.pipeline.agentic_dataset import (
    iter_sealed_rows,
    recover_inprogress_writer,
    verify_record_reference,
)
from analysis.interpretability.pipeline.agentic_task_ledger import (
    LedgerClaim,
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity

ACTIVE_STATES = frozenset({"claimed", "running", "result_saved"})
TERMINAL_SCHEDULER_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT",
})


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _owners(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if snapshot.get("complete") is not True:
        raise ValueError("scheduler snapshot must be marked complete")
    rows = snapshot.get("owners")
    if not isinstance(rows, list):
        raise TypeError("scheduler snapshot requires an owners list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("scheduler owner row must be an object")
        owner_id, state = row.get("owner_id"), row.get("state")
        if not isinstance(owner_id, str) or not isinstance(state, str):
            raise TypeError("scheduler owner row lacks owner_id or state")
        if owner_id in result:
            raise ValueError(f"duplicate scheduler owner: {owner_id}")
        result[owner_id] = {**row, "state": state.upper()}
    return result


@audit_stage("reconciliation")
def reconcile(
    root: Path,
    *,
    scheduler_snapshot: Mapping[str, Any],
    stripe_count: int = 256,
    apply: bool = False,
) -> dict[str, Any]:
    """Plan or apply safe terminal-owner recovery for known task identities."""

    audit_progress(phase="task_registry", tasks_checked=0)
    owners = _owners(scheduler_snapshot)
    identities: dict[str, ClaimIdentity] = {}
    for row in iter_sealed_rows(root, "task_definitions", required=True):
        identity = ClaimIdentity(**row["claim_identity"])
        fingerprint = identity_fingerprint(identity)
        if fingerprint in identities:
            raise ValueError("task definitions contain a duplicate identity")
        identities[fingerprint] = identity
        audit_progress(tasks_checked=len(identities))
    ledger = StripedTaskLedger(root / "control" / "task-ledger", stripe_count=stripe_count)
    audit_progress(phase="ledger")
    latest = ledger.snapshot()["latest"]
    actions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    recovered_writers: dict[str, list[dict[str, Any]]] = {}
    audit_progress(phase="verify_ledger", ledger_total=len(latest), ledger_checked=0)
    for index, (fingerprint, event) in enumerate(sorted(latest.items()), 1):
        audit_progress(ledger_checked=index-1, actions=len(actions), blocked=len(blocked))
        owner_id = event.get("owner_id")
        owner = owners.get(owner_id) if isinstance(owner_id, str) else None
        if owner is None or owner["state"] not in TERMINAL_SCHEDULER_STATES:
            continue
        state = event.get("state")
        if apply and owner_id not in recovered_writers:
            recovered_writers[owner_id] = recover_inprogress_writer(
                root, writer_id=owner_id
            )
        if state in {"completed", "terminal_failed", "result_saved"}:
            references = event.get("record_references", [])
            references_valid = bool(references) and all(
                isinstance(reference, dict)
                and verify_record_reference(root, reference)
                for reference in references
            )
            if state == "result_saved" and not references_valid:
                blocked.append({
                    "fingerprint": fingerprint,
                    "owner_id": owner_id,
                    "reason": "saved_response_references_not_sealed_or_invalid",
                })
                continue
            if state == "result_saved":
                action = "complete_saved_result"
                if apply:
                    claim = LedgerClaim(
                        fingerprint=fingerprint,
                        owner_id=owner_id,
                        generation=event["generation"],
                        token=event["token"],
                    )
                    ledger.transition(
                        claim,
                        state="completed",
                        record_references=references,
                        detail={"reconciled_terminal_owner": dict(owner)},
                    )
                actions.append({
                    "fingerprint": fingerprint,
                    "owner_id": owner_id,
                    "action": action,
                })
            elif references and not references_valid:
                blocked.append({
                    "fingerprint": fingerprint,
                    "owner_id": owner_id,
                    "reason": "terminal_record_references_not_sealed_or_invalid",
                })
            continue
        if state in {"claimed", "running"}:
            if fingerprint not in identities:
                blocked.append({
                    "fingerprint": fingerprint,
                    "owner_id": owner_id,
                    "reason": "ledger_identity_absent_from_task_registry",
                })
                continue
            if apply:
                ledger.release_stale(
                    identities[fingerprint],
                    expected_token=event["token"],
                    scheduler_confirmation={
                        "owner_terminal": True,
                        "scheduler_state": owner["state"],
                        "job_id": owner.get("job_id"),
                    },
                )
            actions.append({
                "fingerprint": fingerprint,
                "owner_id": owner_id,
                "action": "release_uncommitted_claim",
            })
    audit_progress(ledger_checked=len(latest), actions=len(actions), blocked=len(blocked))
    return {
        "format_version": "geodml-agentic-reconciliation-v1",
        "applied": apply,
        "actions": actions,
        "blocked": blocked,
        "recovered_writers": {
            writer: [manifest["path"] for manifest in manifests]
            for writer, manifests in recovered_writers.items()
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scheduler-snapshot", type=Path, required=True)
    parser.add_argument("--stripe-count", type=int, default=256)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = json.loads(args.scheduler_snapshot.read_text(encoding="utf-8"))
    result = reconcile(
        args.dataset_root,
        scheduler_snapshot=snapshot,
        stripe_count=args.stripe_count,
        apply=args.apply,
    )
    if args.receipt:
        if args.receipt.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
        _atomic(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
