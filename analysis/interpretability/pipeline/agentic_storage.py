"""Storage incidents and admission health for the incremental dataset."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STORAGE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
FORMAT_VERSION = "geodml-agentic-storage-snapshot-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def record_storage_incident(
    root: Path,
    error: OSError,
    *,
    operation: str,
    writer_id: str | None = None,
    table: str | None = None,
) -> Path | None:
    """Best-effort durable incident marker; never masks the original error."""

    if error.errno not in STORAGE_ERRNOS:
        return None
    value = {
        "format_version": "geodml-agentic-storage-incident-v1",
        "recorded_at": _now(),
        "recorded_at_epoch_ns": time.time_ns(),
        "errno": error.errno,
        "error": str(error),
        "operation": operation,
        "writer_id": writer_id,
        "table": table,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    incident_id = "incident-" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    path = root / "control" / "storage-incidents" / f"{incident_id}.json"
    try:
        _atomic(path, {**value, "incident_id": incident_id})
    except OSError:
        return None
    return path


def _probe(root: Path) -> None:
    directory = root / "control" / "storage-probes"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=True) as stream:
        stream.write(b"x" * 4096)
        stream.flush()
        os.fsync(stream.fileno())


def acknowledge_incidents(root: Path, incident_ids: Sequence[str]) -> list[str]:
    """Acknowledge named incidents only after a fresh durable write succeeds."""

    _probe(root)
    acknowledged: list[str] = []
    for incident_id in incident_ids:
        source = root / "control" / "storage-incidents" / f"{incident_id}.json"
        if not source.is_file():
            raise ValueError(f"unknown storage incident: {incident_id}")
        source_value = json.loads(source.read_text(encoding="utf-8"))
        target = root / "control" / "storage-acknowledgments" / f"{incident_id}.json"
        value = {
            "format_version": "geodml-agentic-storage-acknowledgment-v1",
            "incident_id": incident_id,
            "incident_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "acknowledged_at": _now(),
            "write_probe": "passed",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != value:
                # The timestamp may differ; a prior valid acknowledgment wins.
                saved = json.loads(target.read_text(encoding="utf-8"))
                if (
                    saved.get("incident_id") != incident_id
                    or saved.get("incident_sha256") != value["incident_sha256"]
                    or saved.get("write_probe") != "passed"
                ):
                    raise ValueError(f"storage acknowledgment conflicts: {incident_id}")
        else:
            _atomic(target, value)
        if source_value.get("incident_id") != incident_id:
            raise ValueError(f"storage incident identity mismatch: {incident_id}")
        acknowledged.append(incident_id)
    return acknowledged


def storage_health(
    root: Path,
    *,
    minimum_free_bytes: int = 20 * 1024**3,
    minimum_free_inodes: int = 100_000,
    quota_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture admission health without attempting to infer site quota limits."""

    if minimum_free_bytes < 0 or minimum_free_inodes < 0:
        raise ValueError("storage reserves must be non-negative")
    stats = os.statvfs(root)
    available_bytes = stats.f_bavail * stats.f_frsize
    available_inodes = stats.f_favail
    incident_root = root / "control" / "storage-incidents"
    acknowledgment_root = root / "control" / "storage-acknowledgments"
    incidents = sorted(incident_root.glob("incident-*.json")) if incident_root.exists() else []
    unacknowledged = [
        path.stem
        for path in incidents
        if not (acknowledgment_root / path.name).is_file()
    ]
    quota_verified = bool(
        isinstance(quota_evidence, Mapping)
        and quota_evidence.get("fresh") is True
        and quota_evidence.get("within_limits") is True
    )
    reasons = []
    if available_bytes < minimum_free_bytes:
        reasons.append("free_bytes_below_reserve")
    if available_inodes < minimum_free_inodes:
        reasons.append("free_inodes_below_reserve")
    if unacknowledged:
        reasons.append("unacknowledged_storage_incident")
    if quota_evidence is not None and not quota_verified:
        reasons.append("quota_evidence_not_fresh_or_over_limit")
    identity = {
        "dataset_root": str(root.resolve()),
        "available_bytes": available_bytes,
        "available_inodes": available_inodes,
        "minimum_free_bytes": minimum_free_bytes,
        "minimum_free_inodes": minimum_free_inodes,
        "unacknowledged_incidents": unacknowledged,
        "quota_evidence": None if quota_evidence is None else dict(quota_evidence),
    }
    return {
        "format_version": FORMAT_VERSION,
        "snapshot_id": "storage-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24],
        "captured_at": _now(),
        "captured_at_epoch": int(time.time()),
        **identity,
        "quota_verified": quota_verified,
        "safe_to_admit": not reasons,
        "reasons": reasons,
    }
