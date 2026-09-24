"""Striped durable task state for batch and interactive GEODML workers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference_claims import ClaimIdentity

FORMAT_VERSION = "geodml-agentic-task-event-v1"
TERMINAL_STATES = frozenset({"completed", "terminal_failed"})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def identity_fingerprint(identity: ClaimIdentity) -> str:
    return hashlib.sha256(_canonical(asdict(identity))).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class LedgerClaim:
    fingerprint: str
    owner_id: str
    generation: int
    token: str


@dataclass(frozen=True)
class ClaimResult:
    status: str
    claim: LedgerClaim | None
    latest_event: Mapping[str, Any] | None


class StripedTaskLedger:
    """Bounded-file journal with short cross-node critical sections."""

    def __init__(self, root: Path, *, stripe_count: int = 256) -> None:
        if type(stripe_count) is not int or stripe_count <= 0 or stripe_count > 4096:
            raise ValueError("stripe_count must be between 1 and 4096")
        self.root = root
        self.stripe_count = stripe_count
        self.events = root / "events"
        self.locks = root / "locks"
        self.events.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)

    def _stripe(self, fingerprint: str) -> int:
        return int(fingerprint[:16], 16) % self.stripe_count

    def _paths(self, fingerprint: str) -> tuple[Path, Path]:
        stripe = self._stripe(fingerprint)
        return (
            self.events / f"stripe-{stripe:04d}.jsonl",
            self.locks / f"stripe-{stripe:04d}.lock",
        )

    @contextmanager
    def _locked(self, fingerprint: str) -> Iterator[Path]:
        event_path, lock_path = self._paths(fingerprint)
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield event_path

    def _history(self, path: Path, fingerprint: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"ledger has an incomplete event at {path}:{number}") from error
                if row.get("format_version") != FORMAT_VERSION:
                    raise ValueError(f"ledger event has an unsupported format at {path}:{number}")
                if row.get("fingerprint") == fingerprint:
                    rows.append(row)
        return rows

    def _append(self, path: Path, row: Mapping[str, Any]) -> None:
        raw = _canonical(dict(row)) + b"\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("ledger append made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)

    def inspect(self, identity: ClaimIdentity) -> Mapping[str, Any] | None:
        fingerprint = identity_fingerprint(identity)
        with self._locked(fingerprint) as path:
            history = self._history(path, fingerprint)
            return None if not history else history[-1]

    def claim(self, identity: ClaimIdentity, *, owner_id: str) -> ClaimResult:
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError("owner_id is required")
        fingerprint = identity_fingerprint(identity)
        with self._locked(fingerprint) as path:
            history = self._history(path, fingerprint)
            latest = None if not history else history[-1]
            if latest and latest["state"] in TERMINAL_STATES:
                return ClaimResult(latest["state"], None, latest)
            if latest and latest["state"] in {"claimed", "running", "result_saved"}:
                return ClaimResult("busy", None, latest)
            generation = max((row.get("generation", 0) for row in history), default=0) + 1
            token = hashlib.sha256(
                _canonical({"fingerprint": fingerprint, "owner_id": owner_id, "generation": generation})
            ).hexdigest()
            claim = LedgerClaim(fingerprint, owner_id, generation, token)
            event = {
                "format_version": FORMAT_VERSION,
                "fingerprint": fingerprint,
                "identity": asdict(identity),
                "state": "claimed",
                "owner_id": owner_id,
                "generation": generation,
                "token": token,
                "recorded_at": _now(),
            }
            self._append(path, event)
            return ClaimResult("owned", claim, event)

    def transition(
        self,
        claim: LedgerClaim,
        *,
        state: str,
        record_references: Sequence[Mapping[str, Any]] = (),
        detail: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if state not in {
            "running", "result_saved", "completed", "checkpointed",
            "retryable", "terminal_failed",
        }:
            raise ValueError(f"unsupported task state: {state}")
        with self._locked(claim.fingerprint) as path:
            history = self._history(path, claim.fingerprint)
            if not history:
                raise RuntimeError("claim has no ledger history")
            latest = history[-1]
            if (
                latest.get("token") != claim.token
                or latest.get("generation") != claim.generation
                or latest.get("owner_id") != claim.owner_id
            ):
                raise RuntimeError("claim generation is no longer current")
            if latest["state"] in TERMINAL_STATES:
                raise RuntimeError("task is already terminal")
            if state == "completed" and not record_references:
                raise ValueError("completion requires durable record references")
            event = {
                "format_version": FORMAT_VERSION,
                "fingerprint": claim.fingerprint,
                "state": state,
                "owner_id": claim.owner_id,
                "generation": claim.generation,
                "token": claim.token,
                "record_references": [dict(row) for row in record_references],
                "detail": {} if detail is None else dict(detail),
                "recorded_at": _now(),
            }
            self._append(path, event)
            return event

    def release_stale(
        self,
        identity: ClaimIdentity,
        *,
        expected_token: str,
        scheduler_confirmation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if scheduler_confirmation.get("owner_terminal") is not True:
            raise ValueError("stale release requires confirmed terminal owner allocation")
        fingerprint = identity_fingerprint(identity)
        with self._locked(fingerprint) as path:
            history = self._history(path, fingerprint)
            if not history or history[-1].get("token") != expected_token:
                raise RuntimeError("stale release token is not current")
            latest = history[-1]
            if latest["state"] in TERMINAL_STATES:
                raise RuntimeError("cannot release a terminal task")
            event = {
                "format_version": FORMAT_VERSION,
                "fingerprint": fingerprint,
                "state": "checkpointed",
                "owner_id": latest["owner_id"],
                "generation": latest["generation"],
                "token": latest["token"],
                "record_references": [],
                "detail": {"release": "confirmed_terminal_owner", **dict(scheduler_confirmation)},
                "recorded_at": _now(),
            }
            self._append(path, event)
            return event

    def snapshot(self) -> dict[str, Any]:
        """Read one cross-stripe snapshot without scanning unrelated artifacts."""

        latest: dict[str, dict[str, Any]] = {}
        event_count = 0
        with ExitStack() as stack:
            locks = [
                stack.enter_context((self.locks / f"stripe-{index:04d}.lock").open("a"))
                for index in range(self.stripe_count)
            ]
            for lock in locks:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for index in range(self.stripe_count):
                path = self.events / f"stripe-{index:04d}.jsonl"
                if not path.exists():
                    continue
                with path.open("rb") as stream:
                    for number, line in enumerate(stream, 1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"ledger has an incomplete event at {path}:{number}"
                            ) from error
                        if row.get("format_version") != FORMAT_VERSION:
                            raise ValueError(
                                f"ledger event has an unsupported format at {path}:{number}"
                            )
                        fingerprint = row.get("fingerprint")
                        if not isinstance(fingerprint, str):
                            raise TypeError("ledger event lacks a fingerprint")
                        latest[fingerprint] = row
                        event_count += 1
        return {
            "format_version": "geodml-agentic-ledger-snapshot-v1",
            "event_count": event_count,
            "latest": latest,
        }
