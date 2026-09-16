"""Shared, durable ownership and completed outcomes for independent batch jobs.

Every writer must use the same root on a filesystem supporting cross-node
``flock`` and atomic rename. Lock files are permanent; deleting or replacing a
live lock file would split ownership. There is no lease timeout or lock stealing.
The operating system releases ownership when a process exits. Commit the shared
outcome before appending a worker-local journal so a crash can reuse the result.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = "geodml-inference-claim-v1"
FAILURE_FORMAT_VERSION = "geodml-inference-failure-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field in claim record: {key}")
        value[key] = item
    return value


@dataclass(frozen=True)
class ClaimIdentity:
    """Scientific identity only; worker layout and output paths are excluded."""

    task_id: str
    model_id: str
    model_revision: str
    protocol: str
    request_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"claim identity {name} must be non-empty text")
        if re.fullmatch(r"[0-9a-f]{64}", self.request_sha256) is None:
            raise ValueError("claim identity request_sha256 must be a SHA-256 digest")


def _identity(value: ClaimIdentity | Mapping[str, str]) -> ClaimIdentity:
    if isinstance(value, ClaimIdentity):
        return value
    if not isinstance(value, Mapping) or set(value) != set(ClaimIdentity.__dataclass_fields__):
        raise ValueError("claim identity must contain exactly the five scientific fields")
    return ClaimIdentity(**value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if path.is_dir():
        return
    _ensure_directory(path.parent)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if not path.is_dir():
            raise
    else:
        _fsync_directory(path.parent)


def _validate_outcome(
    outcome: Any, validate: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    if not isinstance(outcome, dict):
        # Match the JSON-validation error contract used by callers.
        raise ValueError("claim outcome must be a JSON object")  # noqa: TRY004
    original = _canonical(outcome)
    if validate is not None:
        validate(outcome)
        if _canonical(outcome) != original:
            raise ValueError("claim outcome validator must not mutate its input")
    return outcome


class TaskClaim:
    """A claim valid only inside ``try_claim``; commit never releases its lock."""

    def __init__(
        self, *, status: str, identity: ClaimIdentity, path: Path,
        validate: Callable[[dict[str, Any]], None] | None,
        outcome: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
        validate_failure: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._status = status
        self.fingerprint = _digest(asdict(identity))
        self.outcome = outcome
        self.failure = failure
        self._identity = identity
        self._path = path
        self._validate = validate
        self._validate_failure = validate_failure
        self._active = True

    @property
    def status(self) -> str:
        return self._status

    def commit(self, outcome: Mapping[str, Any]) -> None:
        """Validate and atomically persist one outcome while retaining ownership."""
        self._persist(outcome, failed=False)

    def fail(self, failure: Mapping[str, Any]) -> None:
        """Persist a terminal failure after bounded attempts, never on cancellation.

        Subsequent jobs report this failure without repeating inference. A repair
        needs an explicit changed protocol or a separately reviewed recovery;
        new job IDs do not silently reset the retry budget.
        """
        self._persist(failure, failed=True)

    def _persist(self, outcome: Mapping[str, Any], *, failed: bool) -> None:
        if not self._active or self.status != "owned":
            raise RuntimeError("only an active owned task claim can commit")
        if not isinstance(outcome, Mapping):
            # Match the JSON-validation error contract used by callers.
            raise ValueError("claim outcome must be a JSON object")  # noqa: TRY004
        # Copy through JSON so later caller mutations cannot change the record.
        validate = self._validate_failure if failed else self._validate
        saved = _validate_outcome(json.loads(_canonical(dict(outcome))), validate)
        failure_path = self._path.with_suffix(".failed.json")
        target = failure_path if failed else self._path
        record = {
            "format_version": FAILURE_FORMAT_VERSION if failed else FORMAT_VERSION,
            "identity": asdict(self._identity),
            "identity_sha256": self.fingerprint,
            "outcome": saved,
            "outcome_sha256": _digest(saved),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=self._path.parent, prefix=".claim-", suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(_canonical(record) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Under our held lock another compliant writer cannot create this.
            if self._path.exists() or failure_path.exists():
                raise FileExistsError("refusing to overwrite a committed task outcome")
            os.replace(temporary, target)
            _fsync_directory(self._path.parent)
        except BaseException:
            # A failed durability step has uncertain commit status. Never allow
            # this claim to execute/commit again; reacquire and inspect the file.
            self._active = False
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if failed:
            self.failure = saved
            self._status = "failed"
        else:
            self.outcome = saved
            self._status = "completed"


class InferenceClaimStore:
    """Nonblocking task locks plus immutable, fingerprint-checked outcomes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _read(
        self, path: Path, identity: ClaimIdentity,
        validate: Callable[[dict[str, Any]], None] | None,
        *, format_version: str = FORMAT_VERSION,
    ) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"corrupt shared task outcome: {path}") from error
        expected_keys = {"format_version", "identity", "identity_sha256", "outcome", "outcome_sha256"}
        if (
            not isinstance(record, dict) or set(record) != expected_keys
            or record["format_version"] != format_version
            or record["identity"] != asdict(identity)
            or record["identity_sha256"] != _digest(asdict(identity))
            or record["outcome_sha256"] != _digest(record["outcome"])
        ):
            raise ValueError(f"mismatched shared task outcome envelope: {path}")
        return _validate_outcome(record["outcome"], validate)

    @contextmanager
    def try_claim(
        self, identity: ClaimIdentity | Mapping[str, str], *,
        validate: Callable[[dict[str, Any]], None] | None = None,
        validate_failure: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[TaskClaim]:
        """Yield ``busy``, ``completed``, ``failed`` or ``owned`` without waiting.

        ``validate`` must raise for semantically invalid outcomes. It runs on
        both commit and reuse. Without it, the caller must validate semantics;
        the store still validates the durable identity and content fingerprints.
        """
        identity = _identity(identity)
        fingerprint = _digest(asdict(identity))
        directory = self.root / fingerprint[:2]
        _ensure_directory(directory)
        lock_path = directory / f"{fingerprint}.lock"
        path = directory / f"{fingerprint}.json"
        failure_path = path.with_suffix(".failed.json")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        locked = False
        claim = None
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                claim = TaskClaim(status="busy", identity=identity, path=path, validate=validate)
            else:
                locked = True
                if path.exists() and failure_path.exists():
                    raise ValueError(f"both success and failure records exist: {path}")
                outcome = self._read(path, identity, validate) if path.exists() else None
                failure = self._read(
                    failure_path, identity, validate_failure,
                    format_version=FAILURE_FORMAT_VERSION,
                ) if failure_path.exists() else None
                if outcome is not None or failure is not None:
                    # Recover the rename-before-directory-fsync crash window.
                    _fsync_directory(directory)
                claim = TaskClaim(
                    status="completed" if outcome is not None else "failed" if failure is not None else "owned",
                    identity=identity, path=path, validate=validate, outcome=outcome,
                    failure=failure, validate_failure=validate_failure,
                )
            yield claim
        finally:
            if claim is not None:
                claim._active = False
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
