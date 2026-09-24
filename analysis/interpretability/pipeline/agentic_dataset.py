"""Append-safe final-format JSONL shards for Experiment V2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agentic_storage import record_storage_incident

CONTRACT_VERSION = "geodml-incremental-dataset-v1"
TABLE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
WRITER_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
SECRET_PATTERN = re.compile(
    r"hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9_.-]{20,}"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def initialize_dataset(root: Path, *, population_id: str, acceptance_policy_id: str) -> dict[str, Any]:
    """Create or verify the single versioned dataset root."""

    if not population_id or not acceptance_policy_id:
        raise ValueError("population and acceptance policy IDs are required")
    contract = {
        "format_version": CONTRACT_VERSION,
        "population_id": population_id,
        "acceptance_policy_id": acceptance_policy_id,
        "publication_allowlist": [
            "README.md", "contract.json", "schemas", "data", "artifacts",
            "manifests", "plans", "reports",
        ],
        "publication_exclusions": ["control", "local-only", "publication"],
    }
    path = root / "contract.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != contract:
            raise ValueError("dataset root has a different immutable contract")
        required = (root / "schemas" / "records-v1.json", root / "README.md")
        if not all(item.is_file() for item in required):
            raise ValueError("dataset root has an incomplete initialization")
        return saved
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("refusing to initialize a non-empty dataset root")
    for name in (
        "schemas", "data", "artifacts", "manifests", "plans", "reports",
        "control", "publication", "local-only",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    schema = {
        "format_version": "geodml-incremental-record-schema-v1",
        "envelope": {
            "required": ["record_id", "transaction_id", "row"],
            "record_id": "globally stable content/transaction-derived identifier",
            "transaction_id": "one inference attempt generation or registration",
            "row": "table-specific JSON object",
        },
        "tables": {
            "prompts": ["prompt_id", "prompt_text", "prompt_sha256", "source"],
            "keyword_memberships": [
                "prompt_id", "keyword_ids", "primary_keyword_id",
                "primary_priority_rank",
            ],
            "task_definitions": [
                "task_id", "prompt_id", "model", "stage", "claim_identity",
                "identity_fingerprint", "dependency_fingerprints", "runnable_task",
            ],
            "generations": [
                "cell_id", "ranking", "answer", "trace_record_id",
            ],
            "traces": ["trace_sha256", "events"],
            "diagnostics": ["cell_id", "status", "llm_calls"],
            "transport_attempts": ["task_context", "event"],
            "judge_inputs": [
                "judge_role", "task", "rendered_prompt", "request_sha256",
            ],
            "judgments": [
                "judge_task_id", "input_record_id", "raw_output", "parsed_output",
            ],
            "failed_attempts": ["request_sha256", "error"],
            "provenance": ["slurm_job_id"],
            "legacy_imports": [
                "import_id", "artifact_root", "findings", "scientific_completion",
            ],
        },
        "artifact_policy": {
            "included": (
                "frozen prompts, configurations, normalized recovery tables, and "
                "explicitly copied evidence/provenance artifacts"
            ),
            "excluded": [
                "credentials", "model weights", "download caches", "control locks",
                "active shards",
            ],
            "references": (
                "Every externally retained raw artifact must be identified as an "
                "external reference; an index row does not imply copied bytes."
            ),
        },
    }
    readme = f"""---
pretty_name: GEODML Experiment V2 incremental dataset
---

# GEODML Experiment V2

This private dataset is written directly by the Experiment V2 workers.
Immutable JSONL shards contain frozen prompts and keyword memberships, task
definitions, generator rankings and answers, ordered traces, transport attempts,
Nemotron inputs and judgments, failures, diagnostics, and provenance. The
dataset contract is `{CONTRACT_VERSION}`. Files under `control/`, active shards,
credentials, model weights, and caches are excluded from publication.

Scientific completion must be computed from validated task identities and
checksum-verified record references. File counts and transport attempts are not
scientific completion counts.
"""
    _atomic_json(root / "schemas" / "records-v1.json", schema)
    _atomic_bytes(root / "README.md", readme.encode("utf-8"))
    # The immutable contract is the initialization commit marker. Write it last
    # so readers cannot accept a root whose schema or dataset card was interrupted.
    _atomic_json(path, contract)
    _fsync_directory(root)
    return contract


@dataclass(frozen=True)
class RecordReference:
    table: str
    writer_id: str
    shard_sequence: int
    line_number: int
    record_id: str
    transaction_id: str


class JsonlShardWriter:
    """One writer for one table; active shards never enter publication manifests."""

    def __init__(
        self,
        root: Path,
        *,
        table: str,
        writer_id: str,
        shard_sequence: int = 0,
        maximum_shard_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if TABLE_PATTERN.fullmatch(table) is None:
            raise ValueError("invalid table name")
        if WRITER_PATTERN.fullmatch(writer_id) is None:
            raise ValueError("invalid writer ID")
        if type(shard_sequence) is not int or shard_sequence < 0:
            raise ValueError("shard sequence must be a non-negative integer")
        if type(maximum_shard_bytes) is not int or maximum_shard_bytes <= 0:
            raise ValueError("maximum shard bytes must be positive")
        contract = root / "contract.json"
        if not contract.is_file():
            raise ValueError("dataset root is not initialized")
        self.root = root
        self.table = table
        self.writer_id = writer_id
        self.shard_sequence = shard_sequence
        self.maximum_shard_bytes = maximum_shard_bytes
        self.directory = root / "data" / table
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"part-{writer_id}-{shard_sequence:06d}"
        self.active_path = self.directory / f"{stem}.jsonl.inprogress"
        self.sealed_path = self.directory / f"{stem}.jsonl"
        self.manifest_path = self.directory / f"{stem}.manifest.json"
        if self.sealed_path.exists() or self.manifest_path.exists():
            raise FileExistsError("shard sequence is already sealed")
        self.stream = self.active_path.open("ab", buffering=0)
        self._lock = threading.Lock()
        self.line_number = sum(1 for _ in self.active_path.open("rb"))
        self.record_ids: set[str] = set()
        with self.active_path.open("rb") as existing:
            for number, line in enumerate(existing, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"incomplete active shard at line {number}") from error
                record_id = row.get("record_id")
                if not isinstance(record_id, str) or record_id in self.record_ids:
                    raise ValueError("active shard has invalid or duplicate record IDs")
                self.record_ids.add(record_id)

    def append(
        self,
        row: Mapping[str, Any],
        *,
        transaction_id: str,
        record_id: str | None = None,
    ) -> RecordReference:
        with self._lock:
            if not isinstance(transaction_id, str) or not transaction_id:
                raise ValueError("transaction ID is required")
            payload = json.loads(_canonical(dict(row)))
            if record_id is None:
                record_id = "record-" + _digest(
                    {"table": self.table, "transaction_id": transaction_id, "row": payload}
                )
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("record ID is required")
            if record_id in self.record_ids:
                raise ValueError(f"duplicate record ID in active shard: {record_id}")
            envelope = {
                "record_id": record_id,
                "transaction_id": transaction_id,
                "row": payload,
            }
            raw = _canonical(envelope) + b"\n"
            if SECRET_PATTERN.search(raw.decode("utf-8")):
                raise ValueError("credential-shaped content rejected")
            if self.active_path.stat().st_size and (
                self.active_path.stat().st_size + len(raw) > self.maximum_shard_bytes
            ):
                raise OverflowError("active shard reached its configured size; seal it")
            self.stream.write(raw)
            os.fsync(self.stream.fileno())
            self.line_number += 1
            self.record_ids.add(record_id)
            return RecordReference(
                table=self.table,
                writer_id=self.writer_id,
                shard_sequence=self.shard_sequence,
                line_number=self.line_number,
                record_id=record_id,
                transaction_id=transaction_id,
            )

    def seal(self) -> dict[str, Any]:
        with self._lock:
            if self.stream.closed:
                raise RuntimeError("writer is already closed")
            self.stream.close()
            if self.line_number == 0:
                raise ValueError("refusing to seal an empty shard")
            os.replace(self.active_path, self.sealed_path)
            _fsync_directory(self.directory)
            manifest = _sealed_manifest(
                self.root,
                table=self.table,
                writer_id=self.writer_id,
                shard_sequence=self.shard_sequence,
                path=self.sealed_path,
            )
            _atomic_json(self.manifest_path, manifest)
            return manifest

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()

    def __enter__(self) -> JsonlShardWriter:  # noqa: PYI034
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _sealed_manifest(
    root: Path,
    *,
    table: str,
    writer_id: str,
    shard_sequence: int,
    path: Path,
) -> dict[str, Any]:
    raw = path.read_bytes()
    record_ids: list[str] = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"sealed shard has invalid JSON at line {number}") from error
        record_id = value.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise ValueError("sealed shard has invalid or duplicate record IDs")
        record_ids.append(record_id)
    return {
        "format_version": "geodml-jsonl-shard-v1",
        "table": table,
        "writer_id": writer_id,
        "shard_sequence": shard_sequence,
        "path": str(path.relative_to(root)),
        "rows": len(record_ids),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_ids_sha256": _digest(sorted(record_ids)),
    }


def finalize_orphaned_shard(
    root: Path, *, table: str, writer_id: str, shard_sequence: int
) -> dict[str, Any] | None:
    """Finish a checksum manifest after a crash between rename and manifest commit."""

    directory = root / "data" / table
    stem = f"part-{writer_id}-{shard_sequence:06d}"
    shard = directory / f"{stem}.jsonl"
    manifest_path = directory / f"{stem}.manifest.json"
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if value != _sealed_manifest(
            root,
            table=table,
            writer_id=writer_id,
            shard_sequence=shard_sequence,
            path=shard,
        ):
            raise ValueError("sealed shard manifest does not match its bytes")
        return value
    if not shard.exists():
        return None
    manifest = _sealed_manifest(
        root,
        table=table,
        writer_id=writer_id,
        shard_sequence=shard_sequence,
        path=shard,
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def recover_inprogress_writer(
    root: Path, *, writer_id: str
) -> list[dict[str, Any]]:
    """Seal valid shards after the writer allocation is confirmed terminal.

    The caller is responsible for proving that no process can still append to
    this writer ID. Invalid or partial tails fail closed and remain untouched.
    """

    if WRITER_PATTERN.fullmatch(writer_id) is None:
        raise ValueError("invalid writer ID")
    if not (root / "contract.json").is_file():
        raise ValueError("dataset root is not initialized")
    recovery_root = root / "control" / "writer-recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    recovered: list[dict[str, Any]] = []
    lock_path = recovery_root / f"{writer_id}.lock"
    with lock_path.open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data_root = root / "data"
        for directory in sorted(path for path in data_root.iterdir() if path.is_dir()):
            pattern = re.compile(
                rf"part-{re.escape(writer_id)}-([0-9]{{6}})\.jsonl\.inprogress"
            )
            for active in sorted(directory.glob(f"part-{writer_id}-*.jsonl.inprogress")):
                match = pattern.fullmatch(active.name)
                if match is None:
                    raise ValueError(f"invalid active shard name: {active}")
                raw = active.read_bytes()
                if not raw:
                    active.unlink()
                    _fsync_directory(directory)
                    continue
                if not raw.endswith(b"\n"):
                    raise ValueError(f"incomplete active shard tail: {active}")
                record_ids: set[str] = set()
                for number, line in enumerate(raw.splitlines(), 1):
                    try:
                        envelope = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid active shard JSON at {active}:{number}"
                        ) from error
                    record_id = envelope.get("record_id")
                    if not isinstance(record_id, str) or record_id in record_ids:
                        raise ValueError(f"invalid active shard record at {active}:{number}")
                    record_ids.add(record_id)
                sequence = int(match.group(1))
                sealed = directory / active.name.removesuffix(".inprogress")
                manifest_path = directory / (
                    active.name.removesuffix(".jsonl.inprogress") + ".manifest.json"
                )
                if sealed.exists() or manifest_path.exists():
                    raise FileExistsError(f"active shard conflicts with sealed files: {active}")
                os.replace(active, sealed)
                _fsync_directory(directory)
                manifest = _sealed_manifest(
                    root,
                    table=directory.name,
                    writer_id=writer_id,
                    shard_sequence=sequence,
                    path=sealed,
                )
                _atomic_json(manifest_path, manifest)
                recovered.append(manifest)
    return recovered


def verify_record_reference(root: Path, reference: Mapping[str, Any], *, verification=None) -> bool:
    """Verify that a ledger reference resolves to an immutable sealed record."""

    table = reference.get("table")
    writer_id = reference.get("writer_id")
    sequence = reference.get("shard_sequence")
    line_number = reference.get("line_number")
    record_id = reference.get("record_id")
    if (
        not isinstance(table, str)
        or TABLE_PATTERN.fullmatch(table) is None
        or not isinstance(writer_id, str)
        or WRITER_PATTERN.fullmatch(writer_id) is None
        or type(sequence) is not int
        or sequence < 0
        or type(line_number) is not int
        or line_number < 1
        or not isinstance(record_id, str)
        or not record_id
    ):
        return False
    directory = root / "data" / table
    stem = f"part-{writer_id}-{sequence:06d}"
    manifest_path = directory / f"{stem}.manifest.json"
    shard = directory / f"{stem}.jsonl"
    if not manifest_path.is_file() or not shard.is_file():
        return False
    if verification is not None:
        return verification.reference(reference, manifest_path, shard,
                                      lambda: verify_record_reference(root, reference))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = shard.read_bytes()
    if (
        manifest.get("path") != str(shard.relative_to(root))
        or manifest.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        return False
    lines = raw.splitlines()
    if len(lines) != manifest.get("rows") or line_number > len(lines):
        return False
    envelope = json.loads(lines[line_number - 1])
    return (
        envelope.get("record_id") == record_id
        and envelope.get("transaction_id") == reference.get("transaction_id")
    )


def iter_sealed_rows(
    root: Path, table: str, *, required: bool = False, verification=None
) -> Iterator[dict[str, Any]]:
    """Yield verified row payloads from immutable shards in manifest order."""

    if TABLE_PATTERN.fullmatch(table) is None:
        raise ValueError("invalid table name")
    directory = root / "data" / table
    manifests = sorted(directory.glob("*.manifest.json")) if directory.is_dir() else []
    if required and not manifests:
        raise ValueError(f"dataset has no sealed {table} table")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != "geodml-jsonl-shard-v1":
            raise ValueError(f"unsupported shard manifest: {manifest_path}")
        relative = manifest.get("path")
        if not isinstance(relative, str):
            raise TypeError(f"shard manifest lacks its path: {manifest_path}")
        shard = root / relative
        if not shard.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"shard path escapes dataset root: {manifest_path}")
        if verification is not None:
            before = verification.signature(shard)
            if not verification.file(shard, manifest):
                raise ValueError(f"shard checksum mismatch: {shard}")
        raw = shard.read_bytes()
        if verification is not None and before != verification.signature(shard):
            raise ValueError(f"shard changed during verification: {shard}")
        if verification is None and hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
            raise ValueError(f"shard checksum mismatch: {shard}")
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) != manifest.get("rows"):
            raise ValueError(f"shard row count mismatch: {shard}")
        for number, line in enumerate(lines, 1):
            envelope = json.loads(line)
            row = envelope.get("row")
            if not isinstance(row, dict):
                raise TypeError(f"non-object dataset row in {shard}:{number}")
            yield row


class FinalDatasetWriter:
    """Lazy multi-table writer whose sealed files are directly publishable."""

    def __init__(
        self,
        root: Path,
        *,
        writer_id: str,
        maximum_shard_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if not (root / "contract.json").is_file():
            raise ValueError("dataset root is not initialized")
        if WRITER_PATTERN.fullmatch(writer_id) is None:
            raise ValueError("invalid writer ID")
        self.root = root
        self.writer_id = writer_id
        self.maximum_shard_bytes = maximum_shard_bytes
        self._writers: dict[str, JsonlShardWriter] = {}
        self._manifests: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._seal_interval = float(os.environ.get("GEODML_DATASET_SEAL_INTERVAL_SECONDS", "0"))
        if not 0 <= self._seal_interval < float("inf"):
            raise ValueError("dataset seal interval must be finite and non-negative")
        self._last_seal = time.monotonic()

    def _sequence(self, table: str) -> int:
        directory = self.root / "data" / table
        directory.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(
            rf"part-{re.escape(self.writer_id)}-([0-9]{{6}})"
            r"(?:\.jsonl(?:\.inprogress)?|\.manifest\.json)$"
        )
        sequences = {
            int(match.group(1))
            for path in directory.iterdir()
            if (match := pattern.fullmatch(path.name)) is not None
        }
        if not sequences:
            return 0
        latest = max(sequences)
        active = directory / f"part-{self.writer_id}-{latest:06d}.jsonl.inprogress"
        if active.exists():
            return latest
        finalize_orphaned_shard(
            self.root,
            table=table,
            writer_id=self.writer_id,
            shard_sequence=latest,
        )
        return latest + 1

    def _writer(self, table: str) -> JsonlShardWriter:
        writer = self._writers.get(table)
        if writer is None:
            writer = JsonlShardWriter(
                self.root,
                table=table,
                writer_id=self.writer_id,
                shard_sequence=self._sequence(table),
                maximum_shard_bytes=self.maximum_shard_bytes,
            )
            self._writers[table] = writer
        return writer

    def append(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        transaction_id: str,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._seal_interval and time.monotonic() - self._last_seal >= self._seal_interval:
                self.seal()
            try:
                writer = self._writer(table)
                reference = writer.append(
                    row, transaction_id=transaction_id, record_id=record_id
                )
            except OverflowError:
                self._manifests.append(writer.seal())
                writer = JsonlShardWriter(
                    self.root,
                    table=table,
                    writer_id=self.writer_id,
                    shard_sequence=writer.shard_sequence + 1,
                    maximum_shard_bytes=self.maximum_shard_bytes,
                )
                self._writers[table] = writer
                reference = writer.append(
                    row, transaction_id=transaction_id, record_id=record_id
                )
            except OSError as error:
                record_storage_incident(
                    self.root,
                    error,
                    operation="append_final_dataset_record",
                    writer_id=self.writer_id,
                    table=table,
                )
                raise
            return asdict(reference)

    def seal(self) -> list[dict[str, Any]]:
        with self._lock:
            for table, writer in list(self._writers.items()):
                try:
                    if writer.line_number:
                        self._manifests.append(writer.seal())
                    else:
                        writer.close()
                        writer.active_path.unlink(missing_ok=True)
                except OSError as error:
                    record_storage_incident(
                        self.root,
                        error,
                        operation="seal_final_dataset_shard",
                        writer_id=self.writer_id,
                        table=table,
                    )
                    raise
                del self._writers[table]
            self._last_seal = time.monotonic()
            return list(self._manifests)

    def close(self) -> None:
        with self._lock:
            for writer in self._writers.values():
                writer.close()
            self._writers.clear()

    def audit_callback(self, event: Mapping[str, Any]) -> None:
        context = event.get("task_context")
        if not isinstance(context, Mapping):
            raise TypeError("transport event lacks task context")
        transaction_id = context.get("transaction_id")
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError("transport event lacks transaction ID")
        self.append("transport_attempts", event, transaction_id=transaction_id)

    def __enter__(self) -> FinalDatasetWriter:  # noqa: PYI034
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
