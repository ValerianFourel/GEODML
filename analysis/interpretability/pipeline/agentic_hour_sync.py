"""Conflict-checked HF coordination and immutable, checksum-verified transfers.

Only the networked helper uses this module. Never synchronize live lock files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .agentic_dataset import verify_record_reference
from .agentic_hours import canonical, digest, empty_registry, identifier
from .agentic_task_ledger import StripedTaskLedger
from .inference_claims import ClaimIdentity

REGISTRY_PATH = "coordination/hours.json"


def atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def relative_path(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts) or "\\" in name:
        raise ValueError("unsafe bundle path")
    if path.parts[0] not in {"README.md", "contract.json", "schemas", "data", "artifacts", "manifests", "plans", "reports"}:
        raise ValueError("bundle path is outside the dataset publication allowlist")
    if name.endswith((".inprogress", ".tmp")):
        raise ValueError("active files cannot be transferred")
    return name


class ConflictError(RuntimeError):
    pass


class HubStore:
    """Small adapter; tests exercise the same protocol with isolated stores."""

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = HfApi()  # HF_TOKEN or the host's saved login; never persisted.
        info = self.api.repo_info(repo_id, repo_type="dataset")
        if info.private is not True:
            raise ValueError("shared-hour repository must already exist and be private")

    def head(self) -> str:
        return self.api.repo_info(self.repo_id, repo_type="dataset").sha

    def read(self, name: str, revision: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            path = hf_hub_download(self.repo_id, name, repo_type="dataset", revision=revision)
        except EntryNotFoundError:
            return None
        return Path(path).read_bytes()

    def commit(self, revision: str, files: dict[str, bytes], message: str) -> str:
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import HfHubHTTPError
        try:
            result = self.api.create_commit(
                repo_id=self.repo_id, repo_type="dataset", revision="main",
                parent_commit=revision, commit_message=message,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw)
                            for name, raw in files.items()],
            )
        except HfHubHTTPError as error:
            if getattr(error.response, "status_code", None) in {409, 412}:
                raise ConflictError("repository advanced") from error
            raise
        return result.oid


class Exchange:
    def __init__(self, store, journal: Path):
        self.store, self.journal = store, journal

    def snapshot(self) -> tuple[str, dict]:
        revision = self.store.head()
        raw = self.store.read(REGISTRY_PATH, revision)
        return revision, empty_registry() if raw is None else json.loads(raw)

    def transact(self, operation_id: str, payload: dict,
                 change: Callable[[dict], dict], extra: dict[str, bytes] | None = None) -> dict:
        """Journal intent first; a lost response is resolved by the operation ID."""
        identifier(operation_id)
        intent = {"operation_id": operation_id, "payload": payload}
        local = self.journal / f"{operation_id}.json"
        if local.exists() and json.loads(local.read_bytes()) != intent:
            raise ValueError("operation ID already has a different intent")
        atomic(local, canonical(intent))
        remote = f"coordination/operations/{operation_id}.json"
        for _ in range(8):
            revision, state = self.snapshot()
            prior = self.store.read(remote, revision)
            if prior is not None:
                receipt = json.loads(prior)
                if receipt["intent"] != intent:
                    raise ValueError("remote operation ID conflicts")
                return receipt
            updated = change(state)
            receipt = {"intent": intent, "registry_sha256": digest(updated)}
            files = {**(extra or {}), REGISTRY_PATH: canonical(updated), remote: canonical(receipt)}
            try:
                self.store.commit(revision, files, f"GEODML {operation_id}")
                return receipt
            except ConflictError:
                continue
        raise ConflictError("coordination busy; retry the same operation ID")

    def immutable(self, files: dict[str, bytes]) -> str:
        """Bounded commits, safe to resume after partial upload or lost response."""
        revision = self.store.head()
        for offset in range(0, len(files), 32):
            batch = dict(list(files.items())[offset:offset + 32])
            for _ in range(8):
                revision = self.store.head()
                missing = {}
                for name, raw in batch.items():
                    old = self.store.read(name, revision)
                    if old is None:
                        missing[name] = raw
                    elif old != raw:
                        raise ValueError(f"immutable remote file conflicts: {name}")
                if not missing:
                    break
                try:
                    revision = self.store.commit(revision, missing, "GEODML sealed transfer")
                    break
                except ConflictError:
                    continue
            else:
                raise ConflictError("upload busy; resume without changing its files")
        return revision

    def upload(self, root: Path, names: list[str], *, outcomes: dict,
               metadata: dict) -> str:
        from analysis.scripts.publish_agentic_dataset import SECRET_BYTES
        inventory = {}
        for name in sorted(set(names)):
            relative_path(name)
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("transfer path escapes the dataset")
            raw = path.read_bytes()
            if SECRET_BYTES.search(raw):
                raise ValueError("credential-shaped content rejected")
            sha = hashlib.sha256(raw).hexdigest()
            # Keep at most one sealed payload in memory; the manifest is last.
            self.immutable({f"exchange/objects/{sha}": raw})
            inventory[name] = {"sha256": sha, "bytes": len(raw)}
        manifest = {"format_version": "geodml-hour-bundle-v1", "files": inventory,
                    "outcomes": outcomes, "metadata": metadata}
        if SECRET_BYTES.search(canonical(manifest)):
            raise ValueError("credential-shaped metadata rejected")
        bundle_id = "bundle-" + digest(manifest)
        # Data first, marker last: no reader can mistake a partial upload for a bundle.
        self.immutable({f"exchange/bundles/{bundle_id}.json": canonical(manifest)})
        return bundle_id

    def manifest(self, bundle_id: str, revision: str | None = None) -> dict:
        identifier(bundle_id)
        raw = self.store.read(f"exchange/bundles/{bundle_id}.json", revision or self.store.head())
        if raw is None:
            raise ValueError("bundle is not published")
        value = json.loads(raw)
        if "bundle-" + digest(value) != bundle_id or value["format_version"] != "geodml-hour-bundle-v1":
            raise ValueError("bundle manifest checksum mismatch")
        return value

    def download(self, bundle_id: str, root: Path, *, stripes: int = 256,
                 import_outcomes: bool = True, verify_remote: bool = False,
                 revision: str | None = None) -> dict:
        revision = revision or self.store.head()
        value = self.manifest(bundle_id, revision)
        for name, expected in value["files"].items():
            relative_path(name)
            target = root / name
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("download path escapes dataset")
            raw = target.read_bytes() if target.exists() and not verify_remote else self.store.read(
                f"exchange/objects/{expected['sha256']}", revision)
            if raw is None or len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                raise ValueError(f"missing, corrupt, or conflicting artifact: {name}")
            if target.exists() and target.read_bytes() != raw:
                raise ValueError(f"conflicting local artifact: {name}")
            if not target.exists():
                atomic(target, raw)
        for fp, event in value["outcomes"].items():
            refs = event.get("record_references", [])
            if event["state"] == "completed" and not refs:
                raise ValueError(f"completion lacks record references: {fp}")
            if not all(verify_record_reference(root, ref) for ref in refs):
                raise ValueError(f"outcome references failed verification: {fp}")
            if event["state"] not in {"completed", "terminal_failed"}:
                raise ValueError("only verified terminal outcomes can be imported")
            if event["state"] == "terminal_failed" and (not event.get("owner_id") or not event.get("generation")):
                raise ValueError("terminal failure lacks its durable producer event")
        if import_outcomes:
            import_events(root, value["outcomes"], stripes=stripes)
        return value


def import_events(root: Path, outcomes: dict, *, stripes: int) -> None:
    from .agentic_task_ledger import identity_fingerprint
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes)
    for fp, event in outcomes.items():
        identity = ClaimIdentity(**event["identity"])
        if fp != identity_fingerprint(identity):
            raise ValueError("imported task fingerprint mismatch")
        prior = ledger.inspect(identity)
        if prior and prior["state"] in {"completed", "terminal_failed"}:
            if prior["state"] != event["state"] or prior["record_references"] != event["record_references"]:
                raise ValueError("conflicting terminal result; reconciliation required")
            continue
        claim = ledger.claim(identity, owner_id="import-" + digest(event)[:24])
        if claim.status != "owned":
            raise ValueError("cannot import over an active local task")
        ledger.transition(claim.claim, state=event["state"],
                          record_references=event["record_references"],
                          detail={"original_producer": event.get("owner_id"), "imported": True})


def checkpoint_files(root: Path, tasks: dict, *, stripes: int = 256,
                     writer_id: str | None = None) -> tuple[list[str], dict]:
    """Only completed transactions whose entire reference set is sealed."""
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
    names, outcomes = set(), {}
    for fp, task in tasks.items():
        event = latest.get(fp, {})
        if writer_id is not None and event.get("owner_id") != writer_id:
            continue
        if event.get("state") not in {"completed", "terminal_failed"}:
            continue
        refs = event.get("record_references", [])
        if (event["state"] == "completed" and not refs) or not all(verify_record_reference(root, ref) for ref in refs):
            continue
        outcomes[fp] = {**event, "identity": task["claim_identity"]}
        for ref in refs:
            stem = f"data/{ref['table']}/part-{ref['writer_id']}-{ref['shard_sequence']:06d}"
            names.update((stem + ".jsonl", stem + ".manifest.json"))
    # Include transport attempts, failed attempts and diagnostics not referenced by success rows.
    if writer_id is not None:
        identifier(writer_id)
        for path in (root / "data").glob(f"*/part-{writer_id}-*.manifest.json"):
            manifest = json.loads(path.read_bytes())
            names.update((str(path.relative_to(root)), manifest["path"]))
    return sorted(names), outcomes
