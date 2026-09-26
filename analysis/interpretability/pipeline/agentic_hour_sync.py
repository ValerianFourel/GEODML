"""Conflict-checked HF coordination and immutable, checksum-verified transfers.

Only the networked helper uses this module. Never synchronize live lock files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath

from .agentic_audit_progress import audit_progress, audit_stage
from .agentic_dataset import verify_record_reference
from .agentic_hours import canonical, digest, empty_registry, identifier, verify_plan
from .agentic_task_ledger import StripedTaskLedger
from .inference_claims import ClaimIdentity

REGISTRY_PATH = "coordination/hours.json"
UPLOAD_BATCH_BYTES = 64 * 1024 * 1024
UPLOAD_BATCH_FILES = 32
REGISTRY_CACHE_REVISIONS = 4


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


def commit_with_cooldown(operation):
    """Retry one rejected commit after the server's cooldown, without re-auditing."""
    try:
        return operation()
    except Exception as error:
        response = getattr(error, 'response', None)
        if getattr(response, 'status_code', None) != 429:
            raise
        value = response.headers.get('Retry-After')
        delay = 3660.0  # The Hub commit-limit message specifies about one hour.
        if value:
            try:
                delay = float(value)
            except ValueError:
                try:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    pass
        if not 0 <= delay <= 3660:
            raise  # Never retry earlier than a longer server-requested wait.
        deadline = time.monotonic() + max(delay, 1)
        print(f'HF_COMMIT_COOLDOWN seconds={max(delay, 1):.0f} retry=1/1; preserving current batch', flush=True)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            audit_progress(phase='hf_commit_cooldown', retry_in_seconds=round(remaining))
            time.sleep(min(30, max(0, remaining)))
        # Same payload and parent revision: concurrent changes must still fail CAS.
        return operation()


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
            result = commit_with_cooldown(lambda: self.api.create_commit(
                repo_id=self.repo_id, repo_type="dataset", revision="main",
                parent_commit=revision, commit_message=message,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=raw)
                            for name, raw in files.items()],
            ))
        except HfHubHTTPError as error:
            if getattr(error.response, "status_code", None) in {409, 412}:
                raise ConflictError("repository advanced") from error
            raise
        return result.oid


class Exchange:
    def __init__(self, store, journal: Path):
        self.store, self.journal = store, journal
        # A commit revision never changes content, so the registry bytes this
        # process read or committed at a revision are exact. Nothing is persisted.
        self._registry: dict[str, bytes | None] = {}
        # Plan and bundle IDs are content hashes; reuse of their verification is
        # limited to an explicit staging session (see reusing_verification).
        self._session = False
        self._plans: dict[str, dict] = {}
        self._downloaded: dict[tuple, tuple[bool, dict]] = {}

    @contextmanager
    def reusing_verification(self):
        """Verify each plan and bundle once while staging one finite wave."""
        outer, self._session = self._session, True
        try:
            yield self
        finally:
            self._session = outer
            if not outer:
                self._plans.clear()
                self._downloaded.clear()

    def _remember(self, revision, raw: bytes | None) -> None:
        if not isinstance(revision, str):
            return
        self._registry.pop(revision, None)
        self._registry[revision] = raw
        while len(self._registry) > REGISTRY_CACHE_REVISIONS:
            self._registry.pop(next(iter(self._registry)))

    def snapshot(self) -> tuple[str, dict]:
        """Registry at the current head; revisions this process read or wrote cost no download."""
        revision = self.store.head()
        if revision in self._registry:
            raw = self._registry[revision]
        else:
            raw = self.store.read(REGISTRY_PATH, revision)
            self._remember(revision, raw)
        return revision, empty_registry() if raw is None else json.loads(raw)

    def plan(self, plan_id: str) -> dict:
        """Load and verify a published plan; callers must treat it as read-only."""
        if plan_id in self._plans:
            return self._plans[plan_id]
        raw = self.store.read(f"coordination/plans/{plan_id}.json", self.store.head())
        if raw is None:
            raise ValueError("published plan is missing")
        value = json.loads(raw)
        if value.get("plan_id") != plan_id:
            raise ValueError("plan identity mismatch")
        verify_plan(value)
        if self._session:
            self._plans[plan_id] = value
        return value

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
            raw = canonical(updated)
            files = {**(extra or {}), REGISTRY_PATH: raw, remote: canonical(receipt)}
            try:
                committed = self.store.commit(revision, files, f"GEODML {operation_id}")
            except ConflictError:
                continue
            # The server accepted exactly these bytes on top of `revision`.
            self._remember(committed, raw)
            return receipt
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
                    parent = revision
                    revision = self.store.commit(revision, missing, "GEODML sealed transfer")
                except ConflictError:
                    continue
                # A compare-and-set commit without the registry leaves it unchanged.
                if REGISTRY_PATH not in missing and parent in self._registry:
                    self._remember(revision, self._registry[parent])
                break
            else:
                raise ConflictError("upload busy; resume without changing its files")
        return revision

    def upload(self, root: Path, names: list[str], *, outcomes: dict,
               metadata: dict) -> str:
        return self.upload_many([(root, names, outcomes, metadata)])[0][0]

    @audit_stage("bundle_upload")
    def upload_many(self, entries: list[tuple[Path, list[str], dict, dict]], *,
                    isolate_errors: bool = False) -> tuple[list[str | None], dict[int, str]]:
        """Publish several bundles: objects share bounded commits, manifests commit last.

        With ``isolate_errors`` a rejected entry (unsafe path, credential-shaped
        content) publishes no manifest and is reported by index; the others
        proceed. Objects it already flushed are unreferenced content, never a bundle.
        """
        from analysis.scripts.publish_agentic_dataset import SECRET_BYTES
        total = sum(len(set(names)) for _, names, _, _ in entries)
        audit_progress(phase="verify_and_publish_files", files_total=total, files_finished=0, bytes_finished=0)
        progress = {"files": 0, "bytes": 0}
        pending: dict[str, bytes] = {}

        def flush():
            if pending:
                self.immutable(pending)
                pending.clear()
                audit_progress(files_finished=progress["files"], bytes_finished=progress["bytes"])

        manifests: dict[int, dict] = {}
        errors: dict[int, str] = {}
        for index, (root, names, outcomes, metadata) in enumerate(entries):
            added: list[str] = []
            try:
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
                    object_name = f"exchange/objects/{sha}"
                    if pending and (len(pending) >= UPLOAD_BATCH_FILES
                                    or sum(map(len, pending.values())) + len(raw) > UPLOAD_BATCH_BYTES):
                        flush()
                    # Bound buffered payloads; a single oversized file travels alone.
                    # Content-addressed objects already committed are reused on retry.
                    if object_name not in pending:
                        pending[object_name] = raw
                        added.append(object_name)
                    inventory[name] = {"sha256": sha, "bytes": len(raw)}
                    progress["files"] += 1
                    progress["bytes"] += len(raw)
                    if sum(map(len, pending.values())) >= UPLOAD_BATCH_BYTES:
                        flush()
                manifest = {"format_version": "geodml-hour-bundle-v1", "files": inventory,
                            "outcomes": outcomes, "metadata": metadata}
                if SECRET_BYTES.search(canonical(manifest)):
                    raise ValueError("credential-shaped metadata rejected")
            except ValueError as error:
                if not isolate_errors:
                    raise
                for object_name in added:
                    pending.pop(object_name, None)
                errors[index] = str(error)
                continue
            manifests[index] = manifest
        flush()
        bundles: list[str | None] = [None] * len(entries)
        for index, manifest in manifests.items():
            bundles[index] = "bundle-" + digest(manifest)
        # Data first, marker last: no reader can mistake a partial upload for a bundle.
        audit_progress(phase="publish_manifest")
        if manifests:
            self.immutable({f"exchange/bundles/{bundles[index]}.json": canonical(manifest)
                            for index, manifest in manifests.items()})
        return bundles, errors

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
        root = root.resolve()
        key = (bundle_id, str(root), stripes)
        prior = self._downloaded.get(key)
        # Inside a staging session a bundle verified (and imported) into this root
        # needs no second pass; forced remote checks always rerun.
        if prior and not verify_remote and (prior[0] or not import_outcomes):
            return prior[1]
        revision = revision or self.store.head()
        value = self.manifest(bundle_id, revision)
        from .agentic_verification_cache import VerificationCache
        with VerificationCache(root, force=verify_remote) as verification:
            for name, expected in value["files"].items():
                relative_path(name)
                target = root / name
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    raise ValueError("download path escapes dataset")
                if target.exists() and not verify_remote:
                    if not verification.file(target, expected):
                        raise ValueError(f"missing, corrupt, or conflicting artifact: {name}")
                    continue
                raw = self.store.read(f"exchange/objects/{expected['sha256']}", revision)
                if raw is None or len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                    raise ValueError(f"missing, corrupt, or conflicting artifact: {name}")
                if not target.exists():
                    atomic(target, raw)
                if not verification.file(target, expected):
                    raise ValueError(f"conflicting local artifact: {name}")
            for fp, event in value["outcomes"].items():
                refs = event.get("record_references", [])
                if event["state"] == "completed" and not refs:
                    raise ValueError(f"completion lacks record references: {fp}")
                if not all(verify_record_reference(root, ref, verification=verification) for ref in refs):
                    raise ValueError(f"outcome references failed verification: {fp}")
                if event["state"] not in {"completed", "terminal_failed"}:
                    raise ValueError("only verified terminal outcomes can be imported")
                if event["state"] == "terminal_failed" and (not event.get("owner_id") or not event.get("generation")):
                    raise ValueError("terminal failure lacks its durable producer event")
        if import_outcomes:
            import_events(root, value["outcomes"], stripes=stripes)
        if self._session:
            self._downloaded[key] = (import_outcomes or bool(prior and prior[0]), value)
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
                     writer_id: str | None = None, latest: dict | None = None) -> tuple[list[str], dict]:
    """Only completed transactions whose entire reference set is sealed.

    ``latest`` lets a wave sync share one ledger snapshot across its attempts.
    """
    if latest is None:
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
