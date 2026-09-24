#!/usr/bin/env python3
"""Queue and publish sealed Experiment V2 files without rebuilding the dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_BYTES = re.compile(
    rb"hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9_.-]{20,}"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic(path: Path, value: object) -> None:
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


def build_manifest(root: Path) -> dict[str, Any]:
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("format_version") != "geodml-incremental-dataset-v1":
        raise ValueError("unsupported incremental dataset contract")
    allowed = contract.get("publication_allowlist")
    excluded = set(contract.get("publication_exclusions", []))
    if not isinstance(allowed, list) or "contract.json" not in allowed:
        raise ValueError("dataset contract lacks a publication allowlist")
    paths: list[Path] = [contract_path]
    for name in allowed:
        if name == "contract.json":
            continue
        parent = root / name
        if parent.exists():
            paths.extend(path for path in parent.rglob("*") if path.is_file())
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        if relative.parts[0] in excluded:
            raise ValueError(f"excluded directory entered publication set: {relative}")
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"publication file escapes dataset root: {relative}")
        if path.name.endswith(".inprogress") or path.name.endswith(".tmp"):
            continue
        if relative.parent == Path("manifests") and relative.name.startswith("publication-"):
            continue
        raw = path.read_bytes()
        if SECRET_BYTES.search(raw):
            raise ValueError(f"credential-shaped content in publication file: {relative}")
        files[str(relative)] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    for relative in list(files):
        if not relative.startswith("data/") or not relative.endswith(".jsonl"):
            continue
        manifest_name = relative[:-6] + ".manifest.json"
        if manifest_name not in files:
            raise ValueError(f"sealed data shard lacks its checksum manifest: {relative}")
        value = json.loads((root / manifest_name).read_text(encoding="utf-8"))
        if value.get("path") != relative or value.get("sha256") != files[relative]["sha256"]:
            raise ValueError(f"data shard manifest mismatch: {relative}")
    identity = {
        "format_version": "geodml-agentic-publication-manifest-v1",
        "dataset_contract_sha256": _sha(contract_path),
        "files": files,
    }
    return {
        **identity,
        "snapshot_id": "snapshot-" + hashlib.sha256(_canonical(identity)).hexdigest()[:24],
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }


def enqueue(root: Path, *, repo_id: str) -> tuple[Path, dict[str, Any]]:
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id) is None:
        raise ValueError("repo ID must include a valid namespace and dataset name")
    manifest = build_manifest(root)
    queue = {
        "format_version": "geodml-agentic-publication-queue-v1",
        "snapshot_id": manifest["snapshot_id"],
        "repo_id": repo_id,
        "repo_type": "dataset",
        "required_visibility": "private",
        "status": "queued",
        "manifest": manifest,
    }
    path = root / "publication" / f"queue-{manifest['snapshot_id']}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != queue:
            raise ValueError("publication queue identity conflicts with existing request")
        return path, queue
    _atomic(path, queue)
    return path, queue


def _namespace_authorized(identity: dict[str, Any], namespace: str) -> bool:
    names = {identity.get("name")}
    organizations = identity.get("orgs", [])
    if isinstance(organizations, list):
        names.update(
            row.get("name")
            for row in organizations
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        )
    return namespace.casefold() in {
        value.casefold() for value in names if isinstance(value, str)
    }


def publish(root: Path, *, queue_path: Path) -> dict[str, Any]:
    """Run on a networked login/transfer host; inference state is never touched."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required on the publication host")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("status") != "queued":
        raise ValueError("publication queue is not in queued state")
    manifest = build_manifest(root)
    if manifest != queue.get("manifest"):
        raise ValueError("dataset publication set changed after it was queued")
    repo_id = queue["repo_id"]
    namespace = repo_id.split("/", 1)[0]
    receipt_path = root / "publication" / f"receipt-{manifest['snapshot_id']}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "verified":
            raise ValueError("existing publication receipt is not verified")
        return receipt
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    identity = api.whoami()
    if not _namespace_authorized(identity, namespace):
        raise PermissionError(f"authenticated account cannot publish under {namespace}")
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token
    )
    if api.repo_info(repo_id=repo_id, repo_type="dataset", token=token).private is not True:
        raise ValueError("destination dataset is not private")
    prefix = "snapshots/" + manifest["snapshot_id"]
    manifest_path = root / "manifests" / f"publication-{manifest['snapshot_id']}.json"
    marker_relative = str(manifest_path.relative_to(root))
    marker = prefix + "/" + marker_relative
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("saved publication manifest conflicts with queued snapshot")
    else:
        _atomic(manifest_path, manifest)
    upload_files = [*manifest["files"], marker_relative]
    if marker in remote:
        downloaded = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=marker,
            token=token,
        ))
        if downloaded.read_bytes() != manifest_path.read_bytes():
            raise ValueError("remote snapshot marker conflicts with local manifest")
        commit = "already-present"
    else:
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo=prefix,
            allow_patterns=upload_files,
            commit_message=f"Add sealed GEODML snapshot {manifest['snapshot_id']}",
            token=token,
        )
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    expected = {prefix + "/" + name for name in upload_files}
    if not expected <= remote:
        raise ValueError("remote snapshot inventory is incomplete")
    receipt = {
        "format_version": "geodml-agentic-publication-receipt-v1",
        "status": "verified",
        "snapshot_id": manifest["snapshot_id"],
        "repo_id": repo_id,
        "path_in_repo": prefix,
        "commit": str(commit),
        "verified_at": _now(),
    }
    _atomic(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--enqueue", action="store_true")
    action.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_manifest(args.dataset_root)
    if not args.enqueue and not args.publish:
        print(json.dumps({"dry_run": True, "manifest": manifest}, indent=2, sort_keys=True))
        return
    queue_path, queue = enqueue(args.dataset_root, repo_id=args.repo_id)
    if args.enqueue:
        print(json.dumps({"queue_path": str(queue_path), "queue": queue}, indent=2, sort_keys=True))
        return
    receipt = publish(args.dataset_root, queue_path=queue_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
