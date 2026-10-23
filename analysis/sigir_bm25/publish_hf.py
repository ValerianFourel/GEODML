"""Publish a finished SIGIR BM25 run folder to the private Hugging Face dataset (a finite login-node helper).

  python publish_hf.py --folder DIR --path-in-repo sigir-bm25/NAME [--repo OWNER/DATASET]

1. Writes DIR/MANIFEST.json: bytes, sha256 and git blob id of every file in DIR.
2. Asks for a Hugging Face write token in the terminal (hidden, kept in this process only, never written).
3. Refuses unless the dataset exists and is private, and refuses to replace a file already on the Hub whose content
   differs (nothing is overwritten).
4. Uploads DIR in one commit (content already on the Hub is not sent again).
5. Checks every file on the Hub against the manifest: size, and the LFS sha256 or the git blob id. Exit 0 only when
   all match; 1 when the check fails after the upload; 2 when refused before uploading.
Rerun after an interruption: the manifest is rebuilt from the same files, and a fully verified folder is not uploaded
again. Standalone (standard library + huggingface_hub), so it can be copied next to the run and run from there.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = "ValerianFourel/geodml-experiment-v2-paper-private"
MANIFEST = "MANIFEST.json"
CHUNK = 1 << 20


def file_digests(path: Path) -> dict:
    """Bytes, sha256 (the Hub's LFS id) and git blob id (the Hub's id for files stored in git) of one file."""
    size = path.stat().st_size
    sha256, blob = hashlib.sha256(), hashlib.sha1(b"blob %d\0" % size)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return {"bytes": size, "sha256": sha256.hexdigest(), "git_blob": blob.hexdigest()}


def build_manifest(folder: Path) -> dict:
    files = {p.relative_to(folder).as_posix(): file_digests(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.relative_to(folder).as_posix() != MANIFEST}
    return {"run": str(folder), "files": files}


def remote_files(api, repo: str, prefix: str) -> dict:
    """The Hub's files under prefix, by path relative to it: bytes, git blob id and LFS sha256 (None if in git)."""
    try:
        entries = list(api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type="dataset"))
    except Exception as e:  # the Hub answers 404 for a folder that does not exist yet
        if type(e).__name__ == "EntryNotFoundError" or getattr(getattr(e, "response", None), "status_code", 0) == 404:
            return {}
        raise
    out = {}
    for e in entries:
        if not hasattr(e, "blob_id"):  # a folder
            continue
        lfs = getattr(e, "lfs", None)
        sha = (getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)) if lfs else None
        out[e.path[len(prefix) + 1:]] = {"bytes": e.size, "git_blob": e.blob_id, "lfs_sha256": sha}
    return out


def compare(expected: dict, remote: dict) -> dict:
    """Per expected file: 'match', 'missing' or 'differs' on the Hub."""
    out = {}
    for name, d in expected.items():
        r = remote.get(name)
        if r is None:
            out[name] = "missing"
        elif r["bytes"] == d["bytes"] and (r["lfs_sha256"] == d["sha256"] if r["lfs_sha256"]
                                           else r["git_blob"] == d["git_blob"]):
            out[name] = "match"
        else:
            out[name] = "differs"
    return out


def main(argv=None, api=None, ask=getpass.getpass) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True)
    p.add_argument("--path-in-repo", required=True)
    p.add_argument("--repo", default=REPO)
    a = p.parse_args(argv)
    folder, prefix = Path(a.folder).resolve(strict=True), a.path_in_repo.strip("/")
    if not prefix:
        print("--path-in-repo must name a folder in the dataset", file=sys.stderr)
        return 2
    manifest = build_manifest(folder)
    if not manifest["files"]:
        print(f"no files in {folder}", file=sys.stderr)
        return 2
    tmp = folder / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, folder / MANIFEST)
    expected = {**manifest["files"], MANIFEST: file_digests(folder / MANIFEST)}
    total = sum(d["bytes"] for d in expected.values())
    print(json.dumps({"files": len(expected), "bytes": total, "repo": a.repo, "path_in_repo": prefix}), flush=True)
    if api is None:
        os.environ.pop("HF_HUB_OFFLINE", None)  # read when huggingface_hub is imported
        from huggingface_hub import HfApi
        token = ask("Hugging Face WRITE token for the private dataset (hidden): ").strip()
        if not token:
            print("a write token is required", file=sys.stderr)
            return 2
        api = HfApi(token=token)
    if not api.repo_info(a.repo, repo_type="dataset").private:
        print(f"{a.repo} is not private: nothing uploaded", file=sys.stderr)
        return 2
    before = compare(expected, remote_files(api, a.repo, prefix))
    differs = sorted(k for k, v in before.items() if v == "differs")
    if differs:
        print(f"refused: {len(differs)} files under {prefix} already differ on the Hub, e.g. {differs[:3]}",
              file=sys.stderr)
        return 2
    if all(v == "match" for v in before.values()):
        print(json.dumps({"verified": len(before), "uploaded": False, "note": "already on the Hub"}))
        return 0
    commit = api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(folder), path_in_repo=prefix,
                               commit_message=f"{prefix}: {len(expected)} files, {total} bytes")
    after = compare(expected, remote_files(api, a.repo, prefix))
    problems = {k: v for k, v in after.items() if v != "match"}
    print(json.dumps({"commit": getattr(commit, "oid", None), "verified": len(after) - len(problems),
                      "problems": problems}))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
