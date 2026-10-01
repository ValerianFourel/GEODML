#!/usr/bin/env python3
"""Read-only Hub verification of the recorded JUPITER export scopes.

Writes only a new report and small metadata downloads. Never uploads, deletes,
releases ownership, submits work, or claims a full byte restore of activations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.archive_jupiter_to_hub import lfs_sha
from analysis.scripts.archive_geoaxis_prompts_26k import sha256_file, git_blob_sha1


def compare(api, repo, revision, expected):
    """Check fixed-revision metadata against saved bytes and content hashes."""
    infos = {}
    paths = list(expected)
    for start in range(0, len(paths), 100):
        infos.update((i.path, i) for i in api.get_paths_info(
            repo, paths[start:start + 100], repo_type="dataset", revision=revision))
    missing, mismatch, unverified = [], [], []
    for name, entry in expected.items():
        info = infos.get(name)
        if info is None:
            missing.append(name)
        elif info.size != entry["bytes"]:
            mismatch.append(name)
        elif lfs_sha(info):
            if lfs_sha(info) != entry["sha256"]:
                mismatch.append(name)
        elif entry.get("local") and Path(entry["local"]).is_file() and info.size <= 16 * 1024**2:
            local = Path(entry["local"])
            if sha256_file(local) != entry["sha256"] or git_blob_sha1(local) != getattr(info, "blob_id", None):
                mismatch.append(name)
        else:
            unverified.append(name)
    return {"expected_files": len(expected), "missing": missing, "mismatch": mismatch,
            "hash_unverified": unverified,
            "status": "pass" if expected and not (missing or mismatch or unverified) else "attention"}


def geoaxis(api, base, repo, staging_name):
    staging = base / "audits/leave-jupiter" / staging_name
    manifest = staging / "MANIFEST.tsv"
    expected = {}
    with manifest.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name = row["path"]
            if name in expected:
                raise ValueError("duplicate path in saved manifest")
            expected[name] = {"bytes": int(row["bytes"]), "sha256": row["sha256"], "local": str(staging / name)}
    if not expected:
        raise ValueError("empty export manifest cannot establish completion")
    for name in ("MANIFEST.tsv", "README.md"):
        path = staging / name
        expected[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "local": str(path)}
    info = api.repo_info(repo, repo_type="dataset")
    if info.private is not True:
        raise ValueError("expected private GeoAxis repository")
    return {"repo": repo, "revision": info.sha, "scope": "saved manifest, not arbitrary later files",
            **compare(api, repo, info.sha, expected)}


def general_archive(api, base, cache):
    from huggingface_hub import hf_hub_download
    work = base / "audits/leave-jupiter/hf-archive"
    plan_path = work / "plan/summary.json"
    plan = json.loads(plan_path.read_text())
    repo = "ValerianFourel/geodml-jupiter-archive-private"
    info = api.repo_info(repo, repo_type="dataset")
    if info.private is not True:
        raise ValueError("expected private general archive")
    expected, pending, mismatched_manifests, planned = {}, [], [], 0
    for label, root in plan["roots"].items():
        for unit in root["units"]:
            planned += 1
            prefix = label + "/" + unit["unit"]
            marker = work / "done" / label / (unit["unit"] + ".json")
            if not marker.is_file():
                pending.append(prefix)
                continue
            saved = json.loads(marker.read_text())
            paths = [prefix + "/manifest.json", prefix + "/members.txt.gz"]
            meta = {r.path: r for r in api.get_paths_info(repo, paths, repo_type="dataset", revision=info.sha)}
            if not all(p in meta for p in paths) or meta[paths[0]].size > 16 * 1024**2:
                mismatched_manifests.append(prefix)
                continue
            remote = json.loads(Path(hf_hub_download(repo, paths[0], repo_type="dataset",
                               revision=info.sha, cache_dir=cache)).read_text())
            if remote != {k: v for k, v in saved.items() if k != "verified_at"}:
                mismatched_manifests.append(prefix)
            for part in saved["parts"]:
                expected[prefix + "/" + part["name"]] = part
    index_paths = {"index/plan-summary.json": plan_path}
    index_paths.update({f"index/{label}-files.tsv.gz": work / "plan" / label / "files.tsv.gz"
                        for label in plan["roots"]})
    for name, path in index_paths.items():
        expected[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "local": str(path)}
    result = compare(api, repo, info.sha, expected)
    complete = json.loads((work / "COMPLETE.json").read_text()) if (work / "COMPLETE.json").is_file() else {}
    if not planned or pending or mismatched_manifests or complete.get("units") != planned:
        result["status"] = "attention"
    return {"repo": repo, "revision": info.sha, **result, "planned_units": planned,
            "pending_units": pending, "manifest_mismatches": mismatched_manifests,
            "completion_receipt": complete, "scope": "saved filtered plan; not all JUPITER files",
            "exclusions": {label: {k: root.get(k) for k in ("excludes", "exclude_globs", "unreadable", "filtered")}
                           for label, root in plan["roots"].items()}}


def activations(api, root):
    if not root.is_dir():
        raise ValueError("activation source folder absent; cannot prove full upload scope")
    local, symlinks = {}, []
    def fail(error):
        raise error
    for folder, dirs, files in os.walk(root, onerror=fail):
        for name in dirs:
            path = Path(folder) / name
            if path.is_symlink():
                symlinks.append(str(path.relative_to(root)))
        dirs[:] = [d for d in dirs if d != ".cache" and not (Path(folder) / d).is_symlink()]
        for name in files:
            path = Path(folder) / name
            if path.is_symlink():
                symlinks.append(str(path.relative_to(root)))
            else:
                local[str(path.relative_to(root))] = path.stat().st_size
    repo = "ValerianFourel/geodml-emnlp-2026-probing-activations"
    info = api.repo_info(repo, repo_type="dataset")
    remote = {r.path: r.size for r in api.list_repo_tree(repo, repo_type="dataset", revision=info.sha,
                                                       recursive=True) if hasattr(r, "size")}
    missing = sorted(set(local) - remote.keys())
    mismatch = [p for p, size in local.items() if p in remote and remote[p] != size]
    return {"repo": repo, "revision": info.sha, "private": info.private,
            "status": "inventory_match" if local and not (missing or mismatch or symlinks) else "attention",
            "local_files": len(local), "local_bytes": sum(local.values()), "remote_files": len(remote),
            "missing": missing, "size_mismatch": mismatch, "symlinks_not_verified": symlinks,
            "scope": "all local activations including t7_chunks_full; .cache excluded",
            "content_hash_verification": "not_performed", "full_restore": "not_performed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import HfApi
    api = HfApi()
    checks = {}
    operations = {
        "general_archive": lambda: general_archive(api, args.base, args.output / "metadata-cache"),
        "prompts_26k": lambda: geoaxis(api, args.base, "ValerianFourel/geoaxis-prompts-generation-26k", "geoaxis-26k-staging"),
        "readiness_axis": lambda: geoaxis(api, args.base, "ValerianFourel/geoaxis-readiness-axis", "geoaxis-axis-staging"),
        "activations": lambda: activations(api, args.activation_root),
    }
    for name, operation in operations.items():
        print("CHECKING " + name, flush=True)
        try:
            checks[name] = operation()
        except Exception as error:
            checks[name] = {"status": "unknown", "error": f"{type(error).__name__}: {error}"}
        print(name.upper() + " " + json.dumps(checks[name]), flush=True)
    report = {"checked_at_epoch": int(time.time()), "checks": checks,
              "all_content_verified": False,
              "limitation": "Receipt/manifest checks cover recorded exports. Activation comparison checks names/sizes only. "
                            "This does not prove all files on JUPITER were selected, or authorize deletion."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("EXPORT_REPORT " + str(args.output / "report.json"), flush=True)
    return 0 if all(c["status"] in ("pass", "inventory_match") for c in checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
