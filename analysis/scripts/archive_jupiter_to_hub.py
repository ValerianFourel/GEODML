#!/usr/bin/env python3
"""Archive JUPITER folders into a private Hugging Face dataset as tar.gz parts.

plan    walk each root, drop excluded paths and globs, names containing a skipped
        part (default: restricted-local), files over --max-file-mb, secret files
        and files that contain a token; group the rest into units of about
        --unit-gb and save the plan.
run     for each unit not yet verified: tar | gzip -1 into parts of --part-gb,
        sha256 every part, one commit per unit (parts, member list, manifest),
        verify sizes and sha256 on the Hub, then delete the local parts.
status  planned vs verified units.

Restore one unit:  cat part-*.tar.gz | tar -xzf -   (paths relative to its root)
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import gzip
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_TYPE = "dataset"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{30,}|gh[opsu]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}")
RESTRICTED = re.compile(rb"lmsys|wildchat", re.I)
SECRET_NAMES = {"token", "stored_tokens", ".git-credentials", ".netrc", "id_rsa", "id_ed25519"}
SKIP_SUFFIXES = (".safetensors", ".gguf")
SCAN_MAX = 2_000_000


def make_api():
    from huggingface_hub import HfApi
    return HfApi()


def parse_roots(values):
    roots = {}
    for value in values:
        label, _, path = value.partition("=")
        if not label or not path or "/" in label:
            raise SystemExit(f"bad --root {value!r}; use LABEL=/absolute/path")
        roots[label] = os.path.realpath(path)
    return roots


def parse_excludes(values, roots, flag="--exclude"):
    excludes = collections.defaultdict(list)
    for value in values:
        label, _, rel = value.partition(":")
        if label not in roots or not rel:
            raise SystemExit(f"bad {flag} {value!r}; use LABEL:relative/path")
        excludes[label].append(rel.strip("/"))
    return excludes


def walk(root, excludes, globs=(), skip_names=(), max_bytes=None):
    """Sorted (rel, size, mtime, kind) for regular files and symlinks under root.

    Excluded paths, glob matches and paths containing a skipped name part are
    pruned without reading them; files over max_bytes are counted, not listed.
    """
    found, stack, unreadable = [], [root], 0
    filtered = collections.defaultdict(lambda: [0, 0])  # reason -> [files, bytes]
    while stack:
        folder = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            unreadable += 1
            continue
        for entry in entries:
            rel = os.path.relpath(entry.path, root)
            if any(rel == x or rel.startswith(x + "/") for x in excludes):
                continue
            if any(fnmatch.fnmatchcase(rel, pattern) for pattern in globs):
                filtered["glob"][0] += 1
                continue
            if any(part in rel for part in skip_names):
                filtered["skipped name"][0] += 1
                continue
            try:
                if entry.is_symlink():
                    found.append((rel, 0, 0, "link"))
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    if max_bytes is not None and stat.st_size > max_bytes:
                        filtered["over size limit"][0] += 1
                        filtered["over size limit"][1] += stat.st_size
                        continue
                    found.append((rel, stat.st_size, int(stat.st_mtime), "file"))
            except OSError:
                unreadable += 1
    found.sort()
    return found, unreadable, {reason: tuple(row) for reason, row in filtered.items()}


def group(files, unit_bytes):
    units, current, size = [], [], 0
    for item in files:
        if current and size + item[1] > unit_bytes:
            units.append(current)
            current, size = [], 0
        current.append(item)
        size += item[1]
    if current:
        units.append(current)
    return units


def cmd_plan(args):
    roots = parse_roots(args.root)
    excludes = parse_excludes(args.exclude, roots)
    globs = parse_excludes(args.exclude_glob, roots, "--exclude-glob")
    skip_names = args.skip_name_contains if args.skip_name_contains is not None else ["restricted-local"]
    max_bytes = int(args.max_file_mb * 1e6) if args.max_file_mb else None
    work = Path(args.work)
    plan_dir = work / "plan"
    if plan_dir.exists():
        if not args.replan:
            raise SystemExit(f"{plan_dir} exists; pass --replan to rebuild it (only before uploading)")
        if (work / "done").exists() and any((work / "done").rglob("*.json")):
            raise SystemExit("units are already uploaded; a new plan would not match them. Ask before replanning.")
        shutil.rmtree(plan_dir)
    plan_dir.mkdir(parents=True)
    summary = {"format": "geodml-jupiter-archive-plan-v1", "created_at": int(time.time()),
               "host": socket.gethostname(), "unit_gb": args.unit_gb, "part_gb": args.part_gb,
               "max_file_mb": args.max_file_mb, "skip_name_contains": skip_names, "roots": {}}
    for label, root in roots.items():
        print(f"== scanning {label} = {root}", flush=True)
        files, unreadable, filtered = walk(root, excludes[label], globs[label], skip_names, max_bytes)
        keep, secret, skipped, restricted, restricted_files = [], [], [], collections.Counter(), []
        for number, (rel, size, mtime, kind) in enumerate(files, 1):
            if number % 50000 == 0:
                print(f"   {number} / {len(files)} files checked", flush=True)
            name = os.path.basename(rel)
            if name in SECRET_NAMES:
                secret.append(rel)
                continue
            if rel.endswith(SKIP_SUFFIXES):
                skipped.append(rel)
                continue
            if kind == "file" and size <= SCAN_MAX:
                try:
                    data = Path(root, rel).read_bytes()
                except OSError:
                    data = b""
                if TOKEN.search(data):
                    secret.append(rel)
                    continue
                if RESTRICTED.search(data):
                    restricted["/".join(rel.split("/")[:2])] += 1
                    restricted_files.append(rel)
            keep.append((rel, size, mtime, kind))
        units = group(keep, int(args.unit_gb * 1e9))
        label_dir = plan_dir / label
        label_dir.mkdir()
        with gzip.open(label_dir / "files.tsv.gz", "wt") as index:
            index.write("unit\tpath\tbytes\tmtime\tkind\n")
            for number, unit in enumerate(units, 1):
                name = f"unit-{number:04d}"
                (label_dir / f"{name}.list").write_bytes(b"".join(r.encode() + b"\0" for r, *_ in unit))
                for rel, size, mtime, kind in unit:
                    index.write(f"{name}\t{rel}\t{size}\t{mtime}\t{kind}\n")
        tops = collections.Counter()
        for rel, size, *_ in keep:
            tops[rel.split("/")[0]] += size
        biggest = sorted(keep, key=lambda item: -item[1])[:15]
        total = sum(item[1] for item in keep)
        summary["roots"][label] = {
            "path": root, "excludes": excludes[label], "exclude_globs": globs[label], "filtered": filtered,
            "restricted_files": restricted_files, "files": len(keep), "bytes": total,
            "units": [{"unit": f"unit-{n:04d}", "files": len(u), "bytes": sum(i[1] for i in u),
                       "first": u[0][0], "last": u[-1][0]} for n, u in enumerate(units, 1)],
            "secret_files_left_out": secret, "model_weights_left_out": skipped,
            "restricted_mentions": dict(restricted), "unreadable": unreadable}
        print(f"ROOT {label}: {len(keep)} files, {total / 1e9:.2f} GB in {len(units)} units; "
              f"left out {len(secret)} secret files, {len(skipped)} weight files; unreadable {unreadable}")
        for reason, (count, size) in sorted(filtered.items()):
            print(f"   filtered ({reason}): {count} entries" + (f", {size / 1e9:.2f} GB" if size else ""))
        for top, size in tops.most_common(25):
            print(f"   {size / 1e9:9.2f} GB  {top}")
        print("   largest files:")
        for rel, size, *_ in biggest:
            print(f"   {size / 1e9:9.2f} GB  {rel}")
        if secret:
            print("   secret files left out (names only):", *secret[:20], sep="\n     ")
        if restricted:
            print("   files mentioning lmsys/wildchat, by folder:", dict(restricted.most_common(20)))
            print("   files mentioning lmsys/wildchat (names only):", *restricted_files[:40], sep="\n     ")
    (plan_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    total = sum(r["bytes"] for r in summary["roots"].values())
    units = sum(len(r["units"]) for r in summary["roots"].values())
    mentions = sum(sum(r["restricted_mentions"].values()) for r in summary["roots"].values())
    print(f"PLAN_TOTAL {total / 1e9:.1f} GB in {units} units (before gzip); "
          f"upload about {total / 5e6 / 3600:.1f}-{total / 2e6 / 3600:.1f} h at 5-2 MB/s before compression gains")
    print(f"RESTRICTED_MENTIONS {mentions}")
    print("PLAN_READY" if total <= args.max_gb * 1e9 else f"PLAN_TOO_LARGE: over --max-gb {args.max_gb}")


def retry(action, what):
    for attempt in range(1, 7):
        try:
            return action()
        except Exception as error:  # Hub and network errors vary by library version
            text = str(error)
            transient = any(code in text for code in ("429", "500", "502", "503", "504", "timed out", "Connection"))
            if not transient or attempt == 6:
                raise
            wait = 600 if "429" in text else 60 * attempt
            print(f"   retry {attempt}/5 for {what} in {wait} s: {text[:160]}", flush=True)
            time.sleep(wait)


def lfs_sha(info):
    lfs = getattr(info, "lfs", None)
    if lfs is None:
        return None
    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)


def verify(api, repo, base, manifest):
    paths = [f"{base}/{part['name']}" for part in manifest["parts"]] + [f"{base}/manifest.json"]
    infos = {info.path: info for info in retry(lambda: api.get_paths_info(repo, paths, repo_type=REPO_TYPE), "verify")}
    for part in manifest["parts"]:
        info = infos.get(f"{base}/{part['name']}")
        if info is None or getattr(info, "size", None) != part["bytes"]:
            return False
        sha = lfs_sha(info)
        if sha is not None and sha != part["sha256"]:
            return False
    return f"{base}/manifest.json" in infos


def build_unit(root, listing, stage, part_bytes):
    command = 'tar -C "$1" --null --no-recursion -T "$2" -cf - | gzip -1'
    proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", command, "_", root, str(listing)],
                            stdout=subprocess.PIPE)
    parts, stream, out, digest, written = [], hashlib.sha256(), None, None, 0

    def close():
        out.close()
        parts.append({"name": out.name.rsplit("/", 1)[-1], "bytes": written, "sha256": digest.hexdigest()})

    for chunk in iter(lambda: proc.stdout.read(8 << 20), b""):
        stream.update(chunk)
        while chunk:
            if out is None:
                out = open(stage / f"part-{len(parts):03d}.tar.gz", "wb")
                digest, written = hashlib.sha256(), 0
            take = min(len(chunk), part_bytes - written)
            out.write(chunk[:take])
            digest.update(chunk[:take])
            written += take
            chunk = chunk[take:]
            if written == part_bytes:
                close()
                out = None
    if out is not None:
        close()
    if proc.wait() != 0:
        raise RuntimeError(f"tar/gzip failed with exit {proc.returncode}")
    return parts, stream.hexdigest()


def cmd_run(args):
    work = Path(args.work)
    summary = json.loads((work / "plan/summary.json").read_text())
    total = sum(r["bytes"] for r in summary["roots"].values())
    if total > args.max_gb * 1e9:
        raise SystemExit(f"plan is {total / 1e9:.1f} GB, over --max-gb {args.max_gb}")
    mentions = sum(sum(r["restricted_mentions"].values()) for r in summary["roots"].values())
    if mentions and not args.accept_restricted:
        raise SystemExit(f"{mentions} planned files mention lmsys/wildchat; rerun with --accept-restricted "
                         "only after deciding they may go to a private Hub repo")
    api = make_api()
    retry(lambda: api.create_repo(args.repo, repo_type=REPO_TYPE, private=True, exist_ok=True), "create repo")
    part_bytes = int(summary["part_gb"] * 1e9)
    done_count = 0
    units_total = sum(len(r["units"]) for r in summary["roots"].values())
    for label, root in summary["roots"].items():
        for unit in root["units"]:
            base = f"{label}/{unit['unit']}"
            marker = work / "done" / label / f"{unit['unit']}.json"
            if marker.exists():
                done_count += 1
                print(f"SKIP {base} (verified earlier)", flush=True)
                continue
            stage = work / "stage" / label / unit["unit"]
            shutil.rmtree(stage, ignore_errors=True)
            stage.mkdir(parents=True)
            listing = work / "plan" / label / f"{unit['unit']}.list"
            started = time.time()
            print(f"== {base}: {unit['files']} files, {unit['bytes'] / 1e9:.2f} GB", flush=True)
            parts, stream_sha = build_unit(root["path"], listing, stage, part_bytes)
            members = listing.read_bytes().split(b"\0")[:-1]
            with gzip.open(stage / "members.txt.gz", "wb") as handle:
                handle.write(b"\n".join(members) + b"\n")
            manifest = {"format": "geodml-jupiter-archive-unit-v1", "root_label": label, "root_path": root["path"],
                        "unit": unit["unit"], "files": unit["files"], "bytes_before_compression": unit["bytes"],
                        "first": unit["first"], "last": unit["last"], "parts": parts, "stream_sha256": stream_sha,
                        "restore": "cat part-*.tar.gz | tar -xzf -", "host": socket.gethostname(),
                        "created_at": int(time.time())}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
            packed = sum(p["bytes"] for p in parts)
            print(f"   packed {packed / 1e9:.2f} GB in {len(parts)} part(s) in {time.time() - started:.0f} s; uploading",
                  flush=True)
            retry(lambda: api.upload_folder(repo_id=args.repo, repo_type=REPO_TYPE, folder_path=str(stage),
                                            path_in_repo=base, commit_message=f"JUPITER archive {base}"), base)
            if not verify(api, args.repo, base, manifest):
                raise SystemExit(f"VERIFY FAILED for {base}; local parts kept in {stage}")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({**manifest, "verified_at": int(time.time())}))
            shutil.rmtree(stage)
            done_count += 1
            print(f"DONE {base} ({done_count}/{units_total}) {packed / max(time.time() - started, 1) / 1e6:.1f} MB/s",
                  flush=True)
    index_dir = work / "stage" / "_index"
    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True)
    for label in summary["roots"]:
        shutil.copy(work / "plan" / label / "files.tsv.gz", index_dir / f"{label}-files.tsv.gz")
    shutil.copy(work / "plan" / "summary.json", index_dir / "plan-summary.json")
    (index_dir / "README.md").write_text(
        "# GEODML JUPITER archive\n\nFolders from JUPITER (fscratch, home) archived before leaving the cluster.\n"
        "Each `<root>/unit-NNNN/` holds gzip tar parts, `members.txt.gz` and `manifest.json` (sizes, sha256).\n"
        "Restore a unit with `cat part-*.tar.gz | tar -xzf -`. `index/<root>-files.tsv.gz` maps every file to its unit.\n"
        "Secret files and files containing tokens were left out; see `index/plan-summary.json`.\n")
    retry(lambda: api.upload_folder(repo_id=args.repo, repo_type=REPO_TYPE, folder_path=str(index_dir),
                                    path_in_repo="index", commit_message="JUPITER archive index"), "index")
    retry(lambda: api.upload_file(repo_id=args.repo, repo_type=REPO_TYPE, path_or_fileobj=str(index_dir / "README.md"),
                                  path_in_repo="README.md", commit_message="JUPITER archive README"), "README")
    shutil.rmtree(index_dir)
    (work / "COMPLETE.json").write_text(json.dumps({"units": done_count, "completed_at": int(time.time())}))
    print(f"ARCHIVE_COMPLETE {done_count}/{units_total} units in https://huggingface.co/datasets/{args.repo}")


def cmd_status(args):
    work = Path(args.work)
    if not (work / "plan/summary.json").exists():
        print("ARCHIVE_UNITS 0/0 (no plan yet)\nARCHIVE_COMPLETE=no")
        return
    summary = json.loads((work / "plan/summary.json").read_text())
    planned = done = 0
    uploaded = 0
    for label, root in summary["roots"].items():
        for unit in root["units"]:
            planned += 1
            marker = work / "done" / label / f"{unit['unit']}.json"
            if marker.exists():
                done += 1
                uploaded += sum(p["bytes"] for p in json.loads(marker.read_text())["parts"])
    complete = (work / "COMPLETE.json").exists() and done == planned
    print(f"ARCHIVE_UNITS {done}/{planned} verified, {uploaded / 1e9:.2f} GB uploaded (compressed)")
    print("ARCHIVE_COMPLETE=" + ("yes" if complete else "no"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "run", "status"])
    parser.add_argument("--work", required=True)
    parser.add_argument("--repo", default="ValerianFourel/geodml-jupiter-archive-private")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[], help="LABEL:relative/path, pruned")
    parser.add_argument("--exclude-glob", action="append", default=[], help="LABEL:glob on the relative path, pruned")
    parser.add_argument("--skip-name-contains", action="append",
                        help="leave out any path containing this text (default: restricted-local)")
    parser.add_argument("--max-file-mb", type=float, default=0, help="leave out files larger than this (0: no limit)")
    parser.add_argument("--unit-gb", type=float, default=10)
    parser.add_argument("--part-gb", type=float, default=5)
    parser.add_argument("--max-gb", type=float, default=150)
    parser.add_argument("--replan", action="store_true")
    parser.add_argument("--accept-restricted", action="store_true")
    args = parser.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
