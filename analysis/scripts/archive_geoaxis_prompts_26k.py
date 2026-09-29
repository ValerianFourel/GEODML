#!/usr/bin/env python3
"""Archive the 26,009-prompt GeoAxis population into a private Hugging Face dataset.

Collects an allowlist (final audit, axis maps, robustness battery, plan, final
checkpoint manifests, population registration, selection manifest, pointer
files) into a staging tree and writes MANIFEST.tsv (path, bytes, sha256,
source) and README.md. Dry run by default. --apply creates the private dataset
(refusing a public one), uploads the staging tree and verifies every file's size
and hash on the Hub.

By owner decision (Valerian, 2026-09-29) the final audit's restricted-local
embeddings of the selected prompts are included in this private repo. Every
other restricted-local file (e.g. axis exemplar text) is left out.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.archive_jupiter_to_hub import (  # noqa: E402
    REPO_TYPE, SCAN_MAX, SECRET_NAMES, SKIP_SUFFIXES, TOKEN, lfs_sha, retry)

DEFAULT_REPO = "ValerianFourel/geoaxis-prompts-generation-26k"
POPULATION_ROWS = 26009
# Frozen population hashes (analysis/docs/agentic_paper_dataset_contract.md).
CONTRACT = {
    "final-audit/compliant-candidates.jsonl": "1718321f8fc86f63d00aab30e87991ada9b62a59c5ef4ce99b5acef0df4d32e9",
    "final-audit/final-axis-map.jsonl": "43189f68bcafc77f9dceb7a1a8d993251d4c2a739b401ef4fd24cb64e292682e",
}
SKIP_DIRS = {"logs", "quarantine", "__pycache__"}
EMBEDDING_NAME = "question_embeddings.restricted-local.npz"


def make_api():
    from huggingface_hub import HfApi
    return HfApi()


def ensure_token():
    from analysis.scripts.dispatch_threehour_wave import hf_token
    hf_token(write=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def allowed(group, rel, path, size, max_bytes, left_out):
    """Decide whether one file of a group goes into the archive; record why not."""
    name = os.path.basename(rel)
    parts = rel.split("/")
    if name in SECRET_NAMES:
        left_out["secret"].append(f"{group}/{rel}")
        return False
    if name.endswith(SKIP_SUFFIXES):
        left_out["model weights"].append(f"{group}/{rel}")
        return False
    if "restricted-local" in rel:
        embedding = group == "final-audit" and parts[0] == "projections" and name == EMBEDDING_NAME
        if not embedding:
            left_out["restricted-local"].append(f"{group}/{rel}")
            return False
    elif size > max_bytes:
        left_out["over size limit"].append(f"{group}/{rel}")
        return False
    if size <= SCAN_MAX and TOKEN.search(Path(path).read_bytes()):
        left_out["contains a token"].append(f"{group}/{rel}")
        return False
    return True


def sources(args):
    """(destination group, source path, is_directory, skipped top-level dir prefixes)."""
    items = [("final-audit", args.final_audit, True, ())]
    for value in args.map:
        name, _, path = value.partition("=")
        if not name or not path or "/" in name:
            raise SystemExit(f"bad --map {value!r}; use NAME=/path")
        items.append((f"maps/{name}", path, True, ()))
    for group, path in (("battery", args.battery), ("plan", args.plan),
                        ("registration/population-registration-v1", args.registration)):
        if path:
            items.append((group, path, True, ()))
    if args.checkpoint:
        items.append(("checkpoint", args.checkpoint, True, ("final-audit",)))
    if args.selection_manifest:
        items.append(("selection/selection-manifest.json", args.selection_manifest, False, ()))
    for path in args.pointer:
        items.append((f"pointers/{os.path.basename(path)}", path, False, ()))
    for group, path, is_dir, _ in items:
        if not (Path(path).is_dir() if is_dir else Path(path).is_file()):
            raise SystemExit(f"missing {'folder' if is_dir else 'file'} for {group}: {path}")
    return items


def collect(args):
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    for child in staging.iterdir():  # rebuild, but keep the uploader's resume cache
        if child.name != ".cache":
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    max_bytes = int(args.max_file_gb * 1e9)
    left_out = collections.defaultdict(list)
    rows = []
    for group, source, is_dir, skip_prefixes in sources(args):
        if is_dir:
            files = []
            for folder, dirs, names in os.walk(source):
                rel_folder = os.path.relpath(folder, source)
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
                                 and not (rel_folder == "." and d.startswith(skip_prefixes)))
                for name in sorted(names):
                    path = os.path.join(folder, name)
                    if os.path.islink(path) or not os.path.isfile(path):
                        continue
                    files.append((os.path.normpath(os.path.join(rel_folder, name)), path))
        else:
            files = [("", source)]
        for rel, path in files:
            size = os.path.getsize(path)
            dest_rel = f"{group}/{rel}" if rel else group
            if rel and not allowed(group, rel, path, size, max_bytes, left_out):
                continue
            if not rel and size <= SCAN_MAX and TOKEN.search(Path(path).read_bytes()):
                left_out["contains a token"].append(dest_rel)
                continue
            dest = staging / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, dest)
            except OSError:
                shutil.copy2(path, dest)
            rows.append((dest_rel, size, sha256_file(dest), os.path.realpath(path)))
    return staging, rows, left_out


def readme(rows, left_out, contract):
    groups = collections.Counter()
    for rel, size, *_ in rows:
        groups[rel.split("/")[0]] += size
    listing = "\n".join(f"| `{g}/` | {size / 1e6:,.1f} MB |" for g, size in sorted(groups.items()))
    hashes = "\n".join(f"| `{name}` | `{digest}` | {state} |" for name, (digest, state) in contract.items())
    return f"""---
license: other
viewer: false
---
# GeoAxis prompt population (26,009 prompts), generation archive

Private archive of the GEODML Experiment V2 prompt population: every prompt,
its position on the 0–1 semantic axis from information seeking to action
readiness, and what is needed to reuse those positions **without generating or
embedding the prompts again**. Archived from JUPITER on
{time.strftime('%Y-%m-%d', time.gmtime())} before leaving the cluster.

| Folder | Size |
| --- | ---: |
{listing}

- `final-audit/`: `compliant-candidates.jsonl` (the 26,009 prompts),
  `final-axis-map.jsonl` (`consensus_axis_1_z`, `axis_1_rank`,
  `axis_1_percentile_0_1`), summaries, per-view merged projections, comparison,
  and `projections/{{qwen,mistral}}/shard-*/question_embeddings.restricted-local.npz`
  (float32 embeddings of exactly these prompts, 4096 dimensions, arrays
  `candidate_ids` and `embeddings`).
- `maps/`: readiness embedding maps (`readiness_embedding_map.json`,
  `readiness_supervised_subspace_coordinates.jsonl`, `subspace_manifest.json`)
  for the Qwen3-8B and Mistral-7B LLM2Vec views.
- `battery/`: the Qwen/Mistral robustness battery with the cross-embedding alignment.
- `plan/`: the axis-1 target plan and subspace bounds.
- `checkpoint/`: the final global merge (candidates, validation, strict
  selection, manifests) that the final audit re-checked.
- `registration/population-registration-v1/`: the files the axis analysis reads
  (`population-selection-records.jsonl`, `manifest.json`, ...), byte-identical.
- `selection/`, `pointers/`: the pilot selection manifest and cluster pointer files.

## Frozen contract hashes

| File | sha256 | Here |
| --- | --- | --- |
{hashes}

## Reuse without re-embedding

Axis positions are in `final-axis-map.jsonl`. To project a *new* prompt: embed it
with the same LLM2Vec view (models and revisions in the map manifests), L2-normalise,
subtract `embedding_mean`, project on `supervised_subspace_axes`, normalise with
the plan's subspace bounds, align Mistral to Qwen with the battery's
`cross_embedding_alignment`, average the two z-scores and rank it against the
population's `consensus_axis_1_z` (code: `readiness_prompt_population.py`
`project_text_embeddings`, `build_readiness_prompt_population.py compare-projections`).

## Access and provenance

Private. The embeddings are labelled `restricted-local` by the pipeline because
the axis was fit on a restricted-local scope. They are included here by owner
decision (Valerian, 2026-09-29) for this private repository only; do not make
this repository public or redistribute its files. Axis exemplar text and every
other restricted-local file were left out ({sum(len(v) for v in left_out.values())} files left out
in total; see `MANIFEST.tsv` for everything included, with sha256 and source path).
"""


def verify(api, repo, staging, rows):
    infos = {}
    paths = [rel for rel, *_ in rows]
    for start in range(0, len(paths), 100):
        batch = paths[start:start + 100]
        for info in retry(lambda: api.get_paths_info(repo, batch, repo_type=REPO_TYPE), "verify"):
            infos[info.path] = info
    bad = []
    for rel, size, digest, _ in rows:
        info = infos.get(rel)
        if info is None or getattr(info, "size", None) != size:
            bad.append(rel)
            continue
        remote = lfs_sha(info)
        if remote is not None:
            if remote != digest:
                bad.append(rel)
        elif getattr(info, "blob_id", None) != git_blob_sha1(staging / rel):
            bad.append(rel)
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", required=True, help="empty or earlier staging folder, on the same filesystem if possible")
    parser.add_argument("--final-audit", required=True)
    parser.add_argument("--map", action="append", default=[], help="NAME=/path/to/map/folder")
    parser.add_argument("--battery")
    parser.add_argument("--plan")
    parser.add_argument("--checkpoint")
    parser.add_argument("--registration")
    parser.add_argument("--selection-manifest")
    parser.add_argument("--pointer", action="append", default=[])
    parser.add_argument("--max-file-gb", type=float, default=2.0)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--apply", action="store_true", help="upload; without it only the staging tree is built")
    args = parser.parse_args(argv)

    staging, rows, left_out = collect(args)
    by_rel = {rel: digest for rel, _, digest, _ in rows}
    contract = {name: (digest, "MATCHES" if by_rel.get(name) == digest else f"DIFFERS ({by_rel.get(name, 'missing')})")
                for name, digest in CONTRACT.items()}
    with open(staging / "MANIFEST.tsv", "w") as handle:
        handle.write("path\tbytes\tsha256\tsource\n")
        handle.writelines(f"{rel}\t{size}\t{digest}\t{src}\n" for rel, size, digest, src in rows)
    (staging / "README.md").write_text(readme(rows, left_out, contract))
    rows += [(name, (staging / name).stat().st_size, sha256_file(staging / name), "generated")
             for name in ("MANIFEST.tsv", "README.md")]

    groups = collections.defaultdict(lambda: [0, 0])
    for rel, size, *_ in rows:
        key = "/".join(rel.split("/")[:2]) if rel.startswith(("maps/", "registration/")) else rel.split("/")[0]
        groups[key][0] += 1
        groups[key][1] += size
    for key, (count, size) in sorted(groups.items()):
        print(f"  {key:<48} {count:6d} files {size / 1e6:10.1f} MB")
    embeddings = [rel for rel, *_ in rows if rel.endswith(EMBEDDING_NAME)]
    print(f"EMBEDDING_SHARDS {len(embeddings)} (restricted-local, included by owner decision)")
    for reason, items in sorted(left_out.items()):
        print(f"LEFT_OUT {reason}: {len(items)}", *items[:10], sep="\n    ")
    for name, (_, state) in contract.items():
        print(f"CONTRACT {name} {state}")
    total = sum(size for _, size, *_ in rows)
    print(f"STAGED {len(rows)} files, {total / 1e9:.2f} GB in {staging}")
    if any(state != "MATCHES" for _, state in contract.values()):
        raise SystemExit("CONTRACT_MISMATCH: the population files differ from the frozen hashes; nothing uploaded")
    if not args.apply:
        print("DRY_RUN_OK: rerun with --apply to upload")
        return 0

    ensure_token()
    api = make_api()
    retry(lambda: api.create_repo(args.repo, repo_type=REPO_TYPE, private=True, exist_ok=True), "create repo")
    if not getattr(retry(lambda: api.repo_info(args.repo, repo_type=REPO_TYPE), "repo info"), "private", False):
        raise SystemExit(f"{args.repo} is not private; refusing to upload")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=args.repo, repo_type=REPO_TYPE, folder_path=str(staging), private=True)
    else:
        retry(lambda: api.upload_folder(repo_id=args.repo, repo_type=REPO_TYPE, folder_path=str(staging),
                                        commit_message="GeoAxis 26k archive", ignore_patterns=[".cache/**"]), "upload")
    bad = verify(api, args.repo, staging, rows)
    if bad:
        print("VERIFY_FAILED", len(bad), *bad[:20], sep="\n    ")
        return 1
    receipt = {"repo": args.repo, "files": len(rows), "bytes": total, "verified_at": int(time.time()),
               "contract": {name: state for name, (_, state) in contract.items()}}
    (staging.parent / f"{staging.name}.receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"GEOAXIS_26K_COMPLETE {len(rows)} files, {total / 1e9:.2f} GB verified in "
          f"https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
