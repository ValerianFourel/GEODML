#!/usr/bin/env python3
"""Archive how the information-seeking to action-readiness axis was constructed.

Stages explicit groups into a private Hugging Face dataset with MANIFEST.tsv
(path, bytes, sha256, source) and README.md: the prompt corpus, task bank and
codebook, raw judge outputs, the assembled prompt/annotation bundle, LLM2Vec
embeddings, fitted subspace maps, robustness, comparisons, confirmations, and
the manifests and logs of the source-acquisition runs. Dry run by default;
--apply creates the private repo (refusing a public one), uploads and verifies
every file on the Hub.

By owner decision (Valerian, 2026-09-29) the restricted-local scope (WildChat,
MS MARCO and LMSYS-Chat prompts with their grades and embeddings) is included in
this private repository so that the fit is exactly reproducible. Secrets and
files containing tokens are still left out.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.archive_geoaxis_prompts_26k import (  # noqa: E402
    allowed, sha256_file, stage, upload_private, write_manifest)

DEFAULT_REPO = "ValerianFourel/geoaxis-readiness-axis"


def make_api():
    from huggingface_hub import HfApi
    return HfApi()


def ensure_token():
    from analysis.scripts.dispatch_threehour_wave import hf_token
    hf_token(write=True)


def parse_groups(values, manifests_only):
    items = []
    for value in values:
        dest, _, path = value.partition("=")
        if not dest or not path or dest.startswith("/") or ".." in dest.split("/"):
            raise SystemExit(f"bad group {value!r}; use DEST/FOLDER=/source/path")
        items.append((dest.strip("/"), path, Path(path).is_dir(), (), manifests_only))
    return items


def pack(staging, dest, source, manifests_only, max_bytes, left_out):
    """Pack a file-heavy folder into DEST.tar.gz plus DEST.members.tsv (the Hub limits files per folder)."""
    out = staging / f"{dest}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with tarfile.open(out, "w:gz") as tar:
        for folder, dirs, names in os.walk(source):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for name in sorted(names):
                path = os.path.join(folder, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                rel = os.path.normpath(os.path.relpath(path, source))
                size = os.path.getsize(path)
                if not allowed(dest, rel, path, size, max_bytes, left_out, lambda group, rel: True, manifests_only):
                    continue
                tar.add(path, arcname=rel, recursive=False)
                members.append((rel, size, sha256_file(path)))
    listing = staging / f"{dest}.members.tsv"
    listing.write_text("path\tbytes\tsha256\n" + "".join(f"{r}\t{s}\t{d}\n" for r, s, d in members))
    source = os.path.realpath(source)
    return [(f"{dest}.tar.gz", out.stat().st_size, sha256_file(out), source),
            (f"{dest}.members.tsv", listing.stat().st_size, sha256_file(listing), source)], len(members)


def readme(rows, left_out):
    groups = collections.Counter()
    for rel, size, *_ in rows:
        groups["/".join(rel.split("/")[:2])] += size
    listing = "\n".join(f"| `{g}/` | {size / 1e6:,.1f} MB |" for g, size in sorted(groups.items()))
    restricted = sum("restricted-local" in rel for rel, *_ in rows)
    return f"""---
license: other
viewer: false
---
# GeoAxis readiness axis: how it was constructed

Private archive of how GEODML isolated the semantic axis from information
seeking to action readiness. Prompts from six sources were graded by several
judge models; the grades of the completed judges define a supervised subspace
in the latent space of LLM2Vec (McGill/Mila) text embedders. Archived from
JUPITER on {time.strftime('%Y-%m-%d', time.gmtime())} before leaving the cluster.

| Folder | Size |
| --- | ---: |
{listing}

Typical contents: `corpus/` (source corpus), `task-bank/` (blinded judge tasks and
the private codebook), `judges/` (raw judge outputs), `subspace/bundle/`
(prompts, annotations, failures and missing tasks in both scopes),
`subspace/embeddings/` (per-view LLM2Vec embeddings and manifests),
`subspace/maps/` (fitted maps, diagnostics, coordinates),
`subspace/robustness|comparisons|confirmations/`, and `acquisition/` (manifests
and logs of the source-sampling runs). `MANIFEST.tsv` lists every file with its
sha256 and original cluster path. File-heavy folders (for example the judge
queue) are packed as `<name>.tar.gz` with `<name>.members.tsv` listing every
member with its sha256; unpack with `tar -xzf <name>.tar.gz`.

Code: `analysis/scripts/build_readiness_hf_dataset.py` (assemble, embed,
fit-subspace, robustness-battery, finalize) and
`analysis/interpretability/pipeline/readiness_hf_dataset.py`,
`readiness_hf_subspace.py` in the GEODML repository; each manifest records its
git commit, inputs with hashes, and model revisions.

## Access

Private. {restricted} files belong to the `restricted-local` scope (WildChat,
MS MARCO and LMSYS-Chat prompts and derived grades and embeddings). They are
included by owner decision (Valerian, 2026-09-29) so the fit is exactly
reproducible. Their source licenses forbid redistribution: never make this
repository public or share these files. The HF-safe scope is published
separately as `ValerianFourel/geodml-semantic-readiness-20k`.
{sum(len(v) for v in left_out.values())} files were left out (secrets, tokens, bulk files).
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--group", action="append", default=[], help="DEST=/path: file or folder, stored under DEST")
    parser.add_argument("--manifests-only", action="append", default=[],
                        help="DEST=/path: only manifests, logs and small text files (<=50 MB) of this folder")
    parser.add_argument("--pack", action="append", default=[],
                        help="DEST=/folder: pack the whole folder as DEST.tar.gz (for folders with many files)")
    parser.add_argument("--pack-manifests", action="append", default=[],
                        help="DEST=/folder: pack only its manifests, logs and small text files as DEST.tar.gz")
    parser.add_argument("--max-file-gb", type=float, default=5.0)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--apply", action="store_true", help="upload; without it only the staging tree is built")
    args = parser.parse_args(argv)
    items = parse_groups(args.group, False) + parse_groups(args.manifests_only, True)
    packs = parse_groups(args.pack, False) + parse_groups(args.pack_manifests, True)
    if not items and not packs:
        raise SystemExit("nothing to archive: pass --group, --manifests-only, --pack or --pack-manifests")
    for dest, path, is_dir, _, _ in packs:
        if not is_dir:
            raise SystemExit(f"missing folder for pack {dest}: {path}")
    staging = Path(args.staging)
    max_bytes = int(args.max_file_gb * 1e9)
    rows, left_out = stage(staging, items, max_bytes, restricted_ok=lambda group, rel: True)
    for dest, path, _, _, manifests_only in packs:
        packed, count = pack(staging, dest, path, manifests_only, max_bytes, left_out)
        rows += packed
        print(f"PACKED {dest}: {count} files into {packed[0][1] / 1e6:.1f} MB")
    rows = write_manifest(staging, rows, readme(rows, left_out))

    groups = collections.defaultdict(lambda: [0, 0])
    for rel, size, *_ in rows:
        key = "/".join(rel.split("/")[:2])
        groups[key][0] += 1
        groups[key][1] += size
    for key, (count, size) in sorted(groups.items()):
        print(f"  {key:<56} {count:7d} files {size / 1e6:10.1f} MB")
    print(f"RESTRICTED_LOCAL_INCLUDED {sum('restricted-local' in rel for rel, *_ in rows)} files (owner decision)")
    for reason, skipped in sorted(left_out.items()):
        print(f"LEFT_OUT {reason}: {len(skipped)}", *skipped[:10], sep="\n    ")
    total = sum(size for _, size, *_ in rows)
    print(f"STAGED {len(rows)} files, {total / 1e9:.2f} GB in {staging}")
    if not args.apply:
        print("DRY_RUN_OK: rerun with --apply to upload")
        return 0

    ensure_token()
    bad = upload_private(make_api(), args.repo, staging, rows, "GeoAxis readiness axis archive")
    if bad:
        print("VERIFY_FAILED", len(bad), *bad[:20], sep="\n    ")
        return 1
    receipt = {"repo": args.repo, "files": len(rows), "bytes": total, "verified_at": int(time.time())}
    (staging.parent / f"{staging.name}.receipt.json").write_text(json.dumps(receipt, indent=1))
    print(f"GEOAXIS_AXIS_COMPLETE {len(rows)} files, {total / 1e9:.2f} GB verified in "
          f"https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
