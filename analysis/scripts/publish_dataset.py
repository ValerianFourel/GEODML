#!/usr/bin/env python3
"""Publish GEODML data to the Hugging Face Hub.

Subcommands:
  clean   — remove backup/scheduling artifacts (.bak, .done_*, .maxkw0-shrunk, ...)
  stage   — clean + copy canonical files into ./hf_stage/ (verify size before push)
  push    — clean + stage + upload to a HF dataset repo
  pull    — download the published dataset to a local directory

Env required:
  HF_TOKEN          — huggingface write token (set in .env or shell)
  GEODML_DATA_ROOT  — only required on JUPITER for clean/stage/push
                      (defaults to /e/scratch/scifi/fourel1)

Optional:
  HF_REPO           — full repo id, e.g. valerianfourel/geodml-emnlp-2026
                      (default: <hf-username>/geodml-emnlp-2026)
  HF_PRIVATE        — "true" to make the repo private (default: true)

Usage on JUPITER:
  .venv/bin/python scripts/publish_dataset.py clean --dry-run
  .venv/bin/python scripts/publish_dataset.py stage
  .venv/bin/python scripts/publish_dataset.py push --repo valerianfourel/geodml-emnlp-2026

Usage on local Mac:
  python scripts/publish_dataset.py pull --repo valerianfourel/geodml-emnlp-2026 \
                                          --out ~/geodml_data
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("GEODML_DATA_ROOT", "/e/scratch/scifi/fourel1"))
STAGE_DIR = REPO_ROOT / "hf_stage"


# ── Cleanup rules ────────────────────────────────────────────────────────────
# Patterns are evaluated with Path.glob() under each root. Match → delete.

CLEAN_FILE_PATTERNS = {
    DATA_ROOT / "data" / "runs": [
        "*/phase2/keywords.jsonl.cap1k.bak",
        "*/phase2/keywords.jsonl.maxkw0-shrunk",
        "*/phase2/*.tmp",
        "*/phase2/*.partial",
    ],
    DATA_ROOT / "data" / "order_probe": [
        "*.bak-*.jsonl*",
        "*.cap1k-*.jsonl",
        "*.pre-unify-*",
        "*passage*",
        ".done_*passage*",
    ],
    REPO_ROOT / "interpretability" / "output": [
        # empty CSVs from aborted runs are caught by EMPTY_FILE_SWEEP below
    ],
}

# Directories to wipe entirely
CLEAN_DIRS = [
    DATA_ROOT / "data" / "order_probe_capped_attic",
]

# After explicit pattern sweep, also delete any 0-byte CSV under interp output
EMPTY_FILE_SWEEP_ROOTS = [
    REPO_ROOT / "interpretability" / "output",
]


# ── Staging rules ────────────────────────────────────────────────────────────
# (target_subdir_in_stage, source_root, glob_pattern)
STAGE_RULES = [
    ("data/runs",            DATA_ROOT / "data" / "runs",         "*/phase2/keywords.jsonl"),
    ("data/order_probe",     DATA_ROOT / "data" / "order_probe",  "*_seed*.jsonl"),
    ("data/features",        DATA_ROOT / "data" / "features",     "*.parquet"),
    ("data/main",            DATA_ROOT / "data" / "main",         "*.parquet"),
    ("data/dml_results",     DATA_ROOT / "data" / "dml_results",  "*.parquet"),
    ("interpretability/output", REPO_ROOT / "interpretability" / "output", "**/*.csv"),
]

# Single files (source_path, target_path_in_stage)
STAGE_FILES = [
    (REPO_ROOT / "docs" / "dml_summary_long.csv", "docs/dml_summary_long.csv"),
    (REPO_ROOT / "docs" / "dml_summary_wide.csv", "docs/dml_summary_wide.csv"),
    (REPO_ROOT / "docs" / "dml_headline.md",      "docs/dml_headline.md"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def size_h(n: int) -> str:
    f = float(n)
    for u in ("B", "K", "M", "G"):
        if f < 1024:
            return f"{f:.1f}{u}"
        f /= 1024
    return f"{f:.1f}T"


def dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file() and not f.is_symlink():
            total += f.stat().st_size
    return total


def load_env() -> None:
    """Source .env if present so HF_TOKEN is picked up."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().lstrip("export ").strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


# ── clean ────────────────────────────────────────────────────────────────────


def clean(dry_run: bool = False) -> int:
    n_files = 0
    n_dirs = 0
    n_bytes = 0
    print(f"[clean] dry_run={dry_run}  DATA_ROOT={DATA_ROOT}")

    # 1. file patterns
    for root, patterns in CLEAN_FILE_PATTERNS.items():
        if not root.exists():
            print(f"  (skip; missing {root})")
            continue
        for pat in patterns:
            for f in root.glob(pat):
                if not f.is_file():
                    continue
                sz = f.stat().st_size
                print(f"  rm {f}  ({size_h(sz)})")
                n_files += 1
                n_bytes += sz
                if not dry_run:
                    f.unlink(missing_ok=True)

    # 2. directories
    for d in CLEAN_DIRS:
        if d.exists():
            sz = dir_size(d)
            print(f"  rm -rf {d}  ({size_h(sz)})")
            n_dirs += 1
            n_bytes += sz
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)

    # 3. empty-file sweep
    for root in EMPTY_FILE_SWEEP_ROOTS:
        if not root.exists():
            continue
        for f in root.rglob("*.csv"):
            if f.is_file() and f.stat().st_size == 0:
                print(f"  rm {f}  (0B empty)")
                n_files += 1
                if not dry_run:
                    f.unlink(missing_ok=True)

    print(f"[clean] would remove {n_files} files + {n_dirs} dirs "
          f"({size_h(n_bytes)} total)" + (" [DRY-RUN]" if dry_run else ""))
    return 0


# ── stage ────────────────────────────────────────────────────────────────────


def stage(force: bool = False) -> int:
    if STAGE_DIR.exists():
        if not force:
            print(f"[stage] {STAGE_DIR} already exists; use --force to overwrite")
            return 1
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)

    print(f"[stage] target: {STAGE_DIR}")
    n_files = 0
    n_bytes = 0

    # rule-based copies
    for target_sub, src_root, pattern in STAGE_RULES:
        if not src_root.exists():
            print(f"  (skip; missing {src_root})")
            continue
        target = STAGE_DIR / target_sub
        target.mkdir(parents=True, exist_ok=True)
        for src in src_root.glob(pattern):
            if not src.is_file() or src.stat().st_size == 0:
                continue
            rel = src.relative_to(src_root)
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_files += 1
            n_bytes += src.stat().st_size

    # single files
    for src, target_rel in STAGE_FILES:
        if not src.exists():
            print(f"  (skip; missing {src})")
            continue
        dst = STAGE_DIR / target_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_files += 1
        n_bytes += src.stat().st_size

    # write README + manifest
    write_readme(STAGE_DIR, n_files, n_bytes)
    write_manifest(STAGE_DIR)

    print(f"[stage] {n_files} files, {size_h(n_bytes)} -> {STAGE_DIR}")
    return 0


def write_readme(stage_dir: Path, n_files: int, n_bytes: int) -> None:
    md = stage_dir / "README.md"
    txt = f"""---
license: cc-by-4.0
language: en
pretty_name: GEODML — Generative-Engine Optimization Double Machine Learning
size_categories:
- 1M<n<10M
tags:
- llm
- ranking
- causal-inference
- double-ml
- rag
- seo
---

# GEODML — EMNLP 2026 submission

End-to-end pipeline outputs for the GEODML paper.

- {n_files} files, {size_h(n_bytes)}
- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}

## Layout

```
data/runs/             Stage A — LLM rerank outputs per (engine × model × pool × variant)
data/order_probe/      Stage A' — order-sensitivity probe per seed
data/features/         Stage B — engineered features (per engine × pool)
data/main/             Stage C — main long-format experiment table per variant
data/dml_results/      Stage D — DML treatment-effect estimates per variant
interpretability/output/  Stage F — ablation / saliency / weights / probing CSVs
docs/                  paper-ready summaries (dml_summary_*, dml_headline.md)
```

## Headline DML estimates

See `docs/dml_headline.md` for the canonical PLR+LightGBM estimates,
pooled across (engine × pool × model) subsets per (variant × treatment × outcome).

## Citation

If you use this dataset, please cite the GEODML paper (EMNLP 2026).
"""
    md.write_text(txt)


def write_manifest(stage_dir: Path) -> None:
    """Per-subdir file counts and sizes — sanity check on what got included."""
    manifest = {}
    for sub in sorted({"data/runs", "data/order_probe", "data/features",
                       "data/main", "data/dml_results",
                       "interpretability/output", "docs"}):
        d = stage_dir / sub
        if not d.exists():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        manifest[sub] = {
            "n_files": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
        }
    (stage_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


# ── push ─────────────────────────────────────────────────────────────────────


def push(repo_id: str, private: bool = True) -> int:
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("install: pip install huggingface_hub", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var not set (and not found in .env)", file=sys.stderr)
        return 2

    if not STAGE_DIR.exists():
        print(f"[push] {STAGE_DIR} missing; run `stage` first", file=sys.stderr)
        return 1

    login(token=token, add_to_git_credential=False)
    api = HfApi()
    print(f"[push] ensuring repo: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="dataset",
                    private=private, exist_ok=True)

    sz = dir_size(STAGE_DIR)
    print(f"[push] uploading {STAGE_DIR}  ({size_h(sz)}) → {repo_id}")
    api.upload_folder(
        folder_path=str(STAGE_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"sync {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[push] done → {url}")
    print()
    print("Pull locally with:")
    print(f"  python scripts/publish_dataset.py pull --repo {repo_id} "
          f"--out ~/geodml_data")
    print("or with the CLI:")
    print(f"  huggingface-cli download {repo_id} --repo-type dataset "
          f"--local-dir ~/geodml_data")
    return 0


# ── pull (run on Mac) ────────────────────────────────────────────────────────


def pull(repo_id: str, out: Path, revision: str = "main") -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("install: pip install huggingface_hub", file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    print(f"[pull] {repo_id} → {out}  (revision={revision})")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out),
        revision=revision,
    )
    print(f"[pull] done → {path}")
    # quick verify
    manifest = out / "MANIFEST.json"
    if manifest.exists():
        print("\nMANIFEST.json contents:")
        print(manifest.read_text())
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    load_env()

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("clean", help="Remove backup / scheduling artifacts")
    p_clean.add_argument("--dry-run", action="store_true")

    p_stage = sub.add_parser("stage", help="Clean + stage to ./hf_stage/")
    p_stage.add_argument("--force", action="store_true",
                         help="overwrite existing ./hf_stage/")
    p_stage.add_argument("--skip-clean", action="store_true")

    p_push = sub.add_parser("push", help="Clean + stage + upload to HF")
    p_push.add_argument("--repo", default=os.environ.get("HF_REPO"),
                        help="full repo id, e.g. user/geodml-emnlp-2026")
    p_push.add_argument("--public", action="store_true",
                        help="make the repo public (default: private)")
    p_push.add_argument("--skip-clean", action="store_true")
    p_push.add_argument("--skip-stage", action="store_true",
                        help="re-use existing ./hf_stage/ without rebuilding")

    p_pull = sub.add_parser("pull", help="Download dataset (run on Mac)")
    p_pull.add_argument("--repo", required=True)
    p_pull.add_argument("--out", type=Path, default=Path.home() / "geodml_data")
    p_pull.add_argument("--revision", default="main")

    args = ap.parse_args()

    if args.cmd == "clean":
        return clean(dry_run=args.dry_run)

    if args.cmd == "stage":
        if not args.skip_clean:
            clean(dry_run=False)
        return stage(force=args.force)

    if args.cmd == "push":
        if not args.repo:
            print("--repo (or HF_REPO env) required", file=sys.stderr)
            return 2
        if not args.skip_clean:
            clean(dry_run=False)
        if not args.skip_stage:
            stage(force=True)
        return push(repo_id=args.repo, private=not args.public)

    if args.cmd == "pull":
        return pull(repo_id=args.repo, out=args.out, revision=args.revision)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
