"""Download all or a focused component of the GEODML raw archive.

The default remains the complete ``geodml-papersize`` snapshot for backward
compatibility.  Use ``--component serp`` for the small frozen search-result
pools required to inspect the original candidate sets.

Usage:
    python scripts/download_data.py --component serp
    python scripts/download_data.py --component core
    python scripts/download_data.py --component full --extract-html
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keep --dry-run and public downloads usable in minimal envs.
    def load_dotenv() -> bool:
        return False


COMPONENT_PATTERNS: dict[str, tuple[str, ...] | None] = {
    # None asks huggingface_hub for the complete snapshot. Keep this as the
    # default so existing invocations retain their historical behavior.
    "full": None,
    "serp": (
        "README*",
        "data/serp/**",
    ),
    "rerank": (
        "README*",
        "data/runs/*/phase2/keywords.jsonl",
        "data/runs/*/phase2/rankings.csv",
    ),
    "core": (
        "README*",
        "data/serp/**",
        "data/runs/*/phase2/keywords.jsonl",
        "data/runs/*/phase2/rankings.csv",
        "data/features/**",
        "data/main/**",
        "data/dataforseo/**",
        "data/dml_results/**",
        "data/order_probe/**",
    ),
}


def patterns_for_component(component: str) -> tuple[str, ...] | None:
    """Return Hub allow-patterns for a named download component."""

    try:
        return COMPONENT_PATTERNS[component]
    except KeyError as exc:
        choices = ", ".join(sorted(COMPONENT_PATTERNS))
        raise ValueError(f"Unknown component {component!r}; choose one of: {choices}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all or part of the frozen GEODML raw archive."
    )
    parser.add_argument(
        "--repo", default=os.getenv("HF_DATASET_REPO", "ValerianFourel/geodml-papersize")
    )
    parser.add_argument(
        "--local-dir", default=os.getenv("GEODML_DATA_ROOT", "./geodml_data")
    )
    parser.add_argument(
        "--component",
        choices=sorted(COMPONENT_PATTERNS),
        default="full",
        help=(
            "Download scope: serp=frozen search pools, rerank=ranking inputs/results, "
            "core=analysis tables without bulky caches, full=entire raw archive "
            "(default)."
        ),
    )
    parser.add_argument(
        "--extract-html",
        action="store_true",
        help="After a full download, expand data/runs/*/phase2/html_cache.tar.gz in place.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download, only run post-processing (e.g. --extract-html).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved repository, destination, and file patterns without downloading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.extract_html and args.component != "full":
        parser.error("--extract-html requires --component full")

    local_dir = Path(args.local_dir).resolve()
    patterns = patterns_for_component(args.component)
    token = os.getenv("HF_TOKEN") or None

    if args.dry_run:
        print(f"[dry-run] repo={args.repo}")
        print(f"[dry-run] destination={local_dir}")
        print(f"[dry-run] component={args.component}")
        print(f"[dry-run] allow_patterns={list(patterns) if patterns else 'ALL FILES'}")
        print(f"[dry-run] authentication={'HF_TOKEN' if token else 'anonymous'}")
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        auth_mode = "HF_TOKEN" if token else "anonymous access"
        print(
            f"[download] repo={args.repo} component={args.component} "
            f"auth={auth_mode} -> {local_dir}"
        )
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=8,
            allow_patterns=patterns,
        )
        print("[download] done. Size:")
        subprocess.run(["du", "-sh", str(local_dir)], check=False)

    if args.extract_html:
        runs_dir = local_dir / "data" / "runs"
        if not runs_dir.exists():
            print(f"ERROR: {runs_dir} not found. Download first.")
            return 3
        tarballs = sorted(runs_dir.glob("*/phase2/html_cache.tar.gz"))
        print(f"[extract] found {len(tarballs)} html_cache tarballs")
        for tb in tarballs:
            target = tb.parent / "html_cache"
            if target.exists() and any(target.iterdir()):
                print(f"[extract] skip (already unpacked): {target}")
                continue
            print(f"[extract] {tb}")
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(tb.parent)
        print("[extract] done.")

    print(f"OK. Data root: {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
