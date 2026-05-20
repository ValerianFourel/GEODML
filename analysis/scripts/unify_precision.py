#!/usr/bin/env python3
"""Unify ``llm_parameters.precision`` across the GEODML campaign.

Drops records whose ``llm_parameters.precision != target`` from:

  - ``data/runs/*_top10_*/phase2/keywords.jsonl``   (Stage A — rerank)
  - ``data/order_probe/*.jsonl``                    (Stage A' — order_probe)

For every modified JSONL the matching checkpoint is removed so the next
``rerank.py --resume`` / ``order_probe.py --resume`` re-processes the
scrubbed keywords from scratch — both rebuild their ``done`` set by
scanning the JSONL when the ckpt is missing (rerank.py:581-596,
order_probe.py:161-171).

Why per-record scrub instead of per-cell clear (runbook §⑦)
-----------------------------------------------------------
The bf16-reconciliation campaign has already produced a large body of
correct ``bf16-full`` records that interleave with leftover ``4bit-nf4``
and ``api-hf`` records *in the same cells*. Clearing whole cells discards
that work; this script preserves it.

Atomicity / safety
------------------
Each JSONL is backed up to ``<name>.pre-unify-<ts>.bak`` before rewrite,
and the rewrite uses ``tmp + os.replace`` so a concurrent rerank write
never observes a half-written file. Pass ``--no-backup`` to opt out
(irreversible — only if you have a backup elsewhere).

Usage
-----
Dry-run report (default; no writes):
    python scripts/unify_precision.py

Apply in place:
    python scripts/unify_precision.py --execute

Restrict scope:
    python scripts/unify_precision.py --execute --skip-order-probe
    python scripts/unify_precision.py --execute --skip-rerank

Idempotent: a cell already uniform on the target precision is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if USE_COLOR else text


def precision_of(record: dict) -> str:
    return record.get("llm_parameters", {}).get("precision", "MISSING")


def scrub_jsonl(
    path: Path,
    target: str,
    *,
    execute: bool,
    backup: bool,
    timestamp: str,
) -> tuple[int, int, Counter]:
    """Filter ``path`` in place to records with precision == target.

    Returns ``(kept, dropped, before_breakdown)``. When ``dropped == 0`` the
    file is untouched. Otherwise, in execute mode, a backup is created
    (unless ``backup`` is False) and the file is atomically rewritten.
    """
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    before = Counter(precision_of(r) for r in records)
    keep = [r for r in records if precision_of(r) == target]
    dropped = len(records) - len(keep)

    if dropped == 0 or not execute:
        return len(keep), dropped, before

    if backup:
        bak = path.with_name(f"{path.name}.pre-unify-{timestamp}.bak")
        shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".unify_tmp")
    with tmp.open("w") as f:
        for r in keep:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)
    return len(keep), dropped, before


def rerank_ckpt_for(jsonl_path: Path) -> Path:
    return jsonl_path.parent / ".rerank_ckpt.json"


def order_probe_ckpt_for(jsonl_path: Path) -> Path:
    return jsonl_path.parent / f".{jsonl_path.stem}_ckpt.json"


def remove_if_exists(p: Path, *, execute: bool) -> bool:
    if not p.exists():
        return False
    if execute:
        p.unlink()
    return True


def fmt_breakdown(cnt: Counter, target: str) -> str:
    parts = []
    for k, v in cnt.most_common():
        marker = "" if k == target else c("*", RED)
        parts.append(f"{k}{marker}={v}")
    return ", ".join(parts)


def process(
    files: list[Path],
    label: str,
    ckpt_for: Callable[[Path], Path],
    *,
    target: str,
    execute: bool,
    backup: bool,
    timestamp: str,
) -> dict[str, int]:
    totals = {"files": 0, "affected": 0, "kept": 0, "dropped": 0, "ckpts": 0}
    if not files:
        print(c(f"--- {label}: no files ---", DIM))
        return totals
    print(c(f"--- {label} ({len(files)} files) ---", BOLD))
    for p in files:
        totals["files"] += 1
        kept, dropped, before = scrub_jsonl(
            p, target,
            execute=execute, backup=backup, timestamp=timestamp,
        )
        totals["kept"] += kept
        totals["dropped"] += dropped
        if dropped > 0:
            totals["affected"] += 1
            if remove_if_exists(ckpt_for(p), execute=execute):
                totals["ckpts"] += 1
        mark = c(" OK ", GREEN) if dropped == 0 else c(f"-{dropped:>4d}", YELLOW)
        cell_id = (
            p.parent.parent.name if p.name == "keywords.jsonl" else p.stem
        )
        breakdown = fmt_breakdown(before, target)
        print(f"  {mark}  {cell_id:<60s}  kept={kept:>5d}  {breakdown}")
    return totals


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--root",
        default=os.environ.get("GEODML_DATA_ROOT"),
        help="Dataset root (defaults to $GEODML_DATA_ROOT).",
    )
    ap.add_argument(
        "--target",
        default="bf16-full",
        help="Precision label to retain. All other records are dropped. "
             "Default: bf16-full.",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually rewrite files and delete checkpoints. Without this "
             "flag, runs in dry-run mode (report only).",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the .pre-unify-<ts>.bak backup. Irreversible.",
    )
    ap.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip data/runs/*/phase2/keywords.jsonl (Stage A).",
    )
    ap.add_argument(
        "--skip-order-probe",
        action="store_true",
        help="Skip data/order_probe/*.jsonl (Stage A').",
    )
    args = ap.parse_args()

    if not args.root:
        print("ERROR: --root not given and $GEODML_DATA_ROOT is unset.",
              file=sys.stderr)
        return 2
    root = Path(args.root).expanduser().resolve()
    if not (root / "data").is_dir():
        print(f"ERROR: {root}/data does not exist.", file=sys.stderr)
        return 2

    mode = c("EXECUTE", RED) if args.execute else c("DRY-RUN", YELLOW)
    backup = not args.no_backup
    backup_str = "no-backup" if args.no_backup else ".pre-unify-<ts>.bak per modified file"
    print(c(f"=== unify_precision  target={args.target}  mode={mode} ===", BOLD))
    print(f"  data_root: {root}")
    print(f"  backup:    {backup_str}")
    print()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    files_rerank: list[Path] = []
    files_op: list[Path] = []
    if not args.skip_rerank:
        files_rerank = sorted(
            (root / "data" / "runs").glob("*_top10_*/phase2/keywords.jsonl")
        )
    if not args.skip_order_probe:
        op_dir = root / "data" / "order_probe"
        if op_dir.is_dir():
            files_op = sorted(
                p for p in op_dir.glob("*.jsonl")
                if not p.name.startswith(".")
            )

    t_rerank = process(
        files_rerank, "Stage A — rerank (keywords.jsonl)", rerank_ckpt_for,
        target=args.target, execute=args.execute, backup=backup, timestamp=timestamp,
    )
    print()
    t_op = process(
        files_op, "Stage A' — order_probe (*.jsonl)", order_probe_ckpt_for,
        target=args.target, execute=args.execute, backup=backup, timestamp=timestamp,
    )

    print()
    print(c("=== Summary ===", BOLD))
    print(f"  Stage A  : files={t_rerank['files']}  affected={t_rerank['affected']}  "
          f"kept={t_rerank['kept']}  dropped={t_rerank['dropped']}  "
          f"ckpts_cleared={t_rerank['ckpts']}")
    print(f"  Stage A' : files={t_op['files']}  affected={t_op['affected']}  "
          f"kept={t_op['kept']}  dropped={t_op['dropped']}  "
          f"ckpts_cleared={t_op['ckpts']}")

    total_dropped = t_rerank["dropped"] + t_op["dropped"]
    if total_dropped == 0:
        print(c(f"\n  Already uniform on precision={args.target}. Nothing to do.", GREEN))
        return 0

    if not args.execute:
        print(c("\n  DRY-RUN. Re-run with --execute to apply.", YELLOW))
        return 0

    print(c("\n  Done. Next steps:", GREEN))
    print(c("    scancel -u $USER     # if not already done", DIM))
    print(c("    LOCAL_PRECISION=full MAX_KW=0 INCLUDE_RAG_REDO=1 INCLUDE_F_GAPS=1 \\", DIM))
    print(c("      ./scripts/finish_on_gpu.sh", DIM))
    print(c("  After the queue drains:", GREEN))
    print(c("    .venv/bin/python scripts/backfill_precision.py \\", DIM))
    print(c("      --root \"$GEODML_DATA_ROOT\" --include-recent", DIM))
    print(c("    FORCE=1 FEATURES_DEVICE=cuda ./run_full_dml.sh", DIM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
