#!/usr/bin/env python3
"""Resumable dispatcher for the repair manifest.

Reads `manifests/repair_manifest.parquet` and:
  1. Re-audits any cell currently in SUBMITTED/RUNNING by polling squeue, so
     statuses reflect the cluster state.
  2. For cells with gap > 0 and status in (TODO, FAILED): submits an sbatch
     job using the existing per-stage template, then marks status=SUBMITTED.
  3. Writes the manifest back. Safe to Ctrl-C and re-run — each invocation is
     idempotent.

By design this script does NOT loop; you re-invoke it (or put it in cron / a
`watch -n 600 ...`) to keep submitting as cells finish. That avoids long-lived
processes on the login node.

Usage:
  python scripts/repair_dispatch.py                        # submit pending, poll running
  python scripts/repair_dispatch.py --dry-run              # print sbatch commands only
  python scripts/repair_dispatch.py --stage rerank         # only this stage
  python scripts/repair_dispatch.py --max-submissions 8    # rate-limit
  python scripts/repair_dispatch.py --account scifi        # override JUWELS_ACCOUNT
  python scripts/repair_dispatch.py --status               # show status table only
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "manifests" / "repair_manifest.parquet"

SBATCH_TEMPLATES = {
    "rerank":      "scripts/slurm/run_rerank.sbatch",
    "order_probe": "scripts/slurm/run_order_probe.sbatch",
    "probing":     "scripts/slurm/run_probing.sbatch",
}


def squeue_alive(jobids: list[str]) -> set[str]:
    """Return the set of jobids currently in the queue (any state)."""
    jobids = [j for j in jobids if j]
    if not jobids:
        return set()
    try:
        res = subprocess.run(
            ["squeue", "-h", "--jobs", ",".join(jobids), "--format=%i"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    alive = {ln.strip() for ln in res.stdout.splitlines() if ln.strip()}
    return alive


def submit_one(row: pd.Series, account: str, partition: str, dry_run: bool) -> str | None:
    """sbatch one cell; return jobid string or None on failure."""
    stage = row["stage"]
    template = REPO_ROOT / SBATCH_TEMPLATES[stage]
    if not template.exists():
        print(f"  [ERR] template missing: {template}")
        return None

    # build --export string per stage
    if stage == "rerank":
        exp = (
            f"ALL,MODEL={row['model']},ENGINE={row['engine']},"
            f"POOL={int(row['pool'])},PROMPT_VARIANT={row['variant']},"
            f"LOCAL_PRECISION=full"
        )
        jobname = (f"rerank-{row['model'].split('/')[-1]}-"
                   f"{row['engine']}-{int(row['pool'])}-{row['variant']}")
    elif stage == "order_probe":
        exp = (
            f"ALL,MODEL={row['model']},ENGINE={row['engine']},"
            f"POOL={int(row['pool'])},PROMPT_VARIANT={row['variant']},"
            f"SEED={int(row['seed'])},LOCAL_PRECISION=full"
        )
        jobname = (f"op-{row['model'].split('/')[-1]}-{row['engine']}-"
                   f"{int(row['pool'])}-{row['variant']}-s{int(row['seed'])}")
    elif stage == "probing":
        exp = (
            f"ALL,MODEL={row['model']},PROMPT_VARIANT={row['variant']},"
            f"LOCAL_PRECISION=full"
        )
        jobname = f"prob-{row['model'].split('/')[-1]}-{row['variant']}"
    else:
        print(f"  [ERR] unknown stage: {stage}")
        return None

    cmd = [
        "sbatch",
        f"--account={account}",
        f"--partition={partition}",
        "--time=05:55:00",
        f"--job-name={jobname}",
        f"--output=logs/{jobname}-%j.out",
        f"--error=logs/{jobname}-%j.err",
        f"--export={exp}",
        str(template),
    ]
    if dry_run:
        print("  [DRY] " + " ".join(shlex.quote(c) for c in cmd))
        return "DRY-RUN"

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as ex:
        print(f"  [ERR] sbatch failed: {ex}")
        return None
    if res.returncode != 0:
        print(f"  [ERR] sbatch rc={res.returncode}: {res.stderr.strip()}")
        return None
    out = res.stdout.strip()
    # Typical output: "Submitted batch job 12345"
    jobid = out.split()[-1] if out else ""
    print(f"  [OK]  {jobname}  jobid={jobid}")
    return jobid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rerank", "order_probe", "probing", "all"],
                    default="all")
    ap.add_argument("--max-submissions", type=int, default=32,
                    help="Maximum number of new sbatch submissions per invocation.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--account", default=os.environ.get("JUWELS_ACCOUNT", "scifi"))
    ap.add_argument("--partition", default="booster")
    ap.add_argument("--status", action="store_true",
                    help="Print current status table and exit without submitting.")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"[FATAL] manifest not found: {MANIFEST}")
        print("Run `python scripts/repair_audit.py` first.", file=sys.stderr)
        sys.exit(2)

    df = pq.read_table(MANIFEST).to_pandas()

    # ── 1. Poll SLURM and resolve SUBMITTED/RUNNING ─────────────────────────
    in_flight = df[df["status"].isin(["SUBMITTED", "RUNNING"])]
    if len(in_flight):
        alive = squeue_alive(list(in_flight["last_jobid"].astype(str)))
        print(f"[poll] {len(in_flight)} in-flight cells; {len(alive)} still in queue")
        # mark cells whose jobid is no longer in squeue: status TBD — repair_audit
        # will re-check the cell's gap and update DONE/FAILED. For now mark them
        # PENDING_RECHECK so the next audit run can finalise them.
        df.loc[(df["status"].isin(["SUBMITTED", "RUNNING"]))
               & (~df["last_jobid"].astype(str).isin(alive)),
               "status"] = "PENDING_RECHECK"
        df.loc[(df["status"].isin(["SUBMITTED", "RUNNING"]))
               & (df["last_jobid"].astype(str).isin(alive)),
               "status"] = "RUNNING"

    # ── 2. Filter to candidates to submit ───────────────────────────────────
    if args.stage != "all":
        candidates_mask = (df["stage"] == args.stage) & (df["gap"] > 0)
    else:
        candidates_mask = df["gap"] > 0
    candidates_mask &= df["status"].isin(["TODO", "FAILED", "PENDING_RECHECK"])
    candidates = df[candidates_mask].copy()
    # Prioritise largest gaps first
    candidates = candidates.sort_values("gap", ascending=False)

    print(f"\n[plan] {len(candidates)} cells with gap > 0 awaiting submission")

    if args.status:
        _print_status(df)
        return

    # ── 3. Submit up to --max-submissions ───────────────────────────────────
    n_submitted = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for idx, row in candidates.iterrows():
        if n_submitted >= args.max_submissions:
            print(f"  hit --max-submissions={args.max_submissions}; stopping for this cycle")
            break
        jobid = submit_one(row, args.account, args.partition, args.dry_run)
        if jobid is None:
            df.at[idx, "status"] = "FAILED"
            df.at[idx, "last_check"] = now
            continue
        df.at[idx, "status"] = "SUBMITTED"
        df.at[idx, "last_jobid"] = jobid
        df.at[idx, "last_submitted"] = now
        df.at[idx, "last_check"] = now
        n_submitted += 1

    # ── 4. Persist + summary ────────────────────────────────────────────────
    if not args.dry_run:
        df.to_parquet(MANIFEST)
        print(f"\n[manifest] {n_submitted} new submissions; saved → "
              f"{MANIFEST.relative_to(REPO_ROOT)}")

    _print_status(df)


def _print_status(df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("STATUS TABLE")
    print("=" * 88)
    pivot = df.groupby(["stage", "status"]).size().unstack(fill_value=0)
    print(pivot.to_string())
    print()
    print(df.groupby("stage")["gap"].agg(["sum", "count"])
          .rename(columns={"sum": "total_kw_gap", "count": "cells"})
          .to_string())


if __name__ == "__main__":
    main()
