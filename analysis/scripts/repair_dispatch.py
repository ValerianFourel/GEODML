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

    # Single-resubmit-authority: the dispatcher owns ALL resubmission. Force
    # MAX_ATTEMPTS=1 so the sbatch's chain_resubmit short-circuits and does NOT
    # spawn an afterany successor — that successor gets a fresh jobid this
    # dispatcher never learns about, so it would keep re-launching the cell on
    # top of a chain still running. With chaining off there is exactly one
    # resubmit authority (this loop) and duplicates can't pile up.
    exp += ",MAX_ATTEMPTS=1"
    # Re-arm skip_if_at_max for Stage A/A' so an already-complete cell exits
    # without paying the ~5 min 70B model-load. (Bare dispatch never passed
    # MAX_KEYWORDS, so the guard was a no-op.)
    if stage in ("rerank", "order_probe") and pd.notna(row.get("target_kw")):
        exp += f",MAX_KEYWORDS={int(row['target_kw'])}"

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
    ap.add_argument("--max-inflight", type=int,
                    default=int(os.environ.get("REPAIR_MAX_INFLIGHT", "24")),
                    help="Global ceiling on SUBMITTED+RUNNING cells. Never let the "
                         "queue exceed this many GEODML jobs, no matter how many "
                         "cells still have gap > 0.")
    ap.add_argument("--max-attempts-per-cell", type=int,
                    default=int(os.environ.get("REPAIR_MAX_SUBMITS_PER_CELL", "8")),
                    help="Stop resubmitting a cell after this many total launches "
                         "(absolute backstop; audit then marks it STUCK).")
    ap.add_argument("--stuck-threshold", type=int,
                    default=int(os.environ.get("REPAIR_STUCK_THRESHOLD", "2")),
                    help="Stop resubmitting a cell after this many launches in a "
                         "row that produced no new keywords (resets on progress).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--account", default=os.environ.get("JUWELS_ACCOUNT", "scifi"))
    ap.add_argument("--partition", default="booster")
    ap.add_argument("--status", action="store_true",
                    help="Print current status table and exit without submitting.")
    args = ap.parse_args()

    # Clamp to sane minimums: a 0 here would either stall the queue forever
    # (max_inflight=0 → room=0 → nothing ever submits) or quarantine untried
    # cells (stuck_threshold=0 → ceiling hit before the first launch).
    args.stuck_threshold = max(1, args.stuck_threshold)
    args.max_attempts_per_cell = max(1, args.max_attempts_per_cell)
    args.max_inflight = max(1, args.max_inflight)

    if not MANIFEST.exists():
        print(f"[FATAL] manifest not found: {MANIFEST}")
        print("Run `python scripts/repair_audit.py` first.", file=sys.stderr)
        sys.exit(2)

    df = pq.read_table(MANIFEST).to_pandas()

    # Ensure circuit-breaker counter columns exist (older manifests lack them;
    # repair_audit also creates them, but dispatch may run on a stale manifest).
    if "submit_count" not in df.columns:
        df["submit_count"] = 0
    df["submit_count"] = df["submit_count"].fillna(0).astype(int)
    if "submits_since_progress" not in df.columns:
        df["submits_since_progress"] = 0
    df["submits_since_progress"] = df["submits_since_progress"].fillna(0).astype(int)
    if "best_actual_kw" not in df.columns:
        df["best_actual_kw"] = df["actual_kw"]

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
    # Circuit-breaker: never resubmit a cell that has exhausted its attempts.
    # These cells stop being launched here, drain out of the queue, and the next
    # audit marks them STUCK so the loop can terminate. (STUCK / DONE are already
    # excluded by the status filter above; this also catches cells the audit has
    # not re-evaluated yet — e.g. a stale manifest.)
    ceiling_hit = (
        (df["submit_count"] >= args.max_attempts_per_cell)
        | (df["submits_since_progress"] >= args.stuck_threshold)
    )
    blocked = df[candidates_mask & ceiling_hit]
    candidates_mask &= ~ceiling_hit
    candidates = df[candidates_mask].copy()
    # Prioritise largest gaps first
    candidates = candidates.sort_values("gap", ascending=False)

    print(f"\n[plan] {len(candidates)} cells with gap > 0 awaiting submission")
    if len(blocked):
        print(f"[ceiling] {len(blocked)} cell(s) with gap > 0 held back: hit "
              f"--max-attempts-per-cell={args.max_attempts_per_cell} or "
              f"--stuck-threshold={args.stuck_threshold} (audit will mark STUCK).")

    if args.status:
        _print_status(df)
        return

    # ── 3. Submit up to the per-cycle budget AND the global in-flight cap ────
    inflight = int(df["status"].isin(["SUBMITTED", "RUNNING"]).sum())
    room = max(0, args.max_inflight - inflight)
    budget = min(args.max_submissions, room)
    print(f"[inflight] {inflight} cell(s) in queue; room={room} "
          f"(max_inflight={args.max_inflight}); submitting ≤ {budget} this cycle")

    n_submitted = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for idx, row in candidates.iterrows():
        if n_submitted >= budget:
            limiter = ("--max-submissions" if args.max_submissions <= room
                       else "--max-inflight")
            print(f"  hit {limiter} (budget={budget}); stopping for this cycle")
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
        # Circuit-breaker bookkeeping: every launch counts, and counts against
        # the no-progress streak until the next audit observes new keywords.
        df.at[idx, "submit_count"] = int(df.at[idx, "submit_count"]) + 1
        df.at[idx, "submits_since_progress"] = (
            int(df.at[idx, "submits_since_progress"]) + 1)
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
