#!/usr/bin/env python3
"""One-shot, resume-aware view of what's left in the GEODML pipeline.

For every resumable stage (rerank, order_probe, probing) classifies cells into
three buckets:
  - DONE       : gap = 0
  - IN-FLIGHT  : gap > 0 but a job in squeue is working on it
  - READY      : gap > 0 and no job in queue (--resume picks up existing output)

Also shows per-variant kw coverage, slurm queue summary, Stage D headline DML
coefficients, Stage F probing CSV landing status, and the recommended next
action.

Usage (on JUPITER):
    set -a; source .env; set +a
    python scripts/pipeline_status.py            # refreshes manifest then renders
    python scripts/pipeline_status.py --no-refresh   # use existing manifest
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "manifests" / "repair_manifest.parquet"
DATA_ROOT = Path(os.environ.get("GEODML_DATA_ROOT", "/e/scratch/scifi/fourel1"))

G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; B = "\033[1m"
D = "\033[2m"; C = "\033[36m"; X = "\033[0m"
if not sys.stdout.isatty():
    G = Y = R = B = D = C = X = ""


def squeue_jobs() -> list[dict]:
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--noheader",
             "--format=%i|%j|%T|%M|%R"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            rows.append({"jobid": parts[0], "name": parts[1],
                         "state": parts[2], "time": parts[3], "reason": parts[4]})
    return rows


def header(text: str) -> None:
    print(f"\n{B}{'═' * 78}{X}")
    print(f"{B}  {text}{X}")
    print(f"{B}{'═' * 78}{X}")


def refresh_manifest() -> None:
    print(f"{D}  refreshing manifest via repair_audit.py …{X}", file=sys.stderr)
    res = subprocess.run([sys.executable, "scripts/repair_audit.py"],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    if res.returncode != 0:
        print(f"{R}  repair_audit failed (continuing with stale manifest):{X}",
              file=sys.stderr)
        print(res.stderr, file=sys.stderr)


def fmt_cell(row: pd.Series) -> str:
    pool = int(row["pool"]) if pd.notna(row.get("pool")) else "?"
    cell = f"{row.engine}/{str(row.model).split('/')[-1]}/serp{pool}/{row.variant}"
    if pd.notna(row.get("seed")):
        cell += f"/seed{int(row['seed'])}"
    return cell


def main() -> int:
    if "--no-refresh" not in sys.argv and MANIFEST.parent.exists():
        refresh_manifest()

    if not MANIFEST.exists():
        print(f"{R}Manifest missing at {MANIFEST}. "
              f"Run `python scripts/repair_audit.py` first.{X}")
        return 1

    git_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True,
                                cwd=REPO_ROOT).stdout.strip()
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO_ROOT).stdout.strip()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    header(f"PIPELINE STATUS  {ts}")
    print(f"  data_root : {DATA_ROOT}")
    print(f"  git       : {git_branch} @ {git_sha}")
    print(f"  manifest  : {MANIFEST.relative_to(REPO_ROOT)}  "
          f"(mtime {time.strftime('%H:%M', time.localtime(MANIFEST.stat().st_mtime))})")

    df = pd.read_parquet(MANIFEST)
    jobs = squeue_jobs()
    alive = {j["jobid"] for j in jobs}

    # ── Per-stage resumable buckets ─────────────────────────────────────
    header("RESUMABLE STAGES  (DONE / IN-FLIGHT / READY-to-resume)")

    all_ready: list[tuple[str, pd.DataFrame]] = []
    for stage in ("rerank", "order_probe", "probing"):
        sub = df[df["stage"] == stage]
        if sub.empty:
            continue
        done = sub[sub["gap"] == 0]
        in_flight = sub[(sub["gap"] > 0)
                        & (sub["last_jobid"].astype(str).isin(alive))]
        ready = sub[(sub["gap"] > 0)
                    & (~sub["last_jobid"].astype(str).isin(alive))]
        pct = 100.0 * len(done) / len(sub)
        color = G if pct == 100 else (Y if pct >= 50 else R)
        gap_total = int(sub["gap"].sum())

        print(f"\n  {color}{stage:13s}{X}  "
              f"{len(done):>3d}/{len(sub):<3d} done ({pct:5.1f}%)  "
              f"kw_gap={gap_total:>6,d}")
        print(f"    {G}DONE     {X}= {len(done):>3d} cells")
        print(f"    {C}IN-FLIGHT{X}= {len(in_flight):>3d} cells "
              f"(kw_gap_pending={int(in_flight['gap'].sum()):,})")
        print(f"    {Y}READY    {X}= {len(ready):>3d} cells "
              f"(kw_gap_actionable={int(ready['gap'].sum()):,})")

        if len(ready):
            all_ready.append((stage, ready))

    # ── Per-cell READY detail ───────────────────────────────────────────
    if all_ready:
        header("READY-TO-RESUME CELLS  (gap > 0, no job in queue, --resume safe)")
        for stage, ready in all_ready:
            print(f"\n  {Y}{stage}{X}  ({len(ready)} cells, "
                  f"{int(ready['gap'].sum()):,} kw gap):")
            for _, r in ready.sort_values("gap", ascending=False).iterrows():
                print(f"    gap={int(r['gap']):>5,}  "
                      f"({int(r['actual_kw']):>5,}/{int(r['target_kw']):<5,})  "
                      f"{fmt_cell(r)}")
            print(f"    {D}→ python scripts/repair_dispatch.py --stage {stage}{X}")
    else:
        print(f"\n  {G}(no resumable cells outstanding — every gap has a job, "
              f"or every cell is done){X}")

    # ── Per-variant coverage ────────────────────────────────────────────
    header("PER-VARIANT KEYWORD COVERAGE  (from rerank stage)")
    rerank = df[df["stage"] == "rerank"]
    if not rerank.empty:
        for v in sorted(rerank["variant"].unique()):
            s = rerank[rerank["variant"] == v]
            actual = int(s["actual_kw"].sum())
            target = int(s["target_kw"].sum())
            pct = 100.0 * actual / target if target else 0
            color = G if pct >= 95 else (Y if pct >= 80 else R)
            print(f"  {color}{v:14s}{X}  cov={pct:5.1f}%  "
                  f"target={target:>6,d}  actual={actual:>6,d}  "
                  f"missing={target - actual:>5,d}")

    # ── Slurm queue ─────────────────────────────────────────────────────
    header("SLURM QUEUE")
    if not jobs:
        print(f"  {D}(empty){X}")
    else:
        states = Counter(j["state"] for j in jobs)
        for state, n in sorted(states.items()):
            color = G if state == "RUNNING" else (Y if state == "PENDING" else D)
            print(f"  {color}{state:10s}{X} {n:>3d}")
        pd_reasons = Counter(j["reason"] for j in jobs if j["state"] == "PENDING")
        if pd_reasons:
            print(f"\n  {D}pending reasons:{X}")
            for reason, n in pd_reasons.most_common(5):
                print(f"    {n:>3d}×  {reason}")
        # Group running jobs by stage prefix
        run_by_prefix = Counter(j["name"].split("-", 1)[0]
                                for j in jobs if j["state"] == "RUNNING")
        if run_by_prefix:
            print(f"\n  {D}running by stage prefix:{X}")
            for prefix, n in run_by_prefix.most_common():
                print(f"    {n:>3d}×  {prefix}")

    # ── Headline DML ────────────────────────────────────────────────────
    header("STAGE D HEADLINE  (POOLED+plr+lgbm+rank_delta, T7_source_earned)")
    for v in ("biased", "neutral", "biased_rag", "neutral_rag"):
        p = DATA_ROOT / "data" / "dml_results" / f"dml_results_long_{v}.parquet"
        if not p.exists():
            print(f"  {R}{v:14s}  MISSING parquet at {p}{X}")
            continue
        try:
            dml = pd.read_parquet(p)
            hd = dml[(dml.subset == "POOLED") & (dml.method == "plr")
                     & (dml.learner == "lgbm") & (dml.outcome == "rank_delta")
                     & (dml.treatment == "T7_source_earned")]
        except Exception as ex:
            print(f"  {R}{v:14s}  read error: {ex}{X}")
            continue
        if not len(hd):
            print(f"  {R}{v:14s}  no T7 headline row in DML output{X}")
            continue
        r = hd.iloc[0]
        stars = r["sig_stars"] if pd.notna(r["sig_stars"]) else ""
        print(f"  {v:14s}  T7 = {r['coef']:+7.3f}{stars:<3}  "
              f"se={r['se']:.3f}  n={int(r['n_obs']):,}")

    # ── Stage F probing CSVs ────────────────────────────────────────────
    header("STAGE F PROBING  (CSV landing status)")
    pdir = REPO_ROOT / "interpretability" / "output"
    if pdir.exists():
        prob_dirs = sorted(pdir.glob("probing_*"))
        prob_csvs = [d for d in prob_dirs
                     if (d / "probing_results.csv").exists()
                     and (d / "probing_results.csv").stat().st_size > 0]
        color = (G if prob_dirs and len(prob_csvs) == len(prob_dirs)
                 else Y if prob_csvs else R)
        print(f"  {color}{len(prob_csvs)}/{len(prob_dirs)} probing CSVs landed{X}")
        for d in prob_dirs:
            csv = d / "probing_results.csv"
            if csv.exists() and csv.stat().st_size > 0:
                lines = sum(1 for _ in csv.open()) - 1  # minus header
                print(f"    {G}OK {X}  {d.name:50s} rows={lines:>6,d}")
            else:
                print(f"    {Y}-- {X}  {d.name:50s} (empty)")

    # ── Next action ─────────────────────────────────────────────────────
    header("NEXT ACTION")
    total_gap = int(df["gap"].sum())
    if total_gap == 0 and not jobs:
        print(f"  {G}✓ ALL CLEAR — pipeline complete{X}")
        print("    python scripts/publish_dataset.py stage --force")
        print("    python scripts/publish_dataset.py push --repo "
              "ValerianFourel/geodml-emnlp-2026")
    elif all_ready:
        n_ready = sum(len(r) for _, r in all_ready)
        print(f"  {Y}⚠ {n_ready} cells ready to dispatch "
              f"({total_gap:,} kw gap total){X}")
        print("    python scripts/repair_dispatch.py --dry-run | head -30")
        print("    python scripts/repair_dispatch.py")
    else:
        print(f"  {C}… WAITING — {len(jobs)} jobs in flight, total gap={total_gap:,}{X}")
        print("    watch -n 60 'python scripts/pipeline_status.py --no-refresh'")
        print(f"    {D}(or re-run this in ~30 min){X}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
