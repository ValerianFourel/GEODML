#!/usr/bin/env python3
"""Consolidated study report — pastes a single text blob to feed back for analysis.

Sections (each section is wrapped in try/except so one failure doesn't kill the rest):
  - HEADER:    timestamp, git, branch, env
  - JOBS:      squeue + scontrol/.out tail for specific job IDs (default 487548)
  - AUDIT:     per-stage counts (uses scripts/audit_files.py --tsv)
  - STAGE D:   DML estimates per variant — point estimates with CIs
  - ABLATION:  per (treatment, variant, model) effect summary
  - SALIENCY:  per-variant, per-frame summary CSV contents
  - WEIGHTS:   logit_lens.csv numeric describe per variant
  - PROBING:   existence + size status

Usage:
  .venv/bin/python scripts/study_report.py
  .venv/bin/python scripts/study_report.py --job 487548 487547
  .venv/bin/python scripts/study_report.py --skip saliency probing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path(os.environ.get("GEODML_DATA_ROOT", "/e/scratch/scifi/fourel1"))
INTERP_OUT = REPO_ROOT / "interpretability" / "output"

ACTIVE_VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
MODELS_SHORT = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"]
TREATMENTS = [
    "T7_source_earned", "T5_topical_comp", "T3_structured_data_new",
    "T2a_question_headings", "T6_freshness", "T1b_stats_density",
]
FRAMES = [("full", "_full"), ("robust_winners", "_rw")]


def hr(title: str = "", char: str = "=") -> str:
    line = char * 86
    return f"\n{line}\n{title}\n{line}" if title else f"\n{line}"


def sh(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "" if r.returncode else "")
    except Exception as e:
        return f"<sh error: {e}>"


# ── Sections ──────────────────────────────────────────────────────────────────


def section_header(args) -> None:
    print(hr("STUDY REPORT — GEODML"))
    print(f"timestamp_utc : {datetime.utcnow().isoformat(timespec='seconds')}Z")
    print(f"data_root     : {DATA_ROOT}")
    print(f"repo          : {REPO_ROOT}")
    print(f"hostname      : {sh('hostname').strip()}")
    print(f"branch        : {sh(f'git -C {REPO_ROOT} branch --show-current').strip()}")
    print(f"git_head      : {sh(f'git -C {REPO_ROOT} log -1 --format=%h %s').strip()}")
    print(f"slurm R       : {sh('squeue -u $USER -h -t R 2>/dev/null | wc -l').strip()}")
    print(f"slurm PD      : {sh('squeue -u $USER -h -t PD 2>/dev/null | wc -l').strip()}")


def section_jobs(args) -> None:
    print(hr("SLURM JOBS"))
    print(sh('squeue -u $USER -o "%.10i %.20j %.2t %.10M %.6D %R" 2>&1 | head -40'))
    for jid in (args.job or []):
        print(hr(f"JOB {jid} — scontrol + .out tail", char="-"))
        scon = sh(f'scontrol show job {jid} 2>&1 | head -25')
        print(scon)
        log = sh(f'ls -t logs/*-{jid}.out 2>/dev/null | head -1').strip()
        if log:
            print(f"\n--- {log} (tail -40) ---")
            print(sh(f'tail -40 "{log}"'))
        else:
            print(f"(no log file matching *-{jid}.out under logs/)")


def section_audit(args) -> None:
    print(hr("AUDIT — stage counts"))
    out = sh(f'.venv/bin/python {REPO_ROOT}/scripts/audit_files.py --tsv 2>/dev/null', timeout=60)
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) < 2:
        print("(audit_files.py produced no output — fallback to disk scan skipped)")
        return
    headers = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    by_stage: dict = {}
    for r in rows:
        rec = dict(zip(headers, r))
        s = rec["stage"]
        b = by_stage.setdefault(s, {"exists": 0, "total": 0, "flagged": 0, "rows": []})
        b["total"] += 1
        if rec.get("exists") == "1":
            b["exists"] += 1
        flags = rec.get("flags", "")
        if flags and flags != "DONE":
            b["flagged"] += 1
        try:
            n = int(rec.get("rows", "0"))
        except ValueError:
            n = 0
        if n > 0:
            b["rows"].append(n)
    print(f"  {'stage':14s}  {'exists':>10}  {'flagged':>7}  {'total_rows':>12}  "
          f"{'min/p50/max':>22}")
    for s, b in by_stage.items():
        rs = sorted(b["rows"]) or [0]
        med = rs[len(rs) // 2] if rs else 0
        mm = f"{rs[0]}/{med}/{rs[-1]}"
        print(f"  {s:14s}  {b['exists']:>3}/{b['total']:<6}  {b['flagged']:>7}  "
              f"{sum(rs):>12}  {mm:>22}")


def section_stage_d(args) -> None:
    print(hr("STAGE D — DML estimates per variant"))
    try:
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError as e:
        print(f"(skipped: {e})")
        return
    for v in ACTIVE_VARIANTS:
        p = DATA_ROOT / "data" / "dml_results" / f"dml_results_long_{v}.parquet"
        if not p.exists():
            print(f"\n  [{v}] MISSING: {p}")
            continue
        df = pq.read_table(p).to_pandas()
        print(f"\n  ── variant={v}  rows={len(df)} ──")
        print(f"  columns: {list(df.columns)}")
        # Build a tight summary if standard cols are present
        std_cols = [c for c in ("treatment", "estimator", "theta", "se",
                                "ci_lower", "ci_upper", "pvalue", "n")
                    if c in df.columns]
        if {"treatment", "theta"}.issubset(df.columns):
            grp_cols = [c for c in ("treatment", "estimator") if c in df.columns]
            agg = {"theta": ["count", "mean", "std", "min", "max"]}
            if "se" in df.columns:
                agg["se"] = "mean"
            if "pvalue" in df.columns:
                agg["pvalue"] = "median"
            summary = df.groupby(grp_cols).agg(agg).round(4)
            summary.columns = ["_".join(c).strip("_") for c in summary.columns]
            print("\n  per-treatment summary:")
            print(summary.to_string())
        if std_cols:
            head = df[std_cols].head(20)
            print(f"\n  first 20 rows (cols={std_cols}):")
            print(head.round(4).to_string(index=False))
        else:
            print("\n  first 10 rows (raw):")
            print(df.head(10).to_string(index=False))


def section_ablation(args) -> None:
    print(hr("STAGE F — ablation effect per (treatment × variant × model)"))
    try:
        import pandas as pd
    except ImportError as e:
        print(f"(skipped: {e})")
        return
    rows = []
    for v in ACTIVE_VARIANTS:
        for m in MODELS_SHORT:
            for t in TREATMENTS:
                csv = INTERP_OUT / f"ablation_{t}_{m}_{v}" / "ablation_results_full.csv"
                if not csv.exists():
                    continue
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    continue
                row = {
                    "model": m.split("-")[0],
                    "variant": v,
                    "treatment": t.split("_")[0],
                    "n": len(df),
                }
                # surface common metric columns if present
                for col in ("delta_logit", "delta_logit_mean", "delta_rank",
                            "delta_rank_mean", "ablation_effect", "effect"):
                    if col in df.columns:
                        try:
                            row[col] = round(float(df[col].mean()), 4)
                        except Exception:
                            pass
                rows.append(row)
    if not rows:
        print("(no ablation CSVs found)")
        return
    df = pd.DataFrame(rows).sort_values(["treatment", "variant", "model"])
    print(df.to_string(index=False))


def section_saliency(args) -> None:
    print(hr("STAGE F — saliency summary tables"))
    try:
        import pandas as pd
    except ImportError as e:
        print(f"(skipped: {e})")
        return
    for v in ACTIVE_VARIANTS:
        for m in MODELS_SHORT:
            for frame_name, suffix in FRAMES:
                csv = INTERP_OUT / f"saliency_{m}_{v}" / f"saliency_summary{suffix}.csv"
                if not csv.exists():
                    continue
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    continue
                print(f"\n  [{v} / {m.split('-')[0]} / frame={frame_name}]  rows={len(df)}")
                if len(df):
                    print(df.to_string(index=False))


def section_weights(args) -> None:
    print(hr("STAGE F — weights (logit lens) summary"))
    try:
        import pandas as pd
    except ImportError as e:
        print(f"(skipped: {e})")
        return
    for v in ACTIVE_VARIANTS:
        for m in MODELS_SHORT:
            csv = INTERP_OUT / f"weights_{m}_{v}" / "logit_lens.csv"
            if not csv.exists():
                print(f"  [{v} / {m.split('-')[0]}] MISSING")
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as e:
                print(f"  [{v} / {m.split('-')[0]}] READ ERROR: {e}")
                continue
            print(f"\n  ── [{v} / {m.split('-')[0]}]  rows={len(df)}  cols={list(df.columns)[:8]}")
            num = df.select_dtypes("number")
            if len(num.columns):
                desc = num.describe().round(4)
                # keep it tight — only show count/mean/std/min/max
                keep = [r for r in ("count", "mean", "std", "min", "max") if r in desc.index]
                print(desc.loc[keep].to_string())


def section_probing(args) -> None:
    print(hr("STAGE F — probing status"))
    print(f"  {'variant':14s}  {'model':24s}  {'status':<8}  {'size':>8}")
    for v in ACTIVE_VARIANTS:
        for m in MODELS_SHORT:
            csv = INTERP_OUT / f"probing_{m}_{v}" / "probing_results.csv"
            if csv.exists():
                size = csv.stat().st_size
                status = "EXISTS" if size > 0 else "EMPTY"
            else:
                size, status = 0, "MISSING"
            print(f"  {v:14s}  {m.split('-')[0]:24s}  {status:<8}  {size:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────


SECTIONS = [
    ("header",    section_header),
    ("jobs",      section_jobs),
    ("audit",     section_audit),
    ("stage_d",   section_stage_d),
    ("ablation",  section_ablation),
    ("saliency",  section_saliency),
    ("weights",   section_weights),
    ("probing",   section_probing),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", nargs="*", default=["487548"],
                    help="Slurm job IDs to inspect in detail (default: 487548)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=[name for name, _ in SECTIONS],
                    help="Sections to skip")
    args = ap.parse_args()

    for name, fn in SECTIONS:
        if name in args.skip:
            continue
        try:
            fn(args)
        except Exception:
            print(f"\n[{name}] ERROR:")
            traceback.print_exc(limit=3, file=sys.stdout)
    print(hr("END"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
