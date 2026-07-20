#!/usr/bin/env python3
"""Human-readable audit report generated from manifests/repair_manifest.parquet.

Reads the manifest written by `scripts/repair_audit.py` and emits a single
markdown document listing every cell, its gap, and a short rollup at the top.

Designed to answer the question "what's still missing to have all 1011
keywords across every (variant × engine × pool × model [× seed])?".

Usage:
  python scripts/repair_report.py                          # write docs/repair_report_<date>.md
  python scripts/repair_report.py --out FILE.md            # custom output path
  python scripts/repair_report.py --stage rerank           # filter to one stage
  python scripts/repair_report.py --status                 # also print summary to stdout
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "manifests" / "repair_manifest.parquet"


def load() -> pd.DataFrame:
    if not MANIFEST.exists():
        sys.exit(f"[FATAL] manifest not found: {MANIFEST}\n"
                 f"Run scripts/repair_audit.py first.")
    return pq.read_table(MANIFEST).to_pandas()


def fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    if not len(df):
        return "_(none)_\n"
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write_report(df: pd.DataFrame, out_path: Path, stage_filter: str | None) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"# Repair audit — what's missing for full 1011-keyword coverage\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append(f"_Source manifest: `manifests/repair_manifest.parquet` "
                 f"(rows={len(df)})_\n")
    lines.append(f"_DATA_ROOT used by last audit: "
                 f"`{__import__('os').environ.get('GEODML_DATA_ROOT', '~/geodml_data')}`_\n")
    lines.append("\n> **Note:** numbers reflect whatever mirror the audit was "
                 "pointed at. Run `repair_audit.py` on JUPITER for the "
                 "authoritative cluster-side view.\n")

    if stage_filter:
        df = df[df["stage"] == stage_filter]

    # ── 1. Top-line rollup ──
    lines.append("## 1. Rollup\n")
    roll = (df.groupby("stage")
              .agg(cells=("run_id", "count"),
                   done=("gap", lambda s: int((s == 0).sum())),
                   partial=("gap", lambda s: int(((s > 0) &
                            (df.loc[s.index, "actual_kw"] > 0)).sum())),
                   empty=("actual_kw", lambda s: int((s == 0).sum())),
                   total_kw_gap=("gap", "sum"))
              .reset_index())
    lines.append(fmt_table(roll, ["stage", "cells", "done", "partial", "empty",
                                   "total_kw_gap"]))

    # ── 2. Status breakdown ──
    lines.append("## 2. Status (last dispatch state)\n")
    stat = (df.groupby(["stage", "status"]).size().unstack(fill_value=0)
              .reset_index())
    lines.append(fmt_table(stat, list(stat.columns)))

    # ── 3. Per-stage detail ──
    for stage in sorted(df["stage"].unique()):
        sub = df[df["stage"] == stage].copy()
        n_open = int((sub["gap"] > 0).sum())
        lines.append(f"## 3. Stage `{stage}` — {n_open} cells with gap > 0\n")

        if n_open == 0:
            lines.append("All cells at target. _Nothing to do for this stage._\n")
            continue

        sub = sub[sub["gap"] > 0].sort_values("gap", ascending=False)
        cols = ["variant", "engine", "pool", "model", "seed",
                "actual_kw", "target_kw", "gap", "status", "last_jobid"]
        # compact model name
        sub["model"] = sub["model"].astype(str).str.split("/").str[-1]
        for c in ("pool", "seed"):
            if c in sub.columns:
                sub[c] = sub[c].where(sub[c].notna(), "").astype(str)\
                               .str.replace(r"\.0$", "", regex=True)
        cols = [c for c in cols if c in sub.columns]
        lines.append(fmt_table(sub, cols))

    # ── 4. RAG-specific summary (the user's primary concern) ──
    rag = df[df["variant"].astype(str).str.endswith("_rag")
             & (df["gap"] > 0)]
    if len(rag):
        lines.append("## 4. RAG-variant gap (biased_rag + neutral_rag)\n")
        lines.append(f"**{len(rag)} cells** with missing RAG output. "
                     f"Total keyword-gap = **{int(rag['gap'].sum()):,}**.\n")
        agg = (rag.groupby("variant")["gap"]
                  .agg(["sum", "count"])
                  .rename(columns={"sum": "total_kw_gap",
                                   "count": "cells_with_gap"})
                  .reset_index())
        lines.append(fmt_table(agg, ["variant", "cells_with_gap", "total_kw_gap"]))

    # ── 5. What to do next ──
    lines.append("## 5. Next actions\n")
    lines.append(
        "Run on JUPITER:\n\n"
        "```bash\n"
        "set -a; source .env; set +a\n"
        "# Foreground loop (Ctrl-C safe, resumable):\n"
        "./scripts/repair_loop.sh\n"
        "# Or one-shot (re-invoke as needed):\n"
        "./scripts/repair_run.sh\n"
        "```\n\n"
        "After Stage A/A'/probing cells all reach `gap = 0`, re-derive "
        "downstream:\n\n"
        "```bash\n"
        "./scripts/slurm/dispatch_bcd.sh --with-stage-f\n"
        "```\n"
    )
    if any(df["variant"].astype(str).str.endswith("_rag")
           & (df["actual_kw"] == 0)):
        lines.append(
            "\n**Note on RAG cells at 0 keywords:** the underlying rag_index "
            "must be built first (or the rerank job will produce empty "
            "passages). On a machine with internet + an OpenAI key:\n\n"
            "```bash\nOPENAI_API_KEY=sk-... bash scripts/run_rag_embeddings.sh\n"
            "```\n"
        )

    out_path.write_text("\n".join(lines))
    print(f"Wrote → {out_path.relative_to(REPO_ROOT)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Output markdown path. Default: docs/repair_report_<date>.md")
    ap.add_argument("--stage", choices=["rerank", "order_probe", "probing"],
                    default=None, help="Restrict the report to one stage")
    ap.add_argument("--status", action="store_true",
                    help="Also print the short summary to stdout")
    args = ap.parse_args()

    df = load()
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = REPO_ROOT / "docs" / f"repair_report_{datetime.now():%Y-%m-%d}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(df, out_path, args.stage)

    if args.status:
        print()
        for stage in sorted(df["stage"].unique()):
            sub = df[df["stage"] == stage]
            n_open = int((sub["gap"] > 0).sum())
            n_total = len(sub)
            n_done = n_total - n_open
            gap = int(sub["gap"].sum())
            print(f"  {stage:12s} done={n_done}/{n_total}  open={n_open}  "
                  f"kw_gap={gap:,}")


if __name__ == "__main__":
    main()
