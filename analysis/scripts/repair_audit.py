#!/usr/bin/env python3
"""Audit data-completeness gaps and write a repair manifest.

Walks the full (stage × variant × engine × pool × model [× seed]) grid and
computes, per cell:
  - target_kw  : the canonical keyword count from the SERP pool for that
                 (engine, pool_size) — i.e. how many keywords COULD have output
  - actual_kw  : the number of unique keywords with output in the relevant
                 keywords.jsonl or output file
  - gap        : target_kw − actual_kw

Stages audited:
  rerank       : data/runs/{run_id}/phase2/keywords.jsonl       per (variant×engine×pool×model)
  order_probe  : data/order_probe/{run_id}_seed{seed}.jsonl     per (variant×engine×pool×model×seed)
  probing      : interpretability/output/probing_{model}_{variant}/probing_results.csv
                                                                 per (variant×model)

For RAG variants the target is still the FULL canonical keyword set: a
keyword absent from a RAG run means RAG retrieval failed for it, and the
goal here is exactly to retry those keywords.

Writes:
  manifests/repair_manifest.parquet  (read by repair_dispatch.py)

Usage:
  python scripts/repair_audit.py
  python scripts/repair_audit.py --stage rerank
  python scripts/repair_audit.py --print-only      # show summary, don't write
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("GEODML_DATA_ROOT",
                                str(Path.home() / "geodml_data")))
INTERP_OUT = REPO_ROOT / "interpretability" / "output"
MANIFEST = REPO_ROOT / "manifests" / "repair_manifest.parquet"

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
MODELS = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
ENGINES = ["searxng", "ddg"]
POOLS = [20, 50]
SEEDS = [42, 123]

SERP_FILES = {
    ("ddg", 20): "phase0_top20_ddg.parquet",
    ("ddg", 50): "phase0_top50_ddg.parquet",
    ("searxng", 20): "phase0_top20_searxng.parquet",
    ("searxng", 50): "phase0_top50_searxng.parquet",
}


def canonical_keywords() -> dict[tuple[str, int], set[str]]:
    """Return {(engine, pool): set(keyword)} from the SERP pool files."""
    out = {}
    for (e, p), fname in SERP_FILES.items():
        path = DATA_ROOT / "data" / "serp" / fname
        if not path.exists():
            print(f"  [warn] SERP file missing: {path}")
            out[(e, p)] = set()
            continue
        df = pq.read_table(path, columns=["keyword"]).to_pandas()
        out[(e, p)] = set(df["keyword"].dropna().unique())
    return out


def jsonl_keywords(path: Path) -> set[str]:
    """Read unique 'keyword' values from a JSONL file."""
    if not path.exists():
        return set()
    out = set()
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kw = obj.get("keyword")
                if isinstance(kw, str):
                    out.add(kw)
    except OSError:
        pass
    return out


def csv_keyword_count(path: Path) -> int:
    """For probing_results.csv: return row count (we just need non-empty)."""
    if not path.exists():
        return 0
    try:
        # cheap row count without loading
        with path.open("rb") as f:
            return max(0, sum(1 for _ in f) - 1)  # subtract header
    except OSError:
        return 0


def run_id(engine: str, model: str, pool: int, variant: str, top_n: int = 10) -> str:
    return f"{engine}_{model.split('/')[-1]}_serp{pool}_top{top_n}_{variant}"


# ── audits per stage ────────────────────────────────────────────────────────

def audit_rerank(canon: dict) -> list[dict]:
    rows = []
    for v in VARIANTS:
        for m in MODELS:
            for e in ENGINES:
                for p in POOLS:
                    rid = run_id(e, m, p, v)
                    target_kw = canon.get((e, p), set())
                    actual = jsonl_keywords(
                        DATA_ROOT / "data" / "runs" / rid / "phase2" / "keywords.jsonl"
                    )
                    rows.append({
                        "stage": "rerank",
                        "variant": v,
                        "engine": e,
                        "pool": p,
                        "model": m,
                        "seed": None,
                        "run_id": rid,
                        "target_kw": len(target_kw),
                        "actual_kw": len(actual),
                        "gap": len(target_kw - actual),
                        "missing_kw_sample": ",".join(sorted(list(target_kw - actual))[:5]),
                    })
    return rows


def audit_order_probe(canon: dict) -> list[dict]:
    rows = []
    for v in VARIANTS:
        for m in MODELS:
            for e in ENGINES:
                for p in POOLS:
                    for s in SEEDS:
                        rid = run_id(e, m, p, v)
                        target_kw = canon.get((e, p), set())
                        actual = jsonl_keywords(
                            DATA_ROOT / "data" / "order_probe" / f"{rid}_seed{s}.jsonl"
                        )
                        rows.append({
                            "stage": "order_probe",
                            "variant": v,
                            "engine": e,
                            "pool": p,
                            "model": m,
                            "seed": s,
                            "run_id": f"{rid}_seed{s}",
                            "target_kw": len(target_kw),
                            "actual_kw": len(actual),
                            "gap": len(target_kw - actual),
                            "missing_kw_sample": ",".join(sorted(list(target_kw - actual))[:5]),
                        })
    return rows


def audit_probing() -> list[dict]:
    rows = []
    for v in VARIANTS:
        for m in MODELS:
            mt = m.split("/")[-1]
            outdir = INTERP_OUT / f"probing_{mt}_{v}"
            csv = outdir / "probing_results.csv"
            n = csv_keyword_count(csv)
            rows.append({
                "stage": "probing",
                "variant": v,
                "engine": None,
                "pool": None,
                "model": m,
                "seed": None,
                "run_id": f"probing_{mt}_{v}",
                "target_kw": 1,        # probing has no per-kw target; just exist+nonzero
                "actual_kw": int(n > 0),
                "gap": int(n == 0),
                "missing_kw_sample": "" if n > 0 else "ALL",
            })
    return rows


# ── manifest writer ─────────────────────────────────────────────────────────

# Circuit-breaker thresholds (env-overridable). These are what make the repair
# loop CONVERGE instead of resubmitting forever. Some cells can never reach
# gap == 0 — e.g. a RAG variant where retrieval genuinely fails for a handful
# of keywords (see audit_rerank's docstring). Without a give-up rule the loop
# resubmits them every cycle indefinitely.
#
#   STUCK_THRESHOLD       quarantine after this many submissions in a row that
#                         produced NO new keywords (resets whenever a cell makes
#                         progress). This is the primary, progress-aware brake.
#   MAX_SUBMITS_PER_CELL  absolute backstop: quarantine after this many total
#                         submissions regardless of progress, so even a cell
#                         that crawls forward one keyword at a time can't run
#                         unbounded.
# max(1, …): a threshold of 0 would mark an untried cell STUCK before it is ever
# submitted (0 >= 0), making the loop silently no-op and falsely report "converged".
STUCK_THRESHOLD = max(1, int(os.environ.get("REPAIR_STUCK_THRESHOLD", "2")))
MAX_SUBMITS_PER_CELL = max(1, int(os.environ.get("REPAIR_MAX_SUBMITS_PER_CELL", "8")))


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve status + progress counters from an existing manifest when
    re-auditing. Audit *measurements* (target_kw/actual_kw/gap) are recomputed
    fresh every time; bookkeeping (status, jobids, and the circuit-breaker
    counters submit_count / best_actual_kw / submits_since_progress) is carried
    forward so the loop can detect non-convergence and quarantine STUCK cells.

    Progress is measured against a persisted high-water mark (best_actual_kw),
    NOT against transient status: repair_dispatch resubmits a finished cell in
    the same cycle it detects completion, so a status like PENDING_RECHECK is
    usually gone before the next audit sees it. Comparing actual_kw to the
    high-water mark is race-free."""
    if MANIFEST.exists():
        old = pq.read_table(MANIFEST).to_pandas()
        keep = ["run_id", "status", "last_jobid", "last_submitted", "last_check",
                "submit_count", "best_actual_kw", "submits_since_progress"]
        keep = [c for c in keep if c in old.columns]
        new_df = new_df.merge(old[keep], on="run_id", how="left")

    # Defaults for any missing/NaN bookkeeping column. Covers both the
    # no-manifest case and older manifests that predate the counter columns.
    if "status" not in new_df.columns:
        new_df["status"] = "TODO"
    new_df["status"] = new_df["status"].fillna("TODO")
    for c in ("last_jobid", "last_submitted", "last_check"):
        if c not in new_df.columns:
            new_df[c] = ""
        new_df[c] = new_df[c].fillna("")
    if "submit_count" not in new_df.columns:
        new_df["submit_count"] = 0
    new_df["submit_count"] = new_df["submit_count"].fillna(0).astype(int)
    if "submits_since_progress" not in new_df.columns:
        new_df["submits_since_progress"] = 0
    new_df["submits_since_progress"] = (
        new_df["submits_since_progress"].fillna(0).astype(int))
    if "best_actual_kw" not in new_df.columns:
        new_df["best_actual_kw"] = new_df["actual_kw"]
    new_df["best_actual_kw"] = (
        new_df["best_actual_kw"].fillna(new_df["actual_kw"]).astype(int))

    # ── progress accounting (race-free) ─────────────────────────────────────
    # Any cell whose freshly-measured actual_kw beats its high-water mark made
    # progress: raise the mark and clear the no-progress streak.
    improved = new_df["actual_kw"] > new_df["best_actual_kw"]
    new_df.loc[improved, "best_actual_kw"] = new_df.loc[improved, "actual_kw"]
    new_df.loc[improved, "submits_since_progress"] = 0

    # ── status transitions ──────────────────────────────────────────────────
    # Quarantine cells that have stalled, but never override a cell that is
    # currently in flight (it may still improve before it finishes).
    not_inflight = ~new_df["status"].isin(["SUBMITTED", "RUNNING"])
    stuck = (
        not_inflight
        & (new_df["gap"] > 0)
        & (new_df["submit_count"] > 0)   # never quarantine a cell that was never tried
        & (
            (new_df["submits_since_progress"] >= STUCK_THRESHOLD)
            | (new_df["submit_count"] >= MAX_SUBMITS_PER_CELL)
        )
    )
    new_df.loc[stuck, "status"] = "STUCK"
    # gap == 0 always wins: a fully-covered cell is DONE no matter its history.
    new_df.loc[new_df["gap"] == 0, "status"] = "DONE"
    return new_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rerank", "order_probe", "probing", "all"],
                    default="all")
    ap.add_argument("--print-only", action="store_true",
                    help="Compute and print summary without writing the manifest")
    args = ap.parse_args()

    print(f"REPO_ROOT  : {REPO_ROOT}")
    print(f"DATA_ROOT  : {DATA_ROOT}")
    print(f"INTERP_OUT : {INTERP_OUT}")
    print()

    if not DATA_ROOT.exists():
        print(f"[FATAL] GEODML_DATA_ROOT does not exist: {DATA_ROOT}", file=sys.stderr)
        sys.exit(2)

    print("Loading canonical keyword sets from SERP pool files …")
    canon = canonical_keywords()
    for k, v in sorted(canon.items()):
        print(f"  {k[0]:8s} top{k[1]}: {len(v)} unique keywords")
    print()

    all_rows = []
    if args.stage in ("rerank", "all"):
        print("Auditing rerank cells …")
        all_rows.extend(audit_rerank(canon))
    if args.stage in ("order_probe", "all"):
        print("Auditing order_probe cells …")
        all_rows.extend(audit_order_probe(canon))
    if args.stage in ("probing", "all"):
        print("Auditing probing cells …")
        all_rows.extend(audit_probing())

    df = pd.DataFrame(all_rows)
    df = merge_with_existing(df)

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for stage in df["stage"].unique():
        sub = df[df["stage"] == stage]
        n_total = len(sub)
        n_done = int((sub["gap"] == 0).sum())
        n_partial = int(((sub["gap"] > 0) & (sub["actual_kw"] > 0)).sum())
        n_empty = int((sub["actual_kw"] == 0).sum())
        n_stuck = int((sub["status"] == "STUCK").sum())
        n_actionable = int(((sub["gap"] > 0)
                            & (~sub["status"].isin(["STUCK", "DONE"]))).sum())
        total_gap = int(sub["gap"].sum())
        print(f"\n  [{stage}]  cells={n_total}  done={n_done}  "
              f"partial={n_partial}  empty={n_empty}  stuck={n_stuck}  "
              f"actionable={n_actionable}  total_kw_gap={total_gap:,}")
        # print biggest gaps
        worst = sub[sub["gap"] > 0].sort_values("gap", ascending=False).head(10)
        if len(worst):
            print(f"    largest gaps:")
            for _, r in worst.iterrows():
                print(f"      {r['run_id']:60s} gap={int(r['gap']):>4} "
                      f"({int(r['actual_kw']):>4}/{int(r['target_kw']):>4})")

    # ── write manifest ──────────────────────────────────────────────────────
    if not args.print_only:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MANIFEST)
        print(f"\nWrote manifest → {MANIFEST.relative_to(REPO_ROOT)}  rows={len(df)}")

    # ── status counts ───────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("STATUS")
    print("=" * 88)
    print(df.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
