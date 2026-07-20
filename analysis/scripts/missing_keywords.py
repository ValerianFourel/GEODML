#!/usr/bin/env python3
"""Per-cell missing-keyword diagnostic + targeted-rerun manifest.

What this does:

  1. Walks the (variant × engine × pool × model) grid.
  2. For each cell, computes which canonical SERP keywords have NO LLM output.
  3. Cross-references with the rag_coverage table to flag the 267 keywords
     where RAG retrieval itself failed (the irreducible Layer-3 gap).
  4. Writes per-cell missing-keyword lists and a master manifest, both as
     parquet and as a human-readable markdown report.

Outputs:
  manifests/missing_keywords_per_cell.parquet  (one row per missing kw × cell)
  manifests/missing_keywords_summary.parquet   (one row per cell with counts)
  manifests/missing_keywords_per_cell/*.txt    (one file per cell, kw per line)
  docs/missing_keywords_report_<date>.md       (paper-ready summary)

What this does NOT do:
  - Submit any jobs. JUPITER's `scripts/repair_loop.sh` already does that via
    `rerank.py --resume`, which auto-fills the missing keywords per cell. The
    purpose of THIS script is to make the gap visible and to produce keyword
    lists that the operator can hand to a job (e.g. via `--keywords-file=…`
    once rerank.py grows that flag).

Usage:
  python scripts/missing_keywords.py                          # default paths
  python scripts/missing_keywords.py --data-root ~/geodml_data # override
  python scripts/missing_keywords.py --no-write               # report only
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent

# Honor $GEODML_DATA_ROOT (set in JUPITER .env). Fall back to ~/geodml_data
# only if the env var is unset (e.g. when running locally on the Mac).
DEFAULT_DATA_ROOT = Path(os.environ.get(
    "GEODML_DATA_ROOT",
    str(Path.home() / "geodml_data")))

VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
MODELS = ["Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct"]
ENGINES = ["ddg", "searxng"]
POOLS = [20, 50]


def load_canonical(serp_dir: Path) -> dict[tuple[str, int], set[str]]:
    canon: dict[tuple[str, int], set[str]] = {}
    for fn in sorted(serp_dir.glob("phase0_top*.parquet")):
        eng = "ddg" if "ddg" in fn.name else "searxng"
        pool = 20 if "top20" in fn.name else 50
        s = pq.read_table(fn, columns=["keyword"]).to_pandas()
        canon[(eng, pool)] = set(s["keyword"].dropna().unique())
    return canon


def load_no_rag(coverage_dir: Path) -> set[str]:
    """Read the rag_coverage table (built by bridge_dataset_gaps.py).

    Returns the set of keywords flagged `no_rag` (no RAG output anywhere).
    Empty set if the coverage table doesn't exist."""
    p = coverage_dir / "rag_coverage.parquet"
    if not p.exists():
        return set()
    rc = pq.read_table(p).to_pandas()
    return set(rc[rc["rag_coverage"] == "no_rag"]["keyword"])


def per_cell_gaps(main_dir: Path,
                  canon: dict[tuple[str, int], set[str]],
                  no_rag_kw: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-cell missing-keyword lists.

    Returns (long_df, summary_df):
      long_df    — one row per (variant, engine, pool, model, missing_keyword)
      summary_df — one row per cell with counts
    """
    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    for v in VARIANTS:
        p = main_dir / f"full_experiment_data_{v}.parquet"
        if not p.exists():
            print(f"  [warn] {p} missing — skipping variant {v}")
            continue
        d = pq.read_table(p).to_pandas()
        d = d[d["url"] != ""]   # drop empty-URL placeholders
        is_rag = v.endswith("_rag")

        for (e, pool) in sorted(canon):
            target = canon[(e, pool)]
            for m in MODELS:
                sub = d[(d["search_engine"] == e)
                        & (d["pool"] == pool)
                        & (d["llm_model"] == m)]
                actual = set(sub["keyword"].unique())
                missing = target - actual

                # Classify each missing keyword
                for kw in missing:
                    reason = "rag_retrieval_failed" if (is_rag and kw in no_rag_kw) \
                              else ("rerun_needed_rag" if is_rag
                                    else "rerun_needed")
                    long_rows.append({
                        "variant": v, "engine": e, "pool": pool,
                        "model": m.split("-")[0],
                        "keyword": kw,
                        "missing_reason": reason,
                    })

                summary_rows.append({
                    "variant": v, "engine": e, "pool": pool,
                    "model": m.split("-")[0],
                    "target": len(target),
                    "actual": len(actual),
                    "missing": len(missing),
                    "missing_rerun_needed":
                        len([kw for kw in missing
                             if not (is_rag and kw in no_rag_kw)]),
                    "missing_rag_failed":
                        len([kw for kw in missing
                             if is_rag and kw in no_rag_kw]),
                    "coverage_pct": round(len(actual) / len(target) * 100, 1),
                })

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def write_per_cell_files(long_df: pd.DataFrame, out_dir: Path) -> None:
    """One plain-text file per cell, listing missing keywords (one per line).

    Each cell's file can be passed as `--keywords-file` to rerank.py once
    that flag exists; today, just useful for spot-checks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (v, e, pool, m), grp in long_df.groupby(
            ["variant", "engine", "pool", "model"]):
        fn = out_dir / f"{v}__{e}_pool{pool}_{m}.txt"
        kws = sorted(grp["keyword"].tolist())
        fn.write_text("\n".join(kws) + "\n")


def write_markdown(summary: pd.DataFrame, no_rag_kw: set[str],
                   partial_rag_kw: set[str], out_md: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append(f"# Missing-keyword audit\n_Generated: {now}_\n")

    # Roll-up
    roll = (summary.groupby("variant")
            .agg(target=("target", "sum"),
                 actual=("actual", "sum"),
                 missing=("missing", "sum"),
                 rerun_needed=("missing_rerun_needed", "sum"),
                 rag_failed=("missing_rag_failed", "sum"))
            .reset_index())
    roll["cov%"] = (roll["actual"] / roll["target"] * 100).round(1)
    lines.append("## 1. Per-variant roll-up (sum across 8 cells)\n")
    lines.append(_md_table(roll, ["variant", "target", "actual",
                                  "missing", "rerun_needed",
                                  "rag_failed", "cov%"]))

    # Worst cells
    lines.append("## 2. Worst-covered cells (top 12)\n")
    worst = summary.sort_values("coverage_pct").head(12)
    lines.append(_md_table(worst, ["variant", "engine", "pool", "model",
                                   "actual", "target", "missing",
                                   "coverage_pct"]))

    # Layer-3 gaps
    lines.append("## 3. Layer-3 (irreducible) RAG-retrieval failures\n")
    lines.append(f"**{len(no_rag_kw)} keywords** have NO RAG output in any "
                 f"of the 16 RAG cells. Fix: rebuild `rag_index` via "
                 f"`scripts/run_rag_embeddings.sh` (needs OPENAI_API_KEY).\n\n")
    lines.append(f"**{len(partial_rag_kw)} keywords** are partial-RAG "
                 f"(present in one prompt variant but not the other). "
                 f"Re-running `rerank.py --resume --variant neutral_rag` "
                 f"on the affected cells should close most of this.\n\n")
    lines.append("First 20 no-RAG keywords:\n\n```\n"
                 + "\n".join(sorted(no_rag_kw)[:20]) + "\n```\n")

    # Per-cell summary
    lines.append("## 4. Full per-cell summary (32 cells)\n")
    summary_sorted = summary.sort_values(["variant", "engine", "pool",
                                          "model"])
    lines.append(_md_table(summary_sorted,
                           ["variant", "engine", "pool", "model",
                            "actual", "target", "missing",
                            "missing_rerun_needed",
                            "missing_rag_failed",
                            "coverage_pct"]))

    # What to do
    lines.append("## 5. What to do next\n")
    lines.append(
        "On JUPITER, in a fresh shell:\n\n"
        "```bash\n"
        "set -a; source .env; set +a\n"
        "# the existing repair pipeline auto-fills missing keywords via\n"
        "# rerank.py --resume + sbatch dispatch. one cycle:\n"
        "./scripts/repair_run.sh\n"
        "# or run-until-done with downstream chain:\n"
        "./scripts/repair_loop.sh --with-downstream\n"
        "```\n\n"
        "For the 267 Layer-3 no-RAG keywords, the rag_index itself is "
        "incomplete. Run **on a machine with internet + OpenAI key** "
        "(not JUPITER):\n\n"
        "```bash\nOPENAI_API_KEY=sk-... bash scripts/run_rag_embeddings.sh\n"
        "```\n\n"
        "After that, re-run the rerank pipeline on JUPITER so the LLM "
        "sees the freshly-indexed passages.\n"
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if not len(df):
        return "_(empty)_\n"
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols)
                     + " |")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="Base data path (default: $GEODML_DATA_ROOT or "
                         "~/geodml_data)")
    ap.add_argument("--serp-dir", default=None,
                    help="SERP pool dir (default: {data-root}/data/serp/, "
                         "falls back to repo-local mirror if missing)")
    ap.add_argument("--no-write", action="store_true",
                    help="Print summary, write nothing")
    args = ap.parse_args()

    root = Path(args.data_root)
    main_dir = root / "data" / "main"
    cov_dir  = root / "data" / "coverage"
    if args.serp_dir:
        serp_dir = Path(args.serp_dir)
    else:
        # Prefer the data-root's serp/ dir (JUPITER layout). Fall back to the
        # repo-local mirror (local Mac dev layout).
        candidates = [root / "data" / "serp",
                      REPO_ROOT / "geodml_data" / "data" / "serp"]
        serp_dir = next((p for p in candidates if p.exists()),
                        candidates[0])

    print(f"DATA_ROOT : {root}")
    print(f"MAIN      : {main_dir}")
    print(f"COVERAGE  : {cov_dir}")
    print(f"SERP      : {serp_dir}\n")

    if not main_dir.exists():
        sys.exit(f"[FATAL] {main_dir} doesn't exist")

    canon = load_canonical(serp_dir)
    if not canon:
        sys.exit(f"[FATAL] No SERP pool files in {serp_dir}")
    no_rag = load_no_rag(cov_dir)
    print(f"Canonical (engine, pool) → keyword counts:")
    for (e, p), s in sorted(canon.items()):
        print(f"  {e:8s} top{p}: {len(s):>4}")
    print(f"\nLayer-3 no-RAG keywords: {len(no_rag)}\n")

    long_df, summary = per_cell_gaps(main_dir, canon, no_rag)

    print("=" * 72)
    print("ROLL-UP per variant")
    print("=" * 72)
    roll = (summary.groupby("variant")
            .agg(target=("target", "sum"),
                 actual=("actual", "sum"),
                 missing=("missing", "sum"),
                 rerun_needed=("missing_rerun_needed", "sum"),
                 rag_failed=("missing_rag_failed", "sum"))
            .reset_index())
    roll["cov%"] = (roll["actual"] / roll["target"] * 100).round(1)
    print(roll.to_string(index=False))

    # Identify partial-RAG keywords (in one rag-variant but not the other)
    if (cov_dir / "rag_coverage.parquet").exists():
        rc = pq.read_table(cov_dir / "rag_coverage.parquet").to_pandas()
        partial_kw = set(rc[rc["rag_coverage"] == "partial_rag"]["keyword"])
    else:
        partial_kw = set()

    if args.no_write:
        return

    manifests = REPO_ROOT / "manifests"
    manifests.mkdir(exist_ok=True)
    long_df.to_parquet(manifests / "missing_keywords_per_cell.parquet")
    summary.to_parquet(manifests / "missing_keywords_summary.parquet")
    write_per_cell_files(long_df,
                         manifests / "missing_keywords_per_cell")

    md_out = REPO_ROOT / "docs" / \
        f"missing_keywords_report_{datetime.now():%Y-%m-%d}.md"
    write_markdown(summary, no_rag, partial_kw, md_out)

    print(f"\nWrote:")
    print(f"  manifests/missing_keywords_per_cell.parquet  ({len(long_df):,} rows)")
    print(f"  manifests/missing_keywords_summary.parquet   ({len(summary)} rows)")
    print(f"  manifests/missing_keywords_per_cell/         ({len(summary)} txt files)")
    print(f"  {md_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
