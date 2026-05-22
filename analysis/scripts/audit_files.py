#!/usr/bin/env python3
"""Per-file inventory for every expected GEODML output.

Where audit_pipeline.py summarises by cell, this walks the canonical path of
every expected artifact across the 4 active variants and prints one line per
file with: existence, size, rows, mtime, and an optional FLAG.

FLAGS surfaced:
  CAP?    keyword count is in [950..1050] (likely capped at ~1000)
  LOW     rows < 50% of the median for its (stage, variant) cohort
  STALE   exists but mtime older than the newest peer by > 48h
  EMPTY   exists but 0 bytes / 0 rows

Examples:
  python scripts/audit_files.py                       # full inventory
  python scripts/audit_files.py --only stage_a        # one stage
  python scripts/audit_files.py --missing-only        # only files that don't exist
  python scripts/audit_files.py --flags-only          # only files with a FLAG
  python scripts/audit_files.py --variants biased,neutral
  python scripts/audit_files.py --tsv                 # machine-readable

Reads paths from interpretability.pipeline.config so it stays in sync with
the pipeline (no hardcoded directory names).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from interpretability.pipeline import config as C  # noqa: E402

ACTIVE_VARIANTS = ["biased", "neutral", "biased_rag", "neutral_rag"]
MODELS = ["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
ENGINES = ["searxng", "ddg"]
POOLS = [(20, 10), (50, 10)]
SEEDS = [42, 123]
TREATMENTS = [
    "T7_source_earned", "T5_topical_comp", "T3_structured_data_new",
    "T2a_question_headings", "T6_freshness", "T1b_stats_density",
]
FRAMES = ["full", "robust_winners"]
FRAME_SUFFIX = {"full": "_full", "robust_winners": "_rw"}

DATA_ROOT = Path(os.environ.get("GEODML_DATA_ROOT", str(C.DEFAULT_DATA_ROOT)))
INTERP_OUT = REPO_ROOT / "interpretability" / "output"

GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
DIM = "\033[2m"; BOLD = "\033[1m"; RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def c(s: str, color: str) -> str:
    return f"{color}{s}{RESET}" if USE_COLOR else s


def size_h(n: int) -> str:
    if n <= 0:
        return "-"
    for u in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}T"


def jsonl_rows(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return -1


def csv_rows(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        with p.open("rb") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except OSError:
        return -1


def parquet_rows(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(str(p)).num_rows
    except Exception:
        try:
            import pyarrow.parquet as pq
            return pq.ParquetFile(str(p)).metadata.num_rows
        except Exception:
            return -1


def mtime_age(p: Path) -> tuple[float, str]:
    """Returns (mtime_epoch, 'Nh'/'Nd' string). 0 if missing."""
    if not p.exists():
        return 0.0, "-"
    m = p.stat().st_mtime
    age = time.time() - m
    if age < 3600:
        return m, f"{age / 60:.0f}m"
    if age < 86400:
        return m, f"{age / 3600:.1f}h"
    return m, f"{age / 86400:.1f}d"


class Row:
    __slots__ = ("stage", "variant", "label", "path", "rows", "size", "mtime", "age", "flags")

    def __init__(self, stage, variant, label, path, rows, size, mtime, age, flags=()):
        self.stage = stage
        self.variant = variant
        self.label = label
        self.path = path
        self.rows = rows
        self.size = size
        self.mtime = mtime
        self.age = age
        self.flags = list(flags)


# ── Path enumerators ──────────────────────────────────────────────────────────

def stage_a_rows() -> list[Row]:
    out = []
    for v in ACTIVE_VARIANTS:
        for model in MODELS:
            ms = C.short_model_name(model)
            for engine in ENGINES:
                for pool, topn in POOLS:
                    run = C.run_label_with_variant(engine, model, pool, topn, v)
                    p = DATA_ROOT / "data" / "runs" / run / "phase2" / "keywords.jsonl"
                    n = jsonl_rows(p)
                    size = p.stat().st_size if p.exists() else 0
                    mt, age = mtime_age(p)
                    out.append(Row("stage_a", v,
                                   f"{ms}/{engine}/serp{pool}",
                                   p, n, size, mt, age))
    return out


def stage_aprime_rows() -> list[Row]:
    out = []
    op_dir = DATA_ROOT / "data" / "order_probe"
    for v in ACTIVE_VARIANTS:
        for model in MODELS:
            ms = C.short_model_name(model)
            for engine in ENGINES:
                for pool, topn in POOLS:
                    run = C.run_label_with_variant(engine, model, pool, topn, v)
                    for seed in SEEDS:
                        p = op_dir / f"{run}_seed{seed}.jsonl"
                        n = jsonl_rows(p)
                        size = p.stat().st_size if p.exists() else 0
                        mt, age = mtime_age(p)
                        out.append(Row("stage_a_prime", v,
                                       f"{ms}/{engine}/serp{pool}/seed{seed}",
                                       p, n, size, mt, age))
    return out


def stage_b_rows() -> list[Row]:
    out = []
    feat = DATA_ROOT / "data" / "features"
    for engine in ENGINES:
        for pool, _ in POOLS:
            p = feat / f"features_{engine}_top{pool}.parquet"
            n = parquet_rows(p)
            size = p.stat().st_size if p.exists() else 0
            mt, age = mtime_age(p)
            out.append(Row("stage_b", "-",
                           f"{engine}/serp{pool}",
                           p, n, size, mt, age))
    return out


def stage_c_rows() -> list[Row]:
    out = []
    for v in ACTIVE_VARIANTS:
        p = C.main_table_path(v, DATA_ROOT)
        n = parquet_rows(p)
        size = p.stat().st_size if p.exists() else 0
        mt, age = mtime_age(p)
        out.append(Row("stage_c", v, v, p, n, size, mt, age))
    return out


def stage_d_rows() -> list[Row]:
    out = []
    for v in ACTIVE_VARIANTS:
        p = C.dml_results_path(v, DATA_ROOT)
        n = parquet_rows(p)
        size = p.stat().st_size if p.exists() else 0
        mt, age = mtime_age(p)
        out.append(Row("stage_d", v, v, p, n, size, mt, age))
    return out


def stage_f_rows() -> list[Row]:
    out = []
    for v in ACTIVE_VARIANTS:
        for model in MODELS:
            ms = C.short_model_name(model)
            # ablation: per-treatment dir
            for t in TREATMENTS:
                d = INTERP_OUT / f"ablation_{t}_{ms}_{v}"
                full = d / "ablation_results_full.csv"
                marker = d / f".done_{ms}_{t}"
                n = csv_rows(full)
                size = full.stat().st_size if full.exists() else 0
                mt, age = mtime_age(full)
                flags = ["DONE"] if marker.exists() else []
                out.append(Row("F-ablation", v,
                               f"{ms}/{t}", full, n, size, mt, age, flags))
            # saliency: per-frame summary CSV
            for f in FRAMES:
                d = INTERP_OUT / f"saliency_{ms}_{v}"
                p = d / f"saliency_summary{FRAME_SUFFIX[f]}.csv"
                marker = d / f".done_{ms}_{f}"
                n = csv_rows(p)
                size = p.stat().st_size if p.exists() else 0
                mt, age = mtime_age(p)
                flags = ["DONE"] if marker.exists() else []
                out.append(Row("F-saliency", v,
                               f"{ms}/{f}", p, n, size, mt, age, flags))
            # probing
            d = INTERP_OUT / f"probing_{ms}_{v}"
            p = d / "probing_results.csv"
            marker = d / f".done_{ms}"
            n = csv_rows(p)
            size = p.stat().st_size if p.exists() else 0
            mt, age = mtime_age(p)
            flags = ["DONE"] if marker.exists() else []
            out.append(Row("F-probing", v, ms, p, n, size, mt, age, flags))
            # weights
            d = INTERP_OUT / f"weights_{ms}_{v}"
            p = d / "logit_lens.csv"
            marker = d / f".done_{ms}"
            n = csv_rows(p)
            size = p.stat().st_size if p.exists() else 0
            mt, age = mtime_age(p)
            flags = ["DONE"] if marker.exists() else []
            out.append(Row("F-weights", v, ms, p, n, size, mt, age, flags))
    return out


STAGE_FNS = {
    "stage_a":       stage_a_rows,
    "stage_a_prime": stage_aprime_rows,
    "stage_b":       stage_b_rows,
    "stage_c":       stage_c_rows,
    "stage_d":       stage_d_rows,
    "stage_f":       stage_f_rows,
}


# ── Anomaly detection ─────────────────────────────────────────────────────────

CAP_RANGE = (950, 1050)  # rerank/order_probe MAX_KEYWORDS=1000 leaves cells here


def annotate(rows: list[Row]) -> None:
    """Add LOW / CAP? / STALE / EMPTY flags based on peer cohort."""
    by_cohort: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        by_cohort.setdefault((r.stage, r.variant), []).append(r)
    now = time.time()
    for (_stage, _v), bucket in by_cohort.items():
        existing = [r for r in bucket if r.path.exists() and r.rows > 0]
        if not existing:
            continue
        med = median(r.rows for r in existing)
        newest = max(r.mtime for r in existing) or now
        for r in bucket:
            if not r.path.exists():
                continue
            if r.rows == 0 or r.size == 0:
                r.flags.append("EMPTY")
                continue
            # CAP? — only meaningful for keyword-record stages
            if r.stage in ("stage_a", "stage_a_prime") and \
               CAP_RANGE[0] <= r.rows <= CAP_RANGE[1]:
                r.flags.append("CAP?")
            # LOW — much smaller than peers in same cohort
            if med >= 100 and r.rows < 0.5 * med:
                r.flags.append("LOW")
            # STALE — mtime older than newest peer by > 48h
            if newest - r.mtime > 48 * 3600 and r.rows < med:
                r.flags.append("STALE")


# ── Output ────────────────────────────────────────────────────────────────────

def status_tag(r: Row) -> str:
    if not r.path.exists():
        return c(" -- ", RED)
    if r.rows <= 0:
        return c("EMTY", RED)
    if "DONE" in r.flags:
        return c("DONE", GREEN)
    if any(f in r.flags for f in ("LOW", "CAP?", "STALE")):
        return c("WARN", YELLOW)
    return c(" OK ", GREEN)


def fmt_flags(flags: list[str]) -> str:
    if not flags:
        return ""
    color = {
        "DONE": GREEN, "CAP?": YELLOW, "LOW": YELLOW,
        "STALE": YELLOW, "EMPTY": RED,
    }
    return " ".join(c(f, color.get(f, DIM)) for f in flags)


def print_section(title: str, rows: list[Row], rel_to: Path | None = None) -> None:
    bar = "─" * 86
    print(f"\n{c(bar, DIM)}\n{c(title, BOLD)}\n{c(bar, DIM)}")
    print(f"  {'STATUS':4}  {'VARIANT':12} {'CELL':36} "
          f"{'ROWS':>8}  {'SIZE':>6}  {'AGE':>5}  FLAGS  PATH")
    for r in sorted(rows, key=lambda x: (x.variant, x.label)):
        path_disp = str(r.path)
        if rel_to:
            try:
                path_disp = str(r.path.relative_to(rel_to))
            except ValueError:
                pass
        print(f"  {status_tag(r)}  {r.variant:12} {r.label:36} "
              f"{(r.rows if r.rows >= 0 else '?'):>8}  "
              f"{size_h(r.size):>6}  {r.age:>5}  "
              f"{fmt_flags(r.flags):<22}  {path_disp}")


def print_tsv(rows: list[Row]) -> None:
    print("stage\tvariant\tcell\texists\trows\tsize_bytes\tmtime_epoch\tage\tflags\tpath")
    for r in rows:
        print("\t".join((
            r.stage, r.variant, r.label,
            "1" if r.path.exists() else "0",
            str(r.rows), str(r.size),
            f"{r.mtime:.0f}", r.age,
            ",".join(r.flags), str(r.path),
        )))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    global ACTIVE_VARIANTS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(STAGE_FNS.keys()) + ["all"], default="all",
                    help="Run only one stage's inventory.")
    ap.add_argument("--variants", default=",".join(ACTIVE_VARIANTS),
                    help="Comma-separated variants (default: 4 active).")
    ap.add_argument("--missing-only", action="store_true",
                    help="Only show files that don't exist.")
    ap.add_argument("--flags-only", action="store_true",
                    help="Only show files with at least one FLAG (excluding DONE).")
    ap.add_argument("--tsv", action="store_true",
                    help="Machine-readable TSV output (no formatting).")
    ap.add_argument("--no-rel", action="store_true",
                    help="Print full absolute paths instead of repo-relative.")
    args = ap.parse_args()

    ACTIVE_VARIANTS = [v.strip() for v in args.variants.split(",") if v.strip()]

    stages = list(STAGE_FNS.keys()) if args.only == "all" else [args.only]
    all_rows: list[Row] = []
    by_stage: dict[str, list[Row]] = {}
    for s in stages:
        rows = STAGE_FNS[s]()
        by_stage[s] = rows
        all_rows.extend(rows)

    annotate(all_rows)

    def keep(r: Row) -> bool:
        if args.missing_only:
            return not r.path.exists()
        if args.flags_only:
            return bool([f for f in r.flags if f != "DONE"])
        return True

    if args.tsv:
        print_tsv([r for r in all_rows if keep(r)])
        return 0

    # repo-relative paths only for INTERP_OUT; data files are outside the repo
    for s in stages:
        rows = [r for r in by_stage[s] if keep(r)]
        if not rows:
            continue
        rel = None if args.no_rel else None  # use absolute for clarity
        print_section(
            {
                "stage_a":       "Stage A — rerank keywords.jsonl",
                "stage_a_prime": "Stage A' — order_probe keywords.jsonl",
                "stage_b":       "Stage B — features parquets",
                "stage_c":       "Stage C — main table parquets",
                "stage_d":       "Stage D — DML result parquets",
                "stage_f":       "Stage F — interpretability outputs",
            }[s],
            rows, rel_to=rel,
        )

    # Summary
    bar = "─" * 86
    print(f"\n{c(bar, DIM)}\n{c('Summary', BOLD)}\n{c(bar, DIM)}")
    for s in stages:
        rows = by_stage[s]
        exists = sum(1 for r in rows if r.path.exists() and r.rows > 0)
        flagged = sum(1 for r in rows if any(f != "DONE" for f in r.flags))
        print(f"  {s:14}  exists={exists:>3d}/{len(rows):<3d}  flagged={flagged:>3d}")

    # Specific CAP? roll-up
    caps = [r for r in all_rows if "CAP?" in r.flags]
    if caps:
        print(f"\n  {c('Suspected ~1000 keyword cap', YELLOW)} (re-run with MAX_KW=0):")
        for r in caps:
            print(f"    {r.stage:14}  {r.variant:12} {r.label:36} rows={r.rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
