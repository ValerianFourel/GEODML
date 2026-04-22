#!/usr/bin/env python3
"""Consolidate FULL DATA from all experiment runs into a single folder.

Creates a structured archive directory containing:
  - Phase 2 data: rankings, features, HTML metadata per run
  - Phase 3 data: LLM-extracted features per run
  - Final results: analysis outputs, cross-model analysis, merged datasets

Usage:
  python paperSizeExperiment/consolidate_results.py
  python paperSizeExperiment/consolidate_results.py --dest /path/to/archive
  python paperSizeExperiment/consolidate_results.py --no-html-cache   # skip large HTML caches
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "output"
DEFAULT_DEST = SCRIPT_DIR / "consolidated_results"


def copy_file(src: Path, dst: Path, label: str = ""):
    """Copy a file, creating parent dirs as needed. Returns True if copied."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = src.stat().st_size / (1024 * 1024)
    tag = f" [{label}]" if label else ""
    print(f"  {dst.relative_to(dst.parent.parent.parent)}{tag} ({size_mb:.1f} MB)")
    return True


def copy_dir(src: Path, dst: Path, label: str = "", skip_existing: bool = False):
    """Copy an entire directory tree. Returns True if copied (or skipped).

    When skip_existing and dst already exists with matching file count, skip the copy.
    """
    if not src.exists() or not src.is_dir():
        return False
    src_files = sum(1 for _ in src.rglob("*") if _.is_file())
    if skip_existing and dst.exists():
        dst_files = sum(1 for _ in dst.rglob("*") if _.is_file())
        if dst_files == src_files:
            tag = f" [{label}]" if label else ""
            print(f"  {dst.relative_to(dst.parent.parent.parent)}/{tag} ({dst_files} files, SKIPPED — already present)")
            return True
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    n_files = sum(1 for _ in dst.rglob("*") if _.is_file())
    tag = f" [{label}]" if label else ""
    print(f"  {dst.relative_to(dst.parent.parent.parent)}/{tag} ({n_files} files)")
    return True


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file (excluding header)."""
    try:
        with open(path) as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return -1


def discover_runs(output_root: Path) -> list[Path]:
    """Find all experiment run directories (exclude cross_model_analysis, etc.)."""
    skip = {"cross_model_analysis"}
    runs = []
    for d in sorted(output_root.iterdir()):
        if d.is_dir() and d.name not in skip and not d.name.startswith("."):
            # Must have at least rankings.csv or experiment.json to be a real run
            if (d / "rankings.csv").exists() or (d / "experiment.json").exists():
                runs.append(d)
    return runs


def consolidate(dest: Path, include_html_cache: bool = True,
                include_analysis: bool = True, skip_existing: bool = False):
    """Main consolidation logic."""
    if not OUTPUT_ROOT.exists():
        print(f"ERROR: Output root not found: {OUTPUT_ROOT}")
        sys.exit(1)

    runs = discover_runs(OUTPUT_ROOT)
    if not runs:
        print("ERROR: No experiment runs found.")
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nConsolidating {len(runs)} runs -> {dest}/")
    print(f"  include_html_cache={include_html_cache} include_analysis={include_analysis} skip_existing={skip_existing}\n")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(OUTPUT_ROOT),
        "include_analysis": include_analysis,
        "include_html_cache": include_html_cache,
        "runs": [],
        "phase0_files": {},
        "phase2_files": {},
        "phase3_files": {},
        "analysis_files": {},
    }

    # ── Phase 0: raw SERP JSONs (witness of LLM input) ─────────────────────
    # Phase 0 files are produced directly into the destination dir by
    # run_phase0_serp.py, so they already live here — just index them.
    print("── phase 0 (raw SERP / LLM input witness) ──")
    phase0_files = sorted(dest.glob("phase0_*.json"))
    for pf in phase0_files:
        try:
            with open(pf) as f:
                pdata = json.load(f)
            sr = pdata.get("serp_results", {})
            total = len(sr)
            ok = sum(1 for v in sr.values() if len(v.get("raw_results") or []) > 0)
        except Exception:
            total = ok = -1
        size_mb = pf.stat().st_size / (1024 * 1024)
        print(f"  {pf.name}  ({size_mb:.1f} MB, {total} keywords, {ok} with results)")
        manifest["phase0_files"][pf.name] = {
            "path": str(pf.relative_to(dest)),
            "size_bytes": pf.stat().st_size,
            "keywords": total,
            "keywords_with_results": ok,
        }
    print()

    # ── Per-run data ─────────────────────────────────────────────────────────

    for run_dir in runs:
        run_name = run_dir.name
        run_dest = dest / "runs" / run_name
        run_dest.mkdir(parents=True, exist_ok=True)

        print(f"── {run_name} ──")

        run_info = {"run_id": run_name, "phase2": {}, "phase3": {}, "analysis": {}}

        # -- Phase 2: SERP + HTML fetch --
        phase2_dest = run_dest / "phase2"
        phase2_dest.mkdir(parents=True, exist_ok=True)

        phase2_files = {
            "rankings.csv": "SERP rankings + LLM re-rankings",
            "features.csv": "Code-based features from HTML",
            "experiment.json": "Full experiment metadata + per-keyword results",
            "keywords.jsonl": "Keywords with search results",
            "progress.json": "Pipeline progress tracker",
        }

        for fname, desc in phase2_files.items():
            src = run_dir / fname
            if copy_file(src, phase2_dest / fname, desc):
                rows = count_csv_rows(src) if fname.endswith(".csv") else None
                run_info["phase2"][fname] = {
                    "rows": rows,
                    "size_bytes": src.stat().st_size,
                }

        # HTML cache (can be large)
        html_cache = run_dir / "html_cache"
        if html_cache.exists() and include_html_cache:
            copy_dir(html_cache, phase2_dest / "html_cache", "cached HTML pages",
                     skip_existing=skip_existing)
            n_html = sum(1 for _ in html_cache.rglob("*") if _.is_file())
            run_info["phase2"]["html_cache_files"] = n_html
        elif html_cache.exists():
            n_html = sum(1 for _ in html_cache.rglob("*") if _.is_file())
            print(f"  (skipping html_cache: {n_html} files)")
            run_info["phase2"]["html_cache_files"] = n_html
            run_info["phase2"]["html_cache_skipped"] = True

        # -- Phase 3: LLM-extracted features --
        phase3_dest = run_dest / "phase3"
        phase3_dest.mkdir(parents=True, exist_ok=True)

        phase3_files = {
            "features_new.csv": "LLM-extracted treatments + confounders",
            "features_new_moz.csv": "Features with Moz domain authority",
            "features_snapshot.csv": "Features snapshot (intermediate)",
            "features.csv.bak": "Features backup before re-extraction",
        }

        for fname, desc in phase3_files.items():
            src = run_dir / fname
            if copy_file(src, phase3_dest / fname, desc):
                rows = count_csv_rows(src) if fname.endswith(".csv") else None
                run_info["phase3"][fname] = {
                    "rows": rows,
                    "size_bytes": src.stat().st_size,
                }

        # -- Final merged dataset --
        geodml = run_dir / "geodml_dataset.csv"
        if copy_file(geodml, run_dest / "geodml_dataset.csv", "merged final dataset"):
            rows = count_csv_rows(geodml)
            run_info["final_dataset_rows"] = rows

        # -- Analysis results --
        # Collect all analysis directories (analysis, analysis_full, analysis_halo)
        if include_analysis:
            for analysis_dir_name in sorted(run_dir.iterdir()):
                if analysis_dir_name.is_dir() and analysis_dir_name.name.startswith("analysis"):
                    analysis_dest = run_dest / analysis_dir_name.name
                    if copy_dir(analysis_dir_name, analysis_dest, "DML analysis",
                                skip_existing=skip_existing):
                        run_info["analysis"][analysis_dir_name.name] = {
                            "files": [f.name for f in analysis_dir_name.iterdir() if f.is_file()]
                        }

        manifest["runs"].append(run_info)
        print()

    # ── Cross-model analysis ─────────────────────────────────────────────────

    if include_analysis:
        cross_model_src = OUTPUT_ROOT / "cross_model_analysis"
        if cross_model_src.exists():
            print("── cross_model_analysis ──")
            cross_dest = dest / "cross_model_analysis"
            copy_dir(cross_model_src, cross_dest, "cross-model DML results",
                     skip_existing=skip_existing)
            manifest["analysis_files"]["cross_model_analysis"] = {
                "files": [f.name for f in cross_model_src.iterdir() if f.is_file()]
            }
            print()

    # ── Top-level merged files ───────────────────────────────────────────────

    print("── merged datasets ──")
    merged_dest = dest / "merged"
    merged_dest.mkdir(parents=True, exist_ok=True)

    for pattern in ["merged_*.csv", "meta_analysis_summary.csv"]:
        for f in sorted(OUTPUT_ROOT.glob(pattern)):
            if copy_file(f, merged_dest / f.name, "merged dataset"):
                manifest["analysis_files"][f.name] = {
                    "rows": count_csv_rows(f),
                    "size_bytes": f.stat().st_size,
                }

    # Copy experiment manifest and tracker
    for fname in ["experiment_manifest.json", "tracker.json", "presentation_data.json"]:
        src = OUTPUT_ROOT / fname
        if src.exists():
            copy_file(src, dest / fname, "experiment metadata")

    # ── Write consolidation manifest ─────────────────────────────────────────

    manifest_path = dest / "consolidation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")

    # ── Summary ──────────────────────────────────────────────────────────────

    total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    total_files = sum(1 for f in dest.rglob("*") if f.is_file())

    print(f"\n{'='*60}")
    print(f"CONSOLIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Destination:  {dest}/")
    print(f"  Runs:         {len(runs)}")
    print(f"  Total files:  {total_files}")
    print(f"  Total size:   {total_size / (1024*1024):.1f} MB")
    print(f"\nDirectory structure:")
    print(f"  {dest.name}/")
    print(f"    runs/                        # per-run data")
    for run_dir in runs:
        print(f"      {run_dir.name}/")
        print(f"        phase2/                  # SERP + HTML fetch data")
        print(f"        phase3/                  # LLM-extracted features")
        print(f"        geodml_dataset.csv       # merged final dataset")
        print(f"        analysis*/               # DML results")
    print(f"    cross_model_analysis/        # cross-model DML results")
    print(f"    merged/                      # all-run merged CSVs")
    print(f"    consolidation_manifest.json  # this archive's metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate full experiment data into a single archive folder")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST),
                        help=f"Destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--no-html-cache", action="store_true",
                        help="Skip copying HTML cache directories (saves space)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip analysis* dirs and cross_model_analysis")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip copying dirs that already exist with same file count")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing destination before consolidating "
                             "(WARNING: also deletes phase0_*.json files in dest)")
    args = parser.parse_args()

    dest = Path(args.dest)

    if args.clean and dest.exists():
        print(f"Removing existing: {dest}/")
        shutil.rmtree(dest)

    consolidate(dest,
                include_html_cache=not args.no_html_cache,
                include_analysis=not args.no_analysis,
                skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
