#!/usr/bin/env python3
"""Master orchestrator for the paper-size experiment.

Runs the full pipeline across all configured (engine, model, pool_size)
combinations, then merges results and runs cross-model DML analysis.

All progress is tracked in output/tracker.json — the experiment can be
interrupted and resumed safely. Completed runs/phases are skipped on restart.

Usage:
  # Full experiment (all models x all pool sizes)
  python paperSizeExperiment/run_experiment.py

  # Test with 3 keywords
  python paperSizeExperiment/run_experiment.py --keywords 3

  # Single model + pool size
  python paperSizeExperiment/run_experiment.py --models "meta-llama/Llama-3.3-70B-Instruct" --pool-sizes "20,10"

  # Skip data gathering (re-run analysis only)
  python paperSizeExperiment/run_experiment.py --skip-gather --skip-features

  # Skip analysis (data gathering only)
  python paperSizeExperiment/run_experiment.py --skip-analysis

  # Force re-run even if already completed
  python paperSizeExperiment/run_experiment.py --force

  # Check progress
  python paperSizeExperiment/experiment_tracker.py status
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    KEYWORDS_FILE, LLM_MODELS, POOL_SIZES, SEARCH_ENGINE,
    ENABLE_LLM_FEATURES, ENABLE_PAGERANK, ENABLE_WHOIS,
    MOZ_API_KEY, OUTPUT_ROOT, run_label, run_output_dir,
)
import experiment_tracker as tracker


def load_keywords(keywords_file: Path, limit: int = 0) -> list[str]:
    if not keywords_file.exists():
        print(f"Keywords file not found: {keywords_file}")
        sys.exit(1)
    with open(keywords_file) as f:
        keywords = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if limit > 0:
        keywords = keywords[:limit]
    return keywords


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_gather(run_id: str, keywords_file: Path, engine: str, model: str,
               serp_n: int, llm_top_n: int, output_dir: Path,
               keyword_limit: int = 0, extra_args: list[str] = None):
    """Run gather_data.py for one configuration."""
    tracker.update_phase(run_id, "gather", "running")

    gather_script = PROJECT_ROOT / "pipeline" / "gather_data.py"
    progress_file = output_dir / "progress.json"
    cmd = [
        sys.executable, str(gather_script),
        "--engine", engine,
        "--serp-results", str(serp_n),
        "--llm-top-n", str(llm_top_n),
        "--llm-model", model,
        "--keywords-file", str(keywords_file),
        "--output-dir", str(output_dir),
        "--progress-file", str(progress_file),
    ]
    if keyword_limit > 0:
        cmd += ["--keywords", str(keyword_limit)]
    if ENABLE_LLM_FEATURES:
        cmd.append("--llm-features")
    if ENABLE_PAGERANK:
        cmd.append("--pagerank")
    if ENABLE_WHOIS:
        cmd.append("--whois")
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"GATHER: {run_id}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    rc = result.returncode

    # Read back output files + progress file to update tracker
    _update_gather_stats(run_id, output_dir, progress_file)

    if rc == 0:
        tracker.update_phase(run_id, "gather", "completed")
        # Record output files
        for fname in ["experiment.json", "rankings.csv", "features.csv"]:
            fpath = output_dir / fname
            if fpath.exists():
                rows = None
                if fname.endswith(".csv"):
                    try:
                        with open(fpath) as f:
                            rows = sum(1 for _ in f) - 1  # minus header
                    except Exception:
                        pass
                tracker.record_output_file(run_id, fname, str(fpath), rows)
    else:
        tracker.update_phase(run_id, "gather", "failed")
        tracker.record_error(run_id, "gather", f"exit code {rc}")
        print(f"\nWARNING: gather_data.py exited with code {rc}")

    return rc


def _update_gather_stats(run_id: str, output_dir: Path, progress_file: Path = None):
    """Parse output files + progress.json to update tracker with actual progress."""
    # First try progress.json (most accurate, updated live)
    if progress_file and progress_file.exists():
        try:
            with open(progress_file) as f:
                prog = json.load(f)
            tracker.update_gather_progress(
                run_id,
                keywords_attempted=prog.get("phase1_keywords_done", 0),
                keywords_succeeded=prog.get("phase1_keywords_done", 0) - prog.get("phase1_keywords_failed", 0),
                keywords_failed=prog.get("phase1_keywords_failed", 0),
                urls_fetched=prog.get("phase2_urls_done", 0) - prog.get("phase2_urls_failed", 0),
                urls_failed=prog.get("phase2_urls_failed", 0),
                llm_errors=prog.get("phase1_llm_errors", 0),
                llm_fallbacks=prog.get("phase1_llm_fallbacks", 0),
            )
            if prog.get("phase1_current_keyword"):
                tracker.update_gather_progress(
                    run_id, last_keyword=prog["phase1_current_keyword"])
        except Exception:
            pass

    # Also parse experiment.json for keyword stats (authoritative on LLM details)
    json_path = output_dir / "experiment.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                exp = json.load(f)
            kw_results = exp.get("per_keyword_results", [])
            kw_failed = exp.get("failed_keywords", [])

            # Count LLM stats
            llm_calls = 0
            llm_errors = 0
            llm_fallbacks = 0
            for kw in kw_results:
                llm = kw.get("llm", {})
                llm_calls += 1
                if llm.get("error"):
                    llm_errors += 1
                if llm.get("used_fallback"):
                    llm_fallbacks += 1

            tracker.update_gather_progress(
                run_id,
                keywords_attempted=len(kw_results) + len(kw_failed),
                keywords_succeeded=len(kw_results),
                keywords_failed=len(kw_failed),
                llm_calls=llm_calls,
                llm_errors=llm_errors,
                llm_fallbacks=llm_fallbacks,
            )
            if kw_results:
                tracker.update_gather_progress(
                    run_id, last_keyword=kw_results[-1].get("query", ""))
        except Exception:
            pass

    # Parse features.csv for URL stats
    features_path = output_dir / "features.csv"
    if features_path.exists():
        try:
            import pandas as pd
            feat = pd.read_csv(features_path)
            urls_ok = feat["error"].isna().sum() if "error" in feat.columns else len(feat)
            urls_fail = len(feat) - urls_ok
            tracker.update_gather_progress(
                run_id,
                urls_fetched=int(urls_ok),
                urls_failed=int(urls_fail),
            )
        except Exception:
            pass


def run_extract_features(run_id: str, output_dir: Path, extra_args: list[str] = None):
    """Run extract_features.py for enhanced treatment/confounder extraction."""
    tracker.update_phase(run_id, "extract_features", "running")

    extract_script = PROJECT_ROOT / "pipeline" / "extract_features.py"
    experiment_json = output_dir / "experiment.json"

    if not experiment_json.exists():
        print(f"  Skipping extract_features: no experiment.json in {output_dir}")
        tracker.update_phase(run_id, "extract_features", "skipped")
        return 1

    cmd = [
        sys.executable, str(extract_script),
        "--experiment-json", str(experiment_json),
        "--html-cache-dir", str(output_dir / "html_cache"),
        "--existing-dataset", str(output_dir / "geodml_dataset.csv"),
        "--output-csv", str(output_dir / "features_new.csv"),
    ]
    if MOZ_API_KEY:
        cmd += ["--moz-api-key", MOZ_API_KEY]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  EXTRACT FEATURES: {output_dir.name}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    rc = result.returncode

    if rc == 0:
        tracker.update_phase(run_id, "extract_features", "completed")
        fpath = output_dir / "features_new.csv"
        if fpath.exists():
            try:
                with open(fpath) as f:
                    rows = sum(1 for _ in f) - 1
                tracker.record_output_file(run_id, "features_new.csv", str(fpath), rows)
                tracker.update_feature_progress(run_id, urls_processed=rows)
            except Exception:
                pass
    else:
        tracker.update_phase(run_id, "extract_features", "failed")
        tracker.record_error(run_id, "extract_features", f"exit code {rc}")

    return rc


def run_clean(run_id: str, output_dir: Path):
    """Run clean_data.py to merge rankings + features."""
    tracker.update_phase(run_id, "clean", "running")

    clean_script = PROJECT_ROOT / "pipeline" / "clean_data.py"
    cmd = [
        sys.executable, str(clean_script),
        "--input-dir", str(output_dir),
        "--output", str(output_dir / "geodml_dataset.csv"),
    ]
    features_new = output_dir / "features_new.csv"
    if features_new.exists():
        cmd += ["--new-features", str(features_new)]

    print(f"\n  CLEAN DATA: {output_dir.name}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    rc = result.returncode

    if rc == 0:
        tracker.update_phase(run_id, "clean", "completed")
        fpath = output_dir / "geodml_dataset.csv"
        if fpath.exists():
            try:
                with open(fpath) as f:
                    rows = sum(1 for _ in f) - 1
                tracker.record_output_file(run_id, "geodml_dataset.csv", str(fpath), rows)
            except Exception:
                pass
    else:
        tracker.update_phase(run_id, "clean", "failed")
        tracker.record_error(run_id, "clean", f"exit code {rc}")

    return rc


def run_analysis(run_id: str, output_dir: Path, results_dir: Path,
                 extra_args: list[str] = None):
    """Run analyze.py on merged dataset."""
    tracker.update_phase(run_id, "analyze", "running")

    analyze_script = PROJECT_ROOT / "pipeline" / "analyze.py"
    dataset = output_dir / "geodml_dataset.csv"

    if not dataset.exists():
        print(f"  Skipping analysis: no geodml_dataset.csv in {output_dir}")
        tracker.update_phase(run_id, "analyze", "skipped")
        return 1

    cmd = [
        sys.executable, str(analyze_script),
        "--input", str(dataset),
        "--output-dir", str(results_dir),
        "--outcome", "all",
        "--method", "plr",
        "--learner", "all",
        "--measurement", "all",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  ANALYZE: {output_dir.name} -> {results_dir.name}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    rc = result.returncode

    if rc == 0:
        tracker.update_phase(run_id, "analyze", "completed")
        # Record analysis outputs
        for fname in ["all_experiments.csv", "summary.json", "dml_coefficients.png"]:
            fpath = results_dir / fname
            if fpath.exists():
                tracker.record_output_file(run_id, f"analysis/{fname}", str(fpath))
    else:
        tracker.update_phase(run_id, "analyze", "failed")
        tracker.record_error(run_id, "analyze", f"exit code {rc}")

    return rc


def merge_all_datasets(output_root: Path, merged_path: Path):
    """Merge all per-run geodml_dataset.csv into one master CSV."""
    import pandas as pd

    all_dfs = []
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("cross_model_analysis",):
            continue
        dataset = run_dir / "geodml_dataset.csv"
        if not dataset.exists():
            continue
        df = pd.read_csv(dataset)
        df["run_id"] = run_dir.name
        for part in run_dir.name.split("_"):
            if part.startswith("serp"):
                df["serp_pool_size"] = int(part.replace("serp", ""))
            if part.startswith("top"):
                df["llm_pool_size"] = int(part.replace("top", ""))
        all_dfs.append(df)
        print(f"  Loaded {len(df)} rows from {run_dir.name}")

    if not all_dfs:
        print("No datasets found to merge.")
        return None

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(merged_path, index=False)
    print(f"\nMerged dataset: {len(merged)} rows from {len(all_dfs)} runs -> {merged_path}")
    print(f"  Keywords: {merged['keyword'].nunique()}")
    print(f"  Unique domains: {merged['domain'].nunique()}")
    if "llm_model" in merged.columns:
        print(f"  LLM models: {sorted(merged['llm_model'].dropna().unique())}")
    if "serp_pool_size" in merged.columns:
        print(f"  Pool sizes: {sorted(merged['serp_pool_size'].dropna().unique())}")

    return merged


def run_cross_model_analysis(merged_csv: Path, results_dir: Path):
    """Run analysis on the merged cross-model dataset."""
    analyze_script = SCRIPT_DIR / "analyze_cross_model.py"
    if not analyze_script.exists():
        analyze_script = PROJECT_ROOT / "pipeline" / "analyze.py"

    cmd = [
        sys.executable, str(analyze_script),
        "--input", str(merged_csv),
        "--output-dir", str(results_dir),
    ]
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL ANALYSIS")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper-size experiment: multi-model, multi-pool DML pipeline")
    parser.add_argument("--keywords", type=int, default=0,
                        help="Limit to first N keywords (0=all)")
    parser.add_argument("--keywords-file", type=str, default=str(KEYWORDS_FILE),
                        help=f"Keywords file (default: {KEYWORDS_FILE})")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model IDs (overrides config.py)")
    parser.add_argument("--pool-sizes", type=str, default=None,
                        help="Comma-separated pool sizes as 'serp,topN' pairs (e.g. '20,10;50,10')")
    parser.add_argument("--engine", type=str, default=SEARCH_ENGINE,
                        help=f"Search engine (default: {SEARCH_ENGINE})")
    parser.add_argument("--skip-gather", action="store_true",
                        help="Skip data gathering (use existing data)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip DML analysis")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merging and cross-model analysis")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if already completed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan without running")
    args = parser.parse_args()

    # ── Parse experiment grid ─────────────────────────────────────────────
    models = LLM_MODELS
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    pool_sizes = POOL_SIZES
    if args.pool_sizes:
        pool_sizes = []
        for pair in args.pool_sizes.split(";"):
            parts = pair.strip().split(",")
            pool_sizes.append((int(parts[0]), int(parts[1])))

    engine = args.engine
    keywords_file = Path(args.keywords_file)
    keywords = load_keywords(keywords_file, args.keywords)

    # ── Print experiment plan ─────────────────────────────────────────────
    total_runs = len(models) * len(pool_sizes)
    print(f"\n{'='*70}")
    print(f"PAPER-SIZE EXPERIMENT")
    print(f"{'='*70}")
    print(f"  Keywords:     {len(keywords)} (from {keywords_file.name})")
    print(f"  Engine:       {engine}")
    print(f"  LLM models:   {len(models)}")
    for m in models:
        print(f"    - {m}")
    print(f"  Pool sizes:   {len(pool_sizes)}")
    for serp_n, top_n in pool_sizes:
        print(f"    - SERP={serp_n} -> Top-{top_n}")
    print(f"  Total runs:   {total_runs}")
    print(f"  Output:       {OUTPUT_ROOT}/")
    print(f"{'='*70}")

    if args.dry_run:
        print("\n  DRY RUN — experiment plan printed above. Exiting.")
        for model in models:
            for serp_n, top_n in pool_sizes:
                label = run_label(engine, model, serp_n, top_n)
                out_dir = run_output_dir(engine, model, serp_n, top_n)
                # Check if already done
                done = tracker.is_run_complete(label)
                tag = " [DONE]" if done else ""
                print(f"    {label} -> {out_dir}{tag}")
        return

    # ── Initialize tracker ────────────────────────────────────────────────
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tracker.init_experiment(models, pool_sizes, engine, len(keywords), str(keywords_file))

    start_time = time.time()
    run_results = []
    runs_skipped = 0

    for model in models:
        for serp_n, top_n in pool_sizes:
            label = run_label(engine, model, serp_n, top_n)
            out_dir = run_output_dir(engine, model, serp_n, top_n)
            out_dir.mkdir(parents=True, exist_ok=True)
            results_dir = out_dir / "analysis"

            # ── Resume logic: skip completed runs ─────────────────────
            if not args.force and tracker.is_run_complete(label):
                print(f"\n  SKIP (already completed): {label}")
                runs_skipped += 1
                run_results.append({"run": label, "gather": "DONE",
                                    "features": "DONE", "clean": "DONE",
                                    "analysis": "DONE", "elapsed_min": 0})
                continue

            # ── Initialize run tracking ───────────────────────────────
            tracker.init_run(label, model, engine, serp_n, top_n, len(keywords))
            run_start = time.time()
            status = {"run": label, "gather": None, "features": None,
                      "clean": None, "analysis": None}

            # Step 1: Gather data
            if not args.skip_gather:
                if not args.force and tracker.is_phase_complete(label, "gather"):
                    print(f"\n  SKIP gather (already done): {label}")
                    status["gather"] = "DONE"
                else:
                    rc = run_gather(label, keywords_file, engine, model,
                                    serp_n, top_n, out_dir,
                                    keyword_limit=args.keywords)
                    status["gather"] = "OK" if rc == 0 else f"FAIL({rc})"
            else:
                tracker.update_phase(label, "gather", "skipped")
                status["gather"] = "SKIPPED"

            # Step 2: Extract enhanced features
            if not args.skip_features:
                if not args.force and tracker.is_phase_complete(label, "extract_features"):
                    print(f"\n  SKIP extract_features (already done): {label}")
                    status["features"] = "DONE"
                else:
                    rc = run_extract_features(label, out_dir)
                    status["features"] = "OK" if rc == 0 else f"FAIL({rc})"
            else:
                tracker.update_phase(label, "extract_features", "skipped")
                status["features"] = "SKIPPED"

            # Step 3: Clean & merge
            if not args.force and tracker.is_phase_complete(label, "clean"):
                print(f"\n  SKIP clean (already done): {label}")
                status["clean"] = "DONE"
            else:
                rc = run_clean(label, out_dir)
                status["clean"] = "OK" if rc == 0 else f"FAIL({rc})"

            # Step 4: Per-run DML analysis
            if not args.skip_analysis:
                if not args.force and tracker.is_phase_complete(label, "analyze"):
                    print(f"\n  SKIP analyze (already done): {label}")
                    status["analysis"] = "DONE"
                else:
                    rc = run_analysis(label, out_dir, results_dir)
                    status["analysis"] = "OK" if rc == 0 else f"FAIL({rc})"
            else:
                tracker.update_phase(label, "analyze", "skipped")
                status["analysis"] = "SKIPPED"

            elapsed = time.time() - run_start
            status["elapsed_min"] = round(elapsed / 60, 1)
            run_results.append(status)

            # Mark run complete if all phases OK
            all_ok = all(v in ("OK", "DONE", "SKIPPED") for v in
                         [status["gather"], status["features"],
                          status["clean"], status["analysis"]])
            tracker.complete_run(label, "completed" if all_ok else "partial")
            print(f"\n  Run {label} {'completed' if all_ok else 'partial'} in {status['elapsed_min']}m")

    # ── Merge all runs + cross-model analysis ─────────────────────────────
    if not args.skip_merge:
        merged_csv = OUTPUT_ROOT / "merged_all_runs.csv"
        merged = merge_all_datasets(OUTPUT_ROOT, merged_csv)

        if merged is not None and not args.skip_analysis:
            cross_results_dir = OUTPUT_ROOT / "cross_model_analysis"
            cross_results_dir.mkdir(parents=True, exist_ok=True)
            run_cross_model_analysis(merged_csv, cross_results_dir)

    # ── Final summary ─────────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_elapsed / 60:.1f} minutes")
    if runs_skipped:
        print(f"  Runs skipped (already done): {runs_skipped}")
    print(f"\n  Run Summary:")
    for s in run_results:
        print(f"    {s['run']}: gather={s['gather']} features={s['features']} "
              f"clean={s['clean']} analysis={s['analysis']} ({s['elapsed_min']}m)")

    # Save experiment manifest
    manifest = {
        "experiment_start_utc": datetime.now(timezone.utc).isoformat(),
        "keywords_count": len(keywords),
        "keywords_file": str(keywords_file),
        "engine": engine,
        "models": models,
        "pool_sizes": pool_sizes,
        "total_runs": total_runs,
        "runs_skipped": runs_skipped,
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "run_results": run_results,
    }
    manifest_path = OUTPUT_ROOT / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest saved: {manifest_path}")
    print(f"  Tracker:       {tracker.TRACKER_FILE}")
    print(f"  Output root:   {OUTPUT_ROOT}/")

    # Print tracker status
    print()
    tracker.print_status()


if __name__ == "__main__":
    main()
