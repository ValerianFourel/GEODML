#!/usr/bin/env python3
"""Experiment tracker — persistent state tracking across runs.

Tracks progress at every level (run, keyword, phase) so the experiment
can be interrupted and resumed without losing work.

Usage:
  # Check status
  python paperSizeExperiment/experiment_tracker.py status

  # Detailed status for a specific run
  python paperSizeExperiment/experiment_tracker.py status --run searxng_Llama-3.3-70B-Instruct_serp20_top10

  # Reset a failed run
  python paperSizeExperiment/experiment_tracker.py reset --run searxng_Llama-3.3-70B-Instruct_serp20_top10

  # Export summary
  python paperSizeExperiment/experiment_tracker.py export
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACKER_FILE = SCRIPT_DIR / "output" / "tracker.json"


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _load_tracker() -> dict:
    """Load or initialize the tracker state."""
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {
        "created_utc": _utcnow(),
        "last_updated_utc": None,
        "experiment_config": {},
        "runs": {},
        "global_stats": {
            "total_keywords_attempted": 0,
            "total_keywords_succeeded": 0,
            "total_urls_fetched": 0,
            "total_urls_failed": 0,
            "total_llm_calls": 0,
            "total_llm_errors": 0,
            "total_elapsed_seconds": 0,
        },
    }


def _save_tracker(state: dict):
    """Save tracker state to disk."""
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated_utc"] = _utcnow()
    with open(TRACKER_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def init_experiment(models: list, pool_sizes: list, engine: str,
                    keywords_count: int, keywords_file: str):
    """Initialize the tracker for a new experiment."""
    state = _load_tracker()
    state["experiment_config"] = {
        "models": models,
        "pool_sizes": pool_sizes,
        "engine": engine,
        "keywords_count": keywords_count,
        "keywords_file": keywords_file,
        "initialized_utc": _utcnow(),
    }
    _save_tracker(state)
    return state


def init_run(run_id: str, model: str, engine: str,
             serp_n: int, llm_top_n: int, keywords_count: int):
    """Initialize tracking for a specific run."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        state["runs"][run_id] = {
            "model": model,
            "engine": engine,
            "serp_n": serp_n,
            "llm_top_n": llm_top_n,
            "keywords_total": keywords_count,
            "status": "initialized",
            "started_utc": _utcnow(),
            "ended_utc": None,
            "elapsed_seconds": 0,
            "phases": {
                "gather": {"status": "pending", "started_utc": None, "ended_utc": None},
                "extract_features": {"status": "pending", "started_utc": None, "ended_utc": None},
                "clean": {"status": "pending", "started_utc": None, "ended_utc": None},
                "analyze": {"status": "pending", "started_utc": None, "ended_utc": None},
            },
            "gather_progress": {
                "keywords_attempted": 0,
                "keywords_succeeded": 0,
                "keywords_failed": 0,
                "keywords_failed_list": [],
                "urls_fetched": 0,
                "urls_failed": 0,
                "llm_calls": 0,
                "llm_errors": 0,
                "llm_fallbacks": 0,
            },
            "feature_progress": {
                "urls_processed": 0,
                "moz_domains_queried": 0,
            },
            "analysis_progress": {
                "treatments_analyzed": 0,
                "significant_results": 0,
            },
            "errors": [],
            "last_keyword_processed": None,
            "output_files": {},
        }
    _save_tracker(state)
    return state


def update_phase(run_id: str, phase: str, status: str, extra: dict = None):
    """Update the status of a phase within a run."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    run = state["runs"][run_id]
    if phase in run["phases"]:
        run["phases"][phase]["status"] = status
        if status == "running":
            run["phases"][phase]["started_utc"] = _utcnow()
            run["status"] = f"running:{phase}"
        elif status in ("completed", "failed", "skipped"):
            run["phases"][phase]["ended_utc"] = _utcnow()
        if extra:
            run["phases"][phase].update(extra)
    _save_tracker(state)


def update_gather_progress(run_id: str, **kwargs):
    """Update gather phase progress counters."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    progress = state["runs"][run_id]["gather_progress"]
    for k, v in kwargs.items():
        if k in progress:
            if isinstance(progress[k], list) and isinstance(v, str):
                progress[k].append(v)
            else:
                progress[k] = v
    if "last_keyword" in kwargs:
        state["runs"][run_id]["last_keyword_processed"] = kwargs["last_keyword"]
    _save_tracker(state)


def update_feature_progress(run_id: str, **kwargs):
    """Update feature extraction progress."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    progress = state["runs"][run_id]["feature_progress"]
    for k, v in kwargs.items():
        if k in progress:
            progress[k] = v
    _save_tracker(state)


def update_analysis_progress(run_id: str, **kwargs):
    """Update analysis progress."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    progress = state["runs"][run_id]["analysis_progress"]
    for k, v in kwargs.items():
        if k in progress:
            progress[k] = v
    _save_tracker(state)


def record_output_file(run_id: str, name: str, path: str, rows: int = None):
    """Record an output file produced by a run."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    state["runs"][run_id]["output_files"][name] = {
        "path": path,
        "rows": rows,
        "saved_utc": _utcnow(),
    }
    _save_tracker(state)


def record_error(run_id: str, phase: str, error: str):
    """Record an error that occurred during a run."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    state["runs"][run_id]["errors"].append({
        "phase": phase,
        "error": error[:500],
        "timestamp_utc": _utcnow(),
    })
    _save_tracker(state)


def complete_run(run_id: str, status: str = "completed"):
    """Mark a run as completed or failed."""
    state = _load_tracker()
    if run_id not in state["runs"]:
        return
    run = state["runs"][run_id]
    run["status"] = status
    run["ended_utc"] = _utcnow()
    # Calculate elapsed
    if run["started_utc"]:
        try:
            start = datetime.fromisoformat(run["started_utc"])
            end = datetime.fromisoformat(run["ended_utc"])
            run["elapsed_seconds"] = round((end - start).total_seconds())
        except Exception:
            pass
    # Update global stats
    gp = run["gather_progress"]
    gs = state["global_stats"]
    gs["total_keywords_attempted"] += gp["keywords_attempted"]
    gs["total_keywords_succeeded"] += gp["keywords_succeeded"]
    gs["total_urls_fetched"] += gp["urls_fetched"]
    gs["total_urls_failed"] += gp["urls_failed"]
    gs["total_llm_calls"] += gp["llm_calls"]
    gs["total_llm_errors"] += gp["llm_errors"]
    gs["total_elapsed_seconds"] += run["elapsed_seconds"]
    _save_tracker(state)


def reset_run(run_id: str):
    """Reset a run so it can be re-executed."""
    state = _load_tracker()
    if run_id in state["runs"]:
        model = state["runs"][run_id]["model"]
        engine = state["runs"][run_id]["engine"]
        serp_n = state["runs"][run_id]["serp_n"]
        llm_top_n = state["runs"][run_id]["llm_top_n"]
        keywords_total = state["runs"][run_id]["keywords_total"]
        del state["runs"][run_id]
        _save_tracker(state)
        init_run(run_id, model, engine, serp_n, llm_top_n, keywords_total)
        print(f"  Reset run: {run_id}")
    else:
        print(f"  Run not found: {run_id}")


def get_run_status(run_id: str) -> dict | None:
    """Get the current status of a run."""
    state = _load_tracker()
    return state["runs"].get(run_id)


def get_completed_keywords(run_id: str) -> int:
    """Get number of keywords completed for a run (for resume logic)."""
    state = _load_tracker()
    run = state["runs"].get(run_id, {})
    return run.get("gather_progress", {}).get("keywords_succeeded", 0)


def _read_live_progress(run_id: str) -> dict | None:
    """Read the live progress.json written by gather_data.py during execution."""
    # progress.json lives at output/<run_id>/progress.json
    progress_path = TRACKER_FILE.parent / run_id / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def is_run_complete(run_id: str) -> bool:
    """Check if a run has already been completed."""
    state = _load_tracker()
    run = state["runs"].get(run_id, {})
    return run.get("status") == "completed"


def is_phase_complete(run_id: str, phase: str) -> bool:
    """Check if a specific phase of a run is complete."""
    state = _load_tracker()
    run = state["runs"].get(run_id, {})
    return run.get("phases", {}).get(phase, {}).get("status") == "completed"


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_status(run_filter: str = None):
    """Print human-readable experiment status."""
    state = _load_tracker()

    if not state["runs"]:
        print("No runs recorded yet.")
        return

    config = state.get("experiment_config", {})
    if config:
        print(f"\nExperiment initialized: {config.get('initialized_utc', '?')}")
        print(f"  Keywords: {config.get('keywords_count', '?')}")
        print(f"  Models: {config.get('models', [])}")
        print(f"  Pool sizes: {config.get('pool_sizes', [])}")

    print(f"\n{'='*90}")
    print(f"{'Run':<55} {'Status':<15} {'KW OK':>6} {'URLs':>6} {'Time':>8}")
    print(f"{'='*90}")

    for run_id, run in sorted(state["runs"].items()):
        if run_filter and run_filter not in run_id:
            continue

        gp = run["gather_progress"]
        elapsed = run.get("elapsed_seconds", 0)
        elapsed_str = f"{elapsed // 3600}h{(elapsed % 3600) // 60:02d}m" if elapsed else "—"
        kw_str = f"{gp['keywords_succeeded']}/{run['keywords_total']}"
        url_str = f"{gp['urls_fetched']}"

        print(f"{run_id:<55} {run['status']:<15} {kw_str:>6} {url_str:>6} {elapsed_str:>8}")

        # Phase details
        for phase_name, phase in run["phases"].items():
            status = phase["status"]
            marker = {"pending": "  ", "running": ">>", "completed": "OK",
                      "failed": "!!", "skipped": "--"}.get(status, "??")
            print(f"  [{marker}] {phase_name:<20} {status}")

        # ── Live progress from progress.json ──────────────────────
        if run["status"].startswith("running") or run["status"] == "initialized":
            live = _read_live_progress(run_id)
            if live:
                phase = live.get("phase", "?")
                updated = live.get("last_updated_utc", "")
                if updated:
                    updated_short = updated[11:19]  # HH:MM:SS
                else:
                    updated_short = ""

                if phase.startswith("phase1"):
                    kw_done = live.get("phase1_keywords_done", 0)
                    kw_total = live.get("phase1_keywords_total", 0)
                    kw_failed = live.get("phase1_keywords_failed", 0)
                    current = live.get("phase1_current_keyword", "")
                    llm_err = live.get("phase1_llm_errors", 0)
                    llm_fb = live.get("phase1_llm_fallbacks", 0)
                    pct = (kw_done / kw_total * 100) if kw_total else 0
                    bar_len = 30
                    filled = int(bar_len * kw_done / kw_total) if kw_total else 0
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"  ┌─ LIVE Phase 1: SERP + LLM Re-Ranking")
                    print(f"  │  [{bar}] {kw_done}/{kw_total} ({pct:.0f}%)")
                    if current:
                        print(f"  │  Current: \"{current}\"")
                    if kw_failed:
                        print(f"  │  Failed: {kw_failed}")
                    if llm_err or llm_fb:
                        print(f"  │  LLM errors: {llm_err}, fallbacks: {llm_fb}")
                    print(f"  └─ Updated: {updated_short}")

                elif phase.startswith("phase2"):
                    url_done = live.get("phase2_urls_done", 0)
                    url_total = live.get("phase2_urls_total", 0)
                    url_failed = live.get("phase2_urls_failed", 0)
                    current = live.get("phase2_current_url", "")
                    pct = (url_done / url_total * 100) if url_total else 0
                    bar_len = 30
                    filled = int(bar_len * url_done / url_total) if url_total else 0
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"  ┌─ LIVE Phase 2: HTML Feature Extraction")
                    print(f"  │  [{bar}] {url_done}/{url_total} ({pct:.0f}%)")
                    if current:
                        short_url = current[:60] + "..." if len(current) > 60 else current
                        print(f"  │  Current: {short_url}")
                    if url_failed:
                        print(f"  │  Failed: {url_failed}")
                    print(f"  └─ Updated: {updated_short}")

                elif phase.startswith("phase3"):
                    llm_done = live.get("phase3_llm_features_done", 0)
                    llm_total = live.get("phase3_llm_features_total", 0)
                    pct = (llm_done / llm_total * 100) if llm_total else 0
                    print(f"  ┌─ LIVE Phase 3: LLM Feature Extraction")
                    print(f"  │  {llm_done}/{llm_total} ({pct:.0f}%)")
                    print(f"  └─ Updated: {updated_short}")

                elif phase == "phase_pagerank":
                    print(f"  ┌─ LIVE: PageRank API")
                    print(f"  └─ Status: {live.get('phase_pagerank', '?')}")

                elif phase == "phase_whois":
                    print(f"  ┌─ LIVE: WHOIS Lookup")
                    print(f"  └─ Status: {live.get('phase_whois', '?')}")

                elif phase == "done":
                    print(f"  ┌─ LIVE: Gather complete")
                    print(f"  └─ Updated: {updated_short}")

        if run["errors"]:
            print(f"  Errors: {len(run['errors'])} (latest: {run['errors'][-1]['error'][:60]})")

        # Output files
        if run.get("output_files"):
            for fname, info in run["output_files"].items():
                rows_str = f" ({info['rows']} rows)" if info.get("rows") else ""
                print(f"  -> {fname}{rows_str}")

        print()

    # Global stats
    gs = state["global_stats"]
    total_elapsed = gs["total_elapsed_seconds"]
    print(f"{'─'*90}")
    print(f"Global: {gs['total_keywords_succeeded']} keywords, "
          f"{gs['total_urls_fetched']} URLs, "
          f"{gs['total_llm_calls']} LLM calls, "
          f"{gs['total_llm_errors']} LLM errors, "
          f"{total_elapsed // 3600}h{(total_elapsed % 3600) // 60:02d}m total")


def print_export():
    """Export a concise summary for pasting into notes."""
    state = _load_tracker()

    print(f"# Experiment Status Export — {_utcnow()}\n")

    config = state.get("experiment_config", {})
    print(f"Keywords: {config.get('keywords_count', '?')}")
    print(f"Models: {', '.join(m.split('/')[-1] for m in config.get('models', []))}")
    print(f"Pools: {config.get('pool_sizes', [])}\n")

    print(f"| Run | Status | Keywords | URLs | Errors | Time |")
    print(f"|-----|--------|----------|------|--------|------|")

    for run_id, run in sorted(state["runs"].items()):
        gp = run["gather_progress"]
        elapsed = run.get("elapsed_seconds", 0)
        elapsed_str = f"{elapsed // 60}m" if elapsed else "—"
        short_id = run_id.replace("searxng_", "").replace("_Instruct", "")
        print(f"| {short_id} | {run['status']} | "
              f"{gp['keywords_succeeded']}/{run['keywords_total']} | "
              f"{gp['urls_fetched']} | {len(run['errors'])} | {elapsed_str} |")

    gs = state["global_stats"]
    print(f"\nTotal: {gs['total_keywords_succeeded']} kw, "
          f"{gs['total_urls_fetched']} urls, "
          f"{gs['total_llm_calls']} llm calls")


def main():
    parser = argparse.ArgumentParser(description="Experiment tracker CLI")
    parser.add_argument("command", choices=["status", "export", "reset"],
                        help="Command to run")
    parser.add_argument("--run", type=str, default=None,
                        help="Filter to specific run ID")
    args = parser.parse_args()

    if args.command == "status":
        print_status(args.run)
    elif args.command == "export":
        print_export()
    elif args.command == "reset":
        if not args.run:
            print("--run is required for reset")
            sys.exit(1)
        reset_run(args.run)


if __name__ == "__main__":
    main()
