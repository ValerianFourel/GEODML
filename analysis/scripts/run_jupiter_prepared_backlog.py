"""Resume an existing dataset-backed wave without recreating bootstrap plans."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hours import inventory
from analysis.scripts.reconcile_agentic_dataset import reconcile
from analysis.scripts.verify_inference_allocation import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-environment", required=True, type=Path)
    parser.add_argument("--scheduler-snapshot", required=True, type=Path)
    parser.add_argument("--ledger-stripes", type=int, default=256)
    args = parser.parse_args(argv)
    # Must precede artifact reads/recovery; caller activates the site environment.
    verify("jupiter")
    runtime = json.loads(args.runtime_environment.read_bytes())
    if not isinstance(runtime, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                            for k, v in runtime.items()):
        raise ValueError("runtime environment must be a saved string mapping")
    snapshot = json.loads(args.scheduler_snapshot.read_bytes())
    age = time.time() - snapshot["captured_at_epoch"]
    if not 0 <= age <= 120 or snapshot.get("complete") is not True:
        raise ValueError("resume requires a complete scheduler snapshot no older than 120 seconds")
    root = Path(runtime["GEODML_DATASET_ROOT"])
    wave = Path(runtime["GEODML_WAVE_ROOT"])
    manifest = json.loads((wave / "run_manifest.json").read_bytes())
    if manifest.get("format_version") != "geodml-inference-wave-v2" or manifest.get("dispatch_mode") != "backlog":
        raise ValueError("resume requires the existing prepared dataset backlog")
    # Read-only reconciliation. Ambiguous owners require a separate explicit audit.
    report = reconcile(root, scheduler_snapshot=snapshot, stripe_count=args.ledger_stripes)
    if report["blocked"] or report["actions"]:
        raise ValueError("dataset needs terminal-owner reconciliation before resume: " + json.dumps(report))
    tasks, completed, blocked = inventory(root, stripes=args.ledger_stripes)
    print(json.dumps({"current_registered": len(tasks), "current_verified_completed": len(completed),
                      "current_blocked": len(blocked), "reconciliation": report}), flush=True)
    os.environ.update(runtime)
    os.environ["GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"] = "1"
    repository = Path(__file__).resolve().parents[2]
    os.execv("/bin/bash", ["bash", str(repository / "analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch")])


if __name__ == "__main__":
    raise SystemExit(main())
