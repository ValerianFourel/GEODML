"""python -m analysis.fullrun <command>

  plan      --config CONFIG.json --ledger DIR        write the finite task list (once)
  worker    --ledger DIR [--gpu] [--stage S ...]     run ready tasks in this allocation until none is ready or the deadline
  status    --ledger DIR                             counts by state and stage, remaining estimate, disk and inodes
  reconcile --ledger DIR                             release claims of allocations Slurm reports terminal
  merge     funnel|trace --output DIR SHARD...       merge keyword-hash extraction shards
  hub       inventory|missing|fetch ...              reconcile with the Hub and import missing bundles (login node only)

Worker exit codes: 0 nothing left this worker may start; 4 deadline with ready tasks left (reconcile, then resubmit).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.fullrun import ledger as L  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--ledger", type=Path, required=True)
    q = sub.add_parser("worker")
    q.add_argument("--ledger", type=Path, required=True)
    q.add_argument("--gpu", action="store_true", help="run only GPU tasks (else only CPU tasks)")
    q.add_argument("--stage", action="append", help="only tasks of these stages")
    q.add_argument("--cores", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1)))
    q.add_argument("--end-epoch", type=float, help="allocation end (seconds since epoch); default now + --hours")
    q.add_argument("--hours", type=float, default=1.0)
    q.add_argument("--margin-minutes", type=float, default=10.0)
    q.add_argument("--min-start-minutes", type=float, default=10.0, help="start no task with less time left")
    q.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}"))
    q.add_argument("--devices", type=int, default=4, help="GPU worker: device slots (CUDA_VISIBLE_DEVICES 0..N-1)")
    for name in ("status", "reconcile"):
        q = sub.add_parser(name)
        q.add_argument("--ledger", type=Path, required=True)
        q.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", ""))
    q = sub.add_parser("merge")
    q.add_argument("kind", choices=("funnel", "trace"))
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("shards", nargs="+", type=Path)
    q = sub.add_parser("hub")
    q.add_argument("action", choices=("inventory", "missing", "fetch"))
    q.add_argument("--cache", type=Path, required=True, help="folder for bundle descriptors and the inventory")
    q.add_argument("--root", type=Path, action="append", default=[], help="local dataset roots (missing)")
    q.add_argument("--import-root", type=Path, help="fetch: dataset root that receives the imported bundles")
    q.add_argument("--shard", help="fetch: only bundles k, k+n, ... of the missing list (k/n)")
    q.add_argument("--revision")
    a = p.parse_args(argv)

    if a.command == "plan":
        from analysis.fullrun import plan
        tasks = plan.build(plan.load_config(a.config))
        ledger = L.Ledger(a.ledger)
        ledger.write_tasks(tasks)
        (ledger.root / "config.json").write_text(a.config.read_text())
        print(json.dumps({"tasks": len(tasks), "cpu_hours_estimate": round(sum(t.est_cpu_h for t in tasks), 1),
                          "gpu_tasks": sum(t.gpu for t in tasks)}))
        return 0
    if a.command == "worker":
        end = a.end_epoch or time.time() + 3600 * a.hours
        return L.work(L.Ledger(a.ledger), job=a.job, cores=a.cores, end_epoch=end, margin_minutes=a.margin_minutes,
                      stages=set(a.stage) if a.stage else None, gpu=a.gpu, min_start_minutes=a.min_start_minutes,
                      devices=a.devices)
    if a.command == "status":
        print(json.dumps(L.Ledger(a.ledger).status(), indent=1))
        return 0
    if a.command == "reconcile":
        print(json.dumps({"released": L.Ledger(a.ledger).reconcile(a.job)}))
        return 0
    if a.command == "merge":
        from analysis.fullrun import merge
        fn = merge.merge_funnel_extract if a.kind == "funnel" else merge.merge_trace_extract
        print(json.dumps(fn(a.shards, a.output)))
        return 0
    from analysis.fullrun import hub
    if a.action == "inventory":
        print(json.dumps(hub.inventory(a.cache, revision=a.revision)))
    elif a.action == "missing":
        result = hub.missing(a.cache / "inventory.json", hub.local_completed(a.root))
        (a.cache / "missing.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k != "bundles"}))
    else:
        result = hub.fetch(json.loads((a.cache / "missing.json").read_text()), a.import_root, a.shard, revision=a.revision)
        print(json.dumps(result))
        return 0 if not result["failed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
