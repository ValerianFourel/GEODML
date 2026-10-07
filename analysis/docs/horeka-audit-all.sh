#!/bin/bash
# Read-only audit of everything V2 holds, on HoreKa and on the private Hub. Changes nothing.
#   1 generation on HoreKa: Llama and Qwen registered cells by ledger state
#   2 generation on the Hub: published completed / failed per model (bundle manifests)
#   3 Gemma SI-v4 on HoreKa: per run, cells, shards finished, task states summed over shard reports
#   4 Gemma SI-v4 on the Hub: finished shards in each run's export manifest versus HoreKa
# A final verdict line per item says COMPLETE, or what is missing and which page step fixes it.
# Usage: bash <checkout>/analysis/docs/horeka-audit-all.sh   (login shell; a few minutes)
set -euo pipefail
test -z "${SLURM_JOB_ID:-}" || { echo "run on a login shell" >&2; exit 2; }
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
echo "audit $(date -u +%FT%TZ) from $(git -C "$CODE" rev-parse --short HEAD); queue: $(squeue --me -h -o %j | sort | uniq -c | tr '\n' ' ')"
nice -n 10 "$RT/bin/python" -u - "$W" <<'PY'
import collections, json, sys
from pathlib import Path
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.report_inference_hub_progress import collect as hub_counts

W = Path(sys.argv[1])
REPO = "ValerianFourel/geodml-experiment-v2-paper-private"
DATA = {"llama4": W / "llama-hf/dataset", "qwen38": W / "shared-hours/dataset"}
GEMMA = {"llama4": [W / "reviews/gemma-si-v4-llama-reuse-5h-20261004"],
         "qwen38": [W / "reviews/gemma-si-v4-qwen-cpu-5h-20261006", W / "reviews/gemma-si-v4-qwen-topup-cpu-5h-20261007"]}
DONE = ("finished", "finished_with_failures")
verdict = []

print("\n== 1 generation on HoreKa")
local = {}
for model, root in DATA.items():
    tasks = {identity_fingerprint(ClaimIdentity(**r["claim_identity"]))
             for r in iter_sealed_rows(root, "task_definitions", required=True) if r.get("model") == model}
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=256).snapshot()["latest"]
    states = collections.Counter(latest.get(fp, {}).get("state", "never_started") for fp in tasks)
    local[model] = {"registered": len(tasks), **states}
    print(f"  {model}: registered {len(tasks):,} | " + " | ".join(f"{s} {n:,}" for s, n in states.most_common()))
    open_ = len(tasks) - states["completed"] - states["terminal_failed"]
    verdict.append(f"{model} generation on HoreKa: " + ("COMPLETE" if open_ == 0 else
                   f"{open_:,} cells not done (never_started -> page step 2 sync; checkpointed/claimed -> step 3 send)"))

print("\n== 2 generation on the Hub")
store = HubStore(REPO)
rev = store.head()
hub = hub_counts(Exchange(store, Path(".")), rev)
for model in ("llama4", "qwen38"):
    m = hub["models"][model]
    print(f"  {model}: completed {m['completed']} | terminal_failed {m['terminal_failed']} | conflicts {m['conflicts']} | sources {m['sources']}")
    done_here = local[model].get("completed", 0) + local[model].get("terminal_failed", 0)
    done_hub = (m["completed"] or 0) + (m["terminal_failed"] or 0)
    gap = done_here - done_hub
    verdict.append(f"{model} generation on the Hub: " + ("UP TO DATE" if gap <= 0 else
                   f"{gap:,} cells done on HoreKa but not published (page step 4 publishes Qwen)"))
if hub["issues"]:
    print("  hub issues:", hub["issues"][:5])

print("\n== 3 Gemma SI-v4 on HoreKa")
runs = []
for model, roots in GEMMA.items():
    for root in roots:
        if not (root / "plan.json").is_file():
            print(f"  {model}: {root.name}: no plan yet")
            continue
        plan = json.loads((root / "plan.json").read_text())
        status, states, finished = collections.Counter(), collections.Counter(), set()
        for shard in plan["shards"]:
            results = Path(shard["directory"]) / "results"
            latest = results / "reports/latest.json"
            if not latest.is_file():
                status["not_started"] += 1
                continue
            summary = json.loads((results / "reports" / json.loads(latest.read_text())["directory"] / "summary.json").read_text())
            status[summary.get("status")] += 1
            states.update(summary.get("states", {}))
            if summary.get("status") in DONE:
                finished.add(shard["id"])
        runs.append((model, root, plan, finished))
        print(f"  {model}: {root.name}: {plan['cells']:,} cells in {len(plan['shards'])} shards | shards {dict(status)}")
        print(f"      task states: " + " | ".join(f"{s} {n:,}" for s, n in states.most_common()))
        left = len(plan["shards"]) - len(finished)
        verdict.append(f"Gemma {model} {root.name}: " + ("COMPLETE" if left == 0 else f"{left} of {len(plan['shards'])} shards still running or queued"))
    judged = sum(r[2]["cells"] for r in runs if r[0] == model)
    gen = local[model].get("completed", 0)
    print(f"  {model}: cells in Gemma plans {judged:,} vs generated {gen:,} (difference: excluded diagnostics, input problems, or answers not yet frozen)")

print("\n== 4 Gemma SI-v4 on the Hub")
for model, root, plan, finished in runs:
    raw = store.read(f"reviews/gemma-si-v4/{plan['plan_id']}/export-manifest.json", rev)
    if raw is None:
        print(f"  {model}: {root.name}: not published yet")
        verdict.append(f"Gemma {model} {root.name} on the Hub: NOT PUBLISHED (page step 5)")
        continue
    manifest = json.loads(raw)
    published = {s["shard"] for s in manifest["shards"] if s.get("status") in DONE}
    missing = finished - published
    print(f"  {model}: {root.name}: {len(published)} finished shards published, {len(finished)} finished here")
    verdict.append(f"Gemma {model} {root.name} on the Hub: " + ("UP TO DATE" if not missing else f"{len(missing)} finished shards not yet published (page step 5)"))

print("\n== verdict")
for line in verdict:
    print("  " + line)
PY
