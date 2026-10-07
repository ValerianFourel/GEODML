#!/bin/bash
# Make HoreKa's local Qwen and Llama datasets complete, then report what each holds.
# Usage (HoreKa login shell):  bash horeka-sync-full-dataset.sh count   # changes nothing
#                              bash horeka-sync-full-dataset.sh apply   # seal, release, download
# apply, Qwen: (1) for ended Slurm jobs, seal their unsealed output files and release their stale claims
#   (reconcile_agentic_dataset, receipt kept); (2) download every Hub result bundle (coordination/
#   qwen-results.json) whose cells this dataset lacks and import them into its ledger (Exchange.download
#   verifies every file and reference). Llama: counts only. Refuses while a Qwen bout is queued or running.
set -euo pipefail
MODE=${1:-count}
case "$MODE" in count|apply) ;; *) echo "usage: $0 count|apply" >&2; exit 2;; esac
test -z "${SLURM_JOB_ID:-}" || { echo "run on a login shell" >&2; exit 2; }
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
QCODE="$W/checkouts/qwen-recovery-405b408cda3d2bab926c90975cb55476f8cfcd0c"
test -z "$(git -C "$QCODE" status --porcelain --untracked-files=all)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$QCODE"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
QDATA="$W/shared-hours/dataset"; LDATA="$W/llama-hf/dataset"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); OUT="$W/qwen-bouts/sync-$STAMP"; mkdir -p "$OUT"
echo "mode $MODE, receipts in $OUT"
if [ "$MODE" = apply ]; then
  if squeue --me -h -o %j | grep -q '^geodml-qwen-bout'; then
    echo "a Qwen bout is queued or running; apply only after it ends (count is safe)" >&2; exit 1
  fi
  echo "== Qwen 1/2: seal crashed writers and release stale claims of ended jobs"
  "$RT/bin/python" -c '
import json, sys
from analysis.scripts.capture_agentic_scheduler_snapshot import capture
s = capture(plan={"plan_id": "horeka-qwen-manual"}, since="2026-09-15", include_all_jobs=True); s["cluster"] = "horeka"
open(sys.argv[1], "w").write(json.dumps(s)); print(len(s["owners"]), "ended-job owners")' "$OUT/scheduler.json"
  nice -n 10 "$RT/bin/python" "$QCODE/analysis/scripts/reconcile_agentic_dataset.py" --dataset-root "$QDATA" \
    --scheduler-snapshot "$OUT/scheduler.json" --apply --receipt "$OUT/reconcile-receipt.json" > "$OUT/reconcile.log"
  "$RT/bin/python" -c '
import json, sys, collections
r = json.load(open(sys.argv[1]))
print("actions:", dict(collections.Counter(a["action"] for a in r["actions"])),
      "| still blocked:", dict(collections.Counter(b["reason"] for b in r["blocked"])),
      "| writers sealed:", len(r["recovered_writers"]))' "$OUT/reconcile-receipt.json"
fi
if [ "$MODE" = apply ]; then echo "== Qwen 2/2: download missing Hub results"; else echo "== Qwen: Hub results missing here"; fi
nice -n 10 "$RT/bin/python" -u - "$QDATA" "$W/qwen-bouts/import-journal" "$MODE" <<'PY'
import sys
from pathlib import Path
from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.scripts.publish_qwen_results import read_index
root, journal, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
store = HubStore("ValerianFourel/geodml-experiment-v2-paper-private")
ex = Exchange(store, journal)
rev = store.head()
index = read_index(store, rev)
latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=256).snapshot()["latest"]
need = []
for entry in index["bundles"]:
    outcomes = ex.manifest(entry["bundle"], rev)["outcomes"]
    missing = sum(latest.get(fp, {}).get("state") != e["state"] for fp, e in outcomes.items())
    if missing:
        need.append((entry["bundle"], missing, entry.get("writer_id")))
print("Hub bundles:", len(index["bundles"]), "| with cells missing here:", len(need),
      "| cells:", sum(n for _, n, _ in need), flush=True)
if mode == "apply":
    with ex.reusing_verification():
        for i, (bundle, n, writer) in enumerate(need, 1):
            ex.download(bundle, root, stripes=256, revision=rev)
            print("imported", i, "of", len(need), writer, n, "cells", flush=True)
    print("IMPORT DONE", flush=True)
PY
echo "== what each local dataset holds"
for PAIR in "qwen38=$QDATA" "llama4=$LDATA"; do
  nice -n 10 "$RT/bin/python" - "${PAIR%%=*}" "${PAIR#*=}" <<'PY'
import collections, sys
from pathlib import Path
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
model, root = sys.argv[1], Path(sys.argv[2])
tasks = {identity_fingerprint(ClaimIdentity(**r["claim_identity"])) for r in iter_sealed_rows(root, "task_definitions", required=True)
         if r.get("model") == model}
latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=256).snapshot()["latest"]
states = collections.Counter(latest.get(fp, {}).get("state", "never_started") for fp in tasks)
print(f"{model}: registered {len(tasks):,} |", " | ".join(f"{s} {n:,}" for s, n in states.most_common()), flush=True)
PY
done
echo "receipts: $OUT"
