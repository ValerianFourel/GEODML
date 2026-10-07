#!/bin/bash
# Finite wave of full-run CPU allocations with observed-start admission (run on a login node; no computation here).
# Usage: CODE=... LEDGER=... bash horeka-fullrun-wave.sh N        (N <= 4, Valerian's approved concurrency)
# Submits one job, waits until Slurm shows it RUNNING, waits 10 minutes, checks concurrency and storage, then submits
# the next; stops after N submissions, when the ledger has no ready CPU task, or on storage failure. Never cancels jobs.
set -euo pipefail
N=${1:?usage: horeka-fullrun-wave.sh N}
[ "$N" -le 4 ] || { echo "at most 4 concurrent allocations were approved" >&2; exit 2; }
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?}; LEDGER=${LEDGER:?}
SBATCH="$CODE/analysis/docs/horeka-fullrun-cpu.sbatch"
LOGS="$(dirname "$LEDGER")/slurm"; mkdir -p "$LOGS"
running_jobs() { squeue -h -u "$USER" -o %i | wc -l; }
for i in $(seq 1 "$N"); do
  ready=$(cd "$CODE" && "$RT/bin/python" -c "import json,sys; sys.path.insert(0,'.'); from analysis.fullrun.ledger import Ledger; s=Ledger('$LEDGER').status(); print(s['states'].get('ready',0)+s['states'].get('waiting',0))")
  [ "$ready" -gt 0 ] || { echo "no CPU work left to admit"; break; }
  [ "$(running_jobs)" -lt 5 ] || { echo "five allocations already queued or running: stop"; break; }
  free=$(df -P "$W" | awk 'NR==2 {print int($4/1e6)}')
  [ "$free" -ge 100 ] || { echo "storage below 100 GB free: stop"; exit 2; }
  JOB=$(sbatch --parsable --export=ALL,CODE="$CODE",LEDGER="$LEDGER" --output="$LOGS/cpu-%j.out" "$SBATCH")
  echo "submitted $JOB ($i of $N) $(date -u +%FT%TZ)"
  [ "$i" -lt "$N" ] || break
  until [ "$(squeue -h -j "$JOB" -o %T 2>/dev/null)" = RUNNING ] || [ -z "$(squeue -h -j "$JOB" -o %T 2>/dev/null)" ]; do sleep 60; done
  echo "observed start of $JOB $(date -u +%FT%TZ); waiting 10 minutes before the next admission"
  sleep 600
done
squeue -u "$USER" -o '%.10i %.9P %.22j %.8T %.10M %.10l %.20S'
