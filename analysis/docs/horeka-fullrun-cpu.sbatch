#!/bin/bash
#SBATCH --job-name=geodml-fullrun-cpu
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=152
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# One CPU node working through the full-run ledger (analysis/fullrun; plan in
# analysis/docs/handoff/2026-10-08_full-run-plan_handoff.md). Six hours, approved by Valerian on 2026-10-08
# (cap 8 CPU allocations, at most 4 at once, 10 minutes between observed starts: use horeka-fullrun-wave.sh).
# Needs: CODE (pinned clean checkout), LEDGER (planned ledger folder). Optional: SIF (default $W/containers/geodml-cpu.sif;
# RUNTIME=venv uses $W/environment/fullrun-cpu instead), CORES (default 76 physical cores), MARGIN_MIN (default 15),
# MIN_FREE_GB (default 100), MIN_FREE_INODES (default 2000000), IDLE_MIN (default 20: wait that long for dependencies
# another allocation is running before giving the node back).
# Exit: 0 nothing left this node may start; 4 deadline with ready tasks left (resubmit through the wave helper); 2 storage.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
LEDGER=${LEDGER:?LEDGER must point to the planned ledger}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
COMMIT=$(git -C "$CODE" rev-parse HEAD)
CORES=${CORES:-76}; MARGIN_MIN=${MARGIN_MIN:-15}
SIF=${SIF:-$W/containers/geodml-cpu.sif}; RUNTIME=${RUNTIME:-apptainer}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
echo "fullrun cpu: commit $COMMIT job $SLURM_JOB_ID node $(hostname) cores $CORES runtime $RUNTIME end $(date -u -d @"$END" +%FT%TZ)"
df -h "$W" | tail -n 1
# storage gate before any work (AGENTS.md: stop admissions on storage failure or stale evidence)
"$RT/bin/python" - "$LEDGER" "${MIN_FREE_GB:-100}" "${MIN_FREE_INODES:-2000000}" <<'PY'
import os, shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 1e9; inodes = os.statvfs(sys.argv[1]).f_favail
print(f"storage: {free:.0f} GB free, {inodes} inodes free")
sys.exit(0 if free >= float(sys.argv[2]) and inodes >= int(sys.argv[3]) else 2)
PY
# claims of ended allocations are released only after Slurm reports them terminal (host python: sacct)
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun reconcile --ledger "$LEDGER" --job "$SLURM_JOB_ID")
if [ "$RUNTIME" = venv ]; then
  RUN=("$W/environment/fullrun-cpu/bin/python")
else
  RUN=(apptainer exec --no-home --bind "$W" --env "GEODML_GIT_COMMIT=$COMMIT,PYTHONPATH=$CODE" "$SIF" python)
fi
rc=0
(cd "$CODE" && GEODML_GIT_COMMIT=$COMMIT "${RUN[@]}" -u -m analysis.fullrun worker --ledger "$LEDGER" --cores "$CORES" \
  --end-epoch "$END" --margin-minutes "$MARGIN_MIN" --idle-minutes "${IDLE_MIN:-20}" --job "$SLURM_JOB_ID") || rc=$?
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun status --ledger "$LEDGER") || true
echo "fullrun cpu: exit $rc end $(date -u +%FT%TZ)"
exit "$rc"
