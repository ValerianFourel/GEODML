#!/bin/bash
#SBATCH --job-name=geodml-fullrun-gpustats
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=152
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# GPU statistics of the full run (ledger planned with "gpu_stats": true): up to four 1-GPU statistics tasks at once
# (funnel stage models, generator decisions, steelman generator and fixed effects, all with --backend cuda), plus the
# embedding tasks (all four GPUs, validated LLM2Vec profile) when they are ready. Approved by Valerian on 2026-10-08
# ("launch them all"): submitted by horeka-fullrun-launch.sh with --time 03:00:00, at most 1 at once, cap 3 (36 GPU-hours).
# Needs CODE, LEDGER. Optional: GPU_SIF (default $W/containers/geodml-gpu.sif), MARGIN_MIN (default 15), CPU_STEPS (1: a
# second worker runs the CPU-only tasks on this node's cores, CPU_CORES of them, default 72; the GPU worker keeps
# GPU_CORES, default 32; both then wait up to IDLE_MIN=60 minutes for dependencies, since the node is held anyway).
# Exit: 0 nothing left this node may start; 4 deadline with ready tasks left (resubmit after reconcile); 2 storage/CUDA.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
LEDGER=${LEDGER:?LEDGER must point to the planned ledger}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
COMMIT=$(git -C "$CODE" rev-parse HEAD)
GPU_SIF=${GPU_SIF:-$W/containers/geodml-gpu.sif}; MARGIN_MIN=${MARGIN_MIN:-15}; CPU_STEPS=${CPU_STEPS:-0}
if [ "$CPU_STEPS" = 1 ]; then GPU_CORES=${GPU_CORES:-32}; CPU_CORES=${CPU_CORES:-72}; IDLE_MIN=${IDLE_MIN:-60}; else GPU_CORES=${GPU_CORES:-64}; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
echo "fullrun gpu-stats: commit $COMMIT job $SLURM_JOB_ID node $(hostname) end $(date -u -d @"$END" +%FT%TZ)"
nvidia-smi -L
RUN=(apptainer exec --nv --no-home --bind "$W" --env "GEODML_GIT_COMMIT=$COMMIT,PYTHONPATH=$CODE" "$GPU_SIF")
"${RUN[@]}" python -c "import torch,sys; ok=torch.cuda.is_available() and torch.cuda.device_count()==4; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'devices',torch.cuda.device_count()); sys.exit(0 if ok else 2)"
"$RT/bin/python" - "$LEDGER" <<'PY'
import os, shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 1e9; inodes = os.statvfs(sys.argv[1]).f_favail
print(f"storage: {free:.0f} GB free, {inodes} inodes free"); sys.exit(0 if free >= 100 and inodes >= 2_000_000 else 2)
PY
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun reconcile --ledger "$LEDGER" --job "$SLURM_JOB_ID")
# The GPU worker runs on the host: statistics tasks call {PY} = the container's python through the wrapper below; the
# embedding tasks run horeka-fullrun-embed.sh with the validated venvs (no container), as on the embedding node.
WRAP=$(mktemp -d)/py; printf '#!/bin/bash\nexec %s python "$@"\n' "${RUN[*]}" > "$WRAP"; chmod +x "$WRAP"
rc=0; crc=0; CPID=""
if [ "$CPU_STEPS" = 1 ]; then
  CLOG="$(dirname "$LEDGER")/slurm/gpustats-cpu-$SLURM_JOB_ID.out"; mkdir -p "$(dirname "$CLOG")"
  echo "CPU worker beside the GPU worker: $CPU_CORES cores, log $CLOG"
  (cd "$CODE" && "${RUN[@]}" python -u -m analysis.fullrun worker --ledger "$LEDGER" --cores "$CPU_CORES" --end-epoch "$END" \
    --margin-minutes "$MARGIN_MIN" --idle-minutes "$IDLE_MIN" --job "$SLURM_JOB_ID") > "$CLOG" 2>&1 &
  CPID=$!
fi
(cd "$CODE" && "$RT/bin/python" -u -m analysis.fullrun worker --ledger "$LEDGER" --gpu --devices 4 --cores "$GPU_CORES" --python "$WRAP" \
  --end-epoch "$END" --margin-minutes "$MARGIN_MIN" --min-start-minutes 20 --idle-minutes "${IDLE_MIN:-20}" --job "$SLURM_JOB_ID") || rc=$?
if [ -n "$CPID" ]; then wait "$CPID" || crc=$?; echo "CPU worker exit $crc"; tail -n 3 "$CLOG"; fi
[ "$rc" -eq 0 ] && [ "$crc" -ne 0 ] && rc=$crc
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun status --ledger "$LEDGER") || true
echo "fullrun gpu-stats: exit $rc end $(date -u +%FT%TZ)"
exit "$rc"
