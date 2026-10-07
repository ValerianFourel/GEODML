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
# embedding tasks (all four GPUs, validated LLM2Vec profile) when they are ready. Wall time 04:00:00 is a PLACEHOLDER:
# allocations of this script need Valerian's explicit approval (proposed: up to 3 allocations of 4 hours, 4 A100 each).
# Needs CODE, LEDGER. Optional: GPU_SIF (default $W/containers/geodml-gpu.sif), MARGIN_MIN (default 15), CPU_STEPS (1: let
# this node also run CPU-only tasks with its spare cores in a second worker).
# Exit: 0 nothing left this node may start; 4 deadline with ready tasks left (resubmit after reconcile); 2 storage/CUDA.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
source "$W/geodml-nemotron-env.sh"
CODE=${CODE:?CODE must point to the pinned checkout}
LEDGER=${LEDGER:?LEDGER must point to the planned ledger}
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
COMMIT=$(git -C "$CODE" rev-parse HEAD)
GPU_SIF=${GPU_SIF:-$W/containers/geodml-gpu.sif}; MARGIN_MIN=${MARGIN_MIN:-15}
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
rc=0
(cd "$CODE" && "$RT/bin/python" -u -m analysis.fullrun worker --ledger "$LEDGER" --gpu --devices 4 --cores 64 --python "$WRAP" \
  --end-epoch "$END" --margin-minutes "$MARGIN_MIN" --min-start-minutes 20 --job "$SLURM_JOB_ID") || rc=$?
(cd "$CODE" && "$RT/bin/python" -m analysis.fullrun status --ledger "$LEDGER") || true
echo "fullrun gpu-stats: exit $rc end $(date -u +%FT%TZ)"
exit "$rc"
