# Shared SLURM env setup for GEODML interpretability jobs on JUWELS Booster.
# Source me from each *.sbatch wrapper:  source scripts/slurm/_common.sh
#
# Boots the project venv, loads HF + dataset caches off $SCRATCH, and switches
# transformers/HF Hub into offline mode (compute nodes have no internet on
# JUWELS Booster — see https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html).
#
# Required env (set by the wrapper, not here):
#   SLURM_SUBMIT_DIR  — repo root, where .venv/, .env, geodml_data/ live
# Optional:
#   JUWELS_PROJECT    — passed to `jutil env activate -p <project>`
#   HF_HOME           — override the default $SCRATCH/hf_cache
#   GEODML_DATA_ROOT  — override the default $SCRATCH/geodml_data

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Project env (sets PROJECT, SCRATCH, etc. on JSC systems).
if [ -n "${JUWELS_PROJECT:-}" ]; then
  jutil env activate -p "$JUWELS_PROJECT"
fi

# JUPITER compute nodes mount /e/ instead of /p/. Translate SLURM_SUBMIT_DIR
# captured on login under /p/ to the matching /e/ path. Defaults make this
# safe even if jutil failed to set PROJECT/SCRATCH.
_PROJ="${PROJECT:-/e/project1/scifi}"
_SCR="${SCRATCH:-/e/scratch/scifi}"
if [ ! -d "${SLURM_SUBMIT_DIR:-}" ]; then
  ALT="${SLURM_SUBMIT_DIR:-}"
  ALT="${ALT/#\/p\/project1\/scifi/$_PROJ}"
  ALT="${ALT/#\/p\/scratch\/scifi/$_SCR}"
  if [ -n "$ALT" ] && [ -d "$ALT" ]; then
    echo "[common] path-translate /p/ -> $ALT"
    SLURM_SUBMIT_DIR="$ALT"
    export SLURM_SUBMIT_DIR
  fi
fi

# Module stack. JUPITER has Stages/{2025,2026}; 2026 is the default. JUWELS
# has Stages/2024. We standardise on 2026 because JUPITER is the active host.
# IMPORTANT: do NOT pipe `module load` through sed — the leftmost command of a
# pipeline runs in a subshell, so any LD_LIBRARY_PATH it sets is local to that
# subshell and python at runtime fails with `libpython3.13.so.1.0: cannot open
# shared object file` (rc=127). Capture to a tempfile, then pretty-print.
_modlog=$(mktemp -t modlog.XXXXXX)
module load Stages/2026 GCC Python CUDA >"$_modlog" 2>&1 || true
[ -s "$_modlog" ] && sed 's/^/[modules] /' "$_modlog"
rm -f "$_modlog"

# Activate the project venv created on the login node.
if [ ! -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
  echo "[common] ERROR: $SLURM_SUBMIT_DIR/.venv missing — create it on the login node first."
  exit 99
fi
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# Load .env (HF_TOKEN, HF_DATASET_REPO, model lists, ...).
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "$SLURM_SUBMIT_DIR/.env"
  set +o allexport
fi

# Caches on fast scratch. Offline because JUWELS Booster compute nodes
# have no outbound HTTP — pre-populate $HF_HOME on the login node.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GEODML_DATA_ROOT="${GEODML_DATA_ROOT:-$SCRATCH/geodml_data}"

# Make sklearn / numpy stay inside our 48-cpu allocation.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-48}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$SLURM_SUBMIT_DIR/logs"

echo "[common] node=$(hostname) jobid=${SLURM_JOB_ID:-?} attempt=${ATTEMPT:-1}/${MAX_ATTEMPTS:-6}"
echo "[common] HF_HOME=$HF_HOME"
echo "[common] GEODML_DATA_ROOT=$GEODML_DATA_ROOT"
echo "[common] visible GPUs:"
nvidia-smi -L 2>&1 | sed 's/^/[common]   /' || echo "[common] nvidia-smi unavailable"

# Helper: short-circuit if a Stage A / A' output is already at the target
# keyword count. Avoids paying ~5 min of 70B model-load time on a cell that
# the dispatcher submitted broadly. Caller passes (jsonl_path, target_count,
# done_marker_path). On match: touches the done marker and exits 0 cleanly.
skip_if_at_max() {
  local jsonl="$1"
  local target="${2:-0}"
  local marker="${3:-}"
  [ -z "$target" ] && return 0
  [ "$target" -le 0 ] && return 0
  [ -f "$jsonl" ] || return 0
  local n
  n=$(wc -l < "$jsonl" 2>/dev/null | tr -d ' ')
  n=${n:-0}
  if [ "$n" -ge "$target" ]; then
    echo "[skip] $jsonl already has $n ≥ $target keywords; exiting 0 without loading model."
    if [ -n "$marker" ]; then
      mkdir -p "$(dirname "$marker")"
      touch "$marker"
    fi
    exit 0
  fi
}

# Historical callers still invoke this helper after a checkpoint or failure.
# Retain the interface, but never extend a budget by submitting another job.
# Missing work needs a new estimate and explicit allocation-time approval.
chain_resubmit() {
  printf '%s\n' '[chain] No automatic resubmission. Preserve checkpoints and obtain fresh wall-time approval for missing work.'
  return 1
}
