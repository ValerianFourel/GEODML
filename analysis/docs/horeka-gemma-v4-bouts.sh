#!/bin/bash
# Run from a separate HoreKa login shell after sourcing the existing environment.
# Usage: bash horeka-gemma-v4-bouts.sh WORKSPACE QWEN_DATASET LLAMA_DATASET OUTPUT ACCOUNT PRIOR_V4_INPUTS
set -euo pipefail
set +x

if [ "$#" -ne 6 ]; then
  echo 'Usage: WORKSPACE QWEN_DATASET LLAMA_DATASET OUTPUT ACCOUNT PRIOR_V4_INPUTS' >&2
  exit 2
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo 'Run this sender in a separate login shell; preserve the live allocation.' >&2
  exit 2
fi
source "$1/geodml-nemotron-env.sh"
: "${RT:?The existing environment must define RT}"
command -v tmux >/dev/null
GEMMA_CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
GEMMA_ROOT=$4
GEMMA_SESSION=gemma-v4-bouts
mkdir -p "$GEMMA_ROOT"
if tmux has-session -t "$GEMMA_SESSION" 2>/dev/null; then
  echo "Existing sender preserved. Inspect: tmux attach -t $GEMMA_SESSION"
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GEMMA_CODE"
GEMMA_ARGS=("$RT/bin/python" -u "$GEMMA_CODE/analysis/scripts/horeka_gemma_v4.py" start
  --workspace "$1" --source "$2:qwen38" --source "$3:llama4"
  --output "$GEMMA_ROOT" --account "$5" --exclude-inputs "$6")
printf -v GEMMA_COMMAND '%q ' "${GEMMA_ARGS[@]}"
printf -v GEMMA_LOG '%q' "$GEMMA_ROOT/sender.log"
printf -v GEMMA_ENV '%q' "$1/geodml-nemotron-env.sh"
printf -v GEMMA_LAUNCHER '%q' "$GEMMA_ROOT/sender-command.sh"
printf '#!/bin/bash\nset -euo pipefail\nset +x\nsource %s\nexport PYTHONDONTWRITEBYTECODE=1\nunset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE\nexec %s >> %s 2>&1\n' \
  "$GEMMA_ENV" "$GEMMA_COMMAND" "$GEMMA_LOG" > "$GEMMA_ROOT/sender-command.sh"
# Source the environment in the new window as well: an existing tmux server
# can otherwise retain old library paths and offline flags from an earlier job.
tmux new-session -d -s "$GEMMA_SESSION" "bash $GEMMA_LAUNCHER"
printf 'Sender: tmux attach -t %s\nLog: %s/sender.log\nStatus: %s/sender/status.json\n' \
  "$GEMMA_SESSION" "$GEMMA_ROOT" "$GEMMA_ROOT"
