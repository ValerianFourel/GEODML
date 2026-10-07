#!/bin/bash
# Gemma v4 pass over the Qwen answers saved at freeze time: one four-hour cpuonly
# preparation, then the finite five-hour-bout sender (200 in flight, 600 s polls).
# Usage: bash horeka-gemma-v4-qwen.sh WORKSPACE QWEN_DATASET OUTPUT ACCOUNT PRIOR_V4_INPUTS
set -euo pipefail
set +x
if [ "$#" -ne 5 ]; then
  echo 'Usage: WORKSPACE QWEN_DATASET OUTPUT ACCOUNT PRIOR_V4_INPUTS' >&2
  exit 2
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo 'Use a login shell; preserve the live allocation.' >&2
  exit 2
fi
source "$1/geodml-nemotron-env.sh"
: "${RT:?The existing environment must define RT}"
command -v tmux >/dev/null
GEMMA_CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
GEMMA_SESSION=gemma-v4-qwen
if tmux has-session -t "=$GEMMA_SESSION" 2>/dev/null; then
  echo "Existing Qwen sender preserved. Inspect: tmux attach -t $GEMMA_SESSION"
  exit 0
fi
# One Gemma sender per workspace holds the sender lock; let the llama one finish.
if tmux has-session -t '=gemma-v4-bouts' 2>/dev/null; then
  echo 'The gemma-v4-bouts sender session still exists; let it finish first.' >&2
  exit 1
fi
mkdir -p "$3"
GEMMA_ARGS=("$RT/bin/python" -u "$GEMMA_CODE/analysis/scripts/horeka_gemma_v4.py" start
  --workspace "$1" --source "$2:qwen38" --output "$3" --account "$4"
  --exclude-inputs "$5" --prepare-on-cpu)
printf -v GEMMA_COMMAND '%q ' "${GEMMA_ARGS[@]}"
printf -v GEMMA_ENV '%q' "$1/geodml-nemotron-env.sh"
printf -v GEMMA_CODE_Q '%q' "$GEMMA_CODE"
printf -v GEMMA_LOG '%q' "$3/sender.log"
printf -v GEMMA_LAUNCHER '%q' "$3/sender-command.sh"
# The HF write token is typed hidden inside the session: never on disk or in argv.
cat > "$3/sender-command.sh" <<EOF
#!/bin/bash
set -euo pipefail
set +x
source $GEMMA_ENV
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$GEMMA_CODE_Q
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
while [ -z "\${HF_TOKEN:-}" ]; do read -rsp 'HF write token (hidden, memory only; Enter alone asks again): ' HF_TOKEN; echo; done
export HF_TOKEN
echo "Gemma Qwen sender started \$(date -u +%FT%TZ) from \$(git -C $GEMMA_CODE_Q rev-parse HEAD)" | tee -a $GEMMA_LOG
echo 'Detach with Ctrl-b d; the sender keeps running.'
exec $GEMMA_COMMAND >> $GEMMA_LOG 2>&1
EOF
tmux new-session -d -s "$GEMMA_SESSION" "bash $GEMMA_LAUNCHER"
printf 'Log: %s/sender.log\nStatus: %s/sender/status.json\n' "$3" "$3"
if [ -n "${TMUX:-}" ]; then
  tmux switch-client -t "$GEMMA_SESSION"
else
  tmux attach -t "$GEMMA_SESSION"
fi
