#!/bin/bash
# Gemma v4 top-up pass: judge only answers that no earlier pass froze.
# Every EXCLUDE dir is an earlier frozen SI-v4 input (e.g. a finished run's frozen/);
# its cells are skipped and cells sharing its answer maps are only reported.
# One four-hour CPU preparation, then the finite five-hour-bout sender in tmux
# SESSION. It waits for any other Gemma sender in this workspace before sending.
# Usage: bash horeka-gemma-v4-topup.sh WORKSPACE MODEL DATASET OUTPUT ACCOUNT SESSION EXCLUDE...
set -euo pipefail
set +x
if [ "$#" -lt 7 ]; then
  echo 'Usage: WORKSPACE MODEL(qwen38|llama4) DATASET OUTPUT ACCOUNT SESSION EXCLUDE...' >&2
  exit 2
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo 'Use a login shell; preserve the live allocation.' >&2
  exit 2
fi
WORKSPACE=$1 MODEL=$2 DATASET=$3 OUTPUT=$4 ACCOUNT=$5 SESSION=$6
shift 6
case "$MODEL" in qwen38|llama4) ;; *) echo 'MODEL must be qwen38 or llama4' >&2; exit 2;; esac
[[ "$SESSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'bad session name' >&2; exit 2; }
source "$WORKSPACE/geodml-nemotron-env.sh"
: "${RT:?The existing environment must define RT}"
command -v tmux >/dev/null
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "Existing sender preserved. Inspect: tmux attach -t $SESSION"
  exit 0
fi
ARGS=("$RT/bin/python" -u "$CODE/analysis/scripts/horeka_gemma_v4.py" start
  --workspace "$WORKSPACE" --source "$DATASET:$MODEL" --output "$OUTPUT" --account "$ACCOUNT" --prepare-on-cpu)
for EXCLUDE in "$@"; do
  test -f "$EXCLUDE/manifest.json" || { echo "not a frozen input: $EXCLUDE" >&2; exit 2; }
  ARGS+=(--exclude-inputs "$EXCLUDE")
done
mkdir -p "$OUTPUT"
printf -v COMMAND '%q ' "${ARGS[@]}"
printf -v ENV_Q '%q' "$WORKSPACE/geodml-nemotron-env.sh"
printf -v CODE_Q '%q' "$CODE"
printf -v LOG '%q' "$OUTPUT/sender.log"
# The HF write token is typed hidden inside the session: never on disk or in argv.
cat > "$OUTPUT/sender-command.sh" <<EOF
#!/bin/bash
set -euo pipefail
set +x
source $ENV_Q
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$CODE_Q
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
if [ -z "\${HF_TOKEN:-}" ]; then
  read -rsp 'HF write token (hidden, memory only): ' HF_TOKEN
  echo
  export HF_TOKEN
fi
echo "Gemma $MODEL top-up sender started \$(date -u +%FT%TZ) from \$(git -C $CODE_Q rev-parse HEAD)" | tee -a $LOG
echo 'Detach with Ctrl-b d; the sender keeps running.'
exec $COMMAND >> $LOG 2>&1
EOF
printf -v LAUNCHER '%q' "$OUTPUT/sender-command.sh"
tmux new-session -d -s "$SESSION" "bash $LAUNCHER"
printf 'Log: %s/sender.log\n' "$OUTPUT"
if [ -n "${TMUX:-}" ]; then tmux switch-client -t "$SESSION"; else tmux attach -t "$SESSION"; fi
