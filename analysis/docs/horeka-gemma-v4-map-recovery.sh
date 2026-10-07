#!/bin/bash
# Labelled Gemma SI-v4 map recovery for one finished run: re-judge only the cells whose answer map failed,
# allowing up to ATTEMPTS corrective map attempts (new seed and error feedback each time), then their sources.
# One CPU preparation (exact cell list), then five-hour GPU bouts; its own small sender lock, so it runs next to
# the main Gemma sender. Results go to a new run root; the source run is never changed.
# Usage: bash horeka-gemma-v4-map-recovery.sh SOURCE_RUN MODEL DATASET OUTPUT [ATTEMPTS]
set -euo pipefail
set +x
[ "$#" -ge 4 ] || { echo 'Usage: SOURCE_RUN MODEL(llama4|qwen38) DATASET OUTPUT [ATTEMPTS, default 4]' >&2; exit 2; }
test -z "${SLURM_JOB_ID:-}" || { echo 'Use a login shell.' >&2; exit 2; }
SRC=$1 MODEL=$2 DATA=$3 OUT=$4 ATTEMPTS=${5:-4}
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout $CODE" >&2; exit 2; }
SESSION=gemma-v4-map-recovery
if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "Existing recovery sender preserved: tmux attach -t $SESSION"; exit 0; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
mkdir -p "$OUT"
CELLS="$OUT/failed-map-cells.txt"
if [ ! -f "$CELLS" ]; then
  "$RT/bin/python" -u "$CODE/analysis/scripts/select_gemma_v4_failed_maps.py" --run "$SRC" --output "$CELLS"
fi
PLAN_ID=$("$RT/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_id"])' "$SRC/plan.json")
ARGS=("$RT/bin/python" -u "$CODE/analysis/scripts/horeka_gemma_v4.py" start --workspace "$W" --source "$DATA:$MODEL"
  --output "$OUT" --account hk-project-p0026831 --prepare-on-cpu
  --cells "$CELLS" --map-validation-attempts "$ATTEMPTS" --recovery-of "$PLAN_ID")
printf -v COMMAND '%q ' "${ARGS[@]}"
printf -v CODE_Q '%q' "$CODE"
printf -v LOG '%q' "$OUT/sender.log"
cat > "$OUT/sender-command.sh" <<EOF
#!/bin/bash
set -euo pipefail
set +x
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$CODE_Q
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
if [ -z "\${HF_TOKEN:-}" ]; then read -rsp 'HF write token (hidden, memory only): ' HF_TOKEN; echo; export HF_TOKEN; fi
echo "Gemma map recovery started \$(date -u +%FT%TZ) from \$(git -C $CODE_Q rev-parse HEAD)" | tee -a $LOG
echo 'Detach with Ctrl-b d; the sender keeps running.'
exec $COMMAND >> $LOG 2>&1
EOF
tmux new-session -d -s "$SESSION" "bash $(printf %q "$OUT/sender-command.sh")"
echo "Recovery of $PLAN_ID: $(wc -l < "$CELLS") cells, up to $ATTEMPTS map attempts. Log: $OUT/sender.log"
if [ -n "${TMUX:-}" ]; then tmux switch-client -t "$SESSION"; else tmux attach -t "$SESSION"; fi
