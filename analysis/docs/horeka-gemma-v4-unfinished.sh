#!/bin/bash
# Labelled Gemma SI-v4 completion of one run's unfinished cells (Valerian, 2026-10-09): cells whose answer map never
# ran or whose sources were not all judged, e.g. in shards that used their whole allocation budget. Same protocol and
# settings as the source run (two map validation attempts); failed maps are left to the map-recovery launcher.
# The dataset and model come from the source run's preparation.json. One CPU preparation (exact cell list), then
# five-hour GPU bouts under the shared 200-bout ceiling. Results go to a new run root; the source run is never changed.
# Usage: bash horeka-gemma-v4-unfinished.sh SOURCE_RUN OUTPUT
set -euo pipefail
set +x
[ "$#" -eq 2 ] || { echo 'Usage: SOURCE_RUN OUTPUT' >&2; exit 2; }
test -z "${SLURM_JOB_ID:-}" || { echo 'Use a login shell.' >&2; exit 2; }
SRC=$1 OUT=$2
source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout $CODE" >&2; exit 2; }
SESSION=gemma-v4-unfinished
if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "Existing sender preserved: tmux attach -t $SESSION"; exit 0; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE"
SOURCE=$("$RT/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1]))["sources"]; assert len(s)==1, s; print(s[0])' "$SRC/preparation.json")
PLAN_ID=$("$RT/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_id"])' "$SRC/plan.json")
mkdir -p "$OUT"
CELLS="$OUT/unfinished-cells.txt"
if [ ! -f "$CELLS" ]; then
  "$RT/bin/python" -u "$CODE/analysis/scripts/select_gemma_v4_failed_maps.py" --run "$SRC" --output "$CELLS" --unfinished
fi
ARGS=("$RT/bin/python" -u "$CODE/analysis/scripts/horeka_gemma_v4.py" start --workspace "$W" --source "$SOURCE"
  --output "$OUT" --account hk-project-p0026831 --prepare-on-cpu --cells "$CELLS" --recovery-of "$PLAN_ID")
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
while [ -z "\${HF_TOKEN:-}" ]; do read -rsp 'HF write token (hidden, memory only; Enter alone asks again): ' HF_TOKEN; echo; done
export HF_TOKEN
echo "Gemma unfinished-cell completion started \$(date -u +%FT%TZ) from \$(git -C $CODE_Q rev-parse HEAD)" | tee -a $LOG
echo 'Detach with Ctrl-b d; the sender keeps running.'
exec $COMMAND >> $LOG 2>&1
EOF
tmux new-session -d -s "$SESSION" "bash $(printf %q "$OUT/sender-command.sh")"
echo "Completion of $PLAN_ID ($SOURCE): $(wc -l < "$CELLS") cells. Log: $OUT/sender.log"
if [ -n "${TMUX:-}" ]; then tmux switch-client -t "$SESSION"; else tmux attach -t "$SESSION"; fi
