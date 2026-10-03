#!/bin/bash
# Cancel the user's jobs except one explicitly preserved ID, then restart Gemma.
# Usage: WORKSPACE RUN PRESERVED_JOB LOGIN_HOST [LOGIN_HOST...]
set -euo pipefail
set +x
if [ "$#" -lt 4 ] || ! [[ "$3" =~ ^[0-9]+$ ]]; then
  echo 'Usage: WORKSPACE RUN PRESERVED_JOB LOGIN_HOST [LOGIN_HOST...]' >&2
  exit 2
fi
test -z "${SLURM_JOB_ID:-}" || { echo 'Use a login shell.' >&2; exit 2; }
GEMMA_WORKSPACE=$1
GEMMA_RUN=$2
GEMMA_KEEP=$3
shift 3
source "$GEMMA_WORKSPACE/geodml-nemotron-env.sh"
GEMMA_CODE=$(cd -- "$(dirname -- "$0")/../.." && pwd)
GEMMA_SCRIPT="$GEMMA_CODE/analysis/docs/horeka-gemma-v4-prequeue.sh"
# Qwen senders and the preserved job are never stopped here.
for GEMMA_HOST in "$@"; do
  [[ "$GEMMA_HOST" =~ ^[a-zA-Z0-9.-]+$ ]] || exit 2
  if [ "$(hostname -s)" = "$GEMMA_HOST" ]; then
    bash "$GEMMA_SCRIPT" --stop-local
  else
    printf -v GEMMA_REMOTE 'bash %q --stop-local' "$GEMMA_SCRIPT"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$GEMMA_HOST" "$GEMMA_REMOTE"
  fi
done
GEMMA_QUEUE=$(squeue --me --array --noheader --format=%i)
GEMMA_CANCEL=()
while read -r GEMMA_JOB; do
  [ -n "$GEMMA_JOB" ] || continue
  [[ "$GEMMA_JOB" =~ ^[0-9]+(_[0-9]+)?(\+[0-9]+)?$ ]] || exit 2
  if [ "$GEMMA_JOB" != "$GEMMA_KEEP" ]; then
    GEMMA_CANCEL+=("$GEMMA_JOB")
  fi
done <<< "$GEMMA_QUEUE"
if [ "${#GEMMA_CANCEL[@]}" -gt 0 ]; then
  printf 'Cancelling %s jobs; preserving job %s.\n' "${#GEMMA_CANCEL[@]}" "$GEMMA_KEEP"
  scancel "${GEMMA_CANCEL[@]}"
fi
for GEMMA_TRY in {1..10}; do
  GEMMA_QUEUE=$(squeue --me --array --noheader --format=%i)
  GEMMA_LEFT=0
  while read -r GEMMA_JOB; do
    if [ -n "$GEMMA_JOB" ] && [ "$GEMMA_JOB" != "$GEMMA_KEEP" ]; then
      GEMMA_LEFT=$((GEMMA_LEFT + 1))
    fi
  done <<< "$GEMMA_QUEUE"
  if [ "$GEMMA_LEFT" -eq 0 ]; then
    bash "$GEMMA_SCRIPT" "$GEMMA_WORKSPACE" "$GEMMA_RUN" --restart-after-cancel "$GEMMA_KEEP"
    exit 0
  fi
  sleep 30
done
echo 'Cancellation is still settling. Saved work preserved; no relaunch yet.' >&2
exit 1
