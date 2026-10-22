#!/bin/bash
# Qwen BM25 production run: SHARDS parallel chains of JOBS accelerated jobs (full four-A100 node, 6 h each). Finite.
#   CODE=<clean checkout> BUNDLE=<bundle> RUN_ID=<id> BOUT_CONFIG=<qwen bout config.json> ACCOUNT=<account> \
#     bash submit_qwen_chains.sh [SHARDS] [JOBS]          (defaults 5 and 4: at most 20 jobs, 120 node-hours)
# Shard s takes the cells with order % SHARDS == s, so chains never repeat each other's work; all write to one run
# directory. The first job of shard s may start s x 10 minutes after submission (AGENTS.md: about ten minutes between
# starts, at most five allocations at once). Within a chain each job waits for the previous one with afternotok:
# exit 4 (deadline checkpoint) starts it, exit 0 (the shard is done) cancels the rest of the chain, any other failure
# writes CHAIN_STOP and every later job exits at once. Optional: TIME (06:00:00), PHASES (64:128), DEVICE (cuda-pool),
# AFTER (a job id the first job of every chain waits for, afterany). Every submission is appended to submissions.tsv.
set -euo pipefail
W=${GEODML_WORKSPACE:-/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen}
CODE=${CODE:?}; BUNDLE=${BUNDLE:?}; RUN_ID=${RUN_ID:?}; BOUT_CONFIG=${BOUT_CONFIG:?}; ACCOUNT=${ACCOUNT:?}
SHARDS=${1:-5}; JOBS=${2:-4}
TIME=${TIME:-06:00:00}; PHASES=${PHASES:-64:128}; DEVICE=${DEVICE:-cuda-pool}; AFTER=${AFTER:-}
[ "$SHARDS" -ge 1 ] && [ "$SHARDS" -le 5 ] || { echo "SHARDS must be 1..5 (at most five allocations at once)" >&2; exit 2; }
[ "$JOBS" -ge 1 ] && [ "$JOBS" -le 8 ] || { echo "JOBS must be 1..8" >&2; exit 2; }
test -z "$(git -C "$CODE" status --porcelain --untracked-files=all)" || { echo "dirty checkout: $CODE" >&2; exit 2; }
for p in "$BUNDLE/manifest.json" "$BOUT_CONFIG" "$CODE/analysis/sigir_bm25/run_qwen_bm25.sbatch"; do
  test -e "$p" || { echo "missing $p" >&2; exit 2; }
done
RUN="$W/runs/sigir-bm25/$RUN_ID"
[ -s "$RUN/CHAIN_STOP" ] && { echo "$RUN/CHAIN_STOP exists; read it and delete it first" >&2; exit 2; }
mkdir -p "$RUN"
LOG="$RUN/submissions.tsv"
[ -s "$LOG" ] || printf 'submitted_utc\tshard\tshards\tjob_in_chain\tjob_id\tdepends_on\ttime\tphases\tdevice\tcommit\n' > "$LOG"
COMMIT=$(git -C "$CODE" rev-parse HEAD)
for s in $(seq 0 $((SHARDS - 1))); do
  prev=$AFTER
  for j in $(seq 1 "$JOBS"); do
    opts=()
    if [ "$j" -eq 1 ]; then
      [ "$s" -gt 0 ] && opts+=(--begin=now+$((s * 10))minutes)
      [ -n "$prev" ] && opts+=(--dependency=afterany:"$prev")
    else
      opts+=(--dependency=afternotok:"$prev" --kill-on-invalid-dep=yes)
    fi
    id=$(sbatch --parsable --account="$ACCOUNT" --partition=accelerated --time="$TIME" ${opts[@]+"${opts[@]}"} \
      --job-name="geodml-sigir-qwen-s$s" \
      --export="ALL,CODE=$CODE,BUNDLE=$BUNDLE,RUN_ID=$RUN_ID,BOUT_CONFIG=$BOUT_CONFIG,PHASES=$PHASES,DEVICE=$DEVICE,SHARD=$s,SHARDS=$SHARDS" \
      --output="$W/runs/sigir-bm25/$RUN_ID-s$s-%j.out" "$CODE/analysis/sigir_bm25/run_qwen_bm25.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$s" "$SHARDS" "$j" "$id" "${prev:--}" \
      "$TIME" "$PHASES" "$DEVICE" "$COMMIT" >> "$LOG"
    echo "shard $s job $j: $id${prev:+ (after $prev)}"
    prev=$id
  done
done
echo "submitted $((SHARDS * JOBS)) jobs; record: $LOG"
