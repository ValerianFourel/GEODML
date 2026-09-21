#!/usr/bin/env bash
set -euo pipefail

: "${GEODML_ADAPTIVE_PLAN:?}"
: "${GEODML_ADAPTIVE_WORKER_INDEX:?}"
: "${GEODML_ADAPTIVE_ROLE_ROOT:?}"
plan="$GEODML_ADAPTIVE_PLAN"
worker="$GEODML_ADAPTIVE_WORKER_INDEX"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd -P)"
cd "$repository"

json_value() {
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); [v:=v[k] for k in sys.argv[2:]]; print("" if v is None else v)' "$@"
}

paired_root="$(json_value "$plan" overflow paired root)"
if [[ -n "$paired_root" && "$(date +%s)" -lt "$((SLURM_JOB_END_TIME - 1020))" ]]; then
  import_marker="$(dirname "$plan")/claims/paired-qwen38-imported.json"
  import_view="$(dirname "$plan")/paired-import-view"
  if [[ "$worker" == 0 && ! -s "$import_marker" ]]; then
    exec 7>>"$(dirname "$plan")/.paired-import.lock"
    flock 7
    if [[ ! -s "$import_marker" ]]; then
      if [[ ! -d "$import_view" ]]; then
        python3 analysis/scripts/stage_agentic_generator_import_view.py \
          --source "$(json_value "$plan" overflow paired source_output)" \
          --output "$import_view"
      fi
      python3 analysis/scripts/run_agentic_search_integration_smoke.py \
        --output "$import_view" --base-url http://127.0.0.1:1/v1 \
        --model-id Qwen/Qwen3.8-27B \
        --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
        --cross-encoder-snapshot "$(json_value "$plan" overflow paired cross_encoder_snapshot)" \
        --cross-encoder-revision "$(json_value "$plan" overflow paired cross_encoder_revision)" \
        --search-snapshot "duckduckgo=$(json_value "$plan" overflow paired search_snapshots duckduckgo path)" \
        --search-snapshot "searxng=$(json_value "$plan" overflow paired search_snapshots searxng path)" \
        --seed 20260911 --query-max-tokens 256 --final-max-tokens 2048 \
        --request-concurrency 4 --cell-concurrency 12 --disable-thinking \
        --prompts-jsonl "$(json_value "$plan" overflow paired prompt_sources prompts_jsonl path)" \
        --selection-records-jsonl "$(json_value "$plan" overflow paired prompt_sources selection_records_jsonl path)" \
        --cell-ids-jsonl "$(json_value "$plan" overflow paired tasks path)" \
        --shared-claim-root "$(json_value "$plan" overflow paired claim_root)" \
        --worker-index 0 --worker-count 5 --prompt-count 120 \
        --prompt-selection-seed "$(json_value "$plan" overflow paired prompt_selection_seed)" \
        --prompt-shard-index 0 --prompt-shard-count 1 --production-conditions --import-only
      python3 - "$import_marker" <<'PY'
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({"format_version": "agentic-generator-import-v1",
    "status": "complete", "git_commit": os.environ["GEODML_EXECUTION_COMMIT"],
    "finished_at": datetime.now(timezone.utc).isoformat()}, sort_keys=True) + "\n")
temporary.replace(path)
PY
    fi
    flock -u 7
    exec 7>&-
  fi
  for _ in {1..60}; do
    [[ -s "$import_marker" ]] && break
    sleep 5
  done
  test -s "$import_marker"

  paired_role="$GEODML_ADAPTIVE_ROLE_ROOT/paired-qwen38"
  export SEARCH_AGENTIC_OUTPUT="$paired_role/output"
  export SEARCH_AGENTIC_PROFILE="$(json_value "$plan" overflow paired serving_profile)"
  export SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT="$(json_value "$plan" overflow paired cross_encoder_snapshot)"
  export SEARCH_AGENTIC_CROSS_ENCODER_REVISION="$(json_value "$plan" overflow paired cross_encoder_revision)"
  export SEARCH_AGENTIC_DDG_SNAPSHOT="$(json_value "$plan" overflow paired search_snapshots duckduckgo path)"
  export SEARCH_AGENTIC_SEARXNG_SNAPSHOT="$(json_value "$plan" overflow paired search_snapshots searxng path)"
  export SEARCH_AGENTIC_SERVER_LOG="$paired_role/server.log"
  export SEARCH_AGENTIC_GPU_TELEMETRY="$paired_role/gpu.csv"
  export SEARCH_AGENTIC_SHARED_CLAIM_ROOT="$(json_value "$plan" overflow paired claim_root)"
  export SEARCH_AGENTIC_WORKER_INDEX="$worker"
  export SEARCH_AGENTIC_WORKER_COUNT=5
  export SEARCH_AGENTIC_CELL_IDS_JSONL="$(json_value "$plan" overflow paired tasks path)"
  export SEARCH_AGENTIC_EXPECTED_CELL_COUNT=1440
  export SEARCH_AGENTIC_PROMPTS_JSONL="$(json_value "$plan" overflow paired prompt_sources prompts_jsonl path)"
  export SEARCH_AGENTIC_SELECTION_RECORDS_JSONL="$(json_value "$plan" overflow paired prompt_sources selection_records_jsonl path)"
  export SEARCH_AGENTIC_PROMPT_COUNT=120
  export SEARCH_AGENTIC_PROMPT_SELECTION_SEED="$(json_value "$plan" overflow paired prompt_selection_seed)"
  export SEARCH_AGENTIC_PROMPT_SHARD_INDEX=0
  export SEARCH_AGENTIC_PROMPT_SHARD_COUNT=1
  export SEARCH_AGENTIC_PRODUCTION_CONDITIONS=1
  export SEARCH_AGENTIC_REQUEST_CONCURRENCY=4
  export SEARCH_AGENTIC_CELL_CONCURRENCY=12
  mkdir -p "$paired_role"
  bash analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh
fi

if [[ "$(date +%s)" -ge "$((SLURM_JOB_END_TIME - 1020))" ]]; then
  printf 'OVERFLOW=CHECKPOINTED reason=allocation_deadline\n'
  exit 0
fi

if (( worker < 2 )); then
  slug=qwen38
  local_worker="$worker"
  model_workers=2
  launcher=analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh
else
  slug=llama4
  local_worker=$((worker - 2))
  model_workers=3
  launcher=analysis/scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh
fi

python3 - "$plan" "$slug" "$GEODML_ADAPTIVE_ROLE_ROOT/backlog.env" <<'PY'
import json, shlex, sys
from pathlib import Path
plan = json.load(open(sys.argv[1]))
matches = [item for item in plan["overflow"]["backlogs"] if item["slug"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one {sys.argv[2]} backlog, got {len(matches)}")
item = matches[0]
values = {
    "profile": item["serving_profile"], "claim": item["claim_root"],
    "tasks": item["tasks"]["path"], "cross": item["cross_encoder_snapshot"],
    "cross_revision": item["cross_encoder_revision"],
    "ddg": item["search_snapshots"]["duckduckgo"]["path"],
    "searxng": item["search_snapshots"]["searxng"]["path"],
    "prompts": item["prompt_sources"]["prompts_jsonl"]["path"],
    "selection": item["prompt_sources"]["selection_records_jsonl"]["path"],
    "seed": str(item["prompt_selection_seed"]),
}
path = Path(sys.argv[3]); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")
PY
source "$GEODML_ADAPTIVE_ROLE_ROOT/backlog.env"
backlog_role="$GEODML_ADAPTIVE_ROLE_ROOT/backlog-$slug"
export SEARCH_AGENTIC_OUTPUT="$backlog_role/output"
export SEARCH_AGENTIC_PROFILE="$profile"
export SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT="$cross"
export SEARCH_AGENTIC_CROSS_ENCODER_REVISION="$cross_revision"
export SEARCH_AGENTIC_DDG_SNAPSHOT="$ddg"
export SEARCH_AGENTIC_SEARXNG_SNAPSHOT="$searxng"
export SEARCH_AGENTIC_SERVER_LOG="$backlog_role/server.log"
export SEARCH_AGENTIC_GPU_TELEMETRY="$backlog_role/gpu.csv"
export SEARCH_AGENTIC_SHARED_CLAIM_ROOT="$claim"
export SEARCH_AGENTIC_WORKER_INDEX="$local_worker"
export SEARCH_AGENTIC_WORKER_COUNT="$model_workers"
export SEARCH_AGENTIC_CELL_IDS_JSONL="$tasks"
export SEARCH_AGENTIC_EXPECTED_CELL_COUNT=14400
export SEARCH_AGENTIC_PROMPTS_JSONL="$prompts"
export SEARCH_AGENTIC_SELECTION_RECORDS_JSONL="$selection"
export SEARCH_AGENTIC_PROMPT_COUNT=1200
export SEARCH_AGENTIC_PROMPT_SELECTION_SEED="$seed"
export SEARCH_AGENTIC_PROMPT_SHARD_INDEX=0
export SEARCH_AGENTIC_PROMPT_SHARD_COUNT=1
export SEARCH_AGENTIC_PRODUCTION_CONDITIONS=1
export SEARCH_AGENTIC_REQUEST_CONCURRENCY=4
export SEARCH_AGENTIC_CELL_CONCURRENCY=12
mkdir -p "$backlog_role"
bash "$launcher"
