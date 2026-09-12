#!/usr/bin/env bash
set -euo pipefail

: "${GEODML_EXPECTED_JOB_ID:?}"
: "${GEODML_EXECUTION_COMMIT:?}"
: "${GEODML_EXECUTION_REPOSITORY:?}"
: "${SEARCH_AGENTIC_CALIBRATION_ROOT:?}"
: "${SEARCH_AGENTIC_SELECTION_ROOT:?}"
: "${SEARCH_PILOT_ROOT:?}"
: "${SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_CROSS_ENCODER_REVISION:?}"
: "${SEARCH_AGENTIC_DDG_SNAPSHOT:?}"
: "${SEARCH_AGENTIC_SEARXNG_SNAPSHOT:?}"

test "${SLURM_JOB_ID:?}" = "$GEODML_EXPECTED_JOB_ID"
test "$GEODML_EXECUTION_COMMIT" = "$(git -C "$GEODML_EXECUTION_REPOSITORY" rev-parse HEAD)"
test -z "$(git -C "$GEODML_EXECUTION_REPOSITORY" status --porcelain --untracked-files=all)"
test -s "$SEARCH_AGENTIC_SELECTION_ROOT/selection-manifest.json"
test -s "$SEARCH_AGENTIC_SELECTION_ROOT/pilot-prompts.jsonl"
test -s "$SEARCH_AGENTIC_SELECTION_ROOT/selection-records.jsonl"
test -d "$SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT"
test -s "$SEARCH_AGENTIC_DDG_SNAPSHOT"
test -s "$SEARCH_AGENTIC_SEARXNG_SNAPSHOT"

mkdir -p "$SEARCH_AGENTIC_CALIBRATION_ROOT/logs"

python3 - \
  "$SEARCH_AGENTIC_SELECTION_ROOT/selection-manifest.json" \
  "$SEARCH_AGENTIC_CALIBRATION_ROOT/config.json" <<'PY'
import json
import os
import sys
from pathlib import Path

selection_path = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
selection = json.loads(selection_path.read_text())
assert selection["format_version"] == "readiness-axis-balanced-pilot-v1"
assert selection["selection_id"] == "610d4e26814bc0a202300b54729dfc59cb7a526018115f0b56927dde5b524ed0"
assert selection["diagnostics"]["sample_size"] == 500

config = {
    "format_version": "agentic-search-25-prompt-execution-calibration-v1",
    "scientific_result": False,
    "purpose": "execution compatibility and throughput calibration",
    "git_commit": os.environ["GEODML_EXECUTION_COMMIT"],
    "resources": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "512G"},
    "selection_manifest": str(selection_path),
    "selection_id": selection["selection_id"],
    "prompt_count": 25,
    "prompt_selection_seed": 20260912,
    "models": [
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ],
    "methods": ["Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"],
    "engines": ["duckduckgo", "searxng"],
    "conditions": ["natural", "ablated", "shuffled"],
    "cells_per_model": 300,
    "total_cells": 900,
    "retrieval_mode": "frozen-snapshot-deterministic-lexical-v1",
    "condition_mode": "smoke-only-subset-and-order-v1",
    "baseline_started": False,
    "judge_started": False,
}
serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
if config_path.exists():
    assert config_path.read_text() == serialized, "existing calibration config differs"
else:
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(serialized)
    temporary.replace(config_path)
attempt = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "approved_walltime": os.environ.get("ACL_ARR_APPROVED_WALLTIME"),
    "allocation_estimate": os.environ.get("ACL_ARR_ALLOCATION_ESTIMATE"),
    "infrastructure_risk": os.environ.get("ACL_ARR_INFRASTRUCTURE_RISK"),
}
attempt_path = config_path.parent / "attempts" / f"job-{attempt['slurm_job_id']}.json"
attempt_path.parent.mkdir(parents=True, exist_ok=True)
attempt_serialized = json.dumps(attempt, indent=2, sort_keys=True) + "\n"
if attempt_path.exists():
    assert attempt_path.read_text() == attempt_serialized, "allocation attempt differs"
else:
    attempt_path.write_text(attempt_serialized)
print("AGENTIC_25_PROMPT_CONFIG_GATE=PASS")
PY

export SEARCH_AGENTIC_PROMPTS_JSONL="$SEARCH_AGENTIC_SELECTION_ROOT/pilot-prompts.jsonl"
export SEARCH_AGENTIC_SELECTION_RECORDS_JSONL="$SEARCH_AGENTIC_SELECTION_ROOT/selection-records.jsonl"
export SEARCH_AGENTIC_PROMPT_COUNT=25
export SEARCH_AGENTIC_PROMPT_SELECTION_SEED=20260912
export SEARCH_AGENTIC_EXPECTED_CELL_COUNT=300
export SEARCH_AGENTIC_REQUEST_CONCURRENCY="${SEARCH_AGENTIC_REQUEST_CONCURRENCY:-4}"

run_model() {
  local slug="$1"
  local profile="$2"
  local launcher="$3"
  local attempt_record
  export SEARCH_AGENTIC_OUTPUT="$SEARCH_AGENTIC_CALIBRATION_ROOT/models/$slug"
  if python3 - "$SEARCH_AGENTIC_OUTPUT/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text())
if not (
    value.get("status") == "complete"
    and value.get("cell_count") == 300
    and value.get("completed_count") == 300
    and value.get("remaining_count") == 0
    and value.get("scientific_result") is False
):
    raise SystemExit(1)
PY
  then
    printf 'CALIBRATION_MODEL_ALREADY_COMPLETE=%s\n' "$slug"
    return
  fi
  attempt_record="$(mktemp "$SEARCH_AGENTIC_CALIBRATION_ROOT/logs/${slug}.XXXXXX")"
  export SEARCH_AGENTIC_PROFILE="$profile"
  export SEARCH_AGENTIC_SERVER_LOG="$attempt_record.server.log"
  export SEARCH_AGENTIC_GPU_TELEMETRY="$attempt_record.gpu.csv"
  printf 'MODEL=%s\nOUTPUT=%s\nSERVER_LOG=%s\nGPU_TELEMETRY=%s\n' \
    "$slug" "$SEARCH_AGENTIC_OUTPUT" "$SEARCH_AGENTIC_SERVER_LOG" \
    "$SEARCH_AGENTIC_GPU_TELEMETRY" > "$attempt_record"
  printf 'CALIBRATION_MODEL_START=%s\nATTEMPT_RECORD=%s\n' "$slug" "$attempt_record"
  bash "$launcher"
  printf 'CALIBRATION_MODEL_COMPLETE=%s\n' "$slug"
}

run_model \
  qwen38 \
  "$SEARCH_PILOT_ROOT/primary-answer2048-smoke-39aaf746b484/model-config-a58eb7c40e545350abc1.serving-profile.json" \
  "$GEODML_EXECUTION_REPOSITORY/analysis/scripts/slurm/jupiter/run_agentic_search_qwen38_smoke.sh"

run_model \
  qwen25 \
  "$SEARCH_PILOT_ROOT/primary-answer2048-hf-overrides-a5648c6cd160/model-config-fbe629168c7384f96048.serving-profile.json" \
  "$GEODML_EXECUTION_REPOSITORY/analysis/scripts/slurm/jupiter/run_agentic_search_qwen25_smoke.sh"

run_model \
  llama4 \
  "$SEARCH_PILOT_ROOT/primary-answer2048-smoke-39aaf746b484/model-config-2403370677dae49f8cd1.serving-profile.json" \
  "$GEODML_EXECUTION_REPOSITORY/analysis/scripts/slurm/jupiter/run_agentic_search_llama4_smoke.sh"

python3 - "$SEARCH_AGENTIC_CALIBRATION_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
models = {}
for slug in ("qwen38", "qwen25", "llama4"):
    path = root / "models" / slug / "run_manifest.json"
    value = json.loads(path.read_text())
    assert value["status"] == "complete", value
    assert value["cell_count"] == 300, value
    assert value["completed_count"] == 300, value
    assert value["remaining_count"] == 0, value
    assert value["scientific_result"] is False, value
    models[slug] = {"manifest": str(path), "config_sha256": value["config_sha256"]}
manifest = {
    "format_version": "agentic-search-25-prompt-execution-calibration-v1",
    "status": "complete",
    "scientific_result": False,
    "git_commit": os.environ["GEODML_EXECUTION_COMMIT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "prompt_count": 25,
    "model_count": 3,
    "cell_count": 900,
    "completed_count": 900,
    "remaining_count": 0,
    "models": models,
    "baseline_started": False,
    "judge_started": False,
}
temporary = root / "run_manifest.json.tmp"
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "run_manifest.json")
print("AGENTIC_25_PROMPT_CALIBRATION=PASS")
print("SUMMARY=" + json.dumps({
    key: manifest[key]
    for key in ("status", "prompt_count", "model_count", "cell_count", "completed_count")
}, sort_keys=True))
PY
