#!/usr/bin/env bash
# Create the reconnect-safe environment file for the approved ACL ARR pilot.

set -euo pipefail
umask 077

EXPECTED_COMMIT="${1:?usage: $0 EXPECTED_GIT_COMMIT}"
ENVIRONMENT_FILE="${ACL_ARR_ENVIRONMENT_FILE:-$HOME/geodml-acl-arr-pilot.env}"

: "${GEODML_REPOSITORY:?set GEODML_REPOSITORY}"
: "${GEODML_PROJECT_ROOT:?set GEODML_PROJECT_ROOT}"
: "${GEODML_CACHE_ROOT:?set GEODML_CACHE_ROOT}"
: "${USER:?set USER}"

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: expected a 40-character Git commit SHA" >&2
    exit 2
fi

CURRENT_COMMIT="$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "ERROR: checkout $CURRENT_COMMIT does not match $EXPECTED_COMMIT" >&2
    exit 2
fi

AUDIT_ROOT="${AUDIT_ROOT:-}"
if [[ -z "$AUDIT_ROOT" ]]; then
    AUDIT_POINTER="$HOME/geodml-final-audit-latest.txt"
    [[ -s "$AUDIT_POINTER" ]] || {
        echo "ERROR: missing final-audit pointer $AUDIT_POINTER" >&2
        exit 2
    }
    AUDIT_ROOT="$(<"$AUDIT_POINTER")"
fi
[[ -s "$AUDIT_ROOT/compliant-candidates.jsonl" ]]
[[ -s "$AUDIT_ROOT/final-axis-map.jsonl" ]]

SEARCH_SNAPSHOT="${ACL_ARR_SEARCH_SNAPSHOT:-}"
if [[ -n "$SEARCH_SNAPSHOT" && ! -s "$SEARCH_SNAPSHOT" ]]; then
    SEARCH_SNAPSHOT=""
fi
if [[ -z "$SEARCH_SNAPSHOT" && -n "${GEODML_DATA_ROOT:-}" ]]; then
    CANDIDATE="$GEODML_DATA_ROOT/data/serp/phase0_top20_searxng.parquet"
    [[ ! -s "$CANDIDATE" ]] || SEARCH_SNAPSHOT="$CANDIDATE"
fi
for CANDIDATE in \
    "$GEODML_REPOSITORY/geodml_data/data/serp/phase0_top20_searxng.parquet" \
    "${PROJECT:-/path-not-set}/$USER/GEODML_Analysis/geodml_data/data/serp/phase0_top20_searxng.parquet" \
    "${SCRATCH:-/path-not-set}/$USER/data/serp/phase0_top20_searxng.parquet" \
    "${SCRATCH:-/path-not-set}/data/serp/phase0_top20_searxng.parquet" \
    "/e/scratch/scifi/$USER/data/serp/phase0_top20_searxng.parquet" \
    "/p/scratch/scifi/$USER/data/serp/phase0_top20_searxng.parquet" \
    "${FSCRATCH:-/path-not-set}/$USER/geodml_data/data/serp/phase0_top20_searxng.parquet" \
    "$GEODML_CACHE_ROOT/geodml_data/data/serp/phase0_top20_searxng.parquet"
do
    if [[ -z "$SEARCH_SNAPSHOT" && -s "$CANDIDATE" ]]; then
        SEARCH_SNAPSHOT="$CANDIDATE"
    fi
done
if [[ -z "$SEARCH_SNAPSHOT" ]]; then
    for SEARCH_ROOT in \
        "${SCRATCH:-/path-not-set}/$USER" \
        "${SCRATCH:-/path-not-set}" \
        "/e/scratch/scifi/$USER" \
        "/p/scratch/scifi/$USER" \
        "$GEODML_PROJECT_ROOT" \
        "$GEODML_CACHE_ROOT"
    do
        [[ -d "$SEARCH_ROOT" ]] || continue
        CANDIDATE="$(
            find "$SEARCH_ROOT" -maxdepth 8 -type f \
                -name phase0_top20_searxng.parquet -print -quit 2>/dev/null || true
        )"
        if [[ -n "$CANDIDATE" ]]; then
            SEARCH_SNAPSHOT="$CANDIDATE"
            break
        fi
    done
fi
if [[ -z "$SEARCH_SNAPSHOT" ]]; then
    echo "ERROR: could not find phase0_top20_searxng.parquet" >&2
    exit 2
fi

GEODML_DATA_ROOT="${SEARCH_SNAPSHOT%/data/serp/phase0_top20_searxng.parquet}"
ACL_ARR_SEARCH_SNAPSHOT="$SEARCH_SNAPSHOT"
ACL_ARR_RUN_ROOT="${ACL_ARR_PILOT_RUN_ROOT:-$GEODML_PROJECT_ROOT/runs/acl-arr-document-pilot/pilot-128-${EXPECTED_COMMIT:0:7}}"
ACL_ARR_VENV="${ACL_ARR_PILOT_VENV:-$GEODML_CACHE_ROOT/python/.venv-acl-arr-vllm}"
HF_HOME="${ACL_ARR_HF_HOME:-$GEODML_CACHE_ROOT/huggingface-acl-arr}"
HF_HUB_CACHE="$HF_HOME/hub"
GEODML_EXPECTED_COMMIT="$EXPECTED_COMMIT"

export GEODML_EXPECTED_COMMIT GEODML_REPOSITORY GEODML_PROJECT_ROOT
export GEODML_CACHE_ROOT GEODML_DATA_ROOT AUDIT_ROOT
export ACL_ARR_SEARCH_SNAPSHOT ACL_ARR_RUN_ROOT ACL_ARR_VENV
export HF_HOME HF_HUB_CACHE

mkdir -p "$ACL_ARR_RUN_ROOT" "$HF_HUB_CACHE" "$(dirname "$ACL_ARR_VENV")"

TEMPORARY_FILE="${ENVIRONMENT_FILE}.tmp.$$"
trap 'rm -f "$TEMPORARY_FILE"' EXIT
declare -px \
    GEODML_EXPECTED_COMMIT GEODML_REPOSITORY GEODML_PROJECT_ROOT \
    GEODML_CACHE_ROOT GEODML_DATA_ROOT AUDIT_ROOT \
    ACL_ARR_SEARCH_SNAPSHOT ACL_ARR_RUN_ROOT ACL_ARR_VENV \
    HF_HOME HF_HUB_CACHE > "$TEMPORARY_FILE"
chmod 600 "$TEMPORARY_FILE"
mv "$TEMPORARY_FILE" "$ENVIRONMENT_FILE"
trap - EXIT

echo "ACL_ARR_ENVIRONMENT=PASS"
echo "commit=$GEODML_EXPECTED_COMMIT"
echo "audit_root=$AUDIT_ROOT"
echo "search_snapshot=$ACL_ARR_SEARCH_SNAPSHOT"
echo "run_root=$ACL_ARR_RUN_ROOT"
echo "environment_file=$ENVIRONMENT_FILE"
