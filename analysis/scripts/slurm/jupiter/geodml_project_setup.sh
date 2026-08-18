#!/usr/bin/env bash
# Base login-shell setup for geodml on JSC systems.
#
# Install this file in a private location such as ~/geodml_setup.sh, then use:
#
#   source ~/geodml_setup.sh
#
# The script must be sourced so that its exported variables and working-directory
# change remain in the current shell. It does not load compute modules, activate
# a model virtual environment, fetch Git changes, or submit work.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This setup must be sourced: source $0" >&2
    exit 2
fi

_geodml_project_setup() {
    local project_name="${JUPITER_PROJECT:-scifi}"
    local branch
    local commit
    local worktree_state

    if ! command -v jutil >/dev/null 2>&1; then
        echo "geodml setup: jutil is unavailable; run this on a JSC login node" >&2
        return 2
    fi

    if ! jutil env activate -p "$project_name"; then
        echo "geodml setup: could not activate JSC project $project_name" >&2
        return 2
    fi

    if [[ -z "${PROJECT:-}" || -z "${FSCRATCH:-}" || -z "${USER:-}" ]]; then
        echo "geodml setup: PROJECT, FSCRATCH, and USER must be set by jutil" >&2
        return 2
    fi

    export JUPITER_PROJECT="$project_name"
    export JUPITER_ACCOUNT="${JUPITER_ACCOUNT:-$project_name}"
    export GEODML_PROJECT_ROOT="${GEODML_PROJECT_ROOT:-$PROJECT/$USER/geodml}"
    export GEODML_REPOSITORY="${GEODML_REPOSITORY:-$GEODML_PROJECT_ROOT/src/geodml-mono}"
    export GEODML_CACHE_ROOT="${GEODML_CACHE_ROOT:-$FSCRATCH/$USER/geodml}"
    export GEODML_RUNS_ROOT="${GEODML_RUNS_ROOT:-$GEODML_PROJECT_ROOT/runs}"
    export GEODML_MODELS_ROOT="${GEODML_MODELS_ROOT:-$GEODML_PROJECT_ROOT/models}"
    export GEODML_STAGING_ROOT="${GEODML_STAGING_ROOT:-$GEODML_PROJECT_ROOT/staging}"
    export GEODML_RESTRICTED_DATA_ROOT="${GEODML_RESTRICTED_DATA_ROOT:-$GEODML_PROJECT_ROOT/restricted-data}"
    export GEODML_MODEL_VENV="${GEODML_MODEL_VENV:-$GEODML_CACHE_ROOT/python/.venv-model-panel-transformers5141}"

    if ! git -C "$GEODML_REPOSITORY" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "geodml setup: repository not found at $GEODML_REPOSITORY" >&2
        return 2
    fi

    if ! cd "$GEODML_REPOSITORY"; then
        echo "geodml setup: could not enter $GEODML_REPOSITORY" >&2
        return 2
    fi

    branch="$(git branch --show-current)"
    commit="$(git rev-parse HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then
        worktree_state="dirty"
    else
        worktree_state="clean"
    fi
    if [[ -z "$branch" ]]; then
        branch="detached HEAD"
    fi

    echo "geodml environment ready"
    echo "  project:    $JUPITER_PROJECT"
    echo "  repository: $GEODML_REPOSITORY"
    echo "  branch:     $branch"
    echo "  commit:     $commit"
    echo "  worktree:   $worktree_state"
    echo "  runs:       $GEODML_RUNS_ROOT"
    echo "  models:     $GEODML_MODELS_ROOT"
}

_geodml_project_setup
_geodml_setup_status=$?
unset -f _geodml_project_setup
if [[ "$_geodml_setup_status" -ne 0 ]]; then
    unset _geodml_setup_status
    return 2
fi
unset _geodml_setup_status
