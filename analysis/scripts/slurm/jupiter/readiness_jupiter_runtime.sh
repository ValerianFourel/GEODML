#!/bin/bash
# Shared JUPITER Python setup for readiness controllers.

readiness_clear_inherited_python_runtime() {
    local inherited="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin}" cleaned="" entry
    local entries=()
    if [[ -n "$inherited" ]]; then
        IFS=: read -r -a entries <<< "$PATH"
        for entry in "${entries[@]}"; do
            [[ "$entry" == "$inherited" ]] && continue
            cleaned="${cleaned:+$cleaned:}$entry"
        done
        export PATH="$cleaned"
    fi
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
    hash -r
}

readiness_load_jupiter_stack() {
    local restore_nounset=0
    if ! type module >/dev/null 2>&1; then
        [[ $- == *u* ]] && restore_nounset=1
        set +u
        source /etc/profile
        [[ "$restore_nounset" -eq 1 ]] && set -u
    fi
    module --force purge
    module load Stages/2026
    module load GCCcore/14.3.0
    module load SciPy-Stack/2025b
    module load git
    module load PyTorch/2.9.1
    jutil env activate -p "${JUPITER_PROJECT:-scifi}"
    hash -r
}

readiness_bootstrap_jupiter_control_runtime() {
    local success_message="${1:-READINESS_JUPITER_CONTROL_RUNTIME=PASS}"
    readiness_clear_inherited_python_runtime
    readiness_load_jupiter_stack
    python3 -c 'import json, pathlib'
    printf '%s\n' "$success_message"
}
