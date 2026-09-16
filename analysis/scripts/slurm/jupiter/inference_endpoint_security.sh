#!/usr/bin/env bash
# Source after setting REPOSITORY_ROOT. Never put the ephemeral key in argv,
# profiles, manifests, or traced shell assignments. Children inherit it directly.
set +x
umask 077

geodml_init_inference_endpoint() {
    set +x
    umask 077
    if [[ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]]; then
        echo "ERROR: direct inference endpoints require one local node" >&2
        return 2
    fi
    VLLM_API_KEY="$(python3 "$REPOSITORY_ROOT/analysis/scripts/inference_endpoint_security.py" new-key)" || return 2
    if [[ -z "$VLLM_API_KEY" ]]; then
        echo "ERROR: no inference endpoint credential was generated" >&2
        return 2
    fi
    export VLLM_API_KEY GEODML_INFERENCE_AUTH_REQUIRED=1 VLLM_HOST_IP=127.0.0.1
}

geodml_private_server_log() {
    set +x
    if [[ -L "$1" ]]; then
        echo "ERROR: refusing a symlink for the private server log" >&2
        return 2
    fi
    touch -- "$1" || return 2
    chmod -- 600 "$1" || return 2
}

geodml_probe_inference_endpoint() {
    set +x
    python3 "$REPOSITORY_ROOT/analysis/scripts/inference_endpoint_security.py" \
        probe --base-url "$1" --model-id "$2"
}

geodml_safe_server_tail() {
    set +x
    python3 - "$1" <<'PY'
from collections import deque
import os
from pathlib import Path
import sys

key = os.environ.get("VLLM_API_KEY")
if not key:
    raise SystemExit("ERROR: refusing to print server logs without redaction key")
try:
    with Path(sys.argv[1]).open(errors="replace") as source:
        tail = "".join(deque(source, maxlen=120))
except OSError:
    raise SystemExit("ERROR: private server log could not be read") from None
sys.stdout.write(tail.replace(key, "[REDACTED]"))
PY
}
