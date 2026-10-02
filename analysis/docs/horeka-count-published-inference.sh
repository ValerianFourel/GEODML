#!/bin/bash
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  INFERENCE_PIN=7a5d3bb89f2b3349cfdd2c80748f2fa5bf3a3f11
  INFERENCE_CODE="$W/checkouts/inference-status-$INFERENCE_PIN"
  test "$(git -C "$INFERENCE_CODE" rev-parse HEAD)" = "$INFERENCE_PIN"
  test -z "$(git -C "$INFERENCE_CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$INFERENCE_CODE"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  mkdir -p "$W/reviews"
  "$RT/bin/python" -u - "$W" <<'PY'
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings
from huggingface_hub import get_token
from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
from analysis.scripts.report_inference_hub_progress import collect

workspace = Path(sys.argv[1])
output = Path(tempfile.mkdtemp(prefix='inference-hub-counts-', dir=workspace / 'reviews'))
token = get_token() or ''
try:
    if not token:
        warnings.simplefilter('error', getpass.GetPassWarning)
        with open('/dev/tty', 'r+') as terminal:
            token = getpass.getpass('HF token for private dataset read access (hidden): ', stream=terminal).strip()
    if not token:
        raise ValueError('HF read access is required')
    os.environ['HF_TOKEN'] = token
    destination = 'ValerianFourel/geodml-experiment-v2-paper-private'
    report = collect(Exchange(HubStore(destination), output / 'unused-journal'))
    report['repo_id'] = destination
    path = output / 'hub-progress.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print('HF_REPORT', path, flush=True)
    raise SystemExit(0 if report['status'] == 'counted' else 2)
except Exception as error:
    message = str(error).replace(token, '[REDACTED]') if token else str(error)
    print('HF_COUNTS_UNAVAILABLE', type(error).__name__, message, file=sys.stderr)
    raise SystemExit(1)
PY
)
