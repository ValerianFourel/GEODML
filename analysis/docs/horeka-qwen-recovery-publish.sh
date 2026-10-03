#!/bin/bash
(
  set -euo pipefail
  set +x
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  QWEN_RECOVERY_PIN=4a668d72bb2727518e9aa30ee08442d848fbc254
  QWEN_RECOVERY_CODE="$W/checkouts/qwen-recovery-$QWEN_RECOVERY_PIN"
  QWEN_RECOVERY_RUN="$W/reviews/qwen-recovery-20261003"
  test "$(git -C "$QWEN_RECOVERY_CODE" rev-parse HEAD)" = "$QWEN_RECOVERY_PIN"
  test -z "$(git -C "$QWEN_RECOVERY_CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$QWEN_RECOVERY_CODE"
  export GEODML_AUDIT_PROGRESS=1
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  test -f "$QWEN_RECOVERY_RUN/final/coverage.json"
  exec 9>>"$W/reviews/qwen-hf-update.lock"
  flock -n 9 || { echo "Another publisher is running; preserve it."; exit 1; }
  QWEN_PUBLISH_OUT="$(mktemp -d "$W/reviews/qwen-recovery-publish.XXXXXX")"
  read -r -s -p "HF WRITE token (hidden): " HF_TOKEN </dev/tty
  printf '\n'
  test -n "$HF_TOKEN"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  "$RT/bin/python" -u - "$QWEN_RECOVERY_RUN" "$QWEN_PUBLISH_OUT" "$QWEN_RECOVERY_PIN" <<'PY'
from pathlib import Path
import json, os, sys
from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
from analysis.scripts.publish_qwen_results import publish
from analysis.scripts.report_inference_hub_progress import collect
root, output, pin = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
context = json.loads((root / 'recovery.json').read_text())
coverage = json.loads((root / 'final/coverage.json').read_text())
print('RECOVERY_COVERAGE', json.dumps({k: v for k, v in coverage.items() if k != 'blocked_fingerprints'}, indent=2), flush=True)
try:
    exchange = Exchange(HubStore(context['repo_id']), output / 'journal')
    result = publish(exchange, Path(context['dataset_root']), stripes=256, apply=True, source_commit=pin)
    (output / 'publish.json').write_text(json.dumps(result, indent=2) + '\n')
    progress = collect(exchange, revision=exchange.store.head())
    (output / 'hub-progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    print('PUBLISHED_COUNTS', json.dumps(progress, indent=2), flush=True)
    print('PUBLISH_DIRECTORY', output, flush=True)
    print('Recovery complete:', coverage['complete'], '| Publication issues:', len(result['blocked']))
    if result['blocked'] or progress['status'] != 'counted':
        raise SystemExit(2)
except Exception as error:
    message = str(error).replace(os.environ['HF_TOKEN'], '[REDACTED]')
    (output / 'error.json').write_text(json.dumps({'error': message}, indent=2) + '\n')
    print('PUBLISH_FAILED', message, '| Evidence:', output, file=sys.stderr)
    raise SystemExit(1)
PY
)
