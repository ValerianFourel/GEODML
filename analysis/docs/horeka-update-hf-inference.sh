#!/bin/bash
(
  set -euo pipefail
  source /hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/geodml-nemotron-env.sh
  INFERENCE_PIN=7a5d3bb89f2b3349cfdd2c80748f2fa5bf3a3f11
  INFERENCE_CODE="$W/checkouts/inference-status-$INFERENCE_PIN"
  test "$(git -C "$INFERENCE_CODE" rev-parse HEAD)" = "$INFERENCE_PIN"
  test -z "$(git -C "$INFERENCE_CODE" status --porcelain --untracked-files=all)"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$INFERENCE_CODE"
  export GEODML_AUDIT_PROGRESS=1
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  "$RT/bin/python" -u - "$W" "$INFERENCE_CODE" "$INFERENCE_PIN" <<'PY'
from collections import Counter
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

workspace, code, pin = Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(code))
from analysis.scripts import publish_qwen_results as publisher
from analysis.scripts.audit_horeka_saved_progress import qwen_counts
from analysis.scripts.report_inference_hub_progress import collect as published_counts
from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

reviews = workspace / 'reviews'
reviews.mkdir(exist_ok=True)
with (reviews / 'qwen-hf-update.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('An update from this page is already running. Keep it and inspect its UPDATE_DIRECTORY.')
    output = Path(tempfile.mkdtemp(prefix='qwen-hf-update-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ-'), dir=reviews))
    print('UPDATE_DIRECTORY', output, flush=True)
    token = ''
    try:
        division_path = Path((workspace / 'qwen-bouts/CURRENT').read_text().strip())
        if not division_path.is_absolute():
            raise ValueError('CURRENT must contain an absolute division path')
        division_bytes = (division_path / 'division.json').read_bytes()
        division = json.loads(division_bytes)
        if division.get('format_version') != 'geodml-horeka-qwen-bouts-v1' or division.get('model') != 'qwen38':
            raise ValueError('Expected the existing Qwen bout division')
        dataset = Path(division['dataset_root']).resolve(strict=True)
        contract_bytes = (dataset / 'contract.json').read_bytes()
        if json.loads(contract_bytes).get('format_version') != 'geodml-incremental-dataset-v1':
            raise ValueError('Unsupported dataset contract')
        layout = qwen_counts(workspace)
        local = layout['qwen_unique_cells']
        if local.get('unavailable') or layout.get('division') != str(division_path):
            raise ValueError('Cannot establish the current dataset ledger layout: ' + str(local))
        stripes = local['ledger_stripes']
        save(output / 'qwen-counts-before.json', layout)
        destination = 'ValerianFourel/geodml-experiment-v2-paper-private'
        context = {'started_utc': datetime.now(timezone.utc).isoformat(), 'dataset_root': str(dataset),
                   'division': str(division_path), 'division_sha256': hashlib.sha256(division_bytes).hexdigest(),
                   'contract_sha256': hashlib.sha256(contract_bytes).hexdigest(), 'repo_id': destination,
                   'publication_code_commit': pin, 'ledger_stripes': stripes, 'source_commit_meaning': 'publication helper, not generator execution',
                   'source_settings_changed': False, 'new_inference_started': False}
        save(output / 'context.json', context)
        warnings.simplefilter('error', getpass.GetPassWarning)
        # Always ask: the cached login token may be read-only (403 on upload).
        # getpass reads from /dev/tty itself; output may be piped through tee.
        token = getpass.getpass('HF WRITE token for the private dataset (hidden): ').strip()
        if not token:
            raise ValueError('A write token is required in the terminal prompt')
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        # HubStore verifies that the destination already exists and is private.
        store = HubStore(destination)
        exchange = Exchange(store, output / 'journal')
        print('CHECK_AND_PUBLISH', dataset, '->', destination, flush=True)
        print('Checking saved task identities and sealed references. This can take time before upload progress appears.', flush=True)
        result = publisher.publish(exchange, dataset, stripes=stripes, apply=True, source_commit=pin)
        save(output / 'publish.json', result)
        revision = store.head()
        index = publisher.read_index(store, revision)
        save(output / 'index-after.json', index)
        indexed = {entry['bundle']: entry for entry in index['bundles']}
        new_completed = new_failed = 0
        for entry in result['bundles']:
            if indexed.get(entry['bundle']) != entry:
                raise ValueError('Published bundle receipt does not match the saved remote index')
            manifest = exchange.manifest(entry['bundle'], revision=revision)
            counts = Counter(event['state'] for event in manifest['outcomes'].values())
            if counts.get('completed', 0) != entry['completed'] or counts.get('terminal_failed', 0) != entry['failed']:
                raise ValueError('Published outcome counts differ from the remote index')
            if set(counts) - {'completed', 'terminal_failed'}:
                raise ValueError('Unexpected nonterminal outcome in the published bundle')
            new_completed += entry['completed']
            new_failed += entry['failed']
        receipt = {'status': 'partial_with_blocked_writers' if result['blocked'] else
                   'updated' if result['bundles'] else 'nothing_new_at_snapshot',
                   'repo_id': destination, 'revision': revision,
                   'new_completed_published': new_completed, 'new_terminal_failures_recorded': new_failed,
                   'blocked_writers': result['blocked'], 'new_bundles_checked': len(result['bundles']),
                   'remote_data_verification': 'performed by existing publisher before index update',
                   'finished_utc': datetime.now(timezone.utc).isoformat(),
                   'full_population_complete': None}
        save(output / 'verification.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)
        print('UPDATE_DIRECTORY', output, flush=True)
        progress = published_counts(exchange, revision=revision)
        progress['repo_id'] = destination
        save(output / 'hub-progress.json', progress)
        print('PUBLISHED_INFERENCE_COUNTS', json.dumps(progress, indent=2), flush=True)
        print('HF_REPORT', output / 'hub-progress.json', flush=True)
        if result['blocked'] or progress['status'] != 'counted':
            raise SystemExit(2)
    except Exception as error:
        message = str(error).replace(token, '[REDACTED]') if token else str(error)
        save(output / 'error.json', {'error_type': type(error).__name__, 'message': message,
                                   'time_utc': datetime.now(timezone.utc).isoformat(),
                                   'partial_remote_upload_possible': True})
        print('UPDATE_FAILED', type(error).__name__, message, file=sys.stderr, flush=True)
        print('INSPECT_UPDATE_DIRECTORY', output, file=sys.stderr, flush=True)
        raise SystemExit(1)
PY
)
