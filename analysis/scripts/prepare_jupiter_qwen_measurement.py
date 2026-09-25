"""Prepare and submit the first approved Qwen segment from existing registrations."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import HubStore, atomic
from analysis.interpretability.pipeline.agentic_hours import canonical, inventory
from analysis.scripts import prepare_jupiter_llama as first
from analysis.scripts.prepare_horeka_qwen import MODELS
from analysis.scripts.prepare_shared_hour_inputs import FILE_KEYS, SETTING_KEYS
from analysis.scripts.reconcile_agentic_dataset import reconcile

ESTIMATE = ('Qwen segment 1 of the approved five three-hour Qwen segments, alongside five '
            'approved three-hour Llama segments. Total wave ceiling: 30 node-hours / 120 GPU-hours. '
            'This allocation: one node, four GH200, 32 requested CPUs, all node memory, '
            'three hours / 12 GPU-hours. Qwen throughput is not yet measured; admit useful '
            'missing work for up to 175 minutes including startup, then drain and clean up. '
            'No promise that the backlog will finish. Remaining nine segments are not submitted here.')


def validate_source(runtime, tasks):
    profile = first.load_profile(runtime['SEARCH_AGENTIC_PROFILE'])
    selected = [row for row in tasks if row['model'] == 'qwen38']
    if (not selected or profile['model'] != {'model_id': MODELS[0][0], 'model_revision': MODELS[0][1]}
            or any((t['claim_identity']['model_id'], t['claim_identity']['model_revision']) != MODELS[0]
                   for t in selected)):
        raise ValueError('pinned Qwen registration and profile required')
    if (profile['serving']['tensor_parallel_size'] != 4
            or profile['serving']['data_parallel_size'] != 1
            or runtime['SEARCH_AGENTIC_CROSS_ENCODER_REVISION'] != MODELS[1][1]):
        raise ValueError('Qwen serving or compactor differs from the maintained worker')


def prepare(args):
    root, output = args.dataset.resolve(), args.output.resolve()
    if output.exists():
        return first.verify_prepared(output)
    pin = first.clean_commit()
    snapshot = first.current_scheduler(args.since)
    first.scheduler_gate(snapshot, int(time.time()))
    hub = HubStore(args.hf_repo)
    registry = json.loads(hub.read('coordination/hours.json', hub.head()))
    if any(h['model'] == 'qwen38' for h in registry['hours'].values()):
        raise ValueError('Qwen shared packages already exist; use their reservations')
    runtime = first.read(args.reference_runtime)
    if Path(runtime['GEODML_DATASET_ROOT']).resolve() != root:
        raise ValueError('saved Qwen runtime belongs to a different dataset')
    profile = first.load_profile(runtime['SEARCH_AGENTIC_PROFILE'])
    if first.importlib.metadata.version('sentence-transformers') != '6.0.1':
        raise ValueError('sentence-transformers 6.0.1 required')
    if first.importlib.metadata.version('vllm') != profile['runtime']['vllm_version']:
        raise ValueError('vLLM differs from frozen Qwen profile')
    cache = Path(os.environ.get('HF_HUB_CACHE') or runtime['HF_HUB_CACHE'])
    first._snapshot(cache, MODELS[0][0], MODELS[0][1])
    review = reconcile(root, scheduler_snapshot=snapshot, apply=True)
    if review['blocked']:
        raise ValueError('reconciliation blocked; preserve saved results')
    local = inventory(root, reuse_verified=True)
    validate_source(runtime, local[0])
    if any(t['fingerprint'] in local[2] for t in local[0] if t['model'] == 'qwen38'):
        raise ValueError('Qwen has blocked cells; review before measurement')
    environment = {key: runtime[key] for key in FILE_KEYS | SETTING_KEYS if key in runtime}
    for key in ('ACL_ARR_VENV', 'HF_HUB_CACHE', 'GEODML_CACHE_ROOT'):
        environment[key] = os.environ.get(key) or runtime[key]
        if not Path(environment[key]).is_dir():
            raise ValueError('missing directory: ' + key)
    source_files = [Path(runtime[key]).resolve() for key in (*first.FILES, 'SEARCH_AGENTIC_PROFILE')]
    proofs = {str(p): first.identity(p) for p in source_files}
    output.mkdir(parents=True)
    atomic(output / 'profile.json', Path(runtime['SEARCH_AGENTIC_PROFILE']).read_bytes())
    summary = first.freeze_backlog(root, output, model='qwen38', local_inventory=local)
    environment.update({
        'SEARCH_AGENTIC_PROFILE': str(output / 'profile.json'),
        'GEODML_EXECUTION_REPOSITORY': str(first.REPOSITORY), 'GEODML_EXECUTION_COMMIT': pin,
        'GEODML_MODEL_SLUG': 'qwen38', 'GEODML_DATASET_ROOT': str(root),
        'GEODML_WAVE_ROOT': str(output / 'wave'), 'GEODML_WAVE_OUTPUT_ROOT': str(output / 'results'),
        'GEODML_WAVE_LOG_ROOT': str(output / 'logs'),
        'GEODML_WORKER_LAUNCHER': str(first.REPOSITORY / 'analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh'),
        'GEODML_START_MARGIN_SECONDS': '300', 'GEODML_CLEANUP_MARGIN_SECONDS': '120',
        'GEODML_WORKER_INDEX': '0', 'GEODML_WORKER_COUNT': '1',
        'GEODML_WORKER_STDOUT': str(output / 'slurm-%j.out'),
        'GEODML_WORKER_STDERR': str(output / 'slurm-%j.err'),
        'GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY': '1', 'PYTHONDONTWRITEBYTECODE': '1',
    })
    atomic(output / 'runtime.json', canonical(environment))
    script = ('#!/bin/bash\nset -eo pipefail\nsource ' + shlex.quote(str(args.environment_file.resolve())) +
              '\nif ! type module >/dev/null 2>&1; then source /etc/profile; fi\n'
              'module load Stages/2026 GCC Python CUDA git\nsource ' +
              shlex.quote(environment['ACL_ARR_VENV'] + '/bin/activate') +
              '\nexport GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1\nexport PYTHONDONTWRITEBYTECODE=1\n'
              'exec python3 ' + shlex.quote(str(first.REPOSITORY / 'analysis/scripts/prepare_jupiter_llama.py')) +
              ' execute --output ' + shlex.quote(str(output)) + '\n')
    atomic(output / 'run.sh', script.encode())
    for path, expected in proofs.items():
        if first.identity(Path(path)) != expected:
            raise ValueError('frozen input changed during preparation')
    record = {'git_commit': pin, 'dataset_root': str(root), 'model': 'qwen38',
              'approved_walltime': '03:00:00', 'estimate': ESTIMATE,
              'wave_member': 'qwen-1-of-5', 'summary': summary,
              'files': {str(p): first.identity(p) for p in output.rglob('*') if p.is_file()},
              'dataset_files': {p: v for p, v in proofs.items() if Path(p).is_relative_to(root)},
              'external_files': {p: v for p, v in proofs.items() if not Path(p).is_relative_to(root)}}
    atomic(output / 'preparation.json', canonical(record))
    return first.verify_prepared(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'reference-runtime', 'output', 'environment-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('since', 'hf-repo', 'account', 'partition', 'approval'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not args.approval.strip():
        raise ValueError('explicit approval evidence required')
    from huggingface_hub import get_token
    if not get_token():
        import getpass
        os.environ['HF_TOKEN'] = getpass.getpass('HF token (hidden): ').strip()
    record = prepare(args)
    if args.submit:
        args.approved_walltime = '03:00:00'
        result = first.submit(args)
    else:
        result = {'status': 'prepared', **record['summary'], 'allocation_submitted': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
