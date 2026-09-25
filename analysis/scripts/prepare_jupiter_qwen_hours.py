"""Size Qwen shared hours from a saved JUPITER measurement; never submit jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline.agentic_hour_sync import (
    Exchange,
    HubStore,
    atomic,
)
from analysis.interpretability.pipeline.agentic_hours import (
    build_plan,
    canonical,
    digest,
    install_plan,
    inventory,
)
from analysis.scripts.manage_agentic_hours import load_plan
from analysis.scripts.prepare_horeka_qwen import MODELS


def read(path):
    return json.loads(Path(path).read_bytes())


def measured_calibration(directory):
    """Use the same conservative allocation-cost estimate as the Llama wave."""
    prior = read(directory / 'preparation.json')
    runtime = read(directory / 'runtime.json')
    boundary = read(directory / 'boundary.json')
    allocation = read(directory / 'allocation.json')['slurm']
    paths = list((directory / 'results/attempts').glob('*/run_manifest.json'))
    if len(paths) != 1:
        raise ValueError('one reference result manifest required')
    measured = read(paths[0])
    direct = measured.get('direct_dataset', {})
    count = direct.get('committed_count')
    walltime = prior.get('approved_walltime', '01:00:00')
    seconds = {'01:00:00': 3600, '03:00:00': 10800}.get(walltime)
    configuration = prior['summary']['configuration_sha256']
    if (type(count) is not int or count <= 0 or direct.get('reused_count') != 0
            or measured.get('completed_count') != count or measured.get('failed_cell_ids') != []
            or measured.get('status') not in {'checkpointed', 'completed'}
            or boundary.get('status') != 'verified'
            or not seconds or allocation.get('TimeLimit') != walltime or allocation.get('NumNodes') != '1'
            or not re.fullmatch(r'[0-9a-f]{64}', configuration)):
        raise ValueError('clean one-hour reference measurement required')
    if Path(direct.get('root', '')).resolve() != Path(prior['dataset_root']).resolve():
        raise ValueError('measurement belongs to a different dataset')
    profile_path = Path(runtime['SEARCH_AGENTIC_PROFILE'])
    profile = read(profile_path)
    sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if (prior['files'][str(profile_path)]['sha256'] != sha
            or profile['model'] != {'model_id': MODELS[0][0], 'model_revision': MODELS[0][1]}
            or profile['serving']['tensor_parallel_size'] != 4):
        raise ValueError('frozen four-GPU Qwen profile required')
    job = str(boundary['slurm_job_id'])
    # Query just this historical allocation; no population audit or GPU work.
    raw = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                          '--format=JobID,State,AllocTRES,Cluster'],
                         check=True, capture_output=True, text=True).stdout
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    if not any(len(r) >= 4 and r[0] == job and r[1] == 'COMPLETED'
               and 'gres/gpu:gh200=4' in r[2].split(',')
               and 'jupiter' in r[3].lower() for r in rows):
        raise ValueError('completed JUPITER four-GH200 scheduler evidence required')
    return configuration, {
        'cluster': 'jupiter', 'gpus': 4, 'gpu_type': 'GH200',
        'scientific_config_sha256': configuration, 'reference_profile_sha256': sha,
        'seconds_per_task': seconds / count, 'startup_seconds': 0, 'drain_seconds': 300,
        'evidence': {'job_id': job, 'manifest_sha256': digest(measured), 'committed': count,
                     'allocation_seconds': seconds,
                     'method': 'effective full allocation cost per cell; includes startup; additional 300s reserve'},
    }


def prepare_plan(previous, state, local, configuration, timing, contract, pin):
    tasks, completed, blocked = local
    configs = {r['configuration_sha256'] for r in tasks if r['model'] == 'qwen38'}
    if configuration not in configs:
        raise ValueError('Qwen measurement configuration absent from registered tasks')
    calibration = copy.deepcopy(previous['calibration'])
    existing = calibration.get('qwen38', {})
    if existing and 'configurations' not in existing:
        existing = {'configurations': {existing['scientific_config_sha256']: existing}}
    calibration['qwen38'] = existing or {'configurations': {}}
    calibration['qwen38']['configurations'][configuration] = timing
    return build_plan(tasks=tasks, calibration=calibration, registry=state,
                      contract=contract, completed=completed, blocked=blocked,
                      source_commit=pin, input_bundle=previous['input_bundle'],
                      allow_uncalibrated=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    site = read(args.site)
    if site['cluster'] != 'jupiter':
        raise ValueError('only the JUPITER chief may prepare this plan')
    if args.output.exists():
        raise FileExistsError('preserve the existing preparation; publish its plan with publish-plan')
    pin = subprocess.run(['git', '-C', str(REPOSITORY), 'rev-parse', 'HEAD'],
                         check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPOSITORY), 'status', '--porcelain'],
                           check=True, capture_output=True, text=True).stdout
    if dirty:
        raise ValueError('use a clean pinned checkout')
    root = Path(site['dataset_root'])
    configuration, timing = measured_calibration(args.measurement)
    exchange = Exchange(HubStore(args.repo_id), args.output / 'journal')
    _, state = exchange.snapshot()
    previous = load_plan(exchange, state['current_plan'])
    local = inventory(root, stripes=site.get('stripes', 256), reuse_verified=True)
    published = {fp for row in state['hours'].values() for fp in row['completed']}
    published.update(fp for fp, event in exchange.manifest(previous['input_bundle'])['outcomes'].items()
                     if event['state'] == 'completed')
    if local[1] - published:
        raise ValueError('sync saved completions before replanning; no inputs will be re-uploaded')
    plan = prepare_plan(previous, state, local, configuration, timing,
                        read(root / 'contract.json'), pin)
    # Exercise all ownership/history checks before persisting or publishing.
    install_plan(state, plan, cluster='jupiter')
    atomic(args.output / 'plan.json', canonical(plan))
    atomic(args.output / 'calibration.json', canonical(plan['calibration']))
    updated = {**site, 'calibration': str((args.output / 'calibration.json').resolve()),
               'source_commit': pin, 'input_bundle': plan['input_bundle']}
    atomic(args.output / 'site.json', canonical(updated))
    if args.publish:
        exchange.transact('qwen-hours-' + plan['plan_id'],
                          {'action': 'plan', 'plan_sha256': digest(plan)},
                          lambda current: install_plan(current, plan, cluster='jupiter'),
                          {f"coordination/plans/{plan['plan_id']}.json": canonical(plan)})
    rows = [r for r in plan['packages'] if r['model'] == 'qwen38']
    report = {'status': 'published' if args.publish else 'prepared', 'plan_id': plan['plan_id'],
              'qwen_packages': len(rows), 'qwen_cells': sum(len(r['task_fingerprints']) for r in rows),
              'oversized_prompt_groups': sum(r['oversized_prompt_group'] for r in rows),
              'allocation_submitted': False}
    atomic(args.output / 'summary.json', canonical(report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
