"""Publish and submit one explicitly approved five-job JUPITER wave.

The exceptions are scoped to this wave: no start gap and unavailable quota
evidence. They never assert that quota was verified. No retry or resubmission.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from analysis.interpretability.pipeline.agentic_hour_runtime import (
    allocation_command,
    validate_request,
)
from analysis.interpretability.pipeline.agentic_hour_sync import (
    Exchange,
    HubStore,
    atomic,
)
from analysis.interpretability.pipeline.agentic_hour_updates import (
    progress,
    publish_progress,
    replan,
)
from analysis.interpretability.pipeline.agentic_hours import (
    canonical,
    check_owner,
    digest,
    inventory,
)
from analysis.interpretability.pipeline.agentic_storage import storage_health
from analysis.scripts.manage_agentic_hours import stage as stage_attempt
from analysis.scripts.prepare_jupiter_llama import current_scheduler
from analysis.scripts.prepare_shared_hour_inputs import FILE_KEYS, SETTING_KEYS, stage
from analysis.scripts.publish_agentic_dataset import build_manifest
from analysis.scripts.reconcile_agentic_dataset import reconcile


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    atomic(Path(path), canonical(value))


def idle(since):
    snapshot = current_scheduler(since)
    snapshot['cluster'] = 'jupiter'
    if (not snapshot.get('complete') or snapshot['jobs']
            or not 0 <= time.time() - snapshot['captured_at_epoch'] <= 120):
        raise ValueError('Wave requires a fresh idle scheduler; preserve existing allocations')
    return snapshot


def admit_five(state, attempts, snapshot, storage, *, now, evidence):
    """One CAS reserves the entire admission capacity before any sbatch."""
    if len(attempts) != 5 or len({a['attempt_id'] for a in attempts}) != 5:
        raise ValueError('Exactly five distinct attempts required')
    if not evidence.strip():
        raise ValueError('Explicit quota and start-gap exception evidence required')
    if (snapshot.get('cluster') != 'jupiter' or snapshot.get('complete') is not True
            or snapshot.get('jobs') or not 0 <= now - snapshot['captured_at_epoch'] <= 120):
        raise ValueError('Fresh idle JUPITER scheduler required')
    if (storage.get('safe_to_admit') is not True
            or not 0 <= now - storage['captured_at_epoch'] <= 300):
        raise ValueError('Storage failure blocks this wave despite quota exception')
    result = deepcopy(state)
    admission = result['admission'].setdefault('jupiter', {})
    if admission.get('pending_attempt') or admission.get('finite_wave'):
        raise ValueError('Prior admission must be reconciled first')
    selected = {h for a in attempts for h in a['owners']}
    if any(h.get('owner') and key not in selected for key, h in result['hours'].items()):
        raise ValueError('Other owned hours must be reconciled first')
    tickets = admission.setdefault('tickets', {})
    fingerprints = set()
    for attempt in attempts:
        request = attempt['request']
        validate_request(request, attempt['cluster_profile'])
        if (request['cluster'] != 'jupiter' or request['mode'] != 'batch'
                or request['approval']['walltime_seconds'] != 3600
                or attempt['model'] != 'llama4' or attempt['attempt_id'] in tickets):
            raise ValueError('Only five new one-hour Llama batch attempts are approved')
        for hour_id, owner in attempt['owners'].items():
            hour = result['hours'][hour_id]
            check_owner(hour, owner)
            work = set(hour['task_fingerprints'])
            if fingerprints & work:
                raise ValueError('Wave assignments overlap')
            fingerprints.update(work)
        tickets[attempt['attempt_id']] = {
            'request_sha256': digest(request), 'existing_job_id': None,
            'attempt_sha256': digest({**attempt, 'admission_ticket': None}),
            'admitted_at': now, 'scheduler_sha256': digest(snapshot),
            'storage_sha256': digest(storage), 'quota_verified': False,
            'policy_exception': evidence,
        }
    admission['finite_wave'] = {
        'attempt_ids': [a['attempt_id'] for a in attempts], 'maximum_gpu_hours': 20,
        'start_gap_seconds': 0, 'quota_verified': False, 'evidence': evidence,
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prior', 'qwen-runtime', 'priority', 'output', 'environment-file'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--since', required=True)
    p.add_argument('--account', required=True)
    p.add_argument('--partition', default='booster')
    p.add_argument('--repo-id', required=True)
    p.add_argument('--approval', required=True)
    p.add_argument('--allow-stale-quota-and-simultaneous-starts', action='store_true')
    args = p.parse_args()
    if not args.allow_stale_quota_and_simultaneous_starts or not args.approval.strip():
        raise ValueError('The explicit wave-specific policy exceptions are required')
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=REPO, text=True):
        raise ValueError('Use a clean committed checkout')
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'submission-intent.json').exists():
            raise ValueError('Submission already attempted. Inspect saved job receipts; do not resubmit')
        run(args, pin)


def run(args, pin):
    os.environ.setdefault('GEODML_AUDIT_PROGRESS', '1')
    os.environ['HF_TOKEN'] = getpass.getpass('HF write token (hidden): ').strip()
    if not os.environ['HF_TOKEN']:
        raise ValueError('No token entered')
    snapshot = idle(args.since)
    health = storage_health(args.source)
    if not health['safe_to_admit']:
        raise ValueError('Storage failure; no new admissions')
    save(args.output / 'approval.json', {
        'evidence': args.approval, 'jobs': 5, 'walltime_seconds_each': 3600,
        'maximum_gpu_hours': 20, 'quota_verified': False, 'start_gap_seconds': 0,
    })
    exchange = Exchange(HubStore(args.repo_id), args.output / 'journal')
    _, state = exchange.snapshot()
    if any(h.get('owner') for h in state['hours'].values()):
        raise ValueError('Existing shared owners must be reconciled before this bootstrap handoff')
    prior = read(args.prior / 'preparation.json')
    runtime = read(args.prior / 'runtime.json')
    if Path(prior['dataset_root']).resolve() != args.source:
        raise ValueError('Prior measurement used a different dataset')
    manifests = list((args.prior / 'results/attempts').glob('*/run_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected one reference allocation manifest')
    measured = read(manifests[0])
    direct = measured.get('direct_dataset', {})
    count = direct.get('committed_count', 0)
    boundary = read(args.prior / 'boundary.json')
    allocation = read(args.prior / 'allocation.json')['slurm']
    if not any(str(row['job_id']) == str(boundary['slurm_job_id']) and row['state'] == 'COMPLETED'
               for row in snapshot['owners']):
        raise ValueError('Reference job has no completed scheduler confirmation')
    if (measured.get('status') != 'checkpointed' or measured.get('failed_cell_ids')
            or not count or direct.get('reused_count') != 0
            or boundary.get('status') != 'verified'
            or allocation.get('TimeLimit') != '01:00:00' or allocation.get('NumNodes') != '1'):
        raise ValueError('Reference allocation is not a clean one-hour measurement')
    profile_path = Path(runtime['SEARCH_AGENTIC_PROFILE'])
    import hashlib
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if prior['files'][str(profile_path)]['sha256'] != profile_sha:
        raise ValueError('Reference profile changed since preparation')
    configuration = prior['summary']['configuration_sha256']
    review = reconcile(args.source, scheduler_snapshot=snapshot, apply=True)
    if review['blocked']:
        raise ValueError('Result reconciliation blocked')
    local = inventory(args.source, reuse_verified=True)
    if {t['configuration_sha256'] for t in local[0] if t['model'] == 'llama4'} != {configuration}:
        raise ValueError('Registered Llama scientific configuration differs from measurement')
    model_inputs = {}
    for model, path in [('qwen38', args.qwen_runtime), ('llama4', args.prior / 'runtime.json')]:
        values = read(path)
        model_inputs[model] = {'runtime_environment': {k: v for k, v in values.items()
                              if k in FILE_KEYS | SETTING_KEYS}, 'extra_files': {}}
    mirror = args.output / 'dataset'
    print('Preparing shared inputs from existing registrations and saved results', flush=True)
    stage(args.source, mirror, model_inputs, args.priority, idle(args.since), local_inventory=local)
    report = read(mirror / read(mirror / 'local-only/shared-input-owner.json')['manifest'])
    bundle = exchange.upload(mirror, list(build_manifest(mirror)['files']),
                             outcomes=report['outcomes'], metadata={'kind': 'frozen-inputs'})
    calibration = {'llama4': {'configurations': {configuration: {
        'cluster': 'jupiter', 'gpus': 4, 'gpu_type': 'GH200',
        'scientific_config_sha256': configuration, 'reference_profile_sha256': profile_sha,
        'seconds_per_task': 3600 / count, 'startup_seconds': 0, 'drain_seconds': 300,
        'evidence': {'job_id': boundary['slurm_job_id'], 'manifest_sha256': digest(measured),
                     'committed': count, 'method': 'effective full allocation cost per cell; includes startup; additional 300s reserve'},
    }}}}
    save(args.output / 'calibration.json', calibration)
    site = {'cluster': 'jupiter', 'dataset_root': str(mirror), 'input_bundle': bundle,
            'calibration': str(args.output / 'calibration.json'), 'source_commit': pin,
            'plan_dir': str(args.output / 'plans'), 'journal': str(args.output / 'journal'),
            'stripes': 256, 'attempts': []}
    _, state = exchange.snapshot()
    plan_result = replan(exchange, site, state, {}, local_inventory=local)
    plan = read(plan_result['plan_file'])
    rows = [x for x in plan['packages'] if x['model'] == 'llama4']
    groups = [[] for _ in range(5)]
    # Reserve ample next-keyword work per allocation; remaining cells stay owned
    # until terminal reconciliation, never silently reassigned after one hour.
    cursor = 0
    for group in groups:
        size = 0
        while size < 3 * count and cursor < len(rows):
            group.append(rows[cursor]['hour_id'])
            size += len(rows[cursor]['task_fingerprints'])
            cursor += 1
        if not group:
            raise ValueError('Insufficient eligible work for five allocations')
    setup = args.output / 'environment.sh'
    atomic(setup, ('source ' + shlex.quote(str(args.environment_file.resolve())) + '\n'
                  'if ! type module >/dev/null 2>&1; then source /etc/profile; fi\n'
                  'module load Stages/2026 GCC Python CUDA git\n'
                  'source ' + shlex.quote(runtime['ACL_ARR_VENV'] + '/bin/activate') + '\n'
                  'export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1\n').encode())
    profile = {'cluster': 'jupiter', 'account': args.account, 'partition': args.partition,
               'environment_file': str(setup), 'cache_root': runtime['GEODML_CACHE_ROOT'],
               'validated_models': {'llama4': {'evidence': calibration['llama4'],
                    'reference_profile_sha256': profile_sha}}}
    serving = {k: v for k, v in runtime.items() if k in FILE_KEYS | SETTING_KEYS
               and k != 'GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY'}
    attempts = []
    wave_id = 'llama-five-' + digest({'output': str(args.output), 'pin': pin})[:16]
    for number, hours in enumerate(groups, 1):
        attempt_id = f'{wave_id}-{number}'
        request = {'attempt_id': attempt_id, 'cluster': 'jupiter', 'mode': 'batch',
                   'git_commit': pin, 'hour_ids': hours, 'since': args.since,
                   'attempt_dir': str(args.output / attempt_id),
                   'approval': {'status': 'approved', 'evidence': args.approval,
                        'walltime_seconds': 3600, 'maximum_gpu_hours': 4,
                        'resources': {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': 'all'},
                        'estimate': f'Reference {count} cells per one-hour allocation; 300s admission margin; finite five-job wave, total at most 20 GPU-hours'}}
        attempts.append(stage_attempt(exchange, request=request, profile=profile, runtime=serving,
                        reference_profile=profile_path, dataset=mirror, repository=REPO,
                        operation_id='reserve-' + attempt_id))
    snapshot = idle(args.since)
    health = storage_health(mirror)
    exchange.transact('admit-' + wave_id, {'action': 'finite-five', 'attempts': [a['attempt_id'] for a in attempts]},
        lambda state: admit_five(state, attempts, snapshot, health, now=int(time.time()), evidence=args.approval))
    revision, state = exchange.snapshot()
    for a in attempts:
        a['admission_ticket'] = state['admission']['jupiter']['tickets'][a['attempt_id']]
        path = Path(a['request']['attempt_dir']) / 'attempt.json'
        save(path, a)
        site['attempts'].append(str(path))
    save(args.output / 'site.json', site)
    publish_progress(exchange, progress(mirror, state, revision=revision, local_inventory=local,
                      deferred=plan['deferred'], prior_completed=plan['completed_before_plan']))
    submit_held(args.output, attempts, since=args.since, wave_id=wave_id, dataset=mirror)


def submit_held(output, attempts, *, since, wave_id, dataset):
    if len(attempts) != 5 or (output / 'submission-intent.json').exists():
        raise ValueError('Exactly five unsubmitted attempts required; inspect any existing receipt')
    idle(since)
    save(output / 'submission-intent.json', {'wave_id': wave_id,
         'attempts': [str(Path(a['request']['attempt_dir']) / 'attempt.json') for a in attempts]})
    jobs = []
    for number, a in enumerate(attempts, 1):
        if not storage_health(dataset)['safe_to_admit']:
            raise ValueError('Storage failure: no further submissions')
        command = allocation_command(a['request'], a['cluster_profile'], REPO)
        command.insert(1, '--hold')
        receipt = output / f'submission-{number}.json'
        save(receipt, {'status': 'requested', 'command': command})
        environment = {k: v for k, v in os.environ.items()
                       if k not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_API_TOKEN'}}
        raw = subprocess.check_output(command, text=True, env=environment).strip()
        job = raw.split(';')[0]
        if not job.isdigit():
            raise ValueError('Uncertain submission response; inspect Slurm, do not retry')
        jobs.append(job)
        save(receipt, {'status': 'held', 'job_id': job,
                      'attempt': str(Path(a['request']['attempt_dir']) / 'attempt.json')})
        print('HELD_JOB=' + job, flush=True)
    snapshot = current_scheduler(since)
    if (not snapshot.get('complete') or not 0 <= time.time() - snapshot['captured_at_epoch'] <= 120
            or {str(j['job_id']) for j in snapshot['jobs']} != set(jobs)
            or any(j['state'] != 'PENDING' or not j.get('held') for j in snapshot['jobs'])):
        raise ValueError('Scheduler changed; wave remains held for review')
    for job in jobs:
        if not storage_health(dataset)['safe_to_admit']:
            raise ValueError('Storage failure: remaining jobs stay held')
        subprocess.run(['scontrol', 'release', job], check=True)
        print('RELEASED_JOB=' + job, flush=True)
    save(output / 'submitted.json', {'job_ids': jobs, 'maximum_gpu_hours': 20})
    print(json.dumps({'job_ids': jobs, 'site': str(output / 'site.json'), 'maximum_gpu_hours': 20}))


if __name__ == '__main__':
    main()
