"""Finite approved Qwen expansion and coordinated Llama wave on JUPITER.

The existing Qwen measurement is member one unless its cancellation and a new
five-member wave are explicitly confirmed.
Qwen workers share its immutable queue and atomic ledger. Llama stays HF-owned.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import shlex
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
from analysis.interpretability.pipeline.agentic_hours import (
    canonical,
    check_owner,
    digest,
)
from analysis.interpretability.pipeline.agentic_storage import storage_health
from analysis.scripts import prepare_jupiter_llama as first
from analysis.scripts.manage_agentic_hours import scheduler, stage, sync_once
from analysis.scripts.prepare_agentic_qwen_inputs import copy_immutable
from analysis.scripts.prepare_shared_hour_inputs import FILE_KEYS, SETTING_KEYS
from analysis.scripts.reconcile_agentic_dataset import (
    TERMINAL_SCHEDULER_STATES,
    reconcile,
)

read = first.read


def save(path, value):
    atomic(Path(path), canonical(value))


def gate(snapshot, allowed, extra, maximum, *, now):
    if snapshot.get('complete') is not True or not 0 <= now - snapshot['captured_at_epoch'] <= 120:
        raise ValueError('Fresh complete scheduler evidence required')
    ids = {str(j['job_id']) for j in snapshot['jobs']}
    if not ids <= set(allowed) or len(ids) + extra > maximum:
        raise ValueError('Other allocations or concurrency limit block submission; preserve live jobs')


def health(args, root):
    quota = None
    if args.quota:
        quota = read(args.quota)
        quota['fresh'] = (quota.get('cluster') == 'jupiter'
                          and 0 <= time.time() - quota.get('captured_at_epoch', 0) <= 300)
    result = storage_health(root, quota_evidence=quota)
    if not result['safe_to_admit'] or (not result['quota_verified'] and not args.allow_stale_quota):
        raise ValueError('Storage admission blocked: fresh quota evidence or explicit wave exception required')
    # Statvfs cannot detect all user quotas. Exercise actual creation and fsync.
    from analysis.interpretability.pipeline.agentic_storage import _probe
    _probe(root)
    return result


def approval(args, estimate):
    return {'status': 'approved', 'evidence': args.approval, 'walltime': '03:00:00',
            'walltime_seconds': 10800, 'maximum_gpu_hours': 12,
            'extended_walltime_approval': {'walltime_seconds': 10800, 'evidence': args.approval},
            'resources': {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': 'all'},
            'estimate': estimate}


def verify_reference(path):
    record = read(path / 'preparation.json')
    if record.get('model') != 'qwen38' or record.get('approved_walltime') != '03:00:00':
        raise ValueError('Expected the approved three-hour Qwen member one')
    for group in ('files', 'dataset_files', 'external_files'):
        for name, expected in record[group].items():
            if first.identity(Path(name)) != expected:
                raise ValueError('Frozen Qwen reference changed: ' + name)
    submission = read(path / 'submission.json')
    if submission.get('status') != 'submitted' or not str(submission.get('job_id', '')).isdigit():
        raise ValueError('Member one has no confirmed submission')
    return record, submission


def localize_reference(directory, reference, reference_root):
    """Bind frozen member-one files inside the executing member directory.

    ``verify_prepared`` proves ``record['files']`` through a VerificationCache
    rooted at the member directory, so member-one paths outside that root
    crashed the allocation before any cell ran. Copy each frozen file under
    ``inputs/`` preserving its identity, fail closed on any mismatch, and
    return the member-local proof map.
    """
    root = Path(reference_root)
    localized = {}
    for name, expected in reference['files'].items():
        source = Path(name)
        try:
            relative = source.relative_to(root)
        except ValueError:
            raise ValueError('Frozen reference file lies outside member one: ' + name)
        target = Path(directory) / 'inputs' / relative
        copy_immutable(source, target, expected)
        if first.identity(target) != expected:
            raise ValueError('Localized reference mismatch: ' + str(target))
        localized[str(target)] = expected
    return localized


def member_state(job, snapshot):
    for row in snapshot['jobs']:
        if str(row['job_id']) == str(job):
            return row['state']
    row = first.command(['sacct', '-X', '-j', str(job), '--noheader', '--parsable2', '--format=State'])
    return row.split('|')[0].split()[0].rstrip('+').upper() if row.strip() else None


def setup(args, runtime):
    path = args.output / 'environment.sh'
    atomic(path, ('source ' + shlex.quote(str(args.environment_file.resolve())) + '\n'
                 'if ! type module >/dev/null 2>&1; then source /etc/profile; fi\n'
                 'module load Stages/2026 GCC Python CUDA git\nsource ' +
                 shlex.quote(runtime['ACL_ARR_VENV'] + '/bin/activate') + '\n'
                 'export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=1 PYTHONDONTWRITEBYTECODE=1\n').encode())
    return path


def qwen(args, pin, exchange):
    reference, submitted = verify_reference(args.qwen_reference)
    job = str(submitted['job_id'])
    if job != args.existing_qwen_job:
        raise ValueError('Qwen member one differs from approved existing job')
    _, state = exchange.snapshot()
    if any(h['model'] == 'qwen38' for h in state['hours'].values()):
        raise ValueError('Qwen has HF-owned packages; use the shared-hour dispatcher')
    source = read(args.qwen_reference / 'runtime.json')
    root = Path(reference['dataset_root'])
    if Path(source['GEODML_DATASET_ROOT']).resolve() != root.resolve():
        raise ValueError('Qwen ledger differs from member one')
    snapshot = first.current_scheduler(args.since)
    replace_cancelled = args.replace_cancelled_qwen
    allowed = [] if replace_cancelled else [job]
    gate(snapshot, allowed, 5 if replace_cancelled else 4, args.maximum_concurrent, now=time.time())
    known = [j for j in snapshot['jobs'] + snapshot['owners'] if str(j['job_id']) == job]
    health(args, root)
    if replace_cancelled:
        if not known or any(j['state'] != 'CANCELLED' for j in known):
            raise ValueError('Replacement requires Slurm-confirmed cancellation of the original Qwen job')
        accounting = first.command(['sacct', '-X', '-j', job, '--noheader', '--parsable2',
                                    '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])
        save(args.output / 'cancelled-reference.json', {
            'job_id': job, 'scheduler_rows': known, 'accounting': accounting,
            'new_wave_node_hours': 30, 'new_wave_gpu_hours': 120,
            'prior_consumption': 'Recorded separately in accounting; not deducted from the newly approved wave',
            'approval': args.approval})
        review = reconcile(root, scheduler_snapshot=snapshot, apply=True)
        save(args.output / 'cancelled-reconciliation.json', review)
        if review['blocked']:
            raise ValueError('Cancelled Qwen results need review before replacement')
    elif not known or any(j['state'] not in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'COMPLETED'} for j in known):
        raise ValueError('Existing Qwen allocation missing or failed; inspect before expanding')
    environment = setup(args, source)
    entries = []
    estimate = (('Approved new Qwen members 1-5 after explicit cancellation; prior resource use recorded separately. ' if replace_cancelled
                 else 'Approved Qwen members 2-5; member 1 already submitted. ') + 'Throughput unmeasured. '
                'Each allocation admits missing cells for up to 175 minutes including startup, '
                'with five minutes for drain/cleanup; 12 GPU-hours each. '
                'All five share one frozen queue and atomic task ledger; no full completion promise.')
    for number in range(1 if replace_cancelled else 2, 6):
        directory = args.output / f'qwen-{number}'
        directory.mkdir(exist_ok=True)
        runtime = {**source, 'GEODML_EXECUTION_COMMIT': pin, 'GEODML_EXECUTION_REPOSITORY': str(REPO),
                   'GEODML_WORKER_LAUNCHER': str(REPO / 'analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh'),
                   'GEODML_WAVE_OUTPUT_ROOT': str(directory / 'results'),
                   'GEODML_WAVE_LOG_ROOT': str(directory / 'logs'),
                   'GEODML_WORKER_STDOUT': str(directory / 'slurm-%j.out'),
                   'GEODML_WORKER_STDERR': str(directory / 'slurm-%j.err'),
                   'GEODML_WORKER_INDEX': '0', 'GEODML_WORKER_COUNT': '1',
                   'GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY': '1'}
        # All workers traverse forward and use the same atomic ledger. Worker IDs
        # contain the Slurm job ID, so index zero is unique across allocations.
        save(directory / 'runtime.json', runtime)
        script = ('#!/bin/bash\nset -eo pipefail\nsource ' + shlex.quote(str(environment)) +
                  '\nexec python3 ' + shlex.quote(str(REPO / 'analysis/scripts/prepare_jupiter_llama.py')) +
                  ' execute --output ' + shlex.quote(str(directory)) + '\n')
        atomic(directory / 'run.sh', script.encode())
        frozen = {str(directory / n): first.identity(directory / n) for n in ('runtime.json', 'run.sh')}
        frozen.update(localize_reference(directory, reference, args.qwen_reference))
        save(directory / 'preparation.json', {**reference, 'git_commit': pin, 'files': frozen,
             'wave_member': f'qwen-{number}-of-5', 'estimate': estimate})
        command = ['sbatch', '--parsable', '--hold', '--no-requeue', '--nodes=1', '--ntasks=1',
                   '--gres=gpu:4', '--exclusive', '--cpus-per-task=32', '--mem=0', '--time=03:00:00',
                   '--account=' + args.account, '--partition=' + args.partition,
                   f'--job-name=geodml-qwen-threehour-{number}', '--chdir=' + str(REPO),
                   '--output=' + str(directory / 'slurm-%j.out'), '--error=' + str(directory / 'slurm-%j.err'),
                   str(directory / 'run.sh')]
        entries.append((directory, command, approval(args, estimate)))
    submit(args, entries, root, allowed)


def groups(state):
    rows = [(key, row) for key, row in state['hours'].items()
            if row['model'] == 'llama4' and not row['owner'] and row['status'] in {'available', 'partial'}
            and set(row['task_fingerprints']) - set(row['completed']) - set(row['failed'])]
    rows.sort(key=lambda item: (item[1]['priority_rank'], item[0]))
    if len(rows) < 5:
        raise ValueError('Fewer than five eligible Llama packages')
    # Reserve about twice the observed upper three-hour throughput. Keep each
    # group contiguous in keyword priority; remaining packages stay available.
    result, cursor = [], 0
    for _ in range(5):
        selected, size = [], 0
        while cursor < len(rows) and size < 9000:
            key, row = rows[cursor]
            selected.append(key)
            size += len(set(row['task_fingerprints']) - set(row['completed']) - set(row['failed']))
            cursor += 1
        if not selected:
            raise ValueError('Insufficient eligible Llama work for five allocations')
        result.append(selected)
    return result


def admit(state, attempts, snapshot, storage, args):
    if len(attempts) != 5 or len({a['attempt_id'] for a in attempts}) != 5:
        raise ValueError('Exactly five distinct Llama attempts required')
    if (not storage['safe_to_admit'] or not 0 <= time.time() - storage['captured_at_epoch'] <= 300
            or (not storage['quota_verified'] and not args.allow_stale_quota)):
        raise ValueError('Storage evidence blocks admission')
    prior = state['admission'].get('jupiter', {})
    terminal = {o.get('attempt_id') for o in snapshot['owners'] if o['state'] in TERMINAL_SCHEDULER_STATES}
    if (prior.get('pending_attempt') or
            not set(prior.get('finite_wave', {}).get('attempt_ids', [])) <= terminal):
        raise ValueError('Previous admission has not ended')
    gate(snapshot, args.allowed_jobs, 5, args.maximum_concurrent, now=time.time())
    result = deepcopy(state)
    selected = {h for a in attempts for h in a['owners']}
    if any(h.get('owner') and key not in selected for key, h in result['hours'].items()):
        raise ValueError('Other shared owners remain; reconcile them first')
    tickets = result['admission'].setdefault('jupiter', {}).setdefault('tickets', {})
    seen = set()
    for attempt in attempts:
        validate_request(attempt['request'], attempt['cluster_profile'])
        if (attempt['model'] != 'llama4' or attempt['request']['cluster'] != 'jupiter'
                or attempt['request']['approval']['walltime_seconds'] != 10800):
            raise ValueError('Only the five approved three-hour JUPITER Llama members are supported')
        if attempt['attempt_id'] in tickets:
            raise ValueError('Attempt already admitted; inspect saved receipts')
        for key, owner in attempt['owners'].items():
            hour = result['hours'][key]
            check_owner(hour, owner)
            work = set(hour['task_fingerprints'])
            if seen & work:
                raise ValueError('Wave work overlaps')
            seen.update(work)
        tickets[attempt['attempt_id']] = {
            'request_sha256': digest(attempt['request']), 'existing_job_id': None,
            'attempt_sha256': digest({**attempt, 'admission_ticket': None}),
            'admitted_at': int(time.time()), 'scheduler_sha256': digest(snapshot),
            'storage_sha256': digest(storage), 'quota_verified': storage['quota_verified'],
            'policy_exception': args.approval}
    result['admission']['jupiter']['finite_wave'] = {
        'attempt_ids': [a['attempt_id'] for a in attempts], 'maximum_gpu_hours': 60,
        'start_gap_seconds': 0, 'evidence': args.approval}
    return result


def llama(args, pin, exchange):
    gate(first.current_scheduler(args.since), args.allowed_jobs, 5, args.maximum_concurrent, now=time.time())
    site = read(args.llama_site)
    root = Path(site['dataset_root'])
    health(args, root)
    for path in site['attempts']:
        attempt = read(path)
        _, state = exchange.snapshot()
        sync_path = Path(path).parent / 'sync.json'
        if sync_path.exists():
            saved = read(sync_path)
            if saved.get('status') == 'released' and all(saved['bundle'] in state['hours'][h]['checkpoints'] for h in attempt['owners']):
                print('ALREADY SYNCED: ' + attempt['attempt_id'], flush=True)
                continue
        snapshot = scheduler(attempt)
        if not any(o['owner_id'] == attempt['writer_id'] and o['state'] in TERMINAL_SCHEDULER_STATES for o in snapshot['owners']):
            raise ValueError('Previous Llama allocation is not confirmed terminal')
        result = sync_once(exchange, attempt, snapshot, health(args, root))
        if result['status'] != 'released':
            raise ValueError('Previous Llama results could not be released: ' + str(result))
        print(json.dumps(result), flush=True)
    _, state = exchange.snapshot()
    if any(h.get('owner') for h in state['hours'].values()):
        raise ValueError('Existing shared owners remain')
    prior = read(site['attempts'][0])
    if prior['model'] != 'llama4':
        raise ValueError('Expected the previous Llama shared-hour site')
    runtime = read(args.llama_runtime)
    environment = setup(args, runtime)
    profile = {**prior['cluster_profile'], 'account': args.account, 'partition': args.partition,
               'environment_file': str(environment), 'cache_root': runtime['GEODML_CACHE_ROOT']}
    serving = {k: v for k, v in runtime.items() if k in FILE_KEYS | SETTING_KEYS
               and k != 'GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY'}
    attempts = []
    estimate = ('Previous five 55-minute Llama runs committed 1042-1395 cells each; '
                'rough three-hour range 3100-4500 cells per node, workload-dependent. '
                'Three hours approved, actual allocation deadline with drain margin, 12 GPU-hours each; '
                'five Llama members, maximum 60 GPU-hours. Frozen remaining packages only.')
    for number, hours in enumerate(groups(state), 1):
        name = 'llama-threehour-' + digest(str(args.output))[:12] + f'-{number}'
        request = {'attempt_id': name, 'cluster': 'jupiter', 'mode': 'batch', 'git_commit': pin,
                   'hour_ids': hours, 'since': args.since, 'attempt_dir': str(args.output / name),
                   'approval': approval(args, estimate)}
        attempts.append(stage(exchange, request=request, profile=profile, runtime=serving,
                              reference_profile=Path(runtime['SEARCH_AGENTIC_PROFILE']), dataset=root,
                              repository=REPO, operation_id='reserve-' + name))
    snapshot = first.current_scheduler(args.since)
    storage = health(args, root)
    exchange.transact('admit-' + digest(str(args.output))[:20], {'attempts': [a['attempt_id'] for a in attempts]},
                      lambda s: admit(s, attempts, snapshot, storage, args))
    _, state = exchange.snapshot()
    entries = []
    for attempt in attempts:
        attempt['admission_ticket'] = state['admission']['jupiter']['tickets'][attempt['attempt_id']]
        directory = Path(attempt['request']['attempt_dir'])
        save(directory / 'attempt.json', attempt)
        command = allocation_command(attempt['request'], profile, REPO)
        command.insert(1, '--hold')
        entries.append((directory, command, attempt['request']['approval']))
    save(args.output / 'site.json', {**site, 'source_commit': pin,
         'attempts': [str(d / 'attempt.json') for d, _, _ in entries], 'journal': str(args.output / 'journal')})
    submit(args, entries, root, args.allowed_jobs)


def submit(args, entries, root, allowed, round_id=None, receipt_extra=None):
    intent = args.output / ('submission-intent.json' if round_id is None
                            else f'relaunch-intent-{round_id}.json')
    if intent.exists():
        raise ValueError('Submission already attempted; inspect receipts, never resubmit')
    gate(first.current_scheduler(args.since), allowed, len(entries), args.maximum_concurrent, now=time.time())
    health(args, root)
    save(intent, {'members': [str(d) for d, _, _ in entries], 'approval': args.approval,
                  'existing_qwen_job': args.existing_qwen_job, 'maximum_total_gpu_hours': 120,
                  'maximum_concurrent': args.maximum_concurrent, 'start_gap_seconds': 0,
                  'quota_exception': args.allow_stale_quota,
                  'replaces_cancelled_qwen': args.replace_cancelled_qwen})
    jobs = []
    for directory, command, approved in entries:
        health(args, root)
        health(args, Path(read(args.qwen_reference / 'preparation.json')['dataset_root']))
        gate(first.current_scheduler(args.since), [*allowed, *jobs], 1, args.maximum_concurrent, now=time.time())
        receipt = {'status': 'submission_requested', 'command': command, 'approval': approved}
        save(directory / 'submission.json', receipt)
        raw = first.command(command)
        atomic(directory / 'sbatch-response.txt', raw.encode())
        job = raw.split(';')[0]
        if not job.isdigit():
            raise ValueError('Uncertain sbatch response; inspect receipts, never resubmit')
        jobs.append(job)
        save(directory / 'submission.json', {**receipt, 'job_id': job, 'status': 'submitted_held'})
        print('HELD_JOB=' + job, flush=True)
    snapshot = first.current_scheduler(args.since)
    gate(snapshot, [*allowed, *jobs], 0, args.maximum_concurrent, now=time.time())
    held = {str(j['job_id']) for j in snapshot['jobs'] if j['state'] == 'PENDING' and j.get('held')}
    if not set(jobs) <= held:
        raise ValueError('All new allocations must be confirmed held before release')
    for (directory, _, _), job in zip(entries, jobs, strict=True):
        health(args, root)
        health(args, Path(read(args.qwen_reference / 'preparation.json')['dataset_root']))
        gate(first.current_scheduler(args.since), [*allowed, *jobs], 0, args.maximum_concurrent, now=time.time())
        # Save the confirmed job identity before release; execute reads this receipt.
        receipt = read(directory / 'submission.json')
        save(directory / 'submission.json', {**receipt, 'status': 'submitted'})
        first.command(['scontrol', 'release', job])
        print('RELEASED_JOB=' + job, flush=True)
    final = args.output / ('submitted.json' if round_id is None else f'relaunch-{round_id}.json')
    save(final, {'job_ids': jobs, 'maximum_gpu_hours': 12 * len(jobs), **(receipt_extra or {})})
    print(json.dumps({'job_ids': jobs, 'new_gpu_hours': 12 * len(jobs),
                      'relaunch_round': round_id}, default=str), flush=True)


def relaunch_qwen(args, pin):
    """Resubmit only Slurm-confirmed FAILED Qwen members of the dispatched wave.

    Live, completed, cancelled and deadline members are never touched. Old
    receipts rotate to job-suffixed files; the frozen member-one queue and all
    scientific settings are reused unchanged. Requires fresh explicit approval.
    """
    reference, _ = verify_reference(args.qwen_reference)
    root = Path(reference['dataset_root'])
    guard = args.qwen_reference / 'expansion-qwen.json'
    if not guard.exists() or read(guard).get('output') != str(args.output):
        raise ValueError('Relaunch must target the originally dispatched wave directory')
    dispatched = read(args.output / 'submitted.json')
    wave_jobs = {str(j) for j in dispatched['job_ids']}
    members = []
    for number in range(1, 6):
        directory = args.output / f'qwen-{number}'
        receipt = read(directory / 'submission.json')
        job = str(receipt.get('job_id', ''))
        if receipt.get('status') != 'submitted' or job not in wave_jobs:
            raise ValueError(f'qwen-{number} receipt is not part of the dispatched wave')
        members.append((number, directory, job))
    snapshot = first.current_scheduler(args.since)
    if snapshot.get('complete') is not True or not 0 <= time.time() - snapshot['captured_at_epoch'] <= 120:
        raise ValueError('Fresh complete scheduler evidence required')
    failed, live_jobs, states = [], [], {}
    for number, directory, job in members:
        state = member_state(job, snapshot)
        states[job] = state
        if state is None:
            raise ValueError(f'No scheduler evidence for job {job} (qwen-{number}); inspect manually')
        if state not in TERMINAL_SCHEDULER_STATES:
            live_jobs.append(job)
        elif state == 'FAILED':
            failed.append((number, directory, job))
    allowed = [*live_jobs, *(read(args.live_wave)['job_ids'] if args.live_wave else [])]
    if not failed:
        result = {'status': 'no_failed_members', 'member_states': states}
        print(json.dumps(result), flush=True)
        return result
    gate(snapshot, allowed, len(failed), args.maximum_concurrent, now=time.time())
    health(args, root)
    round_id = 1 + len(list(args.output.glob('relaunch-[0-9]*.json')))
    if (args.output / f'relaunch-intent-{round_id}.json').exists():
        raise ValueError('Relaunch round already attempted; inspect receipts, never resubmit')
    environment = args.output / 'environment.sh'
    if not environment.exists():
        raise ValueError('Wave environment.sh is missing; cannot relaunch')
    source = read(args.qwen_reference / 'runtime.json')
    estimate = ('Relaunch of Qwen wave members whose allocations FAILED within seconds at pre-inference '
                'verification (frozen member-one paths outside the member directory crashed '
                'verify_prepared before any cell ran; failed attempts consumed seconds and are recorded '
                'separately). Same frozen queue and atomic ledger as member one; throughput unmeasured. '
                'Each allocation admits missing cells for up to 175 minutes including startup, with five '
                'minutes for drain/cleanup; 12 GPU-hours each.')
    entries, accounting = [], {}
    for number, directory, job in failed:
        accounting[job] = {'member': f'qwen-{number}', 'state': 'FAILED', 'sacct': first.command(
            ['sacct', '-X', '-j', job, '--noheader', '--parsable2',
             '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'])}
        os.replace(directory / 'submission.json', directory / f'submission-failed-{job}.json')
        if (directory / 'preparation.json').exists():
            os.replace(directory / 'preparation.json', directory / f'preparation-failed-{job}.json')
        if (directory / 'sbatch-response.txt').exists():
            os.replace(directory / 'sbatch-response.txt', directory / f'sbatch-response-{job}.txt')
        runtime = {**source, 'GEODML_EXECUTION_COMMIT': pin, 'GEODML_EXECUTION_REPOSITORY': str(REPO),
                   'GEODML_WORKER_LAUNCHER': str(REPO / 'analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh'),
                   'GEODML_WAVE_OUTPUT_ROOT': str(directory / 'results'),
                   'GEODML_WAVE_LOG_ROOT': str(directory / 'logs'),
                   'GEODML_WORKER_STDOUT': str(directory / 'slurm-%j.out'),
                   'GEODML_WORKER_STDERR': str(directory / 'slurm-%j.err'),
                   'GEODML_WORKER_INDEX': '0', 'GEODML_WORKER_COUNT': '1',
                   'GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY': '1'}
        save(directory / 'runtime.json', runtime)
        script = ('#!/bin/bash\nset -eo pipefail\nsource ' + shlex.quote(str(environment)) +
                  '\nexec python3 ' + shlex.quote(str(REPO / 'analysis/scripts/prepare_jupiter_llama.py')) +
                  ' execute --output ' + shlex.quote(str(directory)) + '\n')
        atomic(directory / 'run.sh', script.encode())
        frozen = {str(directory / n): first.identity(directory / n) for n in ('runtime.json', 'run.sh')}
        frozen.update(localize_reference(directory, reference, args.qwen_reference))
        save(directory / 'preparation.json', {**reference, 'git_commit': pin, 'files': frozen,
             'wave_member': f'qwen-{number}-of-5-relaunch-{round_id}', 'estimate': estimate,
             'relaunch_of': job})
        command = ['sbatch', '--parsable', '--hold', '--no-requeue', '--nodes=1', '--ntasks=1',
                   '--gres=gpu:4', '--exclusive', '--cpus-per-task=32', '--mem=0', '--time=03:00:00',
                   '--account=' + args.account, '--partition=' + args.partition,
                   f'--job-name=geodml-qwen-threehour-{number}-r{round_id}', '--chdir=' + str(REPO),
                   '--output=' + str(directory / 'slurm-%j.out'), '--error=' + str(directory / 'slurm-%j.err'),
                   str(directory / 'run.sh')]
        entries.append((directory, command, approval(args, estimate)))
    return submit(args, entries, root, allowed, round_id=round_id, receipt_extra={
        'relaunch_round': round_id, 'failed_jobs': accounting, 'member_states': states,
        'approval': args.approval, 'git_commit': pin})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=['qwen', 'llama', 'relaunch-qwen'])
    for key in ('output', 'qwen-reference'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('environment-file', 'llama-site', 'llama-runtime'):
        p.add_argument('--' + key, type=Path)
    for key in ('since', 'account', 'partition', 'approval'):
        p.add_argument('--' + key, required=True)
    for key in ('existing-qwen-job', 'repo-id'):
        p.add_argument('--' + key)
    p.add_argument('--maximum-concurrent', type=int, choices=[5, 10], default=5)
    p.add_argument('--qwen-wave', type=Path)
    p.add_argument('--live-wave', type=Path,
                   help='submitted.json of a wave allowed to remain live during relaunch-qwen')
    p.add_argument('--replace-cancelled-qwen', action='store_true')
    p.add_argument('--simultaneous-starts', action='store_true')
    p.add_argument('--allow-stale-quota', action='store_true')
    p.add_argument('--quota', type=Path)
    args = p.parse_args()
    if not args.approval.strip() or not args.simultaneous_starts:
        raise ValueError('Explicit finite-wave walltime and simultaneous-start approval required')
    required = {'qwen': ('environment_file', 'existing_qwen_job', 'repo_id'),
                'llama': ('environment_file', 'existing_qwen_job', 'repo_id', 'llama_site', 'llama_runtime'),
                'relaunch-qwen': ()}
    missing = ['--' + name.replace('_', '-') for name in required[args.model]
               if getattr(args, name) in (None, '')]
    if missing:
        raise ValueError('Missing required arguments: ' + ', '.join(missing))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.qwen_reference / 'expansion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin = first.clean_commit()
        args.allowed_jobs = []
        if args.model == 'relaunch-qwen':
            relaunch_qwen(args, pin)
            return
        # One fixed receipt location per model under the original preparation prevents
        # alternate --output paths from duplicating the approved wave.
        guard = args.qwen_reference / ('expansion-' + args.model + '.json')
        if guard.exists() and read(guard)['output'] != str(args.output):
            raise ValueError('This wave already belongs to another output directory')
        if (args.output / 'submission-intent.json').exists():
            raise ValueError('Submission already attempted; inspect saved receipts')
        save(guard, {'output': str(args.output), 'git_commit': pin})
        if args.model == 'llama' and args.maximum_concurrent == 10:
            if not args.qwen_wave:
                raise ValueError('Ten-way mode requires the saved Qwen wave receipt')
            args.allowed_jobs = read(args.qwen_wave / 'submitted.json')['job_ids']
            if not args.replace_cancelled_qwen:
                args.allowed_jobs = [args.existing_qwen_job, *args.allowed_jobs]
        from huggingface_hub import get_token
        if args.model == 'llama' or not get_token():
            os.environ['HF_TOKEN'] = getpass.getpass('HF WRITE token (hidden): ' if args.model == 'llama' else 'HF token (hidden): ').strip()
        exchange = Exchange(HubStore(args.repo_id), args.output / 'journal')
        (qwen if args.model == 'qwen' else llama)(args, pin, exchange)


if __name__ == '__main__':
    main()
