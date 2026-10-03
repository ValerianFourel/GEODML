#!/usr/bin/env python3
"""One finite recovery sweep over the existing HoreKa legacy Qwen dataset.

Login commands transfer verified Hub metadata and submit at most one allocation
per admission. The finite sender drives preparation, submission and final audit.
CPU jobs reconcile ended writers and inventory the dataset; GPU jobs reuse the
unchanged bout executor. No old division, sender, scientific setting or HF hour
assignment is replaced. A second recovery sweep is never created automatically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.agentic_hour_runtime import AdmissionDeferred
from analysis.interpretability.pipeline.agentic_hours import canonical, digest
from analysis.scripts import horeka_qwen_bouts as bouts
from analysis.scripts.capture_agentic_scheduler_snapshot import capture
from analysis.scripts.reconcile_agentic_dataset import TERMINAL_SCHEDULER_STATES, reconcile

FORMAT = 'geodml-horeka-qwen-recovery-v1'
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    atomic(Path(path), canonical(value))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def locked(root, *, blocking=False, name='operator.lock'):
    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise AdmissionDeferred(f'another recovery operator or CPU audit holds {root / name}') from error
        yield


def checked_context(root):
    context = read(root / 'recovery.json')
    if context.get('format_version') != FORMAT:
        raise ValueError('unsupported recovery context')
    pin = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    if dirty or pin != context['git_commit']:
        raise ValueError('use the clean pinned recovery checkout')
    source = Path(context['source_division'])
    if file_hash(source / 'division.json') != context['source_division_sha256']:
        raise ValueError('source division changed')
    if digest(read(context['plan'])) != context['plan_sha256']:
        raise ValueError('original frozen plan changed')
    return context


def scheduler(context):
    value = capture(plan={'plan_id': 'horeka-qwen-recovery'}, since=context['since'], include_all_jobs=True)
    value['cluster'] = 'horeka'
    return value


def assert_old_pass_ended(context, snapshot):
    live = {row['job_id'] for row in snapshot['jobs']}
    terminal = {row['job_id'] for row in snapshot['owners'] if row['state'] in TERMINAL_SCHEDULER_STATES}
    source = Path(context['source_division'])
    for entry in read(source / 'division.json')['bouts']:
        path = source / 'bouts' / f"bout-{entry['number']:04d}" / 'submission.json'
        if not path.is_file():
            raise ValueError(f'original bout {entry["number"]} was not submitted; finish the original sender first')
        receipt = read(path)
        job = str(receipt.get('stdout', '')).strip().split(';')[0]
        if receipt.get('returncode') != 0 or not job.isdigit():
            raise ValueError(f'original bout {entry["number"]} has no confirmed allocation; inspect its receipt')
        if job in live:
            raise AdmissionDeferred(f'original Qwen job {job} is live or lacks terminal accounting; preserve it')
        if job not in terminal:
            raise ValueError(f'original Qwen job {job} is live or lacks terminal accounting; preserve it')


def assert_previous_recovery_ended(context, snapshot):
    """Never overlap the earlier operator page's independently pinned division."""
    previous = context.get('previous_recovery')
    if not previous or not (Path(previous) / 'recovery.json').exists():
        return
    root = Path(previous)
    prior = read(root / 'recovery.json')
    if Path(prior['dataset_root']).resolve() != Path(context['dataset_root']).resolve():
        raise ValueError('previous recovery uses a different dataset')
    for phase in ('preparation', 'final'):
        if (root / phase / 'SUBMISSION_ATTEMPTED').exists():
            cpu_phase_finished(root, phase, snapshot)
    ready = root / 'preparation/ready.json'
    if ready.exists():
        jobs = confirmed_submissions(root, snapshot)
        if len(jobs) != read(ready)['maximum_allocations']:
            raise ValueError(f'previous recovery still has unsubmitted work: {root}; preserve that division')
        if set(jobs) & {r['job_id'] for r in snapshot['jobs']}:
            raise AdmissionDeferred('previous recovery GPU allocations are still live; preserving them')
    elif not (root / 'preparation/coverage.json').exists() or read(root / 'preparation/coverage.json')['eligible_missing']:
        raise ValueError(f'previous recovery is unfinished: {root}; inspect it before creating another division')


def storage(context, directory):
    from analysis.scripts.prepare_horeka_qwen import capture_quota
    from analysis.scripts.manage_agentic_hours import health
    quota = directory / f'quota-{time.time_ns()}.json'
    save(quota, capture_quota(Path(context['workspace']), context['account']))
    value = health({'dataset_root': context['dataset_root'], 'cluster': 'horeka'}, quota)
    if not value.get('safe_to_admit') or not value.get('quota_verified'):
        raise ValueError('storage or quota evidence blocks new work')
    return value


def admit(context, snapshot, health, attempt):
    from analysis.interpretability.pipeline.agentic_hour_runtime import admission
    # Count every user allocation, including interactive allocations with ordinary names.
    if any(row['state'] not in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'STAGE_OUT'}
           for row in snapshot['jobs']):
        raise ValueError('unclassified live scheduler state; inspect it before admission')
    return admission({'admission': {}}, {'cluster': 'horeka', 'attempt_id': attempt}, snapshot, health, now=int(time.time()))


def remote_snapshot(context, directory):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
    from analysis.scripts.publish_qwen_results import read_index
    exchange = Exchange(HubStore(context['repo_id']), directory / 'hub-journal')
    revision, registry = exchange.snapshot()
    states = {}
    for bundle in read_index(exchange.store, revision)['bundles']:
        for fp, event in exchange.manifest(bundle['bundle'], revision=revision)['outcomes'].items():
            state = event['state']
            if state not in {'completed', 'terminal_failed'} or fp in states and states[fp] != state:
                raise ValueError('conflicting or nonterminal published Qwen outcome')
            states[fp] = state
    registered = set()
    for hour in registry['hours'].values():
        if hour['model'] == 'qwen38':
            registered.update(hour['task_fingerprints'])
    result = {'revision': revision, 'captured_at_epoch': int(time.time()), 'states': states,
              'hf_registered_qwen': sorted(registered)}
    save(directory / 'hub-snapshot.json', result)
    return result


def measured_rate(source, terminal_jobs):
    samples = []
    for path in source.glob('bouts/bout-*/attempts/job*/bout-result.json'):
        result = read(path)
        direct = result.get('direct_dataset') or {}
        done = direct.get('committed_count')
        elapsed = result.get('elapsed_seconds')
        job = str(result.get('job_id', ''))
        if (job in terminal_jobs and result.get('status') == 'complete'
                and type(done) is int and done >= 12 and direct.get('reused_count') == 0
                and isinstance(elapsed, (int, float)) and elapsed > 600
                and not result.get('validation_error') and result.get('returncode', 0) == 0):
            samples.append({'job_id': job, 'new_cells': done, 'elapsed_seconds': elapsed,
                            'seconds_per_cell': (elapsed - 360) / done})
    if not samples:
        raise ValueError('no successful unreused Qwen throughput samples; return the original bout logs')
    rates = sorted(row['seconds_per_cell'] for row in samples)
    # One budgeted sweep, sized conservatively from completed allocations.
    return {'seconds_per_cell': rates[math.ceil(.9 * len(rates)) - 1],
            'median_seconds_per_cell': statistics.median(rates), 'samples': samples,
            'startup_seconds': 600, 'historical_startup_seconds': 360}


def freeze_report(context, root, remote, snapshot, *, final=False):
    from analysis.interpretability.pipeline.agentic_hours import inventory
    from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
    assert_old_pass_ended(context, snapshot)
    assert_previous_recovery_ended(context, snapshot)
    dataset = Path(context['dataset_root'])
    review = reconcile(dataset, scheduler_snapshot=snapshot, stripe_count=256, apply=True)
    save(root / 'reconciliation.json', review)
    tasks, completed, blocked = inventory(dataset, stripes=256, reuse_verified=True)
    plan = read(context['plan'])
    published = set(remote['states'])
    rows = bouts.remaining_rows(plan, tasks, completed, blocked, published)
    overlap = set(r['fingerprint'] for r in rows) & set(remote['hf_registered_qwen'])
    if overlap:
        raise ValueError('remaining cells belong to HF hour plans; use their registered workflow instead')
    latest = StripedTaskLedger(dataset / 'control/task-ledger', stripe_count=256).snapshot()['latest']
    qwen = {r['fingerprint'] for r in tasks if r['model'] == 'qwen38'}
    done = completed | {fp for fp, state in remote['states'].items() if state == 'completed'}
    failed = {fp for fp, event in latest.items() if event['state'] == 'terminal_failed'} | {
        fp for fp, state in remote['states'].items() if state == 'terminal_failed'}
    missing = {r['fingerprint'] for r in rows}
    conflicts = qwen & done & failed
    if conflicts:
        raise ValueError('local and published outcomes conflict')
    unresolved = qwen - done - failed - missing
    report = {'registered_qwen': len(qwen), 'verified_completed': len(qwen & done),
              'terminal_failed': len(qwen & failed), 'eligible_missing': len(missing),
              'blocked_or_unverified': len(unresolved), 'blocked_fingerprints': sorted(unresolved),
              'reconciliation_blocked': review['blocked'], 'hub_revision': remote['revision'],
              'complete': qwen <= done and not review['blocked'], 'scientific_acceptance': False}
    save(root / 'coverage.json', report)
    if final or not rows:
        return report
    terminal = {r['job_id'] for r in snapshot['owners'] if r['state'] == 'COMPLETED'}
    rate = measured_rate(Path(context['source_division']), terminal)
    ceiling = context.get('gpu_walltime_seconds', 3600)
    seconds = min(ceiling, max(1200, math.ceil((len(rows) * rate['seconds_per_cell'] + 1020) / 300) * 300))
    walltime = f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00'
    args = SimpleNamespace(output=root.parent / 'division', dataset=dataset, plan=Path(context['plan']),
                           walltime=walltime, seconds_per_cell=rate['seconds_per_cell'], startup_seconds=600,
                           overbook=2.0, measurement='recovery measured-throughput.json: successful terminal jobs')
    bouts.write_division(args, plan, tasks, completed, blocked, published, remote['revision'])
    save(root / 'measured-throughput.json', rate)
    division = read(args.output / 'division.json')
    frozen = {str(p.relative_to(args.output)): file_hash(p) for p in args.output.rglob('*') if p.is_file()}
    estimate = {'resources_per_bout': {'nodes': 1, 'gpus': 4, 'gpu_type': 'A100', 'cpus': 32, 'memory': 'whole node'},
                'primary_runtime_seconds': [
                    600 + min(b['primary_cells'] for b in division['bouts']) * rate['median_seconds_per_cell'],
                    600 + max(b['primary_cells'] for b in division['bouts']) * rate['seconds_per_cell']],
                'walltime_seconds': seconds, 'startup_allowance_seconds': 600,
                'drain_cleanup_seconds': 420,
                'cheaper_alternative': 'Saved answers are finalized first; a backlog smaller than one bout gets a shorter allocation. No automatic retry sweep.'}
    save(root / 'ready.json', {'coverage': report, 'frozen_files': frozen, 'totals': division['totals'],
                             'sizing': division['sizing'], 'estimate': estimate,
                             'maximum_allocations': division['counts']['bouts']})
    return report


def confirmed_submissions(root, snapshot):
    known = {row['job_id'] for row in snapshot['jobs'] + snapshot['owners']}
    jobs = []
    for marker in (root / 'division/bouts').glob('bout-*/SUBMISSION_ATTEMPTED'):
        if bouts.refused_submission(marker.parent):
            continue
        receipt_path = marker.parent / 'submission.json'
        if not receipt_path.exists():
            raise ValueError('ambiguous submission; reconcile its durable marker before any new job')
        receipt = read(receipt_path)
        job = str(receipt.get('stdout', '')).strip().split(';')[0]
        if receipt.get('returncode') != 0 or not job.isdigit() or job not in known:
            raise ValueError('a previous recovery submission lacks confirmed accounting; inspect its receipt')
        jobs.append(job)
    return jobs


def before_gpu_submission(root, numbers):
    context = checked_context(root)
    ready = read(root / 'preparation/ready.json')
    if len(numbers) != 1 or not 1 <= numbers[0] <= ready['maximum_allocations']:
        raise ValueError('submit exactly one member of this finite recovery division')
    for rel, expected in ready['frozen_files'].items():
        if file_hash(root / 'division' / rel) != expected:
            raise ValueError('frozen recovery division changed')
    check = root / f'admission-{time.time_ns()}'
    check.mkdir()
    remote = remote_snapshot(context, check)
    selected = read(root / 'division/bouts' / f'bout-{numbers[0]:04d}' / 'bout.json')
    fingerprints = set(selected['primary_fingerprints'] + selected['spill_fingerprints'])
    if fingerprints & set(remote['hf_registered_qwen']):
        raise ValueError('HF now registers these cells; preserve ownership and stop legacy admission')
    if fingerprints & set(remote['states']):
        raise ValueError('selected cells gained published outcomes; reconcile those outcomes before new inference')
    healthy = storage(context, check)
    snapshot = scheduler(context)
    assert_old_pass_ended(context, snapshot)
    assert_previous_recovery_ended(context, snapshot)
    confirmed_submissions(root, snapshot)
    ticket = admit(context, snapshot, healthy, f'qwen-recovery-{numbers[0]}')
    save(check / 'scheduler.json', snapshot)
    save(check / 'admission.json', ticket)
    print(json.dumps({'finite_budget': ready['totals'], 'next_bout': numbers[0], 'sizing': ready['sizing'],
                      'estimate': ready.get('estimate')}, indent=2), flush=True)


def submit_cpu(root, context, phase):
    directory = root / phase
    if directory.exists():
        if (directory / 'SUBMISSION_ATTEMPTED').exists():
            if not bouts.refused_submission(directory):
                print(f'Existing {phase} preserved. Inspect {directory}; no duplicate CPU job submitted.')
                return
        directory.rename(root / f'{phase}-unsubmitted-{time.time_ns()}')
    directory.mkdir()
    remote_snapshot(context, directory)  # Finite transfer on login, before compute allocation.
    healthy = storage(context, directory)
    snapshot = scheduler(context)
    assert_old_pass_ended(context, snapshot)
    assert_previous_recovery_ended(context, snapshot)
    jobs = confirmed_submissions(root, snapshot)
    if phase == 'final' and (root / 'preparation/ready.json').exists():
        if len(jobs) != read(root / 'preparation/ready.json')['maximum_allocations']:
            raise ValueError('finite sweep still has unsubmitted bouts; use next before finalizing')
    live = {r['job_id'] for r in snapshot['jobs']}
    if set(jobs) & live:
        raise ValueError('recovery GPU jobs are live; wait before final reconciliation')
    save(directory / 'admission.json', admit(context, snapshot, healthy, 'qwen-recovery-' + phase))
    command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
               '--mem=16G', '--time=01:00:00', '--partition=cpuonly', '--account=' + context['account'],
               '--job-name=geodml-qwen-recovery-' + phase, '--chdir=' + str(REPO),
               '--output=' + str(directory / 'slurm-%j.out'), '--error=' + str(directory / 'slurm-%j.err'),
               '--wrap=' + shlex.join([sys.executable, str(Path(__file__).resolve()), 'cpu',
                                       '--output', str(root), '--phase', phase])]
    save(directory / 'submission-command.json', command)
    (directory / 'SUBMISSION_ATTEMPTED').write_text(str(time.time()))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    save(directory / 'submission.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?', result.stdout.strip()):
        raise ValueError('CPU submission not confirmed; inspect its saved receipt before retrying')
    print(f'{phase.upper()}_JOB={result.stdout.strip()} LOGS={directory}', flush=True)


def initialize(args, root):
    if (root / 'recovery.json').exists():
        context = checked_context(root)
        if args.walltime and bouts.walltime_seconds(args.walltime) != context.get('gpu_walltime_seconds', 3600):
            raise ValueError('existing recovery wall-time is frozen; use a separate output and --previous-recovery')
        if args.previous_recovery and str(args.previous_recovery.resolve()) != context.get('previous_recovery'):
            raise ValueError('existing previous-recovery guard is frozen; preserve this context')
        return context
    if not args.workspace or not args.account:
        raise ValueError('prepare/send requires --workspace and --account')
    workspace = args.workspace.resolve(strict=True)
    source = Path((workspace / 'qwen-bouts/CURRENT').read_text().strip()).resolve(strict=True)
    division = read(source / 'division.json')
    if division.get('format_version') != bouts.DIVISION_VERSION or division.get('model') != 'qwen38':
        raise ValueError('expected the existing legacy Qwen bout division')
    from analysis.scripts.audit_horeka_saved_progress import qwen_counts
    layout = qwen_counts(workspace)['qwen_unique_cells']
    if layout.get('ledger_stripes') != 256:
        raise ValueError('existing Qwen executor requires a verified 256-stripe ledger; layout is unavailable or differs')
    save(root / 'initial-counts.json', layout)
    pin = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    seconds = bouts.walltime_seconds(args.walltime or '01:00:00')
    previous = args.previous_recovery.resolve() if args.previous_recovery else None
    if previous == root:
        raise ValueError('previous recovery must be a different directory')
    context = {'format_version': FORMAT, 'workspace': str(workspace), 'source_division': str(source),
               'source_division_sha256': file_hash(source / 'division.json'), 'git_commit': pin,
               'dataset_root': division['dataset_root'], 'plan': division['plan'],
               'plan_sha256': division['plan_sha256'], 'account': args.account, 'repo_id': args.repo_id,
               'since': args.since, 'cpu_budget': {'allocations': 2, 'cpus': 4, 'hours_each': 1, 'memory': '16G'},
               'gpu_walltime_seconds': seconds, 'previous_recovery': str(previous) if previous else None,
               'authorization': f'User requested finalizing Qwen and relaunching missed cells in bouts up to {seconds / 3600:g} hours; one finite sweep, no automatic retries'}
    save(root / 'recovery.json', context)
    return checked_context(root)


def submit_next(root, context, snapshot):
    ready = read(root / 'preparation/ready.json')
    confirmed_submissions(root, snapshot)
    todo = [n for n in range(1, ready['maximum_allocations'] + 1)
            if not (root / 'division/bouts' / f'bout-{n:04d}' / 'SUBMISSION_ATTEMPTED').exists()
            or bouts.refused_submission(root / 'division/bouts' / f'bout-{n:04d}')]
    if not todo:
        return False
    # Defer before model/input verification and Hub transfer; the final hook checks again.
    admit(context, snapshot, storage(context, root), f'qwen-recovery-{todo[0]}')
    bouts.submit(SimpleNamespace(division=root / 'division', workspace=Path(context['workspace']),
        first=todo[0], count=1, account=context['account'], partition='accelerated', reservation=None,
        approved_walltime=ready['sizing']['walltime'], approved_count=ready['maximum_allocations'],
        approval=context['authorization'] + '; maximum GPU-hours ' + str(ready['totals']['gpu_hours']), dry_run=False))
    return True


def cpu_phase_finished(root, phase, snapshot):
    directory = root / phase
    if not (directory / 'SUBMISSION_ATTEMPTED').exists():
        return False
    receipt = directory / 'submission.json'
    if not receipt.exists():
        raise ValueError(f'ambiguous {phase} submission; inspect {directory}')
    value = read(receipt)
    job = str(value.get('stdout', '')).strip().split(';')[0]
    if value.get('returncode') != 0 or not job.isdigit():
        raise ValueError(f'{phase} submission failed or is ambiguous; no automatic retry: {directory}')
    if job in {r['job_id'] for r in snapshot['jobs']}:
        raise AdmissionDeferred(f'{phase} job {job} is still queued/running')
    states = {r['job_id']: r['state'] for r in snapshot['owners']}
    if states.get(job) != 'COMPLETED':
        raise ValueError(f'{phase} job {job} is failed or lacks terminal accounting; inspect {directory}')
    if not (directory / 'coverage.json').exists():
        raise ValueError(f'{phase} job {job} ended without a coverage report; inspect {directory}')
    if phase == 'preparation' and read(directory / 'coverage.json')['eligible_missing'] and not (directory / 'ready.json').exists():
        raise ValueError(f'preparation ended without a frozen division; inspect {directory}')
    return True


def sender_step(root, context):
    """Advance one phase; only valid scheduling waits may be retried automatically."""
    snapshot = scheduler(context)
    assert_old_pass_ended(context, snapshot)
    assert_previous_recovery_ended(context, snapshot)
    if not cpu_phase_finished(root, 'preparation', snapshot):
        admit(context, snapshot, storage(context, root), 'qwen-recovery-preparation')
        print('CPU audit estimate: 5-45 minutes, 4 CPUs, 16 GiB, one-hour ceiling. No GPUs.', flush=True)
        submit_cpu(root, context, 'preparation')
        return None
    if (root / 'preparation/ready.json').exists():
        # Explicit sbatch refusals can be retried by an operator, never by this sender.
        if any(bouts.refused_submission(p) for p in (root / 'division/bouts').glob('bout-*')):
            raise ValueError('a GPU submission was refused; inspect its receipt before any manual retry')
        if submit_next(root, context, snapshot):
            return None
        jobs = confirmed_submissions(root, snapshot)
        if set(jobs) & {r['job_id'] for r in snapshot['jobs']}:
            raise AdmissionDeferred('finite sweep submitted; waiting for its allocations to end')
    if not cpu_phase_finished(root, 'final', snapshot):
        admit(context, snapshot, storage(context, root), 'qwen-recovery-final')
        print('Final CPU audit estimate: 5-45 minutes, 4 CPUs, 16 GiB, one-hour ceiling. No GPUs.', flush=True)
        submit_cpu(root, context, 'final')
        return None
    return read(root / 'final/coverage.json')


def send(root):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('run the sender on a login host outside an allocation')
    context = checked_context(root)
    # One login-side sender per workspace. CPU jobs take only the short operator lock.
    with locked(Path(context['workspace']) / 'qwen-bouts', name='recovery-sender.lock'):
        state_path = root / 'sender.json'
        if not state_path.exists():
            save(state_path, {'started_at_epoch': int(time.time()),
                              'deadline_epoch': int(time.time()) + 7 * 24 * 3600,
                              'poll_seconds': 60, 'automatic_retries': 0})
        state = read(state_path)
        while time.time() < state['deadline_epoch']:
            try:
                with locked(root):
                    report = sender_step(root, checked_context(root))
                if report is not None:
                    save(root / 'sender-status.json', {'state': 'finished', 'coverage': report})
                    print('SWEEP_FINISHED', json.dumps(report, sort_keys=True), flush=True)
                    return 0 if report['complete'] else 2
                status = {'state': 'submitted', 'captured_at_epoch': int(time.time())}
            except AdmissionDeferred as error:
                status = {'state': 'waiting', 'reason': str(error), 'captured_at_epoch': int(time.time())}
            except Exception as error:
                save(root / 'sender-status.json', {'state': 'blocked', 'reason': str(error),
                                                   'captured_at_epoch': int(time.time())})
                raise
            save(root / 'sender-status.json', status)
            print(json.dumps(status, sort_keys=True), flush=True)
            time.sleep(min(state['poll_seconds'], max(0, state['deadline_epoch'] - time.time())))
        save(root / 'sender-status.json', {'state': 'expired', 'deadline_epoch': state['deadline_epoch']})
        print('SENDER_EXPIRED: seven-day limit reached; all allocations and saved work preserved.', flush=True)
        return 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'send', 'next', 'status', 'finalize', 'cpu'])
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--account')
    parser.add_argument('--repo-id', default='ValerianFourel/geodml-experiment-v2-paper-private')
    parser.add_argument('--since', default='2026-09-25')
    parser.add_argument('--phase', choices=['preparation', 'final'])
    parser.add_argument('--walltime', choices=['01:00:00', '05:00:00'],
                        help='Explicitly authorized GPU ceiling, frozen on preparation; default one hour')
    parser.add_argument('--previous-recovery', type=Path,
                        help='Earlier recovery directory whose work must finish before this sweep')
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.command == 'send':
        if os.environ.get('SLURM_JOB_ID'):
            raise ValueError('run the sender on a login host outside an allocation')
        # Existing contexts are immutable and atomically written. A CPU audit may
        # hold operator.lock while a restarted sender reads its pinned context.
        with nullcontext() if (root / 'recovery.json').exists() else locked(root):
            initialize(args, root)
        return send(root)
    with nullcontext() if args.command == 'status' else locked(root, blocking=args.command == 'cpu'):
        if args.command == 'prepare':
            initialize(args, root)
        context = checked_context(root)
        if args.command in {'prepare', 'finalize'}:
            print('CPU audit estimate: 5-45 minutes, 4 CPUs, 16 GiB, one-hour ceiling. No GPUs.', flush=True)
            submit_cpu(root, context, 'preparation' if args.command == 'prepare' else 'final')
        elif args.command == 'cpu':
            if not os.environ.get('SLURM_JOB_ID') or not args.phase:
                raise ValueError('CPU reconciliation requires its Slurm allocation and phase')
            directory = root / args.phase
            receipt = read(directory / 'submission.json')
            if receipt['returncode'] != 0 or receipt['stdout'].strip().split(';')[0] != os.environ['SLURM_JOB_ID']:
                raise ValueError('CPU allocation does not match its submission receipt')
            with (directory / 'STARTED').open('x') as marker:
                marker.write(os.environ['SLURM_JOB_ID'])
            storage(context, directory)
            snapshot = scheduler(context)
            save(directory / 'scheduler.json', snapshot)
            report = freeze_report(context, directory, read(directory / 'hub-snapshot.json'), snapshot, final=args.phase == 'final')
            print(json.dumps(report, indent=2), flush=True)
        elif args.command == 'next':
            if not (root / 'preparation/ready.json').exists():
                if (root / 'preparation/coverage.json').exists():
                    coverage = read(root / 'preparation/coverage.json')
                    if coverage['eligible_missing']:
                        raise ValueError('missing cells were found but preparation failed; inspect the CPU log')
                    print('NO_ELIGIBLE_MISSING_CELLS', json.dumps(coverage, indent=2))
                    return 0
                raise ValueError('CPU preparation is not ready; inspect its Slurm log before requesting a GPU')
            snapshot = scheduler(context)
            if not submit_next(root, context, snapshot):
                print('FINITE_SWEEP_SUBMITTED: inspect status, then finalize when every allocation ends.')
        else:
            snapshot = scheduler(context)
            jobs = confirmed_submissions(root, snapshot)
            states = {row['job_id']: row['state'] for row in snapshot['owners'] + snapshot['jobs']}
            print(json.dumps({'recovery_directory': str(root), 'gpu_jobs': {job: states[job] for job in jobs},
                              'all_user_live_jobs': snapshot['jobs']}, indent=2))
            for phase in ['preparation', 'final']:
                receipt = root / phase / 'submission.json'
                if receipt.exists():
                    print(phase + '_submission', json.dumps(read(receipt)), 'logs', receipt.parent)
            for name in ['preparation/coverage.json', 'preparation/ready.json', 'final/coverage.json']:
                path = root / name
                if path.exists():
                    value = read(path)
                    value.pop('frozen_files', None)
                    value.pop('blocked_fingerprints', None)
                    print(name, json.dumps(value, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
