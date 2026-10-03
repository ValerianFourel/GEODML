#!/usr/bin/env python3
"""Queue a held first wave for an existing pinned Gemma plan's preparation.

This operator adapter leaves preparation and inference code pinned unchanged.
It binds held jobs to verified shards before releasing them, then hands their
ordinary submission receipts to the existing ten-minute finite sender.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]


def git_pin(repository):
    if subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('use a clean committed checkout')
    return subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()


def fields(job):
    raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=30)
    return dict(part.split('=', 1) for part in raw.split() if '=' in part)


def checked_job(job, spec, *, comment=None):
    row = fields(job)
    if (row.get('JobId') != job or row.get('Account') != spec['account']
            or row.get('Partition') != 'accelerated'
            or not row.get('UserId', '').endswith(f'({os.getuid()})')):
        raise ValueError('scheduler owner/account/partition differs; no mutation')
    if comment is not None and (row.get('Comment') != comment
            or row.get('JobName') != 'geodml-gemma-v4-bout'
            or row.get('TimeLimit') != '05:00:00'
            or row.get('NumNodes') not in ('1', '1-1')
            or row.get('NumCPUs') != '32'
            or row.get('OverSubscribe') != 'NO'
            or row.get('TresPerNode') != 'gres/gpu:4'):
        raise ValueError('prequeued job resources or identity changed; no release')
    return row


def held(row):
    return row.get('JobState') == 'PENDING' and row.get('Reason') == 'JobHeldUser'


def command(args):
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)


def slots(root, state):
    return [{'id': f'slot-{i:04d}', 'directory': str(root / 'prequeue' / f'slot-{i:04d}')}
            for i in range(state['slots'])]


def current(slot, observed, sender):
    history = sender.attempts(slot)
    if not history:
        return None, None, None
    path, intent = history[-1]
    job = sender.identify(path, intent, observed, persist=True)
    receipt = sender.read(path / 'receipt.json') if (path / 'receipt.json').exists() else {}
    if not job and receipt.get('disposition') == 'capacity_refused':
        return None, None, None
    if not job:
        raise ValueError(f'{slot["id"]}: ambiguous submission; preserve it and restart after scheduler reconciliation')
    return path, intent, job


def enqueue(root, state, spec, observed, runtime, sender):
    """Every slot gets at most one accepted job; lost acknowledgements reconcile."""
    runtime.storage(spec, root / 'prequeue/admission')
    checked_at = time.time()
    for slot in slots(root, state):
        path, intent, job = current(slot, observed, sender)
        if job:
            continue
        if time.time() >= state['deadline_epoch']:
            raise ValueError('finite preparation deadline reached')
        if time.time() - checked_at >= 120:
            runtime.storage(spec, root / 'prequeue/admission')
            checked_at = time.time()
        folder = Path(slot['directory'])
        history = sender.attempts(slot)
        number = len(history) + 1
        attempt = folder / 'submissions' / f'attempt-{number:04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        token = hashlib.sha256((str(root) + state['helper_pin'] + slot['id'] + str(number)).encode()).hexdigest()[:32]
        comment = 'geodml-gemma-v4:' + token
        wrapper = folder / 'run.sh'
        wrapper.write_text('#!/bin/bash\nset -euo pipefail\nexec bash ' + shlex.quote(str(folder / 'assigned.sh')) + '\n')
        cmd = ['sbatch', '--parsable', '--hold', f'--dependency=afterok:{state["preparation_job"]}',
               '--kill-on-invalid-dep=yes', '--no-requeue', f'--account={spec["account"]}',
               '--partition=accelerated', '--nodes=1', '--ntasks=1', '--cpus-per-task=32',
               '--gres=gpu:4', '--mem=0', '--exclusive', '--time=05:00:00',
               '--job-name=geodml-gemma-v4-bout', f'--comment={comment}',
               f'--chdir={state["repository"]}', '--open-mode=append',
               f'--output={folder}/slurm-%j.out', f'--error={folder}/slurm-%j.err', str(wrapper)]
        intent = {'shard_id': slot['id'], 'comment': comment, 'attempt': number,
                  'created_at_epoch': time.time(), 'command': cmd}
        sender.save(attempt / 'intent.json', intent)
        # If interrupted during sbatch, the durable comment is the only authority
        # for recovery. Never blindly repeat a missing acknowledgement.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        receipt = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?\s*', result.stdout)
        if result.returncode == 0 and match:
            receipt['job_id'] = match[1]
        elif result.returncode and not result.stdout.strip() and re.search(
                r'MaxSubmitJob|maximum number of jobs', result.stderr, re.I):
            receipt['disposition'] = 'capacity_refused'
        sender.save(attempt / 'receipt.json', receipt)
        if receipt.get('disposition') == 'capacity_refused':
            sender.emit({'state': 'capacity_wait', 'reason': result.stderr, 'poll_seconds': 600})
            return False
        if not receipt.get('job_id'):
            raise ValueError('prequeue submission was not acknowledged; reconcile before retrying')
        sender.emit({'state': 'held_first_wave', 'slot': slot['id'], 'job_id': receipt['job_id']})
    return True


def cancel_unused(slot, path, intent, job, observed, spec, sender):
    receipt_path = Path(slot['directory']) / 'unused.json'
    row = observed['jobs'].get(job)
    if row is not None:
        live = checked_job(job, spec, comment=intent['comment'])
        if not held(live):
            raise ValueError('unused slot is not user-held; refusing cancellation')
        command(['scancel', job])
    elif observed['accounting'].get(job, {}).get('state') not in sender.TERMINAL:
        raise ValueError('unused slot has unresolved scheduler state')
    sender.save(receipt_path, {'job_id': job, 'reason': 'no frozen shard assigned', 'cancelled_or_terminal': True})


def activate(root, state, spec, observed, runtime, sender):
    plan = runtime.checked_plan(root)
    sender.validate(plan)
    if len(plan['shards']) > state['slots']:
        raise ValueError('prepared shard count exceeds first-wave slots')
    if time.time() >= plan['deadline_epoch']:
        raise ValueError('frozen plan deadline reached')
    sender_state = sender.state_file(root, plan)
    runtime.storage(plan, root / 'prequeue/release-admission')
    runtime.reserve(plan, root)
    bindings = []
    # Persist every adopted intent before releasing any job. The old sender sees
    # these as its first bouts and counts them against the original shard limits.
    for index, slot in enumerate(slots(root, state)):
        path, intent, job = current(slot, observed, sender)
        if index >= len(plan['shards']):
            if job:
                cancel_unused(slot, path, intent, job, observed, spec, sender)
            continue
        if not job:
            continue  # scheduler capacity refusal: normal sender fills this shard later
        shard = plan['shards'][index]
        directory = Path(shard['directory'])
        attempt = directory / 'submissions/attempt-0001'
        expected = {'shard_id': shard['id'], 'attempt': 1, 'comment': intent['comment'],
                    'git_commit': plan['git_commit'], 'config_sha256': shard['config_sha256'],
                    'plan_sha256': sender_state['plan_sha256'], 'completed_before': 0,
                    'command': intent['command'], 'prequeue_slot': slot['id']}
        if (attempt / 'intent.json').exists():
            previous = sender.read(attempt / 'intent.json')
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError('shard already belongs to another submission')
        else:
            if sender.attempts(shard) or (directory / 'results/control/index.sqlite').exists():
                raise ValueError('cannot adopt a slot over existing shard work')
            sender.save(attempt / 'intent.json', {**expected, 'created_at_epoch': time.time()})
        receipt_path = attempt / 'receipt.json'
        if receipt_path.exists() and sender.read(receipt_path).get('job_id') != job:
            raise ValueError('adopted job receipt changed')
        sender.save(receipt_path, {'job_id': job, 'prequeued': True, 'returncode': 0})
        assigned = Path(slot['directory']) / 'assigned.sh'
        # Atomic replacement prevents a partial script if an operator releases a
        # held job outside this adapter. Runtime still verifies plan/claim/binding.
        runtime.atomic(assigned, ('#!/bin/bash\nset -euo pipefail\nexec bash ' + shlex.quote(str(directory / 'run.sh')) + '\n').encode())
        bindings.append((job, intent, slot))
    # Binding/cancelling hundreds of slots can itself outlive quota freshness.
    runtime.storage(plan, root / 'prequeue/release-admission')
    checked_at = time.time()
    for job, intent, slot in bindings:
        if job not in observed['jobs']:
            if observed['accounting'].get(job, {}).get('state') in sender.TERMINAL:
                continue  # previously released work is reconciled by normal sender
            raise ValueError('assigned job scheduler state unresolved')
        if time.time() - checked_at >= 120:
            runtime.storage(plan, root / 'prequeue/release-admission')
            checked_at = time.time()
        row = checked_job(job, spec, comment=intent['comment'])
        release_path = Path(slot['directory']) / 'release-intent.json'
        if held(row):
            if time.time() >= plan['deadline_epoch']:
                raise ValueError('frozen plan deadline reached before release')
            sender.save(release_path, {'job_id': job, 'plan_sha256': sender_state['plan_sha256']})
            command(['scontrol', 'release', job])
            sender.emit({'state': 'released_first_wave', 'job_id': job, 'slot': slot['id']})
        elif not release_path.exists():
            raise ValueError('job left hold without an adapter release intent')
    sender.save(root / 'prequeue/activated.json', {'plan_sha256': sender_state['plan_sha256'],
                'adopted_jobs': [job for job, _, _ in bindings], 'time': time.time()})


def run(root, repository, helper_pin, runtime, sender):
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('run this controller on a login host')
    spec = sender.read(root / 'preparation.json')
    if (str(root) != spec['root'] or runtime.clean_pin() != spec['git_commit']
            or repository != runtime.REPO or not root.is_relative_to(Path(spec['workspace']).resolve())):
        raise ValueError('existing scientific checkout or preparation root differs')
    submitted = sender.read(root / 'preparation-submission.json')
    match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?\s*', submitted['stdout'])
    if submitted['returncode'] != 0 or not match:
        raise ValueError('existing preparation has no successful submission receipt')
    prep_job = match[1]
    with (root / 'start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sender.sender_lock(spec['workspace']):
            path = root / 'prequeue/state.json'
            expected = {'helper_pin': helper_pin, 'repository': str(repository),
                        'preparation_job': prep_job, 'slots': 200,
                        'deadline_epoch': spec['preparation_deadline_epoch']}
            if path.exists():
                state = sender.read(path)
                if state != expected:
                    raise ValueError('prequeue controller configuration changed')
            else:
                if (root / 'sender/state.json').exists():
                    raise ValueError('existing inference sender state; do not insert a prequeue wave')
                row = checked_job(prep_job, spec)
                if row.get('JobName') != 'geodml-gemma-v4-prepare' or row.get('JobState') not in ('PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED'):
                    raise ValueError('preparation is not an active/successful Gemma preparation job')
                state = expected
                sender.save(path, state)
            since = datetime.fromtimestamp(spec['created_at_epoch'] - 86400).strftime('%Y-%m-%d')
            while not (root / 'prequeue/activated.json').exists():
                observed = sender.snapshot(since)
                live = observed['jobs'].get(prep_job)
                terminal = observed['accounting'].get(prep_job, {}).get('state') if not live else None
                if terminal in sender.TERMINAL and terminal != 'COMPLETED':
                    for slot in slots(root, state):
                        p, intent, job = current(slot, observed, sender)
                        if job:
                            cancel_unused(slot, p, intent, job, observed, spec, sender)
                    raise ValueError('preparation failed; unused held wave retired')
                if time.time() >= state['deadline_epoch']:
                    raise ValueError('finite preparation deadline reached; inspect held jobs')
                if terminal == 'COMPLETED':
                    activate(root, state, spec, observed, runtime, sender)
                    break
                if not live:
                    raise ValueError('preparation scheduler state unresolved; preserve all jobs')
                full = enqueue(root, state, spec, observed, runtime, sender)
                status = {'state': 'waiting_for_preparation', 'preparation_job': prep_job,
                          'poll_seconds': 600, 'time': time.time(), 'slots_requested': 200,
                          'first_wave_fully_submitted': full}
                sender.save(root / 'prequeue/status.json', status)
                sender.emit(status)
                time.sleep(min(600, max(0, state['deadline_epoch'] - time.time())))
    # These receipts are native to the unchanged sender: no second submission
    # for adopted first bouts, and subsequent bouts retain the frozen budget.
    return sender.send(root)


def login_takeover(root, repository, helper_pin, runtime, sender):
    """Retire only pending GPU preparation, then prepare on the authorized login host."""
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('login preparation requires a login shell')
    spec = sender.read(root / 'preparation.json')
    state = sender.read(root / 'prequeue/state.json')
    if (runtime.clean_pin() != spec['git_commit'] or repository != runtime.REPO
            or state['repository'] != str(repository) or spec['root'] != str(root)
            or not root.is_relative_to(Path(spec['workspace']).resolve())):
        raise ValueError('original preparation/inference pin or root changed')
    prep_job = state['preparation_job']
    since = datetime.fromtimestamp(spec['created_at_epoch'] - 86400).strftime('%Y-%m-%d')
    takeover_path = root / 'prequeue/login-takeover.json'
    with (root / 'start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sender.sender_lock(spec['workspace']):
            if takeover_path.exists():
                takeover = sender.read(takeover_path)
                if takeover['helper_pin'] != helper_pin or takeover['preparation_job'] != prep_job:
                    raise ValueError('login takeover pin or preparation job changed')
            else:
                if (root / 'plan.json').exists() or (root / 'sender/state.json').exists():
                    raise ValueError('preparation/inference already progressed; preserve it')
                for name in ('frozen', 'frozen.partial', 'shards', 'shards.partial'):
                    if (root / name).exists():
                        raise ValueError('partial preparation exists; preserve and inspect it')
                row = checked_job(prep_job, spec)
                if row.get('JobState') != 'PENDING' or row.get('JobName') != 'geodml-gemma-v4-prepare':
                    raise ValueError('preparation is no longer pending; preserve its allocation')
                takeover = {'helper_pin': helper_pin, 'preparation_job': prep_job,
                            'authorization': 'Valerian explicitly requested preparation on the login shell now',
                            'started_at_epoch': time.time(), 'original_state': state}
                sender.save(takeover_path, takeover)
            if not (root / 'plan.json').exists():
                observed = sender.snapshot(since)
                if prep_job in observed['jobs']:
                    row = checked_job(prep_job, spec)
                    if row.get('JobState') != 'PENDING':
                        raise ValueError('preparation started; preserve its allocation')
                    command(['scontrol', 'hold', prep_job])
                    if not held(checked_job(prep_job, spec)):
                        raise ValueError('preparation could not be held; no login work started')
                elif observed['accounting'].get(prep_job, {}).get('state') != 'CANCELLED':
                    raise ValueError('preparation state unresolved or already executed')
                # Remove afterok BEFORE cancelling preparation, otherwise Slurm's
                # kill-on-invalid-dep would cancel the 200 accepted inference jobs.
                known = []
                for slot in slots(root, state):
                    path, intent, job = current(slot, observed, sender)
                    if not job:
                        continue
                    row = checked_job(job, spec, comment=intent['comment'])
                    if not held(row):
                        raise ValueError('first-wave job is not held; preserve it')
                    known.append((slot, intent, job))
                for slot, intent, job in known:
                    receipt = Path(slot['directory']) / 'login-dependency.json'
                    sender.save(receipt, {'job_id': job, 'preparation_job': prep_job, 'state': 'clearing'})
                    command(['scontrol', 'update', f'JobId={job}', 'Dependency='])
                    row = checked_job(job, spec, comment=intent['comment'])
                    if not held(row) or row.get('Dependency') not in ('(null)', '', 'None'):
                        raise ValueError('dependency removal not verified; preparation preserved')
                    sender.save(receipt, {'job_id': job, 'preparation_job': prep_job, 'state': 'cleared'})
                if prep_job in observed['jobs']:
                    if not held(checked_job(prep_job, spec)):
                        raise ValueError('preparation no longer held; preserve it')
                    command(['scancel', prep_job])
                # Require terminal confirmation before any input scan. Queries
                # remain at least 30 seconds apart if accounting has not settled.
                for attempt in range(10):
                    ended = sender.snapshot(since)
                    if prep_job not in ended['jobs'] and ended['accounting'].get(prep_job, {}).get('state') == 'CANCELLED':
                        break
                    if attempt == 9:
                        raise ValueError('awaiting terminal preparation accounting; rerun same pinned takeover')
                    time.sleep(30)
                sender.save(root / 'prequeue/login-preparation-retired.json', ended['accounting'][prep_job])
                for name in ('frozen', 'frozen.partial', 'shards', 'shards.partial'):
                    if (root / name).exists():
                        raise ValueError('partial preparation exists; no automatic overwrite')
                runtime.storage(spec, root / 'prequeue/login-admission')
                sender.emit({'state': 'preparing_on_login', 'gpu_used': False, 'maximum_seconds': 3600})
                sender.save(root / 'prequeue/status.json', {'state': 'preparing_on_login', 'time': time.time()})
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
                subprocess.run(['nice', '-n', '10', sys.executable, '-u',
                    str(REPO / 'analysis/scripts/horeka_gemma_v4.py'), 'prepare-login',
                    '--output', str(root), '--inference-repository', str(repository)],
                    check=True, timeout=3600, env=env)
            plan = runtime.checked_plan(root)
            execution = plan.get('preparation_execution', {})
            if execution.get('execution') != 'login' or execution.get('preparation_code_revision') != helper_pin:
                raise ValueError('plan lacks the pinned login preparation record')
            if not (root / 'prequeue/activated.json').exists():
                activate(root, state, spec, sender.snapshot(since), runtime, sender)
    return sender.send(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-on-login', action='store_true')
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args(argv)
    pin = git_pin(REPO)
    repository = args.repository.resolve(strict=True)
    sys.path.insert(0, str(repository))
    from analysis.scripts import horeka_gemma_v4 as runtime
    from analysis.scripts import horeka_gemma_v4_sender as sender
    root = args.output.resolve(strict=True)
    try:
        controller = login_takeover if args.prepare_on_login else run
        return controller(root, repository, pin, runtime, sender)
    except Exception as error:
        sender.save(root / 'prequeue/status.json', {'state': 'blocked', 'reason': str(error),
                    'time': time.time(), 'jobs_preserved': True})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
