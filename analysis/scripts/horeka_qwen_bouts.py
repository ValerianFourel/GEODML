#!/usr/bin/env python3
"""Pre-divided HoreKa Qwen bouts: divide once, submit explicitly approved ranges, never retry.

`divide` splits every remaining qwen38 cell (frozen keyword priority, front first)
into fixed five-hour bouts sized from a measured A100 rate. Each bout owns a
primary list and lists the next bout's first cells as spill-over, so a fast
allocation never idles. All bouts write to one dataset root whose striped task
ledger admits each cell once; unfinished cells stay missing for a later sweep.
`submit` sends an approved range as independent jobs. `execute` runs one bout.
Outputs are diagnostic until verified and published; this never touches the
Hugging Face hour registry.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.agentic_hours import canonical, digest

DIVISION_VERSION = 'geodml-horeka-qwen-bouts-v1'
MODEL = 'qwen38'
START_MARGIN, CLEANUP_MARGIN = 300, 120
BATCH_ONLY = ('--parsable', '--no-requeue', '--chdir=', '--output=', '--error=')


def read(path):
    return json.loads(Path(path).read_bytes())


def walltime_seconds(value):
    hours, minutes, seconds = (int(part) for part in value.split(':'))
    return hours * 3600 + minutes * 60 + seconds


def remaining_rows(plan, tasks, completed, blocked, published):
    """Missing qwen38 cells in the frozen batch order (front of the priority list first)."""
    excluded = set(plan.get('completed_before_plan', [])) | set(completed) | set(blocked) | set(published)
    rows = [r for r in tasks if r['model'] == MODEL and r['fingerprint'] not in excluded
            and plan.get('deferred', {}).get(r['fingerprint']) == 'awaiting_calibration']
    return sorted(rows, key=lambda r: (r['priority_rank'], r['keyword_id'], r['prompt_id'], r['task_id']))


def capacity(walltime, seconds_per_cell, startup_seconds):
    usable = walltime_seconds(walltime) - startup_seconds - START_MARGIN - CLEANUP_MARGIN
    cells = math.floor(usable / seconds_per_cell)
    if cells <= 0:
        raise ValueError('bout has no useful work time')
    return cells


def split(rows, primary_cells, overbook):
    """Whole prompt groups, one scientific configuration per bout; spill = next bout's head."""
    if not rows:
        raise ValueError('no remaining Qwen cells')
    if overbook < 1:
        raise ValueError('overbook must be at least 1')
    groups = []
    for row in rows:
        key = (row['configuration_sha256'], row['prompt_id'])
        if groups and groups[-1][0] == key:
            groups[-1][1].append(row)
        else:
            groups.append((key, [row]))
    bouts, current, config = [], [], None
    for (configuration, _), members in groups:
        if current and (configuration != config or len(current) + len(members) > primary_cells):
            bouts.append(current)
            current = []
        config = configuration
        current.extend(members)
    bouts.append(current)
    extra = math.ceil(primary_cells * (overbook - 1))
    result = []
    for index, primary in enumerate(bouts):
        following = bouts[index + 1] if index + 1 < len(bouts) else []
        same = following and following[0]['configuration_sha256'] == primary[0]['configuration_sha256']
        spill = following[:extra] if same else []
        result.append({'number': index + 1, 'primary': primary, 'spill': spill})
    return result


def divide(args):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
    from analysis.interpretability.pipeline.agentic_hours import inventory
    from analysis.scripts.publish_qwen_results import published_fingerprints, read_index
    out = args.output.resolve()
    if out.exists():
        raise ValueError('division already exists; keep it and submit from it, or choose a new output')
    plan = read(args.plan)
    tasks, completed, blocked = inventory(args.dataset.resolve(), stripes=args.stripes, reuse_verified=True)
    exchange = Exchange(HubStore(args.repo_id), out.parent / (out.name + '-journal'))
    revision = exchange.store.head()
    published = published_fingerprints(exchange, read_index(exchange.store, revision))
    rows = remaining_rows(plan, tasks, completed, blocked, published)
    primary_cells = capacity(args.walltime, args.seconds_per_cell, args.startup_seconds)
    bouts = split(rows, primary_cells, args.overbook)
    out.mkdir(parents=True)
    summary = []
    for bout in bouts:
        directory = out / 'bouts' / f"bout-{bout['number']:04d}"
        directory.mkdir(parents=True)
        cells = [r['runnable_task'] for r in bout['primary'] + bout['spill']]
        atomic(directory / 'cells.jsonl', b''.join(canonical(c) + b'\n' for c in cells))
        record = {'number': bout['number'], 'primary_cells': len(bout['primary']),
                  'spill_cells': len(bout['spill']), 'cell_count': len(cells),
                  'configuration_sha256': bout['primary'][0]['configuration_sha256'],
                  'first_keyword': bout['primary'][0]['keyword_id'], 'last_keyword': bout['primary'][-1]['keyword_id'],
                  'primary_fingerprints': [r['fingerprint'] for r in bout['primary']],
                  'spill_fingerprints': [r['fingerprint'] for r in bout['spill']]}
        atomic(directory / 'bout.json', canonical(record))
        summary.append({k: v for k, v in record.items() if not k.endswith('fingerprints')})
    division = {
        'format_version': DIVISION_VERSION, 'model': MODEL, 'dataset_root': str(args.dataset.resolve()),
        'plan': str(args.plan.resolve()), 'plan_sha256': digest(plan), 'hub_revision': revision,
        'counts': {'registered_qwen': sum(r['model'] == MODEL for r in tasks), 'remaining': len(rows),
                   'published_excluded': len(published), 'bouts': len(bouts)},
        'sizing': {'walltime': args.walltime, 'seconds_per_cell': args.seconds_per_cell,
                   'startup_seconds': args.startup_seconds, 'start_margin_seconds': START_MARGIN,
                   'cleanup_margin_seconds': CLEANUP_MARGIN, 'primary_cells': primary_cells,
                   'overbook': args.overbook, 'measurement': args.measurement},
        'totals': {'node_hours': len(bouts) * walltime_seconds(args.walltime) / 3600,
                   'gpu_hours': len(bouts) * walltime_seconds(args.walltime) * 4 / 3600},
        'bouts': summary}
    atomic(out / 'division.json', canonical(division))
    print(json.dumps({k: division[k] for k in ('counts', 'sizing', 'totals')}, indent=2))


def bout_config(division, bout_dir, *, repo, pin, workspace, runtime, checked, settings, cross, cross_revision,
                approval, walltime):
    number = read(bout_dir / 'bout.json')['number']
    return {'repository': str(repo), 'git_commit': pin, 'job_name': f'geodml-qwen-bout-{number:04d}',
            'bout': number, 'walltime': walltime, 'dataset_root': division['dataset_root'],
            'cell_count': read(bout_dir / 'bout.json')['cell_count'],
            'reference_profile': checked['files']['SEARCH_AGENTIC_PROFILE'], 'inputs': checked['files'],
            'settings': settings, 'cross_encoder': str(cross), 'cross_revision': cross_revision,
            'cache': str(workspace / 'serving-cache' / f'bout-{number:04d}'),
            'approval': approval,
            'environment': {'HF_HUB_CACHE': str(workspace / 'models'),
                            'PATH': str(runtime.parent) + ':' + os.environ['PATH'],
                            'PYTHONPATH': str(repo), 'PYTHONDONTWRITEBYTECODE': '1'}}


def submit(args):
    from analysis.interpretability.pipeline.agentic_hour_runtime import validate_reservation
    from analysis.scripts.manage_agentic_hours import health
    from analysis.scripts.prepare_horeka_qwen import MODELS, capture_quota, snapshot
    from analysis.scripts.prepare_shared_hour_inputs import verify_inputs
    repo = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('clean committed checkout required')
    division = read(args.division / 'division.json')
    if division.get('format_version') != DIVISION_VERSION:
        raise ValueError('unknown division format')
    if args.approved_walltime != division['sizing']['walltime']:
        raise ValueError('approved wall-time differs from the division sizing')
    numbers = list(range(args.first, args.first + args.count))
    if args.count < 1 or args.count > args.approved_count:
        raise ValueError('bout count exceeds the approved number of allocations')
    dirs = [args.division / 'bouts' / f'bout-{n:04d}' for n in numbers]
    missing = [str(d) for d in dirs if not (d / 'bout.json').is_file()]
    if missing:
        raise ValueError(f'bouts not in division: {missing[:3]}')
    done = [d.name for d in dirs if (d / 'SUBMISSION_ATTEMPTED').exists()]
    if done:
        raise ValueError(f'already submitted; inspect their receipts instead of resubmitting: {done[:5]}')
    root = Path(division['dataset_root'])
    if not (root / 'contract.json').is_file():
        raise ValueError('dataset root lacks contract.json; direct dataset mode cannot write there')
    manifests = list((root / 'artifacts/shared-preparations').glob('*.json'))
    if len(manifests) != 1:
        raise ValueError('expected one frozen manifest')
    checked = verify_inputs(root, manifests[0], model=MODEL)
    settings = read(manifests[0])['models'][MODEL]['reference_runtime']
    runtime = args.workspace / 'environment/qwen-runtime/bin/python'
    if not runtime.is_file():
        raise ValueError('prepare inference runtime first')
    cross = snapshot(args.workspace / 'models', {'repo_id': MODELS[1][0], 'revision': MODELS[1][1]})
    if not cross.is_dir():
        raise ValueError('pinned BGE snapshot absent')
    seconds = walltime_seconds(args.approved_walltime)
    approval = {'evidence': args.approval, 'walltime_seconds': seconds, 'maximum_gpu_hours': seconds * 4 / 3600,
                'approved_allocations': args.approved_count, 'bouts': numbers,
                'resources': {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': 'all'},
                'estimate': (f"measured {division['sizing']['seconds_per_cell']} s per cell on 4 x A100; "
                             f"{division['sizing']['primary_cells']} primary cells per bout plus spill-over")}
    receipts = args.division / 'submissions'
    receipts.mkdir(exist_ok=True)
    quota_path = receipts / f'quota-{int(time.time())}.json'
    atomic(quota_path, canonical(capture_quota(args.workspace, args.account)))
    storage = health({'dataset_root': str(root), 'cluster': 'horeka'}, quota_path)
    if storage.get('safe_to_admit') is not True or storage.get('quota_verified') is not True:
        raise ValueError('storage admission blocked; fresh quota evidence required')
    validate_reservation({'approval': approval}, {'cluster': 'horeka', 'partition': args.partition,
                                                  'account': args.account, 'reservation': args.reservation})
    planned = []
    for number, directory in zip(numbers, dirs):
        config = bout_config(division, directory, repo=repo, pin=pin, workspace=args.workspace, runtime=runtime,
                             checked=checked, settings=settings, cross=cross, cross_revision=MODELS[1][1],
                             approval=approval, walltime=args.approved_walltime)
        atomic(directory / 'config.json', canonical(config))
        atomic(directory / 'run.sh', b'#!/bin/bash\nset -euo pipefail\nexec "$1" "$2" execute --config "$3"\n')
        command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--exclusive',
                   '--cpus-per-task=32', '--mem=0', '--time=' + args.approved_walltime,
                   '--account=' + args.account, '--partition=' + args.partition,
                   '--job-name=' + config['job_name'], '--chdir=' + str(repo),
                   '--output=' + str(directory / 'slurm-%j.out'), '--error=' + str(directory / 'slurm-%j.err')]
        if args.reservation:
            command.append('--reservation=' + args.reservation)
        command += [str(directory / 'run.sh'), str(runtime), str(Path(__file__).resolve()), str(directory / 'config.json')]
        atomic(directory / 'submission-command.json', canonical(command))
        planned.append((number, directory, command))
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'bouts': numbers, 'first_command': planned[0][2]}, indent=2))
        return
    results = []
    for number, directory, command in planned:
        # A durable marker survives an ambiguous sbatch response. Never automatically retry.
        with (directory / 'SUBMISSION_ATTEMPTED').open('x') as stream:
            stream.write(str(time.time()))
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        receipt = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        atomic(directory / 'submission.json', canonical(receipt))
        results.append({'bout': number, 'job_id': result.stdout.strip(), 'returncode': result.returncode})
        if result.returncode:
            atomic(receipts / f'submitted-{numbers[0]:04d}-{numbers[-1]:04d}.json', canonical(results))
            raise RuntimeError(f'bout {number}: sbatch failed; later bouts were not submitted: {result.stderr}')
    atomic(receipts / f'submitted-{numbers[0]:04d}-{numbers[-1]:04d}.json', canonical(results))
    print(json.dumps(results, indent=2))


def compiler_environment():
    """FlashInfer JIT on HoreKa: nvcc 12.4 rejects the default Intel icx; use the system GCC."""
    cc, cxx = shutil.which('gcc'), shutil.which('g++')
    if not cc or not cxx:
        raise ValueError('system gcc/g++ required for FlashInfer JIT compilation')
    return {'CC': cc, 'CXX': cxx, 'CUDAHOSTCXX': cxx, 'NVCC_CCBIN': cxx}


def controller_command(config, *, profile, output, writer_id):
    repo, inputs, settings = Path(config['repository']), config['inputs'], config['settings']
    return [sys.executable, str(repo / 'analysis/scripts/run_agentic_search_integration_smoke.py'),
            '--output', str(output), '--base-url', profile['serving']['public_base_url'],
            '--model-id', profile['model']['model_id'], '--model-revision', profile['model']['model_revision'],
            '--cross-encoder-snapshot', config['cross_encoder'], '--cross-encoder-revision', config['cross_revision'],
            '--search-snapshot', 'duckduckgo=' + inputs['SEARCH_AGENTIC_DDG_SNAPSHOT'],
            '--search-snapshot', 'searxng=' + inputs['SEARCH_AGENTIC_SEARXNG_SNAPSHOT'],
            '--prompts-jsonl', inputs['SEARCH_AGENTIC_PROMPTS_JSONL'],
            '--selection-records-jsonl', inputs['SEARCH_AGENTIC_SELECTION_RECORDS_JSONL'],
            '--cell-ids-jsonl', str(Path(config['bout_dir']) / 'cells.jsonl'),
            '--prompt-count', settings['SEARCH_AGENTIC_PROMPT_COUNT'],
            '--prompt-selection-seed', settings['SEARCH_AGENTIC_PROMPT_SELECTION_SEED'],
            '--request-concurrency', settings['SEARCH_AGENTIC_REQUEST_CONCURRENCY'],
            '--cell-concurrency', settings['SEARCH_AGENTIC_CELL_CONCURRENCY'],
            '--seed', '20260911', '--query-max-tokens', '256', '--final-max-tokens', '2048',
            '--disable-thinking', '--production-conditions',
            '--dataset-root', config['dataset_root'], '--dataset-writer-id', writer_id,
            '--dataset-ledger-stripes', '256', '--worker-index', '0', '--worker-count', '1']


def execute(config_path):
    # Compute-node proof precedes configuration, GPU imports and model loading.
    from analysis.scripts.verify_inference_allocation import verify
    boundary = verify('horeka')
    config = read(config_path)
    bout_dir = Path(config_path).parent
    config['bout_dir'] = str(bout_dir)
    repo = Path(config['repository'])
    job = os.environ['SLURM_JOB_ID']
    attempt = bout_dir / 'attempts' / f'job{job}'
    attempt.mkdir(parents=True, exist_ok=False)
    atomic(attempt / 'boundary.json', canonical(boundary))
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    fields = dict(t.split('=', 1) for t in info.split() if '=' in t)
    if fields.get('JobName') != config['job_name'] or fields.get('TimeLimit') != config['walltime']:
        raise ValueError('allocation differs from the approved bout')
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != config['git_commit']:
        raise ValueError('execution commit mismatch')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('execution checkout is dirty')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SEARCH_AGENTIC_', 'GEODML_'))}
    env.update(config['environment'])
    env.update(compiler_environment())
    for field, variable in (('StartTime', 'SLURM_JOB_START_TIME'), ('EndTime', 'SLURM_JOB_END_TIME')):
        env[variable] = str(int(datetime.fromisoformat(fields[field]).timestamp()))
    env.update(GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY='1', GEODML_APPROVED_WALLTIME=config['walltime'],
               GEODML_START_MARGIN_SECONDS=str(START_MARGIN), GEODML_CLEANUP_MARGIN_SECONDS=str(CLEANUP_MARGIN),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('GEODML_PRIVATE_NETWORK_NAMESPACE', None)
    os.environ.update(env)
    for key in list(os.environ):
        if key.startswith(('SEARCH_AGENTIC_', 'GEODML_')) and key not in env:
            del os.environ[key]
    from analysis.scripts.horeka_qwen_probe import local_profile
    from analysis.scripts.search_vllm_stage import (
        create_or_verify_profile,
        discover_visible_gpus,
        inspect_vllm,
        load_profile,
    )
    atomic(attempt / 'nvidia-smi.txt', subprocess.check_output(['nvidia-smi'], text=True).encode())
    frozen = load_profile(config['reference_profile'])
    executable, version, help_text = inspect_vllm(Path(sys.executable).parent / 'vllm')
    profile = local_profile(frozen, executable, version, help_text,
                            discover_visible_gpus(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    profile_path = attempt / 'horeka-profile.json'
    create_or_verify_profile(profile_path, profile)
    writer_id = f"horeka-bout{config['bout']:04d}-job{job}"
    command = [sys.executable, str(repo / 'analysis/scripts/search_vllm_stage.py'), 'run',
               '--profile', str(profile_path), '--server-log', str(attempt / 'server.log'),
               '--cache-base', config['cache'], '--startup-timeout-seconds', '900', '--',
               *controller_command(config, profile=profile, output=attempt / 'results', writer_id=writer_id)]
    started = time.time()
    atomic(attempt / 'execution.json', canonical({'job_id': job, 'bout': config['bout'], 'started_at': started,
           'command': command, 'git_commit': config['git_commit'], 'writer_id': writer_id,
           'compiler': compiler_environment(), 'approval': config['approval'], 'scientific_result': False}))
    telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                  '--format=csv,noheader', '-l', '30'],
                                 stdout=(attempt / 'gpu.csv').open('w'), stderr=subprocess.DEVNULL)
    timeout = int(env['SLURM_JOB_END_TIME']) - time.time() - CLEANUP_MARGIN
    if timeout <= 0:
        raise ValueError('allocation has no remaining work time')
    process = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        summary = {'status': 'failed', 'returncode': returncode}
        manifest_path = attempt / 'results/run_manifest.json'
        if manifest_path.is_file():
            manifest = read(manifest_path)
            summary.update({k: manifest.get(k) for k in ('status', 'completed_count', 'remaining_count', 'stop_reason')})
            summary['direct_dataset'] = {k: manifest.get('direct_dataset', {}).get(k)
                                         for k in ('committed_count', 'reused_count')}
            if returncode == 0:
                from analysis.scripts.run_agentic_search_integration_smoke import validate_manifest_artifacts
                try:
                    validate_manifest_artifacts(manifest_path, config['cell_count'])
                except ValueError as error:
                    summary['validation_error'] = str(error)
            else:
                summary['status'] = 'failed'
        atomic(attempt / 'bout-result.json', canonical({**summary, 'job_id': job, 'bout': config['bout'],
               'elapsed_seconds': time.time() - started, 'scientific_result': False}))
        return returncode
    except subprocess.TimeoutExpired:
        atomic(attempt / 'bout-result.json', canonical({'status': 'deadline', 'job_id': job, 'bout': config['bout'],
               'elapsed_seconds': time.time() - started, 'scientific_result': False}))
        return 124
    finally:
        for proc, group in ((process, True), (telemetry, False)):
            try:
                os.killpg(proc.pid, signal.SIGTERM) if group else proc.terminate()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL) if group else proc.kill()
                proc.wait()


def status(args):
    division = read(args.division / 'division.json')
    rows = []
    for bout in division['bouts']:
        directory = args.division / 'bouts' / f"bout-{bout['number']:04d}"
        if not (directory / 'SUBMISSION_ATTEMPTED').exists():
            continue
        receipt = read(directory / 'submission.json') if (directory / 'submission.json').is_file() else {}
        results = [read(p) for p in sorted((directory / 'attempts').glob('job*/bout-result.json'))]
        live = [read(p) for p in sorted((directory / 'attempts').glob('job*/results/run_manifest.json'))]
        rows.append({'bout': bout['number'], 'job_id': receipt.get('stdout', '').strip(),
                     'completed': sum(m.get('completed_count', 0) for m in live),
                     'results': [{k: r.get(k) for k in ('status', 'stop_reason', 'completed_count', 'returncode')}
                                 for r in results]})
    print(json.dumps({'submitted_bouts': len(rows), 'completed_cells': sum(r['completed'] for r in rows),
                      'bouts': rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('divide')
    for name in ('dataset', 'plan', 'output'):
        d.add_argument('--' + name, type=Path, required=True)
    d.add_argument('--repo-id', default='ValerianFourel/geodml-experiment-v2-paper-private')
    d.add_argument('--walltime', default='05:00:00')
    d.add_argument('--seconds-per-cell', type=float, required=True)
    d.add_argument('--startup-seconds', type=int, default=360)
    d.add_argument('--overbook', type=float, default=1.25)
    d.add_argument('--measurement', required=True, help='Where the per-cell rate comes from')
    d.add_argument('--stripes', type=int, default=256)
    s = sub.add_parser('submit')
    for name in ('division', 'workspace'):
        s.add_argument('--' + name, type=Path, required=True)
    s.add_argument('--first', type=int, required=True)
    s.add_argument('--count', type=int, required=True)
    s.add_argument('--account', required=True)
    s.add_argument('--partition', default='accelerated')
    s.add_argument('--reservation')
    s.add_argument('--approved-walltime', required=True)
    s.add_argument('--approved-count', type=int, required=True)
    s.add_argument('--approval', required=True)
    s.add_argument('--dry-run', action='store_true')
    e = sub.add_parser('execute')
    e.add_argument('--config', type=Path, required=True)
    t = sub.add_parser('status')
    t.add_argument('--division', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'execute':
        return execute(args.config)
    return {'divide': divide, 'submit': submit, 'status': status}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
