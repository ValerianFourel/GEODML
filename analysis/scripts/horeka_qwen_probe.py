#!/usr/bin/env python3
"""One explicitly approved HoreKa compatibility allocation; never production claims."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.agentic_hours import canonical


def read(path):
    return json.loads(Path(path).read_bytes())


def choose_probe(plan, root, *, stripes=256):
    from analysis.interpretability.pipeline.agentic_hours import inventory
    # Uncalibrated plans have no packaged task rows; deferred IDs join the registry.
    tasks, completed, blocked = inventory(root, stripes=stripes)
    excluded = set(plan['completed_before_plan']) | completed | blocked
    rows = sorted((r for r in tasks if r['fingerprint'] not in excluded and r['model'] == 'qwen38'
                   and plan.get('deferred', {}).get(r['fingerprint']) == 'awaiting_calibration'),
                  key=lambda r: (r['priority_rank'], r['keyword_id'], r['prompt_id'], r['task_id']))
    if not rows:
        raise ValueError('no missing Qwen cells for compatibility test')
    first = rows[0]
    return [r['runnable_task'] for r in rows if r['prompt_id'] == first['prompt_id']
            and r['configuration_sha256'] == first['configuration_sha256']]


def local_profile(frozen, executable, version, help_text, gpus, visible):
    from analysis.scripts.search_vllm_stage import build_profile
    if version != frozen['runtime']['vllm_version']:
        raise ValueError('installed vLLM differs from reference')
    serving, features = frozen['serving'], frozen['features']
    return build_profile(stage=frozen['stage'], **frozen['model'],
        vllm_executable=executable, vllm_version=version, vllm_help=help_text,
        visible_gpus=gpus, cuda_visible_devices=visible, expected_gpu_name_pattern='A100',
        **{k: serving[k] for k in ('host', 'port', 'data_parallel_size', 'tensor_parallel_size',
                                  'dtype', 'max_model_len', 'gpu_memory_utilization', 'request_concurrency')},
        **{k: features.get(k) for k in ('language_model_only', 'tokenizer_mode', 'attention_backend',
                                       'config_format', 'load_format', 'structured_outputs_config', 'rope_scaling')},
        enforce_eager='--enforce-eager' in frozen['server_argv'],
        disable_custom_all_reduce='--disable-custom-all-reduce' in frozen['server_argv'])


def execute(config_path):
    # Compute-node proof precedes profile preparation, GPU imports and model loading.
    from analysis.scripts.verify_inference_allocation import verify
    boundary = verify('horeka')
    config = read(config_path)
    out, repo = Path(config_path).parent, Path(config['repository'])
    atomic(out / 'boundary.json', canonical(boundary))
    job = os.environ['SLURM_JOB_ID']
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    fields = dict(t.split('=', 1) for t in info.split() if '=' in t)
    if fields.get('JobName') != config['job_name'] or fields.get('TimeLimit') != '01:00:00':
        raise ValueError('allocation differs from approved compatibility run')
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != config['git_commit']:
        raise ValueError('execution commit mismatch')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('execution checkout is dirty')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SEARCH_AGENTIC_', 'GEODML_'))}
    env.update(config['environment'])
    for field, variable in (('StartTime', 'SLURM_JOB_START_TIME'), ('EndTime', 'SLURM_JOB_END_TIME')):
        env[variable] = str(int(datetime.fromisoformat(fields[field]).timestamp()))
    env.update(GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY='1', GEODML_APPROVED_WALLTIME='01:00:00',
               GEODML_START_MARGIN_SECONDS='300', GEODML_CLEANUP_MARGIN_SECONDS='120',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('GEODML_PRIVATE_NETWORK_NAMESPACE', None)
    os.environ.update(env)
    # Remove inherited scientific routing even from this profile-preparation process.
    for key in list(os.environ):
        if key.startswith(('SEARCH_AGENTIC_', 'GEODML_')) and key not in env:
            del os.environ[key]
    from analysis.scripts.search_vllm_stage import (
        create_or_verify_profile,
        discover_visible_gpus,
        inspect_vllm,
        load_profile,
    )
    gpu_raw = subprocess.check_output(['nvidia-smi'], text=True)
    atomic(out / 'nvidia-smi.txt', gpu_raw.encode())
    frozen = load_profile(config['reference_profile'])
    executable, version, help_text = inspect_vllm(Path(sys.executable).parent / 'vllm')
    profile = local_profile(frozen, executable, version, help_text,
                            discover_visible_gpus(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    profile_path = out / 'horeka-profile.json'
    create_or_verify_profile(profile_path, profile)
    controller = [sys.executable, str(repo / 'analysis/scripts/run_agentic_search_integration_smoke.py'),
        '--output', str(out / 'results'), '--base-url', profile['serving']['public_base_url'],
        '--model-id', frozen['model']['model_id'], '--model-revision', frozen['model']['model_revision'],
        '--cross-encoder-snapshot', config['cross_encoder'], '--cross-encoder-revision', config['cross_revision'],
        '--search-snapshot', 'duckduckgo=' + config['inputs']['SEARCH_AGENTIC_DDG_SNAPSHOT'],
        '--search-snapshot', 'searxng=' + config['inputs']['SEARCH_AGENTIC_SEARXNG_SNAPSHOT'],
        '--prompts-jsonl', config['inputs']['SEARCH_AGENTIC_PROMPTS_JSONL'],
        '--selection-records-jsonl', config['inputs']['SEARCH_AGENTIC_SELECTION_RECORDS_JSONL'],
        '--cell-ids-jsonl', str(out / 'cells.jsonl'), '--prompt-count', config['settings']['SEARCH_AGENTIC_PROMPT_COUNT'],
        '--prompt-selection-seed', config['settings']['SEARCH_AGENTIC_PROMPT_SELECTION_SEED'],
        '--request-concurrency', config['settings']['SEARCH_AGENTIC_REQUEST_CONCURRENCY'],
        '--cell-concurrency', config['settings']['SEARCH_AGENTIC_CELL_CONCURRENCY'],
        '--seed', '20260911', '--query-max-tokens', '256', '--final-max-tokens', '2048',
        '--disable-thinking', '--production-conditions']
    command = [sys.executable, str(repo / 'analysis/scripts/search_vllm_stage.py'), 'run',
        '--profile', str(profile_path), '--server-log', str(out / 'server.log'),
        '--cache-base', config['cache'], '--startup-timeout-seconds', '900', '--', *controller]
    started = time.time()
    atomic(out / 'execution.json', canonical({'job_id': job, 'started_at': started, 'command': command,
           'git_commit': config['git_commit'], 'approved_walltime_seconds': 3600, 'maximum_gpu_hours': 4,
           'scientific_result': False, 'purpose': 'compatibility_only'}))
    timeout = int(env['SLURM_JOB_END_TIME']) - time.time() - 120
    if timeout <= 0:
        raise ValueError('allocation has no remaining work time')
    process = subprocess.Popen(command, cwd=repo, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        status = 'failed'
        manifest = None
        if returncode == 0:
            from analysis.scripts.run_agentic_search_integration_smoke import (
                validate_manifest_artifacts,
            )
            manifest = validate_manifest_artifacts(out / 'results/run_manifest.json', config['cell_count'])
            status = 'compatible' if manifest['status'] == 'complete' else 'checkpointed'
        atomic(out / 'compatibility.json', canonical({'status': status, 'returncode': returncode,
               'elapsed_seconds': time.time()-started, 'completed_cells': (manifest or {}).get('completed_count', 0),
               'scientific_result': False, 'gh200_calibration': False, 'reference_profile': config['reference_profile'],
               'local_profile': str(profile_path), 'job_id': job}))
        return returncode
    except subprocess.TimeoutExpired:
        atomic(out / 'compatibility.json', canonical({'status': 'deadline', 'scientific_result': False, 'job_id': job}))
        return 124
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def prepare(args):
    from analysis.interpretability.pipeline.agentic_hour_runtime import (
        admission,
        validate_reservation,
    )
    from analysis.interpretability.pipeline.agentic_hours import empty_registry
    from analysis.scripts.capture_agentic_scheduler_snapshot import capture
    from analysis.scripts.manage_agentic_hours import health
    from analysis.scripts.prepare_horeka_qwen import MODELS, capture_quota, snapshot
    from analysis.scripts.prepare_shared_hour_inputs import verify_inputs
    repo = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('clean committed checkout required')
    out = args.output.resolve()
    if out.exists():
        raise ValueError('attempt directory already exists; inspect its receipt instead of resubmitting')
    root = args.dataset.resolve()
    manifests = list((root / 'artifacts/shared-preparations').glob('*.json'))
    if len(manifests) != 1:
        raise ValueError('expected one frozen manifest')
    checked = verify_inputs(root, manifests[0], model='qwen38')
    descriptor = read(manifests[0])
    plan = read(args.plan)
    from analysis.interpretability.pipeline.agentic_hours import verify_plan
    verify_plan(plan)
    cells = choose_probe(plan, root)
    settings = descriptor['models']['qwen38']['reference_runtime']
    runtime = args.workspace / 'environment/qwen-runtime/bin/python'
    if not runtime.is_file():
        raise ValueError('prepare inference runtime first')
    quota = capture_quota(args.workspace, args.account)
    out.mkdir(parents=True, exist_ok=False)
    atomic(out / 'quota.json', canonical(quota))
    storage = health({'dataset_root': str(root), 'cluster': 'horeka'}, out / 'quota.json')
    sched = {**capture(plan={'plan_id': 'horeka-compatibility'}, since=args.since), 'cluster': 'horeka'}
    request = {'cluster': 'horeka', 'attempt_id': out.name, 'approval': {'walltime_seconds': 3600}}
    admission(empty_registry(), request, sched, {**storage, 'cluster': 'horeka', 'captured_at_epoch': int(time.time())}, now=int(time.time()))
    site = {'cluster': 'horeka', 'partition': args.partition, 'account': args.account, 'reservation': args.reservation}
    validate_reservation(request, site)
    cross = snapshot(args.workspace / 'models', {'repo_id': MODELS[1][0], 'revision': MODELS[1][1]})
    if not cross.is_dir():
        raise ValueError('pinned BGE snapshot absent')
    config = {'repository': str(repo), 'git_commit': pin, 'job_name': 'geodml-qwen-horeka-compat',
        'cell_count': len(cells), 'reference_profile': checked['files']['SEARCH_AGENTIC_PROFILE'],
        'inputs': checked['files'], 'settings': settings, 'cross_encoder': str(cross), 'cross_revision': MODELS[1][1],
        'cache': str(args.workspace / 'serving-cache' / out.name),
        'approval': {'evidence': args.approval, 'walltime_seconds': 3600, 'maximum_gpu_hours': 4,
                     'resources': {'nodes': 1, 'gpus': 4, 'cpus': 32, 'memory': 'all'},
                     'estimate': '20-45 minutes, unmeasured first A100 start; 15 minute margin; fixed one-prompt compatibility test'},
        'environment': {'HF_HUB_CACHE': str(args.workspace / 'models'), 'PATH': str(runtime.parent) + ':' + os.environ['PATH'],
                        'PYTHONPATH': str(repo), 'PYTHONDONTWRITEBYTECODE': '1'}}
    atomic(out / 'config.json', canonical(config))
    atomic(out / 'cells.jsonl', b''.join(canonical(c) + b'\n' for c in cells))
    atomic(out / 'scheduler.json', canonical(sched))
    script = '#!/bin/bash\nset -euo pipefail\nexec "$1" "$2" execute --config "$3"\n'
    atomic(out / 'run.sh', script.encode())
    command = ['sbatch', '--parsable', '--no-requeue', '--nodes=1', '--ntasks=1', '--gres=gpu:4',
        '--exclusive', '--cpus-per-task=32', '--mem=0', '--time=01:00:00', '--account=' + args.account,
        '--partition=' + args.partition, '--job-name=' + config['job_name'], '--chdir=' + str(repo),
        '--output=' + str(out / 'slurm-%j.out'), '--error=' + str(out / 'slurm-%j.err')]
    if args.reservation:
        command.append('--reservation=' + args.reservation)
    command += [str(out / 'run.sh'), str(runtime), str(Path(__file__).resolve()), str(out / 'config.json')]
    atomic(out / 'submission-command.json', canonical(command))
    # A durable marker survives an ambiguous sbatch response. Never automatically retry.
    with (out / 'SUBMISSION_ATTEMPTED').open('x') as stream:
        stream.write(str(time.time()))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    atomic(out / 'submission.json', canonical({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}))
    if result.returncode:
        raise RuntimeError(result.stderr)
    print(json.dumps({'job_id': result.stdout.strip(), 'attempt': str(out), 'cell_count': len(cells),
                      'walltime': '01:00:00', 'maximum_gpu_hours': 4, 'purpose': 'compatibility_only'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('execute')
    run.add_argument('--config', type=Path, required=True)
    submit = sub.add_parser('submit')
    for name in ('workspace', 'dataset', 'plan', 'output'):
        submit.add_argument('--' + name, type=Path, required=True)
    submit.add_argument('--account', required=True)
    submit.add_argument('--partition', default='accelerated')
    submit.add_argument('--reservation')
    submit.add_argument('--since', required=True)
    submit.add_argument('--approved-walltime', required=True, choices=['01:00:00'])
    submit.add_argument('--approval', required=True)
    args = parser.parse_args()
    return execute(args.config) if args.command == 'execute' else prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())
