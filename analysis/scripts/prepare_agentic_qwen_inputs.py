#!/usr/bin/env python3
"""Copy an immutable Qwen input mirror; never register tasks or launch inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
from analysis.interpretability.pipeline.agentic_hour_sync import (
    checkpoint_files,
    import_events,
)
from analysis.interpretability.pipeline.agentic_hours import (
    digest,
    inventory,
)
from analysis.scripts.prepare_horeka_qwen import MODELS, immutable_json
from analysis.scripts.search_vllm_stage import load_profile

INPUT_KEYS = {
    'SEARCH_AGENTIC_PROFILE': 'reference-profile.json',
    'SEARCH_AGENTIC_DDG_SNAPSHOT': 'duckduckgo.parquet',
    'SEARCH_AGENTIC_SEARXNG_SNAPSHOT': 'searxng.parquet',
    'SEARCH_AGENTIC_PROMPTS_JSONL': 'population-prompts.jsonl',
    'SEARCH_AGENTIC_SELECTION_RECORDS_JSONL': 'population-selection-records.jsonl',
}
SETTING_KEYS = ('SEARCH_AGENTIC_CROSS_ENCODER_REVISION', 'SEARCH_AGENTIC_PROMPT_COUNT',
                'SEARCH_AGENTIC_PROMPT_SELECTION_SEED', 'SEARCH_AGENTIC_REQUEST_CONCURRENCY',
                'SEARCH_AGENTIC_CELL_CONCURRENCY')
MANIFEST = 'artifacts/qwen-preparation.json'


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def identity(path: Path) -> dict:
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def scheduler_gate(value: dict) -> None:
    if (value.get('cluster') != 'jupiter' or value.get('complete') is not True
            or value.get('jobs') != []
            or not 0 <= time.time() - value.get('captured_at_epoch', 0) <= 120):
        raise ValueError('fresh JUPITER proof that legacy allocations have stopped is required')


def copy_immutable(source: Path, target: Path, expected: dict) -> None:
    if target.is_symlink():
        raise ValueError(f'staging target is a symlink: {target}')
    if target.exists():
        if identity(target) != expected:
            raise ValueError(f'existing staged file conflicts: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        if identity(temporary) != expected:
            raise ValueError(f'source changed while staging: {source}')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def stage_inputs(source: Path, output: Path, runtime: dict, priority: Path,
                 scheduler: dict, *, stripes: int = 256) -> dict:
    scheduler_gate(scheduler)
    source, output = source.resolve(), output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('input mirror must be separate from the live dataset')
    profile = load_profile(runtime['SEARCH_AGENTIC_PROFILE'])
    if profile['model'] != {'model_id': MODELS[0][0], 'model_revision': MODELS[0][1]}:
        raise ValueError('reference profile is not the pinned Qwen model')
    settings = {key: str(runtime[key]) for key in SETTING_KEYS}
    if settings['SEARCH_AGENTIC_CROSS_ENCODER_REVISION'] != MODELS[1][1]:
        raise ValueError('compactor revision differs from the pinned workflow')
    if any(int(settings[key]) < 1 for key in SETTING_KEYS if key != 'SEARCH_AGENTIC_CROSS_ENCODER_REVISION'):
        raise ValueError('invalid frozen runtime setting')
    tasks, completed, blocked = inventory(source, stripes=stripes)
    chosen = {row['fingerprint']: row for row in tasks if row['model'] == 'qwen38'}
    if not chosen:
        raise ValueError('no registered Qwen tasks; registration must not be recreated here')
    if any((t['claim_identity']['model_id'], t['claim_identity']['model_revision']) != MODELS[0]
           for t in chosen.values()):
        raise ValueError('registered Qwen tasks have another model revision')
    names, outcomes = checkpoint_files(source, chosen, stripes=stripes)
    if (blocked & chosen.keys()) - outcomes.keys():
        raise ValueError('unreconciled Qwen claims or unverifiable completions; reconcile on JUPITER first')
    names = set(names) | {'README.md', 'contract.json'}
    names.update(str(p.relative_to(source)) for p in (source / 'schemas').rglob('*') if p.is_file())
    # Keep complete sealed shards and their checksums, including mixed registration shards.
    for table in ('prompts', 'keyword_memberships', 'task_definitions'):
        list(iter_sealed_rows(source, table, required=True))
        for manifest_path in (source / 'data' / table).glob('*.manifest.json'):
            manifest = json.loads(manifest_path.read_bytes())
            if table == 'task_definitions':
                with (source / manifest['path']).open() as stream:
                    if not any(json.loads(line)['row']['model'] == 'qwen38' for line in stream if line.strip()):
                        continue
            names.update((str(manifest_path.relative_to(source)), manifest['path']))
    bindings = {key: Path(runtime[key]) for key in INPUT_KEYS}
    bindings['keyword_priority'] = priority
    inputs = {key: identity(path) for key, path in bindings.items()}
    prompt_ids = {t['prompt_id'] for t in chosen.values()}
    if int(settings['SEARCH_AGENTIC_PROMPT_COUNT']) != len(prompt_ids):
        raise ValueError('runtime prompt count differs from registered Qwen population')
    for row in iter_sealed_rows(source, 'prompts', required=True):
        if row['prompt_id'] in prompt_ids:
            origin = row.get('source', {})
            if (origin.get('prompts_jsonl_sha256') != inputs['SEARCH_AGENTIC_PROMPTS_JSONL']['sha256']
                    or origin.get('selection_records_jsonl_sha256') != inputs['SEARCH_AGENTIC_SELECTION_RECORDS_JSONL']['sha256']):
                raise ValueError('frozen prompt files differ from the registered population')
    prefix = 'artifacts/qwen-inputs/' + digest(inputs)[:24]
    local_paths = {key: prefix + '/' + INPUT_KEYS.get(key, 'keyword-priority.json') for key in bindings}
    for key in ('SEARCH_AGENTIC_DDG_SNAPSHOT', 'SEARCH_AGENTIC_SEARXNG_SNAPSHOT'):
        if bindings[key].suffix not in {'.parquet', '.jsonl'}:
            raise ValueError('unsupported frozen search snapshot format')
        local_paths[key] = str(Path(local_paths[key]).with_suffix(bindings[key].suffix))
    files = {name: identity(source / name) for name in sorted(names)}
    files.update({local_paths[key]: inputs[key] for key in bindings})
    result = {'format_version': 'geodml-qwen-preparation-v1', 'files': files,
              'inputs': local_paths, 'runtime_settings': settings,
              'qwen_task_count': len(chosen), 'verified_completed': len(completed & chosen.keys()),
              'blocked': len(blocked & chosen.keys()),
              'eligible_remaining': len(chosen.keys() - completed - blocked),
              'reference_profile_sha256': inputs['SEARCH_AGENTIC_PROFILE']['sha256'],
              'reference_vllm_version': profile['runtime']['vllm_version'],
              'outcomes': outcomes, 'gpu_validation': 'not_performed'}
    if (output / MANIFEST).exists() and json.loads((output / MANIFEST).read_bytes()) != result:
        raise ValueError('source snapshot changed; choose a new input mirror, preserving the old one')
    for name in sorted(names):
        origin, target = source / name, output / name
        if origin.is_symlink() or not origin.resolve().is_relative_to(source):
            raise ValueError('dataset input escapes source')
        if not target.resolve().is_relative_to(output):
            raise ValueError('staging target escapes mirror')
        copy_immutable(origin, target, files[name])
    for key, origin in bindings.items():
        target = output / local_paths[key]
        if not target.resolve().is_relative_to(output):
            raise ValueError('staging target escapes mirror')
        copy_immutable(origin, target, inputs[key])
    import_events(output, outcomes, stripes=stripes)
    immutable_json(output / MANIFEST, result)
    verify_inputs(output, stripes=stripes)
    return result


def verify_inputs(root: Path, *, stripes: int = 256) -> dict:
    value = json.loads((root / MANIFEST).read_bytes())
    if value.get('format_version') != 'geodml-qwen-preparation-v1':
        raise ValueError('unsupported Qwen preparation manifest')
    for name, expected in value['files'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or identity(path) != expected:
            raise ValueError(f'input checksum/path mismatch: {name}')
    tasks, done, blocked = inventory(root, stripes=stripes)
    selected = {r['fingerprint'] for r in tasks if r['model'] == 'qwen38'}
    if len(selected) != value['qwen_task_count']:
        raise ValueError('Qwen task inventory changed')
    return {'status': 'verified', 'dataset_root': str(root.resolve()),
            'manifest_sha256': sha(root / MANIFEST), 'files': len(value['files']),
            'bytes': sum(e['bytes'] for e in value['files'].values()),
            'qwen_task_count': len(selected), 'verified_completed': len(done & selected),
            'blocked': len(blocked & selected), 'eligible_remaining': len(selected - done - blocked),
            'reference_vllm_version': value['reference_vllm_version'],
            'gpu_validation': 'not_performed'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage')
    for key in ('source', 'output', 'runtime', 'keyword-priority', 'scheduler-snapshot'):
        stage.add_argument('--' + key, type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--dataset-root', type=Path, required=True)
    verify.add_argument('--report', type=Path, required=True)
    for p in (stage, verify):
        p.add_argument('--stripes', type=int, default=256)
    args = parser.parse_args(argv)
    if args.command == 'stage':
        result = stage_inputs(args.source, args.output, json.loads(args.runtime.read_bytes()), args.keyword_priority,
                              json.loads(args.scheduler_snapshot.read_bytes()), stripes=args.stripes)
        print(json.dumps({k: v for k, v in result.items() if k not in {'files', 'outcomes'}}, sort_keys=True))
    else:
        result = verify_inputs(args.dataset_root, stripes=args.stripes)
        immutable_json(args.report, result)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
