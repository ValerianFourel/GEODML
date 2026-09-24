#!/usr/bin/env python3
"""Freeze existing registrations and inputs without registering or accepting work."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_audit_progress import audit_stage
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
from analysis.interpretability.pipeline.agentic_hour_sync import (
    checkpoint_files,
    import_events,
)
from analysis.interpretability.pipeline.agentic_hours import digest, inventory
from analysis.scripts.prepare_agentic_qwen_inputs import (
    copy_immutable,
    identity,
    scheduler_gate,
)
from analysis.scripts.prepare_horeka_qwen import immutable_json
from analysis.scripts.search_vllm_stage import load_profile

FILE_KEYS = {'SEARCH_AGENTIC_PROFILE', 'SEARCH_AGENTIC_DDG_SNAPSHOT', 'SEARCH_AGENTIC_SEARXNG_SNAPSHOT',
             'SEARCH_AGENTIC_PROMPTS_JSONL', 'SEARCH_AGENTIC_SELECTION_RECORDS_JSONL',
             'GEODML_JUDGE_MANIFEST', 'GEODML_JUDGE_PROFILE'}
GENERATOR_KEYS = {key for key in FILE_KEYS if key.startswith('SEARCH_')}
SETTING_KEYS = {'SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT', 'SEARCH_AGENTIC_CROSS_ENCODER_REVISION',
                'SEARCH_AGENTIC_PROMPT_COUNT', 'SEARCH_AGENTIC_PROMPT_SELECTION_SEED',
                'SEARCH_AGENTIC_REQUEST_CONCURRENCY', 'SEARCH_AGENTIC_CELL_CONCURRENCY',
                'GEODML_JUDGE_ROLE', 'GEODML_JUDGE_DISABLE_THINKING', 'GEODML_JUDGE_CONCURRENCY',
                'GEODML_JUDGE_MAX_OUTPUT_TOKENS', 'GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY'}


def stage(source, output, model_inputs, priority, snapshot, *, stripes=256):
    scheduler_gate(snapshot)
    source, output = source.resolve(), output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('input mirror must be separate from the live dataset')
    tasks, done, blocked = inventory(source, stripes=stripes)
    if not tasks:
        raise ValueError('no registered work to freeze')
    indexed = {t['fingerprint']: t for t in tasks}
    names, outcomes = checkpoint_files(source, indexed, stripes=stripes)
    if blocked - outcomes.keys():
        raise ValueError('unreconciled claims or invalid completions; reconcile JUPITER first')
    models = {t['model'] for t in tasks}
    if models - model_inputs.keys():
        raise ValueError('missing frozen input configuration for: ' + ', '.join(sorted(models - model_inputs.keys())))
    names = set(names) | {'contract.json', 'README.md'}
    names.update(str(p.relative_to(source)) for p in (source / 'schemas').rglob('*') if p.is_file())
    for table in ('prompts', 'keyword_memberships', 'task_definitions'):
        list(iter_sealed_rows(source, table, required=True))
        for path in (source / 'data' / table).glob('*.manifest.json'):
            value = json.loads(path.read_bytes())
            names.update((str(path.relative_to(source)), value['path']))
    origins = {name: source / name for name in sorted(names)}
    bindings = {}
    for model in sorted(models):
        config = model_inputs[model]
        runtime = config['runtime_environment']
        if set(runtime) - FILE_KEYS - SETTING_KEYS or any(not isinstance(v, str) for v in runtime.values()):
            raise ValueError('runtime values must be strings')
        required = {'GEODML_JUDGE_MANIFEST', 'GEODML_JUDGE_PROFILE'} if model == 'nemotron' else GENERATOR_KEYS
        if not required <= runtime.keys():
            raise ValueError('frozen runtime lacks required inputs for ' + model)
        profile_key = 'GEODML_JUDGE_PROFILE' if model == 'nemotron' else 'SEARCH_AGENTIC_PROFILE'
        profile = load_profile(runtime[profile_key])
        selected = [t for t in tasks if t['model'] == model]
        if any((t['claim_identity']['model_id'], t['claim_identity']['model_revision']) !=
               (profile['model']['model_id'], profile['model']['model_revision']) for t in selected):
            raise ValueError('reference model differs from registered work')
        files = {key: Path(runtime[key]) for key in required}
        for name, path in config.get('extra_files', {}).items():
            if '/' in name or '\\' in name or name in files or name in {'.', '..'}:
                raise ValueError('invalid extra input name')
            files[name] = Path(path)
        if model != 'nemotron':
            prompt_hash = identity(files['SEARCH_AGENTIC_PROMPTS_JSONL'])['sha256']
            selection_hash = identity(files['SEARCH_AGENTIC_SELECTION_RECORDS_JSONL'])['sha256']
            prompts = {t['prompt_id'] for t in selected}
            for row in iter_sealed_rows(source, 'prompts', required=True):
                if row['prompt_id'] in prompts and (
                    row.get('source', {}).get('prompts_jsonl_sha256') != prompt_hash or
                    row.get('source', {}).get('selection_records_jsonl_sha256') != selection_hash):
                    raise ValueError('prompt inputs differ from registered identities')
        mapping = {}
        for key, path in sorted(files.items()):
            entry = identity(path)
            name = f"artifacts/shared-inputs/{model}/{entry['sha256']}/{path.name}"
            origins[name] = path
            mapping[key] = name
        # Preserve original settings for provenance; site paths are rebound separately.
        bindings[model] = {'files': mapping, 'reference_runtime': runtime,
                           'reference_profile_sha256': identity(Path(runtime[profile_key]))['sha256']}
    origins['artifacts/shared-inputs/keyword-priority.json'] = priority
    files = {name: identity(path) for name, path in sorted(origins.items())}
    report = {'format_version': 'geodml-shared-inputs-v1', 'files': files, 'models': bindings,
              'outcomes': outcomes, 'registered_tasks': len(tasks), 'verified_completed': len(done),
              'missing_models': sorted({'qwen38', 'llama4', 'nemotron'} - models),
              'scientific_settings_changed': False}
    manifest_name = f'artifacts/shared-preparations/{digest(report)}.json'
    owner = output / 'local-only/shared-input-owner.json'
    if owner.exists() and json.loads(owner.read_bytes()) != {'manifest': manifest_name}:
        raise ValueError('source changed; preserve this mirror and select a new output')
    immutable_json(owner, {'manifest': manifest_name})
    for name, path in origins.items():
        if name in names and (path.is_symlink() or not path.resolve().is_relative_to(source)):
            raise ValueError('source dataset file escapes its root')
        if not (output / name).resolve().is_relative_to(output):
            raise ValueError('input destination escapes mirror')
        copy_immutable(path, output / name, files[name])
    import_events(output, outcomes, stripes=stripes)
    immutable_json(output / manifest_name, report)
    return {'status': 'frozen', 'manifest_sha256': digest(report), 'registered_tasks': len(tasks),
            'verified_completed': len(done), 'missing_models': report['missing_models']}


@audit_stage("verify_inputs")
def verify_inputs(root: Path, manifest: Path, *, model: str, full_audit: bool = False) -> dict:
    root = root.resolve()
    manifest = manifest if manifest.is_absolute() else root / manifest
    if manifest.is_symlink() or not manifest.resolve().is_relative_to(root):
        raise ValueError('shared input manifest escapes the dataset')
    value = json.loads(manifest.read_bytes())
    if value.get('format_version') != 'geodml-shared-inputs-v1' or manifest.stem != digest(value):
        raise ValueError('shared input manifest checksum mismatch')
    from analysis.interpretability.pipeline.agentic_verification_cache import (
        VerificationCache,
    )
    with VerificationCache(root, force=full_audit) as verification:
        for name, expected in value['files'].items():
            path = root / name
            if not verification.file(path, expected):
                raise ValueError('shared input file checksum/path mismatch: ' + name)
    binding = value['models'][model]
    key = 'GEODML_JUDGE_PROFILE' if model == 'nemotron' else 'SEARCH_AGENTIC_PROFILE'
    profile = root / binding['files'][key]
    if identity(profile)['sha256'] != binding['reference_profile_sha256']:
        raise ValueError('shared reference profile checksum mismatch')
    loaded = load_profile(profile)
    return {'status': 'verified', 'model': model, 'reference_profile_sha256': binding['reference_profile_sha256'],
            'reference_vllm_version': loaded['runtime']['vllm_version'],
            'files': {key: str(root / path) for key, path in binding['files'].items()},
            'gpu_validation': 'not_performed'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'model-inputs', 'keyword-priority', 'scheduler-snapshot'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stripes', type=int, default=256)
    args = parser.parse_args(argv)
    result = stage(args.source, args.output, json.loads(args.model_inputs.read_bytes()), args.keyword_priority,
                   json.loads(args.scheduler_snapshot.read_bytes()), stripes=args.stripes)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
