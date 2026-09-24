#!/usr/bin/env python3
"""Inventory, resume and verify only the frozen Qwen and BGE downloads. No Slurm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import atomic
from analysis.interpretability.pipeline.agentic_hours import canonical

MODELS = (
    ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
    ('BAAI/bge-reranker-v2-m3', '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'),
)
FORMAT = 'geodml-qwen-downloads-v1'


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def selected(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError('unsafe model file path')
    if any(part in {'original', 'onnx', 'openvino', '.git'} for part in path.parts):
        return False
    return path.suffix in {'.json', '.safetensors', '.model', '.txt', '.py', '.jinja', '.tiktoken'}


def validate_manifest(value: dict) -> None:
    if value.get('format_version') != FORMAT or [
        (m.get('repo_id'), m.get('revision')) for m in value.get('models', [])
    ] != list(MODELS):
        raise ValueError('manifest must contain exactly the two pinned models')
    for model in value['models']:
        files = model['files']
        if not {'config.json', 'tokenizer_config.json'} <= files.keys():
            raise ValueError('model configuration/tokenizer files missing')
        if not any(name.endswith('.safetensors') for name in files):
            raise ValueError('model weights missing')
        if not any(name in files for name in ('tokenizer.json', 'tokenizer.model', 'sentencepiece.bpe.model', 'vocab.json')):
            raise ValueError('tokenizer vocabulary missing')
        for name, entry in files.items():
            if not selected(name) or type(entry['bytes']) is not int or entry['bytes'] < 0:
                raise ValueError('invalid model file inventory')
            algorithm = entry.get('hash_kind')
            length = {'sha256': 64, 'git-blob-sha1': 40}.get(algorithm)
            if not length or not re.fullmatch(f'[0-9a-f]{{{length}}}', entry.get('hash', '')):
                raise ValueError('model file has no verifiable remote checksum')


def model_inventory(api) -> dict:
    models = []
    for repo, revision in MODELS:
        info = api.model_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError('Hub returned a different pinned revision')
        files = {}
        for sibling in info.siblings:
            if not selected(sibling.rfilename):
                continue
            sha = field(field(sibling, 'lfs'), 'sha256')
            files[sibling.rfilename] = {
                'bytes': sibling.size, 'hash_kind': 'sha256' if sha else 'git-blob-sha1',
                'hash': sha or field(sibling, 'blob_id'),
            }
        models.append({'repo_id': repo, 'revision': revision, 'files': dict(sorted(files.items()))})
    value = {'format_version': FORMAT, 'models': models}
    validate_manifest(value)
    return value


def immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f'existing preparation artifact conflicts: {path}')
    else:
        atomic(path, canonical(value) + b'\n')


def verify_file(path: Path, entry: dict) -> None:
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        raise ValueError(f'missing model file or size mismatch: {path}')
    hasher = hashlib.sha256() if entry['hash_kind'] == 'sha256' else hashlib.sha1()
    if entry['hash_kind'] == 'git-blob-sha1':
        hasher.update(f"blob {entry['bytes']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            hasher.update(block)
    if hasher.hexdigest() != entry['hash']:
        raise ValueError(f'model checksum mismatch: {path}')


def snapshot(cache: Path, model: dict) -> Path:
    return cache / ('models--' + model['repo_id'].replace('/', '--')) / 'snapshots' / model['revision']


def verify_models(manifest: dict, cache: Path) -> dict:
    validate_manifest(manifest)
    records = []
    for model in manifest['models']:
        root = snapshot(cache, model)
        for name, entry in model['files'].items():
            path = root / name
            if not path.resolve().is_relative_to(cache.resolve()):
                raise ValueError('model cache link escapes configured cache')
            verify_file(path, entry)
            if name.endswith('.safetensors.index.json'):
                index = json.loads(path.read_bytes())
                if not index.get('weight_map') or not set(index['weight_map'].values()) <= model['files'].keys():
                    raise ValueError('weight index references missing shards')
        records.append({'repo_id': model['repo_id'], 'revision': model['revision'],
                        'snapshot': str(root.resolve()), 'files': len(model['files']),
                        'bytes': sum(e['bytes'] for e in model['files'].values())})
    return {'status': 'verified', 'models': records, 'gpu_validation': 'not_performed'}


def check_storage(cache: Path, quota: dict, required_bytes: int, required_files: int) -> None:
    if (quota.get('cluster') != 'horeka' or quota.get('within_limits') is not True
            or not 0 <= time.time() - quota.get('captured_at_epoch', 0) <= 300
            or not cache.resolve().is_relative_to(Path(quota.get('workspace', '/not-configured')).resolve())):
        raise ValueError('fresh matching HoreKa quota evidence required')
    parent = cache
    while not parent.exists():
        parent = parent.parent
    stats = os.statvfs(parent)
    margin = max(5 * 1024**3, required_bytes // 10)
    if (min(shutil.disk_usage(parent).free, quota.get('work_headroom_bytes', 0)) < required_bytes + margin
            or min(stats.f_favail, quota.get('work_headroom_files', 0)) < required_files + 1000):
        raise ValueError('insufficient quota/filesystem byte or file headroom')


def download_models(manifest: dict, cache: Path, quota: dict, *, download=None) -> dict:
    validate_manifest(manifest)
    missing = [(m, name, entry) for m in manifest['models'] for name, entry in m['files'].items()
               if not (snapshot(cache, m) / name).exists()]
    # Include room for partial downloads and the largest transient file.
    sizes = [e['bytes'] for _, _, e in missing]
    check_storage(cache, quota, sum(sizes) + max(sizes, default=0), len(missing) * 3)
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    for model in manifest['models']:
        for name, entry in model['files'].items():
            path = snapshot(cache, model) / name
            if not path.exists():
                print(f"DOWNLOAD {model['repo_id']} {name}", file=sys.stderr, flush=True)
                downloaded = Path(download(model['repo_id'], revision=model['revision'], filename=name,
                                           cache_dir=str(cache)))
                if downloaded.resolve() != path.resolve():
                    raise ValueError('download returned an unexpected cache location')
            if not path.resolve().is_relative_to(cache.resolve()):
                raise ValueError('model cache link escapes configured cache')
            verify_file(path, entry)
    return verify_models(manifest, cache)


def parse_quota(raw: str, kind: str) -> dict:
    rows = [line.split() for line in raw.splitlines() if len(line.split()) >= 12 and line.split()[1] == kind]
    if len(rows) != 1:
        raise ValueError('could not parse one authoritative GPFS quota row')
    row = rows[0]
    used, soft, hard, doubt = (float(v) for v in row[2:6])
    files, fsoft, fhard, fdoubt = (int(v) for v in row[7:11])
    if min(soft, hard, fsoft, fhard) <= 0 or min(used, doubt, files, fdoubt) < 0:
        raise ValueError('quota limits unavailable or invalid')
    return {'headroom_bytes': int((min(soft, hard)-used-doubt) * 1024**3),
            'headroom_files': min(fsoft, fhard)-files-fdoubt, 'raw': raw}


def capture_quota(workspace: Path, project: str, cluster: str = 'hkn.scc.kit.edu') -> dict:
    import getpass
    workspace = workspace.resolve(strict=True)
    # Query the mounted filesystem, not a guessed cross-cluster configuration.
    executable = '/usr/lpp/mmfs/bin/mmlsquota'
    results = {}
    for name, option, identity, fs, kind in (
        ('home', '-j', project, 'hkfs-home', 'FILESET'),
        ('work', '-u', getpass.getuser(), 'hkfs-work', 'USR'),
    ):
        raw = subprocess.check_output([executable, option, identity, '--block-size', 'G', '-C', cluster, fs],
                                      text=True, timeout=60)
        results[name] = parse_quota(raw, kind)
    return {'cluster': 'horeka', 'captured_at_epoch': int(time.time()), 'workspace': str(workspace),
            'within_limits': all(r['headroom_bytes'] > 0 and r['headroom_files'] > 0 for r in results.values()),
            'work_headroom_bytes': results['work']['headroom_bytes'],
            'work_headroom_files': results['work']['headroom_files'], 'quotas': results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inv = commands.add_parser('inventory')
    inv.add_argument('--manifest', type=Path, required=True)
    for command in ('download', 'verify'):
        p = commands.add_parser(command)
        p.add_argument('--manifest', type=Path, required=True)
        p.add_argument('--cache', type=Path, required=True)
        p.add_argument('--report', type=Path, required=True)
        if command == 'download':
            p.add_argument('--quota-evidence', type=Path, required=True)
    quota = commands.add_parser('quota')
    quota.add_argument('--workspace', type=Path, required=True)
    quota.add_argument('--project', required=True)
    quota.add_argument('--quota-cluster', default='hkn.scc.kit.edu')
    quota.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'quota':
        result = capture_quota(args.workspace, args.project, args.quota_cluster)
        atomic(args.output, canonical(result) + b'\n')
    elif args.command == 'inventory':
        from huggingface_hub import HfApi
        result = model_inventory(HfApi())
        immutable_json(args.manifest, result)
        print('REQUIRED_MODEL_BYTES=' + str(sum(e['bytes'] for m in result['models'] for e in m['files'].values())))
    else:
        value = json.loads(args.manifest.read_bytes())
        if args.command == 'download':
            result = download_models(value, args.cache, json.loads(args.quota_evidence.read_bytes()))
        else:
            result = verify_models(value, args.cache)
        immutable_json(args.report, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
