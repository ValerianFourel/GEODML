#!/usr/bin/env python3
"""Download a pinned historical archive, independently of scientific progress."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import (
    Exchange,
    HubStore,
    atomic,
)
from analysis.interpretability.pipeline.agentic_hours import canonical, digest
from analysis.scripts.prepare_horeka_qwen import (
    check_storage,
    field,
    immutable_json,
    verify_file,
)

REPO = 'ValerianFourel/geodml-papersize'
REVISION = '8ba12dcc592e41eda86f349ca3dde97d726e4b4b'


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or '\\' in name or '.git' in path.parts:
        raise ValueError('unsafe archive path')
    return name


def inventory(api):
    info = api.repo_info(REPO, repo_type='dataset', revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('archive revision differs from pin')
    files = {}
    for item in info.siblings:
        name = safe_name(item.rfilename)
        sha = field(field(item, 'lfs'), 'sha256')
        checksum = sha or field(item, 'blob_id')
        if type(item.size) is not int or item.size < 0 or not re.fullmatch(
                '[0-9a-f]{64}' if sha else '[0-9a-f]{40}', checksum or ''):
            raise ValueError('archive file lacks authoritative size/checksum')
        files[name] = {'bytes': item.size, 'hash_kind': 'sha256' if sha else 'git-blob-sha1', 'hash': checksum}
    return {'format_version': 'geodml-historical-archive-v1', 'repo_id': REPO,
            'revision': REVISION, 'files': dict(sorted(files.items())),
            'scientific_completion_imported': False, 'archives_extracted': False}


def download_archive(manifest, root, quota, download):
    if manifest['repo_id'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('unexpected archive source')
    missing = [e['bytes'] for n, e in manifest['files'].items() if not (root / safe_name(n)).exists()]
    check_storage(root, quota, sum(missing) + max(missing, default=0), len(missing) * 4)
    for name, entry in manifest['files'].items():
        target = root / safe_name(name)
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('archive path escapes destination')
        if not target.exists():
            obtained = Path(download(REPO, name, repo_type='dataset', revision=REVISION, local_dir=str(root)))
            if obtained.resolve() != target.resolve():
                raise ValueError('archive download returned another path')
        verify_file(target, entry)
    return {'status': 'verified', 'repo_id': REPO, 'revision': REVISION,
            'inventory_sha256': digest(manifest), 'files': len(manifest['files']),
            'bytes': sum(e['bytes'] for e in manifest['files'].values()),
            'scientific_completion_imported': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--quota-evidence', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--publish-catalog', action='store_true')
    parser.add_argument('--repo-id', default='ValerianFourel/geodml-experiment-v2-paper-private')
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download
    manifest = inventory(HfApi())
    immutable_json(args.report_dir / 'inventory.json', manifest)
    if args.inventory_only:
        print(json.dumps({'files': len(manifest['files']), 'bytes': sum(e['bytes'] for e in manifest['files'].values()),
                          'revision': REVISION}))
        return 0
    report = download_archive(manifest, args.root, json.loads(args.quota_evidence.read_bytes()), hf_hub_download)
    atomic(args.report_dir / 'verified.json', canonical(report))
    if args.publish_catalog:
        exchange = Exchange(HubStore(args.repo_id), args.report_dir / 'journal')
        exchange.immutable({f'artifacts/archive-catalogs/{REVISION}.json': canonical(manifest)})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
