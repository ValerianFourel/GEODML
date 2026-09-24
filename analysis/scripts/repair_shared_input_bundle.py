#!/usr/bin/env python3
"""Repair omitted frozen-input objects without changing an immutable bundle or plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import (
    Exchange,
    HubStore,
    atomic,
    relative_path,
)
from analysis.interpretability.pipeline.agentic_hours import digest
from analysis.scripts.publish_agentic_dataset import SECRET_BYTES


def repair(exchange, bundle, root, *, publish=False):
    root = root.resolve()
    revision = exchange.store.head()
    outer = exchange.manifest(bundle, revision)
    if outer.get('metadata', {}).get('kind') != 'frozen-inputs':
        raise ValueError('expected frozen input bundle')
    descriptors = [n for n in outer['files'] if n.startswith('artifacts/shared-preparations/') and n.endswith('.json')]
    if len(descriptors) != 1:
        raise ValueError('expected one frozen input descriptor')
    name = descriptors[0]
    entry = outer['files'][name]
    raw = exchange.store.read('exchange/objects/' + entry['sha256'], revision)
    if raw is None or len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('invalid frozen descriptor object')
    descriptor = json.loads(raw)
    if descriptor.get('format_version') != 'geodml-shared-inputs-v1' or Path(name).stem != digest(descriptor):
        raise ValueError('invalid frozen descriptor identity')
    omitted = {n: e for n, e in descriptor['files'].items() if n not in outer['files']}
    for name, expected in omitted.items():
        relative_path(name)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('repair path escapes dataset')
        raw = path.read_bytes() if publish or path.exists() else exchange.store.read('exchange/objects/' + expected['sha256'], revision)
        if raw is None or len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError(f'missing or conflicting repair object: {name}; publish from JUPITER first')
        if SECRET_BYTES.search(raw):
            raise ValueError('credential-shaped repair rejected')
        if publish:
            exchange.immutable({'exchange/objects/' + expected['sha256']: raw})
        elif not path.exists():
            atomic(path, raw)
    return {'status': 'repair_objects_published' if publish else 'local_files_repaired',
            'files': sorted(omitted), 'input_bundle': bundle, 'plan_changed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--repo-id', default='ValerianFourel/geodml-experiment-v2-paper-private')
    args = parser.parse_args(argv)
    print(json.dumps(repair(Exchange(HubStore(args.repo_id), args.journal), args.bundle,
                            args.dataset_root, publish=args.publish), indent=2))


if __name__ == '__main__':
    main()
