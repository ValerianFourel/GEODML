"""Capture and inspect artifacts regardless of scheduler state; never mutate sources.

This forensic index is not a protocol-selection or full-population coverage audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline.agentic_generation_tasks import (
    CalibrationPrompt,
    build_cells,
)
from analysis.interpretability.pipeline.inference_claims import (
    InferenceClaimStore,
    _digest,
    _identity,
    _unique_object,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    _validate_completed_cell,
    _validate_shared_generator_bundle,
)
from analysis.scripts.submit_agentic_generator_backlog import (
    _generator_scientific_outcome,
    _portable_generator_trace,
)

HEX64 = re.compile(r'[0-9a-f]{64}')
HEX20 = re.compile(r'[0-9a-f]{20}')
EXCLUDED_DIRS = {'.git', '.venv', '__pycache__', '.cache', 'restricted-local'}
EXCLUDED_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.pem', '.key'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def load_population(selection):
    manifest = read_json(selection)
    entry = manifest['sources']['prompts']
    path = (selection.parent / entry['path']).resolve()
    if sha_file(path) != entry['sha256']:
        raise ValueError('population prompt checksum mismatch')
    result = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line, object_pairs_hook=_unique_object)
            key = row['candidate_id']
            if not isinstance(key, str) or not key or key in result:
                raise ValueError('invalid or duplicate population prompt ID')
            text = row['question']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('empty population prompt')
            digest = hashlib.sha256(text.encode()).hexdigest()
            if row.get('question_sha256', digest) != digest:
                raise ValueError('population question hash mismatch')
            result[key] = CalibrationPrompt(key, text, digest, 0, row['keyword'])
    if len(result) != entry['rows']:
        raise ValueError('population row count mismatch')
    return result


class Packs:
    def __init__(self, output, limit):
        self.output, self.limit = output, limit
        self.stream = None
        self.size = 0
        self.index = 0

    def add(self, path, digest):
        size = path.stat().st_size
        if self.stream is None or self.size + size > self.limit:
            self.close()
            self.index += 1
            self.name = f'artifacts-{self.index:05d}.tar'
            self.stream = tarfile.open(self.output / self.name, 'w')  # noqa: SIM115 -- rotated and closed in finally
            self.size = 0
        info = tarfile.TarInfo(digest)
        info.size, info.mode, info.mtime = size, 0o600, 0
        with path.open('rb') as source:
            reader = HashReader(source)
            self.stream.addfile(info, reader)
            if reader.digest.hexdigest() != digest:
                raise ValueError('archived bytes differ from source checksum')
        self.size += size
        return self.name, digest

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def path_references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and child.startswith('/') and (
                key in {'path', 'root', 'claim_root', 'materialized_root', 'source_result_path'}
                or key.endswith(('_path', '_root', '_jsonl'))
            ):
                yield child
            else:
                yield from path_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from path_references(child)


def candidate_cell(record, population):
    prompt = population[record['prompt_id']]
    matches = [cell for cell in build_cells((prompt,)) if cell.cell_id == record['cell_id']]
    if len(matches) != 1:
        raise ValueError('result cell is outside the population factorial')
    return matches[0], prompt.prompt


def inspect_artifact(path, population):
    """Return validation tier and optional immutable outcome identity."""
    filename = path.name.removesuffix('.failed.json').removesuffix('.json')
    if HEX64.fullmatch(filename) and path.suffix == '.json' and path.parent.name == filename[:2]:
        record = read_json(path)
        identity = _identity(record['identity'])
        if _digest(record['identity']) != filename:
            raise ValueError('claim filename/identity mismatch')
        store = InferenceClaimStore(path.parent.parent)
        state, outcome = store.inspect(identity)
        if state == 'busy':
            return 'busy_claim', None
        expected = 'failed' if path.name.endswith('.failed.json') else 'completed'
        if state != expected:
            raise ValueError('claim status differs from artifact')
        scientific = _generator_scientific_outcome(record)
        tier = 'claim_envelope_valid'
        if identity.protocol == 'agentic-generator-shared-v1':
            payload = outcome['diagnostics'] if state == 'failed' else outcome['result']
            cell, prompt = candidate_cell(payload, population)
            if identity.task_id != cell.cell_id:
                raise ValueError('claim task ID differs from payload')
            _validate_shared_generator_bundle(cell, prompt, outcome, failed=state == 'failed')
            tier = 'generator_payload_valid' if state == 'completed' else 'generator_failure_valid'
        digest = hashlib.sha256(scientific).hexdigest() if scientific is not None else record['outcome_sha256']
        return tier, (filename, state, identity.model_id, identity.protocol, identity.task_id, digest)
    if path.parent.name == 'results' and HEX20.fullmatch(path.stem) and path.suffix == '.json':
        result = read_json(path)
        cell, _ = candidate_cell(result, population)
        output = path.parent.parent
        trace_path = output / 'traces' / path.name
        _validate_completed_cell(cell, trace_path, path)
        trace = read_json(trace_path)
        if trace.get('user_prompt_sha256') != cell.prompt_sha256:
            raise ValueError('legacy trace prompt hash mismatch')
        if not isinstance(result.get('answer'), str) or not result['answer'].strip():
            raise ValueError('legacy answer missing')
        config = read_json(output / 'config.json')
        config_hash = config.get('config_sha256')
        if config_hash != _digest({k: v for k, v in config.items() if k != 'config_sha256'}):
            raise ValueError('legacy config hash mismatch')
        model = config.get('model_id')
        revision = config.get('model_revision')
        if not isinstance(model, str) or not model or not isinstance(revision, str) or not revision:
            raise ValueError('legacy model identity missing')
        portable = _portable_generator_trace({k: v for k, v in result.items() if k != 'trace'}, trace)
        if portable is None:
            raise ValueError('legacy scientific payload is not comparable')
        fingerprint = 'legacy:' + _digest([cell.cell_id, model, revision, config_hash])
        return 'legacy_payload_valid', (fingerprint, 'completed', model, 'legacy-config:' + config_hash, cell.cell_id, _digest(portable))
    if path.suffix == '.json':
        read_json(path)
        return 'json_parse_valid', None
    if path.suffix == '.jsonl':
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith(b'\n'):
                    raise ValueError(f'unterminated JSONL record at line {number}; original preserved')
                if not isinstance(json.loads(line, object_pairs_hook=_unique_object), dict):
                    raise TypeError(f'non-object JSONL record at line {number}')
        return 'jsonl_parse_valid', None
    return 'preserved_uninterpreted', None


def collect(roots, selection, output, *, pack_bytes=512 * 1024 * 1024, files=()):
    roots = sorted({Path(root).resolve() for root in roots})
    roots = [p for p in roots if not any(p != q and p.is_relative_to(q) for q in roots)]
    selection, output = Path(selection).resolve(), Path(output).resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in roots):
        raise ValueError('snapshot output and input roots must not overlap')
    if not all(path.is_dir() for path in roots):
        raise ValueError('every input root must be an existing directory')
    population = load_population(selection)
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    db = sqlite3.connect(output / 'inventory.sqlite')
    db.executescript('''
      CREATE TABLE artifacts(path TEXT PRIMARY KEY, sha256 TEXT, bytes INTEGER,
        pack TEXT, member TEXT, tier TEXT, error TEXT);
      CREATE TABLE contents(sha256 TEXT PRIMARY KEY, pack TEXT, member TEXT);
      CREATE TABLE outcomes(identity TEXT, state TEXT, model TEXT, protocol TEXT,
        task TEXT, payload_sha256 TEXT, path TEXT PRIMARY KEY);
      CREATE TABLE external_references(source TEXT, target TEXT, PRIMARY KEY(source,target));
    ''')
    started = now()
    receipt = {'format_version': 'geodml-forensic-snapshot-v1', 'started_at': started,
               'status': 'collecting', 'roots': [str(p) for p in roots],
               'selection_manifest': str(selection), 'population_prompts': len(population),
               'collector_sha256': sha_file(Path(__file__)),
               'population_scope': 'prompt file hash and row identities verified; axis map requires separate audit'}
    (output / 'snapshot.json').write_text(json.dumps(receipt, indent=2) + '\n')
    packs = Packs(output, pack_bytes)
    count = 0
    try:
        def walk_error(error):
            db.execute('INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?)',
                       (str(error.filename), None, None, None, None, 'unreadable', str(error)))
        walks = (entry for root in roots for entry in os.walk(root, followlinks=False, onerror=walk_error))
        extra_files = sorted({Path(p).resolve() for p in files})
        extra_files = [p for p in extra_files if not any(p.is_relative_to(r) for r in roots)]
        from itertools import chain
        for directory, dirs, names in chain(walks, ((str(p.parent), [], [p.name]) for p in extra_files)):
                parent = Path(directory)
                for name in list(dirs):
                    path = parent / name
                    if name in EXCLUDED_DIRS or path.is_symlink():
                        dirs.remove(name)
                        db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                                   (str(path), None, None, None, None, 'excluded_directory', 'cache, restricted scope, or symlink'))
                for name in sorted(names):
                    path = parent / name
                    if path.is_symlink() or not path.is_file() or name.startswith('.env') or path.suffix in EXCLUDED_SUFFIXES or name.endswith('.lock'):
                        db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                                   (str(path), None, None, None, None, 'excluded_file', 'symlink, special file, lock, model binary, or credentials'))
                        continue
                    digest = pack = member = size = None
                    tier, error, outcome = 'unreadable', None, None
                    try:
                        before = file_state(path)
                        size = before[2]
                        digest = sha_file(path)
                        known = db.execute('SELECT pack,member FROM contents WHERE sha256=?', (digest,)).fetchone()
                        pack, member = known if known else packs.add(path, digest)
                        if before != file_state(path) or (not known and sha_file(path) != digest):
                            raise ValueError('source changed during capture; snapshot is incomplete')
                        if not known:
                            db.execute('INSERT INTO contents VALUES(?,?,?)', (digest, pack, member))
                        try:
                            tier, outcome = inspect_artifact(path, population)
                            if path.name in {'run_manifest.json', 'launch.json', 'selection-manifest.json', 'config.json'}:
                                for target in path_references(read_json(path)):
                                    resolved = Path(target).resolve()
                                    if not any(resolved.is_relative_to(root) for root in roots):
                                        db.execute('INSERT OR IGNORE INTO external_references VALUES(?,?)', (str(path), target))
                        except (OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
                            tier, error = 'unverified_or_invalid', str(exc)
                        if before != file_state(path):
                            raise ValueError('source changed during validation; snapshot is incomplete')
                    except (OSError, ValueError, tarfile.TarError) as exc:
                        tier, error, outcome = 'unreadable_or_changed', str(exc), None
                    db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                               (str(path), digest, size, pack, member, tier, error))
                    if outcome:
                        db.execute('INSERT INTO outcomes VALUES(?,?,?,?,?,?,?)', (*outcome, str(path)))
                    count += 1
                    if count % 1000 == 0:
                        db.commit()
                        print(f'CAPTURED_FILES={count}', file=sys.stderr, flush=True)
        packs.close()
        db.commit()
        archive_errors = []
        for pack in sorted(output.glob('artifacts-*.tar')):
            try:
                with tarfile.open(pack) as stream:
                    for member in stream:
                        reader = HashReader(stream.extractfile(member))
                        while reader.read(1024 * 1024):
                            pass
                        if reader.digest.hexdigest() != member.name:
                            archive_errors.append(f'{pack.name}:{member.name}: checksum mismatch')
            except (OSError, tarfile.TarError) as exc:
                archive_errors.append(f'{pack.name}: {exc}')
        tiers = dict(db.execute('SELECT tier,count(*) FROM artifacts GROUP BY tier'))
        conflicts = db.execute('SELECT count(*) FROM (SELECT identity FROM outcomes GROUP BY identity HAVING count(DISTINCT state || payload_sha256)>1)').fetchone()[0]
        claims = [dict(zip(('model','protocol','state','distinct_identities'), row)) for row in db.execute(
            'SELECT model,protocol,state,count(DISTINCT identity) FROM outcomes GROUP BY model,protocol,state')]
        receipt.update(ended_at=now(), status='captured_with_findings', files=count,
                       unique_contents=db.execute('SELECT count(*) FROM contents').fetchone()[0],
                       validation_tiers=tiers, conflicting_identities=conflicts,
                       identity_counts=claims, full_population_completion_percent=None,
                       archive_errors=archive_errors,
                       external_reference_count=db.execute('SELECT count(DISTINCT target) FROM external_references').fetchone()[0],
                       physical_model_calls=None,
                       coverage_status='pending protocol acceptance, legacy/native reconciliation and judge-to-generation validation',
                       snapshot_consistency='sequential capture with per-file change checks; not a filesystem transaction')
        (output / 'snapshot.json').write_text(json.dumps(receipt, indent=2) + '\n')
        with (output / 'checksums.sha256').open('w') as stream:
            for path in sorted(output.iterdir()):
                if path.name != 'checksums.sha256' and path.is_file():
                    stream.write(f'{sha_file(path)}  {path.name}\n')
        return receipt
    finally:
        packs.close()
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = collect(args.root, args.selection_manifest, args.output)
    print(json.dumps({key: receipt[key] for key in ('status','files','unique_contents','validation_tiers','conflicting_identities','coverage_status')}, indent=2))
    print('SNAPSHOT_DIRECTORY=' + str(args.output.resolve()))
    return 2 if receipt['archive_errors'] or any(
        receipt['validation_tiers'].get(tier, 0)
        for tier in ('unreadable', 'unreadable_or_changed')
    ) else 0


if __name__ == '__main__':
    raise SystemExit(main())
