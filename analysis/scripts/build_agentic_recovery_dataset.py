"""Recover all selected artifacts locally and export population-bound audit tables.

Recovery coverage is distinct from protocol-approved scientific completion.
Raw forensic packs stay local; only validated population-bound records enter Hub tables.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sqlite3
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.interpretability.pipeline.agentic_generation_tasks import build_cells
from analysis.interpretability.pipeline.agentic_judging import AgenticJudgeTask
from analysis.scripts import collect_agentic_forensic_snapshot as capture
from analysis.scripts.run_acl_arr_vllm import (
    _prepare_agentic_judge,
    _request_sha256,
    _validate_shared_outcome,
)

MODELS = {'qwen38': 'Qwen/Qwen3.8-27B', 'llama4': 'meta-llama/Llama-4-Scout-17B-16E-Instruct'}
NEMOTRON = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
SECRET = re.compile(r'hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9_.-]{20,}')


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


class Tables:
    def __init__(self, root):
        self.root, self.stack, self.counts = root, ExitStack(), Counter()
        self.streams = {}
        (root / 'data').mkdir(parents=True)

    def write(self, table, row):
        text = json.dumps(row, ensure_ascii=False, allow_nan=False)
        if SECRET.search(text):
            raise ValueError(f'credential-shaped content in {table}; publication stopped')
        if table not in self.streams:
            self.streams[table] = self.stack.enter_context(gzip.open(self.root / 'data' / (table + '.jsonl.gz'), 'wt'))  # noqa: SIM115 -- ExitStack owns stream
        self.streams[table].write(text + '\n')
        self.counts[table] += 1

    def close(self):
        self.stack.close()


class Archive:
    def __init__(self, root):
        self.root = root
        self.db = sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.packs = {}

    def read(self, path):
        row = self.db.execute('SELECT * FROM artifacts WHERE path=?', (str(path),)).fetchone()
        if row is None or not row['pack']:
            raise ValueError(f'artifact not captured: {path}')
        if row['pack'] not in self.packs:
            if len(self.packs) >= 8:
                self.packs.pop(next(iter(self.packs))).close()
            self.packs[row['pack']] = tarfile.open(self.root / row['pack'])  # noqa: SIM115 -- closed by Archive.close
        stream = self.packs[row['pack']].extractfile(row['member'])
        data = stream.read()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'archive checksum mismatch: {path}')
        return data

    def json(self, path):
        return json.loads(self.read(path), object_pairs_hook=capture._unique_object)

    def close(self):
        for stream in self.packs.values():
            stream.close()
        self.db.close()


def verified_inputs(selection):
    manifest = capture.read_json(selection)
    files, rows = {}, {}
    for key in ('prompts', 'axis_map'):
        entry = manifest['sources'][key]
        path = (selection.parent / entry['path']).resolve()
        if capture.sha_file(path) != entry['sha256']:
            raise ValueError(f'{key} checksum mismatch')
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r['candidate_id'] for r in values]
        if len(values) != entry['rows'] or len(ids) != len(set(ids)):
            raise ValueError(f'{key} count or unique-key mismatch')
        files[key], rows[key] = path, {r['candidate_id']: r for r in values}
    if rows['prompts'].keys() != rows['axis_map'].keys():
        raise ValueError('population/axis-map ID mismatch')
    return files, rows


def export(snapshot, selection, output, *, population_files, population_rows):
    archive = Archive(snapshot)
    receipt = capture.read_json(snapshot / 'snapshot.json')
    if receipt['archive_errors'] or any(receipt['validation_tiers'].get(k) for k in ('unreadable', 'unreadable_or_changed')):
        raise ValueError('incomplete forensic snapshot; originals and partial packs preserved')
    for key, path in population_files.items():
        if hashlib.sha256(archive.read(path)).hexdigest() != capture.read_json(selection)['sources'][key]['sha256']:
            raise ValueError('captured population differs from frozen selection')
    output.mkdir(exist_ok=False)
    tables = Tables(output)
    findings = Counter()
    # This population is generated, not a redistribution of readiness source corpora.
    eligible = set()
    population = capture.load_population(selection)
    for pid, row in population_rows['prompts'].items():
        generated = all(row.get(k) is not None for k in ('generator_model', 'generation_seed', 'proposal_kind', 'question_sha256'))
        restricted = 'wildchat' in json.dumps(row).casefold() or 'local-only' in json.dumps(row).casefold()
        if generated and not restricted:
            eligible.add(pid)
            tables.write('prompts', {'prompt_id': pid, 'prompt_sha256': population[pid].question_sha256,
                                     'prompt': row, 'axis': population_rows['axis_map'][pid]})
        else:
            findings['prompt_transfer_provenance_unverified'] += 1
    seen_cohorts = set()
    for row in archive.db.execute("SELECT path FROM artifacts WHERE path LIKE '%selection-records.jsonl' AND pack IS NOT NULL"):
        try:
            records = [json.loads(line) for line in archive.read(row['path']).splitlines() if line.strip()]
            cohort_id = capture._digest(records)
            if cohort_id not in seen_cohorts:
                seen_cohorts.add(cohort_id)
                for record in records:
                    if record['candidate_id'] in eligible:
                        tables.write('cohorts', {'cohort_id': cohort_id, 'prompt_id': record['candidate_id'], 'selection_record': record})
        except (ValueError, TypeError, KeyError):
            findings['invalid_cohort_records'] += 1
    for row in archive.db.execute("SELECT path FROM artifacts WHERE path LIKE '%accounting.psv' AND pack IS NOT NULL"):
        for attempt in csv.DictReader(io.StringIO(archive.read(row['path']).decode()), delimiter='|'):
            tables.write('allocations', attempt)
    for row in archive.db.execute("SELECT path FROM artifacts WHERE path LIKE '%/run_manifest.json' AND tier='json_parse_valid'"):
        value = archive.json(row['path'])
        if isinstance(value, dict):
            tables.write('runs', {'artifact_sha256': hashlib.sha256(archive.read(row['path'])).hexdigest(),
                'format_version': value.get('format_version'), 'status': value.get('status'),
                'git_commit': value.get('git_commit', value.get('source_git_commit')),
                'prompt_count': value.get('prompt_count'), 'cells_per_model': value.get('cells_per_model'),
                'claim_root': value.get('claim_root'), 'models': value.get('models')})
    slots = defaultdict(set)
    outcomes = {}
    trace_links = defaultdict(set)
    conflicts = {row[0] for row in archive.db.execute('SELECT identity FROM outcomes GROUP BY identity HAVING count(DISTINCT state || payload_sha256)>1')}
    # One row per scientifically identical payload/model/revision; aliases retain every native identity.
    for row in archive.db.execute('SELECT o.*,a.tier FROM outcomes o JOIN artifacts a USING(path) ORDER BY o.path'):
        if row['tier'] not in ('generator_payload_valid', 'legacy_payload_valid') or row['state'] != 'completed':
            continue
        value = archive.json(row['path'])
        if row['tier'] == 'generator_payload_valid':
            result, trace = value['outcome']['result'], value['outcome']['trace']
            revision = value['identity']['model_revision']
            native = value['identity']
        else:
            result = {k: v for k, v in value.items() if k != 'trace'}
            parent = Path(row['path']).parent.parent
            trace = archive.json(parent / 'traces' / Path(row['path']).name)
            config = archive.json(parent / 'config.json')
            revision, native = config['model_revision'], None
        pid = result['prompt_id']
        gid = capture._digest([row['model'], revision, row['task'], row['payload_sha256']])
        slots[row['model'], row['task']].add(gid)
        trace_links[row['model'], row['task'], trace['trace_sha256']].add(gid)
        if gid not in outcomes:
            outcomes[gid] = {'prompt_id': pid, 'cell_id': row['task'], 'model': row['model'], 'answer': result['answer'], 'conflict': False}
            if pid in eligible:
                tables.write('generations', {'generation_id': gid, 'model_id': row['model'], 'model_revision': revision,
                    'validation': 'payload_valid_protocol_acceptance_pending', 'result': result, 'trace': trace})
                for index, event in enumerate(trace.get('events', [])):
                    tables.write('events', {'generation_id': gid, 'event_index': index, 'event': event,
                                           'physical_request_identity_verified': False})
        outcomes[gid]['conflict'] |= row['identity'] in conflicts
        if pid in eligible:
            tables.write('generation_aliases', {'generation_id': gid, 'identity': row['identity'], 'native_identity': native,
                'protocol': row['protocol'], 'source_artifact_sha256': archive.db.execute('SELECT sha256 FROM artifacts WHERE path=?', (row['path'],)).fetchone()[0]})
    # Reconstruct judge requests from archived task banks, not job completion flags.
    task_by_id = {}
    for row in archive.db.execute("SELECT path FROM artifacts WHERE path LIKE '%tasks.jsonl' AND pack IS NOT NULL"):
        for number, line in enumerate(archive.read(row['path']).splitlines(), 1):
            try:
                value = json.loads(line)
                if 'judge_task_id' not in value or 'evidence' not in value:
                    continue
                task = AgenticJudgeTask.from_dict(value)
                if task.judge_task_id in task_by_id and (task_by_id[task.judge_task_id] is None or task_by_id[task.judge_task_id].to_dict() != task.to_dict()):
                    task_by_id[task.judge_task_id] = None
                    findings['conflicting_judge_task'] += 1
                elif task.judge_task_id not in task_by_id:
                    task_by_id[task.judge_task_id] = task
            except (ValueError, TypeError, KeyError, AttributeError):
                findings['unparsed_task_record'] += 1
    mappings = defaultdict(dict)
    for row in archive.db.execute("SELECT path FROM artifacts WHERE path LIKE '%private_mapping.jsonl' AND pack IS NOT NULL"):
        for line in archive.read(row['path']).splitlines():
            try:
                value = json.loads(line)
                mappings[value['judge_task_id']][capture._digest(value)] = value
            except (ValueError, TypeError, KeyError):
                findings['unparsed_mapping_record'] += 1
    judged = defaultdict(set)
    seen_judges = set()
    for row in archive.db.execute("SELECT * FROM outcomes WHERE protocol LIKE 'agentic-judge-shared-v1:%' AND state='completed'"):
        value = archive.json(row['path'])
        jid = capture._digest([row['identity'], row['payload_sha256']])
        if jid in seen_judges:
            continue
        seen_judges.add(jid)
        task = task_by_id.get(row['task'])
        try:
            if task is None or row['identity'] in conflicts:
                raise ValueError('missing/conflicting judge task or outcome')
            item = _prepare_agentic_judge(task, max_tokens=2048)
            _validate_shared_outcome(value['outcome'], item=item, fake=False, pilot_only=False)
            if value['identity']['request_sha256'] != value['outcome']['request_sha256']:
                raise ValueError('judge identity/request hash mismatch')
            links = set()
            for mapping in mappings[row['task']].values():
                for gid in trace_links.get((mapping['generator_model_id'], mapping['source_cell_id'], mapping['source_trace_sha256']), ()):
                    if outcomes[gid]['prompt_id'] == mapping['prompt_id'] and task.prompt_text == population[mapping['prompt_id']].prompt and task.answer == outcomes[gid]['answer']:
                        links.add(gid)
            if len(links) != 1:
                raise ValueError('judge does not bind to exactly one recovered generation')
            gid = next(iter(links))
            if outcomes[gid]['prompt_id'] in eligible:
                tables.write('judgments', {'judgment_id': jid, 'generation_id': gid, 'identity': value['identity'],
                    'task': task.to_dict(), 'outcome': value['outcome'], 'validation': 'request_and_trace_bound_protocol_acceptance_pending'})
            if row['model'] == NEMOTRON:
                judged[gid].add(jid)
        except (ValueError, KeyError, TypeError) as error:
            findings['unbound_or_invalid_judgment'] += 1
            tables.write('exclusions', {'artifact_sha256': archive.db.execute('SELECT sha256 FROM artifacts WHERE path=?', (row['path'],)).fetchone()[0],
                                       'reason': str(error), 'stage': 'judge'})
    # Preserve valid prefixes of interrupted journals. They do not become extra calls
    # or extra completed cells just because several retries copied the same rows.
    prompt_texts = {population[pid].prompt for pid in eligible}
    seen_journal = set()
    for artifact in archive.db.execute("SELECT path,sha256 FROM artifacts WHERE (path LIKE '%/outcomes.jsonl' OR path LIKE '%/failures.jsonl' OR path LIKE '%/attempts.jsonl') AND pack IS NOT NULL"):
        path = Path(artifact['path'])
        for number, line in enumerate(archive.read(path).splitlines(), 1):
            try:
                record = json.loads(line)
                context = record.get('task_context', record)
                task = task_by_id.get(context.get('judge_task_id'))
                if task is None or task.prompt_text not in prompt_texts:
                    continue
                item = _prepare_agentic_judge(task, max_tokens=2048)
                if any(context.get(k) != v for k, v in item['base'].items()):
                    raise ValueError('journal task context mismatch')
                tier = 'population_bound_execution_record'
                if path.name == 'outcomes.jsonl':
                    raw = record['raw_output']
                    if record.get('request_sha256') != _request_sha256(item) or hashlib.sha256(raw.encode()).hexdigest() != record.get('raw_output_sha256') or item['validator'](raw) != record.get('parsed_output'):
                        raise ValueError('journal request or outcome mismatch')
                    tier = 'judge_journal_payload_valid_protocol_acceptance_pending'
                digest = capture._digest(record)
                if digest not in seen_journal:
                    seen_journal.add(digest)
                    tables.write('judge_journals', {'record_id': digest, 'kind': path.name, 'validation': tier,
                        'source_artifact_sha256': artifact['sha256'], 'line': number, 'record': record})
            except (ValueError, KeyError, TypeError, AttributeError):
                findings['unparsed_or_invalid_journal_record'] += 1
    counts = Counter()
    per_prompt = defaultdict(Counter)
    for pid, prompt in population.items():
        for cell in build_cells((prompt,)):
            for alias, model in MODELS.items():
                matches = slots.get((model, cell.cell_id), set())
                state = 'no_validated_payload_found'
                if len(matches) > 1 or any(outcomes[g]['conflict'] for g in matches):
                    state = 'ambiguous'
                elif len(matches) == 1:
                    state = 'recovered'
                    per_prompt[alias][pid] += 1
                counts[alias, state] += 1
                judge_state = 'unverified'
                if state == 'recovered' and any(judged.get(g) for g in matches):
                    judge_state = 'recovered'
                counts['nemotron', judge_state] += 1
                if pid in eligible:
                    tables.write('coverage', {'prompt_id': pid, 'cell_id': cell.cell_id, 'model': alias,
                        'method': cell.core['method'], 'engine': cell.engine, 'condition': cell.condition.value,
                        'generation_recovery': state, 'judgment_recovery': judge_state,
                        'generation_ids': sorted(matches), 'scientific_completion': 'unverified'})
    for row in archive.db.execute('SELECT * FROM artifacts ORDER BY path'):
        tables.write('artifacts', dict(row))
    for row in archive.db.execute('SELECT * FROM external_references ORDER BY target'):
        tables.write('external_references', dict(row))
    stages = []
    for alias in (*MODELS, 'nemotron'):
        target = len(population) * (24 if alias == 'nemotron' else 12)
        stages.append({'model': alias, 'target': target, 'recovered_slots': counts[alias, 'recovered'],
            'recovered_percent': round(100 * counts[alias, 'recovered'] / target, 4),
            'ambiguous_slots': counts[alias, 'ambiguous'], 'exact_scientific_completion_percent': None,
            'prompts_all_12_recovered': sum(v == 12 for v in per_prompt[alias].values()) if alias != 'nemotron' else None})
    original500 = None
    for record in archive.db.execute("SELECT path FROM artifacts WHERE path LIKE '%/original500.json' AND tier='json_parse_valid'"):
        original500 = archive.json(record['path'])
    tables.close()
    summary = {'format_version': 'geodml-recovery-dataset-v1', 'status': 'recovery_audit_complete_scientific_acceptance_pending',
        'population': len(population), 'export_eligible_prompts': len(eligible), 'stages': stages,
        'original500_exact_audit': original500, 'table_rows': dict(tables.counts), 'findings': dict(findings), 'physical_model_calls': None,
        'raw_capture': str(snapshot), 'capture': receipt,
        'limitations': ['Recovery percentages measure validated saved payloads, not accepted Experiment V2 completion.',
                       'Different request/protocol versions remain separate; no latest-job selection.',
                       'Unpersisted inference calls cannot be reconstructed.',
                       'Raw logs, unrestricted archives and unrelated readiness corpora remain local.',
                       'External references are inventoried, not assumed captured.']}
    dump(output / 'summary.json', summary)
    lines = ['GEODML recovery audit', f'Population: {len(population)}; transfer-eligible: {len(eligible)}',
             'Model       recovered / target       recovery %   exact scientific %']
    lines += [f"{r['model']:10} {r['recovered_slots']:9} / {r['target']:<9} {r['recovered_percent']:9.2f}%   unverified" for r in stages]
    lines += [f'Findings: {dict(findings)}', 'Physical inference calls: unknown; recorded events are in data/events.jsonl.gz.']
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n')
    (output / 'README.md').write_text('# GEODML Experiment V2 recovery snapshot\n\n' + '\n'.join(summary['limitations']) + '\n\nEach data/*.jsonl.gz is a separate table. Generations contain original result and trace structures; judgments link to generation IDs. generation_aliases retains native identities and duplicate artifact references. coverage contains every export-eligible factorial slot. artifacts indexes the complete local content-addressed archive. This is a recovery snapshot, not a completed scientific dataset.\n')
    archive.close()
    files = {str(p.relative_to(output)): capture.sha_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    dump(output / 'publication-manifest.json', {'format_version': 'geodml-recovery-publication-v1', 'files': files,
        'state': 'validated_recovery_snapshot', 'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()})
    return summary


def verify_publication(output):
    manifest = capture.read_json(output / 'publication-manifest.json')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'publication-manifest.json'}:
        raise ValueError('unexpected or missing publication files')
    for name, digest in manifest['files'].items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()) or capture.sha_file(path) != digest:
            raise ValueError(f'publication checksum/path mismatch: {name}')
    return manifest


def publish(output, repo):
    from huggingface_hub import HfApi, hf_hub_download
    manifest = verify_publication(output)
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type='dataset', private=True, exist_ok=True)
    if api.repo_info(repo_id=repo, repo_type='dataset').private is not True:
        raise ValueError('destination is not private; no files transferred')
    prefix = 'snapshots/' + output.parent.name
    remote_files = set(api.list_repo_files(repo_id=repo, repo_type='dataset'))
    marker = prefix + '/publication-manifest.json'
    if marker in remote_files:
        previous = Path(hf_hub_download(repo_id=repo, repo_type='dataset', filename=marker))
        if previous.read_bytes() != (output / 'publication-manifest.json').read_bytes():
            raise ValueError('snapshot ID already exists with different contents')
    commit = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(output), path_in_repo=prefix,
                               commit_message='Add verified GEODML recovery snapshot ' + output.parent.name)
    remote_files = set(api.list_repo_files(repo_id=repo, repo_type='dataset'))
    if not {prefix + '/' + name for name in [*manifest['files'], 'publication-manifest.json']} <= remote_files:
        raise ValueError('remote snapshot file inventory incomplete')
    dump(output.parent / 'upload-receipt.json', {'repo_id': repo, 'path': prefix, 'commit': str(commit),
        'publication_manifest_sha256': capture.sha_file(output / 'publication-manifest.json'), 'source_commit': manifest['git_commit']})
    print('HF_UPLOAD=' + str(commit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    collect_parser = sub.add_parser('build')
    collect_parser.add_argument('--root', type=Path, action='append', required=True)
    collect_parser.add_argument('--selection-manifest', type=Path, required=True)
    collect_parser.add_argument('--output', type=Path, required=True)
    collect_parser.add_argument('--file', type=Path, action='append', default=[])
    publish_parser = sub.add_parser('publish')
    publish_parser.add_argument('--output', type=Path, required=True)
    publish_parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    if args.command == 'publish':
        publish(args.output.resolve(), args.repo_id)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    files, rows = verified_inputs(args.selection_manifest)
    capture.collect(args.root, args.selection_manifest, args.output / 'local-forensics', files=[*files.values(), *args.file])
    export(args.output / 'local-forensics', args.selection_manifest, args.output / 'hub', population_files=files, population_rows=rows)
    verify_publication(args.output / 'hub')
    print((args.output / 'hub/summary.txt').read_text())
    print('DATASET_DIRECTORY=' + str(args.output))


if __name__ == '__main__':
    main()
