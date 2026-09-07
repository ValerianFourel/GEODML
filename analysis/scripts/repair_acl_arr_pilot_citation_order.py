"""Create an isolated pilot-only citation-order repair without inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.acl_arr_document_experiment import validate_answer_output
from analysis.scripts.run_acl_arr_pilot_answers import load_answers, verify_hashes, _validate_saved
from analysis.scripts.run_acl_arr_vllm import _sha256, _read_jsonl, _atomic_json, _now


def repair_output(raw, allowed, *, grouped=False, formatting=False):
    if not isinstance(raw, str):
        return None
    try:
        unwrapped = False
        if formatting:
            fence = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', raw, re.DOTALL)
            if fence:
                raw = fence[1]
                unwrapped = True
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {'answer', 'cited_document_ids'}:
            return None
        answer, declared = value['answer'], value['cited_document_ids']
        if not isinstance(answer, str) or not isinstance(declared, list):
            return None
        if any(not isinstance(item, str) for item in declared):
            return None
        validation_answer = answer
        has_group = False
        if grouped:
            brackets = re.findall(r'\[([^\[\]]*)\]', answer)
            if len(brackets) != answer.count('[') or len(brackets) != answer.count(']'):
                return None
            citation_ids = []
            for content in brackets:
                if formatting:
                    content = content.strip()
                if not re.fullmatch(r'[A-Za-z0-9_.:-]+(?:\s*,\s*[A-Za-z0-9_.:-]+)*', content):
                    return None
                ids = [part.strip() for part in content.split(',')]
                has_group |= len(ids) > 1
                citation_ids.extend(ids)
            inline = list(dict.fromkeys(citation_ids))
            validation_answer = re.sub(r'\[([^\[\]]*)\]',
                lambda match: ''.join('[' + part.strip() + ']' for part in match[1].split(',')), answer)
        else:
            inline = list(dict.fromkeys(re.findall(r'\[([A-Za-z0-9_.:-]+)\]', answer)))
        if not inline or (declared == inline and not has_group and not unwrapped
                          and validation_answer == answer) or len(declared) != len(set(declared)):
            return None
        if set(declared) != set(inline) or not set(inline) <= set(allowed):
            return None
        repaired = {'answer': answer, 'cited_document_ids': inline}
        validate_answer_output(json.dumps({'answer': validation_answer, 'cited_document_ids': inline}),
                               allowed_document_ids=allowed)
        return repaired
    except (ValueError, TypeError):
        return None


def audit_and_repair(source, output, *, grouped=False, formatting=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError('output must be a fresh directory outside the source results')
    paths = [source / name for name in ('answer_manifest.json', 'outcomes.jsonl', 'failures.jsonl', 'attempts.jsonl')]
    hashes = {str(path): _sha256(path) for path in paths}
    manifest = json.loads(paths[0].read_text())
    if (manifest.get('format_version') != 'acl-arr-pilot-answer-step-v1'
            or manifest.get('scientific_result') is not False
            or manifest.get('eligible_for_analysis') is not False
            or manifest.get('status') != 'complete_with_failures'):
        raise ValueError('expected a finished, pilot-only answer failure manifest')
    verify_hashes(manifest['source_artifacts_sha256'])
    if hashes[str(paths[1])] != manifest['outcomes_sha256']:
        raise ValueError('source outcomes hash mismatch')
    run_root = Path(manifest['tasks_path']).parents[3]
    items, frozen_hashes, metadata = load_answers(run_root, Path(manifest['rerank_recovery_results']))
    for key in ('tasks_sha256', 'model_id', 'model_revision', 'plan_source_git_commit'):
        if manifest[key] != metadata[key]:
            raise ValueError('source identity mismatch: ' + key)
    prepared = {item['base']['task_id']: item for item in items}
    done = set()
    for row in (_read_jsonl(paths[1]) if paths[1].stat().st_size else []):
        task_id = _validate_saved(row, prepared)
        if task_id in done:
            raise ValueError('duplicate successful task')
        done.add(task_id)
    repairs, unresolved, seen = [], [], set()
    order_count, grouped_count, formatting_count = 0, 0, 0
    policy = 'pilot-only-citation-order-and-groups-v1' if grouped else 'pilot-only-citation-order-v1'
    if formatting:
        policy = 'pilot-only-citation-formatting-v1'
    for row in _read_jsonl(paths[2]):
        task_id = row['task_id']
        if task_id in seen or task_id in done or task_id not in prepared:
            raise ValueError('duplicate, completed, or unknown failed task')
        seen.add(task_id)
        item = prepared[task_id]
        if any(row.get(key) != value for key, value in item['base'].items()):
            raise ValueError('failed task identity mismatch')
        if row.get('scientific_result') is not False or row.get('eligible_for_analysis') is not False:
            raise ValueError('failure must be pilot-only')
        raw = row.get('raw_output')
        if isinstance(raw, str) and hashlib.sha256(raw.encode()).hexdigest() != row.get('raw_output_sha256'):
            raise ValueError('failed raw output hash mismatch')
        repaired = repair_output(raw, item['base']['input_document_ids'])
        repair_kind = 'order_only'
        if repaired is None and grouped:
            repaired = repair_output(raw, item['base']['input_document_ids'], grouped=True)
            repair_kind = 'grouped_citations'
        if repaired is None and formatting:
            repaired = repair_output(raw, item['base']['input_document_ids'], grouped=True, formatting=True)
            repair_kind = 'whitespace_or_json_fence'
        if repaired is None:
            unresolved.append({'task_id': task_id, 'original_error': row.get('error'),
                               'scientific_result': False, 'eligible_for_analysis': False})
        else:
            order_count += repair_kind == 'order_only'
            grouped_count += repair_kind == 'grouped_citations'
            formatting_count += repair_kind == 'whitespace_or_json_fence'
            repairs.append({'task_id': task_id, 'source_failure': row,
                            'repair_policy': policy, 'repair_kind': repair_kind,
                            'repaired_output': repaired, 'scientific_result': False,
                            'eligible_for_analysis': False})
    if len(done) != manifest['completed_count'] or len(done | seen) != manifest['task_count']:
        raise ValueError('source coverage mismatch')
    verify_hashes({**hashes, **frozen_hashes})
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (('repaired_records.jsonl', repairs), ('unresolved.jsonl', unresolved)):
        with (output / name).open('x') as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    verify_hashes(hashes)
    result = {'format_version': ('acl-arr-pilot-citation-groups-repair-v1' if grouped
                                else 'acl-arr-pilot-citation-order-repair-v1'), 'status': 'complete',
              'created_at': _now(), 'source_artifacts_sha256': hashes,
              'execution_git_commit': subprocess.check_output(['git', '-C', str(REPOSITORY_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'repair_policy': policy, 'order_only_repaired_count': order_count,
              'grouped_citation_repaired_count': grouped_count,
              'formatting_repaired_count': formatting_count,
              'counts_scope': 'cumulative over original failures; do not add earlier repair counts',
              'scientific_result': False, 'eligible_for_analysis': False,
              'original_valid_count': len(done), 'repaired_count': len(repairs),
              'unresolved_count': len(unresolved), 'inference_requests': 0,
              'repaired_records_sha256': _sha256(output / 'repaired_records.jsonl'),
              'unresolved_sha256': _sha256(output / 'unresolved.jsonl')}
    _atomic_json(output / 'repair_manifest.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--approve-pilot-only-citation-order-repair', action='store_true', required=True)
    parser.add_argument('--approve-pilot-only-grouped-citations', action='store_true')
    parser.add_argument('--approve-pilot-only-formatting', action='store_true')
    args = parser.parse_args()
    audit_and_repair(args.source_results, args.output_dir,
                     grouped=args.approve_pilot_only_grouped_citations,
                     formatting=args.approve_pilot_only_formatting)
