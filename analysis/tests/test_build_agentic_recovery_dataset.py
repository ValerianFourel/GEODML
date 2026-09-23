# ruff: noqa: F811
import gzip
import json
from types import SimpleNamespace

import pytest

from analysis.scripts import build_agentic_recovery_dataset as dataset
from analysis.scripts import collect_agentic_forensic_snapshot as capture
from analysis.tests.test_collect_agentic_forensic_snapshot import (  # noqa: F401 -- pytest fixture
    put,
    source,
)


def prepare(source, tmp_path):
    root, selection, prompt, *_ = source
    population = root / 'population.jsonl'
    row = json.loads(population.read_text())
    row.update(generator_model='generator', generation_seed=42, proposal_kind='generated', question_sha256=prompt.question_sha256)
    population.write_text(json.dumps(row) + '\n')
    axis = root / 'axis.jsonl'
    axis.write_text(json.dumps({'candidate_id': prompt.prompt_id, 'axis_1': 0.5}) + '\n')
    manifest = json.loads(selection.read_text())
    manifest['sources']['prompts']['sha256'] = capture.sha_file(population)
    manifest['sources']['axis_map'] = {'path': str(axis), 'rows': 1, 'sha256': capture.sha_file(axis)}
    put(selection, manifest)
    return root, selection


def run(source, tmp_path):
    root, selection = prepare(source, tmp_path)
    files, rows = dataset.verified_inputs(selection)
    snapshot = tmp_path / 'snapshot'
    capture.collect([root], selection, snapshot)
    output = tmp_path / 'hub'
    summary = dataset.export(snapshot, selection, output, population_files=files, population_rows=rows)
    return summary, output


def read_table(output, name):
    with gzip.open(output / 'data' / (name + '.jsonl.gz'), 'rt') as stream:
        return [json.loads(line) for line in stream]


def test_failed_job_durable_result_exported_and_coverage_denominator_fixed(source, tmp_path):
    summary, output = run(source, tmp_path)
    assert summary['stages'][0]['target'] == 12
    assert summary['stages'][0]['recovered_slots'] == 1
    assert summary['stages'][0]['exact_scientific_completion_percent'] is None
    assert len(read_table(output, 'coverage')) == 24
    assert len(read_table(output, 'generations')) == 1
    assert read_table(output, 'generations')[0]['result']['answer'] == 'Saved answer'
    assert not list(output.glob('*.tar'))
    dataset.verify_publication(output)


def test_duplicate_native_claim_does_not_double_count(source, tmp_path):
    root = source[0]
    original = next((root / 'failed-job/claims').glob('*/*.json'))
    copy = root / 'retry/claims' / original.parent.name / original.name
    copy.parent.mkdir(parents=True)
    copy.write_bytes(original.read_bytes())
    summary, output = run(source, tmp_path)
    assert summary['stages'][0]['recovered_slots'] == 1
    assert len(read_table(output, 'generations')) == 1
    assert len(read_table(output, 'generation_aliases')) == 2


def test_conflicting_identity_kept_but_not_counted_recovered(source, tmp_path):
    from analysis.interpretability.pipeline.inference_claims import InferenceClaimStore
    root, _, _, _, bundle, identity = source
    bundle['result']['answer'] = 'Competing answer'
    with InferenceClaimStore(root / 'retry/claims').try_claim(identity) as claim:
        claim.commit(bundle)
    summary, _ = run(source, tmp_path)
    assert summary['stages'][0]['recovered_slots'] == 0
    assert summary['stages'][0]['ambiguous_slots'] == 1


def test_publication_rejects_changed_or_extra_file(source, tmp_path):
    _, output = run(source, tmp_path)
    (output / 'secret.txt').write_text('extra')
    with pytest.raises(ValueError, match='unexpected'):
        dataset.verify_publication(output)
    (output / 'secret.txt').unlink()
    (output / 'README.md').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        dataset.verify_publication(output)


def test_existing_public_repo_never_receives_data(source, tmp_path, monkeypatch):
    _, output = run(source, tmp_path)
    calls = []
    class API:
        def create_repo(self, **kwargs):
            calls.append('create')
        def repo_info(self, **kwargs):
            return SimpleNamespace(private=False)
        def upload_folder(self, **kwargs):
            calls.append('upload')
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    with pytest.raises(ValueError, match='not private'):
        dataset.publish(output, 'example/private')
    assert calls == ['create']


def test_population_axis_mismatch_stops(source, tmp_path):
    root, selection = prepare(source, tmp_path)
    (root / 'axis.jsonl').write_text('{"candidate_id":"different"}\n')
    manifest = json.loads(selection.read_text())
    manifest['sources']['axis_map']['sha256'] = capture.sha_file(root / 'axis.jsonl')
    put(selection, manifest)
    with pytest.raises(ValueError, match='ID mismatch'):
        dataset.verified_inputs(selection)


def test_explicit_external_file_is_preserved(source, tmp_path):
    root, selection, *_ = source
    extra = tmp_path / 'external.parquet'
    extra.write_bytes(b'example')
    capture.collect([root], selection, tmp_path / 'snapshot', files=[extra])
    archive = dataset.Archive(tmp_path / 'snapshot')
    assert archive.read(extra) == b'example'
    archive.close()


def test_judge_binds_request_answer_and_trace_to_recovered_generation(source, tmp_path):
    import hashlib

    from analysis.interpretability.pipeline.agentic_judging import (
        FORMAT_VERSION,
        AgenticJudgeTask,
    )
    from analysis.interpretability.pipeline.inference_claims import (
        ClaimIdentity,
        InferenceClaimStore,
    )
    from analysis.scripts.run_acl_arr_vllm import (
        _prepare_agentic_judge,
        _request_sha256,
    )
    root, _, prompt, cell, bundle, _ = source
    task = AgenticJudgeTask('judge-1', FORMAT_VERSION, 'blind-1', prompt.prompt, (), bundle['result']['answer'])
    tasks = root / 'judge/plan/bulk_tasks.jsonl'
    tasks.parent.mkdir(parents=True)
    tasks.write_text(json.dumps(task.to_dict()) + '\n')
    mapping = {'judge_task_id': task.judge_task_id, 'generator_model_id': dataset.MODELS['qwen38'],
               'source_cell_id': cell.cell_id, 'source_trace_sha256': bundle['trace']['trace_sha256'], 'prompt_id': prompt.prompt_id}
    (tasks.parent / 'private_mapping.jsonl').write_text(json.dumps(mapping) + '\n')
    item = _prepare_agentic_judge(task, max_tokens=2048)
    raw = item['fake_output']
    request_hash = _request_sha256(item)
    result = {'ok': True, 'base': item['base'], 'request_sha256': request_hash,
              'fake_backend': False, 'pilot_only': False, 'usage': {}, 'duration_seconds': 1,
              'started_at': '2026-09-23T00:00:00Z', 'finished_at': '2026-09-23T00:00:01Z',
              'producer': {'run_id': 'r1', 'invocation_id': 'i1', 'execution_git_commit': 'a'*40,
                           'slurm': {'SLURM_JOB_ID': '1', 'SLURM_ARRAY_JOB_ID': None, 'SLURM_ARRAY_TASK_ID': None}},
              'raw_output': raw, 'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'parsed_output': item['validator'](raw)}
    identity = ClaimIdentity(task.judge_task_id, dataset.NEMOTRON, 'a'*40, 'agentic-judge-shared-v1:' + 'b'*64, request_hash)
    with InferenceClaimStore(root / 'judge/claims').try_claim(identity) as claim:
        claim.commit(result)
    journal = root / 'failed-judge/outcomes.jsonl'
    journal.parent.mkdir()
    journal_row = {**result['base'], **{k: v for k, v in result.items() if k != 'base'}}
    journal.write_text(json.dumps(journal_row) + '\n{broken')
    summary, output = run(source, tmp_path)
    assert len(read_table(output, 'judge_journals')) == 1
    assert summary['stages'][2]['recovered_slots'] == 1
    judgment = read_table(output, 'judgments')[0]
    assert judgment['generation_id'] == read_table(output, 'generations')[0]['generation_id']


def test_private_upload_uses_content_addressed_snapshot_and_receipt(source, tmp_path, monkeypatch):
    _, output = run(source, tmp_path)
    uploads = []
    class API:
        def __init__(self):
            self.files = []
        def create_repo(self, **kwargs):
            assert kwargs['private'] is True
        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)
        def list_repo_files(self, **kwargs):
            return self.files
        def upload_folder(self, **kwargs):
            uploads.append(kwargs)
            self.files = [kwargs['path_in_repo'] + '/' + str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()]
            return 'https://example.invalid/commit/123'
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    dataset.publish(output, 'example/private')
    receipt = json.loads((output.parent / 'upload-receipt.json').read_text())
    assert receipt['path'].startswith('snapshots/recovery-')
    assert uploads[0]['folder_path'] == str(output)
    assert 'local-forensics' not in uploads[0]['folder_path']
