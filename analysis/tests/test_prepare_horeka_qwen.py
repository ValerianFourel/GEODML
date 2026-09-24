"""Download preparation needs neither GPU execution nor an approved allocation."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import prepare_horeka_qwen as prep

FILES = {"config.json": b'{}', "tokenizer.json": b'{}',
         "tokenizer_config.json": b'{}', "model.safetensors": b'fixture weights'}


class ModelHub:
    def __init__(self):
        self.calls = []

    def model_info(self, repo_id, *, revision, files_metadata):
        self.calls.append((repo_id, revision))
        return SimpleNamespace(sha=revision, siblings=[
            SimpleNamespace(rfilename=name, size=len(raw),
                            lfs=SimpleNamespace(sha256=hashlib.sha256(raw).hexdigest()), blob_id=None)
            for name, raw in {**FILES, 'original/model.pth': b'no', 'onnx/model.onnx': b'no'}.items()
        ])


def quota(root):
    return {"cluster": "horeka", "captured_at_epoch": int(time.time()),
            "within_limits": True, "workspace": str(root.resolve()),
            "work_headroom_bytes": 10**12, "work_headroom_files": 10**6}


def test_only_pinned_qwen_and_required_compactor_are_inventoried():
    hub = ModelHub()
    value = prep.model_inventory(hub)
    assert hub.calls == [
        ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
        ('BAAI/bge-reranker-v2-m3', '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')]
    assert all(set(row['files']) == set(FILES) for row in value['models'])


def test_interrupted_download_resumes_and_verifies_bytes(tmp_path):
    manifest = prep.model_inventory(ModelHub())
    calls = []
    interrupted = [True]

    def download(repo_id, *, revision, filename, cache_dir):
        calls.append((repo_id, filename))
        if filename == 'model.safetensors' and interrupted[0]:
            interrupted[0] = False
            raise ConnectionError('interrupted')
        path = Path(cache_dir) / ('models--' + repo_id.replace('/', '--')) / 'snapshots' / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FILES[filename])
        return str(path)

    with pytest.raises(ConnectionError):
        prep.download_models(manifest, tmp_path / 'models', quota(tmp_path), download=download)
    prep.download_models(manifest, tmp_path / 'models', quota(tmp_path), download=download)
    report = prep.verify_models(manifest, tmp_path / 'models')
    assert report['status'] == 'verified'
    assert calls.count(('Qwen/Qwen3.8-27B', 'config.json')) == 1
    path = next((tmp_path / 'models').glob('models--Qwen*/snapshots/*/model.safetensors'))
    path.write_bytes(b'x' * len(FILES['model.safetensors']))
    with pytest.raises(ValueError, match='checksum|size'):
        prep.verify_models(manifest, tmp_path / 'models')


def test_stale_quota_and_wrong_model_rejected_before_download(tmp_path):
    manifest = prep.model_inventory(ModelHub())
    stale = {**quota(tmp_path), 'captured_at_epoch': 0}
    with pytest.raises(ValueError, match='quota'):
        prep.download_models(manifest, tmp_path / 'models', stale, download=lambda *a, **k: pytest.fail('download'))
    manifest['models'][0]['repo_id'] = 'meta-llama/anything'
    with pytest.raises(ValueError, match='pinned'):
        prep.verify_models(manifest, tmp_path / 'models')


def test_missing_weight_shard_is_not_a_complete_inventory():
    class MissingShard(ModelHub):
        def model_info(self, *args, **kwargs):
            value = super().model_info(*args, **kwargs)
            value.siblings = [row for row in value.siblings if row.rfilename != 'model.safetensors']
            return value
    with pytest.raises(ValueError, match='weights'):
        prep.model_inventory(MissingShard())


def test_small_files_use_git_blob_checksums(tmp_path):
    path = tmp_path / 'config.json'
    path.write_bytes(b'{}')
    expected = {'bytes': 2, 'hash_kind': 'git-blob-sha1',
                'hash': hashlib.sha1(b'blob 2\0{}').hexdigest()}
    prep.verify_file(path, expected)
    path.write_bytes(b'[]')
    with pytest.raises(ValueError, match='checksum'):
        prep.verify_file(path, expected)


def test_weight_index_cannot_hide_missing_shards(tmp_path):
    manifest = prep.model_inventory(ModelHub())
    for model in manifest['models']:
        for name, raw in FILES.items():
            path = prep.snapshot(tmp_path, model) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    raw = json.dumps({'weight_map': {'layer': 'missing.safetensors'}}).encode()
    name = 'model.safetensors.index.json'
    manifest['models'][0]['files'][name] = {'bytes': len(raw), 'hash_kind': 'sha256',
                                         'hash': hashlib.sha256(raw).hexdigest()}
    (prep.snapshot(tmp_path, manifest['models'][0]) / name).write_bytes(raw)
    with pytest.raises(ValueError, match='missing shards'):
        prep.verify_models(manifest, tmp_path)


@pytest.mark.parametrize('separator', [' | ', '|'])
@pytest.mark.parametrize('kind,blocks,files,expected_bytes,expected_files', [
    ('FILESET', '2955 10240 11264 4 none', '2161595 10485760 11534336 271 none',
     7281 * 1024**3, 8323894),
    ('USR', '13 256000 276480 0 none', '37862 52428800 57671680 0 none',
     255987 * 1024**3, 52390938),
])
def test_quota_parser_accepts_gpfs_column_separators(separator, kind, blocks, files,
                                                   expected_bytes, expected_files):
    raw = ('Block Limits | File Limits\n'
           'Filesystem type GB quota limit in_doubt grace | files quota limit in_doubt grace | Remarks\n'
           f'work {kind} {blocks}{separator}{files}{separator}hkfs.scc.kit.edu\n')
    result = prep.parse_quota(raw, kind)
    assert result == {'headroom_bytes': expected_bytes, 'headroom_files': expected_files, 'raw': raw}


def test_quota_parser_includes_in_doubt_and_both_byte_and_file_limits():
    row = prep.parse_quota('home FILESET 2955 10240 11264 4 none 2161595 10485760 11534336 271 none site', 'FILESET')
    assert row['headroom_bytes'] == (10240-2955-4) * 1024**3
    assert row['headroom_files'] == 10485760-2161595-271
    with pytest.raises(ValueError, match='quota'):
        prep.parse_quota('permission denied', 'USR')


def test_input_download_never_requires_approval_or_changes_registry(tmp_path, monkeypatch, capsys):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
    from analysis.interpretability.pipeline.agentic_hours import inventory
    from analysis.scripts import manage_agentic_hours as cli
    from analysis.scripts.publish_agentic_dataset import build_manifest
    from analysis.tests.test_agentic_hours import MemoryHub, complete, data
    root = data(tmp_path / 'jupiter')
    tasks, _, _ = inventory(root, stripes=4)
    complete(root, tasks[0], 'done')
    hub = MemoryHub()
    exchange = Exchange(hub, tmp_path / 'journal')
    from analysis.interpretability.pipeline.agentic_hour_sync import checkpoint_files
    _, outcomes = checkpoint_files(root, {r['fingerprint']: r for r in tasks}, stripes=4)
    bundle = exchange.upload(root, list(build_manifest(root)['files']), outcomes=outcomes,
                             metadata={'kind': 'frozen-inputs'})
    monkeypatch.setattr(cli, 'HubStore', lambda _: hub)
    before = hub.head()
    quota_file = tmp_path / 'quota.json'
    quota_file.write_text(json.dumps(quota(tmp_path)))
    args = ['--journal', str(tmp_path / 'journal2'), 'download-inputs', '--input-bundle', bundle,
            '--dataset-root', str(tmp_path / 'horeka'), '--stripes', '4', '--quota-evidence', str(quota_file)]
    assert cli.main(args) == 0
    assert cli.main(args) == 0
    assert hub.head() == before
    result = inventory(tmp_path / 'horeka', stripes=4)
    assert len(result[1]) == 1
    assert {r['fingerprint'] for r in result[0]} == {r['fingerprint'] for r in tasks}
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['verified_completed'] == 1


def test_frozen_qwen_mirror_preserves_identity_and_excludes_other_results(tmp_path):
    from dataclasses import replace

    from analysis.interpretability.pipeline.agentic_dataset import initialize_dataset
    from analysis.interpretability.pipeline.agentic_hours import inventory
    from analysis.scripts.prepare_agentic_qwen_inputs import (
        MANIFEST,
        stage_inputs,
        verify_inputs,
    )
    from analysis.scripts.register_agentic_dataset_tasks import register_generator_tasks
    from analysis.scripts.run_agentic_search_integration_smoke import (
        _generator_claim_identity,
        describe_queue,
    )
    from analysis.scripts.search_vllm_stage import build_profile
    from analysis.tests.test_agentic_hours import complete
    from analysis.tests.test_register_agentic_dataset_tasks import (
        _generator_inputs,
        _priority,
    )
    root = tmp_path / 'jupiter'
    initialize_dataset(root, population_id='population', acceptance_policy_id='v2')
    qwen = replace(_generator_inputs(tmp_path, model_id=prep.MODELS[0][0], revision=prep.MODELS[0][1]),
                   cross_encoder_revision=prep.MODELS[1][1], cell_concurrency=12)
    priority = _priority(tmp_path)
    register_generator_tasks(dataset_root=root, model_slug='qwen38', inputs=qwen,
                             keyword_priority_path=priority, writer_id='register-qwen')
    llama = replace(qwen, model_id='meta-llama/Llama-4', model_revision='b' * 40)
    register_generator_tasks(dataset_root=root, model_slug='llama4', inputs=llama,
                             keyword_priority_path=priority, writer_id='register-llama')
    tasks, _, _ = inventory(root, stripes=4)
    qtasks = [r for r in tasks if r['model'] == 'qwen38']
    complete(root, qtasks[0], 'done-qwen')
    complete(root, next(r for r in tasks if r['model'] == 'llama4'), 'done-llama')
    profile = build_profile(stage='qwen-generator', model_id=prep.MODELS[0][0], model_revision=prep.MODELS[0][1],
        vllm_executable='/fixture/vllm', vllm_version='0.28.0', vllm_help='',
        visible_gpus=[{'index': i, 'uuid': f'GPU-{i}', 'name': 'GH200', 'memory_total_mib': 97871} for i in range(4)],
        cuda_visible_devices='0,1,2,3', expected_gpu_name_pattern='GH200', max_model_len=4096, request_concurrency=1)
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(profile))
    runtime = {'SEARCH_AGENTIC_PROFILE': str(path),
               'SEARCH_AGENTIC_DDG_SNAPSHOT': str(qwen.search_snapshots['duckduckgo']),
               'SEARCH_AGENTIC_SEARXNG_SNAPSHOT': str(qwen.search_snapshots['searxng']),
               'SEARCH_AGENTIC_PROMPTS_JSONL': str(qwen.prompts_jsonl),
               'SEARCH_AGENTIC_SELECTION_RECORDS_JSONL': str(qwen.selection_records_jsonl),
               'SEARCH_AGENTIC_CROSS_ENCODER_REVISION': prep.MODELS[1][1],
               'SEARCH_AGENTIC_PROMPT_COUNT': '2', 'SEARCH_AGENTIC_PROMPT_SELECTION_SEED': '20260912',
               'SEARCH_AGENTIC_REQUEST_CONCURRENCY': '1', 'SEARCH_AGENTIC_CELL_CONCURRENCY': '12'}
    snapshot = {'cluster': 'jupiter', 'complete': True, 'jobs': [], 'captured_at_epoch': int(time.time())}
    output = tmp_path / 'mirror'
    with pytest.raises(ValueError, match='legacy allocations'):
        stage_inputs(root, output, runtime, priority, {**snapshot, 'jobs': [{'job_id': '1995245'}]}, stripes=4)
    assert not output.exists()
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    first = stage_inputs(root, output, runtime, priority, snapshot, stripes=4)
    assert stage_inputs(root, output, runtime, priority, snapshot, stripes=4) == first
    assert verify_inputs(output, stripes=4)['verified_completed'] == 1
    assert all('llama' not in str(p) for p in output.rglob('*'))
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    binding = json.loads((output / MANIFEST).read_bytes())['inputs']
    relocated = replace(qwen, prompts_jsonl=output / binding['SEARCH_AGENTIC_PROMPTS_JSONL'],
        selection_records_jsonl=output / binding['SEARCH_AGENTIC_SELECTION_RECORDS_JSONL'],
        cross_encoder_snapshot=tmp_path / 'different-cache', search_snapshots={
            'duckduckgo': output / binding['SEARCH_AGENTIC_DDG_SNAPSHOT'],
            'searxng': output / binding['SEARCH_AGENTIC_SEARXNG_SNAPSHOT']})
    descriptions = [describe_queue(inputs) for inputs in (qwen, relocated)]
    identities = [_generator_claim_identity(d['cells'][0], inputs, d['config'], d['legacy_prompt'],
                    d['target_urls'], method_source_sha256='same-source')
                  for d, inputs in zip(descriptions, (qwen, relocated))]
    assert identities[0] == identities[1]
    (output / binding['SEARCH_AGENTIC_PROMPTS_JSONL']).write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        verify_inputs(output, stripes=4)
