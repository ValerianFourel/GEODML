import hashlib
import json
import sqlite3
import tarfile

import pytest

from analysis.interpretability.pipeline.agentic_generation_tasks import (
    CalibrationPrompt,
    build_cells,
)
from analysis.interpretability.pipeline.inference_claims import (
    ClaimIdentity,
    InferenceClaimStore,
    _digest,
)
from analysis.scripts import collect_agentic_forensic_snapshot as audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'runs'
    root.mkdir()
    prompt = CalibrationPrompt('p1', 'A question?', hashlib.sha256(b'A question?').hexdigest(), 0, 'software')
    population = root / 'population.jsonl'
    population.write_text(json.dumps({'candidate_id': prompt.prompt_id, 'question': prompt.prompt, 'keyword': prompt.keyword}) + '\n')
    selection = put(root / 'selection-manifest.json', {'sources': {'prompts': {'path': str(population), 'rows': 1, 'sha256': audit.sha_file(population)}}})
    cell = build_cells((prompt,))[0]
    trace = {'method_id': cell.method_class.method_id, 'condition': cell.condition.value,
             'search_engine': cell.engine, 'user_prompt_sha256': cell.prompt_sha256, 'events': []}
    trace['trace_sha256'] = _digest(trace)
    result = {'cell_id': cell.cell_id, **cell.core, 'answer': 'Saved answer', 'ranking': [], 'trace_sha256': trace['trace_sha256']}
    bundle = {'trace': trace, 'result': result,
              'diagnostics': {'cell_id': cell.cell_id, **cell.core, 'status': 'complete'},
              'producer': {'git_commit': 'a'*40, 'slurm_job_id': '123', 'source_config_sha256': 'b'*64}}
    identity = ClaimIdentity(cell.cell_id, 'Qwen/Qwen3.8-27B', 'c'*40, 'agentic-generator-shared-v1', 'd'*64)
    with InferenceClaimStore(root / 'failed-job/claims').try_claim(identity) as claim:
        claim.commit(bundle)
    return root, selection, prompt, cell, bundle, identity


def run(source, tmp_path):
    root, selection, *_ = source
    return audit.collect([root], selection, tmp_path / 'snapshot', pack_bytes=2048)


def test_recovers_claim_from_failed_job_without_local_results(source, tmp_path):
    root, *_ = source
    put(root / 'failed-job/run_manifest.json', {'status': 'FAILED', 'completed_count': 0})
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['generator_payload_valid'] == 1
    assert receipt['identity_counts'][0]['distinct_identities'] == 1
    assert receipt['full_population_completion_percent'] is None
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_same_claim_in_two_registries_counts_once(source, tmp_path):
    root, _, _, _, _, identity = source
    fingerprint = _digest(identity.__dict__)
    original = root / 'failed-job/claims' / fingerprint[:2] / (fingerprint + '.json')
    copy = root / 'retry/claims' / fingerprint[:2] / original.name
    copy.parent.mkdir(parents=True)
    copy.write_bytes(original.read_bytes())
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['generator_payload_valid'] == 2
    assert receipt['identity_counts'][0]['distinct_identities'] == 1
    assert receipt['conflicting_identities'] == 0


def test_different_payload_same_identity_is_conflict(source, tmp_path):
    root, _, _, _, bundle, identity = source
    bundle['result']['answer'] = 'A different saved answer'
    with InferenceClaimStore(root / 'retry/claims').try_claim(identity) as claim:
        claim.commit(bundle)
    assert run(source, tmp_path)['conflicting_identities'] == 1


def test_corrupt_and_truncated_artifacts_preserved(source, tmp_path):
    root, *_ = source
    malformed = root / 'bad.json'
    malformed.write_bytes(b'{broken')
    journal = root / 'outcomes.jsonl'
    journal.write_bytes(b'{"ok": true}\n{"truncated":')
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['unverified_or_invalid'] == 2
    archived = {}
    for pack in (tmp_path / 'snapshot').glob('*.tar'):
        with tarfile.open(pack) as stream:
            for entry in stream:
                archived[entry.name] = stream.extractfile(entry).read()
    assert archived[audit.sha_file(malformed)] == malformed.read_bytes()
    assert archived[audit.sha_file(journal)] == journal.read_bytes()


def test_legacy_result_inspected_without_successful_manifest(source, tmp_path):
    root, _, _, cell, bundle, _ = source
    output = root / 'old-failed-job/models/qwen38/shard-0'
    trace = put(output / 'traces' / f'{cell.cell_id}.json', bundle['trace'])
    put(output / 'results' / trace.name, {**bundle['result'], 'trace': str(trace.resolve())})
    config = {'model_id': 'Qwen/Qwen3.8-27B', 'model_revision': 'c'*40}
    config['config_sha256'] = _digest(config)
    put(output / 'config.json', config)
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['legacy_payload_valid'] == 1
    assert receipt['full_population_completion_percent'] is None


def test_bad_population_hash_stops_before_snapshot(source, tmp_path):
    root, _selection, *_ = source
    (root / 'population.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        run(source, tmp_path)
    assert not (tmp_path / 'snapshot').exists()


def test_rejects_existing_or_overlapping_output(source, tmp_path):
    root, selection, *_ = source
    with pytest.raises(ValueError, match='overlap'):
        audit.collect([root], selection, root / 'snapshot')
    (tmp_path / 'snapshot').mkdir()
    with pytest.raises(FileExistsError):
        run(source, tmp_path)


def test_symlink_and_locks_are_not_archived(source, tmp_path):
    root, *_ = source
    (root / 'outside-link').symlink_to('/etc/passwd')
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['excluded_file'] >= 2
    db = sqlite3.connect(tmp_path / 'snapshot/inventory.sqlite')
    assert db.execute("SELECT count(*) FROM artifacts WHERE path LIKE '%.lock' AND pack IS NOT NULL").fetchone()[0] == 0
    db.close()


def test_external_references_are_reported_not_silently_ignored(source, tmp_path):
    root, *_ = source
    put(root / 'other/run_manifest.json', {'judge': {'claim_root': '/external/claims'}})
    receipt = run(source, tmp_path)
    assert receipt['external_reference_count'] == 1
    assert receipt['archive_errors'] == []


def test_corrupted_claim_does_not_enter_outcome_counts(source, tmp_path):
    root, *_ = source
    path = next((root / 'failed-job/claims').glob('*/*.json'))
    value = json.loads(path.read_text())
    value['outcome']['result']['answer'] = 'Corrupt payload'
    put(path, value)
    receipt = run(source, tmp_path)
    assert receipt['identity_counts'] == []
    assert receipt['validation_tiers']['unverified_or_invalid'] == 1


def test_changed_source_during_archive_is_a_finding(source, tmp_path, monkeypatch):
    original = audit.Packs.add
    def changing_add(self, path, digest):
        result = original(self, path, digest)
        if path.name == 'change.txt':
            path.write_text('changed after capture')
        return result
    monkeypatch.setattr(audit.Packs, 'add', changing_add)
    root, *_ = source
    (root / 'change.txt').write_text('original')
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['unreadable_or_changed'] == 1


def test_terminal_failure_is_not_a_completed_outcome(source, tmp_path):
    root, _, _, _cell, bundle, identity = source
    fingerprint = _digest(identity.__dict__)
    original = root / 'failed-job/claims' / fingerprint[:2] / (fingerprint + '.json')
    original.unlink()
    failure = {'trace': bundle['trace'], 'diagnostics': {**bundle['diagnostics'], 'status': 'failed'},
               'producer': bundle['producer'], 'error': 'exhausted retries', 'attempts': 3}
    from analysis.scripts.run_agentic_search_integration_smoke import (
        FAILED_CELL_RETRY_PASSES,
    )
    failure['attempts'] = FAILED_CELL_RETRY_PASSES + 1
    with InferenceClaimStore(root / 'failed-job/claims').try_claim(identity) as claim:
        claim.fail(failure)
    receipt = run(source, tmp_path)
    assert receipt['validation_tiers']['generator_failure_valid'] == 1
    assert receipt['identity_counts'][0]['state'] == 'failed'


def test_nested_duplicate_roots_are_traversed_once(source, tmp_path):
    root, selection, *_ = source
    receipt = audit.collect([root, root / 'failed-job', root], selection, tmp_path / 'snapshot')
    assert receipt['validation_tiers']['generator_payload_valid'] == 1
    assert receipt['roots'] == [str(root)]


def test_resume_reuses_closed_packs_and_preserves_old_snapshot(source, tmp_path):
    root, selection, *_ = source
    old = tmp_path / 'old'
    audit.collect([root], selection, old, pack_bytes=128)
    state = json.loads((old / 'snapshot.json').read_text())
    state['status'] = 'collecting'
    put(old / 'snapshot.json', state)
    before = {p.name: p.read_bytes() for p in old.iterdir() if p.is_file()}
    last = max(old.glob('artifacts-*.tar')).name
    new = tmp_path / 'new'
    receipt = audit.collect([root], selection, new, pack_bytes=128, resume_from=old)
    assert receipt['reused_files'] > 0
    assert receipt['new_files'] > 0
    assert receipt['archive_errors'] == []
    assert receipt['identity_counts'][0]['distinct_identities'] == 1
    assert {p.name: p.read_bytes() for p in old.iterdir() if p.is_file()} == before
    assert (new / last).read_bytes()  # New pack may reuse dropped number, never old bytes.


def test_resume_changed_source_is_recaptured(source, tmp_path):
    root, selection, *_ = source
    note = root / 'note.txt'
    note.write_text('before')
    old = tmp_path / 'old'
    audit.collect([root], selection, old)
    note.write_text('after')
    new = tmp_path / 'new'
    receipt = audit.collect([root], selection, new, resume_from=old)
    assert receipt['new_files'] == 1
    db = sqlite3.connect(new / 'inventory.sqlite')
    assert db.execute('SELECT sha256 FROM artifacts WHERE path=?', (str(note),)).fetchone()[0] == audit.sha_file(note)
    db.close()


def test_resume_missing_source_does_not_keep_validated_outcome(source, tmp_path):
    root, selection, *_ = source
    old = tmp_path / 'old'
    audit.collect([root], selection, old)
    next((root / 'failed-job/claims').glob('*/*.json')).unlink()
    receipt = audit.collect([root], selection, tmp_path / 'new', resume_from=old)
    assert receipt['identity_counts'] == []
    assert receipt['validation_tiers']['source_missing_on_resume'] == 1


def test_resume_truncated_last_pack_recaptures_its_sources(source, tmp_path):
    root, selection, *_ = source
    old = tmp_path / 'old'
    audit.collect([root], selection, old, pack_bytes=128)
    last = max(old.glob('artifacts-*.tar'))
    last.write_bytes(last.read_bytes()[:600])
    state = json.loads((old / 'snapshot.json').read_text())
    state['status'] = 'collecting'
    put(old / 'snapshot.json', state)
    receipt = audit.collect([root], selection, tmp_path / 'new', pack_bytes=128, resume_from=old)
    assert receipt['archive_errors'] == []
    assert receipt['identity_counts'][0]['distinct_identities'] == 1


def test_resume_source_replaced_by_symlink_drops_outcome(source, tmp_path):
    root, selection, *_ = source
    old = tmp_path / 'old'
    audit.collect([root], selection, old)
    claim = next((root / 'failed-job/claims').glob('*/*.json'))
    target = tmp_path / 'moved-claim.json'
    claim.rename(target)
    claim.symlink_to(target)
    receipt = audit.collect([root], selection, tmp_path / 'new', resume_from=old)
    assert receipt['identity_counts'] == []
    assert receipt['validation_tiers']['excluded_file'] >= 1
