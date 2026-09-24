"""Public update commands against isolated Hub and cluster fixtures."""
import time
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline import agentic_hour_updates as updates
from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter
from analysis.interpretability.pipeline.agentic_hour_sync import (
    ConflictError,
    Exchange,
    checkpoint_files,
)
from analysis.interpretability.pipeline.agentic_hours import (
    build_plan,
    canonical,
    claim_hours,
    empty_registry,
    finish_hours,
    inventory,
)
from analysis.scripts import manage_agentic_hours as cli
from analysis.scripts.publish_agentic_dataset import build_manifest
from analysis.tests.test_agentic_hours import MemoryHub, calibration, complete, data


def setup(tmp_path):
    root = data(tmp_path / 'dataset')
    writer = FinalDatasetWriter(root, writer_id='prompt-fixture')
    for keyword, axis_bin in [('alpha', 0), ('beta', 1)]:
        writer.append('prompts', {'prompt_id': 'prompt-' + keyword, 'axis_bin': axis_bin}, transaction_id=keyword)
    writer.seal()
    hub = MemoryHub()
    exchange = Exchange(hub, tmp_path / 'journal')
    bundle = exchange.upload(root, list(build_manifest(root)['files']), outcomes={}, metadata={'kind': 'frozen-inputs'})
    (tmp_path / 'calibration.json').write_bytes(canonical(calibration()))
    (tmp_path / 'quota.json').write_bytes(canonical({'cluster': 'horeka', 'captured_at_epoch': int(time.time()),
        'within_limits': True, 'workspace': str(tmp_path), 'work_headroom_bytes': 10**12, 'work_headroom_files': 10**6}))
    site = {'cluster': 'jupiter', 'dataset_root': str(root), 'input_bundle': bundle,
            'calibration': str(tmp_path / 'calibration.json'), 'source_commit': 'b' * 40,
            'plan_dir': str(tmp_path / 'plans'), 'journal': str(tmp_path / 'journal'),
            'quota_evidence': str(tmp_path / 'quota.json'), 'stripes': 4, 'attempts': []}
    first = updates.replan(exchange, site, empty_registry(), {})
    return root, hub, exchange, site, first


def test_replan_unchanged_keeps_ids_and_does_not_commit(tmp_path):
    _, hub, exchange, site, first = setup(tmp_path)
    revision, state = exchange.snapshot()
    plans = {first['plan_id']: updates.load_plan(exchange, first['plan_id'], revision)}
    assert updates.replan(exchange, site, state, plans) == {'status': 'unchanged', 'plan_id': first['plan_id']}
    assert hub.head() == revision


def test_select_explicit_model_both_qwen_clusters_and_keyword_direction(tmp_path):
    _, _, exchange, _, _ = setup(tmp_path)
    _, state = exchange.snapshot()
    batch = updates.select_hours(state, model='qwen38', count=2)
    interactive = updates.select_hours(state, model='qwen38', cluster='jupiter', mode='interactive', count=2)
    assert batch['cluster'] == 'horeka'
    assert interactive['hour_ids'] == batch['hour_ids'][::-1]
    assert batch['reserved'] is False
    assert updates.select_hours(state, model='qwen38', first=batch['hour_ids'][1])['hour_ids'] == batch['hour_ids'][1:]
    with pytest.raises(ValueError, match='not enabled'):
        updates.select_hours(state, model='llama4', cluster='horeka')
    with pytest.raises(SystemExit):
        cli.parser().parse_args(['select', '--site', 'site.json'])


def test_progress_bins_are_not_double_counted_and_snapshot_races_fail(tmp_path):
    root, _, exchange, _, first = setup(tmp_path)
    revision, state = exchange.snapshot()
    report = updates.progress(root, state, revision=revision, stripes=4)
    assert report['task_count'] == 4
    assert [(r['keyword_id'], r['axis_bin'], r['cells']) for r in report['groups']] == [
        ('alpha', 0, {'eligible': 2}), ('beta', 1, {'eligible': 2})]
    updates.publish_progress(exchange, report)
    unchanged = exchange.store.head()
    updates.publish_progress(exchange, report)
    assert exchange.store.head() == unchanged
    plan = updates.load_plan(exchange, first['plan_id'], unchanged)
    exchange.transact('claim', {}, lambda s: claim_hours(s, hour_ids=[plan['packages'][0]['hour_id']],
        cluster='horeka', attempt_id='hk', supported_models=['qwen38']))
    with pytest.raises(ConflictError, match='registry changed'):
        updates.publish_progress(exchange, report)


def test_status_reads_shared_document_without_local_dataset_or_audit(tmp_path, monkeypatch):
    root, _, exchange, site, _ = setup(tmp_path)
    revision, state = exchange.snapshot()
    report = updates.progress(root, state, revision=revision, stripes=4)
    updates.publish_progress(exchange, report)
    site['dataset_root'] = str(tmp_path / 'not-downloaded')
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    def forbidden(*args, **kwargs):
        pytest.fail('status must not audit or download plans')
    monkeypatch.setattr(updates, 'inventory', forbidden)
    monkeypatch.setattr(updates, 'load_plan', forbidden)
    result = updates.run(exchange, SimpleNamespace(command='status', site=path))
    assert result['progress']['cells'] == {'eligible': 4}
    assert result['status'] == 'current'
    assert result['local_audit_performed'] is False
    exchange.transact('change-after-report', {}, lambda s: {**s, 'admission': {'jupiter': {}}})
    result = updates.run(exchange, SimpleNamespace(command='status', site=path))
    assert result['status'] == 'stale'
    assert result['progress']['cells'] == {'eligible': 4}


def test_missing_shared_document_does_not_trigger_an_implicit_audit(tmp_path, monkeypatch):
    _, _, exchange, site, _ = setup(tmp_path)
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    monkeypatch.setattr(updates, 'inventory', lambda *a, **k: pytest.fail('unexpected audit'))
    result = updates.run(exchange, SimpleNamespace(command='status', site=path))
    assert result['status'] == 'progress_not_published'
    assert result['local_audit_performed'] is False


def test_normal_update_reads_inventory_once(tmp_path, monkeypatch):
    _, _, exchange, site, _ = setup(tmp_path)
    site['cluster'] = 'horeka'
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    original, calls = updates.inventory, []
    def once(*args, **kwargs):
        calls.append(kwargs)
        assert len(calls) == 1, 'update repeated the population inventory'
        return original(*args, **kwargs)
    monkeypatch.setattr(updates, 'inventory', once)
    result = updates.run(exchange, SimpleNamespace(command='update', scope='results', site=path))
    assert result['progress']['task_count'] == 4
    assert calls[0]['reuse_verified'] is True


def test_pull_does_not_repeat_inventory_after_verified_download(tmp_path, monkeypatch):
    root, _, exchange, site, _ = setup(tmp_path)
    revision, state = exchange.snapshot()
    updates.publish_progress(exchange, updates.progress(root, state, revision=revision, stripes=4))
    site['cluster'] = 'horeka'
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    monkeypatch.setattr(updates, 'inventory', lambda *a, **k: pytest.fail('redundant post-download audit'))
    result = updates.run(exchange, SimpleNamespace(command='pull', model='qwen38', site=path))
    assert result['progress']['task_count'] == 4
    assert result['scope'] == 'published_shared_state'


def test_local_audit_is_explicit_and_still_available(tmp_path, monkeypatch):
    _, _, exchange, site, _ = setup(tmp_path)
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    original, calls = updates.inventory, []
    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(updates, 'inventory', record)
    args = cli.parser().parse_args(['status', '--site', str(path), '--audit-local'])
    result = updates.run(exchange, args)
    assert result['progress']['task_count'] == 4
    assert calls == [True]


def test_replanning_reuses_the_updates_inventory_snapshot(tmp_path, monkeypatch):
    _, _, exchange, site, first = setup(tmp_path)
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    revision, state = exchange.snapshot()
    plans = {first['plan_id']: updates.load_plan(exchange, first['plan_id'], revision)}
    monkeypatch.setattr(updates, 'pull', lambda *a, **k: (state, plans, revision))
    original, calls = updates.inventory, []
    def once(*args, **kwargs):
        calls.append(True)
        assert len(calls) == 1, 'replanning repeated the already-read inventory'
        return original(*args, **kwargs)
    monkeypatch.setattr(updates, 'inventory', once)
    result = updates.run(exchange, SimpleNamespace(command='update', scope='both', site=path))
    assert result['plan']['status'] == 'unchanged'
    assert result['progress']['task_count'] == 4
    assert calls == [True]


def test_results_update_keeps_plan_and_no_new_payload_transfer(tmp_path, monkeypatch):
    _, hub, exchange, site, first = setup(tmp_path)
    site['cluster'] = 'horeka'
    path = tmp_path / 'site.json'
    path.write_bytes(canonical(site))
    monkeypatch.setattr(cli, 'HubStore', lambda _: hub)
    assert cli.main(['update', '--site', str(path)]) == 0
    after_first = hub.head()
    assert cli.main(['update', '--site', str(path)]) == 0
    assert hub.head() == after_first
    assert exchange.snapshot()[1]['current_plan'] == first['plan_id']
    with pytest.raises(ValueError, match='only JUPITER'):
        cli.main(['update', '--site', str(path), '--scope', 'both'])


def test_missing_calibration_deferred_without_inventing_hours(tmp_path):
    root = data(tmp_path / 'dataset')
    tasks, done, blocked = inventory(root, stripes=4)
    value = build_plan(tasks=tasks, calibration={}, registry=empty_registry(), contract={},
                       completed=done, blocked=blocked, source_commit='b'*40,
                       input_bundle='bundle', allow_uncalibrated=True)
    assert value['packages'] == []
    assert set(value['deferred'].values()) == {'awaiting_calibration'}
    assert len(value['deferred']) == 4


def test_configuration_specific_timing_keeps_packages_homogeneous(tmp_path):
    tasks, done, blocked = inventory(data(tmp_path / 'dataset'), stripes=4)
    tasks[-1]['configuration_sha256'] = 'second-config'
    reference = calibration()['qwen38']
    value = build_plan(tasks=tasks, calibration={'qwen38': {'configurations': {
        'config-qwen': reference, 'second-config': {**reference, 'scientific_config_sha256': 'second-config'}}}},
        registry=empty_registry(), contract={}, completed=done, blocked=blocked,
        source_commit='b'*40, input_bundle='bundle')
    assert len(value['packages']) == 3
    for package in value['packages']:
        assert {value['tasks'][fp]['configuration_sha256'] for fp in package['task_fingerprints']} == {package['configuration_sha256']}


def test_replan_preserves_owned_hours_and_completed_cells(tmp_path):
    root, _, exchange, site, first = setup(tmp_path)
    revision, state = exchange.snapshot()
    plan = updates.load_plan(exchange, first['plan_id'], revision)
    owned = plan['packages'][0]['hour_id']
    exchange.transact('claim', {}, lambda s: claim_hours(s, hour_ids=[owned], cluster='horeka',
                                                      attempt_id='hk', supported_models=['qwen38']))
    tasks, _, _ = inventory(root, stripes=4)
    complete(root, tasks[-1], 'saved')
    _, state = exchange.snapshot()
    with pytest.raises(ValueError, match='await publication'):
        updates.replan(exchange, site, state, {first['plan_id']: plan})
    second = plan['packages'][1]['hour_id']
    exchange.transact('second-claim', {}, lambda s: claim_hours(s, hour_ids=[second], cluster='jupiter',
                                                               attempt_id='jp', supported_models=['qwen38']))
    names, outcomes = checkpoint_files(root, {t['fingerprint']: t for t in tasks}, stripes=4)
    bundle = exchange.upload(root, names, outcomes=outcomes, metadata={})
    owner = exchange.snapshot()[1]['hours'][second]['owner']
    exchange.transact('finished', {}, lambda s: finish_hours(s, owners={second: owner}, checkpoint=bundle,
                                                            outcomes=outcomes, terminal=True))
    _, state = exchange.snapshot()
    result = updates.replan(exchange, site, state, {first['plan_id']: plan})
    new_state = exchange.snapshot()[1]
    assert new_state['hours'][owned] == state['hours'][owned]
    new_plan = updates.load_plan(exchange, result['plan_id'], exchange.store.head())
    assert tasks[-1]['fingerprint'] not in new_plan['tasks']
    assert len(new_plan['tasks']) == 1


def test_partial_horeka_results_are_retrieved_before_jupiter_continuation(tmp_path):
    root, hub, chief, site, first = setup(tmp_path)
    worker = Exchange(hub, tmp_path / 'horeka-journal')
    hk_site = {**site, 'cluster': 'horeka', 'dataset_root': str(tmp_path / 'horeka')}
    state, plans, _ = updates.pull(worker, hk_site, model='qwen38')
    hour_id = plans[first['plan_id']]['packages'][0]['hour_id']
    worker.transact('hk-claim', {}, lambda s: claim_hours(s, hour_ids=[hour_id], cluster='horeka',
                                                        attempt_id='hk-run', supported_models=['qwen38']))
    tasks, _, _ = inventory(tmp_path / 'horeka', stripes=4)
    task = next(t for t in tasks if t['fingerprint'] in state['hours'][hour_id]['task_fingerprints'])
    complete(tmp_path / 'horeka', task, 'hk-output')
    names, outcomes = checkpoint_files(tmp_path / 'horeka', {t['fingerprint']: t for t in tasks}, stripes=4)
    bundle = worker.upload(tmp_path / 'horeka', names, outcomes=outcomes, metadata={})
    owner = worker.snapshot()[1]['hours'][hour_id]['owner']
    worker.transact('hk-terminal', {}, lambda s: finish_hours(s, owners={hour_id: owner}, checkpoint=bundle,
                                                           outcomes=outcomes, terminal=True))
    # Use the HoreKa quota fixture for this local retrieval; ownership remains JUPITER below.
    updates.pull(chief, {**site, 'cluster': 'horeka'}, model='qwen38')
    assert inventory(root, stripes=4)[1] == {task['fingerprint']}
    chief.transact('jp-continuation', {}, lambda s: claim_hours(s, hour_ids=[hour_id], cluster='jupiter',
                                                              attempt_id='jp-run', supported_models=['qwen38']))
    hour = chief.snapshot()[1]['hours'][hour_id]
    assert hour['owner']['cluster'] == 'jupiter'
    assert set(hour['task_fingerprints']) - set(hour['completed']) == set(state['hours'][hour_id]['task_fingerprints']) - {task['fingerprint']}


def test_publish_lost_response_retries_same_saved_plan(tmp_path):
    _, hub, exchange, site, first = setup(tmp_path)
    revision, state = exchange.snapshot()
    plans = {first['plan_id']: updates.load_plan(exchange, first['plan_id'], revision)}
    changed = calibration()
    changed['qwen38']['seconds_per_task'] = 100
    from pathlib import Path
    Path(site['calibration']).write_bytes(canonical(changed))
    hub.lose_response = True
    with pytest.raises(ConnectionError):
        updates.replan(exchange, site, state, plans)
    # Same saved proposal/operation resolves the lost reply without a second plan.
    count = len(hub.versions)
    result = updates.replan(exchange, site, state, plans)
    assert result['status'] == 'published'
    assert len(hub.versions) == count


def test_selection_does_not_mix_scientific_configurations(tmp_path):
    _, _, exchange, _, _ = setup(tmp_path)
    _, state = exchange.snapshot()
    hours = list(state['hours'].values())
    hours[1]['configuration_sha256'] = 'other'
    assert updates.select_hours(state, model='qwen38', count=2)['hour_ids'] == [hours[0]['hour_id']]
    assert updates.select_hours(state, model='qwen38', count=2, configuration='other')['hour_ids'] == [hours[1]['hour_id']]
