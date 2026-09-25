import json

import pytest

from analysis.scripts import manage_agentic_hours as helper
from analysis.scripts.reconcile_agentic_dataset import _owners


@pytest.mark.parametrize('already_named', [True, False])
def test_terminal_execution_receipt_produces_one_shared_owner(tmp_path, monkeypatch, already_named):
    attempt = {'request': {'attempt_dir': str(tmp_path), 'since': '2026-09-01'},
               'cluster': 'jupiter', 'attempt_id': 'wave-1', 'writer_id': 'jupiter-wave-1'}
    (tmp_path / 'execution.json').write_text(json.dumps({'job_id': '2008531'}))
    rows = [{'owner_id': 'job2008531-worker0', 'job_id': '2008531', 'state': 'COMPLETED',
             'cluster': 'jupiter', 'attempt_id': 'wave-1'}]
    if already_named:
        rows.append({**rows[0], 'owner_id': attempt['writer_id']})
    monkeypatch.setattr(helper, 'capture', lambda **kwargs: {'complete': True, 'owners': rows, 'jobs': []})
    snapshot = helper.scheduler(attempt)
    owners = _owners(snapshot)
    assert owners[attempt['writer_id']]['job_id'] == '2008531'
    assert len([row for row in snapshot['owners'] if row['owner_id'] == attempt['writer_id']]) == 1
    assert owners['job2008531-worker0']['state'] == 'COMPLETED'


def test_existing_writer_cannot_be_reassigned_to_another_job(tmp_path, monkeypatch):
    attempt = {'request': {'attempt_dir': str(tmp_path), 'since': '2026-09-01'},
               'cluster': 'jupiter', 'attempt_id': 'wave-1', 'writer_id': 'jupiter-wave-1'}
    (tmp_path / 'execution.json').write_text(json.dumps({'job_id': '2008531'}))
    rows = [{'owner_id': 'jupiter-wave-1', 'job_id': '999', 'state': 'RUNNING', 'cluster': 'jupiter'},
            {'owner_id': 'job2008531-worker0', 'job_id': '2008531', 'state': 'COMPLETED', 'cluster': 'jupiter'}]
    monkeypatch.setattr(helper, 'capture', lambda **kwargs: {'complete': True, 'owners': rows, 'jobs': []})
    with pytest.raises(ValueError, match='conflicting scheduler owner'):
        helper.scheduler(attempt)
