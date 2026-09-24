"""Progress describes real audit work without changing results or stdout."""
import io
import threading

import pytest

from analysis.interpretability.pipeline.agentic_audit_progress import (
    audit_progress,
    audit_stage,
)
from analysis.interpretability.pipeline.agentic_hours import inventory
from analysis.tests.test_prepare_shared_hour_inputs import fixture


def test_inventory_reports_actual_counts_without_changing_results(tmp_path, monkeypatch, capsys):
    root, _, _ = fixture(tmp_path)
    monkeypatch.delenv('GEODML_AUDIT_PROGRESS', raising=False)
    expected = inventory(root, stripes=4)
    assert capsys.readouterr().err == ''
    monkeypatch.setenv('GEODML_AUDIT_PROGRESS', '1')
    assert inventory(root, stripes=4) == expected
    output = capsys.readouterr()
    assert output.out == ''
    assert 'stage=inventory status=started' in output.err
    assert 'status=finished' in output.err
    assert 'tasks_checked=48 verified_completed=0 blocked=0' in output.err


def test_slow_stage_has_heartbeat_and_failure_is_not_success(monkeypatch):
    seen = threading.Event()
    class Stream(io.StringIO):
        def write(self, text):
            if 'status=working' in text:
                seen.set()
            return super().write(text)
    stream = Stream()
    monkeypatch.setenv('GEODML_AUDIT_PROGRESS', '1')
    monkeypatch.setattr('sys.stderr', stream)
    @audit_stage('slow_check', interval=0.01)
    def check():
        audit_progress(tasks_checked=7)
        assert seen.wait(2), 'no heartbeat while audit is blocked'
        raise ValueError('bad reference')
    with pytest.raises(ValueError, match='bad reference'):
        check()
    output = stream.getvalue()
    assert 'status=working' in output and 'tasks_checked=7' in output
    assert 'status=failed' in output and 'status=finished' not in output


def test_upload_reports_confirmed_files_and_keeps_manifest(tmp_path, monkeypatch, capsys):
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
    from analysis.tests.test_agentic_hours import MemoryHub
    monkeypatch.setenv('GEODML_AUDIT_PROGRESS', '1')
    (tmp_path / 'README.md').write_text('hello')
    exchange = Exchange(MemoryHub(), tmp_path / 'journal')
    bundle = exchange.upload(tmp_path, ['README.md'], outcomes={}, metadata={})
    assert exchange.manifest(bundle)['files']['README.md']['bytes'] == 5
    output = capsys.readouterr()
    assert output.out == ''
    assert 'files_total=1 files_finished=1 bytes_finished=5' in output.err
    assert 'phase=publish_manifest' in output.err
