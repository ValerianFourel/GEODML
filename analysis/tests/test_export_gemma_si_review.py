"""Diagnostic export preserves evidence and verifies its private-Hub receipt."""
import hashlib
import io
import json
import zipfile

import pytest

from analysis.scripts import export_gemma_si_review as export


def test_export_preserves_raw_judgments_reports_missing_pass_and_excludes_credentials(tmp_path):
    run = tmp_path / "gemma-run"
    trial = run / "attempts/job5171481/trial/pass1"
    trial.mkdir(parents=True)
    raw = b'{"judge_task_id":"one","ok":true,"duration_seconds":2,"raw_output":"raw reply"}\n'
    (trial / "results.jsonl").write_bytes(raw)
    (trial / "summary.json").write_text('{"seconds":4,"status":"passed"}')
    (run / "token").write_text("must not upload")
    (run / "environment.sh").write_text("must not upload")
    files, report, jobs = export.collect([run])
    assert jobs == ["5171481"]
    first = report["attempts"][0]["passes"]["pass1"]
    assert first["successful_rows"] == 1 and first["judgments_per_second"] == .25
    assert first["latency_seconds_mean"] == 2 and first["failed_rows"] == []
    assert report["attempts"][0]["passes"]["pass2"]["judgments_per_second"] is None
    assert "gemma-run/attempts/job5171481/trial/pass2/results.jsonl" in report["missing"]
    packed = export.package(files, report, {"stdout": "5171481|TIMEOUT|0:0"})
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        assert archive.read("gemma-run/attempts/job5171481/trial/pass1/results.jsonl") == raw
        assert not any(n.endswith(("token", "environment.sh")) for n in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        for name, entry in manifest["files"].items():
            content = archive.read(name)
            assert entry == {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    assert export.package(files, report, {"stdout": "5171481|TIMEOUT|0:0"}) == packed
    (run / "inputs.json").symlink_to(run / "token")
    with pytest.raises(ValueError, match="redirected"):
        export.collect([run])


@pytest.mark.parametrize("corrupt", [False, True])
def test_upload_writes_only_archive_and_requires_exact_remote_verification(monkeypatch, corrupt):
    objects, writes = {}, []
    class Store:
        def __init__(self, repo_id):
            assert repo_id == "owner/private"
        def head(self):
            return "parent"
        def exists(self, name, revision):
            return name in objects
        def commit(self, revision, files, message):
            assert revision == "parent"
            objects.update(files)
            writes.append(files)
            return "saved-commit"
        def read(self, name, revision):
            assert revision == "saved-commit"
            return b"corrupt" if corrupt else objects[name]
    monkeypatch.setattr(export, "HubStore", Store)
    if corrupt:
        with pytest.raises(ValueError, match="remote archive verification failed"):
            export.publish(b"archive", "owner/private")
    else:
        receipt = export.publish(b"archive", "owner/private")
        assert receipt["verified"] and receipt["revision"] == "saved-commit"
        assert receipt["sha256"] == hashlib.sha256(b"archive").hexdigest()
    assert len(writes) == 1
    assert list(objects) == ["reviews/gemma-si-v3/exports/" + hashlib.sha256(b"archive").hexdigest() + ".zip"]
