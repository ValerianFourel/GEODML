"""Historical archive downloads never create current experiment completions."""
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import prepare_horeka_archive as archive


def test_archive_download_resumes_and_preserves_raw_api_errors(tmp_path):
    files = {'data/serp/input.parquet': b'frozen search',
             'data/runs/html_cache.tar.gz': b'archived html',
             'data/dataforseo/raw/backlinks.json': b'{"status_message":"Access denied"}'}
    api = SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(sha=archive.REVISION, siblings=[
        SimpleNamespace(rfilename=n, size=len(raw), lfs={'sha256': hashlib.sha256(raw).hexdigest()})
        for n, raw in files.items()]))
    manifest = archive.inventory(api)
    quota = {'cluster': 'horeka', 'workspace': str(tmp_path), 'captured_at_epoch': int(time.time()),
             'within_limits': True, 'work_headroom_bytes': 10**12, 'work_headroom_files': 10**6}
    root = tmp_path / 'archive'
    calls = []
    def download(repo, name, **kw):
        calls.append(name)
        target = Path(kw['local_dir']) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
        return target
    result = archive.download_archive(manifest, root, quota, download)
    assert result['scientific_completion_imported'] is False
    assert result['files'] == 3
    archive.download_archive(manifest, root, quota, download)
    assert len(calls) == 3
    assert not (root / 'control').exists()
    assert (root / 'data/dataforseo/raw/backlinks.json').read_bytes() == files['data/dataforseo/raw/backlinks.json']
    (root / 'data/serp/input.parquet').write_bytes(b'bad')
    with pytest.raises(ValueError, match='size mismatch'):
        archive.download_archive(manifest, root, quota, download)
    assert len(calls) == 3
    with pytest.raises(ValueError, match='fresh'):
        archive.download_archive(manifest, root, {**quota, 'captured_at_epoch': 0}, download)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'data/../escape', '.git/config'])
def test_archive_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match='unsafe'):
        archive.safe_name(name)
