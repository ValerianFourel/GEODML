"""The JUPITER archiver keeps secrets, restricted-local data and oversize files off the Hub and restores exactly."""

import gzip
import hashlib
import json
import os
import shutil
import subprocess

import pytest

from analysis.scripts import archive_jupiter_to_hub as arch


class FakeHub:
    """Minimal HfApi stand-in that stores uploads in a local folder."""

    def __init__(self, root, fail_first_commit=False):
        self.root, self.fail = root, fail_first_commit
        self.commits = 0

    def create_repo(self, *args, **kwargs):
        pass

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, commit_message):
        self.commits += 1
        if self.fail and self.commits == 1:
            raise RuntimeError("503 Service Unavailable")
        shutil.copytree(folder_path, self.root / path_in_repo, dirs_exist_ok=True)

    def upload_file(self, repo_id, repo_type, path_or_fileobj, path_in_repo, commit_message):
        shutil.copy(path_or_fileobj, self.root / path_in_repo)

    def get_paths_info(self, repo, paths, repo_type):
        out = []
        for path in paths:
            target = self.root / path
            if target.exists():
                data = target.read_bytes()
                out.append(type("Info", (), {"path": path, "size": len(data),
                                             "lfs": {"sha256": hashlib.sha256(data).hexdigest()}})())
        return out


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "fs"
    (root / "geodml/audits/a").mkdir(parents=True)
    (root / "geodml/runs/r/round-01/projections").mkdir(parents=True)
    (root / "geodml/compile-cache").mkdir(parents=True)
    (root / "geodml/jupiter-llama-x-1").mkdir(parents=True)
    for index in range(12):
        (root / f"geodml/audits/a/f{index}.json").write_bytes(os.urandom(3000 + index))
    (root / "geodml/audits/a/env.sh").write_text("export HF_TOKEN=hf_" + "A" * 34 + "\n")
    (root / "geodml/audits/a/token").write_text("x")
    (root / "geodml/audits/a/w.safetensors").write_bytes(b"0" * 10)
    (root / "geodml/audits/a/big.tar").write_bytes(b"1" * 5000)
    (root / "geodml/audits/a/src.json").write_text('{"source": "allenai-wildchat-1m"}')
    (root / "geodml/runs/r/round-01/projections/question_embeddings.restricted-local.npz").write_bytes(b"e")
    (root / "geodml/compile-cache/k.bin").write_bytes(b"c")
    (root / "geodml/jupiter-llama-x-1/part.json").write_text("{}")
    os.symlink("f1.json", root / "geodml/audits/a/link")
    return root


def plan(tmp_path, tree, *extra):
    arch.main(["plan", "--work", str(tmp_path / "work"), "--root", f"fs={tree}",
               "--exclude", "fs:geodml/compile-cache", "--exclude-glob", "fs:geodml/jupiter-llama-*",
               "--max-file-mb", "0.004", "--unit-gb", "0.00002", "--part-gb", "0.00001", *extra])
    return json.loads((tmp_path / "work/plan/summary.json").read_text())


def planned_paths(tmp_path):
    with gzip.open(tmp_path / "work/plan/fs/files.tsv.gz", "rt") as handle:
        return {line.split("\t")[1] for line in list(handle)[1:]}


def test_plan_leaves_out_secrets_restricted_local_oversize_and_excluded_paths(tmp_path, tree):
    summary = plan(tmp_path, tree)
    paths = planned_paths(tmp_path)
    root = summary["roots"]["fs"]
    assert "geodml/audits/a/f0.json" in paths and "geodml/audits/a/link" in paths
    for absent in ("geodml/audits/a/env.sh", "geodml/audits/a/token", "geodml/audits/a/w.safetensors",
                   "geodml/audits/a/big.tar", "geodml/compile-cache/k.bin", "geodml/jupiter-llama-x-1/part.json"):
        assert absent not in paths
    assert not any("restricted-local" in path for path in paths)
    assert sorted(root["secret_files_left_out"]) == ["geodml/audits/a/env.sh", "geodml/audits/a/token"]
    assert root["restricted_files"] == ["geodml/audits/a/src.json"]
    assert root["filtered"]["over size limit"] == [1, 5000]
    assert root["filtered"]["glob"][0] == 1 and root["filtered"]["skipped name"][0] == 1
    with pytest.raises(SystemExit, match="--replan"):
        plan(tmp_path, tree)


def test_run_refuses_restricted_mentions_without_explicit_acceptance(tmp_path, tree, monkeypatch):
    plan(tmp_path, tree)
    monkeypatch.setattr(arch, "make_api", lambda: pytest.fail("no Hub call before acceptance"))
    with pytest.raises(SystemExit, match="accept-restricted"):
        arch.main(["run", "--work", str(tmp_path / "work")])


def test_run_retries_verifies_resumes_and_restores_byte_identical(tmp_path, tree, monkeypatch):
    plan(tmp_path, tree)
    hub_root = tmp_path / "hub"
    hub_root.mkdir()
    hub = FakeHub(hub_root, fail_first_commit=True)
    monkeypatch.setattr(arch, "make_api", lambda: hub)
    monkeypatch.setattr(arch.time, "sleep", lambda seconds: None)
    arch.main(["run", "--work", str(tmp_path / "work"), "--accept-restricted"])
    units = sorted(p for p in (hub_root / "fs").iterdir())
    assert len(units) > 1 and (tmp_path / "work/COMPLETE.json").exists()
    assert not any((tmp_path / "work/stage/fs").iterdir())
    commits = hub.commits
    arch.main(["run", "--work", str(tmp_path / "work"), "--accept-restricted"])
    assert hub.commits == commits + 1  # only the index is committed again; verified units are skipped

    restore = tmp_path / "restore"
    restore.mkdir()
    for unit in units:
        parts = sorted(unit.glob("part-*.tar.gz"))
        subprocess.run(f"cat {' '.join(map(str, parts))} | tar -xzf -", shell=True, cwd=restore, check=True)
    for path in planned_paths(tmp_path):
        original, copy = tree / path, restore / path
        if original.is_symlink():
            assert os.readlink(copy) == os.readlink(original)
        else:
            assert copy.read_bytes() == original.read_bytes()


def test_verify_rejects_size_or_sha_mismatch(tmp_path):
    hub_root = tmp_path / "hub"
    (hub_root / "fs/unit-0001").mkdir(parents=True)
    (hub_root / "fs/unit-0001/part-000.tar.gz").write_bytes(b"abc")
    (hub_root / "fs/unit-0001/manifest.json").write_text("{}")
    hub = FakeHub(hub_root)
    good = {"parts": [{"name": "part-000.tar.gz", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]}
    assert arch.verify(hub, "r", "fs/unit-0001", good)
    assert not arch.verify(hub, "r", "fs/unit-0001", {"parts": [{**good["parts"][0], "bytes": 4}]})
    assert not arch.verify(hub, "r", "fs/unit-0001", {"parts": [{**good["parts"][0], "sha256": "0" * 64}]})
