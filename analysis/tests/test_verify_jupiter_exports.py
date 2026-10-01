"""Export checks refuse vacuous success and separate inventory from content proof."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from analysis.scripts import verify_jupiter_exports as verify


class Hub:
    def __init__(self, files, *, lfs=True):
        self.files, self.lfs = files, lfs

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(sha="frozen-revision", private=True)

    def get_paths_info(self, repo, paths, *, repo_type, revision):
        assert revision == "frozen-revision"
        return [SimpleNamespace(path=p, size=len(self.files[p]),
                lfs={"sha256": hashlib.sha256(self.files[p]).hexdigest()} if self.lfs else None,
                blob_id=hashlib.sha1(f"blob {len(self.files[p])}\0".encode() + self.files[p]).hexdigest())
                for p in paths if p in self.files]

    def list_repo_tree(self, repo, *, repo_type, revision, recursive):
        return self.get_paths_info(repo, self.files, repo_type=repo_type, revision=revision)


@pytest.mark.parametrize("case", ["match", "missing", "same_size_corruption", "no_hash"])
def test_manifest_comparison_distinguishes_missing_corrupt_and_unverified(case):
    files = {} if case == "missing" else {"a": b"bad" if case == "same_size_corruption" else b"yes"}
    result = verify.compare(Hub(files, lfs=case != "no_hash"), "repo", "frozen-revision",
                            {"a": {"bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}})
    assert result["status"] == ("pass" if case == "match" else "attention")
    assert result[{"match": "mismatch", "missing": "missing", "same_size_corruption": "mismatch",
                   "no_hash": "hash_unverified"}[case]] == ([] if case == "match" else ["a"])


def test_plain_git_blob_verifies_against_original_manifest_and_local_content(tmp_path):
    path = tmp_path / "a"
    path.write_bytes(b"yes")
    entry = {"a": {"bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest(), "local": str(path)}}
    assert verify.compare(Hub({"a": b"yes"}, lfs=False), "r", "frozen-revision", entry)["status"] == "pass"
    path.write_bytes(b"bad")
    assert verify.compare(Hub({"a": b"bad"}, lfs=False), "r", "frozen-revision", entry)["mismatch"] == ["a"]


def test_empty_geoaxis_manifest_is_not_complete(tmp_path):
    staging = tmp_path / "audits/leave-jupiter/staging"
    staging.mkdir(parents=True)
    (staging / "MANIFEST.tsv").write_text("path\tbytes\tsha256\tsource\n")
    with pytest.raises(ValueError, match="empty export"):
        verify.geoaxis(Hub({}), tmp_path, "r", "staging")


@pytest.mark.parametrize("full", [False, True])
def test_activation_inventory_includes_large_scope_and_never_claims_hash_verification(tmp_path, full):
    (tmp_path / "t7_chunks_full").mkdir()
    (tmp_path / "t7_chunks_full/a.npz").write_bytes(b"yes")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/internal").write_text("ignored")
    result = verify.activations(Hub({"t7_chunks_full/a.npz": b"yes"} if full else {}), tmp_path)
    assert result["local_files"] == 1
    assert result["status"] == ("inventory_match" if full else "attention")
    assert result["content_hash_verification"] == "not_performed"
    assert result["missing"] == ([] if full else ["t7_chunks_full/a.npz"])


@pytest.mark.parametrize("missing_part", [False, True])
def test_archive_receipt_alone_does_not_prove_remote_parts_exist(tmp_path, monkeypatch, missing_part):
    import huggingface_hub
    work = tmp_path / "audits/leave-jupiter/hf-archive"
    (work / "plan/home").mkdir(parents=True)
    (work / "done/home").mkdir(parents=True)
    plan = {"roots": {"home": {"units": [{"unit": "unit-0001"}]}}}
    (work / "plan/summary.json").write_text(json.dumps(plan))
    (work / "plan/home/files.tsv.gz").write_bytes(b"index")
    manifest = {"parts": [{"name": "part-1.tar.gz", "bytes": 3,
                            "sha256": hashlib.sha256(b"tar").hexdigest()}]}
    (work / "done/home/unit-0001.json").write_text(json.dumps({**manifest, "verified_at": 123}))
    (work / "COMPLETE.json").write_text('{"units":1}')
    remote = {"home/unit-0001/manifest.json": json.dumps(manifest).encode(),
              "home/unit-0001/members.txt.gz": b"names", "home/unit-0001/part-1.tar.gz": b"tar",
              "index/plan-summary.json": json.dumps(plan).encode(), "index/home-files.tsv.gz": b"index"}
    if missing_part:
        del remote["home/unit-0001/part-1.tar.gz"]
    def download(repo, path, *, repo_type, revision, cache_dir):
        assert revision == "frozen-revision"
        target = tmp_path / "downloaded-manifest.json"
        target.write_bytes(remote[path])
        return str(target)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = verify.general_archive(Hub(remote), tmp_path, tmp_path / "cache")
    assert result["status"] == ("attention" if missing_part else "pass")
    assert result["pending_units"] == []
    assert result["missing"] == (["home/unit-0001/part-1.tar.gz"] if missing_part else [])
