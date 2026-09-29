"""The readiness-axis archive keeps the restricted scope by owner decision but never secrets or bulk logs."""

import hashlib

import pytest

from analysis.scripts import archive_geoaxis_readiness_axis as axis
from analysis.tests.test_archive_geoaxis_prompts_26k import FakeHub


@pytest.fixture
def tree(tmp_path):
    subspace = tmp_path / "subspace"
    (subspace / "bundle/restricted-local").mkdir(parents=True)
    (subspace / "bundle/restricted-local/prompts.jsonl").write_text('{"source_name": "allenai-wildchat-1m"}\n')
    (subspace / "maps/qwen").mkdir(parents=True)
    (subspace / "maps/qwen/readiness_axis_exemplars.restricted-local.json").write_text("[]")
    (subspace / "maps/qwen/subspace_manifest.json").write_text("{}")
    (subspace / "embeddings/qwen").mkdir(parents=True)
    (subspace / "embeddings/qwen/part-00000.npz").write_bytes(b"E" * 40)
    (subspace / "embeddings/qwen/.cache").mkdir()
    (subspace / "embeddings/qwen/.cache/x").write_text("cache")
    judges = tmp_path / "judge-queue"
    (judges / "primary-frontier").mkdir(parents=True)
    (judges / "primary-frontier/task-1.json").write_text('{"grade": 4}')
    (judges / "env.sh").write_text("export HF_TOKEN=hf_" + "C" * 34)
    runs = tmp_path / "phase1"
    runs.mkdir()
    (runs / "run_manifest.json").write_text("{}")
    (runs / "job.log").write_text("log")
    (runs / "cache.parquet").write_bytes(b"P" * 10)
    (runs / "big.log").write_bytes(b"L" * 60)
    return tmp_path


def argv(tmp_path, *extra):
    return ["--staging", str(tmp_path / "staging"), "--group", f"subspace={tmp_path / 'subspace'}",
            "--group", f"judges/20k-four-judge-v2={tmp_path / 'judge-queue'}",
            "--manifests-only", f"acquisition/phase1={tmp_path / 'phase1'}", *extra]


def staged(tmp_path):
    root = tmp_path / "staging"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_dry_run_keeps_restricted_scope_drops_secrets_caches_and_bulk(tmp_path, tree, monkeypatch, capsys):
    monkeypatch.setattr(axis, "make_api", lambda: pytest.fail("dry run must not call the Hub"))
    monkeypatch.setattr("analysis.scripts.archive_geoaxis_prompts_26k.MANIFEST_MAX_BYTES", 50)
    assert axis.main(argv(tmp_path)) == 0
    files = staged(tmp_path)
    for kept in ("subspace/bundle/restricted-local/prompts.jsonl",
                 "subspace/maps/qwen/readiness_axis_exemplars.restricted-local.json",
                 "subspace/embeddings/qwen/part-00000.npz", "judges/20k-four-judge-v2/primary-frontier/task-1.json",
                 "acquisition/phase1/run_manifest.json", "acquisition/phase1/job.log", "MANIFEST.tsv", "README.md"):
        assert kept in files
    for dropped in ("judges/20k-four-judge-v2/env.sh", "acquisition/phase1/cache.parquet",
                    "acquisition/phase1/big.log", "subspace/embeddings/qwen/.cache/x"):
        assert dropped not in files
    out = capsys.readouterr().out
    assert "RESTRICTED_LOCAL_INCLUDED 2" in out and "DRY_RUN_OK" in out
    for line in (tmp_path / "staging/MANIFEST.tsv").read_text().splitlines()[1:]:
        rel, size, digest, _ = line.split("\t")
        data = (tmp_path / "staging" / rel).read_bytes()
        assert len(data) == int(size) and hashlib.sha256(data).hexdigest() == digest


def test_apply_uploads_privately_and_verifies(tmp_path, tree, monkeypatch, capsys):
    hub = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(axis, "make_api", lambda: hub)
    monkeypatch.setattr(axis, "ensure_token", lambda: None)
    assert axis.main(argv(tmp_path, "--apply")) == 0
    assert "GEOAXIS_AXIS_COMPLETE" in capsys.readouterr().out
    assert (tmp_path / "hub/subspace/bundle/restricted-local/prompts.jsonl").exists()
    public = FakeHub(tmp_path / "hub2", private=False)
    monkeypatch.setattr(axis, "make_api", lambda: public)
    with pytest.raises(SystemExit, match="not private"):
        axis.main(argv(tmp_path, "--apply"))


def test_pack_keeps_many_files_in_one_verified_tarball(tmp_path, tree, monkeypatch):
    import tarfile
    for index in range(30):
        (tree / "judge-queue/primary-frontier" / f"t{index}.json").write_text(str(index))
    monkeypatch.setattr("analysis.scripts.archive_geoaxis_prompts_26k.MANIFEST_MAX_BYTES", 50)
    assert axis.main(["--staging", str(tmp_path / "staging"), "--pack", f"judges/queue={tree / 'judge-queue'}",
                      "--pack-manifests", f"acquisition/phase1={tree / 'phase1'}"]) == 0
    with tarfile.open(tmp_path / "staging/judges/queue.tar.gz") as tar:
        names = set(tar.getnames())
    assert "primary-frontier/t29.json" in names and "env.sh" not in names and len(names) == 31
    members = (tmp_path / "staging/judges/queue.members.tsv").read_text().splitlines()
    assert len(members) == 32  # header + 31 members
    with tarfile.open(tmp_path / "staging/acquisition/phase1.tar.gz") as tar:
        assert sorted(tar.getnames()) == ["job.log", "run_manifest.json"]
    manifest = (tmp_path / "staging/MANIFEST.tsv").read_text()
    assert "judges/queue.tar.gz" in manifest and "judges/queue.members.tsv" in manifest


def test_rejects_group_destinations_that_escape_the_repo(tmp_path, tree):
    with pytest.raises(SystemExit, match="bad group"):
        axis.main(["--staging", str(tmp_path / "s"), "--group", f"../x={tmp_path / 'subspace'}"])
