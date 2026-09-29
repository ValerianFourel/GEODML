"""The GeoAxis 26k archive keeps exactly the allowlist, checks the frozen hashes and verifies the Hub copy."""

import hashlib
import shutil

import pytest

from analysis.scripts import archive_geoaxis_prompts_26k as geo


class FakeHub:
    def __init__(self, root, private=True):
        self.root, self.private, self.calls = root, private, []

    def create_repo(self, *args, **kwargs):
        self.calls.append("create_repo")

    def repo_info(self, *args, **kwargs):
        return type("Info", (), {"private": self.private})()

    def upload_large_folder(self, repo_id, repo_type, folder_path, private):
        self.calls.append("upload")
        shutil.copytree(folder_path, self.root, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))

    def get_paths_info(self, repo, paths, repo_type):
        out = []
        for path in paths:
            target = self.root / path
            if target.exists():
                data = target.read_bytes()
                lfs = {"sha256": hashlib.sha256(data).hexdigest()} if target.suffix in {".npz", ".jsonl"} else None
                blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                out.append(type("Info", (), {"path": path, "size": len(data), "lfs": lfs, "blob_id": blob})())
        return out


@pytest.fixture
def sources(tmp_path, monkeypatch):
    base = tmp_path / "src"
    audit = base / "checkpoint/final-audit-4gpu-a452213"
    for view in ("qwen", "mistral"):
        (audit / f"projections/{view}/shard-000").mkdir(parents=True)
        (audit / f"projections/{view}/shard-000/question_embeddings.restricted-local.npz").write_bytes(b"E" * 64)
        (audit / f"projections/{view}/shard-000/projection_manifest.json").write_text("{}")
    (audit / "logs").mkdir()
    (audit / "logs/run.log").write_text("log")
    (audit / "compliant-candidates.jsonl").write_text('{"candidate_id": "a"}\n')
    (audit / "final-axis-map.jsonl").write_text('{"candidate_id": "a", "axis_1_percentile_0_1": 0.5}\n')
    (audit / "final-audit-summary.json").write_text("{}")
    (base / "checkpoint/merge_manifest.json").write_text("{}")
    (base / "checkpoint/projections/qwen").mkdir(parents=True)
    (base / "checkpoint/projections/qwen/question_embeddings.restricted-local.npz").write_bytes(b"X" * 10)
    (base / "checkpoint/merged").mkdir()
    (base / "checkpoint/merged/candidates.jsonl").write_bytes(b"c" * 300)
    qmap = base / "maps/qwen"
    qmap.mkdir(parents=True)
    (qmap / "readiness_embedding_map.json").write_text("{}")
    (qmap / "readiness_axis_exemplars.restricted-local.json").write_text('["exemplar text"]')
    (base / "registration").mkdir()
    (base / "registration/manifest.json").write_text("{}")
    (base / "registration/population-selection-records.jsonl").write_text("{}\n")
    (base / "home").mkdir()
    (base / "home/geodml-final-audit-latest.txt").write_text(str(audit) + "\n")
    (base / "home/geodml-acl-arr-pilot.env").write_text("HF_TOKEN=hf_" + "B" * 34 + "\n")
    monkeypatch.setattr(geo, "CONTRACT", {
        "final-audit/compliant-candidates.jsonl": hashlib.sha256((audit / "compliant-candidates.jsonl").read_bytes()).hexdigest(),
        "final-audit/final-axis-map.jsonl": hashlib.sha256((audit / "final-axis-map.jsonl").read_bytes()).hexdigest(),
    })
    return base, audit


def argv(tmp_path, base, audit, *extra):
    return ["--staging", str(tmp_path / "staging"), "--final-audit", str(audit), "--checkpoint", str(base / "checkpoint"),
            "--map", f"qwen={base / 'maps/qwen'}", "--registration", str(base / "registration"),
            "--pointer", str(base / "home/geodml-final-audit-latest.txt"),
            "--pointer", str(base / "home/geodml-acl-arr-pilot.env"), "--max-file-gb", "0.0000002", *extra]


def staged(tmp_path):
    root = tmp_path / "staging"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_dry_run_stages_the_allowlist_only_and_makes_no_hub_calls(tmp_path, sources, monkeypatch, capsys):
    base, audit = sources
    monkeypatch.setattr(geo, "make_api", lambda: pytest.fail("dry run must not call the Hub"))
    assert geo.main(argv(tmp_path, base, audit)) == 0
    files = staged(tmp_path)
    assert "final-audit/projections/qwen/shard-000/question_embeddings.restricted-local.npz" in files
    assert "final-audit/projections/mistral/shard-000/question_embeddings.restricted-local.npz" in files
    assert "final-audit/final-axis-map.jsonl" in files and "maps/qwen/readiness_embedding_map.json" in files
    assert "registration/population-registration-v1/population-selection-records.jsonl" in files
    assert "pointers/geodml-final-audit-latest.txt" in files and "checkpoint/merge_manifest.json" in files
    for absent in ("final-audit/logs/run.log", "maps/qwen/readiness_axis_exemplars.restricted-local.json",
                   "checkpoint/projections/qwen/question_embeddings.restricted-local.npz",
                   "checkpoint/merged/candidates.jsonl", "pointers/geodml-acl-arr-pilot.env"):
        assert absent not in files
    assert not any(f.startswith("checkpoint/final-audit") for f in files)
    manifest = (tmp_path / "staging/MANIFEST.tsv").read_text().splitlines()
    for line in manifest[1:]:
        rel, size, digest, _ = line.split("\t")
        data = (tmp_path / "staging" / rel).read_bytes()
        assert len(data) == int(size) and hashlib.sha256(data).hexdigest() == digest
    out = capsys.readouterr().out
    assert "EMBEDDING_SHARDS 2" in out and "DRY_RUN_OK" in out
    assert "CONTRACT final-audit/final-axis-map.jsonl MATCHES" in out


def test_contract_mismatch_stops_before_any_upload(tmp_path, sources, monkeypatch):
    base, audit = sources
    (audit / "final-axis-map.jsonl").write_text("changed\n")
    monkeypatch.setattr(geo, "make_api", lambda: pytest.fail("no upload on a contract mismatch"))
    with pytest.raises(SystemExit, match="CONTRACT_MISMATCH"):
        geo.main(argv(tmp_path, base, audit, "--apply"))


def test_apply_refuses_a_public_repo(tmp_path, sources, monkeypatch):
    base, audit = sources
    hub = FakeHub(tmp_path / "hub", private=False)
    monkeypatch.setattr(geo, "make_api", lambda: hub)
    monkeypatch.setattr(geo, "ensure_token", lambda: None)
    with pytest.raises(SystemExit, match="not private"):
        geo.main(argv(tmp_path, base, audit, "--apply"))
    assert "upload" not in hub.calls


def test_apply_uploads_and_verifies_every_file(tmp_path, sources, monkeypatch, capsys):
    base, audit = sources
    hub = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(geo, "make_api", lambda: hub)
    monkeypatch.setattr(geo, "ensure_token", lambda: None)
    assert geo.main(argv(tmp_path, base, audit, "--apply")) == 0
    assert "GEOAXIS_26K_COMPLETE" in capsys.readouterr().out
    assert (tmp_path / "staging.receipt.json").exists()
    (tmp_path / "hub/final-audit/final-axis-map.jsonl").write_text("tampered\n")
    rows = [line.split("\t") for line in (tmp_path / "staging/MANIFEST.tsv").read_text().splitlines()[1:]]
    bad = geo.verify(hub, "r", tmp_path / "staging", [(r, int(s), d, src) for r, s, d, src in rows])
    assert bad == ["final-audit/final-axis-map.jsonl"]
