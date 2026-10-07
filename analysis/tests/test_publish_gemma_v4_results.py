"""Gemma v4 export: only finished shards, skip identical files, verify by server hash, manifest last."""
import gzip
import hashlib
import json

import pytest

from analysis.interpretability.pipeline.agentic_hour_sync import ConflictError
from analysis.scripts import publish_gemma_v4_results as pub


class Store:
    """In-memory private repository with git-blob and LFS-style hashes."""
    def __init__(self, conflicts=0, corrupt=None):
        self.files, self.commits, self.rev, self.conflicts, self.corrupt = {}, [], 0, conflicts, corrupt

    def head(self):
        return f"r{self.rev}"

    def hashes(self, names, revision):
        out = {}
        for n in names:
            if n in self.files:
                raw = self.files[n]
                lfs = n.endswith(".gz")
                out[n] = {"size": len(raw), "blob_id": None if lfs else pub.git_blob_id(raw),
                          "sha256": hashlib.sha256(raw).hexdigest() if lfs else None}
        return out

    def commit(self, revision, files, message):
        if self.conflicts:
            self.conflicts -= 1
            raise ConflictError("advanced")
        assert revision == self.head()
        for name, raw in files.items():
            self.files[name] = b"tampered" if name == self.corrupt else raw
        self.commits.append((message, sorted(files)))
        self.rev += 1
        return self.head()


def run(tmp_path, statuses):
    root = tmp_path / "run"
    shards = []
    for i, status in enumerate(statuses):
        d = root / "shards" / f"shard-{i:04d}"
        if status:
            rep = d / "results/reports/w1"
            rep.mkdir(parents=True)
            (rep / "summary.json").write_text(json.dumps({"status": status, "states": {"done": 3}}))
            (rep / "maps.jsonl").write_text('{"judge_task_id": "m"}\n')
            with gzip.open(rep / "cells.jsonl.gz", "wt") as f:
                f.write('{"cell": %d}\n' % i)
            (d / "results/reports/latest.json").write_text(json.dumps({"directory": "w1"}))
        else:
            d.mkdir(parents=True)
        shards.append({"id": f"shard-{i:04d}", "directory": str(d), "cells": 3})
    (root / "frozen").mkdir(parents=True)
    (root / "frozen/manifest.json").write_text("{}")
    (root / "judge-config.json").write_text("{}")
    (root / "plan.json").write_text(json.dumps({"plan_id": "gemma-v4-abc", "shards": shards}))
    return root


def test_publishes_finished_shards_once_and_writes_the_manifest_last(tmp_path):
    root = run(tmp_path, ["finished", "finished_with_failures", "incomplete", None])
    store = Store(conflicts=1)
    runs = [pub.collect(root)]
    dry = pub.publish(store, runs, apply=False)
    assert dry["files_to_upload"] == 3 + 2 * 3 and not store.files
    assert dry["runs"][0]["shards_pending"] == ["shard-0002", "shard-0003"]
    done = pub.publish(store, runs, apply=True)
    assert done["applied"] and done["verified_files"] == 9
    assert store.commits[-1][1] == ["reviews/gemma-si-v4/gemma-v4-abc/export-manifest.json"]
    manifest = json.loads(store.files["reviews/gemma-si-v4/gemma-v4-abc/export-manifest.json"])
    assert manifest["semantic_acceptance"] == "not_established"
    assert {s["shard"]: s["status"] for s in manifest["shards"]}["shard-0003"] == "not_started"
    assert "reviews/gemma-si-v4/gemma-v4-abc/shards/shard-0002/cells.jsonl.gz" not in store.files
    # A rerun uploads nothing new; only the manifest is rewritten.
    commits = len(store.commits)
    again = pub.publish(store, [pub.collect(root)], apply=True)
    assert again["files_to_upload"] == 0 and len(store.commits) == commits + 1


def test_a_server_hash_mismatch_stops_before_the_manifest(tmp_path):
    root = run(tmp_path, ["finished"])
    bad = "reviews/gemma-si-v4/gemma-v4-abc/shards/shard-0000/maps.jsonl"
    store = Store(corrupt=bad)
    with pytest.raises(ValueError, match="server hashes differ"):
        pub.publish(store, [pub.collect(root)], apply=True)
    assert not any(n.endswith("export-manifest.json") for n in store.files)


def test_a_redirected_report_pointer_is_refused(tmp_path):
    root = run(tmp_path, ["finished"])
    (root / "shards/shard-0000/results/reports/latest.json").write_text(json.dumps({"directory": "../x"}))
    with pytest.raises(ValueError, match="invalid report pointer"):
        pub.collect(root)
