"""Registry round-trips per wave: O(1) for staging and syncing, same outcomes."""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter
from analysis.interpretability.pipeline.agentic_hour_sync import REGISTRY_PATH, Exchange
from analysis.interpretability.pipeline.agentic_hours import canonical, claim_hours
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import dispatch_threehour_wave as wave
from analysis.scripts.manage_agentic_hours import stage_wave, sync_wave
from analysis.tests.test_agentic_hours import (
    MemoryHub,
    complete,
    healthy,
    setup_exchange,
    snapshot,
    stage_wave_fixture,
)


class CountingHub(MemoryHub):
    def __init__(self):
        super().__init__()
        self.reads = Counter()

    def read(self, name, revision):
        self.reads[name] += 1
        return super().read(name, revision)


def test_snapshot_reuses_registry_this_process_committed_and_rereads_foreign_commits(tmp_path):
    hub = CountingHub()
    exchange = Exchange(hub, tmp_path / "journal")
    exchange.transact("one", {}, lambda state: {**state, "current_plan": "p1"})
    hub.reads.clear()
    _, state = exchange.snapshot()
    assert state["current_plan"] == "p1" and hub.reads[REGISTRY_PATH] == 0
    state["hours"]["mutated"] = {}
    assert exchange.snapshot()[1]["hours"] == {}  # callers never share a cached object
    exchange.immutable({"exchange/objects/x": b"x"})  # a registry-free commit of our own
    assert exchange.snapshot()[1]["current_plan"] == "p1" and hub.reads[REGISTRY_PATH] == 0
    Exchange(hub, tmp_path / "other").transact("two", {}, lambda state: {**state, "current_plan": "p2"})
    hub.reads.clear()
    assert exchange.snapshot()[1]["current_plan"] == "p2" and hub.reads[REGISTRY_PATH] == 1


def test_stage_wave_reads_registry_plan_and_input_bundle_once(tmp_path):
    hub = CountingHub()
    _, packages, request, kwargs = stage_wave_fixture(tmp_path, hub)
    exchange = Exchange(hub, tmp_path / "dispatch-journal")  # a fresh dispatcher process
    wave_requests = [request("wave-1", [packages[0]["hour_id"]]), request("wave-2", [packages[1]["hour_id"]])]
    hub.reads.clear()
    attempts = stage_wave(exchange, requests=wave_requests, operation_id="reserve-wave", **kwargs)
    assert [attempt["attempt_id"] for attempt in attempts] == ["wave-1", "wave-2"]
    assert hub.reads[REGISTRY_PATH] == 1
    assert [n for name, n in hub.reads.items() if name.startswith("coordination/plans/")] == [1]
    bundles = [name for name in hub.reads if name.startswith("exchange/bundles/")]
    assert [hub.reads[name] for name in bundles] == [1]
    # Reuse ends with the staging session: a later download verifies again.
    exchange.download(bundles[0].removeprefix("exchange/bundles/").removesuffix(".json"),
                      kwargs["dataset"], stripes=4)
    assert hub.reads[bundles[0]] == 2


def reports(root, count, prefix):
    (root / "reports").mkdir(parents=True)
    for i in range(count):
        (root / "reports" / f"{i}.json").write_text(json.dumps({prefix: i}))
    return [f"reports/{i}.json" for i in range(count)]


def test_upload_many_shares_commits_and_matches_single_bundle_ids(tmp_path):
    entries = [(tmp_path / name, reports(tmp_path / name, 10, name), {}, {"part": name}) for name in ("a", "b")]
    hub = MemoryHub()
    bundles, errors = Exchange(hub, tmp_path / "journal").upload_many(entries)
    assert errors == {}
    assert len(hub.versions) - 1 == 2  # one shared object commit, one manifest commit
    separate = Exchange(MemoryHub(), tmp_path / "separate")
    assert bundles == [separate.upload(root, names, outcomes=outcomes, metadata=metadata)
                       for root, names, outcomes, metadata in entries]


def test_upload_many_isolates_a_rejected_entry_only_when_asked(tmp_path):
    good = (tmp_path / "good", reports(tmp_path / "good", 2, "ok"), {}, {})
    bad_root = tmp_path / "bad"
    (bad_root / "reports").mkdir(parents=True)
    token = "hf_" + "x" * 24
    (bad_root / "reports/leak.json").write_text(json.dumps({"value": token}))
    bad = (bad_root, ["reports/leak.json"], {}, {})
    hub = MemoryHub()
    exchange = Exchange(hub, tmp_path / "journal")
    with pytest.raises(ValueError, match="credential-shaped"):
        exchange.upload_many([good, bad])
    bundles, errors = exchange.upload_many([bad, good], isolate_errors=True)
    assert bundles[0] is None and "credential-shaped" in errors[0]
    assert set(exchange.manifest(bundles[1])["files"]) == set(good[1])
    assert not any(token.encode() in raw for raw in hub.versions[-1].values())


def finish(root, task, writer_id, answer):
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
    claim = ledger.claim(ClaimIdentity(**task["claim_identity"]), owner_id=writer_id).claim
    writer = FinalDatasetWriter(root, writer_id=writer_id)
    reference = writer.append("generations", {"answer": answer}, transaction_id=task["task_id"])
    writer.seal()
    ledger.transition(claim, state="completed", record_references=[reference])


def finished_wave(tmp_path, answers=None, unsealed=None):
    """Two attempts, one package each; every task completed by its writer.

    ``unsealed`` names an attempt whose last task stops at result_saved with a
    reference to a shard that was never sealed.
    """
    root, first, _, value = setup_exchange(tmp_path)
    packages = value["packages"][:2]

    def claim_both(state):
        for name, package in zip(("one", "two"), packages):
            state = claim_hours(state, hour_ids=[package["hour_id"]], cluster="jupiter",
                                attempt_id=name, supported_models=["qwen38"])
        return state

    first.transact("reserve-both", {}, claim_both)
    _, state = first.snapshot()
    attempts = []
    for name, package in zip(("one", "two"), packages):
        directory = tmp_path / f"attempt-{name}"
        directory.mkdir()
        tasks = {fp: value["tasks"][fp] for fp in package["task_fingerprints"]}
        (directory / "tasks.json").write_bytes(canonical(tasks))
        for number, task in enumerate(tasks.values(), 1):
            if name == unsealed and number == len(tasks):
                ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
                claim = ledger.claim(ClaimIdentity(**task["claim_identity"]), owner_id=f"jupiter-{name}").claim
                ledger.transition(claim, state="result_saved", record_references=[{
                    "table": "generations", "writer_id": f"jupiter-{name}", "shard_sequence": 99,
                    "line_number": 1, "record_id": "record-never-sealed"}])
            elif answers and name in answers:
                finish(root, task, f"jupiter-{name}", answers[name])
            else:
                complete(root, task, f"jupiter-{name}")
        attempts.append({"dataset_root": str(root), "request": {"attempt_dir": str(directory)},
                         "ledger_stripes": 4, "writer_id": f"jupiter-{name}", "cluster": "jupiter",
                         "attempt_id": name,
                         "owners": {package["hour_id"]: state["hours"][package["hour_id"]]["owner"]}})
    now = int(time.time())
    scheduler = {**snapshot(), "captured_at_epoch": now, "owners": [
        {"owner_id": a["writer_id"], "state": "COMPLETED", "cluster": "jupiter", "job_id": str(70 + i)}
        for i, a in enumerate(attempts)]}
    return first, attempts, scheduler, {**healthy(), "captured_at_epoch": now}


def counted_transacts(exchange):
    calls = []
    original = exchange.transact

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    exchange.transact = counted
    return calls


def test_sync_wave_releases_every_attempt_in_one_registry_commit(tmp_path):
    exchange, attempts, scheduler, storage = finished_wave(tmp_path)
    hub = exchange.store
    calls = counted_transacts(exchange)
    before = len(hub.versions)
    results = sync_wave(exchange, attempts, scheduler, storage)
    assert [result["status"] for result in results] == ["released", "released"]
    assert len(calls) == 1 and calls[0].startswith("checkpoint-wave-")
    assert len(hub.versions) - before == 3  # shared objects, both manifests, one registry commit
    _, state = exchange.snapshot()
    for attempt, result in zip(attempts, results):
        (key,) = attempt["owners"]
        hour = state["hours"][key]
        assert hour["status"] == "complete" and hour["owner"] is None
        assert hour["completed"] == sorted(hour["task_fingerprints"])
        assert hour["checkpoints"] == [result["bundle"]]
        manifest = exchange.manifest(result["bundle"])
        assert set(manifest["outcomes"]) == set(hour["task_fingerprints"])
        assert manifest["metadata"]["terminal"] is True
        assert manifest["metadata"]["scheduler_confirmation"] == [
            row for row in scheduler["owners"] if row["owner_id"] == attempt["writer_id"]]
        receipt = Path(attempt["request"]["attempt_dir"]) / "sync.json"
        assert json.loads(receipt.read_bytes()) == result
    # A rerun returns the saved receipts without touching the Hub.
    calls.clear()
    before = len(hub.versions)
    assert sync_wave(exchange, attempts, scheduler, storage) == results
    assert calls == [] and len(hub.versions) == before
    # A receipt lost after the registry commit is restored, never re-applied.
    (Path(attempts[1]["request"]["attempt_dir"]) / "sync.json").unlink()
    assert sync_wave(exchange, attempts, scheduler, storage) == results
    assert calls == [] and len(hub.versions) == before


def test_sync_wave_keeps_a_rejected_attempt_owned_and_releases_the_rest(tmp_path, monkeypatch):
    import re

    from analysis.interpretability.pipeline import agentic_dataset

    token = "hf_" + "y" * 24
    # Emulate a saved shard the publication scanner rejects although it reached disk.
    monkeypatch.setattr(agentic_dataset, "SECRET_PATTERN", re.compile("(?!)"))
    exchange, attempts, scheduler, storage = finished_wave(tmp_path, answers={"two": "leaked " + token})
    monkeypatch.undo()
    owner = attempts[1]["owners"]
    results = sync_wave(exchange, attempts, scheduler, storage)
    assert results[0]["status"] == "released"
    assert results[1] == {"status": "blocked", "reason": "upload_rejected",
                          "error": "credential-shaped content rejected"}
    _, state = exchange.snapshot()
    (key,) = owner
    assert state["hours"][key]["owner"] == owner[key]
    assert not (Path(attempts[1]["request"]["attempt_dir"]) / "sync.json").exists()
    assert not any(token.encode() in raw for raw in exchange.store.versions[-1].values())


def test_sync_wave_attributes_a_blocked_reconciliation_to_its_own_writer(tmp_path):
    exchange, attempts, scheduler, storage = finished_wave(tmp_path, unsealed="two")
    calls = counted_transacts(exchange)
    results = sync_wave(exchange, attempts, scheduler, storage)
    assert results[0]["status"] == "released"
    assert results[1]["status"] == "blocked"
    assert [row["owner_id"] for row in results[1]["reconciliation"]["blocked"]] == ["jupiter-two"]
    assert all(row["owner_id"] == "jupiter-two" for row in results[1]["reconciliation"]["actions"])
    assert len(calls) == 1
    _, state = exchange.snapshot()
    (key,) = attempts[1]["owners"]
    assert state["hours"][key]["owner"] == attempts[1]["owners"][key]


def site_with(tmp_path, attempts):
    paths = []
    for attempt in attempts:
        path = tmp_path / attempt["attempt_id"] / "attempt.json"
        wave.save(path, attempt)
        paths.append(str(path))
    site = tmp_path / "site.json"
    wave.save(site, {"dataset_root": str(tmp_path), "attempts": paths})
    return site


def test_sync_sites_publishes_finished_attempts_and_reports_live_ones(tmp_path, monkeypatch):
    attempts = [{"attempt_id": name, "writer_id": "jupiter-" + name, "owners": {"h-" + name: {}}}
                for name in ("done", "live")]
    site = site_with(tmp_path, attempts)
    monkeypatch.setattr(wave, "health", lambda *a: {})
    monkeypatch.setattr(wave, "scheduler_wave", lambda pending: {"owners": [
        {"owner_id": "jupiter-done", "state": "COMPLETED"}, {"owner_id": "jupiter-live", "state": "RUNNING"}]})
    synced = []

    def fake_sync(exchange, finished, snapshot, storage):
        synced.append([a["attempt_id"] for a in finished])
        return [{"status": "released", "bundle": "b"} for _ in finished]

    monkeypatch.setattr(wave, "sync_wave", fake_sync)
    exchange = SimpleNamespace(snapshot=lambda: ("rev", {"hours": {}}))
    summary = wave.sync_sites(SimpleNamespace(), exchange, [site], tmp_path, require_finished=False)
    assert synced == [["done"]]
    assert summary["released"] == ["done"]
    assert summary["live"] == [{"attempt_id": "live", "observed": ["RUNNING"]}]
    # Dispatch requires every earlier attempt to be finished before uploading anything.
    synced.clear()
    with pytest.raises(ValueError, match="live observed: RUNNING"):
        wave.sync_sites(SimpleNamespace(), exchange, [site], tmp_path, require_finished=True)
    assert synced == []


def test_dispatch_stops_when_an_earlier_attempt_stays_blocked(tmp_path, monkeypatch):
    attempts = [{"attempt_id": "one", "writer_id": "jupiter-one", "owners": {"h": {}}}]
    site = site_with(tmp_path, attempts)
    monkeypatch.setattr(wave, "health", lambda *a: {})
    monkeypatch.setattr(wave, "scheduler_wave", lambda pending: {"owners": [
        {"owner_id": "jupiter-one", "state": "COMPLETED"}]})
    monkeypatch.setattr(wave, "sync_wave", lambda *a: [{"status": "blocked", "reason": "upload_rejected"}])
    exchange = SimpleNamespace(snapshot=lambda: ("rev", {"hours": {}}))
    with pytest.raises(ValueError, match="could not be released"):
        wave.sync_sites(SimpleNamespace(), exchange, [site], tmp_path, require_finished=True)
    summary = wave.sync_sites(SimpleNamespace(), exchange, [site], tmp_path, require_finished=False)
    assert summary["blocked"] == [{"attempt_id": "one", "result": {"status": "blocked", "reason": "upload_rejected"}}]


def test_sync_llama_cli_needs_no_allocation_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dispatch_threehour_wave.py", "sync-llama", "--since", "2026-09-01",
                                      "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="Missing required arguments") as excinfo:
        wave.main()
    assert all(flag in str(excinfo.value) for flag in ("--qwen-reference", "--repo-id", "--llama-site"))
    monkeypatch.setattr(sys, "argv", ["dispatch_threehour_wave.py", "llama", "--since", "2026-09-01",
                                      "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="approval required"):
        wave.main()


def test_objects_fan_out_below_the_hub_directory_limit_and_old_bundles_still_download(tmp_path):
    from analysis.interpretability.pipeline.agentic_hour_sync import legacy_object_path, object_path
    root = tmp_path / "source"
    names = reports(root, 3, "v")
    hub = MemoryHub()
    exchange = Exchange(hub, tmp_path / "journal")
    bundle = exchange.upload(root, names, outcomes={}, metadata={})
    objects = [name for name in hub.versions[-1] if name.startswith("exchange/objects")]
    assert objects and all(name.startswith("exchange/objects-v2/") and name.count("/") == 3 for name in objects)
    # A bundle published before the fan-out stored its objects flat; it must still download.
    files = exchange.manifest(bundle)["files"]
    moved = {legacy_object_path(entry["sha256"]): hub.versions[-1][object_path(entry["sha256"])]
             for entry in files.values()}
    hub.versions.append({**{k: v for k, v in hub.versions[-1].items() if k not in
                            {object_path(e["sha256"]) for e in files.values()}}, **moved})
    exchange.download(bundle, tmp_path / "mirror", stripes=4, import_outcomes=False)
    assert sorted(p.name for p in (tmp_path / "mirror/reports").iterdir()) == sorted(Path(n).name for n in names)


def test_reads_retry_timeouts_but_not_real_errors():
    from analysis.interpretability.pipeline.agentic_hour_sync import with_network_retries

    class ReadTimeout(Exception):
        pass

    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ReadTimeout("The read operation timed out")
        return b"ok"

    assert with_network_retries(flaky, sleep=waits.append) == b"ok" and waits == [5, 10]
    with pytest.raises(ValueError):
        with_network_retries(lambda: (_ for _ in ()).throw(ValueError("bad")), sleep=waits.append)
    calls.clear()
    with pytest.raises(ReadTimeout):
        with_network_retries(lambda: (calls.append(1), (_ for _ in ()).throw(ReadTimeout()))[1], attempts=2,
                             sleep=lambda _: None)
    assert len(calls) == 2


def test_reupload_checks_presence_instead_of_downloading_objects(tmp_path):
    class PresenceHub(CountingHub):
        def exists(self, name, revision):
            self.reads["exists:" + name] += 1
            return name in self.versions[int(revision)]

    root = tmp_path / "source"
    names = reports(root, 4, "p")
    hub = PresenceHub()
    exchange = Exchange(hub, tmp_path / "journal")
    bundle = exchange.upload(root, names, outcomes={}, metadata={})
    before = len(hub.versions)
    hub.reads.clear()
    assert exchange.upload(root, names, outcomes={}, metadata={}) == bundle
    assert len(hub.versions) == before
    assert not [name for name in hub.reads if name.startswith("exchange/objects-v2/")]
    assert sum(1 for name in hub.reads if name.startswith("exists:")) == 4


class HashHub(CountingHub):
    """Reports Hub-style hashes: git blob ids for even sizes, LFS SHA-256 for odd sizes."""

    def hashes(self, names, revision):
        import hashlib
        result = {}
        for name in names:
            raw = self.versions[int(revision)].get(name)
            if raw is None:
                continue
            if len(raw) % 2:
                result[name] = {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "blob_id": None}
            else:
                result[name] = {"size": len(raw), "sha256": None,
                                "blob_id": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()}
        return result


def test_remote_verification_uses_hub_hashes_instead_of_downloads(tmp_path):
    from analysis.interpretability.pipeline.agentic_hour_sync import object_path
    root = tmp_path / "source"
    names = reports(root, 6, "h")
    hub = HashHub()
    exchange = Exchange(hub, tmp_path / "journal")
    bundle = exchange.upload(root, names, outcomes={}, metadata={})
    hub.reads.clear()
    exchange.download(bundle, root, stripes=4, import_outcomes=False, verify_remote=True)
    assert not [name for name in hub.reads if name.startswith("exchange/objects")]
    # A corrupted Hub copy fails verification even though the local file is intact.
    sha = exchange.manifest(bundle)["files"][names[0]]["sha256"]
    hub.commit(hub.head(), {object_path(sha): b"tampered"}, "corrupt fixture")
    with pytest.raises(ValueError, match="corrupt"):
        exchange.download(bundle, root, stripes=4, import_outcomes=False, verify_remote=True)


def test_each_sealed_shard_is_read_once_and_rechecked_after_a_change(tmp_path, monkeypatch):
    from analysis.interpretability.pipeline import agentic_dataset
    from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
    root = tmp_path / "ds"
    initialize_dataset(root, population_id="p", acceptance_policy_id="a")
    writer = FinalDatasetWriter(root, writer_id="w")
    refs = [writer.append("generations", {"answer": f"a{i}"}, transaction_id=f"t{i}") for i in range(50)]
    writer.seal()
    reads = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(self.name), real(self))[1])
    agentic_dataset._SHARD_INDEX.clear()
    assert all(agentic_dataset.verify_record_reference(root, dict(ref)) for ref in refs)
    assert len([name for name in reads if name.endswith(".jsonl")]) == 1
    shard = next((root / "data/generations").glob("*.jsonl"))
    shard.chmod(0o644)
    shard.write_bytes(shard.read_bytes().replace(b"a1", b"b1", 1))
    assert not agentic_dataset.verify_record_reference(root, dict(refs[0]))
