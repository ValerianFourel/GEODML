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
