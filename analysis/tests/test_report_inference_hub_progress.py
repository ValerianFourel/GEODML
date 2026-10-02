"""Published cell counts use unique identities and a single immutable snapshot."""
from dataclasses import asdict
import hashlib

import pytest

from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, REGISTRY_PATH
from analysis.interpretability.pipeline.agentic_hours import canonical, digest
from analysis.interpretability.pipeline.agentic_task_ledger import identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import report_inference_hub_progress as progress
from analysis.scripts.publish_qwen_results import INDEX_PATH, INDEX_VERSION
from analysis.tests.test_agentic_hours import MemoryHub


def outcome(name, state="completed", model="qwen"):
    identity = ClaimIdentity(name, model, "revision", "generation-v1", hashlib.sha256(name.encode()).hexdigest())
    return identity_fingerprint(identity), {
        "identity": asdict(identity), "state": state, "owner_id": "worker", "generation": 1,
        "record_references": [{"table": "results", "writer_id": name, "shard_sequence": 0,
                               "row_id": name, "row_sha256": hashlib.sha256(name.encode()).hexdigest()}]
        if state == "completed" else [],
    }


def bundle(hub, *events, tag="a"):
    outcomes = dict(events)
    files = {}
    for event in outcomes.values():
        for ref in event["record_references"]:
            stem = f"data/{ref['table']}/part-{ref['writer_id']}-{ref['shard_sequence']:06d}"
            for suffix in (".jsonl", ".manifest.json"):
                files[stem + suffix] = {"sha256": "a" * 64, "bytes": 1}
    manifest = {"format_version": "geodml-hour-bundle-v1", "files": files,
                "outcomes": outcomes, "metadata": {"tag": tag}}
    name = "bundle-" + digest(manifest)
    hub.commit(hub.head(), {f"exchange/bundles/{name}.json": canonical(manifest)}, "fixture bundle")
    return {"bundle": name, "completed": sum(e["state"] == "completed" for e in outcomes.values()),
            "failed": sum(e["state"] == "terminal_failed" for e in outcomes.values())}


def documents(hub, entries, hours):
    hub.commit(hub.head(), {
        INDEX_PATH: canonical({"format_version": INDEX_VERSION, "model": "qwen38", "bundles": entries}),
        REGISTRY_PATH: canonical({"format_version": "geodml-hours-v1", "hours": hours}),
    }, "fixture index")


def hour(model, events, entries, extra=()):
    return {"model": model, "task_fingerprints": [fp for fp, _ in events] + list(extra),
            "completed": [fp for fp, event in events if event["state"] == "completed"],
            "failed": [fp for fp, event in events if event["state"] == "terminal_failed"],
            "checkpoints": [entry["bundle"] for entry in entries]}


def test_overlapping_indexes_and_replans_count_unique_successes_and_failures(tmp_path):
    hub = MemoryHub()
    q1, q2, qfail = outcome("q1"), outcome("q2"), outcome("qfail", "terminal_failed")
    llama = outcome("llama", model="llama")
    qbundle = bundle(hub, q1, qfail)
    overlapping = bundle(hub, q1, q2, tag="other writer")
    lbundle = bundle(hub, llama)
    documents(hub, [qbundle, overlapping], {
        "old-q": hour("qwen38", [q1, qfail], [qbundle], extra=["not-yet-done"]),
        "new-q": hour("qwen38", [q1], [qbundle]),
        "llama": hour("llama4", [llama], [lbundle]),
    })
    report = progress.collect(Exchange(hub, tmp_path))
    assert report["status"] == "counted"
    assert report["models"]["qwen38"]["completed"] == 2
    assert report["models"]["qwen38"]["terminal_failed"] == 1
    assert report["models"]["qwen38"]["registry_expected"] == 3
    assert report["models"]["qwen38"]["registry_unresolved"] == 1
    assert report["models"]["qwen38"]["expected_total"] is None
    assert report["generator_totals"]["completed"] == 3
    assert report["generator_totals"]["terminal_failed"] == 1


def test_conflicting_terminal_outcomes_are_not_counted_as_success_or_failure(tmp_path):
    hub = MemoryHub()
    success = outcome("same")
    failed = outcome("same", "terminal_failed")
    entries = [bundle(hub, success), bundle(hub, failed)]
    documents(hub, entries, {})
    report = progress.collect(Exchange(hub, tmp_path))
    assert report["status"] == "partial"
    assert report["models"]["qwen38"]["completed"] == 0
    assert report["models"]["qwen38"]["terminal_failed"] == 0
    assert report["models"]["qwen38"]["conflicts"] == 1


def test_missing_evidence_remains_partial(tmp_path):
    hub = MemoryHub()
    event = outcome("llama", model="llama")
    missing = {"bundle": "bundle-" + "0" * 64, "completed": 1, "failed": 0}
    hub.commit(hub.head(), {REGISTRY_PATH: canonical({"format_version": "geodml-hours-v1", "hours": {
        "missing": hour("llama4", [event], [missing])}})}, "missing fixture")
    report = progress.collect(Exchange(hub, tmp_path))
    assert report["status"] == "partial"
    assert report["models"]["qwen38"]["completed"] is None
    assert report["models"]["llama4"]["registry_unresolved"] == 1
    assert report["models"]["llama4"]["completed"] == 0
    assert report["generator_totals"]["completed"] is None
    assert len(report["issues"]) >= 2


def test_all_reads_use_requested_revision_even_if_latest_has_new_results(tmp_path):
    hub = MemoryHub()
    first, second = outcome("first"), outcome("second")
    entry = bundle(hub, first)
    documents(hub, [entry], {})
    revision = hub.head()
    next_entry = bundle(hub, second)
    documents(hub, [entry, next_entry], {})
    report = progress.collect(Exchange(hub, tmp_path), revision)
    assert report["revision"] == revision
    assert report["models"]["qwen38"]["completed"] == 1
    assert progress.collect(Exchange(hub, tmp_path))["models"]["qwen38"]["completed"] == 2


@pytest.mark.parametrize("malformed", ["index", "registry", "hour", "outcomes", "event", "references"])
def test_malformed_remote_json_reports_partial_instead_of_crashing(tmp_path, malformed):
    hub = MemoryHub()
    event = outcome("qwen")
    entry = bundle(hub, event)
    index = {"format_version": INDEX_VERSION, "model": "qwen38", "bundles": [entry]}
    registry = {"format_version": "geodml-hours-v1", "hours": {}}
    if malformed == "index":
        index = []
    elif malformed == "registry":
        registry = []
    elif malformed == "hour":
        registry["hours"] = {"malformed": []}
    else:
        # Publish a correctly hashed malformed manifest to reach shape validation.
        import json
        manifest = json.loads(hub.read(f"exchange/bundles/{entry['bundle']}.json", hub.head()))
        if malformed == "outcomes":
            manifest["outcomes"] = []
        elif malformed == "event":
            manifest["outcomes"][event[0]] = []
        else:
            manifest["outcomes"][event[0]]["record_references"] = [[]]
        name = "bundle-" + digest(manifest)
        hub.commit(hub.head(), {f"exchange/bundles/{name}.json": canonical(manifest)}, "malformed bundle")
        index["bundles"][0]["bundle"] = name
    hub.commit(hub.head(), {INDEX_PATH: canonical(index), REGISTRY_PATH: canonical(registry)}, "malformed docs")
    report = progress.collect(Exchange(hub, tmp_path))
    assert report["status"] == "partial"
    assert report["issues"]


def test_no_remote_evidence_has_unknown_generator_total(tmp_path):
    report = progress.collect(Exchange(MemoryHub(), tmp_path))
    assert report["generator_totals"]["completed"] is None
    assert report["generator_totals"]["terminal_failed"] is None
    assert report["generator_totals"]["counts_are_lower_bounds"] is True
