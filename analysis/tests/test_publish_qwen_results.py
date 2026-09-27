import json

from analysis.interpretability.pipeline.agentic_hour_sync import Exchange
from analysis.interpretability.pipeline.agentic_task_ledger import identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import publish_qwen_results as qwen
from analysis.tests.test_agentic_hours import MemoryHub, complete, data, inventory


def fixture(tmp_path):
    root = data(tmp_path / "jupiter")
    tasks, _, _ = inventory(root, stripes=4)
    return root, {row["task_id"]: row for row in tasks}


def test_dry_run_counts_without_publishing(tmp_path):
    root, tasks = fixture(tmp_path)
    complete(root, tasks["alpha-0"], "job1-worker0")
    hub = MemoryHub()
    report = qwen.publish(Exchange(hub, tmp_path / "journal"), root, stripes=4)
    assert (report["verified_on_disk"], report["new_cells"], report["writers"]) == (1, 1, 1)
    assert report["applied"] is False and len(hub.versions) == 1


def test_apply_publishes_verified_bundles_once_and_later_cells_incrementally(tmp_path):
    root, tasks = fixture(tmp_path)
    complete(root, tasks["alpha-0"], "job1-worker0")
    complete(root, tasks["alpha-1"], "job1-worker0")
    complete(root, tasks["beta-0"], "job2-worker0")
    hub = MemoryHub()
    exchange = Exchange(hub, tmp_path / "journal")
    report = qwen.publish(exchange, root, stripes=4, apply=True, source_commit="c" * 40)
    assert report["applied"] and report["completed_total"] == 3 and not report["blocked"]
    assert sorted(entry["writer_id"] for entry in report["bundles"]) == ["job1-worker0", "job2-worker0"]
    index = json.loads(hub.read(qwen.INDEX_PATH, hub.head()))
    assert index["completed_total"] == 3 and len(index["bundles"]) == 2
    manifest = exchange.manifest(report["bundles"][0]["bundle"])
    assert manifest["metadata"]["kind"] == "qwen-results"
    assert all(event["state"] == "completed" and event["record_references"] for event in manifest["outcomes"].values())
    before = len(hub.versions)
    again = qwen.publish(exchange, root, stripes=4, apply=True)
    assert again["new_cells"] == 0 and len(hub.versions) == before
    complete(root, tasks["beta-1"], "job3-worker0")
    later = qwen.publish(exchange, root, stripes=4, apply=True)
    assert later["new_cells"] == 1 and later["completed_total"] == 4
    fingerprint = identity_fingerprint(ClaimIdentity(**tasks["beta-1"]["claim_identity"]))
    assert list(exchange.manifest(later["bundles"][0]["bundle"])["outcomes"]) == [fingerprint]


def test_collect_returns_only_verified_terminal_cells(tmp_path):
    root, tasks = fixture(tmp_path)
    complete(root, tasks["alpha-0"], "job1-worker0")
    outcomes = qwen.collect(root, stripes=4)
    assert len(outcomes) == 1
    assert all(event["identity"]["task_id"] == "alpha-0" for event in outcomes.values())
