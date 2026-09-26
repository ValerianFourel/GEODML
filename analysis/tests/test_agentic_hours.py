"""Exercise portable hours across independent local cluster mirrors."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
)
from analysis.interpretability.pipeline.agentic_hour_runtime import admission
from analysis.interpretability.pipeline.agentic_hour_sync import (
    ConflictError,
    Exchange,
    checkpoint_files,
)
from analysis.interpretability.pipeline.agentic_hours import (
    build_plan,
    canonical,
    claim_hours,
    empty_registry,
    finish_hours,
    install_plan,
    inventory,
    verify_plan,
)
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.manage_agentic_hours import sync_once
from analysis.scripts.publish_agentic_dataset import build_manifest


class MemoryHub:
    """Revisioned remote filesystem with genuine competing compare-and-set writes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.versions = [{}]
        self.lose_response = False

    def head(self):
        with self.lock:
            return str(len(self.versions) - 1)

    def read(self, name, revision):
        return self.versions[int(revision)].get(name)

    def commit(self, revision, files, message):
        with self.lock:
            if int(revision) != len(self.versions) - 1:
                raise ConflictError("advanced")
            self.versions.append({**self.versions[-1], **files})
            if self.lose_response:
                self.lose_response = False
                raise ConnectionError("response lost after server committed")
            return str(len(self.versions) - 1)


def data(root):
    initialize_dataset(root, population_id="frozen", acceptance_policy_id="accepted-v1")
    writer = FinalDatasetWriter(root, writer_id="registration")
    for rank, keyword in enumerate(("alpha", "beta")):
        prompt = f"prompt-{keyword}"
        writer.append("keyword_memberships", {
            "prompt_id": prompt, "primary_keyword_id": keyword,
            "primary_priority_rank": rank, "keyword_ids": [keyword],
        }, transaction_id=prompt)
        for number in range(2):
            task = f"{keyword}-{number}"
            identity = ClaimIdentity(task, "pinned-model", "a" * 40, "protocol-v1",
                                     hashlib.sha256(task.encode()).hexdigest())
            writer.append("task_definitions", {
                "task_id": task, "prompt_id": prompt, "model": "qwen38",
                "claim_identity": asdict(identity), "configuration_sha256": "config-qwen",
                "runnable_task": {"cell_id": task}, "dependency_fingerprints": [],
            }, transaction_id=task)
    writer.seal()
    return root


def calibration():
    return {"qwen38": {"cluster": "jupiter", "gpus": 4, "gpu_type": "GH200",
                       "evidence": "fixture-allocation", "scientific_config_sha256": "config-qwen",
                       "reference_profile_sha256": "reference-hash", "seconds_per_task": 1600,
                       "startup_seconds": 300, "drain_seconds": 100}}


def plan(root, state=None, blocked=None):
    tasks, complete, local_blocked = inventory(root, stripes=4)
    return build_plan(tasks=tasks, calibration=calibration(), registry=state or empty_registry(),
                      contract=json.loads((root / "contract.json").read_bytes()),
                      completed=complete, blocked=local_blocked | (blocked or set()),
                      source_commit="b" * 40, input_bundle="input-bundle")


def setup_exchange(tmp_path):
    root = data(tmp_path / "jupiter")
    hub = MemoryHub()
    first = Exchange(hub, tmp_path / "jupiter-journal")
    second = Exchange(hub, tmp_path / "horeka-journal")
    value = plan(root)
    first.transact("plan-initial", {"action": "plan"}, lambda state: install_plan(state, value, cluster="jupiter"),
                   {f"coordination/plans/{value['plan_id']}.json": canonical(value)})
    return root, first, second, value


def complete(root, task, writer_id):
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
    identity = ClaimIdentity(**task["claim_identity"])
    claim = ledger.claim(identity, owner_id=writer_id).claim
    writer = FinalDatasetWriter(root, writer_id=writer_id)
    reference = writer.append("generations", {"answer": "synthetic test answer"}, transaction_id=task["task_id"])
    writer.seal()
    ledger.transition(claim, state="completed", record_references=[reference])


def test_hour_is_fixed_work_not_an_allocation(tmp_path):
    value = plan(data(tmp_path / "dataset"))
    assert [(p["keyword_id"], len(p["task_fingerprints"]), p["reference_seconds"])
            for p in value["packages"]] == [("alpha", 2, 3600), ("beta", 2, 3600)]
    assert value == plan(tmp_path / "dataset")
    verify_plan(value)
    value["packages"][0]["reference_seconds"] = 1
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_plan(value)


def test_missing_or_wrong_calibration_does_not_guess(tmp_path):
    root = data(tmp_path / "dataset")
    tasks, done, blocked = inventory(root, stripes=4)
    args = {"tasks": tasks, "registry": empty_registry(), "contract": {}, "completed": done,
            "blocked": blocked, "source_commit": "a" * 40, "input_bundle": "input"}
    with pytest.raises(ValueError, match="calibration required"):
        build_plan(calibration={}, **args)
    wrong = calibration()
    wrong["qwen38"]["scientific_config_sha256"] = "different-settings"
    with pytest.raises(ValueError, match="scientific configuration"):
        build_plan(calibration=wrong, **args)


def test_two_clusters_racing_for_an_hour_have_one_owner(tmp_path):
    _, first, second, value = setup_exchange(tmp_path)
    hour_id = value["packages"][0]["hour_id"]
    barrier = threading.Barrier(2)
    def reserve(pair):
        cluster, exchange = pair
        barrier.wait()
        try:
            exchange.transact(cluster, {"action": "claim", "cluster": cluster},
                              lambda state: claim_hours(state, hour_ids=[hour_id], cluster=cluster,
                                                        attempt_id=cluster, supported_models=["qwen38"]))
            return "owned"
        except ValueError:
            return "unavailable"
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(reserve, [("jupiter", first), ("horeka", second)]))
    assert sorted(results) == ["owned", "unavailable"]
    _, state = first.snapshot()
    assert state["hours"][hour_id]["status"] == "reserved"
    assert len(state["hours"][hour_id]["attempts"]) == 1


def test_lost_claim_response_resumes_same_operation_without_second_claim(tmp_path):
    _, first, _, value = setup_exchange(tmp_path)
    key = value["packages"][0]["hour_id"]
    change = lambda state: claim_hours(state, hour_ids=[key], cluster="jupiter", attempt_id="run1", supported_models=["qwen38"])
    first.store.lose_response = True
    with pytest.raises(ConnectionError):
        first.transact("reservation", {"hour": key}, change)
    first.transact("reservation", {"hour": key}, change)
    _, state = first.snapshot()
    assert state["hours"][key]["generation"] == 1
    assert state["hours"][key]["owner"]["attempt_id"] == "run1"
    with pytest.raises(ValueError, match="different intent"):
        first.transact("reservation", {"hour": "different"}, change)


def test_partial_jupiter_results_resume_on_horeka_without_repeating_done_tasks(tmp_path):
    root, first, second, value = setup_exchange(tmp_path)
    initial = first.upload(root, list(build_manifest(root)["files"]), outcomes={}, metadata={"kind": "input"})
    horeka = tmp_path / "horeka"
    second.download(initial, horeka, stripes=4)
    key = value["packages"][0]["hour_id"]
    first.transact("reserve-jupiter", {"hour": key}, lambda state: claim_hours(
        state, hour_ids=[key], cluster="jupiter", attempt_id="run1", supported_models=["qwen38"]))
    _, state = first.snapshot()
    owner = state["hours"][key]["owner"]
    fingerprints = value["packages"][0]["task_fingerprints"]
    complete(root, value["tasks"][fingerprints[0]], "jupiter-run1")
    names, outcomes = checkpoint_files(root, value["tasks"], stripes=4, writer_id="jupiter-run1")
    bundle = first.upload(root, names, outcomes=outcomes, metadata={"attempt": "run1"})
    first.download(bundle, root, stripes=4, import_outcomes=False, verify_remote=True)
    first.transact("finish-jupiter", {"bundle": bundle}, lambda current: finish_hours(
        current, owners={key: owner}, checkpoint=bundle, outcomes=outcomes, terminal=True))
    second.download(bundle, horeka, stripes=4)
    second.download(bundle, horeka, stripes=4)
    tasks, completed, _blocked = inventory(horeka, stripes=4)
    assert [row["task_id"] for row in tasks if row["fingerprint"] in completed] == ["alpha-0"]
    assert [row["task_id"] for row in tasks if row["fingerprint"] in fingerprints and row["fingerprint"] not in completed] == ["alpha-1"]
    second.transact("reserve-horeka", {"hour": key}, lambda current: claim_hours(
        current, hour_ids=[key], cluster="horeka", attempt_id="run2", supported_models=["qwen38"]))
    _, state = second.snapshot()
    assert state["hours"][key]["owner"] == {"cluster": "horeka", "attempt_id": "run2", "generation": 2}
    assert state["hours"][key]["completed"] == [fingerprints[0]]
    with pytest.raises(ValueError, match="stale writer"):
        finish_hours(state, owners={key: owner}, checkpoint=bundle, outcomes=outcomes, terminal=True)


def test_transfer_rejects_corruption_and_conflicting_mirror(tmp_path):
    root, first, second, _ = setup_exchange(tmp_path)
    bundle = first.upload(root, list(build_manifest(root)["files"]), outcomes={}, metadata={})
    destination = tmp_path / "mirror"
    second.download(bundle, destination, stripes=4)
    (destination / "contract.json").write_text("wrong")
    with pytest.raises(ValueError, match="conflicting"):
        second.download(bundle, destination, stripes=4)
    manifest = first.manifest(bundle)
    sha = manifest["files"]["contract.json"]["sha256"]
    first.store.commit(first.store.head(), {f"exchange/objects/{sha}": b"corrupt"}, "corrupt fixture")
    with pytest.raises(ValueError, match="corrupt"):
        second.download(bundle, tmp_path / "another", stripes=4)


def test_replanning_freezes_owned_packages_and_rejects_stale_publication(tmp_path):
    root, first, _, value = setup_exchange(tmp_path)
    key = value["packages"][0]["hour_id"]
    _, initial = first.snapshot()
    stale = plan(root, initial)
    owned = claim_hours(initial, hour_ids=[key], cluster="horeka", attempt_id="run1", supported_models=["qwen38"])
    with pytest.raises(ValueError, match="stale plan"):
        install_plan(owned, stale, cluster="jupiter")
    revised = plan(root, owned)
    assert [p["keyword_id"] for p in revised["packages"]] == ["beta"]
    installed = install_plan(owned, revised, cluster="jupiter")
    assert installed["hours"][key] == owned["hours"][key]
    old_beta = value["packages"][1]["hour_id"]
    assert installed["hours"][old_beta]["status"] == "superseded"
    assert installed["hours"][old_beta]["superseded_by"] == [revised["packages"][0]["hour_id"]]
    with pytest.raises(ValueError, match="JUPITER chief"):
        install_plan(owned, revised, cluster="horeka")


def snapshot(cluster="jupiter", jobs=None):
    return {"cluster": cluster, "complete": True, "captured_at_epoch": 1000,
            "jobs": jobs or [], "owners": []}


def healthy(cluster="jupiter"):
    return {"cluster": cluster, "captured_at_epoch": 1000, "safe_to_admit": True, "quota_verified": True}


def test_admission_limits_are_per_cluster_and_gate_actual_starts():
    state = empty_registry()
    request = {"cluster": "jupiter", "attempt_id": "one"}
    admitted = admission(state, request, snapshot(), healthy(), now=1000)
    with pytest.raises(ValueError, match="unconfirmed"):
        admission(admitted, {**request, "attempt_id": "two"}, snapshot(), healthy(), now=1000)
    other = admission(admitted, {"cluster": "horeka", "attempt_id": "two"}, snapshot("horeka"), healthy("horeka"), now=1000)
    assert set(other["admission"]) == {"jupiter", "horeka"}
    jobs = [{"job_id": "1", "attempt_id": "one", "state": "RUNNING", "start_epoch": 900}]
    with pytest.raises(ValueError, match="ten-minute"):
        admission(admitted, {**request, "attempt_id": "two"}, snapshot(jobs=jobs), healthy(), now=1000)
    jobs = [{"job_id": str(i), "state": "RUNNING", "start_epoch": 100} for i in range(5)]
    with pytest.raises(ValueError, match="concurrency"):
        admission(state, request, snapshot(jobs=jobs), healthy(), now=1000)
    attached = admission(state, request, snapshot(jobs=jobs), healthy(), now=1000, existing_job_id="0")
    assert attached["admission"]["jupiter"]["tickets"]["one"]["existing_job_id"] == "0"


def test_failed_storage_blocks_admission_and_stale_evidence_is_rejected():
    request = {"cluster": "jupiter", "attempt_id": "one"}
    with pytest.raises(ValueError, match="storage admission"):
        admission(empty_registry(), request, snapshot(), {**healthy(), "safe_to_admit": False}, now=1000)
    with pytest.raises(ValueError, match="stale"):
        admission(empty_registry(), request, snapshot(), healthy(), now=2000)


def test_finite_sync_releases_only_after_terminal_scheduler_evidence(tmp_path):
    root, first, _, value = setup_exchange(tmp_path)
    key = value["packages"][0]["hour_id"]
    first.transact("reserve", {}, lambda state: claim_hours(state, hour_ids=[key], cluster="jupiter",
                                                           attempt_id="one", supported_models=["qwen38"]))
    _, state = first.snapshot()
    directory = tmp_path / "attempt"
    directory.mkdir()
    tasks = {fp: value["tasks"][fp] for fp in value["packages"][0]["task_fingerprints"]}
    (directory / "tasks.json").write_bytes(canonical(tasks))
    attempt = {"dataset_root": str(root), "request": {"attempt_dir": str(directory)},
               "ledger_stripes": 4, "writer_id": "jupiter-one", "cluster": "jupiter", "attempt_id": "one",
               "owners": {key: state["hours"][key]["owner"]}}
    for task in tasks.values():
        complete(root, task, "jupiter-one")
    now = int(time.time())
    current = {**snapshot(), "captured_at_epoch": now}
    health = {**healthy(), "captured_at_epoch": now}
    first_result = sync_once(first, attempt, current, health)
    assert first_result["status"] == "owned"
    assert first.snapshot()[1]["hours"][key]["status"] == "awaiting_sync"
    current["owners"] = [{"owner_id": "jupiter-one", "state": "COMPLETED", "cluster": "jupiter", "job_id": "7"}]
    final_result = sync_once(first, attempt, current, health)
    assert final_result["status"] == "released"
    assert first.snapshot()[1]["hours"][key]["status"] == "complete"
    assert first.snapshot()[1]["hours"][key]["owner"] is None


def test_hour_guard_rejects_changed_scientific_identity_before_claim(tmp_path, monkeypatch):
    root = data(tmp_path / "dataset")
    tasks, _, _ = inventory(root, stripes=4)
    allowed = tmp_path / "tasks.json"
    allowed.write_bytes(canonical({tasks[0]["fingerprint"]: tasks[0]}))
    monkeypatch.setenv("GEODML_HOUR_TASKS", str(allowed))
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
    accepted = ledger.claim(ClaimIdentity(**tasks[0]["claim_identity"]), owner_id="jupiter-run")
    assert accepted.status == "owned"
    with pytest.raises(ValueError, match="outside the reserved"):
        ledger.claim(ClaimIdentity(**tasks[1]["claim_identity"]), owner_id="jupiter-run")


def test_terminal_failure_without_result_reference_transfers_as_failure(tmp_path):
    root, first, second, value = setup_exchange(tmp_path)
    inputs = first.upload(root, list(build_manifest(root)["files"]), outcomes={}, metadata={})
    destination = tmp_path / "horeka"
    second.download(inputs, destination, stripes=4)
    task = next(iter(value["tasks"].values()))
    identity = ClaimIdentity(**task["claim_identity"])
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
    claimed = ledger.claim(identity, owner_id="jupiter-failed")
    ledger.transition(claimed.claim, state="terminal_failed", detail={"error": "bounded model failure", "attempts": 3})
    names, outcomes = checkpoint_files(root, value["tasks"], stripes=4, writer_id="jupiter-failed")
    bundle = first.upload(root, names, outcomes=outcomes, metadata={})
    second.download(bundle, destination, stripes=4)
    remote_ledger = StripedTaskLedger(destination / "control/task-ledger", stripe_count=4)
    assert remote_ledger.claim(identity, owner_id="horeka-next").status == "terminal_failed"
    assert inventory(destination, stripes=4)[1] == set()


def test_missing_terminal_artifact_holds_ownership_instead_of_rescheduling(tmp_path):
    root, first, _, value = setup_exchange(tmp_path)
    key = value["packages"][0]["hour_id"]
    first.transact("claim", {}, lambda state: claim_hours(state, hour_ids=[key], cluster="jupiter",
                                                         attempt_id="broken", supported_models=["qwen38"]))
    task = value["tasks"][value["packages"][0]["task_fingerprints"][0]]
    complete(root, task, "jupiter-broken")
    saved = next((root / "data/generations").glob("part-jupiter-broken-*.jsonl"))
    saved.write_text("damaged\n")
    directory = tmp_path / "attempt"
    directory.mkdir()
    (directory / "tasks.json").write_bytes(canonical(value["tasks"]))
    owner = first.snapshot()[1]["hours"][key]["owner"]
    attempt = {"dataset_root": str(root), "request": {"attempt_dir": str(directory)}, "ledger_stripes": 4,
               "writer_id": "jupiter-broken", "cluster": "jupiter", "attempt_id": "broken", "owners": {key: owner}}
    now = int(time.time())
    terminal = {**snapshot(), "captured_at_epoch": now, "owners": [
        {"owner_id": "jupiter-broken", "cluster": "jupiter", "state": "FAILED", "job_id": "8"}]}
    # Corrupt evidence may fail validation before a blocked report can be constructed.
    try:
        report = sync_once(first, attempt, terminal, {**healthy(), "captured_at_epoch": now})
        assert report["status"] == "blocked"
    except ValueError as error:
        assert "hash" in str(error) or "checksum" in str(error)
    assert first.snapshot()[1]["hours"][key]["owner"] == owner


def test_unvalidated_horeka_model_cannot_be_reserved(tmp_path):
    _, first, _, value = setup_exchange(tmp_path)
    key = value["packages"][0]["hour_id"]
    with pytest.raises(ValueError, match="not validated"):
        first.transact("bad-model", {}, lambda state: claim_hours(state, hour_ids=[key], cluster="horeka",
                                                                  attempt_id="one", supported_models=[]))
    assert first.snapshot()[1]["hours"][key]["status"] == "available"


def test_running_without_observed_start_does_not_open_admission_gate():
    state = empty_registry()
    state["admission"]["jupiter"] = {"pending_attempt": "one"}
    request = {"cluster": "jupiter", "attempt_id": "two"}
    jobs = [{"job_id": "1", "attempt_id": "one", "state": "RUNNING", "start_epoch": None}]
    with pytest.raises(ValueError, match="start is unconfirmed"):
        admission(state, request, snapshot(jobs=jobs), healthy(), now=1000)


def test_periodic_sealing_is_opt_in_and_preserves_record_references(tmp_path, monkeypatch):
    root = data(tmp_path / "dataset")
    monkeypatch.setenv("GEODML_DATASET_SEAL_INTERVAL_SECONDS", "0.001")
    writer = FinalDatasetWriter(root, writer_id="periodic")
    first = writer.append("generations", {"answer": "first"}, transaction_id="one")
    time.sleep(0.01)
    second = writer.append("generations", {"answer": "second"}, transaction_id="two")
    from analysis.interpretability.pipeline.agentic_dataset import (
        verify_record_reference,
    )
    assert verify_record_reference(root, first) is True
    assert first["shard_sequence"] == 0
    assert second["shard_sequence"] == 1
    writer.seal()
    assert verify_record_reference(root, second) is True


def test_nemotron_dependencies_wait_for_verified_generations(tmp_path):
    root = data(tmp_path / "dataset")
    tasks, done, blocked = inventory(root, stripes=4)
    source = tasks[0]
    judge = {**source, "model": "nemotron", "configuration_sha256": "judge-config",
             "task_id": "judge-one", "dependency_fingerprints": [source["fingerprint"]],
             "claim_identity": {**source["claim_identity"], "model_id": "judge-model"}}
    from analysis.interpretability.pipeline.agentic_task_ledger import (
        identity_fingerprint,
    )

    judge["fingerprint"] = identity_fingerprint(ClaimIdentity(**judge["claim_identity"]))
    values = {**calibration(), "nemotron": {**calibration()["qwen38"], "scientific_config_sha256": "judge-config"}}
    args = {"tasks": [*tasks, judge], "calibration": values, "registry": empty_registry(),
            "contract": {}, "blocked": blocked, "source_commit": "a" * 40, "input_bundle": "input"}
    waiting = build_plan(completed=done, **args)
    assert waiting["deferred"][judge["fingerprint"]] == "generation_dependency"
    ready = build_plan(completed={source["fingerprint"]}, **args)
    assert [(row["model"], row["task_fingerprints"]) for row in ready["packages"] if row["model"] == "nemotron"] == [
        ("nemotron", [judge["fingerprint"]])]


def test_scheduler_maps_shared_hour_terminal_owner_and_explicit_interactive_job():
    from analysis.scripts.capture_agentic_scheduler_snapshot import capture

    output = iter([
        "21|interactive|RUNNING|2026-09-24T10:00:00|None|\n",
        "20|geodml-hours-run1|TIMEOUT|2026-09-24T09:00:00|geodml-hours:horeka:run1|\n",
    ])
    observed = capture(plan={"plan_id": "shared-hours"}, since="2026-09-24",
                       include_job_ids=["21"], runner=lambda argv: next(output))
    assert [row["job_id"] for row in observed["jobs"]] == ["21"]
    owner = next(row for row in observed["owners"] if row["owner_id"] == "horeka-run1")
    assert (owner["cluster"], owner["attempt_id"], owner["state"]) == ("horeka", "run1", "TIMEOUT")


def test_reserve_execute_and_sync_real_cpu_child_across_two_hour_packages(tmp_path, monkeypatch):
    """Real subprocess/ledger/files; only hardware and scheduler queries are fixtures."""
    import subprocess
    from datetime import datetime, timezone
    from pathlib import Path

    from analysis.interpretability.pipeline import (
        agentic_hour_runtime as runtime_module,
    )
    from analysis.interpretability.pipeline.agentic_hours import digest
    from analysis.scripts.manage_agentic_hours import stage
    from analysis.scripts.search_vllm_stage import build_profile

    root = data(tmp_path / "jupiter")
    reference = build_profile(
        stage="qwen-generator", model_id="pinned-model", model_revision="a" * 40,
        vllm_executable="/fixture/vllm", vllm_version="fixture", vllm_help="",
        visible_gpus=[{"index": i, "uuid": f"GPU-{i}", "name": "GH200", "memory_total_mib": 97871} for i in range(4)],
        cuda_visible_devices="0,1,2,3", expected_gpu_name_pattern="GH200", max_model_len=4096,
        request_concurrency=1,
    )
    reference_path = tmp_path / "reference.json"
    reference_path.write_bytes(canonical(reference))
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    exchange = Exchange(MemoryHub(), tmp_path / "journal")
    bundle = exchange.upload(root, list(build_manifest(root)["files"]), outcomes={}, metadata={})
    values = calibration()
    values["qwen38"]["reference_profile_sha256"] = reference_sha
    tasks, completed, blocked = inventory(root, stripes=4)
    value = build_plan(tasks=tasks, calibration=values, registry=empty_registry(), contract={}, completed=completed,
                       blocked=blocked, source_commit="b" * 40, input_bundle=bundle)
    exchange.transact("plan", {}, lambda state: install_plan(state, value, cluster="jupiter"),
                      {f"coordination/plans/{value['plan_id']}.json": canonical(value)})
    repo = tmp_path / "code"
    leaf = repo / "analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"
    leaf.parent.mkdir(parents=True)
    script = repo / "cpu_worker.py"
    script.write_text('''import json, os
from pathlib import Path
from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
root = Path(os.environ["GEODML_DATASET_ROOT"])
tasks = json.loads(Path(os.environ["GEODML_HOUR_TASKS"]).read_bytes())
writer_id = os.environ["GEODML_DATASET_WRITER_ID"]
ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=4)
writer = FinalDatasetWriter(root, writer_id=writer_id)
completed = []
for line in Path(os.environ["GEODML_WORKER_TASKS"]).read_text().splitlines():
    queued = json.loads(line)
    task = tasks[queued["geodml_task_fingerprint"]]
    claim = ledger.claim(ClaimIdentity(**task["claim_identity"]), owner_id=writer_id)
    if claim.status == "completed":
        continue
    ref = writer.append("generations", {"answer": "CPU fixture"}, transaction_id=task["task_id"])
    ledger.transition(claim.claim, state="completed", record_references=[ref])
    completed.append(task["task_id"])
writer.seal()
output = Path(os.environ["GEODML_WORKER_OUTPUT"])
output.mkdir(parents=True)
(output / "observed.json").write_text(json.dumps({"completed": completed}))
''')
    leaf.write_text('exec python3 cpu_worker.py\n')
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "CPU execution fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    request = {"attempt_id": "cpu-test", "cluster": "jupiter", "mode": "interactive",
               "hour_ids": [p["hour_id"] for p in value["packages"]], "attempt_dir": str(tmp_path / "attempt"),
               "git_commit": commit, "since": "2026-09-24",
               "approval": {"status": "approved", "walltime_seconds": 3600, "maximum_gpu_hours": 4.0,
                            "evidence": "fixture-only", "estimate": "synthetic test; no GPUs used",
                            "resources": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "all"}}}
    cluster = {"cluster": "jupiter", "cache_root": str(tmp_path / "cache"),
               "minimum_cache_free_bytes": 0, "minimum_cache_free_inodes": 0,
               "validated_models": {"qwen38": {"evidence": "fixture", "reference_profile_sha256": reference_sha}}}
    attempt = stage(exchange, request=request, profile=cluster, runtime={"SEARCH_AGENTIC_PROFILE": str(reference_path)},
                    reference_profile=reference_path, dataset=root, repository=repo, operation_id="reserve", stripes=4)
    assert [json.loads(line)["cell_id"] for line in (tmp_path / "attempt/backlog.jsonl").read_text().splitlines()] == [
        "alpha-0", "alpha-1", "beta-0", "beta-1"]
    attempt["admission_ticket"] = {"request_sha256": digest(request), "attempt_sha256": digest(attempt), "existing_job_id": "42"}
    attempt_path = tmp_path / "attempt/attempt.json"
    attempt_path.write_bytes(canonical(attempt))
    original = subprocess.check_output
    now = int(time.time())
    def checked(command, **kwargs):
        if command[0] == "scontrol":
            return f"JobId=42 NumNodes=1 StartTime={datetime.fromtimestamp(now - 10, timezone.utc).isoformat()} EndTime={datetime.fromtimestamp(now + 3500, timezone.utc).isoformat()}"
        if command[0] == "nvidia-smi":
            return "GH200\n" * 4
        return original(command, **kwargs)
    monkeypatch.setattr(runtime_module, "verify_boundary", lambda *args, **kwargs: {"status": "fixture"})
    monkeypatch.setattr(runtime_module.subprocess, "check_output", checked)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    assert runtime_module.execute(attempt_path, expected_job_id="42") == 0
    assert json.loads((tmp_path / "attempt/output/observed.json").read_bytes()) == {
        "completed": ["alpha-0", "alpha-1", "beta-0", "beta-1"]}
    execution = json.loads((tmp_path / "attempt/execution.json").read_bytes())
    assert execution["cache_cleanup"] == "removed_attempt_owned_cache"
    assert (tmp_path / "attempt/output/observed.json").is_file()
    assert not (tmp_path / "cache/jupiter-cpu-test").exists()
    with pytest.raises(ValueError, match="already started"):
        runtime_module.execute(attempt_path, expected_job_id="42")
    terminal = {**snapshot(), "captured_at_epoch": int(time.time()), "owners": [
        {"owner_id": "jupiter-cpu-test", "cluster": "jupiter", "state": "COMPLETED", "job_id": "42"}]}
    result = sync_once(exchange, attempt, terminal, {**healthy(), "captured_at_epoch": int(time.time())})
    assert result["status"] == "released"
    assert [hour["status"] for hour in exchange.snapshot()[1]["hours"].values()] == ["complete", "complete"]

    # A100 may change hardware, but cannot silently shrink the context.
    changed = json.loads(reference_path.read_bytes())
    changed["serving"]["max_model_len"] = 2048
    reference_path.write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="hash mismatch|profile file changed"):
        runtime_module.verify_serving(attempt)


def test_prepared_bootstrap_resume_preserves_artifacts_and_recounts(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from analysis.scripts import run_jupiter_prepared_backlog as resume
    root = data(tmp_path / "dataset")
    wave = tmp_path / "wave"
    wave.mkdir()
    manifest = wave / "run_manifest.json"
    manifest.write_text(json.dumps({"format_version": "geodml-inference-wave-v2", "dispatch_mode": "backlog"}))
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"GEODML_DATASET_ROOT": str(root), "GEODML_WAVE_ROOT": str(wave),
                                   "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY": "0"}))
    scheduler_file = tmp_path / "scheduler.json"
    scheduler_file.write_text(json.dumps({"complete": True, "captured_at_epoch": int(time.time()), "owners": []}))
    preserved = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    preserved[str(manifest)] = manifest.read_bytes()
    monkeypatch.setattr(resume, "verify", lambda cluster: {"status": "fixture"})
    monkeypatch.setenv("GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY", "0")
    monkeypatch.setenv("GEODML_DATASET_ROOT", "before")
    monkeypatch.setenv("GEODML_WAVE_ROOT", "before")
    launched = []
    monkeypatch.setattr(resume.os, "execv", lambda *args: launched.append(args))
    resume.main(["--runtime-environment", str(runtime), "--scheduler-snapshot", str(scheduler_file), "--ledger-stripes", "4"])
    report = json.loads(capsys.readouterr().out)
    assert (report["current_registered"], report["current_verified_completed"], report["current_blocked"]) == (4, 0, 0)
    assert all(Path(name).read_bytes() == raw for name, raw in preserved.items())
    assert len(launched) == 1 and launched[0][1][-1].endswith("run_inference_wave_worker.sbatch")
    assert resume.os.environ["GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"] == "1"


def stage_wave_fixture(tmp_path, hub):
    """Published two-package plan, validated cluster profile and pinned fixture repository."""
    import subprocess

    from analysis.scripts.search_vllm_stage import build_profile

    root = data(tmp_path / "jupiter")
    reference = build_profile(
        stage="qwen-generator", model_id="pinned-model", model_revision="a" * 40,
        vllm_executable="/fixture/vllm", vllm_version="fixture", vllm_help="",
        visible_gpus=[{"index": i, "uuid": f"GPU-{i}", "name": "GH200", "memory_total_mib": 97871} for i in range(4)],
        cuda_visible_devices="0,1,2,3", expected_gpu_name_pattern="GH200", max_model_len=4096,
        request_concurrency=1,
    )
    reference_path = tmp_path / "reference.json"
    reference_path.write_bytes(canonical(reference))
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    exchange = Exchange(hub, tmp_path / "journal")
    bundle = exchange.upload(root, list(build_manifest(root)["files"]), outcomes={}, metadata={})
    values = calibration()
    values["qwen38"]["reference_profile_sha256"] = reference_sha
    tasks, completed, blocked = inventory(root, stripes=4)
    value = build_plan(tasks=tasks, calibration=values, registry=empty_registry(), contract={},
                       completed=completed, blocked=blocked, source_commit="b" * 40, input_bundle=bundle)
    exchange.transact("plan", {}, lambda state: install_plan(state, value, cluster="jupiter"),
                      {f"coordination/plans/{value['plan_id']}.json": canonical(value)})
    repo = tmp_path / "code"
    leaf = repo / "analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("exec true\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    assert len(value["packages"]) >= 2
    cluster = {"cluster": "jupiter", "cache_root": str(tmp_path / "cache"),
               "minimum_cache_free_bytes": 0, "minimum_cache_free_inodes": 0,
               "validated_models": {"qwen38": {"evidence": "fixture", "reference_profile_sha256": reference_sha}}}

    def request(name, hour_ids):
        return {"attempt_id": name, "cluster": "jupiter", "mode": "interactive",
                "hour_ids": hour_ids, "attempt_dir": str(tmp_path / f"attempt-{name}"),
                "git_commit": commit, "since": "2026-09-24",
                "approval": {"status": "approved", "walltime_seconds": 3600, "maximum_gpu_hours": 4.0,
                             "evidence": "fixture-only", "estimate": "synthetic test; no GPUs used",
                             "resources": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "all"}}}

    runtime = {"SEARCH_AGENTIC_PROFILE": str(reference_path)}
    kwargs = {"profile": cluster, "runtime": runtime, "reference_profile": reference_path,
              "dataset": root, "repository": repo, "stripes": 4}
    return exchange, value["packages"], request, kwargs


def test_stage_wave_reserves_the_whole_wave_in_one_transaction(tmp_path):
    from pathlib import Path

    from analysis.scripts.manage_agentic_hours import stage_wave

    hub = MemoryHub()
    exchange, packages, request, kwargs = stage_wave_fixture(tmp_path, hub)
    wave = [request("wave-1", [packages[0]["hour_id"]]), request("wave-2", [packages[1]["hour_id"]])]
    calls = []
    original = exchange.transact

    def counted(*a, **k):
        calls.append(a[0])
        return original(*a, **k)

    exchange.transact = counted
    revision_before = hub.head()
    attempts = stage_wave(exchange, requests=wave, operation_id="reserve-wave", **kwargs)
    assert calls == ["reserve-wave"]
    assert int(hub.head()) == int(revision_before) + 1
    _, registry = exchange.snapshot()
    assert [attempt["attempt_id"] for attempt in attempts] == ["wave-1", "wave-2"]
    for attempt, req in zip(attempts, wave):
        for key in req["hour_ids"]:
            assert registry["hours"][key]["owner"]["attempt_id"] == attempt["attempt_id"]
        assert (Path(req["attempt_dir"]) / "attempt.json").is_file()
        assert (Path(req["attempt_dir"]) / "backlog.jsonl").is_file()
        assert attempt["owners"]
    # Idempotent rerun: same operation id, no new commit, saved manifests returned.
    calls.clear()
    revision_before = hub.head()
    rerun = stage_wave(exchange, requests=wave, operation_id="reserve-wave", **kwargs)
    assert rerun == attempts
    assert calls == ["reserve-wave"]
    assert hub.head() == revision_before
    # Any conflict aborts the whole wave before committing.
    conflict = [request("wave-3", [packages[0]["hour_id"], packages[1]["hour_id"]])]
    with pytest.raises(ValueError):
        stage_wave(exchange, requests=conflict, operation_id="reserve-wave-2", **kwargs)
    _, registry = exchange.snapshot()
    assert registry["hours"][packages[0]["hour_id"]]["owner"]["attempt_id"] == "wave-1"
    assert registry["hours"][packages[1]["hour_id"]]["owner"]["attempt_id"] == "wave-2"
    # Overlapping hour sets inside one wave are rejected before any registry access.
    calls.clear()
    with pytest.raises(ValueError, match="must not share"):
        stage_wave(exchange, requests=[request("wave-4", [packages[0]["hour_id"]]),
                                       request("wave-5", [packages[0]["hour_id"]])],
                   operation_id="reserve-wave-3", **kwargs)
    assert calls == []
