from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    initialize_dataset,
    iter_sealed_rows,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import prepare_jupiter_llama as helper
from analysis.scripts.register_agentic_dataset_tasks import register_generator_tasks
from analysis.tests.test_register_agentic_dataset_tasks import (
    _generator_inputs,
    _priority,
)


def registered(tmp_path):
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    inputs = _generator_inputs(tmp_path, model_id=helper.MODEL["model_id"], revision=helper.MODEL["model_revision"])
    inputs = replace(inputs, final_max_tokens=4096, request_concurrency=4, cell_concurrency=12)
    for slug, values in [("qwen38", replace(inputs, model_id="Qwen/Qwen3.8-27B", model_revision="a" * 40)),
                         ("llama4", inputs)]:
        register_generator_tasks(dataset_root=root, model_slug=slug, inputs=values,
                                 keyword_priority_path=_priority(tmp_path), writer_id="register-" + slug)
    return root, inputs


def test_freezes_all_missing_llama_cells_and_keeps_qwen_separate(tmp_path):
    root, _ = registered(tmp_path)
    tasks = [row for row in iter_sealed_rows(root, "task_definitions") if row["model"] == "llama4"]
    results = FinalDatasetWriter(root, writer_id="old-results")
    ref = results.append("generations", {"answer": "prior result"}, transaction_id="old")
    results.seal()
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=8)
    claim = ledger.claim(ClaimIdentity(**tasks[0]["claim_identity"]), owner_id="old").claim
    ledger.transition(claim, state="completed", record_references=[ref])
    ledger.claim(ClaimIdentity(**tasks[1]["claim_identity"]), owner_id="active")
    output = tmp_path / "prepared"
    summary = helper.freeze_backlog(root, output, stripes=8)
    assert (summary["registered"], summary["completed"], summary["blocked"], summary["eligible"]) == (24, 1, 1, 22)
    assert summary["hour_packages"] is None
    rows = [json.loads(line) for line in (output / "wave/backlog.jsonl").read_text().splitlines()]
    assert len(rows) == 22
    assert {row["cell_id"] for row in rows} == {row["task_id"] for row in tasks[2:]}
    keywords = [row["geodml_keyword_id"] for row in rows]
    assert keywords == sorted(keywords)
    assert len(list(iter_sealed_rows(root, "task_definitions"))) == 48


def test_runtime_preserves_llama_protocol(tmp_path):
    runtime = {key: str(tmp_path / key) for key in helper.FILES}
    runtime.update(SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT=str(tmp_path / helper.BGE_REVISION),
                   SEARCH_AGENTIC_CROSS_ENCODER_REVISION=helper.BGE_REVISION,
                   SEARCH_AGENTIC_PROMPT_COUNT="26008", SEARCH_AGENTIC_PROMPT_SELECTION_SEED="20260912")
    profile = {"model": helper.MODEL, "serving": {"tensor_parallel_size": 4, "data_parallel_size": 1,
               "dtype": "bfloat16", "max_model_len": 40960, "request_concurrency": 4}}
    inputs = helper.inputs_from_runtime(tmp_path, runtime, profile)
    assert (inputs.prompt_count, inputs.final_max_tokens, inputs.disable_thinking) == (26008, 4096, False)
    assert (inputs.cell_concurrency, inputs.request_concurrency) == (12, 4)
    profile["serving"]["tensor_parallel_size"] = 2
    with pytest.raises(ValueError, match="serving settings"):
        helper.inputs_from_runtime(tmp_path, runtime, profile)


def test_prepared_file_tampering_fails_even_after_a_warm_receipt(tmp_path, monkeypatch):
    root, output = tmp_path / "dataset", tmp_path / "out"
    root.mkdir()
    output.mkdir()
    payload = output / "runtime.json"
    payload.write_text('{"original": true}')
    record = {"git_commit": "a" * 40, "dataset_root": str(root),
              "files": {str(payload): helper.identity(payload)}, "dataset_files": {}, "external_files": {}}
    (output / "preparation.json").write_text(json.dumps(record))
    monkeypatch.setattr(helper, "clean_commit", lambda: "a" * 40)
    helper.verify_prepared(output)
    helper.verify_prepared(output)
    payload.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="prepared file changed"):
        helper.verify_prepared(output)


@pytest.mark.parametrize("snapshot", [
    {"complete": False, "captured_at_epoch": 1000, "jobs": [], "owners": []},
    {"complete": True, "captured_at_epoch": 1, "jobs": [], "owners": []},
    {"complete": True, "captured_at_epoch": 1000, "jobs": [{"state": "PENDING"}], "owners": []},
    {"complete": True, "captured_at_epoch": 1000, "jobs": [], "owners": [{"start_epoch": 900}]},
])
def test_scheduler_gate_preserves_live_jobs_and_observed_start_gap(snapshot):
    with pytest.raises(ValueError):
        helper.scheduler_gate(snapshot, 1000)


def submission_fixture(tmp_path, monkeypatch):
    root, output = tmp_path / "dataset", tmp_path / "out"
    (root / "control").mkdir(parents=True)
    output.mkdir()
    record = {"git_commit": "a" * 40, "dataset_root": str(root), "summary": {"eligible": 100}}
    monkeypatch.setattr(helper, "verify_prepared", lambda path: record)
    class Hub:
        def __init__(self, repo):
            pass
        def head(self):
            return "b" * 40
        def read(self, *args):
            return b'{"hours": {}}'
    monkeypatch.setattr(helper, "HubStore", Hub)
    snapshots = iter([[], [{"job_id": "1234", "held": True, "state": "PENDING"}]])
    monkeypatch.setattr(helper, "current_scheduler", lambda since: {
        "complete": True, "captured_at_epoch": int(helper.time.time()), "owners": [], "jobs": next(snapshots)})
    monkeypatch.setattr(helper, "storage_health", lambda root: {"safe_to_admit": True, "quota_verified": False})
    args = SimpleNamespace(output=output, approval="Approved one first one-hour Llama job", approved_walltime="01:00:00",
                           hf_repo="private/test", account="test", partition="booster", since="2026-09-24")
    return args, root, output


def test_submits_one_held_job_records_it_before_release_and_cannot_duplicate(tmp_path, monkeypatch):
    args, root, output = submission_fixture(tmp_path, monkeypatch)
    calls = []
    def run(argv):
        calls.append(argv)
        if argv[0] == "sbatch":
            assert helper.read(output / "submission.json")["status"] == "submission_requested"
            assert "--hold" in argv and "--time=01:00:00" in argv and "--gres=gpu:4" in argv
            return "1234"
        assert argv == ["scontrol", "release", "1234"]
        assert helper.read(output / "submission.json")["job_id"] == "1234"
        return ""
    monkeypatch.setattr(helper, "command", run)
    result = helper.submit(args)
    assert result["job_id"] == "1234" and result["maximum_gpu_hours"] == 4
    with pytest.raises(ValueError, match="already requested"):
        helper.submit(args)
    assert len(calls) == 2
    assert helper.read(root / "control/llama-first-allocation.json")["status"] == "submitted"


def test_uncertain_submission_persists_intent_and_never_retries(tmp_path, monkeypatch):
    args, root, _output = submission_fixture(tmp_path, monkeypatch)
    def broken(argv):
        raise RuntimeError("connection lost after Slurm may have accepted")
    monkeypatch.setattr(helper, "command", broken)
    with pytest.raises(RuntimeError):
        helper.submit(args)
    with pytest.raises(ValueError, match="already requested"):
        helper.submit(args)
    assert helper.read(root / "control/llama-first-allocation.json")["status"] == "submission_requested"


def test_existing_hf_llama_packages_prevent_bootstrap_submission(tmp_path, monkeypatch):
    args, _, _ = submission_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(helper.HubStore, "read", lambda *args: b'{"hours":{"h":{"model":"llama4"}}}')
    with pytest.raises(ValueError, match="coordinated dispatcher"):
        helper.submit(args)


def test_execute_checks_boundary_before_reading_any_prepared_files(tmp_path, monkeypatch):
    from analysis.scripts import verify_inference_allocation
    def fail(cluster):
        assert cluster == "jupiter"
        raise ValueError("not a compute allocation")
    monkeypatch.setattr(verify_inference_allocation, "verify", fail)
    with pytest.raises(ValueError, match="not a compute allocation"):
        helper.execute(tmp_path / "nonexistent")


def test_full_preparation_adds_llama_to_existing_population_without_allocating(tmp_path, monkeypatch):
    from analysis.tests.test_search_vllm_stage import profile
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="population", acceptance_policy_id="v2")
    inputs = _generator_inputs(tmp_path, model_id="Qwen/Qwen3.8-27B", revision="a" * 40)
    priority = _priority(tmp_path)
    register_generator_tasks(dataset_root=root, model_slug="qwen38", inputs=inputs,
                             keyword_priority_path=priority, writer_id="qwen")
    before = {str(p): p.read_bytes() for p in (root / "data").rglob("*") if p.is_file()}
    bge = tmp_path / helper.BGE_REVISION
    bge.mkdir()
    (bge / "model.safetensors").write_bytes(b"test weights")
    runtime = dict(zip(helper.FILES, map(str, [inputs.search_snapshots["duckduckgo"], inputs.search_snapshots["searxng"],
                                             inputs.prompts_jsonl, inputs.selection_records_jsonl]), strict=True))
    runtime.update(SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT=str(bge), SEARCH_AGENTIC_CROSS_ENCODER_REVISION=helper.BGE_REVISION,
                   SEARCH_AGENTIC_PROMPT_COUNT="2", SEARCH_AGENTIC_PROMPT_SELECTION_SEED="20260912")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(runtime))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile(**helper.MODEL, max_model_len=40960, request_concurrency=4,
                                              tokenizer_mode=None, config_format=None, load_format=None, attention_backend=None)))
    cache = tmp_path / "cache"
    snapshot = cache / ("models--" + helper.MODEL["model_id"].replace("/", "--")) / "snapshots" / helper.MODEL["model_revision"]
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}")
    (snapshot / "model.safetensors.index.json").write_text('{"weight_map":{"x":"weights.safetensors"}}')
    (snapshot / "weights.safetensors").write_bytes(b"test model")
    for name in ("HF_HUB_CACHE", "GEODML_CACHE_ROOT", "ACL_ARR_VENV"):
        monkeypatch.setenv(name, str(cache))
    monkeypatch.setattr(helper, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(helper.importlib.metadata, "version", lambda name: "6.0.1" if name == "sentence-transformers" else "0.28.0")
    def no_scheduler(argv):
        raise AssertionError("preparation must not allocate or query Slurm")
    monkeypatch.setattr(helper, "command", no_scheduler)
    hub = tmp_path / "recovery"
    (hub / "data").mkdir(parents=True)
    for name in ("generations", "generation_aliases"):
        with gzip.open(hub / f"data/{name}.jsonl.gz", "wt"):
            pass
    files = {str(p.relative_to(hub)): hashlib.sha256(p.read_bytes()).hexdigest() for p in hub.rglob("*") if p.is_file()}
    (hub / "publication-manifest.json").write_text(json.dumps({"format_version": "geodml-recovery-publication-v1", "files": files}))
    args = SimpleNamespace(dataset=root, output=tmp_path / "prepared", reference_runtime=reference,
                           profile=profile_path, keyword_priority=priority, recovery_hub=hub)
    result = helper.prepare(args)
    assert (result["registered"], result["eligible"], result["allocation_submitted"]) == (24, 24, False)
    assert all(helper.Path(p).read_bytes() == raw for p, raw in before.items())
    saved_runtime = helper.read(args.output / "runtime.json")
    assert saved_runtime["GEODML_DATASET_ROOT"] == str(root)
    assert saved_runtime["GEODML_START_MARGIN_SECONDS"] == "300"
    assert "GEODML_INFERENCE_CLAIM_ROOT" not in saved_runtime
    assert helper.read(args.output / "recovery.json")["ledger_completions_created"] == 0
    with pytest.raises(FileExistsError):
        helper.prepare(args)


def test_execute_uses_slurm_actual_deadline_and_clears_legacy_claims(tmp_path, monkeypatch):
    from analysis.scripts import verify_inference_allocation
    monkeypatch.setattr(verify_inference_allocation, "verify", lambda cluster: {"verified": True})
    monkeypatch.setattr(helper, "verify_prepared", lambda path: {"git_commit": "a" * 40})
    monkeypatch.setattr(helper.os, "environ", dict(helper.os.environ))
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_END_TIME", "9999999999")
    monkeypatch.setenv("GEODML_INFERENCE_CLAIM_ROOT", "/old/claims")
    monkeypatch.setenv("SEARCH_AGENTIC_CELL_IDS_JSONL", "/old/cells")
    monkeypatch.setenv("GEODML_ROLE_END_TIME", "9999999999")
    (tmp_path / "submission.json").write_text(json.dumps({"job_id": "1234", "approval": {
        "walltime": "01:00:00", "estimate": helper.ESTIMATE}}))
    (tmp_path / "runtime.json").write_text(json.dumps({"GEODML_EXECUTION_COMMIT": "a" * 40,
                                                     "GEODML_DATASET_ROOT": "/correct/dataset"}))
    monkeypatch.setattr(helper, "command", lambda args: "TimeLimit=01:00:00 NumNodes=1 StartTime=2026-09-25T12:00:00 EndTime=2026-09-25T13:00:00")
    def execute(path, args):
        assert path == "/bin/bash" and args[-1].endswith("run_inference_wave_worker.sbatch")
        assert helper.os.environ["SLURM_JOB_END_TIME"] == str(int(datetime.fromisoformat("2026-09-25T13:00:00").timestamp()))
        assert "GEODML_INFERENCE_CLAIM_ROOT" not in helper.os.environ
        assert "SEARCH_AGENTIC_CELL_IDS_JSONL" not in helper.os.environ
        assert "GEODML_ROLE_END_TIME" not in helper.os.environ
        assert helper.os.environ["GEODML_DATASET_ROOT"] == "/correct/dataset"
        raise SystemExit(0)
    monkeypatch.setattr(helper.os, "execv", execute)
    with pytest.raises(SystemExit) as stopped:
        helper.execute(tmp_path)
    assert stopped.value.code == 0
