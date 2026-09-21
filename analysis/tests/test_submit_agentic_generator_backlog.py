"""Approved schedules, immutable backlogs, and fail-closed Slurm submission."""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from analysis.scripts import submit_agentic_generator_backlog as module


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"path": str(path), "rows": len(rows), "sha256": module._hash(path)}


@pytest.fixture
def backlog(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    for name in (
        "run_agentic_generation_worker.sh", "run_inference_wave_worker.sbatch",
        "run_agentic_search_qwen38_smoke.sh", "run_agentic_search_llama4_smoke.sh",
    ):
        path = repository / "analysis/scripts/slurm/jupiter" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n")
        path.chmod(0o755)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", repository)
    (repository / "analysis/scripts/inference_network_namespace.py").write_text("# CPU-only namespace probe fixture\n")
    cache, profiles = tmp_path / "cache", tmp_path / "profiles"
    for spec in module.MODELS.values():
        _json(profiles / spec["profile"], {
            "model": {key: spec[key] for key in ("model_id", "model_revision")},
            "serving": {"tensor_parallel_size": 4, "data_parallel_size": 1,
                        "host": "127.0.0.1", "port": 8010},
        })
        snapshot = cache / ("models--" + spec["model_id"].replace("/", "--")) / "snapshots" / spec["model_revision"]
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            _json(snapshot / name, {})
        _json(snapshot / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
    monkeypatch.setattr(module, "load_profile", lambda path: json.loads(path.read_text()))
    bge = cache / "models--BAAI--bge-reranker-v2-m3/snapshots" / module.BGE_REVISION
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        _json(bge / name, {})
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("true\n")
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    rows = {key: [] for key in ("prompts", "axis_map", "selection_records")}
    # Use the planned production backlog size, still a small CPU-only fixture.
    for index in range(1200):
        identity = {"candidate_id": f"new-{index:04d}"}
        question = f"Question {index}?"
        rows["prompts"].append({**identity, "question": question, "keyword": "topic",
                                "question_sha256": hashlib.sha256(question.encode()).hexdigest()})
        rows["axis_map"].append(identity)
        rows["selection_records"].append({**identity, "axis_bin": index % 20})
    exclusions = {}
    for label, count in (("original500", 500), ("prior120", 120)):
        exclusions[label] = {
            "prompts": [{"candidate_id": f"{label}-{index:04d}", "question": f"{label} question {index}?",
                         "keyword": "topic"} for index in range(count)],
            "axis_map": [{"candidate_id": f"{label}-{index:04d}"} for index in range(count)],
            "selection_records": [{"candidate_id": f"{label}-{index:04d}", "axis_bin": index % 20}
                                  for index in range(count)],
        }
    for family in (rows, *exclusions.values()):
        for index, (prompt, axis) in enumerate(zip(family["prompts"], family["axis_map"], strict=True)):
            digest = hashlib.sha256(prompt["question"].encode()).hexdigest()
            percentile = (index % 20 + 0.5) / 20
            prompt.update(keyword_id="topic", question_sha256=digest, target_normalized_axis_1=percentile)
            axis.update(text_sha256=digest, axis_1_percentile_0_1=percentile)
    artifacts = {}
    for key, name in (("prompts", "pilot-prompts.jsonl"), ("axis_map", "pilot-axis.jsonl"),
                      ("selection_records", "selection-records.jsonl")):
        path = cohort / name
        artifacts[key] = {**_jsonl(path, rows[key]), "path": name}
    population_sources = {
        key: _jsonl(tmp_path / "population" / f"{key}.jsonl",
                    rows[key] + exclusions["original500"][key] + exclusions["prior120"][key])
        for key in ("prompts", "axis_map")
    }
    provenance = {}
    for label, values in exclusions.items():
        old_artifacts = {key: _jsonl(tmp_path / label / f"{key}.jsonl", records)
                         for key, records in values.items()}
        old_manifest = tmp_path / label / "selection-manifest.json"
        _json(old_manifest, {
            "format_version": "readiness-axis-balanced-pilot-v1" if label == "original500" else "agentic-new-prompt-cohort-v1",
            "prompt_count": len(values["prompts"]), "sources": population_sources,
            "artifacts": old_artifacts,
        })
        provenance[label] = {
            **old_artifacts, "selection_manifest": {"path": str(old_manifest), "sha256": module._hash(old_manifest)},
            "prompt_count": len(values["prompts"]),
        }
    provenance["original500"].pop("axis_map")
    excluded_ids = sorted(row["candidate_id"] for value in exclusions.values() for row in value["prompts"])
    _json(cohort / "selection-manifest.json", {
        "format_version": "agentic-new-prompt-cohort-v1", "prompt_count": 1200,
        "expected_cells_per_model": 14400, "source_git_commit": "a" * 40,
        "artifacts": artifacts, "sources": population_sources,
        "original_excluded_prompt_count": 500, "additional_excluded_prompt_count": 120,
        "excluded_prompt_count": 620,
        "diagnostics": {"axis_bins": 20},
        "excluded_prompt_ids_sha256": hashlib.sha256(module._canonical(excluded_ids)).hexdigest(),
        "exclusion": provenance["original500"], "additional_exclusions": [provenance["prior120"]],
    })
    environment = {
        "PATH": "/usr/bin:/bin", "GEODML_APPROVED_WALLTIME": "03:00:00",
        "GEODML_MAXIMUM_TOTAL_GPU_HOURS": "48",
        "GEODML_ALLOCATION_ESTIMATE": "Four approved 3h jobs including startup and drain; cap 48 GPU-hours.",
        "GEODML_EXECUTION_REPOSITORY": str(repository), "GEODML_EXECUTION_COMMIT": "a" * 40,
        "HF_HUB_CACHE": str(cache), "GEODML_CACHE_ROOT": str(cache), "ACL_ARR_VENV": str(venv),
    }
    for engine in ("DDG", "SEARXNG"):
        snapshot = tmp_path / f"{engine}.jsonl"
        snapshot.write_text(json.dumps({"keyword": "topic", "title": "source", "position": 1,
                                       "url": "https://example.org/source", "snippet": "evidence"}) + "\n")
        environment[f"SEARCH_AGENTIC_{engine}_SNAPSHOT"] = str(snapshot)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "6.0.1")
    monkeypatch.setattr(module.shutil, "which", lambda *args, **kwargs: "/usr/bin/sbatch")
    calls, syncs, probes = [], [], []
    responses = [(0, "12345\n", ""), (0, "12346;testcluster\n", "")]
    root = tmp_path / "run"
    real_sync = module._sync_directory

    def sync(path):
        syncs.append(path)
        real_sync(path)

    monkeypatch.setattr(module, "_sync_directory", sync)

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "a" * 40 if command[3] == "rev-parse" else "", "")
        if command[0] == module.sys.executable:
            assert command[1:] == [str(repository / "analysis/scripts/inference_network_namespace.py"), "--check"]
            assert kwargs["check"] and kwargs["timeout"] == 30
            assert not calls
            probes.append(command)
            return subprocess.CompletedProcess(command, 0, "NETWORK_NAMESPACE_CHECK=PASS\n", "")
        assert command[0] == "sbatch"
        model_root = root / "models" / kwargs["env"]["GEODML_MODEL_SLUG"]
        assert json.loads((model_root / "submission-intent.json").read_text())["status"] == "submission_requested"
        assert model_root in syncs  # Intent directory was fsynced before invoking Slurm.
        calls.append((command, kwargs))
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return subprocess.CompletedProcess(command, *response)

    monkeypatch.setattr(module.subprocess, "run", run)
    return {
        "arguments": {"cohort_root": cohort, "run_root": root, "profile_root": profiles,
                      "source_git_commit": "a" * 40, "account": "test-account",
                      "partition": "test-partition", "submit": True, "environment": environment},
        "environment": environment, "calls": calls, "responses": responses, "syncs": syncs,
        "probes": probes,
    }


def test_four_allocations_share_backlog_and_claims_with_fixed_resources(backlog):
    result = module.submit_backlog(**backlog["arguments"])
    root = backlog["arguments"]["run_root"]
    assert result["status"] == "submitted"
    assert result["allocation_count"] == 4
    assert result["maximum_total_gpu_hours"] == 48
    assert "network_isolation" not in result
    assert result["cells_per_model"] == 14400
    assert len(backlog["calls"]) == 2
    assert not backlog["probes"]
    assert len((root / "tasks.jsonl").read_text().splitlines()) == 14400
    for (slug, job), (command, options) in zip(
        (("qwen38", "12345"), ("llama4", "12346")), backlog["calls"], strict=True,
    ):
        assert {"--array=0-1%2", "--nodes=1", "--ntasks=1", "--cpus-per-task=32",
                "--mem=512G", "--gres=gpu:4", "--time=03:00:00", "--export=ALL",
                "--exclusive", "--parsable", "--no-requeue", "--account=test-account",
                "--partition=test-partition"} <= set(command)
        env = options["env"]
        assert env["GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"] == "1"
        assert env["GEODML_INFERENCE_CLAIM_ROOT"] == str(root / "claims")
        assert env["SEARCH_AGENTIC_PROMPT_COUNT"] == "1200"
        assert env["SEARCH_AGENTIC_CELL_CONCURRENCY"] == "12"
        assert env["SEARCH_AGENTIC_REQUEST_CONCURRENCY"] == "4"
        assert env["SEARCH_AGENTIC_PROMPT_SHARD_COUNT"] == "1"
        assert env["SEARCH_AGENTIC_PROFILE"] == str(root / "profiles" / f"{slug}.json")
        assert options["timeout"] == 60
        model_root = root / "models" / slug
        assert (model_root / "job-id.txt").read_text() == job + "\n"
        wave = json.loads((model_root / "wave/run_manifest.json").read_text())
        assert wave["format_version"] == "geodml-inference-wave-v2"
        assert wave["dispatch_mode"] == "backlog"
        assert wave["worker_count"] == 2 and wave["backlog"]["task_count"] == 14400
    assert (root / "models/qwen38/wave/backlog.jsonl").read_bytes() == (root / "models/llama4/wave/backlog.jsonl").read_bytes()
    assert result["models"]["llama4"]["slurm_cluster"] == "testcluster"
    with pytest.raises(FileExistsError, match="intent already exists"):
        module.submit_backlog(**backlog["arguments"])
    assert len(backlog["calls"]) == 2


def test_ten_twelve_hour_allocations_use_five_slots_per_model(backlog):
    args = {
        **backlog["arguments"],
        "approved_walltime": "12:00:00",
        "workers_per_model": 5,
        "maximum_total_gpu_hours": 480,
    }
    backlog["environment"].update(
        GEODML_APPROVED_WALLTIME="12:00:00",
        GEODML_MAXIMUM_TOTAL_GPU_HOURS="480",
        GEODML_ALLOCATION_ESTIMATE="Ten explicitly approved 12h jobs; cap 480 GPU-hours.",
    )
    result = module.submit_backlog(**args)
    root = args["run_root"]
    assert result["format_version"] == "agentic-generator-backlog-v2"
    assert result["allocation_count"] == 10
    assert result["workers_per_model"] == 5
    assert result["maximum_total_gpu_hours"] == 480
    assert len(backlog["calls"]) == 2
    for command, options in backlog["calls"]:
        assert {"--array=0-4%5", "--time=12:00:00", "--gres=gpu:4"} <= set(command)
        assert "backlog-12h" in next(value for value in command if value.startswith("--job-name="))
        assert options["env"]["GEODML_MAXIMUM_TOTAL_GPU_HOURS"] == "480"
        wave = json.loads((root / "models" / options["env"]["GEODML_MODEL_SLUG"] /
                           "wave/run_manifest.json").read_text())
        assert wave["worker_count"] == 5


def test_single_qwen_one_hour_schedule_submits_one_bounded_worker(backlog):
    args = {
        **backlog["arguments"],
        "approved_walltime": "01:00:00",
        "workers_per_model": 1,
        "maximum_total_gpu_hours": 4,
        "model_slugs": ("qwen38",),
    }
    backlog["environment"].update(
        GEODML_APPROVED_WALLTIME="01:00:00",
        GEODML_MAXIMUM_TOTAL_GPU_HOURS="4",
        GEODML_ALLOCATION_ESTIMATE=(
            "One approved Qwen3.8 worker for one hour; one exclusive Booster node, "
            "four GH200 GPUs, 32 CPUs, 512G RAM; cap 4 GPU-hours."
        ),
    )
    result = module.submit_backlog(**args)
    assert result["allocation_count"] == 1
    assert result["workers_per_model"] == 1
    assert result["maximum_total_gpu_hours"] == 4
    assert set(result["models"]) == {"qwen38"}
    assert len(backlog["calls"]) == 1
    command, options = backlog["calls"][0]
    assert {"--array=0-0%1", "--time=01:00:00", "--gres=gpu:4", "--exclusive"} <= set(command)
    assert options["env"]["GEODML_MODEL_SLUG"] == "qwen38"
    assert options["env"]["GEODML_MAXIMUM_TOTAL_GPU_HOURS"] == "4"
    assert not (args["run_root"] / "models/llama4").exists()


def test_resume_reuses_only_an_exact_prior_shared_claim_root(backlog):
    args = backlog["arguments"]
    prior = args["run_root"].parent / "prior"
    module.submit_backlog(**{**args, "run_root": prior, "submit": False})
    result = module.submit_backlog(**{
        **args,
        "run_root": args["run_root"],
        "resume_from_run_root": prior,
    })
    assert result["resume_from"]["run_root"] == str(prior)
    assert result["claim_root"] == str(prior / "claims")
    for _, options in backlog["calls"]:
        assert options["env"]["GEODML_INFERENCE_CLAIM_ROOT"] == str(prior / "claims")


def test_resume_rejects_a_different_prior_queue_before_slurm_submission(backlog):
    args = backlog["arguments"]
    prior = args["run_root"].parent / "prior"
    module.submit_backlog(**{**args, "run_root": prior, "submit": False})
    (prior / "tasks.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="task queue differs"):
        module.submit_backlog(**{**args, "resume_from_run_root": prior})
    assert not backlog["calls"]


def test_resume_of_resume_preserves_original_claim_registry(backlog):
    args = {**backlog["arguments"], "submit": False}
    first = args["run_root"].parent / "first"
    second = args["run_root"].parent / "second"
    module.submit_backlog(**{**args, "run_root": first})
    module.submit_backlog(**{**args, "run_root": second, "resume_from_run_root": first})
    result = module.submit_backlog(**{**args, "resume_from_run_root": second})
    assert result["claim_root"] == str(first / "claims")
    assert result["resume_from"]["claim_root"] == str(first / "claims")


def test_prepare_only_can_be_rechecked_then_submitted_once(backlog, monkeypatch):
    monkeypatch.setattr(module.shutil, "which", lambda *args, **kwargs: None)
    args = {**backlog["arguments"], "submit": False}
    prepared = module.submit_backlog(**args)
    repeated = module.submit_backlog(**args)
    assert prepared == repeated
    assert not backlog["calls"]
    assert not backlog["probes"]
    assert not list(args["run_root"].glob("models/*/submission-intent.json"))
    monkeypatch.setattr(module.shutil, "which", lambda *args, **kwargs: "/usr/bin/sbatch")
    assert module.submit_backlog(**backlog["arguments"])["status"] == "submitted"
    assert len(backlog["calls"]) == 2


@pytest.mark.parametrize("change", ["walltime", "estimate", "commit", "llama_weights", "profile", "cohort"])
def test_preflight_rejects_bad_inputs_before_allocating(backlog, change):
    args, env = backlog["arguments"], backlog["environment"]
    if change == "walltime":
        env["GEODML_APPROVED_WALLTIME"] = "04:00:00"
    elif change == "estimate":
        env["GEODML_ALLOCATION_ESTIMATE"] = ""
    elif change == "commit":
        args["source_git_commit"] = "b" * 40
    elif change == "llama_weights":
        next(Path(env["HF_HUB_CACHE"]).glob("models--meta-llama*/snapshots/*/model.safetensors")).unlink()
    elif change == "profile":
        path = args["profile_root"] / module.MODELS["llama4"]["profile"]
        profile = json.loads(path.read_text())
        profile["serving"]["tensor_parallel_size"] = 2
        _json(path, profile)
    else:
        (args["cohort_root"] / "pilot-prompts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        module.submit_backlog(**args)
    assert not backlog["calls"]
    assert not args["run_root"].exists()


@pytest.mark.parametrize("response", [
    (1, "", "PRIVATE_TOKEN should not be persisted"),
    (0, "ambiguous PRIVATE_TOKEN stdout", ""),
    subprocess.TimeoutExpired(["sbatch"], 60, output="PRIVATE_TOKEN"),
])
def test_partial_or_ambiguous_submission_permanently_blocks_retry(backlog, response):
    backlog["responses"][1] = response
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        module.submit_backlog(**backlog["arguments"])
    root = backlog["arguments"]["run_root"]
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "submission_incomplete"
    assert manifest["models"]["qwen38"]["job_id"] == "12345"
    assert manifest["models"]["llama4"]["status"] == "submission_uncertain"
    with pytest.raises(FileExistsError, match="intent already exists"):
        module.submit_backlog(**backlog["arguments"])
    assert len(backlog["calls"]) == 2
    assert not any("PRIVATE_TOKEN" in path.read_text() for path in root.rglob("*.json"))


def test_ambiguous_first_submission_stops_before_second_array(backlog):
    backlog["responses"][0] = (0, "12345\n12346\n", "")
    with pytest.raises(RuntimeError, match="unambiguous"):
        module.submit_backlog(**backlog["arguments"])
    assert len(backlog["calls"]) == 1


def test_concurrent_submitter_cannot_acquire_run_lock(backlog):
    root = backlog["arguments"]["run_root"]
    root.mkdir()
    with (root / ".submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="run lock"):
            module.submit_backlog(**backlog["arguments"])
    assert not backlog["calls"]


def test_changed_frozen_queue_is_rejected_before_submission(backlog):
    module.submit_backlog(**{**backlog["arguments"], "submit": False})
    queue = backlog["arguments"]["run_root"] / "models/llama4/wave/backlog.jsonl"
    queue.write_text("{}\n")
    with pytest.raises(ValueError, match="frozen input changed"):
        module.submit_backlog(**backlog["arguments"])
    assert not backlog["calls"]


def test_prior_allocation_settings_and_credentials_are_not_exported(backlog):
    backlog["environment"].update({
        "SLURM_JOB_ID": "old", "SBATCH_TIMELIMIT": "12:00:00", "HF_TOKEN": "PRIVATE_TOKEN",
        "GEODML_WORKER_INDEX": "7", "GEODML_ALLOCATION_BUDGET_JSON": "wrong",
        "SEARCH_AGENTIC_CELL_IDS_JSONL": "/previous-cells.jsonl", "VLLM_API_KEY": "PRIVATE_TOKEN",
    })
    module.submit_backlog(**backlog["arguments"])
    for _, kwargs in backlog["calls"]:
        env = kwargs["env"]
        assert not any(key.startswith(("SLURM_", "SBATCH_")) for key in env)
        for key in ("HF_TOKEN", "VLLM_API_KEY", "GEODML_WORKER_INDEX", "GEODML_ALLOCATION_BUDGET_JSON",
                    "SEARCH_AGENTIC_CELL_IDS_JSONL"):
            assert key not in env
    assert not any("PRIVATE_TOKEN" in path.read_text() for path in backlog["arguments"]["run_root"].rglob("*.json"))


def test_freezing_preserves_parquet_suffix_for_reader_and_workers(backlog, monkeypatch):
    source_jsonl = Path(backlog["environment"]["SEARCH_AGENTIC_DDG_SNAPSHOT"])
    source_parquet = source_jsonl.with_suffix(".parquet")
    source_parquet.write_bytes(b"PAR1 fixture payload, parsing mocked in this test")
    backlog["environment"]["SEARCH_AGENTIC_DDG_SNAPSHOT"] = str(source_parquet)
    real_adapter = module.FrozenSnapshotSearchAdapter
    seen = []

    def adapter(engine, path):
        seen.append(path)
        if engine == "duckduckgo":
            assert path.suffix == ".parquet"
            assert path.read_bytes() == source_parquet.read_bytes()
            return real_adapter(engine, source_jsonl)
        return real_adapter(engine, path)

    monkeypatch.setattr(module, "FrozenSnapshotSearchAdapter", adapter)
    module.submit_backlog(**backlog["arguments"])
    assert any(path.name == "duckduckgo.parquet" for path in seen)
    for _, options in backlog["calls"]:
        assert options["env"]["SEARCH_AGENTIC_DDG_SNAPSHOT"].endswith("/search/duckduckgo.parquet")


@pytest.mark.parametrize("change", [
    "old120", "missing_prior", "wrong_union_count", "wrong_union_hash",
    "original_manifest_hash", "prior_manifest_hash", "original_prompts_hash",
    "prior_prompts_hash", "population_hash", "id_overlap", "text_overlap",
    "axis_duplicates", "wrong_record_bins",
])
def test_wrong_cohort_or_unverified_exclusions_never_submit(backlog, change):
    cohort = backlog["arguments"]["cohort_root"]
    manifest_path = cohort / "selection-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "old120":
        manifest.update(prompt_count=120, expected_cells_per_model=1440)
        for entry in manifest["artifacts"].values():
            path = cohort / entry["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines()][:120]
            entry.update(_jsonl(path, rows))
    elif change == "missing_prior":
        manifest.pop("additional_exclusions")
    elif change == "wrong_union_count":
        manifest["excluded_prompt_count"] = 500
    elif change == "wrong_union_hash":
        manifest["excluded_prompt_ids_sha256"] = "0" * 64
    elif change in {"original_manifest_hash", "prior_manifest_hash", "original_prompts_hash", "prior_prompts_hash"}:
        provenance = manifest["exclusion"] if change.startswith("original") else manifest["additional_exclusions"][0]
        key = "selection_manifest" if "manifest_hash" in change else "prompts"
        provenance[key]["sha256"] = "0" * 64
    elif change == "population_hash":
        manifest["sources"]["prompts"]["sha256"] = "0" * 64
    elif change in {"axis_duplicates", "wrong_record_bins"}:
        key = "axis_map" if change == "axis_duplicates" else "selection_records"
        entry = manifest["artifacts"][key]
        path = cohort / entry["path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "axis_duplicates":
            rows = [rows[0]] * len(rows)
        else:
            rows = [{**row, "axis_bin": 0} for row in rows]
        entry.update(_jsonl(path, rows))
    else:
        entry = manifest["artifacts"]["prompts"]
        path = cohort / entry["path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "id_overlap":
            rows[0]["candidate_id"] = "original500-0000"
        else:
            rows[0]["question"] = "  PRIOR120   QUESTION 0?  "
            rows[0]["question_sha256"] = hashlib.sha256(rows[0]["question"].encode()).hexdigest()
        entry.update(_jsonl(path, rows))
    _json(manifest_path, manifest)
    with pytest.raises(ValueError):
        module.submit_backlog(**backlog["arguments"])
    assert not backlog["calls"]
    assert not backlog["arguments"]["run_root"].exists()


def test_login_namespace_restriction_does_not_block_compute_job_submission(backlog, monkeypatch):
    real_run = module.subprocess.run

    def run(command, **kwargs):
        if command[0] == module.sys.executable:
            raise subprocess.CalledProcessError(1, command, stderr="network namespace unavailable")
        return real_run(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.submit_backlog(**backlog["arguments"])
    assert result["status"] == "submitted"
    assert len(backlog["calls"]) == 2
    assert not backlog["probes"]


def test_missing_compute_isolation_helper_prevents_submission(backlog):
    repository = Path(backlog["environment"]["GEODML_EXECUTION_REPOSITORY"])
    (repository / "analysis/scripts/inference_network_namespace.py").unlink()
    with pytest.raises(ValueError, match="inference_network_namespace"):
        module.submit_backlog(**backlog["arguments"])
    assert not backlog["calls"]
    assert not backlog["arguments"]["run_root"].exists()
