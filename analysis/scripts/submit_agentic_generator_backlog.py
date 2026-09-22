"""Freeze a supplied cohort and submit a bounded, resumable generator backlog."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline import inference_claims
from analysis.interpretability.pipeline.agentic_generation_tasks import (
    _canonical,
    build_cells,
    load_calibration_prompts,
)
from analysis.interpretability.pipeline.inference_wave import (
    _atomic_json,
    _atomic_jsonl,
    build_inference_wave,
    write_inference_wave,
)
from analysis.scripts.prepare_agentic_new_cohort import (
    _normalize,
    _question_key,
    _validate_excluded_rows,
    _verified_rows,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    FrozenSnapshotSearchAdapter,
    _build_target_urls,
)
from analysis.scripts.search_vllm_stage import load_profile
from analysis.scripts.submit_agentic_paired_trial import (
    BGE_REVISION,
    MODELS,
    PROMPT_SELECTION_SEED,
    _file,
    _hash,
    _now,
    _require,
    _snapshot,
)

# These preserve the already-run four-allocation launcher as the default.  New
# schedules must be explicit at both the CLI and the approved environment.
APPROVED_WALLTIME = "03:00:00"
APPROVED_PROMPT_COUNT = 1200
WORKERS_PER_MODEL = 2
MAXIMUM_GPU_HOURS = 48
WAVE_SEED = 20260915
# Do not export credentials, stale allocation controls, or prior worker settings.
SHELL_ENVIRONMENT = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TMPDIR",
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "MODULEPATH", "MODULESHOME",
    "LOADEDMODULES", "_LMFILES_", "LMOD_CMD", "LMOD_DIR", "LMOD_PKG",
})


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_json(path: Path, value: object) -> None:
    _atomic_json(path, value)
    _sync_directory(path.parent)


def _copy_frozen(source: Path, destination: Path, digest: str) -> None:
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"input changed during preparation: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(destination.parent)


def _schedule(
    approved_walltime: str, workers_per_model: int, maximum_total_gpu_hours: int,
    model_count: int,
) -> dict[str, Any]:
    """Validate the exact user-approved Slurm layout before preparing a run."""
    matched = re.fullmatch(r"([0-9]{2,}):([0-5][0-9]):([0-5][0-9])", approved_walltime)
    if matched is None:
        raise ValueError("approved wall-time must use positive HH:MM:SS")
    hours, minutes, seconds = (int(value) for value in matched.groups())
    total_seconds = 3600 * hours + 60 * minutes + seconds
    if total_seconds <= 0:
        raise ValueError("approved wall-time must be positive")
    if type(workers_per_model) is not int or workers_per_model <= 0:
        raise ValueError("workers per model must be a positive integer")
    if type(maximum_total_gpu_hours) is not int or maximum_total_gpu_hours <= 0:
        raise ValueError("maximum total GPU-hours must be a positive integer")
    if type(model_count) is not int or model_count <= 0:
        raise ValueError("model count must be a positive integer")
    expected = model_count * workers_per_model * 4 * total_seconds / 3600
    if expected != int(expected) or maximum_total_gpu_hours != int(expected):
        raise ValueError(
            "maximum GPU-hours must exactly equal selected models × workers × four GPUs × wall-time"
        )
    walltime_tag = f"{hours}h" if minutes == seconds == 0 else approved_walltime.replace(":", "")
    return {
        "approved_walltime": approved_walltime,
        "workers_per_model": workers_per_model,
        "allocation_count": model_count * workers_per_model,
        "maximum_total_gpu_hours": maximum_total_gpu_hours,
        "array": f"0-{workers_per_model - 1}%{workers_per_model}",
        "job_name_tag": walltime_tag,
    }


def _preflight(
    environment: Mapping[str, str], profile_root: Path, source_git_commit: str,
    *, submit: bool, schedule: Mapping[str, Any], model_slugs: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, Any]]:
    if _require(environment, "GEODML_APPROVED_WALLTIME") != schedule["approved_walltime"]:
        raise ValueError("environment approval wall-time differs from requested schedule")
    if _require(environment, "GEODML_MAXIMUM_TOTAL_GPU_HOURS") != str(schedule["maximum_total_gpu_hours"]):
        raise ValueError("environment GPU-hour cap differs from requested schedule")
    estimate = _require(environment, "GEODML_ALLOCATION_ESTIMATE")
    if re.fullmatch(r"[0-9a-f]{40}", source_git_commit) is None:
        raise ValueError("source Git commit must be a full lowercase Git SHA")
    if environment.get("GEODML_EXECUTION_COMMIT", source_git_commit) != source_git_commit:
        raise ValueError("execution commit differs from the requested source Git commit")
    repository = Path(environment.get("GEODML_EXECUTION_REPOSITORY", str(REPOSITORY_ROOT))).resolve()
    if repository != REPOSITORY_ROOT:
        raise ValueError("run the helper from the pinned execution repository")
    for arguments, expected in (
        (("rev-parse", "HEAD"), source_git_commit),
        (("status", "--porcelain", "--untracked-files=all"), ""),
    ):
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True, text=True, check=True,
        )
        if result.stdout.strip() != expected:
            raise ValueError("execution repository is dirty or has the wrong commit")
    env = {key: value for key, value in environment.items() if key in SHELL_ENVIRONMENT}
    env.update(
        GEODML_EXECUTION_REPOSITORY=str(repository),
        GEODML_EXECUTION_COMMIT=source_git_commit,
        GEODML_APPROVED_WALLTIME=schedule["approved_walltime"],
        GEODML_MAXIMUM_TOTAL_GPU_HOURS=str(schedule["maximum_total_gpu_hours"]),
        GEODML_ALLOCATION_ESTIMATE=estimate,
        GEODML_START_MARGIN_SECONDS="120", GEODML_CLEANUP_MARGIN_SECONDS="45",
    )
    for name in ("HF_HUB_CACHE", "GEODML_CACHE_ROOT", "ACL_ARR_VENV"):
        path = Path(_require(environment, name)).resolve()
        if not path.is_dir():
            raise ValueError(f"configured directory does not exist: {name}")
        env[name] = str(path)
    _file(Path(env["ACL_ARR_VENV"]) / "bin/activate")
    if importlib.metadata.version("sentence-transformers") != "6.0.1":
        raise ValueError("the generator requires sentence-transformers 6.0.1")
    if submit and shutil.which("sbatch", path=env.get("PATH")) is None:
        raise ValueError("sbatch is not available in this shell")
    for name in (
        "run_agentic_generation_worker.sh", "run_inference_wave_worker.sbatch",
        "run_agentic_search_qwen38_smoke.sh", "run_agentic_search_llama4_smoke.sh",
    ):
        path = _file(repository / "analysis/scripts/slurm/jupiter" / name)
        if name == "run_agentic_generation_worker.sh" and not os.access(path, os.X_OK):
            raise ValueError("the generation worker must be executable")
    cache = Path(env["HF_HUB_CACHE"])
    models = {}
    for slug in model_slugs:
        specification = MODELS[slug]
        profile_path = _file(profile_root / specification["profile"])
        profile = load_profile(profile_path)
        expected = {key: specification[key] for key in ("model_id", "model_revision")}
        if profile["model"] != expected or any(
            profile["serving"].get(key) != value for key, value in {
                "tensor_parallel_size": 4, "data_parallel_size": 1,
                "host": "127.0.0.1", "port": 8010,
            }.items()
        ):
            raise ValueError(f"wrong model or serving topology in {profile_path}")
        models[slug] = {
            **expected, "profile": str(profile_path), "profile_sha256": _hash(profile_path),
            "snapshot": _snapshot(cache, expected["model_id"], expected["model_revision"]),
        }
    bge = Path(environment.get("SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT", str(
        cache / "models--BAAI--bge-reranker-v2-m3/snapshots" / BGE_REVISION,
    ))).resolve()
    if (bge.name != BGE_REVISION or
            environment.get("SEARCH_AGENTIC_CROSS_ENCODER_REVISION", BGE_REVISION) != BGE_REVISION):
        raise ValueError("the cross-encoder revision differs from the frozen revision")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        _file(bge / name)
    env.update(SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT=str(bge),
               SEARCH_AGENTIC_CROSS_ENCODER_REVISION=BGE_REVISION)
    for name in ("SEARCH_AGENTIC_DDG_SNAPSHOT", "SEARCH_AGENTIC_SEARXNG_SNAPSHOT"):
        env[name] = str(_file(Path(_require(environment, name)).resolve()))
    # Login-node namespace policy does not establish compute-node capability.
    # The serving stage must create and verify isolation before starting vLLM.
    _file(repository / "analysis/scripts/inference_network_namespace.py")
    return env, models


def _exclusion_sources(
    cohort: Path, manifest: dict[str, Any], selected: dict[str, list[dict[str, Any]]],
) -> dict[str, Path]:
    """Recheck the approved original-500 plus previous-120 exclusion evidence."""
    if any(manifest.get(key) != expected for key, expected in (
        ("original_excluded_prompt_count", 500), ("additional_excluded_prompt_count", 120),
        ("excluded_prompt_count", 620),
    )) or len(manifest.get("additional_exclusions", [])) != 1:
        raise ValueError("the approved cohort requires original-500 plus prior-120 exclusion provenance")
    sources: dict[str, Path] = {}
    population = {}
    for key in ("prompts", "axis_map"):
        rows, verified = _verified_rows(cohort, manifest["sources"][key], f"population {key}")
        population[key] = {row["candidate_id"]: row for row in rows}
        if len(population[key]) != len(rows):
            raise ValueError("population candidate IDs are duplicated")
        sources[f"provenance/population/{key}.jsonl"] = Path(verified["path"])
    if manifest["diagnostics"]["axis_bins"] != 20:
        raise ValueError("the approved cohort requires the frozen 20 axis bins")
    normalized_population = {
        row.candidate_id: row for row in _normalize(
            list(population["prompts"].values()), list(population["axis_map"].values()), 20,
        )
    }
    excluded_ids: set[str] = set()
    excluded_text: set[str] = set()
    for label, provenance, expected_count, expected_format in (
        ("original500", manifest["exclusion"], 500, "readiness-axis-balanced-pilot-v1"),
        ("prior120", manifest["additional_exclusions"][0], 120, "agentic-new-prompt-cohort-v1"),
    ):
        entry = provenance["selection_manifest"]
        path = (cohort / entry["path"]).resolve()
        if _hash(path) != entry["sha256"]:
            raise ValueError(f"{label} exclusion manifest hash mismatch")
        previous = json.loads(path.read_text())
        if previous["format_version"] != expected_format:
            raise ValueError(f"{label} exclusion manifest format differs")
        if label == "prior120" and (
            provenance.get("prompt_count") != 120 or previous.get("prompt_count") != 120
        ):
            raise ValueError("prior exclusion must contain exactly 120 prompts")
        sources[f"provenance/{label}/selection-manifest.json"] = path
        rows_by_key = {}
        for key in ("prompts", "axis_map", "selection_records"):
            rows, verified = _verified_rows(path.parent, previous["artifacts"][key], f"{label} {key}")
            if len(rows) != expected_count:
                raise ValueError(f"{label} exclusion artifact count differs")
            if key != "axis_map" or label == "prior120":
                _, declared = _verified_rows(cohort, provenance[key], f"declared {label} {key}")
                if declared != verified:
                    raise ValueError(f"{label} exclusion artifact provenance differs")
            rows_by_key[key] = rows
            sources[f"provenance/{label}/{key}.jsonl"] = Path(verified["path"])
        ids = {row["candidate_id"] for row in rows_by_key["prompts"]}
        if len(ids) != expected_count or any(
            {row["candidate_id"] for row in rows_by_key[key]} != ids
            for key in ("axis_map", "selection_records")
        ):
            raise ValueError(f"{label} exclusion candidate IDs differ or repeat")
        for key in ("prompts", "axis_map"):
            if any(previous["sources"][key][field] != manifest["sources"][key][field]
                   for field in ("sha256", "rows")):
                raise ValueError(f"{label} exclusion population source differs")
            if any(population[key].get(row["candidate_id"]) != row for row in rows_by_key[key]):
                raise ValueError(f"{label} exclusion rows differ from the frozen population")
        excluded_ids.update(ids)
        excluded_text.update(_question_key(row["question"]) for row in rows_by_key["prompts"])
    if len(excluded_ids) != 620 or hashlib.sha256(_canonical(sorted(excluded_ids))).hexdigest() != manifest.get("excluded_prompt_ids_sha256"):
        raise ValueError("excluded ID union or its hash differs from the approved 620 prompts")
    selected_ids = {row["candidate_id"] for row in selected["prompts"]}
    selected_text = {_question_key(row["question"]) for row in selected["prompts"]}
    if len(selected_ids) != APPROVED_PROMPT_COUNT or len(selected_text) != APPROVED_PROMPT_COUNT:
        raise ValueError("selected prompt IDs or normalized question text are duplicated")
    if selected_ids & excluded_ids or selected_text & excluded_text:
        raise ValueError("selected prompts overlap excluded IDs or normalized question text")
    _validate_excluded_rows(
        selected["prompts"], selected["axis_map"], selected["selection_records"],
        normalized_population,
    )
    return sources


def _cohort_inputs(cohort: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = _file(cohort / "selection-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["format_version"] != "agentic-new-prompt-cohort-v1":
        raise ValueError("a frozen new-prompt cohort is required")
    count = manifest["prompt_count"]
    if type(count) is not int or count != APPROVED_PROMPT_COUNT or manifest["expected_cells_per_model"] != 12 * count:
        raise ValueError("this approval requires exactly 1200 prompts and 14400 cells per model")
    sources = {"cohort/selection-manifest.json": manifest_path}
    selected = {}
    for key, filename in (
        ("prompts", "pilot-prompts.jsonl"), ("axis_map", "pilot-axis.jsonl"),
        ("selection_records", "selection-records.jsonl"),
    ):
        rows, verified = _verified_rows(cohort, manifest["artifacts"][key], key)
        path = Path(verified["path"])
        if len(rows) != count or path != cohort / filename:
            raise ValueError("cohort artifact count or canonical filename differs")
        sources["cohort/" + filename] = path
        selected[key] = rows
    sources.update(_exclusion_sources(cohort, manifest, selected))
    return manifest, sources


def _command(
    run_root: Path, slug: str, account: str, partition: str, schedule: Mapping[str, Any],
) -> list[str]:
    logs = run_root / "models" / slug / "logs"
    return [
        "sbatch", "--parsable", "--no-requeue", f"--array={schedule['array']}",
        f"--account={account}", f"--partition={partition}",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--mem=512G",
        "--gres=gpu:4", "--exclusive", f"--time={schedule['approved_walltime']}", "--export=ALL",
        f"--job-name=geodml-{slug}-backlog-{schedule['job_name_tag']}", f"--chdir={REPOSITORY_ROOT}",
        f"--output={logs}/slurm-%A_%a.out", f"--error={logs}/slurm-%A_%a.err",
        str(REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch"),
    ]


def _resume_claims(
    resume_from_run_root: Path | tuple[Path, ...] | list[Path] | None,
    cohort_root: Path,
    tasks: list[dict[str, Any]],
    destination_claim_root: Path,
) -> dict[str, Any] | None:
    """Accept one or more exact frozen queues as durable claim predecessors."""
    if resume_from_run_root is None:
        return None
    roots = (
        (resume_from_run_root,)
        if isinstance(resume_from_run_root, Path)
        else tuple(resume_from_run_root)
    )
    if not roots:
        raise ValueError("at least one resume source is required")
    expected_tasks = b"".join(
        json.dumps(task, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for task in tasks
    )
    sources = []
    seen_runs: set[Path] = set()
    for supplied_root in roots:
        root = supplied_root.resolve()
        if root in seen_runs:
            continue
        seen_runs.add(root)
        manifest_path = _file(root / "run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") not in {
            "agentic-four-generator-backlog-v1", "agentic-generator-backlog-v2",
        }:
            raise ValueError("resume source is not a compatible generator backlog")
        if manifest.get("request", {}).get("cohort_root") != str(cohort_root):
            raise ValueError("resume source uses a different frozen cohort")
        prior_tasks = _file(root / "tasks.jsonl")
        if prior_tasks.read_bytes() != expected_tasks:
            raise ValueError("resume source task queue differs from the requested queue")
        recorded_claim_root = manifest.get("claim_root")
        if recorded_claim_root is None:
            # Older manifests predate the explicit shared-registry field.
            claim_root = root / "claims"
        elif not isinstance(recorded_claim_root, str) or not Path(recorded_claim_root).is_absolute():
            raise ValueError("resume source claim root must be an absolute path")
        else:
            claim_root = Path(recorded_claim_root).resolve()
        if claim_root.exists() and not claim_root.is_dir():
            raise ValueError("resume source claim root is not a directory")
        sources.append({
            "run_root": str(root),
            "run_manifest_sha256": _hash(manifest_path),
            "tasks_sha256": _hash(prior_tasks),
            "claim_root": str(claim_root),
        })
    if len(sources) == 1:
        return sources[0]
    return {
        "sources": sources,
        "tasks_sha256": hashlib.sha256(expected_tasks).hexdigest(),
        "claim_root": str(destination_claim_root.resolve()),
        "reconciliation": "validated-immutable-union-v1",
    }


def _claim_record(
    path: Path, fingerprint: str, failed: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Read and validate one immutable terminal record while holding its lock."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"claim record is not a regular file: {path}")
    lock_path = path.parent / f"{fingerprint}.lock"
    descriptor = None
    try:
        if lock_path.exists():
            descriptor = os.open(
                lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"resume source has an active claim: {lock_path}") from error
        record = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=inference_claims._unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"corrupt terminal claim record: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    expected_keys = {
        "format_version", "identity", "identity_sha256", "outcome", "outcome_sha256",
    }
    identity = (
        inference_claims._identity(record.get("identity", {}))
        if isinstance(record, dict) else None
    )
    expected_format = (
        inference_claims.FAILURE_FORMAT_VERSION
        if failed else inference_claims.FORMAT_VERSION
    )
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record["format_version"] != expected_format
        or record["identity_sha256"] != fingerprint
        or inference_claims._digest(record["identity"]) != fingerprint
        or record["outcome_sha256"] != inference_claims._digest(record["outcome"])
        or identity is None
    ):
        raise ValueError(f"mismatched terminal claim envelope: {path}")
    return _canonical(record) + b"\n", record


def _portable_generator_trace(
    result: Mapping[str, Any], trace: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return scientific output without derived hashes or absolute paths."""
    portable_result = json.loads(_canonical(result))
    portable_trace = json.loads(_canonical(trace))
    portable_result.pop("trace_sha256", None)
    portable_trace.pop("trace_sha256", None)
    events = portable_trace.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            return None
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        raw_payload = payload.get("raw_payload")
        if not isinstance(raw_payload, dict):
            continue
        if raw_payload.get("format_version") != "frozen-search-snapshot-response-v1":
            continue
        snapshot_sha256 = raw_payload.get("snapshot_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", str(snapshot_sha256)) is None:
            return None
        if "snapshot" in raw_payload:
            raw_payload["snapshot"] = f"sha256:{snapshot_sha256}"
    return {"result": portable_result, "trace": portable_trace}


def _generator_scientific_outcome(record: Mapping[str, Any]) -> bytes | None:
    """Return the generator payload covered by scientific task identity.

    Runtime diagnostics and producer provenance are intentionally excluded.
    They remain auditable in the source records and reconciliation receipt.
    """
    identity = record.get("identity")
    outcome = record.get("outcome")
    if (
        record.get("format_version") != inference_claims.FORMAT_VERSION
        or not isinstance(identity, Mapping)
        or identity.get("protocol") != "agentic-generator-shared-v1"
        or not isinstance(outcome, Mapping)
        or set(outcome) != {"result", "trace", "diagnostics", "producer"}
    ):
        return None
    result, trace = outcome["result"], outcome["trace"]
    if not isinstance(result, Mapping) or not isinstance(trace, Mapping):
        return None
    portable = _portable_generator_trace(result, trace)
    return None if portable is None else _canonical(portable)


def _claim_preference(record: Mapping[str, Any], path: Path) -> tuple[bool, str, str]:
    """Prefer directly committed provenance, then a stable content/path order."""
    outcome = record["outcome"]
    producer = outcome.get("producer") if isinstance(outcome, Mapping) else None
    imported = isinstance(producer, Mapping) and producer.get("imported_legacy_artifact") is True
    return imported, record["outcome_sha256"], str(path)


def _reconcile_claim_roots(
    resume: Mapping[str, Any], destination: Path,
) -> dict[str, Any]:
    """Materialize a fail-closed union without modifying any source registry."""
    if resume.get("reconciliation") != "validated-immutable-union-v1":
        raise ValueError("claim reconciliation requires multiple validated sources")
    source_roots = sorted({
        Path(source["claim_root"]).resolve() for source in resume["sources"]
    })
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"reconciled claim root already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".claims-reconcile-", dir=destination.parent))
    records: dict[str, dict[str, Any]] = {}
    resolutions: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    try:
        for root in source_roots:
            count = 0
            if root.is_dir():
                for directory in sorted(root.iterdir()):
                    if not directory.is_dir() or re.fullmatch(r"[0-9a-f]{2}", directory.name) is None:
                        continue
                    for path in sorted(directory.iterdir()):
                        matched = re.fullmatch(r"([0-9a-f]{64})(\.failed)?\.json", path.name)
                        if matched is None:
                            continue
                        fingerprint, failure_marker = matched.groups()
                        if directory.name != fingerprint[:2]:
                            raise ValueError(f"claim record is in the wrong shard: {path}")
                        failed = failure_marker is not None
                        payload, record = _claim_record(path, fingerprint, failed)
                        previous = records.get(fingerprint)
                        if previous is None:
                            records[fingerprint] = {
                                "payload": payload, "record": record,
                                "failed": failed, "path": path,
                            }
                        elif previous["payload"] != payload or previous["failed"] != failed:
                            if previous["failed"] != failed:
                                raise ValueError(
                                    f"conflicting terminal claim {fingerprint}: "
                                    f"{previous['path']} and {path}"
                                )
                            prior_science = _generator_scientific_outcome(previous["record"])
                            current_science = _generator_scientific_outcome(record)
                            if prior_science is None or current_science is None or prior_science != current_science:
                                raise ValueError(
                                    f"conflicting scientific outcome {fingerprint}: "
                                    f"{previous['path']} and {path}"
                                )
                            resolution = resolutions.setdefault(fingerprint, {
                                "identity_sha256": fingerprint,
                                "scientific_outcome_sha256": hashlib.sha256(prior_science).hexdigest(),
                                "candidates": {},
                            })
                            for candidate_record, candidate_path in (
                                (previous["record"], previous["path"]), (record, path),
                            ):
                                resolution["candidates"].setdefault(
                                    candidate_record["outcome_sha256"], [],
                                ).append(str(candidate_path))
                            if _claim_preference(record, path) < _claim_preference(
                                previous["record"], previous["path"],
                            ):
                                records[fingerprint] = {
                                    "payload": payload, "record": record,
                                    "failed": failed, "path": path,
                                }
                        count += 1
            source_counts[str(root)] = count
        for fingerprint, selected in sorted(records.items()):
            directory = temporary / fingerprint[:2]
            directory.mkdir(exist_ok=True)
            suffix = ".failed.json" if selected["failed"] else ".json"
            with (directory / f"{fingerprint}{suffix}").open("xb") as stream:
                stream.write(selected["payload"])
                stream.flush()
                os.fsync(stream.fileno())
        for directory in temporary.iterdir():
            if directory.is_dir():
                _sync_directory(directory)
        resolution_path = temporary / "reconciliation-resolutions.jsonl"
        with resolution_path.open("xb") as stream:
            for fingerprint, resolution in sorted(resolutions.items()):
                resolution["candidates"] = {
                    digest: sorted(set(paths))
                    for digest, paths in sorted(resolution["candidates"].items())
                }
                resolution["selected_outcome_sha256"] = records[fingerprint][
                    "record"
                ]["outcome_sha256"]
                stream.write(_canonical(resolution) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            "format_version": "geodml-claim-reconciliation-v1",
            "created_at_utc": _now(),
            "mode": "validated-immutable-union",
            "source_claim_roots": [str(root) for root in source_roots],
            "source_terminal_record_counts": source_counts,
            "terminal_record_count": len(records),
            "conflict_count": 0,
            "scientific_duplicate_count": len(resolutions),
            "metadata_variant_count": sum(
                max(0, len(resolution["candidates"]) - 1)
                for resolution in resolutions.values()
            ),
            "resolution_file": {
                "path": "reconciliation-resolutions.jsonl",
                "sha256": _hash(resolution_path),
                "rows": len(resolutions),
            },
            "source_registries_modified": False,
        }
        _durable_json(temporary / "reconciliation.json", receipt)
        _sync_directory(temporary)
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def submit_backlog(
    *, cohort_root: Path, run_root: Path, profile_root: Path,
    source_git_commit: str, account: str, partition: str, submit: bool = False,
    environment: Mapping[str, str] | None = None,
    approved_walltime: str = APPROVED_WALLTIME,
    workers_per_model: int = WORKERS_PER_MODEL,
    maximum_total_gpu_hours: int = MAXIMUM_GPU_HOURS,
    resume_from_run_root: Path | tuple[Path, ...] | list[Path] | None = None,
    model_slugs: tuple[str, ...] = tuple(MODELS),
) -> dict[str, Any]:
    """Prepare once; submit the selected model arrays with no automatic recovery."""
    cohort_root, run_root, profile_root = (
        path.resolve() for path in (cohort_root, run_root, profile_root)
    )
    if (run_root.is_relative_to(REPOSITORY_ROOT) or cohort_root.is_relative_to(run_root)
            or run_root.is_relative_to(cohort_root)):
        raise ValueError("run root must be outside the pinned source repository and supplied cohort")
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None for value in (account, partition)):
        raise ValueError("an explicit account and partition are required")
    # Reject recorded attempts before any preparation or external command.
    if run_root.exists() and any(run_root.glob("models/*/submission-intent.json")):
        raise FileExistsError("submission intent already exists; inspect Slurm, never resubmit this run")
    model_slugs = tuple(model_slugs)
    if not model_slugs or len(set(model_slugs)) != len(model_slugs) or any(
        slug not in MODELS for slug in model_slugs
    ):
        raise ValueError("select one or more distinct supported model slugs")
    schedule = _schedule(
        approved_walltime, workers_per_model, maximum_total_gpu_hours, len(model_slugs),
    )
    env, models = _preflight(
        dict(os.environ if environment is None else environment), profile_root,
        source_git_commit, submit=submit, schedule=schedule, model_slugs=model_slugs,
    )
    cohort, sources = _cohort_inputs(cohort_root)
    source_prompts = load_calibration_prompts(
        cohort_root / "pilot-prompts.jsonl", cohort_root / "selection-records.jsonl",
        prompt_count=cohort["prompt_count"], seed=PROMPT_SELECTION_SEED,
    )
    source_cells = build_cells(source_prompts)
    tasks = [{"cell_id": cell.cell_id, **cell.core} for cell in source_cells]
    if len(tasks) != cohort["expected_cells_per_model"]:
        raise ValueError("cohort cell count differs from its manifest")
    resume = _resume_claims(
        resume_from_run_root, cohort_root, tasks, run_root / "claims",
    )
    for slug, model in models.items():
        sources[f"profiles/{slug}.json"] = Path(model["profile"])
    search_paths = {}
    for name, engine in (
        ("SEARCH_AGENTIC_DDG_SNAPSHOT", "duckduckgo"),
        ("SEARCH_AGENTIC_SEARXNG_SNAPSHOT", "searxng"),
    ):
        source = Path(env[name])
        relative = f"search/{engine}{source.suffix.lower()}"
        sources[relative] = source
        search_paths[engine] = run_root / relative
    source_files = {
        relative: {"path": str(path), "sha256": _hash(path)}
        for relative, path in sources.items()
    }
    request = {
        "cohort_root": str(cohort_root), "git_commit": source_git_commit,
        "account": account, "partition": partition, "models": models,
        "source_files": source_files, "prompt_count": cohort["prompt_count"],
        "runtime": {key: env[key] for key in (
            "HF_HUB_CACHE", "GEODML_CACHE_ROOT", "ACL_ARR_VENV",
            "SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT", "SEARCH_AGENTIC_CROSS_ENCODER_REVISION",
        )},
        "approved_walltime_per_job": schedule["approved_walltime"],
        "workers_per_model": schedule["workers_per_model"],
        "maximum_total_gpu_hours": schedule["maximum_total_gpu_hours"],
        "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
        "resume_from": resume,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another preparation or submission holds this run lock") from error
        if any(run_root.glob("models/*/submission-intent.json")):
            raise FileExistsError("submission intent already exists; inspect Slurm, never resubmit this run")
        manifest_path = run_root / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "prepared" or manifest.get("request") != request:
                raise ValueError("existing run is not the same intact prepared request")
            for relative, digest in manifest["frozen_files"].items():
                if _hash(run_root / relative) != digest:
                    raise ValueError(f"frozen input changed: {relative}")
        else:
            if any(path.name != ".submission.lock" for path in run_root.iterdir()):
                raise FileExistsError("run root contains unrecognized or incomplete preparation")
            claim_reconciliation = None
            if resume is not None and "sources" in resume:
                claim_reconciliation = _reconcile_claim_roots(resume, Path(resume["claim_root"]))
            manifest = {
                "format_version": "agentic-generator-backlog-v2", "status": "preparing",
                "scientific_result": False, "created_at_utc": _now(), "request": request,
                "git_commit": source_git_commit, "prompt_count": cohort["prompt_count"],
                "cells_per_model": cohort["expected_cells_per_model"],
                "approved_walltime_per_job": schedule["approved_walltime"],
                "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
                "allocation_count": schedule["allocation_count"],
                "workers_per_model": schedule["workers_per_model"],
                "maximum_total_gpu_hours": schedule["maximum_total_gpu_hours"],
                "resources_per_job": {"nodes": 1, "gpus": 4, "gpu_type": "GH200", "cpus": 32,
                                      "memory": "512G", "exclusive_node": True},
                "prompt_selection_seed": PROMPT_SELECTION_SEED, "wave_seed": WAVE_SEED,
                "claim_root": resume["claim_root"] if resume else str(run_root / "claims"),
                "resume_from": resume,
                "claim_reconciliation": claim_reconciliation,
                "claim_identity": "agentic-generator-shared-v1: cell, model, revision, request hash",
                "cutoff_policy": "actual Slurm end; 120s admission margin; 45s cleanup margin",
                "automatic_resubmission": False,
                "models": {slug: {**model, "status": "not_submitted"} for slug, model in models.items()},
            }
            _durable_json(manifest_path, manifest)
            try:
                for relative, path in sources.items():
                    _copy_frozen(path, run_root / relative, source_files[relative]["sha256"])
                prompts = load_calibration_prompts(
                    run_root / "cohort/pilot-prompts.jsonl", run_root / "cohort/selection-records.jsonl",
                    prompt_count=cohort["prompt_count"], seed=PROMPT_SELECTION_SEED,
                )
                cells = build_cells(prompts)
                frozen_tasks = [{"cell_id": cell.cell_id, **cell.core} for cell in cells]
                if frozen_tasks != tasks:
                    raise ValueError("copied cohort produced a different task queue")
                adapters = {
                    engine: FrozenSnapshotSearchAdapter(engine, path)
                    for engine, path in search_paths.items()
                }
                if not set.intersection(*(adapter.keywords for adapter in adapters.values())):
                    raise ValueError("search snapshots have no shared keyword")
                selection_audit: dict[str, dict[str, int]] = {}
                targets = _build_target_urls(prompts, adapters, seed=PROMPT_SELECTION_SEED, selection_audit=selection_audit)
                if len(targets) != 2 * len(prompts):
                    raise ValueError("cohort lacks a target URL for each prompt and engine")
                manifest["target_selection"] = {
                    "counts": selection_audit, "target_url_count": len(targets),
                    "sha256": hashlib.sha256(_canonical([
                        {"prompt_id": prompt, "engine": engine, "target_url": url}
                        for (prompt, engine), url in sorted(targets.items())
                    ])).hexdigest(),
                }
                tasks_path = run_root / "tasks.jsonl"
                _atomic_jsonl(tasks_path, tasks)
                for slug in models:
                    model_root = run_root / "models" / slug
                    write_inference_wave(
                        model_root / "wave", source_tasks_path=tasks_path,
                        wave=build_inference_wave(
                            tasks, task_id_field="cell_id", completed_task_ids=set(),
                            worker_count=schedule["workers_per_model"], master_seed=WAVE_SEED,
                            dispatch_mode="backlog",
                        ),
                    )
                    (model_root / "logs").mkdir()
                    manifest["models"][slug]["command"] = _command(
                        run_root, slug, account, partition, schedule,
                    )
                manifest["frozen_files"] = {
                    str(path.relative_to(run_root)): _hash(path)
                    for path in run_root.rglob("*")
                    if (
                        path.is_file()
                        and path not in {manifest_path, run_root / ".submission.lock"}
                        and not path.is_relative_to(run_root / "claims")
                    )
                }
                manifest["status"] = "prepared"
                _durable_json(manifest_path, manifest)
            except BaseException as error:
                manifest.update(status="preparation_failed", error_type=type(error).__name__)
                _durable_json(manifest_path, manifest)
                raise
        if not submit:
            return manifest

        for slug in models:
            model_root = run_root / "models" / slug
            logs = model_root / "logs"
            worker_env = {
                **env, "GEODML_MODEL_SLUG": slug,
                "GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY": "1",
                "GEODML_WAVE_ROOT": str(model_root / "wave"),
                "GEODML_WAVE_OUTPUT_ROOT": str(model_root / "outputs"),
                "GEODML_WAVE_LOG_ROOT": str(logs),
                "GEODML_INFERENCE_CLAIM_ROOT": resume["claim_root"] if resume else str(run_root / "claims"),
                "GEODML_WORKER_LAUNCHER": str(REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"),
                "GEODML_WORKER_STDOUT": str(logs / "slurm-%A_%a.out"),
                "GEODML_WORKER_STDERR": str(logs / "slurm-%A_%a.err"),
                "SEARCH_AGENTIC_PROFILE": str(run_root / "profiles" / f"{slug}.json"),
                "SEARCH_AGENTIC_DDG_SNAPSHOT": str(search_paths["duckduckgo"]),
                "SEARCH_AGENTIC_SEARXNG_SNAPSHOT": str(search_paths["searxng"]),
                "SEARCH_AGENTIC_PROMPTS_JSONL": str(run_root / "cohort/pilot-prompts.jsonl"),
                "SEARCH_AGENTIC_SELECTION_RECORDS_JSONL": str(run_root / "cohort/selection-records.jsonl"),
                "SEARCH_AGENTIC_PROMPT_COUNT": str(cohort["prompt_count"]),
                "SEARCH_AGENTIC_PROMPT_SELECTION_SEED": str(PROMPT_SELECTION_SEED),
                "SEARCH_AGENTIC_PROMPT_SHARD_INDEX": "0", "SEARCH_AGENTIC_PROMPT_SHARD_COUNT": "1",
                "SEARCH_AGENTIC_PRODUCTION_CONDITIONS": "1",
                "SEARCH_AGENTIC_REQUEST_CONCURRENCY": "4", "SEARCH_AGENTIC_CELL_CONCURRENCY": "12",
            }
            command = _command(run_root, slug, account, partition, schedule)
            intent_path = model_root / "submission-intent.json"
            intent = {"requested_at_utc": _now(), "command": command, "status": "submission_requested"}
            # fsync both file and directory before sbatch can create any allocation.
            _durable_json(intent_path, intent)
            manifest["status"] = "submitting"
            manifest["models"][slug]["status"] = "submission_requested"
            _durable_json(manifest_path, manifest)
            try:
                result = subprocess.run(command, env=worker_env, capture_output=True, text=True, check=False, timeout=60)
                intent["returncode"] = result.returncode
                parsed = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", result.stdout.strip())
                if result.returncode != 0 or parsed is None:
                    raise RuntimeError("sbatch did not return an unambiguous job ID; inspect Slurm before any new approval")
                job_id = parsed.group(1)
                with (model_root / "job-id.txt").open("x", encoding="utf-8") as stream:
                    stream.write(job_id + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                _sync_directory(model_root)
                intent.update(status="submitted", job_id=job_id)
                manifest["models"][slug].update(status="submitted", job_id=job_id)
                if parsed.group(2):
                    intent["slurm_cluster"] = parsed.group(2)
                    manifest["models"][slug]["slurm_cluster"] = parsed.group(2)
                _durable_json(intent_path, intent)
                _durable_json(manifest_path, manifest)
            except BaseException as error:
                intent.update(status="submission_uncertain", error_type=type(error).__name__)
                manifest["status"] = "submission_incomplete"
                manifest["models"][slug].update(status="submission_uncertain", error_type=type(error).__name__)
                _durable_json(intent_path, intent)
                _durable_json(manifest_path, manifest)
                raise
        manifest.update(status="submitted", submitted_at_utc=_now())
        _durable_json(manifest_path, manifest)
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cohort-root", "run-root", "profile-root"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("source-git-commit", "account", "partition"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--approved-walltime", default=APPROVED_WALLTIME)
    parser.add_argument("--workers-per-model", type=int, default=WORKERS_PER_MODEL)
    parser.add_argument("--maximum-total-gpu-hours", type=int, default=MAXIMUM_GPU_HOURS)
    parser.add_argument(
        "--resume-from-run-root", type=Path, action="append",
        help=(
            "Reuse a matching prior run's durable shared claims; repeat to create a "
            "validated immutable union before submission."
        ),
    )
    parser.add_argument(
        "--model", dest="model_slugs", action="append", choices=tuple(MODELS),
        help="Submit only this model; repeat for more than one. Defaults to both models.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true", help="Submit the explicitly approved jobs; otherwise prepare only")
    action.add_argument("--prepare-only", dest="submit", action="store_false")
    parser.set_defaults(submit=False)
    arguments = parser.parse_args()
    arguments.model_slugs = tuple(arguments.model_slugs or MODELS)
    try:
        manifest = submit_backlog(**vars(arguments))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError,
            importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(str(error)) from error
    print(f"BACKLOG_STATUS={manifest['status']}")
    print(f"RUN_ROOT={arguments.run_root.resolve()}")
    print(
        f"ALLOCATIONS={manifest['allocation_count']} "
        f"MAXIMUM_GPU_HOURS={manifest['maximum_total_gpu_hours']}"
    )
    for slug, model in manifest["models"].items():
        if "job_id" in model:
            print(f"{slug.upper()}_ARRAY_JOB={model['job_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
