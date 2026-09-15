"""Prepare and submit exactly two approved one-hour, new-prompt generator jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.inference_wave import (
    _atomic_json,
    _atomic_jsonl,
    build_inference_wave,
    write_inference_wave,
)
from analysis.scripts.prepare_agentic_new_cohort import prepare_new_cohort
from analysis.scripts.run_agentic_search_integration_smoke import (
    FrozenSnapshotSearchAdapter,
    _build_target_urls,
    _canonical,
    _cells,
    _load_calibration_prompts,
)
from analysis.scripts.search_vllm_stage import load_profile

MODELS = {
    "qwen38": {
        "model_id": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "profile": "model-config-a58eb7c40e545350abc1.serving-profile.json",
    },
    "llama4": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "model_revision": "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
        "profile": "model-config-2403370677dae49f8cd1.serving-profile.json",
    },
}
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
PROMPT_SELECTION_SEED = 20260912
COHORT_SEED = 20260916


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty file: {path}")
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ValueError(f"required environment variable is empty: {name}")
    return value


def _snapshot(cache: Path, model_id: str, revision: str) -> dict[str, Any]:
    snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        json.loads(_file(snapshot / name).read_text())
    index = json.loads(_file(snapshot / "model.safetensors.index.json").read_text())
    weights = sorted(set(index["weight_map"].values()))
    if not weights:
        raise ValueError(f"model weight map is empty: {snapshot}")
    for name in weights:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"weight path escapes snapshot: {name}")
        _file(snapshot / relative)
    return {"path": str(snapshot), "weight_shards": len(weights)}


def _preflight(environment: dict[str, str], profile_root: Path) -> dict[str, Any]:
    if _require(environment, "GEODML_APPROVED_WALLTIME") != "01:00:00":
        raise ValueError("this paired trial requires approval for two 01:00:00 jobs")
    _require(environment, "GEODML_ALLOCATION_ESTIMATE")
    commit = _require(environment, "GEODML_EXECUTION_COMMIT")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("execution commit must be a full lowercase Git SHA")
    repository = Path(_require(environment, "GEODML_EXECUTION_REPOSITORY")).resolve()
    if repository != REPOSITORY_ROOT:
        raise ValueError("run the submit helper from the pinned execution repository")
    environment["GEODML_EXECUTION_REPOSITORY"] = str(repository)
    for arguments in (("rev-parse", "HEAD"), ("status", "--porcelain", "--untracked-files=all")):
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True, text=True, check=True,
        )
        if result.stdout.strip() != (commit if arguments[0] == "rev-parse" else ""):
            raise ValueError("execution repository is dirty or has the wrong commit")
    for name in ("HF_HUB_CACHE", "GEODML_CACHE_ROOT", "ACL_ARR_VENV"):
        path = Path(_require(environment, name)).resolve()
        if not path.is_dir():
            raise ValueError(f"configured directory does not exist: {name}")
        environment[name] = str(path)
    _file(Path(environment["ACL_ARR_VENV"]) / "bin/activate")
    if importlib.metadata.version("sentence-transformers") != "6.0.1":
        raise ValueError("the generator requires sentence-transformers 6.0.1")
    if shutil.which("sbatch", path=environment.get("PATH")) is None:
        raise ValueError("sbatch is not available in this shell")
    launcher = repository / "analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"
    _file(launcher)
    if not os.access(launcher, os.X_OK):
        raise ValueError("the generation worker must be executable")
    _file(repository / "analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch")
    cache = Path(environment["HF_HUB_CACHE"])
    models = {}
    for slug, specification in MODELS.items():
        profile_path = _file(profile_root / specification["profile"])
        profile = load_profile(profile_path)
        expected = {key: specification[key] for key in ("model_id", "model_revision")}
        serving = profile["serving"]
        if profile["model"] != expected or any(
            serving.get(key) != value
            for key, value in {"tensor_parallel_size": 4, "data_parallel_size": 1,
                               "host": "127.0.0.1", "port": 8010}.items()
        ):
            raise ValueError(f"wrong model or serving topology in {profile_path}")
        models[slug] = {
            **expected,
            "profile": str(profile_path),
            "profile_sha256": _hash(profile_path),
            "snapshot": _snapshot(cache, expected["model_id"], expected["model_revision"]),
        }
    bge = Path(environment.get("SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT", str(
        cache / "models--BAAI--bge-reranker-v2-m3/snapshots" / BGE_REVISION
    ))).resolve()
    if environment.get("SEARCH_AGENTIC_CROSS_ENCODER_REVISION", BGE_REVISION) != BGE_REVISION:
        raise ValueError("the cross-encoder revision differs from the frozen revision")
    if bge.name != BGE_REVISION:
        raise ValueError("the cross-encoder snapshot directory has the wrong revision")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        _file(bge / name)
    environment["SEARCH_AGENTIC_CROSS_ENCODER_SNAPSHOT"] = str(bge)
    environment["SEARCH_AGENTIC_CROSS_ENCODER_REVISION"] = BGE_REVISION
    for name in ("SEARCH_AGENTIC_DDG_SNAPSHOT", "SEARCH_AGENTIC_SEARXNG_SNAPSHOT"):
        environment[name] = str(_file(Path(_require(environment, name)).resolve()))
    return models


def submit_trial(
    *, selection_root: Path, output: Path, profile_root: Path,
    account: str, partition: str, submit: bool,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze one shared cohort, then submit once per model without automatic retries."""
    if not submit:
        raise ValueError("explicit --submit is required after allocation approval")
    output = output.resolve()
    if output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("trial output must be outside the pinned source repository")
    if output.exists():
        raise FileExistsError(f"trial already exists; inspect its jobs instead of resubmitting: {output}")
    if not account.strip() or not partition.strip():
        raise ValueError("an explicit account and partition are required")
    env = dict(os.environ if environment is None else environment)
    models = _preflight(env, profile_root.resolve())
    # Exclusive creation also rejects a concurrent invocation after preflight.
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "run_manifest.json"
    manifest: dict[str, Any] = {
        "format_version": "agentic-paired-new-cohort-trial-v1",
        "status": "preparing", "scientific_result": False,
        "started_at_utc": _now(), "git_commit": env["GEODML_EXECUTION_COMMIT"],
        "approved_walltime_per_job": "01:00:00", "maximum_total_gpu_hours": 8,
        "allocation_estimate": env["GEODML_ALLOCATION_ESTIMATE"],
        "resources_per_job": {"nodes": 1, "gpus": 4, "cpus": 32, "memory": "512G"},
        "account": account, "partition": partition,
        "prompt_count": 120, "cells_per_model": 1440,
        "prompt_selection_seed": PROMPT_SELECTION_SEED, "cohort_seed": COHORT_SEED,
        "request_concurrency": 4, "cell_concurrency": 12,
        "cutoff_policy": "Slurm may time out; retain completed per-cell artifacts, retry only missing cells",
        "models": {slug: {**model, "status": "not_submitted"} for slug, model in models.items()},
    }
    _atomic_json(manifest_path, manifest)
    try:
        cohort = output / "cohort"
        cohort_manifest = prepare_new_cohort(
            selection_root.resolve(), cohort, source_git_commit=env["GEODML_EXECUTION_COMMIT"],
            prompt_count=120, axis_bins=20, master_seed=COHORT_SEED,
        )
        prompts = _load_calibration_prompts(
            cohort / "pilot-prompts.jsonl", cohort / "selection-records.jsonl",
            prompt_count=120, seed=PROMPT_SELECTION_SEED,
        )
        cells = _cells(prompts)
        if len(prompts) != 120 or len(cells) != 1440:
            raise ValueError("paired cohort does not contain 120 prompts and 1440 cells")
        adapters = {
            engine: FrozenSnapshotSearchAdapter(engine, Path(env[name]))
            for engine, name in (("duckduckgo", "SEARCH_AGENTIC_DDG_SNAPSHOT"),
                                 ("searxng", "SEARCH_AGENTIC_SEARXNG_SNAPSHOT"))
        }
        if not set.intersection(*(adapter.keywords for adapter in adapters.values())):
            raise ValueError("search snapshots have no shared keyword")
        selection_audit: dict[str, dict[str, int]] = {}
        targets = _build_target_urls(
            prompts, adapters, seed=PROMPT_SELECTION_SEED, selection_audit=selection_audit,
        )
        if len(targets) != 240:
            raise ValueError("paired cohort must have 240 prompt-engine target URLs")
        manifest["target_selection"] = {
            "policy": "exact-keyword-else-deterministic-lexical-v1",
            "counts": selection_audit,
            "target_url_count": len(targets),
            "sha256": hashlib.sha256(_canonical([
                {"prompt_id": prompt, "engine": engine, "target_url": url}
                for (prompt, engine), url in sorted(targets.items())
            ])).hexdigest(),
        }
        manifest["search_snapshots"] = {
            engine: {"path": str(adapter.path), "sha256": adapter.snapshot_sha256}
            for engine, adapter in adapters.items()
        }
        tasks = [{"cell_id": cell.cell_id, **cell.core} for cell in cells]
        tasks_path = output / "tasks.jsonl"
        _atomic_jsonl(tasks_path, tasks)
        manifest["cohort_manifest"] = {"path": str(cohort_manifest), "sha256": _hash(cohort_manifest)}
        manifest["tasks"] = {"path": str(tasks_path), "sha256": _hash(tasks_path)}
        for slug in MODELS:
            model_root = output / "models" / slug
            write_inference_wave(
                model_root / "wave",
                wave=build_inference_wave(
                    tasks, task_id_field="cell_id", completed_task_ids=set(), worker_count=1,
                ),
                source_tasks_path=tasks_path,
            )
            (model_root / "logs").mkdir()
        manifest["status"] = "prepared"
        _atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="preparation_failed", error=f"{type(error).__name__}: {error}")
        _atomic_json(manifest_path, manifest)
        raise

    for slug, model in models.items():
        model_root = output / "models" / slug
        log_root = model_root / "logs"
        worker_env = {
            **{key: value for key, value in env.items()
               if not key.startswith(("SLURM_", "SBATCH_"))},
            "GEODML_MODEL_SLUG": slug,
            "GEODML_WAVE_ROOT": str(model_root / "wave"),
            "GEODML_WAVE_OUTPUT_ROOT": str(model_root / "outputs"),
            "GEODML_WAVE_LOG_ROOT": str(log_root),
            "GEODML_WORKER_LAUNCHER": str(REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter/run_agentic_generation_worker.sh"),
            "GEODML_WORKER_STDOUT": str(log_root / "slurm-%A_%a.out"),
            "GEODML_WORKER_STDERR": str(log_root / "slurm-%A_%a.err"),
            "SEARCH_AGENTIC_PROFILE": model["profile"],
            "SEARCH_AGENTIC_PROMPTS_JSONL": str(output / "cohort/pilot-prompts.jsonl"),
            "SEARCH_AGENTIC_SELECTION_RECORDS_JSONL": str(output / "cohort/selection-records.jsonl"),
            "SEARCH_AGENTIC_PROMPT_COUNT": "120",
            "SEARCH_AGENTIC_PROMPT_SELECTION_SEED": str(PROMPT_SELECTION_SEED),
            "SEARCH_AGENTIC_PROMPT_SHARD_INDEX": "0", "SEARCH_AGENTIC_PROMPT_SHARD_COUNT": "1",
            "SEARCH_AGENTIC_PRODUCTION_CONDITIONS": "1",
            "SEARCH_AGENTIC_REQUEST_CONCURRENCY": "4", "SEARCH_AGENTIC_CELL_CONCURRENCY": "12",
        }
        command = [
            "sbatch", "--parsable", "--no-requeue", "--array=0-0",
            f"--account={account}", f"--partition={partition}",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--mem=512G",
            "--gres=gpu:4", "--time=01:00:00", "--export=ALL",
            f"--job-name=geodml-{slug}-new120-1h",
            f"--chdir={REPOSITORY_ROOT}",
            f"--output={worker_env['GEODML_WORKER_STDOUT']}",
            f"--error={worker_env['GEODML_WORKER_STDERR']}",
            str(REPOSITORY_ROOT / "analysis/scripts/slurm/jupiter/run_inference_wave_worker.sbatch"),
        ]
        intent = {"requested_at_utc": _now(), "command": command, "status": "submission_requested"}
        _atomic_json(model_root / "submission-intent.json", intent)
        manifest["status"] = "submitting"
        manifest["models"][slug]["status"] = "submission_requested"
        _atomic_json(manifest_path, manifest)
        try:
            result = subprocess.run(
                command, env=worker_env, capture_output=True, text=True,
                check=False, timeout=60,
            )
            intent.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            parsed = re.fullmatch(r"([0-9]+)(?:;([A-Za-z0-9_.-]+))?", result.stdout.strip())
            if result.returncode != 0 or parsed is None:
                raise RuntimeError("sbatch did not return an unambiguous accepted job ID; inspect Slurm before any retry")
            job_id = parsed.group(1)
            with (model_root / "job-id.txt").open("x", encoding="utf-8") as stream:
                stream.write(job_id + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            manifest["models"][slug].update(status="submitted", job_id=job_id)
            if parsed.group(2):
                manifest["models"][slug]["slurm_cluster"] = parsed.group(2)
            intent.update(status="submitted", job_id=job_id)
            _atomic_json(model_root / "submission-intent.json", intent)
            _atomic_json(manifest_path, manifest)
            print(f"{slug.upper()}_JOB={job_id}", flush=True)
        except BaseException as error:
            manifest["status"] = "submission_incomplete"
            manifest["models"][slug].update(status="submission_uncertain", error=f"{type(error).__name__}: {error}")
            intent.update(status="submission_uncertain", error=f"{type(error).__name__}: {error}")
            _atomic_json(model_root / "submission-intent.json", intent)
            _atomic_json(manifest_path, manifest)
            raise
    manifest.update(status="submitted", submitted_at_utc=_now())
    _atomic_json(manifest_path, manifest)
    print(f"TRIAL_ROOT={output}")
    print("PAIRED_TRIAL_SUBMISSION=PASS jobs=2 maximum_gpu_hours=8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--submit", action="store_true")
    arguments = parser.parse_args()
    try:
        submit_trial(
            selection_root=arguments.selection_root, output=arguments.output_dir,
            profile_root=arguments.profile_root, account=arguments.account,
            partition=arguments.partition, submit=arguments.submit,
        )
    except (
        OSError, ValueError, KeyError, TypeError, RuntimeError,
        subprocess.SubprocessError, importlib.metadata.PackageNotFoundError,
    ) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
