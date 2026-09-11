#!/usr/bin/env python3
"""Audit agentic-search launch prerequisites without loading model weights."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_search import (  # noqa: E402
    ContextCompactor,
    ExperimentalCondition,
    IdentityConditionHook,
    LexicalOverlapScorer,
    ParallelExpansionV1,
    ReactiveSnippetLoopV1,
    ScriptedLLM,
    StaticSearchAdapter,
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    revision: str
    configuration_id: str
    max_model_len: int


EXPECTED_MODELS = (
    ModelSpec(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
        "model-config-2403370677dae49f8cd1",
        40960,
    ),
    ModelSpec(
        "Qwen/Qwen2.5-72B-Instruct",
        "495f39366efef23836d0cfae4fbe635880d2be31",
        "model-config-fbe629168c7384f96048",
        41984,
    ),
    ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "2dc98e2afe4face0e4ce40972a915c45368bd34a",
        "model-config-05b4f1c9ea3e2b8524e5",
        43008,
    ),
    ModelSpec(
        "Qwen/Qwen3.8-27B",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "model-config-a58eb7c40e545350abc1",
        41984,
    ),
)
EXPECTED_METHODS = ("Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1")
EXPECTED_CONDITIONS = ("natural", "ablated", "shuffled")
EXPECTED_ENGINES = ("duckduckgo", "searxng")
CROSS_ENCODER_ID = "BAAI/bge-reranker-v2-m3"
SEARCH_COLUMNS = frozenset({"keyword", "position", "title", "url", "snippet"})
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tekken.json",
    "sentencepiece.bpe.model",
    "vocab.json",
)


@dataclass(frozen=True, slots=True)
class ReadinessInputs:
    repository: Path
    model_snapshots: Path
    cross_encoder_snapshot: Path
    search_snapshots: Mapping[str, Path]
    smoke_outputs: Mapping[str, Path]
    check_runtime_packages: bool = True
    require_clean_git: bool = True


def build_experiment_matrix() -> list[dict[str, Any]]:
    """Return the deterministic 48-cell execution matrix."""

    rows: list[dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        for model in EXPECTED_MODELS:
            for condition in EXPECTED_CONDITIONS:
                for engine in EXPECTED_ENGINES:
                    identity = {
                        "method": method,
                        "model_id": model.model_id,
                        "model_revision": model.revision,
                        "model_configuration_id": model.configuration_id,
                        "max_model_len": model.max_model_len,
                        "condition": condition,
                        "search_engine": engine,
                    }
                    digest = hashlib.sha256(
                        _canonical(identity).encode("utf-8")
                    ).hexdigest()[:20]
                    rows.append({"cell_id": f"agentic-cell-{digest}", **identity})
    return rows


def audit_readiness(inputs: ReadinessInputs) -> dict[str, Any]:
    """Run all read-only checks and return a JSON-serializable manifest."""

    checks = [
        _capture_check("matrix", _check_matrix),
        _capture_check("agentic_method_contracts", _check_agentic_contracts),
        _capture_check(
            "model_snapshots",
            lambda: _check_model_snapshots(inputs.model_snapshots),
        ),
        _capture_check(
            "cross_encoder_snapshot",
            lambda: _check_cross_encoder_snapshot(inputs.cross_encoder_snapshot),
        ),
        _capture_check(
            "search_snapshots",
            lambda: _check_search_snapshots(inputs.search_snapshots),
        ),
        _capture_check(
            "smoke_inference",
            lambda: _check_smoke_outputs(inputs.smoke_outputs),
        ),
        _capture_check(
            "repository",
            lambda: _check_repository(
                inputs.repository,
                require_clean_git=inputs.require_clean_git,
            ),
        ),
    ]
    if inputs.check_runtime_packages:
        checks.append(_capture_check("runtime_packages", _check_runtime_packages))
    matrix = build_experiment_matrix()
    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    result = {
        "format_version": "agentic-search-cluster-readiness-v1",
        "status": status,
        "scientific_result": False,
        "weights_loaded": False,
        "inference_started": False,
        "network_requests_started": False,
        "matrix_cell_count": len(matrix),
        "methods": list(EXPECTED_METHODS),
        "models": [asdict(model) for model in EXPECTED_MODELS],
        "conditions": list(EXPECTED_CONDITIONS),
        "search_engines": list(EXPECTED_ENGINES),
        "cross_encoder": CROSS_ENCODER_ID,
        "checks": checks,
        "matrix": matrix,
    }
    result["manifest_sha256"] = hashlib.sha256(
        _canonical(result).encode("utf-8")
    ).hexdigest()
    return result


def _capture_check(name: str, function: Any) -> dict[str, Any]:
    try:
        details = function()
    except Exception as error:
        return {
            "name": name,
            "status": "FAIL",
            "errors": [f"{type(error).__name__}: {error}"],
        }
    return {"name": name, "status": "PASS", "details": details, "errors": []}


def _check_matrix() -> dict[str, Any]:
    matrix = build_experiment_matrix()
    if len(matrix) != 48:
        raise ValueError(f"expected 48 matrix cells, found {len(matrix)}")
    identifiers = [str(row["cell_id"]) for row in matrix]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("matrix cell identifiers are not unique")
    return {
        "cell_count": len(matrix),
        "method_count": len(EXPECTED_METHODS),
        "model_count": len(EXPECTED_MODELS),
        "condition_count": len(EXPECTED_CONDITIONS),
        "search_engine_count": len(EXPECTED_ENGINES),
    }


def _check_agentic_contracts() -> dict[str, Any]:
    result = asyncio.run(_exercise_agentic_contracts())
    expected = len(EXPECTED_CONDITIONS) * len(EXPECTED_ENGINES) * len(EXPECTED_METHODS)
    if len(result) != expected:
        raise ValueError(f"expected {expected} contract exercises, found {len(result)}")
    return {"exercise_count": len(result), "exercises": result}


async def _exercise_agentic_contracts() -> list[dict[str, Any]]:
    exercises: list[dict[str, Any]] = []
    snippets = {
        "query one": [_snippet("one", "target evidence")],
        "query two": [_snippet("two", "other evidence")],
        "query three": [_snippet("three", "supporting evidence")],
        "reactive query": [_snippet("reactive", "target evidence")],
    }
    for engine in EXPECTED_ENGINES:
        for condition_text in EXPECTED_CONDITIONS:
            condition = ExperimentalCondition(condition_text)
            parallel_llm = ScriptedLLM([
                json.dumps({"queries": ["query one", "query two", "query three"]}),
                json.dumps({
                    "ranking": ["https://readiness.invalid/one"],
                    "answer": "parallel contract passed",
                }),
            ])
            parallel = ParallelExpansionV1(
                llm=parallel_llm,
                search=StaticSearchAdapter(engine, snippets),
                compactor=ContextCompactor(LexicalOverlapScorer()),
                condition_hook=IdentityConditionHook(),
            )
            parallel_result = await parallel.run("target", condition)
            _assert_trace(
                parallel_result.trace.to_dict(),
                required=("llm_call", "search", "deduplication", "condition", "compaction"),
            )
            exercises.append({
                "method": parallel_result.method_id,
                "condition": condition.value,
                "search_engine": engine,
                "llm_calls": len(parallel_llm.requests),
                "search_calls": 3,
            })

            reactive_llm = ScriptedLLM([
                json.dumps({"action": "search", "query": "reactive query"}),
                json.dumps({
                    "action": "finish",
                    "ranking": ["https://readiness.invalid/reactive"],
                    "answer": "reactive contract passed",
                }),
            ])
            reactive_search = StaticSearchAdapter(engine, snippets)
            reactive = ReactiveSnippetLoopV1(
                llm=reactive_llm,
                search=reactive_search,
                compactor=ContextCompactor(LexicalOverlapScorer()),
                condition_hook=IdentityConditionHook(),
            )
            reactive_result = await reactive.run("target", condition)
            _assert_trace(
                reactive_result.trace.to_dict(),
                required=("llm_call", "search", "condition", "compaction", "observation"),
            )
            exercises.append({
                "method": reactive_result.method_id,
                "condition": condition.value,
                "search_engine": engine,
                "llm_calls": len(reactive_llm.requests),
                "search_calls": len(reactive_search.calls),
            })
    return exercises


def _snippet(name: str, text: str) -> dict[str, str]:
    return {
        "url": f"https://readiness.invalid/{name}",
        "title": name,
        "text": text,
    }


def _assert_trace(trace: Mapping[str, Any], *, required: Sequence[str]) -> None:
    events = trace.get("events")
    if not isinstance(events, list):
        raise ValueError("agent trace has no events list")
    event_types = {str(row.get("event_type")) for row in events if isinstance(row, dict)}
    missing = set(required).difference(event_types)
    if missing:
        raise ValueError(f"agent trace is missing events: {sorted(missing)}")
    if not isinstance(trace.get("trace_sha256"), str):
        raise ValueError("agent trace has no SHA-256 identity")


def _check_model_snapshots(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    rows = value.get("models")
    if not isinstance(rows, list):
        raise ValueError("model-snapshots.json must contain a models list")
    by_id = {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict)
    }
    expected_ids = {model.model_id for model in EXPECTED_MODELS}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids.difference(by_id))
        extra = sorted(set(by_id).difference(expected_ids))
        raise ValueError(f"model panel mismatch; missing={missing} extra={extra}")
    verified = []
    for model in EXPECTED_MODELS:
        row = by_id[model.model_id]
        revision = str(row.get("revision", ""))
        if revision != model.revision:
            raise ValueError(f"revision mismatch for {model.model_id}: {revision}")
        snapshot = Path(str(row.get("snapshot", "")))
        files = _check_snapshot_files(snapshot, label=model.model_id)
        verified.append({
            "model_id": model.model_id,
            "revision": model.revision,
            "snapshot": str(snapshot.resolve()),
            **files,
        })
    return {
        "lock_file": _file_identity(path),
        "verified_models": verified,
        "weights_loaded": False,
    }


def _check_cross_encoder_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"missing snapshot directory for {CROSS_ENCODER_ID}: {path}")
    revision = path.resolve().name
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            "cross-encoder snapshot directory must be named with its 40-character revision"
        )
    files = _check_snapshot_files(path, label=CROSS_ENCODER_ID)
    return {
        "model_id": CROSS_ENCODER_ID,
        "revision": revision,
        "snapshot": str(path.resolve()),
        **files,
        "weights_loaded": False,
    }


def _check_snapshot_files(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"missing snapshot directory for {label}: {path}")
    config = path / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise ValueError(f"missing config.json for {label}: {path}")
    config_value = _read_object(config)
    architectures = config_value.get("architectures")
    text_config = config_value.get("text_config")
    if not architectures and isinstance(text_config, dict):
        architectures = text_config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError(f"config.json has no model architecture for {label}: {path}")
    tokenizer = next(
        (path / name for name in TOKENIZER_FILES if (path / name).is_file()),
        None,
    )
    if tokenizer is None or tokenizer.stat().st_size == 0:
        raise ValueError(f"missing tokenizer files for {label}: {path}")
    weights = sorted(
        candidate
        for candidate in path.glob("*.safetensors")
        if candidate.is_file() and candidate.stat().st_size > 0
    )
    if not weights:
        raise ValueError(f"missing nonempty safetensors for {label}: {path}")
    return {
        "config_sha256": _sha256(config),
        "architectures": [str(value) for value in architectures],
        "tokenizer_file": tokenizer.name,
        "weight_file_count": len(weights),
        "weight_bytes": sum(candidate.stat().st_size for candidate in weights),
    }


def _check_search_snapshots(paths: Mapping[str, Path]) -> dict[str, Any]:
    normalized = {
        _normalize_engine(name): Path(path)
        for name, path in paths.items()
    }
    missing = [engine for engine in EXPECTED_ENGINES if engine not in normalized]
    if missing:
        raise ValueError(f"missing search snapshot for {', '.join(missing)}")
    extra = sorted(set(normalized).difference(EXPECTED_ENGINES))
    if extra:
        raise ValueError(f"unexpected search engines: {extra}")
    verified = []
    for engine in EXPECTED_ENGINES:
        path = normalized[engine]
        columns = _search_columns(path)
        missing_columns = sorted(SEARCH_COLUMNS.difference(columns))
        if missing_columns:
            raise ValueError(
                f"{engine} snapshot is missing columns {missing_columns}: {path}"
            )
        verified.append({
            "search_engine": engine,
            "columns": sorted(columns),
            **_file_identity(path),
        })
    return {"verified_search_engines": verified, "network_requests_started": False}


def _check_smoke_outputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    expected_ids = {model.configuration_id for model in EXPECTED_MODELS}
    if set(paths) != expected_ids:
        missing = sorted(expected_ids.difference(paths))
        extra = sorted(set(paths).difference(expected_ids))
        raise ValueError(f"smoke-output panel mismatch; missing={missing} extra={extra}")
    verified = []
    for model in EXPECTED_MODELS:
        output = Path(paths[model.configuration_id])
        manifest_path = output / "run_manifest.json"
        outcomes_path = output / "outcomes.jsonl"
        manifest = _read_object(manifest_path)
        expected_identity = {
            "model_id": model.model_id,
            "model_revision": model.revision,
            "model_configuration_id": model.configuration_id,
        }
        actual_identity = {
            key: manifest.get(key)
            for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                f"smoke identity mismatch for {model.configuration_id}: {actual_identity}"
            )
        expected_summary = {
            "status": "checkpointed",
            "answer_max_tokens_override": 2048,
            "completed_count": 4,
            "failures_this_invocation": 0,
            "fake_backend": False,
        }
        actual_summary = {key: manifest.get(key) for key in expected_summary}
        if actual_summary != expected_summary:
            raise ValueError(
                f"smoke summary mismatch for {model.configuration_id}: {actual_summary}"
            )
        outcomes = _read_jsonl_objects(outcomes_path)
        pipelines = Counter(str(row.get("pipeline")) for row in outcomes)
        if pipelines != {"rerank": 3, "answer": 1}:
            raise ValueError(
                f"smoke pipeline coverage mismatch for {model.configuration_id}: "
                f"{dict(pipelines)}"
            )
        binding = manifest.get("serving_profile")
        if not isinstance(binding, dict):
            raise ValueError(f"smoke has no serving-profile binding: {model.configuration_id}")
        profile_path = Path(str(binding.get("path", "")))
        profile_identity = _serving_profile_identity(profile_path)
        if binding.get("sha256") != profile_identity["profile_sha256"]:
            raise ValueError(
                f"serving-profile hash mismatch for {model.configuration_id}"
            )
        verified.append({
            **expected_identity,
            **expected_summary,
            "pipelines": dict(sorted(pipelines.items())),
            "run_manifest": _file_identity(manifest_path),
            "outcomes": _file_identity(outcomes_path),
            "serving_profile": profile_identity,
        })
    return {"verified_models": verified, "inference_started_by_audit": False}


def _serving_profile_identity(path: Path) -> dict[str, Any]:
    """Verify and identify a serving profile by its canonical semantic hash."""

    profile = _read_object(path)
    expected = profile.get("profile_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"serving profile has no valid profile_sha256: {path}")
    core = {key: value for key, value in profile.items() if key != "profile_sha256"}
    actual = hashlib.sha256(_canonical(core).encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError(f"serving profile canonical hash mismatch: {path}")
    return {**_file_identity(path), "profile_sha256": expected}


def _search_columns(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty search snapshot: {path}")
    if path.suffix.casefold() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ValueError("pyarrow is required to inspect Parquet search snapshots") from error
        return set(parquet.read_schema(path).names)
    if path.suffix.casefold() in {".jsonl", ".json"}:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"search snapshot row is not an object: {path}")
                    return set(value)
        raise ValueError(f"search snapshot has no records: {path}")
    raise ValueError(f"unsupported search snapshot format: {path}")


def _check_repository(path: Path, *, require_clean_git: bool) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"repository directory does not exist: {path}")
    git_directory = path / ".git"
    if not git_directory.exists():
        if require_clean_git:
            raise ValueError(f"not a Git checkout: {path}")
        return {
            "path": str(path.resolve()),
            "git_commit": None,
            "git_check": "disabled-for-fixture",
        }
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if require_clean_git:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
        if status.strip():
            raise ValueError("repository has tracked or untracked changes")
        required = (
            "analysis/interpretability/pipeline/agentic_search.py",
            "analysis/scripts/verify_agentic_search_cluster_readiness.py",
        )
        subprocess.run(
            ["git", "-C", str(path), "ls-files", "--error-unmatch", *required],
            check=True,
            capture_output=True,
            text=True,
        )
    return {"path": str(path.resolve()), "git_commit": commit, "tracked_files_clean": True}


def _check_runtime_packages() -> dict[str, Any]:
    packages = {}
    for import_name, distribution in (
        ("vllm", "vllm"),
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "torch"),
    ):
        if util.find_spec(import_name) is None:
            raise ValueError(f"missing Python package: {distribution}")
        packages[distribution] = metadata.version(distribution)
    executable = shutil.which("vllm")
    if executable is None:
        raise ValueError("vllm executable is not on PATH")
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "vllm_executable": executable,
        "weights_loaded": False,
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file has no records: {path}")
    return rows


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_engine(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"ddg", "duckduckgo"}:
        return "duckduckgo"
    if normalized == "searxng":
        return normalized
    raise ValueError(f"unsupported search engine: {value}")


def _parse_search_snapshot(value: str) -> tuple[str, Path]:
    engine, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("use ENGINE=PATH for --search-snapshot")
    try:
        normalized = _normalize_engine(engine)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return normalized, Path(path)


def _parse_key_path(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition("=")
    if not separator or not key.strip() or not path:
        raise argparse.ArgumentTypeError("use KEY=PATH")
    return key.strip(), Path(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite readiness manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--model-snapshots", type=Path, required=True)
    parser.add_argument("--cross-encoder-snapshot", type=Path, required=True)
    parser.add_argument(
        "--search-snapshot",
        action="append",
        type=_parse_search_snapshot,
        required=True,
        metavar="ENGINE=PATH",
    )
    parser.add_argument(
        "--smoke-output",
        action="append",
        type=_parse_key_path,
        required=True,
        metavar="MODEL_CONFIG_ID=PATH",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-runtime-packages", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    search_snapshots = dict(arguments.search_snapshot)
    if len(search_snapshots) != len(arguments.search_snapshot):
        raise SystemExit("duplicate --search-snapshot engine")
    smoke_outputs = dict(arguments.smoke_output)
    if len(smoke_outputs) != len(arguments.smoke_output):
        raise SystemExit("duplicate --smoke-output model configuration")
    result = audit_readiness(ReadinessInputs(
        repository=arguments.repository,
        model_snapshots=arguments.model_snapshots,
        cross_encoder_snapshot=arguments.cross_encoder_snapshot,
        search_snapshots=search_snapshots,
        smoke_outputs=smoke_outputs,
        check_runtime_packages=not arguments.skip_runtime_packages,
        require_clean_git=True,
    ))
    for check in result["checks"]:
        print("READINESS_CHECK=" + json.dumps(check, sort_keys=True), flush=True)
    print(f"MATRIX_CELL_COUNT={result['matrix_cell_count']}", flush=True)
    print(f"AGENTIC_SEARCH_READINESS={result['status']}", flush=True)
    print("SCIENTIFIC_RESULT=false; no model weights loaded and no inference started", flush=True)
    if arguments.output is not None:
        _write_json_atomic(arguments.output, result)
        print(f"READINESS_MANIFEST={arguments.output.resolve()}", flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
