#!/usr/bin/env python3
"""Build and run one profiled vLLM endpoint for a search pipeline stage."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.request import urlopen


FORMAT_VERSION = "search-vllm-serving-profile-v1"
APPROVAL_FORMAT_VERSION = "search-vllm-benchmark-approval-v1"
RUNTIME_FORMAT_VERSION = "search-vllm-runtime-v1"
CACHE_SUBDIRECTORIES = {
    "VLLM_CACHE_ROOT": "vllm",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer-workspace",
    "TRTLLM_DG_CACHE_DIR": "tensorrt-llm",
}
CACHE_VARIABLES = tuple(CACHE_SUBDIRECTORIES)
_PROFILE_KEYS = {
    "format_version",
    "stage",
    "model",
    "runtime",
    "server_argv",
    "serving",
    "features",
    "visible_gpu_assignment",
    "cache_policy",
    "profile_sha256",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _profile_hash(record: Mapping[str, Any]) -> str:
    core = {key: value for key, value in record.items() if key != "profile_sha256"}
    return hashlib.sha256(_canonical(core)).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_gpu_inventory(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(values) != 4:
        raise ValueError("exactly four visible GPUs are required")
    normalized = []
    for value in values:
        index = value.get("index")
        uuid = str(value.get("uuid", "")).strip()
        name = str(value.get("name", "")).strip()
        memory = value.get("memory_total_mib")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("visible GPU indexes must be non-negative integers")
        if not uuid or not name:
            raise ValueError("visible GPU UUID and name must be nonempty")
        if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
            raise ValueError("visible GPU memory must be a positive integer MiB value")
        normalized.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": memory,
            }
        )
    if len({item["index"] for item in normalized}) != 4:
        raise ValueError("visible GPU indexes must be unique")
    if len({item["uuid"] for item in normalized}) != 4:
        raise ValueError("visible GPU UUIDs must be unique")
    return sorted(normalized, key=lambda item: item["index"])


def _validate_gpu_environment(
    values: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str | None,
    expected_gpu_name_pattern: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gpus = _normalize_gpu_inventory(values)
    if (
        not isinstance(expected_gpu_name_pattern, str)
        or not expected_gpu_name_pattern
    ):
        raise ValueError("expected GPU name pattern must be nonempty")
    try:
        expected_gpu_name = re.compile(expected_gpu_name_pattern)
    except re.error as exc:
        raise ValueError(
            "expected GPU name pattern must be a valid regular expression"
        ) from exc
    unexpected_names = [
        gpu["name"] for gpu in gpus if expected_gpu_name.search(gpu["name"]) is None
    ]
    if unexpected_names:
        raise ValueError(
            "every visible GPU must match the expected GPU name pattern "
            f"{expected_gpu_name_pattern!r}; got {unexpected_names!r}"
        )
    if not isinstance(cuda_visible_devices, str) or not cuda_visible_devices.strip():
        raise ValueError("CUDA_VISIBLE_DEVICES is required")
    tokens = tuple(token.strip() for token in cuda_visible_devices.split(","))
    if len(tokens) != 4 or any(not token for token in tokens) or len(set(tokens)) != 4:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly four assigned GPUs as unique tokens"
        )
    indexes = {str(gpu["index"]) for gpu in gpus}
    uuids = {gpu["uuid"] for gpu in gpus}
    if set(tokens) == indexes:
        mode = "inventory-index"
    elif set(tokens) == uuids:
        mode = "inventory-uuid"
    else:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must exactly match all inventory indexes or UUIDs"
        )
    return gpus, {"mode": mode, "tokens": list(tokens)}


def _option(argv: list[str], name: str, value: Any) -> None:
    if value is not None:
        argv.extend((name, str(value)))


def _validated_rope_scaling(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "factor",
        "original_max_position_embeddings",
        "type",
    }:
        raise ValueError(
            "rope scaling must contain exactly factor, "
            "original_max_position_embeddings, and type"
        )
    factor = value["factor"]
    original = value["original_max_position_embeddings"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or factor <= 1
    ):
        raise ValueError("rope scaling factor must be finite and greater than one")
    if (
        isinstance(original, bool)
        or not isinstance(original, int)
        or original < 1
    ):
        raise ValueError("rope scaling original context must be a positive integer")
    if value["type"] != "yarn":
        raise ValueError("rope scaling type must be yarn")
    return {
        "factor": float(factor),
        "original_max_position_embeddings": original,
        "type": "yarn",
    }


def _server_argv(
    *,
    vllm_executable: str,
    model_id: str,
    model_revision: str,
    host: str,
    port: int,
    data_parallel_size: int,
    tensor_parallel_size: int,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    language_model_only: bool,
    tokenizer_mode: str | None,
    attention_backend: str | None,
    config_format: str | None,
    load_format: str | None,
    structured_outputs_config: Mapping[str, Any],
    rope_scaling: Mapping[str, Any] | None = None,
    enforce_eager: bool = False,
    disable_custom_all_reduce: bool = False,
) -> list[str]:
    argv = [
        vllm_executable,
        "serve",
        model_id,
        "--revision",
        model_revision,
        "--served-model-name",
        model_id,
    ]
    if language_model_only:
        argv.append("--language-model-only")
    if enforce_eager:
        argv.append("--enforce-eager")
    if disable_custom_all_reduce:
        argv.append("--disable-custom-all-reduce")
    _option(argv, "--tokenizer-mode", tokenizer_mode)
    _option(argv, "--attention-backend", attention_backend)
    _option(argv, "--config-format", config_format)
    _option(argv, "--load-format", load_format)
    if rope_scaling is not None:
        argv.extend(
            (
                "--rope-scaling",
                json.dumps(rope_scaling, sort_keys=True, separators=(",", ":")),
            )
        )
    argv.extend(("--host", host, "--port", str(port)))
    if data_parallel_size > 1:
        argv.extend(("--data-parallel-size", str(data_parallel_size)))
    argv.extend(
        (
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            dtype,
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            format(gpu_memory_utilization, ".12g"),
            "--enable-prefix-caching",
            "--no-enable-log-requests",
            "--trust-remote-code",
            "--structured-outputs-config",
            json.dumps(
                structured_outputs_config,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    return argv


def build_profile(
    *,
    stage: str,
    model_id: str,
    model_revision: str,
    vllm_executable: str,
    vllm_version: str,
    vllm_help: str,
    visible_gpus: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str | None,
    expected_gpu_name_pattern: str,
    host: str = "127.0.0.1",
    port: int = 8010,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 4,
    dtype: str = "bfloat16",
    max_model_len: int,
    gpu_memory_utilization: float = 0.90,
    request_concurrency: int = 8,
    language_model_only: bool = False,
    tokenizer_mode: str | None = None,
    attention_backend: str | None = None,
    config_format: str | None = None,
    load_format: str | None = None,
    structured_outputs_config: Mapping[str, Any] | None = None,
    rope_scaling: Mapping[str, Any] | None = None,
    enforce_eager: bool = False,
    disable_custom_all_reduce: bool = False,
) -> dict[str, Any]:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", stage) is None:
        raise ValueError("stage must contain lowercase letters, digits, and hyphens")
    if not model_id.strip():
        raise ValueError("model ID must be nonempty")
    if re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("model revision must be an immutable 40-character SHA")
    if not vllm_executable.strip() or not vllm_version.strip():
        raise ValueError("vLLM executable and version must be nonempty")
    if host != "127.0.0.1":
        raise ValueError("the stage endpoint must bind only to 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 through 65535")
    dp = _positive_integer(data_parallel_size, "data_parallel_size")
    tp = _positive_integer(tensor_parallel_size, "tensor_parallel_size")
    concurrency = _positive_integer(request_concurrency, "request_concurrency")
    context = _positive_integer(max_model_len, "max_model_len")
    gpus, _ = _validate_gpu_environment(
        visible_gpus,
        cuda_visible_devices,
        expected_gpu_name_pattern,
    )
    if dp * tp != len(gpus):
        raise ValueError("DP x TP must equal four visible GPUs")
    if not dtype.strip():
        raise ValueError("dtype must be nonempty")
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
    if structured_outputs_config is None:
        structured = {"backend": "xgrammar"}
    elif not isinstance(structured_outputs_config, Mapping):
        raise ValueError("structured outputs configuration must be a JSON object")
    else:
        structured = dict(structured_outputs_config)
    if not structured:
        raise ValueError("structured outputs configuration must be nonempty")
    rope = _validated_rope_scaling(rope_scaling)

    argv = _server_argv(
        vllm_executable=vllm_executable,
        model_id=model_id,
        model_revision=model_revision,
        host=host,
        port=port,
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        dtype=dtype,
        max_model_len=context,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=language_model_only,
        tokenizer_mode=tokenizer_mode,
        attention_backend=attention_backend,
        config_format=config_format,
        load_format=load_format,
        structured_outputs_config=structured,
        rope_scaling=rope,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    if dp > 1:
        if "--data-parallel-size" not in argv:
            raise ValueError("DP argv is missing --data-parallel-size")
        if "--data-parallel-size" not in vllm_help:
            raise ValueError("installed vLLM lacks --data-parallel-size")
    if language_model_only and "--language-model-only" not in vllm_help:
        raise ValueError("installed vLLM lacks --language-model-only")
    if enforce_eager and "--enforce-eager" not in vllm_help:
        raise ValueError("installed vLLM lacks --enforce-eager")
    if disable_custom_all_reduce and "--disable-custom-all-reduce" not in vllm_help:
        raise ValueError("installed vLLM lacks --disable-custom-all-reduce")
    if rope is not None and "--rope-scaling" not in vllm_help:
        raise ValueError("installed vLLM lacks --rope-scaling")

    record: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "model": {"model_id": model_id, "model_revision": model_revision},
        "runtime": {
            "vllm_executable": vllm_executable,
            "vllm_version": vllm_version,
            "vllm_help_sha256": hashlib.sha256(vllm_help.encode()).hexdigest(),
        },
        "server_argv": argv,
        "serving": {
            "host": host,
            "port": port,
            "public_base_url": f"http://{host}:{port}/v1",
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "request_concurrency": concurrency,
            "dtype": dtype,
            "max_model_len": context,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
        "features": {
            "language_model_only": language_model_only,
            "tokenizer_mode": tokenizer_mode,
            "attention_backend": attention_backend,
            "config_format": config_format,
            "load_format": load_format,
            "structured_outputs_config": structured,
            "prefix_caching": True,
            "log_requests": False,
            "trust_remote_code": True,
        },
        "visible_gpu_assignment": {
            "required_gpu_count": len(gpus),
            "expected_gpu_name_pattern": expected_gpu_name_pattern,
            "accepted_cuda_visible_devices": (
                "exact-inventory-indexes-or-exact-inventory-uuids"
            ),
            "parallel_topology": {
                "data_parallel_size": dp,
                "tensor_parallel_size": tp,
            },
        },
        "cache_policy": {
            "scope": "job-node-stage-full-profile-sha256",
            "writable_cache_sharing": "one-stage-server",
            "subdirectories": dict(CACHE_SUBDIRECTORIES),
        },
    }
    if rope is not None:
        record["features"]["rope_scaling"] = rope
    record["profile_sha256"] = _profile_hash(record)
    return record


def verify_profile(record: Mapping[str, Any]) -> dict[str, Any]:
    if set(record) != _PROFILE_KEYS:
        raise ValueError("serving profile fields are incomplete or unknown")
    if record.get("format_version") != FORMAT_VERSION:
        raise ValueError("serving profile format mismatch")
    expected = record.get("profile_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("serving profile hash is invalid")
    if _profile_hash(record) != expected:
        raise ValueError("serving profile hash mismatch")
    stage_name = record.get("stage")
    if not isinstance(stage_name, str) or re.fullmatch(
        r"[a-z0-9][a-z0-9-]*", stage_name
    ) is None:
        raise ValueError("serving profile stage is invalid")

    def nested(name: str, keys: set[str]) -> Mapping[str, Any]:
        value = record.get(name)
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"serving profile {name} fields are invalid")
        return value

    model = nested("model", {"model_id", "model_revision"})
    if not isinstance(model["model_id"], str) or not model["model_id"].strip():
        raise ValueError("serving profile model ID is invalid")
    if not isinstance(model["model_revision"], str) or re.fullmatch(
        r"[0-9a-f]{40}", model["model_revision"]
    ) is None:
        raise ValueError("serving profile model revision is invalid")
    runtime = nested(
        "runtime",
        {"vllm_executable", "vllm_version", "vllm_help_sha256"},
    )
    if not all(
        isinstance(runtime[key], str) and runtime[key]
        for key in ("vllm_executable", "vllm_version")
    ) or re.fullmatch(r"[0-9a-f]{64}", str(runtime["vllm_help_sha256"])) is None:
        raise ValueError("serving profile runtime identity is invalid")
    serving = nested(
        "serving",
        {
            "host",
            "port",
            "public_base_url",
            "data_parallel_size",
            "tensor_parallel_size",
            "request_concurrency",
            "dtype",
            "max_model_len",
            "gpu_memory_utilization",
        },
    )
    if serving["host"] != "127.0.0.1":
        raise ValueError("serving profile host is invalid")
    port = serving["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("serving profile port is invalid")
    dp = _positive_integer(serving["data_parallel_size"], "data_parallel_size")
    tp = _positive_integer(serving["tensor_parallel_size"], "tensor_parallel_size")
    concurrency = _positive_integer(
        serving["request_concurrency"], "request_concurrency"
    )
    context = _positive_integer(serving["max_model_len"], "max_model_len")
    if dp * tp != 4:
        raise ValueError("serving profile topology must use exactly four GPUs")
    dtype = serving["dtype"]
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("serving profile dtype is invalid")
    utilization = serving["gpu_memory_utilization"]
    if (
        isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not math.isfinite(utilization)
        or not 0 < utilization <= 1
    ):
        raise ValueError("serving profile GPU memory utilization is invalid")
    base_url = f"http://{serving['host']}:{port}/v1"
    if serving["public_base_url"] != base_url:
        raise ValueError("serving profile endpoint fields disagree")
    feature_keys = {
            "language_model_only",
            "tokenizer_mode",
            "attention_backend",
            "config_format",
            "load_format",
            "structured_outputs_config",
            "prefix_caching",
            "log_requests",
            "trust_remote_code",
    }
    features_value = record.get("features")
    if not isinstance(features_value, Mapping) or set(features_value) not in (
        feature_keys,
        feature_keys | {"rope_scaling"},
    ):
        raise ValueError("serving profile features fields are invalid")
    features = features_value
    for key in ("language_model_only", "prefix_caching", "log_requests", "trust_remote_code"):
        if not isinstance(features[key], bool):
            raise ValueError(f"serving profile feature {key} is invalid")
    if (
        features["prefix_caching"] is not True
        or features["log_requests"] is not False
        or features["trust_remote_code"] is not True
    ):
        raise ValueError("serving profile fixed feature policy changed")
    for key in ("tokenizer_mode", "attention_backend", "config_format", "load_format"):
        if features[key] is not None and (
            not isinstance(features[key], str) or not features[key]
        ):
            raise ValueError(f"serving profile feature {key} is invalid")
    structured = features["structured_outputs_config"]
    if not isinstance(structured, Mapping) or not structured:
        raise ValueError("structured outputs configuration must be a JSON object")
    rope = _validated_rope_scaling(features.get("rope_scaling"))
    assignment = nested(
        "visible_gpu_assignment",
        {
            "required_gpu_count",
            "expected_gpu_name_pattern",
            "accepted_cuda_visible_devices",
            "parallel_topology",
        },
    )
    if assignment["required_gpu_count"] != 4:
        raise ValueError("serving profile must require four GPUs")
    pattern = assignment["expected_gpu_name_pattern"]
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("serving profile GPU name pattern is invalid")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError("serving profile GPU name pattern is invalid") from exc
    if assignment["accepted_cuda_visible_devices"] != (
        "exact-inventory-indexes-or-exact-inventory-uuids"
    ):
        raise ValueError("serving profile CUDA assignment policy is invalid")
    topology = assignment["parallel_topology"]
    if not isinstance(topology, Mapping) or dict(topology) != {
        "data_parallel_size": dp,
        "tensor_parallel_size": tp,
    }:
        raise ValueError("serving profile topology fields disagree")
    cache_policy = nested(
        "cache_policy",
        {"scope", "writable_cache_sharing", "subdirectories"},
    )
    if (
        cache_policy["scope"] != "job-node-stage-full-profile-sha256"
        or cache_policy["writable_cache_sharing"] != "one-stage-server"
        or cache_policy["subdirectories"] != CACHE_SUBDIRECTORIES
    ):
        raise ValueError("serving profile cache policy is invalid")
    argv = record.get("server_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("serving profile server argv is invalid")
    expected_argv = _server_argv(
        vllm_executable=runtime["vllm_executable"],
        model_id=model["model_id"],
        model_revision=model["model_revision"],
        host=serving["host"],
        port=port,
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        dtype=dtype,
        max_model_len=context,
        gpu_memory_utilization=float(utilization),
        language_model_only=features["language_model_only"],
        tokenizer_mode=features["tokenizer_mode"],
        attention_backend=features["attention_backend"],
        config_format=features["config_format"],
        load_format=features["load_format"],
        structured_outputs_config=structured,
        rope_scaling=rope,
        enforce_eager="--enforce-eager" in argv,
        disable_custom_all_reduce="--disable-custom-all-reduce" in argv,
    )
    if argv != expected_argv:
        raise ValueError("serving profile server argv disagrees with profile fields")
    if concurrency < 1:
        raise ValueError("serving profile request concurrency is invalid")
    return dict(record)


def load_profile(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read serving profile: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("serving profile must be a JSON object")
    return verify_profile(value)


def create_or_verify_profile(path: Path | str, record: Mapping[str, Any]) -> str:
    path = Path(path)
    desired = verify_profile(record)
    if path.exists():
        if load_profile(path) != desired:
            raise ValueError("serving profile changed; use a new output directory")
        return desired["profile_sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(desired, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if load_profile(path) != desired:
                raise ValueError("serving profile changed; use a new output directory")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return desired["profile_sha256"]


def _resolved_cache_base(cache_base: Path | str) -> Path:
    base = Path(cache_base).expanduser().absolute().resolve()
    home = Path.home().resolve()
    if base == home or home in base.parents:
        raise ValueError("serving caches and locks must be outside HOME")
    return base


def cache_environment(
    record: Mapping[str, Any],
    *,
    job_id: str,
    cache_base: Path | str,
    hostname: str | None = None,
) -> dict[str, str]:
    profile = verify_profile(record)
    if re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) is None:
        raise ValueError("Slurm job ID contains unsafe characters")
    node = hostname or socket.gethostname()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", node) is None:
        raise ValueError("hostname contains unsafe characters")
    resolved_base = _resolved_cache_base(cache_base)
    root = (
        resolved_base
        / f"job{job_id}"
        / node
        / profile["stage"]
        / profile["profile_sha256"]
    )
    environment = {
        variable: str(root / directory)
        for variable, directory in CACHE_SUBDIRECTORIES.items()
    }
    if len(set(environment.values())) != len(CACHE_VARIABLES):
        raise ValueError("serving cache directories must be distinct")
    return environment


def prepare_cache_directories(environment: Mapping[str, str]) -> None:
    if set(environment) != set(CACHE_VARIABLES):
        raise ValueError("all six serving cache variables are required")
    for value in environment.values():
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b"search-vllm-cache-check\n")
            probe.flush()
            os.fsync(probe.fileno())


def require_benchmark_approval(
    record: Mapping[str, Any],
    approval_path: Path | str | None,
) -> dict[str, Any] | None:
    profile = verify_profile(record)
    if profile["serving"]["data_parallel_size"] == 1:
        return None
    if approval_path is None:
        raise ValueError("DP scientific use requires a benchmark approval artifact")
    try:
        value = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark approval artifact: {exc}") from exc
    expected_keys = {
        "format_version",
        "profile_sha256",
        "benchmark_result_sha256",
        "approved_for_scientific_use",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("benchmark approval artifact format is invalid")
    if value["format_version"] != APPROVAL_FORMAT_VERSION:
        raise ValueError("benchmark approval artifact format is invalid")
    if value["profile_sha256"] != profile["profile_sha256"]:
        raise ValueError("benchmark approval profile hash does not match")
    if re.fullmatch(r"[0-9a-f]{64}", str(value["benchmark_result_sha256"])) is None:
        raise ValueError("benchmark approval result hash is invalid")
    if value["approved_for_scientific_use"] is not True:
        raise ValueError("benchmark approval does not authorize scientific use")
    return value


def benchmark_approval_binding(
    record: Mapping[str, Any],
    approval_path: Path | str | None,
) -> dict[str, Any] | None:
    value = require_benchmark_approval(record, approval_path)
    if value is None:
        return None
    path = Path(approval_path).expanduser().resolve(strict=True)
    content = path.read_bytes()
    if json.loads(content) != value:
        raise ValueError("benchmark approval artifact changed during validation")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "profile_sha256": value["profile_sha256"],
        "benchmark_result_sha256": value["benchmark_result_sha256"],
    }


def _runtime_gpu_environment(
    profile: Mapping[str, Any],
    visible_gpus: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record = verify_profile(profile)
    requirement = record["visible_gpu_assignment"]
    return _validate_gpu_environment(
        visible_gpus,
        cuda_visible_devices,
        requirement["expected_gpu_name_pattern"],
    )


def build_runtime_record(
    *,
    profile_path: Path | str,
    profile: Mapping[str, Any],
    visible_gpus: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str | None,
    cache_paths: Mapping[str, str],
    hostname: str,
    job_id: str,
    step_id: str | None,
    benchmark_approval: Mapping[str, Any] | None,
) -> dict[str, Any]:
    serving_profile = verify_profile(profile)
    inventory, assignment = _runtime_gpu_environment(
        serving_profile,
        visible_gpus,
        cuda_visible_devices,
    )
    if set(cache_paths) != set(CACHE_VARIABLES):
        raise ValueError("runtime record requires all six serving cache paths")
    if not hostname or not job_id:
        raise ValueError("runtime record requires hostname and Slurm job ID")
    return {
        "format_version": RUNTIME_FORMAT_VERSION,
        "profile_path": str(Path(profile_path).expanduser().resolve()),
        "profile_sha256": serving_profile["profile_sha256"],
        "visible_gpu_inventory": inventory,
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_assignment": assignment,
        "hostname": hostname,
        "slurm_job_id": job_id,
        "slurm_step_id": step_id or None,
        "cache_paths": dict(cache_paths),
        "benchmark_approval": (
            dict(benchmark_approval) if benchmark_approval is not None else None
        ),
        "server_status": "not_started",
        "server_exit_code": None,
        "controller_status": "not_started",
        "controller_exit_code": None,
        "started_at_epoch_seconds": time.time(),
        "finished_at_epoch_seconds": None,
    }


def write_runtime_record(
    server_log: Path | str,
    record: Mapping[str, Any],
) -> Path:
    path = Path(str(Path(server_log)) + ".runtime.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_runtime_binding(
    path: Path | str,
    *,
    expected_profile_sha256: str,
    require_approval: bool,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read serving runtime record: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("serving runtime record must be a JSON object")
    if value.get("format_version") != RUNTIME_FORMAT_VERSION:
        raise ValueError("serving runtime record format mismatch")
    if value.get("profile_sha256") != expected_profile_sha256:
        raise ValueError("serving runtime record profile hash mismatch")
    required_keys = {
        "format_version",
        "profile_path",
        "profile_sha256",
        "visible_gpu_inventory",
        "cuda_visible_devices",
        "cuda_assignment",
        "hostname",
        "slurm_job_id",
        "slurm_step_id",
        "cache_paths",
        "benchmark_approval",
        "server_status",
        "server_exit_code",
        "controller_status",
        "controller_exit_code",
        "started_at_epoch_seconds",
        "finished_at_epoch_seconds",
    }
    optional_keys = {
        "server_pid",
        "controller_pid",
        "signal",
        "failure",
        "cleanup_errors",
    }
    if not required_keys.issubset(value) or not set(value).issubset(
        required_keys | optional_keys
    ):
        raise ValueError("serving runtime record fields are malformed")
    profile_path = Path(str(value["profile_path"])).expanduser().resolve(strict=True)
    if load_profile(profile_path)["profile_sha256"] != expected_profile_sha256:
        raise ValueError("serving runtime record profile sidecar mismatch")
    profile = load_profile(profile_path)
    try:
        inventory, assignment = _runtime_gpu_environment(
            profile,
            value["visible_gpu_inventory"],
            value["cuda_visible_devices"],
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"serving runtime record GPU facts are malformed: {exc}") from exc
    if value["cuda_assignment"] != assignment:
        raise ValueError("serving runtime record CUDA assignment is inconsistent")
    hostname = value.get("hostname")
    if not isinstance(hostname, str) or re.fullmatch(
        r"[A-Za-z0-9_.-]+", hostname
    ) is None:
        raise ValueError("serving runtime record hostname is invalid")
    job_id = value.get("slurm_job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) is None:
        raise ValueError("serving runtime record Slurm job ID is invalid")
    step_id = value.get("slurm_step_id")
    if step_id is not None and (
        not isinstance(step_id, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", step_id) is None
    ):
        raise ValueError("serving runtime record Slurm step ID is invalid")
    cache_paths = value.get("cache_paths")
    if not isinstance(cache_paths, Mapping) or set(cache_paths) != set(CACHE_VARIABLES):
        raise ValueError("serving runtime record cache paths are malformed")
    normalized_cache_paths = {}
    for variable, directory in CACHE_SUBDIRECTORIES.items():
        cache_path = Path(str(cache_paths[variable])).expanduser()
        if not cache_path.is_absolute() or cache_path.name != directory:
            raise ValueError("serving runtime record cache paths are malformed")
        normalized_cache_paths[variable] = str(cache_path)
    if len(set(normalized_cache_paths.values())) != len(CACHE_VARIABLES):
        raise ValueError("serving runtime record cache paths are not distinct")
    if any(
        not isinstance(value.get(key), (int, float))
        or isinstance(value.get(key), bool)
        or not math.isfinite(value[key])
        for key in ("started_at_epoch_seconds",)
    ):
        raise ValueError("serving runtime record start time is invalid")
    finished_at = value.get("finished_at_epoch_seconds")
    if finished_at is not None and (
        not isinstance(finished_at, (int, float))
        or isinstance(finished_at, bool)
        or not math.isfinite(finished_at)
        or finished_at < value["started_at_epoch_seconds"]
    ):
        raise ValueError("serving runtime record finish time is invalid")
    server_states = {"not_started", "starting", "ready", "exited", "signaled", "terminated"}
    controller_states = {
        "not_started",
        "running",
        "exited",
        "signaled",
        "terminating_after_failure",
        "terminated",
    }
    if value.get("server_status") not in server_states:
        raise ValueError("serving runtime record server lifecycle is invalid")
    if value.get("controller_status") not in controller_states:
        raise ValueError("serving runtime record controller lifecycle is invalid")
    for key in ("server_exit_code", "controller_exit_code"):
        if value.get(key) is not None and (
            isinstance(value[key], bool) or not isinstance(value[key], int)
        ):
            raise ValueError("serving runtime record exit code is invalid")
    for key in ("server_pid", "controller_pid", "signal"):
        if key in value and (
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] <= 0
        ):
            raise ValueError(f"serving runtime record {key} is invalid")
    if "failure" in value and (
        not isinstance(value["failure"], str) or not value["failure"]
    ):
        raise ValueError("serving runtime record failure is invalid")
    if "cleanup_errors" in value and (
        not isinstance(value["cleanup_errors"], list)
        or not value["cleanup_errors"]
        or not all(isinstance(item, str) and item for item in value["cleanup_errors"])
    ):
        raise ValueError("serving runtime record cleanup errors are invalid")
    approval = value.get("benchmark_approval")
    if require_approval:
        required = {
            "path",
            "sha256",
            "profile_sha256",
            "benchmark_result_sha256",
        }
        if not isinstance(approval, Mapping) or set(approval) != required:
            raise ValueError("serving runtime record lacks benchmark approval")
        if approval["profile_sha256"] != expected_profile_sha256:
            raise ValueError("serving runtime benchmark approval profile mismatch")
        for key in ("sha256", "benchmark_result_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(approval[key])) is None:
                raise ValueError("serving runtime benchmark approval hash is invalid")
        approval_path = Path(str(approval["path"])).expanduser().resolve(strict=True)
        content = approval_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != approval["sha256"]:
            raise ValueError("serving runtime benchmark approval artifact hash mismatch")
        try:
            approval_value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("serving runtime benchmark approval is invalid JSON") from exc
        if not isinstance(approval_value, Mapping) or (
            approval_value.get("format_version") != APPROVAL_FORMAT_VERSION
            or approval_value.get("profile_sha256") != expected_profile_sha256
            or approval_value.get("benchmark_result_sha256")
            != approval["benchmark_result_sha256"]
            or approval_value.get("approved_for_scientific_use") is not True
        ):
            raise ValueError("serving runtime benchmark approval content mismatch")
        approval = dict(approval)
        approval["path"] = str(approval_path)
    elif approval is not None:
        raise ValueError("DP1 serving runtime must not claim benchmark approval")
    return {
        "path": str(resolved),
        "profile_path": str(profile_path),
        "profile_sha256": expected_profile_sha256,
        "visible_gpu_inventory": inventory,
        "cuda_visible_devices": value["cuda_visible_devices"],
        "cuda_assignment": assignment,
        "hostname": hostname,
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "cache_paths": normalized_cache_paths,
        "benchmark_approval": approval,
        "started_at_epoch_seconds": value["started_at_epoch_seconds"],
    }


def ensure_port_available(host: str, port: int, *, socket_factory=socket.socket) -> None:
    try:
        with socket_factory() as probe:
            probe.bind((host, port))
    except OSError as exc:
        raise ValueError(f"serving port {host}:{port} is already in use") from exc


@contextmanager
def port_ownership(record: Mapping[str, Any], *, cache_base: Path | str):
    profile = verify_profile(record)
    resolved_base = _resolved_cache_base(cache_base)
    host = profile["serving"]["host"]
    port = profile["serving"]["port"]
    node_host = socket.gethostname()
    key = hashlib.sha256(f"{node_host}:{host}:{port}".encode()).hexdigest()
    lock_directory = resolved_base / "port-locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{key}.lock"
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise ValueError(f"serving port {host}:{port} is already owned") from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def discover_visible_gpus() -> tuple[dict[str, Any], ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError("unexpected nvidia-smi GPU inventory")
        records.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(float(fields[3])),
            }
        )
    return tuple(_normalize_gpu_inventory(records))


def inspect_vllm(executable: Path | str) -> tuple[str, str, str]:
    path = str(Path(executable).expanduser().resolve(strict=True))

    def invoke(arguments: Sequence[str]) -> str:
        result = subprocess.run(
            [path, *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip()

    version = invoke(("--version",))
    help_text = invoke(("serve", "--help=all"))
    if not version or not help_text:
        raise ValueError("vLLM version and serve help must be available before load")
    return path, version, help_text


def verify_runtime(profile: Mapping[str, Any]) -> None:
    record = verify_profile(profile)
    executable, version, help_text = inspect_vllm(record["runtime"]["vllm_executable"])
    expected = record["runtime"]
    if executable != expected["vllm_executable"] or version != expected["vllm_version"]:
        raise ValueError("vLLM runtime changed after serving profile creation")
    if hashlib.sha256(help_text.encode()).hexdigest() != expected["vllm_help_sha256"]:
        raise ValueError("vLLM serve interface changed after serving profile creation")
    dp = record["serving"]["data_parallel_size"]
    if dp > 1 and (
        "--data-parallel-size" not in help_text
        or "--data-parallel-size" not in record["server_argv"]
    ):
        raise ValueError("installed vLLM lacks --data-parallel-size")


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_group(
    process: subprocess.Popen[Any] | None,
    grace_seconds: float = 30,
) -> None:
    if process is None:
        return
    process_group_id = process.pid
    group_alive = _group_exists(process_group_id)
    if group_alive:
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            group_alive = False
        deadline = time.monotonic() + max(0, grace_seconds)
        while group_alive and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            group_alive = _group_exists(process_group_id)
        if group_alive:
            group_alive = _group_exists(process_group_id)
        if group_alive:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                group_alive = False
    try:
        process.wait(timeout=max(1, grace_seconds))
    except subprocess.TimeoutExpired:
        pass
    if not group_alive:
        return
    deadline = time.monotonic() + max(1, min(5, grace_seconds))
    while _group_exists(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_exists(process_group_id):
        raise RuntimeError(f"process group {process_group_id} survived cleanup")


def wait_for_controller(
    controller: subprocess.Popen[Any],
    server: subprocess.Popen[Any],
    *,
    poll_seconds: float = 0.2,
) -> int:
    while True:
        controller_status = controller.poll()
        if controller_status is not None:
            return controller_status
        server_status = server.poll()
        if server_status is not None:
            _terminate_group(controller)
            raise RuntimeError(
                f"vLLM server exited with status {server_status} while controller was running"
            )
        time.sleep(poll_seconds)


def _log_tail(path: Path, limit: int = 100) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-limit:])
    except OSError:
        return ""


def wait_until_ready(
    process: subprocess.Popen[Any],
    *,
    base_url: str,
    model_id: str,
    timeout_seconds: float,
) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("startup timeout must be positive and finite")
    deadline = time.monotonic() + timeout_seconds
    last_error = "endpoint was not reachable"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM exited before readiness with status {return_code}")
        try:
            with urlopen(base_url.rstrip("/") + "/models", timeout=2) as response:
                payload = json.loads(response.read())
            served = {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if model_id in served:
                return
            last_error = f"endpoint serves {sorted(served)}, expected {model_id!r}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise RuntimeError(f"vLLM did not become ready before timeout: {last_error}")


class _StageSignal(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def record_stage_signal(
    runtime_record: dict[str, Any],
    signum: int,
    *,
    controller_started: bool,
) -> None:
    runtime_record["signal"] = signum
    runtime_record[
        "controller_status" if controller_started else "server_status"
    ] = "signaled"


def run_stage(
    profile_path: Path,
    server_log: Path,
    controller_command: Sequence[str],
    *,
    cache_base: Path | str,
    startup_timeout_seconds: float,
    benchmark_approval_path: Path | str | None = None,
) -> int:
    profile = load_profile(profile_path)
    approval = benchmark_approval_binding(profile, benchmark_approval_path)
    verify_runtime(profile)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise ValueError("SLURM_JOB_ID is required")
    command = list(controller_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a controller command is required")
    visible_gpus = discover_visible_gpus()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    _runtime_gpu_environment(profile, visible_gpus, cuda_visible_devices)
    caches = cache_environment(profile, job_id=job_id, cache_base=cache_base)
    runtime_record = build_runtime_record(
        profile_path=profile_path,
        profile=profile,
        visible_gpus=visible_gpus,
        cuda_visible_devices=cuda_visible_devices,
        cache_paths=caches,
        hostname=socket.gethostname(),
        job_id=job_id,
        step_id=os.environ.get("SLURM_STEP_ID") or os.environ.get("SLURM_STEPID"),
        benchmark_approval=approval,
    )
    with port_ownership(profile, cache_base=cache_base):
        server_log.parent.mkdir(parents=True, exist_ok=True)
        runtime_path = write_runtime_record(server_log, runtime_record)
        ensure_port_available(profile["serving"]["host"], profile["serving"]["port"])
        prepare_cache_directories(caches)
        server = None
        controller = None
        signaled = None

        def handle_signal(signum, _frame):
            raise _StageSignal(signum)

        previous = {
            signum: signal.signal(signum, handle_signal)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            with server_log.open("ab", buffering=0) as stream:
                server_environment = dict(os.environ)
                server_environment.update(caches)
                server = subprocess.Popen(
                    profile["server_argv"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=server_environment,
                    start_new_session=True,
                )
                runtime_record.update(server_status="starting", server_pid=server.pid)
                write_runtime_record(server_log, runtime_record)
                try:
                    wait_until_ready(
                        server,
                        base_url=profile["serving"]["public_base_url"],
                        model_id=profile["model"]["model_id"],
                        timeout_seconds=startup_timeout_seconds,
                    )
                except Exception as exc:
                    tail = _log_tail(server_log)
                    if tail:
                        raise RuntimeError(
                            f"{exc}\nLAST_SERVER_LOG_LINES\n{tail}"
                        ) from exc
                    raise
                runtime_record["server_status"] = "ready"
                write_runtime_record(server_log, runtime_record)
                controller_environment = dict(os.environ)
                controller_environment["GEODML_SERVING_RUNTIME_RECORD"] = str(
                    runtime_path.resolve()
                )
                controller = subprocess.Popen(
                    command,
                    env=controller_environment,
                    start_new_session=True,
                )
                runtime_record.update(
                    controller_status="running",
                    controller_pid=controller.pid,
                )
                write_runtime_record(server_log, runtime_record)
                try:
                    controller_status = wait_for_controller(controller, server)
                except RuntimeError as exc:
                    tail = _log_tail(server_log)
                    if tail:
                        raise RuntimeError(
                            f"{exc}\nLAST_SERVER_LOG_LINES\n{tail}"
                        ) from exc
                    raise
                runtime_record.update(
                    controller_status="exited",
                    controller_exit_code=controller_status,
                )
                return controller_status
        except _StageSignal as exc:
            signaled = exc.signum
            record_stage_signal(
                runtime_record,
                exc.signum,
                controller_started=controller is not None,
            )
        except BaseException as exc:
            runtime_record["failure"] = f"{type(exc).__name__}: {exc}"
            if server is not None and server.poll() is not None:
                runtime_record.update(
                    server_status="exited",
                    server_exit_code=server.poll(),
                )
            if controller is not None and controller.poll() is None:
                runtime_record["controller_status"] = "terminating_after_failure"
            raise
        finally:
            cleanup_errors = []
            for name, process in (("controller", controller), ("server", server)):
                try:
                    _terminate_group(process)
                except RuntimeError as exc:
                    cleanup_errors.append(str(exc))
                if process is not None:
                    exit_code = process.poll()
                    runtime_record[name + "_exit_code"] = exit_code
                    if runtime_record[name + "_status"] in (
                        "running",
                        "starting",
                        "ready",
                        "terminating_after_failure",
                    ):
                        runtime_record[name + "_status"] = "terminated"
            runtime_record["finished_at_epoch_seconds"] = time.time()
            if cleanup_errors:
                runtime_record["cleanup_errors"] = cleanup_errors
            try:
                write_runtime_record(server_log, runtime_record)
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            if cleanup_errors:
                raise RuntimeError("; ".join(cleanup_errors))
        assert signaled is not None
        return 128 + signaled


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--stage", required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--vllm-executable", type=Path, required=True)
    prepare.add_argument("--cache-base", required=True)
    prepare.add_argument("--expected-gpu-name-pattern", required=True)
    prepare.add_argument("--host", default="127.0.0.1")
    prepare.add_argument("--port", type=int, default=8010)
    prepare.add_argument("--data-parallel-size", type=int, default=1)
    prepare.add_argument("--tensor-parallel-size", type=int, default=4)
    prepare.add_argument("--dtype", default="bfloat16")
    prepare.add_argument("--max-model-len", type=int, required=True)
    prepare.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    prepare.add_argument("--enforce-eager", action="store_true")
    prepare.add_argument("--disable-custom-all-reduce", action="store_true")
    prepare.add_argument("--request-concurrency", type=int, default=8)
    prepare.add_argument("--language-model-only", action="store_true")
    prepare.add_argument("--tokenizer-mode")
    prepare.add_argument("--attention-backend")
    prepare.add_argument("--config-format")
    prepare.add_argument("--load-format")
    prepare.add_argument(
        "--structured-outputs-config",
        type=json.loads,
        default={"backend": "xgrammar"},
    )
    prepare.add_argument("--rope-scaling", type=json.loads)
    run = commands.add_parser("run")
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--server-log", type=Path, required=True)
    run.add_argument("--cache-base", required=True)
    run.add_argument("--startup-timeout-seconds", type=float, required=True)
    run.add_argument("--benchmark-approval", type=Path)
    run.add_argument("controller", nargs=argparse.REMAINDER)
    approval = commands.add_parser("validate-approval")
    approval.add_argument("--profile", type=Path, required=True)
    approval.add_argument("--approval", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        executable, version, help_text = inspect_vllm(args.vllm_executable)
        record = build_profile(
            stage=args.stage,
            model_id=args.model_id,
            model_revision=args.model_revision,
            vllm_executable=executable,
            vllm_version=version,
            vllm_help=help_text,
            visible_gpus=discover_visible_gpus(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            expected_gpu_name_pattern=args.expected_gpu_name_pattern,
            host=args.host,
            port=args.port,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            request_concurrency=args.request_concurrency,
            language_model_only=args.language_model_only,
            tokenizer_mode=args.tokenizer_mode,
            attention_backend=args.attention_backend,
            config_format=args.config_format,
            load_format=args.load_format,
            structured_outputs_config=args.structured_outputs_config,
            rope_scaling=args.rope_scaling,
            enforce_eager=args.enforce_eager,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        )
        job_id = os.environ.get("SLURM_JOB_ID", "")
        if not job_id:
            raise ValueError("SLURM_JOB_ID is required")
        cache_environment(record, job_id=job_id, cache_base=args.cache_base)
        print(create_or_verify_profile(args.profile, record))
        return 0
    if args.command == "validate-approval":
        require_benchmark_approval(load_profile(args.profile), args.approval)
        return 0
    return run_stage(
        args.profile,
        args.server_log,
        args.controller,
        cache_base=args.cache_base,
        startup_timeout_seconds=args.startup_timeout_seconds,
        benchmark_approval_path=args.benchmark_approval,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        raise SystemExit(f"STOP: {exc}") from exc
