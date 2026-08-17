#!/usr/bin/env python3
"""Export and import resumable OpenAI-compatible readiness-judge batches.

This script deliberately does not submit paid provider jobs.  It creates the
portable JSONL request artifact accepted by OpenAI-compatible batch APIs and
imports the downloaded provider result while reusing the frozen readiness
rubric, strict parser, and task-level cache contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    ReadinessLabelTask,
    parse_readiness_judgment,
)
try:  # Support both package imports in tests and direct script execution.
    from analysis.scripts.run_semantic_readiness_judge import _render_retry_prompt  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct cluster invocation
    from run_semantic_readiness_judge import _render_retry_prompt  # noqa: E402


BATCH_FORMAT_VERSION = "semantic-readiness-openai-batch-v1"
BACKEND_NAME = "openai-compatible-batch"
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "max_tokens", "temperature"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    export = stages.add_parser("export")
    export.add_argument("--tasks", required=True)
    export.add_argument("--tasks-sha256", required=True)
    export.add_argument("--expected-tasks", required=True, type=int)
    export.add_argument("--judge-slot", required=True)
    export.add_argument("--provider", required=True)
    export.add_argument("--model", required=True)
    export.add_argument("--model-family", required=True)
    export.add_argument("--model-revision", required=True)
    export.add_argument("--expected-provider-model")
    export.add_argument("--output-dir", required=True)
    export.add_argument("--judge-output-dir", required=True)
    export.add_argument("--batch-endpoint", default="/v1/chat/completions")
    export.add_argument("--max-new-tokens", type=int, default=300)
    export.add_argument("--maximum-attempts", type=int, default=5)
    export.add_argument(
        "--request-options",
        help="Optional JSON object of frozen provider options, excluding reserved keys.",
    )

    ingest = stages.add_parser("import")
    ingest.add_argument("--tasks", required=True)
    ingest.add_argument("--export-manifest", required=True)
    ingest.add_argument(
        "--batch-requests",
        help="Submitted request JSONL; defaults to the path frozen in the export manifest.",
    )
    ingest.add_argument("--batch-output", required=True)
    ingest.add_argument("--provider-batch-id", required=True)
    ingest.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.stage == "export":
        _export_batch(args)
    else:
        _import_batch(args)
    return 0


def _export_batch(args) -> None:
    task_path = Path(args.tasks).resolve()
    actual_task_sha256 = _sha256_file(task_path)
    if actual_task_sha256 != args.tasks_sha256:
        raise SystemExit(
            "task-bank hash mismatch: "
            f"expected {args.tasks_sha256}, found {actual_task_sha256}"
        )
    if args.expected_tasks <= 0:
        raise SystemExit("expected task count must be positive")
    if args.max_new_tokens <= 0 or args.maximum_attempts <= 0:
        raise SystemExit("generation limits must be positive")
    _require_nonempty_identity(args)

    tasks = _load_slot_tasks(task_path, args.judge_slot, args.expected_tasks)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit(f"batch export directory already exists: {output}")
    request_options, request_options_sha256 = _load_request_options(
        getattr(args, "request_options", None)
    )
    endpoint = str(args.batch_endpoint).strip()
    if not endpoint.startswith("/"):
        raise SystemExit("batch endpoint must be an absolute API path")
    judge_output = Path(args.judge_output_dir).resolve()
    judge_output.mkdir(parents=True, exist_ok=True)
    identity = _judge_identity(
        task_file_sha256=actual_task_sha256,
        expected_tasks_for_slot=len(tasks),
        judge_slot=args.judge_slot,
        provider=args.provider,
        model=args.model,
        model_family=args.model_family,
        model_revision=args.model_revision,
        expected_provider_model=getattr(args, "expected_provider_model", None),
    )
    _freeze_identity(judge_output / "judge_identity.json", identity)
    cache_directory = judge_output / "task_cache"

    requests = []
    cached_task_ids = []
    for task in tasks:
        cache_path = _task_cache_path(cache_directory, task.task_id)
        if cache_path.exists():
            cached = _read_json(cache_path)
            _validate_cache_identity(
                cached,
                task,
                args.model,
                args.model_family,
                args.model_revision,
                required_backend=BACKEND_NAME,
                required_provider=args.provider,
            )
            cached_task_ids.append(task.task_id)
            continue
        attempts = _load_failure_attempts(
            cache_path.with_suffix(".failed.json"),
            task=task,
            model=args.model,
            model_family=args.model_family,
            model_revision=args.model_revision,
            provider=args.provider,
        )
        if len(attempts) >= args.maximum_attempts:
            raise SystemExit(
                f"judge task exhausted {args.maximum_attempts} attempts: {task.task_id}"
            )
        prompt = task.prompt
        if attempts and str(attempts[-1].get("raw", "")).strip():
            prompt = _render_retry_prompt(
                prompt,
                str(attempts[-1].get("error", "invalid provider response")),
                str(attempts[-1]["raw"]),
            )
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_new_tokens,
            "temperature": 0,
            **request_options,
        }
        requests.append(
            {
                "custom_id": task.task_id,
                "method": "POST",
                "url": endpoint,
                "body": body,
            }
        )

    if not requests:
        raise SystemExit("no pending tasks: every task in the judge slot is cached")

    output.mkdir(parents=True)
    request_path = output / "batch_requests.jsonl"
    _atomic_jsonl(request_path, requests)
    manifest = {
        "format_version": BATCH_FORMAT_VERSION,
        "artifact_role": "readiness_judge_batch_export",
        "scientific_result": False,
        "created_at": _utc_now(),
        "git_commit_sha": _git_commit_sha(),
        "task_file": str(task_path),
        "task_file_sha256": actual_task_sha256,
        "expected_tasks_for_slot": args.expected_tasks,
        "judge_slot": args.judge_slot,
        "provider": args.provider,
        "model": args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "expected_provider_model": getattr(args, "expected_provider_model", None),
        "backend": BACKEND_NAME,
        "batch_endpoint": endpoint,
        "max_new_tokens": args.max_new_tokens,
        "maximum_attempts": args.maximum_attempts,
        "request_options": request_options,
        "request_options_sha256": request_options_sha256,
        "request_count": len(requests),
        "request_task_ids": [row["custom_id"] for row in requests],
        "cached_task_count": len(cached_task_ids),
        "cached_task_ids": cached_task_ids,
        "request_file": str(request_path),
        "request_file_sha256": _sha256_file(request_path),
        "judge_output_dir": str(judge_output),
    }
    _atomic_json(output / "batch_manifest.json", manifest)
    print(f"requests: {request_path}")
    print(f"manifest: {output / 'batch_manifest.json'}")
    print(f"pending tasks: {len(requests)}")


def _import_batch(args) -> None:
    if not str(args.provider_batch_id).strip():
        raise SystemExit("provider-batch-id must be nonempty")
    manifest_path = Path(args.export_manifest).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != BATCH_FORMAT_VERSION:
        raise SystemExit("unsupported or missing batch export format version")
    for field in ("provider", "model", "model_family", "model_revision"):
        if not str(manifest.get(field, "")).strip():
            raise SystemExit(f"batch export manifest has no {field}")

    task_path = Path(args.tasks).resolve()
    actual_task_sha256 = _sha256_file(task_path)
    if actual_task_sha256 != manifest.get("task_file_sha256"):
        raise SystemExit("task bank does not match the batch export manifest")
    tasks = _load_slot_tasks(
        task_path,
        str(manifest["judge_slot"]),
        int(manifest["expected_tasks_for_slot"]),
    )
    task_by_id = {task.task_id: task for task in tasks}
    request_task_ids = tuple(str(value) for value in manifest["request_task_ids"])
    if len(request_task_ids) != len(set(request_task_ids)):
        raise SystemExit("batch export manifest contains duplicate request task IDs")
    unknown_requested = set(request_task_ids) - set(task_by_id)
    if unknown_requested:
        raise SystemExit("batch export manifest references tasks outside its judge slot")
    request_path = Path(
        args.batch_requests or str(manifest.get("request_file", ""))
    ).resolve()
    if not request_path.is_file():
        raise SystemExit(f"submitted batch request file does not exist: {request_path}")
    if _sha256_file(request_path) != manifest.get("request_file_sha256"):
        raise SystemExit("submitted batch request file does not match its export manifest")
    submitted_requests = _read_jsonl(request_path)
    submitted_task_ids = tuple(
        str(row.get("custom_id", "")) for row in submitted_requests
    )
    if submitted_task_ids != request_task_ids:
        raise SystemExit("submitted batch request rows do not match the export manifest")

    batch_output_path = Path(args.batch_output).resolve()
    batch_output_sha256 = _sha256_file(batch_output_path)
    result_rows = _read_jsonl(batch_output_path)
    result_by_id = {}
    for row in result_rows:
        custom_id = str(row.get("custom_id", ""))
        if not custom_id:
            raise SystemExit("batch result row is missing custom_id")
        if custom_id in result_by_id:
            raise SystemExit(f"duplicate batch result custom_id: {custom_id}")
        result_by_id[custom_id] = row
    unknown_results = set(result_by_id) - set(request_task_ids)
    if unknown_results:
        raise SystemExit("batch output contains results not declared by its export manifest")

    output = Path(args.output_dir).resolve()
    declared_output = Path(str(manifest["judge_output_dir"])).resolve()
    if output != declared_output:
        raise SystemExit(
            "judge output directory does not match the frozen batch export manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    cache_directory = output / "task_cache"
    cache_directory.mkdir(exist_ok=True)
    identity = _judge_identity(
        task_file_sha256=actual_task_sha256,
        expected_tasks_for_slot=len(tasks),
        judge_slot=str(manifest["judge_slot"]),
        provider=str(manifest["provider"]),
        model=str(manifest["model"]),
        model_family=str(manifest["model_family"]),
        model_revision=str(manifest["model_revision"]),
        expected_provider_model=_optional_string(
            manifest.get("expected_provider_model")
        ),
    )
    _freeze_identity(output / "judge_identity.json", identity)

    raw_import_directory = output / "batch_imports"
    raw_import_directory.mkdir(exist_ok=True)
    preserved_batch_path = raw_import_directory / f"{batch_output_sha256}.jsonl"
    import_manifest_path = raw_import_directory / f"{batch_output_sha256}.manifest.json"
    frozen_import_provenance = {
        "export_manifest_sha256": _sha256_file(manifest_path),
        "submitted_request_file_sha256": _sha256_file(request_path),
        "provider": manifest["provider"],
        "provider_batch_id": args.provider_batch_id,
        "batch_output_sha256": batch_output_sha256,
    }
    if import_manifest_path.exists():
        previous_import = _read_json(import_manifest_path)
        for field, value in frozen_import_provenance.items():
            if previous_import.get(field) != value:
                raise SystemExit(
                    f"refusing to change frozen batch-import provenance: {field}"
                )
    if not preserved_batch_path.exists():
        _atomic_text(preserved_batch_path, batch_output_path.read_text(encoding="utf-8"))

    imported_successes = 0
    imported_failures = 0
    already_cached = 0
    for task_id in request_task_ids:
        row = result_by_id.get(task_id)
        if row is None:
            continue
        task = task_by_id[task_id]
        cache_path = _task_cache_path(cache_directory, task_id)
        if cache_path.exists():
            cached = _read_json(cache_path)
            _validate_cache_identity(
                cached,
                task,
                str(manifest["model"]),
                str(manifest["model_family"]),
                str(manifest["model_revision"]),
                required_backend=BACKEND_NAME,
                required_provider=str(manifest["provider"]),
            )
            already_cached += 1
            continue

        raw_response, provider_model, provider_usage, error = _extract_batch_result(row)
        expected_provider_model = manifest.get("expected_provider_model")
        if (
            error is None
            and expected_provider_model
            and provider_model != expected_provider_model
        ):
            error = (
                "provider model mismatch: "
                f"expected {expected_provider_model!r}, found {provider_model!r}"
            )
        if error is None:
            try:
                parse_readiness_judgment(task, raw_response)
            except ValueError as exc:
                error = str(exc)

        failure_path = cache_path.with_suffix(".failed.json")
        attempts = _load_failure_attempts(
            failure_path,
            task=task,
            model=str(manifest["model"]),
            model_family=str(manifest["model_family"]),
            model_revision=str(manifest["model_revision"]),
            provider=str(manifest["provider"]),
        )
        if error is not None:
            attempt_identity = f"{batch_output_sha256}:{task_id}"
            if not any(
                item.get("batch_attempt_identity") == attempt_identity
                for item in attempts
            ):
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "error": error,
                        "raw": raw_response,
                        "provider_model": provider_model,
                        "provider_usage": provider_usage,
                        "provider_response": row,
                        "provider": manifest["provider"],
                        "provider_batch_id": args.provider_batch_id,
                        "batch_attempt_identity": attempt_identity,
                    }
                )
                _atomic_json(
                    failure_path,
                    {
                        "task_id": task.task_id,
                        "model": manifest["model"],
                        "model_family": manifest["model_family"],
                        "model_revision": manifest["model_revision"],
                        "provider": manifest["provider"],
                        "backend": BACKEND_NAME,
                        "attempts": attempts,
                    },
                )
            imported_failures += 1
            continue

        cached = {
            "task_id": task.task_id,
            "item_id": task.item_id,
            "judge_slot": task.judge_slot,
            "model": manifest["model"],
            "model_family": manifest["model_family"],
            "model_revision": manifest["model_revision"],
            "backend": BACKEND_NAME,
            "precision": None,
            "raw_response": raw_response,
            "rejected_attempts": attempts,
            "provider_model": provider_model,
            "provider_usage": provider_usage,
            "provider_response": row,
            "provider": manifest["provider"],
            "provider_batch_id": args.provider_batch_id,
            "batch_output_sha256": batch_output_sha256,
        }
        _atomic_json(cache_path, cached)
        failure_path.unlink(missing_ok=True)
        imported_successes += 1

    responses = []
    failed_task_ids = []
    missing_task_ids = []
    for task in tasks:
        cache_path = _task_cache_path(cache_directory, task.task_id)
        if cache_path.exists():
            cached = _read_json(cache_path)
            _validate_cache_identity(
                cached,
                task,
                str(manifest["model"]),
                str(manifest["model_family"]),
                str(manifest["model_revision"]),
                required_backend=BACKEND_NAME,
                required_provider=str(manifest["provider"]),
            )
            parse_readiness_judgment(task, str(cached["raw_response"]))
            responses.append(cached)
        elif cache_path.with_suffix(".failed.json").exists():
            failed_task_ids.append(task.task_id)
        else:
            missing_task_ids.append(task.task_id)
    _atomic_jsonl(output / "judge_responses.jsonl", responses)

    import_manifest = {
        "format_version": BATCH_FORMAT_VERSION,
        "artifact_role": "readiness_judge_batch_import",
        "scientific_result": False,
        "imported_at": _utc_now(),
        "git_commit_sha": _git_commit_sha(),
        "export_manifest": str(manifest_path),
        "export_manifest_sha256": frozen_import_provenance[
            "export_manifest_sha256"
        ],
        "submitted_request_file": str(request_path),
        "submitted_request_file_sha256": frozen_import_provenance[
            "submitted_request_file_sha256"
        ],
        "provider": manifest["provider"],
        "provider_batch_id": args.provider_batch_id,
        "batch_output": str(batch_output_path),
        "batch_output_sha256": batch_output_sha256,
        "batch_result_row_count": len(result_rows),
        "imported_success_count": imported_successes,
        "imported_failure_count": imported_failures,
        "already_cached_count": already_cached,
    }
    if not import_manifest_path.exists():
        _atomic_json(import_manifest_path, import_manifest)
    _atomic_json(
        output / "run_manifest.json",
        {
            **identity,
            "artifact_role": "raw_judge_responses",
            "scientific_result": False,
            "task_file": str(task_path),
            "task_count_for_slot": len(tasks),
            "selected_task_count": len(tasks),
            "completed_count": len(responses),
            "failed_count": len(failed_task_ids),
            "failed_task_ids": failed_task_ids,
            "missing_count": len(missing_task_ids),
            "missing_task_ids": missing_task_ids,
            "is_complete": len(responses) == len(tasks),
            "updated_at": _utc_now(),
            "runtime_environment": {
                "hostname": os.uname().nodename,
                "python_executable": sys.executable,
            },
        },
    )
    print(f"judge output: {output}")
    print(f"completed: {len(responses)}/{len(tasks)}")
    print(f"failed: {len(failed_task_ids)}")
    print(f"missing: {len(missing_task_ids)}")


def _load_slot_tasks(
    path: Path,
    judge_slot: str,
    expected_tasks: int,
) -> tuple[ReadinessLabelTask, ...]:
    tasks = tuple(
        ReadinessLabelTask(**row)
        for row in _read_jsonl(path)
        if str(row.get("judge_slot", "")) == judge_slot
    )
    if len(tasks) != expected_tasks:
        raise SystemExit(
            f"judge-slot task-count mismatch: expected {expected_tasks}, found {len(tasks)}"
        )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise SystemExit("judge slot contains duplicate task IDs")
    return tasks


def _load_request_options(path_value: str | None) -> tuple[dict[str, object], str | None]:
    if not path_value:
        return {}, None
    path = Path(path_value).resolve()
    value = _read_json(path)
    if not isinstance(value, dict):
        raise SystemExit("request options must contain one JSON object")
    overlap = _RESERVED_BODY_KEYS & set(value)
    if overlap:
        raise SystemExit(f"request options override reserved keys: {sorted(overlap)}")
    return value, _sha256_file(path)


def _require_nonempty_identity(args) -> None:
    for name in ("judge_slot", "provider", "model", "model_family", "model_revision"):
        if not str(getattr(args, name, "")).strip():
            raise SystemExit(f"{name.replace('_', '-')} must be nonempty")


def _task_cache_path(cache_directory: Path, task_id: str) -> Path:
    return cache_directory / f"{task_id.replace(':', '_')}.json"


def _validate_cache_identity(
    cached: dict[str, object],
    task: ReadinessLabelTask,
    model: str,
    model_family: str,
    model_revision: str,
    *,
    required_backend: str | None = None,
    required_provider: str | None = None,
) -> None:
    if cached.get("task_id") != task.task_id:
        raise SystemExit(f"cache task identity mismatch: {task.task_id}")
    if cached.get("model") != model:
        raise SystemExit(f"cache model mismatch: {task.task_id}")
    if cached.get("model_family") != model_family:
        raise SystemExit(f"cache model family mismatch: {task.task_id}")
    if cached.get("model_revision") != model_revision:
        raise SystemExit(f"cache model revision mismatch: {task.task_id}")
    if required_backend is not None and cached.get("backend") != required_backend:
        raise SystemExit(f"cache backend mismatch: {task.task_id}")
    if required_provider is not None and cached.get("provider") != required_provider:
        raise SystemExit(f"cache provider mismatch: {task.task_id}")


def _load_failure_attempts(
    path: Path,
    *,
    task: ReadinessLabelTask,
    model: str,
    model_family: str,
    model_revision: str,
    provider: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    expected = {
        "task_id": task.task_id,
        "model": model,
        "model_family": model_family,
        "model_revision": model_revision,
        "provider": provider,
        "backend": BACKEND_NAME,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise SystemExit(f"failed cache {field} mismatch: {task.task_id}")
    attempts = payload.get("attempts", [])
    if not isinstance(attempts, list):
        raise SystemExit(f"invalid failed-attempt cache: {path}")
    return list(attempts)


def _extract_batch_result(
    row: dict[str, object],
) -> tuple[str, str | None, object, str | None]:
    top_error = row.get("error")
    response = row.get("response")
    if top_error:
        return "", None, None, f"provider batch error: {top_error}"
    if not isinstance(response, dict):
        return "", None, None, "provider batch row has no response object"
    status_code = response.get("status_code")
    body = response.get("body")
    if status_code != 200:
        return "", None, None, f"provider HTTP status is {status_code!r}"
    if not isinstance(body, dict):
        return "", None, None, "provider response body is not an object"
    provider_model = body.get("model")
    usage = body.get("usage")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return (
            "",
            _optional_string(provider_model),
            usage,
            "provider response must contain one choice",
        )
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return "", _optional_string(provider_model), usage, "provider choice has no message"
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        return "", _optional_string(provider_model), usage, "provider message content is empty"
    return content, _optional_string(provider_model), usage, None


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _judge_identity(
    *,
    task_file_sha256: str,
    expected_tasks_for_slot: int,
    judge_slot: str,
    provider: str,
    model: str,
    model_family: str,
    model_revision: str,
    expected_provider_model: str | None,
) -> dict[str, object]:
    return {
        "format_version": BATCH_FORMAT_VERSION,
        "task_file_sha256": task_file_sha256,
        "expected_tasks_for_slot": expected_tasks_for_slot,
        "judge_slot": judge_slot,
        "provider": provider,
        "model": model,
        "model_family": model_family,
        "model_revision": model_revision,
        "expected_provider_model": expected_provider_model,
        "backend": BACKEND_NAME,
    }


def _freeze_identity(path: Path, identity: dict[str, object]) -> None:
    if path.exists():
        if _read_json(path) != identity:
            raise SystemExit(f"refusing to change frozen judge identity: {path}")
        return
    _atomic_json(path, identity)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
