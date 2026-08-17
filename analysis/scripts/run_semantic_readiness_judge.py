#!/usr/bin/env python3
"""Run one pinned high-quality LLM judge slot with resumable task caches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    ReadinessLabelTask,
    parse_readiness_judgment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-slot", required=True)
    parser.add_argument("--backend", choices=("local", "api", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-family",
        help="Frozen model-family label; required for production runs.",
    )
    parser.add_argument(
        "--model-revision",
        help="Immutable model revision; required for production runs.",
    )
    parser.add_argument("--precision", choices=("full", "4bit"), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Local-backend generation batch size; OOM batches split automatically.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-purpose",
        choices=("debug", "production"),
        default="debug",
        help="Production requires an immutable model revision.",
    )
    parser.add_argument(
        "--skip-task-id",
        action="append",
        default=[],
        help="Explicit task ID to retain as missing; may be repeated.",
    )
    args = parser.parse_args()
    _validate_run_contract(args)
    started_at = _utc_now()

    output = Path(args.output_dir).resolve()
    if output.exists() and not args.resume:
        raise SystemExit(f"output directory exists; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "task_cache"
    cache.mkdir(exist_ok=True)
    all_tasks = tuple(
        ReadinessLabelTask(**row) for row in _read_jsonl(Path(args.tasks).resolve())
    )
    slot_tasks = [item for item in all_tasks if item.judge_slot == args.judge_slot]
    if not slot_tasks:
        raise SystemExit(f"no tasks for judge slot {args.judge_slot!r}")
    stop = None if args.limit is None else args.start_index + args.limit
    tasks = slot_tasks[args.start_index:stop]
    skipped_task_ids = _validate_skipped_task_ids(tasks, args.skip_task_id)
    from interpretability.utils import make_ranker

    ranker = make_ranker(args.backend, args.model, precision=args.precision)
    if args.backend == "local" and args.batch_size > 1:
        responses = _run_local_batches(
            ranker,
            tasks,
            cache=cache,
            skipped_task_ids=skipped_task_ids,
            args=args,
        )
    else:
        responses = _run_serial_tasks(
            ranker,
            tasks,
            cache=cache,
            skipped_task_ids=skipped_task_ids,
            args=args,
        )
    _atomic_jsonl(output / "judge_responses.jsonl", responses)
    _atomic_json(
        output / "run_manifest.json",
        {
            "judge_slot": args.judge_slot,
            "model": args.model,
            "model_family": args.model_family,
            "model_revision": args.model_revision,
            "backend": args.backend,
            "precision": args.precision if args.backend == "local" else None,
            "batch_size": args.batch_size,
            "run_purpose": args.run_purpose,
            "task_file": str(Path(args.tasks).resolve()),
            "task_file_sha256": _sha256_file(Path(args.tasks).resolve()),
            "task_count_for_slot": len(slot_tasks),
            "selected_task_count": len(tasks),
            "start_index": args.start_index,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "maximum_attempts": args.maximum_attempts,
            "completed_count": len(responses),
            "skipped_count": len(skipped_task_ids),
            "skipped_task_ids": sorted(skipped_task_ids),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "git_commit_sha": _git_commit_sha(),
            "runtime_environment": _runtime_environment(),
            "artifact_role": "raw_judge_responses",
            "scientific_result": False,
        },
    )
    print(f"output: {output}")
    return 0


def _run_serial_tasks(
    ranker,
    tasks: list[ReadinessLabelTask],
    *,
    cache: Path,
    skipped_task_ids: frozenset[str],
    args,
) -> list[dict[str, object]]:
    responses = []
    for index, task in enumerate(tasks, 1):
        cache_path = cache / f"{task.task_id.replace(':', '_')}.json"
        failure_path = cache_path.with_suffix(".failed.json")
        if task.task_id in skipped_task_ids:
            print(f"[{index}/{len(tasks)}] skipped {task.task_id}", flush=True)
            continue
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("task_id") != task.task_id or cached.get("model") != args.model:
                raise SystemExit(f"cache identity mismatch: {cache_path}")
            cached_family = cached.get("model_family")
            if cached_family is not None and cached_family != args.model_family:
                raise SystemExit(f"cache model family mismatch: {cache_path}")
            cached_revision = cached.get("model_revision")
            if cached_revision is not None and cached_revision != args.model_revision:
                raise SystemExit(f"cache model revision mismatch: {cache_path}")
            parse_readiness_judgment(task, str(cached["raw_response"]))
            responses.append(cached)
            continue
        attempts = _load_rejected_attempts(failure_path) if args.resume else []
        for _ in range(args.maximum_attempts):
            prompt = task.prompt
            if attempts:
                prompt = _render_retry_prompt(
                    prompt,
                    str(attempts[-1]["error"]),
                    str(attempts[-1]["raw"]),
                )
            raw = str(
                ranker.rank(
                    prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
            )
            try:
                parse_readiness_judgment(task, raw)
            except ValueError as exc:
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "error": str(exc),
                        "raw": raw,
                    }
                )
                continue
            row = {
                "task_id": task.task_id,
                "item_id": task.item_id,
                "judge_slot": task.judge_slot,
                "model": args.model,
                "model_family": args.model_family,
                "model_revision": args.model_revision,
                "backend": args.backend,
                "precision": args.precision if args.backend == "local" else None,
                "raw_response": raw,
                "rejected_attempts": attempts,
            }
            _atomic_json(cache_path, row)
            failure_path.unlink(missing_ok=True)
            responses.append(row)
            break
        else:
            _atomic_json(
                failure_path,
                {
                    "task_id": task.task_id,
                    "model": args.model,
                    "model_family": args.model_family,
                    "backend": args.backend,
                    "attempts": attempts,
                },
            )
            raise SystemExit(f"judge exhausted attempts: {failure_path}")
        if index % 50 == 0 or index == len(tasks):
            print(f"[{index}/{len(tasks)}] {args.judge_slot}", flush=True)
    return responses


def _run_local_batches(
    ranker,
    tasks: list[ReadinessLabelTask],
    *,
    cache: Path,
    skipped_task_ids: frozenset[str],
    args,
) -> list[dict[str, object]]:
    completed: dict[str, dict[str, object]] = {}
    pending: list[tuple[ReadinessLabelTask, list[dict[str, object]]]] = []
    for task in tasks:
        if task.task_id in skipped_task_ids:
            continue
        cache_path = cache / f"{task.task_id.replace(':', '_')}.json"
        failure_path = cache_path.with_suffix(".failed.json")
        if cache_path.exists():
            row = json.loads(cache_path.read_text(encoding="utf-8"))
            _validate_cached_response(task, row, cache_path, args)
            completed[task.task_id] = row
        else:
            attempts = _load_rejected_attempts(failure_path) if args.resume else []
            pending.append((task, attempts))

    while pending:
        batch = pending[: args.batch_size]
        del pending[: args.batch_size]
        prompts = [
            _render_retry_prompt(task.prompt, str(attempts[-1]["error"]), str(attempts[-1]["raw"]))
            if attempts
            else task.prompt
            for task, attempts in batch
        ]
        raw_responses = ranker.rank_batch(
            prompts,
            max_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        if len(raw_responses) != len(batch):
            raise SystemExit(
                f"local batch returned {len(raw_responses)} responses for {len(batch)} tasks"
            )
        exhausted = []
        for (task, attempts), raw in zip(batch, raw_responses):
            cache_path = cache / f"{task.task_id.replace(':', '_')}.json"
            failure_path = cache_path.with_suffix(".failed.json")
            try:
                parse_readiness_judgment(task, str(raw))
            except ValueError as exc:
                attempts.append(
                    {"attempt": len(attempts) + 1, "error": str(exc), "raw": str(raw)}
                )
                _atomic_json(
                    failure_path,
                    {
                        "task_id": task.task_id,
                        "model": args.model,
                        "model_family": args.model_family,
                        "backend": args.backend,
                        "attempts": attempts,
                    },
                )
                if len(attempts) >= args.maximum_attempts:
                    exhausted.append(task.task_id)
                else:
                    pending.append((task, attempts))
                continue
            row = _response_row(task, str(raw), attempts, args)
            _atomic_json(cache_path, row)
            failure_path.unlink(missing_ok=True)
            completed[task.task_id] = row
        print(
            f"[{len(completed)}/{len(tasks) - len(skipped_task_ids)}] "
            f"{args.judge_slot}; pending={len(pending)}",
            flush=True,
        )
        if exhausted:
            raise SystemExit(f"judge exhausted attempts for task IDs: {sorted(exhausted)}")
    return [
        completed[task.task_id]
        for task in tasks
        if task.task_id not in skipped_task_ids
    ]


def _validate_cached_response(task, row, cache_path: Path, args) -> None:
    if row.get("task_id") != task.task_id or row.get("model") != args.model:
        raise SystemExit(f"cache identity mismatch: {cache_path}")
    if row.get("model_family") not in {None, args.model_family}:
        raise SystemExit(f"cache model family mismatch: {cache_path}")
    if row.get("model_revision") not in {None, args.model_revision}:
        raise SystemExit(f"cache model revision mismatch: {cache_path}")
    parse_readiness_judgment(task, str(row["raw_response"]))


def _response_row(task, raw: str, attempts, args) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "item_id": task.item_id,
        "judge_slot": task.judge_slot,
        "model": args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "backend": args.backend,
        "precision": args.precision if args.backend == "local" else None,
        "raw_response": raw,
        "rejected_attempts": attempts,
    }


def _validate_run_contract(args) -> None:
    if args.start_index < 0:
        raise SystemExit("start index must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")
    if args.max_new_tokens <= 0 or args.maximum_attempts <= 0:
        raise SystemExit("generation limits must be positive")
    if getattr(args, "batch_size", 1) <= 0:
        raise SystemExit("batch size must be positive")
    if getattr(args, "backend", "local") != "local" and getattr(args, "batch_size", 1) != 1:
        raise SystemExit("batch sizes above one currently require --backend local")
    if args.run_purpose == "production":
        if not str(args.model_family or "").strip():
            raise SystemExit("production runs require --model-family")
        if not str(args.model_revision or "").strip():
            raise SystemExit("production runs require --model-revision")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _runtime_environment() -> dict[str, object]:
    gpu_names: list[str] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_partition": os.getenv("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu_names": gpu_names,
        "python_executable": sys.executable,
    }


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_rejected_attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempts = payload.get("attempts", [])
    if not isinstance(attempts, list):
        raise SystemExit(f"invalid failed-attempt cache: {path}")
    return list(attempts)


def _validate_skipped_task_ids(
    tasks: list[ReadinessLabelTask],
    requested_task_ids: list[str],
) -> frozenset[str]:
    requested = frozenset(
        str(value).strip()
        for value in requested_task_ids
        if str(value).strip()
    )
    available = {task.task_id for task in tasks}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit(f"skip task IDs are outside the selected judge slice: {unknown}")
    return requested


def _render_retry_prompt(
    prompt: str,
    validation_error: str,
    previous_raw_response: str,
) -> str:
    return f'''{prompt}

Your previous response failed validation: {validation_error}
Repair the previous response shown below. Make the minimum correction required
by the validation error and preserve every other semantic judgment whenever it
is already valid.

<previous_invalid_response>
{previous_raw_response}
</previous_invalid_response>

Return exactly one JSON object and nothing else. Use these exact keys without
renaming, shortening, or adding keys:
{{
  "overall_readiness_0_100": <integer 0..100>,
  "information_seeking_1_7": <integer 1..7>,
  "evaluation_1_7": <integer 1..7>,
  "selection_commitment_1_7": <integer 1..7>,
  "action_implementation_1_7": <integer 1..7>,
  "category": <"information"|"criteria"|"comparison"|"selection"|"action"|"mixed"|"not_applicable">,
  "not_applicable": <true|false>,
  "ambiguity_1_7": <integer 1..7>,
  "confidence_0_1": <number 0..1>,
  "brief_reason": <1 to 20 words>
}}
The brief_reason must contain at most 20 whitespace-separated words; count them
before responding. If category and not_applicable previously disagreed, silently
re-read the original text and choose exactly one of these valid forms:
- applicable: category is one of "information", "criteria", "comparison",
  "selection", "action", or "mixed", and not_applicable is false;
- not applicable: category is "not_applicable" and not_applicable is true.
The invalid pair from the previous response must not be repeated. Preserve every
other valid field. Do not use category values such as evaluation or review.'''


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _atomic_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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


if __name__ == "__main__":
    raise SystemExit(main())
