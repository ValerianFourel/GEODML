#!/usr/bin/env python3
"""Run one causal-LM readiness judge as four data-parallel GPU workers.

Launch this script with ``torchrun --nproc-per-node=4``.  Each worker owns one
complete model replica and a deterministic strided shard of the frozen judge
slot.  Workers batch independent prompts, write disjoint task caches, and
merge the validated responses on rank zero.
"""

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
import time
from typing import Iterable, Sequence


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.semantic_readiness_dataset import (  # noqa: E402
    ReadinessLabelTask,
    parse_readiness_judgment,
)

try:  # Support package imports in tests and direct script execution.
    from analysis.scripts.run_semantic_readiness_judge import (  # noqa: E402
        _render_retry_prompt,
    )
except ModuleNotFoundError:  # pragma: no cover - direct cluster invocation
    from run_semantic_readiness_judge import _render_retry_prompt  # noqa: E402


BACKEND_NAME = "local-four-gpu-data-parallel"
FORMAT_VERSION = "semantic-readiness-four-gpu-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--tasks-sha256", required=True)
    parser.add_argument("--expected-tasks", required=True, type=int)
    parser.add_argument("--judge-slot", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--maximum-attempts", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-purpose",
        choices=("debug", "production"),
        default="debug",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    _validate_arguments(args)

    import torch
    import torch.distributed as distributed

    if not distributed.is_available():
        raise SystemExit("torch.distributed is unavailable")
    distributed.init_process_group(backend="gloo")
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    try:
        if world_size != args.expected_world_size:
            raise SystemExit(
                f"expected {args.expected_world_size} workers, found {world_size}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise SystemExit(
                "each torchrun worker must see the complete four-GPU allocation: "
                f"CUDA available={torch.cuda.is_available()}, "
                f"visible={torch.cuda.device_count()}, workers={world_size}"
            )
        torch.cuda.set_device(local_rank)

        task_path = Path(args.tasks).resolve()
        task_sha256 = _sha256_file(task_path)
        if task_sha256 != args.tasks_sha256:
            raise SystemExit(
                "task-bank hash mismatch: "
                f"expected {args.tasks_sha256}, found {task_sha256}"
            )
        slot_tasks = _load_slot_tasks(
            task_path,
            args.judge_slot,
            args.expected_tasks,
        )
        stop = None if args.limit is None else args.start_index + args.limit
        selected_tasks = slot_tasks[args.start_index:stop]
        if not selected_tasks:
            raise SystemExit("selected judge task slice is empty")
        worker_tasks = _shard_tasks(selected_tasks, rank, world_size)

        output = Path(args.output_dir).resolve()
        setup_error = _prepare_output(
            output,
            args=args,
            task_path=task_path,
            task_sha256=task_sha256,
            slot_task_count=len(slot_tasks),
            selected_task_count=len(selected_tasks),
            rank=rank,
            distributed=distributed,
        )
        if setup_error is not None:
            raise SystemExit(setup_error)

        cache_directory = output / "task_cache"
        worker_directory = output / "workers" / f"rank-{rank}"
        worker_directory.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        wall_started = time.monotonic()

        pending = []
        cached_count = 0
        exhausted_task_ids = []
        for task in worker_tasks:
            cache_path = _task_cache_path(cache_directory, task.task_id)
            if cache_path.exists():
                cached = _read_json(cache_path)
                _validate_cache_identity(cached, task, args)
                parse_readiness_judgment(task, str(cached["raw_response"]))
                cached_count += 1
                continue
            attempts = _load_failure_attempts(
                cache_path.with_suffix(".failed.json"), task, args
            )
            if len(attempts) >= args.maximum_attempts:
                exhausted_task_ids.append(task.task_id)
                continue
            pending.append((task, attempts))

        model = None
        load_seconds = 0.0
        generated_count = 0
        rejected_count = 0
        if pending:
            load_started = time.monotonic()
            model = FourGpuCausalJudge(
                args.model,
                local_rank=local_rank,
                max_input_tokens=args.max_input_tokens,
                attention_implementation=args.attention_implementation,
                disable_thinking=args.disable_thinking,
            )
            load_seconds = time.monotonic() - load_started
            print(
                f"[rank {rank}] loaded model on cuda:{local_rank} in "
                f"{load_seconds:.2f}s; pending={len(pending)}",
                flush=True,
            )

        while pending:
            current = pending[: args.batch_size]
            del pending[: args.batch_size]
            prompts = [
                _prompt_for_attempt(task, attempts)
                for task, attempts in current
            ]
            raw_responses = model.generate(
                prompts,
                max_new_tokens=args.max_new_tokens,
            )
            if len(raw_responses) != len(current):
                raise RuntimeError("batched generation changed response cardinality")

            for (task, attempts), raw_response in zip(current, raw_responses):
                cache_path = _task_cache_path(cache_directory, task.task_id)
                failure_path = cache_path.with_suffix(".failed.json")
                try:
                    parse_readiness_judgment(task, raw_response)
                except ValueError as exc:
                    attempts.append(
                        {
                            "attempt": len(attempts) + 1,
                            "error": str(exc),
                            "raw": raw_response,
                        }
                    )
                    rejected_count += 1
                    _atomic_json(
                        failure_path,
                        _failure_payload(task, attempts, args, rank),
                    )
                    if len(attempts) >= args.maximum_attempts:
                        exhausted_task_ids.append(task.task_id)
                    else:
                        pending.append((task, attempts))
                    continue

                _atomic_json(
                    cache_path,
                    {
                        "task_id": task.task_id,
                        "item_id": task.item_id,
                        "judge_slot": task.judge_slot,
                        "model": args.model,
                        "model_family": args.model_family,
                        "model_revision": args.model_revision,
                        "backend": BACKEND_NAME,
                        "precision": "bfloat16",
                        "worker_rank": rank,
                        "raw_response": raw_response,
                        "rejected_attempts": attempts,
                    },
                )
                failure_path.unlink(missing_ok=True)
                generated_count += 1

            completed_now = cached_count + generated_count
            print(
                f"[rank {rank}] completed={completed_now}/{len(worker_tasks)} "
                f"pending={len(pending)} rejected={rejected_count}",
                flush=True,
            )

        worker_manifest = {
            "format_version": FORMAT_VERSION,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "hostname": socket.gethostname(),
            "task_count": len(worker_tasks),
            "cached_count": cached_count,
            "generated_count": generated_count,
            "rejected_attempt_count": rejected_count,
            "exhausted_task_ids": exhausted_task_ids,
            "load_seconds": load_seconds,
            "wall_seconds": time.monotonic() - wall_started,
            "started_at": started_at,
            "completed_at": _utc_now(),
        }
        worker_manifest_path = worker_directory / "worker_manifest.json"
        if not _is_noop_completed_resume(
            output,
            args=args,
            cached_count=cached_count,
            generated_count=generated_count,
            exhausted_task_ids=exhausted_task_ids,
            worker_task_count=len(worker_tasks),
            worker_manifest_path=worker_manifest_path,
        ):
            _atomic_json(worker_manifest_path, worker_manifest)

        all_exhausted = [None for _ in range(world_size)]
        distributed.all_gather_object(all_exhausted, exhausted_task_ids)
        distributed.barrier()

        final_error = [None]
        if rank == 0:
            try:
                _merge_responses(
                    output,
                    selected_tasks,
                    slot_task_count=len(slot_tasks),
                    args=args,
                    task_path=task_path,
                    task_sha256=task_sha256,
                    all_exhausted=all_exhausted,
                )
            except Exception as exc:  # propagate rank-zero merge failures
                final_error[0] = f"{type(exc).__name__}: {exc}"
        distributed.broadcast_object_list(final_error, src=0)
        if final_error[0] is not None:
            raise SystemExit(final_error[0])
        return 0
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


class FourGpuCausalJudge:
    """One bf16 causal-LM replica bound to one torchrun local rank."""

    def __init__(
        self,
        model_path: str,
        *,
        local_rank: int,
        max_input_tokens: int,
        attention_implementation: str,
        disable_thinking: bool,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(f"cuda:{local_rank}")
        self.max_input_tokens = max_input_tokens
        self.disable_thinking = disable_thinking
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        architectures = tuple(config.architectures or ())
        if not any(name.endswith("ForCausalLM") for name in architectures):
            raise RuntimeError(
                "the optimized runner currently requires a causal-LM checkpoint; "
                f"found architectures={architectures!r}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation=attention_implementation,
        )
        self.model.eval()

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        try:
            return self._generate(prompts, max_new_tokens=max_new_tokens)
        except self.torch.OutOfMemoryError:
            if len(prompts) <= 1:
                raise
            self.torch.cuda.empty_cache()
            midpoint = len(prompts) // 2
            print(
                f"[cuda OOM] splitting batch {len(prompts)} into "
                f"{midpoint}+{len(prompts) - midpoint}",
                flush=True,
            )
            return self.generate(
                prompts[:midpoint], max_new_tokens=max_new_tokens
            ) + self.generate(
                prompts[midpoint:], max_new_tokens=max_new_tokens
            )

    def _generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
    ) -> list[str]:
        rendered = _render_chat_prompts(
            self.tokenizer,
            prompts,
            disable_thinking=self.disable_thinking,
        )
        inputs = self.tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_width = int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        continuations = generated[:, prompt_width:]
        return self.tokenizer.batch_decode(
            continuations,
            skip_special_tokens=True,
        )


def _render_chat_prompts(
    tokenizer,
    prompts: Sequence[str],
    *,
    disable_thinking: bool,
) -> list[str]:
    rendered = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        options = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if disable_thinking:
            options["enable_thinking"] = False
        rendered.append(tokenizer.apply_chat_template(messages, **options))
    return rendered


def _shard_tasks(
    tasks: Sequence[ReadinessLabelTask],
    rank: int,
    world_size: int,
) -> tuple[ReadinessLabelTask, ...]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world size")
    return tuple(tasks[rank::world_size])


def _prompt_for_attempt(
    task: ReadinessLabelTask,
    attempts: Sequence[dict[str, object]],
) -> str:
    if not attempts:
        return task.prompt
    previous = attempts[-1]
    return _render_retry_prompt(
        task.prompt,
        str(previous.get("error", "invalid model response")),
        str(previous.get("raw", "")),
    )


def _prepare_output(
    output: Path,
    *,
    args,
    task_path: Path,
    task_sha256: str,
    slot_task_count: int,
    selected_task_count: int,
    rank: int,
    distributed,
) -> str | None:
    status = [None]
    if rank == 0:
        try:
            if output.exists() and not args.resume:
                raise RuntimeError(f"output directory exists; pass --resume: {output}")
            output.mkdir(parents=True, exist_ok=True)
            (output / "task_cache").mkdir(exist_ok=True)
            (output / "workers").mkdir(exist_ok=True)
            identity = {
                "format_version": FORMAT_VERSION,
                "backend": BACKEND_NAME,
                "task_file": str(task_path),
                "task_file_sha256": task_sha256,
                "task_count_for_slot": slot_task_count,
                "selected_task_count": selected_task_count,
                "start_index": args.start_index,
                "limit": args.limit,
                "judge_slot": args.judge_slot,
                "model": args.model,
                "model_family": args.model_family,
                "model_revision": args.model_revision,
                "precision": "bfloat16",
                "world_size": args.expected_world_size,
                "batch_size_per_gpu": args.batch_size,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "maximum_attempts": args.maximum_attempts,
                "attention_implementation": args.attention_implementation,
                "disable_thinking": args.disable_thinking,
                "run_purpose": args.run_purpose,
            }
            identity_path = output / "judge_identity.json"
            if identity_path.exists():
                if _read_json(identity_path) != identity:
                    raise RuntimeError(
                        f"refusing to change frozen judge identity: {identity_path}"
                    )
            else:
                _atomic_json(identity_path, identity)
        except Exception as exc:
            status[0] = f"{type(exc).__name__}: {exc}"
    distributed.broadcast_object_list(status, src=0)
    distributed.barrier()
    return status[0]


def _merge_responses(
    output: Path,
    selected_tasks: Sequence[ReadinessLabelTask],
    *,
    slot_task_count: int,
    args,
    task_path: Path,
    task_sha256: str,
    all_exhausted: Sequence[Sequence[str]],
) -> None:
    cache_directory = output / "task_cache"
    responses = []
    failed_task_ids = []
    missing_task_ids = []
    for task in selected_tasks:
        cache_path = _task_cache_path(cache_directory, task.task_id)
        if cache_path.exists():
            cached = _read_json(cache_path)
            _validate_cache_identity(cached, task, args)
            parse_readiness_judgment(task, str(cached["raw_response"]))
            responses.append(cached)
        elif cache_path.with_suffix(".failed.json").exists():
            failed_task_ids.append(task.task_id)
        else:
            missing_task_ids.append(task.task_id)

    _atomic_jsonl(output / "judge_responses.jsonl", responses)
    worker_manifests = [
        _read_json(output / "workers" / f"rank-{rank}" / "worker_manifest.json")
        for rank in range(args.expected_world_size)
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "artifact_role": "raw_judge_responses",
        "scientific_result": False,
        "backend": BACKEND_NAME,
        "judge_slot": args.judge_slot,
        "model": args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "precision": "bfloat16",
        "task_file": str(task_path),
        "task_file_sha256": task_sha256,
        "task_count_for_slot": slot_task_count,
        "selected_task_count": len(selected_tasks),
        "start_index": args.start_index,
        "limit": args.limit,
        "world_size": args.expected_world_size,
        "batch_size_per_gpu": args.batch_size,
        "maximum_effective_batch_size": args.batch_size * args.expected_world_size,
        "completed_count": len(responses),
        "failed_count": len(failed_task_ids),
        "failed_task_ids": failed_task_ids,
        "missing_count": len(missing_task_ids),
        "missing_task_ids": missing_task_ids,
        "exhausted_task_ids_by_rank": list(all_exhausted),
        "is_complete": len(responses) == len(selected_tasks),
        "worker_manifests": worker_manifests,
        "git_commit_sha": _git_commit_sha(),
        "completed_at": _utc_now(),
    }
    if _completed_resume_matches(
        output,
        args=args,
        responses=responses,
        expected_manifest=manifest,
    ):
        print(
            f"[rank 0] resume validated {len(responses)}/{len(selected_tasks)} "
            f"cached responses; preserving completed artifacts: {output}",
            flush=True,
        )
        return
    _atomic_json(output / "run_manifest.json", manifest)
    if failed_task_ids or missing_task_ids:
        raise RuntimeError(
            "four-GPU judge run is incomplete: "
            f"completed={len(responses)}, failed={len(failed_task_ids)}, "
            f"missing={len(missing_task_ids)}"
        )
    print(
        f"[rank 0] merged {len(responses)}/{len(selected_tasks)} responses: {output}",
        flush=True,
    )


def _is_noop_completed_resume(
    output: Path,
    *,
    args,
    cached_count: int,
    generated_count: int,
    exhausted_task_ids: Sequence[str],
    worker_task_count: int,
    worker_manifest_path: Path,
) -> bool:
    if not args.resume:
        return False
    if generated_count or exhausted_task_ids or cached_count != worker_task_count:
        return False
    run_manifest_path = output / "run_manifest.json"
    if not run_manifest_path.exists() or not worker_manifest_path.exists():
        return False
    manifest = _read_json(run_manifest_path)
    return bool(manifest.get("is_complete"))


def _completed_resume_matches(
    output: Path,
    *,
    args,
    responses: Sequence[dict[str, object]],
    expected_manifest: dict[str, object],
) -> bool:
    if not args.resume:
        return False
    manifest_path = output / "run_manifest.json"
    response_path = output / "judge_responses.jsonl"
    if not manifest_path.exists() or not response_path.exists():
        return False
    existing_manifest = _read_json(manifest_path)
    identity_fields = (
        "format_version",
        "backend",
        "judge_slot",
        "model",
        "model_family",
        "model_revision",
        "task_file_sha256",
        "selected_task_count",
        "start_index",
        "limit",
        "world_size",
        "batch_size_per_gpu",
    )
    if any(
        existing_manifest.get(field) != expected_manifest.get(field)
        for field in identity_fields
    ):
        return False
    if not existing_manifest.get("is_complete"):
        return False
    return _read_jsonl(response_path) == list(responses)


def _validate_arguments(args) -> None:
    for name in ("judge_slot", "model", "model_family", "model_revision"):
        if not str(getattr(args, name, "")).strip():
            raise SystemExit(f"{name.replace('_', '-')} must be nonempty")
    for name in (
        "expected_tasks",
        "batch_size",
        "max_input_tokens",
        "max_new_tokens",
        "maximum_attempts",
        "expected_world_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")
    if args.start_index < 0:
        raise SystemExit("start-index must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")
    if args.run_purpose == "production" and args.limit is not None:
        raise SystemExit("production runs may not set --limit")


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
            f"judge-slot task-count mismatch: expected {expected_tasks}, "
            f"found {len(tasks)}"
        )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise SystemExit("judge slot contains duplicate task IDs")
    return tasks


def _task_cache_path(cache_directory: Path, task_id: str) -> Path:
    return cache_directory / f"{task_id.replace(':', '_')}.json"


def _validate_cache_identity(
    cached: dict[str, object],
    task: ReadinessLabelTask,
    args,
) -> None:
    expected = {
        "task_id": task.task_id,
        "judge_slot": task.judge_slot,
        "model": args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "backend": BACKEND_NAME,
    }
    for field, value in expected.items():
        if cached.get(field) != value:
            raise RuntimeError(f"cache {field} mismatch: {task.task_id}")


def _load_failure_attempts(
    path: Path,
    task: ReadinessLabelTask,
    args,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    expected = {
        "task_id": task.task_id,
        "judge_slot": task.judge_slot,
        "model": args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "backend": BACKEND_NAME,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"failed cache {field} mismatch: {task.task_id}")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeError(f"invalid failure cache: {path}")
    return list(attempts)


def _failure_payload(task, attempts, args, rank: int) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "item_id": task.item_id,
        "judge_slot": task.judge_slot,
        "model": args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "backend": BACKEND_NAME,
        "worker_rank": rank,
        "attempts": attempts,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


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
