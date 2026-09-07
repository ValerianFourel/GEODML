#!/usr/bin/env python3
"""Run resumable ACL ARR primary or blinded-judge tasks through vLLM."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, contextmanager, ExitStack
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping, Sequence

_TASK_CONTEXT = ContextVar("acl_arr_task_context", default={})


class AuditWriteError(RuntimeError):
    """An audit failure must never cause another inference request."""


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ANALYSIS_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.acl_arr_document_experiment import (  # noqa: E402
    BlindedJudgeTask,
    ExperimentTask,
    FORMAT_VERSION,
    FrozenDocument,
    FrozenDocumentSet,
    JUDGE_FORMAT_VERSION,
    load_plan_from_artifacts,
    render_judge_prompt,
    render_primary_prompt,
    validate_answer_output,
    validate_judge_output,
    validate_rerank_output,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"task file is empty: {path}")
    return rows


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _append(stream, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _verify_primary_task_file(manifest: Mapping[str, Any], tasks_path: Path) -> None:
    artifacts = manifest.get("artifacts")
    tasks = artifacts.get("tasks") if isinstance(artifacts, dict) else None
    if not isinstance(tasks, dict):
        raise ValueError("plan manifest lacks task artifacts")
    digest = _sha256(tasks_path)
    identities = [value for value in tasks.values() if isinstance(value, dict)]
    if not any(identity.get("sha256") == digest for identity in identities):
        raise ValueError("task file does not match any plan task SHA-256")


def _primary_context(manifest_path: Path, tasks_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_primary_task_file(manifest, tasks_path)
    plan = load_plan_from_artifacts(manifest_path)
    rows = _read_jsonl(tasks_path)
    tasks = []
    for row in rows:
        task = ExperimentTask(**row)
        if task.format_version != FORMAT_VERSION or task.plan_id != plan.plan_id:
            raise ValueError("primary task does not match the plan")
        tasks.append(task)
    identities = {(item.model_configuration_id, item.pipeline) for item in tasks}
    if len(identities) != 1:
        raise ValueError("primary task file must contain one model and one pipeline")
    model_configuration_id, pipeline = next(iter(identities))
    model = next(
        (item for item in plan.models if item.configuration_id == model_configuration_id),
        None,
    )
    if model is None:
        raise ValueError("primary task file references an unknown model configuration")
    return plan, tasks, model, pipeline


def _judge_context(manifest_path: Path, tasks_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != JUDGE_FORMAT_VERSION:
        raise ValueError("unsupported judge plan format")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("judge manifest lacks artifacts")
    task_identity = artifacts.get("judge_tasks")
    document_identity = artifacts.get("frozen_document_sets")
    if not isinstance(task_identity, dict) or task_identity.get("sha256") != _sha256(
        tasks_path
    ):
        raise ValueError("judge task file does not match the manifest SHA-256")
    if not isinstance(document_identity, dict):
        raise ValueError("judge manifest lacks frozen document sets")
    documents_path = Path(str(document_identity["path"]))
    if _sha256(documents_path) != document_identity.get("sha256"):
        raise ValueError("judge document sets do not match the manifest SHA-256")
    document_sets = {}
    for row in _read_jsonl(documents_path):
        item = FrozenDocumentSet(
            candidate_set_id=str(row["candidate_set_id"]),
            keyword=str(row["keyword"]),
            search_query=str(row["search_query"]),
            search_engine=str(row["search_engine"]),
            search_snapshot_sha256=str(row["search_snapshot_sha256"]),
            documents=tuple(FrozenDocument(**value) for value in row["documents"]),
        )
        document_sets[item.candidate_set_id] = item
    tasks = []
    for row in _read_jsonl(tasks_path):
        tasks.append(
            BlindedJudgeTask(
                **{
                    **row,
                    "judge_document_ids": tuple(row["judge_document_ids"]),
                    "cited_document_ids": tuple(row["cited_document_ids"]),
                }
            )
        )
    judge_identities = {
        (item.judge_model_id, item.judge_model_revision) for item in tasks
    }
    if len(judge_identities) != 1:
        raise ValueError("judge task file must contain one judge configuration")
    judge_model_id, judge_model_revision = next(iter(judge_identities))
    return manifest, tasks, document_sets, judge_model_id, judge_model_revision


def _rerank_schema(count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ranked_document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["ranked_document_ids"],
        "additionalProperties": False,
    }


def _answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "cited_document_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "cited_document_ids"],
        "additionalProperties": False,
    }


def _judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer_quality": {"type": "integer", "minimum": 1, "maximum": 5},
            "evidence_coverage": {"type": "integer", "minimum": 1, "maximum": 5},
            "citation_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "unsupported_claim_count": {"type": "integer", "minimum": 0},
            "realized_document_ranking": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "use_score": {"type": "integer", "minimum": 0, "maximum": 5},
                    },
                    "required": ["document_id", "use_score"],
                    "additionalProperties": False,
                },
                "minItems": 1,
            },
        },
        "required": [
            "answer_quality",
            "evidence_coverage",
            "citation_correctness",
            "unsupported_claim_count",
            "realized_document_ranking",
        ],
        "additionalProperties": False,
    }


class VllmChatClient:
    """A small asynchronous client for vLLM's OpenAI-compatible server."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        server_model_name: str,
        timeout_seconds: float,
        maximum_attempts: int,
        audit_callback=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.server_model_name = server_model_name
        self.timeout_seconds = timeout_seconds
        self.maximum_attempts = maximum_attempts
        self.session = None
        self.audit_callback = audit_callback

    def _audit(self, event):
        if self.audit_callback is not None:
            try:
                self.audit_callback({"task_context": dict(_TASK_CONTEXT.get()), **event})
            except Exception as exc:
                raise AuditWriteError(str(exc)) from exc

    async def __aenter__(self):
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError(
                "aiohttp is required for real vLLM execution; install analysis/requirements.txt"
            ) from exc
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        try:
            await self.verify_server_identity()
        except BaseException:
            await self.session.close()
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self.session is not None:
            await self.session.close()

    async def verify_server_identity(self) -> None:
        assert self.session is not None
        async with self.session.get(f"{self.base_url}/models") as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"vLLM model identity request failed: {response.status} {body}")
            payload = json.loads(body)
        served = {
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict)
        }
        if self.server_model_name not in served:
            raise RuntimeError(
                f"vLLM serves {sorted(served)}, expected {self.server_model_name!r}"
            )

    async def complete(
        self,
        *,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, Any],
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> tuple[str, Mapping[str, Any]]:
        assert self.session is not None
        payload = {
            "model": self.server_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        last_error: Exception | None = None
        for attempt in range(1, self.maximum_attempts + 1):
            started = time.monotonic()
            event = {"attempt": attempt, "started_at": _now(),
                     "request": {k: payload[k] for k in ("model", "temperature", "max_tokens", "seed", "response_format")},
                     "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            self._audit({**event, "event": "start"})
            status_code = None
            content = None
            usage = {}
            error = None
            body = None
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions", json=payload
                ) as response:
                    body = await response.text()
                    status_code = response.status
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
                    value = json.loads(body)
                content = value["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise RuntimeError("vLLM response content is not text")
                usage = value.get("usage", {})
                usage = usage if isinstance(usage, dict) else {}
            except Exception as exc:
                last_error = exc
                error = f"{type(exc).__name__}: {exc}"
            except BaseException as exc:
                self._audit({**event, "event": "end", "finished_at": _now(),
                             "duration_seconds": time.monotonic() - started,
                             "status_code": status_code, "error": type(exc).__name__})
                raise
            self._audit({**event, "event": "end", "finished_at": _now(),
                         "duration_seconds": time.monotonic() - started,
                         "status_code": status_code, "error": error,
                         "raw_output": content, "usage": usage, "response_body": body})
            if error is None:
                return content, usage
            if attempt < self.maximum_attempts:
                await asyncio.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
        assert last_error is not None
        raise last_error


def _request_sha256(item):
    request = {"prompt": str(item["prompt"]), "schema_name": str(item["schema_name"]),
               "schema": item["schema"], "temperature": float(item["temperature"]),
               "max_tokens": int(item["max_tokens"]), "seed": int(item["seed"])}
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


async def _execute_one(item, *, client, fake):
    started = time.monotonic()
    result = {"base": item["base"], "started_at": _now(), "raw_output": None, "usage": {},
              "request_sha256": _request_sha256(item)}
    token = _TASK_CONTEXT.set(item["base"])
    try:
        if fake:
            raw, usage = str(item["fake_output"]), {}
        else:
            raw, usage = await client.complete(
                prompt=str(item["prompt"]), schema_name=str(item["schema_name"]),
                schema=item["schema"], temperature=float(item["temperature"]),
                max_tokens=int(item["max_tokens"]), seed=int(item["seed"]),
            )
        result.update(raw_output=raw, usage=dict(usage))
        result.update(parsed_output=item["validator"](raw), ok=True)
    except AuditWriteError:
        raise
    except Exception as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        _TASK_CONTEXT.reset(token)
    result.update(finished_at=_now(), duration_seconds=time.monotonic() - started)
    return result


async def _iter_execute(prepared, *, client, maximum_concurrency, fake):
    """Yield completed requests with at most C prepared tasks in flight."""
    if maximum_concurrency <= 0:
        raise ValueError("maximum_concurrency must be positive")
    source = iter(prepared)
    active = set()
    try:
        while True:
            while len(active) < maximum_concurrency:
                item = next(source, None)
                if item is None:
                    break
                active.add(asyncio.create_task(_execute_one(item, client=client, fake=fake)))
            if not active:
                break
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                active.remove(future)
                yield future.result()
    finally:
        for future in active:
            future.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def _execute(
    prepared: Sequence[dict[str, Any]],
    *,
    client: VllmChatClient | None,
    maximum_concurrency: int,
    fake: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(maximum_concurrency)

    async def one(item):
        async with semaphore:
            return await _execute_one(item, client=client, fake=fake)

    active = [asyncio.create_task(one(item)) for item in prepared]
    try:
        return await asyncio.gather(*active)
    finally:
        for future in active:
            future.cancel()
        await asyncio.gather(*active, return_exceptions=True)


def _prepare_primary(
    task: ExperimentTask,
    *,
    prompt,
    assignment,
    document_set,
    model,
) -> dict[str, Any]:
    input_ids = assignment.document_ids(task.condition)
    rendered = render_primary_prompt(task, prompt, assignment, document_set)
    base = {
        "task_id": task.task_id,
        "plan_id": task.plan_id,
        "pipeline": task.pipeline,
        "condition": task.condition,
        "prompt_id": task.prompt_id,
        "prompt_sha256": prompt.question_sha256,
        "assigned_readiness_0_1": prompt.assigned_readiness_0_1,
        "consensus_axis_1_z": prompt.consensus_axis_1_z,
        "axis_1_percentile_0_1": prompt.axis_1_percentile_0_1,
        "assignment_id": assignment.assignment_id,
        "candidate_set_id": document_set.candidate_set_id,
        "input_document_ids": list(input_ids),
        "ablation_target_id": assignment.ablation_target_id,
        "permutation_id": assignment.permutation_id,
        "model_configuration_id": model.configuration_id,
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "fake_backend": False,
    }
    if task.pipeline == "rerank":
        schema = _rerank_schema(task.output_document_count)
        validator = lambda raw: validate_rerank_output(
            raw,
            allowed_document_ids=input_ids,
            output_count=task.output_document_count,
        )
        fake_output = json.dumps(
            {"ranked_document_ids": list(input_ids[: task.output_document_count])}
        )
        max_tokens = model.rerank_max_tokens
    else:
        schema = _answer_schema()
        validator = lambda raw: validate_answer_output(
            raw, allowed_document_ids=input_ids
        )
        first = input_ids[0]
        fake_output = json.dumps(
            {
                "answer": f"Synthetic plumbing output [{first}].",
                "cited_document_ids": [first],
            }
        )
        max_tokens = model.answer_max_tokens
    return {
        "base": base,
        "prompt": rendered,
        "schema_name": f"acl_arr_{task.pipeline}",
        "schema": schema,
        "validator": validator,
        "fake_output": fake_output,
        "temperature": model.temperature,
        "max_tokens": max_tokens,
        "seed": task.decoding_seed,
    }


def _prepare_judge(
    task: BlindedJudgeTask, document_set: FrozenDocumentSet
) -> dict[str, Any]:
    ids = task.judge_document_ids
    return {
        "base": {
            "judge_task_id": task.judge_task_id,
            "blind_case_id": task.blind_case_id,
            "judge_model_id": task.judge_model_id,
            "judge_model_revision": task.judge_model_revision,
            "candidate_set_id": task.candidate_set_id,
            "judge_document_ids": list(ids),
            "fake_backend": False,
        },
        "prompt": render_judge_prompt(task, document_set),
        "schema_name": "acl_arr_realized_use_judge",
        "schema": _judge_schema(),
        "validator": lambda raw: validate_judge_output(
            raw, allowed_document_ids=ids
        ),
        "fake_output": json.dumps(
            {
                "answer_quality": 3,
                "evidence_coverage": 3,
                "citation_correctness": 3,
                "unsupported_claim_count": 0,
                "realized_document_ranking": [
                    {"document_id": ids[0], "use_score": 3}
                ],
            }
        ),
        "temperature": 0.0,
        "max_tokens": 512,
        "seed": int(hashlib.sha256(task.judge_task_id.encode()).hexdigest()[:8], 16),
    }


def _journal_rows(path):
    if not path.exists():
        return
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f"unterminated journal tail; original preserved: {path}")
            if not line.strip():
                raise ValueError(f"blank journal record at {path}:{number}")
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"corrupt journal at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object journal at {path}:{number}")
            yield row


def _validate_resume(output, identity, tasks, prepare, id_field):
    manifest_path = output / "run_manifest.json"
    journals = [output / name for name in ("outcomes.jsonl", "failures.jsonl", "attempts.jsonl")]
    if not manifest_path.exists():
        if any(p.exists() for p in journals):
            raise ValueError("resume journals lack a run manifest; originals preserved")
        return set()
    previous = json.loads(manifest_path.read_text())
    if previous.get("resume_identity") != identity:
        raise ValueError("resume identity mismatch or legacy provenance; use a fresh output directory")
    if (any(previous.get(key) != identity[key] for key in
            ("pipeline", "model_id", "model_revision", "fake_backend", "pilot_only", "maximum_attempts"))
            or previous.get("tasks", {}).get("sha256") != identity["tasks_sha256"]):
        raise ValueError("resume manifest identity fields disagree")
    if not all(p.exists() for p in journals):
        raise ValueError("resume is missing an outcome, failure, or attempt journal")
    indexed = {getattr(task, id_field): task for task in tasks}
    completed = set()
    for path in journals:
        for row in _journal_rows(path):
            base = row.get("task_context", {}) if path.name == "attempts.jsonl" else row
            task_id = base.get(id_field)
            if task_id not in indexed:
                raise ValueError(f"unknown resume task in {path}")
            item = prepare(indexed[task_id])
            expected = {**item["base"], "fake_backend": identity["fake_backend"]}
            if any(base.get(k) != v for k, v in expected.items()):
                raise ValueError(f"resume task identity mismatch in {path}")
            if path.name == "attempts.jsonl":
                attempt = row.get("attempt")
                if row.get("event") not in ("start", "end") or type(attempt) is not int or not 1 <= attempt <= identity["maximum_attempts"]:
                    raise ValueError("invalid HTTP attempt journal record")
            elif path.name == "failures.jsonl":
                if not isinstance(row.get("error"), str):
                    raise ValueError("invalid failure journal record")
            else:
                if task_id in completed:
                    raise ValueError("duplicate resume outcome task ID")
                if identity["pilot_only"] and (row.get("scientific_result") is not False or row.get("eligible_for_analysis") is not False):
                    raise ValueError("pilot resume flags mismatch")
                if row.get("request_sha256") != _request_sha256(item):
                    raise ValueError("resume request fingerprint mismatch")
                raw = row.get("raw_output")
                if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != row.get("raw_output_sha256"):
                    raise ValueError("resume raw output hash mismatch")
                if item["validator"](raw) != row.get("parsed_output"):
                    raise ValueError("resume output validation mismatch")
                completed.add(task_id)
    return completed


@contextmanager
def _writer_lock(output):
    import fcntl
    descriptor = os.open(output, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another writer owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


async def _run(args) -> int:
    with ExitStack() as ownership:
        return await _run_locked(args, ownership)


async def _run_locked(args, ownership) -> int:
    tasks_path = Path(args.tasks).resolve()
    output = Path(args.output_dir).resolve()
    outcomes_path = output / "outcomes.jsonl"
    failures_path = output / "failures.jsonl"
    attempts_path = output / "attempts.jsonl"
    run_manifest_path = output / "run_manifest.json"
    if not args.resume and any(
        path.exists() for path in (outcomes_path, failures_path, attempts_path, run_manifest_path)
    ):
        raise ValueError("output exists; use --resume or a new output directory")

    if args.command == "primary":
        plan, tasks, model, pipeline = _primary_context(
            Path(args.plan_manifest).resolve(), tasks_path
        )
        prompts = {item.prompt_id: item for item in plan.prompts}
        assignments = {item.assignment_id: item for item in plan.assignments}
        document_sets = {item.candidate_set_id: item for item in plan.document_sets}
        id_field = "task_id"
        model_id = model.model_id
        model_revision = model.model_revision
        prepare = lambda task: _prepare_primary(
            task,
            prompt=prompts[task.prompt_id],
            assignment=assignments[task.assignment_id],
            document_set=document_sets[
                assignments[task.assignment_id].candidate_set_id
            ],
            model=model,
        )
        source_manifest = str(Path(args.plan_manifest).resolve())
    else:
        judge_manifest, tasks, document_sets, model_id, model_revision = _judge_context(
            Path(args.judge_manifest).resolve(), tasks_path
        )
        del judge_manifest
        pipeline = "judge"
        id_field = "judge_task_id"
        prepare = lambda task: _prepare_judge(
            task, document_sets[task.candidate_set_id]
        )
        source_manifest = str(Path(args.judge_manifest).resolve())

    task_ids = [getattr(task, id_field) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task file contains duplicate task IDs")
    identity = {"tasks_sha256": _sha256(tasks_path),
                "source_manifest_sha256": _sha256(Path(source_manifest)),
                "pipeline": pipeline, "model_id": model_id, "model_revision": model_revision,
                "fake_backend": args.fake, "pilot_only": args.pilot_only,
                "maximum_attempts": args.max_attempts, "request_timeout": args.request_timeout}
    completed = _validate_resume(output, identity, tasks, prepare, id_field) if args.resume else set()
    pending = [task for task in tasks if getattr(task, id_field) not in completed]
    if args.max_tasks:
        pending = pending[: args.max_tasks]
    server_model_name = args.server_model_name or model_id
    if not args.fake and args.server_model_revision != model_revision:
        raise ValueError(
            "--server-model-revision must equal the immutable task revision"
        )
    run_id = "acl-arr-vllm-run-" + hashlib.sha256(
        (
            f"{_sha256(tasks_path)}:{model_id}:{model_revision}:{pipeline}:"
            f"{os.getenv('SLURM_JOB_ID', 'no-slurm')}"
        ).encode()
    ).hexdigest()[:24]
    manifest = {
        "format_version": "acl-arr-vllm-run-v1",
        "run_id": run_id,
        "invocation_id": uuid.uuid4().hex,
        "resume_identity": identity,
        "scheduler": args.scheduler,
        "pilot_only": args.pilot_only,
        "status": "running",
        "scientific_result": False,
        "eligible_for_analysis": False,
        "started_at": _now(),
        "finished_at": None,
        "source_manifest": source_manifest,
        "tasks": {
            "path": str(tasks_path),
            "sha256": _sha256(tasks_path),
            "total_count": len(tasks),
            "already_completed_count": len(completed),
            "attempted_this_invocation": len(pending),
        },
        "pipeline": pipeline,
        "model_id": model_id,
        "model_revision": model_revision,
        "server_model_name": server_model_name,
        "server_model_revision": args.server_model_revision,
        "vllm_base_url": None if args.fake else args.base_url,
        "fake_backend": args.fake,
        "maximum_concurrency": args.max_concurrency,
        "maximum_attempts": args.max_attempts,
        "slurm": {
            key: os.getenv(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_NUM_NODES",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    ownership.enter_context(_writer_lock(output))
    if args.resume:
        if _validate_resume(output, identity, tasks, prepare, id_field) != completed:
            raise ValueError("resume changed while acquiring writer ownership; retry")
    elif any(path.exists() for path in (outcomes_path, failures_path, attempts_path, run_manifest_path)):
        raise ValueError("output exists; use --resume or a new output directory")
    _atomic_json(run_manifest_path, manifest)

    api_key = args.api_key or os.getenv("VLLM_API_KEY")
    client_context = None
    if not args.fake:
        client_context = VllmChatClient(
            base_url=args.base_url,
            api_key=api_key,
            server_model_name=server_model_name,
            timeout_seconds=args.request_timeout,
            maximum_attempts=args.max_attempts,
        )

    succeeded = 0
    failed = 0
    chunk_size = max(args.max_concurrency, args.max_concurrency * 4)
    with (
        outcomes_path.open("a", encoding="utf-8", buffering=1) as outcomes,
        failures_path.open("a", encoding="utf-8", buffering=1) as failures,
        attempts_path.open("a", encoding="utf-8", buffering=1) as attempts,
    ):
        def persist(stream, row):
            _append(stream, row)
            stream.flush()
            os.fsync(stream.fileno())

        if client_context is not None:
            client_context.audit_callback = lambda event: persist(attempts, {
                **event, "run_id": run_id, "invocation_id": manifest["invocation_id"]})

        async def results(client):
            if args.scheduler == "rolling":
                async with aclosing(_iter_execute((prepare(task) for task in pending),
                        client=client, maximum_concurrency=args.max_concurrency, fake=args.fake)) as stream:
                    async for result in stream:
                        yield result
            else:
                for offset in range(0, len(pending), chunk_size):
                    for result in await _execute([prepare(t) for t in pending[offset:offset + chunk_size]],
                            client=client, maximum_concurrency=args.max_concurrency, fake=args.fake):
                        yield result

        async def process(client):
            nonlocal succeeded, failed
            async with aclosing(results(client)) as stream:
                async for result in stream:
                    base = dict(result["base"])
                    base["fake_backend"] = args.fake
                    base["invocation_id"] = manifest["invocation_id"]
                    if args.pilot_only:
                        base.update(scientific_result=False, eligible_for_analysis=False)
                    if result["ok"]:
                        raw = str(result["raw_output"])
                        persist(
                            outcomes,
                            {
                                **base,
                                "run_id": run_id,
                                "raw_output": raw,
                                "raw_output_sha256": hashlib.sha256(
                                    raw.encode()
                                ).hexdigest(),
                                "parsed_output": result["parsed_output"],
                                "usage": result["usage"],
                                "started_at": result["started_at"],
                                "finished_at": result["finished_at"],
                                "duration_seconds": result["duration_seconds"],
                                "request_sha256": result["request_sha256"],
                            },
                        )
                        succeeded += 1
                        completed.add(base[id_field])
                    else:
                        persist(
                            failures,
                            {
                                **base,
                                "run_id": run_id,
                                "error": result["error"],
                                "raw_output": result["raw_output"],
                                "usage": result["usage"],
                                "duration_seconds": result["duration_seconds"],
                                "request_sha256": result["request_sha256"],
                                "started_at": result["started_at"],
                                "finished_at": result["finished_at"],
                            },
                        )
                        failed += 1
                    manifest.update(completed_count=len(completed), remaining_count=len(tasks) - len(completed))
                    _atomic_json(run_manifest_path, manifest)
                    print(f"PROGRESS={succeeded + failed}/{len(pending)} SUCCEEDED={succeeded} FAILED={failed}", flush=True)

        try:
            if client_context is None or not pending:
                await process(None)
            else:
                async with client_context as client:
                    await process(client)
        except BaseException:
            manifest.update(status="interrupted_or_error", finished_at=_now(),
                            completed_count=len(completed), remaining_count=len(tasks) - len(completed))
            _atomic_json(run_manifest_path, manifest)
            raise

    manifest.update(
        {
            "status": "complete" if len(completed) == len(tasks) else "complete_with_failures" if failed else "checkpointed",
            "scientific_result": not args.fake and not args.pilot_only and len(completed) == len(tasks),
            "eligible_for_analysis": not args.fake and not args.pilot_only and len(completed) == len(tasks),
            "completed_count": len(completed), "remaining_count": len(tasks) - len(completed),
            "finished_at": _now(),
            "outcomes_written_this_invocation": succeeded,
            "failures_written_this_invocation": failed,
            "outcomes_sha256": _sha256(outcomes_path),
            "failures_sha256": _sha256(failures_path),
            "attempts_sha256": _sha256(attempts_path),
        }
    )
    _atomic_json(run_manifest_path, manifest)
    print(f"RUN_ID={run_id}")
    print(f"OUTCOMES={succeeded}")
    print(f"FAILURES={failed}")
    print(f"MANIFEST={run_manifest_path}")
    return 0 if len(completed) == len(tasks) else 2 if failed else 3


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--server-model-name", default=None)
    parser.add_argument(
        "--server-model-revision",
        default=None,
        help="Required for real runs and must equal the task revision SHA.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scheduler", choices=("chunked", "rolling"), default="rolling")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="CPU plumbing only. Fake outputs are never eligible for analysis.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    primary = subparsers.add_parser("primary")
    _add_common(primary)
    primary.add_argument("--plan-manifest", required=True)
    judge = subparsers.add_parser("judge")
    _add_common(judge)
    judge.add_argument("--judge-manifest", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be positive")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0 or args.max_attempts <= 0:
        raise SystemExit("timeout and attempts must be positive")
    if args.max_tasks < 0:
        raise SystemExit("--max-tasks must be non-negative")
    try:
        return asyncio.run(_run(args))
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
