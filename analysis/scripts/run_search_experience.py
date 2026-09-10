#!/usr/bin/env python3
"""Prepare and inspect a separate search-experience pilot against frozen evidence."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.scripts.run_acl_arr_vllm import (
    VllmChatClient, _atomic_json, _append, _iter_execute, _journal_rows,
    _now, _prepare_primary, _request_sha256, _sha256, _validate_resume, _writer_lock,
)
from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    iter_experiment_tasks, load_plan_from_artifacts,
)

BUNDLE_VERSION = "search-experience-bundle-v1"
_CURRENT_ANSWER_SCHEMA = object()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _contract():
    from analysis.interpretability.pipeline import search_experience
    return search_experience


def verify_sources(hashes):
    for path, expected in hashes.items():
        if _sha256(Path(path)) != expected:
            raise ValueError(f"source artifact changed: {path}")


def _artifact_hashes(value):
    hashes = {}
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            hashes[str(Path(value["path"]).resolve())] = value["sha256"]
        for item in value.values():
            hashes.update(_artifact_hashes(item))
    return hashes


def _write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            _append(stream, row)
        stream.flush()
        os.fsync(stream.fileno())


def _cases(plan, prompt_ids, captures, query_contract):
    cases = {}
    for key in prompt_ids:
        prompt = next((p for p in plan.prompts if p.prompt_id == key), None)
        if prompt is None:
            raise ValueError("unknown frozen prompt ID")
        matching = [row for row in captures if row.get("keyword") == prompt.keyword]
        if not matching:
            raise ValueError(f"captured search records missing for {prompt.keyword}")
        cases[key] = _contract().prepare_case(plan, key, capture_rows=matching, query_contract=query_contract)
    return cases


def prepare_bundle(plan_manifest, prompt_ids, captures_jsonl, output, *, query_contract="metadata-keyword-v1", synthetic=False):
    contract = _contract()
    plan_manifest, captures_jsonl = Path(plan_manifest).resolve(), Path(captures_jsonl).resolve()
    plan = load_plan_from_artifacts(plan_manifest)
    captures = list(_journal_rows(captures_jsonl))
    if not captures:
        raise ValueError("real captured search records are required; no captures supplied")
    if not synthetic and any(row.get("synthetic") is True or row.get("fake_backend") is True for row in captures):
        raise ValueError("synthetic captures cannot enter real search-experience preparation")
    if not 1 <= len(prompt_ids) <= 3 or len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("select one to three unique existing prompt IDs for this pilot")
    _cases(plan, prompt_ids, captures, query_contract)
    bundle = {"format_version": BUNDLE_VERSION, "plan_manifest": str(plan_manifest),
              "captures_jsonl": str(captures_jsonl), "prompt_ids": list(prompt_ids),
              "source_plan_id": plan.plan_id, "answer_contract": contract.ANSWER_CONTRACT,
              "query_contract": query_contract,
              "synthetic_inputs": synthetic,
              "judge_contract": contract.JUDGE_CONTRACT,
              "source_hashes": {**_artifact_hashes(json.loads(plan_manifest.read_text())["artifacts"]),
                                **{str(p): _sha256(p) for p in (plan_manifest, captures_jsonl)}},
              "scientific_result": False, "eligible_for_analysis": False}
    bundle["bundle_id"] = "search-bundle-" + _digest(bundle)[:24]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "bundle.json", bundle)
    return bundle


def load_bundle(directory):
    path = Path(directory).resolve() / "bundle.json"
    bundle = json.loads(path.read_text())
    if bundle.get("bundle_id") != "search-bundle-" + _digest({k: v for k, v in bundle.items() if k != "bundle_id"})[:24]:
        raise ValueError("search bundle identity hash mismatch")
    contract = _contract()
    if (bundle.get("format_version") != BUNDLE_VERSION
            or bundle.get("answer_contract") != contract.ANSWER_CONTRACT
            or bundle.get("judge_contract") != contract.JUDGE_CONTRACT):
        raise ValueError("search bundle contract mismatch")
    verify_sources(bundle["source_hashes"])
    plan = load_plan_from_artifacts(bundle["plan_manifest"])
    captures = list(_journal_rows(Path(bundle["captures_jsonl"])))
    cases = _cases(plan, bundle["prompt_ids"], captures, bundle["query_contract"])
    hashes = {**bundle["source_hashes"], str(path): _sha256(path)}
    return bundle, plan, cases, hashes


def primary_items(
    bundle_directory,
    model_configuration_id,
    *,
    answer_max_tokens=None,
    answer_schema_contract=_CURRENT_ANSWER_SCHEMA,
):
    bundle, plan, cases, hashes = load_bundle(bundle_directory)
    contract = _contract()
    if answer_schema_contract is _CURRENT_ANSWER_SCHEMA:
        answer_schema_contract = contract.ANSWER_SCHEMA_CONTRACT
    if answer_schema_contract not in (None, contract.ANSWER_SCHEMA_CONTRACT):
        raise ValueError("unknown answer generation schema contract")
    model = next((m for m in plan.models if m.configuration_id == model_configuration_id), None)
    if model is None:
        raise ValueError("model configuration is not in the frozen plan")
    if answer_max_tokens is not None and answer_max_tokens <= 0:
        raise ValueError("answer max tokens must be positive")
    items = []
    for task in iter_experiment_tasks(plan, model_configuration_id=model_configuration_id):
        if task.prompt_id not in cases:
            continue
        case = cases[task.prompt_id]
        item = _prepare_primary(task, prompt=case.prompt, assignment=case.assignment,
                                document_set=case.document_set, model=model)
        item["base"] = {**item["base"], "legacy_task_id": task.task_id,
                        "bundle_id": bundle["bundle_id"], "protocol": contract.ANSWER_CONTRACT}
        if task.pipeline == "answer":
            allowed = case.assignment.document_ids(task.condition)
            constrained = answer_schema_contract == contract.ANSWER_SCHEMA_CONTRACT
            item.update(prompt=contract.render_answer_prompt(case, task.condition),
                        schema=contract.answer_schema(
                            allowed_document_ids=allowed if constrained else None,
                        ),
                        schema_name=("search_experience_answer_v2" if constrained
                                     else "search_experience_answer_v1"),
                        validator=lambda raw, ids=allowed: contract.validate_answer_output(raw, allowed_document_ids=ids))
            if answer_max_tokens is not None:
                item["max_tokens"] = answer_max_tokens
        item["base"]["task_id"] = "search-task-" + _digest({"bundle_id": bundle["bundle_id"],
            "legacy_task_id": task.task_id, "request_sha256": _request_sha256(item)})[:24]
        items.append(item)
    identity = {"model_id": model.model_id, "model_revision": model.model_revision,
                "model_configuration_id": model.configuration_id,
                "synthetic_inputs": bundle["synthetic_inputs"],
                "pipeline": "search-primary-v1", "source_manifest_sha256": _digest(hashes),
                "fake_backend": False, "pilot_only": True, "maximum_attempts": 3}
    if answer_schema_contract is not None:
        identity["answer_schema_contract"] = answer_schema_contract
    if answer_max_tokens is not None:
        identity["answer_max_tokens_override"] = answer_max_tokens
    return items, identity, hashes


def _run_identity(items, identity):
    return {**identity, "tasks_sha256": _digest([
        {"base": i["base"], "request_sha256": _request_sha256(i)} for i in items])}


def _serving_profile_details(path, identity):
    if path is None:
        return None, None
    from analysis.scripts.search_vllm_stage import load_profile
    resolved = Path(path).expanduser().resolve()
    profile = load_profile(resolved)
    if profile["model"] != {
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
    }:
        raise ValueError("serving profile model identity does not match the run")
    return {
        "path": str(resolved),
        "sha256": profile["profile_sha256"],
    }, profile


def _serving_runtime_details(profile):
    if profile is None:
        return None
    from analysis.scripts.search_vllm_stage import load_runtime_binding
    path = os.environ.get("GEODML_SERVING_RUNTIME_RECORD")
    require_approval = profile["serving"]["data_parallel_size"] > 1
    if path is None:
        if require_approval:
            raise ValueError("DP execution requires a serving runtime record")
        return None
    return load_runtime_binding(
        path,
        expected_profile_sha256=profile["profile_sha256"],
        require_approval=require_approval,
    )


def _verify_manifest_serving_runtime(manifest, serving_profile):
    saved = manifest.get("serving_runtime")
    history = manifest.get("serving_invocations")
    if saved is None and history is None:
        return
    if serving_profile is None:
        raise ValueError("run manifest runtime provenance lacks a serving profile")
    from analysis.scripts.search_vllm_stage import load_profile, load_runtime_binding
    profile = load_profile(serving_profile["path"])

    def verify_runtime(runtime):
        if runtime is None:
            if profile["serving"]["data_parallel_size"] > 1:
                raise ValueError("DP run manifest lacks serving runtime provenance")
            return
        if not isinstance(runtime, dict) or "path" not in runtime:
            raise ValueError("run manifest serving runtime provenance is malformed")
        actual = load_runtime_binding(
            runtime["path"],
            expected_profile_sha256=profile["profile_sha256"],
            require_approval=profile["serving"]["data_parallel_size"] > 1,
        )
        if actual != runtime:
            raise ValueError("run manifest serving runtime provenance does not match")

    verify_runtime(saved)
    if history is None:
        return
    if not isinstance(history, list) or not history:
        raise ValueError("run manifest serving invocation history is malformed")
    expected_keys = {
        "run_id",
        "serving_profile",
        "serving_runtime",
        "started_at",
        "finished_at",
    }
    run_ids = set()
    for invocation in history:
        if not isinstance(invocation, dict) or set(invocation) != expected_keys:
            raise ValueError("run manifest serving invocation history is malformed")
        run_id = invocation["run_id"]
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise ValueError("run manifest serving invocation run IDs are invalid")
        run_ids.add(run_id)
        if invocation["serving_profile"] != serving_profile:
            raise ValueError("run manifest serving invocation profile changed")
        if not isinstance(invocation["started_at"], str) or not invocation["started_at"]:
            raise ValueError("run manifest serving invocation start time is invalid")
        if invocation["finished_at"] is not None and (
            not isinstance(invocation["finished_at"], str)
            or not invocation["finished_at"]
        ):
            raise ValueError("run manifest serving invocation finish time is invalid")
        verify_runtime(invocation["serving_runtime"])
    if history[-1]["run_id"] != manifest.get("run_id"):
        raise ValueError("run manifest current serving invocation is inconsistent")
    if history[-1]["serving_runtime"] != saved:
        raise ValueError("run manifest current serving runtime is inconsistent")


def _previous_serving_invocations(manifest):
    history = manifest.get("serving_invocations")
    if history is not None:
        return [dict(invocation) for invocation in history]
    if manifest.get("serving_profile") is None:
        return []
    return [{
        "run_id": manifest["run_id"],
        "serving_profile": manifest["serving_profile"],
        "serving_runtime": manifest.get("serving_runtime"),
        "started_at": manifest["started_at"],
        "finished_at": manifest.get("finished_at"),
    }]


def _verify_serving_journal_run_ids(output, manifest):
    history = manifest.get("serving_invocations")
    if history is None:
        return
    run_ids = {invocation["run_id"] for invocation in history}
    for name in ("outcomes", "failures", "attempts"):
        path = output / f"{name}.jsonl"
        if not path.exists():
            continue
        for row in _journal_rows(path):
            if row.get("run_id") not in run_ids:
                raise ValueError(
                    f"{name} journal run ID lacks serving invocation provenance"
                )


def _verify_manifest_serving_profile(
    manifest,
    identity,
    expected,
    completed,
    total,
    require_explicit_profile,
):
    saved = manifest.get("serving_profile")
    if saved is None:
        if expected is not None and len(completed) != total:
            raise ValueError(
                "a historical partial run cannot acquire new serving profile provenance"
            )
        return None
    if not isinstance(saved, dict) or set(saved) != {"path", "sha256"}:
        raise ValueError("run manifest serving profile provenance is malformed")
    actual, _ = _serving_profile_details(saved.get("path"), identity)
    if actual != saved:
        raise ValueError("run manifest serving profile hash does not match its sidecar")
    if require_explicit_profile and expected is None and len(completed) != total:
        raise ValueError("a bound partial run requires an explicit serving profile")
    if expected is not None and saved != expected:
        raise ValueError("resume serving profile does not exactly match the run manifest")
    return saved


def _resume(
    output,
    items,
    identity,
    *,
    serving_profile=None,
    require_explicit_profile=False,
):
    by_id = {i["base"]["task_id"]: i for i in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate prepared task IDs")
    tasks = [SimpleNamespace(task_id=key) for key in by_id]
    completed = _validate_resume(
        output,
        identity,
        tasks,
        lambda task: by_id[task.task_id],
        "task_id",
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        saved_profile = _verify_manifest_serving_profile(
            manifest,
            identity,
            serving_profile,
            completed,
            len(items),
            require_explicit_profile,
        )
        _verify_manifest_serving_runtime(manifest, saved_profile)
        _verify_serving_journal_run_ids(output, manifest)
    return completed


def preflight_run(
    items,
    output,
    identity,
    source_hashes,
    *,
    resume=False,
    serving_profile=None,
    require_explicit_profile=False,
):
    verify_sources(source_hashes)
    if len({item["base"]["task_id"] for item in items}) != len(items):
        raise ValueError("duplicate prepared task IDs")
    output = Path(output).resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    run_identity = _run_identity(items, identity)
    binding, _ = _serving_profile_details(serving_profile, run_identity)
    return (
        _resume(
            output,
            items,
            run_identity,
            serving_profile=binding,
            require_explicit_profile=require_explicit_profile,
        )
        if resume
        else set()
    )


async def run_prepared(items, output, *, client, identity, source_hashes,
                       max_concurrency=8, max_tasks=0, resume=False,
                       serving_profile=None):
    if max_concurrency < 1 or max_tasks < 0:
        raise ValueError("concurrency must be positive and max_tasks nonnegative")
    identity = _run_identity(items, identity)
    binding, profile = _serving_profile_details(serving_profile, identity)
    runtime_binding = _serving_runtime_details(profile)
    completed = preflight_run(
        items,
        output,
        identity,
        source_hashes,
        resume=resume,
        serving_profile=serving_profile,
        require_explicit_profile=True,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output):
        if resume:
            if completed != _resume(
                output,
                items,
                identity,
                serving_profile=binding,
            ):
                raise ValueError("resume changed while acquiring output ownership")
        elif any(output.iterdir()):
            raise FileExistsError(output)
        if resume and len(completed) == len(items):
            return 0
        current_binding, _ = _serving_profile_details(serving_profile, identity)
        if current_binding != binding:
            raise ValueError("serving profile changed while acquiring output ownership")
        if _serving_runtime_details(profile) != runtime_binding:
            raise ValueError("serving runtime changed while acquiring output ownership")
        previous_manifest = None
        manifest_path = output / "run_manifest.json"
        if resume and manifest_path.exists():
            previous_manifest = json.loads(manifest_path.read_text())
        pending = [i for i in items if i["base"]["task_id"] not in completed]
        if max_tasks:
            pending = pending[:max_tasks]
        run_id = "search-run-" + uuid.uuid4().hex
        started_at = _now()
        manifest = {**identity, "format_version": "search-experience-run-v1",
                    "resume_identity": identity, "source_artifacts_sha256": source_hashes,
                    "run_id": run_id,
                    "status": "running", "started_at": started_at, "finished_at": None,
                    "scientific_result": False, "eligible_for_analysis": False,
                    "task_count": len(items), "completed_count": len(completed),
                    "remaining_count": len(items) - len(completed),
                    "tasks": {"sha256": identity["tasks_sha256"]},
                    "maximum_concurrency": max_concurrency, "attempted_this_invocation": 0}
        if binding is not None:
            manifest["serving_profile"] = binding
        if runtime_binding is not None:
            manifest["serving_runtime"] = runtime_binding
        if binding is not None:
            history = (
                _previous_serving_invocations(previous_manifest)
                if previous_manifest is not None
                else []
            )
            history.append({
                "run_id": run_id,
                "serving_profile": binding,
                "serving_runtime": runtime_binding,
                "started_at": started_at,
                "finished_at": None,
            })
            manifest["serving_invocations"] = history
        _atomic_json(manifest_path, manifest)
        failures = 0
        with (output / "outcomes.jsonl").open("a", encoding="utf-8") as outcomes, \
                (output / "failures.jsonl").open("a", encoding="utf-8") as failed, \
                (output / "attempts.jsonl").open("a", encoding="utf-8") as attempts:
            def persist(stream, row):
                _append(stream, row)
                stream.flush()
                os.fsync(stream.fileno())
            old_audit = getattr(client, "audit_callback", None)
            if client is not None:
                client.audit_callback = lambda event: persist(attempts, {**event, "run_id": manifest["run_id"]})
            try:
                async with aclosing(_iter_execute(pending, client=client,
                        maximum_concurrency=max_concurrency, fake=False)) as results:
                    async for result in results:
                        row = {**result["base"], **{k: v for k, v in result.items() if k not in ("base", "ok")},
                               "run_id": manifest["run_id"], "scientific_result": False,
                               "eligible_for_analysis": False}
                        if isinstance(row.get("raw_output"), str):
                            row["raw_output_sha256"] = hashlib.sha256(row["raw_output"].encode()).hexdigest()
                        persist(outcomes if result["ok"] else failed, row)
                        if result["ok"]:
                            completed.add(row["task_id"])
                        else:
                            failures += 1
                        manifest.update(completed_count=len(completed), remaining_count=len(items) - len(completed),
                            attempted_this_invocation=manifest["attempted_this_invocation"] + 1)
                        _atomic_json(manifest_path, manifest)
                verify_sources(source_hashes)
                if binding is not None:
                    current_binding, _ = _serving_profile_details(
                        binding["path"], identity
                    )
                    if current_binding != binding:
                        raise ValueError("serving profile changed during the run")
                if _serving_runtime_details(profile) != runtime_binding:
                    raise ValueError("serving runtime changed during the run")
                manifest["status"] = "complete" if len(completed) == len(items) else "complete_with_failures" if failures else "checkpointed"
            except BaseException as exc:
                manifest.update(status="interrupted_or_error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if client is not None:
                    client.audit_callback = old_audit
                finished_at = _now()
                manifest.update(finished_at=finished_at, failures_this_invocation=failures,
                    **{name + "_sha256": _sha256(output / (name + ".jsonl")) for name in ("outcomes", "failures", "attempts")})
                if "serving_invocations" in manifest:
                    manifest["serving_invocations"][-1]["finished_at"] = finished_at
                _atomic_json(manifest_path, manifest)
    return 0 if len(completed) == len(items) else 2 if failures else 3


def _validated_primary(bundle_directory, primary_output):
    output = Path(primary_output).resolve()
    manifest = json.loads((output / "run_manifest.json").read_text())
    verify_sources(manifest["source_artifacts_sha256"])
    for name in ("outcomes", "failures", "attempts"):
        expected = manifest.get(name + "_sha256")
        if expected is not None and expected != _sha256(output / (name + ".jsonl")):
            raise ValueError(f"primary {name} journal hash mismatch")
    _, plan, _, _ = load_bundle(bundle_directory)
    model = next((m for m in plan.models if m.configuration_id == manifest.get("model_configuration_id")
                  and m.model_id == manifest.get("model_id")
                  and m.model_revision == manifest.get("model_revision")), None)
    if model is None:
        raise ValueError("primary outputs do not match a frozen model")
    items, identity, hashes = primary_items(
        bundle_directory,
        model.configuration_id,
        answer_max_tokens=manifest.get("answer_max_tokens_override"),
        answer_schema_contract=manifest.get("answer_schema_contract"),
    )
    if manifest.get("fake_backend") is True:
        identity["fake_backend"] = True
        for item in items:
            item["base"]["fake_backend"] = True
    _resume(output, items, _run_identity(items, identity))
    rows = list(_journal_rows(output / "outcomes.jsonl"))
    hashes.update({str(output / name): _sha256(output / name) for name in
                   ("run_manifest.json", "outcomes.jsonl", "failures.jsonl", "attempts.jsonl")})
    return items, rows, manifest, hashes


def validate_judge_model(generator_model, judge_model):
    if generator_model == judge_model:
        raise ValueError("an independent judge model is required; self-judgment cannot be the sole score")


def judge_items(bundle_directory, primary_output, judge_model_id, judge_model_revision, *,
                judge_contract="search-experience-judge-v1", judge_max_tokens=None):
    contract = _contract()
    if judge_contract not in (contract.JUDGE_CONTRACT, contract.QUOTE_JUDGE_CONTRACT):
        raise ValueError("unknown judge contract")
    quotes_only = judge_contract == contract.QUOTE_JUDGE_CONTRACT
    if re.fullmatch(r"[0-9a-f]{40}", judge_model_revision) is None:
        raise ValueError("judge revision must be an immutable 40-character SHA")
    if judge_max_tokens is not None and judge_max_tokens <= 0:
        raise ValueError("judge max tokens must be positive")
    effective_max_tokens = 2048 if judge_max_tokens is None else judge_max_tokens
    primary, rows, manifest, hashes = _validated_primary(bundle_directory, primary_output)
    expected_answers = {item["base"]["task_id"] for item in primary
                        if item["base"]["pipeline"] == "answer"}
    completed_answers = {row["task_id"] for row in rows if row["pipeline"] == "answer"}
    if completed_answers != expected_answers:
        raise ValueError("complete answer coverage is required before compiling judge tasks")
    if manifest.get("status") == "running":
        raise ValueError("primary writer must finish before compiling immutable judge inputs")
    validate_judge_model(manifest["model_id"], judge_model_id)
    bundle, plan, cases, _ = load_bundle(bundle_directory)
    contract = _contract()
    items = []
    for row in rows:
        if row["pipeline"] != "answer":
            continue
        judge_input = contract.prepare_judge_input(cases[row["prompt_id"]], row["condition"], row["parsed_output"],
            source_task_id=row["task_id"], master_seed=plan.master_seed)
        task_identity = {"source": row["task_id"], "input": judge_input,
            "model": judge_model_id, "revision": judge_model_revision, "contract": judge_contract}
        if judge_max_tokens is not None:
            task_identity["max_tokens"] = judge_max_tokens
        task_id = "search-judge-" + _digest(task_identity)[:24]
        items.append({"base": {"task_id": task_id, "pipeline": "judge", "source_task_id": row["task_id"],
            "bundle_id": bundle["bundle_id"], "judge_model_id": judge_model_id,
            "judge_model_revision": judge_model_revision, "fake_backend": manifest["fake_backend"]},
            "prompt": (contract.render_quote_judge_prompt(judge_input) if quotes_only
                       else contract.render_judge_prompt(judge_input)),
            "schema": contract.judge_quote_schema(judge_input) if quotes_only else contract.judge_schema(),
            "schema_name": "search_experience_judge_quotes_v2" if quotes_only else "search_experience_judge_v1",
            "temperature": 0.0, "max_tokens": effective_max_tokens,
            "seed": int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16),
            "validator": lambda raw, value=judge_input: (
                contract.validate_quote_judge_output(raw, judge_input=value) if quotes_only
                else contract.validate_judge_output(raw, judge_input=value))})
    if not items:
        raise ValueError("no validated answers available for judgment")
    identity = {"model_id": judge_model_id, "model_revision": judge_model_revision,
        "pipeline": "search-judge-v1", "source_manifest_sha256": _digest(hashes),
        "fake_backend": manifest["fake_backend"], "pilot_only": True, "maximum_attempts": 3}
    if quotes_only:
        identity.update(pipeline="search-judge-quotes-v2", judge_contract=judge_contract)
    if judge_max_tokens is not None:
        identity["judge_max_tokens_override"] = judge_max_tokens
    return items, identity, hashes


def inspect_bundle(bundle_directory, primary_output, judge_output, output):
    items, rows, manifest, _ = _validated_primary(bundle_directory, primary_output)
    bundle, plan, cases, _ = load_bundle(bundle_directory)
    judgments = {}
    if judge_output:
        judge_path = Path(judge_output)
        judge_manifest = json.loads((judge_path / "run_manifest.json").read_text())
        prepared, identity, _ = judge_items(bundle_directory, primary_output,
            judge_manifest["model_id"], judge_manifest["model_revision"],
            judge_contract=judge_manifest.get("judge_contract", _contract().JUDGE_CONTRACT),
            judge_max_tokens=judge_manifest.get("judge_max_tokens_override"))
        _resume(judge_path, prepared, _run_identity(prepared, identity))
        judgments = {r["source_task_id"]: r["parsed_output"] for r in _journal_rows(judge_path / "outcomes.jsonl")}
    indexed = {(r["prompt_id"], r["condition"], r["pipeline"]): r for r in rows}
    records = []
    for prompt_id, case in cases.items():
        for condition in ("natural", "ablated", "shuffled"):
            ranking = indexed.get((prompt_id, condition, "rerank"))
            answer = indexed.get((prompt_id, condition, "answer"))
            ids = case.assignment.document_ids(condition)
            trace = case.query_intent
            records.append({"prompt_id": prompt_id, "condition": condition,
                "original_request": case.prompt.question, "query_trace": asdict(trace) if is_dataclass(trace) else trace,
                "captured_search_records": list(case.capture_rows), "evidence_representation": "frozen_page_text",
                "documents": [asdict(d) for key in ids for d in case.document_set.documents if d.document_id == key],
                "ranking": ranking["parsed_output"] if ranking else None,
                "answer": answer["parsed_output"] if answer else None,
                "answer_display": _contract().render_answer_display(answer["parsed_output"]) if answer else None,
                "judgment": judgments.get(answer["task_id"]) if answer else None,
                "complete": bool(ranking and answer and answer["task_id"] in judgments)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "report.json", {"bundle_id": bundle["bundle_id"],
        "model_id": manifest["model_id"], "scientific_result": False, "eligible_for_analysis": False,
        "synthetic_inputs": bundle["synthetic_inputs"], "fake_backend": manifest["fake_backend"],
        "complete": len(rows) == len(items) and all(r["complete"] for r in records), "cases": records})
    return records


def export_human(bundle_directory, primary_output, output, *, sample_size=20, seed=20260907):
    _, rows, manifest, hashes = _validated_primary(bundle_directory, primary_output)
    bundle, _, cases, _ = load_bundle(bundle_directory)
    contract = _contract()
    answers = sorted((r for r in rows if r["pipeline"] == "answer"),
                     key=lambda row: _digest([seed, row["task_id"]]))
    if not answers:
        raise ValueError("no validated answers available for human packets")
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    packets, mappings = [], []
    for row in answers[:sample_size]:
        packet_id = "human-case-" + _digest([seed, row["task_id"]])[:24]
        value = contract.prepare_judge_input(cases[row["prompt_id"]], row["condition"], row["parsed_output"],
            source_task_id=row["task_id"], master_seed=seed)
        packets.append({"packet_id": packet_id, "input": value, "rubric": contract.render_judge_prompt(value),
                        "response_schema": contract.judge_schema()})
        mappings.append({"packet_id": packet_id, "source_task_id": row["task_id"],
                         "generator_model_id": row["model_id"], "condition": row["condition"]})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "human_packets.jsonl", packets)
    _write_jsonl(output / "private_mapping.jsonl", mappings)
    _atomic_json(output / "calibration_manifest.json", {"format_version": "search-human-calibration-v1",
        "seed": seed, "selected_count": len(packets), "available_answer_count": len(answers),
        "source_artifacts_sha256": hashes, "judge_contract": contract.JUDGE_CONTRACT,
        "packet_sha256": _sha256(output / "human_packets.jsonl"),
        "synthetic_inputs": bundle["synthetic_inputs"], "fake_backend": manifest["fake_backend"],
        "scientific_result": False, "eligible_for_analysis": False})
    return packets


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--plan-manifest", required=True)
    prepare.add_argument("--prompt-id", action="append", required=True)
    prepare.add_argument("--captures-jsonl", required=True)
    prepare.add_argument("--query-contract", choices=("metadata-keyword-v1", "full-request-v1"), default="metadata-keyword-v1")
    for name in ("run-primary", "run-judge", "inspect", "export-human"):
        sub = commands.add_parser(name)
        sub.add_argument("--bundle-dir", required=True)
        if name != "run-primary":
            sub.add_argument("--primary-output", required=True)
        if name.startswith("run-"):
            sub.add_argument("--base-url", required=True)
            sub.add_argument("--serving-profile")
            sub.add_argument("--max-concurrency", type=int, default=8)
            sub.add_argument("--max-tasks", type=int, default=0)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--preflight-only", action="store_true",
                             help="Validate inputs and saved work without writes or HTTP; exit 0 complete, 3 pending")
        if name == "run-primary":
            sub.add_argument("--model-configuration-id", required=True)
            sub.add_argument("--server-model-revision", required=True)
            sub.add_argument("--answer-max-tokens", type=int)
        elif name == "run-judge":
            sub.add_argument("--judge-contract", choices=(
                _contract().JUDGE_CONTRACT, _contract().QUOTE_JUDGE_CONTRACT),
                default=_contract().JUDGE_CONTRACT)
            sub.add_argument("--judge-model-id", required=True)
            sub.add_argument("--judge-model-revision", required=True)
            sub.add_argument("--judge-max-tokens", type=int)
        elif name == "inspect":
            sub.add_argument("--judge-output")
        else:
            sub.add_argument("--sample-size", type=int, default=20)
            sub.add_argument("--seed", type=int, default=20260907)
        sub.add_argument("--output-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        prepare_bundle(args.plan_manifest, args.prompt_id, args.captures_jsonl, args.output_dir,
                       query_contract=args.query_contract)
    elif args.command == "inspect":
        inspect_bundle(args.bundle_dir, args.primary_output, args.judge_output, args.output_dir)
    elif args.command == "export-human":
        export_human(args.bundle_dir, args.primary_output, args.output_dir, sample_size=args.sample_size, seed=args.seed)
    else:
        if args.command == "run-primary":
            items, identity, hashes = primary_items(
                args.bundle_dir,
                args.model_configuration_id,
                answer_max_tokens=args.answer_max_tokens,
            )
            if args.server_model_revision != identity["model_revision"]:
                raise ValueError("server revision differs from the frozen model")
        else:
            items, identity, hashes = judge_items(args.bundle_dir, args.primary_output,
                args.judge_model_id, args.judge_model_revision, judge_contract=args.judge_contract,
                judge_max_tokens=args.judge_max_tokens)
        _, serving_profile = _serving_profile_details(
            args.serving_profile,
            identity,
        )
        if serving_profile is not None:
            if args.base_url != serving_profile["serving"]["public_base_url"]:
                raise ValueError("base URL does not match the serving profile")
            if args.max_concurrency != serving_profile["serving"]["request_concurrency"]:
                raise ValueError("request concurrency does not match the serving profile")
        completed = preflight_run(
            items,
            args.output_dir,
            identity,
            hashes,
            resume=args.resume,
            serving_profile=args.serving_profile,
        )
        if args.preflight_only:
            print(json.dumps({"status": "complete" if len(completed) == len(items) else "pending",
                              "task_count": len(items), "completed_count": len(completed),
                              "remaining_count": len(items) - len(completed)}, sort_keys=True))
            return 0 if len(completed) == len(items) else 3
        bundle, _, _, _ = load_bundle(args.bundle_dir)
        if bundle["synthetic_inputs"]:
            raise ValueError("synthetic bundles are for injected-client tests, not real endpoint execution")
        async def execute():
            async with VllmChatClient(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY"),
                    server_model_name=identity["model_id"], timeout_seconds=600,
                    maximum_attempts=identity["maximum_attempts"]) as client:
                return await run_prepared(items, args.output_dir, client=client, identity=identity,
                    source_hashes=hashes, max_concurrency=args.max_concurrency,
                    max_tasks=args.max_tasks, resume=args.resume,
                    serving_profile=args.serving_profile)
        return asyncio.run(execute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
