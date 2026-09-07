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


def primary_items(bundle_directory, model_configuration_id):
    bundle, plan, cases, hashes = load_bundle(bundle_directory)
    contract = _contract()
    model = next((m for m in plan.models if m.configuration_id == model_configuration_id), None)
    if model is None:
        raise ValueError("model configuration is not in the frozen plan")
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
            item.update(prompt=contract.render_answer_prompt(case, task.condition),
                        schema=contract.answer_schema(), schema_name="search_experience_answer_v1",
                        validator=lambda raw, ids=allowed: contract.validate_answer_output(raw, allowed_document_ids=ids))
        item["base"]["task_id"] = "search-task-" + _digest({"bundle_id": bundle["bundle_id"],
            "legacy_task_id": task.task_id, "request_sha256": _request_sha256(item)})[:24]
        items.append(item)
    identity = {"model_id": model.model_id, "model_revision": model.model_revision,
                "model_configuration_id": model.configuration_id,
                "synthetic_inputs": bundle["synthetic_inputs"],
                "pipeline": "search-primary-v1", "source_manifest_sha256": _digest(hashes),
                "fake_backend": False, "pilot_only": True, "maximum_attempts": 3}
    return items, identity, hashes


def _run_identity(items, identity):
    return {**identity, "tasks_sha256": _digest([
        {"base": i["base"], "request_sha256": _request_sha256(i)} for i in items])}


def _resume(output, items, identity):
    by_id = {i["base"]["task_id"]: i for i in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate prepared task IDs")
    tasks = [SimpleNamespace(task_id=key) for key in by_id]
    return _validate_resume(output, identity, tasks, lambda task: by_id[task.task_id], "task_id")


def preflight_run(items, output, identity, source_hashes, *, resume=False):
    verify_sources(source_hashes)
    if len({item["base"]["task_id"] for item in items}) != len(items):
        raise ValueError("duplicate prepared task IDs")
    output = Path(output).resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    return _resume(output, items, _run_identity(items, identity)) if resume else set()


async def run_prepared(items, output, *, client, identity, source_hashes,
                       max_concurrency=8, max_tasks=0, resume=False):
    """Run a pilot shard with the generic executor and strict journal validation."""
    if max_concurrency < 1 or max_tasks < 0:
        raise ValueError("concurrency must be positive and max_tasks nonnegative")
    completed = preflight_run(items, output, identity, source_hashes, resume=resume)
    identity = _run_identity(items, identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output):
        if resume:
            if completed != _resume(output, items, identity):
                raise ValueError("resume changed while acquiring output ownership")
        elif any(output.iterdir()):
            raise FileExistsError(output)
        if resume and len(completed) == len(items):
            return 0
        pending = [i for i in items if i["base"]["task_id"] not in completed]
        if max_tasks:
            pending = pending[:max_tasks]
        manifest = {**identity, "format_version": "search-experience-run-v1",
                    "resume_identity": identity, "source_artifacts_sha256": source_hashes,
                    "run_id": "search-run-" + uuid.uuid4().hex,
                    "status": "running", "started_at": _now(), "finished_at": None,
                    "scientific_result": False, "eligible_for_analysis": False,
                    "task_count": len(items), "completed_count": len(completed),
                    "remaining_count": len(items) - len(completed),
                    "tasks": {"sha256": identity["tasks_sha256"]},
                    "maximum_concurrency": max_concurrency, "attempted_this_invocation": 0}
        manifest_path = output / "run_manifest.json"
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
                manifest["status"] = "complete" if len(completed) == len(items) else "complete_with_failures" if failures else "checkpointed"
            except BaseException as exc:
                manifest.update(status="interrupted_or_error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                if client is not None:
                    client.audit_callback = old_audit
                manifest.update(finished_at=_now(), failures_this_invocation=failures,
                    **{name + "_sha256": _sha256(output / (name + ".jsonl")) for name in ("outcomes", "failures", "attempts")})
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
    items, identity, hashes = primary_items(bundle_directory, model.configuration_id)
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


def judge_items(bundle_directory, primary_output, judge_model_id, judge_model_revision):
    if re.fullmatch(r"[0-9a-f]{40}", judge_model_revision) is None:
        raise ValueError("judge revision must be an immutable 40-character SHA")
    primary, rows, manifest, hashes = _validated_primary(bundle_directory, primary_output)
    if len(rows) != len(primary):
        raise ValueError("complete primary coverage is required before compiling judge tasks")
    validate_judge_model(manifest["model_id"], judge_model_id)
    bundle, plan, cases, _ = load_bundle(bundle_directory)
    contract = _contract()
    items = []
    for row in rows:
        if row["pipeline"] != "answer":
            continue
        judge_input = contract.prepare_judge_input(cases[row["prompt_id"]], row["condition"], row["parsed_output"],
            source_task_id=row["task_id"], master_seed=plan.master_seed)
        task_id = "search-judge-" + _digest({"source": row["task_id"], "input": judge_input,
            "model": judge_model_id, "revision": judge_model_revision, "contract": contract.JUDGE_CONTRACT})[:24]
        items.append({"base": {"task_id": task_id, "pipeline": "judge", "source_task_id": row["task_id"],
            "bundle_id": bundle["bundle_id"], "judge_model_id": judge_model_id,
            "judge_model_revision": judge_model_revision, "fake_backend": manifest["fake_backend"]},
            "prompt": contract.render_judge_prompt(judge_input), "schema": contract.judge_schema(),
            "schema_name": "search_experience_judge_v1", "temperature": 0.0, "max_tokens": 2048,
            "seed": int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16),
            "validator": lambda raw, value=judge_input: contract.validate_judge_output(raw, judge_input=value)})
    if not items:
        raise ValueError("no validated answers available for judgment")
    return items, {"model_id": judge_model_id, "model_revision": judge_model_revision,
        "pipeline": "search-judge-v1", "source_manifest_sha256": _digest(hashes),
        "fake_backend": manifest["fake_backend"], "pilot_only": True, "maximum_attempts": 3}, hashes


def inspect_bundle(bundle_directory, primary_output, judge_output, output):
    items, rows, manifest, _ = _validated_primary(bundle_directory, primary_output)
    bundle, plan, cases, _ = load_bundle(bundle_directory)
    judgments = {}
    if judge_output:
        judge_path = Path(judge_output)
        judge_manifest = json.loads((judge_path / "run_manifest.json").read_text())
        prepared, identity, _ = judge_items(bundle_directory, primary_output,
            judge_manifest["model_id"], judge_manifest["model_revision"])
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
            sub.add_argument("--max-concurrency", type=int, default=8)
            sub.add_argument("--max-tasks", type=int, default=0)
            sub.add_argument("--resume", action="store_true")
        if name == "run-primary":
            sub.add_argument("--model-configuration-id", required=True)
            sub.add_argument("--server-model-revision", required=True)
        elif name == "run-judge":
            sub.add_argument("--judge-model-id", required=True)
            sub.add_argument("--judge-model-revision", required=True)
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
            items, identity, hashes = primary_items(args.bundle_dir, args.model_configuration_id)
            if args.server_model_revision != identity["model_revision"]:
                raise ValueError("server revision differs from the frozen model")
        else:
            items, identity, hashes = judge_items(args.bundle_dir, args.primary_output,
                args.judge_model_id, args.judge_model_revision)
        preflight_run(items, args.output_dir, identity, hashes, resume=args.resume)
        bundle, _, _, _ = load_bundle(args.bundle_dir)
        if bundle["synthetic_inputs"]:
            raise ValueError("synthetic bundles are for injected-client tests, not real endpoint execution")
        async def execute():
            async with VllmChatClient(base_url=args.base_url, api_key=os.getenv("VLLM_API_KEY"),
                    server_model_name=identity["model_id"], timeout_seconds=600,
                    maximum_attempts=identity["maximum_attempts"]) as client:
                return await run_prepared(items, args.output_dir, client=client, identity=identity,
                    source_hashes=hashes, max_concurrency=args.max_concurrency,
                    max_tasks=args.max_tasks, resume=args.resume)
        return asyncio.run(execute())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
