"""Read existing frozen plans and growing journals without changing their files.

The adapters deliberately keep direct reranking, agentic generation, and judge
protocols separate. A candidate pool is identified by URLs *and evidence*, not
by an engine name or a shared set of URL strings alone.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from analysis.interpretability.pipeline.agentic_judging import (
    TRANSCRIPT_FORMAT_VERSION,
    AgenticJudgeTask,
    _task_and_mapping,
    validate_agentic_judgment,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def _pool(rows):
    evidence = [{"url": r["url"], "title": r["title"], "text": r.get("text", r.get("snippet", ""))}
                for r in rows]
    if len({r["url"] for r in evidence}) != len(evidence):
        raise ValueError("candidate pool contains duplicate URLs")
    return "evidence-pool:" + _hash(_canonical(sorted(evidence, key=lambda r: r["url"])))


def _axes(row):
    values = dict(row.get("readiness_coordinates", {}))
    values.update({k: v for k, v in row.items() if "normalized_axis_" in k and v is not None})
    if "B" in row:
        values["B"] = row["B"]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
           for v in values.values()):
        raise ValueError("axis coordinates must be finite numbers")
    return values


class _Reader:
    def __init__(self, base, *, cache_limit_bytes=64 * 1024 * 1024):
        if type(cache_limit_bytes) is not int or cache_limit_bytes < 0:
            raise ValueError("cache byte limit must be a non-negative integer")
        self.base = base
        self.sources = {}
        self.issues = []
        self.cache = OrderedDict()
        self.cache_limit_bytes = cache_limit_bytes
        self.cache_size_bytes = 0

    def path(self, value, base=None):
        path = Path(value)
        return path.resolve() if path.is_absolute() else ((base or self.base) / path).resolve()

    def bytes(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        raw = path.read_bytes()
        source = {"path": str(path), "sha256": _hash(raw), "size_bytes": len(raw)}
        previous = self.sources.get(str(path))
        if previous is not None and source != previous:
            raise ValueError(f"source changed during report read: {path}")
        self.sources[str(path)] = source
        if len(raw) <= self.cache_limit_bytes:
            while self.cache and self.cache_size_bytes + len(raw) > self.cache_limit_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cache_size_bytes -= len(evicted)
            self.cache[path] = raw
            self.cache_size_bytes += len(raw)
        return raw

    def obj(self, path):
        value = json.loads(self.bytes(path))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}")  # noqa: TRY004 -- malformed serialized data
        return value

    def rows(self, path, *, journal=False):
        if journal and not path.exists():
            self.issue("pending_journal", path)
            return []
        raw = self.bytes(path)
        lines = raw.splitlines(keepends=True)
        result = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row must be an object")  # noqa: TRY004 -- malformed serialized data
            except (ValueError, UnicodeError) as error:
                if not journal:
                    raise ValueError(f"invalid frozen JSONL {path}:{index + 1}") from error
                partial = index == len(lines) - 1 and not line.endswith(b"\n")
                self.issue("partial_journal_tail" if partial else "invalid_journal_row", path,
                           line=index + 1, detail=str(error))
                continue
            result.append(row)
        return result

    def artifact(self, spec, base):
        path = self.path(spec["path"], base)
        if _hash(self.bytes(path)) != spec["sha256"]:
            raise ValueError(f"frozen artifact hash mismatch: {path}")
        return path

    def issue(self, code, path, **details):
        self.issues.append({"code": code, "path": str(path), **details})


def _index(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"missing or duplicate {key}")
        result[identity] = row
    return result


def _check(value, expected, label):
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(label)


def _store_unique(store, key, value, reader, path, conflicts):
    if key in conflicts:
        return
    if key in store:
        if store[key] == value:
            reader.issue("duplicate_identical_outcome", path, task_id=key)
        else:
            store.pop(key)
            conflicts.add(key)
            reader.issue("conflicting_outcomes_quarantined", path, task_id=key)
    else:
        store[key] = value


def _direct(reader, spec):
    path = reader.path(spec["plan_manifest"])
    plan = reader.obj(path)
    if plan["format_version"] != "readiness-to-permutation-plan-v1":
        raise ValueError("unsupported direct plan format")
    artifacts = plan["artifacts"]
    tasks_path = reader.artifact(artifacts["rerank_tasks"], path.parent)
    pools_path = reader.artifact(artifacts["frozen_candidate_sets"], path.parent)
    pools = _index(reader.rows(pools_path), "candidate_set_id")
    original = _index(reader.rows(tasks_path), "task_id")
    tasks = {}
    for tid, task in original.items():
        pool = pools[task["candidate_set_id"]]
        candidates = pool["candidates"]
        if type(task["top_n"]) is not int or not 1 <= task["top_n"] <= len(candidates):
            raise ValueError("direct top_n is outside its candidate pool")
        _check(task, {"keyword": pool["keyword"], "search_engine": pool["search_engine"],
                     "search_query": pool["search_query"], "search_snapshot_sha256": pool["search_snapshot_sha256"]},
               "direct task references different search evidence")
        pool_identity = {"engine": pool["search_engine"], "search_snapshot_sha256": pool["search_snapshot_sha256"],
                         "query": pool["search_query"], "candidates": candidates}
        if pool["candidate_set_id"] != "candidate-set:" + _hash(_canonical(pool_identity))[:20]:
            raise ValueError("direct pool identity mismatch")
        model_config = {
            "model_id": task["reranker_model"], "model_revision": task["reranker_model_revision"],
            "backend": task["reranker_backend"], "precision": task["reranker_precision"],
            "temperature": task["temperature"], "max_new_tokens": task["max_new_tokens"],
        }
        config_id = "reranker-config:" + _hash(_canonical(model_config))[:20]
        identity = {k: task[k] for k in ("prompt_candidate_id", "prompt_sha256", "candidate_set_id",
                                        "reranker_configuration_id", "rendered_prompt_sha256", "top_n")}
        expected_tid = "readiness-rerank-task:" + _hash(_canonical(identity))[:24]
        _check(task, {"task_id": expected_tid, "reranker_configuration_id": config_id,
                     "format_version": plan["format_version"],
                     "prompt_sha256": _hash(task["question"]),
                     "rendered_prompt_sha256": _hash(task["rendered_prompt"]),
                     "candidate_ids": [c["candidate_id"] for c in candidates],
                     "candidate_source_positions": [c["source_position"] for c in candidates],
                     "model_native_web_search": False}, "invalid direct task identity")
        if model_config not in plan["rerankers"]:
            raise ValueError("direct task model is absent from plan")
        tasks[tid] = {
            "task_key": "direct:" + tid, "source_kind": "direct", "prompt_id": task["prompt_candidate_id"],
            "question": task["question"], "question_sha256": task["prompt_sha256"],
            "keyword": task["keyword"], "model_id": task["reranker_model"],
            "model_revision": task["reranker_model_revision"], "method": "direct-rerank",
            "configuration_id": task["reranker_configuration_id"],
            "engine": task["search_engine"], "condition": "natural",
            "protocol": plan["format_version"], "cohort_id": plan["plan_id"],
            "candidate_ids": [c["url"] for c in candidates], "pool_id": _pool(candidates),
            "fit_pool_id": task["candidate_set_id"],
            "axes": _axes(task), "ranking": None, "source_cell_id": tid,
            "fake": task["reranker_backend"] == "fake", "top_n": task["top_n"],
        }
    outcomes, conflicts = {}, set()
    for name in spec.get("outcome_files", []):
        outcome_path = reader.path(name)
        rows = reader.rows(outcome_path, journal=True)
        if not rows:
            continue
        run = reader.obj(outcome_path.parent / "run_manifest.json")
        _check(run["tasks"], {"sha256": artifacts["rerank_tasks"]["sha256"]}, "direct runtime tasks changed")
        _check(run["plan"], {"plan_id": plan["plan_id"]}, "direct runtime plan mismatch")
        for row in rows:
            tid = row.get("task_id")
            try:
                task = original[tid]
                _check(row, {k: task[k] for k in ("candidate_set_id", "prompt_candidate_id",
                       "reranker_configuration_id", "reranker_model", "reranker_model_revision")},
                       "direct outcome task identity mismatch")
                _check(run["reranker"], {"configuration_id": task["reranker_configuration_id"],
                                        "model": task["reranker_model"], "revision": task["reranker_model_revision"]},
                       "direct runtime model mismatch")
                raw = row["raw_model_output"]
                tokens = re.split(r"[\s,]+", raw.strip())
                if (len(tokens) != task["top_n"] or len(tokens) != len(set(tokens))
                        or set(tokens) - set(task["candidate_ids"])):
                    raise ValueError("invalid direct top-N permutation")
                positions = dict(zip(task["candidate_ids"], task["candidate_source_positions"]))
                _check(row, {"candidate_ids": tokens, "source_position_vector": [positions[t] for t in tokens],
                             "permutation_sha256": _hash("\n".join(tokens)),
                             "raw_model_output_sha256": _hash(raw), "fake_backend": run["fake_backend"]},
                       "direct parsed permutation or provenance mismatch")
                urls = dict(zip(task["candidate_ids"], tasks[tid]["candidate_ids"]))
                value = {"ranking": [urls[t] for t in tokens], "fake": bool(row["fake_backend"]),
                         "source_result_sha256": _hash(_canonical(row)),
                         "source_result_hash_mode": "canonical-outcome-row-v1"}
                _store_unique(outcomes, tid, value, reader, outcome_path, conflicts)
            except (KeyError, TypeError, ValueError) as error:
                reader.issue("invalid_direct_outcome", outcome_path, task_id=tid, detail=str(error))
    for tid, value in outcomes.items():
        tasks[tid].update(value)
    return list(tasks.values())


def _legacy_cells(reader, manifest, base, prompts):
    sources = manifest["prompt_sources"]
    prompt_path = reader.artifact(sources["prompts_jsonl"], base)
    recorded = _index(reader.rows(prompt_path), "candidate_id")
    if recorded != prompts:
        raise ValueError("configured prompts differ from generator manifest")
    records = _index(reader.rows(reader.artifact(sources["selection_records_jsonl"], base)), "candidate_id")
    seed = manifest["prompt_selection_seed"]
    key = lambda *parts: (_hash("\0".join((str(seed), *parts))), *parts)
    bins = sorted({r["axis_bin"] for r in records.values()})
    count = manifest.get("prompt_population_count", manifest["prompt_count"])
    quota, remainder = divmod(count, len(bins))
    extras = set(sorted(bins, key=lambda b: key("axis-bin", str(b)))[:remainder])
    selected = []
    for bucket in bins:
        ids = sorted((pid for pid in prompts if records[pid]["axis_bin"] == bucket), key=lambda pid: key("prompt", pid))
        selected.extend(ids[:quota + int(bucket in extras)])
    selected.sort(key=lambda pid: (records[pid]["axis_bin"], pid))
    selected = selected[manifest.get("prompt_shard_index", 0)::manifest.get("prompt_shard_count", 1)]
    identity = [{"prompt_id": pid, "question_sha256": _hash(prompts[pid]["question"]),
                 "axis_bin": records[pid]["axis_bin"]} for pid in selected]
    if _hash(_canonical(identity)) != manifest["prompt_selection_sha256"]:
        raise ValueError("generator prompt selection hash mismatch")
    cells = []
    for pid, method, engine, condition in itertools.product(selected, manifest["methods"], manifest["engines"], manifest["conditions"]):
        core = {"prompt_id": pid, "prompt_sha256": _hash(prompts[pid]["question"]),
                "method": method, "engine": engine, "condition": condition}
        cells.append({"cell_id": _hash(_canonical(core))[:20], **core})
    if "cell_selection" in manifest:
        chosen = {r["cell_id"] for r in reader.rows(reader.artifact(manifest["cell_selection"], base))}
        cells = [c for c in cells if c["cell_id"] in chosen]
        if len(cells) != len(chosen):
            raise ValueError("cell selection has unknown IDs")
    if len(cells) != manifest["cell_count"]:
        raise ValueError("generator planned cell count mismatch")
    return cells


def _agentic(reader, spec):
    prompt_path = reader.path(spec["prompts_jsonl"])
    prompts = _index(reader.rows(prompt_path), "candidate_id")
    for row in prompts.values():
        if row.get("question_sha256", _hash(row["question"])) != _hash(row["question"]):
            raise ValueError("prompt question hash mismatch")
    axes = {}
    if spec.get("axis_jsonl"):
        axes = _index(reader.rows(reader.path(spec["axis_jsonl"])), "candidate_id")
        for pid, row in axes.items():
            if pid not in prompts or row.get("question_sha256") != _hash(prompts[pid]["question"]):
                raise ValueError("axis join needs exact prompt ID and question hash")
    outer = None
    if spec.get("task_manifest"):
        outer_path = reader.path(spec["task_manifest"])
        outer = reader.obj(outer_path)
        task_spec = outer.get("tasks") or {"path": "tasks.jsonl", "sha256": outer["frozen_files"]["tasks.jsonl"]}
        cells = list(_index(reader.rows(reader.artifact(task_spec, outer_path.parent)), "cell_id").values())
    tasks = []
    for name in spec["generator_roots"]:
        root = reader.path(name)
        if outer:
            if root.name not in outer["models"]:
                raise ValueError("generator_roots must point at model directories under task_manifest")
            if root != outer_path.parent / "models" / root.name:
                raise ValueError("generator root is outside its frozen plan")
            manifest = outer["models"][root.name]
            protocol = outer["format_version"]
            configuration_id = "agentic-plan-config:" + _hash(_canonical({
                "git_commit": outer.get("git_commit"), "model_id": manifest["model_id"],
                "model_revision": manifest["model_revision"], "profile_sha256": manifest.get("profile_sha256"),
                "tasks_sha256": task_spec["sha256"], "protocol": protocol,
            }))
        else:
            manifest = reader.obj(root / "run_manifest.json")
            cells = _legacy_cells(reader, manifest, root, prompts)
            protocol = manifest["format_version"] + ":" + manifest["condition_mode"]
            configuration_id = "agentic-config:" + manifest["config_sha256"]
        model, revision = manifest["model_id"], manifest["model_revision"]
        cohort = "agentic-cohort:" + _hash(_canonical(sorted((pid, _hash(r["question"])) for pid, r in prompts.items())))
        normalized = {}
        for cell in cells:
            pid = cell["prompt_id"]
            prompt = prompts[pid]
            core = {k: cell[k] for k in ("method", "engine", "condition", "prompt_id", "prompt_sha256")}
            _check(cell, {"cell_id": _hash(_canonical(core))[:20], "prompt_sha256": _hash(prompt["question"])},
                   "invalid frozen generator cell")
            cid = cell["cell_id"]
            normalized[cid] = {"task_key": "agentic:" + _hash(_canonical([cohort, model, revision, cid])),
                "source_kind": "agentic",
                "prompt_id": pid, "question": prompt["question"], "question_sha256": cell["prompt_sha256"],
                "keyword": prompt.get("keyword"), "model_id": model, "model_revision": revision,
                "configuration_id": configuration_id,
                "method": cell["method"], "engine": cell["engine"], "condition": cell["condition"],
                "protocol": protocol, "cohort_id": cohort, "candidate_ids": [], "pool_id": None,
                "fit_pool_id": None,
                "axes": {**_axes(prompt), **_axes(axes.get(pid, {}))}, "ranking": None,
                "source_cell_id": cid, "fake": bool(manifest.get("fake_backend", False))}
        completed, conflicts = {}, set()
        paths = sorted(root.glob("results/*.json")) if not outer else sorted(root.glob("outputs/**/results/*.json"))
        for result_path in paths:
            try:
                result = reader.obj(result_path)
                cid = result["cell_id"]
                task = normalized[cid]
                runtime = reader.obj(result_path.parent.parent / "run_manifest.json")
                _check(runtime, {"model_id": model, "model_revision": revision}, "generator runtime model mismatch")
                public, mapping = _task_and_mapping(result_path, prompt={"question": task["question"],
                    "question_sha256": task["question_sha256"], "axis_bin": "report"},
                    generator_model_id=model, master_seed=0)
                if mapping.source_result_sha256 != _hash(reader.bytes(result_path)):
                    raise ValueError("generator source changed while it was being read")
                _check(result, {k: task[k] for k in ("prompt_id", "method", "engine", "condition")}, "result cell factors mismatch")
                trace_path = reader.path(result["trace"], result_path.parent)
                trace = reader.obj(trace_path)
                if (trace.get("trace_sha256") != mapping.source_trace_sha256
                        or _hash(_canonical({k: v for k, v in trace.items() if k != "trace_sha256"})) != mapping.source_trace_sha256):
                    raise ValueError("generator trace changed while it was being read")
                forwarded = []
                for event in trace["events"]:
                    kind = event.get("event_type")
                    if kind not in {"compaction", "observation"}:
                        continue
                    payload = event["payload"]
                    snippets = payload["selected_snippets" if kind == "compaction" else "snippets"]
                    forwarded.append({"kind": kind, "query": payload.get("query"), "snippets": [
                        {k: row[k] for k in ("url", "title", "text")} for row in snippets]})
                value = {"ranking": result["ranking"], "candidate_ids": [e.url for e in public.evidence],
                         "pool_id": _pool([asdict(e) for e in public.evidence]),
                         "fit_pool_id": "agentic-forwarded-input:" + _hash(_canonical({
                             "keyword": task["keyword"], "events": forwarded,
                         })),
                         "source_result_sha256": _hash(_canonical({k: v for k, v in result.items() if k != "trace"})),
                         "source_result_hash_mode": "canonical-result-without-trace-path-v1",
                         "source_result_sha256s": [mapping.source_result_sha256],
                         "source_trace_sha256": mapping.source_trace_sha256,
                         "runtime_config_sha256": runtime.get("config_sha256"),
                         "execution_configuration_id": runtime.get("config_sha256"),
                         "configuration_mixed": bool(runtime.get("resume_migration")),
                         "fake": bool(runtime.get("fake_backend", False) or result.get("fake_backend", False))}
                if runtime.get("resume_migration"):
                    reader.issue("mixed_generator_resume_provenance", result_path,
                                 detail="Current manifest covers preserved results from an earlier execution configuration")
                if not runtime.get("config_sha256"):
                    reader.issue("missing_runtime_configuration", result_path)
                if cid in conflicts:
                    continue
                if cid in completed:
                    fields = {"source_result_sha256s"}
                    if ({k: v for k, v in completed[cid].items() if k not in fields}
                            != {k: v for k, v in value.items() if k not in fields}):
                        completed.pop(cid)
                        conflicts.add(cid)
                        reader.issue("conflicting_outcomes_quarantined", result_path, task_id=cid)
                    else:
                        completed[cid]["source_result_sha256s"] = sorted(set(
                            completed[cid]["source_result_sha256s"] + value["source_result_sha256s"]))
                        reader.issue("duplicate_identical_outcome", result_path, task_id=cid)
                else:
                    completed[cid] = value
            except (OSError, KeyError, TypeError, ValueError) as error:
                reader.issue("invalid_agentic_result", result_path, detail=str(error))
        for cid, value in completed.items():
            normalized[cid].update(value)
        tasks.extend(normalized.values())
    return tasks


def _judges(reader, spec, generated):
    plan_path = reader.path(spec["plan_manifest"])
    plan = reader.obj(plan_path)
    role = spec.get("role", "bulk")
    model = plan[role + "_model"]
    task_path = reader.artifact(plan["artifacts"][role + "_tasks"], plan_path.parent)
    maps_path = reader.artifact(plan["artifacts"]["private_mapping"], plan_path.parent)
    mappings = _index(reader.rows(maps_path), "judge_task_id")
    public = _index(reader.rows(task_path), "judge_task_id")
    expected, source_by_id = {}, {}
    generated_by_source = {}
    for generated_task in generated:
        for result_hash in generated_task.get("source_result_sha256s", []):
            key = (result_hash, generated_task["model_id"], generated_task.get("source_trace_sha256"))
            generated_by_source.setdefault(key, []).append(generated_task)
    for tid, value in public.items():
        # The frozen denominator survives missing or invalid source artifacts.
        # A task with no verified source can never acquire a counted judgment.
        expected[tid] = {"judge_task_id": tid, "task_key": None,
            "judge_model_id": model["model_id"], "judge_model_revision": model["model_revision"],
            "protocol": plan["format_version"], "ranking_visible": plan["format_version"] == TRANSCRIPT_FORMAT_VERSION,
            "fake": False, "source_verified": False}
        try:
            task = AgenticJudgeTask.from_dict(value)
            if task.format_version != plan["format_version"]:
                raise ValueError("judge task protocol mismatch")
            mapping = mappings[tid]
            result_path = reader.path(mapping["source_result_path"], plan_path.parent)
            if _hash(reader.bytes(result_path)) != mapping["source_result_sha256"]:
                raise ValueError("judge source result hash mismatch")
            matched = generated_by_source.get((mapping["source_result_sha256"], mapping["generator_model_id"],
                                               mapping["source_trace_sha256"]), [])
            if len(matched) != 1:
                raise ValueError("judge source must match exactly one verified generator task")
            generated_task = matched[0]
            rebuilt, rebuilt_map = _task_and_mapping(result_path,
                prompt={"question": generated_task["question"], "question_sha256": generated_task["question_sha256"],
                        "axis_bin": mapping["axis_bin"]}, generator_model_id=generated_task["model_id"],
                master_seed=plan["master_seed"], recorded_conversation=task.format_version == TRANSCRIPT_FORMAT_VERSION)
            if rebuilt.to_dict() != value or asdict(rebuilt_map) != {**mapping, "generated_ranking_evidence_ids": tuple(mapping["generated_ranking_evidence_ids"])}:
                raise ValueError("judge plan does not reproduce from its recorded source")
            expected[tid] = {"judge_task_id": tid, "task_key": generated_task["task_key"],
                "judge_model_id": model["model_id"], "judge_model_revision": model["model_revision"],
                "protocol": task.format_version, "ranking_visible": task.format_version == TRANSCRIPT_FORMAT_VERSION,
                "fake": generated_task["fake"], "source_verified": True}
            source_by_id[tid] = task
        except (OSError, KeyError, TypeError, ValueError) as error:
            reader.issue("invalid_judge_task", task_path, task_id=tid, detail=str(error))
    outcomes, conflicts = {}, set()
    for name in spec.get("outcome_files", []):
        outcome_path = reader.path(name)
        rows = reader.rows(outcome_path, journal=True)
        if not rows:
            continue
        runtime = reader.obj(outcome_path.parent / "run_manifest.json")
        _check(runtime, {"model_id": model["model_id"], "model_revision": model["model_revision"],
                         "pipeline": "agentic-judge-" + role}, "judge runtime model or role mismatch")
        resume = runtime["resume_identity"]
        if resume["source_manifest_sha256"] != _hash(reader.bytes(plan_path)):
            raise ValueError("judge runtime plan hash mismatch")
        run_tasks = reader.artifact(runtime["tasks"], outcome_path.parent)
        if resume["tasks_sha256"] != _hash(reader.bytes(run_tasks)):
            raise ValueError("judge runtime queue hash mismatch")
        runtime_public = _index(reader.rows(run_tasks), "judge_task_id")
        if any(public.get(tid) != row for tid, row in runtime_public.items()):
            raise ValueError("judge runtime queue is not a subset of the frozen plan")
        for row in rows:
            tid = row.get("judge_task_id")
            try:
                task = source_by_id[tid]
                if tid not in runtime_public:
                    raise ValueError("judge outcome absent from runtime task queue")
                ids = [e.evidence_id for e in task.evidence]
                _check(row, {"blind_case_id": task.blind_case_id, "evidence_ids": ids,
                             "fake_backend": runtime["fake_backend"],
                             "raw_output_sha256": _hash(row["raw_output"])}, "judge outcome identity mismatch")
                parsed = validate_agentic_judgment(row["raw_output"], allowed_evidence_ids=ids)
                if parsed != row["parsed_output"]:
                    raise ValueError("judge raw and parsed outputs differ")
                urls = {e.evidence_id: e.url for e in task.evidence}
                value = {**expected[tid], "ideal_ranking": [urls[e] for e in parsed["ideal_relevance_ranking"]],
                    "support_ranking": [urls[e["evidence_id"]] for e in parsed["realized_support_ranking"]],
                    "scores": {k: parsed[k] for k in ("request_fulfillment", "evidence_grounding", "judge_confidence", "unsupported_claim_count")},
                    "fake": bool(expected[tid]["fake"] or row["fake_backend"])}
                _store_unique(outcomes, tid, value, reader, outcome_path, conflicts)
            except (KeyError, TypeError, ValueError) as error:
                reader.issue("invalid_judge_outcome", outcome_path, task_id=tid, detail=str(error))
    return list(expected.values()), list(outcomes.values())


def load_study(config: dict, base_dir: Path) -> dict:
    """Load explicit source lists, keeping pending tasks and quarantining bad rows.

    Frozen-input corruption raises ValueError. Invalid mutable journal entries
    become visible issues and never count as completed work. No files are written.
    """
    reader = _Reader(Path(base_dir))
    tasks = []
    for spec in config.get("direct", []):
        tasks.extend(_direct(reader, spec))
    for spec in config.get("agentic", []):
        tasks.extend(_agentic(reader, spec))
    tasks = list(_index(tasks, "task_key").values())
    expected, judgments = [], []
    for spec in config.get("judges", []):
        planned, saved = _judges(reader, spec, tasks)
        expected.extend(planned)
        judgments.extend(saved)
    return {"tasks": tasks, "judge_tasks": expected, "judgments": judgments,
            "sources": sorted(reader.sources.values(), key=lambda row: row["path"]), "issues": reader.issues}
