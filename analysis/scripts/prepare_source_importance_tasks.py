#!/usr/bin/env python3
"""Freeze SI-v3 judge tasks (one answer x one source, plus J1) from sealed generator cells.

Read-only on the generator datasets. Writes a new output folder:
- tasks.jsonl.gz: unique task records (identical semantic tasks appear once);
- cells.jsonl.gz: per cell, factor metadata, presented source order, the cleaned
  and raw generator ranking, repair flags, and the task ID (or status) per source;
- manifest.json: inputs, counts, protocol constants, git commit.
Every observed source is judged, listed or not. Nothing here runs a model.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as si
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.interpretability.pipeline.agentic_cells import completed_generator_refs, iter_cells
from analysis.interpretability.pipeline.agentic_judging import _trace_evidence, _visible_evidence, _digest
from analysis.scripts.run_source_importance_judge import file_hash

FINAL_PURPOSES = ("parallel_final", "reactive_action", "reactive_forced_finish")


def generator_output_record(trace: dict) -> dict:
    """Raw pre-answer ranking as emitted, plus the controller's repair flags."""

    events = trace.get("events", [])
    finals = [e["payload"] for e in events
              if e.get("event_type") == "llm_call" and e["payload"].get("purpose") in FINAL_PURPOSES]
    repairs = [e["payload"] for e in events if e.get("event_type") == "controller_repair"]
    raw_ranking = None
    if finals:
        try:
            raw_ranking = json.loads(finals[-1].get("raw_output") or "").get("ranking")
        except (ValueError, AttributeError):
            raw_ranking = None
    return {
        "raw_ranking": raw_ranking,
        "final_attempts": len(finals),
        "final_validation_failures": sum(bool(p.get("validation_error")) for p in finals),
        "repaired": bool(repairs),
        "dropped_ranking_references": [d for p in repairs for d in p.get("dropped_ranking_references", [])],
        "answer_truncated": any(p.get("answer_truncated") for p in repairs),
        "malformed_json_recovered": any(p.get("malformed_json_recovered") for p in repairs),
    }


SENSITIVITY_SALT = "si-truncation-sensitivity-v1"


def judged_answer(trace: dict, stored: str) -> tuple[str | None, str]:
    """The complete answer the generator wrote, and where it came from.

    The controller stores answers cut to 1,200 characters when the model's valid
    JSON answer was longer; the full text is still the final attempt's raw output.
    Returns (answer, source): ``stored`` (not cut), ``trace_full`` (recovered in
    full and verified to start with the stored text), ``trace_prefix`` (the model
    output was broken JSON, only the stored prefix exists) or (None,
    ``trace_answer_mismatch``) when the trace does not reproduce the stored text.
    """

    events = trace.get("events", [])
    repairs = [e["payload"] for e in events if e.get("event_type") == "controller_repair"]
    if not any(r.get("answer_truncated") for r in repairs):
        return stored, "stored"
    if any(r.get("malformed_json_recovered") for r in repairs):
        return stored, "trace_prefix"
    finals = [e["payload"] for e in events
              if e.get("event_type") == "llm_call" and e["payload"].get("purpose") in FINAL_PURPOSES]
    try:
        full = json.loads(finals[-1]["raw_output"])["answer"]
    except (IndexError, KeyError, TypeError, ValueError):
        return None, "trace_answer_mismatch"
    if not isinstance(full, str) or not full.startswith(stored) or len(full) <= len(stored):
        return None, "trace_answer_mismatch"
    return full, "trace_full"


def in_sensitivity_subset(cell_id: str, fraction: float) -> bool:
    value = int(hashlib.sha256(f"{SENSITIVITY_SALT}:{cell_id}".encode()).hexdigest()[:8], 16) / 16**8
    return value < fraction


def source_tasks(request: str, answer: str, evidence: list, presented: list, max_tokens: int):
    entries, tasks = [], []
    for position, row in enumerate(evidence):
        entry = {"url": row["url"], "presented_position": position,
                 "source_sha256": _digest({"title": row.get("title", ""), "text": row.get("text", "")})}
        if not si.is_assessable(row):
            entry["status"] = "unassessable_input"
        else:
            item = si.prepare_source_task(request=request, answer=answer, source=row,
                                          observed_urls=presented, max_tokens=max_tokens)
            tasks.append(si.task_record(item))
            entry.update(judge_task_id=item["base"]["judge_task_id"],
                         answer_names_source_url=item["base"]["answer_names_source_url"],
                         answer_masked=bool(item["base"]["mask_spans"]))
        entries.append(entry)
    return entries, tasks


def build_cell(cell: dict, *, max_tokens: int, j1_max_tokens: int,
               sensitivity_fraction: float = 0.1) -> tuple[dict, list[dict]]:
    generation = cell["generation"]
    record = {"fingerprint": cell["fingerprint"], "model": cell["model"],
              **{k: generation.get(k) for k in ("cell_id", "prompt_id", "method", "engine", "condition")},
              "condition_audit": generation.get("condition_audit"),
              "generation_record_id": cell["generation_ref"]["record_id"],
              "trace_record_id": cell["trace_ref"]["record_id"]}
    tasks: list[dict] = []
    request, stored = cell.get("prompt_text"), generation.get("answer")
    if not request or not stored:
        return {**record, "status": "missing_request_or_answer"}, tasks
    answer, answer_source = judged_answer(cell["trace"], stored)
    record.update(answer_source=answer_source, truncated=answer_source != "stored",
                  stored_answer_chars=len(stored), judged_answer_chars=len(answer) if answer else None)
    if answer is None:
        return {**record, "status": answer_source}, tasks
    evidence, raw_count = _trace_evidence(cell["trace"], generation["method"])
    if raw_count != generation.get("final_snippet_count"):
        return {**record, "status": "evidence_count_mismatch"}, tasks
    presented = [row["url"] for row in evidence]
    record.update(stored_answer=stored, judged_answer=answer, provenance=provenance(cell, answer, evidence),
                  prompt_metadata=cell.get("prompt_record", {}), task_metadata=cell.get("task_metadata", {}),
                  keyword_memberships=cell.get("keyword_memberships", {}))
    ranking = list(generation.get("ranking") or [])
    if len(ranking) != len(set(ranking)) or not set(ranking) <= set(presented):
        return {**record, "status": "ranking_outside_evidence"}, tasks
    masked, mask_spans = si.mask_answer_citations(answer, presented)
    record.update(answer_mask_spans=mask_spans,
                  judged_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                  masked_answer_sha256=hashlib.sha256(masked.encode()).hexdigest())
    # Primary: the complete answer the generator wrote (J1 sees the same text).
    j1 = si.prepare_fulfilment_task(request=request, answer=answer, max_tokens=j1_max_tokens)
    tasks.append(si.task_record(j1))
    sources, primary = source_tasks(request, answer, evidence, presented, max_tokens)
    tasks += primary
    if answer_source == "trace_full" and in_sensitivity_subset(record["cell_id"], sensitivity_fraction):
        # Sensitivity: the stored 1,200-character version, on a fixed random subset.
        stored_sources, stored_tasks = source_tasks(request, stored, evidence, presented, max_tokens)
        stored_j1 = si.prepare_fulfilment_task(request=request, answer=stored, max_tokens=j1_max_tokens)
        tasks += stored_tasks + [si.task_record(stored_j1)]
        record["stored_answer_sensitivity"] = {"sources": stored_sources,
                                               "answer_mask_spans": si.mask_answer_citations(stored, presented)[1],
                                               "j1_task_id": stored_j1["base"]["judge_task_id"]}
    return {**record, "status": "no_observed_sources" if not evidence else "ok",
            "presented": presented, "generator_ranking": ranking,
            "generator_output": generator_output_record(cell["trace"]),
            "j1_task_id": j1["base"]["judge_task_id"], "sources": sources}, tasks


def provenance(cell, answer, evidence):
    """Compare serialized final evidence and final raw answer, not inferred order."""
    calls = [e["payload"] for e in cell["trace"].get("events", [])
             if e.get("event_type") == "llm_call" and e.get("payload", {}).get("purpose") in FINAL_PURPOSES]
    history, errors = [], []
    ids = {e["url"]: f"S{i}" for i, e in enumerate(evidence, 1)}
    for call in calls:
        request = dict(call.get("request") or {})
        request.setdefault("purpose", call["purpose"])
        saved = {k: request.get(k) for k in ("purpose", "prompt", "response_schema", "force_finish")}
        try:
            visible = _visible_evidence(request, ids)
            history.append({"request": saved, "visible": visible, "raw_output": call.get("raw_output")})
        except (ValueError, TypeError, KeyError) as exc:
            history.append({"request": saved, "visible": None, "raw_output": call.get("raw_output"),
                            "error": str(exc)})
    if not history or history[-1]["visible"] is None:
        errors.append("final_visible_evidence_unavailable")
    else:
        expected = [(e["url"], e["title"], e["text"]) for e in evidence]
        actual = [(e["url"], e["title"], e["text"]) for e in history[-1]["visible"]]
        if actual != expected:
            errors.append("generator_judge_evidence_mismatch")
    try:
        if json.loads(calls[-1]["raw_output"])["answer"] != answer:
            errors.append("final_answer_mismatch")
    except (ValueError, KeyError, TypeError, IndexError):
        errors.append("final_answer_unverifiable")
    repeated = {}
    for event in cell["trace"].get("events", []):
        if event.get("event_type") == "observation":
            for row in event.get("payload", {}).get("snippets", []):
                repeated.setdefault(row.get("url"), []).append({k: row.get(k) for k in ("title", "text")})
    return {"status": "verified" if not errors else "provenance_unresolved", "reasons": errors,
            "source_trace_sha256": _digest(cell["trace"]), "final_calls": history,
            "repeated_observations": {k: v for k, v in repeated.items() if len(v) > 1},
            "attribution_target": "supplied_record_not_independently_verified_webpage"}


def build_v4_cell(cell, *, max_tokens=4096, map_max_tokens=4096, j1_max_tokens=64,
                  sensitivity_fraction=0.0, preprocessing=v4.PREPROCESSING_VERSION):
    # Reuse recovery, frozen evidence selection, source coverage and J1 unchanged.
    record, original = build_cell(cell, max_tokens=max_tokens, j1_max_tokens=j1_max_tokens,
                                  sensitivity_fraction=0.0)
    record.update(protocol=v4.PROTOCOL, eligibility_version=v4.ELIGIBILITY_VERSION,
                  preprocessing=preprocessing,
                  prompt_metadata=cell.get("prompt_record", {}),
                  task_metadata=cell.get("task_metadata", {}))
    if record["status"] != "ok":
        return record, []
    source_records = {t["judge_task_id"]: t for t in original if t["task"] == "source_importance"}
    j1 = next(t for t in original if t["task"] == "fulfilment")
    answer, request = j1["answer"], j1["request"]
    stored = cell["generation"]["answer"]
    evidence, _ = _trace_evidence(cell["trace"], record["method"])
    record.update(stored_answer=stored, judged_answer=answer, provenance=provenance(cell, answer, evidence))

    def bundle(text, j1_record):
        masked, spans = v4.mask_answer(text, record["presented"], preprocessing)
        mapper = v4.prepare_map_task(request=request, answer=masked, max_tokens=map_max_tokens,
                                     preprocessing=preprocessing)["record"]
        sources, tasks = [], [mapper, j1_record]
        for source in record["sources"]:
            entry = {k: v for k, v in source.items() if k != "judge_task_id"}
            if "judge_task_id" in source:
                t = source_records[source["judge_task_id"]]
                dependency = v4.source_dependency(mapper["judge_task_id"], t["source_title"],
                                                   t["source_text"], max_tokens)
                entry.update(dependency_id=dependency["judge_task_id"], map_task_id=mapper["judge_task_id"],
                             status="awaiting_map")
                tasks.append(dependency)
            sources.append(entry)
        return {"map_task_id": mapper["judge_task_id"], "sources": sources,
                "j1_task_id": j1_record["judge_task_id"], "answer_mask_spans": spans,
                "masked_answer_sha256": hashlib.sha256(masked.encode()).hexdigest()}, tasks

    primary, tasks = bundle(answer, j1)
    if record["answer_source"] == "trace_full" and in_sensitivity_subset(record["cell_id"], sensitivity_fraction):
        sensitivity, extra = bundle(stored, si.task_record(si.prepare_fulfilment_task(
            request=request, answer=stored, max_tokens=j1_max_tokens)))
        record["stored_answer_sensitivity"] = sensitivity
        tasks += extra
    record.update(primary)
    return record, tasks


def build_v3_comparison_cell(cell, **kwargs):
    record, tasks = build_cell(cell, **kwargs)
    record.update(protocol=v4.V3_COMPARISON_PROTOCOL, preprocessing=v4.PROSE_MASK_VERSION)
    if record["status"] != "ok":
        return record, tasks
    replacements, new_tasks = {}, []
    for task in tasks:
        if task["task"] != "source_importance":
            new_tasks.append(task)
            continue
        complete = record["judged_answer"]
        old_mask, _ = si.mask_answer_citations(complete, record["presented"])
        original = complete if task["masked_answer"] == old_mask else record["stored_answer"]
        masked, _ = v4.mask_answer(original, record["presented"], v4.PROSE_MASK_VERSION)
        legacy = si.task_record(si._source_item(request=task["request"], masked_answer=masked,
            title=task["source_title"], text=task["source_text"], max_tokens=task["max_tokens"]))
        wrapped = v4.v3_comparison_item(legacy)["record"]
        replacements[task["judge_task_id"]] = wrapped["judge_task_id"]
        new_tasks.append(wrapped)
    for bundle, answer in ((record, record["judged_answer"]),
                            (record.get("stored_answer_sensitivity", {}), record["stored_answer"])):
        if not bundle:
            continue
        masked, spans = v4.mask_answer(answer, record["presented"], v4.PROSE_MASK_VERSION)
        bundle.update(answer_mask_spans=spans, masked_answer_sha256=hashlib.sha256(masked.encode()).hexdigest())
        for source in bundle["sources"]:
            if "judge_task_id" in source:
                source["judge_task_id"] = replacements[source["judge_task_id"]]
    return record, new_tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="DATASET_ROOT:MODEL",
                        help="generator dataset root and model (qwen38 or llama4); repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, help="optional file with one prompt_id per line (a frozen sample)")
    parser.add_argument("--cell-fingerprints", type=Path, help="exact frozen cell selection; one verified fingerprint per line")
    parser.add_argument("--exclude-cell-fingerprints", type=Path,
                        help="previously judged cells to preserve; exclusions are recorded in the manifest")
    parser.add_argument("--prior-map-task-ids", type=Path,
                        help="known SI-v4 maps; overlapping new cells are recorded as blocked for reconciliation")
    parser.add_argument("--protocol", choices=("si-v3", "si-v4"), default="si-v3")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--map-max-tokens", type=int, default=v4.DEFAULT_MAX_TOKENS)
    parser.add_argument("--preprocessing", choices=(v4.PREPROCESSING_VERSION, v4.PROSE_MASK_VERSION),
                        default=v4.PREPROCESSING_VERSION)
    parser.add_argument("--j1-max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, help="at most this many cells per source (development only)")
    parser.add_argument("--truncation-sensitivity-fraction", type=float, default=0.1,
                        help="share of truncated cells whose stored 1,200-character answer is also judged")
    parser.add_argument("--index-directory", type=Path,
                        help="scratch folder (e.g. node-local $TMPDIR) for the temporary deduplication index")
    args = parser.parse_args(argv)
    selected = v4 if args.protocol == "si-v4" else si
    if args.max_tokens is None:
        args.max_tokens = selected.DEFAULT_MAX_TOKENS
    if min(args.max_tokens, args.map_max_tokens, args.j1_max_tokens) <= 0:
        parser.error("token budgets must be positive")
    if not 0 <= args.truncation_sensitivity_fraction <= 1:
        parser.error("sensitivity fraction must be in [0,1]")
    if args.output.exists():
        parser.error("output already exists; use a new folder")
    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = {line.strip() for line in args.prompt_ids.read_text().splitlines() if line.strip()}
    fingerprints = None
    if args.cell_fingerprints:
        selected_lines = [s.strip() for s in args.cell_fingerprints.read_text().splitlines() if s.strip()]
        fingerprints = set(selected_lines)
        if not fingerprints or len(fingerprints) != len(selected_lines):
            parser.error("cell selection must be nonempty and contain no duplicates")
    missing_fingerprints = set(fingerprints or ())
    excluded = set(args.exclude_cell_fingerprints.read_text().split()) if args.exclude_cell_fingerprints else set()
    prior_maps = set(args.prior_map_task_ids.read_text().split()) if args.prior_map_task_ids else set()
    if args.prior_map_task_ids and args.protocol != "si-v4":
        parser.error("prior map identities require SI-v4")
    if fingerprints and fingerprints & excluded:
        parser.error("selected cells overlap the recorded exclusions")
    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    # Disk-backed deduplication avoids retaining millions of task IDs in Python.
    index = (Path(tempfile.mkdtemp(prefix="si-freeze-index-", dir=args.index_directory)) / "task-index.sqlite"
             if args.index_directory else partial / "task-index.sqlite")
    seen = sqlite3.connect(index)
    seen.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
    seen.execute("CREATE TABLE cells (id TEXT PRIMARY KEY)")
    counts: Counter = Counter()
    inputs = []
    with gzip.open(partial / "tasks.jsonl.gz", "wt", encoding="utf-8") as task_file, \
            gzip.open(partial / "cells.jsonl.gz", "wt", encoding="utf-8") as cell_file:
        for spec in args.source:
            root_text, _, model = spec.rpartition(":")
            root = Path(root_text)
            refs, ref_counts = completed_generator_refs(root, model=model, prompt_ids=prompt_ids)
            ref_counts["previously_judged_excluded"] = sum(r["fingerprint"] in excluded for r in refs)
            refs = [r for r in refs if r["fingerprint"] not in excluded]
            if fingerprints is not None:
                refs = [r for r in refs if r["fingerprint"] in fingerprints]
            if args.limit is not None:
                refs = refs[:args.limit]
            inputs.append({"dataset_root": str(root), "model": model, "cell_selection": dict(ref_counts),
                           "cells_used": len(refs)})
            for cell in iter_cells(root, refs):
                seen.execute("INSERT INTO cells VALUES (?)", (cell["fingerprint"],))
                missing_fingerprints.discard(cell["fingerprint"])
                builder = (build_v4_cell if args.protocol == "si-v4" else build_v3_comparison_cell
                           if args.preprocessing == v4.PROSE_MASK_VERSION else build_cell)
                extra = {"map_max_tokens": args.map_max_tokens, "preprocessing": args.preprocessing} if args.protocol == "si-v4" else {}
                record, tasks = builder(cell, max_tokens=args.max_tokens, j1_max_tokens=args.j1_max_tokens,
                                        sensitivity_fraction=args.truncation_sensitivity_fraction, **extra)
                if record.get("map_task_id") in prior_maps:
                    record.update(status="prior_map_requires_reconciliation",
                                  blockage="Earlier diagnostic owns this map; retain this cell for explicit result reuse, without repeating inference.")
                    tasks = []
                counts[f"answer_{record.get('answer_source', 'none')}"] += 1
                counts["stored_answer_sensitivity_cells"] += "stored_answer_sensitivity" in record
                counts[f"cells_{record['status']}"] += 1
                counts["source_tasks_referenced"] += sum("judge_task_id" in s or "dependency_id" in s
                                                         for s in record.get("sources", []))
                counts["sources_unassessable"] += sum(s.get("status") == "unassessable_input"
                                                      for s in record.get("sources", []))
                cell_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                for task in tasks:
                    digest = _digest(task)
                    prior = seen.execute("SELECT digest FROM tasks WHERE id=?", (task["judge_task_id"],)).fetchone()
                    if prior and prior[0] != digest:
                        raise ValueError("conflicting semantic task identity")
                    if not prior:
                        seen.execute("INSERT INTO tasks VALUES (?,?)", (task["judge_task_id"], digest))
                        counts[f"unique_{task['task']}_tasks"] += 1
                        task_file.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    seen.close()
    index.unlink()
    if args.index_directory:
        index.parent.rmdir()
    if missing_fingerprints:
        raise ValueError(f"{len(missing_fingerprints)} selected cells unavailable or excluded; partial freeze was not promoted")
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": selected.PROTOCOL,
                "task_version": selected.TASK_VERSION, "retry_contract": selected.RETRY_CONTRACT,
                "max_tokens": args.max_tokens, "j1_max_tokens": args.j1_max_tokens, "inputs": inputs,
                "judged_answer": "complete generator answer (trace-recovered when stored truncated)",
                "truncation_sensitivity": {"fraction": args.truncation_sensitivity_fraction,
                                           "salt": SENSITIVITY_SALT},
                "prompt_ids_sha256": hashlib.sha256(args.prompt_ids.read_bytes()).hexdigest()
                if args.prompt_ids else None, "limit": args.limit, "counts": dict(counts),
                "files": {name: file_hash(partial / name)
                          for name in ("tasks.jsonl.gz", "cells.jsonl.gz")},
                "git_commit": commit, "scientific_result": False}
    if args.cell_fingerprints:
        manifest["cell_fingerprints_sha256"] = hashlib.sha256(args.cell_fingerprints.read_bytes()).hexdigest()
    if args.exclude_cell_fingerprints:
        manifest["excluded_cells"] = {"count": len(excluded),
            "sha256": file_hash(args.exclude_cell_fingerprints),
            "path": str(args.exclude_cell_fingerprints.resolve())}
    if args.prior_map_task_ids:
        manifest["prior_map_tasks"] = {"count": len(prior_maps), "sha256": file_hash(args.prior_map_task_ids)}
    if args.protocol == "si-v4":
        manifest.update(map_max_tokens=args.map_max_tokens, preprocessing=args.preprocessing,
                        eligibility_version=v4.ELIGIBILITY_VERSION, warm_gpu_hour_ratio_ceiling=3.0)
    elif args.preprocessing == v4.PROSE_MASK_VERSION:
        manifest.update(protocol=v4.V3_COMPARISON_PROTOCOL, preprocessing=v4.PROSE_MASK_VERSION)
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.rename(args.output)
    print(json.dumps({"output": str(args.output), **manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
