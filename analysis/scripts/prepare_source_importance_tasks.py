#!/usr/bin/env python3
"""Freeze SI-v1 judge tasks (one answer x one source, plus J1) from sealed generator cells.

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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as si
from analysis.interpretability.pipeline.agentic_cells import completed_generator_refs, iter_cells
from analysis.interpretability.pipeline.agentic_judging import _trace_evidence

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
        entry = {"url": row["url"], "presented_position": position}
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
    ranking = list(generation.get("ranking") or [])
    if len(ranking) != len(set(ranking)) or not set(ranking) <= set(presented):
        return {**record, "status": "ranking_outside_evidence"}, tasks
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
                                               "j1_task_id": stored_j1["base"]["judge_task_id"]}
    return {**record, "status": "no_observed_sources" if not evidence else "ok",
            "presented": presented, "generator_ranking": ranking,
            "generator_output": generator_output_record(cell["trace"]),
            "j1_task_id": j1["base"]["judge_task_id"], "sources": sources}, tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="DATASET_ROOT:MODEL",
                        help="generator dataset root and model (qwen38 or llama4); repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, help="optional file with one prompt_id per line (a frozen sample)")
    parser.add_argument("--max-tokens", type=int, default=si.DEFAULT_MAX_TOKENS)
    parser.add_argument("--j1-max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, help="at most this many cells per source (development only)")
    parser.add_argument("--truncation-sensitivity-fraction", type=float, default=0.1,
                        help="share of truncated cells whose stored 1,200-character answer is also judged")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; use a new folder")
    prompt_ids = None
    if args.prompt_ids:
        prompt_ids = {line.strip() for line in args.prompt_ids.read_text().splitlines() if line.strip()}
    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    seen: set[str] = set()
    counts: Counter = Counter()
    inputs = []
    with gzip.open(partial / "tasks.jsonl.gz", "wt", encoding="utf-8") as task_file, \
            gzip.open(partial / "cells.jsonl.gz", "wt", encoding="utf-8") as cell_file:
        for spec in args.source:
            root_text, _, model = spec.rpartition(":")
            root = Path(root_text)
            refs, ref_counts = completed_generator_refs(root, model=model, prompt_ids=prompt_ids)
            if args.limit is not None:
                refs = refs[:args.limit]
            inputs.append({"dataset_root": str(root), "model": model, "cell_selection": dict(ref_counts),
                           "cells_used": len(refs)})
            for cell in iter_cells(root, refs):
                record, tasks = build_cell(cell, max_tokens=args.max_tokens, j1_max_tokens=args.j1_max_tokens,
                                           sensitivity_fraction=args.truncation_sensitivity_fraction)
                counts[f"answer_{record.get('answer_source', 'none')}"] += 1
                counts["stored_answer_sensitivity_cells"] += "stored_answer_sensitivity" in record
                counts[f"cells_{record['status']}"] += 1
                counts["source_tasks_referenced"] += sum("judge_task_id" in s for s in record.get("sources", []))
                counts["sources_unassessable"] += sum(s.get("status") == "unassessable_input"
                                                      for s in record.get("sources", []))
                cell_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                for task in tasks:
                    if task["judge_task_id"] not in seen:
                        seen.add(task["judge_task_id"])
                        counts[f"unique_{task['task']}_tasks"] += 1
                        task_file.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": si.PROTOCOL,
                "task_version": si.TASK_VERSION, "retry_contract": si.RETRY_CONTRACT,
                "max_tokens": args.max_tokens, "j1_max_tokens": args.j1_max_tokens, "inputs": inputs,
                "judged_answer": "complete generator answer (trace-recovered when stored truncated)",
                "truncation_sensitivity": {"fraction": args.truncation_sensitivity_fraction,
                                           "salt": SENSITIVITY_SALT},
                "prompt_ids_sha256": hashlib.sha256(args.prompt_ids.read_bytes()).hexdigest()
                if args.prompt_ids else None, "limit": args.limit, "counts": dict(counts),
                "files": {name: hashlib.sha256((partial / name).read_bytes()).hexdigest()
                          for name in ("tasks.jsonl.gz", "cells.jsonl.gz")},
                "git_commit": commit, "scientific_result": False}
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    partial.rename(args.output)
    print(json.dumps({"output": str(args.output), **manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
