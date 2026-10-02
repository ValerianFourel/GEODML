#!/usr/bin/env python3
"""Freeze 48 constructed SI-v4 controls per split, outside the generator corpus.

Expected decisions are design assertions, not labels for real study examples.
No model calls. The two splits are template-related and not population validation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts.run_source_importance_judge import canonical, file_hash


def cases(split):
    names = ("Acme", "Birch") if split == "development" else ("Cedar", "Delta")
    for variant in range(4):
        entity, other = names if variant < 2 else names[::-1]
        answer = f"{entity} exports CSV offline."
        exact = f"{entity} exports CSV offline."
        options = [
            ("exact", answer, exact, "full", [5, 5]),
            ("paraphrase", answer, f"{entity} can export CSV without an internet connection.", "full", [5, 5]),
            ("topic_or_entity", answer, f"{entity} software overview." if variant % 2 == 0 else f"{other} exports CSV offline.", "unsupported", [0, 0]),
            ("contradiction", answer, f"{entity} cannot export CSV offline.", "contradicted", [0, 0]),
            ("qualification", f"{entity} exports CSV offline without a paid subscription.", exact, "partial", [2, 4]),
            ("entity_list", f"{entity} and {other} export CSV offline.", exact, "partial", [2, 4]),
            ("unsupported_advice", f"Choose {entity} for offline CSV export. {entity} exports CSV.", f"{entity} exports CSV.", "background_only", [1, 3]),
            ("equivalent_sources", answer, exact if variant % 2 == 0 else f"{entity} can export CSV without an internet connection.", "full", [5, 5]),
            ("narration", f"The source states that {entity} exports CSV offline.", exact, "attributed_report", [5, 5]),
            ("absence", "None of the supplied sources explains the migration." if variant % 2 == 0 else
             f"None of the supplied sources explains the migration. {entity} exports CSV offline.", exact,
             "global_absence_only" if variant % 2 == 0 else "mixed_absence", None if variant % 2 == 0 else [1, 3]),
            ("distractor_or_removal", answer, ("Today's weather is sunny. " * 80 + exact) if variant % 2 == 0 else
             "Today's weather is sunny.", "full" if variant % 2 == 0 else "unsupported", [5, 5] if variant % 2 == 0 else [0, 0]),
            ("passage_id_rename", answer, exact, "full", [5, 5]),
        ]
        for family, final, text, relation, interval in options:
            cid = f"constructed-{split}-{family}-{variant}"
            yield {"cell_id": cid, "family": family, "variant": variant,
                   "request": "Explain how to implement the migration." if family in ("narration", "absence") else "Which tool exports CSV offline?",
                   "answer": final, "text": text, "expected_relation": relation,
                   "acceptable_grade_range": interval,
                   "diagnostic_passage_ids": {"text1": "text101"} if family == "passage_id_rename" and variant % 2 else None}


def freeze(output, split, *, task_version=v4.LEGACY_TASK_VERSION):
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True)
    unique, expectations, cells = {}, [], []
    for case in cases(split):
        request, answer, text = case["request"], case["answer"], case["text"]
        mapper = v4.prepare_map_task(request=request, answer=answer, task_version=task_version)["record"]
        source = v4.source_dependency(mapper["judge_task_id"], "", text,
                                      diagnostic_passage_ids=case["diagnostic_passage_ids"], diagnostic_seed=20260930)
        j1 = v3.task_record(v3.prepare_fulfilment_task(request=request, answer=answer))
        for task in (mapper, source, j1):
            unique[task["judge_task_id"]] = task
        url = "https://fixture.invalid/" + case["cell_id"]
        source_hash = v4._digest({"title": "", "text": text})
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        expectations.append({**case, "url": url, "source_sha256": source_hash,
                             "masked_answer_sha256": answer_hash,
                             "reference_kind": "constructed_design_assertion_not_human_gold"})
        cells.append({"cell_id": case["cell_id"], "prompt_id": case["cell_id"], "status": "ok",
                      "protocol": v4.PROTOCOL, "model": "constructed", "method": "constructed",
                      "engine": "none", "condition": case["family"], "constructed": True,
                      "judged_answer": answer, "stored_answer": answer,
                      "judged_answer_sha256": answer_hash, "masked_answer_sha256": answer_hash,
                      "map_task_id": mapper["judge_task_id"], "j1_task_id": j1["judge_task_id"],
                      "presented": [url], "generator_ranking": [],
                      "provenance": {"status": "constructed", "attribution_target": "constructed_record"},
                      "sources": [{"url": url, "dependency_id": source["judge_task_id"],
                                   "source_sha256": source_hash, "status": "awaiting_map"}]})
    files = {}
    for name, data in (("tasks", unique.values()), ("cells", cells), ("expectations", expectations)):
        path = partial / f"{name}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in data:
                stream.write(canonical(row) + "\n")
        files[path.name] = file_hash(path)
    manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": v4.PROTOCOL,
                "task_version": task_version, "retry_contract": mapper["retry_contract"],
                "constructed": True, "scientific_result": False, "split": split,
                "diagnostic_source_seed": 20260930,
                "preprocessing": v4.PREPROCESSING_VERSION, "eligibility_version": v4.ELIGIBILITY_VERSION,
                "max_tokens": 4096, "map_max_tokens": 4096, "j1_max_tokens": 64,
                "cells": len(cells), "source_pairs": len(cells), "unique_tasks": len(unique), "files": files}
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partial.rename(output)
    return manifest


def freeze_supplements(output, inputs, cell_id, list_spec):
    """Freeze explicit cyclic list transformations and independent role controls.

    list_spec contains prefix, five item strings (without their numeric markers),
    and suffix. Reconstructing the supplied answer exactly is mandatory. The
    operator identifies independent items; this function never infers that a
    list is semantically reorderable and never consults source records to edit it.
    """
    from analysis.scripts.prepare_si_v4_evaluation import load_freeze, write_jsonl
    _, originals, tasks = load_freeze(inputs)
    found = [cell for cell in originals if cell["cell_id"] == cell_id]
    if len(found) != 1:
        raise ValueError("exactly one original list cell is required")
    original = found[0]
    map_record = tasks[original["map_task_id"]]
    request, answer = (map_record["inputs"][field] for field in ("request", "answer"))
    if set(list_spec) != {"prefix", "items", "suffix", "independent_items_confirmed"}:
        raise ValueError("list specification requires prefix/items/suffix/independent_items_confirmed")
    if list_spec["independent_items_confirmed"] is not True:
        raise ValueError("independent list items must be identified before creating order controls")
    prefix, items, suffix = (list_spec[k] for k in ("prefix", "items", "suffix"))
    if (not isinstance(prefix, str) or not isinstance(suffix, str) or not isinstance(items, list)
            or len(items) != 5 or any(not isinstance(item, str) or not item.strip() for item in items)
            or len(set(items)) != 5):
        raise ValueError("five distinct complete item strings are required")
    render = lambda order: prefix + " ".join(f"{i + 1}) {text}" for i, text in enumerate(order)) + suffix
    if render(items) != answer:
        raise ValueError("list specification does not reconstruct the original masked answer exactly")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    def write_bundle(path, descriptions, *, fixed=False):
        path.mkdir()
        cells, records, expectations, fixed_maps = [], {}, [], []
        for description in descriptions:
            text, question = description["answer"], description["request"]
            mapper = v4.prepare_map_task(request=question, answer=text, diagnostic_seed=20261002,
                                         task_version=v4.TASK_VERSION)["record"]
            j1 = v3.task_record(v3.prepare_fulfilment_task(request=question, answer=text))
            for task in (mapper, j1):
                records[task["judge_task_id"]] = task
            sha = hashlib.sha256(text.encode()).hexdigest()
            source_rows, presented = [], []
            if "source_text" in description:
                source = v4.source_dependency(mapper["judge_task_id"], "", description["source_text"],
                                              diagnostic_seed=20261002)
                records[source["judge_task_id"]] = source
                url = "https://fixture.invalid/" + description["cell_id"]
                presented.append(url)
                source_rows.append({"url": url, "dependency_id": source["judge_task_id"],
                                    "map_task_id": mapper["judge_task_id"], "status": "awaiting_map",
                                    "source_sha256": v4._digest({"title": "", "text": description["source_text"]})})
            cells.append({"cell_id": description["cell_id"], "prompt_id": description["cell_id"],
                          "status": "ok", "protocol": v4.PROTOCOL, "model": "constructed", "method": "constructed",
                          "engine": "none", "condition": description["family"], "constructed": True,
                          "stored_answer": text, "judged_answer": text, "judged_answer_sha256": sha,
                          "masked_answer_sha256": sha, "map_task_id": mapper["judge_task_id"],
                          "j1_task_id": j1["judge_task_id"], "presented": presented, "generator_ranking": [],
                          "provenance": {"status": "constructed", "attribution_target": "constructed_record"},
                          "sources": source_rows})
            expectations.append({**description, "masked_answer_sha256": sha,
                                 "reference_kind": "constructed_design_assertion_not_human_gold"})
            if fixed:
                raw = description["fixed_map"]
                v4.validate_map(raw, answer=text)
                entry = {"judge_task_id": mapper["judge_task_id"], "raw_output": canonical(raw)}
                if entry not in fixed_maps:
                    fixed_maps.append(entry)
        files = {}
        for name, data in (("tasks", records.values()), ("cells", cells), ("expectations", expectations)):
            write_jsonl(path / f"{name}.jsonl.gz", data)
            files[f"{name}.jsonl.gz"] = file_hash(path / f"{name}.jsonl.gz")
        manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": v4.PROTOCOL,
                    "task_version": v4.TASK_VERSION, "retry_contract": v4.RETRY_CONTRACT,
                    "constructed": True, "scientific_result": False, "split": "development-r2-supplement",
                    "preprocessing": v4.PREPROCESSING_VERSION, "eligibility_version": v4.ELIGIBILITY_VERSION,
                    "max_tokens": 4096, "map_max_tokens": 4096, "j1_max_tokens": 64,
                    "cells": len(cells), "source_pairs": sum(len(c["sources"]) for c in cells),
                    "unique_tasks": len(records), "files": files, "diagnostic_map_seed": 20261002,
                    "diagnostic_source_seed": 20261002,
                    "input_manifest_sha256": file_hash(Path(inputs) / "manifest.json"),
                    "independent_map_review": "required_before_using_role_controls" if fixed else "not_applicable"}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_freeze(path)
        if fixed:
            write_jsonl(output / "role-fixed-maps.jsonl", fixed_maps)
        return manifest

    variants = []
    for rotation in range(5):
        order = list(range(rotation, 5)) + list(range(rotation))
        cursor, item_spans = len(prefix), []
        for position, item_index in enumerate(order):
            start = cursor + len(f"{position + 1}) ")
            end = start + len(items[item_index])
            item_spans.append({"item_index": item_index, "position": position,
                               "start": start, "end": end,
                               "text_sha256": hashlib.sha256(items[item_index].encode()).hexdigest()})
            cursor = end + 1
        variants.append({"cell_id": f"constructed-r2-order-{rotation}", "family": "independent_list_order",
                         "variant": rotation, "request": request, "answer": render([items[i] for i in order]),
                         "original_cell_id": cell_id, "original_masked_answer_sha256": original["masked_answer_sha256"],
                         "item_order": order, "item_text_sha256": [hashlib.sha256(item.encode()).hexdigest() for item in items],
                         "item_spans": item_spans,
                         "expected_behavior": "faithful content mapping and substantive-role invariance under cyclic order"})
    order_manifest = write_bundle(output / "order-inputs", variants)

    sentences = ["Choose Acme to migrate your data.",
                 "The migration includes CSV export, field matching, and validation.",
                 "For example, a preview can show the first ten records.",
                 "The preview button is blue."]
    role_answer = " ".join(sentences)
    roles = ("central", "major", "secondary", "peripheral")
    claims = []
    for i, role in enumerate(roles, 1):
        unit = [w for w in v4.words(role_answer) if w["unit_id"] == f"a{i}"]
        claims.append({"spans": [{"first": unit[0]["word_id"], "last": unit[-1]["word_id"]}],
                       "kind": "recommendation" if i == 1 else "assertion", "roles": [role]})
    fixed = {"status": "ready", "claims": claims, "excluded": [], "note": ""}
    intervals = ([4, 4], [3, 4], [2, 2], [1, 1])
    role_cases = [{"cell_id": f"constructed-r2-role-{role}", "family": "fixed_map_role_isolation",
                   "request": "How should I migrate my data?", "answer": role_answer,
                   "source_text": sentences[i], "supported_claim_role": role, "expected_relation": "full",
                   "acceptable_grade_range": intervals[i], "fixed_map": fixed,
                   "independent_map_review": "required; these expectations are design assertions"}
                  for i, role in enumerate(roles)]
    role_manifest = write_bundle(output / "role-inputs", role_cases, fixed=True)
    result = {"order_inputs": str(output / "order-inputs"), "role_inputs": str(output / "role-inputs"),
              "role_fixed_maps": str(output / "role-fixed-maps.jsonl"),
              "order_manifest": order_manifest, "role_manifest": role_manifest}
    (output / "supplements.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"))
    parser.add_argument("--task-version", choices=(v4.LEGACY_TASK_VERSION, v4.TASK_VERSION), default=v4.LEGACY_TASK_VERSION)
    parser.add_argument("--supplement-inputs", type=Path)
    parser.add_argument("--list-cell-id")
    parser.add_argument("--list-spec", type=Path)
    args = parser.parse_args(argv)
    if args.supplement_inputs:
        if not args.list_cell_id or not args.list_spec or args.split:
            parser.error("supplements require --list-cell-id and --list-spec, without --split")
        report = freeze_supplements(args.output, args.supplement_inputs, args.list_cell_id,
                                    json.loads(args.list_spec.read_text()))
    else:
        if not args.split:
            parser.error("--split is required for the original 48-case controls")
        report = freeze(args.output, args.split, task_version=args.task_version)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
