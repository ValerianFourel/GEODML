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


def freeze(output, split):
    output = Path(output)
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True)
    unique, expectations, cells = {}, [], []
    for case in cases(split):
        request, answer, text = case["request"], case["answer"], case["text"]
        mapper = v4.prepare_map_task(request=request, answer=answer)["record"]
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
                "task_version": v4.TASK_VERSION, "retry_contract": v4.RETRY_CONTRACT,
                "constructed": True, "scientific_result": False, "split": split,
                "diagnostic_source_seed": 20260930,
                "preprocessing": v4.PREPROCESSING_VERSION, "eligibility_version": v4.ELIGIBILITY_VERSION,
                "max_tokens": 4096, "map_max_tokens": 4096, "j1_max_tokens": 64,
                "cells": len(cells), "source_pairs": len(cells), "unique_tasks": len(unique), "files": files}
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    partial.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(args.output, args.split), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
