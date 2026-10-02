"""Evaluation preparation preserves frozen identities and independent review inputs."""
from collections import Counter
import hashlib
import json

import pytest

from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts import prepare_si_v4_evaluation as prep
from analysis.scripts.prepare_si_v4_diagnostics import freeze_supplements


def population(root, prompts=61, *, answer=None, keyword_prefix="fresh", task_version=v4.TASK_VERSION):
    cells, tasks = [], {}
    for i in range(prompts):
        for model in ("qwen38", "llama4"):
            text = answer or ("Acme " + "reliably " * i + "exports CSV offline.")
            request = f"Which tool exports CSV offline for project {i}?"
            mapper = v4.prepare_map_task(request=request, answer=text, task_version=task_version)["record"]
            j1 = v3.task_record(v3.prepare_fulfilment_task(request=request, answer=text))
            source = v4.source_dependency(mapper["judge_task_id"], "Product record", text)
            tasks.update({r["judge_task_id"]: r for r in (mapper, j1, source)})
            sha = hashlib.sha256(text.encode()).hexdigest()
            cells.append({"cell_id": f"{model}-{i}", "prompt_id": f"prompt-{i}", "model": model,
                          "status": "ok", "method": "method", "engine": "engine", "condition": "full",
                          "judged_answer": text, "stored_answer": text, "judged_answer_sha256": sha,
                          "masked_answer_sha256": sha, "map_task_id": mapper["judge_task_id"],
                          "j1_task_id": j1["judge_task_id"], "provenance": {"status": "verified"},
                          "keyword_memberships": {"keyword_ids": [f"{keyword_prefix}-{i}"]},
                          "prompt_metadata": {"measured_readiness": (i % 3) / 2},
                          "sources": [{"url": "https://example.invalid/product", "dependency_id": source["judge_task_id"],
                                       "map_task_id": mapper["judge_task_id"],
                                       "source_sha256": prep.digest({"title": "Product record", "text": text})}]})
    root.mkdir()
    for name, values in (("cells", cells), ("tasks", tasks.values())):
        prep.write_jsonl(root / f"{name}.jsonl.gz", values)
    manifest = {"format_version": "source-importance-task-freeze-v1", "protocol": v4.PROTOCOL,
                "task_version": task_version, "files": {name: prep.file_hash(root / name)
                 for name in ("cells.jsonl.gz", "tasks.jsonl.gz")}, "cells": len(cells)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_fresh_selection_is_paired_excluded_and_never_silently_shrinks(tmp_path):
    inputs = population(tmp_path / "population")
    excluded = tmp_path / "development"
    prep.subset_freeze(inputs, excluded, ["qwen38-0", "llama4-0"])
    # The historical development export lacks memberships. Recover its exact
    # prompt membership from this same verified population instead of guessing.
    prior_manifest, prior_cells, _ = prep.load_freeze(excluded)
    for cell in prior_cells:
        cell.pop("keyword_memberships")
    prep.write_jsonl(excluded / "cells.jsonl.gz", prior_cells)
    prior_manifest["files"]["cells.jsonl.gz"] = prep.file_hash(excluded / "cells.jsonl.gz")
    (excluded / "manifest.json").write_text(json.dumps(prior_manifest))
    report = prep.select_fresh(inputs, [excluded], tmp_path / "fresh")
    _, cells, tasks = prep.load_freeze(tmp_path / "fresh/inputs")
    assert report["cells"] == 120
    assert set(report["stress_prompt_ids"]) == {f"prompt-{i}" for i in range(49, 61)}
    assert "prompt-0" not in {c["prompt_id"] for c in cells}
    assert all(n == 2 for n in Counter(c["prompt_id"] for c in cells).values())
    assert Counter(c["model"] for c in cells) == {"qwen38": 60, "llama4": 60}
    original_tasks = prep.load_freeze(inputs)[2]
    assert all(original_tasks[key] == task for key, task in tasks.items())
    repeat = json.loads((tmp_path / "fresh/repeats/selection.json").read_text())
    assert len(repeat["selected_cell_ids"]) == 24
    assert repeat["by_generator"] == {"qwen38": 12, "llama4": 12}
    too_few = tmp_path / "too-few"
    prep.subset_freeze(inputs, too_few, [c["cell_id"] for c in prep.load_freeze(inputs)[1]
                                       if c["prompt_id"] != "prompt-1"])
    with pytest.raises(ValueError, match="60 verified paired prompts"):
        prep.select_fresh(too_few, [excluded], tmp_path / "must-not-shrink")
    assert not (tmp_path / "must-not-shrink").exists()
    legacy = population(tmp_path / "historical-v4", task_version=v4.LEGACY_TASK_VERSION)
    with pytest.raises(ValueError, match="historical versions remain separate"):
        prep.select_fresh(legacy, [excluded], tmp_path / "not-revised")


def test_reference_packets_are_blind_and_bind_complete_input_bytes(tmp_path):
    inputs = population(tmp_path / "population", 1)
    _, cells, _ = prep.load_freeze(inputs)
    # Input metadata is deliberately poisonous: only original inputs enter grade packets.
    cells[0].update(answer_map="SECRET_MAP", grades={"x": 5}, model="SECRET_GENERATOR")
    prep.write_jsonl(inputs / "cells.jsonl.gz", cells)
    manifest = json.loads((inputs / "manifest.json").read_text())
    manifest["files"]["cells.jsonl.gz"] = prep.file_hash(inputs / "cells.jsonl.gz")
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    report = prep.prepare_packets(inputs, tmp_path / "grade")
    packets = list(prep.rows(tmp_path / "grade/packets.jsonl"))
    assert report["packets"] == 2
    for packet in packets:
        assert set(packet["body"]) == {"request", "answer", "source"}
        assert packet["body"]["answer"] == "Acme exports CSV offline."
        assert packet["body"]["source"] == {"title": "Product record", "text": "Acme exports CSV offline."}
        sha = packet.pop("packet_sha256")
        assert prep.digest(packet) == sha
        assert "SECRET_" not in json.dumps(packet)
    # A altered frozen record may not become a apparently independent reference input.
    with (inputs / "tasks.jsonl.gz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prep.prepare_packets(inputs, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_order_controls_preserve_every_item_and_fixed_role_maps_import(tmp_path):
    items = [f"Factor {i} preserves condition {i}." for i in range(5)]
    prefix, suffix = "Independent factors include: ", " These factors all matter."
    answer = prefix + " ".join(f"{i + 1}) {item}" for i, item in enumerate(items)) + suffix
    inputs = population(tmp_path / "population", 1, answer=answer)
    spec = {"prefix": prefix, "items": items, "suffix": suffix, "independent_items_confirmed": True}
    supplement = freeze_supplements(tmp_path / "controls", inputs, "qwen38-0", spec)
    _, cells, tasks = prep.load_freeze(supplement["order_inputs"])
    expectations = list(prep.rows(tmp_path / "controls/order-inputs/expectations.jsonl.gz"))
    assert len(cells) == 5
    for cell, expectation in zip(cells, expectations):
        assert cell["judged_answer"].startswith(prefix) and cell["judged_answer"].endswith(suffix)
        assert all(cell["judged_answer"].count(item) == 1 for item in items)
        assert tasks[cell["map_task_id"]]["seed"] == 20261002
        assert sorted(expectation["item_order"]) == list(range(5))
    for item in range(5):
        assert {e["item_order"].index(item) for e in expectations} == set(range(5))
    _, role_cells, role_tasks = prep.load_freeze(supplement["role_inputs"])
    fixed = list(prep.rows(supplement["role_fixed_maps"]))
    assert len(role_cells) == 4 and len(fixed) == 1
    parsed = v4.reproduce_map_result(role_tasks[fixed[0]["judge_task_id"]], fixed[0])
    assert [claim["roles"] for claim in parsed["claims"]] == [[r] for r in ("central", "major", "secondary", "peripheral")]
    map_review = prep.prepare_packets(supplement["role_inputs"], tmp_path / "map-review", phase="map",
                                      candidate=supplement["role_fixed_maps"])
    assert map_review["packets"] == 4
    for packet in prep.rows(tmp_path / "map-review/packets.jsonl"):
        assert "source" not in packet["body"]
        assert "candidate_output" not in packet["body"]
    with pytest.raises(ValueError, match="reconstruct"):
        freeze_supplements(tmp_path / "changed", inputs, "qwen38-0", {**spec, "suffix": ""})
    assert not (tmp_path / "changed").exists()


def test_support_audit_requires_frozen_blind_grades_and_validates_saved_output(tmp_path):
    inputs = population(tmp_path / "population", 1)
    _, cells, tasks = prep.load_freeze(inputs)
    cell = cells[0]
    mapper = tasks[cell["map_task_id"]]
    raw_map = {"status": "ready", "claims": [{"spans": [{"first": "a1w1", "last": "a1w4"}],
                "kind": "assertion", "roles": ["central"]}], "excluded": [], "note": ""}
    mapped = v4.validate_map(raw_map, answer=mapper["inputs"]["answer"])
    raw = {"status": "scored", "findings": [{"claim_id": "c1", "relation": "full",
           "spans": [{"first": "a1w1", "last": "a1w4"}], "evidence_ids": ["text1"], "note": ""}],
           "importance": 5, "note": "Entire substantive answer is supported."}
    raw_text = json.dumps(raw)
    dep = tasks[cell["sources"][0]["dependency_id"]]
    parsed = v4.materialize_source(dep, mapper, mapped)["validator"](raw_text)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    result_cells = []
    for original in cells:
        result_cells.append({**original, "sources": [{**original["sources"][0], "status": "scored",
                            "parsed_output": parsed, "raw_output": raw_text,
                            "raw_output_sha256": hashlib.sha256(raw_text.encode()).hexdigest()}]})
    prep.write_jsonl(candidate / "cells.jsonl.gz", result_cells)
    prep.write_jsonl(candidate / "maps.jsonl", [{"judge_task_id": mapper["judge_task_id"], "raw_output": json.dumps(raw_map)}])
    with pytest.raises(ValueError, match="freeze independent"):
        prep.prepare_packets(inputs, tmp_path / "premature", phase="support", candidate=candidate)
    assert not (tmp_path / "premature").exists()
    grades = []
    for original in cells:
        for reviewer in ("gpt-6-astra", "gpt-6-sol"):
            grades.append({"record_type": "grade", "cell_id": original["cell_id"],
                           "url": original["sources"][0]["url"], "reviewer_model": reviewer,
                           "masked_answer_sha256": original["masked_answer_sha256"],
                           "source_sha256": original["sources"][0]["source_sha256"],
                           "frozen_before_grade_review": True, "whole_source_checked": True,
                           "acceptable_grade_range": [5, 5]})
    grade_path = tmp_path / "grades.jsonl"
    prep.write_jsonl(grade_path, grades)
    report = prep.prepare_packets(inputs, tmp_path / "support", phase="support", candidate=candidate,
                                  frozen_grades=grade_path)
    assert report["packets"] == 2
    packet = next(prep.rows(tmp_path / "support/packets.jsonl"))
    assert packet["body"]["findings"][0]["evidence"][0]["text"] == "Acme exports CSV offline."
    assert len(packet["body"]["findings"][0]["finding_id"]) == 64
    result_cells[0]["sources"][0]["raw_output"] = raw_text + " "
    prep.write_jsonl(candidate / "cells.jsonl.gz", result_cells)
    with pytest.raises(ValueError, match="does not bind"):
        prep.prepare_packets(inputs, tmp_path / "tampered", phase="support", candidate=candidate,
                             frozen_grades=grade_path)


def test_supplements_keep_missing_cases_and_detect_position_related_role_changes(tmp_path):
    items = [f"Factor {i} preserves condition {i}." for i in range(5)]
    spec = {"prefix": "Factors include: ", "items": items, "suffix": " These factors all matter.",
            "independent_items_confirmed": True}
    answer = spec["prefix"] + " ".join(f"{i+1}) {item}" for i, item in enumerate(items)) + spec["suffix"]
    original = population(tmp_path / "original", 1, answer=answer)
    supplementary = freeze_supplements(tmp_path / "controls", original, "qwen38-0", spec)
    order_inputs, role_inputs = supplementary["order_inputs"], supplementary["role_inputs"]
    missing = prep.summarize_supplements(order_inputs, None, role_inputs, None)
    assert missing["status"] == "incomplete"
    assert len(missing["order_variants"]) == 5 and len(missing["role_controls"]) == 4
    assert missing["counts"]["order_maps_produced"] == 0

    def report(path, inputs, *, first_is_central=False):
        path.mkdir()
        _, cells, tasks = prep.load_freeze(inputs)
        fixed = {r["judge_task_id"]: r for r in prep.rows(supplementary["role_fixed_maps"])}
        maps = {}
        for cell in cells:
            mapper = tasks[cell["map_task_id"]]
            if cell["map_task_id"] in fixed:
                entry = fixed[cell["map_task_id"]]
                mapped = v4.reproduce_map_result(mapper, entry)
            else:
                units = v3.unitize_answer(mapper["inputs"]["answer"])
                claims = []
                for i, unit in enumerate(units):
                    words = [w for w in v4.words(mapper["inputs"]["answer"]) if w["unit_id"] == unit["unit_id"]]
                    role = "peripheral" if i == len(units) - 1 else "central" if i == 0 and first_is_central else "major"
                    claims.append({"spans": [{"first": words[0]["word_id"], "last": words[-1]["word_id"]}],
                                   "kind": "assertion", "roles": [role]})
                raw_map = {"status": "ready", "claims": claims, "excluded": [], "note": ""}
                entry = {"judge_task_id": mapper["judge_task_id"], "raw_output": json.dumps(raw_map)}
                mapped = v4.reproduce_map_result(mapper, entry)
            maps[cell["map_task_id"]] = entry
            cell["map_sha256"] = mapped["map_sha256"]
            for source in cell["sources"]:
                dep = tasks[source["dependency_id"]]
                index = {"central": 0, "major": 1, "secondary": 2, "peripheral": 3}[cell["cell_id"].split("-")[-1]]
                claim = mapped["claims"][index]
                raw = {"status": "scored", "findings": [{"claim_id": claim["claim_id"], "relation": "full",
                       "spans": [{k: s[k] for k in ("first", "last")} for s in claim["spans"]],
                       "evidence_ids": ["text1"], "note": ""}], "importance": [4, 3, 2, 1][index],
                       "note": "The source supports the isolated constructed claim."}
                raw_text = json.dumps(raw)
                parsed = v4.materialize_source(dep, mapper, mapped)["validator"](raw_text)
                source.update(status="scored", importance=raw["importance"], raw_output=raw_text, parsed_output=parsed,
                              raw_output_sha256=hashlib.sha256(raw_text.encode()).hexdigest())
        prep.write_jsonl(path / "cells.jsonl.gz", cells)
        prep.write_jsonl(path / "maps.jsonl", maps.values())
        (path / "summary.json").write_text(json.dumps({"manifest_sha256": prep.file_hash(Path(inputs) / "manifest.json")}))

    from pathlib import Path
    order_report, role_report = tmp_path / "orders", tmp_path / "roles"
    report(order_report, order_inputs)
    report(role_report, role_inputs)

    def references(order_path, destination):
        destination.mkdir()
        prep.prepare_packets(order_inputs, destination / "order", phase="map", candidate=order_path)
        prep.prepare_packets(role_inputs, destination / "role", phase="map", candidate=role_report)
        reviews = []
        for kind in ("order", "role"):
            for packet in prep.rows(destination / kind / "packets.jsonl"):
                for reviewer in ("gpt-6-astra", "gpt-6-sol"):
                    reviews.append({**{k: packet[k] for k in ("record_type", "cell_id", "map_sha256", "masked_answer_sha256", "packet_sha256")},
                                    "reviewer_model": reviewer, "assessment": {"faithful": True, "importance_roles_faithful": True,
                                    "essential_omission": False, "meaning_reversal": False}})
        prep.write_jsonl(destination / "references.jsonl", reviews)
        return destination / "references.jsonl"

    # Matching constructed grades alone do not establish semantic acceptance.
    unreviewed = prep.summarize_supplements(order_inputs, order_report, role_inputs, role_report)
    assert unreviewed["status"] == "needs_review"
    refs = references(order_report, tmp_path / "reviewed")
    passing = prep.summarize_supplements(order_inputs, order_report, role_inputs, role_report,
                                         references=refs, reference_packets=tmp_path / "reviewed")
    assert passing["status"] == "pass"
    changed_report = tmp_path / "position-dependent"
    report(changed_report, order_inputs, first_is_central=True)
    changed_refs = references(changed_report, tmp_path / "changed-reviews")
    changed = prep.summarize_supplements(order_inputs, changed_report, role_inputs, role_report,
                                        references=changed_refs, reference_packets=tmp_path / "changed-reviews")
    assert changed["status"] == "needs_review"
    assert len(changed["semantic_item_role_comparison"]) == 5
    assert all(item["assignment_changed"] for item in changed["semantic_item_role_comparison"])
