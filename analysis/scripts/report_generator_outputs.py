#!/usr/bin/env python3
"""Describe what the saved Experiment V2 generator datasets contain, as Markdown.

CPU-only and read-only. Counts every registered generator task by its latest
ledger state, reads only the verified generation rows of completed cells (never
traces), joins each prompt to its frozen axis position and writes report.md with
CSV/JSON tables. The axis position is a property of the prompt text, not a
randomized treatment: every axis result is a descriptive association. No judge
(Nemotron/Gemma) results are used, so nothing here is about source importance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_cells import GENERATOR_MODELS, read_references
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows, verify_record_reference
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from analysis.interpretability.pipeline.axis_permutation_metrics import ranking_agreement
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.report_latent_ranking_relationship import _association

AXIS = "axis_1_percentile_0_1"
CELL_OUTCOMES = ("ranking_length", "answer_chars", "search_count", "final_snippet_count",
                 "target_selected_if_seen", "target_reciprocal_rank_if_seen")
PAIR_OUTCOMES = ("top1_change", "top3_distance", "kendall_distance_common", "exact_change")
DESIGN = ("method", "engine", "condition")
CELLS_PER_PROMPT = 12  # full design per generator: 26,009 prompts x 12 = 312,108 cells
LIMITATION = ("Descriptive only. The axis position is a measured property of each prompt, not a randomized "
              "treatment; prompts differ in wording and topic as well as position. Cells of one prompt share "
              "its position, so associations are computed on prompt means. Permutation p-values shuffle "
              "positions within keyword and are not corrected for the many outcomes reported.")


def log(phase, **fields):
    print(json.dumps({"phase": phase, **fields}), file=sys.stderr, flush=True)


def load_axis(path):
    path = Path(path).resolve()
    positions = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, value = row.get("candidate_id"), row.get(AXIS)
            if not isinstance(prompt, str) or prompt in positions:
                raise ValueError(f"missing or duplicate candidate_id at line {number}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{AXIS} must lie in [0, 1] at line {number}")
            positions[prompt] = float(value)
    if not positions:
        raise ValueError("axis map is empty")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return positions, {"path": str(path), "sha256": digest, "prompts": len(positions)}


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def compact_cell(task, generation, membership):
    ranking = [url for url in generation.get("ranking") or [] if isinstance(url, str) and url]
    audit = generation.get("condition_audit") or {}
    target, seen = audit.get("target_url"), audit.get("target_observed") is True
    rank = ranking.index(target) + 1 if target in ranking else None
    return {"model": task["model"], "prompt_id": task["prompt_id"],
            "keyword_id": membership.get("primary_keyword_id", "unassigned"),
            **{field: task.get(field) for field in DESIGN},
            "ranking": ranking, "ranking_length": len(ranking),
            "answer_chars": len(generation.get("answer") or ""),
            "search_count": _number(generation.get("search_count")),
            "final_snippet_count": _number(generation.get("final_snippet_count")),
            "target_selected_if_seen": float(rank is not None) if seen else None,
            "target_reciprocal_rank_if_seen": (1.0 / rank if rank else 0.0) if seen else None}


def load_cells(root, model, *, stripes=256, batch=5000):
    """Compact verified completed cells and the latest ledger state of every registered task."""
    root = Path(root)
    if model not in GENERATOR_MODELS:
        raise ValueError(f"unknown generator model: {model}")
    memberships = {row["prompt_id"]: row for row in iter_sealed_rows(root, "keyword_memberships")}
    tasks = {}
    for row in iter_sealed_rows(root, "task_definitions", required=True):
        if row.get("model") == model and row.get("stage", "generation") == "generation":
            tasks[identity_fingerprint(ClaimIdentity(**row["claim_identity"]))] = row
    log("tasks", model=model, registered=len(tasks))
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
    states, anomalies, completed = Counter(), Counter(), []
    for fingerprint, task in tasks.items():
        event = latest.get(fingerprint)
        state = event.get("state", "unknown") if isinstance(event, dict) else "never_claimed"
        states[state] += 1
        if state != "completed":
            continue
        generations = [r for r in event.get("record_references") or [] if r.get("table") == "generations"]
        if len(generations) != 1:
            anomalies["completed_without_exactly_one_generation"] += 1
        elif not verify_record_reference(root, generations[0]):
            anomalies["completed_with_unverified_generation"] += 1
        else:
            completed.append((task, generations[0]))
    completed.sort(key=lambda item: (item[1]["writer_id"], item[1]["shard_sequence"], item[1]["line_number"]))
    cells = []
    for start in range(0, len(completed), batch):
        chunk = completed[start:start + batch]
        rows = read_references(root, [reference for _, reference in chunk])
        for task, reference in chunk:
            generation = rows[reference["record_id"]]
            if generation.get("cell_id") != task.get("task_id"):
                raise ValueError(f"generation cell differs from its task: {task.get('task_id')}")
            cells.append(compact_cell(task, generation, memberships.get(task["prompt_id"], {})))
        log("generations", model=model, read=len(cells), total=len(completed))
    return cells, {"registered_tasks": len(tasks), "latest_states": dict(sorted(states.items())),
                   "anomalies": dict(sorted(anomalies.items())), "verified_cells": len(cells)}


def compare(cells, *, side, sides, key):
    """Ranking differences between matched cells that differ only in `side`."""
    groups = defaultdict(dict)
    for cell in cells:
        if cell[side] in sides:
            groups[tuple(cell[k] for k in key)][cell[side]] = cell
    rows = []
    for identity, matched in sorted(groups.items()):
        if set(matched) != set(sides):
            continue
        left, right = matched[sides[0]], matched[sides[1]]
        if not left["ranking"] or not right["ranking"]:
            continue
        agreement = ranking_agreement(left["ranking"], right["ranking"])
        tau = agreement["kendall_tau_common"]
        rows.append({**dict(zip(key, identity)), "keyword_id": left["keyword_id"],
                     **({AXIS: left[AXIS]} if AXIS in left else {}),
                     "top1_change": 1.0 - float(agreement["top1_match"]),
                     "top3_distance": 1.0 - agreement["topk_overlap"],
                     "kendall_distance_common": None if tau is None else (1.0 - tau) / 2.0,
                     "exact_change": 1.0 - float(agreement["exact_match"])})
    return rows


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def describe(cells):
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["model"], *(str(cell[f]) for f in DESIGN))].append(cell)
    rows = []
    for (model, *design), group in sorted(groups.items()):
        seen = [c for c in group if c["target_selected_if_seen"] is not None]
        rows.append({"model": model, **dict(zip(DESIGN, design)), "cells": len(group),
                     "prompts": len({c["prompt_id"] for c in group}),
                     "mean_ranking_length": _mean([c["ranking_length"] for c in group]),
                     "empty_ranking_share": _mean([float(c["ranking_length"] == 0) for c in group]),
                     "median_answer_chars": _median([c["answer_chars"] for c in group]),
                     "mean_search_count": _mean([c["search_count"] for c in group]),
                     "mean_final_snippet_count": _mean([c["final_snippet_count"] for c in group]),
                     "target_seen_cells": len(seen),
                     "target_selected_if_seen": _mean([c["target_selected_if_seen"] for c in seen]),
                     "target_mrr_if_seen": _mean([c["target_reciprocal_rank_if_seen"] for c in seen])})
    return rows


def deciles(rows, outcomes, strata):
    groups = defaultdict(list)
    for row in rows:
        if row.get(AXIS) is not None:
            groups[(*(str(row[s]) for s in strata), min(9, int(row[AXIS] * 10)))].append(row)
    result = []
    for (*identity, decile), group in sorted(groups.items()):
        result.append({**dict(zip(strata, identity)), "decile": decile + 1,
                       "axis_range": f"{decile / 10:.1f}-{(decile + 1) / 10:.1f}", "rows": len(group),
                       "prompts": len({r["prompt_id"] for r in group}),
                       **{outcome: _mean([r[outcome] for r in group]) for outcome in outcomes}})
    return result


def associations(rows, outcomes, strata, *, scope, permutations, seed):
    """Spearman association of prompt-mean outcomes with axis position, keyword-blocked permutation."""
    result = []
    groups = defaultdict(list)
    for row in rows:
        if row.get(AXIS) is not None:
            groups[tuple(str(row[s]) for s in strata)].append(row)
    for identity, group in sorted(groups.items()):
        label = {**dict(zip(strata, identity)), "scope": scope}
        for outcome in outcomes:
            by_prompt = defaultdict(list)
            for row in group:
                if row[outcome] is not None:
                    by_prompt[(row["prompt_id"], row["keyword_id"], row[AXIS])].append(row[outcome])
            prompts = [{"prompt_id": p, "keyword_id": k, AXIS: x, outcome: statistics.fmean(v)}
                       for (p, k, x), v in sorted(by_prompt.items())]
            result.append(_association(prompts, coordinate=AXIS, coordinate_role="observed_prompt_property",
                                       outcome=outcome, permutations=permutations, seed=seed, identity=label))
    return result


def _cell(value):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_cell(v)}" for k, v in value.items()) or "none"
    return str(value)


def table(rows, columns):
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def headline(inventory, comparisons, links):
    bullets = []
    for model, entry in inventory.items():
        expected = entry.get("expected_cells")
        share = f" ({entry['verified_cells'] / expected:.1%} of {expected:,} design cells)" if expected else ""
        bullets.append(f"**{model}**: {entry['verified_cells']:,} verified completed cells{share}, "
                       f"{entry['prompts']:,} prompts, {entry['keywords']:,} keywords.")
    for name, rows in comparisons.items():
        if rows:
            bullets.append(f"**{name}**: {len(rows):,} matched pairs; top-1 differs in "
                           f"{_mean([r['top1_change'] for r in rows]):.1%}, top-3 distance "
                           f"{_mean([r['top3_distance'] for r in rows]):.3f}.")
    strongest = sorted((r for r in links if r.get("spearman_rho") is not None),
                       key=lambda r: -abs(r["spearman_rho"]))[:5]
    for r in strongest:
        where = ", ".join(f"{k}={r[k]}" for k in ("model", "comparison") if k in r)
        bullets.append(f"Axis vs `{r['outcome']}` ({where}): Spearman ρ = {r['spearman_rho']:+.3f} over "
                       f"{r['n']:,} prompts, blocked-permutation p = {_cell(r['blocked_permutation_p_two_sided'])}.")
    return "\n".join("- " + b for b in bullets) + "\n"


def render(summary, tables):
    out = ["# Experiment V2 generator outputs: what the saved datasets show\n",
           f"_Generated by `analysis/scripts/report_generator_outputs.py` at commit `{summary['git_commit']}`. "
           "Saved generator outputs only; no judge results._\n",
           "> " + LIMITATION + "\n", "## Headline numbers\n", summary["headline"],
           "## 1. Inventory\n", table(tables["inventory"], ["model", "registered_tasks", "expected_cells",
                                       "verified_cells", "prompts", "keywords", "prompts_without_axis",
                                       "latest_states", "anomalies"]),
           "Latest ledger states are counted for every registered task; only `completed` cells with a "
           "verified generation record are analysed.\n",
           "## 2. Outputs by design cell\n",
           table(tables["descriptives"], ["model", *DESIGN, "cells", "mean_ranking_length", "empty_ranking_share",
                                          "median_answer_chars", "mean_search_count", "mean_final_snippet_count",
                                          "target_selected_if_seen", "target_mrr_if_seen"]),
           "`answer_chars` is the stored answer, which the dataset caps (about 1,200 characters); "
           "target columns use only cells whose condition audit observed the target URL.\n",
           "## 3. Ranking differences between matched cells\n"]
    for name, rows in tables["comparison_summaries"].items():
        out += [f"### {name}\n", table(rows, ["group", "pairs", *PAIR_OUTCOMES])]
    if summary["axis"]:
        out += ["## 4. Along the information-seeking → action-readiness axis\n",
                f"Axis map `{summary['axis']['path']}` (sha256 `{summary['axis']['sha256'][:12]}…`, "
                f"{summary['axis']['prompts']:,} prompts). Deciles of `{AXIS}`.\n",
                "### Outputs by axis decile\n",
                table(tables["cell_deciles"], ["model", "decile", "axis_range", "prompts", *CELL_OUTCOMES]),
                "### Ranking differences by axis decile\n",
                table(tables["pair_deciles"], ["comparison", "decile", "axis_range", "prompts", *PAIR_OUTCOMES]),
                "### Associations with axis position (prompt means)\n",
                table([r for r in tables["associations"] if r["scope"] != "one design cell"],
                      ["scope", "model", "comparison", "outcome", "n", "spearman_rho",
                       "linear_slope_per_axis_unit", "blocked_permutation_p_two_sided", "permutable_keywords"]),
                "Per design cell (model × method × engine × condition) associations are in `associations.csv`.\n"]
    else:
        out += ["## 4. Along the axis\n", "_Not computed: no `--axis-map` was given._\n"]
    out += ["## Provenance\n", "```json\n" + json.dumps(summary["provenance"], indent=2) + "\n```\n"]
    return "\n".join(out)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            fields = sorted({k for row in rows for k in row}, key=lambda k: list(rows[0]).index(k)
                            if k in rows[0] else len(rows[0]))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def build(datasets, output, *, axis_map=None, permutations=200, seed=20261004, stripes=256):
    output = Path(output)
    if output.exists():
        raise ValueError("output directory already exists; use a new path")
    positions, axis = load_axis(axis_map) if axis_map else ({}, None)
    cells, inventory = [], {}
    for model, root in datasets.items():
        model_cells, entry = load_cells(root, model, stripes=stripes)
        for cell in model_cells:
            if cell["prompt_id"] in positions:
                cell[AXIS] = positions[cell["prompt_id"]]
        prompts = {c["prompt_id"] for c in model_cells}
        entry.update(model=model, dataset_root=str(Path(root).resolve()), prompts=len(prompts),
                     keywords=len({c["keyword_id"] for c in model_cells}),
                     expected_cells=len(positions) * CELLS_PER_PROMPT if positions else None,
                     prompts_without_axis=len(prompts - set(positions)) if positions else None)
        inventory[model] = entry
        cells += model_cells
    conditions = {c["condition"] for c in cells}
    comparisons = {}
    if {"natural", "shuffled"} <= conditions:
        comparisons["natural vs shuffled evidence order"] = compare(
            cells, side="condition", sides=("natural", "shuffled"), key=("model", "prompt_id", "method", "engine"))
    if set(GENERATOR_MODELS) <= set(datasets):
        comparisons["llama4 vs qwen38 on the same cell"] = compare(
            cells, side="model", sides=("llama4", "qwen38"), key=("prompt_id", *DESIGN))
    pairs = [{**row, "comparison": name} for name, rows in comparisons.items() for row in rows]
    summaries = {}
    for name, rows in comparisons.items():
        field = "model" if "model" in (rows[0] if rows else {}) else "condition"
        grouped = defaultdict(list)
        for row in rows:
            grouped["all"].append(row)
            grouped[f"{field}={row[field]}"].append(row)
            grouped[f"method={row['method']}"].append(row)
        summaries[name] = [{"group": group, "pairs": len(items),
                            **{o: _mean([r[o] for r in items]) for o in PAIR_OUTCOMES}}
                           for group, items in sorted(grouped.items(), key=lambda kv: (kv[0] != "all", kv[0]))]
    tables = {"inventory": list(inventory.values()), "descriptives": describe(cells),
              "comparison_summaries": summaries, "cell_deciles": [], "pair_deciles": [], "associations": []}
    if positions:
        log("associations", cells=len(cells), pairs=len(pairs), permutations=permutations)
        tables["cell_deciles"] = deciles(cells, CELL_OUTCOMES, ("model",))
        tables["pair_deciles"] = deciles(pairs, PAIR_OUTCOMES, ("comparison",))
        tables["associations"] = (
            associations(cells, CELL_OUTCOMES, ("model",), scope="all cells of the model",
                         permutations=permutations, seed=seed)
            + associations(cells, CELL_OUTCOMES, ("model", *DESIGN), scope="one design cell",
                           permutations=permutations, seed=seed)
            + associations(pairs, PAIR_OUTCOMES, ("comparison",), scope="matched pairs",
                           permutations=permutations, seed=seed))
    commit = _git_commit()
    summary = {"format_version": "geodml-generator-output-report-v1", "scientific_result": False,
               "git_commit": commit, "axis": axis, "limitation": LIMITATION,
               "headline": headline(inventory, comparisons, tables["associations"]),
               "provenance": {"git_commit": commit, "datasets": {m: e["dataset_root"] for m, e in inventory.items()},
                              "axis_map": axis, "permutations": permutations, "seed": seed,
                              "stripes": stripes, "reads": "verified generation rows only; traces not read"}}
    output.mkdir(parents=True)
    (output / "report.md").write_text(render(summary, tables), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({**summary, "inventory": inventory,
        "comparison_summaries": summaries, "associations": tables["associations"]},
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("descriptives", "cell_deciles", "pair_deciles", "associations"):
        write_csv(output / f"{name.replace('_', '-')}.csv", tables[name])
    write_csv(output / "matched-pairs.csv", pairs)
    return summary


def _git_commit():
    import subprocess
    result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() or None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="MODEL=DATASET_ROOT",
                        help="qwen38=... and/or llama4=...; repeatable")
    parser.add_argument("--axis-map", type=Path, help="final-audit/final-axis-map.jsonl (26,009 prompts)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--stripes", type=int, default=256)
    args = parser.parse_args(argv)
    datasets = {}
    for item in args.dataset:
        model, sep, root = item.partition("=")
        if not sep or model not in GENERATOR_MODELS or model in datasets:
            parser.error("each --dataset must be qwen38=ROOT or llama4=ROOT, once per model")
        datasets[model] = Path(root)
    if args.permutations < 0:
        parser.error("permutations must be non-negative")
    build(datasets, args.output_dir, axis_map=args.axis_map, permutations=args.permutations,
          seed=args.seed, stripes=args.stripes)
    print("REPORT=" + str(args.output_dir / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
