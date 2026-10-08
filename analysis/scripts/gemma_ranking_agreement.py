#!/usr/bin/env python3
"""Does Gemma's judgment of source importance match the generator's own ranking? Read-only, diagnostic.

For every Gemma SI-v4 run root given, reads the latest finished report of each shard (cells.jsonl.gz) and pools the
per-answer alignment measures the judge already stored (source_importance.cell_metrics, v4 reporting fix):
  top_source_alignment       the generator's first listed source is in Gemma's top importance group
  generator_chance_baseline  the same if the first source were a random listed one
  first_presented_alignment  the first shown source is in Gemma's top group (the shown-order baseline)
  presented_chance_baseline  the same for a random shown source
  ordering_tau_b             Kendall tau-b between the generator's list order and Gemma's grades
  presented_order_tau_b      Kendall tau-b between the generator's list order and the shown order
  important_source_omission  share of grade 4-5 sources the generator left out of its list
Only complete cells (every shown source graded) enter the means. An answer judged in two runs counts once (first
run given wins). Grouped by model, model x method, model x engine, model x condition. Intervals: 95% percentile
bootstrap over prompts (200 draws, seed 20261008). Writes agreement.json and agreement.md. Semantic acceptance of the
judge is not established: these are diagnostic numbers, not scientific results.

    python -m analysis.scripts.gemma_ranking_agreement --run ROOT [--run ROOT ...] --output DIR
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DONE = ("finished", "finished_with_failures")
MEASURES = ("top_source_alignment", "generator_chance_baseline", "first_presented_alignment", "presented_chance_baseline",
            "ordering_tau_b", "presented_order_tau_b", "important_source_omission", "coverage")
GROUPS = (("model",), ("model", "method"), ("model", "engine"), ("model", "condition"))


def finished_reports(root: Path):
    plan = json.loads((root / "plan.json").read_text())
    for shard in plan["shards"]:
        reports = Path(shard["directory"]) / "results/reports"
        if not (reports / "latest.json").is_file():
            yield shard["id"], None, "not_started"
            continue
        report = reports / json.loads((reports / "latest.json").read_text())["directory"]
        status = json.loads((report / "summary.json").read_text()).get("status")
        yield shard["id"], report if status in DONE else None, status


def collect(roots: list[Path]) -> tuple[list[dict], dict]:
    rows, seen, audit = [], set(), {}
    for root in roots:
        a = audit.setdefault(root.name, {"shards": collections.Counter(), "cells": collections.Counter(),
                                         "sources": collections.Counter(), "duplicates_skipped": 0})
        for _, report, status in finished_reports(root):
            a["shards"][status] += 1
            if report is None:
                continue
            with gzip.open(report / "cells.jsonl.gz", "rt", encoding="utf-8") as stream:
                for line in stream:
                    cell = json.loads(line)
                    if cell.get("status") != "ok":
                        a["cells"][f"input_{cell.get('status')}"] += 1
                        continue
                    for s in cell.get("sources", []):
                        a["sources"][s.get("status")] += 1
                    m = cell.get("metrics") or {}
                    a["cells"]["complete" if m.get("complete") else "incomplete"] += 1
                    if cell["fingerprint"] in seen:
                        a["duplicates_skipped"] += 1
                        continue
                    seen.add(cell["fingerprint"])
                    if m.get("complete"):
                        rows.append({**{k: str(cell.get(k)) for k in ("model", "method", "engine", "condition", "prompt_id")},
                                     **{k: m.get(k) for k in MEASURES}})
    return rows, {k: {"shards": dict(v["shards"]), "cells": dict(v["cells"]), "sources": dict(v["sources"]),
                      "duplicates_skipped": v["duplicates_skipped"]} for k, v in audit.items()}


def summarise(rows: list[dict], draws: int = 200, seed: int = 20261008) -> dict:
    prompts = sorted({r["prompt_id"] for r in rows})
    index = {p: i for i, p in enumerate(prompts)}
    weights = np.random.default_rng(seed).multinomial(len(prompts), np.full(len(prompts), 1 / len(prompts)), size=draws) \
        if prompts else np.zeros((draws, 0))
    out = {"answers": len(rows)}
    for name in MEASURES:
        v = np.array([np.nan if r[name] is None else float(r[name]) for r in rows])
        ok = ~np.isnan(v)
        if not ok.any():
            out[name] = {"mean": None, "n": 0}
            continue
        p = np.array([index[r["prompt_id"]] for r in rows])[ok]
        w = weights[:, p]
        boot = (w * v[ok]).sum(1) / np.maximum(w.sum(1), 1)
        out[name] = {"mean": float(v[ok].mean()), "n": int(ok.sum()),
                     "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


def render(result: dict) -> str:
    f = lambda e: "—" if not e or e.get("mean") is None else (f"{e['mean']:.3f}" + (f" [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]" if "ci95" in e else ""))  # noqa: E731
    lines = ["# Gemma SI-v4 judgments against the generator's own ranking (diagnostic)", "",
             "Complete answers only; 95% prompt-bootstrap intervals. Semantic acceptance of the judge is not established.", "",
             "| group | answers | first listed in Gemma top | chance (random listed) | first shown in Gemma top | chance (random shown) "
             "| tau-b list vs grades | tau-b list vs shown order | important sources omitted |", "|---|---|---|---|---|---|---|---|---|"]
    for key, groups in result["groups"].items():
        for g, s in groups.items():
            lines.append(f"| {g} | {s['answers']:,} | {f(s['top_source_alignment'])} | {f(s['generator_chance_baseline'])} | "
                         f"{f(s['first_presented_alignment'])} | {f(s['presented_chance_baseline'])} | {f(s['ordering_tau_b'])} | "
                         f"{f(s['presented_order_tau_b'])} | {f(s['important_source_omission'])} |")
    lines += ["", "## Audit per run", "", "```", json.dumps(result["audit"], indent=1), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=200)
    a = p.parse_args(argv)
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    rows, audit = collect([r.resolve() for r in a.run])
    groups = {}
    for keys in GROUPS:
        buckets = collections.defaultdict(list)
        for r in rows:
            buckets[" · ".join(r[k] for k in keys)].append(r)
        groups[" × ".join(keys)] = {g: summarise(v, a.draws) for g, v in sorted(buckets.items())}
    result = {"scientific_result": False, "semantic_acceptance": "not_established", "runs": [str(r) for r in a.run],
              "audit": audit, "groups": groups}
    partial = a.output.with_name(a.output.name + ".partial")
    partial.mkdir(parents=True)
    (partial / "agreement.json").write_text(json.dumps(result, indent=1) + "\n")
    (partial / "agreement.md").write_text(render(result))
    partial.rename(a.output)
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
