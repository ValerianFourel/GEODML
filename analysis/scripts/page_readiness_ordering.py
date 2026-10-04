#!/usr/bin/env python3
"""Relocate the frozen readiness subspace, place evidence pages on it, and relate
generator rankings to page coordinates along the prompt axis.

Stages (each writes a new directory, never overwriting):
  extract         CPU   answers (presented pages, ranking, prompt axis) + unique page texts
  sample-prompts  CPU   deterministic sample of archived prompt texts for re-embedding
  embed           1 GPU one LLM2Vec view; sharded, resumable, one process per GPU
  merge           CPU   join shards into the archived projection format with a manifest
  relocate        CPU   prove the archived prompt coordinates are reproduced
  analyze         CPU   page coordinates, ranking models, bootstrap/permutation, HTML

All results are observational. The axis is a measured prompt property; page
coordinates are out-of-domain descriptions of page text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY, REPOSITORY / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.interpretability.pipeline import page_readiness_ordering as ordering  # noqa: E402

SHARD_SIZE = 20000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def new_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError(f"interrupted output preserved for inspection: {partial}")
    partial.mkdir(parents=True)
    return partial


def axis_by_prompt(final_axis_map: Path) -> dict[str, float]:
    return {str(r["candidate_id"]): float(r["axis_1_percentile_0_1"]) for r in read_jsonl(final_axis_map)}


# ---------------------------------------------------------------- extract

def extract(args) -> int:
    from analysis.interpretability.pipeline.agentic_cells import completed_generator_refs, iter_cells
    from analysis.interpretability.pipeline.agentic_judging import _trace_evidence

    prompt_axis = axis_by_prompt(args.final_axis_map)
    partial = new_directory(args.output)
    counts, inputs = Counter(), []
    pages: dict[str, dict] = {}
    observations = []
    for spec in args.source:
        root, _, model = spec.rpartition(":")
        root = Path(root).resolve(strict=True)
        refs, ref_counts = completed_generator_refs(root, model=model)
        inputs.append({"dataset_root": str(root), "model": model, "cells_completed": len(refs),
                       "selection": dict(ref_counts)})
        for cell in iter_cells(root, refs):
            generation = cell["generation"]
            prompt_id = generation.get("prompt_id")
            keyword = (cell.get("keyword_memberships") or {}).get("primary_keyword_id")
            if prompt_id not in prompt_axis:
                counts["skipped_prompt_without_axis"] += 1
                continue
            if not keyword:
                counts["skipped_without_keyword"] += 1
                continue
            try:
                # The final evidence the generator saw, in presented order (shared with SI judging).
                evidence, _ = _trace_evidence(cell["trace"], generation["method"])
            except (ValueError, KeyError, TypeError):
                counts["skipped_unreadable_evidence"] += 1
                continue
            urls = [row["url"] for row in evidence]
            ranking = list(generation.get("ranking") or [])
            if len(set(urls)) != len(urls):
                counts["skipped_duplicate_presented_url"] += 1
                continue
            if len(set(ranking)) != len(ranking) or not set(ranking) <= set(urls):
                counts["skipped_ranking_outside_evidence"] += 1
                continue
            by_url = {}
            for row in evidence:
                text = ordering.page_text(row["title"], row["text"])
                identifier = ordering.page_id(text)
                by_url[row["url"]] = identifier
                entry = pages.setdefault(identifier, {"page_id": identifier, "text": text, "occurrences": 0, "urls": set()})
                entry["occurrences"] += 1
                entry["urls"].add(row["url"])
            observations.append({
                "generation_id": cell["fingerprint"], "prompt_id": prompt_id, "keyword": keyword, "model": model,
                **{k: generation.get(k) for k in ("method", "engine", "condition")},
                "prompt_axis": prompt_axis[prompt_id], "presented": [by_url[u] for u in urls],
                "ranking": [by_url[u] for u in ranking]})
            counts[f"answers_{model}"] += 1
    write_jsonl(partial / "observations.jsonl.gz", observations)
    write_jsonl(partial / "pages.jsonl.gz", ({"page_id": p["page_id"], "text": p["text"], "text_sha256": p["page_id"],
                                              "occurrences": p["occurrences"], "urls": sorted(p["urls"])}
                                             for p in sorted(pages.values(), key=lambda r: r["page_id"])))
    write_json(partial / "manifest.json", {
        "format_version": ordering.FORMAT_VERSION, "stage": "extract", "created_at": now(),
        "git_commit": git_commit(), "inputs": inputs, "final_axis_map": identity(args.final_axis_map),
        "prompt_axis_field": "axis_1_percentile_0_1", "page_text_rule": "title.strip() + newline + snippet.strip()",
        "counts": {**counts, "answers": len(observations), "unique_pages": len(pages)}})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"answers": len(observations), "unique_pages": len(pages), **counts}), flush=True)
    return 0


def sample_prompts(args) -> int:
    rows = read_jsonl(args.prompts)
    chosen = sorted(rows, key=lambda r: hashlib.sha256(f"{args.seed}:{r['candidate_id']}".encode()).hexdigest())[:args.count]
    partial = new_directory(args.output)
    write_jsonl(partial / "pages.jsonl.gz", ({"page_id": r["candidate_id"], "text": r["question"],
                                              "text_sha256": r["question_sha256"]} for r in chosen))
    write_json(partial / "manifest.json", {"stage": "sample-prompts", "created_at": now(), "seed": args.seed,
                                           "count": len(chosen), "prompts": identity(args.prompts)})
    partial.rename(Path(args.output).resolve())
    return 0


# ---------------------------------------------------------------- embed / merge

def _readiness_modules():
    from interpretability.pipeline import readiness_prompt_population as population
    return population


def embed(args) -> int:
    """Embed this worker's shards with one LLM2Vec view and project through its frozen map."""

    population = _readiness_modules()
    from interpretability.pipeline.two_axis_prompt_population import LLM2VecPromptEmbedder
    from analysis.scripts.build_readiness_prompt_population import _validate_embedding_model_revision

    fitted = population.load_readiness_embedding_map(args.map / "readiness_embedding_map.json")
    _validate_embedding_model_revision(fitted, args.embedding_model)
    bounds = population.fit_reference_bounds(read_jsonl(args.map / "readiness_supervised_subspace_coordinates.jsonl"))
    pages = read_jsonl(args.pages)
    for row in pages:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError(f"page text hash differs: {row['page_id']}")
    output = Path(args.output).resolve()
    (output / "shards").mkdir(parents=True, exist_ok=True)
    settings = {"view": args.view, "map_id": fitted.map_id, "map": identity(args.map / "readiness_embedding_map.json"),
                "pages": identity(args.pages), "embedding_model": str(Path(args.embedding_model).resolve()),
                "mntp_model": str(Path(args.mntp_model).resolve()) if args.mntp_model else None,
                "peft_model": str(Path(args.peft_model).resolve()) if args.peft_model else None,
                "max_length": args.max_length, "attention_implementation": args.attention_implementation,
                "shard_size": args.shard_size}
    settings_path = output / "embed-settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("existing embedding output used different settings; use a new output")
    else:
        try:
            with settings_path.open("x") as stream:
                stream.write(json.dumps(settings, indent=2, sort_keys=True) + "\n")
        except FileExistsError:
            if json.loads(settings_path.read_text()) != settings:
                raise ValueError("existing embedding output used different settings") from None
    embedder = None
    shards = range(0, len(pages), args.shard_size)
    for number, start in enumerate(shards):
        if number % args.workers != args.worker_index:
            continue
        target = output / "shards" / f"{number:05d}.jsonl.gz"
        if target.exists():
            continue
        chunk = pages[start:start + args.shard_size]
        if embedder is None:
            embedder = LLM2VecPromptEmbedder(args.embedding_model, mntp_model_name_or_path=args.mntp_model,
                                             peft_model_name_or_path=args.peft_model, batch_size=args.batch_size,
                                             max_length=args.max_length,
                                             attention_implementation=args.attention_implementation)
        projections = population.project_text_embeddings(
            fitted, bounds, item_ids=[r["page_id"] for r in chunk], text_sha256s=[r["text_sha256"] for r in chunk],
            embeddings=embedder.embed([r["text"] for r in chunk]))
        write_jsonl(target, ({"candidate_id": p.item_id, "projection": asdict(p)} for p in projections))
        print(json.dumps({"view": args.view, "shard": number, "pages": len(chunk), "time": now()}), flush=True)
    return 0


def merge(args) -> int:
    settings = json.loads((args.input / "embed-settings.json").read_text())
    pages = [r["page_id"] for r in read_jsonl(args.pages)]
    if identity(args.pages)["sha256"] != settings["pages"]["sha256"]:
        raise ValueError("pages file differs from the embedded one")
    expected = (len(pages) + settings["shard_size"] - 1) // settings["shard_size"]
    rows = []
    for number in range(expected):
        path = args.input / "shards" / f"{number:05d}.jsonl.gz"
        if not path.exists():
            raise ValueError(f"missing shard {number}; rerun embed with the same settings")
        rows.extend(read_jsonl(path))
    if [r["candidate_id"] for r in rows] != pages:
        raise ValueError("merged projections do not cover the pages exactly once, in order")
    partial = new_directory(args.output)
    write_jsonl(partial / "question_projections.jsonl", rows)
    write_json(partial / "projection_manifest.json", {
        "format_version": ordering.FORMAT_VERSION, "created_at": now(), "git_commit_sha": git_commit(),
        "map_id": settings["map_id"], "map": settings["map"], "embedding": settings,
        "candidate_count": len(rows), "embedding_arrays_included": False})
    partial.rename(Path(args.output).resolve())
    return 0


def aligned(qwen: Path, mistral: Path, battery: Path) -> list[dict]:
    """The archived Qwen/Mistral alignment code, applied unchanged."""

    from analysis.scripts.build_readiness_prompt_population import _aligned_projection_rows
    rows, _, _ = _aligned_projection_rows(Path(qwen).resolve(), Path(mistral).resolve(), Path(battery).resolve())
    return rows


# ---------------------------------------------------------------- relocate

def relocate(args) -> int:
    final_rows = read_jsonl(args.final_axis_map)
    report = {"format_version": ordering.FORMAT_VERSION, "stage": "relocate", "created_at": now(),
              "git_commit": git_commit(), "final_axis_map": identity(args.final_axis_map),
              "battery": identity(args.battery / "readiness_robustness_battery.json"),
              "archived_coordinates": ordering.relocation_audit(
                  final_rows, aligned(args.qwen_projections, args.mistral_projections, args.battery))}
    fresh = {}
    for view, archived_dir, fresh_dir in (("qwen", args.qwen_projections, args.fresh_qwen),
                                          ("mistral", args.mistral_projections, args.fresh_mistral)):
        if fresh_dir is None:
            continue
        archived = {r["candidate_id"]: r["projection"]["raw_axis_1"]
                    for r in read_jsonl(archived_dir / "question_projections.jsonl")}
        observed = {r["candidate_id"]: r["projection"]["raw_axis_1"]
                    for r in read_jsonl(fresh_dir / "question_projections.jsonl")}
        fresh[view] = ordering.projection_agreement(archived, observed)
    report["fresh_reembedding"] = fresh
    report["passed"] = report["archived_coordinates"]["passed"] and all(v["passed"] for v in fresh.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], **report["archived_coordinates"],
                      **{f"fresh_{k}": v["spearman"] for k, v in fresh.items()}}), flush=True)
    return 0 if report["passed"] else 2


# ---------------------------------------------------------------- analyze

def analyze(args) -> int:
    extract_manifest = json.loads((args.extract / "manifest.json").read_text())
    observations = read_jsonl(args.extract / "observations.jsonl.gz")
    scale = ordering.PromptScale(read_jsonl(args.final_axis_map))
    rows = aligned(args.qwen, args.mistral, args.battery)
    views = {"qwen": {r["candidate_id"]: r["reference_axis_1_z"] for r in rows},
             "mistral": {r["candidate_id"]: r["candidate_aligned_axis_1_z"] for r in rows}}
    ids = sorted(views["qwen"])
    views["consensus"] = dict(zip(ids, ordering.consensus([views["qwen"][i] for i in ids],
                                                         [views["mistral"][i] for i in ids])))
    percentile, outside = scale.percentile(np.asarray([views["consensus"][i] for i in ids]))
    page_percentile, page_outside = dict(zip(ids, map(float, percentile))), dict(zip(ids, map(bool, outside)))
    missing = {p for o in observations for p in o["presented"]} - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} presented pages lack projections")
    partial = new_directory(args.output)
    write_jsonl(partial / "page_coordinates.jsonl.gz", ({"page_id": i, "qwen_axis_1_z": views["qwen"][i],
        "mistral_aligned_axis_1_z": views["mistral"][i], "consensus_axis_1_z": views["consensus"][i],
        "prompt_scale_percentile_0_1": page_percentile[i], "outside_prompt_range": page_outside[i]} for i in ids))
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    results = {"format_version": ordering.FORMAT_VERSION, "stage": "analyze", "created_at": now(),
               "git_commit": git_commit(), "extract": extract_manifest,
               "inputs": {"final_axis_map": identity(args.final_axis_map),
                          "qwen": identity(args.qwen / "projection_manifest.json"),
                          "mistral": identity(args.mistral / "projection_manifest.json"),
                          "battery": identity(args.battery / "battery_manifest.json")},
               "settings": {"bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed,
                            "max_position": ordering.MAX_POSITION},
               "page_scale": ordering.page_scale_diagnostics(views["consensus"], page_percentile, page_outside, observations),
               "models": {}}
    for model in sorted({o["model"] for o in observations}):
        selected = [o for o in observations if o["model"] == model and o["ranking"]]
        statistics = [ordering.generation_statistics(o, views["consensus"]) for o in selected]
        entry = {"answers": len(selected), "keywords": len({o["keyword"] for o in selected}), "fits": {}}
        for view in ordering.VIEWS:
            data = ordering.ChoiceData(selected, views[view])
            fit = ordering.fit_plackett_luce(data)
            if view == "consensus":
                fit["keyword_bootstrap"] = ordering.keyword_bootstrap(
                    data, replicates=args.bootstrap, seed=args.seed, start=fit["theta"], workers=workers)
                fit["within_keyword_permutation"] = ordering.within_keyword_permutation(
                    data, observed=fit["page_z_x_prompt_axis"], replicates=args.permutations,
                    seed=args.seed + 1, start=fit["theta"], workers=workers)
            entry["fits"][view] = fit
        entry["by_method"] = {}
        for method in sorted({o["method"] for o in selected}):
            subset = [o for o in selected if o["method"] == method]
            entry["by_method"][method] = ordering.fit_plackett_luce(ordering.ChoiceData(subset, views["consensus"]))
        entry["binned"] = {field: ordering.binned_contrasts(selected, statistics, field=field, seed=args.seed)
                           for field in ("ranked_minus_presented_z", "top1_minus_presented_z")}
        valid = [(o["prompt_axis"], s["ranked_minus_presented_z"]) for o, s in zip(selected, statistics)
                 if s.get("ranked_minus_presented_z") is not None]
        entry["axis_vs_ranked_minus_presented_spearman"] = ordering._spearman(*map(np.asarray, zip(*valid))) if len(valid) > 2 else None
        results["models"][model] = entry
        print(json.dumps({"model": model, "answers": len(selected),
                          "page_z": entry["fits"]["consensus"]["page_z"],
                          "interaction": entry["fits"]["consensus"]["page_z_x_prompt_axis"]}), flush=True)
    write_json(partial / "results.json", results)
    (partial / "report.html").write_text(render(results), encoding="utf-8")
    partial.rename(Path(args.output).resolve())
    print(f"REPORT {Path(args.output).resolve() / 'report.html'}", flush=True)
    return 0


def _chart(bins: list[dict], title: str) -> str:
    points = [b for b in bins if "mean" in b]
    if not points:
        return "<p>No data.</p>"
    low = min(min(b["ci95"][0] for b in points), 0.0)
    high = max(max(b["ci95"][1] for b in points), 0.0)
    span = (high - low) or 1.0
    width, height, pad = 560, 220, 36

    def x(value):
        return pad + value * (width - 2 * pad)

    def y(value):
        return height - pad - (value - low) / span * (height - 2 * pad)

    centers = [(b["prompt_axis_low"] + b["prompt_axis_high"]) / 2 for b in points]
    band = " ".join(f"{x(c):.1f},{y(b['ci95'][1]):.1f}" for c, b in zip(centers, points))
    band += " " + " ".join(f"{x(c):.1f},{y(b['ci95'][0]):.1f}" for c, b in reversed(list(zip(centers, points))))
    line = " ".join(f"{x(c):.1f},{y(b['mean']):.1f}" for c, b in zip(centers, points))
    return (f'<figure><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'<line x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>'
            f'<polygon points="{band}" class="band"/><polyline points="{line}" class="line"/>'
            f'<text x="{pad}" y="{height - 8}">information seeking (0)</text>'
            f'<text x="{width - pad}" y="{height - 8}" text-anchor="end">action ready (1)</text>'
            f'<text x="4" y="{y(high) + 4:.1f}">{high:+.2f}</text><text x="4" y="{y(low):.1f}">{low:+.2f}</text>'
            f'</svg><figcaption>{html.escape(title)}</figcaption></figure>')


def render(results: dict) -> str:
    def number(value, digits=3):
        return "—" if value is None else f"{value:+.{digits}f}"

    def pvalue(permutation):
        value = permutation.get("two_sided_p")
        return "—" if value is None else f"{value:.3f}"

    sections = []
    for model, entry in results["models"].items():
        rows = []
        for view, fit in entry["fits"].items():
            boot = fit.get("keyword_bootstrap", {})
            perm = fit.get("within_keyword_permutation", {})
            ci = lambda key: ("[" + ", ".join(f"{v:+.3f}" for v in boot[key]["ci95"]) + "]") if key in boot else "—"
            rows.append(f"<tr><td>{view}</td><td>{number(fit['page_z'])}</td><td>{ci('page_z')}</td>"
                        f"<td>{number(fit['page_z_x_prompt_axis'])}</td><td>{ci('page_z_x_prompt_axis')}</td>"
                        f"<td>{pvalue(perm)}</td>"
                        f"<td>{fit['choice_sets']:,}</td></tr>")
        methods = "".join(f"<tr><td>{html.escape(m)}</td><td>{number(f['page_z'])}</td>"
                          f"<td>{number(f['page_z_x_prompt_axis'])}</td><td>{f['choice_sets']:,}</td></tr>"
                          for m, f in entry["by_method"].items())
        sections.append(f"""
<section><h2>{html.escape(model)}</h2>
<p>{entry['answers']:,} answers across {entry['keywords']:,} keywords. Spearman between prompt axis and
(mean z of ranked pages − mean z of presented pages): <b>{number(entry['axis_vs_ranked_minus_presented_spearman'])}</b>.</p>
<h3>Ranking model (Plackett–Luce over the ranked prefix, presented-position effects)</h3>
<table><thead><tr><th>page view</th><th>page z</th><th>95% CI</th><th>page z × prompt axis</th><th>95% CI</th>
<th>permutation p</th><th>choices</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h3>By search method (consensus view)</h3>
<table><thead><tr><th>method</th><th>page z</th><th>page z × prompt axis</th><th>choices</th></tr></thead><tbody>{methods}</tbody></table>
{_chart(entry['binned']['ranked_minus_presented_z'], 'Ranked minus presented page z, by prompt-axis decile (keyword-cluster 95% CI)')}
{_chart(entry['binned']['top1_minus_presented_z'], 'Top-ranked page minus presented mean z, by prompt-axis decile')}
</section>""")
    scale = results["page_scale"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Page Readiness Ordering</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5d5a52;--line:#2f5d8a;--band:#2f5d8a33;--rule:#d8d4ca}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16171a;--fg:#ecebe6;--muted:#a9a69c;--line:#8cb8e6;--band:#8cb8e633;--rule:#3a3b40}}}}
:root[data-theme="dark"]{{--bg:#16171a;--fg:#ecebe6;--muted:#a9a69c;--line:#8cb8e6;--band:#8cb8e633;--rule:#3a3b40}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:860px;padding:24px 16px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 16px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
svg{{width:100%;height:auto}} .line{{fill:none;stroke:var(--line);stroke-width:2}} .band{{fill:var(--band)}}
.zero{{stroke:var(--muted);stroke-dasharray:4 4}} svg text{{fill:var(--muted);font-size:11px}} .note{{color:var(--muted)}}
</style></head><body>
<h1>Page readiness and generator ordering</h1>
<p class="note">Observational. The prompt axis is a measured prompt property, not a randomized treatment. Page
coordinates come from a subspace fitted on prompts and are out-of-domain descriptions of page text. Generated
{html.escape(results['created_at'])} from commit {html.escape(results['git_commit'][:12])}.</p>
<h2>How pages sit on the prompt scale</h2>
<p>{scale['pages']:,} unique pages; {scale['share_outside_prompt_range']:.1%} fall outside the prompt range.
Median page position on the prompt percentile scale: {scale['page_percentile_quantiles']['0.5']:.2f}.
Mean within-answer variance of page z: {number(scale['mean_within_answer_variance_z'])} (total {number(scale['page_z_variance'])}).</p>
<p class="note">Positive page z: generators rank action-ready pages higher. Positive page z × prompt axis: that
preference grows as the prompt moves toward action readiness. Position effects absorb presented order, which
the evidence-order conditions randomize.</p>
{''.join(sections)}
</body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    p = stages.add_parser("extract")
    p.add_argument("--source", action="append", required=True, metavar="DATASET_ROOT:MODEL")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("sample-prompts")
    p.add_argument("--prompts", type=Path, required=True, help="compliant-candidates.jsonl")
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("embed")
    p.add_argument("--pages", type=Path, required=True)
    p.add_argument("--view", choices=("qwen", "mistral"), required=True)
    p.add_argument("--map", type=Path, required=True, help="directory with readiness_embedding_map.json")
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--mntp-model")
    p.add_argument("--peft-model")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--attention-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("merge")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--pages", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("relocate")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--qwen-projections", type=Path, required=True, help="final audit merged/qwen")
    p.add_argument("--mistral-projections", type=Path, required=True, help="final audit merged/mistral")
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--fresh-qwen", type=Path)
    p.add_argument("--fresh-mistral", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("analyze")
    p.add_argument("--extract", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--mistral", type=Path, required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"extract": extract, "sample-prompts": sample_prompts, "embed": embed, "merge": merge,
            "relocate": relocate, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
