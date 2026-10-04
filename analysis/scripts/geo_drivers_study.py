#!/usr/bin/env python3
"""GEO drivers study (protocol: analysis/docs/geo_drivers_study.md).

  assemble  CPU   join answers (extract), corpus snippets and vectors, archived prompt vectors,
                  frozen maps and prompt keywords into flat arrays: per presented pair the
                  on-keyword flag and the topic similarity with the intent subspace removed
  analyze   CPU   E1 (prompt position -> ranking / pool / reordering intent) and E2 (drivers of
                  rank) per model x engine and pooled per model, with keyword-cluster bootstrap,
                  within-keyword permutation null and the four-strata replication rule; HTML report

Observational. Writes new directories only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.scripts.page_readiness_ordering import (  # noqa: E402
    EMBEDDING_FILES, git_commit, identity, new_directory, read_jsonl, write_json)

FORMAT_VERSION = "geo-drivers-study-v1"
VIEWS = ("qwen", "mistral")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prompt_vectors(directory: Path) -> tuple[list[str], np.ndarray]:
    ids, blocks = [], []
    for path in sorted(Path(directory).glob("shard-*/question_embeddings.restricted-local.npz")):
        with np.load(path, allow_pickle=False) as archive:
            ids += [str(i) for i in archive["candidate_ids"]]
            blocks.append(np.asarray(archive["embeddings"], np.float32))
    if not blocks:
        raise ValueError(f"no archived prompt embeddings under {directory}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate prompt ids in {directory}")
    return ids, np.concatenate(blocks)


def intent_free(matrix: np.ndarray, map_path: Path, chunk: int = 4096) -> np.ndarray:
    """Normalize, center with the frozen map's mean, remove the span of its two supervised axes,
    and renormalize: what is left describes the text apart from the intent subspace."""

    frozen = json.loads(Path(map_path).read_text())
    mean = np.asarray(frozen["embedding_mean"], np.float32)
    basis, _ = np.linalg.qr(np.asarray(frozen["supervised_subspace_axes"], np.float64).T)
    basis = basis.astype(np.float32)
    out = np.empty(matrix.shape, np.float32)
    for start in range(0, len(matrix), chunk):
        block = np.asarray(matrix[start:start + chunk], np.float32)
        block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        block = block - mean
        block = block - (block @ basis) @ basis.T
        out[start:start + chunk] = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
    return out


def assemble(args) -> int:
    import pyarrow.parquet as pq

    snippets = pq.read_table(args.corpus_package / "snippets.parquet").to_pylist()
    doc_index = {r["snippet_id"]: r["row"] for r in snippets}
    if [r["row"] for r in snippets] != list(range(len(snippets))):
        raise ValueError("corpus rows are not in vector order")
    doc_z = np.asarray([r["consensus_axis_1_z"] for r in snippets])
    doc_u = np.asarray([r["prompt_scale_percentile_0_1"] for r in snippets])
    doc_keywords = [frozenset(k.casefold() for k in (r.get("keywords") or [])) for r in snippets]
    prompt_keyword = {str(r["candidate_id"]): str(r["keyword"]).casefold() for r in read_jsonl(args.population_prompts)}

    codes = {name: {} for name in ("model", "engine", "method", "condition", "keyword", "prompt")}

    def code(kind, value):
        return codes[kind].setdefault(value, len(codes[kind]))

    columns = {name: [] for name in ("model", "engine", "method", "condition", "keyword", "prompt", "x")}
    p_offsets, p_docs, p_on, r_offsets, r_docs, pair_prompt = [0], [], [], [0], [], []
    counts = Counter()
    with gzip.open(args.extract / "observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            o = json.loads(line)
            presented = [doc_index.get(p) for p in o["presented"]]
            if any(d is None for d in presented):
                counts["skipped_presented_page_not_in_corpus"] += 1
                continue
            keyword_text = prompt_keyword.get(o["prompt_id"])
            if keyword_text is None:
                counts["skipped_prompt_without_keyword_text"] += 1
                continue
            prompt_code = code("prompt", o["prompt_id"])
            for kind in ("model", "engine", "method", "condition", "keyword"):
                columns[kind].append(code(kind, o[kind]))
            columns["prompt"].append(prompt_code)
            columns["x"].append(float(o["prompt_axis"]))
            p_docs += presented
            p_on += [float(keyword_text in doc_keywords[d]) for d in presented]
            pair_prompt += [prompt_code] * len(presented)
            p_offsets.append(len(p_docs))
            r_docs += [doc_index[p] for p in o["ranking"]]
            r_offsets.append(len(r_docs))
    counts["answers"] = len(columns["x"])
    counts["prompt_keyword_found_in_corpus"] = sum(
        any(k in kws for kws in doc_keywords) for k in {prompt_keyword[p] for p in codes["prompt"]})
    counts["prompts"] = len(codes["prompt"])
    pair_prompt, p_docs_array = np.asarray(pair_prompt), np.asarray(p_docs)
    unique_pairs, pair_inverse = np.unique(np.stack([pair_prompt, p_docs_array], axis=1), axis=0, return_inverse=True)
    prompt_ids = sorted(codes["prompt"], key=codes["prompt"].get)
    topic, diagnostics = np.zeros(len(unique_pairs)), {}
    for view, prompt_dir, map_dir in (("qwen", args.qwen_prompts, args.qwen_map), ("mistral", args.mistral_prompts, args.mistral_map)):
        ids, vectors = load_prompt_vectors(prompt_dir)
        where = {pid: i for i, pid in enumerate(ids)}
        missing = [p for p in prompt_ids if p not in where]
        if missing:
            raise ValueError(f"{len(missing)} prompts lack archived {view} embeddings")
        prompts = intent_free(vectors[[where[p] for p in prompt_ids]], map_dir / "readiness_embedding_map.json")
        docs = intent_free(np.load(args.corpus_package / EMBEDDING_FILES[view], mmap_mode="r"),
                           map_dir / "readiness_embedding_map.json")
        similarity = np.empty(len(unique_pairs), np.float64)
        for start in range(0, len(unique_pairs), 50000):
            block = unique_pairs[start:start + 50000]
            similarity[start:start + 50000] = np.einsum("ij,ij->i", prompts[block[:, 0]], docs[block[:, 1]])
        diagnostics[view] = {"mean": float(similarity.mean()), "sd": float(similarity.std())}
        topic += similarity / len(VIEWS)
        if view == "qwen":
            first = similarity
    diagnostics["view_agreement_pearson"] = float(np.corrcoef(first, similarity)[0, 1])
    partial = new_directory(args.output)
    arrays = {name: np.asarray(values, float if name == "x" else np.int64) for name, values in columns.items()}
    np.savez(partial / "answers.npz", **arrays, p_offsets=np.asarray(p_offsets), p_docs=p_docs_array,
             p_on_keyword=np.asarray(p_on), p_topic=topic[pair_inverse.reshape(-1)], r_offsets=np.asarray(r_offsets),
             r_docs=np.asarray(r_docs, np.int64), doc_z=doc_z, doc_u=doc_u)
    write_json(partial / "codes.json", {kind: sorted(values, key=values.get) for kind, values in codes.items() if kind != "prompt"})
    write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "stage": "assemble", "created_at": now(), "git_commit": git_commit(),
        "inputs": {"extract": identity(args.extract / "manifest.json"),
                   "corpus": identity(args.corpus_package / "manifest.json"),
                   "population_prompts": identity(args.population_prompts),
                   "maps": {v: identity(d / "readiness_embedding_map.json") for v, d in (("qwen", args.qwen_map), ("mistral", args.mistral_map))}},
        "counts": dict(counts), "presented_pairs": len(p_docs), "unique_prompt_page_pairs": int(len(unique_pairs)),
        "on_keyword_share": float(np.mean(p_on)) if p_on else None, "topic_similarity": diagnostics,
        "topic_definition": "cosine after normalizing, centering with each frozen map's mean and removing its two supervised axes; mean of both views"})
    partial.rename(args.output.resolve())
    print(json.dumps({"answers": counts["answers"], "pairs": len(p_docs), "unique_pairs": int(len(unique_pairs)),
                      "on_keyword_share": float(np.mean(p_on)) if p_on else None, **diagnostics}), flush=True)
    return 0


def load_answers(directory: Path) -> tuple[geo.Answers, dict]:
    with np.load(directory / "answers.npz", allow_pickle=False) as data:
        a = geo.Answers(*(data[name] for name in ("model", "engine", "keyword", "prompt", "x", "p_offsets", "p_docs",
                                                   "p_on_keyword", "p_topic", "r_offsets", "r_docs", "doc_z", "doc_u")))
    return a, json.loads((directory / "codes.json").read_text())


def analyze(args) -> int:
    a, codes = load_answers(args.assembled)
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    rows = geo.choice_rows(a)
    strata = {}
    for m, model in enumerate(codes["model"]):
        for e, engine in [*enumerate(codes["engine"]), (None, "all engines")]:
            keep = np.flatnonzero((a.model == m) & ((a.engine == e) if e is not None else True))
            if not len(keep):
                continue
            name = f"{model} · {engine}"
            sub = a.subset(np.isin(np.arange(len(a.x)), keep))
            seed = args.seed + 101 * m + (7 * e if e is not None else 3)
            strata[name] = {"model": model, "engine": engine,
                            "e1": geo.estimate_e1(sub, bootstrap=args.bootstrap, permutations=args.permutations, seed=seed),
                            "e2": geo.estimate_e2(sub, bootstrap=args.bootstrap, permutations=args.permutations, seed=seed,
                                                  workers=workers, rows=geo.restrict(rows, keep))}
            print(json.dumps({"stratum": name, "e1_ranking_slope": strata[name]["e1"]["ranking"]["slope"],
                              **{k: v["beta_per_sd"] for k, v in strata[name]["e2"]["drivers"].items()}}), flush=True)
    primary = {k: v for k, v in strata.items() if v["engine"] != "all engines"}
    results = {"format_version": FORMAT_VERSION, "stage": "analyze", "created_at": now(), "git_commit": git_commit(),
               "assembled": identity(args.assembled / "manifest.json"),
               "settings": {"bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed},
               "replication": {"e1": geo.replication(primary, section="e1", terms=("ranking", "pool", "reordering")),
                               "e2": geo.replication(primary, section="e2", terms=geo.DRIVERS)},
               "strata": strata}
    partial = new_directory(args.output)
    write_json(partial / "results.json", results)
    (partial / "report.html").write_text(render(results), encoding="utf-8")
    partial.rename(args.output.resolve())
    print(f"REPORT {args.output.resolve() / 'report.html'}", flush=True)
    return 0


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:+.{digits}f}"


def _ci(ci):
    return "—" if not ci or ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _p(value):
    return "" if value is None else f" · p {value:.3f}"


def _lines(strata: dict, series: str, title: str) -> str:
    entries = [(name, s["e1"]["bins"]) for name, s in strata.items() if s["engine"] != "all engines"]
    values = [b[series] for _, bins in entries for b in bins]
    if not values:
        return ""
    low, high = min(values + [0.0]), max(values + [0.0])
    span = (high - low) or 1.0
    width, height, pad = 600, 240, 40
    x = lambda v: pad + v * (width - 2 * pad)
    y = lambda v: height - pad - (v - low) / span * (height - 2 * pad)
    lines = []
    for index, (name, bins) in enumerate(entries):
        points = " ".join(f"{x((b['low'] + b['high']) / 2):.1f},{y(b[series]):.1f}" for b in bins)
        lines.append(f'<polyline points="{points}" class="s{index}"/>'
                     f'<text x="{width - pad + 4}" y="{y(bins[-1][series]) + 4:.1f}" class="lab s{index}t">{html.escape(name)}</text>')
    return (f'<figure><div class="scroll"><svg viewBox="0 0 {width + 150} {height}" role="img" aria-label="{html.escape(title)}">'
            f'<line x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>{"".join(lines)}'
            f'<text x="{pad}" y="{height - 10}" class="ax">information seeking (0)</text>'
            f'<text x="{width - pad}" y="{height - 10}" class="ax" text-anchor="end">action ready (1)</text>'
            f'<text x="4" y="{y(high) + 4:.1f}" class="ax">{high:+.2f}</text><text x="4" y="{y(low):.1f}" class="ax">{low:+.2f}</text>'
            f'</svg></div><figcaption>{html.escape(title)}</figcaption></figure>')


def render(results: dict) -> str:
    replication_rows = []
    for section, label in (("e1", "E1"), ("e2", "E2")):
        for term, entry in results["replication"][section].items():
            cells = "".join(f"<td>{_fmt(r['estimate'])}<br><small>{_ci(r['ci95'])}{_p(r['p'])}</small></td>"
                            for r in entry["strata"])
            replication_rows.append(f"<tr><td>{label} {html.escape(term.replace('_', ' '))}</td>"
                                    f"<td class='{'yes' if entry['replicates'] else 'no'}'>{'replicates' if entry['replicates'] else 'not replicated'}</td>{cells}</tr>")
    names = [r["stratum"] for r in next(iter(results["replication"]["e1"].values()))["strata"]]
    driver_rows = []
    for name, stratum in results["strata"].items():
        e2 = stratum["e2"]
        cells = "".join(f"<td>{d['odds_ratio_per_sd']:.2f}<br><small>{_ci(d['ci95'])} · share {d['fit_share'] or 0:.0%}</small></td>"
                        for d in e2["drivers"].values())
        driver_rows.append(f"<tr><td>{html.escape(name)}</td><td>{e2['answers']:,}</td>{cells}</tr>")
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    drivers_head = "".join(f"<th>{d.replace('_', ' ')}<br><small>odds ratio per SD</small></th>" for d in geo.DRIVERS)
    return f"""<title>GEO Drivers Study</title>
<style>
:root{{--bg:#f7f7f4;--fg:#1c1f22;--muted:#5b6268;--rule:#d7dad6;--yes:#1d6b45;--no:#9a3412;
--s0:#2f5d8a;--s1:#b8572a;--s2:#5b7f2e;--s3:#7a4b9c;--font:"Source Sans 3","Helvetica Neue",Arial,sans-serif}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181b;--fg:#e8e9e6;--muted:#a3a9ae;--rule:#343a40;--yes:#6fcf97;--no:#f4a261;--s0:#8cb8e6;--s1:#f0a07a;--s2:#a9cf6f;--s3:#c9a3e6;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#16181b;--fg:#e8e9e6;--muted:#a3a9ae;--rule:#343a40;--yes:#6fcf97;--no:#f4a261;--s0:#8cb8e6;--s1:#f0a07a;--s2:#a9cf6f;--s3:#c9a3e6;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 var(--font);margin:0 auto;max-width:1040px;padding-inline:16px;padding-block:28px 56px}}
h1,h2{{line-height:1.2;text-wrap:balance}} .note,small,figcaption{{color:var(--muted)}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}}
.yes{{color:var(--yes);font-weight:600}} .no{{color:var(--no);font-weight:600}}
svg{{width:100%;height:auto;min-width:560px}} polyline{{fill:none;stroke-width:2}} .zero{{stroke:var(--muted);stroke-dasharray:4 4}}
.ax,.lab{{font-size:11px;fill:var(--muted)}} .s0{{stroke:var(--s0)}} .s1{{stroke:var(--s1)}} .s2{{stroke:var(--s2)}} .s3{{stroke:var(--s3)}}
.s0t{{fill:var(--s0)}} .s1t{{fill:var(--s1)}} .s2t{{fill:var(--s2)}} .s3t{{fill:var(--s3)}}
</style>
<h1>GEO drivers: prompt intent, page intent and keyword</h1>
<p class="note">Observational, closed frozen corpus. Protocol fixed before this run (analysis/docs/geo_drivers_study.md).
A result replicates only if all four model × engine strata agree in sign, exclude 0 in their 95% keyword-bootstrap interval and,
for terms involving the prompt position, pass the within-keyword permutation null (p &lt; 0.05).
Generated {html.escape(results['created_at'])} from commit {html.escape(results['git_commit'][:12])}.</p>
<h2>Replication summary</h2>
<div class="scroll"><table><thead><tr><th>term</th><th>verdict</th>{head}</tr></thead><tbody>{''.join(replication_rows)}</tbody></table></div>
<p class="note">E1 slopes are in consensus-z units: change from the most informational to the most action-ready prompt of a keyword.
Ranking = top-weighted intent of the ranked pages; pool = intent of the presented pages; reordering = ranking − pool.</p>
<h2>E1: where the ranking lands on the axis</h2>
{_lines(results['strata'], 'ranking', 'Ranking intent (top-weighted, consensus z) by prompt position')}
{_lines(results['strata'], 'pool', 'Pool intent (presented pages) by prompt position')}
<h2>E2: what drives a page's rank</h2>
<div class="scroll"><table><thead><tr><th>stratum</th><th>answers</th>{drivers_head}</tr></thead><tbody>{''.join(driver_rows)}</tbody></table></div>
<p class="note">Odds ratio for being picked next per standard deviation of each driver, holding the others and presented position fixed.
Share = that driver's part of the fit lost when it is dropped.</p>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    p = stages.add_parser("assemble")
    p.add_argument("--extract", type=Path, required=True, help="extract-all-v1 (llama and Qwen answers)")
    p.add_argument("--corpus-package", type=Path, required=True, help="snippet-embeddings-corpus-v1")
    p.add_argument("--qwen-prompts", type=Path, required=True, help="final-audit/projections/qwen")
    p.add_argument("--mistral-prompts", type=Path, required=True)
    p.add_argument("--qwen-map", type=Path, required=True)
    p.add_argument("--mistral-map", type=Path, required=True)
    p.add_argument("--population-prompts", type=Path, required=True, help="population-prompts.jsonl (prompt keyword text)")
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("analyze")
    p.add_argument("--assembled", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"assemble": assemble, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
