#!/usr/bin/env python3
"""Where does intent surface? Protocol: analysis/docs/intent_surfacing_study.md.

  trace-extract  CPU  per answer, from the traces: the AI's queries, every retrieved page, the
                      reranker's scored candidates (score, kept or not), the shortlist and the
                      ranking as corpus rows, with consistency checks; query texts for embedding
  replay         CPU  rerun the frozen search with each prompt's own text and with its bare
                      keyword; stops unless recorded AI queries replay to their recorded rows
  analyze        CPU  stage slopes, chain decomposition, mediation, oracles, dispersion,
                      shortlisting and ranking drivers, reranker score drivers, answer stage,
                      contrasts and replication: results.json and final-report.html

Queries are embedded with page_readiness_ordering.py embed/merge on queries.jsonl.gz.
Observational. Writes new directories only.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")  # forked bootstrap workers must not inherit BLAS thread pools

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
import hashlib  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.interpretability.pipeline import page_readiness_ordering as ordering  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402
from analysis.scripts.geo_drivers_study import VIEWS, odds_interval, pair_topic_similarity  # noqa: E402

FORMAT_VERSION = "intent-stages-study-v1"
FIDELITY_PREFIX = "00"  # recorded searches with sha256(engine NUL query) starting so are replayed (1/256)
STAGE_ARRAYS = ("model", "engine", "method", "condition", "keyword", "prompt", "x", "ret_offsets", "ret_docs",
                "cand_offsets", "cand_docs", "pool_offsets", "pool_docs", "rank_offsets", "rank_docs",
                "q_offsets", "q_ids")
ANSWER_KEYS = ("model", "prompt_id", "method", "engine", "condition")
DERIVED_REPLICATION = ("pool_minus_reordering", "mediated_by_pool", "utilization", "query_rewriting")
RANKING_CHANGE_METRICS = ("top1_change", "top3_set_distance", "url_set_distance", "kendall_distance_common")


def corpus_index(package: Path) -> dict[str, int]:
    import pyarrow.parquet as pq
    table = pq.read_table(package / "snippets.parquet", columns=["row", "snippet_id"]).to_pydict()
    if table["row"] != list(range(len(table["row"]))):
        raise ValueError("corpus rows are not in vector order")
    return dict(zip(table["snippet_id"], table["row"]))


def refuse_existing(path: Path) -> None:
    """Fail before any work when the output (or an interrupted attempt at it) already exists."""
    path = Path(path).resolve()
    for candidate in (path, path.with_name(path.name + ".partial")):
        if candidate.exists():
            raise ValueError(f"refusing to overwrite {candidate}")


def _map(function, jobs, workers):
    if workers <= 1 or len(jobs) < 2:
        return list(map(function, jobs))
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return pool.map(function, jobs, chunksize=max(1, len(jobs) // (workers * 8)))


# ---------------------------------------------------------------- trace-extract

def _trace_chunk(job):
    """One contiguous slice of completed cells; runs in a forked worker."""
    from analysis.interpretability.pipeline.agentic_cells import iter_cells

    root, model, refs, (prompt_axis, doc_index) = job
    counts = Counter()
    meta, queries, prompts, searches, snapshots = [], [], {}, defaultdict(set), set()
    sizes, docs, events = defaultdict(list), defaultdict(list), defaultdict(list)

    def rows_of(items):
        return [doc_index.get(ordering.page_id(ordering.page_text(item[1], item[2]))) for item in items]

    for cell in iter_cells(root, refs):
        fields, skipped = readiness.checked_cell(cell, prompt_axis)
        if skipped:
            counts[skipped] += 1
            continue
        generation = cell["generation"]
        try:
            parsed = stages.trace_stages(cell["trace"], generation["method"], fields["urls"])
        except (KeyError, TypeError, ValueError):
            counts["skipped_unreadable_trace_stages"] += 1
            continue
        by_url = {row["url"]: doc_index.get(ordering.page_id(ordering.page_text(row["title"], row["text"])))
                  for row in fields["evidence"]}
        lists = {"ret": rows_of(parsed["retrieved"]), "pool": [by_url[u] for u in fields["urls"]],
                 "rank": [by_url[u] for u in fields["ranking"]]}
        scored = [rows_of(event["rows"]) for event in parsed["events"]]
        if None in lists["ret"] or None in lists["pool"] or any(None in s for s in scored):
            counts["skipped_page_not_in_corpus"] += 1
            continue
        lists["ret"] = list(dict.fromkeys(lists["ret"]))
        lists["cand"] = list(dict.fromkeys(d for s in scored for d in s))
        for name, values in lists.items():
            sizes[name].append(len(values))
            docs[name] += values
        for event, rows in zip(parsed["events"], scored):
            events["answer"].append(len(meta))
            events["rows"].append(len(rows))
            events["doc"] += rows
            events["score"] += [r[3] for r in event["rows"]]
            events["selected"] += [r[4] for r in event["rows"]]
        meta.append((cell["fingerprint"], model, generation.get("engine"), generation["method"],
                     generation.get("condition"), fields["keyword"], fields["prompt_id"], prompt_axis[fields["prompt_id"]]))
        queries.append(parsed["queries"])
        record = prompts.setdefault(fields["prompt_id"], {"prompt_text": cell.get("prompt_text"), "trace_sha256": set()})
        record["trace_sha256"].add(cell["trace"].get("user_prompt_sha256"))
        for engine, query, digest in parsed["searches"]:
            if hashlib.sha256(f"{engine}\0{query}".encode()).hexdigest().startswith(FIDELITY_PREFIX):
                searches[(engine, query)].add(digest)
        snapshots |= parsed["snapshots"]
        for name, passed in parsed["checks"].items():
            counts[f"check_failed:{name}"] += not passed
        counts[f"answers_{model}"] += 1
    as_int = lambda values: np.asarray(values, np.int64)
    return {"meta": meta, "queries": queries, "prompts": prompts, "searches": dict(searches), "snapshots": snapshots,
            "counts": counts, "sizes": {k: as_int(v) for k, v in sizes.items()}, "docs": {k: as_int(v) for k, v in docs.items()},
            "events": {"answer": as_int(events["answer"]), "rows": as_int(events["rows"]), "doc": as_int(events["doc"]),
                       "score": np.asarray(events["score"], float), "selected": as_int(events["selected"])}}


def trace_extract(args) -> int:
    prompt_axis = readiness.axis_by_prompt(args.final_axis_map)
    doc_index = corpus_index(args.corpus_package)
    population = {str(r["candidate_id"]): r for r in readiness.read_jsonl(args.population_prompts)}
    workers = max(1, args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    partial = readiness.new_directory(args.output)
    codes = {kind: {} for kind in ("model", "engine", "method", "condition", "keyword", "prompt")}
    columns, generation_ids, query_ids = defaultdict(list), [], {}
    parts = defaultdict(list)
    prompts, searches, snapshots, counts = {}, defaultdict(set), defaultdict(set), Counter()
    inputs, results = readiness.parallel_chunks(args.source, _trace_chunk, (prompt_axis, doc_index), workers)
    for number, total, chunk in results:
        base = len(generation_ids)
        for (generation_id, *labels, x), queries in zip(chunk["meta"], chunk["queries"]):
            generation_ids.append(generation_id)
            for kind, value in zip(("model", "engine", "method", "condition", "keyword", "prompt"), labels):
                columns[kind].append(codes[kind].setdefault(value, len(codes[kind])))
            columns["x"].append(x)
            parts["q_sizes"].append(len(queries))
            columns["q_ids"] += [query_ids.setdefault(q, len(query_ids)) for q in queries]
        for name in ("ret", "cand", "pool", "rank"):
            parts[f"{name}_sizes"].append(chunk["sizes"].get(name, np.zeros(0, np.int64)))
            parts[f"{name}_docs"].append(chunk["docs"].get(name, np.zeros(0, np.int64)))
        parts["event_answer"].append(chunk["events"]["answer"] + base)
        for key in ("rows", "doc", "score", "selected"):
            parts[f"event_{key}"].append(chunk["events"][key])
        for prompt_id, record in chunk["prompts"].items():
            merged = prompts.setdefault(prompt_id, {"prompt_text": record["prompt_text"], "trace_sha256": set()})
            merged["trace_sha256"] |= record["trace_sha256"]
        for key, digests in chunk["searches"].items():
            searches[key] |= digests
        for engine, path, sha256 in chunk["snapshots"]:
            snapshots[engine].add((path, sha256))
        counts.update(chunk["counts"])
        print(json.dumps({"chunk": number, "of": total, "answers": len(generation_ids), "time": readiness.now()}), flush=True)
    if any(len({sha for _, sha in found}) != 1 for found in snapshots.values()):
        raise ValueError(f"traces used more than one snapshot for an engine: {dict(snapshots)}")
    if not generation_ids:
        raise ValueError("no answers passed the checks")
    joined = {name: np.concatenate(parts[name]) for name in parts if name != "q_sizes"}
    offsets = lambda sizes: np.r_[0, np.cumsum(sizes)].astype(np.int64)
    prompt_rows = []
    for prompt_id in sorted(codes["prompt"], key=codes["prompt"].get):
        record, known = prompts[prompt_id], population.get(prompt_id, {})
        hashes = record["trace_sha256"]
        text = next((t for t in (record["prompt_text"], known.get("question"))
                     if isinstance(t, str) and len(hashes) == 1 and hashlib.sha256(t.encode()).hexdigest() in hashes), None)
        counts["prompts_without_verified_text"] += text is None
        counts["prompts_with_conflicting_trace_hashes"] += len(hashes) > 1
        counts["prompts_without_keyword_text"] += not known.get("keyword")
        prompt_rows.append({"prompt_id": prompt_id, "prompt_text": text, "keyword_text": known.get("keyword")})
    readiness.write_npz(partial / "stages.npz", **{k: np.asarray(columns[k], float if k == "x" else np.int64)
                                                   for k in ("model", "engine", "method", "condition", "keyword", "prompt", "x")},
                        **{f"{n}_offsets": offsets(joined[f"{n}_sizes"]) for n in ("ret", "cand", "pool", "rank")},
                        **{f"{n}_docs": joined[f"{n}_docs"] for n in ("ret", "cand", "pool", "rank")},
                        q_offsets=offsets(parts["q_sizes"]), q_ids=np.asarray(columns["q_ids"], np.int64),
                        event_answer=joined["event_answer"], event_row_offsets=offsets(joined["event_rows"]),
                        event_doc=joined["event_doc"], event_score=joined["event_score"], event_selected=joined["event_selected"])
    readiness.write_json(partial / "codes.json", {kind: sorted(values, key=values.get) for kind, values in codes.items()})
    readiness.write_jsonl(partial / "prompts.jsonl.gz", prompt_rows)
    readiness.write_jsonl(partial / "answers.jsonl.gz", ({"answer": i, "generation_id": g} for i, g in enumerate(generation_ids)))
    readiness.write_jsonl(partial / "queries.jsonl.gz", ({"page_id": h, "text": q, "text_sha256": h}
                                                         for q, h in ((q, hashlib.sha256(q.encode()).hexdigest())
                                                                      for q in sorted(query_ids, key=query_ids.get))))
    sample = [{"engine": e, "query": q, "rows_sha256": next(iter(d))} for (e, q), d in sorted(searches.items()) if len(d) == 1]
    readiness.write_jsonl(partial / "fidelity-sample.jsonl.gz", sample)
    checks = {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("check_failed:")}
    sizes = {n: float(np.mean(joined[f"{n}_sizes"])) for n in ("ret", "cand", "pool", "rank")}
    readiness.write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "stage": "trace-extract", "created_at": readiness.now(),
        "git_commit": readiness.git_commit(), "inputs": inputs, "final_axis_map": readiness.identity(args.final_axis_map),
        "corpus": readiness.identity(args.corpus_package / "manifest.json"),
        "population_prompts": readiness.identity(args.population_prompts),
        "snapshots": {engine: {"path": sorted(found)[0][0], "sha256": sorted(found)[0][1]} for engine, found in snapshots.items()},
        "page_rule": "corpus row of sha256(title.strip() + newline + text.strip())",
        "counts": {**{k: v for k, v in counts.items() if not k.startswith("check_failed:")}, "answers": len(generation_ids),
                   "prompts": len(codes["prompt"]), "unique_queries": len(query_ids), "query_uses": len(columns["q_ids"]),
                   "compaction_events": int(len(joined["event_answer"])), "scored_candidates": int(len(joined["event_doc"])),
                   "fidelity_sample": len(sample), "fidelity_sample_conflicting_recordings": sum(len(d) > 1 for d in searches.values())},
        "consistency_failures": checks, "consistency_passed": not any(checks.values()),
        "mean_pages_per_answer": {"retrieved_distinct": sizes["ret"], "candidates_distinct": sizes["cand"],
                                  "presented": sizes["pool"], "ranked": sizes["rank"]}})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"answers": len(generation_ids), "unique_queries": len(query_ids), "consistency_failures": checks}), flush=True)
    return 0


# ---------------------------------------------------------------- replay

_ADAPTERS: dict = {}


def _replayed_rows(job):
    from analysis.interpretability.pipeline.agentic_search import SEARCH_RESULT_LIMIT
    engine, text = job
    return [(r["url"], r["title"], r["snippet"]) for r in _ADAPTERS[engine]._select_rows(text, SEARCH_RESULT_LIMIT)]


def replay(args) -> int:
    """Rerun the frozen search (the adapter's own selection code) after a fidelity gate."""
    from analysis.scripts import audit_snapshot_glued_rows as snapshot_files
    from analysis.scripts.run_agentic_search_integration_smoke import FrozenSnapshotSearchAdapter

    refuse_existing(args.output)
    manifest = json.loads((args.trace_extract / "manifest.json").read_text())
    recorded = manifest["snapshots"]
    files = dict(str(spec).split("=", 1) for spec in args.snapshot)
    for engine, record in recorded.items():
        if engine not in files:
            files[engine] = snapshot_files.resolve_snapshot(record, args.search_root or Path.cwd(), roots=args.dataset_root)
    _ADAPTERS.clear()
    for engine, path in sorted(files.items()):
        adapter = FrozenSnapshotSearchAdapter(engine, Path(path))
        if engine not in recorded or adapter.snapshot_sha256 != recorded[engine]["sha256"]:
            raise ValueError(f"{engine} snapshot {path} is not the one the traces used")
        _ADAPTERS[engine] = adapter
    workers = max(1, args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    sample = readiness.read_jsonl(args.trace_extract / "fidelity-sample.jsonl.gz")
    replayed = _map(_replayed_rows, [(s["engine"], s["query"]) for s in sample], workers)
    mismatched = [s for s, rows in zip(sample, replayed) if stages.rows_digest(rows) != s["rows_sha256"]]
    gate = {"recorded_searches_replayed": len(sample), "identical": len(sample) - len(mismatched),
            "passed": bool(sample) and not mismatched, "mismatch_examples": mismatched[:5]}
    print(json.dumps({"fidelity_gate": {k: v for k, v in gate.items() if k != "mismatch_examples"}}), flush=True)
    if not gate["passed"]:
        output = Path(args.output).resolve()
        readiness.write_json(output.with_name(output.name + ".fidelity-failed.json"), gate)
        return 2
    doc_index = corpus_index(args.corpus_package)
    prompts = readiness.read_jsonl(args.trace_extract / "prompts.jsonl.gz")
    keywords = sorted({p["keyword_text"] for p in prompts if p["keyword_text"]})
    jobs = sorted({(e, p["prompt_text"]) for e in _ADAPTERS for p in prompts if p["prompt_text"]}
                  | {(e, k) for e in _ADAPTERS for k in keywords})
    found = dict(zip(jobs, _map(_replayed_rows, jobs, workers)))
    counts = Counter()

    def docs(engine, text):
        rows = [doc_index.get(ordering.page_id(ordering.page_text(title, snippet))) for _, title, snippet in found[(engine, text)]]
        counts["pages_not_in_corpus"] += sum(d is None for d in rows)
        return [d for d in rows if d is not None]

    out = []
    for engine in sorted(_ADAPTERS):
        out += [{"engine": engine, "kind": "prompt", "key": p["prompt_id"], "docs": docs(engine, p["prompt_text"])}
                for p in prompts if p["prompt_text"]]
        out += [{"engine": engine, "kind": "keyword", "key": k, "docs": docs(engine, k)} for k in keywords]
    counts.update(prompt_replays=sum(r["kind"] == "prompt" for r in out), keyword_replays=sum(r["kind"] == "keyword" for r in out),
                  prompts_without_verified_text=sum(not p["prompt_text"] for p in prompts))
    partial = readiness.new_directory(args.output)
    readiness.write_jsonl(partial / "replay.jsonl.gz", out)
    readiness.write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "stage": "replay", "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "trace_extract": readiness.identity(args.trace_extract / "manifest.json"),
        "snapshots": {e: {"path": str(a.path), "sha256": a.snapshot_sha256, "rows": a.snapshot_rows} for e, a in _ADAPTERS.items()},
        "selection": "FrozenSnapshotSearchAdapter._select_rows (deterministic-lexical-v1), top 20",
        "fidelity_gate": gate, "counts": dict(counts)})
    partial.rename(Path(args.output).resolve())
    print(json.dumps(dict(counts)), flush=True)
    return 0


# ---------------------------------------------------------------- analyze

def load_stages(directory: Path):
    with np.load(directory / "stages.npz", allow_pickle=False) as data:
        st = stages.Stages(*(data[name] for name in STAGE_ARRAYS))
        ev = stages.Events(data["event_answer"], data["event_row_offsets"], data["event_doc"], data["event_score"],
                           data["event_selected"])
    return st, ev, json.loads((directory / "codes.json").read_text())


def load_corpus(package: Path) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(package / "snippets.parquet", columns=[
        "row", "keywords", "consensus_axis_1_z", "prompt_scale_percentile_0_1", "qwen_axis_1_z",
        "mistral_aligned_axis_1_z", "outside_prompt_range"]).to_pydict()
    if table["row"] != list(range(len(table["row"]))):
        raise ValueError("corpus rows are not in vector order")
    return table


def view_agreement(qwen, mistral, outside) -> dict:
    qwen, mistral = np.asarray(qwen, float), np.asarray(mistral, float)
    enough = len(qwen) > 2
    return {"texts": int(len(qwen)), "pearson": float(np.corrcoef(qwen, mistral)[0, 1]) if enough else None,
            "spearman": ordering._spearman(qwen, mistral) if enough else None,
            "share_outside_prompt_range": float(np.mean(outside)) if len(outside) else None}


def text_coordinates(qwen_dir: Path, mistral_dir: Path, args, ids) -> tuple[np.ndarray, dict]:
    """Consensus z of embedded texts (queries, answers) in the order of ``ids``, and view agreement."""
    found, views, _, outside = readiness.page_coordinates(qwen_dir, mistral_dir, args.battery, args.final_axis_map)
    z = np.asarray([views["consensus"].get(i, np.nan) for i in ids], float)
    return z, {**view_agreement([views["qwen"][i] for i in found], [views["mistral"][i] for i in found],
                                [outside[i] for i in found]), "requested": len(ids), "missing": int(np.isnan(z).sum())}


def answer_values(args, st, codes) -> tuple[np.ndarray | None, dict | None]:
    """Answer-text z joined by (model, prompt, method, engine, condition); NaN where not embedded."""
    if not args.answers_index:
        return None, None
    index = {}
    for row in readiness.read_jsonl(args.answers_index):
        index.setdefault(tuple(str(row[k]) for k in ANSWER_KEYS), str(row[args.answers_id_field]))
    keys = [(codes["model"][m], codes["prompt"][p], codes["method"][s], codes["engine"][e], codes["condition"][c])
            for m, p, s, e, c in zip(st.model, st.prompt, st.method, st.engine, st.condition)]
    ids = [index.get(k) for k in keys]
    wanted = sorted({i for i in ids if i is not None})
    z, validity = text_coordinates(args.answers_qwen, args.answers_mistral, args, wanted)
    by_id = dict(zip(wanted, z))
    values = np.asarray([by_id.get(i, np.nan) if i is not None else np.nan for i in ids], float)
    return values, {**validity, "answers_joined": int(np.isfinite(values).sum()), "answers_index_rows": len(index)}


def replay_values(directory: Path, st, codes, prompts, doc_z) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = {}
    for row in readiness.read_jsonl(directory / "replay.jsonl.gz"):
        mean[(row["engine"], row["kind"], row["key"])] = float(np.mean(doc_z[row["docs"]])) if row["docs"] else np.nan
    keyword_text = [p["keyword_text"] for p in prompts]
    r0 = np.asarray([mean.get((codes["engine"][e], "prompt", codes["prompt"][p]), np.nan) for e, p in zip(st.engine, st.prompt)])
    rk = np.asarray([mean.get((codes["engine"][e], "keyword", keyword_text[p]), np.nan) for e, p in zip(st.engine, st.prompt)])
    return r0, rk, json.loads((directory / "manifest.json").read_text())


def pair_features(st, ev, rows: np.ndarray, codes, corpus, prompts, args):
    """On-keyword flag and intent-free topic similarity of every presented page and of the scored
    candidates ``rows``, relative to the answer's prompt."""
    n_docs = len(corpus["row"])
    row_answer = ev.answer[stages.owner(ev.row_offsets)][rows]
    pair_prompt = np.r_[st.prompt[stages.owner(st.pool_offsets)], st.prompt[row_answer]].astype(np.int64)
    pair_doc = np.r_[st.pool_docs, ev.doc[rows]].astype(np.int64)
    unique, inverse = np.unique(pair_prompt * n_docs + pair_doc, return_inverse=True)
    topic, diagnostics = pair_topic_similarity(
        np.stack([unique // n_docs, unique % n_docs], axis=1), codes["prompt"],
        {"qwen": args.qwen_prompts, "mistral": args.mistral_prompts},
        {view: args.corpus_package / readiness.EMBEDDING_FILES[view] for view in VIEWS},
        {"qwen": args.qwen_map, "mistral": args.mistral_map})
    topic = topic[inverse.reshape(-1)]
    keyword_code = {}
    members = np.unique(np.asarray([keyword_code.setdefault(k.casefold(), len(keyword_code)) * n_docs + d
                                    for d, keywords in enumerate(corpus["keywords"]) for k in (keywords or [])], np.int64))
    prompt_keyword = np.asarray([keyword_code.get((p["keyword_text"] or "").casefold(), -1) for p in prompts], np.int64)
    keyword = prompt_keyword[pair_prompt]
    on = (keyword >= 0) & np.isin(keyword * n_docs + pair_doc, members)
    split = len(st.pool_docs)
    diagnostics["unique_prompt_page_pairs"] = int(len(unique))
    return (on[:split].astype(float), topic[:split]), (on[split:].astype(float), topic[split:]), diagnostics


def strata_masks(st, codes) -> tuple[dict, dict, dict]:
    primary, by_method, labels = {}, {}, {}
    for m, model in enumerate(codes["model"]):
        for e, engine in enumerate(codes["engine"]):
            mask = (st.model == m) & (st.engine == e)
            if not mask.any():
                continue
            primary[f"{model} · {engine}"] = mask
            labels[f"{model} · {engine}"] = {"model": model, "engine": engine, "method": "all"}
            for s, method in enumerate(codes["method"]):
                if (mask & (st.method == s)).any():
                    name = f"{model} · {engine} · {method.split('-')[0]}"
                    by_method[name] = mask & (st.method == s)
                    labels[name] = {"model": model, "engine": engine, "method": method}
    return primary, by_method, labels


def ranking_change(directories) -> list[dict]:
    out = []
    for directory in directories:
        report = json.loads((Path(directory) / "report.json").read_text())
        out.append({"source": str(directory), "observations": report.get("observations"),
                    "summaries": [s for s in report["summaries"] if s["metric"] in RANKING_CHANGE_METRICS]})
    return out


def _correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None


def finite(value):
    """JSON-safe copy: non-finite numbers become null, numpy scalars become Python ones."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


class ResultCache:
    """Finished per-stratum results of a long analysis, reused by a rerun with the same code, inputs and
    settings, so a one-hour allocation that ends mid-analysis loses at most one stratum. Each set of
    code, inputs and settings gets its own folder; entries are written atomically."""

    def __init__(self, output: Path, key: dict):
        output = Path(output).resolve()
        fingerprint = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        self.directory = output.with_name(output.name + ".cache") / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        readiness.write_json(self.directory / "key.json", key)
        self.reused = 0

    def get(self, name: str, compute):
        path = self.directory / f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.json"
        if path.exists():
            self.reused += 1
            return json.loads(path.read_text())["value"]
        value = finite(compute())
        readiness.write_json(path, {"name": name, "value": value})
        return value


_STRATUM: dict = {}


def _stratum(mask):
    """Stage analysis of one stratum; runs in a forked worker."""
    s = _STRATUM
    return {**stages.analyse_stratum(s["st"], mask, s["values"], s["oracles"], draws=s["draws"], shuffles=s["shuffles"]),
            "dispersion": stages.dispersion(s["st"], s["doc_z"], mask)}


def analyze(args) -> int:
    refuse_existing(args.output)
    st, ev, codes = load_stages(args.trace_extract)
    extract_manifest = json.loads((args.trace_extract / "manifest.json").read_text())
    corpus = load_corpus(args.corpus_package)
    doc_z = np.asarray(corpus["consensus_axis_1_z"], float)
    doc_u = np.asarray(corpus["prompt_scale_percentile_0_1"], float)
    prompts = readiness.read_jsonl(args.trace_extract / "prompts.jsonl.gz")
    workers = max(1, args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    validity = {"pages": view_agreement(corpus["qwen_axis_1_z"], corpus["mistral_aligned_axis_1_z"], corpus["outside_prompt_range"])}
    query_z = answer_z = r0 = rk = None
    if args.queries_qwen:
        ids = [r["page_id"] for r in readiness.read_jsonl(args.trace_extract / "queries.jsonl.gz")]
        query_z, validity["queries"] = text_coordinates(args.queries_qwen, args.queries_mistral, args, ids)
    answer_z, answer_validity = answer_values(args, st, codes)
    if answer_validity:
        validity["answers"] = answer_validity
    replay_manifest = None
    if args.replay:
        r0, rk, replay_manifest = replay_values(args.replay, st, codes, prompts, doc_z)
    if args.relocation:
        relocation = json.loads(args.relocation.read_text())
        validity["relocation"] = {"passed": relocation.get("passed"), "archived_coordinates": relocation.get("archived_coordinates"),
                                  "fresh_reembedding_spearman": {k: v.get("spearman") for k, v in (relocation.get("fresh_reembedding") or {}).items()}}

    natural = st.condition == (codes["condition"].index(args.natural_condition) if args.natural_condition in codes["condition"] else -1)
    event_of_row = stages.owner(ev.row_offsets)
    rows = np.flatnonzero(natural[ev.answer[event_of_row]])  # scored candidates of natural answers
    candidate = SimpleNamespace(event=event_of_row[rows], answer=ev.answer[event_of_row[rows]], doc=ev.doc[rows],
                                score=ev.score[rows], selected=ev.selected[rows])
    (pool_on, pool_topic), (candidate.on_keyword, candidate.topic), topic_diagnostics = pair_features(
        st, ev, rows, codes, corpus, prompts, args)
    print(json.dumps({"pairs": topic_diagnostics, "time": readiness.now()}), flush=True)
    values = stages.stage_values(st, doc_z, query_z=query_z, answer_z=answer_z, r0=r0, rk=rk)
    oracles = stages.oracle_values(st, doc_z, doc_u, pool_topic)
    prompt_x, prompt_keyword = stages.prompt_table(st)
    draws = stages.keyword_draws(len(codes["keyword"]), args.bootstrap, args.seed)
    shuffles = stages.shuffle_draws(prompt_x, prompt_keyword, args.permutations, args.seed + 1)
    primary, by_method, labels = strata_masks(st, codes)
    every = {**primary, **by_method}

    _STRATUM.update(st=st, values=values, oracles=oracles, draws=draws, shuffles=shuffles, doc_z=doc_z)
    strata = {}
    for name, result in zip(every, _map(_stratum, list(every.values()), min(workers, len(every)))):
        strata[name] = {**labels[name], **result}
        print(json.dumps({"stratum": name, **{t: round(v["slope"], 4) for t, v in result["slopes"].items()
                                              if not t.startswith("oracle")}}), flush=True)

    # The reranker and the ranking step on natural answers, drivers on the candidates' scale.
    features = stages.driver_columns(doc_z[candidate.doc], doc_u[candidate.doc], st.x[candidate.answer],
                                     candidate.on_keyword, candidate.topic)
    stats = {name: geo._standardize(column)[1] for name, column in features.items()}
    standardized = {name: geo._standardize(column, stats[name])[0] for name, column in features.items()}
    validity["axis_topic_orthogonality"] = {
        "candidate_rows": int(len(rows)),
        "pearson_page_intent_vs_topic": _correlation(features["page_intent"], features["topic_similarity"]),
        "pearson_intent_alignment_vs_topic": _correlation(features["intent_alignment"], features["topic_similarity"]),
        "topic_similarity": topic_diagnostics}
    natural_index = np.flatnonzero(natural)
    natural_answers = geo.Answers(st.model, st.engine, st.keyword, st.prompt, st.x, st.pool_offsets, st.pool_docs, pool_on,
                                  pool_topic, st.rank_offsets, st.rank_docs, doc_z, doc_u).subset(natural)
    ranking_rows = geo.choice_rows(natural_answers)
    cache = ResultCache(args.output, {
        "git_commit": readiness.git_commit(), "trace_extract": readiness.identity(args.trace_extract / "manifest.json")["sha256"],
        "corpus": readiness.identity(args.corpus_package / "manifest.json")["sha256"],
        "prompt_vectors_and_maps": [str(Path(p).resolve()) for p in (args.qwen_prompts, args.mistral_prompts, args.qwen_map, args.mistral_map)],
        "settings": [args.bootstrap, args.permutations, args.seed, args.natural_condition]})

    def reranker(name, mask, seed):
        keep = np.flatnonzero(mask & natural)
        inside = mask[candidate.answer]
        selection = stages.selection_rows(candidate.event[inside], candidate.selected[inside],
                                          np.searchsorted(keep, candidate.answer[inside]), z=features["page_intent"][inside],
                                          u=doc_u[candidate.doc[inside]], on_keyword=candidate.on_keyword[inside],
                                          topic=candidate.topic[inside])
        within = np.searchsorted(natural_index, keep)
        return {"selection": geo.estimate_e2(
                    SimpleNamespace(x=st.x[keep], keyword=st.keyword[keep], prompt=st.prompt[keep]), bootstrap=args.bootstrap,
                    permutations=args.permutations, seed=seed, workers=workers, rows=selection, stats=stats),
                "ranking_step": geo.estimate_e2(
                    natural_answers.subset(np.isin(np.arange(len(natural_index)), within)), bootstrap=args.bootstrap,
                    permutations=args.permutations, seed=seed + 1, workers=workers, rows=geo.restrict(ranking_rows, within),
                    stats=stats),
                "score_drivers": stages.score_drivers(
                    candidate.event[inside], candidate.score[inside], {n: standardized[n][inside] for n in geo.DRIVERS},
                    st.keyword[candidate.answer[inside]], draws)}

    for number, (name, mask) in enumerate(every.items()):
        if not (mask & natural).any():
            continue
        strata[name].update(cache.get(f"reranker:{name}", lambda: reranker(name, mask, args.seed + 1009 * (number + 1))))
        print(json.dumps({"stratum": name, "shortlisting_odds_ratio_per_sd": {
            k: round(v["odds_ratio_per_sd"], 3) for k, v in strata[name]["selection"]["drivers"].items()},
            "reused_from_cache": cache.reused, "time": readiness.now()}), flush=True)

    terms = stages.available_terms(values)
    contrasts = {}
    if len(codes["model"]) == 2:
        for e, engine in enumerate(codes["engine"]):
            a, b = ((st.model == m) & (st.engine == e) for m in (0, 1))
            contrasts[f"{codes['model'][0]} − {codes['model'][1]} · {engine}"] = stages.contrast(st, a, b, values, terms, draws)
    if {stages.REACTIVE, stages.PARALLEL} <= set(codes["method"]):
        reactive, parallel = codes["method"].index(stages.REACTIVE), codes["method"].index(stages.PARALLEL)
        for name, mask in primary.items():
            contrasts[f"Reactive − Parallel · {name}"] = stages.contrast(st, mask & (st.method == reactive),
                                                                         mask & (st.method == parallel), values, terms, draws)
    first = {name: strata[name] for name in primary}
    replication = {"slopes": geo.replication(first, section="slopes", terms=[t for t in terms if t != "Rk"], x_terms=set(terms)),
                   "derived": geo.replication(first, section="derived", terms=DERIVED_REPLICATION, x_terms=set()),
                   "selection": geo.replication(first, section="selection", terms=geo.DRIVERS),
                   "ranking_step": geo.replication(first, section="ranking_step", terms=geo.DRIVERS)}
    verdicts = {}
    for name in primary:
        entry = strata[name]["derived"].get("pool_minus_reordering")
        verdicts[name] = {"pool_carries_more_than_reordering": bool(entry and entry["estimate"] > 0 and entry["ci95"][0] is not None
                                                                    and entry["ci95"][0] > 0)}
    null_anchor = max((abs(strata[n]["slopes"]["Rk"]["slope"]) for n in strata if "Rk" in strata[n]["slopes"]), default=None)
    # Descriptive curves for the paper figure: stage value by prompt-position bin, per model x engine and per model.
    # "z" holds every available stage on the consensus z scale; "u" the page stages on the prompt percentile scale.
    page_u = {k: v for k, v in stages.stage_values(st, doc_u).items() if k in ("R", "C", "P", "K")}
    groups = dict(primary)
    for m, model in enumerate(codes["model"]):
        groups[f"{model} · both engines"] = st.model == m
    curves = {"bins": 20, "note": "descriptive; natural condition and all conditions; keyword-clustered standard errors",
              "groups": {name: {condition: {"z": stages.stage_curves(st.x, st.keyword, values, mask & extra),
                                            "u": stages.stage_curves(st.x, st.keyword, page_u, mask & extra)}
                                for condition, extra in (("natural", natural), ("all", np.ones_like(natural)))}
                         for name, mask in groups.items()}}
    results = {
        "format_version": FORMAT_VERSION, "stage": "analyze", "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "inputs": {"trace_extract": readiness.identity(args.trace_extract / "manifest.json"),
                   "corpus": readiness.identity(args.corpus_package / "manifest.json"),
                   "replay": readiness.identity(args.replay / "manifest.json") if args.replay else None,
                   "queries": {v: readiness.identity(d / "projection_manifest.json") for v, d in
                               (("qwen", args.queries_qwen), ("mistral", args.queries_mistral)) if d},
                   "answers_index": readiness.identity(args.answers_index) if args.answers_index else None,
                   "final_axis_map": readiness.identity(args.final_axis_map)},
        "settings": {"bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed,
                     "reranker_condition": args.natural_condition, "driver_standardization": {k: list(v) for k, v in stats.items()}},
        "coverage": {"trace_extract": extract_manifest["counts"], "consistency_failures": extract_manifest["consistency_failures"],
                     "consistency_passed": extract_manifest["consistency_passed"],
                     "mean_pages_per_answer": extract_manifest["mean_pages_per_answer"],
                     "replay": replay_manifest and {"fidelity_gate": replay_manifest["fidelity_gate"], "counts": replay_manifest["counts"]},
                     "stages_available": [t for t in stages.STAGES if t in values], "null_anchor_max_abs_slope": null_anchor},
        "validity": validity, "ranking_change": ranking_change(args.ranking_change), "strata": strata,
        "contrasts": contrasts, "replication": replication, "verdicts": verdicts, "curves": curves}
    results = finite(results)
    partial = readiness.new_directory(args.output)
    readiness.write_json(partial / "results.json", results)
    (partial / "final-report.html").write_text(render(results), encoding="utf-8")
    partial.rename(Path(args.output).resolve())
    print(f"REPORT {Path(args.output).resolve() / 'final-report.html'}", flush=True)
    return 0


# ---------------------------------------------------------------- report

STAGE_LABELS = {"Q": "AI queries", "R0": "search with the prompt's own text (replay)", "Rk": "search with the bare keyword (replay, null anchor)",
                "R": "retrieved by the AI's searches", "C": "reranker candidates", "P": "shortlist shown to the model",
                "K": "final ranking (top-weighted)", "A": "answer text", "C-R": "deduplication and condition",
                "P-C": "reranker selection", "K-P": "reordering by the model", "R-R0": "query rewriting"}
LADDER = ("Q", "R0", "R", "C", "P", "K", "A")


def _f(value, digits=3):
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:+.{digits}f}"


def _num(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _share(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.0%}"


def _ci(ci, share=False):
    if not ci or ci[0] is None:
        return "—"
    return f"[{_share(ci[0])}, {_share(ci[1])}]" if share else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _p(value) -> str:
    return "" if value is None else f" · p {value:.3f}"


def _value(entry, share=False) -> str:
    """Estimate with its interval and permutation p, for a table cell."""
    if not entry:
        return "—"
    value = entry.get("slope", entry.get("estimate", entry.get("difference")))
    shown = _share(value) if share else _f(value)
    return f"{shown}<br><small>{_ci(entry.get('ci95'), share)}{_p(entry.get('permutation_p'))}</small>"


def _cell(entry, share=False) -> str:
    return f"<td>{_value(entry, share)}</td>"


def _driver_cell(driver: dict) -> str:
    share = _share(driver["fit_share"] or 0)
    return (f"<td>{driver['odds_ratio_per_sd']:.2f}<br><small>{odds_interval(driver['ci95'])} · share {share}"
            f"{_p(driver['permutation_p'])}</small></td>")


def _replication_cell(row: dict) -> str:
    return f"<td>{_f(row['estimate'])}<br><small>{_ci(row['ci95'])}{_p(row['p'])}</small></td>"


def _dispersion_row(name: str, block: dict) -> str:
    cells = "".join(f"<td>{_share(block['shares'][k])}</td>"
                    for k in ("within_pool", "between_pools_same_keyword", "between_keywords"))
    return f"<tr><td>{html.escape(name)}</td>{cells}<td>{_num(block['mean_within_pool_sd'], 3)}</td></tr>"


def _table(head, rows) -> str:
    return (f"<div class='scroll'><table><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _ladder(strata: dict) -> str:
    names = [n for n, s in strata.items() if s["method"] == "all"]
    stages_shown = [s for s in LADDER if any(s in strata[n]["slopes"] for n in names)]
    points = [strata[n]["slopes"][s]["slope"] for n in names for s in stages_shown if s in strata[n]["slopes"]]
    if not points:
        return ""
    low, high = min(points + [0.0]), max(points + [0.0])
    span = (high - low) or 1.0
    width, height, pad = 640, 260, 46
    x = lambda i: pad + i * (width - 2 * pad) / max(len(stages_shown) - 1, 1)
    y = lambda v: height - pad - (v - low) / span * (height - 2 * pad)
    lines = []
    for index, name in enumerate(names):
        pts = [(x(i), y(strata[name]["slopes"][s]["slope"])) for i, s in enumerate(stages_shown) if s in strata[name]["slopes"]]
        lines.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" class="s{index % 4}"/>'
                     + "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" class="d{index % 4}"/>' for a, b in pts)
                     + f'<text x="{width - pad + 6}" y="{pts[-1][1] + 4:.1f}" class="lab t{index % 4}">{html.escape(name)}</text>')
    ticks = "".join(f'<text x="{x(i):.1f}" y="{height - 18}" text-anchor="middle" class="ax">{s}</text>' for i, s in enumerate(stages_shown))
    title = "Within-keyword slope of intent on the prompt position at each point of the pipeline"
    return (f'<figure><div class="scroll"><svg viewBox="0 0 {width + 170} {height}" role="img" aria-label="{title}">'
            f'<line x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>{"".join(lines)}{ticks}'
            f'<text x="4" y="{y(high) + 4:.1f}" class="ax">{high:+.2f}</text><text x="4" y="{y(low):.1f}" class="ax">{low:+.2f}</text>'
            f'</svg></div><figcaption>{title}. {", ".join(f"{s}: {html.escape(STAGE_LABELS[s])}" for s in stages_shown)}.'
            f'</figcaption></figure>')


def render(results: dict) -> str:
    strata, primary = results["strata"], [n for n, s in results["strata"].items() if s["method"] == "all"]
    head = ["", *map(html.escape, primary)]

    def row(label, getter, share=False):
        return f"<tr><td>{label}</td>{''.join(_cell(getter(strata[n]), share) for n in primary)}</tr>"

    derived = lambda key: (lambda s: s["derived"].get(key))
    verdict_rows = []
    for n in primary:
        shown = results["verdicts"][n]["pool_carries_more_than_reordering"]
        retrieval = (strata[n]["derived"].get("increment:retrieval") or {}).get("estimate")
        verdict_rows.append(f"<tr><td>{html.escape(n)}</td><td>{_f(retrieval)}</td>"
                            f"{_cell(strata[n]['derived'].get('pool_minus_reordering'))}"
                            f"<td class='{'yes' if shown else 'no'}'>{'pool &gt; reordering' if shown else 'not shown'}</td></tr>")
    slope_rows = [row(f"{s} · {html.escape(STAGE_LABELS[s])}", lambda st, s=s: st["slopes"].get(s))
                  for s in (*stages.STAGES, *stages.INCREMENTS) if any(s in strata[n]["slopes"] for n in primary)]
    chain_rows = [row(label, derived(key), share=key.startswith("share") or key == "mediated_by_pool")
                  for key, label in (("increment:retrieval", "β_R retrieval (AI queries → frozen search)"),
                                     ("increment:deduplication_and_condition", "β_C − β_R deduplication and condition"),
                                     ("increment:reranker", "β_P − β_C reranker selection"),
                                     ("increment:reordering", "β_K − β_P reordering by the model"),
                                     ("share_of_ranking:pool", "share of β_K already in the pool, β_P / β_K"),
                                     ("share_of_ranking:reordering", "share of β_K from reordering"),
                                     ("pool_minus_reordering", "β_P − (β_K − β_P)"),
                                     ("mediated_by_pool", "share of β_K mediated by the pool (regression)"))]
    split_rows = [row(label, derived(key), share=True)
                  for key, label in (("share_of_pool:retrieval", "retrieval: the AI's queries through the frozen search"),
                                     ("share_of_pool:prompt_words", "… of which the prompt's own words (replay)"),
                                     ("share_of_pool:query_rewriting", "… of which query rewriting by the AI"),
                                     ("share_of_pool:deduplication_and_condition", "deduplication and condition"),
                                     ("share_of_pool:reranker", "reranker selection"))]
    oracle_rows = [row(label, derived(key), share=key == "utilization")
                   for key, label in (("oracle_reordering:intent", "intent oracle − pool (closest on the axis first)"),
                                      ("oracle_reordering:topic", "topic oracle − pool (most topical first)"),
                                      ("oracle_reordering:presented_order", "presented order − pool"),
                                      ("increment:reordering", "observed ranking − pool"),
                                      ("utilization", "utilization U = observed / intent oracle"))]
    dispersion_rows = [_dispersion_row(n, strata[n]['dispersion']) for n in primary]

    def driver_table(section):
        out = []
        for name, s in strata.items():
            block = s.get(section)
            if not block or not block.get("drivers"):
                continue
            cells = "".join(_driver_cell(d) for d in block["drivers"].values())
            out.append(f"<tr><td>{html.escape(name)}</td><td>{block['answers']:,}</td>{cells}</tr>")
        return _table(["stratum", "answers", *(d.replace("_", " ") + "<br><small>odds ratio per SD</small>" for d in geo.DRIVERS)], out)

    score_rows = []
    for name, s in strata.items():
        block = s.get("score_drivers")
        if block:
            score_rows.append(f"<tr><td>{html.escape(name)}</td><td>{block['events']:,}</td><td>{_num(block['within_event_r2'])}</td>"
                              f"<td>{_num(block['within_event_score_sd'])}</td>"
                              + "".join(f"<td>{_f(d['score_per_sd'])}<br><small>{_ci(d['ci95'])}</small></td>"
                                        for d in block["drivers"].values()) + "</tr>")
    score_head = ["stratum", "events", "within-event R²", "within-event score SD",
                  *(d.replace("_", " ") + "<br><small>score change per driver SD</small>" for d in geo.DRIVERS)]
    answer_rows = [row(label, getter, share=share) for label, getter, share in (
        ("β_A answer intent on prompt position", lambda s: s["slopes"].get("A"), False),
        ("answer intent on ranking intent (within keyword)", derived("answer_on_ranking"), False),
        ("share of β_A mediated by the ranking", derived("answer_mediated_by_ranking"), True))]
    contrast_rows = []
    for name, block in results["contrasts"].items():
        for term in ("P", "K-P", "R", "P-C", "Q", "A"):
            if term in block:
                contrast_rows.append(f"<tr><td>{html.escape(name)}</td><td>{term} · {html.escape(STAGE_LABELS[term])}</td>{_cell(block[term])}</tr>")
    replication_rows = []
    for section, title in (("slopes", "stage slope"), ("derived", "chain quantity"),
                           ("selection", "shortlisting driver (log-odds per SD)"), ("ranking_step", "ranking-step driver (log-odds per SD)")):
        for term, entry in results["replication"][section].items():
            replication_rows.append(f"<tr><td>{title}: {html.escape(STAGE_LABELS.get(term, term).replace('_', ' '))}</td>"
                                    f"<td class='{'yes' if entry['replicates'] else 'no'}'>{'replicates' if entry['replicates'] else 'not replicated'}</td>"
                                    + "".join(_replication_cell(r) for r in entry["strata"]) + "</tr>")
    validity = results["validity"]
    validity_rows = [f"<tr><td>{kind}</td><td>{v.get('texts', 0):,}</td><td>{_f(v.get('pearson'))}</td><td>{_f(v.get('spearman'))}</td>"
                     f"<td>{_share(v.get('share_outside_prompt_range'))}</td></tr>" for kind, v in validity.items()
                     if kind in ("pages", "queries", "answers")]
    orthogonality = validity.get("axis_topic_orthogonality", {})
    relocation = validity.get("relocation")
    change_rows = []
    for report in results["ranking_change"]:
        groups = defaultdict(list)
        for s in report["summaries"]:
            if s.get("median_within_keyword_rho") is not None:
                groups[(s["model"], s["metric"])].append(s["median_within_keyword_rho"])
        change_rows += [f"<tr><td>{html.escape(m)}</td><td>{html.escape(metric)}</td><td>{_f(float(np.mean(v)))}</td>"
                        f"<td>{_f(min(v))} … {_f(max(v))}</td><td>{len(v)}</td></tr>" for (m, metric), v in sorted(groups.items())]
    coverage = results["coverage"]
    gate = (coverage.get("replay") or {}).get("fidelity_gate")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Intent Surfacing Study</title>
<style>
:root{{--bg:#f7f7f4;--fg:#1c1f22;--muted:#5b6268;--rule:#d7dad6;--yes:#1d6b45;--no:#9a3412;--panel:#eef0ec;
--s0:#2f5d8a;--s1:#b8572a;--s2:#5b7f2e;--s3:#7a4b9c;--font:"Source Sans 3","Helvetica Neue",Arial,sans-serif}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181b;--fg:#e8e9e6;--muted:#a3a9ae;--rule:#343a40;--yes:#6fcf97;--no:#f4a261;--panel:#1f2226;--s0:#8cb8e6;--s1:#f0a07a;--s2:#a9cf6f;--s3:#c9a3e6;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#16181b;--fg:#e8e9e6;--muted:#a3a9ae;--rule:#343a40;--yes:#6fcf97;--no:#f4a261;--panel:#1f2226;--s0:#8cb8e6;--s1:#f0a07a;--s2:#a9cf6f;--s3:#c9a3e6;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 var(--font);margin:0 auto;max-width:1080px;padding-inline:16px;padding-block:28px 56px}}
h1,h2,h3{{line-height:1.2;text-wrap:balance}} .note,small,figcaption{{color:var(--muted)}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:8px 0 18px}}
th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}}
.yes{{color:var(--yes);font-weight:600}} .no{{color:var(--no);font-weight:600}}
.math{{background:var(--panel);padding:12px 16px;border-radius:6px}} .math p{{margin:6px 0}}
svg{{width:100%;height:auto;min-width:600px}} polyline{{fill:none;stroke-width:2}} .zero{{stroke:var(--muted);stroke-dasharray:4 4}}
.ax,.lab{{font-size:11px;fill:var(--muted)}} .s0{{stroke:var(--s0)}} .s1{{stroke:var(--s1)}} .s2{{stroke:var(--s2)}} .s3{{stroke:var(--s3)}}
.d0{{fill:var(--s0)}} .d1{{fill:var(--s1)}} .d2{{fill:var(--s2)}} .d3{{fill:var(--s3)}} .t0{{fill:var(--s0)}} .t1{{fill:var(--s1)}} .t2{{fill:var(--s2)}} .t3{{fill:var(--s3)}}
</style></head><body>
<h1>Where does intent surface?</h1>
<p class="note">Observational, closed frozen corpus searched by a deterministic lexical retriever. Protocol fixed before this run
(analysis/docs/intent_surfacing_study.md, with its addendum on the shortlisting step). The prompt position x is a measured text
property, not a randomized treatment; positions of queries, pages and answers are out-of-domain descriptions on an axis fitted on
prompts. Generated {html.escape(results['created_at'])} from commit {html.escape(results['git_commit'][:12])};
keyword-bootstrap {results['settings']['bootstrap']} and within-keyword permutations {results['settings']['permutations']}.</p>

<h2>1. Is intent carried by the pool more than by the ranking?</h2>
{_table(["stratum", "β_R retrieval", "β_P − (β_K − β_P), 95% CI", "verdict"], verdict_rows)}
<p class="note">Verdict rule (protocol): the pool increment β_P exceeds the reordering increment β_K − β_P and their difference has a
95% interval excluding 0.</p>

<h2>2. How much does intent match raise a page's chance of being shortlisted?</h2>
<h3>Shortlisting step: reranker candidates → shortlist (natural condition)</h3>
{driver_table("selection")}
<h3>Ranking step on the same answers and the same driver scale: shortlist → ranking</h3>
{driver_table("ranking_step")}
<p class="note">Odds ratio of being picked next per standard deviation of each driver (SD over the reranker's candidates), holding the
other three drivers fixed: top-k Plackett–Luce over the reranker's kept candidates in score order (no position effects; the cross-encoder
scores each candidate on its own), and over the model's ranking with presented-position effects. Share = that driver's part of the fit
lost when it is dropped. p = within-keyword permutation p for intent alignment.</p>

<h2>3. Where does the pool's intent come from?</h2>
<h3>Split of the shortlist slope β_P</h3>
{_table(head, split_rows)}
<h3>What the reranker's score responds to (within compaction event)</h3>
{_table(score_head, score_rows)}
<p class="note">Parallel scores candidates against the prompt; Reactive against the AI's own query.</p>

<h2>4. Intent at every point of the pipeline</h2>
{_ladder(strata)}
{_table(head, slope_rows)}
<p class="note">Within-keyword slope of the stage's consensus intent z on x, in z units from the most informational to the most
action-ready prompt of a keyword; 95% keyword-bootstrap interval; within-keyword permutation p. The bare-keyword replay is identical
within keyword and engine, so its slope is an exact null anchor (largest |slope| {_f(coverage.get('null_anchor_max_abs_slope'), 6)}).</p>

<h2>5. The math: pool versus ranking</h2>
<div class="math">
<p><b>Slope operator.</b> For a stage value S per answer, β_S = Σ x̃ S̃ / Σ x̃², where ~ removes the keyword mean. It is linear in S.</p>
<p><b>Chain identity.</b> K = R + (C − R) + (P − C) + (K − P) per answer, so β_K = β_R + β_(C−R) + β_(P−C) + β_(K−P) exactly.
The first three terms are the pool β_P; the last is what the model's ordering adds.</p>
<p><b>Random-order baseline.</b> If the model ordered its pool at random, every rank would hold the pool mean in expectation, so the
expected ranking intent is P. β_(K−P) is therefore the intent the ordering adds beyond chance.</p>
<p><b>Oracles and utilization.</b> Re-ranking each pool with the same length by closeness on the axis (intent oracle) gives the most
intent the ordering could add; U = β_(K−P) / β_(oracle−P) is the share of that leverage the model uses. The topic oracle orders by
intent-free topic similarity: if it reproduces β_(K−P), topic explains the reordering.</p>
<p><b>Mediation.</b> K̃ = a·x̃ + b·P̃ + e within keyword; share mediated by the pool = 1 − a / β_K.</p>
<p><b>Dispersion bound.</b> Var(page z) = within pool + between pools of a keyword + between keywords. Reordering can only move
intent within a pool; the pool can move it between pools.</p>
<p><b>Shortlisting.</b> P(candidate j picked next) = exp(θ·d_j) / Σ over not-yet-picked candidates exp(θ·d_l); the odds ratio per SD is exp(θ).</p>
</div>
{_table(head, chain_rows)}
<h3>Oracles and utilization</h3>
{_table(head, oracle_rows)}
<h3>Within-pool dispersion of page intent</h3>
{_table(["stratum", "within pool", "between pools, same keyword", "between keywords", "mean within-pool SD (z)"], dispersion_rows)}

<h2>6. Answer stage (natural condition)</h2>
{_table(head, answer_rows)}

<h2>7. Contrasts</h2>
{_table(["contrast", "term", "difference, 95% CI"], contrast_rows)}

<h2>8. Replication across the four model × engine strata</h2>
{_table(["term", "verdict", *map(html.escape, primary)], replication_rows)}
<p class="note">Replicates = same sign, 95% interval excluding 0 and (stage slopes, intent alignment) permutation p &lt; 0.05 in all four strata.</p>

<h2>9. Instrument validity and coverage</h2>
{_table(["texts", "count", "Qwen vs Mistral Pearson", "Spearman", "outside prompt range"], validity_rows)}
<p>Axis–topic orthogonality over reranker candidates: page intent vs topic similarity r = {_f(orthogonality.get('pearson_page_intent_vs_topic'))};
intent alignment vs topic r = {_f(orthogonality.get('pearson_intent_alignment_vs_topic'))}.
Relocation: {html.escape(json.dumps(relocation)) if relocation else "not supplied"}.</p>
<p>Trace consistency (all must be 0): {html.escape(json.dumps(coverage['consistency_failures']))}.
Search replay fidelity: {html.escape(json.dumps(gate)) if gate else "no replay supplied"}.
Answers: {html.escape(json.dumps({k: coverage['trace_extract'].get(k) for k in ('answers', 'prompts', 'unique_queries')}))};
mean pages per answer {html.escape(json.dumps(coverage['mean_pages_per_answer']))}.</p>
<h3>Prompt axis → ranking change (pairs of prompts within keyword)</h3>
{_table(["model", "metric", "mean of median within-keyword ρ", "range over cells", "cells"], change_rows) if change_rows else "<p class='note'>Not supplied.</p>"}
</body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="stage", required=True)
    p = commands.add_parser("trace-extract")
    p.add_argument("--source", action="append", required=True, metavar="DATASET_ROOT:MODEL")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--corpus-package", type=Path, required=True, help="snippet-embeddings-corpus-v1")
    p.add_argument("--population-prompts", type=Path, required=True, help="population-prompts.jsonl (keyword text, question)")
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("replay")
    p.add_argument("--trace-extract", type=Path, required=True)
    p.add_argument("--corpus-package", type=Path, required=True)
    p.add_argument("--snapshot", action="append", default=[], metavar="ENGINE=PATH")
    p.add_argument("--dataset-root", type=Path, action="append", default=[], help="datasets whose shared inputs mirror the snapshots")
    p.add_argument("--search-root", type=Path, help="where to look for a snapshot with the recorded SHA-256")
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("analyze")
    p.add_argument("--trace-extract", type=Path, required=True)
    p.add_argument("--corpus-package", type=Path, required=True)
    p.add_argument("--qwen-prompts", type=Path, required=True, help="final-audit/projections/qwen")
    p.add_argument("--mistral-prompts", type=Path, required=True)
    p.add_argument("--qwen-map", type=Path, required=True)
    p.add_argument("--mistral-map", type=Path, required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--replay", type=Path)
    p.add_argument("--queries-qwen", type=Path, help="merged Qwen projections of queries.jsonl.gz")
    p.add_argument("--queries-mistral", type=Path)
    p.add_argument("--answers-index", type=Path, help="JSONL with model, prompt_id, method, engine, condition and an answer id")
    p.add_argument("--answers-id-field", default="answer_id")
    p.add_argument("--answers-qwen", type=Path)
    p.add_argument("--answers-mistral", type=Path)
    p.add_argument("--relocation", type=Path, help="relocation report with the fresh re-embedding check")
    p.add_argument("--ranking-change", type=Path, action="append", default=[], help="report_axis_ranking_change.py output")
    p.add_argument("--natural-condition", default="natural")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "analyze" and bool(args.queries_qwen) != bool(args.queries_mistral):
        parser.error("give both query views or neither")
    if args.stage == "analyze" and args.answers_index and not (args.answers_qwen and args.answers_mistral):
        parser.error("--answers-index needs --answers-qwen and --answers-mistral")
    return {"trace-extract": trace_extract, "replay": replay, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
