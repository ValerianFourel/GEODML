#!/usr/bin/env python3
"""Export the full run's processed traces as release tables: both generators, every keyword, all three conditions.

    python -m analysis.scripts.export_release_traces --run RUN --corpus-package DIR --battery DIR --final-axis-map FILE \
        --output DIR [--skip-trace] [--skip-funnel]

RUN is the full run's output folder. Read (never written): trace-extract/, intent-replay/, embed-queries-{qwen,mistral}.merged/,
embed-answers-{qwen,mistral}.merged/, answers-export/observations.jsonl.gz, funnel-extract/, funnel-assembled/, funnel-replay/
and, for the check, intent-stages/results.json. Writes OUTPUT once (atomically; refuses to overwrite):

  trace_answers.parquet   one row per answer of the trace extract: identity, prompt position x, keyword, split, page counts
                          per stage, and every per-answer stage value exactly as `intent_stages_study.py analyze` computes it:
                          Q, R0, Rk, R, C, P, K, A on the consensus-z scale and R_u, C_u, P_u, K_u on the prompt scale;
                          and the stage chain's values chain_R0 ... chain_K as `steelman/tables.py` computes them from the
                          funnel extract (page intent u of the snapshot rows: means over the prompt-text search, retrieved,
                          scored and shown rows; rank-weighted over the first L shown slots (I) and the L cited rows (K))
  queries.parquet         one row per distinct agent query: id, text, both views' axis-1 z, consensus z, prompt-scale position
  answer_queries.parquet  answer_id, search_index, query_id (the agent's queries of every answer, in search order)
  stage_items.parquet     funnel extract: per answer every retrieved snapshot row with its search, rank, scored flag,
                          shown slot and cited rank
  rerank_events.parquet   every cross-encoder candidate: event, answer, search, row, score, selection order, topic, on_keyword
  stage_u.parquet         the keyword's own rows per answer (U) with retrieved/scored/presented/ranked/replay flags, topic
  shown.parquet           shown rows per answer with slot, cited rank, topic, on_keyword
  figure_check.json       the stage curves of the paper's stage figure recomputed from trace_answers.parquet (read back from
                          disk) and compared with intent-stages/results.json: every bin mean, standard error and count
  manifest.json           inputs (sha256), counts, row counts, bytes and sha256 of every table, git commit

answer_id is the first 24 hex characters of sha256("model|prompt_id|engine|method|condition"), as in the release's
answers table. Large tables store answer_id dictionary-encoded. Exit 0 when the figure check passes (or is skipped
with --skip-trace), 3 when it fails (the tables are still written, for inspection).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.scripts import intent_stages_study as isd  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402
from analysis.scripts.funnel_study import exploration_keyword  # noqa: E402
from analysis.steelman import tables as chain_tables  # noqa: E402

FORMAT_VERSION = "release-traces-v1"
Z_TERMS = ("Q", "R0", "Rk", "R", "C", "P", "K", "A")
U_TERMS = ("R", "C", "P", "K")
CHAIN_TERMS = chain_tables.CHAIN


def answer_id(model: str, prompt_id: str, engine: str, method: str, condition: str) -> str:
    return hashlib.sha256(f"{model}|{prompt_id}|{engine}|{method}|{condition}".encode()).hexdigest()[:24]


def _pa():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def dictionary(values: list[str], index: np.ndarray):
    """A dictionary-encoded string column: ``values[index]`` without materialising one string per row."""
    pa, _ = _pa()
    return pa.DictionaryArray.from_arrays(pa.array(np.asarray(index, np.int32)), pa.array(values, pa.string()))


def write_table(columns: dict, path: Path) -> dict:
    pa, pq = _pa()
    table = pa.table(columns)
    pq.write_table(table, path, compression="zstd", use_dictionary=True, row_group_size=1_000_000)
    return {"rows": table.num_rows, "bytes": path.stat().st_size, "sha256": readiness.sha256_file(path)}


# ---------------------------------------------------------------- the intent-stages traces

def trace_tables(args, out: Path, chain: SimpleNamespace | None = None) -> tuple[dict, dict]:
    """trace_answers, queries and answer_queries from the trace extract, with the stage chain's values joined on the
    answer id when ``chain`` (``chain_values``) is given; returns (table info, figure check)."""
    run = Path(args.run)
    trace = run / "trace-extract"
    st, ev, codes = isd.load_stages(trace)
    corpus = isd.load_corpus(args.corpus_package)
    doc_z = np.asarray(corpus["consensus_axis_1_z"], float)
    doc_u = np.asarray(corpus["prompt_scale_percentile_0_1"], float)
    prompts = readiness.read_jsonl(trace / "prompts.jsonl.gz")
    generation_ids = [r["generation_id"] for r in readiness.read_jsonl(trace / "answers.jsonl.gz")]
    query_rows = readiness.read_jsonl(trace / "queries.jsonl.gz")
    query_ids = [r["page_id"] for r in query_rows]
    coords = SimpleNamespace(battery=Path(args.battery), final_axis_map=Path(args.final_axis_map),
                             answers_index=run / "answers-export/observations.jsonl.gz", answers_id_field="answer_id",
                             answers_qwen=run / "embed-answers-qwen.merged", answers_mistral=run / "embed-answers-mistral.merged")
    # the same calls as intent_stages_study.py analyze, so every value is the one the stage figure plots
    query_z, query_validity = isd.text_coordinates(run / "embed-queries-qwen.merged", run / "embed-queries-mistral.merged", coords, query_ids)
    answer_z, answer_validity = isd.answer_values(coords, st, codes)
    r0, rk, _ = isd.replay_values(run / "intent-replay", st, codes, prompts, doc_z)
    values = stages.stage_values(st, doc_z, query_z=query_z, answer_z=answer_z, r0=r0, rk=rk)
    page_u = stages.stage_values(st, doc_u)
    print(json.dumps({"stage_values": sorted(values), "answers": len(st.x), "time": readiness.now()}), flush=True)

    # per query: both views, consensus, prompt-scale position (the views behind Q)
    _, views, percentile, outside = readiness.page_coordinates(run / "embed-queries-qwen.merged", run / "embed-queries-mistral.merged",
                                                               Path(args.battery), Path(args.final_axis_map))
    nan = float("nan")
    info = {}
    info["queries"] = write_table({
        "query_id": np.arange(len(query_ids), dtype=np.int32),
        "text": [r["text"] for r in query_rows], "text_sha256": [r["text_sha256"] for r in query_rows],
        "qwen_axis_1_z": np.asarray([views["qwen"].get(i, nan) for i in query_ids], float),
        "mistral_aligned_axis_1_z": np.asarray([views["mistral"].get(i, nan) for i in query_ids], float),
        "consensus_axis_1_z": np.asarray(query_z, float),
        "prompt_scale_x": np.asarray([percentile.get(i, nan) for i in query_ids], float),
        "outside_prompt_range": [outside.get(i) for i in query_ids]}, out / "queries.parquet")

    model, engine, method, condition = (np.asarray(codes[k], object)[getattr(st, k)] for k in ("model", "engine", "method", "condition"))
    prompt_id = np.asarray(codes["prompt"], object)[st.prompt]
    keyword_text = np.asarray([p["keyword_text"] for p in prompts], object)[st.prompt]
    ids = [answer_id(*key) for key in zip(model, prompt_id, engine, method, condition)]
    if len(set(ids)) != len(ids):
        raise ValueError("answer identities are not unique in the trace extract")
    counts = {name: np.diff(getattr(st, f"{name}_offsets")).astype(np.int16) for name in ("q", "ret", "cand", "pool", "rank")}
    columns = {"answer_id": ids, "generation_id": generation_ids,
               "model": dictionary(list(codes["model"]), st.model), "engine": dictionary(list(codes["engine"]), st.engine),
               "method": dictionary(list(codes["method"]), st.method), "condition": dictionary(list(codes["condition"]), st.condition),
               "prompt_id": list(prompt_id), "keyword_id": dictionary(list(codes["keyword"]), st.keyword),
               "keyword": list(keyword_text),
               "split": ["exploration" if exploration_keyword(k or "") else "held-out" for k in keyword_text],
               "x": np.asarray(st.x, float),
               "n_queries": counts["q"], "n_retrieved": counts["ret"], "n_scored": counts["cand"], "n_shown": counts["pool"],
               "n_cited": counts["rank"]}
    for term in Z_TERMS:
        columns[term] = np.asarray(values.get(term, np.full(len(st.x), np.nan)), float)
    for term in U_TERMS:
        columns[f"{term}_u"] = np.asarray(page_u[term], float)
    info["validity"] = {"queries": query_validity, "answers": answer_validity}
    if chain is not None:
        where = {a: i for i, a in enumerate(chain.ids)}
        pos = np.asarray([where.get(a, -1) for a in ids])
        found = pos >= 0
        for term in CHAIN_TERMS:
            columns[f"chain_{term}"] = np.where(found, chain.values[term][np.maximum(pos, 0)], np.nan)
        info["validity"]["chain"] = {
            "trace_answers_without_funnel_answer": int((~found).sum()), "funnel_answers": len(chain.ids),
            "x_max_abs_difference": float(np.max(np.abs(chain.x[pos[found]] - columns["x"][found]), initial=0.0)),
            "keyword_mismatches": int(sum(chain.keyword[j] != k for j, k in zip(pos[found], keyword_text[found])))}
    info["trace_answers"] = write_table(columns, out / "trace_answers.parquet")
    owner = stages.owner(st.q_offsets)
    info["answer_queries"] = write_table({
        "answer_id": dictionary(ids, owner), "search_index": (np.arange(len(st.q_ids)) - st.q_offsets[owner]).astype(np.int8),
        "query_id": np.asarray(st.q_ids, np.int32)}, out / "answer_queries.parquet")
    return info, figure_check(out / "trace_answers.parquet", run / "intent-stages/results.json")


def chain_values(extract: Path, assembled: Path, replay: Path) -> SimpleNamespace:
    """The stage chain's per-answer values for every answer of the funnel extract, computed by the chain's own functions
    exactly as `steelman.tables.load` does (page intent u from the assembly's row features, R0 from the funnel replay)."""
    answers = readiness.read_jsonl(assembled / "answers.jsonl.gz")
    extracted = readiness.read_jsonl(extract / "answers.jsonl.gz")
    if len(extracted) != len(answers) or any(a["fingerprint"] != b["fingerprint"] for a, b in zip(extracted, answers)):
        raise ValueError("extract and assembled answers are not aligned")
    with np.load(assembled / "row_features.npz", allow_pickle=False) as z:
        u = z["u"]
    with np.load(extract / "items.npz") as z:
        values = chain_tables.stage_values(z["offsets"], z["row"], z["scored"], z["presented"], z["ranked"], u)
    with np.load(replay / "replay.npz") as z:
        replay_rows = z["rows"]
    values["R0"] = chain_tables.replay_values([a["prompt_id"] for a in answers], [a["engine"] for a in answers],
                                              readiness.read_jsonl(replay / "prompts.jsonl.gz"), replay_rows, u)
    return SimpleNamespace(ids=[answer_id(a["model"], a["prompt_id"], a["engine"], a["method"], a["condition"]) for a in answers],
                           values={term: np.asarray(values[term], float) for term in CHAIN_TERMS},
                           x=np.asarray([a["x"] for a in answers], float), keyword=[a["keyword_text"] for a in answers])


def curves_from_table(path: Path, *, natural_condition: str = "natural") -> dict:
    """The stage figure's curves from trace_answers.parquet alone (the computation of `analyze`, on the table)."""
    _, pq = _pa()
    return curves_from_frame(pq.read_table(path).to_pandas(), natural_condition=natural_condition)


def curves_from_frame(t, *, natural_condition: str = "natural") -> dict:
    """Curves per model · engine and per model (both engines), natural condition and all conditions, from a frame with
    the trace_answers columns. With only natural answers in the frame, the "all" curves equal the natural ones."""
    x = t["x"].to_numpy(float)
    keyword = t["keyword_id"].astype(str).to_numpy()
    natural = (t["condition"].astype(str) == natural_condition).to_numpy()
    z_values = {term: t[term].to_numpy(float) for term in Z_TERMS if t[term].notna().any()}
    u_values = {term: t[f"{term}_u"].to_numpy(float) for term in U_TERMS}
    groups = {}
    models, engines = t["model"].astype(str).to_numpy(), t["engine"].astype(str).to_numpy()
    for m in sorted(set(models)):
        for e in sorted(set(engines)):
            groups[f"{m} · {e}"] = (models == m) & (engines == e)
        groups[f"{m} · both engines"] = models == m
    return {name: {condition: {"z": stages.stage_curves(x, keyword, z_values, mask & extra),
                               "u": stages.stage_curves(x, keyword, u_values, mask & extra)}
                   for condition, extra in (("natural", natural), ("all", np.ones(len(x), bool)))}
            for name, mask in groups.items()}


def _difference(a, b) -> float:
    a = np.asarray([np.nan if v is None else v for v in a], float)
    b = np.asarray([np.nan if v is None else v for v in b], float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    both = ~np.isnan(a)
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0


def figure_check(table: Path, results_path: Path, tolerance: float = 1e-12) -> dict:
    """Compare every curve the stored results hold with the curves recomputed from the table."""
    if not results_path.exists():
        return {"skipped": f"{results_path} not found"}
    stored = json.loads(results_path.read_text())["curves"]["groups"]
    again = curves_from_table(table)
    worst, compared, missing = 0.0, 0, []
    for name, by_condition in stored.items():
        for condition, scales in by_condition.items():
            for scale, curve in scales.items():
                for term, series in curve["terms"].items():
                    if term in ("G", "G-K"):  # development judge stage, not exported
                        continue
                    mine = again.get(name, {}).get(condition, {}).get(scale, {}).get("terms", {}).get(term)
                    if mine is None:
                        missing.append(f"{name}/{condition}/{scale}/{term}")
                        continue
                    compared += 1
                    if list(mine["n"]) != list(series["n"]):
                        worst = float("inf")
                    worst = max(worst, _difference(mine["mean"], series["mean"]),
                                _difference(mine["se_keyword_cluster"], series["se_keyword_cluster"]))
    return {"curves_compared": compared, "missing": missing, "max_abs_difference": worst, "tolerance": tolerance,
            "passed": compared > 0 and not missing and worst <= tolerance}


# ---------------------------------------------------------------- the funnel traces (row level)

def funnel_tables(extract: Path, assembled: Path, out: Path) -> dict:
    """stage_items, rerank_events, stage_u and shown with release answer ids, from the funnel extract and assembly."""
    answers = readiness.read_jsonl(extract / "answers.jsonl.gz")
    ids = [answer_id(a["model"], a["prompt_id"], a["engine"], a["method"], a["condition"]) for a in answers]
    if len(set(ids)) != len(ids):
        raise ValueError("answer identities are not unique in the funnel extract")
    info = {}
    with np.load(extract / "items.npz") as z:
        items = {k: z[k] for k in z.files}
    offsets = items["offsets"]
    if len(offsets) != len(ids) + 1:
        raise ValueError("items.npz offsets do not match the extract's answers")
    who = np.repeat(np.arange(len(ids)), np.diff(offsets))
    info["stage_items"] = write_table({
        "answer_id": dictionary(ids, who), "row_id": items["row"].astype(np.int32), "search_index": items["search"].astype(np.int8),
        "rank_in_search": items["rank"].astype(np.int8), "scored": items["scored"].astype(np.int8),
        "shown_slot": items["presented"].astype(np.int8), "cited_rank": items["ranked"].astype(np.int8)}, out / "stage_items.parquet")
    del items, who
    with np.load(extract / "events.npz") as z:
        ev = {k: z[k] for k in z.files}
    sizes = np.diff(ev["offsets"])
    event = np.repeat(np.arange(len(sizes)), sizes)
    columns = {"event_id": event.astype(np.int32), "answer_id": dictionary(ids, ev["answer"][event]),
               "search_index": ev["search"][event].astype(np.int8), "row_id": ev["row"].astype(np.int32),
               "score": ev["score"].astype(float), "selected_order": ev["selected"].astype(np.int8)}
    with np.load(assembled / "candidates.npz") as z:
        if len(z["row"]) != len(ev["row"]) or not np.array_equal(z["row"], ev["row"]) or not np.array_equal(z["event"], event):
            raise ValueError("assembled candidates do not align with the extract's events")
        columns["topic"] = z["topic"].astype(float)
        columns["on_keyword"] = z["on_keyword"].astype(np.int8)
    info["rerank_events"] = write_table(columns, out / "rerank_events.parquet")
    del ev, columns, event
    with np.load(assembled / "u.npz") as z:
        u = {k: z[k] for k in z.files}
    info["stage_u"] = write_table({"answer_id": dictionary(ids, u["answer"]), "row_id": u["row"].astype(np.int32),
                                   "topic": u["topic"].astype(float),
                                   **{k: u[k].astype(np.int8) for k in ("retrieved", "scored", "presented", "ranked", "replay")}},
                                  out / "stage_u.parquet")
    with np.load(assembled / "presented.npz") as z:
        p = {k: z[k] for k in z.files}
    info["shown"] = write_table({"answer_id": dictionary(ids, p["answer"]), "row_id": p["row"].astype(np.int32),
                                 "slot": p["slot"].astype(np.int8), "cited_rank": p["rank"].astype(np.int8),
                                 "topic": p["topic"].astype(float), "on_keyword": p["on_keyword"].astype(np.int8)}, out / "shown.parquet")
    info["funnel_answers"] = len(ids)
    return info


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="the full run's output folder")
    p.add_argument("--corpus-package", type=Path, help="snippet-embeddings-corpus-v1 (page positions)")
    p.add_argument("--battery", type=Path, help="the robustness battery folder (view alignment)")
    p.add_argument("--final-axis-map", type=Path, help="final-axis-map.jsonl (the prompt scale)")
    p.add_argument("--extract", type=Path, help="funnel extract (default RUN/funnel-extract)")
    p.add_argument("--assembled", type=Path, help="funnel assembly (default RUN/funnel-assembled)")
    p.add_argument("--replay", type=Path, help="funnel replay of the prompt texts (default RUN/funnel-replay)")
    p.add_argument("--skip-trace", action="store_true", help="only the funnel tables")
    p.add_argument("--skip-funnel", action="store_true", help="only the intent-stages tables")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if not a.skip_trace and not (a.corpus_package and a.battery and a.final_axis_map):
        p.error("--corpus-package, --battery and --final-axis-map are needed for the trace tables")
    started = time.time()
    partial = readiness.new_directory(a.output)
    manifest = {"format_version": FORMAT_VERSION, "created_at": readiness.now(), "git_commit": readiness.git_commit(),
                "answer_id": 'first 24 hex characters of sha256("model|prompt_id|engine|method|condition")', "tables": {}}
    check = {"skipped": "--skip-trace"}
    extract = a.extract or a.run / "funnel-extract"
    assembled = a.assembled or a.run / "funnel-assembled"
    replay = a.replay or a.run / "funnel-replay"
    if not a.skip_trace:
        info, check = trace_tables(a, partial, None if a.skip_funnel else chain_values(extract, assembled, replay))
        manifest["validity"] = info.pop("validity")
        manifest["tables"].update(info)
        manifest["inputs_trace"] = {name: readiness.identity(path) for name, path in (
            ("trace_extract", a.run / "trace-extract/manifest.json"), ("intent_replay", a.run / "intent-replay/manifest.json"),
            ("corpus", a.corpus_package / "manifest.json"), ("final_axis_map", a.final_axis_map),
            ("queries_qwen", a.run / "embed-queries-qwen.merged/projection_manifest.json"),
            ("queries_mistral", a.run / "embed-queries-mistral.merged/projection_manifest.json"),
            ("answers_index", a.run / "answers-export/observations.jsonl.gz")) if Path(path).exists()}
        readiness.write_json(partial / "figure_check.json", check)
        print(json.dumps({"figure_check": check, "time": readiness.now()}), flush=True)
    if not a.skip_funnel:
        manifest["tables"].update(funnel_tables(extract, assembled, partial))
        manifest["inputs_funnel"] = {"extract": readiness.identity(extract / "manifest.json"),
                                     "assembled": readiness.identity(assembled / "manifest.json"),
                                     "replay": readiness.identity(replay / "manifest.json")}
    manifest["figure_check"] = check
    manifest["seconds"] = round(time.time() - started, 1)
    readiness.write_json(partial / "manifest.json", manifest)
    partial.rename(Path(a.output).resolve())
    print(json.dumps({"output": str(a.output), "tables": {k: v.get("rows") for k, v in manifest["tables"].items() if isinstance(v, dict)},
                      "seconds": manifest["seconds"]}), flush=True)
    return 0 if check.get("passed", "skipped" in check) else 3


if __name__ == "__main__":
    raise SystemExit(main())
