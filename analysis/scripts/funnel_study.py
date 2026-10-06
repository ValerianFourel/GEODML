#!/usr/bin/env python3
"""Funnel study (``analysis/docs/funnel_study.md``): admission into the pool and ranking within it.

  extract   read every completed generator cell once; map retrieved / scored / presented / ranked
            evidence to exact snapshot rows (funnel_rows.answer_items)            [HoreKa, CPU]
  replay    the frozen search on each prompt's own text (R0), exact and vectorised  [any]
  assemble  stage tables: U (keyword rows per answer, with r, c, p, k, r0 flags), reranker
            candidates, presented items; topic similarity; row features            [HoreKa, CPU]
  analyze   one task per (stratum, stage, specification) with shared keyword draws and shuffles;
            ``--shard k/n``; finished tasks are cached                              [HoreKa, CPU]
  report    results.json + final-report.html (exits 3 if tasks are missing)          [any]
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
import hashlib  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.interpretability.pipeline import funnel_models as fm  # noqa: E402
from analysis.interpretability.pipeline import funnel_rows as fr  # noqa: E402
from analysis.interpretability.pipeline import geo_drivers as geo  # noqa: E402
from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402
from analysis.scripts.geo_drivers_study import VIEWS, pair_topic_similarity  # noqa: E402
from analysis.scripts.intent_stages_study import ResultCache, refuse_existing  # noqa: E402

FORMAT_VERSION = "funnel-study-v1"
STAGE_NAMES = ("R|U", "R0|U", "P|C", "K|P")
LOG1P = "log1p"

# (name, block, source table, source column, transform). Blocks: A1 intent, A2 topic, A3 snippet surface,
# A4 URL string, B search, C1 off-page domain/SEO, C2 page body. Missing values become the mean; blocks with
# incomplete coverage get an indicator (added below).
ROW_FEATURES = [
    ("page_intent_z", "A1", "docs", "consensus_axis_1_z", None),
    ("snip_title_chars", "A3", "docs", "snip_title_chars", LOG1P), ("snip_text_words", "A3", "docs", "snip_text_words", LOG1P),
    ("snip_digits", "A3", "docs", "snip_digits", LOG1P), ("snip_percent", "A3", "docs", "snip_percent", None),
    ("snip_year", "A3", "docs", "snip_year", None), ("snip_currency", "A3", "docs", "snip_currency", None),
    ("snip_title_question", "A3", "docs", "snip_title_question", None), ("snip_title_listicle", "A3", "docs", "snip_title_listicle", None),
    ("snip_names_domain", "A3", "docs", "snip_names_domain", None), ("snip_glued", "A3", "docs", "snip_glued", None),
    ("url_https", "A4", "urls", "url_https", None), ("url_path_depth", "A4", "urls", "url_path_depth", None),
    ("url_length", "A4", "urls", "url_length", LOG1P), ("url_has_query", "A4", "urls", "url_has_query", None),
    ("url_subdomain", "A4", "urls", "url_subdomain", None), ("url_tld_com", "A4", "urls", "url_tld_com", None),
    ("url_tld_org", "A4", "urls", "url_tld_org", None), ("url_tld_edu_gov", "A4", "urls", "url_tld_edu_gov", None),
    ("url_user_content", "A4", "urls", "url_user_content", None), ("url_wikipedia", "A4", "urls", "url_wikipedia", None),
    ("url_ad_redirect", "A4", "urls", "url_ad_redirect", None),
    ("stored_position", "B", "rows", "position", None), ("searxng_score", "B", "rows", "searxng_score", LOG1P),
    ("searxng_engine_count", "B", "rows", "searxng_engine_count", None),
    ("dfs_organic_count", "C1", "domains", "dfs_organic_count", LOG1P), ("dfs_organic_top1", "C1", "domains", "dfs_organic_pos_1", LOG1P),
    ("dfs_traffic_value", "C1", "domains", "dfs_organic_etv", LOG1P), ("dfs_paid_count", "C1", "domains", "dfs_paid_count", LOG1P),
    ("dfs_domain_age", "C1", "domains", "dfs_age_years_at_reference", None), ("open_pagerank", "C1", "domains", "opr", None),
    ("has_llms_txt", "C1", "domains", "has_llms_txt", None), ("brand_list", "C1", "domains", "brand_list", None),
    ("earned_list", "C1", "domains", "earned_list", None), ("google_top20_url", "C1", "rows", "google_top20_url", None),
    ("google_top20_domain", "C1", "rows", "google_top20_domain", None),
    ("body_stats_density", "C2", "urls", "body_stats_density", LOG1P), ("body_question_headings", "C2", "urls", "body_question_headings", None),
    ("body_modularity", "C2", "urls", "body_modularity", LOG1P), ("body_structured_data", "C2", "urls", "body_structured_data", None),
    ("body_ext_citations", "C2", "urls", "body_ext_citations", None), ("body_auth_citations", "C2", "urls", "body_auth_citations", LOG1P),
    ("body_word_count", "C2", "urls", "body_word_count", LOG1P), ("body_readability", "C2", "urls", "body_readability", None),
    ("body_internal_links", "C2", "urls", "body_internal_links", LOG1P), ("body_outbound_links", "C2", "urls", "body_outbound_links", LOG1P),
    ("body_images_alt", "C2", "urls", "body_images_alt", LOG1P), ("body_freshness", "C2", "urls", "body_freshness", None),
]
MISSING_INDICATORS = {"c1_dfs_missing": "dfs_organic_count", "c1_opr_missing": "open_pagerank", "c1_llms_missing": "has_llms_txt",
                      "c2_html_missing": "body_word_count", "c2_readability_missing": "body_readability"}
VISIBLE_BLOCKS = ("A1", "A2", "A3", "A4", "B")
PRIMARY_C1 = "dfs_organic_count"


# ---------------------------------------------------------------- extract

_EXTRACT: dict = {}


def _extract_chunk(job):
    from analysis.interpretability.pipeline.agentic_cells import iter_cells
    root, model, refs, (prompt_axis, keyword_text) = job
    rows, lookup, indices = _EXTRACT["rows"], _EXTRACT["lookup"], _EXTRACT["indices"]
    counts, snapshot_hashes = Counter(), set()
    meta, items, events = [], [], []
    for cell in iter_cells(root, refs):
        fields, skipped = readiness.checked_cell(cell, prompt_axis)
        if skipped:
            counts[skipped] += 1
            continue
        generation = cell["generation"]
        engine = generation.get("engine")
        try:
            out = fr.answer_items(cell["trace"], generation["method"], engine, fields["urls"], fields["ranking"], lookup, rows,
                                  index=indices.get(engine))
        except (KeyError, TypeError, ValueError) as error:
            counts[f"skipped_trace:{str(error)[:60]}"] += 1
            continue
        for s in (e["payload"] for e in cell["trace"].get("events", []) if isinstance(e, dict) and e.get("event_type") == "search"):
            snapshot_hashes.add((engine, (s.get("raw_payload") or {}).get("snapshot_sha256")))
        counts.update({f"searches:{k}": v for k, v in out["counts"].items()})
        meta.append((cell["fingerprint"], model, engine, generation["method"], generation.get("condition"), fields["prompt_id"],
                     keyword_text.get(fields["prompt_id"]), prompt_axis[fields["prompt_id"]]))
        items.append(np.asarray(out["items"], np.int64).reshape(-1, 6))
        events.append([(e["search"], np.asarray(e["candidates"], float).reshape(-1, 3)) for e in out["events"]])
        counts[f"answers_{model}"] += 1
    return {"meta": meta, "items": items, "events": events, "counts": counts, "snapshots": snapshot_hashes}


def extract(args) -> int:
    refuse_existing(args.output)
    paths = dict(s.split("=", 1) for s in args.snapshot)
    rows = fr.snapshot_rows(paths)
    _EXTRACT.update(rows=rows, lookup=rows.lookup(), indices={e: fr.LexicalIndex(rows, e) for e in paths})
    prompt_axis = readiness.axis_by_prompt(args.final_axis_map)
    keyword_text = {str(r["candidate_id"]): r.get("keyword") for r in readiness.read_jsonl(args.population_prompts)}
    workers = max(1, args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    partial = readiness.new_directory(args.output)
    meta, item_parts, item_sizes, event_rows = [], [], [], []
    counts, snapshot_hashes = Counter(), set()
    inputs, results = readiness.parallel_chunks(args.source, _extract_chunk, (prompt_axis, keyword_text), workers)
    for number, total, chunk in results:
        for m, it, ev in zip(chunk["meta"], chunk["items"], chunk["events"]):
            answer = len(meta)
            meta.append(m)
            item_parts.append(it)
            item_sizes.append(len(it))
            for e, (search, cand) in enumerate(ev):
                event_rows.append((answer, search, cand))
        counts.update(chunk["counts"])
        snapshot_hashes |= chunk["snapshots"]
        print(json.dumps({"chunk": number, "of": total, "answers": len(meta), "time": readiness.now()}), flush=True)
    if not meta:
        raise ValueError("no answers passed the checks")
    recorded = {e: {h for en, h in snapshot_hashes if en == e} for e in paths}
    mismatch = {e: sorted(h) for e, h in recorded.items() if h - {rows.snapshot_sha256[e], "sha256:" + rows.snapshot_sha256[e]}}
    items = np.concatenate(item_parts)
    columns = ("fingerprint", "model", "engine", "method", "condition", "prompt_id", "keyword_text", "x")
    readiness.write_jsonl(partial / "answers.jsonl.gz", (dict(zip(columns, m)) for m in meta))
    readiness.write_npz(partial / "items.npz", offsets=np.r_[0, np.cumsum(item_sizes)].astype(np.int64),
                        row=items[:, 0], search=items[:, 1], rank=items[:, 2], scored=items[:, 3], presented=items[:, 4],
                        ranked=items[:, 5])
    cand_sizes = [len(c) for _, _, c in event_rows]
    cands = np.concatenate([c for _, _, c in event_rows]) if event_rows else np.zeros((0, 3))
    readiness.write_npz(partial / "events.npz", answer=np.asarray([a for a, _, _ in event_rows], np.int64),
                        search=np.asarray([s for _, s, _ in event_rows], np.int64),
                        offsets=np.r_[0, np.cumsum(cand_sizes)].astype(np.int64),
                        row=cands[:, 0].astype(np.int64), score=cands[:, 1], selected=cands[:, 2].astype(np.int64))
    readiness.write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "stage": "extract", "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "inputs": inputs, "snapshot_sha256": rows.snapshot_sha256, "row_table_digest": fr.row_table_digest(rows),
        "snapshot_hash_mismatch": mismatch, "counts": {**counts, "answers": len(meta), "items": int(len(items)),
                                                       "events": len(event_rows)}})
    if mismatch:
        raise ValueError(f"traces recorded other snapshots than the row table: {mismatch}")
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"answers": len(meta), "items": int(len(items)), "events": len(event_rows),
                      "skipped": {k: v for k, v in counts.items() if k.startswith("skipped")}}), flush=True)
    return 0


# ---------------------------------------------------------------- replay

def replay(args) -> int:
    refuse_existing(args.output)
    paths = dict(s.split("=", 1) for s in args.snapshot)
    rows = fr.snapshot_rows(paths)
    population = readiness.read_jsonl(args.population_prompts)
    partial = readiness.new_directory(args.output)
    prompt_ids, engines, chosen = [], [], []
    for engine in sorted(paths):
        index = fr.LexicalIndex(rows, engine)
        for record in population:
            prompt_ids.append(str(record["candidate_id"]))
            engines.append(engine)
            chosen.append(index.select(record["question"]))
    readiness.write_jsonl(partial / "prompts.jsonl.gz", ({"prompt_id": p, "engine": e} for p, e in zip(prompt_ids, engines)))
    readiness.write_npz(partial / "replay.npz", rows=np.asarray(chosen, np.int64))
    readiness.write_json(partial / "manifest.json", {"format_version": FORMAT_VERSION, "stage": "replay", "created_at": readiness.now(),
                                                     "git_commit": readiness.git_commit(), "row_table_digest": fr.row_table_digest(rows),
                                                     "prompts": len(population), "engines": sorted(paths)})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"replays": len(chosen)}), flush=True)
    return 0


# ---------------------------------------------------------------- assemble

def row_feature_table(features_dir: Path, n_rows: int):
    """One row per snapshot row id: every ROW_FEATURES column (transformed) plus missing indicators."""
    import pandas as pd
    rows = pd.read_parquet(features_dir / "rows.parquet")
    if len(rows) != n_rows or not np.array_equal(rows["row_id"].to_numpy(), np.arange(n_rows)):
        raise ValueError("feature rows do not match the snapshot row table")
    docs = pd.read_parquet(features_dir / "docs.parquet")
    urls = pd.read_parquet(features_dir / "urls.parquet")
    domains = pd.read_parquet(features_dir / "domains.parquet")
    merged = (rows.merge(docs, on="corpus_row", how="left", suffixes=("", "_doc"))
              .merge(urls, on="url", how="left", suffixes=("", "_url"))
              .merge(domains, left_on="domain", right_on="domain", how="left", suffixes=("", "_dom")))
    merged = merged.sort_values("row_id")
    out = {}
    for name, _, _, column, transform in ROW_FEATURES:
        values = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
        out[name] = np.log1p(np.clip(values, 0, None)) if transform == LOG1P else values
    for name, source in MISSING_INDICATORS.items():
        out[name] = np.isnan(out[source]).astype(float)
    out["u"] = pd.to_numeric(merged["prompt_scale_percentile_0_1"], errors="coerce").to_numpy(float)
    out["corpus_row"] = merged["corpus_row"].to_numpy(np.int64)
    out["keyword"] = merged["keyword"].to_numpy(object)
    out["ad"] = out["url_ad_redirect"]
    out["glued"] = out["snip_glued"]
    return out


def assemble(args) -> int:
    refuse_existing(args.output)
    extract_dir, features_dir = Path(args.extract), Path(args.features)
    manifest = json.loads((extract_dir / "manifest.json").read_text())
    features_manifest = json.loads((features_dir / "manifest.json").read_text())
    if manifest["row_table_digest"] != features_manifest["row_table_digest"]:
        raise ValueError("feature table and extract use different snapshot row tables")
    answers = readiness.read_jsonl(extract_dir / "answers.jsonl.gz")
    items = np.load(extract_dir / "items.npz")
    events = np.load(extract_dir / "events.npz")
    n_rows = int(features_manifest["counts"]["rows"])
    rf = row_feature_table(features_dir, n_rows)
    keyword_rows = defaultdict(list)
    import pandas as pd
    rows = pd.read_parquet(features_dir / "rows.parquet", columns=["row_id", "engine", "keyword"])
    for r, e, k in zip(rows["row_id"], rows["engine"], rows["keyword"]):
        keyword_rows[(e, str(k).casefold())].append(int(r))
    replayed = {}
    if args.replay:
        rp = readiness.read_jsonl(Path(args.replay) / "prompts.jsonl.gz")
        chosen = np.load(Path(args.replay) / "replay.npz")["rows"]
        replayed = {(p["prompt_id"], p["engine"]): set(map(int, c)) for p, c in zip(rp, chosen)}
    partial = readiness.new_directory(args.output)

    # U: the keyword's own rows of the answer's engine, with stage flags
    u_answer, u_row, flags = [], [], defaultdict(list)
    offsets = items["offsets"]
    missing_keyword = 0
    for i, a in enumerate(answers):
        own = keyword_rows.get((a["engine"], (a["keyword_text"] or "").casefold()))
        if not own:
            missing_keyword += 1
            continue
        lo, hi = offsets[i], offsets[i + 1]
        seen = {int(r): (int(s), int(p), int(k)) for r, s, p, k in zip(items["row"][lo:hi], items["scored"][lo:hi],
                                                                         items["presented"][lo:hi], items["ranked"][lo:hi])}
        r0 = replayed.get((a["prompt_id"], a["engine"]), set())
        for row in own:
            s, p, k = seen.get(row, (0, -1, -1))
            u_answer.append(i)
            u_row.append(row)
            flags["retrieved"].append(int(row in seen))
            flags["scored"].append(s)
            flags["presented"].append(int(p >= 0))
            flags["ranked"].append(int(k >= 0))
            flags["replay"].append(int(row in r0))
    # reranker candidates and presented items
    c_event, c_answer, c_row, c_score, c_selected = [], [], [], [], []
    for e, (a, lo, hi) in enumerate(zip(events["answer"], events["offsets"][:-1], events["offsets"][1:])):
        for r, sc, sel in zip(events["row"][lo:hi], events["score"][lo:hi], events["selected"][lo:hi]):
            c_event.append(e); c_answer.append(int(a)); c_row.append(int(r)); c_score.append(float(sc)); c_selected.append(int(sel))
    p_answer, p_row, p_slot, p_rank = [], [], [], []
    for i in range(len(answers)):
        lo, hi = offsets[i], offsets[i + 1]
        shown = [(int(p), int(r), int(k)) for r, p, k in zip(items["row"][lo:hi], items["presented"][lo:hi], items["ranked"][lo:hi]) if p >= 0]
        for p, r, k in sorted(shown):
            p_answer.append(i); p_row.append(r); p_slot.append(p); p_rank.append(k)

    # pair features: on-keyword and intent-free topic similarity for every (prompt, page) pair used
    prompt_code = {}
    answer_prompt = np.asarray([prompt_code.setdefault(a["prompt_id"], len(prompt_code)) for a in answers], np.int64)
    all_answer = np.r_[np.asarray(u_answer, np.int64), np.asarray(c_answer, np.int64), np.asarray(p_answer, np.int64)]
    all_row = np.r_[np.asarray(u_row, np.int64), np.asarray(c_row, np.int64), np.asarray(p_row, np.int64)]
    corpus_row = rf["corpus_row"][all_row]
    n_docs = int(rf["corpus_row"].max()) + 1
    key = answer_prompt[all_answer] * n_docs + corpus_row
    unique, inverse = np.unique(key, return_inverse=True)
    topic_unique, topic_diag = pair_topic_similarity(
        np.stack([unique // n_docs, unique % n_docs], axis=1), sorted(prompt_code, key=prompt_code.get),
        {"qwen": args.qwen_prompts, "mistral": args.mistral_prompts},
        {view: Path(args.corpus_package) / readiness.EMBEDDING_FILES[view] for view in VIEWS},
        {"qwen": args.qwen_map, "mistral": args.mistral_map})
    topic = topic_unique[inverse.reshape(-1)]
    answer_keyword = np.asarray([(a["keyword_text"] or "").casefold() for a in answers], object)
    row_keyword = np.asarray([str(k).casefold() for k in rf["keyword"]], object)
    on_keyword = (row_keyword[all_row] == answer_keyword[all_answer]).astype(float)
    nu, nc = len(u_answer), len(c_answer)
    readiness.write_npz(partial / "u.npz", answer=np.asarray(u_answer, np.int64), row=np.asarray(u_row, np.int64),
                        topic=topic[:nu], **{k: np.asarray(v, np.int64) for k, v in flags.items()})
    readiness.write_npz(partial / "candidates.npz", event=np.asarray(c_event, np.int64), answer=np.asarray(c_answer, np.int64),
                        row=np.asarray(c_row, np.int64), score=np.asarray(c_score), selected=np.asarray(c_selected, np.int64),
                        topic=topic[nu:nu + nc], on_keyword=on_keyword[nu:nu + nc])
    readiness.write_npz(partial / "presented.npz", answer=np.asarray(p_answer, np.int64), row=np.asarray(p_row, np.int64),
                        slot=np.asarray(p_slot, np.int64), rank=np.asarray(p_rank, np.int64), topic=topic[nu + nc:],
                        on_keyword=on_keyword[nu + nc:])
    np.savez_compressed(partial / "row_features.npz", **{k: v for k, v in rf.items() if k != "keyword"})
    readiness.write_jsonl(partial / "answers.jsonl.gz", answers)
    readiness.write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "stage": "assemble", "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "extract": readiness.identity(extract_dir / "manifest.json"), "features": readiness.identity(features_dir / "manifest.json"),
        "replay": readiness.identity(Path(args.replay) / "manifest.json") if args.replay else None,
        "counts": {"answers": len(answers), "u_items": nu, "candidates": nc, "presented": len(p_answer),
                   "answers_without_keyword_rows": missing_keyword, "unique_prompt_page_pairs": int(len(unique))},
        "topic_similarity": topic_diag})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"u_items": nu, "candidates": nc, "presented": len(p_answer)}), flush=True)
    return 0


# ---------------------------------------------------------------- analyze

def load_assembled(directory: Path):
    d = Path(directory)
    answers = readiness.read_jsonl(d / "answers.jsonl.gz")
    return SimpleNamespace(answers=answers, u=dict(np.load(d / "u.npz")), cand=dict(np.load(d / "candidates.npz")),
                           pres=dict(np.load(d / "presented.npz")), rf=dict(np.load(d / "row_features.npz", allow_pickle=False)),
                           manifest=json.loads((d / "manifest.json").read_text()))


def answer_arrays(data):
    keyword_code, prompt_code = {}, {}
    a = data.answers
    return SimpleNamespace(
        x=np.asarray([r["x"] for r in a], float),
        keyword=np.asarray([keyword_code.setdefault(r["keyword_text"], len(keyword_code)) for r in a], np.int64),
        prompt=np.asarray([prompt_code.setdefault(r["prompt_id"], len(prompt_code)) for r in a], np.int64),
        model=np.asarray([r["model"] for r in a]), method=np.asarray([r["method"] for r in a]),
        engine=np.asarray([r["engine"] for r in a]), condition=np.asarray([r["condition"] for r in a]),
        keywords=len(keyword_code))


def strata(ans) -> dict:
    out = {}
    for model in sorted(set(ans.model)):
        for method in sorted(set(ans.method)):
            out[f"{model} · {method.split('-')[0]}"] = (ans.model == model) & (ans.method == method)
    return out


def _features(spec: str, stage: str) -> list:
    """Feature list of a specification at a stage."""
    names = [fm.Feature("page_intent_z", "A1"), fm.Feature("intent_x_prompt", "A1", "interaction", "page_intent_z"),
             fm.Feature("intent_alignment", "A1", "alignment", "u"), fm.Feature("topic_similarity", "A2")]
    if stage in ("P|C", "K|P"):
        names.append(fm.Feature("on_keyword", "A2"))
    for name, block, *_ in ROW_FEATURES:
        if name == "page_intent_z":
            continue
        if spec == "visible" and block not in VISIBLE_BLOCKS:
            continue
        names.append(fm.Feature(name, block))
    if spec == "main":
        names += [fm.Feature(n, "C1" if n.startswith("c1") else "C2") for n in MISSING_INDICATORS]
    return names


def stage_task(data, ans, stratum: str, stage: str, spec: str, *, condition="natural"):
    """(Stage object, the answers it uses) for one task; None if the stratum has no usable data."""
    rf = data.rf
    mask = strata(ans)[stratum] & (ans.condition == condition if condition != "all" else True)
    columns = {name: rf[name] for name, *_ in ROW_FEATURES}
    columns.update({k: rf[k] for k in MISSING_INDICATORS})
    columns["u"] = rf["u"]
    features = _features(spec, stage)
    if stage in ("R|U", "R0|U"):
        u = data.u
        keep = mask[u["answer"]]
        if spec == "complete":
            keep &= (rf["c1_dfs_missing"][u["row"]] == 0) & (rf["c2_html_missing"][u["row"]] == 0)
        order = np.argsort(u["answer"][keep], kind="stable")
        answer = u["answer"][keep][order]
        row = u["row"][keep][order]
        admitted = (u["retrieved"] if stage == "R|U" else u["replay"])[keep][order]
        topic = u["topic"][keep][order]
        adm, rows_kept = fm.admission_data(answer, admitted)
        if adm.groups == 0:
            return None
        cols = {k: v[row[rows_kept]] for k, v in columns.items()}
        cols["topic_similarity"] = topic[rows_kept]
        design = fm.Design(features, cols)
        return fm.Stage("admission", design, adm, answer[rows_kept])
    if stage == "P|C":
        c = data.cand
        keep = mask[c["answer"]]
        if spec == "complete":
            keep &= (rf["c1_dfs_missing"][c["row"]] == 0) & (rf["c2_html_missing"][c["row"]] == 0)
        idx = np.flatnonzero(keep)
        if not len(idx):
            return None
        sel = stages.selection_rows(c["event"][idx], c["selected"][idx], c["answer"][idx], z=np.zeros(len(idx)), u=np.zeros(len(idx)),
                                    on_keyword=np.zeros(len(idx)), topic=np.arange(len(idx), dtype=float))
        source = idx[sel.topic.astype(np.int64)]  # selection_rows keeps per-row values; the index rides in 'topic'
        cols = {k: v[c["row"][source]] for k, v in columns.items()}
        cols["topic_similarity"] = c["topic"][source]
        cols["on_keyword"] = c["on_keyword"][source]
        design = fm.Design(features, cols)
        set_answer = np.zeros(sel.sets, np.int64)
        set_answer[sel.set] = c["answer"][source]
        return fm.Stage("choice", design, sel, c["answer"][source], set_answer)
    if stage == "K|P":
        p = data.pres
        keep = mask[p["answer"]]
        if spec == "complete":
            keep &= (rf["c1_dfs_missing"][p["row"]] == 0) & (rf["c2_html_missing"][p["row"]] == 0)
        idx = np.flatnonzero(keep)
        rows_out, sets, chosen, positions, set_answer = [], [], [], [], []
        count = 0
        by_answer = defaultdict(list)
        for j in idx:
            by_answer[int(p["answer"][j])].append(j)
        for a, members in by_answer.items():
            ranked = sorted((m for m in members if p["rank"][m] >= 0), key=lambda m: p["rank"][m])
            remaining = list(members)
            for pick in ranked:
                for alt in remaining:
                    rows_out.append(alt); sets.append(count); chosen.append(alt == pick)
                    positions.append(min(int(p["slot"][alt]), 20))
                remaining.remove(pick)
                set_answer.append(a)
                count += 1
        if not count:
            return None
        src = np.asarray(rows_out, np.int64)
        position = np.asarray(positions, np.int64)
        data_rows = SimpleNamespace(z=np.zeros(len(src)), set=np.asarray(sets, np.int64), chosen=np.asarray(chosen, bool),
                                    position=position, sets=count, levels=int(position.max()) + 1)
        cols = {k: v[p["row"][src]] for k, v in columns.items()}
        cols["topic_similarity"] = p["topic"][src]
        cols["on_keyword"] = p["on_keyword"][src]
        design = fm.Design(features, cols)
        return fm.Stage("choice", design, data_rows, p["answer"][src], np.asarray(set_answer, np.int64))
    raise ValueError(stage)


def standardisation(data, ans) -> dict:
    """Mean and SD of every feature over the U population (natural answers), fixed for all tasks."""
    stats = {}
    u = data.u
    natural = ans.condition[u["answer"]] == "natural"
    rows = u["row"][natural]
    x = ans.x[u["answer"][natural]]
    cols = {name: data.rf[name][rows] for name, *_ in ROW_FEATURES}
    cols.update({k: data.rf[k][rows] for k in MISSING_INDICATORS})
    cols["u"] = data.rf["u"][rows]
    cols["topic_similarity"] = u["topic"][natural]
    cols["on_keyword"] = np.ones(len(rows))
    design = fm.Design(_features("main", "K|P"), cols)
    stats = design.fit_stats(x)
    if stats["on_keyword"][1] == 0 or stats["on_keyword"] == (1.0, 1.0):
        c = data.cand
        stats["on_keyword"] = (float(c["on_keyword"].mean()), float(c["on_keyword"].std()) or 1.0)
    return stats


def tasks(data, ans, specs) -> list:
    return [(s, st, sp) for s in strata(ans) for st in STAGE_NAMES for sp in specs]


def analyze(args) -> int:
    data = load_assembled(args.assembled)
    ans = answer_arrays(data)
    stats = standardisation(data, ans)
    specs = args.specs.split(",")
    every = tasks(data, ans, specs)
    shard, of = (int(v) for v in args.shard.split("/"))
    mine = [t for n, t in enumerate(every) if n % of == shard - 1]
    workers = max(1, args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    draws = stages.keyword_draws(ans.keywords, args.bootstrap, args.seed)
    prompt_x, prompt_keyword = np.full(ans.prompt.max() + 1, np.nan), np.zeros(ans.prompt.max() + 1, np.int64)
    prompt_x[ans.prompt], prompt_keyword[ans.prompt] = ans.x, ans.keyword
    shuffles = [s[ans.prompt] for s in stages.shuffle_draws(prompt_x, prompt_keyword, args.permutations, args.seed + 1)]
    cache = ResultCache(Path(args.output), {"git_commit": readiness.git_commit(), "assembled": data.manifest.get("created_at"),
                                            "settings": [args.bootstrap, args.permutations, args.seed]})
    for stratum, stage, spec in mine:
        name = f"{stratum}|{stage}|{spec}"

        def compute():
            task = stage_task(data, ans, stratum, stage, spec)
            if task is None:
                return {"empty": True}
            task.design.stats = dict(stats)
            return fm.estimate_blocks(task, answer_x=ans.x, answer_keyword=ans.keyword, draws=draws, shuffles=shuffles,
                                      workers=workers)

        result = cache.get(name, compute)
        print(json.dumps({"task": name, "reused": cache.reused, "time": readiness.now(),
                          "or": {k: round(v["odds_ratio_per_sd"], 3) for k, v in result.get("features", {}).items()
                                 if k in ("intent_alignment", "topic_similarity", PRIMARY_C1)}}), flush=True)
    print(json.dumps({"shard": args.shard, "tasks": len(mine), "cache": str(cache.directory)}), flush=True)
    return 0


# ---------------------------------------------------------------- report

DECOMPOSITION_FEATURES = ("intent_alignment", "topic_similarity", "page_intent_z", PRIMARY_C1, "open_pagerank",
                          "google_top20_url", "stored_position", "body_structured_data", "body_word_count")


def decomposition_groups(data, ans, name: str) -> np.ndarray:
    """+1 top quartile / −1 bottom quartile within each (answer) of the U items; binaries 1 vs 0."""
    u = data.u
    if name == "intent_alignment":
        values = -np.abs(data.rf["u"][u["row"]] - ans.x[u["answer"]])
    elif name == "topic_similarity":
        values = u["topic"]
    else:
        values = data.rf[name][u["row"]].astype(float)
    group = np.zeros(len(values), np.int64)
    finite = np.isfinite(values)
    if set(np.unique(values[finite])) <= {0.0, 1.0}:
        group[finite & (values == 1)] = 1
        group[finite & (values == 0)] = -1
        return group
    order = np.lexsort((values, u["answer"]))
    for a, idx in _groups(u["answer"][order], order):
        idx = idx[np.isfinite(values[idx])]
        q = max(1, int(np.ceil(len(idx) / 4)))
        if len(idx) >= 2:
            group[idx[:q]] = -1
            group[idx[-q:]] = 1
    return group


def _groups(keys, payload):
    if not len(keys):
        return []
    cuts = np.flatnonzero(np.diff(keys)) + 1
    return zip(keys[np.r_[0, cuts]], np.split(payload, cuts))


def report(args) -> int:
    data = load_assembled(args.assembled)
    ans = answer_arrays(data)
    specs = args.specs.split(",")
    every = tasks(data, ans, specs)
    cache = ResultCache(Path(args.output), {"git_commit": readiness.git_commit(), "assembled": data.manifest.get("created_at"),
                                            "settings": [args.bootstrap, args.permutations, args.seed]})
    results, missing = {}, []
    for stratum, stage, spec in every:
        name = f"{stratum}|{stage}|{spec}"
        path = cache.directory / f"{hashlib.sha256(name.encode()).hexdigest()[:16]}.json"
        if path.exists():
            results.setdefault(spec, {}).setdefault(stratum, {})[stage] = json.loads(path.read_text())["value"]
        else:
            missing.append(name)
    if missing:
        print(json.dumps({"missing_tasks": missing}), flush=True)
        return 3
    main = results["main"]
    replication = {}
    for stage in STAGE_NAMES:
        per = {s: {"drivers": {k: {"beta_per_sd": v["beta_per_sd"], "ci95": v["ci95"], "permutation_p": v["permutation_p"]}
                               for k, v in main[s][stage].get("features", {}).items()}} for s in main}
        names = sorted({k for s in per.values() for k in s["drivers"]})
        replication[stage] = geo.replication(per, section="drivers", terms=names, x_terms={"intent_alignment", "intent_x_prompt"})
    contrasts = {}
    for s in main:
        r, r0, pc, kp = (main[s].get(st, {}) for st in STAGE_NAMES)
        if r.get("features") and r0.get("features"):
            contrasts.setdefault("R − R0", {})[s] = fm.contrast_from_replicates(r, r0, "intent_alignment")
        if kp.get("features") and pc.get("features"):
            contrasts.setdefault("K|P − P|C (domain authority)", {})[s] = fm.contrast_from_replicates(kp, pc, PRIMARY_C1)
    nulls = {s: {st: fm.empirical_null(main[s][st], "C2") for st in STAGE_NAMES if main[s][st].get("features")} for s in main}
    draws = stages.keyword_draws(ans.keywords, args.bootstrap, args.seed)
    natural = ans.condition[data.u["answer"]] == "natural"
    decomposition = {}
    for s, smask in strata(ans).items():
        keep = natural & smask[data.u["answer"]]
        order = np.argsort(data.u["answer"][keep], kind="stable")
        sub = {k: v[keep][order] for k, v in data.u.items()}
        decomposition[s] = {}
        for feature in DECOMPOSITION_FEATURES:
            sub_data = SimpleNamespace(u=sub, rf=data.rf)
            group = decomposition_groups(sub_data, ans, feature)
            flags = {f: sub[f] for f in fm.STAGE_FLAGS}
            decomposition[s][feature] = fm.rr_decomposition(sub["answer"], flags, group, ans.keyword, draws)
    confirmatory = {
        "P1 alignment at R|U": replication["R|U"].get("intent_alignment"),
        "P2 alignment R − R0": {"strata": contrasts.get("R − R0"),
                                "replicates": bool(contrasts.get("R − R0")) and all(
                                    c["ci95"][0] is not None and c["ci95"][0] > 0 for c in contrasts["R − R0"].values())},
        "P3 domain authority K|P − P|C": {"strata": contrasts.get("K|P − P|C (domain authority)"), "c2_null": nulls},
        "P4 page intent × x at R|U": replication["R|U"].get("intent_x_prompt")}
    out = {"format_version": FORMAT_VERSION, "created_at": readiness.now(), "git_commit": readiness.git_commit(),
           "settings": {"bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed, "specs": specs},
           "assembled": data.manifest, "models": results, "replication": replication, "contrasts": contrasts,
           "negative_control_null": nulls, "decomposition": decomposition, "confirmatory": confirmatory,
           "scientific_result": True, "observational": True}
    out = _finite(out)
    target = Path(args.report)
    refuse_existing(target)
    partial = readiness.new_directory(target)
    readiness.write_json(partial / "results.json", out)
    (partial / "final-report.html").write_text(render(out), encoding="utf-8")
    partial.rename(target.resolve())
    print(f"REPORT {target.resolve() / 'final-report.html'}", flush=True)
    return 0


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def render(out: dict) -> str:
    def cell(entry):
        if not entry or entry.get("odds_ratio_per_sd") is None:
            return "<td>—</td>"
        lo, hi = entry["ci95"]
        ci = f" [{np.exp(lo):.2f}, {np.exp(hi):.2f}]" if lo is not None else ""
        p = f"; p={entry['permutation_p']:.3f}" if entry.get("permutation_p") is not None else ""
        return f"<td>{entry['odds_ratio_per_sd']:.2f}{ci}{p}</td>"

    main = out["models"]["main"]
    blocks = []
    for s, by_stage in main.items():
        names = sorted({k for st in by_stage.values() for k in st.get("features", {})},
                       key=lambda n: next((st["features"][n]["block"] for st in by_stage.values() if n in st.get("features", {})), "") + n)
        head = "".join(f"<th>{html.escape(st)}</th>" for st in STAGE_NAMES)
        body = "".join(
            f"<tr><td>{html.escape(n)}</td>" + "".join(cell(by_stage.get(st, {}).get("features", {}).get(n)) for st in STAGE_NAMES) + "</tr>"
            for n in names)
        blocks.append(f"<h3>{html.escape(s)}</h3><table><tr><th>feature (odds ratio per SD [95% CI])</th>{head}</tr>{body}</table>")
    conf = html.escape(json.dumps(out["confirmatory"], indent=1)[:20000])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Funnel Study</title>
<style>body{{font:14px/1.5 -apple-system,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}}table{{border-collapse:collapse;margin:1em 0}}
td,th{{border:1px solid #ccc;padding:3px 7px;font-size:12px}}pre{{background:#f5f5f5;padding:1em;overflow:auto}}</style></head><body>
<h1>Funnel study: admission into the pool and ranking within it</h1>
<p>Observational. Generated {html.escape(out['created_at'])} from commit {html.escape(out['git_commit'][:12])};
{out['settings']['bootstrap']} keyword-bootstrap draws, {out['settings']['permutations']} within-keyword shuffles.
Stages: R|U retrieval admission among the keyword's own rows; R0|U the frozen search on the prompt text itself;
P|C reranker selection; K|P generator ranking.</p>
<h2>Confirmatory family</h2><pre>{conf}</pre>
<h2>Stage models (main specification)</h2>{''.join(blocks)}
</body></html>"""


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--source", action="append", required=True, metavar="DATASET_ROOT:MODEL")
    p.add_argument("--snapshot", action="append", required=True, metavar="ENGINE=PATH")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--population-prompts", type=Path, required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("replay")
    p.add_argument("--snapshot", action="append", required=True, metavar="ENGINE=PATH")
    p.add_argument("--population-prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("assemble")
    p.add_argument("--extract", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--replay", type=Path)
    p.add_argument("--corpus-package", type=Path, required=True)
    p.add_argument("--qwen-prompts", type=Path, required=True)
    p.add_argument("--mistral-prompts", type=Path, required=True)
    p.add_argument("--qwen-map", type=Path, required=True)
    p.add_argument("--mistral-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for name in ("analyze", "report"):
        p = sub.add_parser(name)
        p.add_argument("--assembled", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True, help="analysis folder; its .cache holds finished tasks")
        p.add_argument("--specs", default="main,visible,complete")
        p.add_argument("--bootstrap", type=int, default=200)
        p.add_argument("--permutations", type=int, default=200)
        p.add_argument("--seed", type=int, default=20261007)
        if name == "analyze":
            p.add_argument("--shard", default="1/1")
            p.add_argument("--workers", type=int)
        else:
            p.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"extract": extract, "replay": replay, "assemble": assemble, "analyze": analyze, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
