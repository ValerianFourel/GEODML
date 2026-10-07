"""Merge keyword-hash shards of the two extraction stages into the single folders the analyses read.

Shards come from ``funnel_study.py extract --prompt-shard K/N`` and ``intent_stages_study.py trace-extract
--prompt-shard K/N``. They are concatenated in the given order; a cell present twice (the same model, prompt, engine,
method and condition, e.g. a HoreKa dataset root and a Hub import) keeps its first occurrence. The merged folders have
exactly the layout of an unsharded run, plus ``merge`` provenance in the manifest.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from analysis.scripts import page_readiness_ordering as readiness


def _csr_take(offsets: np.ndarray, keep: np.ndarray, *arrays: np.ndarray):
    """Rows of the kept groups of a CSR layout: (sizes of kept groups, concatenated arrays)."""
    sizes = np.diff(offsets)
    mask = np.repeat(keep, sizes)
    return sizes[keep], [a[mask] for a in arrays]


def _offsets(sizes) -> np.ndarray:
    return np.r_[0, np.cumsum(np.concatenate(sizes) if sizes else np.zeros(0, np.int64))].astype(np.int64)


def non_empty(shards) -> list[Path]:
    """Shard folders that hold answers (an empty keyword-hash shard writes only a manifest with ``empty_shard``)."""
    out = [Path(s) for s in shards if not json.loads((Path(s) / "manifest.json").read_text()).get("empty_shard")]
    if not out:
        raise ValueError("every shard is empty")
    return out


def merge_funnel_extract(shards: list[Path], output: Path) -> dict:
    shards = non_empty(shards)
    partial = readiness.new_directory(output)
    seen, answers = set(), []
    item_sizes, items = [], {k: [] for k in ("row", "search", "rank", "scored", "presented", "ranked")}
    ev_answer, ev_search, cand_sizes, cands = [], [], [], {k: [] for k in ("row", "score", "selected")}
    counts, manifests, digests = Counter(), [], set()
    for shard in map(Path, shards):
        manifest = json.loads((shard / "manifest.json").read_text())
        manifests.append({"path": str(shard), "prompt_shard": manifest.get("prompt_shard"), "answers": manifest["counts"]["answers"]})
        digests.add((manifest["row_table_digest"], json.dumps(manifest["snapshot_sha256"], sort_keys=True)))
        counts.update({k: v for k, v in manifest["counts"].items() if k not in ("answers", "items", "events")})
        rows = readiness.read_jsonl(shard / "answers.jsonl.gz")
        keep = np.zeros(len(rows), bool)
        new_index = np.full(len(rows), -1, np.int64)
        for i, r in enumerate(rows):
            key = (r["model"], r["prompt_id"], r["engine"], r["method"], r["condition"])
            if key in seen:
                counts["duplicate_cells_dropped"] += 1
                continue
            seen.add(key)
            keep[i] = True
            new_index[i] = len(answers)
            answers.append(r)
        it = np.load(shard / "items.npz")
        sizes, arrays = _csr_take(it["offsets"], keep, *(it[k] for k in items))
        item_sizes.append(sizes)
        for k, a in zip(items, arrays):
            items[k].append(a)
        ev = np.load(shard / "events.npz")
        ev_keep = keep[ev["answer"]] if len(ev["answer"]) else np.zeros(0, bool)
        sizes, arrays = _csr_take(ev["offsets"], ev_keep, ev["row"], ev["score"], ev["selected"])
        cand_sizes.append(sizes)
        for k, a in zip(cands, arrays):
            cands[k].append(a)
        ev_answer.append(new_index[ev["answer"][ev_keep]])
        ev_search.append(ev["search"][ev_keep])
    if len(digests) != 1:
        raise ValueError(f"shards were extracted from different snapshot row tables: {digests}")
    cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)  # noqa: E731
    readiness.write_jsonl(partial / "answers.jsonl.gz", answers)
    readiness.write_npz(partial / "items.npz", offsets=_offsets(item_sizes), **{k: cat(v, np.int64) for k, v in items.items()})
    readiness.write_npz(partial / "events.npz", answer=cat(ev_answer, np.int64), search=cat(ev_search, np.int64),
                        offsets=_offsets(cand_sizes), row=cat(cands["row"], np.int64), score=cat(cands["score"], float),
                        selected=cat(cands["selected"], np.int64))
    digest, snapshots = next(iter(digests))
    first = json.loads((Path(shards[0]) / "manifest.json").read_text())
    summary = {**counts, "answers": len(answers), "items": int(sum(len(a) for a in items["row"])),
               "events": int(sum(len(a) for a in ev_answer))}
    readiness.write_json(partial / "manifest.json", {
        "format_version": first.get("format_version"), "stage": "extract", "created_at": readiness.now(),
        "git_commit": readiness.git_commit(), "inputs": first.get("inputs"), "snapshot_sha256": json.loads(snapshots),
        "row_table_digest": digest, "snapshot_hash_mismatch": {}, "counts": summary,
        "merge": {"shards": manifests, "rule": "concatenate in order; first occurrence of (model, prompt, engine, method, condition)"}})
    partial.rename(Path(output).resolve())
    return summary


STAGE_CODES = ("model", "engine", "method", "condition", "keyword", "prompt")
CSR = ("ret", "cand", "pool", "rank")


def merge_trace_extract(shards: list[Path], output: Path) -> dict:
    shards = non_empty(shards)
    partial = readiness.new_directory(output)
    codes = {k: {} for k in STAGE_CODES}
    columns = {k: [] for k in (*STAGE_CODES, "x")}
    csr_sizes, csr_docs = {n: [] for n in CSR}, {n: [] for n in CSR}
    q_sizes, q_ids, query_index, queries = [], [], {}, []
    ev = {k: [] for k in ("answer", "row_sizes", "doc", "score", "selected")}
    generation_ids, prompts, sample, counts, manifests = [], {}, {}, Counter(), []
    seen = set()
    for shard in map(Path, shards):
        manifest = json.loads((shard / "manifest.json").read_text())
        manifests.append({"path": str(shard), "prompt_shard": manifest.get("prompt_shard"), "answers": manifest["counts"]["answers"]})
        counts.update({k: v for k, v in manifest["counts"].items() if isinstance(v, (int, float)) and k not in
                       ("answers", "prompts", "unique_queries", "query_uses", "compaction_events", "scored_candidates", "fidelity_sample")})
        st = np.load(shard / "stages.npz")
        shard_codes = json.loads((shard / "codes.json").read_text())
        shard_q = [r["text"] for r in readiness.read_jsonl(shard / "queries.jsonl.gz")]
        gen = [r["generation_id"] for r in readiness.read_jsonl(shard / "answers.jsonl.gz")]
        labels = {k: np.asarray(shard_codes[k], object)[st[k]] for k in STAGE_CODES}
        n = len(gen)
        keep = np.zeros(n, bool)
        new_index = np.full(n, -1, np.int64)
        for i in range(n):
            key = tuple(labels[k][i] for k in ("model", "prompt", "engine", "method", "condition"))
            if key in seen:
                counts["duplicate_cells_dropped"] += 1
                continue
            seen.add(key)
            keep[i] = True
            new_index[i] = len(generation_ids)
            generation_ids.append(gen[i])
            for k in STAGE_CODES:
                columns[k].append(codes[k].setdefault(labels[k][i], len(codes[k])))
            columns["x"].append(float(st["x"][i]))
        for name in CSR:
            sizes, (docs,) = _csr_take(st[f"{name}_offsets"], keep, st[f"{name}_docs"])
            csr_sizes[name].append(sizes)
            csr_docs[name].append(docs)
        sizes, (qs,) = _csr_take(st["q_offsets"], keep, st["q_ids"])
        q_sizes.append(sizes)
        for q in qs:
            text = shard_q[q]
            if text not in query_index:
                query_index[text] = len(queries)
                queries.append(text)
        q_ids.append(np.asarray([query_index[shard_q[q]] for q in qs], np.int64))
        ev_keep = keep[st["event_answer"]] if len(st["event_answer"]) else np.zeros(0, bool)
        sizes, (doc, score, sel) = _csr_take(st["event_row_offsets"], ev_keep, st["event_doc"], st["event_score"], st["event_selected"])
        ev["row_sizes"].append(sizes)
        ev["doc"].append(doc); ev["score"].append(score); ev["selected"].append(sel)
        ev["answer"].append(new_index[st["event_answer"][ev_keep]])
        for r in readiness.read_jsonl(shard / "prompts.jsonl.gz"):
            prompts.setdefault(r["prompt_id"], r)
        for r in readiness.read_jsonl(shard / "fidelity-sample.jsonl.gz"):
            sample.setdefault((r["engine"], r["query"]), r)
    cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)  # noqa: E731
    readiness.write_npz(partial / "stages.npz", **{k: np.asarray(columns[k], float if k == "x" else np.int64) for k in (*STAGE_CODES, "x")},
                        **{f"{n}_offsets": _offsets(csr_sizes[n]) for n in CSR}, **{f"{n}_docs": cat(csr_docs[n], np.int64) for n in CSR},
                        q_offsets=_offsets(q_sizes), q_ids=cat(q_ids, np.int64), event_answer=cat(ev["answer"], np.int64),
                        event_row_offsets=_offsets(ev["row_sizes"]), event_doc=cat(ev["doc"], np.int64),
                        event_score=cat(ev["score"], float), event_selected=cat(ev["selected"], np.int64))
    readiness.write_json(partial / "codes.json", {k: sorted(v, key=v.get) for k, v in codes.items()})
    readiness.write_jsonl(partial / "prompts.jsonl.gz", (prompts[p] for p in sorted(codes["prompt"], key=codes["prompt"].get)))
    readiness.write_jsonl(partial / "answers.jsonl.gz", ({"answer": i, "generation_id": g} for i, g in enumerate(generation_ids)))
    import hashlib
    readiness.write_jsonl(partial / "queries.jsonl.gz", ({"page_id": h, "text": q, "text_sha256": h}
                                                         for q, h in ((q, hashlib.sha256(q.encode()).hexdigest()) for q in queries)))
    readiness.write_jsonl(partial / "fidelity-sample.jsonl.gz", sample.values())
    first = json.loads((Path(shards[0]) / "manifest.json").read_text())
    summary = {**counts, "answers": len(generation_ids), "prompts": len(codes["prompt"]), "unique_queries": len(queries),
               "query_uses": int(sum(len(a) for a in q_ids)), "compaction_events": int(sum(len(a) for a in ev["answer"])),
               "scored_candidates": int(sum(len(a) for a in ev["doc"])), "fidelity_sample": len(sample)}
    manifest = {k: v for k, v in first.items() if k not in ("counts", "created_at", "git_commit", "prompt_shard")}
    readiness.write_json(partial / "manifest.json", {**manifest, "created_at": readiness.now(), "git_commit": readiness.git_commit(),
                                                     "counts": summary, "merge": {"shards": manifests}})
    partial.rename(Path(output).resolve())
    return summary
