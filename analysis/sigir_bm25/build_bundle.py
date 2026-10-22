#!/usr/bin/env python3
"""Inputs for the BM25 generation rerun, built on the Mac from the release dataset (no model is called).

  cells       --release D --out B             cells-<model>.parquet per generator (natural condition, both methods,
                                              both engines, every prompt; Parallel cells carry their three logged
                                              queries and are left out when none were logged), prompts.parquet,
                                              manifest.json; a seeded random order interleaves methods and engines,
                                              so any finished prefix is an unbiased sample
  check-bm25  --release D --searches S        the vendored BM25 against the ARR rebuttal's bm25l top-20 lists
                                              (searches_bm25l.parquet): every list must be identical
  pairs       --release D --bundle B --out P [--scores RUN ...]
                                              the (prompt, row) reranker pairs of every Parallel cell under BM25
                                              that no logged or listed score covers, as a score_pairs.py bundle with
                                              its shard plan (P/shards.parquet goes to RUN/results/)
  cache       --release D --bundle B --out C [--scores RUN ...]
                                              score-cache parquet (key, score, source): logged Parallel scores first,
                                              then each score_pairs run's shards

cell_id is the original controller's: sha256 of the canonical {condition, engine, method, prompt_id,
prompt_sha256}[:20] (it reproduces the release's cell_id for every answer); answer_id is the release's
sha256("model|prompt_id|engine|method|condition")[:24].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

from analysis.interpretability.pipeline.agentic_generation_tasks import _canonical
from analysis.interpretability.pipeline.agentic_search import Snippet
from analysis.sigir_bm25.bm25 import ANALYZER, SELECTION, Bm25Index
from analysis.sigir_bm25.score_cache import KEY_BYTES, pair_key

MODELS = ("qwen38", "llama4")
ENGINES = ("duckduckgo", "searxng")
PARALLEL, REACTIVE = "Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"
ORDER_SEED = 20261022
SCORE_TABLES = ("pages", "prompts", "queries", "answers", "answer_queries", "pairs_to_score", "pairs_logged")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_id(method: str, engine: str, prompt_id: str, prompt_sha256: str) -> str:
    core = {"method": method, "engine": engine, "condition": "natural", "prompt_id": prompt_id,
            "prompt_sha256": prompt_sha256}
    return hashlib.sha256(_canonical(core)).hexdigest()[:20]


def answer_id(model: str, prompt_id: str, engine: str, method: str) -> str:
    return hashlib.sha256(f"{model}|{prompt_id}|{engine}|{method}|natural".encode()).hexdigest()[:24]


def logged_queries(release: Path) -> dict[str, list[str]]:
    """answer_id -> the three logged queries in search order, for every natural Parallel trace."""
    ta = pd.read_parquet(release / "trace_answers.parquet", columns=["answer_id", "method", "condition"])
    keep = set(ta.loc[(ta["method"].astype(str) == PARALLEL) & (ta["condition"].astype(str) == "natural"),
                      "answer_id"].astype(str))
    aq = pd.read_parquet(release / "answer_queries.parquet")
    aq = aq[aq["answer_id"].astype(str).isin(keep)]
    text = pd.read_parquet(release / "queries.parquet", columns=["query_id", "text"]).set_index("query_id")["text"]
    aq = aq.assign(text=text.reindex(aq["query_id"]).to_numpy()).sort_values(["answer_id", "search_index"])
    out = {}
    for aid, f in aq.groupby("answer_id", sort=False):
        q = f["text"].astype(str).tolist()
        if f["search_index"].tolist() == [0, 1, 2] and len({x.casefold().strip() for x in q}) == 3:
            out[str(aid)] = q
    return out


def cmd_cells(a) -> int:
    release, out = Path(a.release), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = pd.read_parquet(release / "prompts.parquet", columns=["prompt_id", "text"])
    prompts = prompts.assign(prompt_id=prompts["prompt_id"].astype(str), text=prompts["text"].astype(str))
    prompts["prompt_sha256"] = [hashlib.sha256(t.encode()).hexdigest() for t in prompts["text"]]
    queries = logged_queries(release)
    released = pd.read_parquet(release / "answers.parquet", columns=["cell_id", "condition"])
    released_cells = set(released.loc[released["condition"].astype(str) == "natural", "cell_id"].astype(str))
    manifest = {"format_version": "sigir-bm25-bundle-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "release": str(release), "condition": "natural", "order_seed": ORDER_SEED,
                "sources": json.loads((release / "trace_export.json").read_text()).get("tables"),
                "analyzer": ANALYZER, "selection": SELECTION, "unidata_version": unicodedata.unidata_version,
                "cells": {}, "cells_sha256": {}}
    pq_path = out / "prompts.parquet"
    prompts[["prompt_id", "text", "prompt_sha256"]].to_parquet(pq_path, index=False)
    manifest["prompts_sha256"] = sha256_file(pq_path)
    for model in a.models.split(","):
        rows = []
        for p, sha in zip(prompts["prompt_id"], prompts["prompt_sha256"]):
            for engine in ENGINES:
                for method in (PARALLEL, REACTIVE):
                    aid = answer_id(model, p, engine, method)
                    q = queries.get(aid) if method == PARALLEL else None
                    if method == PARALLEL and q is None:
                        continue
                    rows.append((cell_id(method, engine, p, sha), aid, model, p, sha, method, engine, q))
        cells = pd.DataFrame(rows, columns=["cell_id", "answer_id", "model", "prompt_id", "prompt_sha256", "method",
                                            "engine", "queries"])
        cells["order"] = np.random.default_rng(ORDER_SEED).permutation(len(cells))
        cells = cells.sort_values("order").reset_index(drop=True)
        in_release = cells["cell_id"].isin(released_cells)
        path = out / f"cells-{model}.parquet"
        cells.to_parquet(path, index=False)
        design = len(prompts) * len(ENGINES)
        manifest["cells"][model] = {
            "total": int(len(cells)),
            "by_method_engine": {f"{m} · {e}": int(n) for (m, e), n in cells.groupby(["method", "engine"]).size().items()},
            "parallel_without_logged_queries": int(design - (cells["method"] == PARALLEL).sum()),
            "cell_ids_in_release_answers": int(in_release.sum())}
        manifest["cells_sha256"][model] = sha256_file(path)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("cells", "cells_sha256")}, indent=1))
    return 0


def engine_indexes(release: Path) -> dict[str, tuple[pd.DataFrame, Bm25Index]]:
    pages = pd.read_parquet(release / "pages.parquet",
                            columns=["row_id", "engine", "keyword", "position", "url", "title", "snippet"])
    out = {}
    for e in ENGINES:
        rows = pages[pages["engine"].astype(str) == e].sort_values("row_id").reset_index(drop=True)
        out[e] = (rows, Bm25Index.from_rows(rows.to_dict("records")))
    return out


def cmd_check_bm25(a) -> int:
    release = Path(a.release)
    searches = pd.read_parquet(a.searches)
    qtext = pd.read_parquet(release / "queries.parquet", columns=["query_id", "text"])
    qtext = dict(zip(qtext["query_id"].astype(str), qtext["text"].astype(str)))
    ptext = pd.read_parquet(release / "prompts.parquet", columns=["prompt_id", "text"])
    ptext = dict(zip(ptext["prompt_id"].astype(str), ptext["text"].astype(str)))
    report, ok = {}, True
    for e, (rows, index) in engine_indexes(release).items():
        rid = rows["row_id"].to_numpy()
        groups = searches[searches["engine"].astype(str) == e].sort_values(["kind", "key", "rank"]) \
            .groupby(["kind", "key"], sort=False)["row_id"].apply(list)
        if a.sample:
            groups = groups.sample(min(a.sample, len(groups)), random_state=1)
        pos = {int(r): i for i, r in enumerate(rid)}
        t0, bad, ties = time.time(), [], []
        for (kind, key), want in groups.items():
            text = qtext[str(key)] if kind == "q" else ptext[str(key)]
            got = [int(rid[i]) for i, _ in index.top(text, 20)]
            want = [int(x) for x in want]
            if got == want:
                continue
            # A swap of rows whose scores differ only in the last bits (summation order) is a tie, not a mismatch.
            score = index.scores(text)
            swapped = [(g, w) for g, w in zip(got, want) if g != w]
            if sorted(got) == sorted(want) and all(abs(score[pos[g]] - score[pos[w]]) <= 1e-12 * max(abs(score[pos[g]]), 1)
                                                   for g, w in swapped):
                ties.append(f"{kind}:{key}")
            else:
                bad.append(f"{kind}:{key}")
        report[e] = {"searches": int(len(groups)), "identical": int(len(groups) - len(bad) - len(ties)),
                     "ulp_ties": ties[:20], "n_ulp_ties": len(ties), "mismatches": bad[:20],
                     "fingerprint": index.fingerprint, "seconds": round(time.time() - t0, 1)}
        ok &= not bad
    print(json.dumps({"passed": ok, **report}, indent=1))
    return 0 if ok else 3


def bm25_parallel_pairs(release: Path, bundle: Path, prompt_ids: set[str] | None = None) -> pd.DataFrame:
    """(prompt_id, row_id) of every Parallel cell's BM25 pool: the three logged queries' top 20, first row per URL."""
    idx = engine_indexes(release)
    pairs = set()
    for path in sorted(bundle.glob("cells-*.parquet")):
        cells = pd.read_parquet(path)
        cells = cells[cells["method"] == PARALLEL]
        if prompt_ids is not None:
            cells = cells[cells["prompt_id"].isin(prompt_ids)]
        for e, f in cells.groupby("engine"):
            rows, index = idx[e]
            rid, url = rows["row_id"].to_numpy(), rows["url"].tolist()
            for p, queries in zip(f["prompt_id"], f["queries"]):
                seen = set()
                for q in queries:
                    for i, _ in index.top(str(q), 20):
                        if url[i] not in seen:
                            seen.add(url[i])
                            pairs.add((str(p), int(rid[i])))
    return pd.DataFrame(sorted(pairs), columns=["prompt_id", "row_id"])


def known_scores(release: Path, runs: list[str]) -> pd.DataFrame:
    """(prompt_id, row_id, score, source): logged Parallel scores, then each score_pairs run's shards."""
    ta = pd.read_parquet(release / "trace_answers.parquet", columns=["answer_id", "prompt_id", "method"])
    ta = ta[ta["method"].astype(str) == PARALLEL]
    pid = dict(zip(ta["answer_id"].astype(str), ta["prompt_id"].astype(str)))
    ev = pd.read_parquet(release / "rerank_events.parquet", columns=["answer_id", "row_id", "score"])
    ev = ev.assign(prompt_id=ev["answer_id"].astype(str).map(pid)).dropna(subset=["prompt_id"])
    frames = [ev[["prompt_id", "row_id", "score"]].assign(source="logged")]
    for run in runs:
        shards = sorted((Path(run) / "results" / "scores").glob("shard-*.parquet"))
        if shards:
            s = pd.concat([pd.read_parquet(x, columns=["prompt_id", "row_id", "score", "fake"]) for x in shards])
            if s["fake"].any():
                raise SystemExit(f"fake scores (tests only) in {run}")
            frames.append(s[["prompt_id", "row_id", "score"]].assign(source=f"gpu:{Path(run).name}"))
    k = pd.concat(frames, ignore_index=True)
    k = k.assign(prompt_id=k["prompt_id"].astype(str), row_id=k["row_id"].astype(np.int64))
    return k.drop_duplicates(["prompt_id", "row_id"], keep="first")


def cmd_pairs(a) -> int:
    release, bundle, out = Path(a.release), Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    keep = ({x.strip() for x in Path(a.prompt_ids).read_text().splitlines() if x.strip()} if a.prompt_ids else None)
    need = bm25_parallel_pairs(release, bundle, keep)
    known = known_scores(release, a.scores)
    todo = need.merge(known[["prompt_id", "row_id"]], how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge")
    todo = todo.sort_values(["prompt_id", "row_id"]).reset_index(drop=True).assign(purpose="new")
    todo["shard"] = np.arange(len(todo)) // a.shard_size
    pages = pd.read_parquet(release / "pages.parquet", columns=["row_id", "engine", "keyword", "position", "url",
                                                                 "title", "snippet"])
    prompts = pd.read_parquet(bundle / "prompts.parquet")
    tables = {"pages": pages, "prompts": prompts[prompts["prompt_id"].isin(set(todo["prompt_id"]))][["prompt_id", "text"]],
              "queries": pd.DataFrame({"query_id": pd.Series([], dtype="int64"), "text": pd.Series([], dtype=str)}),
              "answers": pd.DataFrame({"answer_id": pd.Series([], dtype=str)}),
              "answer_queries": pd.DataFrame({"answer_id": pd.Series([], dtype=str)}),
              "pairs_to_score": todo[["prompt_id", "row_id", "purpose"]],
              "pairs_logged": pd.DataFrame({"prompt_id": pd.Series([], dtype=str)})}
    files = {}
    for name in SCORE_TABLES:
        path = out / f"{name}.parquet"
        tables[name].to_parquet(path, index=False)
        files[path.name] = {"sha256": sha256_file(path), "rows": int(len(tables[name]))}
    todo[["prompt_id", "row_id", "purpose", "shard"]].to_parquet(out / "shards.parquet", index=False)
    summary = {"pairs_needed": int(len(need)), "covered": int(len(need) - len(todo)), "to_score": int(len(todo)),
               "shards": int(todo["shard"].nunique()), "shard_size": a.shard_size}
    (out / "manifest.json").write_text(json.dumps({"format_version": "sigir-bm25-pairs-v1", "files": files,
                                                   **summary}, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


def cmd_cache(a) -> int:
    release, out = Path(a.release), Path(a.out)
    known = known_scores(release, a.scores)
    pages = pd.read_parquet(release / "pages.parquet", columns=["row_id", "url", "title", "snippet"]).set_index("row_id")
    prompts = pd.read_parquet(Path(a.bundle) / "prompts.parquet")
    ptext = dict(zip(prompts["prompt_id"], prompts["text"]))
    snip = {int(r): Snippet.from_mapping({"url": u, "title": t, "text": s})
            for r, u, t, s in zip(pages.index, pages["url"], pages["title"], pages["snippet"])}
    keys = [pair_key(ptext[p], snip[r].url, snip[r].title, snip[r].text)
            for p, r in zip(known["prompt_id"], known["row_id"])]
    table = pd.DataFrame({"key": keys, "score": known["score"].astype(np.float64).to_numpy(),
                          "source": known["source"].to_numpy()})
    if table["key"].map(len).ne(KEY_BYTES).any():
        raise SystemExit("cache keys must be 16-byte digests")
    # The same page can be a row of both engine snapshots: one key, so keep the first score (logged before GPU).
    dup = table[table["key"].duplicated(keep=False)]
    spread = float(dup.groupby("key")["score"].agg(lambda s: s.max() - s.min()).max()) if len(dup) else 0.0
    table = table.drop_duplicates("key", keep="first")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(json.dumps({"entries": int(len(table)), "by_source": table["source"].value_counts().to_dict(),
                      "duplicate_keys_dropped": int(len(dup) - dup["key"].nunique()),
                      "max_score_spread_within_duplicates": spread, "sha256": sha256_file(out)}, indent=1))
    return 0


def cmd_sample_prompts(a) -> int:
    """A fixed random sample of prompts (one split), as a prompt-id list for generate.py --prompt-ids."""
    release, bundle, out = Path(a.release), Path(a.bundle), Path(a.out)
    ans = pd.read_parquet(release / "answers.parquet", columns=["prompt_id", "split"]).drop_duplicates()
    order = pd.read_parquet(bundle / "prompts.parquet", columns=["prompt_id"])["prompt_id"].tolist()  # release order
    allowed = set(order) if a.split == "any" else set(ans.loc[ans["split"].astype(str) == a.split, "prompt_id"].astype(str))
    ids = [p for p in order if p in allowed]
    if a.n > len(ids):
        raise SystemExit(f"only {len(ids)} {a.split} prompts")
    if a.order == "first":
        pick = ids[:a.n]
    else:
        ids = sorted(ids)
        pick = [ids[i] for i in np.random.default_rng(a.seed).permutation(len(ids))[:a.n]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(p + "\n" for p in pick))
    print(json.dumps({"prompts": len(pick), "split": a.split, "order": a.order, "pool": len(ids), "seed": a.seed,
                      "sha256": sha256_file(out), "out": str(out)}, indent=1))
    return 0


def cmd_sample_cells(a) -> int:
    """A seeded random sample of one model's cells, N per method, from one split's prompts (cell-id lists per method)."""
    release, bundle, out = Path(a.release), Path(a.bundle), Path(a.out)
    ans = pd.read_parquet(release / "answers.parquet", columns=["prompt_id", "split"]).drop_duplicates()
    allowed = set(ans.loc[ans["split"].astype(str) == a.split, "prompt_id"].astype(str)) if a.split != "any" else None
    cells = pd.read_parquet(bundle / f"cells-{a.model}.parquet")
    if allowed is not None:
        cells = cells[cells["prompt_id"].isin(allowed)]
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": a.model, "split": a.split, "seed": a.seed, "per_method": a.n_per_method, "files": {}}
    for method, short in ((PARALLEL, "parallel"), (REACTIVE, "reactive")):
        pool = cells[cells["method"] == method].sort_values("cell_id")
        pick = pool.iloc[np.random.default_rng(a.seed).permutation(len(pool))[:a.n_per_method]]
        path = out / f"{a.model}-{short}-{len(pick)}.txt"
        path.write_text("".join(c + "\n" for c in pick["cell_id"]))
        report["files"][short] = {"path": str(path), "cells": int(len(pick)), "pool": int(len(pool)),
                                  "engines": pick["engine"].value_counts().to_dict(),
                                  "prompts": int(pick["prompt_id"].nunique()), "sha256": sha256_file(path)}
    print(json.dumps(report, indent=1))
    return 0


def cmd_snapshots(a) -> int:
    """Per-engine snapshot JSONL from the release rows (row_id order), the frozen search's own input format.
    The rows are the inference-time snapshot's usable rows (gate G1; check-orig reproduced the logged shortlists),
    but the file bytes differ from the HoreKa originals, so traces carry this file's own sha256."""
    release, out = Path(a.release), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = pd.read_parquet(release / "pages.parquet", columns=["row_id", "engine", "keyword", "position", "url",
                                                                 "title", "snippet"]).sort_values("row_id")
    report = {}
    for e in ENGINES:
        rows = pages[pages["engine"].astype(str) == e][["keyword", "position", "title", "url", "snippet"]]
        path = out / f"{e}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.to_dict("records")))
        report[e] = {"rows": int(len(rows)), "sha256": sha256_file(path), "path": str(path)}
    print(json.dumps(report, indent=1))
    return 0


def cmd_check_orig(a) -> int:
    """Original search + logged queries + cached scores, no LLM: must reproduce each sampled logged shortlist."""
    import asyncio
    import re
    import tempfile

    from analysis.interpretability.pipeline.agentic_search import (ContextCompactor, ExperimentalCondition,
                                                                   IdentityConditionHook, ParallelExpansionV1)
    from analysis.scripts.run_agentic_search_integration_smoke import FrozenSnapshotSearchAdapter
    from analysis.sigir_bm25.score_cache import CachedScorer, load_cache

    class Stub:  # logged queries for the expansion; any valid ranking for the final call
        def __init__(self, queries):
            self.queries = queries

        async def generate(self, request):
            if request.purpose == "parallel_query_expansion":
                return json.dumps({"queries": self.queries})
            ids = re.findall(r'"evidence_id": ?"([^"]+)"', request.prompt)
            return json.dumps({"ranking": ids[:1], "answer": "check"})

    release, bundle = Path(a.release), Path(a.bundle)
    pages = pd.read_parquet(release / "pages.parquet", columns=["row_id", "engine", "keyword", "position", "url",
                                                                 "title", "snippet"]).sort_values("row_id")
    url_of = dict(zip(pages["row_id"].astype(int), pages["url"]))
    shown = pd.read_parquet(release / "shown.parquet", columns=["answer_id", "row_id", "slot"])
    prompts = pd.read_parquet(bundle / "prompts.parquet")
    ptext = dict(zip(prompts["prompt_id"], prompts["text"]))
    scorer = CachedScorer(None, load_cache([Path(a.cache)]), model_id="score-cache")
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        adapters = {}
        for e in ENGINES:
            path = Path(tmp) / f"{e}.jsonl"
            rows = pages[pages["engine"].astype(str) == e][["keyword", "position", "title", "url", "snippet"]]
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.to_dict("records")))
            adapters[e] = FrozenSnapshotSearchAdapter(e, path)
        for path in sorted(bundle.glob("cells-*.parquet")):
            cells = pd.read_parquet(path)
            cells = cells[(cells["method"] == PARALLEL) & cells["answer_id"].isin(set(shown["answer_id"].astype(str)))]
            cells = cells.sample(min(a.sample, len(cells)), random_state=1)
            want = shown[shown["answer_id"].astype(str).isin(set(cells["answer_id"]))].sort_values("slot") \
                .groupby("answer_id")["row_id"].apply(lambda s: [url_of[int(r)] for r in s])
            same, bad = 0, []
            for c in cells.itertuples(index=False):
                method = ParallelExpansionV1(llm=Stub(list(c.queries)), search=adapters[c.engine],
                                             compactor=ContextCompactor(scorer), condition_hook=IdentityConditionHook())
                result = asyncio.run(method.run(ptext[c.prompt_id], ExperimentalCondition.NATURAL))
                got = [s.url for s in result.final_snippets]
                if got == want.get(c.answer_id):
                    same += 1
                else:
                    bad.append(c.answer_id)
            report[path.stem] = {"cells": int(len(cells)), "identical_shortlists": same, "mismatches": bad[:20]}
    ok = all(r["identical_shortlists"] == r["cells"] for r in report.values())
    print(json.dumps({"passed": ok, **report, "score_cache": dict(scorer.counts)}, indent=1))
    return 0 if ok else 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cells")
    c.add_argument("--release", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--models", default=",".join(MODELS))
    k = sub.add_parser("check-bm25")
    k.add_argument("--release", required=True)
    k.add_argument("--searches", required=True)
    k.add_argument("--sample", type=int, default=0)
    for name in ("pairs", "cache"):
        s = sub.add_parser(name)
        s.add_argument("--release", required=True)
        s.add_argument("--bundle", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--scores", action="append", default=[], help="a score_pairs.py run directory (repeatable)")
        if name == "pairs":
            s.add_argument("--shard-size", type=int, default=25000)
            s.add_argument("--prompt-ids", help="only these prompts' Parallel cells")
    sp = sub.add_parser("sample-prompts")
    sp.add_argument("--release", required=True)
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n", type=int, default=1000)
    sp.add_argument("--split", default="held-out", choices=("held-out", "exploration", "any"))
    sp.add_argument("--order", default="random", choices=("random", "first"),
                    help="random (seeded) or the first N in the release's prompt order")
    sp.add_argument("--seed", type=int, default=ORDER_SEED)
    sc = sub.add_parser("sample-cells")
    sc.add_argument("--release", required=True)
    sc.add_argument("--bundle", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--model", default="llama4", choices=MODELS)
    sc.add_argument("--n-per-method", type=int, default=5000)
    sc.add_argument("--split", default="held-out", choices=("held-out", "exploration", "any"))
    sc.add_argument("--seed", type=int, default=ORDER_SEED)
    sn = sub.add_parser("snapshots")
    sn.add_argument("--release", required=True)
    sn.add_argument("--out", required=True)
    o = sub.add_parser("check-orig")
    o.add_argument("--release", required=True)
    o.add_argument("--bundle", required=True)
    o.add_argument("--cache", required=True)
    o.add_argument("--sample", type=int, default=2000)
    a = p.parse_args(argv)
    return {"cells": cmd_cells, "check-bm25": cmd_check_bm25, "pairs": cmd_pairs, "cache": cmd_cache,
            "check-orig": cmd_check_orig, "sample-prompts": cmd_sample_prompts, "snapshots": cmd_snapshots,
            "sample-cells": cmd_sample_cells}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
