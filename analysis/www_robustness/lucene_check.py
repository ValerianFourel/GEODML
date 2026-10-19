"""ARR author response: exact Lucene BM25 for a sample of searches, to validate the paper repository's bm25l (CPU).

bm25l (paper repository, www/analysis/t1_build.py) re-implements Pyserini's default BM25 (Lucene BM25Similarity,
k1 = 0.9, b = 0.4, DefaultEnglishAnalyzer, bag-of-words query) in Python. This step runs the real thing, Anserini's
fat jar (the Lucene toolkit Pyserini wraps; same analyzer and defaults), on one index per engine snapshot whose
documents are exactly bm25l's: id = row_id, contents = title + "\\n" + snippet.

  prepare --bundle B --run-dir R    docs/<engine>/docs.jsonl and topics/<engine>.tsv: a seeded sample of 1,000 logged
                                    queries and 500 held-out prompt texts per engine (whitespace collapsed, because the
                                    topic file is tab-separated; the analyzer treats any whitespace alike)
  collect --run-dir R               runs/<engine>.trec -> results/lucene/hits.parquet (engine, topic, rank, row_id,
                                    score) and sample.parquet (engine, topic, kind, key, text)

The Java steps (IndexCollection, SearchCollection -bm25 -bm25.k1 0.9 -bm25.b 0.4 -hits 100) run between the two, from
run_rebuttal.sbatch. The comparison with bm25l runs on the Mac (www/analysis/rebuttal_lucene_compare.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.www_robustness.score_pairs import atomic_bytes, atomic_json, atomic_parquet, git_commit, load_bundle  # noqa: E402

SEED = 20261019
N_QUERIES, N_PROMPTS = 1000, 500
WS = re.compile(r"\s+")


def one_line(text: str) -> str:
    return WS.sub(" ", str(text)).strip()


def prepare(bundle: Path, run: Path) -> dict:
    t = load_bundle(bundle)
    pages, answers, aq = t["pages"], t["answers"], t["answer_queries"]
    queries = t["queries"].set_index("query_id")["text"]
    prompts = t["prompts"].set_index("prompt_id")["text"]
    engine_of = dict(zip(answers["answer_id"].astype(str), answers["engine"].astype(str)))
    used = pd.DataFrame({"engine": aq["answer_id"].astype(str).map(engine_of), "query_id": aq["query_id"]})
    rng = np.random.default_rng(SEED)
    sample, counts = [], {}
    for e in sorted(pages["engine"].astype(str).unique()):
        rows = pages[pages["engine"].astype(str) == e].sort_values("row_id")
        docs = "".join(json.dumps({"id": str(int(r)), "contents": f"{ti}\n{sn}"}, ensure_ascii=False) + "\n"
                       for r, ti, sn in zip(rows["row_id"], rows["title"], rows["snippet"]))
        atomic_bytes(run / "docs" / e / "docs.jsonl", docs.encode())
        qids = np.sort(used.loc[used["engine"] == e, "query_id"].unique())
        pids = np.sort(answers.loc[answers["engine"].astype(str) == e, "prompt_id"].astype(str).unique())
        pick_q = rng.choice(qids, size=min(N_QUERIES, len(qids)), replace=False)
        pick_p = rng.choice(pids, size=min(N_PROMPTS, len(pids)), replace=False)
        topics = []
        for i, q in enumerate(pick_q):
            topics.append({"engine": e, "topic": f"q{i:05d}", "kind": "q", "key": str(q), "text": str(queries.loc[q])})
        for i, p in enumerate(pick_p):
            topics.append({"engine": e, "topic": f"p{i:05d}", "kind": "p", "key": str(p), "text": str(prompts.loc[p])})
        atomic_bytes(run / "topics" / f"{e}.tsv",
                     "".join(f"{x['topic']}\t{one_line(x['text'])}\n" for x in topics).encode())
        sample += topics
        counts[e] = {"rows": int(len(rows)), "queries": int(len(pick_q)), "prompts": int(len(pick_p))}
    atomic_parquet(run / "results" / "lucene" / "sample.parquet", pd.DataFrame(sample))
    info = {"step": "prepare", "seed": SEED, "counts": counts, "git_commit": git_commit(),
            "bundle_manifest": t["manifest"].get("created_by")}
    atomic_json(run / "diagnostics" / "lucene-prepare.json", info)
    return info


def collect(run: Path) -> dict:
    recs = []
    for f in sorted((run / "runs").glob("*.trec")):
        e = f.stem
        for line in f.read_text().splitlines():
            topic, _, docid, rank, score, _tag = line.split()
            recs.append((e, topic, int(rank), int(docid), float(score)))
    hits = pd.DataFrame(recs, columns=["engine", "topic", "rank", "row_id", "score"])
    atomic_parquet(run / "results" / "lucene" / "hits.parquet", hits)
    info = {"step": "collect", "hits": int(len(hits)), "topics": int(hits.groupby(["engine", "topic"]).ngroups),
            "engines": sorted(hits["engine"].unique().tolist())}
    atomic_json(run / "diagnostics" / "lucene-collect.json", info)
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    a1 = sub.add_parser("prepare")
    a1.add_argument("--bundle", type=Path, required=True)
    a1.add_argument("--run-dir", type=Path, required=True)
    a2 = sub.add_parser("collect")
    a2.add_argument("--run-dir", type=Path, required=True)
    a = p.parse_args(argv)
    info = prepare(a.bundle, a.run_dir) if a.step == "prepare" else collect(a.run_dir)
    print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
