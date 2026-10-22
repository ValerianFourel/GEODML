"""WWW 2027 brief, T1 (retriever robustness): the GPU part, inside one four-A100 HoreKa allocation.

Input: a bundle written by the paper repository's www/analysis/t1_build.py (manifest.json with the sha256 of every table:
pages, prompts, queries, answers, answer_queries, pairs_to_score, pairs_logged). No generator is called.

  embed     --worker i --workers n   intfloat/e5-large-v2 embeddings (sentence-transformers, normalized) of a 1/n share
                                     of the texts: "passage: " + title + "\\n" + snippet for every snapshot row and
                                     "query: " + text for every logged query and held-out prompt
  retrieve                           per engine, cosine top 20 for every logged query and prompt, ties broken like the
                                     frozen search (stored position, sha256(query NUL url), snapshot index); per answer
                                     the three queries pooled in order, a row listed once, a repeated URL not scored;
                                     the pairs still to score and their shard plan
  score     --worker i --workers n   BAAI/bge-reranker-v2-m3 through the pipeline's own scorer
                                     (SentenceTransformersCrossEncoderScorer: batch 32, fp32) on (prompt text,
                                     title + "\\n" + snippet); a shard's pairs are batched by length across prompts
                                     (score_pairs; until 22 Oct one prompt at a time, which padded every batch to its
                                     longest snippet); shard j belongs to worker j mod n; no shard is started that
                                     could not finish before --deadline-epoch; --only-shards limits a side check
  status                             shards done and the measured pairs per second
  compare   --against RUN_DIR        the --only-shards scored under --run-dir against the same shards of RUN_DIR:
                                     every pair matched and max |difference| <= --tolerance, else exit 3

Outputs under --run-dir: results/ (embed/, dense_searches.parquet, dense_items.parquet, shards.parquet,
scores/shard-NNNNN.parquet with .json timing), diagnostics/ (one manifest per step and worker). Every file is written
atomically and never overwritten, so a rerun resumes. Exit codes: 0 done, 4 deadline checkpoint (rerun in a new
allocation), 2 input or environment error. --fake-models (tests only) swaps in hashing embeddings and the pipeline's
LexicalOverlapScorer and marks every output as fake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

K = 20
KINDS = ("row", "query", "prompt")
TABLES = ("pages", "prompts", "queries", "answers", "answer_queries", "pairs_to_score", "pairs_logged")
PURPOSE_ORDER = {"validation": 0, "new": 1, "dense": 2}
E5_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


# ---------------------------------------------------------------- io

def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, indent=1, sort_keys=True) + "\n").encode())


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle(bundle: Path) -> dict:
    """The bundle's tables, after checking every file against manifest.json."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    tables = {}
    for name in TABLES:
        path = bundle / f"{name}.parquet"
        want = manifest["files"].get(path.name, {}).get("sha256")
        if not path.exists() or want != sha256_file(path):
            _fail(f"bundle file missing or changed: {path}")
        tables[name] = pd.read_parquet(path)
    tables["manifest"] = manifest
    return tables


def _fail(message: str):
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(2)


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def environment(model: str | None, fake: bool) -> dict:
    info = {"git_commit": git_commit(), "host": socket.gethostname(), "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model": model, "fake_models": fake}
    if not fake:
        import sentence_transformers
        import torch
        import transformers
        info.update({"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    return info


# ---------------------------------------------------------------- texts and ranking

def texts(tables: dict) -> dict[str, tuple[list, list[str]]]:
    """Keys and model inputs per kind, in a fixed order."""
    pages = tables["pages"].sort_values("row_id")
    queries = tables["queries"].sort_values("query_id")
    prompts = tables["prompts"].sort_values("prompt_id")
    return {"row": (pages["row_id"].tolist(), [f"passage: {t}\n{s}" for t, s in zip(pages["title"], pages["snippet"])]),
            "query": (queries["query_id"].tolist(), [f"query: {t}" for t in queries["text"]]),
            "prompt": (prompts["prompt_id"].tolist(), [f"query: {t}" for t in prompts["text"]])}


def digest(query: str, url: str) -> str:
    return hashlib.sha256(f"{query}\0{url}".encode()).hexdigest()


def top(score: np.ndarray, position: np.ndarray, url: list[str], query: str, k: int = K) -> list[int]:
    """Indices of the k best rows by (-score, position, sha256(query NUL url), index): the frozen search's tie-break."""
    n = len(score)
    if n > k:
        threshold = np.partition(score, n - k)[n - k]
        above = np.flatnonzero(score > threshold)
        tie = np.flatnonzero(score == threshold)
        need = k - len(above)
        if len(tie) > need:
            pos = position[tie]
            cut = np.partition(pos, need - 1)[need - 1]
            sure, border = tie[pos < cut], tie[pos == cut]
            rest = need - len(sure)
            if len(border) > rest:
                border = np.asarray(sorted(border, key=lambda i: (digest(query, url[i]), i))[:rest], np.int64)
            tie = np.concatenate([sure, border])
        candidates = np.concatenate([above, tie])
    else:
        candidates = np.arange(n)
    return sorted((int(i) for i in candidates), key=lambda i: (-score[i], position[i], digest(query, url[i]), i))


def pool(lists: list[list[int]], row_id: np.ndarray, url: list[str]) -> list[tuple[int, int, int, int]]:
    """(row_id, search_index, rank_in_search, scored): a row listed once; a URL listed before is not scored."""
    seen_rows, seen_urls, items = set(), set(), []
    for s, lst in enumerate(lists):
        for r, i in enumerate(lst):
            rid = int(row_id[i])
            if rid in seen_rows:
                continue
            seen_rows.add(rid)
            items.append((rid, s, r, int(url[i] not in seen_urls)))
            seen_urls.add(url[i])
    return items


# ---------------------------------------------------------------- models

class HashingEmbedder:
    """Tests only: deterministic token-hashing vectors, L2-normalized."""

    def encode(self, values: list[str]) -> np.ndarray:
        out = np.zeros((len(values), 64), np.float32)
        for i, v in enumerate(values):
            for token in v.casefold().split():
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm > 0, norm, 1.0)


class E5Embedder:
    def __init__(self, path: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(path, device="cuda", local_files_only=True)
        self.model.max_seq_length = 512

    def encode(self, values: list[str]) -> np.ndarray:
        return self.model.encode(values, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)


def scorer(path: str | None, fake: bool):
    if fake:
        from analysis.interpretability.pipeline.agentic_search import LexicalOverlapScorer
        return LexicalOverlapScorer()
    from analysis.interpretability.pipeline.agentic_search import SentenceTransformersCrossEncoderScorer
    return SentenceTransformersCrossEncoderScorer(path, device="cuda", batch_size=32, local_files_only=True)


# ---------------------------------------------------------------- steps

def step_embed(a) -> int:
    tables = load_bundle(a.bundle)
    out = a.run_dir / "results" / "embed"
    todo = [kind for kind in KINDS if not (out / f"{kind}-{a.worker}.npy").exists()]
    if not todo:
        return 0
    t0 = time.time()
    model = HashingEmbedder() if a.fake_models else E5Embedder(a.model)
    by_kind = texts(tables)
    for kind in todo:
        values = by_kind[kind][1][a.worker::a.workers]
        atomic_npy(out / f"{kind}-{a.worker}.npy", model.encode(values) if values else np.zeros((0, 1), np.float32))
    atomic_json(a.run_dir / "diagnostics" / f"embed-{a.worker}.json",
                {**environment(a.model, a.fake_models), "kinds": todo, "seconds": round(time.time() - t0, 1),
                 "prefixes": {"row": "passage: ", "query": "query: ", "prompt": "query: "}, "revision": E5_REVISION})
    return 0


def merged(run_dir: Path, kind: str, n_items: int, workers: int) -> np.ndarray:
    parts = [np.load(run_dir / "results" / "embed" / f"{kind}-{w}.npy") for w in range(workers)]
    dim = max(p.shape[1] for p in parts if len(p))
    out = np.zeros((n_items, dim), np.float64)
    for w, p in enumerate(parts):
        out[w::workers] = p
    return out


def step_retrieve(a) -> int:
    results = a.run_dir / "results"
    if (results / "shards.parquet").exists():
        return 0
    t0 = time.time()
    tables = load_bundle(a.bundle)
    by_kind = texts(tables)
    emb = {kind: merged(a.run_dir, kind, len(by_kind[kind][0]), a.workers) for kind in KINDS}
    index = {kind: {key: i for i, key in enumerate(by_kind[kind][0])} for kind in KINDS}
    pages = tables["pages"].sort_values("row_id").reset_index(drop=True)
    answers = tables["answers"].sort_values("answer_id").reset_index(drop=True)
    aq = tables["answer_queries"].sort_values(["answer_id", "search_index"])
    per_answer = aq.groupby("answer_id")["query_id"].apply(list)
    qtext = dict(zip(tables["queries"]["query_id"], tables["queries"]["text"]))
    ptext = dict(zip(tables["prompts"]["prompt_id"], tables["prompts"]["text"]))
    searches, lists = [], {}
    for engine in sorted(answers["engine"].unique()):
        sub = pages[pages["engine"] == engine].reset_index(drop=True)
        rows = emb["row"][[index["row"][r] for r in sub["row_id"]]]
        position, url, rid = sub["position"].to_numpy(np.int64), sub["url"].tolist(), sub["row_id"].to_numpy(np.int64)
        mine = answers[answers["engine"] == engine]
        keys = sorted({("q", q) for qs in per_answer.loc[mine["answer_id"]] for q in qs}) + \
            sorted({("p", p) for p in mine["prompt_id"]})
        for start in range(0, len(keys), 2048):
            block = keys[start:start + 2048]
            vec = np.stack([emb["query"][index["query"][k]] if kind == "q" else emb["prompt"][index["prompt"][k]]
                            for kind, k in block])
            score = vec @ rows.T
            for r, (kind, k) in enumerate(block):
                text = qtext[k] if kind == "q" else ptext[k]
                best = top(score[r], position, url, text)
                lists[(kind, engine, k)] = best
                searches.extend((kind, engine, str(k), rank, int(rid[i])) for rank, i in enumerate(best))
    items = []
    by_engine = {e: pages[pages["engine"] == e].reset_index(drop=True) for e in answers["engine"].unique()}
    for aid, engine in zip(answers["answer_id"], answers["engine"]):
        sub = by_engine[engine]
        pooled = pool([lists[("q", engine, q)] for q in per_answer.loc[aid]], sub["row_id"].to_numpy(np.int64),
                      sub["url"].tolist())
        items.extend((aid, *it) for it in pooled)
    items = pd.DataFrame(items, columns=["answer_id", "row_id", "search_index", "rank_in_search", "scored"])
    prompt_of = answers.set_index("answer_id")["prompt_id"]
    scored = items[items["scored"] == 1]
    dense = pd.DataFrame({"prompt_id": prompt_of.loc[scored["answer_id"]].to_numpy(), "row_id": scored["row_id"]})
    dense = dense.drop_duplicates()
    planned = tables["pairs_to_score"]
    known = set(zip(tables["pairs_logged"]["prompt_id"], tables["pairs_logged"]["row_id"])) | \
        set(zip(planned["prompt_id"], planned["row_id"]))
    dense = dense[[p not in known for p in zip(dense["prompt_id"], dense["row_id"])]].assign(purpose="dense")
    pairs = pd.concat([planned[["prompt_id", "row_id", "purpose"]], dense], ignore_index=True)
    pairs["order"] = pairs["purpose"].map(PURPOSE_ORDER)
    pairs = pairs.sort_values(["order", "prompt_id", "row_id"]).drop(columns="order").reset_index(drop=True)
    pairs["shard"] = np.arange(len(pairs)) // a.shard_size
    atomic_parquet(results / "dense_searches.parquet",
                   pd.DataFrame(searches, columns=["kind", "engine", "key", "rank", "row_id"]))
    atomic_parquet(results / "dense_items.parquet", items)
    atomic_parquet(results / "shards.parquet", pairs)  # written last: marks the step done
    atomic_json(a.run_dir / "diagnostics" / "retrieve.json",
                {**environment(None, True), "fake_models": a.fake_models, "seconds": round(time.time() - t0, 1),
                 "searches": len(lists), "answers": int(len(answers)), "pairs": pairs["purpose"].value_counts().to_dict(),
                 "shards": int(pairs["shard"].max() + 1) if len(pairs) else 0, "shard_size": a.shard_size})
    return 0


def step_score(a) -> int:
    results = a.run_dir / "results"
    pairs = pd.read_parquet(results / "shards.parquet")
    tables = load_bundle(a.bundle)
    by_shard = {int(s): f for s, f in pairs.groupby("shard", sort=True)}
    only = {int(x) for x in a.only_shards.split(",")} if a.only_shards else None
    shards = [s for s in by_shard if s % a.workers == a.worker and (only is None or s in only)
              and not (results / "scores" / f"shard-{s:05d}.parquet").exists()]
    if not shards:
        return 0
    ptext = dict(zip(tables["prompts"]["prompt_id"], tables["prompts"]["text"]))
    page = tables["pages"].set_index("row_id")
    from analysis.interpretability.pipeline.agentic_search import Snippet
    t_load = time.time()
    model = scorer(a.model, a.fake_models)
    load_seconds = round(time.time() - t_load, 1)
    durations, status = [], 0
    for s in shards:
        expected = 1.25 * (np.mean(durations) if durations else a.first_shard_seconds)
        if a.deadline_epoch and time.time() + expected > a.deadline_epoch:
            status = 4
            break
        t0 = time.time()
        part = by_shard[s]
        rows = page.loc[part["row_id"].to_numpy()]
        pairs_in = [(ptext[p], Snippet.from_mapping({"url": u, "title": t, "text": x}))
                    for p, u, t, x in zip(part["prompt_id"], rows["url"], rows["title"], rows["snippet"])]
        if hasattr(model, "score_pairs"):  # the cross-encoder: the whole shard, batched by length
            values = model.score_pairs(pairs_in)
        else:  # the fake lexical scorer scores each pair on its own
            values = [v for q, snip in pairs_in for v in model.score(q, [snip])]
        out = part[["prompt_id", "row_id", "purpose"]].assign(score=np.asarray(values, np.float64),
                                                                fake=bool(a.fake_models))
        atomic_parquet(results / "scores" / f"shard-{s:05d}.parquet", out)
        durations.append(time.time() - t0)
        atomic_json(results / "scores" / f"shard-{s:05d}.json",
                    {"pairs": int(len(out)), "seconds": round(durations[-1], 3), "worker": a.worker})
    atomic_json(a.run_dir / "diagnostics" / f"score-{a.worker}-{os.environ.get('SLURM_JOB_ID', 'local')}.json",
                {**environment(a.model, a.fake_models), "revision": BGE_REVISION, "batch_size": 32,
                 "model_load_seconds": load_seconds, "shards_done": len(durations),
                 "shards_assigned": len(shards), "pairs_per_second": (
                     float(sum(len(by_shard[s]) for s in shards[:len(durations)]) / sum(durations))
                     if durations else None), "deadline_checkpoint": status == 4})
    return status


def step_status(a) -> int:
    results = a.run_dir / "results"
    if not (results / "shards.parquet").exists():
        print("retrieve not done")
        return 0
    pairs = pd.read_parquet(results / "shards.parquet", columns=["shard", "purpose"])
    done = sorted(results.glob("scores/shard-*.json"))
    meta = [json.loads(p.read_text()) for p in done]
    n, secs = sum(m["pairs"] for m in meta), sum(m["seconds"] for m in meta)
    print(json.dumps({"shards_total": int(pairs["shard"].nunique()), "shards_done": len(meta),
                      "pairs_total": int(len(pairs)), "pairs_done": n,
                      "pairs_by_purpose": pairs["purpose"].value_counts().to_dict(),
                      "pairs_per_second_per_worker": round(n / secs, 1) if secs else None}, indent=1))
    return 0


def step_compare(a) -> int:
    """The --only-shards rescored under --run-dir against the same shards of --against; exit 3 if any differs."""
    if not a.only_shards or a.against is None:
        _fail("compare needs --only-shards and --against")
    rows = []
    for s in sorted({int(x) for x in a.only_shards.split(",")}):
        name = f"scores/shard-{s:05d}"
        old, new = (pd.read_parquet(r / "results" / f"{name}.parquet") for r in (a.against, a.run_dir))
        m = old.merge(new, on=["prompt_id", "row_id", "purpose"], suffixes=("_old", "_new"))
        diff = float((m["score_old"] - m["score_new"]).abs().max()) if len(m) else float("inf")
        old_s, new_s = (json.loads((r / "results" / f"{name}.json").read_text())["seconds"] for r in (a.against, a.run_dir))
        rows.append({"shard": s, "pairs": len(old), "matched": len(m), "max_abs_diff": diff, "seconds_against": old_s,
                     "seconds": new_s, "passed": bool(len(m) == len(old) == len(new) and diff <= a.tolerance)})
    out = {"git_commit": git_commit(), "against": str(a.against), "tolerance": a.tolerance, "shards": rows,
           "passed": all(r["passed"] for r in rows)}
    atomic_json(a.run_dir / "results" / "compare.json", out)
    print(json.dumps(out, indent=1))
    return 0 if out["passed"] else 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["embed", "retrieve", "score", "status", "compare"])
    p.add_argument("--bundle", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", help="local snapshot directory of the step's model")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard-size", type=int, default=25000)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    p.add_argument("--first-shard-seconds", type=float, default=90.0)
    p.add_argument("--only-shards", default="", help="score: comma-separated shard numbers (a side check of shards "
                   "already scored, in a separate --run-dir holding a copy of results/shards.parquet)")
    p.add_argument("--against", type=Path, help="compare: the run directory that scored the same shards first")
    p.add_argument("--tolerance", type=float, default=1e-4, help="compare: largest allowed |score difference| "
                   "(gate G2's TOL_SCORE in the paper repository)")
    p.add_argument("--fake-models", action="store_true")
    a = p.parse_args(argv)
    if a.step not in ("status", "compare") and a.bundle is None:
        _fail("--bundle is required")
    if a.step in ("embed", "score") and not a.fake_models and not a.model:
        _fail("--model is required")
    return {"embed": step_embed, "retrieve": step_retrieve, "score": step_score, "status": step_status,
            "compare": step_compare}[a.step](a)


if __name__ == "__main__":
    raise SystemExit(main())
