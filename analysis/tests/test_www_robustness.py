"""WWW brief T1 GPU step (analysis/www_robustness/score_pairs.py) end to end on a tiny bundle with the fake models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis.www_robustness import score_pairs as sp

WORDS = ["best", "crm", "price", "buy", "now", "review", "free", "trial", "cloud", "payroll", "plan", "the"]


def write_bundle(root: Path, seed: int = 3) -> Path:
    rng = np.random.default_rng(seed)
    b = root / "bundle"
    b.mkdir()
    pages = []
    for engine in ("duckduckgo", "searxng"):
        for i in range(30):
            pages.append({"row_id": len(pages), "engine": engine, "keyword": "best crm", "position": int(rng.integers(1, 4)),
                          "url": f"https://s{int(rng.integers(0, 20))}.example/{engine}",
                          "title": " ".join(rng.choice(WORDS, 3)), "snippet": " ".join(rng.choice(WORDS, 8))})
    pages = pd.DataFrame(pages)
    prompts = pd.DataFrame({"prompt_id": [f"p{i}" for i in range(4)],
                            "text": [" ".join(rng.choice(WORDS, 6)) for _ in range(4)]})
    queries = pd.DataFrame({"query_id": np.arange(24, dtype=np.int32),
                            "text": [" ".join(rng.choice(WORDS, 3)) for _ in range(24)]})
    answers, aq = [], []
    for i, pid in enumerate(prompts["prompt_id"]):
        for j, engine in enumerate(("duckduckgo", "searxng")):
            aid = f"a{i}{j}"
            answers.append({"answer_id": aid, "prompt_id": pid, "engine": engine, "model": "qwen38"})
            aq.extend({"answer_id": aid, "search_index": s, "query_id": np.int32(3 * (2 * i + j) + s)} for s in range(3))
    logged = pd.DataFrame({"prompt_id": ["p0", "p0", "p1"], "row_id": [0, 1, 2]})
    to_score = pd.DataFrame({"prompt_id": ["p0", "p0", "p2", "p3"], "row_id": [0, 1, 5, 40],
                             "purpose": ["validation", "validation", "new", "new"]})
    tables = {"pages": pages, "prompts": prompts, "queries": queries, "answers": pd.DataFrame(answers),
              "answer_queries": pd.DataFrame(aq), "pairs_to_score": to_score, "pairs_logged": logged}
    files = {}
    for name, frame in tables.items():
        frame.to_parquet(b / f"{name}.parquet", index=False)
        files[f"{name}.parquet"] = {"sha256": sp.sha256_file(b / f"{name}.parquet"), "rows": len(frame)}
    (b / "manifest.json").write_text(json.dumps({"files": files}))
    return b


class ScorePairsTest(unittest.TestCase):
    def test_end_to_end_resume_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, run = write_bundle(root), root / "run"
            base = ["--bundle", str(bundle), "--run-dir", str(run), "--fake-models", "--workers", "2"]
            for w in ("0", "1"):
                self.assertEqual(sp.main(["embed", *base, "--worker", w]), 0)
            self.assertEqual(sp.main(["retrieve", *base, "--shard-size", "7"]), 0)
            # a deadline already past admits no shard
            self.assertEqual(sp.main(["score", *base, "--worker", "0", "--deadline-epoch", "1"]), 4)
            self.assertEqual(list((run / "results").glob("scores/*.parquet")), [])
            for w in ("0", "1"):
                self.assertEqual(sp.main(["score", *base, "--worker", w]), 0)
            shards = pd.read_parquet(run / "results/shards.parquet")
            scores = pd.concat(pd.read_parquet(p) for p in sorted((run / "results").glob("scores/shard-*.parquet")))
            self.assertEqual(len(scores), len(shards))
            self.assertTrue(scores["fake"].all())
            merged = shards.merge(scores, on=["prompt_id", "row_id", "purpose"])
            self.assertEqual(len(merged), len(shards))
            # the fake scorer is the pipeline's lexical overlap: recompute one value
            page = pd.read_parquet(bundle / "pages.parquet").set_index("row_id")
            prompt = dict(pd.read_parquet(bundle / "prompts.parquet").values)
            r = merged.iloc[-1]
            want = len(set(prompt[r.prompt_id].casefold().split())
                       & set(f"{page.title[r.row_id]} {page.snippet[r.row_id]}".casefold().split()))
            self.assertEqual(r.score, float(want))
            # dense pairs exclude logged and already planned pairs
            dense = shards[shards["purpose"] == "dense"]
            self.assertFalse(set(zip(dense.prompt_id, dense.row_id)) & {("p0", 0), ("p0", 1), ("p1", 2), ("p2", 5)})
            # items: each answer lists a row once; a URL repeated within an answer is scored once
            items = pd.read_parquet(run / "results/dense_items.parquet").merge(
                page[["url"]].reset_index(), on="row_id")
            for _, f in items.groupby("answer_id"):
                self.assertEqual(f["row_id"].nunique(), len(f))
                self.assertEqual(f.loc[f["scored"] == 1, "url"].nunique(), (f["scored"] == 1).sum())
                self.assertEqual(set(f["url"]), set(f.loc[f["scored"] == 1, "url"]))
            # a rerun changes nothing
            before = {p: p.stat().st_mtime_ns for p in (run / "results").rglob("*.parquet")}
            for w in ("0", "1"):
                self.assertEqual(sp.main(["score", *base, "--worker", w]), 0)
            self.assertEqual(sp.main(["retrieve", *base, "--shard-size", "7"]), 0)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in (run / "results").rglob("*.parquet")})

    def test_top_matches_naive_order_with_ties(self):
        rng = np.random.default_rng(11)
        for trial in range(200):
            n = int(rng.integers(5, 60))
            score = rng.integers(0, 4, n).astype(float)
            position = rng.integers(1, 4, n)
            url = [f"https://u{int(v)}.example" for v in rng.integers(0, 15, n)]
            query = f"q{trial}"

            def key(i):
                return (-score[i], position[i], hashlib.sha256(f"{query}\0{url[i]}".encode()).hexdigest(), i)

            self.assertEqual(sp.top(score, position, url, query), sorted(range(n), key=key)[:sp.K])

    def test_bundle_change_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = write_bundle(Path(tmp))
            pd.DataFrame({"prompt_id": ["p9"], "row_id": [1]}).to_parquet(bundle / "pairs_logged.parquet", index=False)
            with self.assertRaises(SystemExit) as caught:
                sp.load_bundle(bundle)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
