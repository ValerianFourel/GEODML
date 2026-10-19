"""ARR rebuttal Lucene check (analysis/www_robustness/lucene_check.py): documents, topics and the run-file parser."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from analysis.tests.test_www_robustness import write_bundle
from analysis.www_robustness import lucene_check as lc


class LuceneCheckTest(unittest.TestCase):
    def test_prepare_and_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, run = write_bundle(root), root / "run"
            info = lc.prepare(bundle, run)
            pages = pd.read_parquet(bundle / "pages.parquet")
            for e in ("duckduckgo", "searxng"):
                docs = [json.loads(x) for x in (run / "docs" / e / "docs.jsonl").read_text().splitlines()]
                want = pages[pages["engine"] == e].sort_values("row_id")
                self.assertEqual([d["id"] for d in docs], [str(r) for r in want["row_id"]])
                self.assertEqual(docs[0]["contents"], f"{want['title'].iloc[0]}\n{want['snippet'].iloc[0]}")
                topics = (run / "topics" / f"{e}.tsv").read_text().splitlines()
                self.assertTrue(all(len(t.split("\t")) == 2 for t in topics))
                self.assertEqual(info["counts"][e]["queries"] + info["counts"][e]["prompts"], len(topics))
            sample = pd.read_parquet(run / "results" / "lucene" / "sample.parquet")
            self.assertEqual(set(sample["kind"]), {"q", "p"})
            # the same seed gives the same sample
            lc.prepare(bundle, root / "run2")
            pd.testing.assert_frame_equal(sample, pd.read_parquet(root / "run2" / "results" / "lucene" / "sample.parquet"))
            (run / "runs").mkdir()
            (run / "runs" / "duckduckgo.trec").write_text("q00000 Q0 3 1 2.500000 Anserini\nq00000 Q0 7 2 1.25 Anserini\n")
            out = lc.collect(run)
            hits = pd.read_parquet(run / "results" / "lucene" / "hits.parquet")
            self.assertEqual(out["hits"], 2)
            self.assertEqual(hits["row_id"].tolist(), [3, 7])
            self.assertEqual(hits["score"].tolist(), [2.5, 1.25])

    def test_one_line(self):
        self.assertEqual(lc.one_line("a\tb\n c  "), "a b c")


if __name__ == "__main__":
    unittest.main()
