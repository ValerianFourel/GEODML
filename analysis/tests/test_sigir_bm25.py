"""SIGIR BM25 rerun: BM25 port, search adapter, score cache, generation runner, HoreKa command composition."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd

from analysis.interpretability.pipeline.agentic_search import LexicalOverlapScorer, Snippet
from analysis.sigir_bm25 import generate, horeka_qwen
from analysis.sigir_bm25.bm25 import B, K1, SELECTION, Bm25Index, Bm25SnapshotSearchAdapter, lucene_norm_length, \
    lucene_tokens
from analysis.sigir_bm25.build_bundle import answer_id, cell_id
from analysis.sigir_bm25.score_cache import CachedScorer, pair_key

PARALLEL, REACTIVE = "Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"
TOPICS = {"crm software": ["crm pricing for small teams", "crm software reviews", "best crm tools"],
          "payroll service": ["payroll service cost", "payroll software comparison", "outsourced payroll"]}


def snapshot_rows():
    rows = []
    for keyword, words in TOPICS.items():
        for p in range(1, 13):
            rows.append({"keyword": keyword, "position": p, "url": f"https://{keyword.split()[0]}{p}.example.com/x",
                         "title": f"{keyword.title()} guide {p}", "snippet": f"{words[p % 3]} and {keyword} notes {p}"})
    return rows


def write_snapshot(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class AnalyzerTest(unittest.TestCase):
    def test_possessives_case_stopwords_and_dotted_tokens(self):
        self.assertEqual(lucene_tokens("The Company's apps’s Running U.S.A. and IT"), ["compani", "app", "run", "u.s.a"])

    def test_vendored_porter_equals_nltk(self):
        try:
            from nltk.stem.porter import PorterStemmer as Nltk
        except ImportError:
            self.skipTest("nltk not installed")
        from analysis.sigir_bm25.porter import PorterStemmer as Vendored
        words = "running generalization recovery caresses ponies agreed happy sky relational conditional hopeful " \
                "electrical adjustable software analyses business dying lying sized hoping".split()
        n, v = Nltk(mode=Nltk.MARTIN_EXTENSIONS), Vendored(mode=Vendored.MARTIN_EXTENSIONS)
        self.assertEqual([v.stem(w, to_lowercase=False) for w in words], [n.stem(w, to_lowercase=False) for w in words])

    def test_smallfloat_norms(self):
        self.assertEqual([lucene_norm_length(n) for n in range(25)], list(range(25)))
        for n in (25, 40, 77, 300, 1000, 12345):
            m = lucene_norm_length(n)
            self.assertLessEqual(m, n)
            self.assertEqual(lucene_norm_length(m), m)
        self.assertEqual(sorted(lucene_norm_length(n) for n in range(1, 2000)),
                         [lucene_norm_length(n) for n in range(1, 2000)])


class Bm25IndexTest(unittest.TestCase):
    def reference(self, rows, query, k):
        docs = [Counter(lucene_tokens(f"{r['title']}\n{r['snippet']}")) for r in rows]
        lengths = [sum(d.values()) for d in docs]
        n = sum(x > 0 for x in lengths)
        avgdl = sum(lengths) / n
        q = Counter(lucene_tokens(query))
        scores = []
        for d, length in zip(docs, lengths):
            s = 0.0
            for t, c in q.items():
                if t in d:
                    df = sum(t in e for e in docs)
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    s += c * idf * d[t] / (d[t] + K1 * (1 - B + B * lucene_norm_length(length) / avgdl))
            scores.append(s)
        digest = lambda i: hashlib.sha256(f"{query}\0{rows[i]['url']}".encode()).hexdigest()  # noqa: E731
        order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["position"], digest(i), i))[:k]
        return [(i, scores[i]) for i in order]

    def test_matches_naive_reference_with_ties_and_zero_filler(self):
        rows = snapshot_rows()
        index = Bm25Index.from_rows(rows)
        for query in ("crm pricing", "payroll payroll cost", "notes", "nothing matches here", "crm's teams"):
            got, want = index.top(query, 20), self.reference(rows, query, 20)
            self.assertEqual([i for i, _ in got], [i for i, _ in want], query)
            for (_, a), (_, b) in zip(got, want):
                self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(len(index.top("nothing matches here", 20)), 20)
        self.assertTrue(all(s == 0 for _, s in index.top("nothing matches here", 20)))

    def test_fingerprint_is_stable_and_content_bound(self):
        rows = snapshot_rows()
        self.assertEqual(Bm25Index.from_rows(rows).fingerprint, Bm25Index.from_rows(rows).fingerprint)
        rows[0] = {**rows[0], "snippet": rows[0]["snippet"] + " extra"}
        self.assertNotEqual(Bm25Index.from_rows(snapshot_rows()).fingerprint, Bm25Index.from_rows(rows).fingerprint)


class AdapterTest(unittest.TestCase):
    def test_search_payload_and_urls_for_query(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_snapshot(Path(d) / "ddg.jsonl", snapshot_rows())
            adapter = Bm25SnapshotSearchAdapter("duckduckgo", path)
            response = asyncio.run(adapter.search("payroll software comparison", 20))
            payload = response.raw_payload
            self.assertEqual(payload["selection"], SELECTION)
            self.assertEqual(payload["snapshot_sha256"], adapter.snapshot_sha256)
            self.assertEqual(len(payload["rows"]), 20)
            self.assertEqual(len(payload["scores"]), 20)
            self.assertEqual(payload["scores"], sorted(payload["scores"], reverse=True))
            self.assertEqual([s.url for s in response.snippets], [r["url"] for r in payload["rows"]])
            self.assertTrue(response.snippets[0].url.startswith("https://payroll"))
            self.assertEqual(list(adapter.urls_for_query("payroll software comparison")),
                             list(dict.fromkeys(r["url"] for r in payload["rows"])))
            with self.assertRaises(ValueError):
                Bm25SnapshotSearchAdapter("duckduckgo", path, expected_fingerprint="0" * 64)


class CachedScorerTest(unittest.TestCase):
    def test_hits_misses_and_key_fields(self):
        a = Snippet.from_mapping({"url": "https://a.example.com/", "title": "A", "text": "alpha"})
        b = Snippet.from_mapping({"url": "https://a.example.com/", "title": "A changed", "text": "alpha"})
        table = {pair_key("q", a.url, a.title, a.text): (0.75, "logged")}
        scorer = CachedScorer(LexicalOverlapScorer(), table)
        self.assertEqual(scorer.score("q", [a])[0], 0.75)
        self.assertNotEqual(scorer.score("q", [b])[0], 0.75)
        self.assertEqual(scorer.counts, Counter({"logged": 1, "live": 1}))
        with self.assertRaises(KeyError):
            CachedScorer(None, table, model_id="m").score("q", [b])


class DevicePoolTest(unittest.TestCase):
    class Replica:
        deterministic, model_id, model_revision = True, "/m/bge", None

        def __init__(self, fail=None):
            self.fail, self.calls = fail, 0

        def score(self, query, snippets):
            self.calls += 1
            if self.fail:
                raise self.fail
            return [float(len(s.text)) for s in snippets]

    def test_oom_falls_back_to_cpu_and_other_errors_propagate(self):
        from analysis.sigir_bm25.reranker import DevicePoolScorer
        rows = [Snippet.from_mapping({"url": "https://a.example.com/", "title": "A", "text": "abc"})]
        cpu = self.Replica()
        oom = DevicePoolScorer([self.Replica(RuntimeError("CUDA out of memory. Tried to allocate"))],
                               fallback_factory=lambda: cpu)
        self.assertEqual(oom.score("q", rows), [3.0])
        self.assertEqual(oom.counts, Counter({"cpu_after_oom": 1}))
        self.assertEqual(oom.model_id, "/m/bge")
        ok = DevicePoolScorer([self.Replica(), self.Replica()])
        self.assertEqual([ok.score("q", rows) for _ in range(3)], [[3.0]] * 3)
        self.assertEqual(sum(ok.counts.values()), 3)
        broken = DevicePoolScorer([self.Replica(ValueError("bad input"))], fallback_factory=lambda: cpu)
        with self.assertRaises(ValueError):
            broken.score("q", rows)
        self.assertEqual(broken.score.__self__._free.qsize(), 1)  # the replica is returned after an error


class FakeClient:
    """An OpenAI-compatible client double: answers by request purpose, records purposes and seeds."""

    def __init__(self, mode="ok"):
        self.mode, self.calls = mode, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def complete(self, *, prompt, schema_name, schema, temperature, max_tokens, seed):
        purpose = re.fullmatch(r"agentic_(.+)_\d+", schema_name).group(1)
        self.calls.append((purpose, seed))
        if self.mode == "transport":
            raise RuntimeError("HTTP 503: response body omitted")
        ids = re.findall(r'"evidence_id": ?"([^"]+)"', prompt)
        if purpose == "parallel_query_expansion":
            return json.dumps({"queries": ["generated one", "generated two", "generated three"]}), {}
        if purpose == "reactive_action" and "ITERATION: 1/" in prompt:
            return json.dumps({"action": "search", "query": "crm pricing for small teams"}), {}
        if self.mode == "invalid":
            return "not json", {}
        finish = {"ranking": ids[:2], "answer": "An answer."}
        return json.dumps({"action": "finish", **finish} if purpose.startswith("reactive") else finish), {}


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.snapshot = write_snapshot(root / "ddg.jsonl", snapshot_rows())
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        prompts = pd.DataFrame({"prompt_id": ["p1", "p2"],
                                "text": ["Which CRM software should a small team buy?", "How do payroll services work?"]})
        prompts["prompt_sha256"] = [hashlib.sha256(t.encode()).hexdigest() for t in prompts["text"]]
        prompts.to_parquet(self.bundle / "prompts.parquet", index=False)
        rows = []
        for p, sha in zip(prompts["prompt_id"], prompts["prompt_sha256"]):
            for method in (PARALLEL, REACTIVE):
                q = TOPICS["crm software" if p == "p1" else "payroll service"] if method == PARALLEL else None
                rows.append((cell_id(method, "duckduckgo", p, sha), answer_id("llama4", p, "duckduckgo", method),
                             "llama4", p, sha, method, "duckduckgo", q))
        cells = pd.DataFrame(rows, columns=["cell_id", "answer_id", "model", "prompt_id", "prompt_sha256", "method",
                                            "engine", "queries"]).assign(order=[2, 0, 3, 1])
        cells.to_parquet(self.bundle / "cells-llama4.parquet", index=False)
        (self.bundle / "manifest.json").write_text(json.dumps({"cells_sha256": {"llama4": "x"}}))
        self.cells = cells
        self.run_dir = root / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, client, *extra, env=None):
        argv = ["run", "--bundle", str(self.bundle), "--run-dir", str(self.run_dir), "--model", "llama4",
                "--retriever", "bm25", "--snapshot", f"duckduckgo={self.snapshot}", "--base-url",
                "http://127.0.0.1:8010/v1", "--served-model", "fake", "--final-max-tokens", "4096",
                "--cell-concurrency", "2", *extra]
        with mock.patch.object(generate, "make_client", lambda a: client), \
                mock.patch.dict(os.environ, env or {}, clear=False):
            return generate.main(argv)

    def records(self):
        out = []
        for part in sorted((self.run_dir / "parts").glob("part-*.jsonl")):
            out += [json.loads(x) for x in part.read_text().splitlines() if x.strip()]
        return out

    def test_parallel_replays_logged_queries_and_keeps_the_seed(self):
        client = FakeClient()
        self.assertEqual(self.run_cli(client), 0)
        recs = {r["cell_id"]: r for r in self.records()}
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(r["status"] == "complete" for r in recs.values()))
        self.assertNotIn("parallel_query_expansion", [p for p, _ in client.calls])
        for _, cell in self.cells.iterrows():
            rec = recs[cell["cell_id"]]
            searches = [e["payload"]["query"] for e in rec["trace"]["events"] if e["event_type"] == "search"]
            if cell["method"] == PARALLEL:
                self.assertEqual(searches, list(cell["queries"]))
                self.assertEqual(rec["diagnostics"]["queries_source"], "logged")
                final = [c for c in rec["diagnostics"]["llm_calls"] if c["purpose"] == "parallel_final"]
                self.assertEqual(final[0]["call_index"], 2)
            else:
                self.assertEqual(searches, ["crm pricing for small teams"])
                self.assertEqual(rec["diagnostics"]["queries_source"], "generated")
            self.assertEqual(rec["trace"]["events"][next(i for i, e in enumerate(rec["trace"]["events"])
                                                         if e["event_type"] == "search")]["payload"]["raw_payload"]
                             ["selection"], SELECTION)
        base = generate.SEED
        finals = {s for p, s in client.calls if p == "parallel_final"}
        want = {base + int(c[:8], 16) + 2 for c in self.cells.loc[self.cells.method == PARALLEL, "cell_id"]}
        self.assertEqual(finals, want)

    def test_resume_skips_finished_cells_and_ignores_a_torn_line(self):
        self.assertEqual(self.run_cli(FakeClient(), "--limit", "2"), 0)
        self.assertEqual(len(self.records()), 2)
        part = next((self.run_dir / "parts").glob("part-*.jsonl"))
        with part.open("a") as f:
            f.write('{"format_version": "sigir-bm25-generation-v1", "status": "comp')
        client = FakeClient()
        self.assertEqual(self.run_cli(client), 0)
        self.assertEqual(len(generate.finished(self.run_dir)), 4)
        self.assertLessEqual(len({p for p, _ in client.calls}), 3)

    def test_shards_are_disjoint_and_cover_every_cell(self):
        first, second = FakeClient(), FakeClient()
        self.assertEqual(self.run_cli(first, "--shard", "0", "--shards", "2"), 0)
        after_first = set(generate.finished(self.run_dir))
        self.assertEqual(after_first, set(self.cells.loc[self.cells["order"] % 2 == 0, "cell_id"]))
        self.assertEqual(self.run_cli(second, "--shard", "1", "--shards", "2"), 0)
        self.assertEqual(set(generate.finished(self.run_dir)), set(self.cells["cell_id"]))
        self.assertEqual(len(self.records()), 4)
        with self.assertRaises(SystemExit):
            self.run_cli(FakeClient(), "--shard", "2", "--shards", "2")

    def test_prompt_ids_limit_the_run_to_those_prompts(self):
        ids = Path(self.tmp.name) / "ids.txt"
        ids.write_text("p2\n")
        self.assertEqual(self.run_cli(FakeClient(), "--prompt-ids", str(ids)), 0)
        self.assertEqual({r["prompt_id"] for r in self.records()}, {"p2"})
        self.assertEqual(len(self.records()), 2)
        self.assertEqual(self.run_cli(FakeClient()), 0)  # the same run-dir continues with every prompt
        self.assertEqual(len(generate.finished(self.run_dir)), 4)

    def test_stop_file_checkpoints_without_writing(self):
        stop = Path(self.tmp.name) / "STOP"
        stop.write_text("")
        self.assertEqual(self.run_cli(FakeClient(), env={"GEODML_ADMISSION_STOP_FILE": str(stop)}), 4)
        self.assertEqual(self.records(), [])

    def test_invalid_output_is_recorded_once_after_the_retry(self):
        self.assertEqual(self.run_cli(FakeClient("invalid")), 0)
        recs = self.records()
        self.assertEqual(len(recs), 4)
        self.assertEqual(Counter(r["status"] for r in recs), Counter({"failed": 4}))

    def test_transport_failures_are_not_results_and_trip_the_breaker(self):
        self.assertEqual(self.run_cli(FakeClient("transport"), "--breaker", "2", "--cell-concurrency", "1"), 3)
        self.assertEqual(self.records(), [])

    def test_resume_with_other_settings_is_refused(self):
        self.assertEqual(self.run_cli(FakeClient(), "--limit", "1"), 0)
        argv_extra = ("--query-max-tokens", "128")
        self.assertEqual(self.run_cli(FakeClient(), *argv_extra), 2)

    def test_external_api_needs_https_and_a_key(self):
        a = generate.main.__globals__["argparse"].Namespace(
            api_key_env="NOPE_KEY", base_url="http://api.example.com/v1", served_model="m", timeout=10,
            disable_thinking=False)
        with self.assertRaises(SystemExit):
            generate.make_client(a)
        a.base_url, a.api_key_env = "https://api.example.com/v1", None
        with self.assertRaises(SystemExit):
            generate.make_client(a)


class HorekaCommandTest(unittest.TestCase):
    def test_phases_and_generate_command(self):
        self.assertEqual(horeka_qwen.parse_phase("4:12"), (4, 12, None))
        self.assertEqual(horeka_qwen.parse_phase("32:64:900"), (32, 64, 900))
        with self.assertRaises(Exception):
            horeka_qwen.parse_phase("4")
        a = horeka_qwen.main.__globals__["argparse"].Namespace(
            bundle=Path("/b"), run_dir=Path("/r"), retriever="bm25", score_cache=[Path("/c.parquet")], limit=0,
            shard=3, shards=8, device="cuda-pool")
        config = {"inputs": {"SEARCH_AGENTIC_DDG_SNAPSHOT": "/s/ddg.parquet",
                             "SEARCH_AGENTIC_SEARXNG_SNAPSHOT": "/s/sx.parquet"}, "cross_encoder": "/m/bge"}
        profile = {"serving": {"public_base_url": "http://127.0.0.1:8010/v1"}, "model": {"model_id": "Qwen/X"}}
        cmd = horeka_qwen.generate_command("/py", Path("/repo"), config, profile, a, 32, 64)
        self.assertIn("--disable-thinking", cmd)
        for flag, value in (("--final-max-tokens", "2048"), ("--query-max-tokens", "256"), ("--model", "qwen38"),
                            ("--served-model", "Qwen/X"), ("--request-concurrency", "32"), ("--device", "cuda-pool"),
                            ("--score-cache", "/c.parquet"), ("--shard", "3"), ("--shards", "8")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        script = horeka_qwen.phase_script([(["echo", "a b"], 900), (["echo", "c"], None)])
        self.assertIn("GEODML_ROLE_END_TIME=$(( $(date +%s) + 900 )) echo 'a b'; status=$?", script)
        self.assertTrue(script.endswith("exit $status"))


if __name__ == "__main__":
    unittest.main()
