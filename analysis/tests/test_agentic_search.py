"""Deterministic contract tests for bounded agentic search methods."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from analysis.interpretability.pipeline.agentic_search import (
    AgentExecutionError,
    ContextCompactor,
    ExperimentalCondition,
    FINAL_ANSWER_MAX_CHARACTERS,
    IdentityConditionHook,
    MemoizingSnippetScorer,
    ParallelExpansionV1,
    ReactiveSnippetLoopV1,
    ScriptedLLM,
    Snippet,
    StaticSearchAdapter,
    compact_snippets,
    write_trace_atomic,
)


class PositionScorer:
    model_id = "test-position-scorer"
    model_revision = "test-v1"
    deterministic = True

    def score(self, query: str, snippets: list[Snippet]) -> list[float]:
        del query
        return [float(snippet.text.removeprefix("score=")) for snippet in snippets]


class CountingPositionScorer(PositionScorer):
    def __init__(self):
        self.calls: list[list[Snippet]] = []

    def score(self, query: str, snippets: list[Snippet]) -> list[float]:
        self.calls.append(list(snippets))
        return super().score(query, snippets)


class BarrierSearchAdapter(StaticSearchAdapter):
    """Fail by timeout unless all three searches enter concurrently."""

    def __init__(self, results):
        super().__init__("searxng", results)
        self._entered = 0
        self._barrier = asyncio.Event()

    async def search(self, query: str, limit: int):
        self._entered += 1
        if self._entered == 3:
            self._barrier.set()
        await asyncio.wait_for(self._barrier.wait(), timeout=0.25)
        return await super().search(query, limit)


class MutatingConditionHook:
    def apply(self, condition, snippets):
        del condition
        return [
            Snippet(url=row.url, title=row.title, text="mutated evidence")
            for row in snippets
        ]


class UndeclaredScorer:
    model_id = "test-undeclared-scorer"
    model_revision = "v1"

    def score(self, query: str, snippets: list[Snippet]) -> list[float]:
        del query
        return [0.0 for _ in snippets]


def snippet(index: int, score: int | None = None) -> dict[str, str]:
    value = index if score is None else score
    return {
        "url": f"https://example.test/{index}",
        "title": f"Result {index}",
        "text": f"score={value}",
    }


class AgenticSearchTests(unittest.TestCase):
    def test_memoizing_scorer_requires_deterministic_contract(self):
        with self.assertRaisesRegex(ValueError, "deterministic=True"):
            MemoizingSnippetScorer(UndeclaredScorer())

    def test_memoizing_scorer_only_scores_exact_cache_misses(self):
        underlying = CountingPositionScorer()
        scorer = MemoizingSnippetScorer(underlying)
        rows = [
            Snippet.from_mapping(snippet(1, 5)),
            Snippet.from_mapping(snippet(2, 9)),
            Snippet.from_mapping(snippet(3, 3)),
        ]

        self.assertEqual(scorer.score("query", rows), [5.0, 9.0, 3.0])
        self.assertEqual(scorer.score("query", [rows[2], rows[0]]), [3.0, 5.0])
        changed = Snippet(rows[1].url, rows[1].title, "score=8")
        self.assertEqual(scorer.score("query", [rows[1], changed]), [9.0, 8.0])

        self.assertEqual([len(call) for call in underlying.calls], [3, 1])
        self.assertEqual(scorer.stats.hits, 3)
        self.assertEqual(scorer.stats.misses, 4)
        self.assertEqual(scorer.stats.entries, 4)

    def test_schema_retry_configuration_cannot_exceed_two(self):
        dependencies = {
            "llm": ScriptedLLM([]),
            "search": StaticSearchAdapter("searxng", {}),
            "compactor": ContextCompactor(PositionScorer()),
            "condition_hook": IdentityConditionHook(),
        }
        with self.assertRaisesRegex(ValueError, "at most 2"):
            ParallelExpansionV1(**dependencies, schema_retries=3)

    def test_context_compaction_is_score_sorted_and_stable(self):
        compactor = ContextCompactor(PositionScorer())
        rows = [
            Snippet.from_mapping(snippet(1, 5)),
            Snippet.from_mapping(snippet(2, 9)),
            Snippet.from_mapping(snippet(3, 9)),
            Snippet.from_mapping(snippet(4, 1)),
        ]

        selected = compactor.compact("query", rows, top_k=3)
        self.assertEqual([row.url for row in selected], [
            "https://example.test/2",
            "https://example.test/3",
            "https://example.test/1",
        ])
        scored = compactor.score_and_compact("query", rows, top_k=2)
        self.assertEqual([row.score for row in scored.selected], [9.0, 9.0])
        self.assertEqual(len(scored.scored), 4)
        with self.assertRaisesRegex(ValueError, "top_k"):
            compactor.compact("query", rows, top_k=0)

    def test_public_compaction_utility_accepts_and_returns_mappings(self):
        selected = compact_snippets(
            "query",
            [snippet(1, 2), snippet(2, 8), snippet(3, 5)],
            top_k=2,
            compactor=ContextCompactor(PositionScorer()),
        )
        self.assertEqual(selected, [snippet(2, 8), snippet(3, 5)])

    def test_parallel_expansion_fetches_concurrently_and_deduplicates_in_query_order(self):
        results = {
            "alpha": [snippet(i) for i in range(1, 6)],
            "beta": [snippet(i) for i in range(4, 9)],
            "gamma": [snippet(i) for i in range(8, 11)],
        }
        llm = ScriptedLLM([
            json.dumps({"queries": ["alpha", "beta", "gamma"]}),
            json.dumps({
                "ranking": ["https://example.test/10", "https://example.test/9"],
                "answer": "The compacted evidence supports the result.",
            }),
        ])
        search = BarrierSearchAdapter(results)
        condition = IdentityConditionHook()
        method = ParallelExpansionV1(
            llm=llm,
            search=search,
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=condition,
        )

        result = asyncio.run(method.run(
            "Find the strongest result.", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.method_id, "Parallel-Expansion-v1")
        self.assertEqual(result.ranking, (
            "https://example.test/10", "https://example.test/9"
        ))
        self.assertCountEqual(
            search.calls,
            [("alpha", 20), ("beta", 20), ("gamma", 20)],
        )
        self.assertEqual(len(condition.calls), 1)
        self.assertEqual(len(condition.calls[0][1]), 10)
        self.assertEqual([row.url for row in result.final_snippets], [
            f"https://example.test/{i}" for i in range(10, 3, -1)
        ])
        event_types = [event.event_type for event in result.trace.events]
        self.assertEqual(event_types.count("llm_call"), 2)
        self.assertEqual(event_types.count("search"), 3)
        self.assertEqual(event_types.count("deduplication"), 1)
        self.assertEqual(event_types.count("condition"), 1)
        self.assertEqual(event_types.count("compaction"), 1)
        search_events = [event for event in result.trace.events if event.event_type == "search"]
        self.assertEqual([event.payload["query"] for event in search_events], [
            "alpha", "beta", "gamma"
        ])
        self.assertIn("raw_payload", search_events[0].payload)

    def test_condition_hook_cannot_mutate_search_evidence(self):
        llm = ScriptedLLM([
            json.dumps({"queries": ["one", "two", "three"]}),
        ])
        method = ParallelExpansionV1(
            llm=llm,
            search=StaticSearchAdapter("searxng", {
                "one": [snippet(1)],
                "two": [],
                "three": [],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=MutatingConditionHook(),
        )

        with self.assertRaisesRegex(AgentExecutionError, "new evidence"):
            asyncio.run(method.run("Question", ExperimentalCondition.ABLATED))

    def test_parallel_expansion_retries_invalid_query_schema_twice(self):
        llm = ScriptedLLM([
            "not json",
            json.dumps({"queries": ["same", "same", "third"]}),
            json.dumps({"queries": ["one", "two", "three"]}),
            json.dumps({"ranking": [], "answer": "No retained evidence."}),
        ])
        method = ParallelExpansionV1(
            llm=llm,
            search=StaticSearchAdapter("duckduckgo", {
                "one": [], "two": [], "three": [],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.SHUFFLED
        ))

        self.assertEqual(len(llm.requests), 4)
        query_events = [
            event for event in result.trace.events
            if event.event_type == "llm_call"
            and event.payload["purpose"] == "parallel_query_expansion"
        ]
        self.assertEqual(len(query_events), 3)
        self.assertTrue(query_events[0].payload["validation_error"])
        self.assertTrue(query_events[1].payload["validation_error"])
        self.assertIsNone(query_events[2].payload["validation_error"])

    def test_parallel_final_ranks_short_evidence_ids_and_returns_urls(self):
        llm = ScriptedLLM([
            json.dumps({"queries": ["one", "two", "three"]}),
            json.dumps({"ranking": ["S2", "S1"], "answer": "Bounded answer."}),
        ])
        method = ParallelExpansionV1(
            llm=llm,
            search=StaticSearchAdapter("duckduckgo", {
                "one": [snippet(1), snippet(2)],
                "two": [],
                "three": [],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.ranking, (
            "https://example.test/1", "https://example.test/2"
        ))
        request = llm.requests[-1]
        ranking = request.response_schema["properties"]["ranking"]
        self.assertEqual(ranking, {
            "type": "array", "items": {"type": "string"}
        })
        self.assertIn('"evidence_id": "S1"', request.prompt)

    def test_final_schema_avoids_xgrammar_fragile_semantic_keywords(self):
        llm = ScriptedLLM([
            json.dumps({"queries": ["one", "two", "three"]}),
            json.dumps({"ranking": ["S1"], "answer": "Answer."}),
        ])
        method = ParallelExpansionV1(
            llm=llm,
            search=StaticSearchAdapter("duckduckgo", {
                "one": [snippet(1)],
                "two": [],
                "three": [],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        asyncio.run(method.run("Question", ExperimentalCondition.NATURAL))

        schema = llm.requests[-1].response_schema
        encoded = json.dumps(schema, sort_keys=True)
        self.assertNotIn('"enum"', encoded)
        self.assertNotIn('"uniqueItems"', encoded)
        self.assertNotIn('"maxLength"', encoded)

    def test_reactive_forced_finish_ranks_unique_evidence_ids(self):
        llm = ScriptedLLM([
            json.dumps({"action": "search", "query": "first"}),
            json.dumps({"action": "search", "query": "second"}),
            json.dumps({"action": "search", "query": "third"}),
            json.dumps({
                "action": "finish",
                "ranking": ["S2", "S1"],
                "answer": "Bounded answer.",
            }),
        ])
        repeated = [snippet(1), snippet(2), snippet(1)]
        method = ReactiveSnippetLoopV1(
            llm=llm,
            search=StaticSearchAdapter("searxng", {
                "first": repeated,
                "second": repeated,
                "third": repeated,
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.ranking, (
            "https://example.test/1", "https://example.test/2"
        ))
        request = llm.requests[-1]
        ranking = request.response_schema["properties"]["ranking"]
        self.assertEqual(ranking, {
            "type": "array", "items": {"type": "string"}
        })
        self.assertIn('"evidence_id": "S1"', request.prompt)

    def test_final_attempt_repairs_ranking_and_bounds_answer(self):
        invalid = json.dumps({
            "ranking": ["S2", "https://outside.test/", "S2", "S1"],
            "answer": "x" * 1300,
        })
        llm = ScriptedLLM([
            json.dumps({"queries": ["one", "two", "three"]}),
            invalid,
            invalid,
            invalid,
        ])
        method = ParallelExpansionV1(
            llm=llm,
            search=StaticSearchAdapter("duckduckgo", {
                "one": [snippet(1), snippet(2)],
                "two": [],
                "three": [],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.ranking, (
            "https://example.test/1", "https://example.test/2"
        ))
        self.assertEqual(result.answer, "x" * 1200)
        repairs = [
            event for event in result.trace.events
            if event.event_type == "controller_repair"
        ]
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0].payload["dropped_ranking_references"], [
            "https://outside.test/", "S2"
        ])
        self.assertTrue(repairs[0].payload["answer_truncated"])

    def test_final_attempt_recovers_answer_prefix_from_malformed_json(self):
        malformed_outputs = (
            '{"ranking":["S2","S1"],"answer":"Useful answer without a terminator',
            '{"ranking":["S2","S1"],"answer":"Useful answer before a "   trailing text',
        )
        for malformed in malformed_outputs:
            with self.subTest(malformed=malformed):
                llm = ScriptedLLM([
                    json.dumps({"queries": ["one", "two", "three"]}),
                    malformed,
                    malformed,
                    malformed,
                ])
                method = ParallelExpansionV1(
                    llm=llm,
                    search=StaticSearchAdapter("duckduckgo", {
                        "one": [snippet(1), snippet(2)],
                        "two": [],
                        "three": [],
                    }),
                    compactor=ContextCompactor(PositionScorer()),
                    condition_hook=IdentityConditionHook(),
                )

                result = asyncio.run(method.run(
                    "Question", ExperimentalCondition.NATURAL
                ))

                self.assertEqual(result.ranking, (
                    "https://example.test/1", "https://example.test/2"
                ))
                self.assertTrue(result.answer.startswith("Useful answer"))
                repairs = [
                    event for event in result.trace.events
                    if event.event_type == "controller_repair"
                ]
                self.assertEqual(len(repairs), 1)
                self.assertTrue(repairs[0].payload["malformed_json_recovered"])
                self.assertIn("JSONDecodeError", repairs[0].payload[
                    "trigger_validation_error"
                ])

    def test_reactive_final_attempt_repairs_forced_finish(self):
        invalid_finish = json.dumps({
            "action": "finish",
            "ranking": ["S1", "https://outside.test/"],
            "answer": "Bounded answer.",
        })
        llm = ScriptedLLM([
            json.dumps({"action": "search", "query": "first"}),
            json.dumps({"action": "search", "query": "second"}),
            json.dumps({"action": "search", "query": "third"}),
            invalid_finish,
            invalid_finish,
            invalid_finish,
        ])
        method = ReactiveSnippetLoopV1(
            llm=llm,
            search=StaticSearchAdapter("searxng", {
                "first": [snippet(1), snippet(2)],
                "second": [snippet(1), snippet(2)],
                "third": [snippet(1), snippet(2)],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.ranking, ("https://example.test/2",))
        repairs = [
            event for event in result.trace.events
            if event.event_type == "controller_repair"
        ]
        self.assertEqual(len(repairs), 1)
        self.assertTrue(repairs[0].payload["forced_finish"])
        self.assertEqual(
            repairs[0].payload["dropped_ranking_references"],
            ["https://outside.test/"],
        )

    def test_reactive_recovers_malformed_finish_on_last_attempt(self):
        answers = (
            "Useful answer.", 'Quoted "evidence" and café. ',
            "Astral \U0001f600 evidence.", "Note. " * 2200,
        )
        for forced in (False, True):
            for answer in answers:
                for closed_quote in (False, True):
                    with self.subTest(
                        forced=forced, answer=answer[:30], closed=closed_quote,
                    ):
                        raw = (
                            '{"action":"finish","ranking":["S1","outside","S1"],'
                            '"answer":' + json.dumps(answer, ensure_ascii=True)
                        )
                        if not closed_quote:
                            raw = raw[:-1]
                        search_count = 3 if forced else 1
                        llm = ScriptedLLM([
                            json.dumps({"action": "search", "query": "first"})
                        ] * search_count + [raw] * 3)
                        method = ReactiveSnippetLoopV1(
                            llm=llm,
                            search=StaticSearchAdapter(
                                "searxng", {"first": [snippet(1)]},
                            ),
                            compactor=ContextCompactor(PositionScorer()),
                            condition_hook=IdentityConditionHook(),
                        )

                        result = asyncio.run(method.run(
                            "Question", ExperimentalCondition.NATURAL,
                        ))

                        self.assertEqual(
                            result.answer, answer[:FINAL_ANSWER_MAX_CHARACTERS],
                        )
                        self.assertEqual(result.ranking, ("https://example.test/1",))
                        self.assertEqual(len(llm.requests), search_count + 3)
                        calls = [
                            e for e in result.trace.events if e.event_type == "llm_call"
                        ]
                        self.assertEqual(
                            [e.payload["attempt"] for e in calls[-3:]], [1, 2, 3],
                        )
                        self.assertTrue(all(
                            e.payload["raw_output"] == raw for e in calls[-3:]
                        ))
                        repairs = [
                            e.payload for e in result.trace.events
                            if e.event_type == "controller_repair"
                        ]
                        self.assertEqual(len(repairs), 1)
                        repair = repairs[0]
                        self.assertEqual(
                            repair["purpose"],
                            "reactive_forced_finish" if forced else "reactive_action",
                        )
                        self.assertTrue(repair["malformed_json_recovered"])
                        self.assertEqual(repair["forced_finish"], forced)
                        self.assertEqual(
                            repair["recovered_answer_quote_observed"], closed_quote,
                        )
                        self.assertEqual(
                            repair["recovered_answer_prefix_characters"], len(answer),
                        )
                        self.assertEqual(
                            repair["answer_truncated"],
                            len(answer) > FINAL_ANSWER_MAX_CHARACTERS,
                        )
                        self.assertEqual(
                            repair["dropped_ranking_references"], ["outside", "S1"],
                        )
                        self.assertIn(
                            "JSONDecodeError", repair["trigger_validation_error"],
                        )
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "trace.json"
                            digest = write_trace_atomic(path, result.trace)
                            saved = json.loads(path.read_text(encoding="utf-8"))
                            self.assertEqual(saved["trace_sha256"], digest)

    def test_reactive_rejects_ambiguous_malformed_actions(self):
        prefix = '{"action":"finish","ranking":["S1"],"answer":'
        malformed_outputs = (
            '{"action":"search","query":"unfinished',
            '{"action":"search","ranking":["S1"],"answer":"text',
            '{"ranking":["S1"],"answer":"text',
            '{"action":"finish","ranking":["S1",',
            '{"action":"finish","ranking":"S1","answer":"text',
            '{"action":"finish","ranking":[1],"answer":"text',
            '{"action":"finish","ranking":["S1"]',
            '{"action":"finish","action":"search","ranking":["S1"],"answer":"text',
            '{"action":"finish","ranking":[],"ranking":["S1"],"answer":"text',
            prefix + '"',
            prefix + '"   ',
            prefix + 'null',
            prefix + '"text","action":"search"',
            prefix + '"text"} {"action":"search"}',
            prefix + '"text","answer":"other"',
            prefix + '"text" unexpected content',
            prefix + '"invalid\\q',
            prefix + '"invalid\\ud800',
            prefix + '"invalid\\ud800"',
        )
        for forced in (False, True):
            for raw in malformed_outputs:
                with self.subTest(forced=forced, raw=raw):
                    search_count = 3 if forced else 1
                    llm = ScriptedLLM([
                        json.dumps({"action": "search", "query": "first"})
                    ] * search_count + [raw] * 3)
                    method = ReactiveSnippetLoopV1(
                        llm=llm,
                        search=StaticSearchAdapter(
                            "searxng", {"first": [snippet(1)]},
                        ),
                        compactor=ContextCompactor(PositionScorer()),
                        condition_hook=IdentityConditionHook(),
                    )

                    with self.assertRaises(AgentExecutionError) as raised:
                        asyncio.run(method.run(
                            "Question", ExperimentalCondition.NATURAL,
                        ))

                    events = raised.exception.trace.events
                    self.assertFalse(any(
                        e.event_type == "controller_repair" for e in events
                    ))
                    rejected = [
                        e for e in events if e.event_type == "controller_repair_rejected"
                    ]
                    self.assertEqual(len(rejected), 1)
                    self.assertEqual(len(llm.requests), search_count + 3)

    def test_schema_retry_exhaustion_exposes_complete_trace(self):
        llm = ScriptedLLM(["bad", "still bad", "also bad"])
        method = ParallelExpansionV1(
            llm=llm,
            search=StaticSearchAdapter("searxng", {}),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        with self.assertRaises(AgentExecutionError) as raised:
            asyncio.run(method.run("Question", ExperimentalCondition.ABLATED))

        self.assertEqual(len(llm.requests), 3)
        self.assertEqual(len(raised.exception.trace.events), 3)
        self.assertIn("schema validation failed", str(raised.exception))

    def test_reactive_loop_forces_fourth_finish_call_after_three_searches(self):
        llm = ScriptedLLM([
            json.dumps({"action": "search", "query": "first"}),
            json.dumps({"action": "search", "query": "second"}),
            json.dumps({"action": "search", "query": "third"}),
            json.dumps({
                "action": "finish",
                "ranking": ["https://example.test/9"],
                "answer": "Finished after the bounded loop.",
            }),
        ])
        search = StaticSearchAdapter("duckduckgo", {
            "first": [snippet(1), snippet(2), snippet(3)],
            "second": [snippet(4), snippet(5), snippet(6)],
            "third": [snippet(7), snippet(8), snippet(9)],
        })
        condition = IdentityConditionHook()
        method = ReactiveSnippetLoopV1(
            llm=llm,
            search=search,
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=condition,
        )

        result = asyncio.run(method.run(
            "Investigate this.", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.method_id, "Reactive-Snippet-Loop-v1")
        self.assertEqual(len(llm.requests), 4)
        self.assertEqual(search.calls, [("first", 20), ("second", 20), ("third", 20)])
        self.assertEqual(len(condition.calls), 3)
        self.assertEqual(len(result.final_snippets), 9)
        self.assertTrue(llm.requests[-1].force_finish)
        self.assertEqual(
            [event.event_type for event in result.trace.events].count("observation"),
            3,
        )

    def test_reactive_loop_requires_search_before_finish(self):
        llm = ScriptedLLM([json.dumps({
            "action": "finish", "ranking": [], "answer": "No search needed."
        })])
        search = StaticSearchAdapter("searxng", {})
        method = ReactiveSnippetLoopV1(
            llm=llm,
            search=search,
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        with self.assertRaises(AgentExecutionError):
            asyncio.run(method.run(
                "Answer directly.", ExperimentalCondition.NATURAL
            ))
        self.assertEqual(search.calls, [])
        self.assertEqual(llm.requests[0].response_schema["required"], ["action", "query"])

    def test_reactive_finish_rejects_unobserved_urls_and_retries(self):
        llm = ScriptedLLM([
            json.dumps({"action": "search", "query": "known"}),
            json.dumps({
                "action": "finish",
                "ranking": ["https://unknown.test/result"],
                "answer": "Invalid reference.",
            }),
            json.dumps({
                "action": "finish",
                "ranking": ["https://example.test/3"],
                "answer": "Known reference.",
            }),
        ])
        method = ReactiveSnippetLoopV1(
            llm=llm,
            search=StaticSearchAdapter("searxng", {
                "known": [snippet(1), snippet(2), snippet(3)],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )

        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.NATURAL
        ))

        self.assertEqual(result.ranking, ("https://example.test/3",))
        self.assertEqual(len(llm.requests), 3)

    def test_trace_is_written_atomically_with_a_stable_hash(self):
        llm = ScriptedLLM([
            json.dumps({"action": "search", "query": "known"}),
            json.dumps({
                "action": "finish",
                "ranking": ["https://example.test/1"],
                "answer": "Done.",
            }),
        ])
        method = ReactiveSnippetLoopV1(
            llm=llm,
            search=StaticSearchAdapter("searxng", {
                "known": [snippet(1)],
            }),
            compactor=ContextCompactor(PositionScorer()),
            condition_hook=IdentityConditionHook(),
        )
        result = asyncio.run(method.run(
            "Question", ExperimentalCondition.NATURAL
        ))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            digest = write_trace_atomic(path, result.trace)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["trace_sha256"], digest)
            self.assertEqual(payload["method_id"], "Reactive-Snippet-Loop-v1")
            with self.assertRaises(FileExistsError):
                write_trace_atomic(path, result.trace)


if __name__ == "__main__":
    unittest.main()
