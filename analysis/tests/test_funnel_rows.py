"""Funnel row bookkeeping: snapshot rows, the vectorised frozen search, and exact trace-to-row mapping."""

import asyncio
import copy
import json

import pytest

from analysis.interpretability.pipeline import agentic_search as search
from analysis.interpretability.pipeline import funnel_rows as fr
from analysis.scripts.run_agentic_search_integration_smoke import FrozenSnapshotSearchAdapter, TargetUrlConditionHook
from analysis.tests.test_intent_stages import ACTION, INFORMATION, KEYWORDS, OverlapScorer

ENGINE = "duckduckgo"


def fixture_rows():
    rows = []
    for k, keyword in enumerate(KEYWORDS):
        for i in range(12):
            words = ACTION if i % 2 else INFORMATION
            rows.append({"keyword": keyword, "position": i + 1, "url": f"https://site{k}-{i}.example/page",
                         "title": f"{keyword.title()} {words[0]} {i}", "snippet": f"{keyword} {' '.join(words)} item {i}"})
    # the same page (url, title, text) under a second keyword, and one url with a second text
    rows.append({**rows[1], "keyword": KEYWORDS[1], "position": 13})
    rows.append({**rows[2], "keyword": KEYWORDS[1], "position": 14, "snippet": rows[2]["snippet"] + " revised"})
    rows.append({"keyword": KEYWORDS[1], "position": 15, "url": "https://bad.example", "title": "t", "snippet": " "})  # unusable
    return rows


@pytest.fixture()
def world(tmp_path):
    path = tmp_path / f"{ENGINE}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in fixture_rows()))
    rows = fr.snapshot_rows({ENGINE: path})
    return {"path": path, "rows": rows, "lookup": rows.lookup(), "index": fr.LexicalIndex(rows, ENGINE),
            "adapter": FrozenSnapshotSearchAdapter(ENGINE, path)}


def run(method, adapter, condition="natural", hook=None, ranking=("S2", "S1")):
    words = ACTION
    if method == fr.PARALLEL:
        outputs = [{"queries": [f"{KEYWORDS[0]} {words[0]}", f"{KEYWORDS[0]} {words[1]}", f"{words[2]} {KEYWORDS[1]}"]},
                   {"ranking": list(ranking), "answer": "a"}]
        cls = search.ParallelExpansionV1
    else:
        outputs = [{"action": "search", "query": f"{KEYWORDS[0]} {words[0]}"}, {"action": "search", "query": f"{words[1]} {KEYWORDS[1]}"},
                   {"action": "finish", "ranking": list(ranking), "answer": "a"}]
        cls = search.ReactiveSnippetLoopV1
    agent = cls(llm=search.ScriptedLLM([json.dumps(o) for o in outputs]), search=adapter,
                compactor=search.ContextCompactor(OverlapScorer()), condition_hook=hook or search.IdentityConditionHook())
    result = asyncio.run(agent.run("where to buy tax filing", search.ExperimentalCondition(condition)))
    return result.trace.to_dict(), [s.url for s in result.final_snippets], list(result.ranking)


def test_snapshot_rows_follow_the_adapter_and_identify_rows(world):
    rows, adapter = world["rows"], world["adapter"]
    assert len(rows) == len(adapter.rows) == 26 and rows.exclusions[ENGINE] == {"invalid_snippet": 1}
    assert [(r["keyword"], r["position"], r["url"]) for r in adapter.rows] == list(zip(rows.keyword, rows.position.tolist(), rows.url))
    assert len(world["lookup"]) == 26 and len(fr.row_table_digest(rows)) == 64
    duplicated = fixture_rows()[1]
    assert rows.doc_id[1] == rows.doc_id[24]  # same page under two keywords, two rows
    assert rows.doc_id[2] != rows.doc_id[25] and rows.url[2] == rows.url[25]


@pytest.mark.parametrize("query", [KEYWORDS[0], KEYWORDS[1].upper(), "tax buy price", "deal school item 3", "nothing matches",
                                   "guide", "school scheduling buy", ""])
def test_lexical_index_reproduces_the_adapter_exactly(world, query):
    adapter, rows = world["adapter"], world["rows"]
    expected = [(r["keyword"], r["position"]) for r in adapter._select_rows(query, 20)]
    got = [(rows.keyword[i], int(rows.position[i])) for i in world["index"].select(query, 20)]
    assert got == expected


@pytest.mark.parametrize("method", [fr.PARALLEL, fr.REACTIVE])
@pytest.mark.parametrize("condition", ["natural", "shuffled", "ablated"])
def test_answer_items_map_every_stage_to_exact_rows(world, method, condition):
    rows = world["rows"]
    hook = TargetUrlConditionHook(target_url=rows.url[0], seed=7) if condition != "natural" else None
    trace, presented, ranking = run(method, world["adapter"], condition, hook)
    out = fr.answer_items(trace, method, ENGINE, presented, ranking, world["lookup"], rows)
    assert out["counts"].get("searches_from_raw_rows") == len(out["queries"])
    items = out["items"]
    by_row = {i[0]: i for i in items}
    # retrieved rows are exactly the recorded searches' rows
    searches = [e["payload"] for e in trace["events"] if e.get("event_type") == "search"]
    assert set(by_row) == {world["lookup"][(ENGINE, r["keyword"], r["position"])] for s in searches for r in s["raw_payload"]["rows"]}
    # presented and ranked map back to the same urls, in order
    shown = sorted((i for i in items if i[4] >= 0), key=lambda i: i[4])
    assert [rows.url[i[0]] for i in shown] == presented
    ranked = sorted((i for i in items if i[5] >= 0), key=lambda i: i[5])
    assert [rows.url[i[0]] for i in ranked] == ranking
    # nesting: ranked => presented => scored => retrieved
    assert all(i[3] == 1 for i in items if i[4] >= 0) and all(i[4] >= 0 for i in items if i[5] >= 0)
    # each event's kept candidates are its top-k by score
    for e in out["events"]:
        kept = sorted((c for c in e["candidates"] if c[2] >= 0), key=lambda c: c[2])
        best = sorted(e["candidates"], key=lambda c: -c[1])[:len(kept)]
        assert [c[0] for c in kept] == [c[0] for c in best]
    if condition == "ablated":
        assert all(rows.url[c[0]] != rows.url[0] for e in out["events"] for c in e["candidates"])
    if method == fr.REACTIVE:
        assert [e["search"] for e in out["events"]] == list(range(len(out["queries"])))
    else:
        assert [e["search"] for e in out["events"]] == [-1]


def test_missing_raw_rows_fall_back_to_an_exact_replay(world):
    trace, presented, ranking = run(fr.PARALLEL, world["adapter"])
    stripped = copy.deepcopy(trace)
    for e in stripped["events"]:
        if e.get("event_type") == "search":
            e["payload"]["raw_payload"] = {"snapshot": "sha256:x"}
    with pytest.raises(ValueError, match="no replay index"):
        fr.answer_items(stripped, fr.PARALLEL, ENGINE, presented, ranking, world["lookup"], world["rows"])
    out = fr.answer_items(stripped, fr.PARALLEL, ENGINE, presented, ranking, world["lookup"], world["rows"], world["index"])
    assert out["counts"]["searches_from_replay"] == 3
    reference = fr.answer_items(trace, fr.PARALLEL, ENGINE, presented, ranking, world["lookup"], world["rows"])
    assert out["items"] == reference["items"] and out["events"] == reference["events"]


def test_inconsistent_traces_are_refused(world):
    trace, presented, ranking = run(fr.REACTIVE, world["adapter"])
    with pytest.raises(ValueError, match="presented"):
        fr.answer_items(trace, fr.REACTIVE, ENGINE, presented + ["https://elsewhere.example/"], ranking, world["lookup"], world["rows"])
    bad = copy.deepcopy(trace)
    for e in bad["events"]:
        if e.get("event_type") == "compaction":
            e["payload"]["query"] = "something else"
            break
    with pytest.raises(ValueError, match="own search"):
        fr.answer_items(bad, fr.REACTIVE, ENGINE, presented, ranking, world["lookup"], world["rows"])
