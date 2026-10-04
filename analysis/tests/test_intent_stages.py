"""Intent surfacing study: trace stages from real agentic traces, the chain math in planted worlds,
the shortlisting model, the reranker score drivers, the replay fidelity gate, and the end-to-end run."""
import asyncio
from dataclasses import asdict
import hashlib
import json
import shutil

import numpy as np
import pytest

from analysis.interpretability.pipeline import agentic_search as search
from analysis.interpretability.pipeline import geo_drivers as geo
from analysis.interpretability.pipeline import intent_stages as stages
from analysis.interpretability.pipeline import page_readiness_ordering as ordering
from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter, initialize_dataset
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts import intent_stages_study as study
from analysis.scripts.run_agentic_search_integration_smoke import FrozenSnapshotSearchAdapter
from analysis.tests.test_page_readiness_ordering import battery, projections

KEYWORDS = ("tax filing", "school scheduling")
ACTION, INFORMATION = ("buy", "price", "deal"), ("guide", "explained", "history")
ENGINES = ("duckduckgo", "searxng")


# ---------------------------------------------------------------- planted worlds (math only)

def csr(lists):
    return (np.r_[0, np.cumsum([len(items) for items in lists])].astype(np.int64),
            np.asarray([d for items in lists for d in items], np.int64))


def world(kind, *, keywords=30, prompts=8, docs=400, seed=0):
    """Answers whose pool follows the prompt (kind 'pool', ranking a random order of the pool) or whose
    ordering does (kind 'ranking', pool unrelated to the prompt, ranking = closest on the axis first)."""
    rng = np.random.default_rng(seed)
    doc_z = rng.normal(size=docs)
    doc_u = (np.argsort(np.argsort(doc_z)) + 0.5) / docs
    keyword, prompt, x, lists = [], [], [], {name: [] for name in ("ret", "cand", "pool", "rank")}
    for k in range(keywords):
        for p in range(prompts):
            position = rng.uniform()
            if kind == "pool":
                weight = np.exp(3.0 * doc_z * (position - 0.5))
                pool = rng.choice(docs, 8, replace=False, p=weight / weight.sum())
                ranked = rng.permutation(pool)[:5]
            else:
                pool = rng.choice(docs, 8, replace=False)
                ranked = pool[np.argsort(np.abs(doc_u[pool] - position), kind="stable")][:5]
            retrieved = list(dict.fromkeys([*pool, *rng.choice(docs, 12, replace=False)]))
            for name, items in (("ret", retrieved), ("cand", retrieved[:16]), ("pool", pool), ("rank", ranked)):
                lists[name].append(list(items))
            keyword.append(k), prompt.append(k * prompts + p), x.append(position)
    n = len(x)
    st = stages.Stages(np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64),
                       np.asarray(keyword), np.asarray(prompt), np.asarray(x), *csr(lists["ret"]), *csr(lists["cand"]),
                       *csr(lists["pool"]), *csr(lists["rank"]), np.zeros(n + 1, np.int64), np.zeros(0, np.int64))
    return st, doc_z, doc_u


def analysed(kind, seed=0):
    st, doc_z, doc_u = world(kind, seed=seed)
    values = stages.stage_values(st, doc_z)
    oracles = stages.oracle_values(st, doc_z, doc_u, np.zeros(len(st.pool_docs)))
    draws = stages.keyword_draws(int(st.keyword.max()) + 1, 40, seed + 1)
    shuffles = stages.shuffle_draws(*stages.prompt_table(st), 39, seed + 2)
    return stages.analyse_stratum(st, np.ones(len(st.x), bool), values, oracles, draws=draws, shuffles=shuffles)


def test_chain_increments_add_up_to_the_ranking_slope_exactly():
    result = analysed("pool")
    d = {k: v["estimate"] for k, v in result["derived"].items()}
    increments = sum(d[f"increment:{s}"] for s in ("retrieval", "deduplication_and_condition", "reranker", "reordering"))
    assert increments == pytest.approx(result["slopes"]["K"]["slope"], abs=1e-12)
    shares = sum(d[f"share_of_ranking:{s}"] for s in ("retrieval", "deduplication_and_condition", "reranker", "reordering"))
    assert shares == pytest.approx(1.0, abs=1e-12)
    assert result["slopes"]["K-P"]["slope"] == pytest.approx(d["increment:reordering"], abs=1e-12)


def test_pool_driven_world_puts_the_intent_shift_in_the_pool():
    result = analysed("pool")
    slopes, d = result["slopes"], result["derived"]
    assert slopes["P"]["slope"] > 0.5 and slopes["P"]["permutation_p"] < 0.05
    assert d["share_of_ranking:pool"]["estimate"] == pytest.approx(1.0, abs=0.25)
    assert abs(d["increment:reordering"]["estimate"]) < 0.1 * slopes["P"]["slope"]  # random order adds ~nothing
    assert abs(d["utilization"]["estimate"]) < 0.3
    assert d["pool_minus_reordering"]["ci95"][0] > 0
    assert d["mediated_by_pool"]["estimate"] == pytest.approx(1.0, abs=0.25)


def test_ranking_driven_world_puts_it_in_the_ordering_and_uses_the_whole_oracle():
    result = analysed("ranking")
    slopes, d = result["slopes"], result["derived"]
    assert d["increment:reordering"]["estimate"] > 0.5 and d["increment:reordering"]["ci95"][0] > 0
    assert abs(slopes["P"]["slope"]) < 0.15 * d["increment:reordering"]["estimate"]  # pools ignore the prompt
    assert d["utilization"]["estimate"] == pytest.approx(1.0, abs=1e-9)  # the model behaves as the intent oracle
    assert d["pool_minus_reordering"]["ci95"][1] < 0


def test_mediation_separates_the_direct_path_from_the_pool_path():
    rng = np.random.default_rng(6)
    n = 4000
    keyword = rng.integers(0, 40, n)
    x = rng.uniform(size=n)
    pool = 0.5 * x + rng.normal(scale=0.3, size=n) + 0.1 * keyword
    ranking = 0.3 * x + 0.7 * pool + rng.normal(scale=0.1, size=n) - 0.2 * keyword
    direct, via = stages.mediation(ranking, pool, x, keyword)
    assert direct == pytest.approx(0.3, abs=0.03) and via == pytest.approx(0.7, abs=0.03)


def test_top_weighted_orders_within_answer_and_keeps_the_ranking_length():
    offsets, values = np.array([0, 3, 5, 5]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = 1 / np.log2(np.arange(2, 5))
    plain = stages.top_weighted(offsets, values)
    assert plain[0] == pytest.approx(np.dot(w, [1, 2, 3]) / w.sum())
    assert plain[1] == pytest.approx(np.dot(w[:2], [4, 5]) / w[:2].sum()) and np.isnan(plain[2])
    ordered = stages.top_weighted(offsets, values, sort_key=-values, length=np.array([2, 1, 0]))
    assert ordered[0] == pytest.approx(np.dot(w[:2], [3, 2]) / w[:2].sum()) and ordered[1] == 5.0
    means = stages.csr_mean(offsets, np.array([1.0, np.nan, 3.0, 4.0, 6.0]))
    assert np.isnan(means[0]) and means[1] == 5.0 and np.isnan(means[2])


def test_dispersion_splits_page_intent_variance_exactly():
    rng = np.random.default_rng(8)
    keyword_effect, pools, z, keyword = rng.normal(size=50), [], [], []
    for k in range(50):
        for _ in range(20):
            pool_effect = rng.normal(scale=0.5)
            z += list(keyword_effect[k] + pool_effect + rng.normal(scale=0.5, size=8))
            pools.append(list(range(len(z) - 8, len(z))))
            keyword.append(k)
    n = len(pools)
    offsets, docs = csr(pools)
    empty = np.zeros(n + 1, np.int64)
    st = stages.Stages(*(np.zeros(n, np.int64),) * 4, np.asarray(keyword), np.arange(n), np.zeros(n),
                       empty, docs[:0], empty, docs[:0], offsets, docs, empty, docs[:0], empty, docs[:0])
    shares = stages.dispersion(st, np.asarray(z), np.ones(n, bool))["shares"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
    expected = np.array([0.25, 0.25, 1.0]) / 1.5  # within pool, between pools, between keywords
    assert np.allclose([shares["within_pool"], shares["between_pools_same_keyword"], shares["between_keywords"]],
                       expected, atol=0.08)


def test_contrast_recovers_a_slope_difference_under_shared_resamples():
    rng = np.random.default_rng(9)
    n = 3000
    keyword, x, model = rng.integers(0, 60, n), rng.uniform(size=n), rng.integers(0, 2, n)
    y = np.where(model == 0, 0.3, 0.1) * x + 0.05 * keyword + rng.normal(scale=0.05, size=n)
    st = stages.Stages(model, *(np.zeros(n, np.int64),) * 3, keyword, np.arange(n), x, *(np.zeros(1, np.int64),) * 10)
    draws = stages.keyword_draws(60, 50, 1)
    result = stages.contrast(st, model == 0, model == 1, {"P": y}, ["P"], draws)["P"]
    assert result["difference"] == pytest.approx(0.2, abs=0.02)
    assert result["ci95"][0] < 0.2 < result["ci95"][1]


def test_shuffles_permute_positions_within_each_keyword_only_and_repeat_with_the_seed():
    x, keyword = np.arange(12, dtype=float), np.repeat([0, 1, 2], 4)
    draws = stages.shuffle_draws(x, keyword, 20, seed=3)
    for draw in draws:
        for k in range(3):
            assert sorted(draw[keyword == k]) == sorted(x[keyword == k])
    assert any(not np.array_equal(draw, x) for draw in draws)
    assert all(np.array_equal(a, b) for a, b in zip(draws, stages.shuffle_draws(x, keyword, 20, seed=3)))


# ---------------------------------------------------------------- reranker

def test_selection_rows_expand_each_pick_over_the_candidates_not_yet_picked():
    event = np.array([0, 0, 0, 0, 1, 1, 1])
    selected = np.array([1, -1, 0, -1, -1, 0, -1])  # event 0 keeps candidate 2 then 0; event 1 keeps 5
    values = np.arange(7, dtype=float)
    rows = stages.selection_rows(event, selected, np.array([5, 5, 5, 5, 6, 6, 6]), z=values, u=values,
                                 on_keyword=values, topic=values)
    assert rows.sets == 3 and rows.levels == 1 and not rows.position.any()
    assert rows.set.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert rows.z.tolist() == [0, 1, 2, 3, 0, 1, 3, 4, 5, 6]
    assert rows.chosen.tolist() == [False, False, True, False, True, False, False, False, True, False]
    assert rows.generation.tolist() == [5] * 7 + [6] * 3


def test_shortlisting_model_recovers_planted_preferences():
    rng = np.random.default_rng(4)
    theta, events, n, k = np.array([0.8, 1.5, 0.0, -0.6]), 800, 15, 4
    event = np.repeat(np.arange(events), n)
    features = rng.normal(size=(len(event), 4))
    utility = features @ theta + rng.gumbel(size=len(event))  # top-k of Gumbel utilities = top-k Plackett–Luce
    order = np.lexsort((-utility, event))
    rank = np.empty(len(event), np.int64)
    rank[order] = np.arange(len(event)) - np.repeat(np.arange(events) * n, n)
    rows = stages.selection_rows(event, np.where(rank < k, rank, -1), event, z=features[:, 2], u=features[:, 3],
                                 on_keyword=features[:, 0], topic=features[:, 1])
    fit = ordering.fit_choice_model(np.column_stack([rows.on_keyword, rows.topic, rows.z, rows.u]), rows)
    assert np.allclose(fit.x, theta, atol=0.12)


def test_score_drivers_recover_within_event_effects_and_reweight_keywords_exactly():
    rng = np.random.default_rng(5)
    events, n = 400, 12
    event = np.repeat(np.arange(events), n)
    keyword = event // 8
    drivers = {name: rng.normal(size=len(event)) for name in geo.DRIVERS}
    score = (2.0 * rng.normal(size=events)[event] + 0.5 * drivers["topic_similarity"] - 0.2 * drivers["page_intent"]
             + 0.05 * rng.normal(size=len(event)))
    result = stages.score_drivers(event, score, drivers, keyword, [])
    assert result["drivers"]["topic_similarity"]["score_per_sd"] == pytest.approx(0.5, abs=0.01)
    assert result["drivers"]["page_intent"]["score_per_sd"] == pytest.approx(-0.2, abs=0.01)
    assert result["drivers"]["on_keyword"]["score_per_sd"] == pytest.approx(0.0, abs=0.01)
    draw = stages.keyword_draws(50, 1, 7)[0]
    replicate = stages.score_drivers(event, score, drivers, keyword, [draw])
    centered = lambda v: v - (np.bincount(event, v) / np.bincount(event))[event]
    root = np.sqrt(draw[keyword])
    direct = np.linalg.lstsq(np.column_stack([centered(drivers[d]) for d in geo.DRIVERS]) * root[:, None],
                             centered(score) * root, rcond=None)[0]
    assert [replicate["drivers"][d]["ci95"][0] for d in geo.DRIVERS] == pytest.approx(list(direct), abs=1e-10)


# ---------------------------------------------------------------- real traces

class OverlapScorer:
    model_id, model_revision, deterministic = "overlap-scorer", None, True

    def score(self, query, snippets):
        words = set(query.casefold().split())
        return [len(words & set(f"{s.title} {s.text}".casefold().split()))
                + int(hashlib.sha256(s.url.encode()).hexdigest()[:6], 16) / 1e9 for s in snippets]


def snapshot_rows():
    rows = []
    for k, keyword in enumerate(KEYWORDS):
        for i in range(12):
            words = ACTION if i % 2 else INFORMATION
            rows.append({"keyword": keyword, "position": i + 1, "url": f"https://site{k}-{i}.example/page",
                         "title": f"{keyword.title()} {words[0]} {i}", "snippet": f"{keyword} {' '.join(words)} item {i}"})
    return rows


def run_agent(method, adapter, prompt, condition, keyword, action):
    words = ACTION if action else INFORMATION
    if method == stages.PARALLEL:
        outputs = [{"queries": [f"{keyword} {words[0]}", f"{keyword} {words[1]}", f"{words[2]} {keyword} options"]},
                   {"ranking": ["S2", "S1", "S3"], "answer": f"An answer about {keyword}."}]
        agent_class = search.ParallelExpansionV1
    else:
        outputs = [{"action": "search", "query": f"{keyword} {words[0]}"}, {"action": "search", "query": f"{words[1]} {keyword}"},
                   {"action": "finish", "ranking": ["S1", "S3"], "answer": f"An answer about {keyword}."}]
        agent_class = search.ReactiveSnippetLoopV1
    agent = agent_class(llm=search.ScriptedLLM([json.dumps(o) for o in outputs]), search=adapter,
                        compactor=search.ContextCompactor(OverlapScorer()), condition_hook=search.IdentityConditionHook())
    return asyncio.run(agent.run(prompt, search.ExperimentalCondition(condition)))


def build_inputs(tmp_path):
    """A generator dataset of real Parallel and Reactive traces over a frozen snapshot, plus the corpus,
    archived prompt vectors, maps, battery, prompt axis and population files the study reads."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(11)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    adapters = {}
    for engine in ENGINES:
        study.readiness.write_jsonl(snapshots / f"{engine}.jsonl", snapshot_rows())
        adapters[engine] = FrozenSnapshotSearchAdapter(engine, snapshots / f"{engine}.jsonl")
    prompt_ids = [f"q{i:03d}" for i in range(6)]
    keyword_of = {p: KEYWORDS[i % 2] for i, p in enumerate(prompt_ids)}
    action_of = {p: i >= 3 for i, p in enumerate(prompt_ids)}
    text_of = {p: (f"where to buy {keyword_of[p]} at the best price" if action_of[p] else f"how does {keyword_of[p]} work")
               for p in prompt_ids}
    root = tmp_path / "dataset"
    initialize_dataset(root, population_id="p", acceptance_policy_id="a")
    writer = FinalDatasetWriter(root, writer_id="fixture")
    ledger = StripedTaskLedger(root / "control/task-ledger", stripe_count=256)
    for p in prompt_ids:
        writer.append("prompts", {"prompt_id": p, "prompt_text": text_of[p]}, transaction_id=f"prompt-{p}")
        kid = f"kw{KEYWORDS.index(keyword_of[p])}"
        writer.append("keyword_memberships", {"prompt_id": p, "keyword_ids": [kid], "primary_keyword_id": kid,
                                              "primary_priority_rank": 1}, transaction_id=f"member-{p}")
    results, pending = {}, []
    for model in ("llama4", "qwen38"):
        for engine in ENGINES:
            for method in (stages.PARALLEL, stages.REACTIVE):
                for condition in ("natural", "ablated"):
                    for p in prompt_ids:
                        key = (model, engine, method, condition, p)
                        result = results[key] = run_agent(method, adapters[engine], text_of[p], condition,
                                                          keyword_of[p], action_of[p])
                        task_id = "-".join((model, engine, method[:8], condition, p))
                        identity = ClaimIdentity(task_id=task_id, model_id=model, model_revision="a" * 40, protocol="p",
                                                 request_sha256=hashlib.sha256(task_id.encode()).hexdigest())
                        writer.append("task_definitions", {"task_id": task_id, "prompt_id": p, "model": model,
                                                           "stage": "generation", "method": method, "engine": engine,
                                                           "condition": condition, "claim_identity": asdict(identity)},
                                      transaction_id="d" + task_id)
                        claim = ledger.claim(identity, owner_id="fixture").claim
                        t = writer.append("traces", result.trace.to_dict(), transaction_id=task_id, record_id="t-" + task_id)
                        g = writer.append("generations", {"cell_id": task_id, "prompt_id": p, "method": method,
                                                          "engine": engine, "condition": condition, "answer": result.answer,
                                                          "ranking": list(result.ranking)},
                                          transaction_id=task_id, record_id="g-" + task_id)
                        pending.append((claim, [t, g]))
    writer.seal()
    for claim, refs in pending:
        ledger.transition(claim, state="completed", record_references=refs)

    pages = {}
    for row in snapshot_rows():
        text = ordering.page_text(row["title"], row["snippet"])
        pages.setdefault(ordering.page_id(text), {"keyword": row["keyword"], "action": row["title"].split()[2] in ACTION})
    ids = sorted(pages)
    z = np.asarray([(1.0 if pages[i]["action"] else -1.0) + 0.3 * rng.normal() for i in ids])
    corpus = tmp_path / "corpus"
    (corpus / "embeddings").mkdir(parents=True)
    pq.write_table(pa.table({
        "row": list(range(len(ids))), "snippet_id": ids, "keywords": [[pages[i]["keyword"]] for i in ids],
        "consensus_axis_1_z": list(z), "prompt_scale_percentile_0_1": list((np.argsort(np.argsort(z)) + 0.5) / len(z)),
        "qwen_axis_1_z": list(z + 0.1 * rng.normal(size=len(z))), "mistral_aligned_axis_1_z": list(z + 0.1 * rng.normal(size=len(z))),
        "outside_prompt_range": [False] * len(ids)}), corpus / "snippets.parquet")
    study.readiness.write_json(corpus / "manifest.json", {"rows": len(ids)})
    dim = 6
    for name in study.readiness.EMBEDDING_FILES.values():
        np.save(corpus / name, rng.normal(size=(len(ids), dim)).astype(np.float32))
    maps = {}
    for view in ("qwen", "mistral"):
        shard = tmp_path / f"prompts-{view}/shard-000"
        shard.mkdir(parents=True)
        np.savez(shard / "question_embeddings.restricted-local.npz", candidate_ids=np.asarray(prompt_ids),
                 embeddings=rng.normal(size=(len(prompt_ids), dim)).astype(np.float32))
        maps[view] = tmp_path / f"map-{view}"
        maps[view].mkdir()
        study.readiness.write_json(maps[view] / "readiness_embedding_map.json", {
            "embedding_mean": [0.0] * dim, "supervised_subspace_axes": [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]})
    final = tmp_path / "final-axis-map.jsonl"
    study.readiness.write_jsonl(final, [{"candidate_id": p, "axis_1_percentile_0_1": (i + 0.5) / 6,
                                         "consensus_axis_1_z": float(i) - 2.5} for i, p in enumerate(prompt_ids)])
    population = tmp_path / "population-prompts.jsonl"
    study.readiness.write_jsonl(population, [{"candidate_id": p, "keyword": keyword_of[p].upper(), "question": text_of[p]}
                                             for p in prompt_ids])
    return {"root": root, "results": results, "adapters": adapters, "corpus": corpus, "maps": maps, "final": final,
            "population": population, "battery": battery(tmp_path / "battery"), "text_of": text_of,
            "keyword_of": keyword_of, "doc_index": {i: n for n, i in enumerate(ids)}, "rng": rng}


def extract(tmp_path, inputs, monkeypatch, workers="1", name="extract"):
    monkeypatch.setattr(study, "FIDELITY_PREFIX", "")  # replay every recorded search in the fixture
    output = tmp_path / name
    assert study.main(["trace-extract", "--source", f"{inputs['root']}:llama4", "--source", f"{inputs['root']}:qwen38",
                       "--final-axis-map", str(inputs["final"]), "--corpus-package", str(inputs["corpus"]),
                       "--population-prompts", str(inputs["population"]), "--workers", workers, "--output", str(output)]) == 0
    return output


def test_trace_extract_reads_every_stage_of_real_traces(tmp_path, monkeypatch):
    inputs = build_inputs(tmp_path)
    output = extract(tmp_path, inputs, monkeypatch)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["consistency_passed"] and manifest["counts"]["answers"] == 96
    assert set(manifest["consistency_failures"]) == {"selection_is_top_k_by_score", "shortlist_equals_presented",
                                                     "candidates_within_retrieved", "one_compaction_if_parallel"}
    assert set(manifest["snapshots"]) == set(ENGINES)
    st, ev, codes = study.load_stages(output)
    queries = [r["text"] for r in study.readiness.read_jsonl(output / "queries.jsonl.gz")]
    doc_index = inputs["doc_index"]
    row_of = lambda s: doc_index[ordering.page_id(ordering.page_text(s["title"], s["text"]))]
    for method, events_expected, kept in ((stages.PARALLEL, 1, 7), (stages.REACTIVE, 2, 3)):
        key = ("qwen38", "searxng", method, "natural", "q004")
        i = int(np.flatnonzero((st.model == codes["model"].index("qwen38")) & (st.engine == codes["engine"].index("searxng"))
                               & (st.method == codes["method"].index(method)) & (st.condition == codes["condition"].index("natural"))
                               & (st.prompt == codes["prompt"].index("q004")))[0])
        trace = inputs["results"][key].trace.to_dict()
        searches = [e["payload"] for e in trace["events"] if e["event_type"] == "search"]
        assert [queries[q] for q in st.q_ids[st.q_offsets[i]:st.q_offsets[i + 1]]] == [s["query"] for s in searches]
        retrieved = list(dict.fromkeys(row_of(s) for payload in searches for s in payload["snippets"]))
        assert st.ret_docs[st.ret_offsets[i]:st.ret_offsets[i + 1]].tolist() == retrieved
        final = inputs["results"][key].final_snippets
        by_url = {s.url: row_of(s.to_dict()) for s in final}
        presented = list(dict.fromkeys(s.url for s in final))
        assert st.pool_docs[st.pool_offsets[i]:st.pool_offsets[i + 1]].tolist() == [by_url[u] for u in presented]
        assert st.rank_docs[st.rank_offsets[i]:st.rank_offsets[i + 1]].tolist() == [by_url[u] for u in inputs["results"][key].ranking]
        own = np.flatnonzero(ev.answer == i)
        assert len(own) == events_expected
        for e in own:
            selected = ev.selected[ev.row_offsets[e]:ev.row_offsets[e + 1]]
            assert sorted(selected[selected >= 0].tolist()) == list(range(kept))
    prompts = study.readiness.read_jsonl(output / "prompts.jsonl.gz")
    assert [p["prompt_id"] for p in prompts] == codes["prompt"]
    assert all(p["prompt_text"] == inputs["text_of"][p["prompt_id"]] and p["keyword_text"] == inputs["keyword_of"][p["prompt_id"]].upper()
               for p in prompts)
    parallel = extract(tmp_path, inputs, monkeypatch, workers="2", name="extract-parallel")
    for name in ("prompts.jsonl.gz", "queries.jsonl.gz", "fidelity-sample.jsonl.gz", "answers.jsonl.gz"):
        assert study.readiness.read_jsonl(parallel / name) == study.readiness.read_jsonl(output / name)
    with np.load(output / "stages.npz") as a, np.load(parallel / "stages.npz") as b:
        assert all(np.array_equal(a[k], b[k]) for k in a.files)


def test_trace_checks_flag_a_shortlist_that_is_not_the_reranker_top_k(tmp_path):
    adapter = FrozenSnapshotSearchAdapter("duckduckgo", _snapshot(tmp_path))
    result = run_agent(stages.PARALLEL, adapter, "how does tax filing work", "natural", "tax filing", False)
    trace = result.trace.to_dict()
    presented = list(dict.fromkeys(s.url for s in result.final_snippets))
    assert all(stages.trace_stages(trace, stages.PARALLEL, presented)["checks"].values())
    compaction = next(e["payload"] for e in trace["events"] if e["event_type"] == "compaction")
    compaction["selected_snippets"] = compaction["selected_snippets"][::-1]
    checks = stages.trace_stages(trace, stages.PARALLEL, presented)["checks"]
    assert not checks["selection_is_top_k_by_score"] and not checks["shortlist_equals_presented"]


def _snapshot(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    study.readiness.write_jsonl(path, snapshot_rows())
    return path


def test_replay_reruns_the_frozen_search_only_after_recorded_queries_replay_identically(tmp_path, monkeypatch):
    inputs = build_inputs(tmp_path)
    output = extract(tmp_path, inputs, monkeypatch)
    replay = tmp_path / "replay"
    assert study.main(["replay", "--trace-extract", str(output), "--corpus-package", str(inputs["corpus"]),
                       "--workers", "2", "--output", str(replay)]) == 0
    manifest = json.loads((replay / "manifest.json").read_text())
    assert manifest["fidelity_gate"]["passed"] and manifest["counts"]["pages_not_in_corpus"] == 0
    rows = {(r["engine"], r["kind"], r["key"]): r["docs"] for r in study.readiness.read_jsonl(replay / "replay.jsonl.gz")}
    doc_of = lambda r: inputs["doc_index"][ordering.page_id(ordering.page_text(r["title"], r["snippet"]))]
    for engine, adapter in inputs["adapters"].items():
        assert rows[(engine, "prompt", "q001")] == [doc_of(r) for r in adapter._select_rows(inputs["text_of"]["q001"], 20)]
        keyword_rows = rows[(engine, "keyword", "SCHOOL SCHEDULING")]
        assert keyword_rows == [doc_of(r) for r in adapter._select_rows("SCHOOL SCHEDULING", 20)]
        own = {doc_of(r) for r in snapshot_rows() if r["keyword"] == "school scheduling"}
        assert len(keyword_rows) == 20 and set(keyword_rows[:12]) == own  # exact keyword matches come first
    tampered = tmp_path / "extract-tampered"
    shutil.copytree(output, tampered)
    sample = study.readiness.read_jsonl(tampered / "fidelity-sample.jsonl.gz")
    sample[0]["rows_sha256"] = "0" * 64
    study.readiness.write_jsonl(tampered / "fidelity-sample.jsonl.gz", sample)
    refused = tmp_path / "replay-refused"
    assert study.main(["replay", "--trace-extract", str(tampered), "--corpus-package", str(inputs["corpus"]),
                       "--workers", "1", "--output", str(refused)]) == 2
    assert not refused.exists()
    report = json.loads((tmp_path / "replay-refused.fidelity-failed.json").read_text())
    assert report["identical"] == report["recorded_searches_replayed"] - 1 and not report["passed"]


def test_analyze_end_to_end_reports_every_section(tmp_path, monkeypatch):
    inputs = build_inputs(tmp_path)
    output = extract(tmp_path, inputs, monkeypatch)
    replay = tmp_path / "replay"
    assert study.main(["replay", "--trace-extract", str(output), "--corpus-package", str(inputs["corpus"]),
                       "--workers", "1", "--output", str(replay)]) == 0
    rng = inputs["rng"]
    query_ids = [r["page_id"] for r in study.readiness.read_jsonl(output / "queries.jsonl.gz")]
    queries = {view: projections(tmp_path / f"queries-{view}", f"map-{view[0]}", {q: rng.normal(size=2) for q in query_ids}, rng)
               for view in ("qwen", "mistral")}
    index = [{"model": key[0], "engine": key[1], "method": key[2], "condition": key[3], "prompt_id": key[4],
              "answer_id": "answer-" + "-".join(key)} for key in inputs["results"] if key[3] == "natural"]
    study.readiness.write_jsonl(tmp_path / "answers-index.jsonl.gz", index)
    answers = {view: projections(tmp_path / f"answers-{view}", f"map-{view[0]}",
                                 {row["answer_id"]: rng.normal(size=2) for row in index}, rng) for view in ("qwen", "mistral")}
    analysis = tmp_path / "analysis"
    arguments = ["analyze", "--trace-extract", str(output), "--corpus-package", str(inputs["corpus"]),
                 "--qwen-prompts", str(tmp_path / "prompts-qwen"), "--mistral-prompts", str(tmp_path / "prompts-mistral"),
                 "--qwen-map", str(inputs["maps"]["qwen"]), "--mistral-map", str(inputs["maps"]["mistral"]),
                 "--battery", str(inputs["battery"]), "--final-axis-map", str(inputs["final"]), "--replay", str(replay),
                 "--queries-qwen", str(queries["qwen"]), "--queries-mistral", str(queries["mistral"]),
                 "--answers-index", str(tmp_path / "answers-index.jsonl.gz"), "--answers-qwen", str(answers["qwen"]),
                 "--answers-mistral", str(answers["mistral"]), "--bootstrap", "4", "--permutations", "4",
                 "--workers", "2", "--output", str(analysis)]
    assert study.main(arguments) == 0
    results = json.loads((analysis / "results.json").read_text())
    assert results["coverage"]["consistency_passed"] and results["coverage"]["replay"]["fidelity_gate"]["passed"]
    assert results["coverage"]["stages_available"] == list(stages.STAGES)
    assert results["coverage"]["null_anchor_max_abs_slope"] < 1e-9  # the bare-keyword replay is constant within keyword
    primary = [f"{m} · {e}" for m in ("llama4", "qwen38") for e in ENGINES]
    assert set(results["strata"]) == set(primary) | {f"{n} · {m}" for n in primary for m in ("Parallel", "Reactive")}
    for name in primary:
        stratum = results["strata"][name]
        assert set(stratum["selection"]["drivers"]) == set(geo.DRIVERS) == set(stratum["ranking_step"]["drivers"])
        assert stratum["selection"]["feature_means_sd"] == stratum["ranking_step"]["feature_means_sd"]  # one driver scale
        assert stratum["selection"]["position_effects"] == [0.0]
        assert {"pool_minus_reordering", "share_of_pool:reranker", "share_of_pool:query_rewriting", "utilization",
                "answer_on_ranking"} <= set(stratum["derived"])
    assert results["validity"]["answers"]["answers_joined"] == len(index)
    assert results["validity"]["queries"]["missing"] == 0
    assert set(results["replication"]) == {"slopes", "derived", "selection", "ranking_step"}
    assert "Rk" not in results["replication"]["slopes"]
    assert {"llama4 − qwen38 · duckduckgo", "Reactive − Parallel · llama4 · searxng"} <= set(results["contrasts"])
    report = (analysis / "final-report.html").read_text()
    assert "<title>Intent Surfacing Study</title>" in report and "How much does intent match raise" in report
    assert "NaN" not in (analysis / "results.json").read_text()
    with pytest.raises(ValueError, match="overwrite"):
        study.main(arguments)
    # A rerun (e.g. after an allocation ended mid-analysis) reuses every finished reranker stratum.
    shutil.rmtree(analysis)
    monkeypatch.setattr(geo, "estimate_e2", lambda *a, **k: pytest.fail("finished strata must come from the cache"))
    assert study.main(arguments) == 0
    rerun = json.loads((analysis / "results.json").read_text())
    assert all(rerun["strata"][n][part] == results["strata"][n][part] for n in results["strata"]
               for part in ("selection", "ranking_step", "score_drivers"))
