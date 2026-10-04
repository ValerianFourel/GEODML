"""GEO drivers study: planted-effect recovery, replication rule, row bookkeeping, assemble→analyze."""
import gzip
import json

import numpy as np
import pytest

from analysis.interpretability.pipeline import geo_drivers as geo
from analysis.scripts import geo_drivers_study as study


def synthetic(seed=0, align=1.5, models=(0,), engines=(0,), per_keyword=12, keywords=20, answers_per_prompt=2):
    rng = np.random.default_rng(seed)
    n_docs = 300
    doc_z = rng.normal(size=n_docs)
    doc_u = np.argsort(np.argsort(doc_z)) / (n_docs - 1)
    cols = {k: [] for k in ("model", "engine", "keyword", "prompt", "x")}
    p_off, p_docs, p_on, p_top, r_off, r_docs = [0], [], [], [], [0], []
    prompt = 0
    for m in models:
        for e in engines:
            for k in range(keywords):
                for _ in range(per_keyword):
                    x = rng.uniform()
                    for _ in range(answers_per_prompt):
                        docs = rng.choice(n_docs, 7, replace=False)
                        on = (rng.uniform(size=7) < 0.4).astype(float)
                        topic = rng.normal(size=7)
                        utility = (1.0 * on + 0.8 * topic + 0.3 * doc_z[docs] - align * np.abs(doc_u[docs] - x)
                                   - 0.2 * np.arange(7) + rng.gumbel(size=7))
                        for name, value in zip(cols, (m, e, k, prompt, x)):
                            cols[name].append(value)
                        p_docs += list(docs); p_on += list(on); p_top += list(topic); p_off.append(len(p_docs))
                        r_docs += list(docs[np.argsort(-utility)[:4]]); r_off.append(len(r_docs))
                    prompt += 1
    a = np.asarray
    return geo.Answers(a(cols["model"]), a(cols["engine"]), a(cols["keyword"]), a(cols["prompt"]), a(cols["x"], float),
                       a(p_off), a(p_docs), a(p_on, float), a(p_top, float), a(r_off), a(r_docs), doc_z, doc_u)


def test_planted_drivers_and_intent_shift_are_recovered():
    a = synthetic(keywords=30, per_keyword=20, answers_per_prompt=3)
    e1 = geo.estimate_e1(a, bootstrap=20, permutations=19, seed=1)
    assert e1["ranking"]["slope"] > 0.3 and e1["ranking"]["ci95"][0] > 0 and e1["ranking"]["permutation_p"] == pytest.approx(0.05)
    assert abs(e1["pool"]["slope"]) < 0.1  # random pools: the shift is in the reordering
    assert e1["reordering"]["slope"] == pytest.approx(e1["ranking"]["slope"] - e1["pool"]["slope"], abs=1e-9)
    e2 = geo.estimate_e2(a, bootstrap=10, permutations=9, seed=1, workers=2)
    drivers = e2["drivers"]
    assert drivers["on_keyword"]["beta_per_sd"] == pytest.approx(0.49, abs=0.08)   # 1.0 × sd of a 0.4 Bernoulli
    assert drivers["topic_similarity"]["beta_per_sd"] == pytest.approx(0.8, abs=0.08)
    assert drivers["page_intent"]["beta_per_sd"] == pytest.approx(0.3, abs=0.08)
    assert drivers["intent_alignment"]["beta_per_sd"] > 0.25 and drivers["intent_alignment"]["permutation_p"] == pytest.approx(0.1)
    assert max(drivers, key=lambda d: drivers[d]["fit_share"]) == "topic_similarity"
    assert sum(d["fit_share"] for d in drivers.values()) == pytest.approx(1.0)


def test_replication_needs_every_stratum_and_the_permutation_null():
    real = synthetic(models=(0, 1), engines=(0, 1), keywords=15, per_keyword=12)
    null = synthetic(seed=5, align=0.0, models=(0, 1), engines=(0, 1), keywords=15, per_keyword=12)
    for data, expected in ((real, True), (null, False)):
        strata = {}
        for m in (0, 1):
            for e in (0, 1):
                sub = data.subset((data.model == m) & (data.engine == e))
                # 39 shuffles: the smallest attainable p (1/40) can pass the protocol's p < 0.05.
                strata[f"{m}-{e}"] = {"e1": geo.estimate_e1(sub, bootstrap=20, permutations=39, seed=m * 10 + e),
                                      "e2": geo.estimate_e2(sub, bootstrap=10, permutations=39, seed=m * 10 + e)}
        e1 = geo.replication(strata, section="e1", terms=("ranking",))
        e2 = geo.replication(strata, section="e2", terms=geo.DRIVERS)
        assert e1["ranking"]["replicates"] is expected and e2["intent_alignment"]["replicates"] is expected
        assert e2["topic_similarity"]["replicates"] and len(e2["topic_similarity"]["strata"]) == 4


def test_rows_restricted_to_a_stratum_match_rows_built_for_it():
    a = synthetic(models=(0, 1), keywords=4, per_keyword=3)
    keep = np.flatnonzero(a.model == 1)
    sub = a.subset(a.model == 1)
    restricted, rebuilt = geo.restrict(geo.choice_rows(a), keep), geo.choice_rows(sub)
    for field in ("z", "on_keyword", "topic", "position", "set", "generation", "chosen"):
        assert np.array_equal(getattr(restricted, field), getattr(rebuilt, field)), field
    assert restricted.sets == rebuilt.sets


def write_fixture(tmp_path, rng):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from analysis.scripts.page_readiness_ordering import EMBEDDING_FILES, write_json, write_jsonl
    dim, n_docs = 6, 40
    corpus = tmp_path / "corpus"
    (corpus / "embeddings").mkdir(parents=True)
    keywords = ["tax filing", "school scheduling"]
    pq.write_table(pa.table({"row": list(range(n_docs)), "snippet_id": [f"d{i}" for i in range(n_docs)],
                             "consensus_axis_1_z": list(rng.normal(size=n_docs)),
                             "prompt_scale_percentile_0_1": list(rng.uniform(size=n_docs)),
                             "keywords": [[keywords[i % 2]] for i in range(n_docs)]}), corpus / "snippets.parquet")
    write_json(corpus / "manifest.json", {"rows": n_docs})
    docs = rng.normal(size=(n_docs, dim)).astype(np.float32)
    for name in EMBEDDING_FILES.values():
        np.save(corpus / name, docs)
    prompts = [f"q{i}" for i in range(12)]
    prompt_vectors = rng.normal(size=(12, dim)).astype(np.float32)
    prompt_vectors[0] = docs[0]  # same text as page d0: identical intent-free vector
    maps = {}
    for view in ("qwen", "mistral"):
        shard = tmp_path / f"prompts-{view}/shard-000"
        shard.mkdir(parents=True)
        np.savez(shard / "question_embeddings.restricted-local.npz", candidate_ids=np.asarray(prompts), embeddings=prompt_vectors)
        maps[view] = tmp_path / f"map-{view}"
        maps[view].mkdir()
        write_json(maps[view] / "readiness_embedding_map.json",
                   {"embedding_mean": [0.0] * dim, "supervised_subspace_axes": [[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]]})
    population = tmp_path / "population-prompts.jsonl"
    write_jsonl(population, [{"candidate_id": p, "keyword": keywords[i % 2].upper()} for i, p in enumerate(prompts)])
    extract = tmp_path / "extract"
    extract.mkdir()
    observations = []
    for i, p in enumerate(prompts):
        for model in ("llama4", "qwen38"):
            for engine in ("duckduckgo", "searxng"):
                presented = [f"d{j}" for j in rng.choice(n_docs, 5, replace=False)]
                if i == 0:
                    presented[0] = "d0"
                observations.append({"generation_id": f"{p}-{model}-{engine}", "prompt_id": p, "keyword": f"k{i % 2}",
                                     "model": model, "engine": engine, "method": "Parallel-Expansion-v1", "condition": "natural",
                                     "prompt_axis": (i + 0.5) / 12, "presented": presented, "ranking": presented[:3]})
    write_jsonl(extract / "observations.jsonl.gz", observations)
    write_json(extract / "manifest.json", {"stage": "extract"})
    return extract, corpus, maps, population, docs


def test_assemble_joins_keyword_and_intent_free_topic_then_analyze_reports(tmp_path):
    rng = np.random.default_rng(3)
    extract, corpus, maps, population, docs = write_fixture(tmp_path, rng)
    assembled = tmp_path / "assembled"
    assert study.main(["assemble", "--extract", str(extract), "--corpus-package", str(corpus),
                       "--qwen-prompts", str(tmp_path / "prompts-qwen"), "--mistral-prompts", str(tmp_path / "prompts-mistral"),
                       "--qwen-map", str(maps["qwen"]), "--mistral-map", str(maps["mistral"]),
                       "--population-prompts", str(population), "--output", str(assembled)]) == 0
    a, codes = study.load_answers(assembled)
    assert len(a.x) == 48 and set(codes["model"]) == {"llama4", "qwen38"}
    first = a.p_offsets[0]
    assert a.p_topic[first] == pytest.approx(1.0, abs=1e-5)  # prompt q0 and page d0 share the same vector
    on = a.p_on_keyword[a.p_offsets[0]:a.p_offsets[1]]
    docs_first = a.p_docs[a.p_offsets[0]:a.p_offsets[1]]
    # Prompt q0's keyword is "TAX FILING": matched case-insensitively to the even documents' "tax filing".
    assert list(on) == [float(d % 2 == 0) for d in docs_first]
    manifest = json.loads((assembled / "manifest.json").read_text())
    assert manifest["counts"]["answers"] == 48 and manifest["counts"]["prompt_keyword_found_in_corpus"] == 2
    output = tmp_path / "analysis"
    assert study.main(["analyze", "--assembled", str(assembled), "--bootstrap", "3", "--permutations", "3",
                       "--workers", "1", "--output", str(output)]) == 0
    results = json.loads((output / "results.json").read_text())
    assert set(results["strata"]) == {f"{m} · {e}" for m in ("llama4", "qwen38") for e in ("duckduckgo", "searxng", "all engines")}
    assert set(results["replication"]["e2"]) == set(geo.DRIVERS)
    assert "<title>GEO Drivers Study</title>" in (output / "report.html").read_text()
