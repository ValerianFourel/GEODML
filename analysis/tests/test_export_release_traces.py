"""export_release_traces.py on the synthetic full-run fixture: every table is written, the per-answer stage values
reproduce the stage figure's curves exactly, and the trace and funnel tables share their answer ids."""

import json

import numpy as np
import pandas as pd
import pytest

from analysis.interpretability.pipeline import intent_stages as stages
from analysis.scripts import export_release_traces as export
from analysis.scripts import funnel_study
from analysis.scripts import intent_stages_study as study
from analysis.steelman import tables as chain_tables
from analysis.tests.test_funnel_study import pipeline  # noqa: F401  (fixture)
from analysis.tests.test_page_readiness_ordering import projections


@pytest.fixture()
def run(pipeline, monkeypatch):  # noqa: F811
    """A run folder laid out like the cluster's: trace extract, replay, query and answer projections, answer index,
    the stage analysis, and the funnel extract and assembly of the same cells."""
    inputs, tmp = pipeline["inputs"], pipeline["tmp"]
    root = tmp / "run"
    root.mkdir()
    monkeypatch.setattr(study, "FIDELITY_PREFIX", "")
    assert study.main(["trace-extract", "--source", f"{inputs['root']}:llama4", "--source", f"{inputs['root']}:qwen38",
                       "--final-axis-map", str(inputs["final"]), "--corpus-package", str(inputs["corpus"]),
                       "--population-prompts", str(inputs["population"]), "--workers", "1", "--output", str(root / "trace-extract")]) == 0
    assert study.main(["replay", "--trace-extract", str(root / "trace-extract"), "--corpus-package", str(inputs["corpus"]),
                       "--workers", "1", "--output", str(root / "intent-replay")]) == 0
    rng = inputs["rng"]
    query_ids = [r["page_id"] for r in study.readiness.read_jsonl(root / "trace-extract/queries.jsonl.gz")]
    for view in ("qwen", "mistral"):
        projections(root / f"embed-queries-{view}.merged", f"map-{view[0]}", {q: rng.normal(size=2) for q in query_ids}, rng)
    index = [{"model": k[0], "engine": k[1], "method": k[2], "condition": k[3], "prompt_id": k[4],
              "answer_id": "answer-" + "-".join(k)} for k in inputs["results"] if k[3] == "natural"]
    (root / "answers-export").mkdir()
    study.readiness.write_jsonl(root / "answers-export/observations.jsonl.gz", index)
    for view in ("qwen", "mistral"):
        projections(root / f"embed-answers-{view}.merged", f"map-{view[0]}", {r["answer_id"]: rng.normal(size=2) for r in index}, rng)
    assert study.main(["analyze", "--trace-extract", str(root / "trace-extract"), "--corpus-package", str(inputs["corpus"]),
                       "--qwen-prompts", str(tmp / "prompts-qwen"), "--mistral-prompts", str(tmp / "prompts-mistral"),
                       "--qwen-map", str(inputs["maps"]["qwen"]), "--mistral-map", str(inputs["maps"]["mistral"]),
                       "--battery", str(inputs["battery"]), "--final-axis-map", str(inputs["final"]), "--replay", str(root / "intent-replay"),
                       "--queries-qwen", str(root / "embed-queries-qwen.merged"), "--queries-mistral", str(root / "embed-queries-mistral.merged"),
                       "--answers-index", str(root / "answers-export/observations.jsonl.gz"),
                       "--answers-qwen", str(root / "embed-answers-qwen.merged"), "--answers-mistral", str(root / "embed-answers-mistral.merged"),
                       "--bootstrap", "3", "--permutations", "3", "--workers", "1", "--output", str(root / "intent-stages")]) == 0
    (root / "funnel-extract").symlink_to(pipeline["extract"], target_is_directory=True)
    (root / "funnel-assembled").symlink_to(pipeline["assembled"], target_is_directory=True)
    (root / "funnel-replay").symlink_to(pipeline["replay"], target_is_directory=True)
    return {"root": root, "inputs": inputs, "tmp": tmp}


def export_run(run, *extra):
    out = run["tmp"] / "release-traces"
    rc = export.main(["--run", str(run["root"]), "--corpus-package", str(run["inputs"]["corpus"]), "--battery", str(run["inputs"]["battery"]),
                      "--final-axis-map", str(run["inputs"]["final"]), "--output", str(out), *extra])
    return rc, out


def test_export_reproduces_the_stage_figure_and_links_both_extracts(run):
    rc, out = export_run(run)
    assert rc == 0
    check = json.loads((out / "figure_check.json").read_text())
    assert check["passed"] and check["max_abs_difference"] <= 1e-12 and not check["missing"] and check["curves_compared"] > 50
    answers = pd.read_parquet(out / "trace_answers.parquet")
    assert len(answers) == 96 and answers["answer_id"].is_unique
    assert set(answers["split"]) <= {"exploration", "held-out"}
    natural = answers[answers["condition"].astype(str) == "natural"]
    assert natural["A"].notna().all() and answers.loc[answers["condition"].astype(str) != "natural", "A"].isna().all()
    assert answers[["Q", "R0", "R", "C", "P", "K", "R_u", "P_u", "K_u"]].notna().all().all()
    # every answer's agent queries, with their positions
    links = pd.read_parquet(out / "answer_queries.parquet")
    queries = pd.read_parquet(out / "queries.parquet")
    assert links.groupby(links["answer_id"].astype(str)).size().reindex(answers["answer_id"]).tolist() == answers["n_queries"].tolist()
    q = links.merge(queries, on="query_id").groupby(links["answer_id"].astype(str))["consensus_axis_1_z"].mean()
    assert np.allclose(q.reindex(answers["answer_id"]).to_numpy(), answers["Q"].to_numpy())
    # the funnel tables describe the same answers, and their rows give the same retrieved and cited pages
    items = pd.read_parquet(out / "stage_items.parquet")
    assert set(items["answer_id"].astype(str)) == set(answers["answer_id"])
    corpus = pd.read_parquet(run["inputs"]["corpus"] / "snippets.parquet")
    rows = pd.read_parquet(run["tmp"] / "features/rows.parquet")
    doc_z = corpus["consensus_axis_1_z"].to_numpy()[rows.sort_values("row_id")["corpus_row"].to_numpy()]
    items["z"] = doc_z[items["row_id"].to_numpy()]
    items["doc"] = rows.sort_values("row_id")["corpus_row"].to_numpy()[items["row_id"].to_numpy()]
    retrieved = items.drop_duplicates(["answer_id", "doc"]).groupby(items["answer_id"].astype(str))["z"].mean()
    assert np.allclose(retrieved.reindex(answers["answer_id"]).to_numpy(), answers["R"].to_numpy())
    cited = items[items["cited_rank"] >= 0].sort_values(["answer_id", "cited_rank"])
    k = cited.groupby(cited["answer_id"].astype(str)).apply(
        lambda g: np.average(g["z"], weights=1 / np.log2(np.arange(len(g)) + 2)))
    assert np.allclose(k.reindex(answers["answer_id"]).to_numpy(), answers["K"].to_numpy())
    for name in ("rerank_events", "stage_u", "shown"):
        table = pd.read_parquet(out / f"{name}.parquet")
        assert set(table["answer_id"].astype(str)) <= set(answers["answer_id"]) and len(table) > 0
    # the stage chain's values are the ones the chain itself loads, joined on the answer id
    t = chain_tables.load(run["root"], prompts=run["inputs"]["population"], assembled=run["root"] / "funnel-assembled",
                          extract=run["root"] / "funnel-extract", replay=run["root"] / "funnel-replay")
    mine = answers.set_index("answer_id").loc[[export.answer_id(*k) for k in zip(t.model, t.prompt_id, t.engine, t.method, t.condition)]]
    for term in chain_tables.CHAIN:
        np.testing.assert_array_equal(mine[f"chain_{term}"].to_numpy(), t.values[term])
    assert mine["chain_K"].notna().all() and np.array_equal(mine["x"].to_numpy(), t.x)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tables"]["trace_answers"]["rows"] == 96 and manifest["figure_check"]["passed"]
    assert manifest["validity"]["chain"] == {"trace_answers_without_funnel_answer": 0, "funnel_answers": 96,
                                             "x_max_abs_difference": 0.0, "keyword_mismatches": 0}
    with pytest.raises(ValueError, match="overwrite"):
        export_run(run)


def test_figure_check_fails_when_a_stage_value_changes(run, tmp_path):
    _, out = export_run(run, "--skip-funnel")
    table = pd.read_parquet(out / "trace_answers.parquet")
    table.loc[table.index[0], "K_u"] += 0.01
    tampered = tmp_path / "tampered.parquet"
    table.to_parquet(tampered, index=False)
    check = export.figure_check(tampered, run["root"] / "intent-stages/results.json")
    assert not check["passed"] and check["max_abs_difference"] > 1e-4


def test_answer_ids_match_the_release_answers_table():
    import hashlib
    assert export.answer_id("qwen38", "readiness-question:abc", "searxng", "Parallel-Expansion-v1", "natural") == \
        hashlib.sha256(b"qwen38|readiness-question:abc|searxng|Parallel-Expansion-v1|natural").hexdigest()[:24]
    assert set(export.Z_TERMS) <= set(stages.STAGES) and set(export.U_TERMS) == set(stages.CHAIN)
    assert funnel_study.exploration_keyword("crm software") == funnel_study.exploration_keyword("crm software")
