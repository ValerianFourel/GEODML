"""The steelman on HoreKa-shaped inputs: funnel_study extract/replay/assemble over real Parallel and Reactive
traces of two models (no query file in the extract), the agent's queries from intent_stages_study trace-extract,
the registration's flat population-prompts file, the confirmation-style split argument. Every part, then the report."""

import gzip
import json

import numpy as np
import pytest

from analysis.scripts import intent_stages_study as intent
from analysis.steelman import __main__ as cli
from analysis.steelman import tables as T
from analysis.tests.test_funnel_study import pipeline  # noqa: F401  (fixture)


@pytest.fixture()
def horeka(pipeline, tmp_path):  # noqa: F811
    inputs = pipeline["inputs"]
    trace = tmp_path / "trace-extract"
    assert intent.main(["trace-extract", "--source", f"{inputs['root']}:llama4", "--source", f"{inputs['root']}:qwen38",
                        "--final-axis-map", str(inputs["final"]), "--corpus-package", str(inputs["corpus"]),
                        "--population-prompts", str(inputs["population"]), "--workers", "1", "--output", str(trace)]) == 0
    common = ["--assembled", str(pipeline["assembled"]), "--extract", str(pipeline["extract"]), "--replay", str(pipeline["replay"]),
              "--trace-extract", str(trace), "--features", str(pipeline["features"]), "--prompts", str(inputs["population"]),
              "--split", "all", "--bootstrap", "4", "--permutations", "3", "--model-draws", "2", "--mc", "8",
              "--selection-draws", "2", "--workers", "1", "--output", str(tmp_path / "steelman")]
    return {**pipeline, "trace": trace, "common": common, "out": tmp_path / "steelman"}


def test_loader_joins_queries_and_reads_flat_prompts(horeka):
    t = T.load(prompts=horeka["inputs"]["population"], assembled=horeka["assembled"], extract=horeka["extract"],
               replay=horeka["replay"], trace_extract=horeka["trace"], split="all")
    assert set(t.model) == {"llama4", "qwen38"}
    assert t.queries is not None and all(len(q) >= 1 for q in t.queries)
    assert "joined on fingerprint" in t.manifest["queries"]
    first = json.loads(gzip.open(horeka["extract"] / "answers.jsonl.gz", "rt").readline())
    assert first["fingerprint"]
    assert np.isfinite(t.meta["words"]).all()
    assert t.common().any()


def test_every_part_runs_and_reports(horeka):
    for part in ("chain", "generator", "fe", "pairs", "followup", "lexical"):
        assert cli.main([part, *horeka["common"]]) == 0, part
    with pytest.raises(SystemExit):
        cli.main(["chain", *horeka["common"]])          # refuses to overwrite
    assert cli.main(["report", "--output", str(horeka["out"])]) == 0
    text = (horeka["out"] / "RESULTS.md").read_text()
    assert "llama4" in text and "qwen38" in text
    gen = json.loads((horeka["out"] / "generator.json").read_text())
    assert all("keep_modelled" in e for e in gen["strata"].values())
    lex = json.loads((horeka["out"] / "lexical.json").read_text())
    assert "joined on fingerprint" in lex["agent_queries"] and lex["lexicon_source"].startswith("frozen")
    manifest = json.loads((horeka["out"] / "manifest.json").read_text())
    assert set(manifest["verdicts"]["C2"]) == {"llama4", "qwen38"}


def test_full_run_parts_on_two_models(horeka):
    from analysis.steelman import decide
    for part in ("census", "supply", "ablation", "queries"):
        assert cli.main([part, *horeka["common"]]) == 0, part
    census = json.loads((horeka["out"] / "census.json").read_text())
    assert {r["model"] for r in census["table"]} == {"llama4", "qwen38"}
    conditions = {json.loads(line)["condition"] for line in gzip.open(horeka["extract"] / "answers.jsonl.gz", "rt")}
    assert {r["condition"] for r in census["table"]} == conditions and "natural" in conditions
    supply = json.loads((horeka["out"] / "supply.json").read_text())
    for e in supply["strata"].values():
        assert set(e["slopes"]) >= {"K", "oracle_U", "oracle_R", "oracle_C", "oracle_P", "oracle_snapshot"}
        assert len(e["deciles"]) == 10
    assert decide.verdict_supply(supply["strata"])["verdict"] in ("supported", "narrowed", "failed")
    ablation = json.loads((horeka["out"] / "ablation.json").read_text())
    assert ablation["strata"]
    queries = json.loads((horeka["out"] / "queries.json").read_text())
    assert queries["strata"]


def test_oracle_picks_the_closest_rows_by_hand():
    from analysis.steelman.fullparts import oracle_k, oracle_k_snapshot
    u = np.array([0.1, 0.5, 0.9, 0.45])
    x = np.array([0.5, 0.95])
    L = np.array([2, 1])
    got = oracle_k(np.array([0, 0, 0, 0, 1, 1]), np.array([0, 1, 2, 3, 0, 2]), x, u, L, 2)
    w = 1 / np.log2(np.arange(2) + 2)
    assert got[0] == pytest.approx((w[0] * 0.5 + w[1] * 0.45) / w.sum())
    assert got[1] == pytest.approx(0.9)
    snap = oracle_k_snapshot(x, L, np.sort(u))
    assert snap[0] == pytest.approx(got[0]) and snap[1] == pytest.approx(0.9)


def test_generator_and_fe_split_by_stratum_then_assemble(horeka):
    out = horeka["out"]
    for part in ("generator", "fe"):
        for stratum in ("llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive"):
            assert cli.main([part, *horeka["common"], "--stratum", stratum]) == 0
        assert not (out / f"{part}.json").exists()
        assert cli.main([part, *horeka["common"]]) == 0      # assembles from the per-stratum cache
        body = json.loads((out / f"{part}.json").read_text())
        assert set(body["strata"]) == {"llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive"}
    caches = list((out / "generator.cache").iterdir())
    assert len(caches) == 1 and (caches[0] / "key.json").exists()
