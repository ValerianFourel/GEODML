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
