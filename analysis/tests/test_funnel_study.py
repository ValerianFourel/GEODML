"""Funnel study end to end on real Parallel and Reactive traces over a frozen snapshot fixture:
extract -> replay -> assemble -> analyze (two shards, cached) -> report."""

import json

import numpy as np
import pandas as pd
import pytest

from analysis.interpretability.pipeline import funnel_rows as fr
from analysis.scripts import funnel_study as study
from analysis.tests.test_intent_stages import ENGINES, build_inputs


def snapshot_args(tmp_path):
    return [arg for engine in ENGINES for arg in ("--snapshot", f"{engine}={tmp_path / 'snapshots' / f'{engine}.jsonl'}")]


def synthetic_features(tmp_path, corpus_dir, rng):
    """A features folder with the columns funnel_features.py writes, for the fixture's snapshot rows."""
    paths = {e: tmp_path / "snapshots" / f"{e}.jsonl" for e in ENGINES}
    rows = fr.snapshot_rows(paths)
    corpus = pd.read_parquet(corpus_dir / "snippets.parquet")
    corpus_row = {s: i for i, s in enumerate(corpus["snippet_id"])}
    n = len(rows)
    domain = [u.split("/")[2] for u in rows.url]
    table = pd.DataFrame({"row_id": np.arange(n), "engine": rows.engine, "keyword": rows.keyword, "position": rows.position,
                          "url": rows.url, "title": rows.title, "snippet": rows.snippet, "doc_id": rows.doc_id,
                          "searxng_score": np.where(np.asarray(rows.engine) == "searxng", rng.uniform(0, 1, n), np.nan),
                          "searxng_engine_count": np.where(np.asarray(rows.engine) == "searxng", 1.0, np.nan),
                          "corpus_row": [corpus_row[d] for d in rows.doc_id], "url_normalized": rows.url, "domain": domain,
                          "google_rank_url": np.nan, "google_rank_domain": np.nan,
                          "google_top20_url": rng.integers(0, 2, n), "google_top20_domain": rng.integers(0, 2, n)})
    out = tmp_path / "features"
    out.mkdir()
    table.to_parquet(out / "rows.parquet", index=False)
    docs = corpus[["consensus_axis_1_z", "prompt_scale_percentile_0_1"]].copy()
    docs["corpus_row"] = np.arange(len(docs))
    for name, *_ in study.ROW_FEATURES:
        source = next(c for (nm, _, t, c, _) in study.ROW_FEATURES if nm == name)
        table_name = next(t for (nm, _, t, c, _) in study.ROW_FEATURES if nm == name)
        if table_name == "docs" and source not in docs:
            docs[source] = rng.integers(0, 3, len(docs)).astype(float)
    docs.to_parquet(out / "docs.parquet", index=False)
    urls = pd.DataFrame({"url": sorted(set(rows.url))})
    domains = pd.DataFrame({"domain": sorted(set(domain))})
    for name, _, table_name, column, _ in study.ROW_FEATURES:
        target = {"urls": urls, "domains": domains}.get(table_name)
        if target is not None and column not in target:
            values = rng.normal(size=len(target)) if column not in ("has_llms_txt", "brand_list", "earned_list") else rng.integers(0, 2, len(target))
            target[column] = values.astype(float)
    urls.loc[urls.index[:3], "body_word_count"] = np.nan  # some pages without usable HTML
    domains.loc[domains.index[:2], "dfs_organic_count"] = np.nan
    urls.to_parquet(out / "urls.parquet", index=False)
    domains.to_parquet(out / "domains.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps({"row_table_digest": fr.row_table_digest(rows), "counts": {"rows": n}}))
    return out


@pytest.fixture()
def pipeline(tmp_path):
    inputs = build_inputs(tmp_path)
    rng = np.random.default_rng(3)
    features = synthetic_features(tmp_path, inputs["corpus"], rng)
    snaps = snapshot_args(tmp_path)
    extract = tmp_path / "extract"
    assert study.main(["extract", "--source", f"{inputs['root']}:llama4", "--source", f"{inputs['root']}:qwen38", *snaps,
                       "--final-axis-map", str(inputs["final"]), "--population-prompts", str(inputs["population"]),
                       "--workers", "1", "--output", str(extract)]) == 0
    replay = tmp_path / "replay"
    assert study.main(["replay", *snaps, "--population-prompts", str(inputs["population"]), "--output", str(replay)]) == 0
    assembled = tmp_path / "assembled"
    assert study.main(["assemble", "--extract", str(extract), "--features", str(features), "--replay", str(replay),
                       "--corpus-package", str(inputs["corpus"]), "--qwen-prompts", str(tmp_path / "prompts-qwen"),
                       "--mistral-prompts", str(tmp_path / "prompts-mistral"), "--qwen-map", str(inputs["maps"]["qwen"]),
                       "--mistral-map", str(inputs["maps"]["mistral"]), "--output", str(assembled)]) == 0
    return {"inputs": inputs, "extract": extract, "replay": replay, "assembled": assembled, "features": features, "tmp": tmp_path}


def test_extract_maps_every_answer_and_keeps_stages_nested(pipeline):
    manifest = json.loads((pipeline["extract"] / "manifest.json").read_text())
    assert manifest["counts"]["answers"] == 96 and not manifest["snapshot_hash_mismatch"]
    assert not any(k.startswith("skipped") for k in manifest["counts"])
    items = np.load(pipeline["extract"] / "items.npz")
    assert np.all(items["ranked"][items["ranked"] >= 0] >= 0)
    assert np.all((items["presented"] >= 0) <= (items["scored"] == 1))
    assert np.all((items["ranked"] >= 0) <= (items["presented"] >= 0))
    assembled = json.loads((pipeline["assembled"] / "manifest.json").read_text())
    assert assembled["counts"]["answers_without_keyword_rows"] == 0 and assembled["counts"]["u_items"] == 96 * 12
    u = np.load(pipeline["assembled"] / "u.npz")
    assert set(np.unique(u["replay"])) <= {0, 1} and u["replay"].sum() > 0
    assert np.all(u["ranked"] <= u["presented"]) and np.all(u["presented"] <= u["scored"]) and np.all(u["scored"] <= u["retrieved"])


def run_analysis(pipeline, shard, output="analysis", bootstrap="3", permutations="3", extra=()):
    return study.main(["analyze", "--assembled", str(pipeline["assembled"]), "--output", str(pipeline["tmp"] / output),
                       "--specs", "main,visible", "--bootstrap", bootstrap, "--permutations", permutations,
                       "--secondary-bootstrap", "2", "--split", "all", "--shard", shard, "--workers", "1", *extra])


def test_analyze_in_shards_then_report(pipeline, monkeypatch):
    report_args = ["report", "--assembled", str(pipeline["assembled"]), "--output", str(pipeline["tmp"] / "analysis"),
                   "--specs", "main,visible", "--bootstrap", "3", "--permutations", "3", "--secondary-bootstrap", "2",
                   "--split", "all", "--report", str(pipeline["tmp"] / "report")]
    assert run_analysis(pipeline, "1/2", extra=("--stop-after-minutes", "-1")) == 0  # the guard starts no task
    assert study.main(report_args) == 3
    assert run_analysis(pipeline, "1/2") == 0
    assert study.main(report_args) == 3  # shard 2 is missing
    assert run_analysis(pipeline, "2/2") == 0
    assert study.main(report_args) == 0
    results = json.loads((pipeline["tmp"] / "report" / "results.json").read_text())
    main = results["models"]["main"]
    assert set(main) == {"llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive"}
    for stratum in main.values():
        assert set(stratum) == set(study.STAGE_NAMES)
        for stage in stratum.values():
            if stage.get("features"):
                assert {"intent_alignment", "topic_similarity", study.PRIMARY_C1, "body_word_count"} <= set(stage["features"])
    assert set(results["confirmatory"]) == {"P1 alignment at R|U", "P2 alignment R − R0", "P3 domain authority K|P − P|C",
                                            "P4 page intent × x at R|U"}
    for s, by_feature in results["decomposition"].items():
        for feature, d in by_feature.items():
            if d["answers"]:
                logs = [v for v in d["log_rr"].values() if v is not None]
                assert d["total_log_rr_K_given_U"] == pytest.approx(sum(logs))
    assert "NaN" not in (pipeline["tmp"] / "report" / "results.json").read_text()
    visible = results["models"]["visible"]["llama4 · Parallel"]["K|P"]
    assert len(visible["replicates"]) == 2 and all(f["permutation_p"] is None for f in visible["features"].values())
    assert not any(f["block"] in ("C1", "C2") for f in visible["features"].values())
    assert "<title>Funnel Study</title>" in (pipeline["tmp"] / "report" / "final-report.html").read_text()
    # a rerun reuses every cached task and refits nothing
    monkeypatch.setattr(study.fm, "estimate_blocks", lambda *a, **k: pytest.fail("cached tasks must not be refitted"))
    assert run_analysis(pipeline, "1/1") == 0
    with pytest.raises(ValueError, match="overwrite"):
        study.main(report_args)


def test_assemble_refuses_a_feature_table_from_other_snapshots(pipeline):
    manifest = json.loads((pipeline["features"] / "manifest.json").read_text())
    manifest["row_table_digest"] = "0" * 64
    (pipeline["features"] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different snapshot row tables"):
        study.main(["assemble", "--extract", str(pipeline["extract"]), "--features", str(pipeline["features"]),
                    "--corpus-package", str(pipeline["inputs"]["corpus"]), "--qwen-prompts", "x", "--mistral-prompts", "x",
                    "--qwen-map", "x", "--mistral-map", "x", "--output", str(pipeline["tmp"] / "again")])


def test_the_exploration_split_is_fixed_and_about_thirty_percent():
    import hashlib
    words = [f"keyword {i}" for i in range(4000)]
    share = sum(study.exploration_keyword(w) for w in words) / len(words)
    assert 0.27 < share < 0.33
    assert study.exploration_keyword("robinhood vs etrade") == (
        int(hashlib.sha256(b"funnel-exploration-v1:robinhood vs etrade").hexdigest()[:8], 16) / 2 ** 32 < 0.30)
