"""Subspace relocation, page placement and ranking-model contracts on synthetic data."""
import gzip
import json

import numpy as np
import pytest

from analysis.interpretability.pipeline import page_readiness_ordering as ordering
from analysis.interpretability.pipeline.agentic_dataset import FinalDatasetWriter
from analysis.scripts import page_readiness_ordering as cli
from analysis.tests.test_source_importance_pipeline import A, B, C, dataset

ALIGNMENT = {"orthogonal_rotation": [[0.8, -0.6], [0.6, 0.8]],
             "reference_development_mean": [0.1, -0.2], "reference_development_scale": [2.0, 1.5],
             "candidate_development_mean": [-0.3, 0.4], "candidate_development_scale": [0.5, 3.0]}


def battery(root):
    root.mkdir()
    cli.write_json(root / "battery_manifest.json", {"reference_map_id": "map-q", "candidate_map_id": "map-m"})
    cli.write_json(root / "readiness_robustness_battery.json", {"cross_embedding_alignment": ALIGNMENT})
    return root


def projections(root, map_id, raw, rng):
    root.mkdir()
    rows = [{"candidate_id": item, "projection": {
        "item_id": item, "text_sha256": "h" + item, "raw_axis_1": float(a), "raw_axis_2": float(b),
        "normalized_axis_1": 0.0, "normalized_axis_2": 0.0, "predicted_scalar_readiness_0_1": float(rng.uniform())}}
        for item, (a, b) in raw.items()]
    cli.write_jsonl(root / "question_projections.jsonl", rows)
    cli.write_json(root / "projection_manifest.json", {"map_id": map_id})
    return root


def archived_final_axis_map(path, aligned_rows):
    """The final-audit rule: consensus of both axis-1 z values, sorted by (z, id), percentile rank."""
    rows = [{**r, "consensus_axis_1_z": (r["reference_axis_1_z"] + r["candidate_aligned_axis_1_z"]) / 2}
            for r in aligned_rows]
    rows.sort(key=lambda r: (r["consensus_axis_1_z"], r["candidate_id"]))
    for rank, row in enumerate(rows):
        row["axis_1_rank"], row["axis_1_percentile_0_1"] = rank, rank / (len(rows) - 1)
    cli.write_jsonl(path, rows)
    return rows


def prompt_archive(tmp_path, count=40):
    rng = np.random.default_rng(3)
    ids = [f"q{i:03d}" for i in range(count)]
    qwen = projections(tmp_path / "merged-qwen", "map-q", {i: rng.normal(size=2) for i in ids}, rng)
    mistral = projections(tmp_path / "merged-mistral", "map-m", {i: rng.normal(size=2) for i in ids}, rng)
    bat = battery(tmp_path / "battery")
    final = tmp_path / "final-axis-map.jsonl"
    archived_final_axis_map(final, cli.aligned(qwen, mistral, bat))
    return qwen, mistral, bat, final


def test_relocation_reproduces_archived_prompt_coordinates_and_detects_drift(tmp_path):
    qwen, mistral, bat, final = prompt_archive(tmp_path)
    output = tmp_path / "relocation.json"
    assert cli.main(["relocate", "--final-axis-map", str(final), "--qwen-projections", str(qwen),
                     "--mistral-projections", str(mistral), "--battery", str(bat),
                     "--fresh-qwen", str(qwen), "--fresh-mistral", str(mistral), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["passed"] and report["archived_coordinates"]["max_abs_consensus_z_difference"] < 1e-12
    with pytest.raises(ValueError, match="overwrite"):
        cli.main(["relocate", "--final-axis-map", str(final), "--qwen-projections", str(qwen),
                  "--mistral-projections", str(mistral), "--battery", str(bat), "--output", str(output)])
    rows = cli.read_jsonl(final)
    rows[5]["consensus_axis_1_z"] += 0.01  # an archived coordinate the frozen chain no longer reproduces
    audit = ordering.relocation_audit(rows, cli.aligned(qwen, mistral, bat))
    assert not audit["passed"] and audit["max_abs_consensus_z_difference"] == pytest.approx(0.01)


def test_fresh_reembedding_agreement_flags_reordered_items():
    archived = {f"q{i}": float(i) for i in range(10)}
    assert ordering.projection_agreement(archived, {k: v + 1e-4 for k, v in archived.items()})["passed"]
    swapped = {**archived, "q0": 9.5}
    assert not ordering.projection_agreement(archived, swapped)["passed"]


def test_prompt_scale_places_pages_on_the_prompt_percentile_scale():
    scale = ordering.PromptScale([{"consensus_axis_1_z": v} for v in (-2.0, 0.0, 2.0)])
    values, outside = scale.percentile(np.asarray([-3.0, -1.0, 0.0, 2.0, 5.0]))
    assert values.tolist() == [0.0, 0.25, 0.5, 1.0, 1.0]
    assert outside.tolist() == [True, False, False, False, True]


def planted(interaction, *, answers=1500, seed=0):
    rng = np.random.default_rng(seed)
    pages = {f"p{i}": float(rng.normal()) for i in range(200)}
    observations = []
    for g in range(answers):
        axis = rng.uniform()
        presented = list(rng.choice(list(pages), 8, replace=False))
        utility = np.array([0.4 * pages[p] + interaction * pages[p] * (axis - 0.5) - 0.3 * k
                            for k, p in enumerate(presented)]) + rng.gumbel(size=8)
        observations.append({"keyword": f"k{g % 20}", "prompt_id": f"q{g}", "prompt_axis": axis, "model": "llama4",
                             "method": "m", "presented": presented,
                             "ranking": [presented[i] for i in np.argsort(-utility)[:4]]})
    return observations, pages


def test_plackett_luce_recovers_planted_page_and_interaction_effects():
    observations, pages = planted(1.5)
    data = ordering.ChoiceData(observations, pages)
    fit = ordering.fit_plackett_luce(data)
    assert fit["page_z"] == pytest.approx(0.4, abs=0.08)
    assert fit["page_z_x_prompt_axis"] == pytest.approx(1.5, abs=0.25)
    assert fit["position_effects"][1] == pytest.approx(-0.3, abs=0.1)
    permutation = ordering.within_keyword_permutation(data, observed=fit["page_z_x_prompt_axis"], replicates=19,
                                                      seed=1, start=fit["theta"])
    assert permutation["two_sided_p"] == pytest.approx(0.05)
    boot = ordering.keyword_bootstrap(data, replicates=10, seed=1, start=fit["theta"], workers=2)
    low, high = boot["page_z_x_prompt_axis"]["ci95"]
    assert low < fit["page_z_x_prompt_axis"] < high


def test_no_planted_interaction_is_not_detected():
    observations, pages = planted(0.0, seed=4)
    data = ordering.ChoiceData(observations, pages)
    fit = ordering.fit_plackett_luce(data)
    assert abs(fit["page_z_x_prompt_axis"]) < 0.3


def test_unranked_pages_remain_alternatives_and_each_set_has_one_choice():
    data = ordering.ChoiceData([{"keyword": "k", "prompt_id": "q", "prompt_axis": 0.2,
                                 "presented": ["a", "b", "c"], "ranking": ["c", "a"]}], {"a": 0, "b": 1, "c": 2})
    assert data.sets == 2 and len(data.z) == 3 + 2  # {a,b,c} then {a,b}
    assert data.position.tolist() == [0, 1, 2, 0, 1]


def test_extract_reads_presented_snippets_rankings_and_keywords(tmp_path):
    root = dataset(tmp_path, model="llama4")
    writer = FinalDatasetWriter(root, writer_id="keywords")
    writer.append("keyword_memberships", {"prompt_id": "q1", "keyword_ids": ["kw1"], "primary_keyword_id": "kw1",
                                          "primary_priority_rank": 1}, transaction_id="kw-q1")
    writer.seal()
    axis = tmp_path / "final-axis-map.jsonl"
    cli.write_jsonl(axis, [{"candidate_id": "q1", "axis_1_percentile_0_1": 0.3, "consensus_axis_1_z": 0.0}])
    output = tmp_path / "extract"
    assert cli.main(["extract", "--source", f"{root}:llama4", "--final-axis-map", str(axis),
                     "--output", str(output)]) == 0
    observations = cli.read_jsonl(output / "observations.jsonl.gz")
    pages = {p["page_id"]: p for p in cli.read_jsonl(output / "pages.jsonl.gz")}
    assert len(observations) == 3 and {o["keyword"] for o in observations} == {"kw1"}
    expected = {ordering.page_id(ordering.page_text(r["title"], r["text"])): r["url"] for r in (A, B, C)}
    assert {pid: p["urls"] for pid, p in pages.items()} == {pid: [url] for pid, url in expected.items()}
    natural = next(o for o in observations if o["condition"] == "natural")
    assert [expected[p] for p in natural["presented"]] == [A["url"], B["url"]]
    assert [expected[p] for p in natural["ranking"]] == [B["url"], A["url"]]
    assert all(o["prompt_axis"] == 0.3 for o in observations)
    with pytest.raises(ValueError, match="overwrite"):
        cli.main(["extract", "--source", f"{root}:llama4", "--final-axis-map", str(axis), "--output", str(output)])


def test_analyze_end_to_end_writes_coordinates_fits_and_report(tmp_path):
    _, _, bat, final = prompt_archive(tmp_path)
    rng = np.random.default_rng(7)
    page_ids = [f"page{i:03d}" for i in range(60)]
    qwen = projections(tmp_path / "pages-qwen", "map-q", {p: rng.normal(size=2) for p in page_ids}, rng)
    mistral = projections(tmp_path / "pages-mistral", "map-m", {p: rng.normal(size=2) for p in page_ids}, rng)
    rows = cli.aligned(qwen, mistral, bat)
    z = {r["candidate_id"]: (r["reference_axis_1_z"] + r["candidate_aligned_axis_1_z"]) / 2 for r in rows}
    observations = []
    for g in range(400):
        axis = rng.uniform()
        presented = list(rng.choice(page_ids, 6, replace=False))
        utility = np.array([1.2 * z[p] * (axis - 0.5) for p in presented]) + rng.gumbel(size=6)
        observations.append({"generation_id": f"g{g}", "prompt_id": f"q{g}", "keyword": f"k{g % 8}",
                             "model": "llama4", "method": "Parallel-Expansion-v1", "engine": "ddg",
                             "condition": "natural", "prompt_axis": axis, "presented": presented,
                             "ranking": [presented[i] for i in np.argsort(-utility)[:3]]})
    extract = tmp_path / "extract"
    extract.mkdir()
    cli.write_jsonl(extract / "observations.jsonl.gz", observations)
    cli.write_json(extract / "manifest.json", {"stage": "extract"})
    output = tmp_path / "analysis"
    assert cli.main(["analyze", "--extract", str(extract), "--qwen", str(qwen), "--mistral", str(mistral),
                     "--battery", str(bat), "--final-axis-map", str(final), "--bootstrap", "5",
                     "--permutations", "5", "--workers", "1", "--output", str(output)]) == 0
    results = json.loads((output / "results.json").read_text())
    consensus = results["models"]["llama4"]["fits"]["consensus"]
    assert consensus["page_z_x_prompt_axis"] > 0.5
    assert set(results["models"]["llama4"]["fits"]) == set(ordering.VIEWS)
    with gzip.open(output / "page_coordinates.jsonl.gz", "rt") as stream:
        coordinates = [json.loads(line) for line in stream]
    assert len(coordinates) == 60 and all(0 <= c["prompt_scale_percentile_0_1"] <= 1 for c in coordinates)
    report = (output / "report.html").read_text()
    assert "<title>Page Readiness Ordering</title>" in report and "Observational" in report


def test_embed_shards_split_across_workers_resume_and_merge_exactly(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from interpretability.pipeline import readiness_prompt_population as population
    from interpretability.pipeline import two_axis_prompt_population as embedding
    from analysis.scripts import build_readiness_prompt_population as builder

    texts = [f"page text {i}" for i in range(7)]
    pages = tmp_path / "pages.jsonl.gz"
    cli.write_jsonl(pages, ({"page_id": ordering.page_id(t), "text": t, "text_sha256": ordering.page_id(t)}
                            for t in sorted(texts, key=ordering.page_id)))
    (tmp_path / "map").mkdir()
    (tmp_path / "map/readiness_embedding_map.json").write_text("{}")
    embedded = []

    class Embedder:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, batch):
            embedded.extend(batch)
            return np.ones((len(batch), 2))

    def project(fitted, bounds, *, item_ids, text_sha256s, embeddings):
        return [population.ReadinessTextProjection(i, h, 1.0, 2.0, 0.5, 0.5, 0.5) for i, h in zip(item_ids, text_sha256s)]

    monkeypatch.setattr(embedding, "LLM2VecPromptEmbedder", Embedder)
    monkeypatch.setattr(population, "load_readiness_embedding_map", lambda path: SimpleNamespace(map_id="map-q"))
    monkeypatch.setattr(population, "fit_reference_bounds", lambda rows: None)
    monkeypatch.setattr(population, "project_text_embeddings", project)
    monkeypatch.setattr(builder, "_validate_embedding_model_revision", lambda fitted, model: None)
    monkeypatch.setattr(cli, "read_jsonl", lambda path, _read=cli.read_jsonl: [] if path.name.endswith("coordinates.jsonl") else _read(path))
    output = tmp_path / "embed-qwen"
    base = ["embed", "--pages", str(pages), "--view", "qwen", "--map", str(tmp_path / "map"),
            "--embedding-model", "model", "--shard-size", "3", "--workers", "2", "--output", str(output)]
    assert cli.main([*base, "--worker-index", "0"]) == 0
    assert sorted(p.name for p in (output / "shards").iterdir()) == ["00000.jsonl.gz", "00002.jsonl.gz"]
    with pytest.raises(ValueError, match="missing shard 1"):
        cli.main(["merge", "--input", str(output), "--pages", str(pages), "--output", str(tmp_path / "merged")])
    assert cli.main([*base, "--worker-index", "1"]) == 0
    before = len(embedded)
    assert cli.main([*base, "--worker-index", "0"]) == 0 and len(embedded) == before == 7  # resumed, nothing redone
    with pytest.raises(ValueError, match="different settings"):
        cli.main([*base[:-2], "--max-length", "256", "--output", str(output), "--worker-index", "0"])
    merged = tmp_path / "merged"
    assert cli.main(["merge", "--input", str(output), "--pages", str(pages), "--output", str(merged)]) == 0
    rows = cli.read_jsonl(merged / "question_projections.jsonl")
    assert [r["candidate_id"] for r in rows] == [r["page_id"] for r in cli.read_jsonl(pages)]
    assert json.loads((merged / "projection_manifest.json").read_text())["map_id"] == "map-q"
