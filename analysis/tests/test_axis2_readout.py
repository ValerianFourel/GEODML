"""Synthetic tests of analysis/scripts/axis2_readout.py: placement conventions, random directions, and the readout
end to end on a toy world written in the real input formats (no scientific result)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from analysis.scripts import axis2_readout as ax  # noqa: E402
from analysis.steelman import tables as chain_tables  # noqa: E402

D = 12
RUBRICS = ["information_seeking_reverse", "evaluation", "selection_commitment", "action_implementation"]


def basis(seed: int = 0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((D - 1, D - 1)))
    vectors = [np.r_[0.0, q[:, k]] for k in range(D - 1)]
    e0 = np.zeros(D)
    e0[0] = 1.0
    return e0, vectors


def rubric_matrix(a1, a2, evaluation_sign: float):
    v, _ = np.linalg.qr(np.asarray([[0.6, 0.1, 0.5, 0.6], [0.2, -0.9, -0.3, 0.2], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]).T)
    v1, v2 = v[:, 0], v[:, 1]
    v2 = v2 * (np.sign(v2[1]) * evaluation_sign)  # sign of the evaluation entry of a_2's loading
    return 5.0 * np.outer(a1, v1) + 2.0 * np.outer(a2, v2)


def write_map(path: Path, mean, a1, a2, w, map_id):
    path.mkdir(parents=True, exist_ok=True)
    (path / "readiness_embedding_map.json").write_text(json.dumps({
        "embedding_mean": list(mean), "supervised_subspace_axes": [list(a1), list(a2)],
        "rubric_coefficient_matrix": w.tolist(), "rubric_names": RUBRICS, "map_id": map_id}))


def dictionary(values, index):
    return pa.DictionaryArray.from_arrays(pa.array(np.asarray(index, np.int32)), pa.array(list(values), pa.string()))


def test_percentile_rank_breaks_ties_by_id():
    values = np.array([0.5, 0.1, 0.5, 0.9])
    ids = ["d", "a", "b", "c"]
    np.testing.assert_allclose(ax.percentile_rank(values, ids), [2 / 3, 0.0, 1 / 3, 1.0])


def test_sign_follows_the_evaluation_loading():
    e0, (a1, a2, *_) = basis()
    for wanted in (-1.0, 1.0):
        fmap = ax.FrozenMap(e0, np.vstack([a1, a2]), rubric_matrix(a1, a2, wanted), tuple(RUBRICS), "m")
        assert ax.evaluation_sign(fmap) == wanted
        loadings = ax.rubric_loadings(fmap)["axis2"]
        assert np.sign(loadings["evaluation"]) == wanted


def test_random_directions_are_unit_orthogonal_and_seeded():
    _, (a1, a2, *_) = basis()
    axes = np.vstack([a1, a2])
    d = ax.random_directions(axes, 5, 20261012, 0)
    assert d.shape == (5, D)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    assert np.max(np.abs(d @ axes.T)) < 1e-12
    np.testing.assert_array_equal(d, ax.random_directions(axes, 5, 20261012, 0))
    assert not np.allclose(d, ax.random_directions(axes, 5, 20261012, 1))


def test_place_matches_the_alignment_rule():
    rng = np.random.default_rng(3)
    rq, rm = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
    al = ax.Alignment(np.array([[0.999, -0.04], [-0.04, -0.999]]), np.array([0.1, -0.2]), np.array([2.0, 0.5]),
                      np.array([0.0, 0.3]), np.array([1.5, 0.25]))
    placed = ax.place(rq, rm, al, -1.0)
    zq = (rq - al.ref_mean) / al.ref_scale
    zm = ((rm - al.cand_mean) / al.cand_scale) @ al.rotation
    np.testing.assert_allclose(placed.zq, zq * [1, -1])
    np.testing.assert_allclose(placed.zm, zm * [1, -1])
    np.testing.assert_allclose(placed.consensus, (zq + zm) / 2 * [1, -1])


# ---------------------------------------------------------------- toy world in the real formats

def build_world(root: Path, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    e0, (a1, a2, *others) = basis()
    w_qwen = rubric_matrix(a1, a2, -1.0)  # the Qwen view's a_2 loads negatively on evaluation: sign -1
    write_map(root / "maps" / "qwen", e0, a1, a2, w_qwen, "map-q")
    write_map(root / "maps" / "mistral", e0, a1, -a2, w_qwen, "map-m")  # mistral a_2 points the other way
    battery = {"cross_embedding_alignment": {
        "orthogonal_rotation": [[1.0, 0.0], [0.0, -1.0]], "reference_development_mean": [0.0, 0.0],
        "reference_development_scale": [0.05, 0.05], "candidate_development_mean": [0.0, 0.0],
        "candidate_development_scale": [0.05, 0.05], "confirmation_axis_1": {"spearman": 1.0},
        "confirmation_axis_2": {"spearman": 1.0}}}
    (root / "battery.json").write_text(json.dumps(battery))
    keywords = ["kw-a", "kw-b", "kw-c", "kw-d"]
    split = {"kw-a": "held-out", "kw-b": "held-out", "kw-c": "exploration", "kw-d": "held-out"}
    base = {k: 0.4 * sum(rng.normal() * o for o in others[:4]) for k in keywords}

    def embed(kw, t1, t2):
        noise = 0.05 * sum(rng.normal() * o for o in others[4:])
        return 10.0 * e0 + base[kw] + t1 * a1 + t2 * a2 + noise

    prompts = []
    for kw in keywords:
        for j in range(10):
            t1, t2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
            prompts.append({"id": f"p-{kw}-{j:02d}", "keyword": kw, "t1": t1, "t2": t2, "e": embed(kw, t1, t2),
                            "writer": "writer-a" if j % 2 else "writer-b"})
    prompts.append({"id": "p-zz-unmapped", "keyword": None, "t1": 0.0, "t2": 0.0, "e": embed("kw-a", 0.0, 0.0), "writer": None})
    for view in ("qwen", "mistral"):
        shard = root / "prompts" / view / "shard-000"
        shard.mkdir(parents=True)
        np.savez(shard / ax.PROMPT_FILE, candidate_ids=np.asarray([p["id"] for p in prompts]),
                 embeddings=np.asarray([p["e"] for p in prompts], np.float32))
    pages = []
    for kw in keywords:
        for j in range(12):
            s1, s2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
            pages.append({"keyword": kw, "s2": s2, "e": embed(kw, s1, s2), "text": f"{kw} page {j}"})
    corpus = root / "corpus"
    (corpus / "embeddings").mkdir(parents=True)
    vectors = np.asarray([p["e"] for p in pages], np.float32)
    np.save(corpus / "embeddings" / "qwen3-8b-llm2vec.npy", vectors)
    np.save(corpus / "embeddings" / "mistral-7b-instruct-v0.2-llm2vec.npy", vectors)
    maps = {v: ax.load_map(root / "maps" / v / "readiness_embedding_map.json") for v in ax.VIEWS}
    al = ax.load_alignment(root / "battery.json")
    raw = {v: ax.coordinates(vectors, maps[v].mean, maps[v].axes) for v in ax.VIEWS}
    placed_pages = ax.place(raw["qwen"], raw["mistral"], al, 1.0)
    praw = {v: ax.coordinates(np.asarray([p["e"] for p in prompts], np.float32), maps[v].mean, maps[v].axes) for v in ax.VIEWS}
    placed_prompts = ax.place(praw["qwen"], praw["mistral"], al, 1.0)
    ids = [p["id"] for p in prompts]
    x_all = ax.percentile_rank(placed_prompts.consensus[:, 0], ids)
    mapped = [i for i, p in enumerate(prompts) if p["keyword"] is not None]
    scale = ax.prompt_scale(placed_prompts.consensus[:, 0])  # every embedded prompt, as the stored page positions
    u, _ = scale.percentile(placed_pages.consensus[:, 0])
    pq.write_table(pa.table({"row": np.arange(len(pages)), "text": [p["text"] for p in pages],
                             "qwen_raw_axis_1": raw["qwen"][:, 0], "qwen_raw_axis_2": raw["qwen"][:, 1],
                             "mistral_raw_axis_1": raw["mistral"][:, 0], "mistral_raw_axis_2": raw["mistral"][:, 1],
                             "consensus_axis_1_z": placed_pages.consensus[:, 0], "prompt_scale_percentile_0_1": u}),
                   corpus / "snippets.parquet")
    with open(root / "final-axis-map.jsonl", "w") as f:
        for i in mapped:
            f.write(json.dumps({"candidate_id": ids[i], "consensus_axis_1_z": float(placed_prompts.consensus[i, 0]),
                                "axis_1_percentile_0_1": float(x_all[i])}) + "\n")
    with gzip.open(root / "prompts.jsonl.gz", "wt") as f:
        for p in prompts[:-1]:
            f.write(json.dumps({"prompt_id": p["id"], "prompt": {"candidate_id": p["id"], "keyword": p["keyword"],
                                                                 "question": f"question {p['id']}", "generator_id": p["writer"]}}) + "\n")
    # snapshot rows: one per page, then two more rows that repeat pages
    corpus_row = list(range(len(pages))) + [0, 13]
    pq.write_table(pa.table({"row_id": np.arange(len(corpus_row)), "corpus_row": np.asarray(corpus_row)}), root / "rows.parquet")
    by_kw = {kw: [i for i, p in enumerate(pages) if p["keyword"] == kw] for kw in keywords}
    replay_prompts, replay_rows = [], []
    for p in prompts[:-1]:
        for engine in ("duckduckgo", "searxng"):
            replay_prompts.append({"engine": engine, "prompt_id": p["id"]})
            replay_rows.append(rng.choice(by_kw[p["keyword"]], 3, replace=False))
    (root / "replay").mkdir()
    with gzip.open(root / "replay" / "prompts.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in replay_prompts)
    np.savez(root / "replay" / "replay.npz", rows=np.asarray(replay_rows, np.int64))
    # answers: retrieved = the keyword's pages; shown = the 3 closest to the prompt on a_2; cited = the first two shown
    answers = []
    for p in prompts[:-1]:
        for model in ("llama4", "qwen38"):
            for method in ("Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"):
                for engine in ("duckduckgo", "searxng"):
                    for condition in ("natural", "ablated"):
                        pool = by_kw[p["keyword"]]
                        if condition == "natural":
                            close = sorted(pool, key=lambda i: abs(pages[i]["s2"] - p["t2"]))[:3]
                        else:
                            close = list(rng.choice(pool, 3, replace=False))
                        answers.append({"key": (model, p["id"], engine, method, condition), "prompt": p, "pool": pool, "shown": close})
    for a in answers:
        a["answer_id"] = hashlib.sha256("|".join(a["key"]).encode()).hexdigest()[:24]
    funnel = list(range(len(answers)))
    trace = list(rng.permutation(len(answers)))  # the trace order differs from the funnel order
    items = {"who": [], "row": [], "scored": [], "shown": [], "cited": []}
    for f_index in funnel:
        a = answers[f_index]
        for r in a["pool"]:
            items["who"].append(f_index)
            items["row"].append(r)
            items["scored"].append(1)
            slot = a["shown"].index(r) if r in a["shown"] else -1
            items["shown"].append(slot)
            items["cited"].append(slot if 0 <= slot < 2 else -1)
    traces = root / "traces"
    traces.mkdir()
    funnel_ids = [answers[i]["answer_id"] for i in funnel]
    pq.write_table(pa.table({"answer_id": dictionary(funnel_ids, items["who"]), "row_id": np.asarray(items["row"], np.int32),
                             "search_index": np.zeros(len(items["row"]), np.int8), "rank_in_search": np.zeros(len(items["row"]), np.int8),
                             "scored": np.asarray(items["scored"], np.int8), "shown_slot": np.asarray(items["shown"], np.int8),
                             "cited_rank": np.asarray(items["cited"], np.int8)}), traces / "stage_items.parquet", row_group_size=200)
    # chain values with the chain's own functions, as the release export computes them (axis-1 page percentile u)
    offsets = np.r_[0, np.cumsum(np.bincount(items["who"], minlength=len(answers)))]
    u_row = u[np.asarray(corpus_row)]
    chain = chain_tables.stage_values(offsets, np.asarray(items["row"]), np.asarray(items["scored"]), np.asarray(items["shown"]),
                                      np.asarray(items["cited"]), u_row)
    chain["R0"] = chain_tables.replay_values([a["key"][1] for a in answers], [a["key"][2] for a in answers], replay_prompts,
                                             np.asarray(replay_rows), u_row)
    x_of = dict(zip(ids, x_all))
    columns = {"answer_id": [answers[i]["answer_id"] for i in trace],
               "generation_id": [f"g{i}" for i in trace], "model": [answers[i]["key"][0] for i in trace],
               "engine": [answers[i]["key"][2] for i in trace], "method": [answers[i]["key"][3] for i in trace],
               "condition": [answers[i]["key"][4] for i in trace], "prompt_id": [answers[i]["key"][1] for i in trace],
               "keyword": [answers[i]["prompt"]["keyword"] for i in trace],
               "split": [split[answers[i]["prompt"]["keyword"]] for i in trace],
               "x": [x_of[answers[i]["key"][1]] for i in trace]}
    for s in ax.CHAIN:
        columns[f"chain_{s}"] = np.asarray(chain[s])[trace]
    columns["Q"] = np.full(len(trace), np.nan)
    columns["A"] = np.full(len(trace), np.nan)
    pq.write_table(pa.table(columns), traces / "trace_answers.parquet")
    # one agent query per answer (search order 0); its a_2 coordinate follows the prompt's
    pq.write_table(pa.table({"answer_id": dictionary(columns["answer_id"], np.arange(len(trace))),
                             "search_index": np.zeros(len(trace), np.int8), "query_id": np.arange(len(trace), dtype=np.int32)}),
                   traces / "answer_queries.parquet")
    pq.write_table(pa.table({"query_id": np.arange(len(trace), dtype=np.int32),
                             "text_sha256": [f"q-{i}" for i in range(len(trace))]}), traces / "queries.parquet")
    (traces / "manifest.json").write_text("{}")
    # merged projections of queries and answers, and the answers' index (cluster formats)
    t2 = np.asarray([answers[i]["prompt"]["t2"] for i in trace])
    for kind, values, ids_ in (("queries", 0.08 * t2, [f"q-{i}" for i in range(len(trace))]),
                               ("answers", 0.05 * t2, [f"answer-{columns['answer_id'][i]}" for i in range(len(trace))])):
        for view, sign in (("qwen", 1.0), ("mistral", -1.0)):
            out = root / f"embed-{kind}-{view}.merged"
            out.mkdir()
            (out / "projection_manifest.json").write_text("{}")
            with open(out / "question_projections.jsonl", "w") as f:
                for i, (cid, v) in enumerate(zip(ids_, values)):
                    if kind == "answers" and columns["condition"][i] != "natural":
                        continue
                    f.write(json.dumps({"candidate_id": cid, "projection": {
                        "text_sha256": cid if kind == "queries" else f"sha-{cid}", "raw_axis_1": 0.01 * i / len(trace),
                        "raw_axis_2": sign * float(v) + 0.002 * rng.normal()}}) + "\n")
    with gzip.open(root / "observations.jsonl.gz", "wt") as f:
        for i in range(len(trace)):
            if columns["condition"][i] == "natural":
                f.write(json.dumps({"model": columns["model"][i], "prompt_id": columns["prompt_id"][i], "method": columns["method"][i],
                                    "engine": columns["engine"][i], "condition": "natural",
                                    "answer_id": f"answer-{columns['answer_id'][i]}"}) + "\n")
    # calibration archive: evaluation rises with the signed axis 2 (the negative of the Qwen a_2 coordinate)
    cal = root / "calibration"
    for view in ax.VIEWS:
        (cal / "maps" / view).mkdir(parents=True)
    texts = cal / "extra" / "axisgeo-unified-hf-dataset" / "data" / "likert_prompts"
    texts.mkdir(parents=True)
    items_cal = []
    for i in range(80):
        r1, r2 = rng.normal(0, 0.05, size=2)
        items_cal.append({"item_id": f"item-{i:03d}", "split": "confirmation" if i % 2 else "development", "r1": r1, "r2": r2,
                          "evaluation": float(np.clip(4 - 30 * r2 + rng.normal(0, 0.3), 1, 7))})
    for view, sign in (("qwen", 1.0), ("mistral", -1.0)):
        with open(cal / "maps" / view / "readiness_supervised_subspace_coordinates.jsonl", "w") as f:
            for it in items_cal:
                f.write(json.dumps({"item_id": it["item_id"], "split": it["split"], "source_name": "toy", "axis_1": it["r1"],
                                    "axis_2": sign * it["r2"], "consensus_readiness_0_1": 0.5, "usable_for_axis": True}) + "\n")
    with open(cal / "maps" / "qwen" / "readiness_consensus.jsonl", "w") as f:
        for it in items_cal:
            f.write(json.dumps({"item_id": it["item_id"], "overall_readiness_0_100": 50 + 300 * it["r1"],
                                "information_seeking_1_7": 4.0 - 20 * it["r1"], "evaluation_1_7": it["evaluation"],
                                "selection_commitment_1_7": 3.0, "action_implementation_1_7": 4 + 20 * it["r1"]}) + "\n")
    pq.write_table(pa.table({"item_id": [it["item_id"] for it in items_cal], "text": [f"text {it['item_id']}" for it in items_cal]}),
                   texts / "confirmation-00000.parquet")
    return {"prompts": prompts, "t2_by_id": {p["id"]: p["t2"] for p in prompts}}


def test_readout_end_to_end_on_a_toy_world(tmp_path):
    world = build_world(tmp_path)
    assert ax.main(["texts", "--traces", str(tmp_path / "traces"),
                    "--queries-qwen", str(tmp_path / "embed-queries-qwen.merged"), "--queries-mistral", str(tmp_path / "embed-queries-mistral.merged"),
                    "--answers-qwen", str(tmp_path / "embed-answers-qwen.merged"), "--answers-mistral", str(tmp_path / "embed-answers-mistral.merged"),
                    "--answers-index", str(tmp_path / "observations.jsonl.gz"), "--output", str(tmp_path / "texts")]) == 0
    manifest = json.loads((tmp_path / "texts" / "manifest.json").read_text())
    assert manifest["counts"]["queries_missing"] == 0 and manifest["counts"]["answers_joined"] > 0
    common = ["--traces", str(tmp_path / "traces"), "--replay", str(tmp_path / "replay"), "--rows", str(tmp_path / "rows.parquet"),
              "--corpus", str(tmp_path / "corpus"), "--prompt-embeddings", str(tmp_path / "prompts"), "--maps", str(tmp_path / "maps"),
              "--battery", str(tmp_path / "battery.json"), "--final-axis-map", str(tmp_path / "final-axis-map.jsonl"),
              "--prompt-records", str(tmp_path / "prompts.jsonl.gz"), "--calibration", str(tmp_path / "calibration"),
              "--bootstrap", "20", "--permutations", "20", "--random-directions", "3"]
    out = tmp_path / "out"
    assert ax.main(["run", *common, "--text-coordinates", str(tmp_path / "texts" / "text_coordinates.parquet"), "--output", str(out)]) == 0
    with pytest.raises(ValueError, match="refusing to overwrite"):
        ax.main(["run", *common, "--output", str(out)])
    results = json.loads((out / "results.json").read_text())
    checks = results["checks"]
    # sign: the Qwen a_2 loads negatively on evaluation, so axis 2 is flipped; the Mistral view agrees after rotation
    assert checks["axis2_sign"] == -1.0 and checks["mistral_sign_after_rotation_agrees"]
    assert max(checks["random_max_abs_cosine_with_axes"].values()) < 1e-12
    # the readout reproduces the stored axis-1 chain values exactly and recomputes x
    assert max(checks["chain_values_max_abs_difference"].values()) < 1e-12
    assert max(checks["chain_values_nan_mismatch"].values()) == 0
    assert checks["prompts"]["max_abs_x_difference"] == 0.0 and checks["answers_x_max_abs_difference"] == 0.0
    assert checks["pages"]["max_abs_u_difference"] < 1e-12
    # prompt positions: percentiles on [0, 1], ordered against the planted t2 (flipped by the sign)
    import pandas as pd
    table = pd.read_csv(out / "prompts-axis2.csv.gz")
    assert table["x2"].min() == 0.0 and table["x2"].max() == 1.0 and table["x2"].is_unique
    t2 = table["prompt_id"].map(world["t2_by_id"]).to_numpy()
    assert np.corrcoef(table["x2"], -t2)[0, 1] > 0.9
    # calibration: the signed axis 2 rises with the evaluation score
    assert results["calibration"]["rubrics_confirmation"]["axis2_consensus"]["evaluation"] > 0.5
    assert results["calibration"]["battery_check_max_abs_difference"] < 1e-9
    # planted pass-through on axis 2: shortlist and cited sources follow the prompt; retrieval (the whole keyword) does not
    for split in ("heldout", "all"):
        for cell in ("LP", "LR", "QP", "QR"):
            entry = results["readout"][split][cell]["directions"]
            k = entry["axis2"]["percentile"]["raw"]["slopes"]["K"]
            assert k["estimate"] > 0.3 and k["ci95"][0] > 0 and k["permutation_p"] is not None
            assert abs(entry["axis2"]["percentile"]["raw"]["slopes"]["R"]["estimate"]) < 1e-9
            assert entry["axis2"]["percentile"]["controls"]["slopes"]["K"]["estimate"] > 0.3
            assert entry["axis2"]["sd"]["controls"]["slopes"]["Q"]["estimate"] > 0.3
            assert entry["axis2"]["sd"]["controls"]["slopes"]["A"]["estimate"] > 0.3
            assert "Q" not in entry["random_qwen_1"]["sd"]["raw"]["slopes"]  # no embeddings of queries
            assert set(entry["random_mistral_3"]) == {"sd"} and set(entry["random_mistral_3"]["sd"]) == {"raw", "controls"}
            assert set(entry["axis1"]) == {"percentile", "sd"} and set(entry["axis1"]["sd"]) == {"raw"}
    held = results["readout"]["heldout"]["LP"]
    assert held["keywords"] == 3  # kw-c is an exploration keyword; only natural answers enter
    assert held["answers"] == 30 * 2  # 30 held-out prompts x 2 engines
    summary = results["random_summary"]["heldout"]["LP"]["K"]
    assert summary["directions"] == 6
    # outputs: macros, the generated table, the CSV and the face-validity file
    tex = (out / "numbers-axis2.tex").read_text()
    assert "pending" not in tex and "\\begin{table" not in tex
    table_file = (out / "table-axis2.tex").read_text()
    assert "\\label{tab:axis2}" in table_file and "\\AxTwoSlKLP{}" in table_file and "\\AxTwoSlQQR{}" in table_file
    import re
    body = table_file.split("\n", 2)[2]
    assert not re.search(r"[+-]0\.\d", body)  # no number of its own: every cell is a macro
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
    assert set(re.findall(r"\\((?:AxOne|AxTwo|RndSd|NAxTwo)[A-Za-z]*)", body)) <= defined
    assert len((out / "stages-axis2.csv").read_text().splitlines()) > 100
    for macro in ("\\AxTwoSlKLP", "\\AxTwoCISlKQR", "\\AxTwoViewPrompts", "\\AxTwoRhoEval",
                  "\\AxTwoKwRangeShare", "\\AxTwoKwRangeHalfShare", "\\AxTwoWriterR", "\\RndSdMedKLP", "\\AxTwoSlQLP",
                  "\\AxTwoViewAnswers", "\\AxTwoSlKMin", "\\AxTwoSlKMax", "\\RndSdMedKMin", "\\RndSdTopK", "\\AxTwoAboveRandom",
                  "\\AxOneWithinRandom", "\\NAxTwoStageCells", "\\AxOneSdKMax", "\\AxOneAboveRandomMedianAtK", "\\NAxTwoCells"):
        assert macro in tex, macro
    spec = results["specificity"]["heldout"]
    assert spec["pairs"] == 4 * len(ax.TABLE_STAGES) and 0 <= spec["axis2_controls_above_all_random"] <= spec["pairs"]
    assert set(results["prompt_diagnostics"]["all"]["spread"]["x2"]["share_range_at_least"]) == {"0.3", "0.5"}
    # the macros subcommand rewrites the same file from results.json and refuses to overwrite
    again = tmp_path / "again.tex"
    assert ax.main(["macros", "--results", str(out / "results.json"), "--output", str(again), "--table", str(tmp_path / "table.tex")]) == 0
    assert again.read_text().splitlines()[2:] == tex.splitlines()[2:]
    table = (tmp_path / "table.tex").read_text()
    assert table.count("\\begin{table*}") == 1 and "\\label{tab:axis2}" in table and "\\newcommand" not in table
    assert table.splitlines()[2:] == table_file.splitlines()[2:]
    assert "\\AxTwoBinFirstHigh" in tex and "\\AxTwoBinLastLow" in tex
    with pytest.raises(ValueError, match="refusing to overwrite"):
        ax.main(["macros", "--results", str(out / "results.json"), "--output", str(again)])
    face = (out / "face-validity.txt").read_text()
    assert "15 prompts highest on axis 2" in face and "calibration items" in face and "nearly equal x" in face


def test_run_without_queries_and_answers_reads_pages_only(tmp_path):
    build_world(tmp_path, seed=11)
    out = tmp_path / "out"
    assert ax.main(["run", "--traces", str(tmp_path / "traces"), "--replay", str(tmp_path / "replay"), "--rows", str(tmp_path / "rows.parquet"),
                    "--corpus", str(tmp_path / "corpus"), "--prompt-embeddings", str(tmp_path / "prompts"), "--maps", str(tmp_path / "maps"),
                    "--battery", str(tmp_path / "battery.json"), "--final-axis-map", str(tmp_path / "final-axis-map.jsonl"),
                    "--bootstrap", "10", "--permutations", "0", "--random-directions", "2", "--output", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    slopes = results["readout"]["heldout"]["QR"]["directions"]["axis2"]["percentile"]["controls"]["slopes"]
    assert "Q" not in slopes and "A" not in slopes and "K" in slopes
    assert results["calibration"] is None and "queries" not in results["positions"]
    assert "writer" not in results["prompt_diagnostics"]["all"]
    tex = (out / "numbers-axis2.tex").read_text()
    assert "\\AxTwoSlKQR" in tex and "\\AxTwoSlQQR" not in tex and "AxTwoViewCalib" not in tex


def test_horeka_runner_parses_and_stays_on_cpu_inside_its_allocation():
    import subprocess
    script = Path(__file__).resolve().parents[1] / "docs" / "horeka-axis2-readout.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = "\n".join(line for line in script.read_text().splitlines() if not line.lstrip().startswith("#"))
    for banned in ("cpuonly", "scancel", "sbatch", "salloc", "--gres=gpu"):
        assert banned not in text, banned
    assert "axis2_readout texts" in text and "axis2_readout run" in text and "refusing to overwrite" in text
