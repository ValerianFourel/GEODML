"""Local exploratory review helpers: per-keyword slopes and shrinkage, query lexicon, per-answer metrics."""

import numpy as np
import pytest

from analysis.scripts import funnel_explore as explore
from analysis.scripts import funnel_keywords as kw
from analysis.scripts import funnel_review as review


def test_keyword_slopes_match_a_direct_regression():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 60)
    k = np.repeat([0, 1, 2], 20)
    y = np.r_[0.1 * x[:20], -0.2 * x[20:40], 0.3 * x[40:]] + rng.normal(0, 0.01, 60)
    slope, se, n = kw.keyword_slopes(x, y, k, 4)
    for j, start in enumerate((0, 20, 40)):
        b, a = np.polyfit(x[start:start + 20], y[start:start + 20], 1)
        assert slope[j] == pytest.approx(b, rel=1e-9) and se[j] > 0 and n[j] == 20
    assert np.isnan(slope[3]) and n[3] == 0


def test_shrinkage_pulls_noisy_keywords_and_detects_heterogeneity():
    rng = np.random.default_rng(1)
    same = 0.08 + rng.normal(0, 0.03, 400)          # identical true slopes, sampling noise only
    eb = kw.shrink(same, np.full(400, 0.03))
    assert eb["tau2"] < 2e-4 and abs(eb["mu"] - 0.08) < 0.01 and np.all(eb["shrinkage"] < 0.5)
    true = rng.normal(0.08, 0.1, 400)               # real between-keyword spread
    observed = true + rng.normal(0, 0.03, 400)
    eb2 = kw.shrink(observed, np.full(400, 0.03))
    assert 0.006 < eb2["tau2"] < 0.014 and eb2["i2"] > 0.8
    noisy = np.full(400, 0.03)
    noisy[0] = 0.3
    eb3 = kw.shrink(observed, noisy)
    assert abs(eb3["mean"][0] - eb3["mu"]) < abs(observed[0] - eb3["mu"])  # a noisy keyword is pulled to the mean
    se_mixed = np.r_[np.full(200, 0.01), np.full(200, 0.06)]            # precise keywords have smaller slopes
    slopes = np.r_[rng.normal(0.02, 0.03, 200), rng.normal(0.12, 0.07, 200)]
    eb4 = kw.shrink(slopes, se_mixed)
    wr = 1 / (se_mixed ** 2 + eb4["tau2"])
    assert eb4["mu"] == pytest.approx(np.sum(wr * slopes) / np.sum(wr))  # the centre is the random-effects mean
    assert eb4["fixed_effect_mean"] < eb4["mu"] and eb4["fixed_effect_mean"] < 0.04


def test_query_lexicon_metrics():
    m = explore.query_metrics(["buy crm software price", "crm software guide", "crm software guide"],
                              "Where can I buy CRM software today?", "crm software")
    assert m["q_count"] == 3 and m["q_keyword_inclusion"] == 1.0 and m["q_distinct_share"] == pytest.approx(2 / 3)
    assert m["q_action_share"] > 0 and m["q_information_share"] > 0 and 0 < m["q_prompt_reuse"] <= 1
    assert explore.query_metrics([], "p", "k") == {}


def test_answer_metrics_weight_ranks_and_compare_with_the_keyword_rows():
    table = {"https://a.com/1": {"u": 0.9, "glued": 0.0, "domain": "a.com", "url_normalized": "https://a.com/1",
                                 **{f: 1.0 for f in review.URL_FEATURES + review.DOMAIN_FEATURES}},
             "https://b.com/2": {"u": 0.1, "glued": 1.0, "domain": "b.com", "url_normalized": "https://b.com/2",
                                 **{f: 0.0 for f in review.URL_FEATURES + review.DOMAIN_FEATURES}}}
    t = {"url": table, "keyword_rows": {("searxng", "crm"): {"https://a.com/1"}}, "keyword_u": {("searxng", "crm"): [0.2, 0.4]},
         "google_url": {("crm", "https://a.com/1")}, "google_domain": set(), "r0": {("p1", "searxng"): ({"https://b.com/2"}, 0.3)}}
    prompts = {"p1": ("crm", 0.8, "buy crm now")}
    row = {"prompt_id": "p1", "engine": "searxng", "method": "Parallel-Expansion-v1", "condition": "natural",
           "ranking": ["https://a.com/1", "https://b.com/2", "https://unknown.example"], "search_count": 3,
           "final_snippet_count": 7, "answer": "1) Go to the site. 2) Click sign up. It costs $5."}
    m = review.answer_metrics("llama4", row, prompts, t)
    w = np.array([1.0, 1 / np.log2(3)])
    assert m["cited_u"] == pytest.approx((w[0] * 0.9 + w[1] * 0.1) / w.sum())
    assert m["cited_on_topic"] == pytest.approx(w[0] / w.sum()) and m["cited_google_url"] == pytest.approx(w[0] / w.sum())
    assert m["top1_gap"] == pytest.approx(0.1) and m["null_gap"] == pytest.approx(np.mean([0.6, 0.4]))
    assert m["r0_overlap"] == pytest.approx(1 / 3) and m["r0_u"] == 0.3
    assert m["answer_steps"] == 1 and m["answer_imperatives"] >= 2 and m["answer_currency"] == 1 and m["ranking_len"] == 3


def test_merge_keeps_folder_order_and_drops_republished_cells(tmp_path, monkeypatch):
    import gzip
    import json
    from types import SimpleNamespace
    from analysis.scripts import funnel_local as local

    def record(model, prompt, rows):
        return {"fingerprint": f"{model}-{prompt}", "model": model, "engine": "searxng", "method": "Parallel-Expansion-v1",
                "condition": "natural", "prompt_id": prompt, "keyword_text": "crm", "x": 0.5,
                "items": [[r, 0, i, 1, i, -1] for i, r in enumerate(rows)], "events": [[0, [[rows[0], 0.9, 1]]]], "queries": ["crm"]}

    def chunk(folder, name, model, records):
        folder.mkdir(exist_ok=True)
        with gzip.open(folder / f"{name}.json.gz", "wt") as stream:
            json.dump({"bundle": name, "model": model, "records": records, "counts": {"seen": len(records)}}, stream)

    qwen, llama = tmp_path / "qwen.chunks", tmp_path / "llama.chunks"
    chunk(qwen, "b", "qwen38", [record("qwen38", "p1", [1, 2])])
    chunk(qwen, "a", "qwen38", [record("qwen38", "p2", [3])])
    chunk(llama, "c", "llama4", [record("llama4", "p1", [4, 5, 6]), record("llama4", "p1", [7])])  # republished cell
    monkeypatch.setattr(local.fr, "row_table_digest", lambda rows: "digest")
    out = tmp_path / "merged"
    local.merge_chunks([qwen, llama], out, SimpleNamespace(snapshot_sha256={"searxng": "s"}))
    answers = [json.loads(line) for line in gzip.open(out / "answers.jsonl.gz", "rt")]
    assert [a["prompt_id"] + a["model"] for a in answers] == ["p2qwen38", "p1qwen38", "p1llama4"]
    items = np.load(out / "items.npz")
    assert items["offsets"].tolist() == [0, 1, 3, 6] and items["row"].tolist() == [3, 1, 2, 4, 5, 6]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"]["duplicate_cells"] == 1 and manifest["answers_by_model"] == {"qwen38": 2, "llama4": 1}
    assert manifest["chunks"] == 3 and manifest["chunk_dirs"] == ["qwen.chunks", "llama.chunks"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        local.merge_chunks([qwen], out, SimpleNamespace(snapshot_sha256={}))  # never overwrites


def test_gemma_support_joins_on_the_generation_fingerprint_and_reports_the_join(tmp_path):
    import gzip
    import json
    fp = ["a" * 64, "b" * 64, "c" * 64]
    rows = [{"record_id": f"generation-{fp[0]}-g1", "row": {"prompt_id": "p1", "condition": "natural"}},
            {"record_id": f"generation-{fp[1]}-g1", "row": {"prompt_id": "p2", "condition": "natural"}},
            {"record_id": f"generation-{fp[2]}-g1", "row": {"prompt_id": "p1", "condition": "shuffled"}}]
    (tmp_path / "objects").mkdir()
    (tmp_path / "objects" / "gen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "generation-objects.json").write_text(json.dumps({"objects/gen.jsonl": "meta-llama/Llama-4-Scout-17B-16E-Instruct"}))
    gemma = tmp_path / "gemma"
    gemma.mkdir()
    grades = [{"fingerprint": fp[0], "doc": 1, "grade": 3}, {"fingerprint": fp[0], "doc": 2, "grade": 1},
              {"fingerprint": fp[2], "doc": 1, "grade": 5}, {"fingerprint": "d" * 64, "doc": 1, "grade": 4}]
    with gzip.open(gemma / "grades.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(g) + "\n" for g in grades))
    support, info = kw.gemma_support(gemma, tmp_path, {"p1": "crm", "p2": "erp"})
    assert support == {"crm": 2.0}  # natural answers only; the shuffled cell's grade is not used
    assert info["graded_answers"] == 3 and info["graded_answers_joined"] == 2 and info["graded_answers_unjoined"] == 1
    assert info["natural_answers_with_grades"] == 1 and info["natural_answers_seen"] == 2
