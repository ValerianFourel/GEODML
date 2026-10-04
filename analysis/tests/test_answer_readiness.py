"""Answer-text readiness stages on synthetic data (no scientific result)."""
import hashlib
import json

import pytest

from analysis.scripts import answer_readiness as answers
from analysis.tests.test_source_importance_pipeline import ANSWER, dataset


def axis_map(tmp_path, rows):
    path = tmp_path / "final-axis-map.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_export_keeps_natural_answers_deduplicated_in_embed_page_format(tmp_path):
    roots = {m: dataset(tmp_path / m, model=m) for m in ("qwen38", "llama4")}
    path = axis_map(tmp_path, [{"candidate_id": "q1", answers.AXIS: 0.3}])
    manifest = answers.export(roots, path, tmp_path / "export")
    assert manifest["observations"] == 2 and manifest["unique_answers"] == 1  # both models wrote ANSWER
    _, texts, observations = answers.load_export(tmp_path / "export")
    page = answers.read_jsonl(tmp_path / "export/answers.jsonl.gz")[0]
    assert page["text"] == ANSWER and page["text_sha256"] == hashlib.sha256(ANSWER.encode()).hexdigest()
    assert {o["condition"] for o in observations} == {"natural"}
    assert all(o[answers.AXIS] == 0.3 and texts[o["answer_id"]] == ANSWER for o in observations)
    with pytest.raises(ValueError, match="already exists"):
        answers.export(roots, path, tmp_path / "export")
    report = tmp_path / "text"
    answers.text_report(tmp_path / "export", report, permutations=5)
    assert "# How the answer text changes" in (report / "report.md").read_text()
    (tmp_path / "export/observations.jsonl.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="export file changed"):
        answers.load_export(tmp_path / "export")


def test_text_features_count_markers_per_100_words():
    features = answers.text_features("You should buy it now. Visit www.shop.com today!\n1. Order online for $20.")
    # you should buy it now visit www shop com today order online for
    assert features["words"] == 13 and features["sentences"] == 3
    assert features["list_lines"] == 1 and features["url_mentions"] == 1
    assert features["second_person_per_100w"] == pytest.approx(100 / 13)
    assert features["immediacy_per_100w"] == pytest.approx(200 / 13)  # now, today
    assert features["currency_per_100w"] == pytest.approx(100 / 13)
    assert answers.text_features("")["words_per_sentence"] is None


def synthetic_rows():
    rows, texts = [], {}
    for keyword in range(4):
        for index, x in enumerate((0.1, 0.2, 0.8, 0.9)):
            text = ("buy now you order " * 30) if x > 0.5 else ("history overview because means " * 30)
            identity = answers.answer_id(f"{keyword}-{index}-{text}")
            texts[identity] = text
            rows.append({"model": "llama4", "method": "M", "engine": "e", "prompt_id": f"k{keyword}p{index}",
                         "keyword_id": f"k{keyword}", answers.AXIS: x, "answer_id": identity,
                         **answers.text_features(text)})
    return rows, texts


def test_within_keyword_contrast_and_distinctive_words_hold_topic_fixed():
    rows, texts = synthetic_rows()
    contrast = {r["outcome"]: r for r in answers.within_keyword_contrasts(rows, answers.FEATURES, ("model",))}
    assert contrast["action_verbs_per_100w"]["keywords"] == 4
    assert contrast["action_verbs_per_100w"]["mean_high_minus_low"] == pytest.approx(50.0)
    assert contrast["action_verbs_per_100w"]["share_keywords_higher"] == 1.0
    high, low = answers.distinctive_words(rows, texts, model="llama4")
    assert {r["word"] for r in high[:4]} == {"buy", "now", "you", "order"}
    assert {r["word"] for r in low[:4]} == {"history", "overview", "because", "means"}
    assert high[0]["z"] > 0 > low[0]["z"]


def test_analyze_places_answers_on_the_prompt_scale(tmp_path, monkeypatch):
    rows, texts = synthetic_rows()
    export = tmp_path / "export"
    export.mkdir()
    answers.write_jsonl(export / "answers.jsonl.gz", ({"page_id": i, "text": t, "text_sha256": "x"}
                                                       for i, t in texts.items()))
    answers.write_jsonl(export / "observations.jsonl.gz",
                        ({k: r[k] for k in ("model", "method", "engine", "prompt_id", "keyword_id", answers.AXIS,
                                            "answer_id")} for r in rows))
    (export / "manifest.json").write_text(json.dumps({"files": {
        name: answers.sha256_file(export / name) for name in ("answers.jsonl.gz", "observations.jsonl.gz")}}))
    # Answers to higher-axis prompts get higher z in both views.
    z = {r["answer_id"]: r[answers.AXIS] * 2 - 1 for r in rows}
    from analysis.scripts import page_readiness_ordering
    monkeypatch.setattr(page_readiness_ordering, "aligned", lambda *a: [
        {"candidate_id": i, "reference_axis_1_z": v, "candidate_aligned_axis_1_z": v} for i, v in z.items()])
    path = axis_map(tmp_path, [{"candidate_id": f"p{i}", "consensus_axis_1_z": -1 + i / 50} for i in range(101)])
    tables = answers.analyze(export, tmp_path / "q", tmp_path / "m", tmp_path / "b", path, tmp_path / "out",
                             permutations=20)
    model = next(r for r in tables["associations"] if r["scope"] == "all answers of the model"
                 and r["outcome"] == "answer_axis_percentile")
    assert model["spearman_rho"] == pytest.approx(1.0) and model["n"] == 16
    placed = {r["answer_id"]: r for r in answers.read_jsonl(tmp_path / "out/answer_coordinates.jsonl.gz")}
    assert all(abs(p["answer_axis_percentile"] - (p["consensus_axis_1_z"] + 1) / 2) < 1e-9 for p in placed.values())
    monkeypatch.setattr(page_readiness_ordering, "aligned", lambda *a: [])
    with pytest.raises(ValueError, match="lack projections"):
        answers.analyze(export, tmp_path / "q", tmp_path / "m", tmp_path / "b", path, tmp_path / "out2")
