"""Local funnel report: every section renders from synthetic inputs, numbers land in their own columns,
figures are embedded, CSV tables are written, and nothing is overwritten."""

import csv
import json
import struct
import zlib

import pytest

from analysis.scripts import funnel_local_report as report

STRATA = list(report.STRATA) + list(report.ENGINE_STRATA)


def entry(slope):
    return {"n": 100, "mean": 0.4, "slope": slope, "ci95": [slope - 0.01, slope + 0.01], "low_x_mean": 0.38, "high_x_mean": 0.45}


def curves(terms):
    return {"bin_lower": [0.0, 0.5], "bin_upper": [0.5, 1.0],
            "terms": {t: {"mean": [0.38, 0.45], "se_keyword_cluster": [0.01, 0.01], "n": [50, 50]} for t in terms}}


def png(path):
    raw = b"\x00\xff\xff\xff"
    chunks = [(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)), (b"IDAT", zlib.compress(raw)), (b"IEND", b"")]
    data = b"\x89PNG\r\n\x1a\n" + b"".join(struct.pack(">I", len(c)) + t + c + struct.pack(">I", zlib.crc32(t + c)) for t, c in chunks)
    path.write_bytes(data)


@pytest.fixture
def inputs(tmp_path):
    metrics = ["ranking_len", "search_count", "cited_u", "cited_dfs_organic_count", "alignment_gain", "cited_on_topic", "cited_glued", "r0_u"]
    review = {"answers": 1200, "answers_by_model": {"llama4": 600, "qwen38": 600}, "natural_answers": 400,
              "slopes": {s: {m: entry(0.05) for m in metrics} for s in STRATA}, "curves": {s: curves(metrics) for s in STRATA}}
    (tmp_path / "review").mkdir()
    (tmp_path / "review" / "review.json").write_text(json.dumps(review))
    stage_terms = ["R_u", "P_u", "K_u", "R0_u", "R_offtopic", "q_count"]
    explore = {"answers": 300, "natural": 100, "keywords": 7, "slopes": {s: {m: entry(0.02) for m in stage_terms} for s in STRATA[2:4]},
               "curves": {s: curves(stage_terms) for s in STRATA[2:4]}}
    (tmp_path / "explore").mkdir()
    (tmp_path / "explore" / "explore.json").write_text(json.dumps(explore))
    feature = {"block": "C1", "beta_per_sd": 0.1, "odds_ratio_per_sd": 1.105, "ci95": [0.05, 0.15], "permutation_p": None}
    stage = {"features": {"dfs_organic_count": feature, "intent_alignment": {**feature, "block": "A1", "permutation_p": 0.01}},
             "blocks": {"A1": {"fit_share": 0.4}, "C1": {"fit_share": 0.6}}}
    steps = {"retrieval|U": 0.1, "reranker_candidate|R": 0.0, "shortlist|C": 0.5, "ranking|P": 0.1}
    funnel = {"models": {"main": {s: {st: stage for st in ("R|U", "R0|U", "P|C", "K|P")} for s in STRATA[2:4]}},
              "decomposition": {s: {"topic_similarity": {"answers": 10, "log_rr": steps, "ci95": {k: [v - 0.1, v + 0.1] for k, v in steps.items()},
                                                         "share": steps, "total_log_rr_K_given_U": 0.7, "ci95_total": [0.5, 0.9]}} for s in STRATA[2:4]},
              "negative_control_null": {s: {st: {"features": 3, "quantile": 0.05} for st in ("R|U", "R0|U", "P|C", "K|P")} for s in STRATA[2:4]},
              "contrasts": {"R − R0": {STRATA[2]: {"estimate": 0.02, "ci95": [0.01, 0.03]}}},
              "assembled": {"counts": {"u_items": 10, "candidates": 20, "presented": 5}}}
    (tmp_path / "funnel").mkdir()
    (tmp_path / "funnel" / "results.json").write_text(json.dumps(funnel))
    contrast = {"slope_a": 0.09, "slope_b": 0.06, "difference": 0.03, "ci95": [0.02, 0.04], "answers_a": 10, "answers_b": 10}
    (tmp_path / "contrasts").mkdir()
    (tmp_path / "contrasts" / "contrasts.json").write_text(json.dumps({"answers": 400, "bootstrap": 20,
                                                                        "contrasts": {"Llama − Qwen · Parallel": {"cited_u": contrast}}}))
    (tmp_path / "contrasts" / "contrasts.csv").write_text("contrast,measure\nLlama − Qwen · Parallel,cited_u\n")
    kw_row = {"keyword": "crm", "slope_shrunk": 0.1, "slope_shrunk_lo95": 0.05, "slope_shrunk_hi95": 0.15, "kw_main_intent": "commercial"}
    summary = {"keywords": 2, "keywords_with_slope": 2, "pooled_mean_slope": 0.06, "pooled_mean_slope_ci95": [0.05, 0.07],
               "fixed_effect_mean_slope": 0.04, "raw_slope_mean_by_se_quintile": [0.02, 0.1], "between_keyword_variance_tau2": 0.001,
               "i2": 0.8, "heterogeneity_q": 50.0, "heterogeneity_df": 1, "heterogeneity_permutation_p": 0.05, "permutations": 19,
               "null_q_mean": 1.1, "null_q_max": 9.0, "share_keywords_shrunk_interval_above_0": 0.5, "share_keywords_shrunk_interval_below_0": 0.0,
               "by_intent_class": {"commercial": {"keywords": 2, "answers": 9, "slope": 0.06, "ci95": [0.05, 0.07]}},
               "by_difficulty_tercile": {"0": {"keywords": 1, "answers": 4, "slope": 0.05, "ci95": [0.04, 0.06]}},
               "correlations_with_shrunk_slope": {"kw_cpc": 0.1}, "top20": [kw_row], "bottom20": [kw_row], "gemma": {"supplied": False}}
    (tmp_path / "keywords").mkdir()
    (tmp_path / "keywords" / "keywords-summary.json").write_text(json.dumps(summary))
    with open(tmp_path / "keywords" / "keywords.csv", "w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(kw_row))
        w.writeheader()
        w.writerow(kw_row)
    coverage = {"by_engine": {"duckduckgo": {"dfs_domain": 0.956, "google_top20_domain": 0.332, "google_top20_url": 0.255, "html_usable": 0.784,
                                             "llms_txt": 0.921, "open_pagerank": 0.56, "rows": 10227}},
                "validation_against_experiment1": {"body_word_count": {"n": 8084, "spearman": 0.993}}}
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "coverage.json").write_text(json.dumps(coverage))
    (tmp_path / "figures").mkdir()
    png(tmp_path / "figures" / "fig5-funnel-odds-ratios.png")
    return tmp_path


def test_report_renders_every_section_with_aligned_columns_figures_and_tables(inputs):
    out = inputs / "out" / "report.html"
    out.parent.mkdir()
    args = ["--review", str(inputs / "review"), "--explore", str(inputs / "explore"), "--funnel", str(inputs / "funnel"),
            "--contrasts", str(inputs / "contrasts"), "--keywords", str(inputs / "keywords"), "--features", str(inputs / "features"),
            "--figures", str(inputs / "figures"), "--commit", "abcdef0", "--date", "2026-10-07", "--output", str(out)]
    assert report.main(args) == 0
    page = out.read_text()
    order = ["data", "behaviour", "cited", "stages", "seo", "decomposition", "contrasts", "keywords", "gemma", "limits", "files"]
    positions = [page.index(f'<section id="{sid}">') for sid in order]
    assert positions == sorted(positions)
    head = page[page.index("Feature coverage"):]
    cells = head[head.index("<td>DuckDuckGo</td>"):head.index("</tr>", head.index("<td>DuckDuckGo</td>"))]
    assert cells.index("78.4%") < cells.index("95.6%") < cells.index("56.0%") < cells.index("92.1%") < cells.index("25.5%") < cells.index("33.2%")
    assert page.count("data:image/png;base64,") == 1  # only the figure that exists is embedded
    tables = out.with_name("report-tables")
    assert {"review_slopes.csv", "stage_slopes.csv", "stage_models.csv", "block_fit_shares.csv", "decomposition.csv",
            "contrasts.csv", "keywords.csv"} <= {p.name for p in tables.iterdir()}
    bundle = json.loads(out.with_name("report-results.json").read_text())
    assert bundle["exploratory"] is True and bundle["commit"] == "abcdef0"
    with pytest.raises(ValueError, match="overwrite"):
        report.main(args)
