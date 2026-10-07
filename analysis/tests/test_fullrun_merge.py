"""Keyword-hash shards of both extraction stages merge into exactly the unsharded data (two models, real traces)."""

import gzip
import json

import numpy as np

from analysis.fullrun import merge
from analysis.scripts import funnel_study as study
from analysis.scripts import intent_stages_study as intent
from analysis.tests.test_funnel_study import pipeline, snapshot_args  # noqa: F401


def per_answer(folder):
    answers = [json.loads(line) for line in gzip.open(folder / "answers.jsonl.gz", "rt")]
    it = np.load(folder / "items.npz")
    ev = np.load(folder / "events.npz")
    out = {}
    for i, a in enumerate(answers):
        lo, hi = it["offsets"][i], it["offsets"][i + 1]
        items = tuple(map(tuple, np.column_stack([it[k][lo:hi] for k in ("row", "search", "rank", "scored", "presented", "ranked")]).tolist()))
        events = []
        for e in np.flatnonzero(ev["answer"] == i):
            s, t = ev["offsets"][e], ev["offsets"][e + 1]
            events.append((int(ev["search"][e]), tuple(ev["row"][s:t].tolist()), tuple(np.round(ev["score"][s:t], 12).tolist()),
                           tuple(ev["selected"][s:t].tolist())))
        out[(a["model"], a["prompt_id"], a["engine"], a["method"], a["condition"])] = (a["x"], items, tuple(events))
    return out


def test_funnel_extract_shards_merge_to_the_unsharded_extract(pipeline, tmp_path):  # noqa: F811
    inputs = pipeline["inputs"]
    shards = []
    for k in (1, 2, 3):
        out = tmp_path / f"shard{k}"
        assert study.main(["extract", "--source", f"{inputs['root']}:llama4", "--source", f"{inputs['root']}:qwen38",
                           *snapshot_args(pipeline["tmp"]), "--final-axis-map", str(inputs["final"]),
                           "--population-prompts", str(inputs["population"]), "--workers", "1", "--prompt-shard", f"{k}/3",
                           "--output", str(out)]) == 0
        shards.append(out)
    # the last shard is given twice: its cells must be dropped as duplicates
    summary = merge.merge_funnel_extract(shards + [shards[-1]], tmp_path / "merged")
    assert summary["duplicate_cells_dropped"] == json.loads((shards[-1] / "manifest.json").read_text())["counts"]["answers"]
    assert per_answer(tmp_path / "merged") == per_answer(pipeline["extract"])


def test_trace_extract_shards_merge_to_the_unsharded_trace_extract(pipeline, tmp_path, monkeypatch):  # noqa: F811
    inputs = pipeline["inputs"]
    common = ["--source", f"{inputs['root']}:llama4", "--source", f"{inputs['root']}:qwen38", "--final-axis-map", str(inputs["final"]),
              "--corpus-package", str(inputs["corpus"]), "--population-prompts", str(inputs["population"]), "--workers", "1"]
    assert intent.main(["trace-extract", *common, "--output", str(tmp_path / "full")]) == 0
    shards = []
    for k in (1, 2):
        assert intent.main(["trace-extract", *common, "--prompt-shard", f"{k}/2", "--output", str(tmp_path / f"t{k}")]) == 0
        shards.append(tmp_path / f"t{k}")
    merge.merge_trace_extract(shards, tmp_path / "merged")

    def view(folder):
        st = np.load(folder / "stages.npz")
        codes = json.loads((folder / "codes.json").read_text())
        gen = [json.loads(line)["generation_id"] for line in gzip.open(folder / "answers.jsonl.gz", "rt")]
        q = [json.loads(line)["text"] for line in gzip.open(folder / "queries.jsonl.gz", "rt")]
        out = {}
        for i, g in enumerate(gen):
            entry = [tuple(codes[k][st[k][i]] for k in merge.STAGE_CODES), float(st["x"][i])]
            for n in merge.CSR:
                o = st[f"{n}_offsets"]
                entry.append(tuple(st[f"{n}_docs"][o[i]:o[i + 1]].tolist()))
            entry.append(tuple(q[j] for j in st["q_ids"][st["q_offsets"][i]:st["q_offsets"][i + 1]]))
            evs = np.flatnonzero(st["event_answer"] == i)
            ro = st["event_row_offsets"]
            entry.append(tuple((tuple(st["event_doc"][ro[e]:ro[e + 1]].tolist()), tuple(st["event_selected"][ro[e]:ro[e + 1]].tolist()))
                               for e in evs))
            out[g] = tuple(entry)
        return out

    assert view(tmp_path / "merged") == view(tmp_path / "full")
    full = json.loads((tmp_path / "full" / "manifest.json").read_text())["counts"]
    merged = json.loads((tmp_path / "merged" / "manifest.json").read_text())["counts"]
    assert merged["answers"] == full["answers"] and merged["unique_queries"] == full["unique_queries"]
