#!/usr/bin/env python3
"""Local (Mac, CPU) exploratory review for the funnel study (addendum A1 of funnel_study.md).

  extract-hf  stream trace objects of the published Hugging Face bundles, keep answers of the
              exploration keywords, map every stage to exact snapshot rows (funnel_rows.answer_items)
              and write the same extract folder as ``funnel_study.py extract``. Each trace file is
              deleted after use; finished bundles are skipped on rerun. ``--model`` keeps one model's
              bundles (the trace budget then counts only those).
  merge       one extract folder from several chunk folders (e.g. a Qwen and a Llama extraction).
  review      analyses of the published generation rows (all keywords): behaviour, cited-source intent,
              on-topic share, cited-source SEO and page features, alignment against the keyword's own
              rows, prompt-text search overlap, cross-model agreement, ranking change, moderators.
Everything here is exploratory. No GPU is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline import funnel_rows as fr  # noqa: E402
from analysis.scripts import funnel_study as study  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

REPO = "ValerianFourel/geodml-experiment-v2-paper-private"
SHORT = {"meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4", "Qwen/Qwen3.8-27B": "qwen38"}


def load_prompts(path: Path) -> dict:
    """prompt id -> (keyword text, axis percentile, question) from the archived prompt snapshot."""
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            out[r["prompt"]["candidate_id"]] = (r["prompt"]["keyword"], float(r["axis"]["axis_1_percentile_0_1"]), r["prompt"]["question"])
    return out


# ---------------------------------------------------------------- extract-hf

def extract_hf(args) -> int:
    from huggingface_hub import hf_hub_download
    hf = Path(args.hf_root)
    names = json.loads((hf / "object-index.json").read_text())
    order = json.loads((hf / "bundles-by-exploration-density.json").read_text())
    prompts = load_prompts(args.prompts)
    split = {k: study.exploration_keyword(k) for k, _, _ in prompts.values()}
    rows = fr.snapshot_rows(dict(s.split("=", 1) for s in args.snapshot))
    lookup = rows.lookup()
    indices = {e: fr.LexicalIndex(rows, e) for e in rows.engine_offset}
    prompt_axis = {p: v[1] for p, v in prompts.items()}
    out = Path(args.output)
    chunks = out.with_name(out.name + ".chunks")
    chunks.mkdir(parents=True, exist_ok=True)
    spent, done = 0.0, 0
    for bundle_path in order:
        bundle_path = hf / bundle_path if not Path(bundle_path).is_absolute() else Path(bundle_path)
        bundle = json.loads(bundle_path.read_text())
        models = Counter((o.get("identity") or {}).get("model_id") for o in bundle.get("outcomes", {}).values())
        model = SHORT.get(models.most_common(1)[0][0]) if models else None
        if args.model and model != args.model:
            continue
        files = bundle.get("files", {})
        trace_bytes = sum(m["bytes"] for p, m in files.items() if "/traces/" in p and p.endswith(".jsonl"))
        if spent + trace_bytes > args.budget_gb * 1e9:
            break
        spent += trace_bytes
        target = chunks / (Path(bundle_path).stem + ".json.gz")
        if target.exists():
            done += 1
            continue
        wanted = {}
        for p, m in files.items():
            if "/generations/" in p and p.endswith(".jsonl") and m["sha256"] in names and (hf / names[m["sha256"]]).exists():
                for line in open(hf / names[m["sha256"]], encoding="utf-8"):
                    rec = json.loads(line)
                    row = rec["row"]
                    keyword = prompts.get(row.get("prompt_id"), (None,))[0]
                    if keyword is not None and split.get(keyword) == (args.split == "exploration"):
                        wanted[row["trace_record_id"]] = (rec["record_id"], row, keyword)
        records, counts = [], Counter()
        for p, m in files.items():
            if not ("/traces/" in p and p.endswith(".jsonl")) or not wanted:
                continue
            local = hf_hub_download(REPO, names[m["sha256"]], repo_type="dataset", local_dir=str(chunks / "tmp"))
            try:
                with open(local, encoding="utf-8") as stream:
                    for line in stream:
                        rec = json.loads(line)
                        hit = wanted.get(rec["record_id"])
                        if hit is None:
                            continue
                        record_id, generation, keyword = hit
                        cell = {"generation": generation, "trace": rec["row"],
                                "keyword_memberships": {"primary_keyword_id": keyword}}
                        fields, skipped = readiness.checked_cell(cell, prompt_axis)
                        if skipped:
                            counts[skipped] += 1
                            continue
                        engine = generation.get("engine")
                        try:
                            items = fr.answer_items(rec["row"], generation["method"], engine, fields["urls"], fields["ranking"],
                                                    lookup, rows, index=indices.get(engine))
                        except (KeyError, TypeError, ValueError) as error:
                            counts[f"skipped_trace:{str(error)[:60]}"] += 1
                            continue
                        counts.update({f"searches:{k}": v for k, v in items["counts"].items()})
                        records.append({"fingerprint": record_id, "model": model, "engine": engine, "method": generation["method"],
                                        "condition": generation.get("condition"), "prompt_id": generation["prompt_id"],
                                        "keyword_text": keyword, "x": prompt_axis[generation["prompt_id"]],
                                        "items": items["items"], "events": [[e["search"], e["candidates"]] for e in items["events"]],
                                        "queries": items["queries"]})
            finally:
                Path(local).unlink(missing_ok=True)
        temporary = target.with_suffix(".partial")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump({"bundle": Path(bundle_path).name, "model": model, "records": records, "counts": counts}, stream)
        os.replace(temporary, target)
        done += 1
        print(json.dumps({"bundles": done, "trace_gb": round(spent / 1e9, 2), "answers": len(records), "time": readiness.now()}), flush=True)
    merge_chunks([chunks], out, rows)
    return 0


def merge(args) -> int:
    merge_chunks([Path(c) for c in args.chunks], Path(args.output), fr.snapshot_rows(dict(s.split("=", 1) for s in args.snapshot)))
    return 0


def merge_chunks(chunk_dirs: list, out: Path, rows) -> None:
    """Write the extract folder (answers, items, events, manifest) from the finished chunks of one or
    more chunk folders, in the given folder order (files sorted within each folder)."""
    study.refuse_existing(out)
    answers, item_parts, sizes, events, counts, queries = [], [], [], [], Counter(), []
    seen = set()
    paths = [path for chunks in chunk_dirs for path in sorted(Path(chunks).glob("*.json.gz"))]
    for path in paths:
        chunk = json.load(gzip.open(path, "rt", encoding="utf-8"))
        counts.update(chunk["counts"])
        for r in chunk["records"]:
            key = (r["model"], r["prompt_id"], r["engine"], r["method"], r["condition"])
            if key in seen:  # the same cell published twice (recovery): keep the first
                counts["duplicate_cells"] += 1
                continue
            seen.add(key)
            i = len(answers)
            answers.append({k: r[k] for k in ("fingerprint", "model", "engine", "method", "condition", "prompt_id", "keyword_text", "x")})
            queries.append({"answer": i, "queries": r.get("queries", [])})
            item_parts.append(np.asarray(r["items"], np.int64).reshape(-1, 6))
            sizes.append(len(r["items"]))
            for search, cands in r["events"]:
                events.append((i, search, np.asarray(cands, float).reshape(-1, 3)))
    partial = readiness.new_directory(out)
    items = np.concatenate(item_parts) if item_parts else np.zeros((0, 6), np.int64)
    readiness.write_jsonl(partial / "answers.jsonl.gz", answers)
    readiness.write_jsonl(partial / "queries.jsonl.gz", queries)
    readiness.write_npz(partial / "items.npz", offsets=np.r_[0, np.cumsum(sizes)].astype(np.int64), row=items[:, 0], search=items[:, 1],
                        rank=items[:, 2], scored=items[:, 3], presented=items[:, 4], ranked=items[:, 5])
    cand_sizes = [len(c) for _, _, c in events]
    cands = np.concatenate([c for _, _, c in events]) if events else np.zeros((0, 3))
    readiness.write_npz(partial / "events.npz", answer=np.asarray([a for a, _, _ in events], np.int64),
                        search=np.asarray([s for _, s, _ in events], np.int64), offsets=np.r_[0, np.cumsum(cand_sizes)].astype(np.int64),
                        row=cands[:, 0].astype(np.int64), score=cands[:, 1], selected=cands[:, 2].astype(np.int64))
    readiness.write_json(partial / "manifest.json", {
        "format_version": study.FORMAT_VERSION, "stage": "extract-hf", "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "source": REPO, "split": "exploration", "snapshot_sha256": rows.snapshot_sha256, "row_table_digest": fr.row_table_digest(rows),
        "snapshot_hash_mismatch": {}, "counts": {**counts, "answers": len(answers), "items": int(len(items)), "events": len(events)},
        "chunk_dirs": [Path(c).name for c in chunk_dirs], "chunks": len(paths),
        "answers_by_model": dict(Counter(a["model"] for a in answers))})
    partial.rename(out.resolve())
    print(json.dumps({"answers": len(answers), "items": int(len(items)), "events": len(events)}), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    home = Path.home() / "Hamburg"
    hf = home / "GEODML_Unified/ARR_ACL_CycleOct2026/hf-min"
    serp = home / "GEODML_Analysis/geodml_data/data/serp"
    p = sub.add_parser("extract-hf")
    p.add_argument("--hf-root", type=Path, default=hf)
    p.add_argument("--prompts", type=Path, default=hf / "snapshots/recovery-5ad9bf081e45d0d0a3131b51/data/prompts.jsonl.gz")
    p.add_argument("--snapshot", action="append", default=[f"duckduckgo={serp / 'phase0_top20_ddg.parquet'}",
                                                            f"searxng={serp / 'phase0_top20_searxng.parquet'}"])
    p.add_argument("--split", choices=["exploration"], default="exploration")
    p.add_argument("--budget-gb", type=float, default=12.0)
    p.add_argument("--model", choices=sorted(SHORT.values()), help="keep only this model's bundles")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("merge")
    p.add_argument("--chunks", type=Path, action="append", required=True, help="chunk folder (repeat; order kept)")
    p.add_argument("--snapshot", action="append", default=[f"duckduckgo={serp / 'phase0_top20_ddg.parquet'}",
                                                            f"searxng={serp / 'phase0_top20_searxng.parquet'}"])
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"extract-hf": extract_hf, "merge": merge}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
