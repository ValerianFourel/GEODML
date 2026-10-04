#!/usr/bin/env python3
"""Relocate the frozen readiness subspace, place evidence pages on it, and relate
generator rankings to page coordinates along the prompt axis.

Stages (each writes a new directory, never overwriting):
  extract         CPU   answers (presented pages, ranking, prompt axis) + unique page texts
  corpus          CPU   every servable snapshot row (both engines), deduplicated like shown snippets
  sample-prompts  CPU   deterministic sample of archived prompt texts for re-embedding
  embed           1 GPU one LLM2Vec view; sharded, resumable, one process per GPU
  merge           CPU   join shards into the archived projection format with a manifest
  relocate        CPU   prove the archived prompt coordinates are reproduced
  analyze         CPU   page coordinates, ranking models, bootstrap/permutation, HTML
  package         CPU   snippet table + full vectors per view + manifest, for the dataset
  publish         login upload a verified package to the private HF dataset

All results are observational. The axis is a measured prompt property; page
coordinates are out-of-domain descriptions of page text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY, REPOSITORY / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.interpretability.pipeline import page_readiness_ordering as ordering  # noqa: E402

SHARD_SIZE = 20000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def write_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(path)


def new_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError(f"interrupted output preserved for inspection: {partial}")
    partial.mkdir(parents=True)
    return partial


def axis_by_prompt(final_axis_map: Path) -> dict[str, float]:
    return {str(r["candidate_id"]): float(r["axis_1_percentile_0_1"]) for r in read_jsonl(final_axis_map)}


# ---------------------------------------------------------------- extract

def checked_cell(cell, prompt_axis):
    """Shared validity rules for one generator cell. Returns (fields, None) or (None, skip reason).
    Fields: prompt_id, keyword, evidence (presented order, URL-deduplicated), urls, ranking."""
    from analysis.interpretability.pipeline.agentic_judging import _trace_evidence

    generation = cell["generation"]
    prompt_id = generation.get("prompt_id")
    keyword = (cell.get("keyword_memberships") or {}).get("primary_keyword_id")
    if prompt_id not in prompt_axis:
        return None, "skipped_prompt_without_axis"
    if not keyword:
        return None, "skipped_without_keyword"
    try:
        # The final evidence the generator saw, in presented order (shared with SI judging).
        evidence, _ = _trace_evidence(cell["trace"], generation["method"])
    except (ValueError, KeyError, TypeError):
        return None, "skipped_unreadable_evidence"
    urls = [row["url"] for row in evidence]
    ranking = list(generation.get("ranking") or [])
    if len(set(urls)) != len(urls):
        return None, "skipped_duplicate_presented_url"
    if len(set(ranking)) != len(ranking) or not set(ranking) <= set(urls):
        return None, "skipped_ranking_outside_evidence"
    return {"prompt_id": prompt_id, "keyword": keyword, "evidence": evidence, "urls": urls, "ranking": ranking}, None


def parallel_chunks(sources, worker, extra, workers):
    """Completed cells of each DATASET_ROOT:MODEL source, read in contiguous slices by forked workers.
    Returns (inputs, iterator of (chunk number, chunk count, worker result)) in deterministic order."""
    from analysis.interpretability.pipeline.agentic_cells import completed_generator_refs

    inputs, jobs = [], []
    for spec in sources:
        root, _, model = spec.rpartition(":")
        root = Path(root).resolve(strict=True)
        refs, ref_counts = completed_generator_refs(root, model=model)
        inputs.append({"dataset_root": str(root), "model": model, "cells_completed": len(refs),
                       "selection": dict(ref_counts)})
        size = max(1, -(-len(refs) // (workers * 2)))
        jobs += [(root, model, refs[start:start + size], extra) for start in range(0, len(refs), size)]
    print(json.dumps({"cells": sum(len(j[2]) for j in jobs), "chunks": len(jobs), "workers": workers}), flush=True)

    def results():
        if workers == 1:
            for number, result in enumerate(map(worker, jobs), 1):
                yield number, len(jobs), result
            return
        import multiprocessing
        pool = multiprocessing.get_context("fork").Pool(workers)
        try:
            for number, result in enumerate(pool.imap(worker, jobs), 1):  # ordered: deterministic output
                yield number, len(jobs), result
        finally:
            pool.close()
            pool.join()

    return inputs, results()


def _extract_chunk(job):
    """One contiguous slice of completed cells (keeps shard reads local); runs in a forked worker."""
    from analysis.interpretability.pipeline.agentic_cells import iter_cells

    root, model, refs, prompt_axis = job
    counts, pages, observations = Counter(), {}, []
    for cell in iter_cells(root, refs):
        generation = cell["generation"]
        fields, skipped = checked_cell(cell, prompt_axis)
        if skipped:
            counts[skipped] += 1
            continue
        by_url = {}
        for row in fields["evidence"]:
            text = ordering.page_text(row["title"], row["text"])
            identifier = ordering.page_id(text)
            by_url[row["url"]] = identifier
            entry = pages.setdefault(identifier, {"page_id": identifier, "text": text, "occurrences": 0,
                                                  "urls": set(), "engines": set(), "models": set()})
            entry["occurrences"] += 1
            entry["urls"].add(row["url"])
            entry["engines"].add(generation.get("engine"))
            entry["models"].add(model)
        observations.append({
            "generation_id": cell["fingerprint"], "prompt_id": fields["prompt_id"], "keyword": fields["keyword"],
            "model": model, **{k: generation.get(k) for k in ("method", "engine", "condition")},
            "prompt_axis": prompt_axis[fields["prompt_id"]], "presented": [by_url[u] for u in fields["urls"]],
            "ranking": [by_url[u] for u in fields["ranking"]]})
        counts[f"answers_{model}"] += 1
    return observations, pages, counts


def extract(args) -> int:
    prompt_axis = axis_by_prompt(args.final_axis_map)
    partial = new_directory(args.output)
    workers = max(1, args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    counts = Counter()
    pages: dict[str, dict] = {}
    observations = []
    inputs, results = parallel_chunks(args.source, _extract_chunk, prompt_axis, workers)
    for number, total, (chunk_observations, chunk_pages, chunk_counts) in results:
        observations += chunk_observations
        counts += chunk_counts
        for identifier, entry in chunk_pages.items():
            merged = pages.setdefault(identifier, {**entry, "occurrences": 0, "urls": set(),
                                                   "engines": set(), "models": set()})
            merged["occurrences"] += entry["occurrences"]
            for key in ("urls", "engines", "models"):
                merged[key] |= entry[key]
        print(json.dumps({"chunk": number, "of": total, "answers": len(observations), "time": now()}), flush=True)
    write_jsonl(partial / "observations.jsonl.gz", observations)
    write_jsonl(partial / "pages.jsonl.gz", ({"page_id": p["page_id"], "text": p["text"], "text_sha256": p["page_id"],
                                              "occurrences": p["occurrences"], "urls": sorted(p["urls"]),
                                              "engines": sorted(e for e in p["engines"] if e), "models": sorted(p["models"])}
                                             for p in sorted(pages.values(), key=lambda r: r["page_id"])))
    write_json(partial / "manifest.json", {
        "format_version": ordering.FORMAT_VERSION, "stage": "extract", "created_at": now(),
        "git_commit": git_commit(), "inputs": inputs, "final_axis_map": identity(args.final_axis_map),
        "prompt_axis_field": "axis_1_percentile_0_1", "page_text_rule": "title.strip() + newline + snippet.strip()",
        "counts": {**counts, "answers": len(observations), "unique_pages": len(pages)}})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"answers": len(observations), "unique_pages": len(pages), **counts}), flush=True)
    return 0


def corpus(args) -> int:
    """Every servable row of both frozen search snapshots, deduplicated exactly like shown snippets."""

    from analysis.scripts import audit_snapshot_glued_rows as snapshots
    from analysis.scripts.run_agentic_search_integration_smoke import _normalize_usable_row

    files = {}
    for spec in args.snapshot:
        engine, _, path = str(spec).partition("=")
        files[engine] = Path(path).resolve(strict=True)
    if args.locate_from:
        for engine, path in snapshots.locate(args.locate_from, args.search_root or args.locate_from[0].parent,
                                             report=lambda line: print(line, flush=True)).items():
            files.setdefault(engine, path)
    if not files:
        raise ValueError("give --snapshot ENGINE=PATH or --locate-from DATASET_ROOT")
    shown = {r["page_id"]: r for r in read_jsonl(args.shown / "pages.jsonl.gz")} if args.shown else {}
    partial = new_directory(args.output)
    counts, pages = Counter(), {}
    for engine, path in sorted(files.items()):
        rows = snapshots.read_snapshot(path)
        counts[f"rows_{engine}"] = len(rows)
        # Other same-keyword results' titles glued into this row (the DuckDuckGo scrape defect).
        glued = snapshots.glued_titles(rows)
        for row, glued_count in zip(rows, glued):
            usable, reason = _normalize_usable_row(row)  # the search adapter's own servability check
            if usable is None:
                counts[f"excluded_{engine}_{reason}"] += 1
                continue
            text = ordering.page_text(usable["title"], usable["snippet"])
            identifier = ordering.page_id(text)
            entry = pages.setdefault(identifier, {"page_id": identifier, "text": text, "text_sha256": identifier,
                                                  "urls": set(), "engines": set(), "keywords": set(), "best_position": None,
                                                  "glued_titles": 0})
            entry["glued_titles"] = max(entry["glued_titles"], glued_count)
            counts[f"glued_rows_{engine}"] += glued_count >= 2
            entry["urls"].add(usable["url"])
            entry["engines"].add(engine)
            entry["keywords"].add(usable["keyword"])
            entry["best_position"] = min(filter(None, (entry["best_position"], usable["position"])))
    missing = set(shown) - set(pages)
    rows = []
    for identifier in sorted(pages):
        entry, seen = pages[identifier], shown.get(identifier)
        rows.append({**entry, "urls": sorted(entry["urls"]), "engines": sorted(entry["engines"]),
                     "keywords": sorted(entry["keywords"]), "shown": seen is not None,
                     "occurrences": seen["occurrences"] if seen else 0, "models": seen.get("models", []) if seen else []})
    write_jsonl(partial / "pages.jsonl.gz", rows)
    counts.update(unique_pages=len(rows), shown_in_corpus=len(set(shown) & set(pages)), shown_not_in_corpus=len(missing))
    write_json(partial / "manifest.json", {
        "format_version": ordering.FORMAT_VERSION, "stage": "corpus", "created_at": now(), "git_commit": git_commit(),
        "inputs": [{"model": "snapshot-corpus"}],
        "snapshots": {engine: {"path": str(path), "sha256": sha256_file(path)} for engine, path in files.items()},
        "shown_extract": identity(args.shown / "manifest.json") if args.shown else None,
        "page_text_rule": "title.strip() + newline + snippet.strip(), rows the search adapter can serve",
        "counts": dict(counts)})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"unique_snippets": len(rows), **counts}), flush=True)
    return 0


def sample_prompts(args) -> int:
    rows = read_jsonl(args.prompts)
    chosen = sorted(rows, key=lambda r: hashlib.sha256(f"{args.seed}:{r['candidate_id']}".encode()).hexdigest())[:args.count]
    partial = new_directory(args.output)
    write_jsonl(partial / "pages.jsonl.gz", ({"page_id": r["candidate_id"], "text": r["question"],
                                              "text_sha256": r["question_sha256"]} for r in chosen))
    write_json(partial / "manifest.json", {"stage": "sample-prompts", "created_at": now(), "seed": args.seed,
                                           "count": len(chosen), "prompts": identity(args.prompts)})
    partial.rename(Path(args.output).resolve())
    return 0


# ---------------------------------------------------------------- embed / merge

def _readiness_modules():
    from interpretability.pipeline import readiness_prompt_population as population
    return population


def embed(args) -> int:
    """Embed this worker's shards with one LLM2Vec view and project through its frozen map."""

    population = _readiness_modules()
    from interpretability.pipeline.two_axis_prompt_population import LLM2VecPromptEmbedder
    from analysis.scripts.build_readiness_prompt_population import _validate_embedding_model_revision

    fitted = population.load_readiness_embedding_map(args.map / "readiness_embedding_map.json")
    _validate_embedding_model_revision(fitted, args.embedding_model)
    bounds = population.fit_reference_bounds(read_jsonl(args.map / "readiness_supervised_subspace_coordinates.jsonl"))
    pages = read_jsonl(args.pages)
    for row in pages:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError(f"page text hash differs: {row['page_id']}")
    output = Path(args.output).resolve()
    (output / "shards").mkdir(parents=True, exist_ok=True)
    settings = {"view": args.view, "map_id": fitted.map_id, "map": identity(args.map / "readiness_embedding_map.json"),
                "pages": identity(args.pages), "embedding_model": str(Path(args.embedding_model).resolve()),
                "mntp_model": str(Path(args.mntp_model).resolve()) if args.mntp_model else None,
                "peft_model": str(Path(args.peft_model).resolve()) if args.peft_model else None,
                "max_length": args.max_length, "attention_implementation": args.attention_implementation,
                "shard_size": args.shard_size}
    if args.save_embeddings:  # recorded only when on, so earlier outputs keep resuming
        settings["save_embeddings"] = True
    settings_path = output / "embed-settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("existing embedding output used different settings; use a new output")
    else:
        try:
            with settings_path.open("x") as stream:
                stream.write(json.dumps(settings, indent=2, sort_keys=True) + "\n")
        except FileExistsError:
            if json.loads(settings_path.read_text()) != settings:
                raise ValueError("existing embedding output used different settings") from None
    embedder = None
    shards = range(0, len(pages), args.shard_size)
    for number, start in enumerate(shards):
        if number % args.workers != args.worker_index:
            continue
        target = output / "shards" / f"{number:05d}.jsonl.gz"
        if target.exists():
            continue
        chunk = pages[start:start + args.shard_size]
        if embedder is None:
            embedder = LLM2VecPromptEmbedder(args.embedding_model, mntp_model_name_or_path=args.mntp_model,
                                             peft_model_name_or_path=args.peft_model, batch_size=args.batch_size,
                                             max_length=args.max_length,
                                             attention_implementation=args.attention_implementation)
        vectors = embedder.embed([r["text"] for r in chunk])
        projections = population.project_text_embeddings(
            fitted, bounds, item_ids=[r["page_id"] for r in chunk], text_sha256s=[r["text_sha256"] for r in chunk],
            embeddings=vectors)
        if args.save_embeddings:  # the projections file below marks the shard complete
            write_npz(output / "shards" / f"{number:05d}.npz", ids=np.asarray([r["page_id"] for r in chunk]),
                      embeddings=np.asarray(vectors, dtype=np.float32))
        write_jsonl(target, ({"candidate_id": p.item_id, "projection": asdict(p)} for p in projections))
        print(json.dumps({"view": args.view, "shard": number, "pages": len(chunk), "time": now()}), flush=True)
    return 0


def merge(args) -> int:
    settings = json.loads((args.input / "embed-settings.json").read_text())
    pages = [r["page_id"] for r in read_jsonl(args.pages)]
    if identity(args.pages)["sha256"] != settings["pages"]["sha256"]:
        raise ValueError("pages file differs from the embedded one")
    expected = (len(pages) + settings["shard_size"] - 1) // settings["shard_size"]
    rows = []
    for number in range(expected):
        path = args.input / "shards" / f"{number:05d}.jsonl.gz"
        if not path.exists():
            raise ValueError(f"missing shard {number}; rerun embed with the same settings")
        rows.extend(read_jsonl(path))
    if [r["candidate_id"] for r in rows] != pages:
        raise ValueError("merged projections do not cover the pages exactly once, in order")
    vectors = None
    if settings.get("save_embeddings"):
        parts = []
        for number in range(expected):
            with np.load(args.input / "shards" / f"{number:05d}.npz", allow_pickle=False) as archive:
                start = number * settings["shard_size"]
                if archive["ids"].tolist() != pages[start:start + settings["shard_size"]]:
                    raise ValueError(f"shard {number} vectors are not aligned with its pages")
                parts.append(archive["embeddings"])
        vectors = np.concatenate(parts)
    partial = new_directory(args.output)
    write_jsonl(partial / "question_projections.jsonl", rows)
    arrays = None
    if vectors is not None:
        write_npy(partial / "embeddings.npy", vectors)
        arrays = {"file": "embeddings.npy", "shape": list(vectors.shape), "dtype": "float32",
                  "row_order": "pages file order, identical to question_projections.jsonl",
                  "sha256": sha256_file(partial / "embeddings.npy")}
    write_json(partial / "projection_manifest.json", {
        "format_version": ordering.FORMAT_VERSION, "created_at": now(), "git_commit_sha": git_commit(),
        "map_id": settings["map_id"], "map": settings["map"], "embedding": settings,
        "candidate_count": len(rows), "embedding_arrays_included": vectors is not None, "embedding_arrays": arrays})
    partial.rename(Path(args.output).resolve())
    return 0


def aligned(qwen: Path, mistral: Path, battery: Path) -> list[dict]:
    """The archived Qwen/Mistral alignment code, applied unchanged."""

    from analysis.scripts.build_readiness_prompt_population import _aligned_projection_rows
    rows, _, _ = _aligned_projection_rows(Path(qwen).resolve(), Path(mistral).resolve(), Path(battery).resolve())
    return rows


# ---------------------------------------------------------------- relocate

def replay_map(map_dir: Path, embeddings_dir: Path) -> dict[str, tuple[float, float]]:
    """Project archived prompt embeddings (shard-*/question_embeddings.restricted-local.npz)."""

    population = _readiness_modules()
    fitted = population.load_readiness_embedding_map(map_dir / "readiness_embedding_map.json")
    bounds = population.fit_reference_bounds(read_jsonl(map_dir / "readiness_supervised_subspace_coordinates.jsonl"))
    result = {}
    shards = sorted(Path(embeddings_dir).glob("shard-*/question_embeddings.restricted-local.npz"))
    if not shards:
        raise ValueError(f"no archived embedding shards under {embeddings_dir}")
    for path in shards:
        with np.load(path, allow_pickle=False) as archive:
            ids = [str(i) for i in archive["candidate_ids"]]
            rows = population.project_text_embeddings(fitted, bounds, item_ids=ids, text_sha256s=ids,
                                                      embeddings=archive["embeddings"])
        for row in rows:
            if row.item_id in result:
                raise ValueError(f"prompt embedded in two archived shards: {row.item_id}")
            result[row.item_id] = (row.raw_axis_1, row.raw_axis_2)
    return result


def relocate(args) -> int:
    final_rows = read_jsonl(args.final_axis_map)
    report = {"format_version": ordering.FORMAT_VERSION, "stage": "relocate", "created_at": now(),
              "git_commit": git_commit(), "final_axis_map": identity(args.final_axis_map),
              "battery": identity(args.battery / "readiness_robustness_battery.json"),
              "archived_coordinates": ordering.relocation_audit(
                  final_rows, aligned(args.qwen_projections, args.mistral_projections, args.battery))}
    replay = {}
    for view, archived_dir, map_dir, embeddings_dir in (
            ("qwen", args.qwen_projections, args.qwen_map, args.qwen_embeddings),
            ("mistral", args.mistral_projections, args.mistral_map, args.mistral_embeddings)):
        if embeddings_dir is None:
            continue
        archived = {r["candidate_id"]: (r["projection"]["raw_axis_1"], r["projection"]["raw_axis_2"])
                    for r in read_jsonl(archived_dir / "question_projections.jsonl")}
        replay[view] = ordering.map_replay(archived, replay_map(map_dir, embeddings_dir))
    report["map_replay"] = replay
    fresh = {}
    for view, archived_dir, fresh_dir in (("qwen", args.qwen_projections, args.fresh_qwen),
                                          ("mistral", args.mistral_projections, args.fresh_mistral)):
        if fresh_dir is None:
            continue
        archived = {r["candidate_id"]: r["projection"]["raw_axis_1"]
                    for r in read_jsonl(archived_dir / "question_projections.jsonl")}
        observed = {r["candidate_id"]: r["projection"]["raw_axis_1"]
                    for r in read_jsonl(fresh_dir / "question_projections.jsonl")}
        fresh[view] = ordering.projection_agreement(archived, observed)
    report["fresh_reembedding"] = fresh
    report["passed"] = (report["archived_coordinates"]["passed"]
                        and all(v["passed"] for v in (*replay.values(), *fresh.values())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], **report["archived_coordinates"],
                      **{f"replay_{k}": v["passed"] for k, v in replay.items()},
                      **{f"fresh_{k}": v["spearman"] for k, v in fresh.items()}}), flush=True)
    return 0 if report["passed"] else 2


# ---------------------------------------------------------------- analyze

def page_coordinates(qwen: Path, mistral: Path, battery: Path, final_axis_map: Path):
    """Per-view axis-1 z, consensus z and prompt-scale percentile, via the archived alignment."""

    scale = ordering.PromptScale(read_jsonl(final_axis_map))
    rows = aligned(qwen, mistral, battery)
    views = {"qwen": {r["candidate_id"]: r["reference_axis_1_z"] for r in rows},
             "mistral": {r["candidate_id"]: r["candidate_aligned_axis_1_z"] for r in rows}}
    ids = sorted(views["qwen"])
    views["consensus"] = dict(zip(ids, ordering.consensus([views["qwen"][i] for i in ids],
                                                         [views["mistral"][i] for i in ids])))
    percentile, outside = scale.percentile(np.asarray([views["consensus"][i] for i in ids]))
    return ids, views, dict(zip(ids, map(float, percentile))), dict(zip(ids, map(bool, outside)))


def analyze(args) -> int:
    extract_manifest = json.loads((args.extract / "manifest.json").read_text())
    observations = read_jsonl(args.extract / "observations.jsonl.gz")
    ids, views, page_percentile, page_outside = page_coordinates(args.qwen, args.mistral, args.battery,
                                                                 args.final_axis_map)
    missing = {p for o in observations for p in o["presented"]} - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} presented pages lack projections")
    partial = new_directory(args.output)
    write_jsonl(partial / "page_coordinates.jsonl.gz", ({"page_id": i, "qwen_axis_1_z": views["qwen"][i],
        "mistral_aligned_axis_1_z": views["mistral"][i], "consensus_axis_1_z": views["consensus"][i],
        "prompt_scale_percentile_0_1": page_percentile[i], "outside_prompt_range": page_outside[i]} for i in ids))
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    results = {"format_version": ordering.FORMAT_VERSION, "stage": "analyze", "created_at": now(),
               "git_commit": git_commit(), "extract": extract_manifest,
               "inputs": {"final_axis_map": identity(args.final_axis_map),
                          "qwen": identity(args.qwen / "projection_manifest.json"),
                          "mistral": identity(args.mistral / "projection_manifest.json"),
                          "battery": identity(args.battery / "battery_manifest.json")},
               "settings": {"bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed,
                            "max_position": ordering.MAX_POSITION},
               "page_scale": ordering.page_scale_diagnostics(views["consensus"], page_percentile, page_outside, observations),
               "models": {}}
    for model in sorted({o["model"] for o in observations}):
        selected = [o for o in observations if o["model"] == model and o["ranking"]]
        statistics = [ordering.generation_statistics(o, views["consensus"]) for o in selected]
        entry = {"answers": len(selected), "keywords": len({o["keyword"] for o in selected}), "fits": {}}
        for view in ordering.VIEWS:
            data = ordering.ChoiceData(selected, views[view])
            fit = ordering.fit_plackett_luce(data)
            if view == "consensus":
                fit["keyword_bootstrap"] = ordering.keyword_bootstrap(
                    data, replicates=args.bootstrap, seed=args.seed, start=fit["theta"], workers=workers)
                fit["within_keyword_permutation"] = ordering.within_keyword_permutation(
                    data, observed=fit["page_z_x_prompt_axis"], replicates=args.permutations,
                    seed=args.seed + 1, start=fit["theta"], workers=workers)
            entry["fits"][view] = fit
        entry["by_method"] = {}
        for method in sorted({o["method"] for o in selected}):
            subset = [o for o in selected if o["method"] == method]
            entry["by_method"][method] = ordering.fit_plackett_luce(ordering.ChoiceData(subset, views["consensus"]))
        entry["binned"] = {field: ordering.binned_contrasts(selected, statistics, field=field, seed=args.seed)
                           for field in ("ranked_minus_presented_z", "top1_minus_presented_z")}
        valid = [(o["prompt_axis"], s["ranked_minus_presented_z"]) for o, s in zip(selected, statistics)
                 if s.get("ranked_minus_presented_z") is not None]
        entry["axis_vs_ranked_minus_presented_spearman"] = ordering._spearman(*map(np.asarray, zip(*valid))) if len(valid) > 2 else None
        results["models"][model] = entry
        print(json.dumps({"model": model, "answers": len(selected),
                          "page_z": entry["fits"]["consensus"]["page_z"],
                          "interaction": entry["fits"]["consensus"]["page_z_x_prompt_axis"]}), flush=True)
    write_json(partial / "results.json", results)
    (partial / "report.html").write_text(render(results), encoding="utf-8")
    partial.rename(Path(args.output).resolve())
    print(f"REPORT {Path(args.output).resolve() / 'report.html'}", flush=True)
    return 0


# ---------------------------------------------------------------- package / publish

EMBEDDING_FILES = {"qwen": "embeddings/qwen3-8b-llm2vec.npy", "mistral": "embeddings/mistral-7b-instruct-v0.2-llm2vec.npy"}
DEFAULT_REPO = "ValerianFourel/geodml-experiment-v2-paper-private"


def package(args) -> int:
    """One self-describing folder: snippet table, full vectors per view in the same row order, manifest."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    relocation = json.loads(args.relocation.read_text())
    if not relocation.get("passed") or not relocation.get("fresh_reembedding"):
        raise ValueError("package requires a passed relocation report that includes the fresh re-embedding check")
    pages = read_jsonl(args.extract / "pages.jsonl.gz")
    order = [p["page_id"] for p in pages]
    manifests, raw = {}, {}
    for view, directory in (("qwen", args.qwen), ("mistral", args.mistral)):
        manifests[view] = json.loads((directory / "projection_manifest.json").read_text())
        if not manifests[view].get("embedding_arrays_included"):
            raise ValueError(f"{view} projections were embedded without --save-embeddings")
        rows = read_jsonl(directory / "question_projections.jsonl")
        if [r["candidate_id"] for r in rows] != order:
            raise ValueError(f"{view} projections are not in the pages order")
        raw[view] = [r["projection"] for r in rows]
    ids, views, percentile, outside = page_coordinates(args.qwen, args.mistral, args.battery, args.final_axis_map)
    if set(ids) != set(order):
        raise ValueError("coordinates do not cover the extracted snippets exactly")
    partial = new_directory(args.output)
    (partial / "embeddings").mkdir()
    for view, directory in (("qwen", args.qwen), ("mistral", args.mistral)):
        vectors = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if vectors.shape[0] != len(order) or sha256_file(directory / "embeddings.npy") != manifests[view]["embedding_arrays"]["sha256"]:
            raise ValueError(f"{view} vectors do not match their manifest")
        shutil.copyfile(directory / "embeddings.npy", partial / EMBEDDING_FILES[view])
    titles, snippets = zip(*(p["text"].split("\n", 1) if "\n" in p["text"] else (p["text"], "") for p in pages))
    table = pa.table({
        "row": list(range(len(pages))), "snippet_id": order, "title": list(titles), "snippet": list(snippets),
        "text": [p["text"] for p in pages], "text_sha256": [p["text_sha256"] for p in pages],
        "urls": [p["urls"] for p in pages], "engines": [p.get("engines", []) for p in pages],
        "models": [p.get("models", []) for p in pages], "times_shown": [p["occurrences"] for p in pages],
        **({"shown": [p["shown"] for p in pages], "keywords": [p["keywords"] for p in pages],
            "best_position": [p["best_position"] for p in pages],
            "glued_titles": [p["glued_titles"] for p in pages]} if "shown" in pages[0] else {}),
        "qwen_raw_axis_1": [r["raw_axis_1"] for r in raw["qwen"]], "qwen_raw_axis_2": [r["raw_axis_2"] for r in raw["qwen"]],
        "mistral_raw_axis_1": [r["raw_axis_1"] for r in raw["mistral"]],
        "mistral_raw_axis_2": [r["raw_axis_2"] for r in raw["mistral"]],
        "qwen_axis_1_z": [views["qwen"][i] for i in order],
        "mistral_aligned_axis_1_z": [views["mistral"][i] for i in order],
        "consensus_axis_1_z": [views["consensus"][i] for i in order],
        "prompt_scale_percentile_0_1": [percentile[i] for i in order],
        "outside_prompt_range": [outside[i] for i in order]})
    pq.write_table(table, partial / "snippets.parquet")
    extract_manifest = json.loads((args.extract / "manifest.json").read_text())
    readme = f"""# Evidence snippet embeddings (Experiment V2)

Every distinct evidence snippet shown to a generator in its final answering context
({len(order):,} snippets; sources: {", ".join(i["model"] for i in extract_manifest["inputs"])}).
A snippet is `title + newline + snippet text` exactly as presented, deduplicated by SHA-256.

- `snippets.parquet`: one row per snippet; `row` is the row index into each embedding file.
- `{EMBEDDING_FILES["qwen"]}`, `{EMBEDDING_FILES["mistral"]}`: float32 LLM2Vec vectors, same row order.
- Axis columns place each snippet on the frozen 26,009-prompt information-seeking to
  action-readiness axis (Qwen and aligned Mistral axis-1 z, their consensus, and the
  percentile on the prompt scale). The axis was fitted on prompts, so snippet positions are
  out-of-domain descriptions; they are not treatments.
- Evidence came from a frozen snapshot served by corpus-wide lexical search; some snippets
  were shown under unrelated keywords. `times_shown`, `engines` and `models` record exposure.
- `manifest.json`: inputs, model revisions, code commit and SHA-256 of every file.
"""
    (partial / "README.md").write_text(readme, encoding="utf-8")
    files = {str(path.relative_to(partial)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
             for path in sorted(partial.rglob("*")) if path.is_file()}
    write_json(partial / "manifest.json", {
        "format_version": "geodml-snippet-embeddings-v1", "created_at": now(), "git_commit": git_commit(),
        "rows": len(order), "embedding_dimension": int(np.load(partial / EMBEDDING_FILES["qwen"], mmap_mode="r").shape[1]),
        "views": {view: {"file": EMBEDDING_FILES[view], "embedding": manifests[view]["embedding"],
                         "map_id": manifests[view]["map_id"]} for view in manifests},
        "extract": {"manifest": extract_manifest, "manifest_sha256": sha256_file(args.extract / "manifest.json"),
                    "pages_sha256": sha256_file(args.extract / "pages.jsonl.gz")},
        "final_axis_map": identity(args.final_axis_map), "battery": identity(args.battery / "battery_manifest.json"),
        "relocation": {"sha256": sha256_file(args.relocation), "passed": relocation["passed"],
                       "fresh_reembedding": {k: v["spearman"] for k, v in relocation["fresh_reembedding"].items()}},
        "files": files})
    partial.rename(Path(args.output).resolve())
    print(json.dumps({"package": str(Path(args.output).resolve()), "rows": len(order), "files": len(files) + 1}), flush=True)
    return 0


def verify_package(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"package file changed: {name}")
    return manifest


def publish(args) -> int:
    """Upload a verified package to the private dataset and check every file on the Hub."""

    import getpass
    from huggingface_hub import HfApi

    directory = Path(args.package).resolve()
    manifest = verify_package(directory)
    target = args.path_in_repo or f"derived/snippet-embeddings/{directory.name}"
    receipt_path = directory.with_name(directory.name + ".published.json")
    if receipt_path.exists():
        raise ValueError(f"already published; see {receipt_path}")
    token = getpass.getpass("HF WRITE token for the private dataset (hidden): ").strip()
    if not token:
        raise ValueError("a write token is required")
    try:
        api = HfApi(token=token)
        if not api.repo_info(args.repo_id, repo_type="dataset").private:
            raise ValueError("destination dataset is not private; refusing to upload")
        if any(name.startswith(target + "/") for name in api.list_repo_files(args.repo_id, repo_type="dataset")):
            raise ValueError(f"{target} already exists on the Hub; refusing to overwrite")
        commit = api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(directory),
                                   path_in_repo=target, commit_message=f"Add evidence snippet embeddings ({manifest['rows']} rows)")
        names = [*manifest["files"], "manifest.json"]
        remote = {info.path: info for info in api.get_paths_info(args.repo_id, [f"{target}/{n}" for n in names],
                                                                 repo_type="dataset", expand=True)}
        for name in names:
            info = remote.get(f"{target}/{name}")
            local = directory / name
            if info is None or info.size != local.stat().st_size:
                raise ValueError(f"remote file missing or wrong size: {name}")
            lfs = getattr(info, "lfs", None)
            if lfs is not None and getattr(lfs, "sha256", None) != sha256_file(local):
                raise ValueError(f"remote file checksum differs: {name}")
    except Exception as error:
        raise RuntimeError(str(error).replace(token, "[REDACTED]")) from None
    receipt = {"repo_id": args.repo_id, "path_in_repo": target, "commit": getattr(commit, "oid", None),
               "files_verified": len(names), "published_at": now()}
    write_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)
    return 0


def _chart(bins: list[dict], title: str) -> str:
    points = [b for b in bins if "mean" in b]
    if not points:
        return "<p>No data.</p>"
    low = min(min(b["ci95"][0] for b in points), 0.0)
    high = max(max(b["ci95"][1] for b in points), 0.0)
    span = (high - low) or 1.0
    width, height, pad = 560, 220, 36

    def x(value):
        return pad + value * (width - 2 * pad)

    def y(value):
        return height - pad - (value - low) / span * (height - 2 * pad)

    centers = [(b["prompt_axis_low"] + b["prompt_axis_high"]) / 2 for b in points]
    band = " ".join(f"{x(c):.1f},{y(b['ci95'][1]):.1f}" for c, b in zip(centers, points))
    band += " " + " ".join(f"{x(c):.1f},{y(b['ci95'][0]):.1f}" for c, b in reversed(list(zip(centers, points))))
    line = " ".join(f"{x(c):.1f},{y(b['mean']):.1f}" for c, b in zip(centers, points))
    return (f'<figure><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'<line x1="{pad}" x2="{width - pad}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>'
            f'<polygon points="{band}" class="band"/><polyline points="{line}" class="line"/>'
            f'<text x="{pad}" y="{height - 8}">information seeking (0)</text>'
            f'<text x="{width - pad}" y="{height - 8}" text-anchor="end">action ready (1)</text>'
            f'<text x="4" y="{y(high) + 4:.1f}">{high:+.2f}</text><text x="4" y="{y(low):.1f}">{low:+.2f}</text>'
            f'</svg><figcaption>{html.escape(title)}</figcaption></figure>')


def render(results: dict) -> str:
    def number(value, digits=3):
        return "—" if value is None else f"{value:+.{digits}f}"

    def pvalue(permutation):
        value = permutation.get("two_sided_p")
        return "—" if value is None else f"{value:.3f}"

    sections = []
    for model, entry in results["models"].items():
        rows = []
        for view, fit in entry["fits"].items():
            boot = fit.get("keyword_bootstrap", {})
            perm = fit.get("within_keyword_permutation", {})
            ci = lambda key: ("[" + ", ".join(f"{v:+.3f}" for v in boot[key]["ci95"]) + "]") if key in boot else "—"
            rows.append(f"<tr><td>{view}</td><td>{number(fit['page_z'])}</td><td>{ci('page_z')}</td>"
                        f"<td>{number(fit['page_z_x_prompt_axis'])}</td><td>{ci('page_z_x_prompt_axis')}</td>"
                        f"<td>{pvalue(perm)}</td>"
                        f"<td>{fit['choice_sets']:,}</td></tr>")
        methods = "".join(f"<tr><td>{html.escape(m)}</td><td>{number(f['page_z'])}</td>"
                          f"<td>{number(f['page_z_x_prompt_axis'])}</td><td>{f['choice_sets']:,}</td></tr>"
                          for m, f in entry["by_method"].items())
        sections.append(f"""
<section><h2>{html.escape(model)}</h2>
<p>{entry['answers']:,} answers across {entry['keywords']:,} keywords. Spearman between prompt axis and
(mean z of ranked pages − mean z of presented pages): <b>{number(entry['axis_vs_ranked_minus_presented_spearman'])}</b>.</p>
<h3>Ranking model (Plackett–Luce over the ranked prefix, presented-position effects)</h3>
<table><thead><tr><th>page view</th><th>page z</th><th>95% CI</th><th>page z × prompt axis</th><th>95% CI</th>
<th>permutation p</th><th>choices</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h3>By search method (consensus view)</h3>
<table><thead><tr><th>method</th><th>page z</th><th>page z × prompt axis</th><th>choices</th></tr></thead><tbody>{methods}</tbody></table>
{_chart(entry['binned']['ranked_minus_presented_z'], 'Ranked minus presented page z, by prompt-axis decile (keyword-cluster 95% CI)')}
{_chart(entry['binned']['top1_minus_presented_z'], 'Top-ranked page minus presented mean z, by prompt-axis decile')}
</section>""")
    scale = results["page_scale"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Page Readiness Ordering</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5d5a52;--line:#2f5d8a;--band:#2f5d8a33;--rule:#d8d4ca}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16171a;--fg:#ecebe6;--muted:#a9a69c;--line:#8cb8e6;--band:#8cb8e633;--rule:#3a3b40}}}}
:root[data-theme="dark"]{{--bg:#16171a;--fg:#ecebe6;--muted:#a9a69c;--line:#8cb8e6;--band:#8cb8e633;--rule:#3a3b40}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:860px;padding:24px 16px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 16px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
svg{{width:100%;height:auto}} .line{{fill:none;stroke:var(--line);stroke-width:2}} .band{{fill:var(--band)}}
.zero{{stroke:var(--muted);stroke-dasharray:4 4}} svg text{{fill:var(--muted);font-size:11px}} .note{{color:var(--muted)}}
</style></head><body>
<h1>Page readiness and generator ordering</h1>
<p class="note">Observational. The prompt axis is a measured prompt property, not a randomized treatment. Page
coordinates come from a subspace fitted on prompts and are out-of-domain descriptions of page text. Generated
{html.escape(results['created_at'])} from commit {html.escape(results['git_commit'][:12])}.</p>
<h2>How pages sit on the prompt scale</h2>
<p>{scale['pages']:,} unique pages; {scale['share_outside_prompt_range']:.1%} fall outside the prompt range.
Median page position on the prompt percentile scale: {scale['page_percentile_quantiles']['0.5']:.2f}.
Mean within-answer variance of page z: {number(scale['mean_within_answer_variance_z'])} (total {number(scale['page_z_variance'])}).</p>
<p class="note">Positive page z: generators rank action-ready pages higher. Positive page z × prompt axis: that
preference grows as the prompt moves toward action readiness. Position effects absorb presented order, which
the evidence-order conditions randomize.</p>
{''.join(sections)}
</body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    p = stages.add_parser("extract")
    p.add_argument("--source", action="append", required=True, metavar="DATASET_ROOT:MODEL")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--workers", type=int, help="parallel readers (default: SLURM_CPUS_PER_TASK or 1)")
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("corpus")
    p.add_argument("--snapshot", action="append", default=[], metavar="ENGINE=PATH")
    p.add_argument("--locate-from", type=Path, action="append", default=[], metavar="DATASET_ROOT",
                   help="find each engine's snapshot from the path and SHA-256 recorded in the traces")
    p.add_argument("--search-root", type=Path, help="where to look for a same-named copy of a recorded snapshot")
    p.add_argument("--shown", type=Path, help="extract directory whose snippets were shown to generators")
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("sample-prompts")
    p.add_argument("--prompts", type=Path, required=True, help="compliant-candidates.jsonl")
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("embed")
    p.add_argument("--pages", type=Path, required=True)
    p.add_argument("--view", choices=("qwen", "mistral"), required=True)
    p.add_argument("--map", type=Path, required=True, help="directory with readiness_embedding_map.json")
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--mntp-model")
    p.add_argument("--peft-model")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--attention-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--save-embeddings", action="store_true", help="also keep the full float32 vectors per shard")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("merge")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--pages", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("relocate")
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--qwen-projections", type=Path, required=True, help="final audit merged/qwen")
    p.add_argument("--mistral-projections", type=Path, required=True, help="final audit merged/mistral")
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--qwen-map", type=Path, help="maps/<qwen view>; with --qwen-embeddings replays the map")
    p.add_argument("--mistral-map", type=Path)
    p.add_argument("--qwen-embeddings", type=Path, help="final-audit/projections/qwen")
    p.add_argument("--mistral-embeddings", type=Path)
    p.add_argument("--fresh-qwen", type=Path)
    p.add_argument("--fresh-mistral", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("analyze")
    p.add_argument("--extract", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--mistral", type=Path, required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--workers", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("package")
    p.add_argument("--extract", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True, help="merged Qwen view embedded with --save-embeddings")
    p.add_argument("--mistral", type=Path, required=True)
    p.add_argument("--battery", type=Path, required=True)
    p.add_argument("--final-axis-map", type=Path, required=True)
    p.add_argument("--relocation", type=Path, required=True, help="relocation report with the fresh re-embedding check")
    p.add_argument("--output", type=Path, required=True)
    p = stages.add_parser("publish")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--path-in-repo")
    args = parser.parse_args(argv)
    return {"extract": extract, "corpus": corpus, "sample-prompts": sample_prompts, "embed": embed, "merge": merge,
            "relocate": relocate, "analyze": analyze, "package": package, "publish": publish}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
