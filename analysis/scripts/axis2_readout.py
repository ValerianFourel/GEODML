#!/usr/bin/env python3
"""Read the pipeline on a second direction of the frozen readiness maps: axis 2, with reliability checks and
random-direction specificity controls.

    python -m analysis.scripts.axis2_readout run --traces DIR --replay DIR --rows FILE --corpus DIR \
        --prompt-embeddings DIR --maps DIR --battery FILE --final-axis-map FILE [--prompt-records FILE] \
        [--calibration DIR] [--text-coordinates FILE] [--stored-chain FILE] --output DIR
    python -m analysis.scripts.axis2_readout texts --traces DIR --queries-qwen DIR --queries-mistral DIR \
        --answers-qwen DIR --answers-mistral DIR --answers-index FILE --output DIR

Placement. Every text is placed on axis 2 exactly as on axis 1: the L2-normalised embedding minus the map's
development mean, the raw coordinate on the view's second supervised axis a_v2 (the first two left singular vectors
of the ridge rubric coefficients are a_v1 and a_v2), the battery's development z-scaling, the battery's Procrustes
rotation of the Mistral view onto the Qwen view, the mean of the two views (consensus) and, for prompts, the
percentile rank among all embedded prompts with ties broken by prompt id; pages, queries and answers are interpolated
on the sorted consensus values of the prompts in the final axis map (`PromptScale`, as for axis 1). The sign of axis 2
is fixed so that higher means a higher evaluation rubric score: the sign of the evaluation column's loading on the
Qwen view's a_2 (the rotation carries the Mistral view onto the Qwen orientation).

Readout. Per generator x search method (natural condition, engines pooled), on the held-out keywords and on all
keywords: within-keyword slopes of each stage's mean position on the prompt's position, with the stage chain's own
functions (`steelman.chain`: R0, R, C, P, I, K on the common sample of answers citing at least one source) and the
same slope for the agent's queries Q and the answer text A when their coordinates are given (`texts` on the cluster).
200 keyword-bootstrap draws (seed 20261007, keyword codes in the funnel answers' order as in the chain run) and
200 within-keyword shuffles (seed 20261008). Two units: prompt percentile (axis 1 and axis 2) and SD units, where
the prompt's coordinate and the stage's mean coordinate are both divided by the SD of the prompts' coordinate, so a
slope of 1 means the stage moves as far along the direction as the prompt. Each slope raw and with the axis-1
position x and x^2 as controls. Specificity: five random unit directions per view in the orthogonal complement of
span(a_v1, a_v2) (seed 20261012), read the same way in SD units (pages and prompts only: the agent's queries and the
answers have stored axis coordinates, not embeddings).

Observational throughout: positions are measured properties of text; the prompt population was written to span
axis 1 only, so variation along axis 2 within a keyword is incidental. Page, query and answer positions describe
out-of-domain text with maps fitted on requests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.interpretability.pipeline import intent_stages as stages  # noqa: E402
from analysis.interpretability.pipeline import page_readiness_ordering as ordering  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402
from analysis.scripts import paper_numbers_fullrun as fmt  # noqa: E402  (the paper's macro formatting)
from analysis.steelman import chain as steelman  # noqa: E402
from analysis.steelman import tables as chain_tables  # noqa: E402

FORMAT_VERSION = "axis2-readout-v1"
VIEWS = ("qwen", "mistral")
CHAIN = chain_tables.CHAIN  # R0, R, C, P, I, K
TEXT_STAGES = ("Q", "A")
CELLS = (("llama4", "Parallel-Expansion-v1", "LP"), ("llama4", "Reactive-Snippet-Loop-v1", "LR"),
         ("qwen38", "Parallel-Expansion-v1", "QP"), ("qwen38", "Reactive-Snippet-Loop-v1", "QR"))
CELL_NAMES = {"LP": "Llama $\\cdot$ Parallel", "LR": "Llama $\\cdot$ Reactive", "QP": "Qwen $\\cdot$ Parallel",
              "QR": "Qwen $\\cdot$ Reactive"}
SPLITS = (("heldout", "held-out"), ("all", None))
RUBRICS = ("information_seeking_reverse", "evaluation", "selection_commitment", "action_implementation")
SEED, SHUFFLE_SEED, RANDOM_SEED = 20261007, 20261008, 20261012
AXES = ("axis1", "axis2")
PROMPT_FILE = "question_embeddings.restricted-local.npz"


def log(message: str, **fields) -> None:
    print(json.dumps({"time": readiness.now(), "message": message, **fields}, default=str), flush=True)


def read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- maps, alignment, coordinates

@dataclass
class FrozenMap:
    mean: np.ndarray          # (D,) mean of the normalised development embeddings
    axes: np.ndarray          # (2, D): a_1, a_2
    rubric: np.ndarray        # (D, 4) ridge coefficients of the rescaled rubric scores
    rubric_names: tuple
    map_id: str


def load_map(path: Path) -> FrozenMap:
    m = json.loads(Path(path).read_text())
    axes = np.asarray(m["supervised_subspace_axes"], float)
    if axes.shape[0] != 2:
        raise ValueError("the readiness map must hold exactly two supervised axes")
    return FrozenMap(np.asarray(m["embedding_mean"], float), axes, np.asarray(m["rubric_coefficient_matrix"], float),
                     tuple(m["rubric_names"]), str(m["map_id"]))


def rubric_loadings(fmap: FrozenMap) -> dict:
    """a_k^T W per rubric: the coefficient of each axis coordinate in the map's predicted rubric scores."""
    return {f"axis{k + 1}": dict(zip(fmap.rubric_names, map(float, fmap.axes[k] @ fmap.rubric))) for k in range(2)}


def evaluation_sign(fmap: FrozenMap) -> float:
    """+1 or -1: the sign that makes the a_2 coordinate rise with the map's predicted evaluation score."""
    loading = float(fmap.axes[1] @ fmap.rubric[:, fmap.rubric_names.index("evaluation")])
    if loading == 0:
        raise ValueError("axis 2 carries no evaluation loading; the sign is undefined")
    return 1.0 if loading > 0 else -1.0


@dataclass
class Alignment:
    rotation: np.ndarray
    ref_mean: np.ndarray
    ref_scale: np.ndarray
    cand_mean: np.ndarray
    cand_scale: np.ndarray
    battery: dict = field(default_factory=dict)


def load_alignment(battery: Path) -> Alignment:
    path = Path(battery)
    path = path / "readiness_robustness_battery.json" if path.is_dir() else path
    data = json.loads(path.read_text())
    a = data["cross_embedding_alignment"]
    return Alignment(np.asarray(a["orthogonal_rotation"], float), np.asarray(a["reference_development_mean"], float),
                     np.asarray(a["reference_development_scale"], float), np.asarray(a["candidate_development_mean"], float),
                     np.asarray(a["candidate_development_scale"], float), data)


def coordinates(matrix, mean: np.ndarray, directions: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """(n, k) coordinates on ``directions`` (k, D) of the L2-normalised rows minus the map mean, with the arithmetic of
    `readiness_prompt_population.project_text_embeddings` (float64: normalise, subtract the mean, project). ``matrix``
    may be a memory map; it is read in chunks."""
    out = []
    for start in range(0, len(matrix), chunk):
        block = np.asarray(matrix[start:start + chunk], dtype=np.float64)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if np.any(norms <= 1e-12) or not np.isfinite(block).all():
            raise ValueError("embeddings must be finite and nonzero")
        out.append((block / norms - mean) @ directions.T)
    return np.concatenate(out) if out else np.zeros((0, len(directions)))


def random_directions(axes: np.ndarray, count: int, seed: int, view_index: int) -> np.ndarray:
    """``count`` unit directions drawn from a standard normal and projected onto the orthogonal complement of
    span(a_1, a_2); the draw of view v uses the seed sequence (seed, v)."""
    basis, _ = np.linalg.qr(np.asarray(axes, float).T)  # (D, 2), orthonormal basis of the span
    rng = np.random.default_rng([seed, view_index])
    draws = rng.standard_normal((count, axes.shape[1]))
    draws -= (draws @ basis) @ basis.T
    draws -= (draws @ basis) @ basis.T  # second pass: orthogonal to machine precision
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


def views_z(raw_qwen: np.ndarray, raw_mistral: np.ndarray, al: Alignment) -> tuple[np.ndarray, np.ndarray]:
    """Development z of the Qwen view and the Mistral view rotated onto it, as `_aligned_projection_rows`."""
    zq = (np.asarray(raw_qwen, float) - al.ref_mean) / al.ref_scale
    zm = ((np.asarray(raw_mistral, float) - al.cand_mean) / al.cand_scale) @ al.rotation
    return zq, zm


@dataclass
class Placed:
    """Axis z of both views (axis 2 signed) and their consensus; optional raw coordinates on random directions."""

    zq: np.ndarray
    zm: np.ndarray
    consensus: np.ndarray
    random: dict = field(default_factory=dict)  # view -> (n, k)

    def __len__(self) -> int:
        return len(self.consensus)


def place(raw_qwen, raw_mistral, al: Alignment, sign: float, random: dict | None = None) -> Placed:
    zq, zm = views_z(raw_qwen, raw_mistral, al)
    s = np.array([1.0, sign])
    zq, zm = zq * s, zm * s
    return Placed(zq, zm, ordering.consensus(zq, zm), dict(random or {}))


def percentile_rank(values: np.ndarray, ids) -> np.ndarray:
    """rank / (N - 1), ranks from 0, ties broken by id (the final audit's rule for x)."""
    values = np.asarray(values, float)
    order = np.lexsort((np.asarray(ids), values))
    out = np.empty(len(values))
    out[order] = np.arange(len(values)) / max(1, len(values) - 1)
    return out


def prompt_scale(values: np.ndarray) -> ordering.PromptScale:
    """The page percentile rule of axis 1 (`PromptScale`, which reads the key consensus_axis_1_z) on any axis."""
    return ordering.PromptScale([{"consensus_axis_1_z": float(v)} for v in values])


# directions: the two consensus axes (percentile and SD units), each axis per view, random directions per view
def direction_names(random_count: int) -> list[str]:
    return [*AXES, "axis1_qwen", "axis1_mistral", "axis2_qwen", "axis2_mistral",
            *(f"random_{view}_{k + 1}" for view in VIEWS for k in range(random_count))]


def units_of(direction: str) -> tuple[str, ...]:
    return ("percentile", "sd") if direction in AXES else ("sd",)


def variants_of(direction: str) -> tuple[str, ...]:
    """Axis 1 is read raw only: x and x^2 are functions of its own position."""
    return ("raw",) if direction.startswith("axis1") else ("raw", "controls")


def coordinate(placed: Placed, direction: str) -> np.ndarray | None:
    """The unscaled coordinate of each text on a direction (None where the texts have no such coordinate)."""
    if direction in AXES:
        return placed.consensus[:, AXES.index(direction)]
    if direction.startswith("axis"):
        axis, view = direction.split("_")
        return (placed.zq if view == "qwen" else placed.zm)[:, int(axis[-1]) - 1]
    _, view, k = direction.split("_")
    block = placed.random.get(view)
    return None if block is None else block[:, int(k) - 1]


class Scales:
    """Per direction: the prompts' SD (SD units) and, for the two axes, the prompts' percentile scale."""

    def __init__(self, prompts: Placed, prompt_ids, scale_mask: np.ndarray, directions: list[str]):
        self.sd = {d: float(np.std(coordinate(prompts, d))) for d in directions}
        self.percentile_scale = {a: prompt_scale(coordinate(prompts, a)[scale_mask]) for a in AXES}
        self.prompt_percentile = {a: percentile_rank(coordinate(prompts, a), prompt_ids) for a in AXES}

    def prompt_values(self, prompts: Placed, direction: str, unit: str) -> np.ndarray:
        if unit == "percentile":
            return self.prompt_percentile[direction]
        return coordinate(prompts, direction) / self.sd[direction]

    def text_values(self, texts: Placed, direction: str, unit: str):
        value = coordinate(texts, direction)
        if value is None:
            return None, None
        if unit == "percentile":
            return self.percentile_scale[direction].percentile(value)
        return value / self.sd[direction], None


def agreement(left: np.ndarray, right: np.ndarray) -> dict:
    left, right = np.asarray(left, float), np.asarray(right, float)
    ok = np.isfinite(left) & np.isfinite(right)
    if ok.sum() < 3 or np.std(left[ok]) <= 1e-12 or np.std(right[ok]) <= 1e-12:
        return {"texts": int(ok.sum()), "spearman": None, "pearson": None}
    return {"texts": int(ok.sum()), "spearman": ordering._spearman(left[ok], right[ok]),
            "pearson": float(np.corrcoef(left[ok], right[ok])[0, 1])}


def view_agreement(placed: Placed, keep: np.ndarray | None = None) -> dict:
    keep = np.ones(len(placed), bool) if keep is None else keep
    return {axis: agreement(placed.zq[keep, k], placed.zm[keep, k]) for k, axis in enumerate(AXES)}


# ---------------------------------------------------------------- prompts and pages from embeddings

def load_prompt_embeddings(root: Path, maps: dict, extra: dict) -> tuple[list[str], dict]:
    """Prompt ids (sorted) and per view the (n, 2 + k) coordinates on the view's axes and its random directions."""
    ids, raw = None, {}
    for view in VIEWS:
        directions = np.vstack([maps[view].axes, extra[view]])
        chunks, view_ids = [], []
        shards = sorted((Path(root) / view).glob(f"shard-*/{PROMPT_FILE}"))
        if not shards:
            raise ValueError(f"no prompt embedding shards under {Path(root) / view}")
        for path in shards:
            with np.load(path, allow_pickle=False) as archive:
                view_ids += [str(i) for i in archive["candidate_ids"]]
                chunks.append(coordinates(archive["embeddings"], maps[view].mean, directions))
        order = np.argsort(np.asarray(view_ids), kind="stable")
        view_ids = [view_ids[i] for i in order]
        if len(set(view_ids)) != len(view_ids):
            raise ValueError(f"a prompt is embedded twice in the {view} view")
        if ids is not None and view_ids != ids:
            raise ValueError("the two views embed different prompts")
        ids, raw[view] = view_ids, np.concatenate(chunks)[order]
    return ids, raw


def load_page_embeddings(corpus: Path, maps: dict, extra: dict) -> dict:
    out = {}
    for view in VIEWS:
        directions = np.vstack([maps[view].axes, extra[view]])
        vectors = np.load(Path(corpus) / readiness.EMBEDDING_FILES[view], mmap_mode="r", allow_pickle=False)
        out[view] = coordinates(vectors, maps[view].mean, directions)
    return out


def split_random(raw: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    return raw["qwen"][:, :2], raw["mistral"][:, :2], {view: raw[view][:, 2:] for view in VIEWS}


# ---------------------------------------------------------------- trace tables

def _pd():
    import pandas as pd
    return pd


def _pq():
    import pyarrow.parquet as pq
    return pq


@dataclass
class Traces:
    answers: object           # trace_answers frame (trace order)
    funnel_order: np.ndarray  # trace index of each funnel answer, in the funnel extract's order
    offsets: np.ndarray       # stage_items CSR by funnel answer
    row: np.ndarray
    scored: np.ndarray
    shown: np.ndarray
    cited: np.ndarray


def load_traces(directory: Path) -> Traces:
    pd, pq = _pd(), _pq()
    directory = Path(directory)
    answers = pq.read_table(directory / "trace_answers.parquet", columns=[
        "answer_id", "model", "engine", "method", "condition", "prompt_id", "keyword", "split", "x",
        *(f"chain_{s}" for s in CHAIN), "Q", "A"]).to_pandas()
    for column in ("answer_id", "model", "engine", "method", "condition", "prompt_id", "keyword", "split"):
        answers[column] = answers[column].astype(str)
    index = pd.Index(answers["answer_id"])
    if not index.is_unique:
        raise ValueError("answer ids are not unique in trace_answers")
    table = pq.read_table(directory / "stage_items.parquet", columns=["answer_id", "row_id", "scored", "shown_slot", "cited_rank"])
    column = table.column("answer_id")
    first, mapping, parts = None, None, []
    for chunk in column.chunks:
        if first is None or not chunk.dictionary.equals(first):
            first = chunk.dictionary
            mapping = index.get_indexer(first.to_pandas())
            if (mapping < 0).any():
                raise ValueError("stage_items holds answers missing from trace_answers")
        parts.append(mapping[chunk.indices.to_numpy(zero_copy_only=False)])
    who = np.concatenate(parts).astype(np.int64)
    start = np.flatnonzero(np.r_[True, who[1:] != who[:-1]])
    order = who[start]
    if len(np.unique(order)) != len(order):
        raise ValueError("stage_items rows of an answer are not contiguous")
    offsets = np.r_[start, len(who)].astype(np.int64)
    as_np = lambda name, dtype: table.column(name).to_numpy().astype(dtype)  # noqa: E731
    return Traces(answers, order, offsets, as_np("row_id", np.int64), as_np("scored", np.int64),
                  as_np("shown_slot", np.int64), as_np("cited_rank", np.int64))


@dataclass
class Replay:
    prompts: list
    rows: np.ndarray


def load_replay(directory: Path) -> Replay:
    with np.load(Path(directory) / "replay.npz", allow_pickle=False) as z:
        rows = z["rows"]
    return Replay(list(read_jsonl(Path(directory) / "prompts.jsonl.gz")), rows)


def chain_values(tr: Traces, replay: Replay, row_value: np.ndarray) -> dict:
    """R0 ... K per answer (trace order) with the stage chain's own functions, for one value per snapshot row."""
    funnel = chain_tables.stage_values(tr.offsets, tr.row, tr.scored, tr.shown, tr.cited, row_value)
    prompt_ids = tr.answers["prompt_id"].to_numpy()[tr.funnel_order]
    engines = tr.answers["engine"].to_numpy()[tr.funnel_order]
    funnel["R0"] = chain_tables.replay_values(prompt_ids, engines, replay.prompts, replay.rows, row_value)
    n = len(tr.answers)
    out = {}
    for name in CHAIN:
        values = np.full(n, np.nan)
        values[tr.funnel_order] = funnel[name]
        out[name] = values
    return out


def first_appearance_codes(values: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, int]:
    """Codes by first appearance along ``order`` (the funnel answers' order, as `funnel_study.answer_arrays`)."""
    code = {}
    for value in values[order]:
        code.setdefault(value, len(code))
    return np.asarray([code[v] for v in values], np.int64), len(code)


# ---------------------------------------------------------------- queries and answers (cluster export)

def read_projections(directory: Path) -> dict:
    """candidate id -> (text_sha256, raw axis 1, raw axis 2) from a merged `question_projections.jsonl`."""
    directory = Path(directory)
    path = directory / "question_projections.jsonl"
    if not path.exists():
        path = directory / "question_projections.jsonl.gz"
    out = {}
    for row in read_jsonl(path):
        p = row["projection"]
        out[str(row["candidate_id"])] = (str(p["text_sha256"]), float(p["raw_axis_1"]), float(p["raw_axis_2"]))
    return out


def export_texts(args) -> int:
    """Per-text raw coordinates of both views for the agent's queries (by release query id) and the answers (by
    release answer id), from the full run's merged projections. No embedding, no model: CPU only."""
    pd, pq = _pd(), _pq()
    import pyarrow as pa
    started = time.time()
    partial = readiness.new_directory(args.output)
    traces = Path(args.traces)
    queries = pq.read_table(traces / "queries.parquet", columns=["query_id", "text_sha256"]).to_pandas()
    columns = {"kind": [], "key": [], "text_id": []}
    raw = {f"{view}_raw_axis_{k}": [] for view in VIEWS for k in (1, 2)}
    counts = {}
    by_sha = {}
    for view, directory in (("qwen", args.queries_qwen), ("mistral", args.queries_mistral)):
        projections = read_projections(directory)
        sha = {}
        for candidate, (digest, a1, a2) in projections.items():
            if digest in sha and sha[digest][1:] != (a1, a2):
                raise ValueError(f"one query text with two {view} projections")
            sha[digest] = (candidate, a1, a2)
        by_sha[view] = sha
        counts[f"queries_{view}_projections"] = len(projections)
    missing = 0
    for query_id, digest in zip(queries["query_id"].to_numpy(), queries["text_sha256"].astype(str)):
        hit = [by_sha[view].get(digest) for view in VIEWS]
        if any(h is None for h in hit):
            missing += 1
            continue
        columns["kind"].append("query")
        columns["key"].append(str(int(query_id)))
        columns["text_id"].append(hit[0][0])
        for view, h in zip(VIEWS, hit):
            raw[f"{view}_raw_axis_1"].append(h[1])
            raw[f"{view}_raw_axis_2"].append(h[2])
    counts.update(queries=len(queries), queries_missing=missing)
    log("queries joined", **counts)
    index = {}
    for row in read_jsonl(args.answers_index):
        index.setdefault(tuple(str(row[k]) for k in ("model", "prompt_id", "method", "engine", "condition")), str(row["answer_id"]))
    projections = {view: read_projections(directory) for view, directory in (("qwen", args.answers_qwen), ("mistral", args.answers_mistral))}
    answers = pq.read_table(traces / "trace_answers.parquet", columns=["answer_id", "model", "prompt_id", "method", "engine", "condition"]).to_pandas()
    joined = unmatched = 0
    for row in answers.astype(str).itertuples(index=False):
        text = index.get((row.model, row.prompt_id, row.method, row.engine, row.condition))
        if text is None:
            continue
        hit = [projections[view].get(text) for view in VIEWS]
        if any(h is None for h in hit):
            unmatched += 1
            continue
        joined += 1
        columns["kind"].append("answer")
        columns["key"].append(row.answer_id)
        columns["text_id"].append(text)
        for view, h in zip(VIEWS, hit):
            raw[f"{view}_raw_axis_1"].append(h[1])
            raw[f"{view}_raw_axis_2"].append(h[2])
    counts.update(answers_index_rows=len(index), answers_joined=joined, answers_without_projection=unmatched,
                  answer_texts=len({t for t, k in zip(columns["text_id"], columns["kind"]) if k == "answer"}))
    log("answers joined", **counts)
    table = pa.table({**columns, **{k: np.asarray(v, float) for k, v in raw.items()}})
    pq.write_table(table, partial / "text_coordinates.parquet", compression="zstd")
    inputs = {"queries_table": readiness.identity(traces / "queries.parquet"),
              "trace_answers": readiness.identity(traces / "trace_answers.parquet"),
              "answers_index": readiness.identity(args.answers_index)}
    for name, directory in (("queries_qwen", args.queries_qwen), ("queries_mistral", args.queries_mistral),
                            ("answers_qwen", args.answers_qwen), ("answers_mistral", args.answers_mistral)):
        inputs[name] = readiness.identity(Path(directory) / "projection_manifest.json")
    readiness.write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "stage": "texts", "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "inputs": inputs, "counts": counts, "rows": table.num_rows, "seconds": round(time.time() - started, 1),
        "file": readiness.identity(partial / "text_coordinates.parquet") | {"path": "text_coordinates.parquet"}})
    partial.rename(Path(args.output).resolve())
    log("texts written", output=str(args.output), rows=table.num_rows)
    return 0


@dataclass
class Texts:
    query_ids: np.ndarray     # release query id of each row of ``queries``
    queries: Placed
    answer_ids: np.ndarray    # release answer id of each row of ``answers``
    answer_text_ids: np.ndarray
    answers: Placed


def load_texts(path: Path, al: Alignment, sign: float) -> Texts:
    t = _pq().read_table(path).to_pandas()
    out = {}
    for kind in ("query", "answer"):
        g = t[t["kind"] == kind]
        raw = {view: g[[f"{view}_raw_axis_1", f"{view}_raw_axis_2"]].to_numpy(float) for view in VIEWS}
        out[kind] = (g["key"].astype(str).to_numpy(), g["text_id"].astype(str).to_numpy(), place(raw["qwen"], raw["mistral"], al, sign))
    return Texts(out["query"][0].astype(np.int64), out["query"][2], out["answer"][0], out["answer"][1], out["answer"][2])


def query_layout(traces_dir: Path, answer_index, n_answers: int) -> tuple[np.ndarray, np.ndarray]:
    """The agent's queries of every answer as CSR (offsets by trace answer, query id of each search)."""
    pq = _pq()
    table = pq.read_table(Path(traces_dir) / "answer_queries.parquet", columns=["answer_id", "query_id"])
    column = table.column("answer_id")
    first, mapping, parts = None, None, []
    for chunk in column.chunks:
        if first is None or not chunk.dictionary.equals(first):
            first = chunk.dictionary
            mapping = answer_index.get_indexer(first.to_pandas())
        parts.append(mapping[chunk.indices.to_numpy(zero_copy_only=False)])
    who = np.concatenate(parts).astype(np.int64)
    qid = table.column("query_id").to_numpy().astype(np.int64)
    if (who < 0).any():
        raise ValueError("answer_queries holds answers missing from trace_answers")
    order = np.argsort(who, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(who, minlength=n_answers))]
    return offsets, qid[order]


def query_means(layout: tuple[np.ndarray, np.ndarray], query_value: np.ndarray) -> np.ndarray:
    """Q per answer: the mean over its searches of the query's value (`intent_stages.stage_values`' Q)."""
    offsets, qid = layout
    return stages.csr_mean(offsets, query_value[qid])


# ---------------------------------------------------------------- calibration items

def load_calibration(root: Path, al: Alignment, sign: float) -> dict | None:
    root = Path(root)
    paths = {view: root / "maps" / view / "readiness_supervised_subspace_coordinates.jsonl" for view in VIEWS}
    labels_path = root / "maps" / "qwen" / "readiness_consensus.jsonl"
    if not all(p.exists() for p in [*paths.values(), labels_path]):
        return None
    rows = {view: {r["item_id"]: r for r in read_jsonl(p)} for view, p in paths.items()}
    ids = sorted(rows["qwen"])
    if set(ids) != set(rows["mistral"]):
        raise ValueError("the calibration views hold different items")
    raw = {view: np.asarray([[rows[view][i]["axis_1"], rows[view][i]["axis_2"]] for i in ids], float) for view in VIEWS}
    placed = place(raw["qwen"], raw["mistral"], al, sign)
    labels = {r["item_id"]: r for r in read_jsonl(labels_path)}
    meta = rows["qwen"]
    texts = {}
    for part in sorted((root / "extra" / "axisgeo-unified-hf-dataset" / "data" / "likert_prompts").glob("*.parquet")):
        frame = _pq().read_table(part, columns=["item_id", "text"]).to_pandas()
        texts.update(zip(frame["item_id"].astype(str), frame["text"].astype(str)))
    return {"ids": ids, "placed": placed, "split": np.asarray([meta[i]["split"] for i in ids]),
            "usable": np.asarray([bool(meta[i]["usable_for_axis"]) for i in ids]),
            "source": np.asarray([meta[i]["source_name"] for i in ids]), "labels": labels, "texts": texts,
            "inputs": {name: readiness.identity(p) for name, p in (*((f"coordinates_{v}", p) for v, p in paths.items()),
                                                                     ("consensus_labels", labels_path))}}


def rubric_table(cal: dict, keep: np.ndarray) -> dict:
    """Spearman of each axis (consensus and each view) with the consensus label and each rubric score."""
    ids = [i for i, k in zip(cal["ids"], keep) if k]
    lab = cal["labels"]

    def column(key, transform=lambda v: v):
        return np.asarray([transform(lab[i][key]) if i in lab and lab[i].get(key) is not None else np.nan for i in ids], float)

    labels = {"overall": column("overall_readiness_0_100"),
              "information_seeking_reverse": column("information_seeking_1_7", lambda v: -v),
              "evaluation": column("evaluation_1_7"), "selection_commitment": column("selection_commitment_1_7"),
              "action_implementation": column("action_implementation_1_7")}
    out = {"items": int(keep.sum())}
    for name, values in (("consensus", cal["placed"].consensus), ("qwen", cal["placed"].zq), ("mistral_aligned", cal["placed"].zm)):
        for k, axis in enumerate(AXES):
            v = values[keep, k]
            out[f"{axis}_{name}"] = {label: agreement(v, y)["spearman"] for label, y in labels.items()}
    return out


# ---------------------------------------------------------------- prompt-level diagnostics

def group_mean(values: np.ndarray, keyword: np.ndarray) -> np.ndarray:
    size = int(keyword.max()) + 1
    return (np.bincount(keyword, values, minlength=size) / np.maximum(np.bincount(keyword, minlength=size), 1))[keyword]


def within_keyword_correlation(a: np.ndarray, b: np.ndarray, keyword: np.ndarray) -> float:
    a = np.asarray(a, float) - group_mean(np.asarray(a, float), keyword)
    b = np.asarray(b, float) - group_mean(np.asarray(b, float), keyword)
    denominator = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denominator) if denominator > 0 else float("nan")


def within_share(values: np.ndarray, keyword: np.ndarray) -> float:
    values = np.asarray(values, float)
    total = float(np.sum((values - values.mean()) ** 2))
    return float(np.sum((values - group_mean(values, keyword)) ** 2) / total) if total > 0 else float("nan")


def keyword_spread(values: np.ndarray, keyword: np.ndarray, threshold: float = 0.3) -> dict:
    """Per keyword: SD, interquartile range and range of the values, summarised over keywords."""
    order = np.argsort(keyword, kind="stable")
    groups = np.split(np.asarray(values, float)[order], np.flatnonzero(np.diff(keyword[order])) + 1)
    sd = np.asarray([g.std() for g in groups])
    iqr = np.asarray([np.subtract(*np.percentile(g, [75, 25])) for g in groups])
    span = np.asarray([g.max() - g.min() for g in groups])

    def summary(v):
        return {"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
                "mean": float(v.mean())}

    return {"keywords": len(groups), "sd": summary(sd), "iqr": summary(iqr), "range": summary(span),
            "share_range_at_least": {"threshold": threshold, "share": float(np.mean(span >= threshold))}}


def quadratic_within(x: np.ndarray, y: np.ndarray, keyword: np.ndarray) -> dict:
    """Within-keyword least squares of y on x and x^2: coefficients, vertex and within-keyword R^2."""
    X = np.column_stack([x, x * x])
    beta = steelman.within_keyword_ols(X, y, keyword)
    Xt = X - np.column_stack([group_mean(X[:, j], keyword) for j in range(2)])
    yt = y - group_mean(y, keyword)
    residual = yt - Xt @ beta
    linear = steelman.within_keyword_ols(x[:, None], y, keyword)[0]
    return {"x": float(beta[0]), "x_squared": float(beta[1]),
            "vertex_x": float(-beta[0] / (2 * beta[1])) if beta[1] != 0 else None,
            "within_r2": float(1 - np.sum(residual ** 2) / np.sum(yt ** 2)), "linear_slope": float(linear)}


def prompt_diagnostics(x: np.ndarray, x2: np.ndarray, c2: np.ndarray, keyword: np.ndarray, writer: np.ndarray | None) -> dict:
    out = {"prompts": int(len(x)), "keywords": int(len(np.unique(keyword))),
           "x2_vs_x": {"spearman": agreement(x, x2)["spearman"], "pearson": agreement(x, x2)["pearson"],
                       "within_keyword_pearson": within_keyword_correlation(x, x2, keyword),
                       "within_keyword_slope": float(steelman.within_keyword_ols(x[:, None], x2, keyword)[0]),
                       "quadratic": quadratic_within(x, x2, keyword)},
           "binned": stages.stage_curves(x, keyword, {"x2": x2, "consensus_z2": c2}, np.ones(len(x), bool), bins=20),
           "spread": {"x2": keyword_spread(x2, keyword), "x": keyword_spread(x, keyword)},
           "within_keyword_variance_share": {"x2": within_share(x2, keyword), "x": within_share(x, keyword)}}
    if writer is not None and np.isfinite(writer).all() and len(np.unique(writer)) == 2:
        out["writer"] = {"share_second_writer": float(writer.mean()),
                         "x2_within_keyword_pearson": within_keyword_correlation(writer, x2, keyword),
                         "x_within_keyword_pearson": within_keyword_correlation(writer, x, keyword),
                         "x2_difference_within_keyword": float(steelman.within_keyword_ols(writer[:, None], x2, keyword)[0]),
                         "x_difference_within_keyword": float(steelman.within_keyword_ols(writer[:, None], x, keyword)[0])}
    return out


def load_prompt_records(path: Path | None) -> dict:
    """prompt id -> question, keyword, writer from the archived snapshot ({"prompt": {...}}) or a flat population file."""
    out = {}
    if path is None:
        return out
    for r in read_jsonl(path):
        p = r.get("prompt", r)
        pid = str(r.get("prompt_id") or p.get("candidate_id"))
        out[pid] = {"question": p.get("question"), "keyword": p.get("keyword"), "writer": p.get("generator_id")}
    return out


# ---------------------------------------------------------------- readout

def term_slope(y, x, keyword, prompt, *, draws, shuffles, controls) -> dict:
    """Within-keyword slope of one per-answer series on x (optionally with controls), as `steelman.chain.simple_slope`."""
    ok = np.isfinite(y)
    if ok.sum() < 10:
        return {"skipped": int(ok.sum())}
    k, yy = keyword[ok], np.asarray(y, float)[ok]
    ctrl = None if controls is None else controls[ok]
    X = steelman.design(x[ok], ctrl)
    estimate = float(steelman.within_keyword_ols(X, yy, k)[0])
    boot = [float(steelman.within_keyword_ols(X, yy, k, d[k])[0]) for d in draws]
    null = [float(steelman.within_keyword_ols(steelman.design(s[prompt[ok]], ctrl), yy, k)[0]) for s in shuffles] if shuffles else None
    out = steelman.summarise(estimate, boot, null)
    out["answers"] = int(ok.sum())
    return out


def readout(values: dict, text_values: dict, x, keyword, prompt, *, draws, shuffles, controls) -> dict:
    result = steelman.chain({n: values[n] for n in CHAIN}, x, keyword, prompt, draws=draws, shuffles=shuffles, controls=controls)
    for name, y in text_values.items():
        result["slopes"][name] = term_slope(y, x, keyword, prompt, draws=draws, shuffles=shuffles, controls=controls)
    return result


def run_readout(masks: dict, stage_by: dict, text_by: dict, prompt_by: dict, x, keyword, prompt, prompt_keyword,
                *, draws, permutations: int, shuffle_seed: int, directions: list[str]) -> dict:
    """Every (split, cell, direction, unit, variant) of the readout. ``stage_by[(d, u)]`` holds per-answer chain values,
    ``text_by[(d, u)]`` Q and A, ``prompt_by[(d, u)]`` the prompt-code indexed position (for shuffles and per-answer x)."""
    out = {}
    shuffles = {}
    for (direction, unit), prompt_x in prompt_by.items():
        if direction == "axis2" and unit == "percentile" and permutations:
            shuffles[(direction, unit)] = stages.shuffle_draws(prompt_x, prompt_keyword, permutations, shuffle_seed)
    for split, cells in masks.items():
        out[split] = {}
        for cell, mask in cells.items():
            k, p = keyword[mask], prompt[mask]
            controls = np.column_stack([x[mask], x[mask] ** 2])
            entry = {"answers": int(mask.sum()), "keywords": int(len(np.unique(k))), "directions": {}}
            started = time.time()
            for direction in directions:
                entry["directions"][direction] = {}
                for unit in units_of(direction):
                    xd = prompt_by[(direction, unit)][p]
                    vals = {n: v[mask] for n, v in stage_by[(direction, unit)].items()}
                    texts = {n: v[mask] for n, v in text_by.get((direction, unit), {}).items()}
                    entry["directions"][direction][unit] = {}
                    for variant in variants_of(direction):
                        use = shuffles.get((direction, unit), []) if variant == "raw" else []
                        entry["directions"][direction][unit][variant] = readout(
                            vals, texts, xd, k, p, draws=draws, shuffles=use, controls=None if variant == "raw" else controls)
            log("readout", split=split, cell=cell, answers=entry["answers"], seconds=round(time.time() - started, 1))
            out[split][cell] = entry
    return out


def random_summary(readout_cell: dict, random_names: list[str], stage: str, variant: str) -> dict:
    values = [readout_cell["directions"][d]["sd"][variant]["slopes"].get(stage, {}).get("estimate") for d in random_names]
    values = np.asarray([v for v in values if v is not None], float)
    if not len(values):
        return {}
    return {"median": float(np.median(values)), "max_abs": float(np.max(np.abs(values))), "min": float(values.min()),
            "max": float(values.max()), "directions": int(len(values))}


def stored_chain_check(readout_split: dict, stored_path: Path | None) -> dict | None:
    """The axis-1 percentile readout against the stored stage chain of the same split (estimates and intervals)."""
    if stored_path is None or not Path(stored_path).exists():
        return None
    stored = json.loads(Path(stored_path).read_text())["strata"]
    worst, compared = 0.0, 0
    per_cell = {}
    for model, method, code in CELLS:
        name = f"{model} · {method.split('-')[0]}"
        if name not in stored or code not in readout_split:
            continue
        mine = readout_split[code]["directions"]["axis1"]["percentile"]["raw"]
        theirs = stored[name]["chain"]["common"]
        diff = max(abs(mine["answers"] - theirs["answers"]),
                   *(abs(mine["slopes"][s]["estimate"] - theirs["slopes"][s]["estimate"]) for s in CHAIN),
                   *(abs(a - b) for s in CHAIN for a, b in zip(mine["slopes"][s]["ci95"], theirs["slopes"][s]["ci95"])))
        per_cell[code] = diff
        worst = max(worst, diff)
        compared += 1
    return {"stored": str(stored_path), "cells_compared": compared, "max_abs_difference": worst, "per_cell": per_cell}


# ---------------------------------------------------------------- outputs

def _stage_macro(stage: str) -> str:
    return {"R0": "Rz"}.get(stage, stage)


def _cell(entry: dict, direction: str, unit: str, variant: str, stage: str) -> dict | None:
    try:
        value = entry["directions"][direction][unit][variant]["slopes"][stage]
    except KeyError:
        return None
    return value if "estimate" in value else None


def write_macros(results: dict, path: Path) -> list[str]:
    """numbers-axis2.tex: one macro per number the appendix quotes, and the table as \\AxTwoStageTable."""
    lines = []

    def m(name: str, value: str, src: str):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % src: {src}")

    def r3(v):
        return "--" if v is None else fmt.neg(f"{v:.3f}")

    def r2(v):
        return "--" if v is None else fmt.neg(f"{v:.2f}")

    pos = results["positions"]
    m("AxTwoViewPrompts", r3(pos["prompts"]["view_agreement"]["axis2"]["spearman"]), "positions.prompts.view_agreement.axis2.spearman")
    m("AxTwoViewPages", r3(pos["pages"]["view_agreement"]["axis2"]["spearman"]), "positions.pages.view_agreement.axis2.spearman")
    m("AxTwoViewPagesPearson", r3(pos["pages"]["view_agreement"]["axis2"]["pearson"]), "positions.pages.view_agreement.axis2.pearson")
    m("AxOneViewPagesSpearman", r3(pos["pages"]["view_agreement"]["axis1"]["spearman"]), "positions.pages.view_agreement.axis1.spearman")
    for kind, name in (("queries", "Queries"), ("answers", "Answers")):
        if kind in pos:
            m(f"AxTwoView{name}", r3(pos[kind]["view_agreement"]["axis2"]["spearman"]), f"positions.{kind}.view_agreement.axis2.spearman")
    cal = results.get("calibration")
    if cal:
        m("AxTwoViewCalib", r3(cal["view_agreement_confirmation"]["axis2"]["spearman"]), "calibration.view_agreement_confirmation.axis2.spearman")
        m("AxTwoViewCalibPearson", r3(cal["view_agreement_confirmation"]["axis2"]["pearson"]), "calibration.view_agreement_confirmation.axis2.pearson")
        rub = cal["rubrics_confirmation"]["axis2_consensus"]
        for key, name in (("evaluation", "Eval"), ("selection_commitment", "Sel"), ("action_implementation", "Act"),
                          ("information_seeking_reverse", "Info"), ("overall", "Overall")):
            m(f"AxTwoRho{name}", r2(rub[key]), f"calibration.rubrics_confirmation.axis2_consensus.{key} (Spearman)")
        m("NAxTwoCalibConfirm", fmt.num(cal["rubrics_confirmation"]["items"]), "calibration.rubrics_confirmation.items")
    d = results["prompt_diagnostics"]["all"]
    m("AxTwoRhoX", r2(d["x2_vs_x"]["spearman"]), "prompt_diagnostics.all.x2_vs_x.spearman")
    m("AxTwoRhoXWithin", r2(d["x2_vs_x"]["within_keyword_pearson"]), "prompt_diagnostics.all.x2_vs_x.within_keyword_pearson")
    quad = d["x2_vs_x"]["quadratic"]
    m("AxTwoQuadRsq", r2(quad["within_r2"]), "prompt_diagnostics.all.x2_vs_x.quadratic.within_r2")
    m("AxTwoQuadVertex", r2(quad["vertex_x"]), "prompt_diagnostics.all.x2_vs_x.quadratic.vertex_x")
    means = [v for v in d["binned"]["terms"]["x2"]["mean"] if v is not None]
    peak = int(np.nanargmax(np.asarray([np.nan if v is None else v for v in d["binned"]["terms"]["x2"]["mean"]], float)))
    m("AxTwoBinFirst", r2(means[0]), "prompt_diagnostics.all.binned.terms.x2.mean[0] (x below 0.05)")
    m("AxTwoBinLast", r2(means[-1]), "prompt_diagnostics.all.binned.terms.x2.mean[-1] (x at least 0.95)")
    m("AxTwoBinPeak", r2(d["binned"]["terms"]["x2"]["mean"][peak]), "prompt_diagnostics.all.binned.terms.x2.mean (largest bin mean)")
    m("AxTwoBinPeakLow", f"{d['binned']['bin_lower'][peak]:.2f}", "prompt_diagnostics.all.binned.bin_lower[peak]")
    m("AxTwoBinPeakHigh", f"{d['binned']['bin_upper'][peak]:.2f}", "prompt_diagnostics.all.binned.bin_upper[peak]")
    for axis, name in (("x2", "Two"), ("x", "One")):
        sp = d["spread"][axis]
        m(f"Ax{name}KwSD", r2(sp["sd"]["median"]), f"prompt_diagnostics.all.spread.{axis}.sd.median")
        m(f"Ax{name}KwIQR", r2(sp["iqr"]["median"]), f"prompt_diagnostics.all.spread.{axis}.iqr.median")
        m(f"Ax{name}KwRange", r2(sp["range"]["median"]), f"prompt_diagnostics.all.spread.{axis}.range.median")
        m(f"Ax{name}KwRangeShare", fmt.pct(sp["share_range_at_least"]["share"]), f"prompt_diagnostics.all.spread.{axis}.share_range_at_least (0.3)")
        m(f"Ax{name}WithinVar", fmt.pct(d["within_keyword_variance_share"][axis]), f"prompt_diagnostics.all.within_keyword_variance_share.{axis}")
    if "writer" in d:
        m("AxTwoWriterR", r2(d["writer"]["x2_within_keyword_pearson"]), "prompt_diagnostics.all.writer.x2_within_keyword_pearson")
        m("AxOneWriterR", r2(d["writer"]["x_within_keyword_pearson"]), "prompt_diagnostics.all.writer.x_within_keyword_pearson")
    m("NAxTwoRandom", str(results["settings"]["random_directions_per_view"] * len(VIEWS)), "settings.random_directions_per_view x 2 views")
    ro = results["readout"]["heldout"]
    for code in ro:
        e = ro[code]
        m(f"AxTwoN{code}", fmt.num(e["directions"]["axis2"]["percentile"]["controls"]["answers"]), f"readout.heldout.{code}...controls.answers")
        for stage in (*TEXT_STAGES, *CHAIN):
            v = _cell(e, "axis2", "percentile", "controls", stage)
            if v is None:
                continue
            m(f"AxTwoSl{_stage_macro(stage)}{code}", fmt.slope(v["estimate"]), f"readout.heldout.{code}.directions.axis2.percentile.controls.slopes.{stage}")
            m(f"AxTwoCISl{_stage_macro(stage)}{code}", fmt.ci(*v["ci95"]), f"... {stage} ci95")
            raw = _cell(e, "axis2", "percentile", "raw", stage)
            m(f"AxTwoRawSl{_stage_macro(stage)}{code}", fmt.slope(raw["estimate"]), f"readout.heldout.{code}.directions.axis2.percentile.raw.slopes.{stage}")
            for unit_d, variant, name in (("axis1", "raw", "AxOneSd"), ("axis2", "controls", "AxTwoSd")):
                w = _cell(e, unit_d, "sd", variant, stage)
                if w is not None:
                    m(f"{name}{_stage_macro(stage)}{code}", fmt.slope(w["estimate"], 2), f"readout.heldout.{code}.directions.{unit_d}.sd.{variant}.slopes.{stage}")
            rnd = results["random_summary"]["heldout"][code].get(stage, {})
            if rnd:
                m(f"RndSdMed{_stage_macro(stage)}{code}", fmt.slope(rnd["median"], 2), f"random_summary.heldout.{code}.{stage}.median")
                m(f"RndSdMax{_stage_macro(stage)}{code}", f"{rnd['max_abs']:.2f}", f"random_summary.heldout.{code}.{stage}.max_abs")
    table = stage_table(results)
    header = ("% numbers-axis2.tex -- generated by analysis/scripts/axis2_readout.py (the readout on axis 2); do not edit by hand.\n"
              f"% Results file: {results['output_name']}/results.json, created {results['created_at']}. Cell suffixes LP, LR, QP, QR.\n")
    path.write_text(header + "\n".join(lines) + "\n\n" + "\\newcommand{\\AxTwoStageTable}{%\n" + table + "}\n")
    return lines


def stage_table(results: dict) -> str:
    ro = results["readout"]["heldout"]
    cells = [code for _, _, code in CELLS if code in ro]
    have = [s for s in ("Q", "R0", "R", "P", "I", "K", "A") if any(_cell(ro[c], "axis2", "percentile", "controls", s) for c in cells)]
    labels = {"Q": "$Q$\\ \\ agent's queries", "R0": "$R_0$\\ frozen search on the prompt text", "R": "$R$\\ \\ rows the agent's queries retrieve",
              "P": "$P$\\ \\ shortlist", "I": "$I$\\ \\ \\ first $L$ shown pages, shown order", "K": "$K$\\ \\ cited sources", "A": "$A$\\ \\ answer text"}
    rows = ["\\begin{table*}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{0.95}",
            "\\begin{tabular}{@{}l" + "c" * len(cells) + "@{}}", "\\toprule",
            " & " + " & ".join(CELL_NAMES[c] for c in cells) + " \\\\", "\\midrule",
            f"\\multicolumn{{{len(cells) + 1}}}{{@{{}}l}}{{\\textit{{Slope of the stage's axis-2 position on the prompt's, with $x$ and $x^2$ as controls (percentile units)}}}} \\\\"]
    for s in have:
        vals = []
        for c in cells:
            v = _cell(ro[c], "axis2", "percentile", "controls", s)
            vals.append("--" if v is None else f"{fmt.slope(v['estimate'])} {fmt.ci(*v['ci95'])}")
        rows.append(f"{labels[s]} & " + " & ".join(vals) + " \\\\")
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{{len(cells) + 1}}}{{@{{}}l}}{{\\textit{{SD units: axis 1 / axis 2 (with controls) / random directions, median (largest $|\\beta|$)}}}} \\\\")
    for s in have:
        vals = []
        for c in cells:
            a1 = _cell(ro[c], "axis1", "sd", "raw", s)
            a2 = _cell(ro[c], "axis2", "sd", "controls", s)
            rnd = results["random_summary"]["heldout"][c].get(s, {})
            part = [fmt.slope(a1["estimate"], 2) if a1 else "--", fmt.slope(a2["estimate"], 2) if a2 else "--",
                    (f"{fmt.slope(rnd['median'], 2)} ({rnd['max_abs']:.2f})" if rnd else "--")]
            vals.append(" / ".join(part))
        rows.append(f"{labels[s]} & " + " & ".join(vals) + " \\\\")
    rows.append("\\midrule")
    rows.append("Answers & " + " & ".join(fmt.num(ro[c]["directions"]["axis2"]["percentile"]["controls"]["answers"]) for c in cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}",
             "\\caption{The stage readout on axis 2, held-out keywords (natural condition, answers citing at least one source). "
             "Upper block: within-keyword change in the stage's mean axis-2 position from $x_2=0$ to $x_2=1$, with the axis-1 position $x$ and $x^2$ "
             "as controls; brackets are 95\\% keyword-bootstrap intervals. Lower block: the same slopes with positions in standard deviations of the prompts' "
             "coordinate (a slope of 1 means the stage moves as far along the direction as the prompt): axis 1 without controls, axis 2 with them, and "
             "\\NAxTwoRandom{} random directions orthogonal to both axes, five per encoder view, read with the same controls"
             + (" (page stages only: queries and answers have stored axis positions, not embeddings)" if any(t in have for t in TEXT_STAGES) else "")
             + ". Associations only.}", "\\label{tab:axis2}", "\\end{table*}"]
    return "\n".join(rows) + "\n"


def write_csv(results: dict, path: Path) -> int:
    import csv
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["split", "cell", "direction", "unit", "variant", "term", "kind", "estimate", "ci95_low", "ci95_high",
                    "permutation_p", "answers", "keywords"])
        for split, cells in results["readout"].items():
            for cell, entry in cells.items():
                for direction, units in entry["directions"].items():
                    for unit, variants in units.items():
                        for variant, r in variants.items():
                            for kind in ("slopes", "increments", "shares"):
                                for term, v in r.get(kind, {}).items():
                                    if "estimate" not in v:
                                        continue
                                    w.writerow([split, cell, direction, unit, variant, term, kind[:-1], v["estimate"],
                                                *(v["ci95"] or [None, None]), v.get("permutation_p"),
                                                v.get("answers", r.get("answers")), r.get("keywords")])
                                    count += 1
    return count


def face_validity(path: Path, prompts: dict, pages: dict, cal: dict | None, keyword_split: dict, seed: int) -> None:
    lines = ["Face validity of axis 2 (higher = higher evaluation rubric). Local file: calibration texts are restricted-local "
             "and must not be published.", ""]

    def block(title, rows):
        lines.append(f"== {title}")
        lines.extend(rows)
        lines.append("")

    order = np.argsort(prompts["x2"], kind="stable")
    fmt_prompt = lambda i: (f"  x2={prompts['x2'][i]:.3f}  x={prompts['x'][i]:.3f}  [{prompts['keyword'][i]}] "  # noqa: E731
                            f"({prompts['writer'][i]})  {prompts['question'][i]}")
    block("15 prompts lowest on axis 2", [fmt_prompt(i) for i in order[:15]])
    block("15 prompts highest on axis 2", [fmt_prompt(i) for i in order[::-1][:15]])
    rng = np.random.default_rng(seed)
    held = sorted(k for k, s in keyword_split.items() if s == "held-out")
    chosen = rng.choice(len(held), size=min(10, len(held)), replace=False)
    rows = []
    for j in sorted(chosen):
        kw = held[j]
        idx = np.flatnonzero(prompts["keyword"] == kw)
        if len(idx) < 2:
            continue
        lo, hi = idx[np.argmin(prompts["x2"][idx])], idx[np.argmax(prompts["x2"][idx])]
        rows += [f"  [{kw}]", "   low " + fmt_prompt(lo).strip(), "   high " + fmt_prompt(hi).strip()]
    block("Within keyword: lowest and highest prompt on axis 2 (10 held-out keywords, seeded)", rows)
    order = np.argsort(pages["u2"], kind="stable")
    page_row = lambda i: f"  u2={pages['u2'][i]:.3f}  u={pages['u'][i]:.3f}  {pages['text'][i][:220]!r}"  # noqa: E731
    block("10 pages lowest on axis 2", [page_row(i) for i in order[:10]])
    block("10 pages highest on axis 2", [page_row(i) for i in order[::-1][:10]])
    if cal is not None:
        keep = np.flatnonzero(cal["usable"] & np.asarray([i in cal["texts"] for i in cal["ids"]]))
        order = keep[np.argsort(cal["placed"].consensus[keep, 1], kind="stable")]

        def cal_row(i):
            item = cal["ids"][i]
            lab = cal["labels"].get(item, {})
            return (f"  z2={cal['placed'].consensus[i, 1]:+.2f}  z1={cal['placed'].consensus[i, 0]:+.2f}  "
                    f"eval={lab.get('evaluation_1_7')} sel={lab.get('selection_commitment_1_7')} act={lab.get('action_implementation_1_7')} "
                    f"info={lab.get('information_seeking_1_7')}  ({cal['split'][i]}, {cal['source'][i]})  {cal['texts'][item][:220]!r}")

        block("15 usable calibration items lowest on axis 2", [cal_row(i) for i in order[:15]])
        block("15 usable calibration items highest on axis 2", [cal_row(i) for i in order[::-1][:15]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


# ---------------------------------------------------------------- run

def run(args) -> int:
    pd = _pd()
    started = time.time()
    partial = readiness.new_directory(args.output)
    maps = {view: load_map(Path(args.maps) / view / "readiness_embedding_map.json") for view in VIEWS}
    al = load_alignment(args.battery)
    sign = evaluation_sign(maps["qwen"])
    mistral_sign = evaluation_sign(maps["mistral"]) * float(np.sign(al.rotation[1, 1]))
    extra = {view: random_directions(maps[view].axes, args.random_directions, args.random_seed, i) for i, view in enumerate(VIEWS)}
    directions = direction_names(args.random_directions)
    random_names = [d for d in directions if d.startswith("random_")]
    checks = {"axis2_sign": sign, "mistral_sign_after_rotation_agrees": mistral_sign == sign,
              "random_max_abs_cosine_with_axes": {view: float(np.max(np.abs(extra[view] @ maps[view].axes.T))) for view in VIEWS},
              "random_max_abs_cosine_between": {view: float(np.max(np.abs(np.triu(extra[view] @ extra[view].T, 1)))) for view in VIEWS}}
    log("maps", sign=sign, loadings={v: rubric_loadings(maps[v]) for v in VIEWS})

    # prompts
    prompt_ids, raw = load_prompt_embeddings(args.prompt_embeddings, maps, extra)
    rq, rm, rnd = split_random(raw)
    prompts = place(rq, rm, al, sign, rnd)
    prompt_index = pd.Index(prompt_ids)
    final_rows = list(read_jsonl(args.final_axis_map))
    in_map = prompt_index.get_indexer([str(r["candidate_id"]) for r in final_rows])
    if (in_map < 0).any():
        raise ValueError("the final axis map holds prompts without embeddings")
    scale_mask = np.zeros(len(prompt_ids), bool)
    scale_mask[in_map] = True
    scales = Scales(prompts, prompt_ids, scale_mask, directions)
    archived_z = np.asarray([float(r["consensus_axis_1_z"]) for r in final_rows])
    archived_x = np.asarray([float(r["axis_1_percentile_0_1"]) for r in final_rows])
    checks["prompts"] = {"embedded": len(prompt_ids), "in_final_axis_map": int(scale_mask.sum()),
                         "max_abs_consensus_axis_1_z_difference": float(np.max(np.abs(prompts.consensus[in_map, 0] - archived_z))),
                         "max_abs_x_difference": float(np.max(np.abs(scales.prompt_percentile["axis1"][in_map] - archived_x)))}
    records = load_prompt_records(args.prompt_records)
    if records:
        checks["prompts"]["records"] = len(records)
    log("prompts placed", **checks["prompts"])

    # pages (corpus rows) and snapshot rows
    corpus = _pq().read_table(Path(args.corpus) / "snippets.parquet", columns=[
        "row", "text", "qwen_raw_axis_1", "qwen_raw_axis_2", "mistral_raw_axis_1", "mistral_raw_axis_2",
        "consensus_axis_1_z", "prompt_scale_percentile_0_1"]).to_pandas()
    if not np.array_equal(corpus["row"].to_numpy(), np.arange(len(corpus))):
        raise ValueError("corpus rows are not in vector order")
    praw = load_page_embeddings(args.corpus, maps, extra)
    pq_, pm_, prnd = split_random(praw)
    pages = place(pq_, pm_, al, sign, prnd)
    checks["pages"] = {"pages": len(corpus),
                       "max_abs_raw_difference": float(max(np.max(np.abs(praw[v][:, k] - corpus[f"{v}_raw_axis_{k + 1}"].to_numpy()))
                                                           for v in VIEWS for k in (0, 1))),
                       "max_abs_consensus_axis_1_z_difference": float(np.max(np.abs(pages.consensus[:, 0] - corpus["consensus_axis_1_z"].to_numpy()))),
                       "max_abs_u_difference": float(np.nanmax(np.abs(scales.text_values(pages, "axis1", "percentile")[0]
                                                                      - corpus["prompt_scale_percentile_0_1"].to_numpy())))}
    log("pages placed", **checks["pages"])
    rows = _pq().read_table(args.rows, columns=["row_id", "corpus_row"]).to_pandas()
    if not np.array_equal(rows["row_id"].to_numpy(), np.arange(len(rows))):
        raise ValueError("snapshot rows are not in row order")
    corpus_row = rows["corpus_row"].to_numpy().astype(np.int64)
    if (corpus_row < 0).any() or (corpus_row >= len(corpus)).any():
        raise ValueError("a snapshot row has no page")

    # traces
    tr = load_traces(args.traces)
    a = tr.answers
    n = len(a)
    answer_prompt = prompt_index.get_indexer(a["prompt_id"])
    if (answer_prompt < 0).any():
        raise ValueError("answers whose prompt has no embedding")
    x = a["x"].to_numpy(float)
    checks["answers_x_max_abs_difference"] = float(np.max(np.abs(scales.prompt_percentile["axis1"][answer_prompt] - x)))
    replay = load_replay(args.replay)
    keyword_code, keyword_count = first_appearance_codes(a["keyword"].to_numpy(), tr.funnel_order)
    prompt_code, prompt_count = first_appearance_codes(a["prompt_id"].to_numpy(), tr.funnel_order)
    draws = stages.keyword_draws(keyword_count, args.bootstrap, args.seed)

    # per (direction, unit): chain values, Q and A, prompt positions by prompt code
    texts = load_texts(args.text_coordinates, al, sign) if args.text_coordinates else None
    query_count, layout = None, None
    answer_lookup = pd.Index(a["answer_id"])
    if texts is not None:
        query_count = int(texts.query_ids.max()) + 1
        if len(np.unique(texts.query_ids)) != len(texts.query_ids):
            raise ValueError("a query appears twice in the text coordinates")
        layout = query_layout(args.traces, answer_lookup, n)
    stage_by, text_by, prompt_by = {}, {}, {}
    code_to_prompt = np.full(prompt_count, -1, np.int64)
    code_to_prompt[prompt_code] = answer_prompt
    for direction in directions:
        for unit in units_of(direction):
            key = (direction, unit)
            prompt_values = scales.prompt_values(prompts, direction, unit)
            prompt_by[key] = prompt_values[code_to_prompt]
            page_values, _ = scales.text_values(pages, direction, unit)
            stage_by[key] = chain_values(tr, replay, page_values[corpus_row])
            if texts is not None:
                q, _ = scales.text_values(texts.queries, direction, unit)
                av, _ = scales.text_values(texts.answers, direction, unit)
                if q is not None and av is not None:
                    by_query = np.full(query_count, np.nan)
                    by_query[texts.query_ids] = q
                    answer_values = np.full(n, np.nan)
                    where = answer_lookup.get_indexer(texts.answer_ids)
                    answer_values[where[where >= 0]] = av[where >= 0]
                    text_by[key] = {"Q": query_means(layout, by_query), "A": answer_values}
    reference = stage_by[("axis1", "percentile")]
    checks["chain_values_max_abs_difference"] = {s: float(np.nanmax(np.abs(reference[s] - a[f"chain_{s}"].to_numpy(float)))) for s in CHAIN}
    checks["chain_values_nan_mismatch"] = {s: int((np.isfinite(reference[s]) != np.isfinite(a[f"chain_{s}"].to_numpy(float))).sum()) for s in CHAIN}
    if texts is not None:
        zq = {"Q": text_by[("axis1", "sd")]["Q"] * scales.sd["axis1"], "A": text_by[("axis1", "sd")]["A"] * scales.sd["axis1"]}
        both = {s: np.isfinite(zq[s]) & np.isfinite(a[s].to_numpy(float)) for s in TEXT_STAGES}
        checks["text_values_axis1_z_max_abs_difference"] = {
            s: float(np.max(np.abs(zq[s][both[s]] - a[s].to_numpy(float)[both[s]]))) if both[s].any() else None for s in TEXT_STAGES}
        checks["text_values_nan_mismatch"] = {s: int((np.isfinite(zq[s]) != np.isfinite(a[s].to_numpy(float))).sum()) for s in TEXT_STAGES}
    log("stage values", seconds=round(time.time() - started, 1), checks=checks["chain_values_max_abs_difference"])

    # masks: natural answers of each cell, common sample of the chain
    natural = (a["condition"] == "natural").to_numpy()
    common = np.logical_and.reduce([np.isfinite(reference[s]) for s in CHAIN])
    masks = {}
    for split, value in SPLITS:
        in_split = np.ones(n, bool) if value is None else (a["split"] == value).to_numpy()
        masks[split] = {code: natural & in_split & common & (a["model"] == model).to_numpy() & (a["method"] == method).to_numpy()
                        for model, method, code in CELLS}
    _, first_answer = np.unique(prompt_code, return_index=True)  # codes are 0..prompt_count-1
    readout_results = run_readout(masks, stage_by, text_by, prompt_by, x, keyword_code, prompt_code, keyword_code[first_answer],
                                  draws=draws, permutations=args.permutations, shuffle_seed=args.shuffle_seed, directions=directions)
    summaries = {split: {code: {s: random_summary(entry, random_names, s, "controls") for s in (*CHAIN, *TEXT_STAGES)}
                         for code, entry in cells.items()} for split, cells in readout_results.items()}
    summaries_raw = {split: {code: {s: random_summary(entry, random_names, s, "raw") for s in (*CHAIN, *TEXT_STAGES)}
                             for code, entry in cells.items()} for split, cells in readout_results.items()}
    checks["stored_chain_heldout"] = stored_chain_check(readout_results["heldout"], args.stored_chain)

    # positions, reliability and prompt diagnostics
    keyword_of_prompt = dict(zip(a["prompt_id"], a["keyword"]))
    split_of_keyword = dict(zip(a["keyword"], a["split"]))
    for pid, rec in records.items():
        if pid not in keyword_of_prompt and rec.get("keyword"):
            keyword_of_prompt[pid] = rec["keyword"]
    has_kw = np.asarray([p in keyword_of_prompt for p in prompt_ids])
    kw_text = np.asarray([keyword_of_prompt.get(p, "") for p in prompt_ids], object)
    x1p, x2p = scales.prompt_percentile["axis1"], scales.prompt_percentile["axis2"]
    writer = None
    if records and all(records.get(p, {}).get("writer") for p in np.asarray(prompt_ids)[has_kw]):
        names = sorted({records[p]["writer"] for p in np.asarray(prompt_ids)[has_kw]})
        if len(names) == 2:
            writer = np.asarray([float(records[p]["writer"] == names[1]) if p in records else np.nan for p in prompt_ids])
            checks["writer_coding"] = {"zero": names[0], "one": names[1]}
    diagnostics = {}
    for split, value in SPLITS:
        keep = has_kw & (np.ones(len(prompt_ids), bool) if value is None else
                         np.asarray([split_of_keyword.get(k) == value for k in kw_text]))
        codes = np.unique(kw_text[keep], return_inverse=True)[1]
        diagnostics[split] = prompt_diagnostics(x1p[keep], x2p[keep], prompts.consensus[keep, 1], codes,
                                                None if writer is None else writer[keep])
    positions = {"prompts": {"view_agreement": view_agreement(prompts), "sd": scales.sd},
                 "pages": {"view_agreement": view_agreement(pages)}}
    if texts is not None:
        _, first_text = np.unique(texts.answer_text_ids, return_index=True)
        keep = np.zeros(len(texts.answers), bool)
        keep[first_text] = True
        positions["queries"] = {"view_agreement": view_agreement(texts.queries)}
        positions["answers"] = {"view_agreement": view_agreement(texts.answers, keep), "answers": int(len(texts.answers))}
    calibration = None
    cal = load_calibration(args.calibration, al, sign) if args.calibration else None
    if cal is not None:
        confirm = (cal["split"] == "confirmation") & cal["usable"]
        develop = (cal["split"] == "development") & cal["usable"]
        battery = al.battery["cross_embedding_alignment"]
        va = view_agreement(cal["placed"], confirm)
        calibration = {"items": len(cal["ids"]), "usable_confirmation": int(confirm.sum()), "usable_development": int(develop.sum()),
                       "view_agreement_confirmation": va,
                       "battery_check_max_abs_difference": float(max(abs(va[f"axis{k}"]["spearman"] - battery[f"confirmation_axis_{k}"]["spearman"])
                                                                     for k in (1, 2))),
                       "rubrics_confirmation": rubric_table(cal, confirm), "rubrics_development": rubric_table(cal, develop),
                       "inputs": cal["inputs"]}
        log("calibration", agreement=va, battery_check=calibration["battery_check_max_abs_difference"])
    u1, _ = scales.text_values(pages, "axis1", "percentile")
    u2, _ = scales.text_values(pages, "axis2", "percentile")
    readiness.write_json(partial / "readout-progress.json", {"stage": "outputs"})
    (partial / "readout-progress.json").unlink()

    results = {"format_version": FORMAT_VERSION, "created_at": readiness.now(), "git_commit": readiness.git_commit(),
               "output_name": Path(args.output).name, "observational": True,
               "settings": {"bootstrap": args.bootstrap, "permutations": args.permutations, "seed": args.seed,
                            "shuffle_seed": args.shuffle_seed, "random_seed": args.random_seed,
                            "random_directions_per_view": args.random_directions, "keywords_coded": keyword_count,
                            "controls": ["x", "x^2"], "units": {"percentile": "prompt percentile on the direction",
                                                                "sd": "coordinate / SD of the prompts' coordinate"},
                            "axis2_sign": "higher = higher evaluation rubric (sign of the evaluation loading on the Qwen a_2)",
                            "queries_and_answers": texts is not None},
               "maps": {v: {"map_id": maps[v].map_id, "rubric_loadings": rubric_loadings(maps[v])} for v in VIEWS},
               "checks": checks, "positions": positions, "calibration": calibration, "prompt_diagnostics": diagnostics,
               "readout": readout_results, "random_summary": summaries, "random_summary_raw": summaries_raw}
    inputs = {"traces_manifest": readiness.identity(Path(args.traces) / "manifest.json"),
              "trace_answers": readiness.identity(Path(args.traces) / "trace_answers.parquet"),
              "stage_items": readiness.identity(Path(args.traces) / "stage_items.parquet"),
              "replay": readiness.identity(Path(args.replay) / "replay.npz"), "rows": readiness.identity(args.rows),
              "corpus_manifest": readiness.identity(Path(args.corpus) / "manifest.json") if (Path(args.corpus) / "manifest.json").exists() else None,
              "maps": {v: readiness.identity(Path(args.maps) / v / "readiness_embedding_map.json") for v in VIEWS},
              "battery": readiness.identity(Path(args.battery) / "readiness_robustness_battery.json" if Path(args.battery).is_dir() else args.battery),
              "final_axis_map": readiness.identity(args.final_axis_map),
              "prompt_records": readiness.identity(args.prompt_records) if args.prompt_records else None,
              "text_coordinates": readiness.identity(args.text_coordinates) if args.text_coordinates else None,
              "stored_chain": readiness.identity(args.stored_chain) if args.stored_chain else None}
    results["inputs"] = inputs
    results["seconds"] = round(time.time() - started, 1)
    readiness.write_json(partial / "results.json", json.loads(json.dumps(results, default=_json_default)))
    results = json.loads((partial / "results.json").read_text())
    rows_written = write_csv(results, partial / "stages-axis2.csv")
    write_macros(results, partial / "numbers-axis2.tex")
    question = [records.get(p, {}).get("question") or "" for p in prompt_ids]
    writer_names = [records.get(p, {}).get("writer") or "" for p in prompt_ids]
    face_validity(partial / "face-validity.txt",
                  {"x": x1p[has_kw], "x2": x2p[has_kw], "keyword": kw_text[has_kw], "question": np.asarray(question, object)[has_kw],
                   "writer": np.asarray(writer_names, object)[has_kw]},
                  {"u": u1, "u2": u2, "text": corpus["text"].to_numpy()}, cal, split_of_keyword, args.random_seed)
    pd.DataFrame({"prompt_id": prompt_ids, "keyword": kw_text, "x": x1p, "x2": x2p, "consensus_z1": prompts.consensus[:, 0],
                  "consensus_z2": prompts.consensus[:, 1], "qwen_z2": prompts.zq[:, 1], "mistral_aligned_z2": prompts.zm[:, 1]}
                 ).to_csv(partial / "prompts-axis2.csv.gz", index=False)
    pd.DataFrame({"corpus_row": np.arange(len(corpus)), "u": u1, "u2": u2, "consensus_z2": pages.consensus[:, 1],
                  "qwen_z2": pages.zq[:, 1], "mistral_aligned_z2": pages.zm[:, 1]}).to_csv(partial / "pages-axis2.csv.gz", index=False)
    partial.rename(Path(args.output).resolve())
    log("done", output=str(args.output), csv_rows=rows_written, seconds=round(time.time() - started, 1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="positions, checks, prompt diagnostics, stage readout and random directions")
    r.add_argument("--traces", type=Path, required=True, help="release-traces-v1 (trace_answers, stage_items, answer_queries)")
    r.add_argument("--replay", type=Path, required=True, help="funnel replay of the prompt texts (prompts.jsonl.gz, replay.npz)")
    r.add_argument("--rows", type=Path, required=True, help="funnel-features-v1/rows.parquet (snapshot row -> corpus row)")
    r.add_argument("--corpus", type=Path, required=True, help="snippet-embeddings-corpus-v1 (snippets.parquet, embeddings/)")
    r.add_argument("--prompt-embeddings", type=Path, required=True, help="final-audit/projections with <view>/shard-*/" + PROMPT_FILE)
    r.add_argument("--maps", type=Path, required=True, help="folder with <view>/readiness_embedding_map.json")
    r.add_argument("--battery", type=Path, required=True, help="readiness_robustness_battery.json or its folder")
    r.add_argument("--final-axis-map", type=Path, required=True, help="final-axis-map.jsonl(.gz): the prompts of the percentile scale")
    r.add_argument("--prompt-records", type=Path, help="prompt texts, keywords and writers (snapshot prompts.jsonl.gz)")
    r.add_argument("--calibration", type=Path, help="archive root with maps/<view>/readiness_supervised_subspace_coordinates.jsonl")
    r.add_argument("--text-coordinates", type=Path, help="text_coordinates.parquet of the `texts` stage (queries, answers)")
    r.add_argument("--stored-chain", type=Path, help="steelman-confirmation/chain.json: reproduction check of the held-out axis-1 chain")
    r.add_argument("--bootstrap", type=int, default=200)
    r.add_argument("--permutations", type=int, default=200)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    r.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    r.add_argument("--random-directions", type=int, default=5)
    r.add_argument("--output", type=Path, required=True)
    t = sub.add_parser("texts", help="per-text raw coordinates of queries and answers from the merged projections")
    t.add_argument("--traces", type=Path, required=True, help="release-traces-v1 (queries.parquet, trace_answers.parquet)")
    t.add_argument("--queries-qwen", type=Path, required=True)
    t.add_argument("--queries-mistral", type=Path, required=True)
    t.add_argument("--answers-qwen", type=Path, required=True)
    t.add_argument("--answers-mistral", type=Path, required=True)
    t.add_argument("--answers-index", type=Path, required=True, help="answers-export/observations.jsonl.gz")
    t.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    return run(args) if args.command == "run" else export_texts(args)


if __name__ == "__main__":
    raise SystemExit(main())
