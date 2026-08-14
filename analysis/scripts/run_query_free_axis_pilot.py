#!/usr/bin/env python3
"""Prepare, smoke-test, embed, and fit the query-free LLM2Vec A1 pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.query_free_decision_readiness import (  # noqa: E402
    QUERY_FREE_AXIS_VERSION,
    REPRESENTATION_VIEWS,
    ContentContext,
    FakeQueryFreeObjectiveGenerator,
    QueryFreeGenerationRequest,
    QueryFreeStimulus,
    RealizationPlan,
    build_generation_requests,
    build_ordinal_judge_tasks,
    build_pairwise_judge_tasks,
    fake_query_free_embeddings,
    fit_query_free_axis,
    generate_query_free_stimuli,
    load_query_free_specification,
    measure_query_free_geometry,
    project_query_free_axis,
    render_query_free_generation_prompt,
    representation_texts,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecPromptEmbedder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    prepare = stages.add_parser("prepare")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--specification")
    prepare.add_argument("--master-seed", type=int, default=20260817)

    compile_stage = stages.add_parser("compile")
    compile_stage.add_argument("--output-dir", required=True)
    compile_stage.add_argument("--objectives-jsonl", required=True)
    compile_stage.add_argument("--specification")
    compile_stage.add_argument("--master-seed", type=int, default=20260817)

    export = stages.add_parser("export-judging")
    export.add_argument("--output-dir", required=True)
    export.add_argument("--stimuli-jsonl", required=True)
    export.add_argument("--master-seed", type=int, default=20260817)

    embed = stages.add_parser("embed")
    embed.add_argument("--output-dir", required=True)
    embed.add_argument("--stimuli-jsonl", required=True)
    embed.add_argument("--embedding-model", required=True)
    embed.add_argument("--mntp-model")
    embed.add_argument("--peft-model")
    embed.add_argument("--encode-batch-size", type=int, default=8)
    embed.add_argument("--encode-max-length", type=int, default=512)

    fit = stages.add_parser("fit")
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--stimuli-jsonl", required=True)
    fit.add_argument("--embeddings-npz", required=True)

    smoke = stages.add_parser("fake-smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--specification")
    smoke.add_argument("--master-seed", type=int, default=20260817)
    return parser


class _PersistedObjectiveGenerator:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = {}
        backends = set()
        models = set()
        for row in rows:
            request_id = str(row.get("request_id", ""))
            objective = str(row.get("objective_clause", ""))
            if not request_id or not objective:
                raise ValueError("every objective row needs request_id and objective_clause")
            if request_id in self._rows:
                raise ValueError(f"duplicate objective request_id: {request_id}")
            self._rows[request_id] = objective
            backends.add(str(row.get("generator_backend", "external-llm")))
            models.add(str(row.get("generator_model", "unrecorded")))
        if len(backends) != 1 or len(models) != 1:
            raise ValueError("one compile artifact must use one generator backend/model")
        self.backend_name = next(iter(backends))
        self.model_name = next(iter(models))

    def generate(self, request: QueryFreeGenerationRequest) -> str:
        try:
            return self._rows.pop(request.request_id)
        except KeyError as exc:
            raise ValueError(f"missing objective for request {request.request_id}") from exc

    def assert_consumed(self) -> None:
        if self._rows:
            raise ValueError(
                "objective file contains unknown request IDs: "
                + ", ".join(sorted(self._rows)[:5])
            )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        output = Path(args.output_dir).resolve()
        _prepare_new_output(output)
        if args.stage == "prepare":
            _prepare(args, output)
        elif args.stage == "compile":
            _compile(args, output)
        elif args.stage == "export-judging":
            _export_judging(args, output)
        elif args.stage == "embed":
            _embed(args, output)
        elif args.stage == "fit":
            _fit(args, output)
        else:
            _fake_smoke(args, output)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"output: {output}")
    return 0


def _load_design(args):
    specification = args.specification if args.specification else None
    if specification:
        return load_query_free_specification(specification)
    return load_query_free_specification()


def _prepare(args, output: Path) -> None:
    contexts, plans, specification = _load_design(args)
    requests = build_generation_requests(
        contexts,
        plans,
        master_seed=args.master_seed,
    )
    rows = []
    for request in requests:
        row = asdict(request)
        row["generation_prompt"] = render_query_free_generation_prompt(request)
        rows.append(row)
    _atomic_jsonl(output / "generation_requests.jsonl", rows)
    _atomic_json(output / "construct_and_population_spec.json", specification)
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            status="generation-requests-prepared",
            scientific_result=False,
            master_seed=args.master_seed,
            request_count=len(requests),
            development_request_count=sum(
                request.context.split == "development" for request in requests
            ),
            confirmation_request_count=sum(
                request.context.split == "confirmation" for request in requests
            ),
        ),
    )


def _compile(args, output: Path) -> None:
    contexts, plans, specification = _load_design(args)
    requests = build_generation_requests(
        contexts,
        plans,
        master_seed=args.master_seed,
    )
    objective_path = Path(args.objectives_jsonl).resolve()
    objective_rows = _read_jsonl(objective_path)
    generator = _PersistedObjectiveGenerator(objective_rows)
    stimuli = generate_query_free_stimuli(requests, generator=generator)
    generator.assert_consumed()
    _atomic_jsonl(output / "query_free_stimuli.jsonl", map(asdict, stimuli))
    _atomic_json(output / "construct_and_population_spec.json", specification)
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            status="stimuli-compiled-unembedded",
            scientific_result=False,
            master_seed=args.master_seed,
            objective_file=str(objective_path),
            objective_file_sha256=_sha256_file(objective_path),
            stimulus_count=len(stimuli),
            structurally_valid_count=sum(item.structural_valid for item in stimuli),
            generator_backend=generator.backend_name,
            generator_model=generator.model_name,
        ),
    )


def _export_judging(args, output: Path) -> None:
    stimulus_path = Path(args.stimuli_jsonl).resolve()
    stimuli = _load_stimuli(stimulus_path)
    ordinal = build_ordinal_judge_tasks(stimuli)
    pairwise, codebook = build_pairwise_judge_tasks(
        stimuli,
        master_seed=args.master_seed,
    )
    _atomic_jsonl(output / "ordinal_judge_tasks_blinded.jsonl", map(asdict, ordinal))
    _atomic_jsonl(output / "pairwise_judge_tasks_blinded.jsonl", map(asdict, pairwise))
    _atomic_jsonl(
        output / "judge_task_codebook.jsonl",
        ({"task_id": task_id, **row} for task_id, row in sorted(codebook.items())),
    )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            status="judge-tasks-exported-unjudged",
            scientific_result=False,
            master_seed=args.master_seed,
            stimuli_file=str(stimulus_path),
            stimuli_file_sha256=_sha256_file(stimulus_path),
            ordinal_task_count=len(ordinal),
            pairwise_task_count=len(pairwise),
        ),
    )


def _embed(args, output: Path) -> None:
    stimulus_path = Path(args.stimuli_jsonl).resolve()
    stimuli = _load_stimuli(stimulus_path)
    if any(not item.structural_valid for item in stimuli):
        raise ValueError("refusing to embed structurally invalid stimuli")
    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    texts = representation_texts(stimuli)
    matrices = {}
    for view in REPRESENTATION_VIEWS:
        matrices[view] = np.asarray(embedder.embed(texts[view]), dtype=np.float32)
        print(f"embedded {view}: {matrices[view].shape}", flush=True)
    _atomic_npz(
        output / "query_free_llm2vec_embeddings.npz",
        stimulus_ids=np.asarray([item.stimulus_id for item in stimuli], dtype=str),
        embedding_model=np.asarray(embedder.model_name),
        **{_npz_view_name(view): matrix for view, matrix in matrices.items()},
    )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            status="embedded-unfitted",
            scientific_result=False,
            stimuli_file=str(stimulus_path),
            stimuli_file_sha256=_sha256_file(stimulus_path),
            embedding_model=embedder.model_name,
            embedding_dimension=int(next(iter(matrices.values())).shape[1]),
            representation_views=REPRESENTATION_VIEWS,
            encode_batch_size=args.encode_batch_size,
            encode_max_length=args.encode_max_length,
        ),
    )


def _fit(args, output: Path) -> None:
    stimulus_path = Path(args.stimuli_jsonl).resolve()
    embedding_path = Path(args.embeddings_npz).resolve()
    stimuli = _load_stimuli(stimulus_path)
    model_name, matrices = _load_embeddings(embedding_path, stimuli)
    _fit_and_write(
        output,
        stimuli,
        matrices,
        embedding_model=model_name,
        scientific_result=False,
        source_files={
            "stimuli_file": str(stimulus_path),
            "stimuli_file_sha256": _sha256_file(stimulus_path),
            "embeddings_file": str(embedding_path),
            "embeddings_file_sha256": _sha256_file(embedding_path),
        },
    )


def _fake_smoke(args, output: Path) -> None:
    contexts, plans, specification = _load_design(args)
    requests = build_generation_requests(
        contexts,
        plans,
        master_seed=args.master_seed,
    )
    stimuli = generate_query_free_stimuli(
        requests,
        generator=FakeQueryFreeObjectiveGenerator(),
    )
    matrices = fake_query_free_embeddings(stimuli)
    _atomic_json(output / "construct_and_population_spec.json", specification)
    _atomic_jsonl(output / "query_free_stimuli.jsonl", map(asdict, stimuli))
    _atomic_npz(
        output / "query_free_fake_embeddings.npz",
        stimulus_ids=np.asarray([item.stimulus_id for item in stimuli], dtype=str),
        embedding_model=np.asarray("fake-query-free-embedding-v1"),
        **{_npz_view_name(view): matrix for view, matrix in matrices.items()},
    )
    _fit_and_write(
        output,
        stimuli,
        matrices,
        embedding_model="fake-query-free-embedding-v1",
        scientific_result=False,
        source_files={"smoke_mode": True, "master_seed": args.master_seed},
    )


def _fit_and_write(
    output,
    stimuli,
    matrices,
    *,
    embedding_model,
    scientific_result,
    source_files,
) -> None:
    development_indices = [
        index for index, item in enumerate(stimuli) if item.context_split == "development"
    ]
    confirmation_indices = [
        index for index, item in enumerate(stimuli) if item.context_split == "confirmation"
    ]
    development = tuple(stimuli[index] for index in development_indices)
    confirmation = tuple(stimuli[index] for index in confirmation_indices)
    development_matrices = {
        view: matrix[development_indices] for view, matrix in matrices.items()
    }
    confirmation_matrices = {
        view: matrix[confirmation_indices] for view, matrix in matrices.items()
    }
    axis = fit_query_free_axis(
        development,
        development_matrices,
        embedding_model=embedding_model,
    )
    development_diagnostics = measure_query_free_geometry(
        axis,
        development,
        development_matrices,
    )
    confirmation_diagnostics = measure_query_free_geometry(
        axis,
        confirmation,
        confirmation_matrices,
    )
    development_coordinates = project_query_free_axis(
        axis,
        development,
        development_matrices,
    )
    confirmation_coordinates = project_query_free_axis(
        axis,
        confirmation,
        confirmation_matrices,
    )
    _atomic_json(output / "query_free_axis.json", asdict(axis))
    _atomic_json(
        output / "query_free_axis_diagnostics.json",
        {
            "development": asdict(development_diagnostics),
            "confirmation": asdict(confirmation_diagnostics),
        },
    )
    _atomic_jsonl(
        output / "query_free_axis_coordinates.jsonl",
        (
            {"evaluation_split": "development", **asdict(item)}
            for item in development_coordinates
        ),
        append_rows=(
            {"evaluation_split": "confirmation", **asdict(item)}
            for item in confirmation_coordinates
        ),
    )
    _atomic_json(
        output / "run_manifest.json",
        _manifest(
            status="axis-fitted-unreviewed",
            scientific_result=scientific_result,
            axis_id=axis.axis_id,
            embedding_model=embedding_model,
            development_stimulus_count=len(development),
            confirmation_stimulus_count=len(confirmation),
            representation_views=REPRESENTATION_VIEWS,
            **source_files,
        ),
    )


def _load_stimuli(path: Path) -> tuple[QueryFreeStimulus, ...]:
    rows = []
    for row in _read_jsonl(path):
        row["contract_failures"] = tuple(row.get("contract_failures", ()))
        rows.append(QueryFreeStimulus(**row))
    if not rows:
        raise ValueError("stimulus file is empty")
    return tuple(rows)


def _load_embeddings(path: Path, stimuli) -> tuple[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        ids = tuple(str(value) for value in payload["stimulus_ids"])
        expected = tuple(item.stimulus_id for item in stimuli)
        if ids != expected:
            raise ValueError("embedding stimulus IDs do not align with stimulus JSONL")
        model_name = str(payload["embedding_model"].item())
        matrices = {
            view: np.asarray(payload[_npz_view_name(view)], dtype=np.float64)
            for view in REPRESENTATION_VIEWS
        }
    return model_name, matrices


def _npz_view_name(view: str) -> str:
    return view.replace("-", "_")


def _prepare_new_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)


def _manifest(**values) -> dict[str, object]:
    return {
        "artifact_version": QUERY_FREE_AXIS_VERSION,
        "git_commit_sha": _git_sha(),
        "generated_at": _now(),
        "candidate_sets_bound": False,
        "reranking_outcomes_observed": False,
        "environment": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        **values,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows, *, append_rows=()) -> None:
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    serialized += "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in append_rows
    )
    _atomic_text(path, serialized)


def _atomic_npz(path: Path, **arrays) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
