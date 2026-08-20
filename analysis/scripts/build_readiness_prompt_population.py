#!/usr/bin/env python3
"""Plan, generate, score, and refine natural questions on a readiness subspace."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_hf_dataset import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    atomic_text,
    read_json,
    read_jsonl,
    sha256_file,
)
from interpretability.pipeline.readiness_prompt_population import (  # noqa: E402
    READINESS_PROMPT_POPULATION_VERSION,
    FakeReadinessQuestionGenerator,
    LocalReadinessQuestionGenerator,
    ReadinessGenerationTask,
    ReadinessPromptTarget,
    ReadinessQuestionCandidate,
    ReadinessSubspaceBounds,
    build_generation_tasks,
    build_refinement_tasks,
    build_target_grid,
    fit_reference_bounds,
    generate_question_candidates,
    load_readiness_embedding_map,
    project_questions,
    select_diverse_questions,
    validate_generated_question,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecPromptEmbedder,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    plan = stages.add_parser(
        "plan", help="freeze bounds, 30-cell grid, and round tasks"
    )
    plan.add_argument("--keywords", required=True)
    plan.add_argument("--map", required=True)
    plan.add_argument("--reference-coordinates", required=True)
    plan.add_argument("--generator-ids", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--axis-1-points", type=int, default=6)
    plan.add_argument("--axis-2-points", type=int, default=5)
    plan.add_argument("--lower-quantile", type=float, default=0.05)
    plan.add_argument("--upper-quantile", type=float, default=0.95)
    plan.add_argument("--reference-split", default="development")
    plan.add_argument("--round-index", type=int, default=0)
    plan.add_argument("--candidates-per-task", type=int, default=3)
    plan.add_argument("--master-seed", type=int, default=20260820)

    generate = stages.add_parser(
        "generate", help="run one generator model's assigned tasks"
    )
    generate.add_argument("--tasks", required=True)
    generate.add_argument("--generator-id", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument(
        "--backend",
        choices=("local", "api", "openai", "fake"),
        default="local",
    )
    generate.add_argument("--precision", choices=("full", "4bit"), default="full")
    generate.add_argument("--cache-dir", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--max-new-tokens", type=int, default=180)
    generate.add_argument("--maximum-attempts", type=int, default=5)
    generate.add_argument("--start-index", type=int, default=0)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--resume", action="store_true")

    imported = stages.add_parser(
        "import-proposals",
        help="validate external/LLM2Vec-Gen proposals before frozen-map scoring",
    )
    imported.add_argument("--tasks", required=True)
    imported.add_argument("--proposals", required=True)
    imported.add_argument("--generator-id", required=True)
    imported.add_argument("--model", required=True)
    imported.add_argument("--proposal-kind", default="llm2vec-gen-decoded-proposal")
    imported.add_argument("--output", required=True)

    score = stages.add_parser(
        "score-select",
        help="LLM2Vec-project, diversify, and plan refinement",
    )
    score.add_argument("--plan-dir", required=True)
    score.add_argument("--map", required=True)
    score.add_argument("--candidates", nargs="+", required=True)
    score.add_argument("--embedding-model", required=True)
    score.add_argument("--mntp-model")
    score.add_argument("--peft-model")
    score.add_argument("--embedding-batch-size", type=int, default=8)
    score.add_argument("--embedding-max-length", type=int, default=512)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--generator-ids")
    score.add_argument("--next-round-index", type=int, default=1)
    score.add_argument("--distance-tolerance", type=float, default=0.22)
    score.add_argument("--novelty-weight", type=float, default=0.05)
    score.add_argument("--generator-balance-weight", type=float, default=0.02)
    score.add_argument("--candidates-per-task", type=int, default=3)
    score.add_argument("--master-seed", type=int, default=20260820)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.stage == "plan":
        return _plan(args)
    if args.stage == "generate":
        return _generate(args)
    if args.stage == "import-proposals":
        return _import_proposals(args)
    if args.stage == "score-select":
        return _score_select(args)
    raise AssertionError(args.stage)


def _plan(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite plan directory: {output}")
    fitted = load_readiness_embedding_map(args.map)
    coordinate_rows = read_jsonl(args.reference_coordinates)
    bounds = fit_reference_bounds(
        coordinate_rows,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        reference_split=args.reference_split,
    )
    targets = build_target_grid(
        bounds,
        axis_1_points=args.axis_1_points,
        axis_2_points=args.axis_2_points,
    )
    keywords = _read_keywords(args.keywords)
    generator_ids = _csv(args.generator_ids)
    tasks = build_generation_tasks(
        keywords,
        targets,
        generator_ids,
        round_index=args.round_index,
        master_seed=args.master_seed,
        requested_candidate_count=args.candidates_per_task,
    )
    output.mkdir(parents=True)
    atomic_json(output / "subspace_bounds.json", asdict(bounds))
    atomic_jsonl(output / "target_grid.jsonl", (asdict(row) for row in targets))
    atomic_jsonl(output / f"generation_tasks_round_{args.round_index:02d}.jsonl", (_task_row(row) for row in tasks))
    atomic_json(
        output / "plan_manifest.json",
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "map_id": fitted.map_id,
            "map_version": fitted.map_version,
            "embedding_model": fitted.embedding_model,
            "inputs": {
                "keywords": _file_identity(args.keywords),
                "map": _file_identity(args.map),
                "reference_coordinates": _file_identity(
                    args.reference_coordinates
                ),
            },
            "keyword_count": len(keywords),
            "target_count_per_keyword": len(targets),
            "task_count": len(tasks),
            "generator_ids": list(generator_ids),
            "round_index": args.round_index,
            "master_seed": args.master_seed,
            "scientific_guard": (
                "The grid describes prompt embeddings. It does not define or replace B."
            ),
        },
    )
    print(f"keywords={len(keywords)} targets_per_keyword={len(targets)} tasks={len(tasks)}")
    print(f"output={output}")
    return 0


def _generate(args) -> int:
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"refusing to overwrite candidate file: {output}")
    tasks = [task for task in _read_tasks(args.tasks) if task.generator_id == args.generator_id]
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid task slice")
    tasks = tasks[args.start_index : None if args.limit is None else args.start_index + args.limit]
    if not tasks:
        raise ValueError(f"no tasks assigned to generator {args.generator_id}")
    generator = (
        FakeReadinessQuestionGenerator(args.generator_id)
        if args.backend == "fake"
        else LocalReadinessQuestionGenerator.from_model(
            args.model,
            generator_id=args.generator_id,
            cache_directory=args.cache_dir,
            backend=args.backend,
            precision=args.precision,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            maximum_attempts=args.maximum_attempts,
        )
    )
    rows = generate_question_candidates(tasks, generator)
    if output.exists():
        existing = tuple(ReadinessQuestionCandidate(**row) for row in read_jsonl(output))
        by_id = {row.candidate_id: row for row in (*existing, *rows)}
        rows = tuple(by_id[key] for key in sorted(by_id))
    atomic_jsonl(output, (asdict(row) for row in rows))
    atomic_json(
        output.with_suffix(output.suffix + ".manifest.json"),
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "completed_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "tasks": _file_identity(args.tasks),
            "generator_id": args.generator_id,
            "generator_model": generator.model_name,
            "generator_backend": args.backend,
            "generator_precision": args.precision,
            "task_start_index": args.start_index,
            "task_limit": args.limit,
            "task_count": len(tasks),
            "candidate_count": len(rows),
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "maximum_attempts": args.maximum_attempts,
        },
    )
    print(f"generator={args.generator_id} tasks={len(tasks)} candidates={len(rows)}")
    print(f"output={output}")
    return 0


def _import_proposals(args) -> int:
    tasks = {task.task_id: task for task in _read_tasks(args.tasks)}
    rows = []
    for slot, proposal in enumerate(read_jsonl(args.proposals)):
        task_id = str(proposal.get("task_id", ""))
        task = tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown proposal task_id: {task_id}")
        question = " ".join(str(proposal.get("question", "")).split())
        validate_generated_question(question, task.keyword)
        identity = {
            "version": READINESS_PROMPT_POPULATION_VERSION,
            "task_id": task_id,
            "question": question,
            "model": args.model,
            "proposal_kind": args.proposal_kind,
        }
        digest = _stable_hash(identity)
        rows.append(
            ReadinessQuestionCandidate(
                candidate_id=f"readiness-question:{digest[:24]}",
                task_id=task.task_id,
                keyword_id=task.keyword_id,
                keyword=task.keyword,
                target_id=task.target.target_id,
                target_index=task.target.target_index,
                target_normalized_axis_1=task.target.normalized_axis_1,
                target_normalized_axis_2=task.target.normalized_axis_2,
                target_raw_axis_1=task.target.raw_axis_1,
                target_raw_axis_2=task.target.raw_axis_2,
                round_index=task.round_index,
                generator_id=args.generator_id,
                generator_model=args.model,
                candidate_slot=int(proposal.get("candidate_slot", slot)),
                generation_seed=task.generation_seed,
                question=question,
                question_sha256=_sha256_text(question),
                proposal_kind=args.proposal_kind,
            )
        )
    output = Path(args.output).resolve()
    atomic_jsonl(output, (asdict(row) for row in rows))
    atomic_json(
        output.with_suffix(output.suffix + ".manifest.json"),
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "completed_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "tasks": _file_identity(args.tasks),
            "proposals": _file_identity(args.proposals),
            "generator_id": args.generator_id,
            "generator_model": args.model,
            "proposal_kind": args.proposal_kind,
            "candidate_count": len(rows),
        },
    )
    print(f"imported_candidates={len(rows)} output={output}")
    return 0


def _score_select(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite score directory: {output}")
    output.mkdir(parents=True)
    plan = Path(args.plan_dir).resolve()
    plan_manifest = read_json(plan / "plan_manifest.json")
    bounds = ReadinessSubspaceBounds(**read_json(plan / "subspace_bounds.json"))
    targets = tuple(
        ReadinessPromptTarget(**row)
        for row in read_jsonl(plan / "target_grid.jsonl")
    )
    fitted = load_readiness_embedding_map(args.map)
    if plan_manifest.get("map_id") != fitted.map_id:
        raise ValueError("plan and scoring map ids differ")
    _validate_embedding_model_revision(fitted, args.embedding_model)
    candidates = tuple(
        ReadinessQuestionCandidate(**row)
        for path in args.candidates
        for row in read_jsonl(path)
    )
    if not candidates:
        raise ValueError("no candidates to score")
    if len({row.candidate_id for row in candidates}) != len(candidates):
        raise ValueError("candidate files contain duplicate candidate ids")
    targets_by_id = {row.target_id: row for row in targets}
    for candidate in candidates:
        target = targets_by_id.get(candidate.target_id)
        if target is None or (
            candidate.target_normalized_axis_1 != target.normalized_axis_1
            or candidate.target_normalized_axis_2 != target.normalized_axis_2
            or candidate.target_raw_axis_1 != target.raw_axis_1
            or candidate.target_raw_axis_2 != target.raw_axis_2
        ):
            raise ValueError(
                f"candidate target does not match frozen plan: {candidate.candidate_id}"
            )
    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    )
    embeddings = embedder.embed([row.question for row in candidates])
    projections = project_questions(fitted, bounds, candidates, embeddings)
    selected, diagnostics = select_diverse_questions(
        candidates,
        projections,
        embeddings,
        novelty_weight=args.novelty_weight,
        generator_balance_weight=args.generator_balance_weight,
    )
    generator_ids = (
        _csv(args.generator_ids)
        if args.generator_ids
        else tuple(read_json(plan / "plan_manifest.json")["generator_ids"])
    )
    next_tasks = build_refinement_tasks(
        selected,
        targets,
        generator_ids,
        next_round_index=args.next_round_index,
        distance_tolerance=args.distance_tolerance,
        master_seed=args.master_seed,
        requested_candidate_count=args.candidates_per_task,
    )
    atomic_jsonl(output / "candidate_projections.jsonl", (asdict(row) for row in projections))
    atomic_jsonl(output / "selected_questions.jsonl", (asdict(row) for row in selected))
    atomic_jsonl(output / f"generation_tasks_round_{args.next_round_index:02d}.jsonl", (_task_row(row) for row in next_tasks))
    atomic_npz(
        output / "candidate_embeddings.restricted-local.npz",
        candidate_ids=np.asarray([row.candidate_id for row in candidates]),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    diagnostics.update(
        {
            "map_id": fitted.map_id,
            "embedding_backend": embedder.model_name,
            "next_round_task_count": len(next_tasks),
            "distance_tolerance": args.distance_tolerance,
        }
    )
    atomic_json(output / "selection_diagnostics.json", diagnostics)
    atomic_json(
        output / "run_manifest.json",
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "map_id": fitted.map_id,
            "map": _file_identity(args.map),
                "plan_manifest": _file_identity(plan / "plan_manifest.json"),
            "candidate_files": [
                _file_identity(path) for path in args.candidates
            ],
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "next_round_task_count": len(next_tasks),
            "proposal_kinds": sorted({row.proposal_kind for row in candidates}),
            "embedding": {
                "model": str(Path(args.embedding_model).resolve()),
                "mntp_model": (
                    str(Path(args.mntp_model).resolve()) if args.mntp_model else None
                ),
                "peft_model": (
                    str(Path(args.peft_model).resolve()) if args.peft_model else None
                ),
                "batch_size": args.embedding_batch_size,
                "max_length": args.embedding_max_length,
            },
            "selection": {
                "distance_tolerance": args.distance_tolerance,
                "novelty_weight": args.novelty_weight,
                "generator_balance_weight": args.generator_balance_weight,
            },
            "scientific_guard": (
                "Generated texts are re-embedded before acceptance; LLM2Vec-Gen decoder "
                "states are never treated as the frozen readiness coordinate system."
            ),
        },
    )
    atomic_text(output / "readiness_question_population_report.md", _report(diagnostics, len(next_tasks)))
    print(f"candidates={len(candidates)} selected={len(selected)} next_round_tasks={len(next_tasks)}")
    print(f"output={output}")
    return 0


def _read_keywords(path: str | Path) -> tuple[tuple[str, str], ...]:
    path = Path(path)
    if path.suffix == ".jsonl":
        values = []
        for row in read_jsonl(path):
            keyword = " ".join(str(row.get("keyword", "")).split())
            keyword_id = str(row.get("keyword_id") or f"keyword:{_sha256_text(keyword)[:24]}")
            values.append((keyword_id, keyword))
    else:
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            keyword = " ".join(line.split())
            if keyword:
                values.append((f"keyword:{_sha256_text(keyword)[:24]}", keyword))
    if not values or any(not keyword for _, keyword in values):
        raise ValueError("keyword input must contain at least one nonempty keyword")
    if len({keyword_id for keyword_id, _ in values}) != len(values):
        raise ValueError("keyword ids must be unique")
    return tuple(values)


def _read_tasks(path: str | Path) -> tuple[ReadinessGenerationTask, ...]:
    rows = []
    for row in read_jsonl(path):
        payload = dict(row)
        payload["target"] = ReadinessPromptTarget(**payload["target"])
        rows.append(ReadinessGenerationTask(**payload))
    return tuple(rows)


def _task_row(task: ReadinessGenerationTask) -> dict[str, object]:
    return asdict(task)


def _report(diagnostics: dict[str, object], next_task_count: int) -> str:
    return f"""# Readiness-question population

- Candidates scored: {diagnostics['candidate_count']}
- Questions selected: {diagnostics['selected_count']}
- Keywords: {diagnostics['keyword_count']}
- Mean normalized target distance: {diagnostics['mean_target_distance']}
- Maximum normalized target distance: {diagnostics['maximum_target_distance']}
- Cells scheduled for the next round: {next_task_count}

Generator models propose text; the frozen LLM2Vec map measures the result.  The
coordinates describe question semantics and do not define the experimental
policy variable B.  LLM2Vec-Gen output, when supplied, is only a proposal source:
decoded text must be re-embedded and pass the same selection contract.
"""


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated value must not be empty")
    return values


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _stable_hash(value: object) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _slurm_environment() -> dict[str, str | None]:
    import os

    names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
    )
    return {name.lower(): os.getenv(name) for name in names}


def _validate_embedding_model_revision(fitted, model_path: str | Path) -> None:
    """Reject an obvious base-model/map mismatch before loading a large model."""

    reference = fitted.embedding_model.rsplit("@", 1)
    if len(reference) != 2:
        raise ValueError(
            "frozen map lacks an @revision embedding identity; refusing to guess"
        )
    expected_revision = reference[1]
    path = Path(model_path)
    if path.exists() and path.resolve().name != expected_revision:
        raise ValueError(
            "embedding base-model revision does not match frozen map: "
            f"expected {expected_revision}, got {path.resolve().name}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
