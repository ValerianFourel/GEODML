#!/usr/bin/env python3
"""Plan, generate, score, and refine natural questions on a readiness subspace."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

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
    LocalSearchQuestionValidator,
    LocalReadinessQuestionGenerator,
    QuestionGenerationExhaustedError,
    ReadinessGenerationTask,
    ReadinessPromptTarget,
    ReadinessQuestionCandidate,
    ReadinessSubspaceBounds,
    audit_question_diversity,
    build_generation_tasks,
    build_refinement_tasks,
    build_support_aware_keyword_targets,
    build_target_grid,
    fit_reference_bounds,
    generate_question_candidates,
    load_readiness_embedding_map,
    project_questions,
    project_text_embeddings,
    select_diverse_questions,
    select_spatially_matched_questions,
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
    plan.add_argument(
        "--target-design",
        choices=("rectangular-grid", "support-aware-random"),
        default="rectangular-grid",
    )
    plan.add_argument("--targets-per-keyword", type=int, default=30)
    plan.add_argument("--support-grid-resolution", type=int, default=20)
    plan.add_argument("--minimum-support-bin-count", type=int, default=3)
    plan.add_argument(
        "--support-include-unusable",
        action="store_true",
        help="include development rows that were not usable for axis labels",
    )
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
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--shard-count", type=int, default=1)
    generate.add_argument(
        "--maximum-runtime-seconds",
        type=float,
        help="stop cleanly between tasks after this elapsed runtime",
    )
    generate.add_argument("--resume", action="store_true")
    generate.add_argument(
        "--allow-failed-tasks",
        action="store_true",
        help=(
            "record tasks that exhaust generation validation and continue the "
            "slice; unexpected model/runtime failures still abort"
        ),
    )

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

    validate = stages.add_parser(
        "validate-candidates",
        help="independently judge topic fidelity and online-search usefulness",
    )
    validate.add_argument("--candidates", nargs="+", required=True)
    validate.add_argument("--judge-id", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument(
        "--backend", choices=("local", "api", "openai"), default="local"
    )
    validate.add_argument("--precision", choices=("full", "4bit"), default="full")
    validate.add_argument("--cache-dir", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--maximum-attempts", type=int, default=3)
    validate.add_argument("--shard-count", type=int, default=1)
    validate.add_argument("--shard-index", type=int, default=0)
    validate.add_argument("--resume", action="store_true")

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

    project = stages.add_parser(
        "project-candidates",
        help="project the same candidate texts through one frozen embedding map",
    )
    project.add_argument("--candidates", nargs="+", required=True)
    project.add_argument("--map", required=True)
    project.add_argument("--reference-coordinates", required=True)
    project.add_argument("--embedding-model", required=True)
    project.add_argument("--mntp-model")
    project.add_argument("--peft-model")
    project.add_argument("--embedding-batch-size", type=int, default=8)
    project.add_argument("--embedding-max-length", type=int, default=512)
    project.add_argument("--output-dir", required=True)

    compare = stages.add_parser(
        "compare-projections",
        help="align and compare Qwen/Mistral projections of identical questions",
    )
    compare.add_argument("--reference-projections", required=True)
    compare.add_argument("--candidate-projections", required=True)
    compare.add_argument("--robustness-battery", required=True)
    compare.add_argument("--output-dir", required=True)

    spatial = stages.add_parser(
        "spatial-select",
        help="globally match validated candidates to the grid in both embeddings",
    )
    spatial.add_argument("--plan-dir", required=True)
    spatial.add_argument("--candidates", nargs="+", required=True)
    spatial.add_argument("--reference-projections", required=True)
    spatial.add_argument("--candidate-projections", required=True)
    spatial.add_argument("--robustness-battery", required=True)
    spatial.add_argument("--validations", nargs="+", required=True)
    spatial.add_argument("--generator-ids", required=True)
    spatial.add_argument("--next-round-index", type=int, default=1)
    spatial.add_argument("--distance-tolerance", type=float, default=0.22)
    spatial.add_argument(
        "--require-both-views-within-tolerance",
        action="store_true",
        help=(
            "accept a target assignment only when both frozen embedding views "
            "independently fall within the distance tolerance"
        ),
    )
    spatial.add_argument(
        "--require-delexicalized-template-uniqueness",
        action="store_true",
        help=(
            "retain at most one selected question for each exact template after "
            "removing its keyword phrase"
        ),
    )
    spatial.add_argument("--disagreement-weight", type=float, default=0.10)
    spatial.add_argument("--candidates-per-task", type=int, default=3)
    spatial.add_argument("--master-seed", type=int, default=20260820)
    spatial.add_argument("--output-dir", required=True)

    diversity = stages.add_parser(
        "audit-diversity",
        help="fail closed on repeated cross-keyword question templates",
    )
    diversity.add_argument("--questions", nargs="+", required=True)
    diversity.add_argument("--output-dir", required=True)
    diversity.add_argument(
        "--minimum-delexicalized-unique-fraction", type=float, default=0.90
    )
    diversity.add_argument("--maximum-template-fraction", type=float, default=0.01)
    diversity.add_argument(
        "--minimum-median-keyword-unique-fraction", type=float, default=0.90
    )
    diversity.add_argument(
        "--minimum-keyword-unique-fraction", type=float, default=0.70
    )
    diversity.add_argument(
        "--maximum-opening-frame-fraction", type=float, default=0.05
    )
    diversity.add_argument("--opening-frame-tokens", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.stage == "plan":
        return _plan(args)
    if args.stage == "generate":
        return _generate(args)
    if args.stage == "import-proposals":
        return _import_proposals(args)
    if args.stage == "validate-candidates":
        return _validate_candidates(args)
    if args.stage == "score-select":
        return _score_select(args)
    if args.stage == "project-candidates":
        return _project_candidates(args)
    if args.stage == "compare-projections":
        return _compare_projections(args)
    if args.stage == "spatial-select":
        return _spatial_select(args)
    if args.stage == "audit-diversity":
        return _audit_diversity(args)
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
    keywords = _read_keywords(args.keywords)
    generator_ids = _csv(args.generator_ids)
    support_diagnostics = None
    if args.target_design == "support-aware-random":
        targets, support_diagnostics = build_support_aware_keyword_targets(
            coordinate_rows,
            bounds,
            keywords,
            targets_per_keyword=args.targets_per_keyword,
            support_grid_resolution=args.support_grid_resolution,
            minimum_support_bin_count=args.minimum_support_bin_count,
            master_seed=args.master_seed,
            require_usable_for_axis=not args.support_include_unusable,
        )
        target_count_per_keyword = args.targets_per_keyword
    else:
        targets = build_target_grid(
            bounds,
            axis_1_points=args.axis_1_points,
            axis_2_points=args.axis_2_points,
        )
        target_count_per_keyword = len(targets)
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
    if isinstance(targets, dict):
        keyword_text = dict(keywords)
        atomic_jsonl(
            output / "keyword_target_grid.jsonl",
            (
                {
                    "keyword_id": keyword_id,
                    "keyword": keyword_text[keyword_id],
                    "target": asdict(target),
                }
                for keyword_id in keyword_text
                for target in targets[keyword_id]
            ),
        )
        atomic_json(output / "support_design.json", support_diagnostics)
    else:
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
            "target_design": args.target_design,
            "generation_control": (
                "continuous-adjacent-anchor-blend-v1"
                if args.target_design == "support-aware-random"
                else "legacy-categorical-v1"
            ),
            "target_count_per_keyword": target_count_per_keyword,
            "task_count": len(tasks),
            "requested_candidates_per_task": args.candidates_per_task,
            "maximum_planned_candidate_count": len(tasks)
            * args.candidates_per_task,
            "generator_ids": list(generator_ids),
            "round_index": args.round_index,
            "master_seed": args.master_seed,
            "scientific_guard": (
                "The target design describes prompt embeddings. It does not define "
                "or replace B."
            ),
        },
    )
    print(
        f"keywords={len(keywords)} "
        f"targets_per_keyword={target_count_per_keyword} tasks={len(tasks)}"
    )
    print(f"output={output}")
    return 0


def _generate(args) -> int:
    started = time.monotonic()
    output = Path(args.output).resolve()
    failure_output = output.with_suffix(output.suffix + ".failures.jsonl")
    if output.exists() and not args.resume:
        raise ValueError(f"refusing to overwrite candidate file: {output}")
    if failure_output.exists() and not args.resume:
        raise ValueError(f"refusing to overwrite task-failure file: {failure_output}")
    tasks = [task for task in _read_tasks(args.tasks) if task.generator_id == args.generator_id]
    if (
        args.start_index < 0
        or (args.limit is not None and args.limit <= 0)
        or args.shard_count <= 0
        or not 0 <= args.shard_index < args.shard_count
        or (
            args.maximum_runtime_seconds is not None
            and args.maximum_runtime_seconds <= 0
        )
    ):
        raise ValueError("invalid task slice")
    tasks = tasks[args.shard_index :: args.shard_count]
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
    existing = tuple(
        ReadinessQuestionCandidate(**row)
        for row in (read_jsonl(output) if output.exists() else [])
    )
    expected_task_ids = {task.task_id for task in tasks}
    if any(row.task_id not in expected_task_ids for row in existing):
        raise ValueError("resume output contains candidates outside the requested slice")
    by_id = {row.candidate_id: row for row in existing}
    existing_counts = Counter(row.task_id for row in existing)
    requested_counts = {
        task.task_id: task.requested_candidate_count for task in tasks
    }
    if any(
        count > requested_counts[task_id]
        for task_id, count in existing_counts.items()
    ):
        raise ValueError("resume output contains too many candidates for a task")
    completed_task_ids = {
        task_id
        for task_id, count in existing_counts.items()
        if count == requested_counts[task_id]
    }
    existing_failures = {
        str(row["task_id"]): row
        for row in (read_jsonl(failure_output) if failure_output.exists() else [])
    }
    if any(task_id not in expected_task_ids for task_id in existing_failures):
        raise ValueError("resume failure file contains tasks outside the requested slice")
    if existing_failures and not args.allow_failed_tasks:
        raise ValueError("resume failure file requires --allow-failed-tasks")
    failed_by_task = dict(existing_failures)
    generated_task_count = 0
    for task in tasks:
        if task.task_id in completed_task_ids or task.task_id in failed_by_task:
            continue
        if (
            args.maximum_runtime_seconds is not None
            and time.monotonic() - started >= args.maximum_runtime_seconds
        ):
            break
        try:
            new_rows = generate_question_candidates((task,), generator)
        except QuestionGenerationExhaustedError as exc:
            if not args.allow_failed_tasks:
                raise
            failed_by_task[task.task_id] = {
                "task_id": task.task_id,
                "keyword_id": task.keyword_id,
                "keyword": task.keyword,
                "generator_id": task.generator_id,
                "round_index": task.round_index,
                "failure_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at": _now(),
            }
            atomic_jsonl(
                failure_output,
                (failed_by_task[key] for key in sorted(failed_by_task)),
            )
            print(
                f"failed_task={task.task_id} failed_tasks={len(failed_by_task)} "
                f"error={exc}",
                flush=True,
            )
            continue
        by_id.update((row.candidate_id, row) for row in new_rows)
        completed_task_ids.add(task.task_id)
        generated_task_count += 1
        if generated_task_count % 10 == 0:
            print(
                f"generated_tasks={len(completed_task_ids)}/{len(tasks)} "
                f"candidates={len(by_id)}"
            )
    rows = tuple(by_id[key] for key in sorted(by_id))
    atomic_jsonl(output, (asdict(row) for row in rows))
    atomic_jsonl(
        failure_output,
        (failed_by_task[key] for key in sorted(failed_by_task)),
    )
    elapsed_seconds = time.monotonic() - started
    slice_complete = len(completed_task_ids) == len(tasks)
    slice_terminal = len(completed_task_ids) + len(failed_by_task) == len(tasks)
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
            "task_shard_index": args.shard_index,
            "task_shard_count": args.shard_count,
            "requested_task_count": len(tasks),
            "completed_task_count": len(completed_task_ids),
            "failed_task_count": len(failed_by_task),
            "failed_task_ids": sorted(failed_by_task),
            "generated_task_count_this_invocation": generated_task_count,
            "slice_complete": slice_complete,
            "slice_terminal": slice_terminal,
            "candidate_count": len(rows),
            "elapsed_seconds": elapsed_seconds,
            "maximum_runtime_seconds": args.maximum_runtime_seconds,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "maximum_attempts": args.maximum_attempts,
            "allow_failed_tasks": args.allow_failed_tasks,
        },
    )
    print(
        f"generator={args.generator_id} tasks={len(completed_task_ids)}/{len(tasks)} "
        f"failed_tasks={len(failed_by_task)} candidates={len(rows)} "
        f"elapsed_seconds={elapsed_seconds:.1f}"
    )
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


def _validate_candidates(args) -> int:
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise ValueError(f"refusing to overwrite validation file: {output}")
    all_candidates = tuple(
        ReadinessQuestionCandidate(**row)
        for path in args.candidates
        for row in read_jsonl(path)
    )
    if not all_candidates or len(
        {row.candidate_id for row in all_candidates}
    ) != len(all_candidates):
        raise ValueError("validation candidates must be nonempty and uniquely identified")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("validation shard must satisfy 0 <= index < count")
    candidates = tuple(
        row
        for row in all_candidates
        if int(_sha256_text(row.candidate_id)[:16], 16) % args.shard_count
        == args.shard_index
    )
    existing = {
        str(row["candidate_id"]): row
        for row in (read_jsonl(output) if output.exists() else [])
    }
    candidate_ids = {row.candidate_id for row in candidates}
    if not set(existing).issubset(candidate_ids):
        raise ValueError("existing validation contains unknown candidate ids")
    pending = [row for row in candidates if row.candidate_id not in existing]
    if pending:
        validator = LocalSearchQuestionValidator.from_model(
            args.model,
            judge_id=args.judge_id,
            cache_directory=args.cache_dir,
            backend=args.backend,
            precision=args.precision,
            maximum_attempts=args.maximum_attempts,
        )
        for index, candidate in enumerate(pending, start=1):
            review = validator.review(candidate)
            existing[candidate.candidate_id] = asdict(review)
            if index % 10 == 0 or index == len(pending):
                print(
                    f"validated={len(existing)}/{len(candidates)} "
                    f"accepted={sum(bool(row['accepted']) for row in existing.values())}"
                )
    rows = [existing[candidate_id] for candidate_id in sorted(existing)]
    atomic_jsonl(output, rows)
    atomic_json(
        output.with_suffix(output.suffix + ".manifest.json"),
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "completed_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "candidate_files": [_file_identity(path) for path in args.candidates],
            "candidate_count": len(candidates),
            "total_candidate_count": len(all_candidates),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "reviewed_count": len(rows),
            "accepted_count": sum(bool(row["accepted"]) for row in rows),
            "judge_id": args.judge_id,
            "judge_model": args.model,
            "judge_backend": args.backend,
            "judge_precision": args.precision,
            "acceptance_contract": (
                "Exact keyword, one question, topic relevance, search intent, web "
                "answerability, standalone wording, natural language, and score >= 4/5."
            ),
        },
    )
    print(
        f"reviewed={len(rows)} accepted="
        f"{sum(bool(row['accepted']) for row in rows)} output={output}"
    )
    return 0


def _score_select(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite score directory: {output}")
    output.mkdir(parents=True)
    plan = Path(args.plan_dir).resolve()
    plan_manifest = read_json(plan / "plan_manifest.json")
    bounds = ReadinessSubspaceBounds(**read_json(plan / "subspace_bounds.json"))
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
    keywords = sorted({(row.keyword_id, row.keyword) for row in candidates})
    targets, _ = _read_plan_targets(plan, keywords)
    if isinstance(targets, dict):
        targets_by_key = {
            (keyword_id, target.target_id): target
            for keyword_id, keyword_targets in targets.items()
            for target in keyword_targets
        }
    else:
        targets_by_key = {
            (keyword_id, target.target_id): target
            for keyword_id, _ in keywords
            for target in targets
        }
    for candidate in candidates:
        target = targets_by_key.get((candidate.keyword_id, candidate.target_id))
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


def _project_candidates(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite projection directory: {output}")
    output.mkdir(parents=True)
    fitted = load_readiness_embedding_map(args.map)
    _validate_embedding_model_revision(fitted, args.embedding_model)
    bounds = fit_reference_bounds(read_jsonl(args.reference_coordinates))
    candidates = tuple(
        ReadinessQuestionCandidate(**row)
        for path in args.candidates
        for row in read_jsonl(path)
    )
    if not candidates or len({row.candidate_id for row in candidates}) != len(candidates):
        raise ValueError("projection candidates must be nonempty and uniquely identified")
    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    )
    embeddings = embedder.embed([row.question for row in candidates])
    projections = project_text_embeddings(
        fitted,
        bounds,
        item_ids=[row.candidate_id for row in candidates],
        text_sha256s=[row.question_sha256 for row in candidates],
        embeddings=embeddings,
    )
    projection_by_id = {row.item_id: row for row in projections}
    atomic_jsonl(
        output / "question_projections.jsonl",
        (
            {
                **asdict(candidate),
                "projection": asdict(projection_by_id[candidate.candidate_id]),
            }
            for candidate in candidates
        ),
    )
    atomic_npz(
        output / "question_embeddings.restricted-local.npz",
        candidate_ids=np.asarray([row.candidate_id for row in candidates]),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )
    atomic_json(
        output / "projection_manifest.json",
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "map_id": fitted.map_id,
            "map": _file_identity(args.map),
            "reference_coordinates": _file_identity(args.reference_coordinates),
            "candidate_files": [_file_identity(path) for path in args.candidates],
            "candidate_count": len(candidates),
            "embedding": {
                "model": str(Path(args.embedding_model).resolve()),
                "mntp_model": str(Path(args.mntp_model).resolve()) if args.mntp_model else None,
                "peft_model": str(Path(args.peft_model).resolve()) if args.peft_model else None,
                "batch_size": args.embedding_batch_size,
                "max_length": args.embedding_max_length,
            },
        },
    )
    print(f"map_id={fitted.map_id} candidates={len(candidates)}")
    print(f"output={output}")
    return 0


def _compare_projections(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite comparison directory: {output}")
    reference_root = Path(args.reference_projections).resolve()
    candidate_root = Path(args.candidate_projections).resolve()
    battery_root = Path(args.robustness_battery).resolve()
    rows, identities, summary = _aligned_projection_rows(
        reference_root, candidate_root, battery_root
    )
    output.mkdir(parents=True)
    atomic_jsonl(output / "aligned_question_projections.jsonl", rows)
    atomic_json(output / "projection_comparison.json", summary)
    atomic_text(
        output / "projection_comparison_report.md",
        _projection_comparison_report(summary),
    )
    atomic_json(
        output / "comparison_manifest.json",
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            **identities,
            "candidate_count": len(rows),
        },
    )
    print(
        f"axis_1_spearman={summary['axis_1']['spearman']:.4f} "
        f"axis_2_spearman={summary['axis_2']['spearman']:.4f} "
        f"scalar_spearman={summary['scalar_readiness']['spearman']:.4f}"
    )
    print(f"output={output}")
    return 0


def _aligned_projection_rows(reference_root, candidate_root, battery_root):
    reference_manifest = read_json(reference_root / "projection_manifest.json")
    candidate_manifest = read_json(candidate_root / "projection_manifest.json")
    battery_manifest = read_json(battery_root / "battery_manifest.json")
    if reference_manifest["map_id"] != battery_manifest["reference_map_id"]:
        raise ValueError("reference projection map does not match battery")
    if candidate_manifest["map_id"] != battery_manifest["candidate_map_id"]:
        raise ValueError("candidate projection map does not match battery")
    battery = read_json(battery_root / "readiness_robustness_battery.json")
    alignment = battery["cross_embedding_alignment"]
    reference = _projection_index(reference_root / "question_projections.jsonl")
    candidate = _projection_index(candidate_root / "question_projections.jsonl")
    if set(reference) != set(candidate):
        raise ValueError("projection candidate identities differ")
    rotation = np.asarray(alignment["orthogonal_rotation"], dtype=np.float64)
    reference_mean = np.asarray(alignment["reference_development_mean"])
    reference_scale = np.asarray(alignment["reference_development_scale"])
    candidate_mean = np.asarray(alignment["candidate_development_mean"])
    candidate_scale = np.asarray(alignment["candidate_development_scale"])
    rows = []
    reference_z_rows = []
    candidate_z_rows = []
    reference_scalar = []
    candidate_scalar = []
    for candidate_id in sorted(reference):
        left = reference[candidate_id]
        right = candidate[candidate_id]
        if left["text_sha256"] != right["text_sha256"]:
            raise ValueError(f"projection text hash differs: {candidate_id}")
        left_raw = np.asarray([left["raw_axis_1"], left["raw_axis_2"]])
        right_raw = np.asarray([right["raw_axis_1"], right["raw_axis_2"]])
        left_z = (left_raw - reference_mean) / reference_scale
        right_z = ((right_raw - candidate_mean) / candidate_scale) @ rotation
        right_aligned_raw = reference_mean + right_z * reference_scale
        reference_z_rows.append(left_z)
        candidate_z_rows.append(right_z)
        reference_scalar.append(left["predicted_scalar_readiness_0_1"])
        candidate_scalar.append(right["predicted_scalar_readiness_0_1"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "text_sha256": left["text_sha256"],
                "reference_raw_axis_1": float(left_raw[0]),
                "reference_raw_axis_2": float(left_raw[1]),
                "candidate_aligned_raw_axis_1": float(right_aligned_raw[0]),
                "candidate_aligned_raw_axis_2": float(right_aligned_raw[1]),
                "reference_axis_1_z": float(left_z[0]),
                "reference_axis_2_z": float(left_z[1]),
                "candidate_aligned_axis_1_z": float(right_z[0]),
                "candidate_aligned_axis_2_z": float(right_z[1]),
                "axis_1_absolute_difference": float(abs(left_z[0] - right_z[0])),
                "axis_2_absolute_difference": float(abs(left_z[1] - right_z[1])),
                "reference_scalar_readiness": float(reference_scalar[-1]),
                "candidate_scalar_readiness": float(candidate_scalar[-1]),
            }
        )
    reference_z = np.asarray(reference_z_rows)
    candidate_z = np.asarray(candidate_z_rows)
    summary = {
        "format_version": READINESS_PROMPT_POPULATION_VERSION,
        "candidate_count": len(rows),
        "reference_map_id": reference_manifest["map_id"],
        "candidate_map_id": candidate_manifest["map_id"],
        "axis_1": _vector_agreement(reference_z[:, 0], candidate_z[:, 0]),
        "axis_2": _vector_agreement(reference_z[:, 1], candidate_z[:, 1]),
        "scalar_readiness": _vector_agreement(
            np.asarray(reference_scalar), np.asarray(candidate_scalar)
        ),
        "interpretation_guard": (
            "Alignment was learned only from the original development corpus; "
            "these generated questions were not used to fit it."
        ),
    }
    identities = {
        "reference_projection_manifest": _file_identity(
            reference_root / "projection_manifest.json"
        ),
        "candidate_projection_manifest": _file_identity(
            candidate_root / "projection_manifest.json"
        ),
        "robustness_battery_manifest": _file_identity(
            battery_root / "battery_manifest.json"
        ),
    }
    return rows, identities, summary


def _spatial_select(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite spatial selection directory: {output}")
    plan = Path(args.plan_dir).resolve()
    bounds = ReadinessSubspaceBounds(**read_json(plan / "subspace_bounds.json"))
    candidates = tuple(
        ReadinessQuestionCandidate(**row)
        for path in args.candidates
        for row in read_jsonl(path)
    )
    if not candidates or len({row.candidate_id for row in candidates}) != len(candidates):
        raise ValueError("spatial selection candidates must be nonempty and unique")
    keywords = sorted({(row.keyword_id, row.keyword) for row in candidates})
    targets, target_design = _read_plan_targets(plan, keywords)

    aligned_rows, identities, agreement = _aligned_projection_rows(
        Path(args.reference_projections).resolve(),
        Path(args.candidate_projections).resolve(),
        Path(args.robustness_battery).resolve(),
    )
    aligned = {str(row["candidate_id"]): row for row in aligned_rows}
    candidate_ids = {row.candidate_id for row in candidates}
    if set(aligned) != candidate_ids:
        raise ValueError("aligned projections and candidate identities differ")

    accepted_sets = []
    validation_identities = []
    for path in args.validations:
        reviews = {str(row["candidate_id"]): row for row in read_jsonl(path)}
        if set(reviews) != candidate_ids:
            raise ValueError(f"validation does not cover the exact candidate set: {path}")
        accepted_sets.append(
            {candidate_id for candidate_id, row in reviews.items() if bool(row["accepted"])}
        )
        validation_identities.append(_file_identity(path))
    accepted = set.intersection(*accepted_sets)

    coordinates = {}
    for candidate_id, row in aligned.items():
        reference_axis_1 = _normalize_coordinate(
            row["reference_raw_axis_1"], bounds.axis_1_low, bounds.axis_1_high
        )
        reference_axis_2 = _normalize_coordinate(
            row["reference_raw_axis_2"], bounds.axis_2_low, bounds.axis_2_high
        )
        candidate_axis_1 = _normalize_coordinate(
            row["candidate_aligned_raw_axis_1"], bounds.axis_1_low, bounds.axis_1_high
        )
        candidate_axis_2 = _normalize_coordinate(
            row["candidate_aligned_raw_axis_2"], bounds.axis_2_low, bounds.axis_2_high
        )
        coordinates[candidate_id] = {
            "reference_normalized_axis_1": reference_axis_1,
            "reference_normalized_axis_2": reference_axis_2,
            "candidate_aligned_normalized_axis_1": candidate_axis_1,
            "candidate_aligned_normalized_axis_2": candidate_axis_2,
            "consensus_normalized_axis_1": (reference_axis_1 + candidate_axis_1) / 2,
            "consensus_normalized_axis_2": (reference_axis_2 + candidate_axis_2) / 2,
            "cross_embedding_disagreement": float(
                np.hypot(
                    reference_axis_1 - candidate_axis_1,
                    reference_axis_2 - candidate_axis_2,
                )
            ),
        }

    selected, diagnostics = select_spatially_matched_questions(
        candidates,
        targets,
        coordinates,
        accepted_candidate_ids=accepted,
        disagreement_weight=args.disagreement_weight,
        distance_tolerance=args.distance_tolerance,
        target_design=target_design,
        require_both_views_within_tolerance=getattr(
            args, "require_both_views_within_tolerance", False
        ),
        require_delexicalized_template_uniqueness=getattr(
            args, "require_delexicalized_template_uniqueness", False
        ),
    )
    generators = _csv(args.generator_ids)
    targets_by_keyword = (
        targets
        if isinstance(targets, dict)
        else {keyword_id: targets for keyword_id, _ in keywords}
    )
    selected_by_key = {(row.keyword_id, row.target_id): row for row in selected}
    bad_pairs = set()
    feedback = {}
    accepted_candidates_by_keyword = {}
    for candidate in candidates:
        if candidate.candidate_id in accepted:
            accepted_candidates_by_keyword.setdefault(candidate.keyword_id, []).append(
                candidate
            )
    for keyword_id, _ in keywords:
        for target in targets_by_keyword[keyword_id]:
            row = selected_by_key.get((keyword_id, target.target_id))
            if (
                row is None
                or row.target_distance > args.distance_tolerance
                or (
                    getattr(args, "require_both_views_within_tolerance", False)
                    and not row.both_views_within_tolerance
                )
            ):
                bad_pairs.add((keyword_id, target.target_id))
                if row is None:
                    measured_feedback = _dual_view_refinement_feedback(
                        target,
                        accepted_candidates_by_keyword.get(keyword_id, ()),
                        coordinates,
                    )
                    feedback[(keyword_id, target.target_id)] = (
                        measured_feedback
                        or "No independently validated candidate covered this cell."
                    )
                else:
                    delta_1 = target.normalized_axis_1 - row.consensus_normalized_axis_1
                    delta_2 = target.normalized_axis_2 - row.consensus_normalized_axis_2
                    feedback[(keyword_id, target.target_id)] = (
                        f"The closest validated question landed at "
                        f"({row.consensus_normalized_axis_1:.3f}, "
                        f"{row.consensus_normalized_axis_2:.3f}); shift axis 1 by "
                        f"{delta_1:+.3f} and axis 2 by {delta_2:+.3f}. "
                        f"Rewrite this closest question with the smallest semantic "
                        f"change while preserving the exact keyword: {row.question}"
                    )
    all_next_tasks = build_generation_tasks(
        keywords,
        targets,
        generators,
        round_index=args.next_round_index,
        master_seed=args.master_seed,
        requested_candidate_count=args.candidates_per_task,
        feedback_by_keyword_target=feedback,
    )
    next_tasks = tuple(
        row
        for row in all_next_tasks
        if (row.keyword_id, row.target.target_id) in bad_pairs
    )

    diagnostics.update(
        {
            "validation_file_count": len(args.validations),
            "jointly_accepted_candidate_count": len(accepted),
            "next_round_task_count": len(next_tasks),
            "distance_tolerance": args.distance_tolerance,
            "disagreement_weight": args.disagreement_weight,
            "require_both_views_within_tolerance": getattr(
                args, "require_both_views_within_tolerance", False
            ),
            "require_delexicalized_template_uniqueness": getattr(
                args, "require_delexicalized_template_uniqueness", False
            ),
            "cross_embedding_agreement": agreement,
        }
    )
    output.mkdir(parents=True)
    atomic_jsonl(output / "spatially_selected_questions.jsonl", (asdict(row) for row in selected))
    atomic_jsonl(
        output / f"generation_tasks_round_{args.next_round_index:02d}.jsonl",
        (_task_row(row) for row in next_tasks),
    )
    atomic_json(output / "spatial_coverage_diagnostics.json", diagnostics)
    atomic_text(output / "spatial_coverage_report.md", _spatial_report(diagnostics))
    atomic_json(
        output / "run_manifest.json",
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "plan_manifest": _file_identity(plan / "plan_manifest.json"),
            "candidate_files": [_file_identity(path) for path in args.candidates],
            "validation_files": validation_identities,
            **identities,
            "candidate_count": len(candidates),
            "jointly_accepted_candidate_count": len(accepted),
            "selected_count": len(selected),
            "next_round_task_count": len(next_tasks),
            "coordinate_acceptance_contract": {
                "version": "strict-dual-frozen-view-target-verification-v1",
                "enabled": getattr(
                    args, "require_both_views_within_tolerance", False
                ),
                "distance_tolerance": args.distance_tolerance,
                "rule": (
                    "Both the frozen reference-view coordinate and the "
                    "development-aligned second-view coordinate must be within "
                    "the preregistered Euclidean tolerance of the assigned target."
                ),
            },
            "surface_acceptance_contract": {
                "version": "exact-delexicalized-template-uniqueness-v1",
                "enabled": getattr(
                    args, "require_delexicalized_template_uniqueness", False
                ),
                "rule": (
                    "After replacing the exact keyword phrase with one sentinel "
                    "and normalizing case, punctuation, and numbers, no selected "
                    "question template may occur more than once."
                ),
            },
            "scientific_guard": (
                "Aligned prompt coordinates describe text and do not define the randomized policy B."
            ),
        },
    )
    print(
        f"accepted={len(accepted)}/{len(candidates)} selected={len(selected)} "
        f"next_round_tasks={len(next_tasks)}"
    )
    print(f"spacing_gate_passed={diagnostics['overall_spacing_gate_passed']}")
    print(f"output={output}")
    return 0


def _dual_view_refinement_feedback(target, candidates, coordinates):
    """Describe the closest measured proposal without accepting it."""

    measured = []
    for candidate in candidates:
        coordinate = coordinates.get(candidate.candidate_id)
        if coordinate is None:
            continue
        reference = np.asarray(
            [
                coordinate["reference_normalized_axis_1"],
                coordinate["reference_normalized_axis_2"],
            ],
            dtype=np.float64,
        )
        second_view = np.asarray(
            [
                coordinate["candidate_aligned_normalized_axis_1"],
                coordinate["candidate_aligned_normalized_axis_2"],
            ],
            dtype=np.float64,
        )
        target_coordinate = np.asarray(
            [target.normalized_axis_1, target.normalized_axis_2],
            dtype=np.float64,
        )
        reference_distance = float(np.linalg.norm(target_coordinate - reference))
        second_view_distance = float(
            np.linalg.norm(target_coordinate - second_view)
        )
        measured.append(
            (
                max(reference_distance, second_view_distance),
                reference_distance + second_view_distance,
                candidate.candidate_id,
                candidate,
                reference,
                second_view,
                target_coordinate,
            )
        )
    if not measured:
        return None
    (
        _,
        _,
        _,
        candidate,
        reference,
        second_view,
        target_coordinate,
    ) = min(measured, key=lambda row: row[:3])
    reference_delta = target_coordinate - reference
    second_view_delta = target_coordinate - second_view
    return (
        f"The closest independently validated question landed in frozen Qwen "
        f"LLM2Vec at ({reference[0]:.3f}, {reference[1]:.3f}) and in the "
        f"development-aligned Mistral LLM2Vec view at "
        f"({second_view[0]:.3f}, {second_view[1]:.3f}), while the target is "
        f"({target_coordinate[0]:.3f}, {target_coordinate[1]:.3f}). "
        f"The Qwen-view shift needed is ({reference_delta[0]:+.3f}, "
        f"{reference_delta[1]:+.3f}); the aligned Mistral-view shift needed is "
        f"({second_view_delta[0]:+.3f}, {second_view_delta[1]:+.3f}). Rewrite "
        f"with the smallest semantic change that moves both views toward the "
        f"target, preserve the exact keyword, and do not reuse this question "
        f"frame: {candidate.question}"
    )


def _audit_diversity(args) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite diversity audit directory: {output}")
    question_paths = [Path(path).resolve() for path in args.questions]
    rows = [row for path in question_paths for row in read_jsonl(path)]
    diagnostics = audit_question_diversity(
        rows,
        minimum_delexicalized_unique_fraction=(
            args.minimum_delexicalized_unique_fraction
        ),
        maximum_template_fraction=args.maximum_template_fraction,
        minimum_median_keyword_unique_fraction=(
            args.minimum_median_keyword_unique_fraction
        ),
        minimum_keyword_unique_fraction=args.minimum_keyword_unique_fraction,
        maximum_opening_frame_fraction=args.maximum_opening_frame_fraction,
        opening_frame_tokens=args.opening_frame_tokens,
    )
    output.mkdir(parents=True)
    atomic_json(output / "question_diversity_audit.json", diagnostics)
    atomic_text(
        output / "question_diversity_report.md",
        _diversity_report(diagnostics),
    )
    atomic_json(
        output / "run_manifest.json",
        {
            "format_version": READINESS_PROMPT_POPULATION_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "question_files": [_file_identity(path) for path in question_paths],
            "row_count": diagnostics["row_count"],
            "keyword_count": diagnostics["keyword_count"],
            "all_checks_passed": diagnostics["all_checks_passed"],
            "scientific_guard": diagnostics["scientific_guard"],
        },
    )
    print(
        f"rows={diagnostics['row_count']} keywords={diagnostics['keyword_count']} "
        f"delexicalized_unique_fraction="
        f"{diagnostics['delexicalized_unique_fraction']:.4f}"
    )
    print(f"diversity_gate_passed={diagnostics['all_checks_passed']}")
    print(f"output={output}")
    return 0 if diagnostics["all_checks_passed"] else 2


def _diversity_report(diagnostics: dict[str, object]) -> str:
    checks = diagnostics["checks"]
    lines = [
        "# Readiness-question diversity audit",
        "",
        f"Gate passed: **{str(diagnostics['all_checks_passed']).upper()}**",
        "",
        f"- Questions: {diagnostics['row_count']}",
        f"- Keywords: {diagnostics['keyword_count']}",
        "- Exact-question unique fraction: "
        f"{diagnostics['exact_question_unique_fraction']:.4f}",
        "- Delexicalized-template unique fraction: "
        f"{diagnostics['delexicalized_unique_fraction']:.4f}",
        "- Largest delexicalized-template fraction: "
        f"{diagnostics['maximum_template_fraction']:.4f}",
        "- Median within-keyword unique fraction: "
        f"{diagnostics['median_keyword_unique_fraction']:.4f}",
        "- Minimum within-keyword unique fraction: "
        f"{diagnostics['minimum_keyword_unique_fraction']:.4f}",
        "- Largest opening-frame fraction: "
        f"{diagnostics['maximum_opening_frame_fraction']:.4f}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in checks.items()
    )
    lines.extend(["", str(diagnostics["scientific_guard"]), ""])
    return "\n".join(lines)


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


def _read_plan_targets(
    plan: Path,
    keywords: Sequence[tuple[str, str]],
) -> tuple[
    tuple[ReadinessPromptTarget, ...]
    | dict[str, tuple[ReadinessPromptTarget, ...]],
    str,
]:
    manifest = read_json(plan / "plan_manifest.json")
    target_design = str(manifest.get("target_design", "rectangular-grid"))
    keyword_path = plan / "keyword_target_grid.jsonl"
    if keyword_path.is_file():
        expected_keywords = dict(keywords)
        grouped: dict[str, list[ReadinessPromptTarget]] = {
            keyword_id: [] for keyword_id in expected_keywords
        }
        for row in read_jsonl(keyword_path):
            keyword_id = str(row["keyword_id"])
            if keyword_id not in grouped:
                continue
            if row["keyword"] != expected_keywords[keyword_id]:
                raise ValueError("keyword target plan does not match candidate keywords")
            grouped[keyword_id].append(ReadinessPromptTarget(**row["target"]))
        resolved = {
            keyword_id: tuple(keyword_targets)
            for keyword_id, keyword_targets in grouped.items()
        }
        expected_count = int(manifest["target_count_per_keyword"])
        if any(len(values) != expected_count for values in resolved.values()):
            raise ValueError("keyword target plan has an inconsistent target count")
        return resolved, target_design
    targets = tuple(
        ReadinessPromptTarget(**row) for row in read_jsonl(plan / "target_grid.jsonl")
    )
    return targets, target_design


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


def _projection_index(path: str | Path) -> dict[str, dict[str, object]]:
    indexed = {}
    for row in read_jsonl(path):
        candidate_id = str(row["candidate_id"])
        if candidate_id in indexed:
            raise ValueError(f"duplicate projection candidate: {candidate_id}")
        projection = dict(row["projection"])
        if projection.get("item_id") != candidate_id:
            raise ValueError(f"nested projection identity differs: {candidate_id}")
        indexed[candidate_id] = projection
    return indexed


def _vector_agreement(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    from interpretability.pipeline.readiness_embedding_map import _ranks

    if left.shape != right.shape or left.ndim != 1 or len(left) < 3:
        raise ValueError("projection agreement needs at least three aligned values")
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        raise ValueError("projection agreement values must vary")
    return {
        "item_count": len(left),
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(np.corrcoef(_ranks(left), _ranks(right))[0, 1]),
        "mean_absolute_difference": float(np.mean(np.abs(left - right))),
    }


def _projection_comparison_report(summary: dict[str, object]) -> str:
    axis_1 = summary["axis_1"]
    axis_2 = summary["axis_2"]
    scalar = summary["scalar_readiness"]
    return f"""# Cross-embedding generated-question projections

- Questions: {summary['candidate_count']}
- Aligned axis 1 Spearman: {axis_1['spearman']:.4f}
- Aligned axis 2 Spearman: {axis_2['spearman']:.4f}
- Scalar-readiness Spearman: {scalar['spearman']:.4f}
- Axis 1 mean absolute z-coordinate difference: {axis_1['mean_absolute_difference']:.4f}
- Axis 2 mean absolute z-coordinate difference: {axis_2['mean_absolute_difference']:.4f}

{summary['interpretation_guard']}
"""


def _spatial_report(diagnostics: dict[str, object]) -> str:
    lines = [
        "# Cross-embedding spatial prompt coverage",
        "",
        f"- Candidates: {diagnostics['candidate_count']}",
        f"- Jointly accepted by validators: {diagnostics['jointly_accepted_candidate_count']}",
        f"- Selected: {diagnostics['selected_count']}",
        f"- Mean target distance: {_metric_text(diagnostics['mean_target_distance'])}",
        f"- Maximum target distance: {_metric_text(diagnostics['maximum_target_distance'])}",
        f"- Strict dual-view tolerance required: {diagnostics['require_both_views_within_tolerance']}",
        f"- Jointly coordinate-verified: {diagnostics['verified_selected_count']}/{diagnostics['selected_count']}",
        f"- Mean Qwen-view target distance: {_metric_text(diagnostics['mean_reference_target_distance'])}",
        f"- Maximum Qwen-view target distance: {_metric_text(diagnostics['maximum_reference_target_distance'])}",
        f"- Mean aligned Mistral-view target distance: {_metric_text(diagnostics['mean_candidate_aligned_target_distance'])}",
        f"- Maximum aligned Mistral-view target distance: {_metric_text(diagnostics['maximum_candidate_aligned_target_distance'])}",
        f"- Exact delexicalized-template uniqueness required: {diagnostics['require_delexicalized_template_uniqueness']}",
        f"- Template-duplicate assignments rejected: {diagnostics['template_duplicate_rejection_count']}",
        f"- Selected delexicalized templates unique: {diagnostics['selected_delexicalized_templates_are_unique']}",
        f"- Mean cross-embedding disagreement: {_metric_text(diagnostics['mean_cross_embedding_disagreement'])}",
        f"- Next-round cells: {diagnostics['next_round_task_count']}",
        f"- Target design: {diagnostics.get('target_design', 'rectangular-grid')}",
        f"- Keyword spacing-gate pass fraction: {diagnostics['keyword_spacing_gate_pass_fraction']:.2%}",
        f"- Overall spacing gate: {'PASS' if diagnostics['overall_spacing_gate_passed'] else 'REFINE'}",
        "",
    ]
    pooled = diagnostics["pooled_support_coverage"]
    lines.extend(
        [
            "## Pooled support coverage",
            "",
            f"- Targets/selected: {pooled['target_count']}/{pooled['selected_count']}",
            f"- Within tolerance: {pooled['within_distance_tolerance_fraction']:.2%}",
            f"- Target axis spans: ({pooled['target_axis_1_span']:.4f}, {pooled['target_axis_2_span']:.4f})",
            f"- Observed axis spans: ({pooled['observed_axis_1_span']:.4f}, {pooled['observed_axis_2_span']:.4f})",
            f"- Occupied target bins: {pooled['observed_target_grid_bin_count']}/{pooled['target_occupied_grid_bin_count']}",
            f"- Target/observed histogram total variation: {pooled['histogram_total_variation']:.4f}",
            "",
        ]
    )
    lines.extend(
        f"- [{'x' if passed else ' '}] {name}"
        for name, passed in pooled["gate_checks"].items()
    )
    lines.append("")
    for keyword_id, item in diagnostics["keywords"].items():
        lines.extend(
            [
                f"## {keyword_id}",
                "",
                f"- Selected: {item['selected_count']}",
                f"- Within tolerance: {item['within_distance_tolerance_fraction']:.2%}",
                f"- Axis 1 span: {item['axis_1_span']:.4f}",
                f"- Axis 2 span: {item['axis_2_span']:.4f}",
                f"- Median nearest-neighbor distance: {item['median_nearest_neighbor_distance']:.4f}",
                f"- Occupied grid bins: {item['occupied_grid_bin_count']}/{item['target_count']}",
                f"- Target axis spans: ({item['target_axis_1_span']:.4f}, {item['target_axis_2_span']:.4f})",
                f"- Target median nearest-neighbor distance: {item['target_median_nearest_neighbor_distance']:.4f}",
                f"- Target occupied diagnostic bins: {item['target_occupied_grid_bin_count']}",
                "",
            ]
        )
        lines.extend(
            f"- [{'x' if passed else ' '}] {name}"
            for name, passed in item["gate_checks"].items()
        )
        lines.append("")
    lines.extend(
        [
            "The target design is a diagnostic over prompt embeddings. It does not define B.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_coordinate(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("coordinate bounds must have positive width")
    return float((float(value) - low) / (high - low))


def _metric_text(value) -> str:
    return "not available" if value is None else f"{float(value):.4f}"


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
