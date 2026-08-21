#!/usr/bin/env python3
"""Generate readiness-targeted questions with LLM2Vec-Gen latent feedback.

This is a proposal stage.  It calibrates a bridge on frozen development rows,
steers LLM2Vec-Gen reconstruction states, decodes questions, independently
validates them, and re-embeds every final text with the frozen readiness map.
Decoder-state coordinates are never accepted as readiness coordinates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_hf_dataset import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    read_json,
    read_jsonl,
    sha256_file,
)
from interpretability.pipeline.readiness_latent_feedback import (  # noqa: E402
    LATENT_FEEDBACK_VERSION,
    fit_latent_coordinate_bridge,
    run_latent_feedback,
)
from interpretability.pipeline.readiness_prompt_population import (  # noqa: E402
    LocalSearchQuestionValidator,
    ReadinessGenerationTask,
    ReadinessPromptTarget,
    ReadinessQuestionCandidate,
    ReadinessSubspaceBounds,
    load_readiness_embedding_map,
    project_text_embeddings,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecPromptEmbedder,
)
from scripts.validate_llm2vec_gen_axis import LLM2VecGenBackend  # noqa: E402


class FrozenReadinessScorer:
    """Re-embed final text and project it through the frozen readiness map."""

    def __init__(self, fitted, bounds, embedder) -> None:
        self._fitted = fitted
        self._bounds = bounds
        self._embedder = embedder
        self.model_name = embedder.model_name

    def score(self, texts: Sequence[str]) -> np.ndarray:
        values = tuple(str(text) for text in texts)
        embeddings = self._embedder.embed(values)
        identities = [
            hashlib.sha256(f"{index}\0{text}".encode()).hexdigest()
            for index, text in enumerate(values)
        ]
        projected = project_text_embeddings(
            self._fitted,
            self._bounds,
            item_ids=[f"latent-feedback-score:{value}" for value in identities],
            text_sha256s=[hashlib.sha256(text.encode()).hexdigest() for text in values],
            embeddings=embeddings,
        )
        return np.asarray(
            [
                (row.normalized_axis_1, row.normalized_axis_2)
                for row in projected
            ],
            dtype=np.float64,
        )


class IndependentValidatorAdapter:
    """Adapt the existing independent LLM rubric to the feedback protocol."""

    def __init__(self, validator: LocalSearchQuestionValidator) -> None:
        self._validator = validator
        self.model_name = validator.model_name

    def review(self, question: str, keyword: str) -> tuple[bool, str]:
        digest = hashlib.sha256(f"{keyword}\0{question}".encode()).hexdigest()
        candidate = ReadinessQuestionCandidate(
            candidate_id=f"latent-feedback-review:{digest[:24]}",
            task_id=f"latent-feedback-task:{digest[:24]}",
            keyword_id=f"keyword:{hashlib.sha256(keyword.encode()).hexdigest()[:24]}",
            keyword=keyword,
            target_id="latent-feedback-review-target",
            target_index=0,
            target_normalized_axis_1=0.0,
            target_normalized_axis_2=0.0,
            target_raw_axis_1=0.0,
            target_raw_axis_2=0.0,
            round_index=0,
            generator_id="llm2vec-gen-latent-feedback",
            generator_model="llm2vec-gen-latent-feedback",
            candidate_slot=0,
            generation_seed=int(digest[:16], 16),
            question=question,
            question_sha256=hashlib.sha256(question.encode()).hexdigest(),
            proposal_kind="llm2vec-gen-latent-feedback",
        )
        review = self._validator.review(candidate)
        return review.accepted, review.concise_reason


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--initial-candidates", nargs="+", required=True)
    parser.add_argument(
        "--calibration",
        help=(
            "development-only JSONL with item_id, split, text, "
            "normalized_axis_1, and normalized_axis_2"
        ),
    )
    parser.add_argument(
        "--calibration-corpus",
        help="semantic-readiness corpus JSONL containing item_id, split, and text",
    )
    parser.add_argument(
        "--calibration-coordinates",
        help="frozen readiness_supervised_subspace_coordinates.jsonl",
    )
    parser.add_argument("--map", required=True)
    parser.add_argument("--bounds", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--mntp-model")
    parser.add_argument("--peft-model")
    parser.add_argument("--llm2vec-gen-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-backend", choices=("local", "api", "openai"), default="api"
    )
    parser.add_argument("--judge-precision", choices=("full", "4bit"), default="full")
    parser.add_argument("--judge-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-rounds", type=int, default=3)
    parser.add_argument("--step-scales", default="0.5,1.0,1.5")
    parser.add_argument("--coordinate-step-limit", type=float, default=0.35)
    parser.add_argument("--distance-tolerance", type=float, default=0.12)
    parser.add_argument("--bridge-ridge-penalty", type=float, default=1e-3)
    parser.add_argument("--minimum-calibration-items", type=int, default=10)
    parser.add_argument("--maximum-calibration-items", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--reconstruction-batch-size", type=int, default=8)
    parser.add_argument("--reconstruction-max-length", type=int, default=512)
    parser.add_argument("--decode-max-new-tokens", type=int, default=96)
    parser.add_argument("--judge-maximum-attempts", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--emit-best-effort",
        action="store_true",
        help="emit semantically valid closest proposals even outside tolerance",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    _validate_control_args(args)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("invalid candidate slice")
    step_scales = _positive_csv(args.step_scales)

    tasks = {row.task_id: row for row in _read_tasks(args.tasks)}
    initial = tuple(
        ReadinessQuestionCandidate(**row)
        for path in args.initial_candidates
        for row in read_jsonl(path)
    )
    if not initial or len({row.candidate_id for row in initial}) != len(initial):
        raise ValueError("initial candidates must be nonempty and uniquely identified")
    for candidate in initial:
        task = tasks.get(candidate.task_id)
        if task is None or candidate.keyword != task.keyword:
            raise ValueError(f"initial candidate does not match tasks: {candidate.candidate_id}")
    initial = initial[
        args.start_index : (
            None if args.limit is None else args.start_index + args.limit
        )
    ]
    if not initial:
        raise ValueError("candidate slice is empty")

    generator_models = {row.generator_model for row in initial}
    if args.judge_model == args.llm2vec_gen_model or args.judge_model in generator_models:
        raise ValueError(
            "judge model must be independent of LLM2Vec-Gen and initial generators"
        )

    fitted = load_readiness_embedding_map(args.map)
    _validate_embedding_model_revision(fitted, args.embedding_model)
    bounds = ReadinessSubspaceBounds(**read_json(args.bounds))
    prepared_calibration = bool(args.calibration)
    joined_calibration = bool(
        args.calibration_corpus and args.calibration_coordinates
    )
    incomplete_join = bool(args.calibration_corpus) != bool(
        args.calibration_coordinates
    )
    if incomplete_join or prepared_calibration == joined_calibration:
        raise ValueError(
            "provide exactly one calibration source: --calibration, or both "
            "--calibration-corpus and --calibration-coordinates"
        )
    calibration_rows = (
        _read_calibration(
            args.calibration,
            minimum_items=args.minimum_calibration_items,
            maximum_items=args.maximum_calibration_items,
        )
        if prepared_calibration
        else _join_calibration(
            args.calibration_corpus,
            args.calibration_coordinates,
            bounds,
            minimum_items=args.minimum_calibration_items,
            maximum_items=args.maximum_calibration_items,
        )
    )

    reconstruction_backend = LLM2VecGenBackend(args.llm2vec_gen_model)
    calibration_texts = [
        _reconstruction_request(str(row["text"])) for row in calibration_rows
    ]
    _, calibration_states = reconstruction_backend.encode(
        calibration_texts,
        batch_size=args.reconstruction_batch_size,
        max_length=args.reconstruction_max_length,
    )
    calibration_coordinates = np.asarray(
        [
            (row["normalized_axis_1"], row["normalized_axis_2"])
            for row in calibration_rows
        ],
        dtype=np.float64,
    )
    bridge = fit_latent_coordinate_bridge(
        calibration_states,
        calibration_coordinates,
        ridge_penalty=args.bridge_ridge_penalty,
        minimum_items=args.minimum_calibration_items,
    )

    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    )
    scorer = FrozenReadinessScorer(fitted, bounds, embedder)
    validator = IndependentValidatorAdapter(
        LocalSearchQuestionValidator.from_model(
            args.judge_model,
            judge_id="independent-latent-feedback-validator",
            cache_directory=args.judge_cache_dir,
            backend=args.judge_backend,
            precision=args.judge_precision,
            maximum_attempts=args.judge_maximum_attempts,
        )
    )

    result_rows = []
    trace_rows = []
    proposal_rows = []
    for index, candidate in enumerate(initial, start=1):
        result = run_latent_feedback(
            initial_question=candidate.question,
            keyword=candidate.keyword,
            target_coordinates=(
                candidate.target_normalized_axis_1,
                candidate.target_normalized_axis_2,
            ),
            bridge=bridge,
            reconstruction_backend=reconstruction_backend,
            scorer=scorer,
            validator=validator,
            maximum_rounds=args.maximum_rounds,
            step_scales=step_scales,
            distance_tolerance=args.distance_tolerance,
            coordinate_step_limit=args.coordinate_step_limit,
            encode_batch_size=args.reconstruction_batch_size,
            encode_max_length=args.reconstruction_max_length,
            decode_max_new_tokens=args.decode_max_new_tokens,
        )
        result_rows.append(
            {
                "source_candidate_id": candidate.candidate_id,
                "task_id": candidate.task_id,
                "keyword_id": candidate.keyword_id,
                "keyword": candidate.keyword,
                "target_id": candidate.target_id,
                "candidate_slot": candidate.candidate_slot,
                **{key: value for key, value in asdict(result).items() if key != "attempts"},
            }
        )
        for attempt in result.attempts:
            trace_rows.append(
                {
                    "source_candidate_id": candidate.candidate_id,
                    "task_id": candidate.task_id,
                    "target_id": candidate.target_id,
                    **asdict(attempt),
                }
            )
        should_emit = result.best_question is not None and (
            result.accepted_within_tolerance or args.emit_best_effort
        )
        if should_emit:
            proposal_rows.append(
                {
                    "task_id": candidate.task_id,
                    "question": result.best_question,
                    "candidate_slot": candidate.candidate_slot,
                    "source_candidate_id": candidate.candidate_id,
                    "observed_normalized_axis_1": result.best_normalized_axis_1,
                    "observed_normalized_axis_2": result.best_normalized_axis_2,
                    "target_distance": result.best_target_distance,
                    "within_tolerance": result.accepted_within_tolerance,
                }
            )
        print(
            f"feedback={index}/{len(initial)} task={candidate.task_id} "
            f"distance={result.best_target_distance} stop={result.stop_reason}"
        )

    output.mkdir(parents=True)
    atomic_jsonl(output / "feedback_proposals.jsonl", proposal_rows)
    atomic_jsonl(output / "feedback_results.jsonl", result_rows)
    atomic_jsonl(output / "feedback_trace.jsonl", trace_rows)
    atomic_json(
        output / "bridge_diagnostics.json",
        {
            key: value
            for key, value in asdict(bridge).items()
            if key not in {"coordinate_mean", "state_mean", "directions"}
        }
        | {"coordinate_mean": bridge.coordinate_mean.tolist()},
    )
    atomic_npz(
        output / "bridge_state.restricted-local.npz",
        coordinate_mean=np.asarray(bridge.coordinate_mean, dtype=np.float32),
        state_mean=np.asarray(bridge.state_mean, dtype=np.float32),
        directions=np.asarray(bridge.directions, dtype=np.float32),
    )
    atomic_json(
        output / "run_manifest.json",
        {
            "format_version": LATENT_FEEDBACK_VERSION,
            "created_at": _now(),
            "git_commit_sha": _git_sha(),
            "slurm": _slurm_environment(),
            "inputs": {
                "tasks": _file_identity(args.tasks),
                "initial_candidates": [
                    _file_identity(path) for path in args.initial_candidates
                ],
                "calibration": (
                    {"prepared": _file_identity(args.calibration)}
                    if prepared_calibration
                    else {
                        "corpus": _file_identity(args.calibration_corpus),
                        "coordinates": _file_identity(args.calibration_coordinates),
                    }
                ),
                "map": _file_identity(args.map),
                "bounds": _file_identity(args.bounds),
            },
            "models": {
                "llm2vec_gen": args.llm2vec_gen_model,
                "frozen_readiness_embedding": fitted.embedding_model,
                "judge": args.judge_model,
                "judge_backend": args.judge_backend,
            },
            "bridge": {
                "bridge_hash": bridge.bridge_hash,
                "calibration_item_count": bridge.calibration_item_count,
                "ridge_penalty": bridge.ridge_penalty,
                "coordinate_condition_number": bridge.coordinate_condition_number,
                "state_reconstruction_rmse": bridge.state_reconstruction_rmse,
            },
            "control": {
                "maximum_rounds": args.maximum_rounds,
                "step_scales": list(step_scales),
                "coordinate_step_limit": args.coordinate_step_limit,
                "distance_tolerance": args.distance_tolerance,
                "emit_best_effort": args.emit_best_effort,
            },
            "counts": {
                "initial_candidates": len(initial),
                "decoded_attempts": len(trace_rows),
                "proposals": len(proposal_rows),
                "within_tolerance": sum(
                    bool(row["accepted_within_tolerance"]) for row in result_rows
                ),
            },
            "scientific_guard": (
                "The bridge is calibrated only on frozen development rows. Decoder "
                "states propose text; only independent validation and frozen "
                "re-embedding determine acceptance. Coordinates describe question "
                "semantics and do not define or replace policy variable B."
            ),
        },
    )
    print(f"proposals={len(proposal_rows)} output={output}")
    return 0


def _reconstruction_request(text: str) -> str:
    from interpretability.pipeline.llm2vec_gen_axis import (
        build_realization_reconstruction_text,
    )

    return build_realization_reconstruction_text(text)


def _read_tasks(path: str | Path) -> tuple[ReadinessGenerationTask, ...]:
    rows = []
    for row in read_jsonl(path):
        payload = dict(row)
        payload["target"] = ReadinessPromptTarget(**payload["target"])
        rows.append(ReadinessGenerationTask(**payload))
    if not rows or len({row.task_id for row in rows}) != len(rows):
        raise ValueError("tasks must be nonempty and uniquely identified")
    return tuple(rows)


def _read_calibration(
    path: str | Path, *, minimum_items: int, maximum_items: int
) -> tuple[dict, ...]:
    rows = read_jsonl(path)
    required = {
        "item_id",
        "split",
        "text",
        "normalized_axis_1",
        "normalized_axis_2",
    }
    if maximum_items < minimum_items:
        raise ValueError("maximum calibration items must cover the minimum")
    if len({str(row.get("item_id")) for row in rows}) != len(rows):
        raise ValueError("calibration item ids must be unique")
    normalized = []
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"calibration row {index} lacks required fields")
        if row["split"] != "development":
            raise ValueError("latent bridge calibration must be development-only")
        text = " ".join(str(row["text"]).split())
        coordinate = np.asarray(
            [row["normalized_axis_1"], row["normalized_axis_2"]],
            dtype=np.float64,
        )
        if not text or coordinate.shape != (2,) or not np.isfinite(coordinate).all():
            raise ValueError(f"calibration row {index} is invalid")
        if np.any((coordinate < 0.0) | (coordinate > 1.0)):
            raise ValueError("calibration coordinates must lie in [0, 1]")
        normalized.append(
            {
                **row,
                "text": text,
                "normalized_axis_1": float(coordinate[0]),
                "normalized_axis_2": float(coordinate[1]),
            }
        )
    return _stable_calibration_subset(
        normalized, minimum_items=minimum_items, maximum_items=maximum_items
    )


def _join_calibration(
    corpus_path: str | Path,
    coordinate_path: str | Path,
    bounds: ReadinessSubspaceBounds,
    *,
    minimum_items: int,
    maximum_items: int,
) -> tuple[dict, ...]:
    """Join existing frozen-map artifacts into bridge calibration rows."""

    corpus = {str(row["item_id"]): row for row in read_jsonl(corpus_path)}
    if not corpus:
        raise ValueError("calibration corpus is empty")
    rows = []
    for coordinate in read_jsonl(coordinate_path):
        if coordinate.get("split") != "development" or not bool(
            coordinate.get("usable_for_axis", False)
        ):
            continue
        item_id = str(coordinate.get("item_id", ""))
        item = corpus.get(item_id)
        if item is None or item.get("split") != "development":
            continue
        text = " ".join(str(item.get("text", "")).split())
        raw_axis_1 = float(coordinate["axis_1"])
        raw_axis_2 = float(coordinate["axis_2"])
        normalized_axis_1 = _normalize(
            raw_axis_1, bounds.axis_1_low, bounds.axis_1_high
        )
        normalized_axis_2 = _normalize(
            raw_axis_2, bounds.axis_2_low, bounds.axis_2_high
        )
        if not text or not (
            0.0 <= normalized_axis_1 <= 1.0
            and 0.0 <= normalized_axis_2 <= 1.0
        ):
            continue
        rows.append(
            {
                "item_id": item_id,
                "split": "development",
                "text": text,
                "normalized_axis_1": normalized_axis_1,
                "normalized_axis_2": normalized_axis_2,
            }
        )
    return _stable_calibration_subset(
        rows, minimum_items=minimum_items, maximum_items=maximum_items
    )


def _stable_calibration_subset(
    rows: Sequence[dict], *, minimum_items: int, maximum_items: int
) -> tuple[dict, ...]:
    if maximum_items < minimum_items:
        raise ValueError("maximum calibration items must cover the minimum")
    if len(rows) < minimum_items:
        raise ValueError(f"calibration requires at least {minimum_items} usable rows")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["item_id"]).encode()).hexdigest(),
    )
    return tuple(ordered[:maximum_items])


def _normalize(value: float, low: float, high: float) -> float:
    if high - low <= 1e-12:
        raise ValueError("readiness bounds must span each axis")
    return float((value - low) / (high - low))


def _positive_csv(value: str) -> tuple[float, ...]:
    try:
        rows = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step scales must be numeric") from exc
    if not rows or any(not np.isfinite(item) or item <= 0 for item in rows):
        raise ValueError("step scales must be positive and finite")
    return rows


def _validate_control_args(args) -> None:
    if args.maximum_rounds < 0:
        raise ValueError("maximum rounds must be nonnegative")
    if args.minimum_calibration_items < 3:
        raise ValueError("minimum calibration items must be at least three")
    if args.maximum_calibration_items < args.minimum_calibration_items:
        raise ValueError("maximum calibration items must cover the minimum")
    positive = (
        "coordinate_step_limit",
        "bridge_ridge_penalty",
        "embedding_batch_size",
        "embedding_max_length",
        "reconstruction_batch_size",
        "reconstruction_max_length",
        "decode_max_new_tokens",
        "judge_maximum_attempts",
    )
    if any(float(getattr(args, name)) <= 0 for name in positive):
        raise ValueError("batch sizes, lengths, attempts, and control scales must be positive")
    if args.distance_tolerance < 0:
        raise ValueError("distance tolerance must be nonnegative")


def _validate_embedding_model_revision(fitted, model_path: str | Path) -> None:
    reference = fitted.embedding_model.rsplit("@", 1)
    if len(reference) != 2:
        raise ValueError("frozen map lacks an @revision embedding identity")
    path = Path(model_path)
    if path.exists() and path.resolve().name != reference[1]:
        raise ValueError(
            "embedding base-model revision does not match frozen map: "
            f"expected {reference[1]}, got {path.resolve().name}"
        )


def _file_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ANALYSIS_ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slurm_environment() -> dict[str, str | None]:
    return {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "job_name": os.getenv("SLURM_JOB_NAME"),
        "partition": os.getenv("SLURM_JOB_PARTITION"),
        "account": os.getenv("SLURM_JOB_ACCOUNT"),
        "node_list": os.getenv("SLURM_JOB_NODELIST"),
        "allocated_cpus": os.getenv("SLURM_CPUS_ON_NODE"),
        "allocated_gpus": os.getenv("SLURM_GPUS"),
        "allocated_memory_per_node": os.getenv("SLURM_MEM_PER_NODE"),
        "slurm_time_limit": os.getenv("SLURM_TIMELIMIT"),
        "approved_walltime": os.getenv("GEODML_APPROVED_WALLTIME"),
        "runtime_estimate": os.getenv("GEODML_RUNTIME_ESTIMATE"),
        "runtime_estimate_basis": os.getenv("GEODML_RUNTIME_ESTIMATE_BASIS"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
