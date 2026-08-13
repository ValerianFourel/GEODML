#!/usr/bin/env python3
"""Run staged A1-only generation, calibration, dual embedding, and selection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.a1_prompt_manifold import (  # noqa: E402
    A1Calibration,
    A1Candidate,
    A1ComparisonRequest,
    A1Embedding,
    A1Judgment,
    A1_MANIFOLD_VERSION,
    FakeA1CandidateGenerator,
    FakeA1Embedder,
    FakeA1PairwiseJudge,
    LocalLLMA1CandidateGenerator,
    LocalLLMA1PairwiseJudge,
    build_a1_comparison_requests,
    calibrate_a1_candidates,
    embed_a1_candidates,
    generate_a1_candidate_bank,
    judge_a1_comparisons,
    select_a1_manifold,
    stratified_random_a1_grid,
)
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    LLM2VecGenPromptEmbedder,
    LLM2VecPromptEmbedder,
)


def _grid(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must contain numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("grid must not be empty")
    return result


def _integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("style seeds must be integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("style seeds must not be empty")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    generate = stages.add_parser("generate")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--search-term", required=True)
    generate.add_argument("--generator-model", required=True)
    generate.add_argument("--precision", choices=("full", "4bit"), default="full")
    grid = generate.add_mutually_exclusive_group()
    grid.add_argument(
        "--a1-grid",
        type=_grid,
        help="Explicit comma-separated A1 grid (default: seven equally spaced levels).",
    )
    grid.add_argument(
        "--randomized-a1-levels",
        type=int,
        help=(
            "Build this many reproducible stratified-random A1 levels, including "
            "fixed endpoints 0 and 1; uses --master-seed."
        ),
    )
    generate.add_argument("--style-seeds", type=_integers, default=(0, 1, 2, 3))
    generate.add_argument("--number-candidates", type=int, default=12)
    generate.add_argument("--master-seed", type=int, default=20260817)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--max-new-tokens", type=int, default=500)
    generate.add_argument("--maximum-attempts", type=int, default=8)
    generate.add_argument(
        "--resume",
        action="store_true",
        help="reuse a validated partial generation cache in an existing output directory",
    )

    judge = stages.add_parser("judge")
    judge.add_argument("--output-dir", required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--judge-id", required=True)
    judge.add_argument("--precision", choices=("full", "4bit"), default="full")
    judge.add_argument("--max-new-tokens", type=int, default=80)
    judge.add_argument("--start-index", type=int, default=0)
    judge.add_argument("--limit", type=int)

    input_embed = stages.add_parser("embed-input")
    input_embed.add_argument("--output-dir", required=True)
    input_embed.add_argument("--embedding-model", required=True)
    input_embed.add_argument("--mntp-model")
    input_embed.add_argument("--peft-model")
    input_embed.add_argument("--encode-batch-size", type=int, default=1)
    input_embed.add_argument("--encode-max-length", type=int, default=512)

    response_embed = stages.add_parser("embed-response")
    response_embed.add_argument("--output-dir", required=True)
    response_embed.add_argument("--embedding-model", required=True)
    response_embed.add_argument("--encode-batch-size", type=int, default=1)
    response_embed.add_argument("--encode-max-length", type=int, default=512)

    select = stages.add_parser("select")
    select.add_argument("--output-dir", required=True)
    select.add_argument("--minimum-judges-per-comparison", type=int, default=2)
    select.add_argument("--minimum-distinct-judge-models", type=int, default=2)
    select.add_argument("--calibration-weight", type=float, default=4.0)
    select.add_argument("--smoothness-weight", type=float, default=1.0)
    select.add_argument("--curvature-weight", type=float, default=0.5)
    select.add_argument("--diversity-weight", type=float, default=0.25)
    select.add_argument("--minimum-realized-step", type=float, default=0.0)

    smoke = stages.add_parser("fake-smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--search-term", default="abandoned cart recovery")
    return parser


def _paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "manifest": root / "run_manifest.json",
        "candidates": root / "a1_candidates.jsonl",
        "comparisons": root / "a1_pairwise_comparison_requests.jsonl",
        "judgments": root / "a1_pairwise_judgments.jsonl",
        "calibrations": root / "a1_candidate_calibrations.jsonl",
        "input": root / "a1_llm2vec_input_embeddings.jsonl",
        "response": root / "a1_llm2vec_gen_response_embeddings.jsonl",
        "selected": root / "selected_a1_prompt_manifold.jsonl",
        "diagnostics": root / "a1_manifold_diagnostics.json",
        "report": root / "a1_manifold_report.md",
    }


def _prepare_generation_root(paths: dict[str, Path], *, resume: bool) -> int:
    root = paths["root"]
    if root.resolve() == Path.cwd().resolve():
        raise ValueError("refusing to use the current working directory as output-dir")
    if not root.exists():
        root.mkdir(parents=True)
        return 0
    if not resume:
        raise ValueError(f"output directory already exists: {root}")
    if not root.is_dir():
        raise ValueError(f"output path is not a directory: {root}")
    if paths["manifest"].exists():
        manifest = _read_json(paths["manifest"])
        if manifest.get("status") == "generated-unjudged":
            raise ValueError("generation is already complete; continue with the judge stage")
        raise ValueError("existing run manifest makes generation resume unsafe")
    later_stage_paths = (
        paths["judgments"],
        paths["calibrations"],
        paths["input"],
        paths["response"],
        paths["selected"],
        paths["diagnostics"],
        paths["report"],
    )
    if any(path.exists() for path in later_stage_paths):
        raise ValueError("later-stage artifacts make generation resume unsafe")
    allowed_names = {"cache", "logs", paths["candidates"].name, paths["comparisons"].name}
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in allowed_names)
    if unexpected:
        raise ValueError(
            "unexpected files make generation resume unsafe: " + ", ".join(unexpected)
        )
    cache_directory = root / "cache" / "generation"
    if cache_directory.exists() and not cache_directory.is_dir():
        raise ValueError(f"generation cache path is not a directory: {cache_directory}")
    return sum(
        1
        for path in cache_directory.glob("*.json")
        if not path.name.endswith(".failed.json")
    ) if cache_directory.exists() else 0


def _generate(args, paths) -> None:
    reused_cache_entries = _prepare_generation_root(paths, resume=args.resume)
    if args.randomized_a1_levels is not None:
        a1_grid = stratified_random_a1_grid(
            args.randomized_a1_levels,
            master_seed=args.master_seed,
        )
        a1_grid_design = "stratified-random-fixed-endpoints"
    else:
        a1_grid = args.a1_grid or tuple(step / 6 for step in range(7))
        a1_grid_design = "explicit" if args.a1_grid is not None else "default-seven-level"
    generator = LocalLLMA1CandidateGenerator.from_model(
        args.generator_model,
        precision=args.precision,
        cache_directory=paths["root"] / "cache" / "generation",
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        maximum_attempts=args.maximum_attempts,
    )
    candidates = generate_a1_candidate_bank(
        search_term=args.search_term,
        a1_grid=a1_grid,
        style_seeds=args.style_seeds,
        number_candidates=args.number_candidates,
        master_seed=args.master_seed,
        generator=generator,
    )
    comparisons = build_a1_comparison_requests(candidates)
    _jsonl(paths["candidates"], map(asdict, candidates))
    _jsonl(paths["comparisons"], map(asdict, comparisons))
    _json(
        paths["manifest"],
        {
            "artifact_version": A1_MANIFOLD_VERSION,
            "scientific_result": False,
            "status": "generated-unjudged",
            "git_commit_sha": _git_sha(),
            "generated_at": _now(),
            "search_term": args.search_term,
            "a1_grid": a1_grid,
            "a1_grid_design": a1_grid_design,
            "a1_level_count": len(a1_grid),
            "style_seeds": args.style_seeds,
            "number_candidates_per_level": args.number_candidates,
            "master_seed": args.master_seed,
            "generator_model": args.generator_model,
            "generator_precision": args.precision,
            "generation_temperature": args.temperature,
            "generation_max_new_tokens": args.max_new_tokens,
            "generation_maximum_attempts": args.maximum_attempts,
            "resumed_existing_output": bool(args.resume and reused_cache_entries),
            "preexisting_success_cache_file_count": reused_cache_entries,
            "environment": _environment(),
            "judge_runs": [],
        },
    )
    print(f"generated {len(candidates)} A1 candidates and {len(comparisons)} comparisons")


def _judge(args, paths) -> None:
    candidates = tuple(A1Candidate(**item) for item in _read_jsonl(paths["candidates"]))
    comparisons = tuple(A1ComparisonRequest(**item) for item in _read_jsonl(paths["comparisons"]))
    manifest = _read_json(paths["manifest"])
    if any(run["judge_id"] == args.judge_id for run in manifest["judge_runs"]):
        raise ValueError(f"judge_id already exists: {args.judge_id}")
    subset = comparisons[args.start_index : None if args.limit is None else args.start_index + args.limit]
    judge = LocalLLMA1PairwiseJudge.from_model(
        args.judge_model,
        judge_id=args.judge_id,
        precision=args.precision,
        cache_directory=paths["root"] / "cache" / "judgments" / args.judge_id,
        max_new_tokens=args.max_new_tokens,
    )
    rows = judge_a1_comparisons(subset, candidates, (judge,))
    existing = _read_jsonl(paths["judgments"]) if paths["judgments"].exists() else []
    _jsonl(paths["judgments"], (*existing, *(asdict(item) for item in rows)))
    manifest["judge_runs"].append(
        {"judge_id": args.judge_id, "model": args.judge_model, "comparison_count": len(rows), "start_index": args.start_index}
    )
    manifest["status"] = "partially-judged"
    _json(paths["manifest"], manifest)
    print(f"appended {len(rows)} judgments from {args.judge_id}")


def _embed_input(args, paths) -> None:
    candidates = tuple(A1Candidate(**item) for item in _read_jsonl(paths["candidates"]))
    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    rows = embed_a1_candidates(candidates, embedder=embedder, representation="input")
    _jsonl(paths["input"], map(asdict, rows))
    manifest = _read_json(paths["manifest"])
    manifest.update(
        {"input_embedding_model": args.embedding_model, "input_mntp_model": args.mntp_model, "input_peft_model": args.peft_model, "input_embedding_git_commit_sha": _git_sha()}
    )
    _json(paths["manifest"], manifest)
    print(f"wrote {len(rows)} input-text embeddings")


def _embed_response(args, paths) -> None:
    candidates = tuple(A1Candidate(**item) for item in _read_jsonl(paths["candidates"]))
    embedder = LLM2VecGenPromptEmbedder(args.embedding_model, batch_size=args.encode_batch_size, max_length=args.encode_max_length)
    rows = embed_a1_candidates(candidates, embedder=embedder, representation="anticipated-response")
    _jsonl(paths["response"], map(asdict, rows))
    manifest = _read_json(paths["manifest"])
    manifest.update({"response_embedding_model": args.embedding_model, "response_embedding_git_commit_sha": _git_sha()})
    _json(paths["manifest"], manifest)
    print(f"wrote {len(rows)} anticipated-response embeddings")


def _select(args, paths) -> None:
    candidates = tuple(A1Candidate(**item) for item in _read_jsonl(paths["candidates"]))
    comparisons = tuple(A1ComparisonRequest(**item) for item in _read_jsonl(paths["comparisons"]))
    judgments = tuple(A1Judgment(**item) for item in _read_jsonl(paths["judgments"]))
    input_rows = tuple(A1Embedding(**item) for item in _read_jsonl(paths["input"]))
    response_rows = tuple(A1Embedding(**item) for item in _read_jsonl(paths["response"]))
    observations: dict[str, set[str]] = {}
    for item in judgments:
        observations.setdefault(item.comparison_id, set()).add(item.judge_id)
    incomplete = [item.comparison_id for item in comparisons if len(observations.get(item.comparison_id, ())) < args.minimum_judges_per_comparison]
    if incomplete:
        raise ValueError(f"{len(incomplete)} comparisons lack the minimum number of judges")
    manifest = _read_json(paths["manifest"])
    if len({run["model"] for run in manifest["judge_runs"]}) < args.minimum_distinct_judge_models:
        raise ValueError("too few distinct judge models")
    calibrations = calibrate_a1_candidates(candidates, comparisons, judgments)
    selected, diagnostics = select_a1_manifold(
        candidates,
        calibrations,
        input_rows,
        response_rows,
        calibration_weight=args.calibration_weight,
        smoothness_weight=args.smoothness_weight,
        curvature_weight=args.curvature_weight,
        diversity_weight=args.diversity_weight,
        minimum_realized_step=args.minimum_realized_step,
    )
    _jsonl(paths["calibrations"], map(asdict, calibrations))
    _jsonl(paths["selected"], map(asdict, selected))
    _json(paths["diagnostics"], asdict(diagnostics))
    manifest.update(
        {
            "status": "complete-unreviewed",
            "selection_git_commit_sha": _git_sha(),
            "selection_weights": {
                "calibration": args.calibration_weight,
                "smoothness": args.smoothness_weight,
                "curvature": args.curvature_weight,
                "diversity": args.diversity_weight,
                "minimum_realized_step": args.minimum_realized_step,
            },
            "completed_at": _now(),
        }
    )
    _json(paths["manifest"], manifest)
    _write_report(paths, manifest, diagnostics)
    print(f"selected {len(selected)} prompts on the A1 manifold")


def _fake_smoke(args, paths) -> None:
    if paths["root"].exists():
        raise ValueError(f"output directory already exists: {paths['root']}")
    paths["root"].mkdir(parents=True)
    candidates = generate_a1_candidate_bank(search_term=args.search_term, style_seeds=(0, 1), number_candidates=4, generator=FakeA1CandidateGenerator())
    comparisons = build_a1_comparison_requests(candidates)
    judgments = judge_a1_comparisons(comparisons, candidates, (FakeA1PairwiseJudge("one"), FakeA1PairwiseJudge("two")))
    calibrations = calibrate_a1_candidates(candidates, comparisons, judgments)
    input_rows = embed_a1_candidates(candidates, embedder=FakeA1Embedder("fake-input"), representation="input")
    response_rows = embed_a1_candidates(candidates, embedder=FakeA1Embedder("fake-response", response=True), representation="anticipated-response")
    selected, diagnostics = select_a1_manifold(candidates, calibrations, input_rows, response_rows)
    _jsonl(paths["candidates"], map(asdict, candidates))
    _jsonl(paths["comparisons"], map(asdict, comparisons))
    _jsonl(paths["judgments"], map(asdict, judgments))
    _jsonl(paths["calibrations"], map(asdict, calibrations))
    _jsonl(paths["input"], map(asdict, input_rows))
    _jsonl(paths["response"], map(asdict, response_rows))
    _jsonl(paths["selected"], map(asdict, selected))
    _json(paths["diagnostics"], asdict(diagnostics))
    _json(paths["manifest"], {"artifact_version": A1_MANIFOLD_VERSION, "scientific_result": False, "status": "complete-unreviewed", "search_term": args.search_term})
    print(f"fake smoke selected {len(selected)} prompts; no scientific claim")


def _write_report(paths, manifest, diagnostics) -> None:
    _atomic_text(
        paths["report"],
        f"""# A1 decision-readiness prompt manifold pilot

- Search term inserted structurally: `{manifest['search_term']}`
- Selected prompts: `{diagnostics.selected_count}`
- Strictly monotone style rate: `{diagnostics.fully_strict_monotone_style_rate}`
- Adjacent reversal rate: `{diagnostics.adjacent_reversal_rate}`
- Mean A1 calibration error: `{diagnostics.mean_realized_a1_absolute_error}`
- Mean within-style Spearman: `{diagnostics.mean_style_spearman}`
- LLM2Vec input tortuosity: `{diagnostics.input_mean_tortuosity}`
- LLM2Vec-Gen response tortuosity: `{diagnostics.response_mean_tortuosity}`
- Mean lexical similarity: `{diagnostics.mean_pairwise_lexical_similarity}`

Assigned A1 remains the treatment. Pairwise scores and both embedding spaces are
pre-outcome manipulation checks and selection diagnostics. No latent state is decoded.
""",
    )


def _json(path, payload) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _jsonl(path, rows) -> None:
    _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))


def _atomic_text(path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment() -> dict:
    return {key.lower(): os.getenv(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES")}


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    paths = _paths(Path(args.output_dir))
    try:
        if args.stage == "generate":
            _generate(args, paths)
        elif args.stage == "judge":
            _judge(args, paths)
        elif args.stage == "embed-input":
            _embed_input(args, paths)
        elif args.stage == "embed-response":
            _embed_response(args, paths)
        elif args.stage == "select":
            _select(args, paths)
        else:
            _fake_smoke(args, paths)
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
