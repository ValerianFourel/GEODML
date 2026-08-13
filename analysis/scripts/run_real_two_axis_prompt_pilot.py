#!/usr/bin/env python3
"""Run staged real-LLM generation, judging, and LLM2Vec prompt diagnostics."""

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

from interpretability.pipeline.prompt_continuum import StylePlan  # noqa: E402
from interpretability.pipeline.two_axis_prompt_population import (  # noqa: E402
    FakePairwiseJudge,
    FakeTwoAxisCandidateGenerator,
    FakeTwoAxisPromptEmbedder,
    LLM2VecGenPromptEmbedder,
    LLM2VecPromptEmbedder,
    LocalLLMPairwiseJudge,
    LocalLLMTwoAxisCandidateGenerator,
    PairwiseComparisonRequest,
    PairwiseJudgment,
    SEMANTIC_CONTRACT_VERSION,
    TwoAxisCandidate,
    build_pairwise_comparison_requests,
    calibrate_candidates,
    diagnose_pairwise_judgments,
    generate_candidate_bank,
    judge_comparison_requests,
    measure_selected_latent_population,
    select_prompt_population,
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
        raise argparse.ArgumentTypeError("value must contain integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("value must not be empty")
    return result


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _jsonl(path: Path, rows) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _candidate(payload: dict[str, object]) -> TwoAxisCandidate:
    value = dict(payload)
    value["style_plan"] = StylePlan(**value["style_plan"])
    value["contract_failures"] = tuple(value["contract_failures"])
    return TwoAxisCandidate(**value)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _environment_manifest() -> dict[str, object]:
    return {
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME"),
        "slurm_job_partition": os.getenv("SLURM_JOB_PARTITION"),
        "slurm_nnodes": os.getenv("SLURM_NNODES"),
        "slurm_ntasks": os.getenv("SLURM_NTASKS"),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "python_version": sys.version.split()[0],
        "hostname": os.uname().nodename,
    }


def _paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "run_manifest.json",
        "candidates": root / "two_axis_candidates.jsonl",
        "comparisons": root / "pairwise_comparison_requests.jsonl",
        "judgments": root / "pairwise_judgments.jsonl",
        "judgment_diagnostics": root / "pairwise_judgment_diagnostics.json",
        "judgment_report": root / "pairwise_judgment_report.md",
        "calibrations": root / "candidate_calibrations.jsonl",
        "selected": root / "selected_prompt_population.jsonl",
        "selection": root / "selection_diagnostics.json",
        "latent": root / "llm2vec_latent_diagnostics.json",
        "report": root / "real_two_axis_prompt_report.md",
    }


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--search-term", required=True)
    generate.add_argument("--generator-model", required=True)
    generate.add_argument("--precision", choices=("full", "4bit"), default="full")
    generate.add_argument("--a1-grid", type=_grid, default=(0.0, 0.5, 1.0))
    generate.add_argument("--a2-grid", type=_grid, default=(0.0, 0.5, 1.0))
    generate.add_argument("--style-seeds", type=_integers, default=(0,))
    generate.add_argument("--number-candidates", type=int, default=4)
    generate.add_argument("--master-seed", type=int, default=20260813)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--max-new-tokens", type=int, default=1200)
    generate.add_argument("--maximum-attempts", type=int, default=5)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--output-dir", required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--judge-id", required=True)
    judge.add_argument("--precision", choices=("full", "4bit"), default="full")
    judge.add_argument("--max-new-tokens", type=int, default=80)
    judge.add_argument("--start-index", type=int, default=0)
    judge.add_argument("--limit", type=int)

    embed = subparsers.add_parser("embed-select")
    embed.add_argument("--output-dir", required=True)
    embed.add_argument("--embedding-model", required=True)
    embed.add_argument("--mntp-model")
    embed.add_argument("--peft-model")
    embed.add_argument("--encode-batch-size", type=int, default=1)
    embed.add_argument("--encode-max-length", type=int, default=512)
    embed.add_argument("--monotonic-tolerance", type=float, default=0.02)
    embed.add_argument("--maximum-neighbor-distance", type=float)
    embed.add_argument("--minimum-judges-per-comparison", type=int, default=2)
    embed.add_argument("--minimum-distinct-judge-models", type=int, default=2)

    diagnose = subparsers.add_parser("diagnose-judgments")
    diagnose.add_argument("--output-dir", required=True)

    response = subparsers.add_parser("response-diagnostics")
    response.add_argument("--output-dir", required=True)
    response.add_argument("--embedding-model", required=True)
    response.add_argument("--encode-batch-size", type=int, default=1)
    response.add_argument("--encode-max-length", type=int, default=512)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", required=True)

    smoke = subparsers.add_parser("fake-smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--search-term", default="abandoned cart recovery")
    return parser


def _generate(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    existing = [path for key, path in paths.items() if key != "judgments" and path.exists()]
    if existing:
        raise ValueError("generation output already exists: " + ", ".join(map(str, existing)))
    generator = LocalLLMTwoAxisCandidateGenerator.from_model(
        args.generator_model,
        cache_directory=Path(args.output_dir) / "cache" / "generation",
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        maximum_attempts=args.maximum_attempts,
    )
    candidates = generate_candidate_bank(
        search_term=args.search_term,
        a1_grid=args.a1_grid,
        a2_grid=args.a2_grid,
        style_seeds=args.style_seeds,
        number_candidates=args.number_candidates,
        master_seed=args.master_seed,
        generator=generator,
    )
    comparisons = build_pairwise_comparison_requests(candidates)
    valid = sum(item.structural_valid for item in candidates)
    if valid != len(candidates):
        failures: dict[str, int] = {}
        for item in candidates:
            for failure in item.contract_failures:
                failures[failure] = failures.get(failure, 0) + 1
        raise ValueError(f"{len(candidates) - valid} generated candidates failed hard invariants: {failures}")
    manifest = {
        "artifact_version": "real-two-axis-prompt-pilot-v1",
        "scientific_result": False,
        "status": "generated-unjudged",
        "git_commit_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "search_term": " ".join(args.search_term.split()),
        "a1_grid": list(args.a1_grid),
        "a2_grid": list(args.a2_grid),
        "style_seeds": list(args.style_seeds),
        "number_candidates_per_cell": args.number_candidates,
        "master_seed": args.master_seed,
        "generator_model": args.generator_model,
        "generator_precision": args.precision,
        "generator_temperature": args.temperature,
        "generator_max_new_tokens": args.max_new_tokens,
        "generator_maximum_attempts": args.maximum_attempts,
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "judge_runs": [],
        "environment": _environment_manifest(),
    }
    _json(paths["manifest"], manifest)
    _jsonl(paths["candidates"], (asdict(item) for item in candidates))
    _jsonl(paths["comparisons"], (asdict(item) for item in comparisons))
    print(f"generated {len(candidates)} real LLM candidates and {len(comparisons)} blind comparisons")


def _judge(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    candidates = tuple(_candidate(item) for item in _read_jsonl(paths["candidates"]))
    comparisons = tuple(PairwiseComparisonRequest(**item) for item in _read_jsonl(paths["comparisons"]))
    existing = [PairwiseJudgment(**item) for item in _read_jsonl(paths["judgments"])] if paths["judgments"].exists() else []
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("judge start-index must be nonnegative and limit positive")
    selected_comparisons = comparisons[
        args.start_index : None if args.limit is None else args.start_index + args.limit
    ]
    if not selected_comparisons:
        raise ValueError("selected judge comparison slice is empty")
    existing_keys = {(item.comparison_id, item.judge_id) for item in existing}
    overlap = [
        item.comparison_id
        for item in selected_comparisons
        if (item.comparison_id, args.judge_id) in existing_keys
    ]
    if overlap:
        raise ValueError(
            f"judge slice overlaps {len(overlap)} existing judgments for {args.judge_id}"
        )
    judge = LocalLLMPairwiseJudge.from_model(
        args.judge_model,
        judge_id=args.judge_id,
        cache_directory=Path(args.output_dir) / "cache" / "judging" / args.judge_id,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
    )
    new = judge_comparison_requests(selected_comparisons, candidates, (judge,))
    _jsonl(paths["judgments"], (asdict(item) for item in (*existing, *new)))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["status"] = "judged-unembedded"
    manifest["judge_runs"].append(
        {
            "judge_id": args.judge_id,
            "model": args.judge_model,
            "precision": args.precision,
            "start_index": args.start_index,
            "comparison_count": len(selected_comparisons),
        }
    )
    _json(paths["manifest"], manifest)
    print(f"appended {len(new)} judgments from {args.judge_id}")


def _embed_select(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    candidates = tuple(_candidate(item) for item in _read_jsonl(paths["candidates"]))
    comparisons = tuple(PairwiseComparisonRequest(**item) for item in _read_jsonl(paths["comparisons"]))
    judgments = tuple(PairwiseJudgment(**item) for item in _read_jsonl(paths["judgments"]))
    if not judgments:
        raise ValueError("no pairwise judgments are available")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if args.minimum_judges_per_comparison <= 0 or args.minimum_distinct_judge_models <= 0:
        raise ValueError("minimum judge requirements must be positive")
    judgment_counts: dict[str, set[str]] = {}
    for item in judgments:
        judgment_counts.setdefault(item.comparison_id, set()).add(item.judge_id)
    missing = [
        item.comparison_id
        for item in comparisons
        if len(judgment_counts.get(item.comparison_id, set()))
        < args.minimum_judges_per_comparison
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} comparisons have fewer than "
            f"{args.minimum_judges_per_comparison} distinct judge observations"
        )
    judge_models = {item["model"] for item in manifest.get("judge_runs", ())}
    if len(judge_models) < args.minimum_distinct_judge_models:
        raise ValueError(
            f"found {len(judge_models)} distinct judge models; require "
            f"{args.minimum_distinct_judge_models}"
        )
    calibrations = calibrate_candidates(candidates, comparisons, judgments)
    embedder = LLM2VecPromptEmbedder(
        args.embedding_model,
        mntp_model_name_or_path=args.mntp_model,
        peft_model_name_or_path=args.peft_model,
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    selected, selection = select_prompt_population(
        candidates,
        calibrations,
        embedder=embedder,
        monotonic_tolerance=args.monotonic_tolerance,
        maximum_neighbor_embedding_distance=args.maximum_neighbor_distance,
    )
    latent = measure_selected_latent_population(selected)
    _jsonl(paths["calibrations"], (asdict(item) for item in calibrations))
    _jsonl(paths["selected"], (asdict(item) for item in selected))
    _json(paths["selection"], asdict(selection))
    _json(paths["latent"], asdict(latent))
    manifest.update(
        {
            "status": "input-embedded-selected",
            "input_embedding_model": args.embedding_model,
            "input_mntp_model": args.mntp_model,
            "input_peft_model": args.peft_model,
            "embedding_git_commit_sha": _git_sha(),
            "encode_batch_size": args.encode_batch_size,
            "encode_max_length": args.encode_max_length,
            "minimum_judges_per_comparison": args.minimum_judges_per_comparison,
            "minimum_distinct_judge_models": args.minimum_distinct_judge_models,
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    _json(paths["manifest"], manifest)
    _write_report(paths, manifest, candidates, judgments, selection, latent)
    print(f"embedded and selected {len(selected)} prompts")


def _diagnose_judgments(paths: dict[str, Path]) -> None:
    candidates = tuple(_candidate(item) for item in _read_jsonl(paths["candidates"]))
    comparisons = tuple(
        PairwiseComparisonRequest(**item) for item in _read_jsonl(paths["comparisons"])
    )
    judgments = tuple(PairwiseJudgment(**item) for item in _read_jsonl(paths["judgments"]))
    diagnostics = diagnose_pairwise_judgments(candidates, comparisons, judgments)
    _json(paths["judgment_diagnostics"], diagnostics)
    failing = diagnostics["failing_endpoint_slices"]
    lines = [
        "# Pairwise semantic-judgment diagnostics",
        "",
        "This is a pre-embedding manipulation-check diagnostic. It does not alter",
        "assigned A1/A2, flip judge labels, or relax endpoint calibration.",
        "",
        f"- Candidates: `{diagnostics['candidate_count']}`",
        f"- Comparisons: `{diagnostics['comparison_count']}`",
        f"- Judgments: `{diagnostics['judgment_count']}`",
        f"- Judges: `{diagnostics['judge_ids']}`",
        f"- Cross-judge agreement: `{diagnostics['cross_judge_agreement_rate']}`",
        f"- All endpoint slices ordered: `{diagnostics['all_endpoint_slices_ordered']}`",
        f"- Failing endpoint slices: `{len(failing)}`",
        "",
        "## Endpoint slices",
        "",
        "| Axis | Style | Fixed | Pooled gap | Pooled direct score | Order consistency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in diagnostics["slices"]:
        pooled = item["pooled_endpoint_fit"]
        direct = item["pooled_direct_endpoint_evidence"]
        lines.append(
            f"| {item['axis']} | {item['style_seed']} | {item['fixed_coordinate']:.6g} | "
            f"{pooled['upper_minus_lower']:.6g} | "
            f"{direct['expected_direction_half_tie_score']} | "
            f"{direct['presentation_order_consistency_rate']} |"
        )
        if not pooled["ordered_expected_direction"]:
            lines.extend(["", "### Failing slice endpoint prompts", ""])
            for candidate in item["endpoint_candidates"]:
                lines.extend(
                    [
                        f"- `{candidate['candidate_id']}` "
                        f"(A1={candidate['assigned_a1']}, A2={candidate['assigned_a2']}, "
                        f"candidate={candidate['candidate_index']})",
                        f"  - Objective: {candidate['search_objective_clause']}",
                        f"  - Source preference: {candidate['source_preference_clause']}",
                    ]
                )
            lines.extend(["", "Per-judge details are preserved in the JSON artifact.", ""])
    _atomic_text(paths["judgment_report"], "\n".join(lines) + "\n")
    print(f"wrote {paths['judgment_diagnostics']}")
    print(f"wrote {paths['judgment_report']}")
    if failing:
        print("endpoint calibration failures:")
        for item in failing:
            print(
                f"  {item['axis']} style_seed={item['style_seed']} "
                f"fixed_coordinate={item['fixed_coordinate']:.12g} "
                f"upper_minus_lower={item['upper_minus_lower']:.6g}"
            )
    else:
        print("all endpoint slices order in the expected direction")


def _response_diagnostics(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    selected_payloads = _read_jsonl(paths["selected"])
    selected = []
    from interpretability.pipeline.two_axis_prompt_population import SelectedTwoAxisPrompt

    for payload in selected_payloads:
        payload = dict(payload)
        payload["prompt_embedding"] = tuple(payload["prompt_embedding"])
        selected.append(SelectedTwoAxisPrompt(**payload))
    embedder = LLM2VecGenPromptEmbedder(
        args.embedding_model,
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_length,
    )
    query_bound = [
        item.prompt_template.replace("{QUERY}", item.search_term) for item in selected
    ]
    embeddings = embedder.embed(query_bound)
    response_selected = [
        item.__class__(
            **{
                **asdict(item),
                "embedding_model": embedder.model_name,
                "prompt_embedding": tuple(float(value) for value in embedding),
                "embedding_hash": __import__("hashlib").sha256(
                    embedding.astype("<f8").tobytes()
                ).hexdigest(),
            }
        )
        for item, embedding in zip(selected, embeddings)
    ]
    diagnostics = measure_selected_latent_population(response_selected)
    _json(Path(args.output_dir) / "llm2vec_gen_response_diagnostics.json", asdict(diagnostics))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete-unreviewed",
            "response_embedding_model": args.embedding_model,
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    _json(paths["manifest"], manifest)
    print(f"wrote LLM2Vec-Gen response diagnostics for {len(response_selected)} prompts")


def _write_report(paths, manifest, candidates, judgments, selection, latent) -> None:
    report = f"""# Real two-axis prompt semantic pilot

This run used a real local LLM candidate generator, blind pairwise judge runs,
and primary LLM2Vec input-text representations. It remains an unreviewed pilot,
not a downstream reranking or causal result. A separate stage may add
LLM2Vec-Gen anticipated-response diagnostics.

- Search term inserted structurally: `{manifest['search_term']}`
- Generator model: `{manifest['generator_model']}`
- Judge runs: `{manifest['judge_runs']}`
- Primary LLM2Vec model: `{manifest['input_embedding_model']}`
- Primary LLM2Vec MNTP model: `{manifest.get('input_mntp_model')}`
- Primary LLM2Vec PEFT model: `{manifest['input_peft_model']}`
- Candidate prompts: `{len(candidates)}`
- Pairwise judgments: `{len(judgments)}`
- Selected prompts: `{selection.selected_count}`
- Exact-query structural retention: `{latent.exact_query_structural_retention_rate}`
- A1 judge-coordinate adjacent reversal rate: `{selection.a1_adjacent_reversal_rate}`
- A2 judge-coordinate adjacent reversal rate: `{selection.a2_adjacent_reversal_rate}`
- A1 input-space LLM2Vec slice Spearman mean: `{latent.a1_slice_spearman_mean}`
- A2 input-space LLM2Vec slice Spearman mean: `{latent.a2_slice_spearman_mean}`
- Input-space LLM2Vec direction cosine: `{latent.a1_a2_direction_cosine}`
- A1 cross-axis slope ratio: `{latent.a1_cross_axis_slope_ratio}`
- A2 cross-axis slope ratio: `{latent.a2_cross_axis_slope_ratio}`
- Adjacent/distant embedding-distance ratio: `{latent.adjacent_over_distant_distance_ratio}`

Assigned A1/A2 remain treatment coordinates. Judge-derived realized coordinates
and LLM2Vec representations are manipulation checks. No latent state is decoded.
"""
    _atomic_text(paths["report"], report)


def _validate(paths: dict[str, Path]) -> None:
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    candidates = _read_jsonl(paths["candidates"])
    selected = _read_jsonl(paths["selected"])
    latent = json.loads(paths["latent"].read_text(encoding="utf-8"))
    response_path = Path(paths["manifest"].parent) / "llm2vec_gen_response_diagnostics.json"
    checks = {
        "complete_status": manifest.get("status") == "complete-unreviewed",
        "real_generator": manifest.get("generator_model") not in (None, "fake-two-axis-generator-v1"),
        "judge_runs_present": bool(manifest.get("judge_runs")),
        "all_candidates_structurally_valid": all(item["structural_valid"] for item in candidates),
        "exact_query_not_generated": all(
            manifest["search_term"].casefold() not in item["prompt_template"].casefold()
            for item in candidates
        ),
        "query_placeholder_once": all(item["prompt_template"].count("{QUERY}") == 1 for item in selected),
        "unique_selected_hashes": len({item["candidate_hash"] for item in selected}) == len(selected),
        "query_retention_is_one": latent["exact_query_structural_retention_rate"] == 1.0,
        "input_axis_directions_nonzero": latent["a1_endpoint_distance"] > 0
        and latent["a2_endpoint_distance"] > 0,
        "response_diagnostics_present": response_path.exists(),
    }
    print(json.dumps(checks, indent=2))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("validation failed: " + ", ".join(failed))
    print("validation: PASS")


def _fake_smoke(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    root = Path(args.output_dir)
    if root.exists():
        raise ValueError(f"fake smoke output already exists: {root}")
    candidates = generate_candidate_bank(
        search_term=args.search_term,
        a1_grid=(0.0, 0.5, 1.0),
        a2_grid=(0.0, 0.5, 1.0),
        style_seeds=(0,),
        number_candidates=3,
        generator=FakeTwoAxisCandidateGenerator(),
    )
    comparisons = build_pairwise_comparison_requests(candidates)
    judgments = judge_comparison_requests(
        comparisons, candidates, (FakePairwiseJudge("fake-judge"),)
    )
    calibrations = calibrate_candidates(candidates, comparisons, judgments)
    selected, selection = select_prompt_population(
        candidates, calibrations, embedder=FakeTwoAxisPromptEmbedder()
    )
    latent = measure_selected_latent_population(selected)
    manifest = {
        "artifact_version": "real-two-axis-prompt-pilot-v1",
        "scientific_result": False,
        "status": "complete-unreviewed",
        "git_commit_sha": _git_sha(),
        "search_term": args.search_term,
        "generator_model": "fake-two-axis-generator-v1",
        "judge_runs": [{"judge_id": "fake-judge", "model": "fake"}],
        "embedding_model": "fake-two-axis-prompt-embedder-v1",
    }
    _json(paths["manifest"], manifest)
    _jsonl(paths["candidates"], (asdict(item) for item in candidates))
    _jsonl(paths["comparisons"], (asdict(item) for item in comparisons))
    _jsonl(paths["judgments"], (asdict(item) for item in judgments))
    _jsonl(paths["calibrations"], (asdict(item) for item in calibrations))
    _jsonl(paths["selected"], (asdict(item) for item in selected))
    _json(paths["selection"], asdict(selection))
    _json(paths["latent"], asdict(latent))
    print(f"fake smoke wrote {root}")


def main() -> int:
    parser = _base_parser()
    args = parser.parse_args()
    paths = _paths(Path(args.output_dir))
    try:
        if args.stage == "generate":
            _generate(args, paths)
        elif args.stage == "judge":
            _judge(args, paths)
        elif args.stage == "embed-select":
            _embed_select(args, paths)
        elif args.stage == "diagnose-judgments":
            _diagnose_judgments(paths)
        elif args.stage == "response-diagnostics":
            _response_diagnostics(args, paths)
        elif args.stage == "validate":
            _validate(paths)
        else:
            _fake_smoke(args, paths)
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
