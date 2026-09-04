#!/usr/bin/env python3
"""Resume the relaxed selected-prompt audit and dual-view map on four GPUs.

This command does not allocate resources.  It is intended to run inside an
already approved four-GPU Slurm allocation.  Projection work is split into
small immutable shards so completed shards survive a time limit or interruption.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from interpretability.pipeline.readiness_hf_dataset import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
)
from interpretability.pipeline.readiness_prompt_population import (  # noqa: E402
    delexicalize_question,
    search_review_passes_contract,
)


FORMAT_VERSION = "readiness-final-audit-4gpu-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _explicit_queries(*rows: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for row in rows:
        for key in ("search_query", "query", "search_queries"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values.extend(str(item) for item in value if str(item).strip())
    return tuple(values)


def _query_is_bound(keyword: object, queries: Sequence[str]) -> bool:
    if not queries:
        return True
    normalized_keyword = _normalized(keyword)
    return bool(normalized_keyword) and any(
        normalized_keyword in _normalized(query) for query in queries
    )


def audit_relaxed_selected_prompts(
    *,
    selected_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Recheck selected prompts under the metadata-bound search-trigger-v2 rule."""

    candidate_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    validation_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_groups[str(row.get("candidate_id", ""))].append(row)
    for row in validation_rows:
        validation_groups[str(row.get("candidate_id", ""))].append(row)

    failures: list[set[str]] = [set() for _ in selected_rows]
    candidate_id_groups: dict[str, list[int]] = defaultdict(list)
    template_groups: dict[str, list[int]] = defaultdict(list)

    for index, selected in enumerate(selected_rows):
        candidate_id = str(selected.get("candidate_id", "")).strip()
        candidate_id_groups[candidate_id].append(index)
        candidates = candidate_groups.get(candidate_id, [])
        validations = validation_groups.get(candidate_id, [])
        if not candidate_id:
            failures[index].add("nonempty_candidate_id")
        if len(candidates) != 1:
            failures[index].add("candidate_present_once")
        if len(validations) != 1:
            failures[index].add("validation_present_once")
        if selected.get("both_views_within_tolerance") is not True:
            failures[index].add("both_views_within_tolerance")

        candidate = candidates[0] if len(candidates) == 1 else None
        if candidate is not None:
            for field in ("keyword_id", "keyword", "question"):
                if str(selected.get(field, "")) != str(candidate.get(field, "")):
                    failures[index].add("selected_candidate_identity")
            if str(selected.get("question_sha256", "")) and str(
                selected.get("question_sha256")
            ) != str(candidate.get("question_sha256", "")):
                failures[index].add("selected_candidate_identity")
            if not _query_is_bound(
                candidate.get("keyword"), _explicit_queries(selected, candidate)
            ):
                failures[index].add("search_query_keyword_binding")

        if len(validations) == 1 and not search_review_passes_contract(
            validations[0], contract="search-trigger-v2"
        ):
            failures[index].add("validation_search_trigger_v2")

        keyword = str(selected.get("keyword", "")).strip()
        question = str(selected.get("question", "")).strip()
        if not keyword or not question:
            failures[index].add("nonempty_keyword_and_question")
        else:
            template = delexicalize_question(
                question, keyword, require_keyword=False
            )
            template_groups[template].append(index)

    for indices in candidate_id_groups.values():
        if len(indices) > 1:
            for index in indices:
                failures[index].add("unique_selected_candidate_id")
    for indices in template_groups.values():
        if len(indices) > 1:
            for index in indices:
                failures[index].add("unique_delexicalized_template")

    passed = tuple(
        dict(candidate_groups[str(selected["candidate_id"])][0])
        for index, selected in enumerate(selected_rows)
        if not failures[index]
    )
    gate_failures = Counter(
        failure for row_failures in failures for failure in row_failures
    )
    summary = {
        "format_version": FORMAT_VERSION,
        "acceptance_contract": "search-trigger-v2",
        "selected_prompt_count": len(selected_rows),
        "relaxed_contract_pass_count": len(passed),
        "failed_prompt_count": len(selected_rows) - len(passed),
        "gate_failures": dict(sorted(gate_failures.items())),
        "literal_keyword_in_prompt_required": False,
        "question_mark_required": False,
        "keyword_metadata_binding_required": True,
        "explicit_search_query_keyword_binding_required": True,
        "search_intent_required": True,
        "web_answerable_required": True,
        "delexicalized_template_unique": True,
    }
    return passed, summary


def partition_candidate_rows(
    rows: Sequence[Mapping[str, object]], *, shard_count: int
) -> tuple[tuple[dict[str, object], ...], ...]:
    """Split unique candidates into deterministic, balanced round-robin shards."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["candidate_id"]))
    candidate_ids = [str(row.get("candidate_id", "")) for row in ordered]
    if not candidate_ids or any(not value for value in candidate_ids):
        raise ValueError("candidate rows must be nonempty and identified")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    shards: list[list[dict[str, object]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(ordered):
        shards[index % shard_count].append(row)
    return tuple(tuple(shard) for shard in shards)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _candidate_identity(row: Mapping[str, object]) -> tuple[str, str]:
    return (
        str(row.get("candidate_id", "")),
        str(row.get("question_sha256", "")),
    )


def projection_shard_state(input_path: str | Path, output_root: str | Path) -> str:
    """Return complete only for a manifest plus the exact projected row count."""

    input_path = Path(input_path)
    output_root = Path(output_root)
    if not output_root.exists():
        return "not_started"
    projection_path = output_root / "question_projections.jsonl"
    manifest_path = output_root / "projection_manifest.json"
    if not projection_path.is_file() or not manifest_path.is_file():
        return "incomplete"
    expected = _line_count(input_path)
    observed = _line_count(projection_path)
    if expected == 0 or observed != expected:
        return "incomplete"
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return "incomplete"
    manifest_count = manifest.get("candidate_count")
    if manifest_count is not None and int(manifest_count) != expected:
        return "incomplete"
    try:
        input_identities = [
            _candidate_identity(row) for row in read_jsonl(input_path)
        ]
        output_identities = [
            _candidate_identity(row) for row in read_jsonl(projection_path)
        ]
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return "incomplete"
    if (
        input_identities != output_identities
        or any(not candidate_id for candidate_id, _ in input_identities)
        or len(set(input_identities)) != len(input_identities)
    ):
        return "incomplete"
    return "complete"


def _write_or_verify_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    expected_identities = [_candidate_identity(row) for row in rows]
    if path.exists():
        observed_identities = [
            _candidate_identity(row) for row in read_jsonl(path)
        ]
        if observed_identities != expected_identities:
            raise ValueError(f"existing shard belongs to different candidates: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(path, rows)


def _quarantine(path: Path, quarantine_root: Path) -> None:
    if not path.exists():
        return
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / f"{path.parent.name}-{path.name}-{_now().replace(':', '')}"
    path.replace(target)
    print(f"quarantined_incomplete={path} -> {target}", flush=True)


def _gpu_tokens() -> tuple[str, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        tokens = tuple(value.strip() for value in visible.split(",") if value.strip())
    else:
        descriptor = os.environ.get("READINESS_ALLOCATED_GPU_COUNT") or os.environ.get(
            "SLURM_GPUS_ON_NODE", ""
        )
        counts = re.findall(r"[0-9]+", descriptor)
        count = int(counts[-1]) if counts else 0
        tokens = tuple(str(index) for index in range(count))
    if len(tokens) != 4:
        raise ValueError(
            "exactly four visible GPUs are required; set CUDA_VISIBLE_DEVICES or "
            "READINESS_ALLOCATED_GPU_COUNT=4"
        )
    return tokens


def _projection_command(
    *,
    python: str,
    repository: Path,
    candidate_path: Path,
    output_root: Path,
    map_root: Path,
    embedding_model: Path,
    mntp_model: Path,
    peft_model: Path,
    batch_size: int,
) -> list[str]:
    return [
        python,
        str(repository / "analysis/scripts/build_readiness_prompt_population.py"),
        "project-candidates",
        "--candidates",
        str(candidate_path),
        "--map",
        str(map_root / "readiness_embedding_map.json"),
        "--reference-coordinates",
        str(map_root / "readiness_supervised_subspace_coordinates.jsonl"),
        "--embedding-model",
        str(embedding_model),
        "--mntp-model",
        str(mntp_model),
        "--peft-model",
        str(peft_model),
        "--embedding-batch-size",
        str(batch_size),
        "--output-dir",
        str(output_root),
    ]


def _run_view(
    *,
    view: str,
    shard_paths: Sequence[Path],
    output_root: Path,
    repository: Path,
    python: str,
    map_root: Path,
    embedding_model: Path,
    mntp_model: Path,
    peft_model: Path,
    batch_size: int,
    gpu_tokens: Sequence[str],
) -> tuple[Path, ...]:
    projection_root = output_root / "projections" / view
    log_root = output_root / "logs"
    quarantine_root = output_root / "quarantine"
    log_root.mkdir(parents=True, exist_ok=True)
    outputs = tuple(projection_root / f"shard-{index:03d}" for index in range(len(shard_paths)))

    pending: list[int] = []
    for index, (candidate_path, shard_output) in enumerate(zip(shard_paths, outputs)):
        state = projection_shard_state(candidate_path, shard_output)
        print(f"view={view} shard={index:03d} state={state}", flush=True)
        if state == "complete":
            continue
        if state == "incomplete":
            _quarantine(shard_output, quarantine_root)
        pending.append(index)

    assignments = [pending[index::4] for index in range(4)]

    def worker(gpu_index: int, indices: Sequence[int]) -> None:
        for index in indices:
            candidate_path = shard_paths[index]
            shard_output = outputs[index]
            command = _projection_command(
                python=python,
                repository=repository,
                candidate_path=candidate_path,
                output_root=shard_output,
                map_root=map_root,
                embedding_model=embedding_model,
                mntp_model=mntp_model,
                peft_model=peft_model,
                batch_size=batch_size,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_tokens[gpu_index]
            log_path = log_root / f"{view}-shard-{index:03d}.log"
            print(
                f"START view={view} shard={index:03d} gpu={gpu_tokens[gpu_index]} "
                f"log={log_path}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=repository,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{view} shard {index:03d} failed with exit {result.returncode}; "
                    f"inspect {log_path}"
                )
            if projection_shard_state(candidate_path, shard_output) != "complete":
                raise RuntimeError(f"{view} shard {index:03d} produced incomplete artifacts")
            print(f"DONE view={view} shard={index:03d}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(worker, gpu_index, indices)
            for gpu_index, indices in enumerate(assignments)
            if indices
        ]
        for future in futures:
            future.result()
    return outputs


def _merge_view(
    *,
    view: str,
    candidate_rows: Sequence[Mapping[str, object]],
    shard_paths: Sequence[Path],
    shard_outputs: Sequence[Path],
    output_root: Path,
) -> Path:
    merged_root = output_root / "merged" / view
    expected_ids = [str(row["candidate_id"]) for row in candidate_rows]
    if projection_shard_state(output_root / "compliant-candidates.jsonl", merged_root) == "complete":
        observed = [str(row["candidate_id"]) for row in read_jsonl(merged_root / "question_projections.jsonl")]
        if set(observed) == set(expected_ids):
            return merged_root
    if merged_root.exists():
        _quarantine(merged_root, output_root / "quarantine")

    by_id: dict[str, dict[str, object]] = {}
    manifests = []
    stable_map_id = None
    for shard_path, shard_output in zip(shard_paths, shard_outputs):
        if projection_shard_state(shard_path, shard_output) != "complete":
            raise RuntimeError(f"cannot merge incomplete shard: {shard_output}")
        manifest = read_json(shard_output / "projection_manifest.json")
        map_id = str(manifest["map_id"])
        if stable_map_id is None:
            stable_map_id = map_id
        elif map_id != stable_map_id:
            raise ValueError(f"{view} shards use different map ids")
        manifests.append(_identity(shard_output / "projection_manifest.json"))
        for row in read_jsonl(shard_output / "question_projections.jsonl"):
            candidate_id = str(row["candidate_id"])
            if candidate_id in by_id:
                raise ValueError(f"duplicate projected candidate: {candidate_id}")
            by_id[candidate_id] = row
    if set(by_id) != set(expected_ids):
        raise ValueError(f"{view} projection ids do not equal audited candidate ids")

    temporary = merged_root.parent / f".{view}-merge-{os.getpid()}"
    if temporary.exists():
        _quarantine(temporary, output_root / "quarantine")
    temporary.mkdir(parents=True)
    atomic_jsonl(temporary / "question_projections.jsonl", (by_id[value] for value in expected_ids))
    first_manifest = read_json(shard_outputs[0] / "projection_manifest.json")
    atomic_json(
        temporary / "projection_manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "created_at": _now(),
            "git_commit_sha": first_manifest.get("git_commit_sha", "unavailable"),
            "map_id": stable_map_id,
            "map": first_manifest.get("map"),
            "reference_coordinates": first_manifest.get("reference_coordinates"),
            "candidate_files": [_identity(output_root / "compliant-candidates.jsonl")],
            "candidate_count": len(expected_ids),
            "embedding": first_manifest.get("embedding"),
            "embedding_arrays_included": False,
            "source_projection_manifests": manifests,
            "merge_contract": "exact-audited-id-union-v1",
        },
    )
    temporary.replace(merged_root)
    return merged_root


def _git_sha(repository_or_output: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_or_output if (repository_or_output / ".git").exists() else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty axis")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _finalize_axis(output_root: Path, comparison_root: Path) -> dict[str, object]:
    aligned = read_jsonl(comparison_root / "aligned_question_projections.jsonl")
    mapped = []
    for row in aligned:
        consensus = (
            float(row["reference_axis_1_z"])
            + float(row["candidate_aligned_axis_1_z"])
        ) / 2.0
        mapped.append({**row, "consensus_axis_1_z": consensus})
    mapped.sort(key=lambda row: (float(row["consensus_axis_1_z"]), str(row["candidate_id"])))
    denominator = max(1, len(mapped) - 1)
    for rank, row in enumerate(mapped):
        row["axis_1_rank"] = rank
        row["axis_1_percentile_0_1"] = rank / denominator
    atomic_jsonl(output_root / "final-axis-map.jsonl", mapped)
    values = [float(row["consensus_axis_1_z"]) for row in mapped]
    return {
        "count": len(values),
        "minimum": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "maximum": max(values),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository", default=str(Path.cwd()))
    parser.add_argument("--qwen-map-root", required=True)
    parser.add_argument("--qwen-embedding-model", required=True)
    parser.add_argument("--qwen-mntp-model", required=True)
    parser.add_argument("--qwen-peft-model", required=True)
    parser.add_argument("--mistral-map-root", required=True)
    parser.add_argument("--mistral-embedding-model", required=True)
    parser.add_argument("--mistral-mntp-model", required=True)
    parser.add_argument("--mistral-peft-model", required=True)
    parser.add_argument("--robustness-battery", required=True)
    parser.add_argument("--qwen-python", default=sys.executable)
    parser.add_argument("--mistral-python", default=sys.executable)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    checkpoint = Path(args.checkpoint_root).resolve()
    output = Path(args.output_root).resolve()
    repository = Path(args.repository).resolve()
    output.mkdir(parents=True, exist_ok=True)

    selected_path = checkpoint / "strict-selection/spatially_selected_questions.jsonl"
    candidates_path = checkpoint / "merged/candidates.jsonl"
    validation_path = checkpoint / "merged/validation.jsonl"
    selected = read_jsonl(selected_path)
    candidates = read_jsonl(candidates_path)
    validations = read_jsonl(validation_path)
    passed, compliance = audit_relaxed_selected_prompts(
        selected_rows=selected,
        candidate_rows=candidates,
        validation_rows=validations,
    )
    compliance.update(
        {
            "created_at": _now(),
            "source_selection": _identity(selected_path),
            "source_candidates": _identity(candidates_path),
            "source_validation": _identity(validation_path),
        }
    )
    atomic_json(output / "compliance-summary.json", compliance)
    if len(passed) != len(selected):
        raise ValueError(
            f"relaxed audit failed for {len(selected) - len(passed)} selected prompts; "
            f"inspect {output / 'compliance-summary.json'}"
        )
    _write_or_verify_jsonl(output / "compliant-candidates.jsonl", passed)
    print(f"RELAXED_COMPLIANCE=PASS prompts={len(passed)}", flush=True)
    if args.audit_only:
        return 0
    if "SLURM_JOB_ID" not in os.environ:
        raise ValueError("mapping requires an active Slurm allocation")
    gpu_tokens = _gpu_tokens()
    if args.shard_count < 4 or args.shard_count % 4:
        raise ValueError("shard-count must be a positive multiple of four")

    shards = partition_candidate_rows(passed, shard_count=args.shard_count)
    shard_paths = tuple(
        output / "inputs" / f"candidates-{index:03d}.jsonl"
        for index in range(args.shard_count)
    )
    for path, rows in zip(shard_paths, shards):
        _write_or_verify_jsonl(path, rows)
    atomic_json(
        output / "partition-manifest.json",
        {
            "format_version": FORMAT_VERSION,
            "created_at": _now(),
            "candidate_count": len(passed),
            "shard_count": len(shards),
            "shard_sizes": [len(rows) for rows in shards],
            "source": _identity(output / "compliant-candidates.jsonl"),
        },
    )

    run_manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": _now(),
        "git_commit_sha": _git_sha(repository),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "approved_walltime": os.environ.get("READINESS_APPROVED_WALLTIME"),
        "allocation_estimate": os.environ.get("READINESS_ALLOCATION_ESTIMATE"),
        "visible_gpus": list(gpu_tokens),
        "candidate_count": len(passed),
        "shard_count": len(shards),
        "embedding_batch_size": args.embedding_batch_size,
    }
    atomic_json(output / "run-manifest.json", run_manifest)

    views = (
        (
            "qwen",
            args.qwen_python,
            Path(args.qwen_map_root),
            Path(args.qwen_embedding_model),
            Path(args.qwen_mntp_model),
            Path(args.qwen_peft_model),
        ),
        (
            "mistral",
            args.mistral_python,
            Path(args.mistral_map_root),
            Path(args.mistral_embedding_model),
            Path(args.mistral_mntp_model),
            Path(args.mistral_peft_model),
        ),
    )
    merged_roots = {}
    for view, python, map_root, embedding_model, mntp_model, peft_model in views:
        shard_outputs = _run_view(
            view=view,
            shard_paths=shard_paths,
            output_root=output,
            repository=repository,
            python=python,
            map_root=map_root,
            embedding_model=embedding_model,
            mntp_model=mntp_model,
            peft_model=peft_model,
            batch_size=args.embedding_batch_size,
            gpu_tokens=gpu_tokens,
        )
        merged_roots[view] = _merge_view(
            view=view,
            candidate_rows=passed,
            shard_paths=shard_paths,
            shard_outputs=shard_outputs,
            output_root=output,
        )

    comparison = output / "comparison"
    comparison_manifest = comparison / "comparison_manifest.json"
    comparison_complete = False
    if comparison_manifest.is_file():
        try:
            comparison_complete = int(read_json(comparison_manifest)["candidate_count"]) == len(passed)
        except (KeyError, TypeError, ValueError):
            comparison_complete = False
    if not comparison_complete:
        if comparison.exists():
            _quarantine(comparison, output / "quarantine")
        subprocess.run(
            [
                sys.executable,
                str(repository / "analysis/scripts/build_readiness_prompt_population.py"),
                "compare-projections",
                "--reference-projections",
                str(merged_roots["qwen"]),
                "--candidate-projections",
                str(merged_roots["mistral"]),
                "--robustness-battery",
                str(Path(args.robustness_battery).resolve()),
                "--output-dir",
                str(comparison),
            ],
            cwd=repository,
            check=True,
        )
    axis_summary = _finalize_axis(output, comparison)
    comparison_summary = read_json(comparison / "projection_comparison.json")
    final_summary = {
        "format_version": FORMAT_VERSION,
        "created_at": _now(),
        "candidate_count": len(passed),
        "relaxed_compliance": compliance,
        "projection_comparison": comparison_summary,
        "consensus_axis_1_z": axis_summary,
        "final_axis_map": _identity(output / "final-axis-map.jsonl"),
        "result": "PASS",
    }
    atomic_json(output / "final-audit-summary.json", final_summary)
    print(f"FINAL_AUDIT=PASS prompts={len(passed)}", flush=True)
    print(f"output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
