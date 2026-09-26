#!/usr/bin/env python3
"""Relate frozen prompt-axis coordinates to verified Experiment V2 rankings.

This is a CPU-only, saved-artifact analysis. It does not run inference or
compute embeddings. Assigned coordinates and observed latent coordinates are
reported separately. The latter describe prompts and are not treatments or
confounders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.interpretability.pipeline.agentic_audit_progress import (
    audit_progress,
    audit_stage,
)
from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.axis_permutation_metrics import ranking_agreement
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity

FORMAT_VERSION = "geodml-latent-ranking-relationship-v1"
GENERATOR_MODELS = ("qwen38", "llama4")
COORDINATES = (
    ("assigned_axis_1_0_1", "assigned"),
    ("observed_axis_1_percentile_0_1", "observed_latent"),
)
TASK_OUTCOMES = (
    "ranking_length",
    "target_selected_if_seen",
    "target_reciprocal_rank_if_seen",
)
PAIR_OUTCOMES = (
    "natural_vs_shuffled_top1_change",
    "natural_vs_shuffled_topk_change",
    "natural_vs_shuffled_kendall_distance_common",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load_coordinates(
    path: Path, manifest_path: Path | None
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = path.resolve()
    digest = _sha256(path)
    provenance: dict[str, Any] = {"path": str(path), "sha256": digest}
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get("files", {}).get(path.name)
        if not isinstance(entry, dict) or entry.get("sha256") != digest:
            raise ValueError("coordinate file does not match its registration manifest")
        if entry.get("bytes") is not None and entry["bytes"] != path.stat().st_size:
            raise ValueError("coordinate file size does not match its registration manifest")
        provenance["manifest_path"] = str(manifest_path)
        provenance["manifest_sha256"] = _sha256(manifest_path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row.get("candidate_id")
            if not isinstance(prompt_id, str) or not prompt_id or prompt_id in rows:
                raise ValueError(f"invalid or duplicate coordinate prompt at line {number}")
            axis_bin = row.get("axis_bin")
            if type(axis_bin) is not int or axis_bin < 0:
                raise ValueError(f"invalid axis bin at line {number}")
            normalized = {"prompt_id": prompt_id, "axis_bin": axis_bin}
            for field, _ in COORDINATES:
                value = _finite_number(row.get(field), f"{field} at line {number}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field} must lie in [0, 1]")
                normalized[field] = value
            rows[prompt_id] = normalized
    if not rows:
        raise ValueError("coordinate file is empty")
    provenance["rows"] = len(rows)
    return rows, provenance


def _verified_envelopes(
    root: Path, references: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Resolve ledger references while hashing each immutable shard once."""

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        table = reference.get("table")
        writer = reference.get("writer_id")
        sequence = reference.get("shard_sequence")
        if not isinstance(table, str) or not isinstance(writer, str) or type(sequence) is not int:
            raise ValueError("completed ledger event contains an invalid record reference")
        grouped[(table, writer, sequence)].append(reference)
    resolved: dict[str, dict[str, Any]] = {}
    for shard_index, ((table, writer, sequence), shard_references) in enumerate(
        sorted(grouped.items()), 1
    ):
        audit_progress(
            phase="verify_references",
            referenced_shards=shard_index,
            referenced_shards_total=len(grouped),
        )
        stem = f"part-{writer}-{sequence:06d}"
        directory = root / "data" / table
        manifest_path = directory / f"{stem}.manifest.json"
        shard_path = directory / f"{stem}.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = shard_path.read_bytes()
        lines = raw.splitlines()
        if (
            manifest.get("format_version") != "geodml-jsonl-shard-v1"
            or manifest.get("path") != str(shard_path.relative_to(root))
            or manifest.get("sha256") != hashlib.sha256(raw).hexdigest()
            or manifest.get("rows") != len(lines)
        ):
            raise ValueError(f"referenced shard failed verification: {shard_path}")
        for reference in shard_references:
            line_number = reference.get("line_number")
            if type(line_number) is not int or not 1 <= line_number <= len(lines):
                raise ValueError("record reference line number is invalid")
            envelope = json.loads(lines[line_number - 1])
            record_id = reference.get("record_id")
            if (
                envelope.get("record_id") != record_id
                or envelope.get("transaction_id") != reference.get("transaction_id")
                or not isinstance(envelope.get("row"), dict)
            ):
                raise ValueError("record reference does not match its sealed envelope")
            if table == "generations":
                if record_id in resolved and resolved[record_id] != envelope:
                    raise ValueError("one record ID resolves to conflicting envelopes")
                resolved[record_id] = envelope
    return resolved


def _load_observations(
    root: Path,
    coordinates: Mapping[str, Mapping[str, Any]],
    models: set[str],
    *,
    stripe_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    memberships = {
        row["prompt_id"]: row
        for row in iter_sealed_rows(root, "keyword_memberships", required=True)
    }
    tasks: dict[str, dict[str, Any]] = {}
    task_counts = defaultdict(int)
    for task_index, row in enumerate(
        iter_sealed_rows(root, "task_definitions", required=True), 1
    ):
        if task_index % 1000 == 0:
            audit_progress(phase="task_definitions", tasks_checked=task_index)
        if row.get("model") not in models:
            continue
        identity = ClaimIdentity(**row["claim_identity"])
        fingerprint = identity_fingerprint(identity)
        if fingerprint in tasks:
            raise ValueError("duplicate task fingerprint")
        prompt_id = row.get("prompt_id")
        if prompt_id not in coordinates or prompt_id not in memberships:
            raise ValueError(f"task lacks frozen coordinate or membership: {prompt_id}")
        tasks[fingerprint] = row
        task_counts[row["model"]] += 1
    snapshot = StripedTaskLedger(
        root / "control" / "task-ledger", stripe_count=stripe_count
    ).snapshot()
    audit_progress(
        phase="ledger", tasks_checked=len(tasks),
        ledger_events=snapshot["event_count"],
    )
    completed: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    all_references: list[Mapping[str, Any]] = []
    for fingerprint, task in tasks.items():
        event = snapshot["latest"].get(fingerprint)
        if not isinstance(event, dict) or event.get("state") != "completed":
            continue
        references = event.get("record_references")
        if not isinstance(references, list) or not references:
            raise ValueError("completed task has no record references")
        generation_references = [row for row in references if row.get("table") == "generations"]
        if len(generation_references) != 1:
            raise ValueError("completed generator task must reference exactly one generation")
        completed.append((task, generation_references[0]))
        all_references.extend(references)
    envelopes = _verified_envelopes(root, all_references)
    observations = []
    completed_counts = defaultdict(int)
    for observation_index, (task, reference) in enumerate(completed, 1):
        if observation_index % 1000 == 0:
            audit_progress(
                phase="observations",
                observations_built=observation_index,
                observations_total=len(completed),
            )
        generation = envelopes[reference["record_id"]]["row"]
        if generation.get("cell_id") != task.get("task_id"):
            raise ValueError("generation cell ID differs from its task definition")
        prompt_id = task["prompt_id"]
        membership = memberships[prompt_id]
        ranking = generation.get("ranking")
        if (
            not isinstance(ranking, list)
            or any(not isinstance(item, str) or not item for item in ranking)
            or len(set(ranking)) != len(ranking)
        ):
            raise ValueError("generation ranking is invalid")
        condition_audit = generation.get("condition_audit")
        target_url = None
        target_seen = False
        if condition_audit is not None:
            if not isinstance(condition_audit, dict):
                raise TypeError("condition audit must be an object")
            target_url = condition_audit.get("target_url")
            target_seen = condition_audit.get("target_observed") is True
            if not isinstance(target_url, str) or not target_url:
                raise ValueError("condition audit lacks its target URL")
        target_rank = ranking.index(target_url) + 1 if target_url in ranking else None
        observations.append({
            "task_id": task["task_id"],
            "prompt_id": prompt_id,
            "model": task["model"],
            "method": task.get("method"),
            "engine": task.get("engine"),
            "condition": task.get("condition"),
            "keyword_id": membership["primary_keyword_id"],
            **coordinates[prompt_id],
            "ranking": ranking,
            "ranking_length": len(ranking),
            "target_seen": target_seen,
            "target_selected_if_seen": (
                float(target_rank is not None) if target_seen else None
            ),
            "target_reciprocal_rank_if_seen": (
                0.0 if target_seen and target_rank is None
                else 1.0 / target_rank if target_seen else None
            ),
        })
        completed_counts[task["model"]] += 1
    accounting = {
        "ledger_event_count": snapshot["event_count"],
        "registered_tasks": dict(sorted(task_counts.items())),
        "verified_completed_tasks": dict(sorted(completed_counts.items())),
    }
    return observations, accounting


def _paired_permutation_rows(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in observations:
        if row["condition"] not in {"natural", "shuffled"}:
            continue
        key = tuple(str(row[field]) for field in ("model", "prompt_id", "method", "engine"))
        if row["condition"] in groups[key]:
            raise ValueError("duplicate natural or shuffled task in a pairing cell")
        groups[key][row["condition"]] = row
    paired = []
    for key, values in sorted(groups.items()):
        if set(values) != {"natural", "shuffled"}:
            continue
        natural, shuffled = values["natural"], values["shuffled"]
        agreement = ranking_agreement(natural["ranking"], shuffled["ranking"])
        paired.append({
            "model": key[0], "prompt_id": key[1], "method": key[2],
            "engine": key[3], "keyword_id": natural["keyword_id"],
            "axis_bin": natural["axis_bin"],
            **{field: natural[field] for field, _ in COORDINATES},
            "natural_vs_shuffled_top1_change": (
                None if agreement["top1_match"] is None
                else 1.0 - float(agreement["top1_match"])
            ),
            "natural_vs_shuffled_topk_change": (
                None if agreement["topk_overlap"] is None
                else 1.0 - agreement["topk_overlap"]
            ),
            "natural_vs_shuffled_kendall_distance_common": (
                None if agreement["kendall_tau_common"] is None
                else (1.0 - agreement["kendall_tau_common"]) / 2.0
            ),
        })
    return paired


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    ranked_x, ranked_y = rankdata(x), rankdata(y)
    if np.std(ranked_x) == 0 or np.std(ranked_y) == 0:
        return float("nan")
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def _association(
    rows: Sequence[Mapping[str, Any]],
    *,
    coordinate: str,
    coordinate_role: str,
    outcome: str,
    permutations: int,
    seed: int,
    identity: Mapping[str, str],
) -> dict[str, Any]:
    selected = [row for row in rows if row.get(outcome) is not None]
    result: dict[str, Any] = {
        **identity,
        "coordinate": coordinate,
        "coordinate_role": coordinate_role,
        "outcome": outcome,
        "n": len(selected),
        "unique_prompts": len({row["prompt_id"] for row in selected}),
        "unique_keywords": len({row["keyword_id"] for row in selected}),
        "spearman_rho": None,
        "linear_slope_per_axis_unit": None,
        "blocked_permutation_p_two_sided": None,
        "permutations": permutations,
        "permutable_keywords": 0,
        "status": "insufficient_variation",
    }
    if len(selected) < 3:
        return result
    x = np.asarray([row[coordinate] for row in selected], dtype=float)
    y = np.asarray([row[outcome] for row in selected], dtype=float)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return result
    rho = _correlation(x, y)
    if not math.isfinite(rho):
        return result
    design = np.column_stack((np.ones(len(x)), x))
    slope = float(np.linalg.lstsq(design, y, rcond=None)[0][1])
    blocks: list[np.ndarray] = []
    by_keyword: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(selected):
        by_keyword[str(row["keyword_id"])].append(index)
    for indices in by_keyword.values():
        block = np.asarray(indices, dtype=int)
        if len(block) > 1 and len(np.unique(x[block])) > 1:
            blocks.append(block)
    result.update(
        status="descriptive_association",
        spearman_rho=rho,
        linear_slope_per_axis_unit=slope,
        permutable_keywords=len(blocks),
    )
    if permutations == 0 or not blocks:
        return result
    identity_seed = int(
        hashlib.sha256(
            _canonical([seed, identity, coordinate, outcome])
        ).hexdigest()[:16],
        16,
    )
    rng = np.random.default_rng(identity_seed)
    extreme = 0
    permuted = x.copy()
    for _ in range(permutations):
        permuted[:] = x
        for block in blocks:
            permuted[block] = rng.permutation(x[block])
        statistic = _correlation(permuted, y)
        if math.isfinite(statistic) and abs(statistic) >= abs(rho) - 1e-15:
            extreme += 1
    result["blocked_permutation_p_two_sided"] = (extreme + 1) / (permutations + 1)
    return result


def _associations(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[str],
    strata: Sequence[str],
    *,
    permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in strata)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        identity = dict(zip(strata, key, strict=True))
        for coordinate, role in COORDINATES:
            for outcome in outcomes:
                output.append(_association(
                    group, coordinate=coordinate, coordinate_role=role,
                    outcome=outcome, permutations=permutations, seed=seed,
                    identity=identity,
                ))
    return output


def _bin_summaries(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("model", "method", "engine", "condition", "axis_bin")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        seen = [row for row in rows if row["target_seen"]]
        output.append({
            **dict(zip(fields, key, strict=True)),
            "tasks": len(rows),
            "mean_observed_latent_coordinate": float(np.mean([
                row["observed_axis_1_percentile_0_1"] for row in rows
            ])),
            "mean_ranking_length": float(np.mean([row["ranking_length"] for row in rows])),
            "target_seen_tasks": len(seen),
            "target_selection_rate_if_seen": (
                float(np.mean([row["target_selected_if_seen"] for row in seen]))
                if seen else None
            ),
            "mean_target_reciprocal_rank_if_seen": (
                float(np.mean([row["target_reciprocal_rank_if_seen"] for row in seen]))
                if seen else None
            ),
        })
    return output


def _permutation_bin_summaries(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = ("model", "method", "engine", "axis_bin")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        summary = {
            **dict(zip(fields, key, strict=True)),
            "pairs": len(rows),
            "mean_observed_latent_coordinate": float(np.mean([
                row["observed_axis_1_percentile_0_1"] for row in rows
            ])),
        }
        for outcome in PAIR_OUTCOMES:
            values = [row[outcome] for row in rows if row[outcome] is not None]
            summary[f"{outcome}_n"] = len(values)
            summary[f"mean_{outcome}"] = (
                float(np.mean(values)) if values else None
            )
        output.append(summary)
    return output


def permutation_sniff_check(
    observations: Sequence[Mapping[str, Any]],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize how natural-versus-shuffled ranking change varies by axis."""

    pairs = _paired_permutation_rows(observations)
    return {
        "pairs": pairs,
        "associations": _associations(
            pairs, PAIR_OUTCOMES, ("model", "method", "engine"),
            permutations=permutations, seed=seed,
        ),
        "bin_summaries": _permutation_bin_summaries(pairs),
    }


@audit_stage("latent_ranking_report")
def build_report(
    dataset_root: Path,
    coordinate_path: Path,
    *,
    coordinate_manifest: Path | None,
    models: Sequence[str],
    axis_construct: str,
    permutations: int = 0,
    seed: int = 20260926,
    stripe_count: int = 256,
) -> dict[str, Any]:
    """Build one deterministic report from a ledger snapshot and sealed records."""

    if not axis_construct.strip():
        raise ValueError("axis construct must be named")
    if permutations < 0 or stripe_count < 1:
        raise ValueError("permutations must be non-negative and stripe count positive")
    selected_models = set(models)
    if not selected_models or not selected_models <= set(GENERATOR_MODELS):
        raise ValueError("select at least one supported generator model")
    dataset_root = dataset_root.resolve()
    contract_path = dataset_root / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("format_version") != "geodml-incremental-dataset-v1":
        raise ValueError("unsupported incremental dataset contract")
    audit_progress(phase="coordinates")
    coordinates, coordinate_provenance = _load_coordinates(
        coordinate_path, coordinate_manifest
    )
    observations, accounting = _load_observations(
        dataset_root, coordinates, selected_models, stripe_count=stripe_count
    )
    sniff = permutation_sniff_check(
        observations, permutations=permutations, seed=seed
    )
    pairs = sniff["pairs"]
    audit_progress(
        phase="associations", observations=len(observations), pairs=len(pairs)
    )
    task_associations = _associations(
        observations, TASK_OUTCOMES, ("model", "method", "engine", "condition"),
        permutations=permutations, seed=seed,
    )
    report = {
        "format_version": FORMAT_VERSION,
        "scientific_result": False,
        "analysis_kind": "exploratory_saved_artifact_descriptive_association",
        "axis_construct": axis_construct,
        "warnings": [
            (
                "Observed latent coordinates describe prompts; they do not define "
                "the assigned treatment and are not confounders."
            ),
            (
                "Associations are exploratory and do not identify causal "
                "page-feature or prompt-policy effects."
            ),
            (
                "Incomplete, priority-ordered task completion can induce selection "
                "bias; rerun on the frozen complete population for final reporting."
            ),
            (
                "Raw blocked-permutation p-values are descriptive diagnostics and "
                "are not corrected for repeated looks or multiple outcomes."
            ),
            (
                "Natural-versus-shuffled comparisons use only jointly ranked URLs; "
                "absent URLs receive no invented rank."
            ),
            (
                "Top-k change is one minus overlap, with k=min(3, natural "
                "ranking length, shuffled ranking length) for each pair."
            ),
        ],
        "settings": {
            "models": sorted(selected_models), "permutations": permutations,
            "seed": seed, "stripe_count": stripe_count,
            "permutation_blocks": "primary_keyword_id",
        },
        "provenance": {
            "dataset_root": str(dataset_root),
            "contract_sha256": _sha256(contract_path),
            "coordinates": coordinate_provenance,
        },
        "accounting": accounting,
        "observation_count": len(observations),
        "natural_shuffled_pair_count": len(pairs),
        "task_associations": task_associations,
        "permutation_associations": sniff["associations"],
        "permutation_bin_summaries": sniff["bin_summaries"],
        "bin_summaries": _bin_summaries(observations),
    }
    audit_progress(phase="done", observations=len(observations), pairs=len(pairs))
    return report


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Latent-axis permutation sniff check",
        "",
        "> Exploratory saved-artifact report. This is not a causal or confirmatory result.",
        "",
        f"Axis construct: `{report['axis_construct']}`.",
        f"Verified completed task observations: {report['observation_count']:,}.",
        f"Complete natural/shuffled pairs: {report['natural_shuffled_pair_count']:,}.",
        "",
        "## Interpretation limits",
        "",
    ]
    lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend([
        "", "## Ranking change along the observed latent axis", "",
        (
            "Each outcome is a distance from 0 to 1. Zero means no measured "
            "change and one means the largest measured change."
        ),
        "",
        "| Model | Method | Engine | Change outcome | pairs | Spearman rho | Blocked permutation p |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for row in report["permutation_associations"]:
        if row["coordinate_role"] != "observed_latent":
            continue
        lines.append("| " + " | ".join(_format(row.get(field)) for field in (
            "model", "method", "engine", "outcome", "n", "spearman_rho",
            "blocked_permutation_p_two_sided",
        )) + " |")
    lines.extend([
        "", "## Other observed latent-coordinate associations", "",
        (
            "| Model | Method | Engine | Condition | Outcome | n | Spearman rho "
            "| Blocked permutation p |"
        ),
        "|---|---|---|---|---|---:|---:|---:|",
    ])
    rows = [
        row for row in report["task_associations"]
        if row["coordinate_role"] == "observed_latent"
        and row["outcome"] in {"ranking_length", "target_reciprocal_rank_if_seen"}
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in (
            "model", "method", "engine", "condition", "outcome", "n",
            "spearman_rho", "blocked_permutation_p_two_sided",
        )) + " |")
    lines.extend([
        "", "Full assigned-versus-observed results are in `associations.csv`.",
        "Permutation change by axis bin is in `permutation-bin-summary.csv`.",
        "Other axis-bin outcome summaries are in `bin-summary.csv`.", "",
    ])
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output: Path) -> Path:
    """Atomically publish a new report directory; never replace prior output."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "report.md").write_text(_markdown(report), encoding="utf-8")
        _write_csv(
            temporary / "associations.csv",
            [*report["task_associations"], *report["permutation_associations"]],
        )
        _write_csv(temporary / "bin-summary.csv", report["bin_summaries"])
        _write_csv(
            temporary / "permutation-bin-summary.csv",
            report["permutation_bin_summaries"],
        )
        os.replace(temporary, output)
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--coordinate-manifest", type=Path)
    parser.add_argument("--model", action="append", choices=GENERATOR_MODELS)
    parser.add_argument("--axis-construct", required=True)
    parser.add_argument(
        "--permutations", type=int, default=0,
        help="Within-keyword permutations per association. Zero reports effect sizes only.",
    )
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--stripe-count", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registration = args.dataset_root / "local-only" / "population-registration-v1"
    coordinates = args.coordinates or registration / "population-selection-records.jsonl"
    manifest = args.coordinate_manifest
    if manifest is None and args.coordinates is None:
        manifest = registration / "manifest.json"
    report = build_report(
        args.dataset_root, coordinates, coordinate_manifest=manifest,
        models=args.model or GENERATOR_MODELS, axis_construct=args.axis_construct,
        permutations=args.permutations, seed=args.seed, stripe_count=args.stripe_count,
    )
    destination = write_report(report, args.output_dir)
    print(f"REPORT={destination / 'report.md'}")
    print(f"OBSERVATIONS={report['observation_count']}")
    print(f"NATURAL_SHUFFLED_PAIRS={report['natural_shuffled_pair_count']}")
    print("SCIENTIFIC_RESULT=false")
    print("INFERENCE_OR_ALLOCATION_STARTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
