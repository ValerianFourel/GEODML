"""Paired outcome analysis for the ACL ARR document experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .acl_arr_document_experiment import (
    AclArrExperimentPlan,
    CONDITIONS,
    iter_experiment_tasks,
)


ANALYSIS_FORMAT_VERSION = "acl-arr-paired-analysis-v1"


@dataclass(frozen=True, slots=True)
class AclArrAnalysis:
    summary: Mapping[str, Any]
    paired_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AclArrAnalysisArtifacts:
    summary_path: Path
    paired_rows_path: Path
    report_path: Path


def analyze_acl_arr_outcomes(
    rerank_outcomes: Sequence[Mapping[str, Any]],
    answer_outcomes: Sequence[Mapping[str, Any]],
    judge_outcomes: Sequence[Mapping[str, Any]],
    judge_mappings: Sequence[Mapping[str, Any]],
    *,
    plan: AclArrExperimentPlan,
    allow_fake: bool = False,
) -> AclArrAnalysis:
    """Audit complete paired cells and calculate preregisterable contrasts."""

    expected_rerank = {
        task.task_id
        for task in iter_experiment_tasks(plan)
        if task.pipeline == "rerank"
    }
    expected_answer = {
        task.task_id
        for task in iter_experiment_tasks(plan)
        if task.pipeline == "answer"
    }
    rerank_by_id = _unique_rows(rerank_outcomes, "task_id", "rerank")
    answer_by_id = _unique_rows(answer_outcomes, "task_id", "answer")
    if set(rerank_by_id) != expected_rerank:
        raise ValueError(_coverage_error("rerank", expected_rerank, set(rerank_by_id)))
    if set(answer_by_id) != expected_answer:
        raise ValueError(_coverage_error("answer", expected_answer, set(answer_by_id)))
    if not allow_fake and any(
        row.get("fake_backend") is True
        for row in (*rerank_outcomes, *answer_outcomes, *judge_outcomes)
    ):
        raise ValueError("fake outputs are not eligible for scientific analysis")

    mappings_by_judge_id = _unique_rows(
        judge_mappings, "judge_task_id", "judge mapping"
    )
    judge_by_id = _unique_rows(judge_outcomes, "judge_task_id", "judge")
    if set(judge_by_id) != set(mappings_by_judge_id):
        raise ValueError(
            _coverage_error(
                "judge", set(mappings_by_judge_id), set(judge_by_id)
            )
        )
    mapped_generation_ids = {
        str(row["source_generation_task_id"])
        for row in mappings_by_judge_id.values()
    }
    if mapped_generation_ids != expected_answer:
        raise ValueError(
            _coverage_error("judge mapping", expected_answer, mapped_generation_ids)
        )

    judge_by_generation: dict[str, Mapping[str, Any]] = {}
    for judge_task_id, mapping in mappings_by_judge_id.items():
        generation_id = str(mapping["source_generation_task_id"])
        if generation_id in judge_by_generation:
            raise ValueError("multiple judge outcomes map to one generated answer")
        judgment = judge_by_id[judge_task_id].get("parsed_output")
        if not isinstance(judgment, dict):
            raise ValueError("judge outcome lacks parsed_output")
        judge_by_generation[generation_id] = judgment

    prompt_by_id = {item.prompt_id: item for item in plan.prompts}
    rerank_cells = _condition_cells(rerank_outcomes, "rerank")
    answer_cells = _condition_cells(answer_outcomes, "answer")
    expected_pairs = {
        (prompt.prompt_id, model.configuration_id)
        for prompt in plan.prompts
        for model in plan.models
    }
    if set(rerank_cells) != expected_pairs or set(answer_cells) != expected_pairs:
        raise ValueError("prompt-model paired cell coverage is incomplete")

    paired_rows: list[dict[str, Any]] = []
    for pair in sorted(expected_pairs):
        prompt_id, model_configuration_id = pair
        prompt = prompt_by_id[prompt_id]
        rerank = rerank_cells[pair]
        answer = answer_cells[pair]
        rerank_rankings = {
            condition: _string_list(
                row, "parsed_output", "ranked_document_ids"
            )
            for condition, row in rerank.items()
        }
        citations = {
            condition: _string_list(row, "parsed_output", "cited_document_ids")
            for condition, row in answer.items()
        }
        judgments = {
            condition: judge_by_generation[str(row["task_id"])]
            for condition, row in answer.items()
        }
        realized = {
            condition: [
                str(item["document_id"])
                for item in judgment["realized_document_ranking"]
            ]
            for condition, judgment in judgments.items()
        }
        target = str(rerank["natural"]["ablation_target_id"])
        paired_rows.append(
            {
                "prompt_id": prompt_id,
                "model_configuration_id": model_configuration_id,
                "model_id": rerank["natural"]["model_id"],
                "assigned_readiness_0_1": prompt.assigned_readiness_0_1,
                "consensus_axis_1_z": prompt.consensus_axis_1_z,
                "axis_1_percentile_0_1": prompt.axis_1_percentile_0_1,
                "ablation_target_id": target,
                "rerank_shuffle_top_k_jaccard": _jaccard(
                    rerank_rankings["natural"], rerank_rankings["shuffled"]
                ),
                "rerank_shuffle_kendall_common": _kendall_common(
                    rerank_rankings["natural"], rerank_rankings["shuffled"]
                ),
                "rerank_ablation_top_k_jaccard": _jaccard(
                    [item for item in rerank_rankings["natural"] if item != target],
                    rerank_rankings["ablated"],
                ),
                "rerank_ablation_kendall_common": _kendall_common(
                    [item for item in rerank_rankings["natural"] if item != target],
                    rerank_rankings["ablated"],
                ),
                "answer_shuffle_citation_jaccard": _jaccard(
                    citations["natural"], citations["shuffled"]
                ),
                "answer_ablation_citation_jaccard": _jaccard(
                    [item for item in citations["natural"] if item != target],
                    citations["ablated"],
                ),
                "target_cited_natural": target in citations["natural"],
                "target_cited_shuffled": target in citations["shuffled"],
                "judge_shuffle_realized_use_jaccard": _jaccard(
                    realized["natural"], realized["shuffled"]
                ),
                "judge_ablation_realized_use_jaccard": _jaccard(
                    [item for item in realized["natural"] if item != target],
                    realized["ablated"],
                ),
                "judge_shuffle_answer_quality_delta": int(
                    judgments["shuffled"]["answer_quality"]
                )
                - int(judgments["natural"]["answer_quality"]),
                "judge_ablation_answer_quality_delta": int(
                    judgments["ablated"]["answer_quality"]
                )
                - int(judgments["natural"]["answer_quality"]),
                "judge_shuffle_evidence_coverage_delta": int(
                    judgments["shuffled"]["evidence_coverage"]
                )
                - int(judgments["natural"]["evidence_coverage"]),
                "judge_ablation_evidence_coverage_delta": int(
                    judgments["ablated"]["evidence_coverage"]
                )
                - int(judgments["natural"]["evidence_coverage"]),
            }
        )

    metric_names = sorted(
        key
        for key in paired_rows[0]
        if key.startswith(("rerank_", "answer_", "judge_"))
    )
    metric_means = {
        key: _mean(
            row[key]
            for row in paired_rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
        )
        for key in metric_names
    }
    fake = any(
        row.get("fake_backend") is True
        for row in (*rerank_outcomes, *answer_outcomes, *judge_outcomes)
    )
    summary = {
        "format_version": ANALYSIS_FORMAT_VERSION,
        "result": "PASS",
        "scientific_result": not fake,
        "source_plan_id": plan.plan_id,
        "prompt_count": len(plan.prompts),
        "model_count": len(plan.models),
        "paired_prompt_model_count": len(paired_rows),
        "rerank_outcome_count": len(rerank_outcomes),
        "answer_outcome_count": len(answer_outcomes),
        "judge_outcome_count": len(judge_outcomes),
        "metric_means": metric_means,
        "interpretation_guard": (
            "Assigned readiness is the prompt variable. Consensus axis coordinates "
            "are measurement fields, not confounders. Page-feature effects remain "
            "observational unless page content or features are randomized."
        ),
    }
    return AclArrAnalysis(summary=summary, paired_rows=tuple(paired_rows))


def write_acl_arr_analysis(
    output_directory: str | Path, *, analysis: AclArrAnalysis
) -> AclArrAnalysisArtifacts:
    """Write paired rows and an auditable aggregate summary."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")
    output.mkdir(parents=True)
    summary_path = output / "analysis_summary.json"
    rows_path = output / "paired_prompt_model_outcomes.jsonl"
    report_path = output / "README.md"
    _atomic_text(summary_path, json.dumps(analysis.summary, indent=2, sort_keys=True) + "\n")
    _atomic_text(
        rows_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in analysis.paired_rows
        ),
    )
    _atomic_text(
        report_path,
        "\n".join(
            (
                "# ACL ARR paired document experiment analysis",
                "",
                f"- Result: {analysis.summary['result']}",
                f"- Prompts: {analysis.summary['prompt_count']}",
                f"- Models: {analysis.summary['model_count']}",
                f"- Paired prompt-model rows: {analysis.summary['paired_prompt_model_count']}",
                f"- Scientific result: {analysis.summary['scientific_result']}",
                "",
                str(analysis.summary["interpretation_guard"]),
                "",
            )
        ),
    )
    return AclArrAnalysisArtifacts(
        summary_path=summary_path,
        paired_rows_path=rows_path,
        report_path=report_path,
    )


def _condition_cells(
    rows: Sequence[Mapping[str, Any]], pipeline: str
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    cells: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("pipeline") != pipeline:
            raise ValueError(f"{pipeline} outcome file contains another pipeline")
        key = (str(row["prompt_id"]), str(row["model_configuration_id"]))
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError("outcome has an unknown condition")
        if condition in cells.setdefault(key, {}):
            raise ValueError("paired cell contains a duplicate condition")
        cells[key][condition] = row
    for key, values in cells.items():
        if set(values) != set(CONDITIONS):
            raise ValueError(f"paired cell is incomplete: {key}")
    return cells


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], id_field: str, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(id_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} outcome lacks {id_field}")
        if value in output:
            raise ValueError(f"{label} outcomes contain duplicate {id_field}")
        output[value] = row
    return output


def _coverage_error(label: str, expected: set[str], observed: set[str]) -> str:
    return (
        f"{label} coverage mismatch: expected={len(expected)} observed={len(observed)} "
        f"missing={len(expected - observed)} unexpected={len(observed - expected)}"
    )


def _string_list(
    row: Mapping[str, Any], object_field: str, list_field: str
) -> list[str]:
    value = row.get(object_field)
    if not isinstance(value, dict):
        raise ValueError(f"outcome lacks {object_field}")
    items = value.get(list_field)
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ValueError(f"outcome lacks {list_field}")
    return items


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _kendall_common(left: Sequence[str], right: Sequence[str]) -> float | None:
    common = set(left) & set(right)
    if len(common) < 2:
        return None
    left_order = [item for item in left if item in common]
    right_position = {item: index for index, item in enumerate(right) if item in common}
    concordant = 0
    discordant = 0
    for first in range(len(left_order)):
        for second in range(first + 1, len(left_order)):
            if right_position[left_order[first]] < right_position[left_order[second]]:
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _atomic_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


__all__ = [
    "ANALYSIS_FORMAT_VERSION",
    "AclArrAnalysis",
    "AclArrAnalysisArtifacts",
    "analyze_acl_arr_outcomes",
    "write_acl_arr_analysis",
]
