"""Contracts for the ACL ARR document-order and ablation experiment.

The module prepares paired Natural, Ablated, and Shuffled inputs for two
pipelines: document reranking and cited answer generation. Search is never run
here. Every task refers to an immutable, previously collected document set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence


FORMAT_VERSION = "acl-arr-document-experiment-v1"
JUDGE_FORMAT_VERSION = "acl-arr-realized-use-judge-v1"
CONDITIONS = ("natural", "ablated", "shuffled")
PIPELINES = ("rerank", "answer")


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """One immutable generation or reranking model arm."""

    model_id: str
    model_revision: str
    architecture: str
    total_parameters_b: float
    active_parameters_b: float
    precision: str
    rerank_max_tokens: int = 256
    answer_max_tokens: int = 768
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.model_id.startswith("UNRESOLVED_"):
            raise ValueError("resolve the model_id before preparing a run")
        if re.fullmatch(r"[0-9a-f]{40}", self.model_revision) is None:
            raise ValueError("model_revision must be an immutable 40-character SHA")
        if self.architecture not in {"dense", "moe"}:
            raise ValueError("architecture must be dense or moe")
        if not math.isfinite(self.total_parameters_b) or self.total_parameters_b <= 0:
            raise ValueError("total_parameters_b must be positive and finite")
        if not math.isfinite(self.active_parameters_b) or self.active_parameters_b <= 0:
            raise ValueError("active_parameters_b must be positive and finite")
        if self.active_parameters_b > self.total_parameters_b:
            raise ValueError("active_parameters_b cannot exceed total_parameters_b")
        if not self.precision.strip():
            raise ValueError("precision must be non-empty")
        if self.rerank_max_tokens <= 0 or self.answer_max_tokens <= 0:
            raise ValueError("model output token limits must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")

    @property
    def configuration_id(self) -> str:
        return "model-config-" + _hash(_canonical(asdict(self)))[:20]


@dataclass(frozen=True, slots=True)
class FrozenDocument:
    document_id: str
    natural_position: int
    title: str
    url: str
    text: str
    text_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenDocumentSet:
    candidate_set_id: str
    keyword: str
    search_query: str
    search_engine: str
    search_snapshot_sha256: str
    documents: tuple[FrozenDocument, ...]


@dataclass(frozen=True, slots=True)
class ExperimentPrompt:
    prompt_id: str
    keyword_id: str
    keyword: str
    target_id: str
    target_index: int
    question: str
    question_sha256: str
    assigned_readiness_0_1: float
    consensus_axis_1_z: float
    axis_1_rank: int
    axis_1_percentile_0_1: float
    qwen_axis_1_z: float
    mistral_aligned_axis_1_z: float


@dataclass(frozen=True, slots=True)
class ConditionAssignment:
    assignment_id: str
    prompt_id: str
    candidate_set_id: str
    natural_document_ids: tuple[str, ...]
    ablation_target_id: str
    ablated_document_ids: tuple[str, ...]
    shuffled_document_ids: tuple[str, ...]
    shuffle_policy: str
    permutation_id: str

    def document_ids(self, condition: str) -> tuple[str, ...]:
        if condition == "natural":
            return self.natural_document_ids
        if condition == "ablated":
            return self.ablated_document_ids
        if condition == "shuffled":
            return self.shuffled_document_ids
        raise ValueError(f"unsupported condition: {condition}")


@dataclass(frozen=True, slots=True)
class ExperimentTask:
    task_id: str
    format_version: str
    plan_id: str
    model_configuration_id: str
    pipeline: str
    condition: str
    prompt_id: str
    assignment_id: str
    output_document_count: int
    decoding_seed: int


@dataclass(frozen=True, slots=True)
class AclArrExperimentPlan:
    plan_id: str
    format_version: str
    master_seed: int
    top_n: int
    prompt_source_sha256: str | None
    axis_source_sha256: str | None
    document_source_sha256: str | None
    source_git_commit: str | None
    prompts: tuple[ExperimentPrompt, ...]
    document_sets: tuple[FrozenDocumentSet, ...]
    assignments: tuple[ConditionAssignment, ...]
    models: tuple[ModelConfiguration, ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AclArrPlanArtifacts:
    manifest_path: Path
    prompts_path: Path
    document_sets_path: Path
    assignments_path: Path
    report_path: Path
    task_files: Mapping[tuple[str, str], Path]


@dataclass(frozen=True, slots=True)
class BlindedJudgeTask:
    judge_task_id: str
    format_version: str
    blind_case_id: str
    judge_model_id: str
    judge_model_revision: str
    prompt_text: str
    candidate_set_id: str
    judge_document_ids: tuple[str, ...]
    answer: str
    cited_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgeTaskMapping:
    blind_case_id: str
    judge_task_id: str
    source_generation_task_id: str
    prompt_id: str
    assignment_id: str
    generator_model_configuration_id: str
    generator_model_id: str
    condition: str
    ablation_target_id: str


@dataclass(frozen=True, slots=True)
class BlindedJudgePlan:
    judge_plan_id: str
    format_version: str
    master_seed: int
    judge_model_id: str
    judge_model_revision: str
    document_sets: tuple[FrozenDocumentSet, ...]
    tasks: tuple[BlindedJudgeTask, ...]
    mappings: tuple[JudgeTaskMapping, ...]
    summary: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BlindedJudgeArtifacts:
    manifest_path: Path
    tasks_path: Path
    private_mapping_path: Path
    report_path: Path


def build_acl_arr_experiment_plan(
    prompt_rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
    document_set_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[ModelConfiguration],
    top_n: int,
    master_seed: int = 20260904,
    prompt_source_sha256: str | None = None,
    axis_source_sha256: str | None = None,
    document_source_sha256: str | None = None,
    source_git_commit: str | None = None,
) -> AclArrExperimentPlan:
    """Build one deterministic paired experiment without running inference."""

    if not prompt_rows:
        raise ValueError("prompt population must not be empty")
    if not axis_rows:
        raise ValueError("final axis map must not be empty")
    if not document_set_rows:
        raise ValueError("frozen document sets must not be empty")
    if not models:
        raise ValueError("at least one model configuration is required")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    _validate_optional_digest(prompt_source_sha256, "prompt_source_sha256")
    _validate_optional_digest(axis_source_sha256, "axis_source_sha256")
    _validate_optional_digest(document_source_sha256, "document_source_sha256")
    if source_git_commit is not None and re.fullmatch(r"[0-9a-f]{40}", source_git_commit) is None:
        raise ValueError("source_git_commit must be a full lowercase Git SHA")
    if len({item.configuration_id for item in models}) != len(models):
        raise ValueError("model configurations must be unique")

    prompts = _normalize_prompts(prompt_rows, axis_rows)
    document_sets = _normalize_document_sets(document_set_rows)
    documents_by_keyword = {item.keyword: item for item in document_sets}
    missing = sorted({item.keyword for item in prompts} - set(documents_by_keyword))
    if missing:
        raise ValueError(
            "missing frozen document set for keyword(s): "
            + ", ".join(repr(value) for value in missing[:8])
        )
    undersized = sorted(
        item.keyword
        for item in prompts
        if len(documents_by_keyword[item.keyword].documents) <= top_n
    )
    if undersized:
        raise ValueError(
            "frozen document set must retain at least top_n documents after ablation: "
            + ", ".join(repr(value) for value in sorted(set(undersized))[:8])
        )

    assignments = _build_assignments(
        prompts,
        documents_by_keyword=documents_by_keyword,
        master_seed=master_seed,
    )
    plan_identity = {
        "format_version": FORMAT_VERSION,
        "master_seed": master_seed,
        "top_n": top_n,
        "prompt_source_sha256": prompt_source_sha256,
        "axis_source_sha256": axis_source_sha256,
        "document_source_sha256": document_source_sha256,
        "source_git_commit": source_git_commit,
        "prompt_ids": [item.prompt_id for item in prompts],
        "document_set_ids": [item.candidate_set_id for item in document_sets],
        "assignment_ids": [item.assignment_id for item in assignments],
        "models": [asdict(item) for item in models],
    }
    plan_id = "acl-arr-plan-" + _hash(_canonical(plan_identity))[:24]
    prompt_count = len(prompts)
    model_count = len(models)
    tasks_per_pipeline = prompt_count * model_count * len(CONDITIONS)
    summary = {
        "prompt_count": prompt_count,
        "keyword_count": len({item.keyword for item in prompts}),
        "document_set_count": len(document_sets),
        "condition_assignment_count": len(assignments),
        "condition_count": len(CONDITIONS),
        "model_count": model_count,
        "pipeline_count": len(PIPELINES),
        "tasks_per_model_pipeline": prompt_count * len(CONDITIONS),
        "tasks_per_pipeline": tasks_per_pipeline,
        "primary_task_count": tasks_per_pipeline * len(PIPELINES),
        "planned_judge_task_count": tasks_per_pipeline,
        "planned_total_inference_count": tasks_per_pipeline * 3,
    }
    return AclArrExperimentPlan(
        plan_id=plan_id,
        format_version=FORMAT_VERSION,
        master_seed=master_seed,
        top_n=top_n,
        prompt_source_sha256=prompt_source_sha256,
        axis_source_sha256=axis_source_sha256,
        document_source_sha256=document_source_sha256,
        source_git_commit=source_git_commit,
        prompts=prompts,
        document_sets=document_sets,
        assignments=assignments,
        models=tuple(models),
        summary=summary,
    )


def iter_experiment_tasks(
    plan: AclArrExperimentPlan,
    *,
    model_configuration_id: str | None = None,
    pipeline: str | None = None,
) -> Iterator[ExperimentTask]:
    """Yield stable tasks in a condition-interleaved schedule."""

    if pipeline is not None and pipeline not in PIPELINES:
        raise ValueError(f"unsupported pipeline: {pipeline}")
    selected_models = tuple(
        item
        for item in plan.models
        if model_configuration_id is None
        or item.configuration_id == model_configuration_id
    )
    if model_configuration_id is not None and not selected_models:
        raise ValueError("unknown model configuration ID")
    selected_pipelines = (pipeline,) if pipeline is not None else PIPELINES
    assignment_by_prompt = {item.prompt_id: item for item in plan.assignments}
    scheduled_prompts = sorted(
        plan.prompts,
        key=lambda item: _stable_digest(plan.master_seed, "schedule", item.prompt_id),
    )
    for prompt in scheduled_prompts:
        assignment = assignment_by_prompt[prompt.prompt_id]
        conditions = sorted(
            CONDITIONS,
            key=lambda value: _stable_digest(
                plan.master_seed, "condition-order", prompt.prompt_id, value
            ),
        )
        for model in selected_models:
            for selected_pipeline in selected_pipelines:
                for condition in conditions:
                    input_ids = assignment.document_ids(condition)
                    identity = {
                        "plan_id": plan.plan_id,
                        "prompt_id": prompt.prompt_id,
                        "assignment_id": assignment.assignment_id,
                        "model_configuration_id": model.configuration_id,
                        "pipeline": selected_pipeline,
                        "condition": condition,
                    }
                    yield ExperimentTask(
                        task_id="acl-arr-task-" + _hash(_canonical(identity))[:24],
                        format_version=FORMAT_VERSION,
                        plan_id=plan.plan_id,
                        model_configuration_id=model.configuration_id,
                        pipeline=selected_pipeline,
                        condition=condition,
                        prompt_id=prompt.prompt_id,
                        assignment_id=assignment.assignment_id,
                        output_document_count=min(plan.top_n, len(input_ids)),
                        decoding_seed=_stable_integer(
                            plan.master_seed,
                            "decode",
                            prompt.prompt_id,
                            model.configuration_id,
                            selected_pipeline,
                            condition,
                        ),
                    )


def write_acl_arr_experiment_plan(
    output_directory: str | Path,
    *,
    plan: AclArrExperimentPlan,
) -> AclArrPlanArtifacts:
    """Write an immutable, hash-addressed plan and eight model-pipeline shards."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment plan: {output}")
    output.mkdir(parents=True)
    prompts_path = output / "prompts.jsonl"
    document_sets_path = output / "frozen_document_sets.jsonl"
    assignments_path = output / "condition_assignments.jsonl"
    manifest_path = output / "run_manifest.json"
    report_path = output / "README.md"
    _atomic_jsonl(prompts_path, (asdict(item) for item in plan.prompts))
    _atomic_jsonl(document_sets_path, (asdict(item) for item in plan.document_sets))
    _atomic_jsonl(assignments_path, (asdict(item) for item in plan.assignments))

    task_files: dict[tuple[str, str], Path] = {}
    task_artifacts: dict[str, dict[str, Any]] = {}
    for model in plan.models:
        model_root = output / "tasks" / model.configuration_id
        model_root.mkdir(parents=True)
        for pipeline in PIPELINES:
            path = model_root / f"{pipeline}.jsonl"
            _atomic_jsonl(
                path,
                (
                    asdict(item)
                    for item in iter_experiment_tasks(
                        plan,
                        model_configuration_id=model.configuration_id,
                        pipeline=pipeline,
                    )
                ),
            )
            task_files[(model.configuration_id, pipeline)] = path
            task_artifacts[f"{model.configuration_id}/{pipeline}"] = _file_identity(path)

    manifest = {
        "plan_id": plan.plan_id,
        "format_version": plan.format_version,
        "status": "planned",
        "scientific_result": False,
        "master_seed": plan.master_seed,
        "top_n": plan.top_n,
        "source_git_commit": plan.source_git_commit,
        "sources": {
            "prompts_sha256": plan.prompt_source_sha256,
            "axis_sha256": plan.axis_source_sha256,
            "documents_sha256": plan.document_source_sha256,
        },
        "conditions": list(CONDITIONS),
        "pipelines": list(PIPELINES),
        "ablation_policy": "balanced-single-target-by-document-count-v1",
        "shuffle_policy": "balanced-cyclic-derangement-by-document-count-v1",
        "model_native_web_search": False,
        "models": [
            {**asdict(item), "configuration_id": item.configuration_id}
            for item in plan.models
        ],
        "summary": dict(plan.summary),
        "artifacts": {
            "prompts": _file_identity(prompts_path),
            "frozen_document_sets": _file_identity(document_sets_path),
            "condition_assignments": _file_identity(assignments_path),
            "tasks": task_artifacts,
        },
    }
    _atomic_json(manifest_path, manifest)
    report = "\n".join(
        (
            "# ACL ARR document experiment plan",
            "",
            "> This artifact contains no model inference and no scientific result.",
            "",
            f"- Plan ID: `{plan.plan_id}`",
            f"- Prompts: {plan.summary['prompt_count']}",
            f"- Models: {plan.summary['model_count']}",
            f"- Conditions: {', '.join(CONDITIONS)}",
            f"- Primary inference tasks: {plan.summary['primary_task_count']}",
            f"- Planned judge tasks: {plan.summary['planned_judge_task_count']}",
            "- Search during inference: disabled",
            "- Ablation: one balanced preregistered target per prompt",
            "- Shuffle: deterministic balanced cyclic derangement",
            "- Invalid outputs: fail closed; no source-order fallback",
            "",
        )
    )
    _atomic_text(report_path, report)
    return AclArrPlanArtifacts(
        manifest_path=manifest_path,
        prompts_path=prompts_path,
        document_sets_path=document_sets_path,
        assignments_path=assignments_path,
        report_path=report_path,
        task_files=task_files,
    )


def render_primary_prompt(
    task: ExperimentTask,
    prompt: ExperimentPrompt,
    assignment: ConditionAssignment,
    document_set: FrozenDocumentSet,
) -> str:
    """Render one primary request from references stored in the plan."""

    if task.prompt_id != prompt.prompt_id or task.assignment_id != assignment.assignment_id:
        raise ValueError("task references do not match prompt and condition assignment")
    ids = assignment.document_ids(task.condition)
    documents = _documents_in_order(document_set, ids)
    evidence = "\n\n".join(
        f'<document id="{item.document_id}">\n'
        f"Title: {item.title}\nURL: {item.url}\nText: {item.text}\n</document>"
        for item in documents
    )
    if task.pipeline == "rerank":
        instruction = (
            "Rank the supplied documents by how well they support the exact user request. "
            "Use only the supplied documents. Return one JSON object with exactly one key, "
            f'\"ranked_document_ids\", containing exactly {task.output_document_count} '
            "unique document IDs in best-first order."
        )
    elif task.pipeline == "answer":
        instruction = (
            "Answer the exact user request using only the supplied documents. Cite each "
            "supported claim inline with stable document IDs such as [C003]. Return one "
            "JSON object with exactly the keys \"answer\" and \"cited_document_ids\". "
            "List cited IDs in first-appearance order."
        )
    else:
        raise ValueError(f"unsupported pipeline: {task.pipeline}")
    return (
        f"{instruction}\n\n"
        f"USER REQUEST:\n{prompt.question}\n\n"
        f"SUPPLIED DOCUMENTS:\n{evidence}"
    )


def validate_rerank_output(
    raw_output: str,
    *,
    allowed_document_ids: Sequence[str],
    output_count: int,
) -> dict[str, Any]:
    """Validate a strict JSON document ranking without repair."""

    value = _json_object(raw_output)
    if set(value) != {"ranked_document_ids"}:
        raise ValueError("rerank output must contain only ranked_document_ids")
    ranked = value["ranked_document_ids"]
    if not isinstance(ranked, list) or any(not isinstance(item, str) for item in ranked):
        raise ValueError("ranked_document_ids must be a string list")
    if len(ranked) != output_count:
        raise ValueError(f"expected {output_count} ranked document IDs")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked document IDs contain a duplicate")
    unknown = sorted(set(ranked) - set(allowed_document_ids))
    if unknown:
        raise ValueError("ranking contains unknown document IDs: " + ", ".join(unknown))
    return {"ranked_document_ids": ranked}


def validate_answer_output(
    raw_output: str,
    *,
    allowed_document_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate a cited answer and its deterministic citation index."""

    value = _json_object(raw_output)
    if set(value) != {"answer", "cited_document_ids"}:
        raise ValueError("answer output must contain only answer and cited_document_ids")
    answer = value["answer"]
    cited = value["cited_document_ids"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(cited, list) or any(not isinstance(item, str) for item in cited):
        raise ValueError("cited_document_ids must be a string list")
    parsed = _ordered_unique(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", answer))
    if cited != parsed:
        raise ValueError("declared citations do not match inline citations")
    if not cited:
        raise ValueError("answer must cite at least one supplied document")
    unknown = sorted(set(cited) - set(allowed_document_ids))
    if unknown:
        raise ValueError("answer cites unknown document IDs: " + ", ".join(unknown))
    return {"answer": answer, "cited_document_ids": cited}


def validate_judge_output(
    raw_output: str,
    *,
    allowed_document_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate the blinded realized-use judgment."""

    required = {
        "answer_quality",
        "evidence_coverage",
        "citation_correctness",
        "unsupported_claim_count",
        "realized_document_ranking",
    }
    value = _json_object(raw_output)
    if set(value) != required:
        raise ValueError("judge output has incorrect keys")
    for key in ("answer_quality", "evidence_coverage", "citation_correctness"):
        score = value[key]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{key} must be an integer from 1 to 5")
    unsupported = value["unsupported_claim_count"]
    if isinstance(unsupported, bool) or not isinstance(unsupported, int) or unsupported < 0:
        raise ValueError("unsupported_claim_count must be a non-negative integer")
    ranking = value["realized_document_ranking"]
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("realized_document_ranking must be a non-empty list")
    document_ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for item in ranking:
        if not isinstance(item, dict) or set(item) != {"document_id", "use_score"}:
            raise ValueError("each realized-use row needs document_id and use_score")
        document_id = item["document_id"]
        score = item["use_score"]
        if not isinstance(document_id, str):
            raise ValueError("realized-use document_id must be a string")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
            raise ValueError("realized-use score must be an integer from 0 to 5")
        document_ids.append(document_id)
        normalized.append({"document_id": document_id, "use_score": score})
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("realized document ranking contains a duplicate")
    unknown = sorted(set(document_ids) - set(allowed_document_ids))
    if unknown:
        raise ValueError("judge ranked unknown document IDs: " + ", ".join(unknown))
    return {
        "answer_quality": value["answer_quality"],
        "evidence_coverage": value["evidence_coverage"],
        "citation_correctness": value["citation_correctness"],
        "unsupported_claim_count": unsupported,
        "realized_document_ranking": normalized,
    }


def build_blinded_judge_tasks(
    answer_outcomes: Sequence[Mapping[str, Any]],
    *,
    plan: AclArrExperimentPlan,
    judge_model_id: str,
    judge_model_revision: str,
    master_seed: int,
    allow_fake: bool = False,
) -> BlindedJudgePlan:
    """Create judge inputs with generator and condition metadata held separately."""

    if not answer_outcomes:
        raise ValueError("answer outcomes must not be empty")
    if not judge_model_id.strip():
        raise ValueError("judge model ID must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}", judge_model_revision) is None:
        raise ValueError("judge model revision must be an immutable 40-character SHA")
    prompt_by_id = {item.prompt_id: item for item in plan.prompts}
    assignment_by_id = {item.assignment_id: item for item in plan.assignments}
    document_set_by_id = {item.candidate_set_id: item for item in plan.document_sets}
    tasks: list[BlindedJudgeTask] = []
    mappings: list[JudgeTaskMapping] = []
    source_ids: set[str] = set()
    for index, row in enumerate(answer_outcomes, 1):
        source_task_id = _required_string(row, "task_id", index)
        if source_task_id in source_ids:
            raise ValueError("answer outcomes contain duplicate task IDs")
        source_ids.add(source_task_id)
        if row.get("pipeline") != "answer":
            raise ValueError("judge input contains a non-answer outcome")
        if row.get("fake_backend") is True and not allow_fake:
            raise ValueError("fake answer outcomes cannot enter a scientific judge plan")
        prompt_id = _required_string(row, "prompt_id", index)
        assignment_id = _required_string(row, "assignment_id", index)
        model_configuration_id = _required_string(
            row, "model_configuration_id", index
        )
        model_id = _required_string(row, "model_id", index)
        condition = _required_string(row, "condition", index)
        candidate_set_id = _required_string(row, "candidate_set_id", index)
        if condition not in CONDITIONS:
            raise ValueError("answer outcome has an unknown condition")
        try:
            prompt = prompt_by_id[prompt_id]
            assignment = assignment_by_id[assignment_id]
            document_set = document_set_by_id[candidate_set_id]
        except KeyError as exc:
            raise ValueError("answer outcome does not match the frozen plan") from exc
        parsed = row.get("parsed_output")
        if not isinstance(parsed, dict):
            raise ValueError("answer outcome lacks parsed_output")
        input_ids = assignment.document_ids(condition)
        answer = validate_answer_output(
            _canonical(parsed), allowed_document_ids=input_ids
        )
        judge_ids = _independent_judge_order(
            input_ids, master_seed=master_seed, source_task_id=source_task_id
        )
        blind_case_id = "blind-case-" + _stable_digest(
            master_seed, "blind-case", source_task_id
        )[:24]
        judge_task_id = "acl-arr-judge-task-" + _hash(
            _canonical(
                {
                    "blind_case_id": blind_case_id,
                    "judge_model_id": judge_model_id,
                    "judge_model_revision": judge_model_revision,
                    "judge_document_ids": judge_ids,
                    "answer_sha256": _hash(str(answer["answer"])),
                }
            )
        )[:24]
        tasks.append(
            BlindedJudgeTask(
                judge_task_id=judge_task_id,
                format_version=JUDGE_FORMAT_VERSION,
                blind_case_id=blind_case_id,
                judge_model_id=judge_model_id,
                judge_model_revision=judge_model_revision,
                prompt_text=prompt.question,
                candidate_set_id=document_set.candidate_set_id,
                judge_document_ids=judge_ids,
                answer=str(answer["answer"]),
                cited_document_ids=tuple(answer["cited_document_ids"]),
            )
        )
        mappings.append(
            JudgeTaskMapping(
                blind_case_id=blind_case_id,
                judge_task_id=judge_task_id,
                source_generation_task_id=source_task_id,
                prompt_id=prompt_id,
                assignment_id=assignment_id,
                generator_model_configuration_id=model_configuration_id,
                generator_model_id=model_id,
                condition=condition,
                ablation_target_id=assignment.ablation_target_id,
            )
        )
    judge_identity = {
        "format_version": JUDGE_FORMAT_VERSION,
        "source_plan_id": plan.plan_id,
        "master_seed": master_seed,
        "judge_model_id": judge_model_id,
        "judge_model_revision": judge_model_revision,
        "judge_task_ids": [item.judge_task_id for item in tasks],
    }
    return BlindedJudgePlan(
        judge_plan_id="acl-arr-judge-plan-" + _hash(_canonical(judge_identity))[:24],
        format_version=JUDGE_FORMAT_VERSION,
        master_seed=master_seed,
        judge_model_id=judge_model_id,
        judge_model_revision=judge_model_revision,
        document_sets=plan.document_sets,
        tasks=tuple(tasks),
        mappings=tuple(mappings),
        summary={
            "judge_task_count": len(tasks),
            "source_answer_count": len(answer_outcomes),
        },
    )


def render_judge_prompt(
    task: BlindedJudgeTask, document_set: FrozenDocumentSet
) -> str:
    """Render a judge request without generator or condition labels."""

    documents = _documents_in_order(document_set, task.judge_document_ids)
    evidence = "\n\n".join(
        f'<document id="{item.document_id}">\n'
        f"Title: {item.title}\nURL: {item.url}\nText: {item.text}\n</document>"
        for item in documents
    )
    return (
        "Evaluate the answer only against the supplied documents. Do not infer "
        "anything about the system that generated it. Score answer quality, evidence "
        "coverage, and citation correctness from 1 to 5. Count unsupported claims. "
        "Rank only documents that materially supported the answer, best first, and "
        "assign each an integer use_score from 0 to 5. Return one JSON object with "
        "exactly these keys: answer_quality, evidence_coverage, citation_correctness, "
        "unsupported_claim_count, realized_document_ranking.\n\n"
        f"USER REQUEST:\n{task.prompt_text}\n\n"
        f"SUPPLIED DOCUMENTS:\n{evidence}\n\n"
        f"ANSWER TO EVALUATE:\n{task.answer}"
    )


def write_blinded_judge_plan(
    output_directory: str | Path,
    *,
    judge_plan: BlindedJudgePlan,
) -> BlindedJudgeArtifacts:
    """Write public judge tasks and private experimental mappings separately."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite judge plan: {output}")
    output.mkdir(parents=True)
    tasks_path = output / "judge_tasks.jsonl"
    mapping_path = output / "private_judge_mapping.jsonl"
    document_sets_path = output / "frozen_document_sets.jsonl"
    manifest_path = output / "judge_manifest.json"
    report_path = output / "README.md"
    _atomic_jsonl(tasks_path, (asdict(item) for item in judge_plan.tasks))
    _atomic_jsonl(mapping_path, (asdict(item) for item in judge_plan.mappings))
    _atomic_jsonl(
        document_sets_path, (asdict(item) for item in judge_plan.document_sets)
    )
    _atomic_json(
        manifest_path,
        {
            "judge_plan_id": judge_plan.judge_plan_id,
            "format_version": judge_plan.format_version,
            "status": "planned",
            "scientific_result": False,
            "master_seed": judge_plan.master_seed,
            "judge_model_id": judge_plan.judge_model_id,
            "judge_model_revision": judge_plan.judge_model_revision,
            "blinding": "generator-and-condition-labels-excluded-from-judge-input-v1",
            "judge_document_order": "independent-cyclic-derangement-v1",
            "summary": dict(judge_plan.summary),
            "artifacts": {
                "judge_tasks": _file_identity(tasks_path),
                "private_judge_mapping": _file_identity(mapping_path),
                "frozen_document_sets": _file_identity(document_sets_path),
            },
        },
    )
    _atomic_text(
        report_path,
        "\n".join(
            (
                "# Blinded ACL ARR realized-use judge plan",
                "",
                "> Planning produced no judge inference and no scientific result.",
                "",
                f"- Judge plan ID: `{judge_plan.judge_plan_id}`",
                f"- Judge tasks: {judge_plan.summary['judge_task_count']}",
                "- Generator and condition labels are absent from rendered judge inputs.",
                "- The private mapping must not be supplied to the judge server.",
                "",
            )
        ),
    )
    return BlindedJudgeArtifacts(
        manifest_path=manifest_path,
        tasks_path=tasks_path,
        private_mapping_path=mapping_path,
        report_path=report_path,
    )


def load_plan_from_artifacts(manifest_path: str | Path) -> AclArrExperimentPlan:
    """Reconstruct and verify a plan from its written artifacts."""

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported ACL ARR plan format")
    artifact_root = manifest_file.parent
    artifact_map = manifest.get("artifacts")
    if not isinstance(artifact_map, dict):
        raise ValueError("plan manifest lacks artifacts")
    prompt_rows = _verified_artifact_rows(artifact_root, artifact_map, "prompts")
    document_rows = _verified_artifact_rows(
        artifact_root, artifact_map, "frozen_document_sets"
    )
    assignment_rows = _verified_artifact_rows(
        artifact_root, artifact_map, "condition_assignments"
    )
    models = tuple(
        ModelConfiguration(
            **{key: value for key, value in row.items() if key != "configuration_id"}
        )
        for row in manifest["models"]
    )
    prompts = tuple(ExperimentPrompt(**row) for row in prompt_rows)
    document_sets = tuple(
        FrozenDocumentSet(
            candidate_set_id=row["candidate_set_id"],
            keyword=row["keyword"],
            search_query=row["search_query"],
            search_engine=row["search_engine"],
            search_snapshot_sha256=row["search_snapshot_sha256"],
            documents=tuple(FrozenDocument(**item) for item in row["documents"]),
        )
        for row in document_rows
    )
    assignments = tuple(
        ConditionAssignment(
            **{
                **row,
                "natural_document_ids": tuple(row["natural_document_ids"]),
                "ablated_document_ids": tuple(row["ablated_document_ids"]),
                "shuffled_document_ids": tuple(row["shuffled_document_ids"]),
            }
        )
        for row in assignment_rows
    )
    return AclArrExperimentPlan(
        plan_id=manifest["plan_id"],
        format_version=manifest["format_version"],
        master_seed=int(manifest["master_seed"]),
        top_n=int(manifest["top_n"]),
        prompt_source_sha256=manifest["sources"].get("prompts_sha256"),
        axis_source_sha256=manifest["sources"].get("axis_sha256"),
        document_source_sha256=manifest["sources"].get("documents_sha256"),
        source_git_commit=manifest.get("source_git_commit"),
        prompts=prompts,
        document_sets=document_sets,
        assignments=assignments,
        models=models,
        summary=manifest["summary"],
    )


def _normalize_prompts(
    prompt_rows: Sequence[Mapping[str, Any]],
    axis_rows: Sequence[Mapping[str, Any]],
) -> tuple[ExperimentPrompt, ...]:
    axis_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(axis_rows, 1):
        candidate_id = _required_string(row, "candidate_id", index)
        if candidate_id in axis_by_id:
            raise ValueError("final axis map contains duplicate candidate IDs")
        axis_by_id[candidate_id] = row
    prompts: list[ExperimentPrompt] = []
    seen: set[str] = set()
    for index, row in enumerate(prompt_rows, 1):
        prompt_id = _required_string(row, "candidate_id", index)
        if prompt_id in seen:
            raise ValueError("prompt population contains duplicate candidate IDs")
        seen.add(prompt_id)
        if prompt_id not in axis_by_id:
            raise ValueError(f"prompt is missing from final axis map: {prompt_id}")
        axis = axis_by_id[prompt_id]
        question = _required_string(row, "question", index)
        declared_hash = _required_string(row, "question_sha256", index)
        if declared_hash != _hash(question):
            raise ValueError(f"prompt question hash does not match text: {prompt_id}")
        if str(axis.get("text_sha256", "")) != declared_hash:
            raise ValueError(f"axis question hash does not match prompt: {prompt_id}")
        assigned = row.get("target_normalized_axis_1")
        if isinstance(assigned, bool) or not isinstance(assigned, (int, float)):
            raise ValueError(f"prompt lacks assigned readiness coordinate: {prompt_id}")
        assigned_float = float(assigned)
        percentile = float(axis["axis_1_percentile_0_1"])
        if not 0 <= assigned_float <= 1 or not 0 <= percentile <= 1:
            raise ValueError("readiness coordinates must be within [0, 1]")
        prompts.append(
            ExperimentPrompt(
                prompt_id=prompt_id,
                keyword_id=_required_string(row, "keyword_id", index),
                keyword=_required_string(row, "keyword", index),
                target_id=_required_string(row, "target_id", index),
                target_index=int(row["target_index"]),
                question=question,
                question_sha256=declared_hash,
                assigned_readiness_0_1=assigned_float,
                consensus_axis_1_z=float(axis["consensus_axis_1_z"]),
                axis_1_rank=int(axis["axis_1_rank"]),
                axis_1_percentile_0_1=percentile,
                qwen_axis_1_z=float(axis["reference_axis_1_z"]),
                mistral_aligned_axis_1_z=float(
                    axis["candidate_aligned_axis_1_z"]
                ),
            )
        )
    if set(axis_by_id) != seen:
        raise ValueError("prompt population and final axis map candidate IDs differ")
    return tuple(sorted(prompts, key=lambda item: item.prompt_id))


def _normalize_document_sets(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[FrozenDocumentSet, ...]:
    output: list[FrozenDocumentSet] = []
    keywords: set[str] = set()
    set_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        keyword = _required_string(row, "keyword", index)
        if keyword in keywords:
            raise ValueError(f"multiple frozen document sets for keyword: {keyword}")
        keywords.add(keyword)
        raw_documents = row.get("documents", row.get("candidates"))
        if not isinstance(raw_documents, list) or len(raw_documents) < 2:
            raise ValueError(f"document set {keyword!r} must contain at least two documents")
        documents: list[FrozenDocument] = []
        document_ids: set[str] = set()
        urls: set[str] = set()
        positions: set[int] = set()
        for raw in raw_documents:
            if not isinstance(raw, dict):
                raise ValueError(f"document set {keyword!r} contains a non-object")
            document_id = str(raw.get("document_id", raw.get("candidate_id", ""))).strip()
            position_value = raw.get("natural_position", raw.get("source_position"))
            if position_value is None:
                raise ValueError(
                    f"document {document_id!r} for keyword {keyword!r} has no position"
                )
            text = str(
                raw.get("text", raw.get("content", raw.get("snippet", ""))) or ""
            ).strip()
            url = str(raw.get("url", "") or "").strip()
            if not document_id or not text or not url:
                raise ValueError(f"document set {keyword!r} contains incomplete evidence")
            try:
                position = int(position_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"document set {keyword!r} has an invalid position") from exc
            if position <= 0:
                raise ValueError("document natural positions must be positive")
            if document_id in document_ids or url in urls or position in positions:
                raise ValueError(f"document set {keyword!r} contains duplicate identity")
            document_ids.add(document_id)
            urls.add(url)
            positions.add(position)
            text_hash = _hash(text)
            declared_hash = raw.get("text_sha256", raw.get("content_sha256"))
            if declared_hash is not None and str(declared_hash) != text_hash:
                raise ValueError(f"document text hash mismatch: {document_id}")
            documents.append(
                FrozenDocument(
                    document_id=document_id,
                    natural_position=position,
                    title=str(raw.get("title", "") or ""),
                    url=url,
                    text=text,
                    text_sha256=text_hash,
                )
            )
        documents.sort(key=lambda item: item.natural_position)
        set_identity = {
            "keyword": keyword,
            "search_query": str(row.get("search_query", keyword)),
            "documents": [asdict(item) for item in documents],
        }
        candidate_set_id = str(row.get("candidate_set_id", "")).strip()
        if not candidate_set_id:
            candidate_set_id = "document-set-" + _hash(_canonical(set_identity))[:20]
        if candidate_set_id in set_ids:
            raise ValueError("frozen document set IDs must be unique")
        set_ids.add(candidate_set_id)
        output.append(
            FrozenDocumentSet(
                candidate_set_id=candidate_set_id,
                keyword=keyword,
                search_query=str(row.get("search_query", keyword)),
                search_engine=str(row.get("search_engine", "unknown")),
                search_snapshot_sha256=str(
                    row.get("search_snapshot_sha256", "unavailable")
                ),
                documents=tuple(documents),
            )
        )
    return tuple(sorted(output, key=lambda item: item.keyword))


def _build_assignments(
    prompts: Sequence[ExperimentPrompt],
    *,
    documents_by_keyword: Mapping[str, FrozenDocumentSet],
    master_seed: int,
) -> tuple[ConditionAssignment, ...]:
    groups: dict[int, list[ExperimentPrompt]] = {}
    for prompt in prompts:
        count = len(documents_by_keyword[prompt.keyword].documents)
        groups.setdefault(count, []).append(prompt)
    ablation_index: dict[str, int] = {}
    shuffle_offset: dict[str, int] = {}
    for count, group in groups.items():
        ablation_order = sorted(
            group,
            key=lambda item: _stable_digest(
                master_seed, "ablation-order", item.prompt_id
            ),
        )
        for index, prompt in enumerate(ablation_order):
            ablation_index[prompt.prompt_id] = index % count
        shuffle_order = sorted(
            group,
            key=lambda item: _stable_digest(
                master_seed, "shuffle-order", item.prompt_id
            ),
        )
        for index, prompt in enumerate(shuffle_order):
            shuffle_offset[prompt.prompt_id] = 1 + (index % (count - 1))

    assignments: list[ConditionAssignment] = []
    for prompt in prompts:
        document_set = documents_by_keyword[prompt.keyword]
        natural = tuple(item.document_id for item in document_set.documents)
        target = natural[ablation_index[prompt.prompt_id]]
        ablated = tuple(item for item in natural if item != target)
        offset = shuffle_offset[prompt.prompt_id]
        shuffled = natural[offset:] + natural[:offset]
        if set(shuffled) != set(natural) or any(
            left == right for left, right in zip(natural, shuffled)
        ):
            raise AssertionError("shuffle policy failed to create a derangement")
        permutation_id = "permutation-" + _hash("\n".join(shuffled))[:20]
        identity = {
            "prompt_id": prompt.prompt_id,
            "candidate_set_id": document_set.candidate_set_id,
            "natural": natural,
            "ablation_target_id": target,
            "ablated": ablated,
            "shuffled": shuffled,
            "master_seed": master_seed,
        }
        assignments.append(
            ConditionAssignment(
                assignment_id="condition-assignment-"
                + _hash(_canonical(identity))[:24],
                prompt_id=prompt.prompt_id,
                candidate_set_id=document_set.candidate_set_id,
                natural_document_ids=natural,
                ablation_target_id=target,
                ablated_document_ids=ablated,
                shuffled_document_ids=shuffled,
                shuffle_policy="balanced-cyclic-derangement-by-document-count-v1",
                permutation_id=permutation_id,
            )
        )
    return tuple(sorted(assignments, key=lambda item: item.prompt_id))


def _independent_judge_order(
    document_ids: Sequence[str], *, master_seed: int, source_task_id: str
) -> tuple[str, ...]:
    values = tuple(document_ids)
    if len(values) < 2:
        return values
    offset = 1 + _stable_integer(
        master_seed, "judge-order", source_task_id
    ) % (len(values) - 1)
    return values[offset:] + values[:offset]


def _documents_in_order(
    document_set: FrozenDocumentSet, ids: Sequence[str]
) -> tuple[FrozenDocument, ...]:
    by_id = {item.document_id: item for item in document_set.documents}
    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        raise ValueError("condition references unknown documents: " + ", ".join(unknown))
    if len(set(ids)) != len(ids):
        raise ValueError("condition contains duplicate document IDs")
    return tuple(by_id[value] for value in ids)


def _verified_artifact_rows(
    artifact_root: Path,
    artifacts: Mapping[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    identity = artifacts.get(key)
    if not isinstance(identity, dict):
        raise ValueError(f"plan manifest lacks {key} artifact")
    path_value = identity.get("path")
    if not isinstance(path_value, str):
        raise ValueError(f"plan manifest has invalid {key} path")
    path = Path(path_value)
    if not path.is_absolute():
        path = artifact_root / path
    if _sha256_file(path) != identity.get("sha256"):
        raise ValueError(f"{key} artifact does not match manifest SHA-256")
    return list(_read_jsonl(path))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def _required_string(
    row: Mapping[str, Any], field: str, line_number: int
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {field}")
    return value


def _json_object(raw_output: str) -> dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("model output is empty")
    try:
        value = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("model output is not one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return value


def _ordered_unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _validate_optional_digest(value: str | None, name: str) -> None:
    if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _stable_digest(master_seed: int, *parts: str) -> str:
    return _hash(":".join((str(master_seed), *parts)))


def _stable_integer(master_seed: int, *parts: str) -> int:
    return int(_stable_digest(master_seed, *parts)[:8], 16)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


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


__all__ = [
    "AclArrExperimentPlan",
    "AclArrPlanArtifacts",
    "BlindedJudgeArtifacts",
    "BlindedJudgePlan",
    "BlindedJudgeTask",
    "CONDITIONS",
    "ConditionAssignment",
    "ExperimentPrompt",
    "ExperimentTask",
    "FORMAT_VERSION",
    "FrozenDocument",
    "FrozenDocumentSet",
    "JUDGE_FORMAT_VERSION",
    "JudgeTaskMapping",
    "ModelConfiguration",
    "PIPELINES",
    "build_acl_arr_experiment_plan",
    "build_blinded_judge_tasks",
    "iter_experiment_tasks",
    "load_plan_from_artifacts",
    "render_judge_prompt",
    "render_primary_prompt",
    "validate_answer_output",
    "validate_judge_output",
    "validate_rerank_output",
    "write_acl_arr_experiment_plan",
    "write_blinded_judge_plan",
]
