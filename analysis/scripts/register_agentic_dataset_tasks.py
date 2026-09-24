#!/usr/bin/env python3
"""Register immutable generator or Nemotron tasks in the final dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    iter_sealed_rows,
)
from analysis.interpretability.pipeline.agentic_generation_tasks import (
    load_calibration_prompts,
)
from analysis.interpretability.pipeline.agentic_judging import AgenticJudgeTask
from analysis.interpretability.pipeline.agentic_task_ledger import identity_fingerprint
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.scripts.run_acl_arr_vllm import (
    _prepare_agentic_judge,
    agentic_judge_claim_identity,
)
from analysis.scripts.run_agentic_search_integration_smoke import (
    ENGINES,
    SmokeInputs,
    _generator_claim_identity,
    describe_queue,
)

KEYWORD_FORMAT = "geodml-keyword-priority-v1"
MODEL_SLUGS = frozenset({"qwen38", "llama4"})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected an object at {path}:{number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _keyword_priority(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format_version") != KEYWORD_FORMAT:
        raise ValueError("unsupported keyword priority format")
    rows = value.get("keywords")
    if not isinstance(rows, list) or not rows:
        raise ValueError("keyword priority requires a non-empty keywords list")
    by_source: dict[str, dict[str, Any]] = {}
    ranks: set[int] = set()
    keyword_ids: set[str] = set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise TypeError(f"keyword priority row {number} is not an object")
        source = row.get("source_keyword")
        keyword_id = row.get("keyword_id")
        rank = row.get("priority_rank")
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(keyword_id, str)
            or not keyword_id.strip()
            or type(rank) is not int
            or rank < 0
        ):
            raise ValueError(f"invalid keyword priority row {number}")
        if source in by_source or keyword_id in keyword_ids or rank in ranks:
            raise ValueError("keyword sources, IDs, and priority ranks must be unique")
        by_source[source] = dict(row)
        keyword_ids.add(keyword_id)
        ranks.add(rank)
    return by_source


def _existing(root: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    prompts = {row["prompt_id"]: row for row in iter_sealed_rows(root, "prompts")}
    memberships = {
        row["prompt_id"]: row for row in iter_sealed_rows(root, "keyword_memberships")
    }
    tasks_by_fingerprint: dict[str, dict[str, Any]] = {}
    tasks_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_sealed_rows(root, "task_definitions"):
        identity = ClaimIdentity(**row["claim_identity"])
        fingerprint = identity_fingerprint(identity)
        key = (row["model"], row["task_id"])
        if fingerprint in tasks_by_fingerprint or key in tasks_by_key:
            raise ValueError("existing task registry contains a duplicate identity")
        tasks_by_fingerprint[fingerprint] = row
        tasks_by_key[key] = row
    return prompts, memberships, tasks_by_fingerprint, tasks_by_key


def _append_if_missing(
    writer: FinalDatasetWriter,
    *,
    table: str,
    key: str,
    row: Mapping[str, Any],
    existing: dict[str, dict[str, Any]],
) -> bool:
    prior = existing.get(key)
    normalized = json.loads(_canonical(row))
    if prior is not None:
        if prior != normalized:
            raise ValueError(f"existing {table} row conflicts for {key}")
        return False
    writer.append(
        table,
        normalized,
        transaction_id=f"registration-{key}",
        record_id=f"{table[:-1]}-{hashlib.sha256(key.encode()).hexdigest()}",
    )
    existing[key] = normalized
    return True


def register_generator_tasks(
    *,
    dataset_root: Path,
    model_slug: str,
    inputs: SmokeInputs,
    keyword_priority_path: Path,
    writer_id: str,
) -> dict[str, Any]:
    """Register one generator model's frozen prompts, memberships, and cells."""

    if model_slug not in MODEL_SLUGS:
        raise ValueError("generator model slug must be qwen38 or llama4")
    if inputs.prompts_jsonl is None or inputs.selection_records_jsonl is None:
        raise ValueError("generator registration requires frozen prompt inputs")
    if not inputs.production_conditions:
        raise ValueError("generator registration requires production conditions")
    priority = _keyword_priority(keyword_priority_path)
    prompts = load_calibration_prompts(
        inputs.prompts_jsonl,
        inputs.selection_records_jsonl,
        prompt_count=inputs.prompt_count,
        seed=inputs.prompt_selection_seed,
    )
    missing_keywords = sorted({row.keyword for row in prompts} - set(priority))
    if missing_keywords:
        raise ValueError(
            "keyword priority does not cover frozen prompt keywords: "
            + ", ".join(missing_keywords)
        )
    description = describe_queue(inputs)
    cells = description["cells"]
    config = description["config"]
    config_sha256 = hashlib.sha256(_canonical(config)).hexdigest()
    method_sha256 = _sha256(
        Path(__file__).resolve().parents[1]
        / "interpretability/pipeline/agentic_search.py"
    )
    prompts_by_id = {row.prompt_id: row for row in prompts}
    existing_prompts, existing_memberships, tasks_by_fingerprint, tasks_by_key = (
        _existing(dataset_root)
    )
    writer = FinalDatasetWriter(dataset_root, writer_id=writer_id)
    counts = {"prompts": 0, "keyword_memberships": 0, "task_definitions": 0}
    for prompt in prompts:
        prompt_row = {
            "prompt_id": prompt.prompt_id,
            "prompt_text": prompt.prompt,
            "prompt_sha256": prompt.question_sha256,
            "axis_bin": prompt.axis_bin,
            "source_keyword": prompt.keyword,
            "source": {
                "prompts_jsonl_sha256": _sha256(inputs.prompts_jsonl),
                "selection_records_jsonl_sha256": _sha256(
                    inputs.selection_records_jsonl
                ),
                "prompt_selection_seed": inputs.prompt_selection_seed,
            },
        }
        counts["prompts"] += _append_if_missing(
            writer,
            table="prompts",
            key=prompt.prompt_id,
            row=prompt_row,
            existing=existing_prompts,
        )
        keyword = priority[prompt.keyword]
        membership = {
            "prompt_id": prompt.prompt_id,
            "source_keyword": prompt.keyword,
            "keyword_ids": [keyword["keyword_id"]],
            "primary_keyword_id": keyword["keyword_id"],
            "primary_priority_rank": keyword["priority_rank"],
            "membership_policy": "frozen-source-keyword-v1",
        }
        counts["keyword_memberships"] += _append_if_missing(
            writer,
            table="keyword_memberships",
            key=prompt.prompt_id,
            row=membership,
            existing=existing_memberships,
        )
    for cell in cells:
        if cell.prompt_id is None:
            raise ValueError("production generator cell lacks a prompt ID")
        identity = _generator_claim_identity(
            cell,
            inputs,
            config,
            description["legacy_prompt"],
            description["target_urls"],
            method_source_sha256=method_sha256,
        )
        fingerprint = identity_fingerprint(identity)
        task = {
            "task_id": cell.cell_id,
            "prompt_id": cell.prompt_id,
            "model": model_slug,
            "stage": "generation",
            "method": cell.core["method"],
            "engine": cell.engine,
            "condition": cell.condition.value,
            "claim_identity": asdict(identity),
            "identity_fingerprint": fingerprint,
            "dependency_fingerprints": [],
            "runnable_task": {"cell_id": cell.cell_id},
            "configuration_sha256": config_sha256,
            "prompt_sha256": prompts_by_id[cell.prompt_id].question_sha256,
        }
        prior = tasks_by_fingerprint.get(fingerprint)
        key = (model_slug, cell.cell_id)
        keyed = tasks_by_key.get(key)
        if prior is not None or keyed is not None:
            if prior != task or keyed != task:
                raise ValueError(f"existing task definition conflicts for {key}")
            continue
        writer.append(
            "task_definitions",
            task,
            transaction_id=f"registration-{fingerprint}",
            record_id=f"task-{fingerprint}",
        )
        tasks_by_fingerprint[fingerprint] = task
        tasks_by_key[key] = task
        counts["task_definitions"] += 1
    manifests = writer.seal()
    return {
        "format_version": "geodml-agentic-task-registration-v1",
        "model": model_slug,
        "model_id": inputs.model_id,
        "model_revision": inputs.model_revision,
        "configuration_sha256": config_sha256,
        "selected_prompt_count": len(prompts),
        "task_count": len(cells),
        "appended": counts,
        "sealed_manifests": manifests,
    }


def register_nemotron_tasks(
    *,
    dataset_root: Path,
    manifest_path: Path,
    tasks_path: Path,
    mapping_path: Path,
    writer_id: str,
    disable_thinking: bool = False,
    maximum_attempts: int = 3,
    request_timeout: float = 600.0,
    max_output_tokens: int = 512,
) -> dict[str, Any]:
    """Register bulk Nemotron tasks linked to exact generator fingerprints."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    model = manifest.get("bulk_model")
    if not isinstance(artifacts, dict) or not isinstance(model, dict):
        raise TypeError("judge manifest lacks artifacts or bulk model")
    for name, path in (("bulk_tasks", tasks_path), ("private_mapping", mapping_path)):
        identity = artifacts.get(name)
        if not isinstance(identity, dict) or identity.get("sha256") != _sha256(path):
            raise ValueError(f"judge {name} does not match its manifest")
    tasks = [AgenticJudgeTask.from_dict(row) for row in _read_jsonl(tasks_path)]
    mappings = _read_jsonl(mapping_path)
    mapping_by_task: dict[str, dict[str, Any]] = {}
    for row in mappings:
        task_id = row.get("judge_task_id")
        if not isinstance(task_id, str) or task_id in mapping_by_task:
            raise ValueError("private mapping has a missing or duplicate judge task ID")
        mapping_by_task[task_id] = row
    if {task.judge_task_id for task in tasks} != set(mapping_by_task):
        raise ValueError("bulk judge tasks and private mappings differ")
    _, memberships, tasks_by_fingerprint, tasks_by_key = _existing(dataset_root)
    writer = FinalDatasetWriter(dataset_root, writer_id=writer_id)
    appended = 0
    for task in tasks:
        mapping = mapping_by_task[task.judge_task_id]
        prompt_id = mapping.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id not in memberships:
            raise ValueError("judge task prompt lacks a registered keyword membership")
        source = [
            row
            for row in tasks_by_fingerprint.values()
            if row["task_id"] == mapping.get("source_cell_id")
            and row["claim_identity"]["model_id"] == mapping.get("generator_model_id")
            and row.get("stage") == "generation"
        ]
        if len(source) != 1:
            raise ValueError(
                f"judge task {task.judge_task_id} does not resolve one generator task"
            )
        source_identity = ClaimIdentity(**source[0]["claim_identity"])
        prepared = _prepare_agentic_judge(task, max_tokens=max_output_tokens)
        identity = agentic_judge_claim_identity(
            prepared,
            judge_role="bulk",
            disable_thinking=disable_thinking,
            fake_backend=False,
            pilot_only=False,
            maximum_attempts=maximum_attempts,
            request_timeout=request_timeout,
            model_id=model["model_id"],
            model_revision=model["model_revision"],
        )
        fingerprint = identity_fingerprint(identity)
        row = {
            "task_id": task.judge_task_id,
            "prompt_id": prompt_id,
            "model": "nemotron",
            "stage": "bulk_judging",
            "method": mapping.get("method"),
            "engine": mapping.get("engine"),
            "condition": mapping.get("condition"),
            "claim_identity": asdict(identity),
            "identity_fingerprint": fingerprint,
            "dependency_fingerprints": [identity_fingerprint(source_identity)],
            "runnable_task": task.to_dict(),
            "judge_plan_id": manifest.get("judge_plan_id"),
            "source_cell_id": mapping.get("source_cell_id"),
        }
        prior = tasks_by_fingerprint.get(fingerprint)
        key = ("nemotron", task.judge_task_id)
        keyed = tasks_by_key.get(key)
        if prior is not None or keyed is not None:
            if prior != row or keyed != row:
                raise ValueError(f"existing task definition conflicts for {key}")
            continue
        writer.append(
            "task_definitions",
            row,
            transaction_id=f"registration-{fingerprint}",
            record_id=f"task-{fingerprint}",
        )
        tasks_by_fingerprint[fingerprint] = row
        tasks_by_key[key] = row
        appended += 1
    manifests = writer.seal()
    return {
        "format_version": "geodml-agentic-task-registration-v1",
        "model": "nemotron",
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "task_count": len(tasks),
        "appended": {"task_definitions": appended},
        "sealed_manifests": manifests,
    }


def _binding(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in ENGINES or not path:
        raise argparse.ArgumentTypeError("search snapshot must be ENGINE=PATH")
    return name, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generator = subparsers.add_parser("generator")
    generator.add_argument("--dataset-root", type=Path, required=True)
    generator.add_argument("--writer-id", required=True)
    generator.add_argument("--model-slug", choices=sorted(MODEL_SLUGS), required=True)
    generator.add_argument("--model-id", required=True)
    generator.add_argument("--model-revision", required=True)
    generator.add_argument("--cross-encoder-snapshot", type=Path, required=True)
    generator.add_argument("--cross-encoder-revision", required=True)
    generator.add_argument("--search-snapshot", action="append", type=_binding, required=True)
    generator.add_argument("--prompts-jsonl", type=Path, required=True)
    generator.add_argument("--selection-records-jsonl", type=Path, required=True)
    generator.add_argument("--keyword-priority", type=Path, required=True)
    generator.add_argument("--prompt-count", type=int, required=True)
    generator.add_argument("--seed", type=int, default=20260911)
    generator.add_argument("--prompt-selection-seed", type=int, default=20260912)
    generator.add_argument("--query-max-tokens", type=int, default=256)
    generator.add_argument("--final-max-tokens", type=int, default=2048)
    generator.add_argument("--request-concurrency", type=int, default=1)
    generator.add_argument("--cell-concurrency", type=int)
    generator.add_argument("--disable-thinking", action="store_true")
    judge = subparsers.add_parser("nemotron")
    judge.add_argument("--dataset-root", type=Path, required=True)
    judge.add_argument("--writer-id", required=True)
    judge.add_argument("--judge-manifest", type=Path, required=True)
    judge.add_argument("--tasks", type=Path, required=True)
    judge.add_argument("--private-mapping", type=Path, required=True)
    judge.add_argument("--disable-thinking", action="store_true")
    judge.add_argument("--max-attempts", type=int, default=3)
    judge.add_argument("--request-timeout", type=float, default=600.0)
    judge.add_argument("--max-output-tokens", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generator":
        if re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None:
            raise SystemExit("model revision must be a 40-character SHA")
        if re.fullmatch(r"[0-9a-f]{40}", args.cross_encoder_revision) is None:
            raise SystemExit("cross-encoder revision must be a 40-character SHA")
        snapshots = dict(args.search_snapshot)
        if len(snapshots) != len(args.search_snapshot):
            raise SystemExit("search engines must be unique")
        inputs = SmokeInputs(
            output=args.dataset_root / "local-only" / "registration",
            base_url="http://127.0.0.1:1/v1",
            model_id=args.model_id,
            model_revision=args.model_revision,
            cross_encoder_snapshot=args.cross_encoder_snapshot,
            cross_encoder_revision=args.cross_encoder_revision,
            search_snapshots=snapshots,
            seed=args.seed,
            max_tokens=1024,
            query_max_tokens=args.query_max_tokens,
            final_max_tokens=args.final_max_tokens,
            request_concurrency=args.request_concurrency,
            cell_concurrency=args.cell_concurrency,
            disable_thinking=args.disable_thinking,
            prompts_jsonl=args.prompts_jsonl,
            selection_records_jsonl=args.selection_records_jsonl,
            prompt_count=args.prompt_count,
            prompt_selection_seed=args.prompt_selection_seed,
            production_conditions=True,
        )
        result = register_generator_tasks(
            dataset_root=args.dataset_root,
            model_slug=args.model_slug,
            inputs=inputs,
            keyword_priority_path=args.keyword_priority,
            writer_id=args.writer_id,
        )
    else:
        result = register_nemotron_tasks(
            dataset_root=args.dataset_root,
            manifest_path=args.judge_manifest,
            tasks_path=args.tasks,
            mapping_path=args.private_mapping,
            writer_id=args.writer_id,
            disable_thinking=args.disable_thinking,
            maximum_attempts=args.max_attempts,
            request_timeout=args.request_timeout,
            max_output_tokens=args.max_output_tokens,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
