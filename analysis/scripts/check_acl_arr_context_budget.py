#!/usr/bin/env python3
"""Measure frozen primary chat requests using a pinned, locally cached tokenizer."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.scripts.run_acl_arr_vllm import (  # noqa: E402
    _prepare_primary,
    _primary_context,
    _sha256,
)


def _input_token_count(encoded: Any) -> int:
    """Count one chat's token IDs, never encoding fields or batch rows."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer output is missing input_ids")
        encoded = encoded["input_ids"]
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
        if isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
    if (
        not isinstance(encoded, (list, tuple))
        or not encoded
        or any(type(token) is not int or token < 0 for token in encoded)
    ):
        raise ValueError("input_ids must contain one nonempty sequence of integer token IDs")
    return len(encoded)


def _native_context(snapshot: Path) -> tuple[int | None, str | None]:
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("snapshot config.json must contain an object")
    text = config.get("text_config")
    candidates = [("text_config.", text)] if isinstance(text, dict) else []
    candidates.append(("", config))
    for prefix, candidate in candidates:
        for field in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
            limit = candidate.get(field)
            if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                return limit, prefix + field
    return None, None


def check_context_budget(
    plan_manifest: Path,
    tasks_path: Path,
    model_snapshots: Path,
    *,
    tokenizer_loader: Callable[..., Any],
    tokenizer_version: str,
) -> dict[str, Any]:
    """Read every task without changing evidence, output budgets, or artifacts.

    This checks the Hugging Face tokenizer/chat-template serving path. It does
    not infer a native-Mistral tokenizer budget or apply RoPE scaling overrides.
    The reported maximum is an input requirement, not a GPU memory estimate.
    """
    plan_manifest = Path(plan_manifest).resolve()
    tasks_path = Path(tasks_path).resolve()
    model_snapshots = Path(model_snapshots).resolve()
    plan, tasks, model, pipeline = _primary_context(plan_manifest, tasks_path)
    if model.model_id.startswith("mistralai/Mistral-Small-4-"):
        raise ValueError(
            "the pilot serves this model with a native Mistral tokenizer; "
            "the Hugging Face context preflight does not support that backend"
        )
    payload = json.loads(model_snapshots.read_text(encoding="utf-8"))
    locks = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(locks, list):
        raise ValueError("model-snapshots must contain a models list")
    matches = [
        row for row in locks
        if isinstance(row, dict)
        and row.get("model_id") == model.model_id
        and row.get("revision") == model.model_revision
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one snapshot matching model ID and revision")
    snapshot_value = matches[0].get("snapshot")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise ValueError("matching model snapshot lacks a local path")
    snapshot = Path(snapshot_value).resolve()
    if not snapshot.is_dir():
        raise ValueError(f"missing cached model snapshot: {snapshot}")
    native_context, context_source = _native_context(snapshot)
    tokenizer = tokenizer_loader(
        str(snapshot), local_files_only=True, trust_remote_code=True,
    )
    prompts = {item.prompt_id: item for item in plan.prompts}
    assignments = {item.assignment_id: item for item in plan.assignments}
    documents = {item.candidate_set_id: item for item in plan.document_sets}
    max_prompt_tokens = 0
    max_required_tokens = 0
    worst_task_id = None
    for task in tasks:
        assignment = assignments[task.assignment_id]
        prepared = _prepare_primary(
            task,
            prompt=prompts[task.prompt_id],
            assignment=assignment,
            document_set=documents[assignment.candidate_set_id],
            model=model,
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prepared["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=True,
        )
        prompt_tokens = _input_token_count(tokens)
        required_tokens = prompt_tokens + prepared["max_tokens"]
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        if required_tokens > max_required_tokens:
            max_required_tokens = required_tokens
            worst_task_id = task.task_id
    rounded_budget = ((max_required_tokens + 1023) // 1024) * 1024
    if native_context is None:
        status = "UNKNOWN_NATIVE_CONTEXT"
    elif max_required_tokens > native_context:
        status = "EXCEEDS_NATIVE_CONTEXT"
    elif rounded_budget > native_context:
        status = "ROUNDED_BUDGET_EXCEEDS_NATIVE_CONTEXT"
    else:
        status = "PASS"
    return {
        "format_version": "acl-arr-context-budget-v1",
        "status": status,
        "scientific_result": False,
        "plan_manifest": str(plan_manifest),
        "plan_manifest_sha256": _sha256(plan_manifest),
        "source_git_commit": plan.source_git_commit,
        "tasks": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "model_snapshots_sha256": _sha256(model_snapshots),
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "pipeline": pipeline,
        "snapshot": str(snapshot),
        "config_sha256": _sha256(snapshot / "config.json"),
        "task_count": len(tasks),
        "max_prompt_tokens": max_prompt_tokens,
        "max_required_tokens": max_required_tokens,
        "worst_task_id": worst_task_id,
        "suggested_max_model_len": rounded_budget if status == "PASS" else None,
        "native_context_tokens": native_context,
        "native_context_source": context_source,
        "tokenizer_backend": "transformers.AutoTokenizer",
        "tokenizer_version": tokenizer_version,
        "truncation": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--model-snapshots", type=Path, required=True)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    report = check_context_budget(
        args.plan_manifest, args.tasks, args.model_snapshots,
        tokenizer_loader=AutoTokenizer.from_pretrained,
        tokenizer_version=version("transformers"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
