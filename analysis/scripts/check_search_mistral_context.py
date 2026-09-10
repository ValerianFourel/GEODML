#!/usr/bin/env python3
"""Measure search requests with the pinned native Mistral tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.scripts.run_search_experience import primary_items


def resolve_snapshot(path: Path, model_id: str, revision: str) -> Path:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("model-snapshots must contain a models list")
    matches = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("model_id") == model_id
        and row.get("revision") == revision
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one pinned Mistral snapshot")
    value = matches[0].get("snapshot")
    if not isinstance(value, str) or not value:
        raise ValueError("pinned Mistral snapshot lacks a local path")
    snapshot = Path(value).resolve()
    if not snapshot.is_dir():
        raise ValueError(f"missing pinned Mistral snapshot: {snapshot}")
    return snapshot


def check_context_budget(items, *, token_counter, max_model_len: int) -> dict:
    if max_model_len <= 0:
        raise ValueError("max model length must be positive")
    if not items:
        raise ValueError("at least one request is required")
    worst = None
    for item in items:
        prompt_tokens = token_counter(str(item["prompt"]))
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError("token counter must return a positive integer")
        required = prompt_tokens + int(item["max_tokens"])
        candidate = (required, prompt_tokens, item["base"]["task_id"])
        if worst is None or candidate[0] > worst[0]:
            worst = candidate
    assert worst is not None
    required, prompt_tokens, task_id = worst
    remaining = max_model_len - required
    return {
        "format_version": "search-mistral-context-budget-v1",
        "status": "PASS" if remaining >= 0 else "EXCEEDS_MODEL_CONTEXT",
        "task_count": len(items),
        "max_model_len": max_model_len,
        "max_prompt_tokens": prompt_tokens,
        "max_required_tokens": required,
        "remaining_tokens": max(0, remaining),
        "overflow_tokens": max(0, -remaining),
        "worst_task_id": task_id,
        "scientific_result": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--model-configuration-id", required=True)
    parser.add_argument("--model-snapshots", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--answer-max-tokens", type=int)
    args = parser.parse_args(argv)
    items, identity, _ = primary_items(
        args.bundle_dir,
        args.model_configuration_id,
        answer_max_tokens=args.answer_max_tokens,
    )
    snapshot = resolve_snapshot(
        args.model_snapshots,
        identity["model_id"],
        identity["model_revision"],
    )
    tokenizer_path = snapshot / "tekken.json"
    if not tokenizer_path.is_file():
        raise ValueError(f"native Mistral tokenizer missing: {tokenizer_path}")
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(str(tokenizer_path))

    def count(prompt: str) -> int:
        encoded = tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        )
        return len(encoded.tokens)

    report = check_context_budget(
        items,
        token_counter=count,
        max_model_len=args.max_model_len,
    )
    report.update({
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
        "snapshot": str(snapshot),
        "tokenizer": str(tokenizer_path),
        "tokenizer_backend": "mistral_common.MistralTokenizer",
    })
    print("MISTRAL_CONTEXT_PREFLIGHT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
