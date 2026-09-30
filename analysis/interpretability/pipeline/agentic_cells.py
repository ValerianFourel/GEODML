"""Read completed generator cells (generation + trace + prompt) from a sealed dataset.

Read-only. A cell counts only when its latest ledger state is ``completed`` and both
record references verify against sealed shards; everything else is skipped and
counted, never guessed. Each sealed shard is read once, not once per record.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .agentic_dataset import iter_sealed_rows, verify_record_reference
from .agentic_task_ledger import StripedTaskLedger, identity_fingerprint
from .inference_claims import ClaimIdentity

GENERATOR_MODELS = ("qwen38", "llama4")


def _shard_path(root: Path, reference: Mapping[str, Any]) -> Path:
    return root / "data" / reference["table"] / f"part-{reference['writer_id']}-{reference['shard_sequence']:06d}.jsonl"


def read_references(root: Path, references: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rows for verified references keyed by record_id, reading each shard once."""

    by_shard: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        if not verify_record_reference(root, reference):
            raise ValueError(f"record reference failed verification: {reference.get('record_id')}")
        by_shard[_shard_path(root, reference)].append(reference)
    rows: dict[str, dict[str, Any]] = {}
    for path, wanted in by_shard.items():
        lines = path.read_bytes().splitlines()
        for reference in wanted:
            envelope = json.loads(lines[reference["line_number"] - 1])
            if envelope.get("record_id") != reference["record_id"]:
                raise ValueError(f"record line does not match its reference: {reference['record_id']}")
            rows[reference["record_id"]] = envelope["row"]
    return rows


def completed_generator_refs(
    root: Path, *, model: str, prompt_ids: set[str] | None = None, stripes: int = 256,
) -> tuple[list[dict[str, Any]], Counter]:
    """Verified generation/trace references of completed cells for one generator."""

    if model not in GENERATOR_MODELS:
        raise ValueError(f"unknown generator model: {model}")
    tasks = {}
    for row in iter_sealed_rows(root, "task_definitions", required=True):
        if row.get("model") == model and row.get("stage", "generation") == "generation":
            tasks[identity_fingerprint(ClaimIdentity(**row["claim_identity"]))] = row
    latest = StripedTaskLedger(root / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
    counts: Counter = Counter()
    selected = []
    for fingerprint, task in sorted(tasks.items()):
        if prompt_ids is not None and task.get("prompt_id") not in prompt_ids:
            continue
        event = latest.get(fingerprint)
        if not event or event.get("state") != "completed":
            counts["not_completed"] += 1
            continue
        refs = {ref["table"]: ref for ref in event.get("record_references", [])}
        if "generations" not in refs or "traces" not in refs:
            counts["missing_references"] += 1
            continue
        if not all(verify_record_reference(root, refs[t]) for t in ("generations", "traces")):
            counts["unverified_references"] += 1
            continue
        counts["completed"] += 1
        selected.append({"fingerprint": fingerprint, "model": model,
                         **{k: task.get(k) for k in ("prompt_id", "method", "engine", "condition")},
                         "task_metadata": task,
                         "generation_ref": refs["generations"], "trace_ref": refs["traces"]})
    return selected, counts


def iter_cells(root: Path, refs: list[dict[str, Any]], *, batch: int = 2000) -> Iterator[dict[str, Any]]:
    """Yield cells with generation, trace and prompt text, in batches of shard reads."""

    prompts = {row["prompt_id"]: row for row in iter_sealed_rows(root, "prompts")}
    memberships = {row["prompt_id"]: row for row in iter_sealed_rows(root, "keyword_memberships")}
    for start in range(0, len(refs), batch):
        chunk = refs[start:start + batch]
        rows = read_references(root, [r[k] for r in chunk for k in ("generation_ref", "trace_ref")])
        for ref in chunk:
            generation = rows[ref["generation_ref"]["record_id"]]
            prompt = prompts.get(generation.get("prompt_id"), {})
            yield {**ref, "generation": generation, "trace": rows[ref["trace_ref"]["record_id"]],
                   "prompt_text": prompt.get("prompt_text"), "prompt_record": prompt,
                   "keyword_memberships": memberships.get(generation.get("prompt_id"), {})}
