"""Scheduling policy for a frozen multi-model backlog, independent of Slurm/GPU imports.

Counts are verified claim envelopes, never manifest counters. A scheduling
decision grants no ownership: the existing inference runner still claims each
task. No tasks, model settings, allocations or retries are invented here.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.interpretability.pipeline.inference_claims import (
    ClaimIdentity,
    InferenceClaimStore,
)


@dataclass(frozen=True)
class Queue:
    name: str
    model: str
    priority: int
    claim_root: Path
    identities: tuple[ClaimIdentity, ...]
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.model
            or type(self.priority) is not int
            or self.priority < 0
        ):
            raise ValueError("queue name, model and non-negative priority are required")
        if not self.identities or len({x.task_id for x in self.identities}) != len(
            self.identities
        ):
            raise ValueError("queue must contain unique scientific task identities")
        if (
            len({(x.model_id, x.model_revision, x.protocol) for x in self.identities})
            != 1
        ):
            raise ValueError("a queue must contain one model revision and protocol")


@dataclass(frozen=True)
class Progress:
    expected: int
    completed: int = 0
    failed: int = 0
    busy: int = 0
    missing: int = 0

    @property
    def complete(self) -> bool:
        return self.completed == self.expected and self.expected > 0


def inspect_queue(queue: Queue) -> Progress:
    store = InferenceClaimStore(queue.claim_root)
    counts = Counter(store.inspect(identity)[0] for identity in queue.identities)
    return Progress(expected=len(queue.identities), **counts)


def validate_queues(queues: Sequence[Queue]) -> None:
    indexed = {q.name: q for q in queues}
    if len(indexed) != len(queues):
        raise ValueError("duplicate queue name")
    visited: set[str] = set()

    def visit(name: str, ancestors: frozenset[str]) -> None:
        if name not in indexed:
            raise ValueError(f"unknown prerequisite queue: {name}")
        if name in ancestors:
            raise ValueError("cyclic queue prerequisites")
        if name in visited:
            return
        for dependency in indexed[name].requires:
            visit(dependency, ancestors | {name})
        visited.add(name)

    for name in indexed:
        visit(name, frozenset())
    owners: dict[ClaimIdentity, Path] = {}
    for queue in queues:
        for identity in queue.identities:
            prior = owners.setdefault(identity, queue.claim_root.resolve())
            if prior != queue.claim_root.resolve():
                raise ValueError(
                    "the same scientific task uses separate claim roots; audit before launch"
                )


def choose_queue(
    queues: Sequence[Queue],
    progress: Mapping[str, Progress],
    *,
    current: str | None = None,
    residence_seconds: float = 0,
    minimum_residence_seconds: float = 300,
) -> Queue | None:
    """Prioritize original generation, gated judging, then approved overflow.

    Busy work is not runnable and does not satisfy a dependency. It may justify
    a bounded wait by the supervisor if there is no other eligible work.
    """
    eligible = [
        q
        for q in queues
        if progress[q.name].missing > 0
        and all(progress[name].complete for name in q.requires)
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda q: (q.priority, q.name))
    incumbent = next((q for q in eligible if q.name == current), None)
    best = eligible[0]
    if incumbent is not None and (
        incumbent.priority == best.priority
        or residence_seconds < minimum_residence_seconds
    ):
        return incumbent
    return best


def audit_registries(
    roots: Sequence[Path],
    identities: Sequence[ClaimIdentity],
) -> dict[str, Any]:
    """Count the verified union without merging or rewriting any registry.

    Different request hashes are deliberately not treated as interchangeable.
    Conflicting successful outcomes are surfaced, never resolved by recency.
    """
    unique_roots = sorted({root.resolve() for root in roots})
    counts: Counter[str] = Counter()
    conflicts: list[str] = []
    for identity in identities:
        observations = [
            InferenceClaimStore(root).inspect(identity) for root in unique_roots
        ]
        successes = [outcome for state, outcome in observations if state == "completed"]
        failures = [outcome for state, outcome in observations if state == "failed"]
        if successes and (
            failures or any(value != successes[0] for value in successes[1:])
        ):
            conflicts.append(identity.task_id)
        counts[
            "completed"
            if successes
            else "failed"
            if failures
            else "busy"
            if any(state == "busy" for state, _ in observations)
            else "missing"
        ] += 1
        counts["duplicate_success_records"] += max(0, len(successes) - 1)
    return {
        "expected": len(identities),
        **counts,
        "conflicts": conflicts,
        "claim_roots": [str(root) for root in unique_roots],
        "modified": False,
    }


SWEEP_CONCURRENCIES = (4, 8, 16, 32)


def stratified_sweep_sample(
    token_counts: Mapping[str, int],
    *,
    seed: int = 20260921,
) -> dict[str, list[str]]:
    """Freeze 16 tasks per input-token-length quartile and four disjoint warmups."""
    if len(token_counts) < 68 or any(
        not key or type(value) is not int or value <= 0
        for key, value in token_counts.items()
    ):
        raise ValueError(
            "sweep requires at least 68 tasks with positive measured input-token counts"
        )
    ordered = sorted(token_counts, key=lambda key: (token_counts[key], key))
    measured, warmup = [], []
    for quartile in range(4):
        group = ordered[
            len(ordered) * quartile // 4 : len(ordered) * (quartile + 1) // 4
        ]
        rng = random.Random(f"{seed}:nemotron-length-quartile:{quartile}")
        chosen = rng.sample(group, 17)
        measured.extend(chosen[:16])
        warmup.append(chosen[16])
    return {"measured": measured, "warmup": warmup}


def select_sweep_concurrency(
    results: Sequence[Mapping[str, Any]],
    *,
    sample_sha256: str,
    profile_sha256: str,
) -> dict[str, Any]:
    """Only fully completed, failure-free, matching measurements may win."""
    accepted = []
    seen = set()
    for result in results:
        concurrency = result.get("concurrency")
        if concurrency not in SWEEP_CONCURRENCIES or concurrency in seen:
            raise ValueError("unknown or duplicate sweep candidate")
        seen.add(concurrency)
        seconds = result.get("measurement_seconds")
        if (
            result.get("sample_sha256") != sample_sha256
            or result.get("profile_sha256") != profile_sha256
        ):
            raise ValueError(
                "sweep candidates must use the same sample and serving profile"
            )
        if (
            result.get("status") != "complete"
            or result.get("succeeded") != 64
            or result.get("failed") != 0
            or result.get("warmup_succeeded") != 4
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            continue
        accepted.append((64 / seconds, -concurrency))
    best = max(accepted) if accepted else None
    return {
        "concurrency": -best[1] if best else 4,
        "valid_judgments_per_second": best[0] if best else None,
        "reason": "best_complete_candidate"
        if best
        else "no_complete_candidate_fallback",
        "sample_sha256": sample_sha256,
        "profile_sha256": profile_sha256,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
