#!/usr/bin/env python3
"""Accept recovered generator payloads only when exact registered identities match."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter,
    iter_sealed_rows,
)
from analysis.interpretability.pipeline.agentic_task_ledger import (
    StripedTaskLedger,
    identity_fingerprint,
)
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"recovery row is not an object: {path}:{number}")
            yield value


def _verify_hub(hub: Path) -> str:
    manifest_path = hub / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != "geodml-recovery-publication-v1":
        raise ValueError("unsupported recovery publication manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("recovery publication manifest lacks files")
    actual = {str(path.relative_to(hub)) for path in hub.rglob("*") if path.is_file()}
    if actual != set(files) | {"publication-manifest.json"}:
        raise ValueError("recovery hub has unexpected or missing files")
    for name, digest in files.items():
        if _sha256(hub / name) != digest:
            raise ValueError(f"recovery publication checksum mismatch: {name}")
    return _sha256(manifest_path)


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def accept(
    dataset_root: Path,
    recovery_hub: Path,
    *,
    model: str,
    stripe_count: int = 256,
) -> dict[str, Any]:
    if model not in {"qwen38", "llama4"}:
        raise ValueError("recovery acceptance supports qwen38 or llama4")
    manifest_sha256 = _verify_hub(recovery_hub)
    acceptance_id = f"recovery-generator-{model}-{manifest_sha256[:24]}"
    receipt_path = dataset_root / "control" / "imports" / f"{acceptance_id}.json"
    lock_path = dataset_root / "control" / "imports" / ".recovery-acceptance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt_path.exists():
            return json.loads(receipt_path.read_text())
        tasks: dict[str, tuple[dict[str, Any], ClaimIdentity]] = {}
        for row in iter_sealed_rows(dataset_root, "task_definitions", required=True):
            if row.get("model") != model:
                continue
            identity = ClaimIdentity(**row["claim_identity"])
            fingerprint = identity_fingerprint(identity)
            if fingerprint in tasks:
                raise ValueError("registered tasks duplicate an identity")
            tasks[fingerprint] = row, identity
        prompts = {
            row["prompt_id"]: row
            for row in iter_sealed_rows(dataset_root, "prompts", required=True)
        }
        model_id = next(iter(tasks.values()))[1].model_id if tasks else None
        aliases: dict[str, set[str]] = defaultdict(set)
        for row in _rows(recovery_hub / "data" / "generation_aliases.jsonl.gz"):
            native = row.get("native_identity")
            if not isinstance(native, dict):
                continue
            try:
                identity = ClaimIdentity(**native)
            except (TypeError, ValueError):
                continue
            fingerprint = identity_fingerprint(identity)
            if fingerprint in tasks:
                aliases[fingerprint].add(str(row.get("generation_id", "")))
        generations = {
            row["generation_id"]: row
            for row in _rows(recovery_hub / "data" / "generations.jsonl.gz")
            if row.get("model_id") == model_id
        }
        accepted: list[tuple[str, dict[str, Any], ClaimIdentity, dict[str, Any]]] = []
        rejected: dict[str, int] = defaultdict(int)
        for fingerprint, (task, identity) in sorted(tasks.items()):
            generation_ids = {item for item in aliases.get(fingerprint, set()) if item}
            if not generation_ids:
                rejected["no_exact_native_identity"] += 1
                continue
            if len(generation_ids) != 1:
                rejected["ambiguous_exact_native_identity"] += 1
                continue
            generation = generations.get(next(iter(generation_ids)))
            if generation is None:
                rejected["generation_payload_missing"] += 1
                continue
            result, trace = generation.get("result"), generation.get("trace")
            prompt = prompts.get(task["prompt_id"])
            if not isinstance(result, dict) or not isinstance(trace, dict) or prompt is None:
                rejected["invalid_payload_shape"] += 1
                continue
            expected = {
                "cell_id": task["task_id"],
                "prompt_id": task["prompt_id"],
                "method": task["method"],
                "engine": task["engine"],
                "condition": task["condition"],
            }
            if any(result.get(key) != value for key, value in expected.items()):
                rejected["result_task_identity_mismatch"] += 1
                continue
            if (
                generation.get("model_revision") != identity.model_revision
                or trace.get("trace_sha256") != result.get("trace_sha256")
                or trace.get("user_prompt_sha256") != prompt.get("prompt_sha256")
                or generation.get("validation") != "payload_valid_protocol_acceptance_pending"
            ):
                rejected["payload_protocol_mismatch"] += 1
                continue
            accepted.append((fingerprint, task, identity, generation))
        writer_id = f"accept-{model}-{manifest_sha256[:16]}"
        writer = FinalDatasetWriter(dataset_root, writer_id=writer_id)
        references: dict[str, list[dict[str, Any]]] = {}
        for fingerprint, task, _, generation in accepted:
            transaction = f"recovery-acceptance-{fingerprint}"
            trace_reference = writer.append(
                "traces",
                {
                    "trace_sha256": generation["trace"]["trace_sha256"],
                    "events": generation["trace"].get("events", []),
                    "recovery_generation_id": generation["generation_id"],
                },
                transaction_id=transaction,
                record_id=f"trace-{transaction}",
            )
            generation_reference = writer.append(
                "generations",
                {
                    **generation["result"],
                    "trace_record_id": trace_reference["record_id"],
                    "recovery_generation_id": generation["generation_id"],
                    "validation": "exact_registered_identity_accepted",
                },
                transaction_id=transaction,
                record_id=f"generation-{transaction}",
            )
            provenance_reference = writer.append(
                "provenance",
                {
                    "cell_id": task["task_id"],
                    "source": "verified-recovery-hub",
                    "recovery_publication_manifest_sha256": manifest_sha256,
                    "recovery_generation_id": generation["generation_id"],
                    "native_identity_fingerprint": fingerprint,
                },
                transaction_id=transaction,
                record_id=f"provenance-{transaction}",
            )
            references[fingerprint] = [
                trace_reference, generation_reference, provenance_reference
            ]
        manifests = writer.seal()
        ledger = StripedTaskLedger(
            dataset_root / "control" / "task-ledger", stripe_count=stripe_count
        )
        completed = 0
        for fingerprint, _, identity, _ in accepted:
            ownership = ledger.claim(identity, owner_id=writer_id)
            if ownership.status == "completed":
                continue
            if ownership.status != "owned" or ownership.claim is None:
                raise RuntimeError(f"recovery identity is currently owned: {fingerprint}")
            ledger.transition(
                ownership.claim,
                state="completed",
                record_references=references[fingerprint],
                detail={"acceptance_id": acceptance_id},
            )
            completed += 1
        result = {
            "format_version": "geodml-recovery-generation-acceptance-v1",
            "status": "complete",
            "acceptance_id": acceptance_id,
            "model": model,
            "registered_tasks": len(tasks),
            "exact_candidates": len(accepted),
            "ledger_completions_created": completed,
            "rejected": dict(sorted(rejected.items())),
            "recovery_publication_manifest_sha256": manifest_sha256,
            "sealed_manifests": manifests,
        }
        _atomic(receipt_path, result)
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recovery-hub", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen38", "llama4"), required=True)
    parser.add_argument("--stripe-count", type=int, default=256)
    args = parser.parse_args(argv)
    print(json.dumps(accept(
        args.dataset_root.resolve(), args.recovery_hub.resolve(),
        model=args.model, stripe_count=args.stripe_count,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
