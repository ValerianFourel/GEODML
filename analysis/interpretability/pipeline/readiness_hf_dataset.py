"""Reproducible local and Hugging Face exports for readiness annotations.

The complete panel may contain source text that is not redistributable.  This
module therefore creates a restricted local bundle and a publication-safe
subset in one deterministic pass.  It never uploads anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .semantic_readiness_dataset import (
    ReadinessLabelTask,
    SemanticReadinessItem,
    normalize_semantic_readiness_text,
    parse_readiness_judgment,
)
from .semantic_readiness_transfer import (
    DEFAULT_TRANSFER_SPEC,
    load_transfer_source_specification,
)


READINESS_HF_FORMAT_VERSION = "semantic-readiness-hf-bundle-v1"
LOCAL_SCOPE = "restricted-local"
HUB_SCOPE = "huggingface-safe"


def assemble_readiness_export(
    *,
    corpus_path: str | Path,
    tasks_path: str | Path,
    codebook_path: str | Path,
    queue_root: str | Path,
    output_dir: str | Path,
    transfer_spec_path: str | Path = DEFAULT_TRANSFER_SPEC,
    expected_judge_slots: Sequence[str] = (),
    git_commit_sha: str,
) -> dict[str, object]:
    """Assemble immutable JSONL inputs for later Parquet/HF finalization."""

    corpus_path = Path(corpus_path).resolve()
    tasks_path = Path(tasks_path).resolve()
    codebook_path = Path(codebook_path).resolve()
    queue_root = Path(queue_root).resolve()
    output = Path(output_dir).resolve()
    for path in (corpus_path, tasks_path, codebook_path):
        if not path.is_file():
            raise ValueError(f"missing required input: {path}")
    if not queue_root.is_dir():
        raise ValueError(f"missing judge queue: {queue_root}")
    if output.exists():
        raise ValueError(f"refusing to overwrite export directory: {output}")
    if not git_commit_sha.strip():
        raise ValueError("git_commit_sha must be nonempty")

    corpus = tuple(SemanticReadinessItem(**row) for row in read_jsonl(corpus_path))
    tasks = tuple(ReadinessLabelTask(**row) for row in read_jsonl(tasks_path))
    codebook = read_jsonl(codebook_path)
    _validate_corpus(corpus)
    tasks_by_id = _validate_tasks(tasks, corpus, expected_judge_slots)
    _validate_codebook(codebook, tasks_by_id)

    source_policy = _source_publication_policy(transfer_spec_path)
    restricted_sources = {
        item.source_name
        for item in corpus
        if not _source_is_publishable(item, source_policy)
    }
    publishable_item_ids = {
        item.item_id for item in corpus if item.source_name not in restricted_sources
    }

    model_outputs = _discover_model_outputs(queue_root)
    annotations, failures, missing = _collect_task_results(
        tasks,
        model_outputs=model_outputs,
    )
    prompt_rows = [_prompt_row(item) for item in corpus]
    scopes = {
        LOCAL_SCOPE: {
            "item_ids": {item.item_id for item in corpus},
            "redistributable": False,
        },
        HUB_SCOPE: {
            "item_ids": publishable_item_ids,
            "redistributable": True,
        },
    }
    output.mkdir(parents=True)
    scope_manifests = {}
    for scope_name, scope in scopes.items():
        scope_dir = output / scope_name
        scope_dir.mkdir()
        item_ids = scope["item_ids"]
        selected_prompts = [row for row in prompt_rows if row["item_id"] in item_ids]
        selected_annotations = [
            row for row in annotations if row["item_id"] in item_ids
        ]
        selected_failures = [row for row in failures if row["item_id"] in item_ids]
        selected_missing = [row for row in missing if row["item_id"] in item_ids]
        atomic_jsonl(scope_dir / "prompts.jsonl", selected_prompts)
        atomic_jsonl(scope_dir / "annotations.jsonl", selected_annotations)
        atomic_jsonl(scope_dir / "failures.jsonl", selected_failures)
        atomic_jsonl(scope_dir / "missing_tasks.jsonl", selected_missing)
        artifact_files = {
            name: {
                "sha256": sha256_file(scope_dir / name),
                "size_bytes": (scope_dir / name).stat().st_size,
            }
            for name in (
                "prompts.jsonl",
                "annotations.jsonl",
                "failures.jsonl",
                "missing_tasks.jsonl",
            )
        }
        scope_manifest = {
            "scope": scope_name,
            "redistributable": bool(scope["redistributable"]),
            "prompt_count": len(selected_prompts),
            "annotation_count": len(selected_annotations),
            "failure_count": len(selected_failures),
            "missing_task_count": len(selected_missing),
            "source_counts": dict(
                sorted(Counter(row["source_name"] for row in selected_prompts).items())
            ),
            "answer_type_counts": dict(
                sorted(
                    Counter(row["answer_type"] for row in selected_annotations).items()
                )
            ),
            "artifacts": artifact_files,
        }
        atomic_json(scope_dir / "scope_manifest.json", scope_manifest)
        scope_manifests[scope_name] = scope_manifest

    manifest = {
        "format_version": READINESS_HF_FORMAT_VERSION,
        "git_commit_sha": git_commit_sha,
        "inputs": {
            "corpus": _file_identity(corpus_path),
            "tasks": _file_identity(tasks_path),
            "codebook": _file_identity(codebook_path),
            "queue_root": str(queue_root),
        },
        "corpus_count": len(corpus),
        "task_count": len(tasks),
        "judge_slots": sorted({task.judge_slot for task in tasks}),
        "restricted_sources": sorted(restricted_sources),
        "restricted_prompt_count": len(corpus) - len(publishable_item_ids),
        "publication_guard": (
            "Only the huggingface-safe scope may be finalized or uploaded. "
            "Restricted prompt text, annotations, failures, rationales, and "
            "derived embeddings remain local."
        ),
        "scopes": scope_manifests,
    }
    atomic_json(output / "assembly_manifest.json", manifest)
    return manifest


def embed_prompt_shards(
    *,
    prompts_path: str | Path,
    output_dir: str | Path,
    view: Mapping[str, object],
    shard_size: int,
    embedder_factory: Callable[[Mapping[str, object]], object],
) -> dict[str, object]:
    """Embed prompts into restart-safe, identity-checked NumPy shards."""

    prompts_path = Path(prompts_path).resolve()
    output = Path(output_dir).resolve()
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    view_name = str(view.get("view_name", "")).strip()
    allowed_view_characters = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not view_name or any(
        character not in allowed_view_characters for character in view_name
    ):
        raise ValueError(
            "view_name must use lowercase letters, digits, hyphens, or underscores"
        )
    prompts = read_jsonl(prompts_path)
    if not prompts:
        raise ValueError("prompt input is empty")
    item_ids = [str(row["item_id"]) for row in prompts]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("prompt input contains duplicate item IDs")
    input_identity = _file_identity(prompts_path)
    frozen_view = json.loads(json.dumps(dict(view), sort_keys=True))
    config_sha256 = _sha256_json(frozen_view)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "embedding_identity.json"
    identity = {
        "format_version": READINESS_HF_FORMAT_VERSION,
        "prompts": input_identity,
        "view": frozen_view,
        "view_config_sha256": config_sha256,
        "item_count": len(prompts),
        "shard_size": shard_size,
    }
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError(f"refusing to change frozen embedding identity: {identity_path}")
    else:
        atomic_json(identity_path, identity)

    embedder = None
    completed = []
    dimension = None
    for shard_index, start in enumerate(range(0, len(prompts), shard_size)):
        rows = prompts[start : start + shard_size]
        shard_path = output / f"part-{shard_index:05d}.npz"
        expected_ids = tuple(str(row["item_id"]) for row in rows)
        expected_hashes = tuple(str(row["text_sha256"]) for row in rows)
        if shard_path.exists():
            shard_dimension = _validate_embedding_shard(
                shard_path,
                expected_ids=expected_ids,
                expected_hashes=expected_hashes,
                view_config_sha256=config_sha256,
            )
        else:
            if embedder is None:
                embedder = embedder_factory(frozen_view)
            matrix = np.asarray(
                embedder.embed([str(row["text"]) for row in rows]),
                dtype=np.float32,
            )
            if matrix.ndim != 2 or matrix.shape[0] != len(rows) or matrix.shape[1] <= 0:
                raise RuntimeError(
                    f"embedder returned invalid shape {matrix.shape} for {len(rows)} prompts"
                )
            atomic_npz(
                shard_path,
                item_ids=np.asarray(expected_ids, dtype=str),
                text_sha256s=np.asarray(expected_hashes, dtype=str),
                view_config_sha256=np.asarray(config_sha256),
                embeddings=matrix,
            )
            shard_dimension = int(matrix.shape[1])
        if dimension is None:
            dimension = shard_dimension
        elif dimension != shard_dimension:
            raise ValueError("embedding dimension changed across shards")
        completed.append(
            {
                "path": shard_path.name,
                "sha256": sha256_file(shard_path),
                "row_count": len(rows),
            }
        )
        atomic_json(
            output / "embedding_manifest.json",
            {
                **identity,
                "embedding_dimension": dimension,
                "completed_shard_count": len(completed),
                "expected_shard_count": (len(prompts) + shard_size - 1) // shard_size,
                "completed_item_count": sum(row["row_count"] for row in completed),
                "is_complete": sum(row["row_count"] for row in completed) == len(prompts),
                "shards": completed,
            },
        )
    return read_json(output / "embedding_manifest.json")


def load_complete_embedding_view(
    embedding_dir: str | Path,
) -> tuple[dict[str, object], dict[str, tuple[str, np.ndarray]]]:
    """Load and validate one complete embedding view by item identity."""

    root = Path(embedding_dir).resolve()
    manifest = read_json(root / "embedding_manifest.json")
    if not manifest.get("is_complete"):
        raise ValueError(f"embedding view is incomplete: {root}")
    rows: dict[str, tuple[str, np.ndarray]] = {}
    for shard in manifest.get("shards", ()):
        path = root / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"embedding shard checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            ids = [str(value) for value in payload["item_ids"]]
            hashes = [str(value) for value in payload["text_sha256s"]]
            matrix = np.asarray(payload["embeddings"], dtype=np.float32)
        if matrix.shape != (len(ids), int(manifest["embedding_dimension"])):
            raise ValueError(f"embedding shard shape mismatch: {path}")
        for item_id, text_hash, vector in zip(ids, hashes, matrix):
            if item_id in rows:
                raise ValueError(f"duplicate embedding item ID: {item_id}")
            rows[item_id] = (text_hash, vector)
    if len(rows) != int(manifest["item_count"]):
        raise ValueError("embedding manifest/item count mismatch")
    return manifest, rows


def _collect_task_results(
    tasks: Sequence[ReadinessLabelTask],
    *,
    model_outputs: Mapping[str, tuple[Path, Mapping[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    annotations = []
    failures = []
    missing = []
    for task in sorted(tasks, key=lambda value: value.task_id):
        discovered = model_outputs.get(task.judge_slot)
        if discovered is None:
            missing.append(_task_status_row(task, status="model_not_started"))
            continue
        output_dir, identity = discovered
        cache_path = output_dir / "task_cache" / f"{task.task_id.replace(':', '_')}.json"
        failure_path = cache_path.with_suffix(".failed.json")
        if cache_path.exists():
            cache = read_json(cache_path)
            _validate_cache_task(cache, task, identity)
            judgment = parse_readiness_judgment(task, str(cache["raw_response"]))
            row = asdict(judgment)
            row.update(
                {
                    "rubric_version": task.rubric_version,
                    "model": str(cache.get("model", identity.get("model", ""))),
                    "model_family": str(
                        cache.get("model_family", identity.get("model_family", ""))
                    ),
                    "model_revision": str(
                        cache.get("model_revision", identity.get("model_revision", ""))
                    ),
                    "backend": str(cache.get("backend", identity.get("backend", ""))),
                    "precision": cache.get("precision", identity.get("precision")),
                    "worker_rank": cache.get("worker_rank"),
                    "rejected_attempts_json": json.dumps(
                        cache.get("rejected_attempts", ()),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "rejected_attempt_count": len(cache.get("rejected_attempts", ())),
                }
            )
            annotations.append(row)
        elif failure_path.exists():
            failure = read_json(failure_path)
            _validate_cache_task(failure, task, identity)
            attempts = failure.get("attempts", ())
            if not isinstance(attempts, list):
                raise ValueError(f"invalid attempts in failure cache: {failure_path}")
            failures.append(
                {
                    **_task_status_row(task, status="failed_validation"),
                    "model": str(failure.get("model", identity.get("model", ""))),
                    "model_family": str(
                        failure.get("model_family", identity.get("model_family", ""))
                    ),
                    "model_revision": str(
                        failure.get("model_revision", identity.get("model_revision", ""))
                    ),
                    "backend": str(
                        failure.get("backend", identity.get("backend", ""))
                    ),
                    "attempt_count": len(attempts),
                    "attempts_json": json.dumps(
                        attempts, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        else:
            missing.append(_task_status_row(task, status="not_attempted"))
    return annotations, failures, missing


def _discover_model_outputs(
    queue_root: Path,
) -> dict[str, tuple[Path, Mapping[str, object]]]:
    outputs = {}
    full_root = queue_root / "full"
    if not full_root.exists():
        return outputs
    for directory in sorted(path for path in full_root.iterdir() if path.is_dir()):
        identity_path = directory / "judge_identity.json"
        if not identity_path.exists():
            identity_path = directory / "run_manifest.json"
        if identity_path.exists():
            identity = read_json(identity_path)
        else:
            caches = sorted((directory / "task_cache").glob("*.json"))
            if not caches:
                continue
            identity = read_json(caches[0])
        slot = str(identity.get("judge_slot", "")).strip()
        if not slot:
            raise ValueError(f"judge output omits judge_slot: {identity_path}")
        if slot in outputs:
            raise ValueError(f"multiple full outputs found for judge slot {slot}")
        outputs[slot] = (directory, identity)
    return outputs


def _validate_corpus(corpus: Sequence[SemanticReadinessItem]) -> None:
    if not corpus:
        raise ValueError("corpus is empty")
    if len({item.item_id for item in corpus}) != len(corpus):
        raise ValueError("corpus contains duplicate item IDs")
    if len({item.text_sha256 for item in corpus}) != len(corpus):
        raise ValueError("corpus contains duplicate text hashes")
    for item in corpus:
        actual = hashlib.sha256(
            normalize_semantic_readiness_text(item.text).encode("utf-8")
        ).hexdigest()
        if actual != item.text_sha256:
            raise ValueError(f"corpus text hash mismatch: {item.item_id}")


def _validate_tasks(
    tasks: Sequence[ReadinessLabelTask],
    corpus: Sequence[SemanticReadinessItem],
    expected_judge_slots: Sequence[str],
) -> dict[str, ReadinessLabelTask]:
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("tasks must be nonempty and uniquely identified")
    item_ids = {item.item_id for item in corpus}
    if {task.item_id for task in tasks} != item_ids:
        raise ValueError("task bank and corpus item IDs differ")
    slots = {task.judge_slot for task in tasks}
    expected_slots = {str(value).strip() for value in expected_judge_slots if str(value).strip()}
    if expected_slots and slots != expected_slots:
        raise ValueError(
            f"judge slots differ: expected={sorted(expected_slots)} actual={sorted(slots)}"
        )
    counts = Counter(task.judge_slot for task in tasks)
    if set(counts.values()) != {len(corpus)}:
        raise ValueError("judge slots do not cover the same corpus")
    return {task.task_id: task for task in tasks}


def _validate_codebook(
    rows: Sequence[Mapping[str, object]],
    tasks_by_id: Mapping[str, ReadinessLabelTask],
) -> None:
    by_task = {str(row.get("task_id", "")): row for row in rows}
    if len(by_task) != len(rows) or set(by_task) != set(tasks_by_id):
        raise ValueError("private codebook and task bank differ")
    for task_id, task in tasks_by_id.items():
        if by_task[task_id].get("item_id") != task.item_id:
            raise ValueError(f"private codebook item mismatch: {task_id}")


def _source_publication_policy(path: str | Path) -> dict[str, bool]:
    return {
        source.source_id: not source.redistribution_policy.startswith("local-only")
        for source in load_transfer_source_specification(path)
    }


def _source_is_publishable(
    item: SemanticReadinessItem,
    transfer_policy: Mapping[str, bool],
) -> bool:
    if item.source_name in transfer_policy:
        return transfer_policy[item.source_name]
    if item.source_name in {"databricks-dolly-15k", "anthropic-hh-helpful-base"}:
        return True
    if item.source_name.startswith("stackexchange:"):
        return item.license.casefold() != "unknown"
    return False


def _validate_cache_task(
    cache: Mapping[str, object],
    task: ReadinessLabelTask,
    identity: Mapping[str, object],
) -> None:
    expected = {
        "task_id": task.task_id,
        "item_id": task.item_id,
        "judge_slot": task.judge_slot,
    }
    for field, value in expected.items():
        cached = cache.get(field)
        if cached is not None and cached != value:
            raise ValueError(f"cache {field} mismatch for {task.task_id}")
    for field in ("model", "model_family", "model_revision", "backend"):
        if cache.get(field) is not None and identity.get(field) is not None:
            if cache[field] != identity[field]:
                raise ValueError(f"cache {field} disagrees with judge identity")


def _prompt_row(item: SemanticReadinessItem) -> dict[str, object]:
    return {"dataset_format_version": READINESS_HF_FORMAT_VERSION, **asdict(item)}


def _task_status_row(task: ReadinessLabelTask, *, status: str) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "item_id": task.item_id,
        "judge_slot": task.judge_slot,
        "presentation_variant": task.presentation_variant,
        "rubric_version": task.rubric_version,
        "status": status,
    }


def _validate_embedding_shard(
    path: Path,
    *,
    expected_ids: Sequence[str],
    expected_hashes: Sequence[str],
    view_config_sha256: str,
) -> int:
    with np.load(path, allow_pickle=False) as payload:
        ids = tuple(str(value) for value in payload["item_ids"])
        hashes = tuple(str(value) for value in payload["text_sha256s"])
        config = str(payload["view_config_sha256"].item())
        matrix = np.asarray(payload["embeddings"])
    if ids != tuple(expected_ids) or hashes != tuple(expected_hashes):
        raise ValueError(f"embedding shard input identities changed: {path}")
    if config != view_config_sha256:
        raise ValueError(f"embedding shard view identity changed: {path}")
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] <= 0:
        raise ValueError(f"invalid embedding shard shape: {path}")
    return int(matrix.shape[1])


def read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def atomic_json(path: str | Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def atomic_npz(path: str | Path, **arrays: object) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: str | Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
