"""Phase-1 freeze audit for the semantic-readiness corpora.

This module performs no annotation, embedding, or model inference.  It verifies
that the frozen Layer-1 corpus can be reconstructed byte-for-row from its
recorded surface and web inputs, and records whether each separately assigned
Layer-2 transfer source has an acquired, revision-pinned local snapshot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .semantic_readiness_dataset import (
    SEMANTIC_DATASET_VERSION,
    SemanticReadinessItem,
    WebTextRecord,
    build_semantic_readiness_corpus,
    is_semantic_readiness_text_eligible,
    load_web_retrieval_specification,
    normalize_semantic_readiness_text,
)
from .semantic_readiness_transfer import (
    TRANSFER_PANEL_VERSION,
    TransferSource,
    load_transfer_source_specification,
)


PHASE1_FREEZE_VERSION = "semantic-readiness-phase1-freeze-v1"

FROZEN_BASE_COUNTS: dict[tuple[str, str], int] = {
    ("anthropic-hh-helpful-base", "confirmation"): 418,
    ("anthropic-hh-helpful-base", "development"): 1_582,
    ("databricks-dolly-15k", "confirmation"): 489,
    ("databricks-dolly-15k", "development"): 1_511,
    ("stackexchange:askubuntu", "development"): 190,
    ("stackexchange:diy", "confirmation"): 214,
    ("stackexchange:stackoverflow", "development"): 195,
    ("stackexchange:superuser", "development"): 179,
    ("stackexchange:travel", "confirmation"): 162,
    ("stackexchange:workplace", "development"): 151,
}


@dataclass(frozen=True, slots=True)
class TransferSnapshotInput:
    source_id: str
    path: Path
    revision: str


@dataclass(frozen=True, slots=True)
class SurfaceSnapshotInput:
    """Portable replacement for a machine-local provenance path."""

    source_id: str
    path: Path


def build_phase1_freeze_audit(
    *,
    base_corpus_path: str | Path,
    base_manifest_path: str | Path,
    surface_corpus_path: str | Path,
    surface_manifest_path: str | Path,
    surface_provenance_path: str | Path,
    web_records_path: str | Path,
    web_manifest_path: str | Path,
    web_raw_responses_dir: str | Path,
    web_specification_path: str | Path,
    transfer_specification_path: str | Path,
    surface_inputs: Sequence[SurfaceSnapshotInput] = (),
    transfer_inputs: Sequence[TransferSnapshotInput] = (),
    expected_counts: Mapping[tuple[str, str], int] = FROZEN_BASE_COUNTS,
) -> dict[str, object]:
    """Return an immutable-data audit without performing any model work."""

    paths = {
        "base_corpus": Path(base_corpus_path).resolve(),
        "base_manifest": Path(base_manifest_path).resolve(),
        "surface_corpus": Path(surface_corpus_path).resolve(),
        "surface_manifest": Path(surface_manifest_path).resolve(),
        "surface_provenance": Path(surface_provenance_path).resolve(),
        "web_records": Path(web_records_path).resolve(),
        "web_manifest": Path(web_manifest_path).resolve(),
        "web_raw_responses": Path(web_raw_responses_dir).resolve(),
        "web_specification": Path(web_specification_path).resolve(),
        "transfer_specification": Path(transfer_specification_path).resolve(),
    }
    missing = sorted(name for name, path in paths.items() if not path.exists())
    if missing:
        raise FileNotFoundError("missing Phase-1 inputs: " + ", ".join(missing))

    base_rows = _read_jsonl(paths["base_corpus"])
    items = tuple(SemanticReadinessItem(**row) for row in base_rows)
    surface_rows = _read_jsonl(paths["surface_corpus"])
    web_rows = tuple(_web_record(row) for row in _read_jsonl(paths["web_records"]))
    reconstructed = build_semantic_readiness_corpus(surface_rows, web_rows)

    base_manifest = _read_json(paths["base_manifest"])
    surface_manifest = _read_json(paths["surface_manifest"])
    surface_provenance = _read_json(paths["surface_provenance"])
    web_manifest = _read_json(paths["web_manifest"])
    web_probes = load_web_retrieval_specification(paths["web_specification"])
    transfer_sources = load_transfer_source_specification(
        paths["transfer_specification"]
    )

    base_blockers: list[str] = []
    actual_counts = _counts_by_source_and_split(items)
    expected_counts = dict(expected_counts)
    if actual_counts != expected_counts:
        base_blockers.append("source/split counts differ from the frozen design")
    expected_total = sum(expected_counts.values())
    if len(items) != expected_total:
        base_blockers.append(
            f"expected {expected_total} base rows, found {len(items)}"
        )
    if len({item.item_id for item in items}) != len(items):
        base_blockers.append("base corpus contains duplicate item IDs")
    if len({item.text_sha256 for item in items}) != len(items):
        base_blockers.append("base corpus contains duplicate exact-text hashes")
    text_hash_mismatches = tuple(
        item.item_id
        for item in items
        if item.text_sha256 != _hash_text(normalize_semantic_readiness_text(item.text))
    )
    if text_hash_mismatches:
        base_blockers.append("base corpus contains normalized-text hash mismatches")
    ineligible = tuple(
        item.item_id
        for item in items
        if not is_semantic_readiness_text_eligible(item.text)
    )
    if ineligible:
        base_blockers.append("base corpus contains text outside the 3--100 word rule")
    unknown_licenses = tuple(
        item.item_id
        for item in items
        if not item.license.strip() or item.license.casefold() == "unknown"
    )
    if unknown_licenses:
        base_blockers.append("base corpus contains missing or unknown licenses")
    cross_split_groups = _cross_split_groups(items)
    if cross_split_groups:
        base_blockers.append("development and confirmation share group IDs")
    if tuple(items) != reconstructed:
        base_blockers.append("base corpus is not an exact rebuild of surface + web inputs")

    base_sha = _sha256_file(paths["base_corpus"])
    if base_manifest.get("artifact_version") != SEMANTIC_DATASET_VERSION:
        base_blockers.append("base manifest has an unexpected artifact version")
    if base_manifest.get("corpus_sha256") != base_sha:
        base_blockers.append("base manifest corpus hash does not match the file")
    if base_manifest.get("corpus_count") != len(items):
        base_blockers.append("base manifest corpus count does not match the file")
    if bool(base_manifest.get("semantic_labels_present")):
        base_blockers.append("base manifest unexpectedly declares semantic labels")
    if bool(base_manifest.get("reranking_outcomes_observed")):
        base_blockers.append("base manifest unexpectedly declares reranking outcomes")

    surface_sha = _sha256_file(paths["surface_corpus"])
    if surface_manifest.get("corpus_sha256") != surface_sha:
        base_blockers.append("surface manifest corpus hash does not match the file")
    if base_manifest.get("surface_corpus_sha256") != surface_sha:
        base_blockers.append("base manifest does not pin the supplied surface corpus")
    surface_sources, surface_source_blockers = _audit_surface_sources(
        surface_provenance,
        surface_inputs,
    )
    base_blockers.extend(surface_source_blockers)

    web_sha = _sha256_file(paths["web_records"])
    if base_manifest.get("web_records_sha256") != web_sha:
        base_blockers.append("base manifest does not pin the supplied web records")
    if web_manifest.get("unique_web_record_count") != len(web_rows):
        base_blockers.append("web manifest count does not match the web record file")
    expected_probe_files = {f"{probe.probe_id}.json" for probe in web_probes}
    actual_probe_files = {
        path.name
        for path in paths["web_raw_responses"].iterdir()
        if path.is_file() and path.suffix == ".json"
    }
    if actual_probe_files != expected_probe_files:
        base_blockers.append("raw Stack Exchange response files differ from the probe spec")
    for raw_path in sorted(paths["web_raw_responses"].glob("*.json")):
        _read_json(raw_path)

    licensed_web = sum(item.license.casefold() != "unknown" for item in web_rows)
    excluded_web = len(web_rows) - licensed_web
    if licensed_web != base_manifest.get("web_included_count"):
        base_blockers.append("licensed web count differs from the base manifest")
    if excluded_web != base_manifest.get("web_excluded_count"):
        base_blockers.append("excluded web count differs from the base manifest")

    transfer_status, transfer_blockers = _audit_transfer_inputs(
        transfer_sources,
        transfer_inputs,
    )
    registry_sha = _sha256_file(paths["transfer_specification"])
    base_ready = not base_blockers
    all_transfer_snapshots_present = all(
        bool(row["snapshot_present"]) for row in transfer_status
    )
    full_phase2_ready = base_ready and all_transfer_snapshots_present
    split_counts = _counts_by_split(items)

    return {
        "artifact_version": PHASE1_FREEZE_VERSION,
        "scientific_result": False,
        "annotation_performed": False,
        "embedding_performed": False,
        "model_inference_performed": False,
        "base_corpus": {
            "path": str(paths["base_corpus"]),
            "sha256": base_sha,
            "record_count": len(items),
            "development_count": split_counts.get("development", 0),
            "confirmation_count": split_counts.get("confirmation", 0),
            "unique_item_id_count": len({item.item_id for item in items}),
            "unique_text_sha256_count": len({item.text_sha256 for item in items}),
            "normalized_text_hash_mismatch_count": len(text_hash_mismatches),
            "ineligible_text_count": len(ineligible),
            "unknown_license_count": len(unknown_licenses),
            "cross_split_group_count": len(cross_split_groups),
            "exact_rebuild_matches": tuple(items) == reconstructed,
            "counts_by_source_and_split": {
                f"{source}|{split}": count
                for (source, split), count in sorted(actual_counts.items())
            },
            "manifest_path": str(paths["base_manifest"]),
            "manifest_sha256": _sha256_file(paths["base_manifest"]),
            "blockers": base_blockers,
            "frozen": base_ready,
        },
        "surface_inputs": {
            "corpus_path": str(paths["surface_corpus"]),
            "corpus_sha256": surface_sha,
            "manifest_path": str(paths["surface_manifest"]),
            "manifest_sha256": _sha256_file(paths["surface_manifest"]),
            "provenance_path": str(paths["surface_provenance"]),
            "provenance_sha256": _sha256_file(paths["surface_provenance"]),
            "sources": surface_sources,
        },
        "web_inputs": {
            "records_path": str(paths["web_records"]),
            "records_sha256": web_sha,
            "record_count": len(web_rows),
            "licensed_included_count": licensed_web,
            "unknown_license_excluded_count": excluded_web,
            "probe_count": len(web_probes),
            "raw_response_directory": str(paths["web_raw_responses"]),
            "raw_response_directory_sha256": _sha256_directory(
                paths["web_raw_responses"]
            ),
            "specification_path": str(paths["web_specification"]),
            "specification_sha256": _sha256_file(paths["web_specification"]),
            "manifest_path": str(paths["web_manifest"]),
            "manifest_sha256": _sha256_file(paths["web_manifest"]),
        },
        "transfer_registry": {
            "panel_version": TRANSFER_PANEL_VERSION,
            "specification_path": str(paths["transfer_specification"]),
            "specification_sha256": registry_sha,
            "source_count": len(transfer_sources),
            "development_source_count": sum(
                source.split == "development" for source in transfer_sources
            ),
            "confirmation_source_count": sum(
                source.split == "confirmation" for source in transfer_sources
            ),
            "sources": transfer_status,
            "blockers": transfer_blockers,
            "all_snapshots_present": all_transfer_snapshots_present,
        },
        "phase_gate": {
            "layer1_frozen": base_ready,
            "layer2_registry_frozen": True,
            "layer2_snapshots_frozen": all_transfer_snapshots_present,
            "safe_to_begin_base_only_annotation": base_ready,
            "safe_to_begin_full_phase2": full_phase2_ready,
            "status": (
                "phase1-complete"
                if full_phase2_ready
                else "phase1-transfer-snapshots-pending"
                if base_ready
                else "phase1-blocked"
            ),
        },
    }


def _audit_surface_sources(
    provenance: Mapping[str, object],
    inputs: Sequence[SurfaceSnapshotInput] = (),
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    blockers: list[str] = []
    raw_sources = provenance.get("sources", ())
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        return rows, ["surface provenance sources are malformed"]
    provenance_source_ids = {
        str(value.get("source_id", "")).strip()
        for value in raw_sources
        if isinstance(value, Mapping)
    }
    input_by_id: dict[str, SurfaceSnapshotInput] = {}
    for item in inputs:
        if item.source_id not in provenance_source_ids:
            raise ValueError(f"unknown surface source input: {item.source_id}")
        if item.source_id in input_by_id:
            raise ValueError(f"duplicate surface source input: {item.source_id}")
        input_by_id[item.source_id] = item
    for value in raw_sources:
        if not isinstance(value, Mapping):
            blockers.append("surface provenance contains a non-object source")
            continue
        source_id = str(value.get("source_id", "")).strip()
        provenance_path = Path(str(value.get("resolved_file", ""))).expanduser()
        supplied = input_by_id.get(source_id)
        raw_path = supplied.path.expanduser().resolve() if supplied else provenance_path
        expected_sha = str(value.get("verified_sha256", value.get("sha256", "")))
        present = raw_path.is_file()
        actual_sha = _sha256_file(raw_path) if present else None
        verified = bool(present and expected_sha and actual_sha == expected_sha)
        if not source_id or not str(value.get("revision", "")).strip():
            blockers.append("surface provenance omits source ID or revision")
        if not verified:
            blockers.append(f"surface raw snapshot is missing or changed: {source_id}")
        rows.append(
            {
                "source_id": source_id,
                "revision": value.get("revision"),
                "license": value.get("license"),
                "dataset_card": value.get("dataset_card"),
                "local_path": str(raw_path),
                "provenance_resolved_file": str(provenance_path),
                "path_override_used": supplied is not None,
                "snapshot_present": present,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "verified": verified,
            }
        )
    return rows, blockers


def _audit_transfer_inputs(
    sources: Sequence[TransferSource],
    inputs: Sequence[TransferSnapshotInput],
) -> tuple[list[dict[str, object]], list[str]]:
    source_by_id = {source.source_id: source for source in sources}
    input_by_id: dict[str, TransferSnapshotInput] = {}
    blockers: list[str] = []
    for item in inputs:
        if item.source_id not in source_by_id:
            raise ValueError(f"unknown transfer source input: {item.source_id}")
        if item.source_id in input_by_id:
            raise ValueError(f"duplicate transfer source input: {item.source_id}")
        input_by_id[item.source_id] = item
    rows: list[dict[str, object]] = []
    for source in sources:
        supplied = input_by_id.get(source.source_id)
        path = supplied.path.expanduser().resolve() if supplied else None
        present = bool(path and path.exists())
        snapshot_size = _source_size(path) if present and path else None
        revision = supplied.revision.strip() if supplied else ""
        frozen = bool(present and snapshot_size and revision)
        if not frozen:
            blockers.append(f"transfer snapshot pending: {source.source_id}")
        rows.append(
            {
                **asdict(source),
                "local_path": str(path) if path else None,
                "source_revision": revision or None,
                "snapshot_present": present,
                "snapshot_size_bytes": snapshot_size,
                "snapshot_sha256": (
                    _sha256_source(path) if present and path else None
                ),
                "frozen": frozen,
            }
        )
    return rows, blockers


def _counts_by_source_and_split(
    items: Sequence[SemanticReadinessItem],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.source_name, item.split)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _counts_by_split(items: Sequence[SemanticReadinessItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.split] = counts.get(item.split, 0) + 1
    return counts


def _cross_split_groups(items: Sequence[SemanticReadinessItem]) -> tuple[str, ...]:
    splits: dict[str, set[str]] = {}
    for item in items:
        splits.setdefault(item.group_id, set()).add(item.split)
    return tuple(sorted(group for group, values in splits.items() if len(values) > 1))


def _web_record(row: Mapping[str, object]) -> WebTextRecord:
    values = dict(row)
    for key in ("tags", "retrieval_probe_ids", "retrieval_sampling_regions"):
        values[key] = tuple(values.get(key, ()))
    return WebTextRecord(**values)  # type: ignore[arg-type]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _source_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    return tuple(sorted(item for item in path.rglob("*") if item.is_file()))


def _source_size(path: Path) -> int:
    return sum(item.stat().st_size for item in _source_files(path))


def _sha256_source(path: Path) -> str:
    return _sha256_file(path) if path.is_file() else _sha256_directory(path)


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in _source_files(path):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
