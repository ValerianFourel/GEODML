"""Versioned external-source transfer panel for semantic readiness mapping.

The source datasets are sampling frames, never readiness labels.  Every source
is assigned wholly to development or confirmation before labels are observed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .semantic_readiness_dataset import (
    LABEL_RUBRIC_VERSION,
    SEMANTIC_DATASET_VERSION,
    SemanticReadinessItem,
    is_semantic_readiness_text_eligible,
    normalize_semantic_readiness_text,
)


TRANSFER_PANEL_VERSION = "semantic-readiness-transfer-panel-v1"
DEFAULT_TRANSFER_SPEC = (
    Path(__file__).resolve().parent
    / "specs"
    / "semantic_readiness_transfer_sources_v1.json"
)


@dataclass(frozen=True, slots=True)
class TransferSource:
    source_id: str
    adapter: str
    source_kind: str
    split: str
    license: str
    redistribution_policy: str
    access: str
    source_url: str
    sampling_role: str


@dataclass(frozen=True, slots=True)
class RawTransferPrompt:
    source_record_id: str
    text: str
    group_id: str
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class TransferPromptRecord:
    transfer_record_id: str
    source_id: str
    source_record_id: str
    text: str
    text_sha256: str
    split: str
    group_id: str
    source_url: str | None
    license: str
    redistribution_policy: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class TransferSourceDiagnostics:
    source_id: str
    split: str
    raw_row_count: int
    extracted_prompt_count: int
    eligible_prompt_count: int
    selected_prompt_count: int


@dataclass(frozen=True, slots=True)
class TransferMergeDiagnostics:
    base_count: int
    proposed_transfer_count: int
    included_transfer_count: int
    duplicate_of_base_count: int
    cross_source_duplicate_count: int
    expanded_count: int


def load_transfer_source_specification(
    path: str | Path = DEFAULT_TRANSFER_SPEC,
) -> tuple[TransferSource, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("specification_version") != TRANSFER_PANEL_VERSION:
        raise ValueError("unexpected readiness transfer specification version")
    if payload.get("base_corpus_version") != SEMANTIC_DATASET_VERSION:
        raise ValueError("transfer specification targets another base corpus version")
    if payload.get("label_rubric_version") != LABEL_RUBRIC_VERSION:
        raise ValueError("transfer specification targets another label rubric")
    sources = tuple(TransferSource(**row) for row in payload.get("sources", ()))
    if not sources or len({item.source_id for item in sources}) != len(sources):
        raise ValueError("transfer sources must be non-empty and uniquely identified")
    if any(item.split not in {"development", "confirmation"} for item in sources):
        raise ValueError("every transfer source needs a valid split")
    allowed_adapters = {
        "amazon-shopping-queries",
        "ccpe",
        "chat-conversations",
        "ms-marco",
        "oasst1",
        "schema-guided-dialogue",
        "taskmaster",
    }
    if any(item.adapter not in allowed_adapters for item in sources):
        raise ValueError("transfer specification contains an unknown adapter")
    required_text = (
        "source_id",
        "source_kind",
        "license",
        "redistribution_policy",
        "access",
        "source_url",
        "sampling_role",
    )
    if any(
        not str(getattr(item, field)).strip()
        for item in sources
        for field in required_text
    ):
        raise ValueError("transfer source metadata must be non-empty")
    return sources


def build_transfer_prompt_panel(
    rows_by_source: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    source_revisions: Mapping[str, str],
    sources: Sequence[TransferSource],
    maximum_per_source: int = 1_000,
    master_seed: int = 20260817,
) -> tuple[tuple[TransferPromptRecord, ...], tuple[TransferSourceDiagnostics, ...]]:
    """Extract and deterministically bottom-hash sample each supplied source."""

    if maximum_per_source < 1:
        raise ValueError("maximum_per_source must be positive")
    source_by_id = {item.source_id: item for item in sources}
    unknown = sorted(set(rows_by_source) - set(source_by_id))
    if unknown:
        raise ValueError(f"unknown transfer sources: {unknown}")
    missing_revisions = sorted(
        source_id
        for source_id in rows_by_source
        if not str(source_revisions.get(source_id, "")).strip()
    )
    if missing_revisions:
        raise ValueError(f"missing source revisions: {missing_revisions}")

    records: list[TransferPromptRecord] = []
    diagnostics: list[TransferSourceDiagnostics] = []
    for source in sources:
        if source.source_id not in rows_by_source:
            continue
        raw_count = 0
        extracted_count = 0
        eligible_count = 0
        selected_by_hash: dict[str, tuple[int, RawTransferPrompt]] = {}
        maximum_priority_heap: list[tuple[int, str]] = []
        for row_index, row in enumerate(rows_by_source[source.source_id]):
            raw_count += 1
            for prompt in extract_transfer_prompts(source, row, row_index=row_index):
                extracted_count += 1
                text = normalize_semantic_readiness_text(prompt.text)
                if not is_semantic_readiness_text_eligible(text):
                    continue
                eligible_count += 1
                text_hash = _hash(text)
                priority = int(
                    _hash(f"{master_seed}:{source.source_id}:{text_hash}"),
                    16,
                )
                if text_hash in selected_by_hash:
                    continue
                candidate = RawTransferPrompt(
                    source_record_id=prompt.source_record_id,
                    text=text,
                    group_id=prompt.group_id,
                    source_url=prompt.source_url,
                )
                if len(selected_by_hash) < maximum_per_source:
                    selected_by_hash[text_hash] = (priority, candidate)
                    heapq.heappush(maximum_priority_heap, (-priority, text_hash))
                    continue
                worst_priority = -maximum_priority_heap[0][0]
                if priority < worst_priority:
                    _, removed_hash = heapq.heapreplace(
                        maximum_priority_heap,
                        (-priority, text_hash),
                    )
                    del selected_by_hash[removed_hash]
                    selected_by_hash[text_hash] = (priority, candidate)
        selected = sorted(
            selected_by_hash.items(),
            key=lambda item: (item[1][0], item[0]),
        )
        revision = str(source_revisions[source.source_id]).strip()
        for text_hash, (_, prompt) in selected:
            identity = f"{source.source_id}:{prompt.source_record_id}:{text_hash}"
            records.append(
                TransferPromptRecord(
                    transfer_record_id=f"transfer-prompt:{_hash(identity)[:24]}",
                    source_id=source.source_id,
                    source_record_id=prompt.source_record_id,
                    text=prompt.text,
                    text_sha256=text_hash,
                    split=source.split,
                    group_id=prompt.group_id,
                    source_url=prompt.source_url,
                    license=source.license,
                    redistribution_policy=source.redistribution_policy,
                    source_revision=revision,
                )
            )
        diagnostics.append(
            TransferSourceDiagnostics(
                source_id=source.source_id,
                split=source.split,
                raw_row_count=raw_count,
                extracted_prompt_count=extracted_count,
                eligible_prompt_count=eligible_count,
                selected_prompt_count=len(selected),
            )
        )
    return (
        tuple(sorted(records, key=lambda item: (item.source_id, item.transfer_record_id))),
        tuple(diagnostics),
    )


def extend_semantic_readiness_corpus(
    base_items: Sequence[SemanticReadinessItem],
    transfer_records: Sequence[TransferPromptRecord],
    *,
    sources: Sequence[TransferSource],
) -> tuple[
    tuple[SemanticReadinessItem, ...],
    tuple[SemanticReadinessItem, ...],
    TransferMergeDiagnostics,
]:
    """Append exact-new transfer texts without changing any frozen base row."""

    source_by_id = {item.source_id: item for item in sources}
    source_order = {item.source_id: index for index, item in enumerate(sources)}
    if len({item.item_id for item in base_items}) != len(base_items):
        raise ValueError("base corpus contains duplicate item IDs")
    if len({item.text_sha256 for item in base_items}) != len(base_items):
        raise ValueError("base corpus contains duplicate text hashes")
    unknown = sorted({item.source_id for item in transfer_records} - set(source_by_id))
    if unknown:
        raise ValueError(f"transfer records use unknown sources: {unknown}")
    if len({item.transfer_record_id for item in transfer_records}) != len(
        transfer_records
    ):
        raise ValueError("transfer records contain duplicate record IDs")
    revisions_by_source = {}
    for item in transfer_records:
        revisions_by_source.setdefault(item.source_id, set()).add(item.source_revision)
    inconsistent_revisions = sorted(
        source_id
        for source_id, revisions in revisions_by_source.items()
        if len(revisions) != 1 or not str(next(iter(revisions), "")).strip()
    )
    if inconsistent_revisions:
        raise ValueError(
            f"transfer sources mix or omit revisions: {inconsistent_revisions}"
        )

    seen_hashes = {item.text_sha256 for item in base_items}
    duplicate_of_base = 0
    cross_source_duplicate = 0
    transfer_items = []
    ordered_records = sorted(
        transfer_records,
        key=lambda item: (
            source_order[item.source_id],
            item.transfer_record_id,
        ),
    )
    for record in ordered_records:
        source = source_by_id[record.source_id]
        if (
            record.split != source.split
            or record.license != source.license
            or record.redistribution_policy != source.redistribution_policy
        ):
            raise ValueError(f"transfer provenance disagrees for {record.source_id}")
        actual_hash = _hash(normalize_semantic_readiness_text(record.text))
        if record.text_sha256 != actual_hash:
            raise ValueError(f"transfer text hash mismatch: {record.transfer_record_id}")
        if record.text_sha256 in seen_hashes:
            if any(
                item.text_sha256 == record.text_sha256 for item in base_items
            ):
                duplicate_of_base += 1
            else:
                cross_source_duplicate += 1
            continue
        seen_hashes.add(record.text_sha256)
        transfer_items.append(
            SemanticReadinessItem(
                item_id=f"semantic-item:{record.text_sha256[:24]}",
                source_kind=source.source_kind,
                source_name=source.source_id,
                source_record_id=record.source_record_id,
                text=record.text,
                text_sha256=record.text_sha256,
                split=source.split,
                group_id=record.group_id,
                source_url=record.source_url or source.source_url,
                author_name=None,
                author_url=None,
                license=source.license,
            )
        )
    transfer = tuple(sorted(transfer_items, key=lambda item: item.item_id))
    expanded = (*base_items, *transfer)
    diagnostics = TransferMergeDiagnostics(
        base_count=len(base_items),
        proposed_transfer_count=len(transfer_records),
        included_transfer_count=len(transfer),
        duplicate_of_base_count=duplicate_of_base,
        cross_source_duplicate_count=cross_source_duplicate,
        expanded_count=len(expanded),
    )
    return transfer, tuple(expanded), diagnostics


def extract_transfer_prompts(
    source: TransferSource,
    row: Mapping[str, object],
    *,
    row_index: int,
) -> tuple[RawTransferPrompt, ...]:
    """Normalize one dataset row into at most one first-user text."""

    adapter = source.adapter
    if adapter == "oasst1":
        if str(row.get("role", "")).casefold() not in {"prompter", "user"}:
            return ()
        if row.get("parent_id") not in {None, ""}:
            return ()
        language = str(row.get("lang", row.get("language", "en"))).casefold()
        if language and not language.startswith("en"):
            return ()
        if bool(row.get("synthetic", False)):
            return ()
        record_id = _record_id(row, row_index, "message_id", "id")
        return (_prompt(record_id, row.get("text", "")),)

    if adapter == "chat-conversations":
        language = str(row.get("language", row.get("lang", "en"))).casefold()
        if language and language not in {"en", "eng", "english"}:
            return ()
        conversations = row.get("conversation", row.get("conversations", ()))
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except json.JSONDecodeError:
                return ()
        record_id = _record_id(
            row,
            row_index,
            "conversation_hash",
            "conversation_id",
            "id",
        )
        text = _first_turn_text(conversations, roles={"user", "human", "prompter"})
        return (_prompt(record_id, text),) if text else ()

    if adapter in {"ccpe", "taskmaster", "schema-guided-dialogue"}:
        utterances = row.get("utterances", row.get("turns", ()))
        record_id = _record_id(
            row,
            row_index,
            "conversationId",
            "conversation_id",
            "dialogue_id",
            "id",
        )
        text = _first_turn_text(utterances, roles={"user", "human"})
        return (_prompt(record_id, text),) if text else ()

    if adapter == "ms-marco":
        record_id = _record_id(row, row_index, "query_id", "queryId", "id")
        return (_prompt(record_id, row.get("query", row.get("question", ""))),)

    if adapter == "amazon-shopping-queries":
        locale = str(row.get("product_locale", row.get("locale", "us"))).casefold()
        if locale not in {"us", "en_us", "en-us"}:
            return ()
        record_id = _record_id(row, row_index, "query_id", "queryId", "id")
        return (_prompt(record_id, row.get("query", "")),)

    raise ValueError(f"unsupported transfer adapter: {adapter}")


def _prompt(record_id: str, text: object) -> RawTransferPrompt:
    return RawTransferPrompt(
        source_record_id=record_id,
        text=str(text),
        group_id=f"source-record:{record_id}",
    )


def _first_turn_text(value: object, *, roles: set[str]) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ""
    for turn in value:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("role", turn.get("speaker", ""))).casefold()
        if role not in roles:
            continue
        return str(turn.get("content", turn.get("text", turn.get("utterance", ""))))
    return ""


def _record_id(
    row: Mapping[str, object],
    row_index: int,
    *keys: str,
) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    stable_row = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return f"row-{row_index:09d}-{_hash(stable_row)[:12]}"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
