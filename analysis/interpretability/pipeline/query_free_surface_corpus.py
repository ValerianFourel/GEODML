"""Acquire and characterize natural-language surface forms for Stage A.

The source prompts are a nuisance-coverage reservoir only. Their naturally
occurring intent is never an assigned decision-readiness treatment or a label.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence


SURFACE_CORPUS_VERSION = "query-free-surface-corpus-v2"

_SPACE = re.compile(r"\s+")
_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_QUESTION_OPENING = re.compile(
    r"^(?:who|what|when|where|why|how|which|is|are|am|was|were|do|does|did|"
    r"can|could|would|should|will|may|have|has|had)\b",
    re.IGNORECASE,
)
_POLITE_OPENING = re.compile(
    r"^(?:please|could you|would you|can you|may i ask you to)\b",
    re.IGNORECASE,
)
_CONDITIONAL_OPENING = re.compile(
    r"^(?:if|when|whenever|before|after|given|assuming|suppose|while|once)\b",
    re.IGNORECASE,
)
_IMPERATIVE_OPENINGS = {
    "analyze",
    "calculate",
    "classify",
    "complete",
    "compose",
    "create",
    "define",
    "describe",
    "design",
    "determine",
    "draft",
    "evaluate",
    "explain",
    "extract",
    "find",
    "generate",
    "give",
    "identify",
    "list",
    "provide",
    "rewrite",
    "show",
    "summarize",
    "translate",
    "write",
}
_READINESS_LEXICON = re.compile(
    r"\b(?:understand|learn|explore|explain|identify|compare|evaluate|shortlist|"
    r"select|choose|recommend|buy|purchase|acquire|adopt|deploy|execute|"
    r"implement|install|register|subscribe)\w*\b",
    re.IGNORECASE,
)
_HUMAN_TURN = re.compile(
    r"(?:^|\n\n)Human:\s*(.*?)(?=(?:\n\n)(?:Assistant|Human):|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RawSurfacePrompt:
    source_id: str
    source_record_id: str
    original_split: str
    source_category: str
    text: str
    has_attached_context: bool


@dataclass(frozen=True, slots=True)
class SurfaceCoverageRecord:
    surface_record_id: str
    source_id: str
    source_record_id: str
    original_split: str
    corpus_split: str
    source_category: str
    text: str
    text_sha256: str
    word_count: int
    length_band: str
    sentence_form: str
    perspective: str
    opening_pattern: str
    clause_count_proxy: int
    clause_band: str
    surface_family_id: str
    has_attached_context: bool
    readiness_lexicon_hits: tuple[str, ...]
    intended_use: str = "surface-style-coverage-only"
    eligible_as_semantic_label: bool = False


@dataclass(frozen=True, slots=True)
class SurfaceCorpusDiagnostics:
    raw_record_count: int
    unique_eligible_count: int
    selected_count: int
    development_count: int
    confirmation_count: int
    exact_duplicate_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    coverage_cells: tuple[tuple[str, int], ...]


def read_dolly_surface_prompts(path: str | Path) -> tuple[RawSurfacePrompt, ...]:
    """Read instruction fields from the pinned Dolly JSONL snapshot."""

    rows: list[RawSurfacePrompt] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            instruction = _normalize_text(payload.get("instruction", ""))
            rows.append(
                RawSurfacePrompt(
                    source_id="databricks-dolly-15k",
                    source_record_id=f"dolly:{index:06d}",
                    original_split="train",
                    source_category=str(payload.get("category", "unknown")),
                    text=instruction,
                    has_attached_context=bool(
                        _normalize_text(payload.get("context", ""))
                    ),
                )
            )
    return tuple(rows)


def read_hh_surface_prompts(path: str | Path) -> tuple[RawSurfacePrompt, ...]:
    """Read unique human turns from the pinned HH helpful-base gzip JSONL."""

    rows: list[RawSurfacePrompt] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            seen_in_pair: set[str] = set()
            turn_index = 0
            for field in ("chosen", "rejected"):
                for match in _HUMAN_TURN.finditer(str(payload.get(field, ""))):
                    text = _normalize_text(match.group(1))
                    normalized = text.casefold()
                    if not text or normalized in seen_in_pair:
                        continue
                    seen_in_pair.add(normalized)
                    rows.append(
                        RawSurfacePrompt(
                            source_id="anthropic-hh-helpful-base",
                            source_record_id=f"hh:{index:06d}:{turn_index:02d}",
                            original_split="train",
                            source_category="helpful-base-human-turn",
                            text=text,
                            has_attached_context=turn_index > 0,
                        )
                    )
                    turn_index += 1
    return tuple(rows)


def build_surface_coverage_corpus(
    raw_prompts: Sequence[RawSurfacePrompt],
    *,
    master_seed: int = 20260817,
    maximum_per_source: int = 2_000,
    confirmation_fraction: float = 0.20,
    minimum_words: int = 4,
    maximum_words: int = 80,
) -> tuple[tuple[SurfaceCoverageRecord, ...], SurfaceCorpusDiagnostics]:
    """Filter, exact-deduplicate, stratify, and freeze a coverage corpus."""

    if maximum_per_source < 1:
        raise ValueError("maximum_per_source must be positive")
    if not 0.0 < confirmation_fraction < 0.5:
        raise ValueError("confirmation_fraction must be in (0, 0.5)")
    if minimum_words < 1 or maximum_words < minimum_words:
        raise ValueError("invalid word-count bounds")

    rejection_counts: dict[str, int] = {}
    unique: dict[str, RawSurfacePrompt] = {}
    exact_duplicates = 0
    for item in raw_prompts:
        reason = _rejection_reason(
            item.text,
            minimum_words=minimum_words,
            maximum_words=maximum_words,
        )
        if reason:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        key = _normalized_dedup_key(item.text)
        if key in unique:
            exact_duplicates += 1
            continue
        unique[key] = item

    by_source: dict[str, list[RawSurfacePrompt]] = {}
    for item in unique.values():
        by_source.setdefault(item.source_id, []).append(item)
    selected_raw: list[RawSurfacePrompt] = []
    for source_id, items in sorted(by_source.items()):
        selected_raw.extend(
            _round_robin_surface_cells(
                items,
                limit=maximum_per_source,
                master_seed=master_seed,
                source_id=source_id,
            )
        )

    feature_rows = [_surface_features(item) for item in selected_raw]
    split_by_id = _assign_confirmation_splits(
        feature_rows,
        master_seed=master_seed,
        confirmation_fraction=confirmation_fraction,
    )
    records = tuple(
        SurfaceCoverageRecord(
            surface_record_id=row["surface_record_id"],
            source_id=row["source_id"],
            source_record_id=row["source_record_id"],
            original_split=row["original_split"],
            corpus_split=split_by_id[row["surface_record_id"]],
            source_category=row["source_category"],
            text=row["text"],
            text_sha256=row["text_sha256"],
            word_count=row["word_count"],
            length_band=row["length_band"],
            sentence_form=row["sentence_form"],
            perspective=row["perspective"],
            opening_pattern=row["opening_pattern"],
            clause_count_proxy=row["clause_count_proxy"],
            clause_band=row["clause_band"],
            surface_family_id=row["surface_family_id"],
            has_attached_context=row["has_attached_context"],
            readiness_lexicon_hits=row["readiness_lexicon_hits"],
        )
        for row in sorted(feature_rows, key=lambda value: value["surface_record_id"])
    )
    coverage: dict[str, int] = {}
    for item in records:
        key = "|".join(
            (
                item.source_id,
                item.corpus_split,
                item.sentence_form,
                item.length_band,
                item.perspective,
                item.opening_pattern,
                item.clause_band,
            )
        )
        coverage[key] = coverage.get(key, 0) + 1
    diagnostics = SurfaceCorpusDiagnostics(
        raw_record_count=len(raw_prompts),
        unique_eligible_count=len(unique),
        selected_count=len(records),
        development_count=sum(item.corpus_split == "development" for item in records),
        confirmation_count=sum(item.corpus_split == "confirmation" for item in records),
        exact_duplicate_count=exact_duplicates,
        rejection_counts=tuple(sorted(rejection_counts.items())),
        coverage_cells=tuple(sorted(coverage.items())),
    )
    return records, diagnostics


def _round_robin_surface_cells(
    items: Sequence[RawSurfacePrompt],
    *,
    limit: int,
    master_seed: int,
    source_id: str,
) -> tuple[RawSurfacePrompt, ...]:
    cells: dict[tuple[str, str, str], list[RawSurfacePrompt]] = {}
    for item in items:
        features = _surface_features(item)
        cell = (
            features["sentence_form"],
            features["length_band"],
            features["perspective"],
        )
        cells.setdefault(cell, []).append(item)
    for cell, rows in cells.items():
        rows.sort(
            key=lambda item: _hash(
                f"{SURFACE_CORPUS_VERSION}:{master_seed}:{source_id}:{cell}:"
                f"{item.source_record_id}:{item.text}"
            )
        )
    selected: list[RawSurfacePrompt] = []
    ordered_cells = sorted(cells)
    while len(selected) < min(limit, len(items)):
        added = False
        for cell in ordered_cells:
            if cells[cell] and len(selected) < limit:
                selected.append(cells[cell].pop(0))
                added = True
        if not added:
            break
    return tuple(selected)


def _assign_confirmation_splits(
    rows: Sequence[dict[str, object]],
    *,
    master_seed: int,
    confirmation_fraction: float,
) -> dict[str, str]:
    families: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        families.setdefault(str(row["surface_family_id"]), []).append(row)
    by_form: dict[str, list[str]] = {}
    for family_id, members in families.items():
        by_form.setdefault(str(members[0]["sentence_form"]), []).append(family_id)
    confirmation_families: set[str] = set()
    for sentence_form, family_ids in sorted(by_form.items()):
        ordered = sorted(
            family_ids,
            key=lambda family_id: _hash(
                f"{SURFACE_CORPUS_VERSION}:{master_seed}:family-split:"
                f"{sentence_form}:{family_id}"
            ),
        )
        confirmation_count = int(round(len(ordered) * confirmation_fraction))
        if len(ordered) >= 2:
            confirmation_count = max(1, min(len(ordered) - 1, confirmation_count))
        confirmation_families.update(ordered[:confirmation_count])
    assignments: dict[str, str] = {}
    for family_id, members in families.items():
        for row in members:
            record_id = row["surface_record_id"]
            assignments[record_id] = (
                "confirmation"
                if family_id in confirmation_families
                else "development"
            )
    return assignments


def _surface_features(item: RawSurfacePrompt) -> dict[str, object]:
    text = _normalize_text(item.text)
    words = _WORD.findall(text)
    count = len(words)
    lower = text.casefold()
    first = words[0].casefold() if words else ""
    if text.endswith("?") or _QUESTION_OPENING.search(text):
        sentence_form = "question"
    elif _POLITE_OPENING.search(text):
        sentence_form = "polite-request"
    elif first in _IMPERATIVE_OPENINGS:
        sentence_form = "imperative"
    else:
        sentence_form = "declarative-or-fragment"
    if count <= 12:
        length_band = "short"
    elif count <= 30:
        length_band = "medium"
    else:
        length_band = "long"
    if re.search(r"\b(?:i|me|my|mine|we|our|ours)\b", lower):
        perspective = "first-person"
    elif re.search(r"\b(?:you|your|yours)\b", lower):
        perspective = "second-person"
    else:
        perspective = "impersonal"
    if _CONDITIONAL_OPENING.search(text):
        opening_pattern = "condition-first"
    elif _POLITE_OPENING.search(text):
        opening_pattern = "politeness-first"
    elif sentence_form == "question":
        opening_pattern = "interrogative-first"
    elif sentence_form == "imperative":
        opening_pattern = "verb-first"
    else:
        opening_pattern = "other"
    readiness_hits = tuple(
        sorted({match.group(0).casefold() for match in _READINESS_LEXICON.finditer(text)})
    )
    clause_count = 1 + len(
        re.findall(
            r"[,;:]|\b(?:and|but|because|while|although)\b",
            lower,
        )
    )
    if clause_count <= 1:
        clause_band = "single"
    elif clause_count <= 3:
        clause_band = "few"
    else:
        clause_band = "many"
    family_signature = "|".join(
        (sentence_form, length_band, perspective, opening_pattern, clause_band)
    )
    text_hash = _hash(text)
    return {
        "surface_record_id": f"qf-surface:{text_hash[:24]}",
        "source_id": item.source_id,
        "source_record_id": item.source_record_id,
        "original_split": item.original_split,
        "source_category": item.source_category,
        "text": text,
        "text_sha256": text_hash,
        "word_count": count,
        "length_band": length_band,
        "sentence_form": sentence_form,
        "perspective": perspective,
        "opening_pattern": opening_pattern,
        "clause_count_proxy": clause_count,
        "clause_band": clause_band,
        "surface_family_id": f"qf-surface-family:{_hash(family_signature)[:24]}",
        "has_attached_context": item.has_attached_context,
        "readiness_lexicon_hits": readiness_hits,
    }


def _rejection_reason(
    text: str,
    *,
    minimum_words: int,
    maximum_words: int,
) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return "empty"
    words = _WORD.findall(normalized)
    if len(words) < minimum_words:
        return "too-short"
    if len(words) > maximum_words:
        return "too-long"
    if _URL.search(normalized) or _EMAIL.search(normalized):
        return "external-identifier"
    if "```" in normalized or "\x00" in normalized:
        return "code-or-control-content"
    printable = sum(character.isprintable() for character in normalized)
    if printable / len(normalized) < 0.98:
        return "non-printable-content"
    latin_letters = sum("a" <= character.casefold() <= "z" for character in normalized)
    all_letters = sum(character.isalpha() for character in normalized)
    if all_letters and latin_letters / all_letters < 0.80:
        return "non-english-script"
    return None


def _normalized_dedup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_text(text).casefold()).strip()


def _normalize_text(value: object) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
