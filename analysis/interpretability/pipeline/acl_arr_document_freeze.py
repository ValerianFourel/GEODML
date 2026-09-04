"""Freeze extracted page text onto an immutable search-result snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


def build_frozen_document_sets(
    serp_rows: Sequence[Mapping[str, Any]],
    page_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_documents: int,
    maximum_documents: int,
    max_document_characters: int,
    allow_snippet_fallback: bool,
    search_snapshot_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join extracted text by URL and fail if any keyword pool is incomplete."""

    if minimum_documents <= 0 or maximum_documents < minimum_documents:
        raise ValueError("document count limits are invalid")
    if max_document_characters <= 0:
        raise ValueError("max_document_characters must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", search_snapshot_sha256) is None:
        raise ValueError("search_snapshot_sha256 must be a lowercase SHA-256 digest")
    page_by_url: dict[str, str] = {}
    for index, row in enumerate(page_rows, 1):
        url = str(row.get("url", "") or "").strip()
        text = _page_text(row)
        if not url or not text:
            continue
        if url in page_by_url and page_by_url[url] != text:
            raise ValueError(f"conflicting extracted page text for URL at row {index}")
        page_by_url[url] = text

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for index, row in enumerate(serp_rows, 1):
        keyword = str(row.get("keyword", "") or "").strip()
        url = str(row.get("url", "") or "").strip()
        position_value = row.get("position")
        if position_value is None:
            raise ValueError(f"SERP row {index} has no position")
        try:
            position = int(position_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SERP row {index} has an invalid position") from exc
        if not keyword or not url or position <= 0:
            raise ValueError(f"SERP row {index} is incomplete")
        grouped.setdefault(keyword, []).append(row)
    if not grouped:
        raise ValueError("search snapshot contains no keyword rows")

    output: list[dict[str, Any]] = []
    incomplete: dict[str, int] = {}
    fallback_count = 0
    missing_page_count = 0
    for keyword in sorted(grouped):
        valid: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for row in sorted(grouped[keyword], key=lambda value: int(value["position"])):
            url = str(row["url"]).strip()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            full_text = page_by_url.get(url, "").strip()
            text_source = "extracted_page"
            if not full_text:
                missing_page_count += 1
                snippet = str(row.get("snippet", "") or "").strip()
                if not allow_snippet_fallback or not snippet:
                    continue
                full_text = snippet
                text_source = "serp_snippet"
                fallback_count += 1
            used_text = full_text[:max_document_characters]
            valid.append(
                {
                    "document_id": f"C{len(valid) + 1:03d}",
                    "natural_position": int(row["position"]),
                    "title": str(row.get("title", "") or ""),
                    "url": url,
                    "text": used_text,
                    "text_sha256": _hash(used_text),
                    "full_text_sha256": _hash(full_text),
                    "full_text_characters": len(full_text),
                    "used_text_characters": len(used_text),
                    "truncated": len(used_text) < len(full_text),
                    "text_source": text_source,
                }
            )
            if len(valid) >= maximum_documents:
                break
        if len(valid) < minimum_documents:
            incomplete[keyword] = len(valid)
            continue
        identity = {
            "keyword": keyword,
            "search_snapshot_sha256": search_snapshot_sha256,
            "documents": valid,
        }
        first = grouped[keyword][0]
        output.append(
            {
                "candidate_set_id": "document-set-"
                + _hash(_canonical(identity))[:20],
                "keyword": keyword,
                "search_query": keyword,
                "search_engine": str(
                    first.get("search_engine", first.get("search_backend", "unknown"))
                ),
                "search_snapshot_sha256": search_snapshot_sha256,
                "documents": valid,
            }
        )
    if incomplete:
        preview = ", ".join(
            f"{keyword!r}:{count}" for keyword, count in list(incomplete.items())[:8]
        )
        raise ValueError(
            f"incomplete frozen document pools for {len(incomplete)} keyword(s): {preview}"
        )
    summary = {
        "format_version": "acl-arr-frozen-document-sets-v1",
        "search_keyword_count": len(grouped),
        "complete_keyword_count": len(output),
        "document_count": sum(len(row["documents"]) for row in output),
        "minimum_documents": minimum_documents,
        "maximum_documents": maximum_documents,
        "max_document_characters": max_document_characters,
        "missing_extracted_page_count": missing_page_count,
        "snippet_fallback_enabled": allow_snippet_fallback,
        "snippet_fallback_count": fallback_count,
        "model_native_web_search": False,
    }
    return output, summary


def _page_text(row: Mapping[str, Any]) -> str:
    for field in ("text", "extracted_text", "body_text", "content", "page_text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = ["build_frozen_document_sets"]
