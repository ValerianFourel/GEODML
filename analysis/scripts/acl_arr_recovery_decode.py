"""Pilot-only constrained rerank recovery; the primary protocol is unchanged."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable


PROTOCOL = "acl-arr-pilot-enum-prefix-recovery-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _object_without_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _checked_ranking(raw: str, allowed: list[str], count: int, prefix: list[str]) -> list[str]:
    value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    if not isinstance(value, dict) or set(value) != {"ranked_document_ids"}:
        raise ValueError("rerank output must contain only ranked_document_ids")
    ranked = value["ranked_document_ids"]
    if not isinstance(ranked, list) or any(not isinstance(item, str) for item in ranked):
        raise ValueError("ranked_document_ids must be a string list")
    if len(ranked) != count:
        raise ValueError(f"expected {count} ranked document IDs")
    unknown = sorted(set(ranked) - set(allowed))
    if unknown:
        raise ValueError("ranking contains unknown document IDs: " + ", ".join(unknown))
    if ranked[:len(prefix)] != prefix:
        raise ValueError("ranking violates its frozen prefix")
    if set(ranked[len(prefix):]) & set(prefix):
        raise ValueError("ranking suffix repeats a frozen prefix ID")
    return ranked


def _schema(item: dict[str, Any], allowed: list[str], count: int, prefix: list[str]):
    schema = deepcopy(item["schema"])
    array = schema["properties"]["ranked_document_ids"]
    array["items"] = {"type": "string", "enum": [value for value in allowed if value not in prefix]}
    array["minItems"] = array["maxItems"] = count
    if prefix:
        array["prefixItems"] = [{"const": value} for value in prefix]
    return schema


async def recover_one(
    item: dict[str, Any],
    *,
    client,
    emit_attempt: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Regenerate a whole ranking, with at most output-count server requests.

    This explicitly changes the pilot decoding protocol, not the prompt or seed.
    Every response must obey its requested schema. Only duplicate IDs in the
    unconstrained suffix trigger regeneration. The unique prefix before the
    first duplicate becomes fixed, and cannot appear in the next suffix.

    ``emit_attempt`` must synchronously persist each record or raise. Logging
    errors deliberately propagate, so unaudited outputs cannot become successes.
    The supplied client must use one network attempt per completion.
    """
    started_at = _now()
    attempts = 0

    def failed(exc: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "base": item["base"],
            "error": f"{type(exc).__name__}: {exc}",
            "decode_attempt_count": attempts,
            "started_at": started_at,
            "finished_at": _now(),
        }

    try:
        if item["base"].get("pipeline") != "rerank":
            raise ValueError("recovery supports primary rerank tasks only")
        if getattr(client, "maximum_attempts", None) != 1:
            raise ValueError("recovery requires client maximum_attempts=1 for complete audit logging")
        allowed = item["base"]["input_document_ids"]
        if (not isinstance(allowed, list) or not allowed
                or any(not isinstance(value, str) or not value for value in allowed)
                or len(set(allowed)) != len(allowed)):
            raise ValueError("input_document_ids must be distinct nonempty strings")
        array = item["schema"]["properties"]["ranked_document_ids"]
        count = array["minItems"]
        if type(count) is not int or not 1 <= count <= len(allowed) or array["maxItems"] != count:
            raise ValueError("recovery requires a feasible exact output count")
        if "uniqueItems" in array or "prefixItems" in array:
            raise ValueError("recovery expects the unchanged primary rerank schema")
    except Exception as exc:
        return failed(exc)

    prefix: list[str] = []
    for attempt in range(1, count + 1):
        schema = _schema(item, allowed, count, prefix)
        attempt_start = _now()
        raw = None
        usage: dict[str, Any] = {}
        parsed = None
        next_prefix = None
        error = None
        attempts = attempt
        try:
            raw, response_usage = await client.complete(
                prompt=str(item["prompt"]),
                schema_name=str(item["schema_name"]),
                schema=schema,
                temperature=float(item["temperature"]),
                max_tokens=int(item["max_tokens"]),
                seed=int(item["seed"]),
            )
            usage = dict(response_usage)
            ranked = _checked_ranking(raw, allowed, count, prefix)
            unique_prefix: list[str] = []
            for document_id in ranked:
                if document_id in unique_prefix:
                    break
                unique_prefix.append(document_id)
            if len(unique_prefix) < count:
                if len(unique_prefix) <= len(prefix):
                    raise ValueError("duplicate regeneration did not grow its unique prefix")
                next_prefix = unique_prefix
                error = ValueError("ranked document IDs contain a duplicate")
            else:
                parsed = item["validator"](raw)
        except Exception as exc:
            error = exc

        finished_at = _now()
        emit_attempt({
            "protocol": PROTOCOL,
            "task_id": item["base"]["task_id"],
            "attempt": attempt,
            "maximum_decode_attempts": count,
            "schema": schema,
            "schema_sha256": _hash_text(json.dumps(schema, sort_keys=True, separators=(",", ":"))),
            "prefix": list(prefix),
            "next_prefix": next_prefix,
            "raw_output": raw,
            "raw_output_sha256": _hash_text(raw) if isinstance(raw, str) else None,
            "usage": usage,
            "validator_error": f"{type(error).__name__}: {error}" if error is not None else None,
            "started_at": attempt_start,
            "finished_at": finished_at,
        })
        if error is None:
            return {
                "ok": True,
                "base": item["base"],
                "raw_output": raw,
                "parsed_output": parsed,
                "usage": usage,
                "decode_attempt_count": attempts,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        if next_prefix is None:
            return failed(error)
        prefix = next_prefix
    return failed(ValueError("bounded duplicate regeneration exhausted"))
