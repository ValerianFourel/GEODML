"""Did Shuffled or Ablated change what the generator actually received? (from traces only)

Classifies each matched Natural/Shuffled and Natural/Ablated pair of the same
(generator, prompt, method, engine) into one observed change class instead of a
single "shuffle happened" flag. Reads recorded traces; runs no generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .agentic_judging import _trace_evidence

FINAL_PURPOSES = ("parallel_final", "reactive_action", "reactive_forced_finish")
# First matching class wins, from the largest to the smallest observed change.
CHANGE_CLASSES = ("not_assessable", "trajectory_divergence", "membership_or_content", "order",
                  "prompt_text_only", "none_at_generator_input")


def _events(trace: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    return [e.get("payload", {}) for e in trace.get("events", []) if e.get("event_type") == kind]


def trace_view(trace: Mapping[str, Any], method: str) -> dict[str, Any]:
    """What the generator saw: search queries, incoming order, final evidence and request."""

    calls = [p for p in _events(trace, "llm_call") if p.get("purpose") in FINAL_PURPOSES]
    evidence, _ = _trace_evidence(trace, method)
    return {
        "queries": [p.get("query") for p in _events(trace, "search")],
        "incoming_urls": [[row.get("url") for row in p.get("input_snippets", [])]
                          for p in _events(trace, "condition")],
        "conditioned_urls": [list(p.get("output_urls", [])) for p in _events(trace, "condition")],
        "searched_urls": {row.get("url") for p in _events(trace, "search") for row in p.get("snippets", [])},
        "evidence": [(row["url"], row["title"], row["text"]) for row in evidence],
        "final_prompt": (calls[-1].get("request") or {}).get("prompt") if calls else None,
    }


def classify_pair(natural: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Change class of ``other`` relative to ``natural`` (both ``trace_view`` results)."""

    if natural.get("final_prompt") is None or other.get("final_prompt") is None:
        change = "not_assessable"
    elif natural["queries"] != other["queries"]:
        change = "trajectory_divergence"
    elif set(natural["evidence"]) != set(other["evidence"]):
        change = "membership_or_content"
    elif natural["evidence"] != other["evidence"]:
        change = "order"
    elif natural["final_prompt"] != other["final_prompt"]:
        change = "prompt_text_only"
    else:
        change = "none_at_generator_input"
    incoming_changed = natural["conditioned_urls"] != other["conditioned_urls"]
    return {"change": change, "incoming_order_changed": incoming_changed,
            "erased_by_reranking": incoming_changed and change == "none_at_generator_input"}


def ablation_exposure(natural: Mapping[str, Any], ablated: Mapping[str, Any], target_url: str | None
                      ) -> dict[str, Any]:
    """Target exposure on the Natural path and what replaced it under ablation."""

    if not target_url:
        return {"exposure": "not_assessable"}
    natural_urls = [url for url, _, _ in natural["evidence"]]
    if target_url in natural_urls:
        exposure = "shown"
    elif target_url in natural["searched_urls"]:
        exposure = "retrieved_not_shown"
    else:
        exposure = "target_not_retrieved"
    ablated_urls = [url for url, _, _ in ablated["evidence"]]
    return {"exposure": exposure,
            "target_in_ablated_evidence": target_url in ablated_urls,
            "replacement_urls": [url for url in ablated_urls if url not in natural_urls],
            "removed_urls": [url for url in natural_urls if url not in ablated_urls]}


def audit_group(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit one (generator, prompt, method, engine) group of cells.

    Each cell needs ``condition``, ``method``, ``trace`` and optionally
    ``condition_audit`` (for the target URL). Missing conditions are reported.
    """

    by_condition = {cell["condition"]: cell for cell in cells}
    if "natural" not in by_condition:
        return {"status": "missing_natural", "conditions": sorted(by_condition)}
    views = {c: trace_view(cell["trace"], cell["method"]) for c, cell in by_condition.items()}
    out: dict[str, Any] = {"status": "ok", "conditions": sorted(by_condition)}
    if "shuffled" in views:
        out["shuffled"] = classify_pair(views["natural"], views["shuffled"])
    if "ablated" in views:
        target = (by_condition["ablated"].get("condition_audit") or {}).get("target_url")
        out["ablated"] = {**classify_pair(views["natural"], views["ablated"]),
                          **ablation_exposure(views["natural"], views["ablated"], target)}
    return out
