"""Claim-attribution judge v3: tasks, validators, aggregation and runner path, no LLM."""

import asyncio
import json

import pytest

from analysis.interpretability.pipeline import agentic_judging as judging
from analysis.interpretability.pipeline.agentic_judging import (
    CLAIMS_PROTOCOL,
    JudgeOutputError,
    JudgeOutputSchemaError,
    JudgeOutputSyntaxError,
    aggregate_claim_attribution,
    claim_span_match,
    claims_evidence,
    evidence_set_key,
    prepare_attribution,
    prepare_claim_extraction,
    prepare_fulfilment,
    prepare_relevance,
    presented_evidence_order,
    segment_sentences,
    validate_agentic_judgment,
    validate_attribution,
    validate_claim_extraction,
    validate_fulfilment,
    validate_relevance,
)
from analysis.scripts import run_acl_arr_vllm as runner

ANSWER = ("Asana's free plan allows up to 15 users. Trello uses boards and cards.\n"
          "- ClickUp has a free tier!\nIs that enough? Most tools offer mobile apps.")
ROWS = [{"url": f"https://site{i}.example/page", "title": f"Title {i}", "text": f"Snippet {i}"} for i in range(1, 5)]
REQUEST = "Which free project management tools suit a small team?"


def claim(cid, full=(), partial=(), against=()):
    return {"claim_id": cid, "full_support_evidence_ids": list(full),
            "partial_support_evidence_ids": list(partial), "contradicting_evidence_ids": list(against)}


def evidence():
    return claims_evidence(ROWS, master_seed=7)


# Segmentation, claim extraction and IDs ----------------------------------------------------

def test_segmentation_is_deterministic_with_exact_spans():
    sentences = segment_sentences(ANSWER)
    assert [row["sentence_id"] for row in sentences] == ["S001", "S002", "S003", "S004", "S005"]
    assert [row["text"] for row in sentences] == [
        "Asana's free plan allows up to 15 users.", "Trello uses boards and cards.",
        "- ClickUp has a free tier!", "Is that enough?", "Most tools offer mobile apps."]
    for row in sentences:
        assert ANSWER[row["start"]:row["end"]] == row["text"]
    assert segment_sentences(ANSWER) == sentences


def test_claim_extraction_assigns_ids_and_maps_back_to_the_answer():
    sentences = segment_sentences(ANSWER)
    raw = json.dumps({"claims": [
        {"sentence_id": "S001", "claim_text": "Asana's free plan allows up to 15 users"},
        {"sentence_id": "S002", "claim_text": "Trello uses boards"},
        {"sentence_id": "S002", "claim_text": "Trello uses cards"},
    ]})
    parsed = validate_claim_extraction(raw, sentences=sentences)
    assert [(c["claim_id"], c["source_sentence_id"], c["span_match"]) for c in parsed["claims"]] == [
        ("C001", "S001", "exact"), ("C002", "S002", "exact"), ("C003", "S002", "near")]


def test_claim_extraction_rejects_invented_or_disordered_claims():
    sentences = segment_sentences(ANSWER)
    with pytest.raises(JudgeOutputError, match="does not map back"):
        validate_claim_extraction({"claims": [{"sentence_id": "S001", "claim_text": "Jira is the best tool overall"}]},
                                  sentences=sentences)
    with pytest.raises(JudgeOutputError, match="sentence order"):
        validate_claim_extraction({"claims": [{"sentence_id": "S002", "claim_text": "Trello uses boards"},
                                              {"sentence_id": "S001", "claim_text": "Asana has a free plan"}]},
                                  sentences=sentences)
    with pytest.raises(JudgeOutputError, match="unknown sentence"):
        validate_claim_extraction({"claims": [{"sentence_id": "S099", "claim_text": "x"}]}, sentences=sentences)


def test_no_factual_claims_is_legal_and_aggregates_to_nothing():
    parsed = validate_claim_extraction({"claims": []}, sentences=segment_sentences("Hello! Is that enough?"))
    assert parsed == {"claims": []}
    result = aggregate_claim_attribution({"claims": []}, evidence_ids=["EA", "EB"])
    assert result["claim_count"] == 0 and result["realized_support_tiers"] == []
    assert result["unsupported_fraction"] is None


def test_span_match_levels():
    assert claim_span_match("free plan allows up to 15 users", "Asana's free plan allows up to 15 users.") == "exact"
    assert claim_span_match("Asana free plan allows 15 users", "Asana's free plan allows up to 15 users.") == "near"
    assert claim_span_match("Jira costs nothing", "Asana's free plan allows up to 15 users.") is None


def test_evidence_ids_are_opaque_stable_and_independent_of_order():
    first = evidence()
    again = claims_evidence(list(reversed(ROWS)), master_seed=7)
    assert {row.url: row.evidence_id for row in first} == {row.url: row.evidence_id for row in again}
    assert all(row.evidence_id.startswith("E") and "site" not in row.evidence_id for row in first)
    duplicates = claims_evidence([*ROWS, ROWS[0]], master_seed=7)
    assert len(duplicates) == len(ROWS)
    assert evidence_set_key(first) == evidence_set_key(again)


# Presentation order and blinding ------------------------------------------------------------

def test_presentation_order_is_deterministic_and_reverse_is_its_mirror():
    rows = evidence()
    primary = presented_evidence_order(rows, master_seed=7, order_key="k")
    assert presented_evidence_order(rows, master_seed=7, order_key="k") == primary
    reverse = presented_evidence_order(rows, master_seed=7, order_key="k", variant="reverse")
    assert reverse == primary[::-1]
    with pytest.raises(ValueError):
        presented_evidence_order(rows, master_seed=7, order_key="k", variant="random")


def test_attribution_hides_urls_and_relevance_shows_them_and_nothing_leaks():
    claims = [{"claim_id": "C001", "claim_text": "Asana has a free plan"}]
    j2 = prepare_attribution(claims, evidence(), master_seed=7, max_tokens=4096)
    j3 = prepare_relevance(prompt_text=REQUEST, evidence=evidence(), master_seed=7, max_tokens=4096)
    j1 = prepare_fulfilment(prompt_text=REQUEST, answer=ANSWER, max_tokens=64)
    assert "https://" not in j2["prompt"] and "site1.example" not in j2["prompt"]
    assert "https://site1.example/page" in j3["prompt"]
    assert "Snippet" not in j1["prompt"] and ANSWER in j1["prompt"]
    for item in (j1, j2, j3):
        for leak in ("qwen", "llama", "Parallel-Expansion", "Reactive-Snippet", "duckduckgo", "searxng",
                     "shuffled", "ablated", "axis", "GENERATOR RANKING"):
            assert leak.lower() not in item["prompt"].lower()
        assert item["base"]["protocol"] == CLAIMS_PROTOCOL and item["temperature"] == 0.0


def test_task_ids_are_stable_and_variant_specific():
    claims = [{"claim_id": "C001", "claim_text": "Asana has a free plan"}]
    first = prepare_attribution(claims, evidence(), master_seed=7, max_tokens=4096)
    again = prepare_attribution(claims, evidence(), master_seed=7, max_tokens=4096)
    reverse = prepare_attribution(claims, evidence(), master_seed=7, max_tokens=4096, variant="reverse")
    assert first["base"]["judge_task_id"] == again["base"]["judge_task_id"]
    assert runner._request_sha256(first) == runner._request_sha256(again)
    assert reverse["base"]["judge_task_id"] != first["base"]["judge_task_id"]
    assert reverse["base"]["presented_evidence_ids"] == first["base"]["presented_evidence_ids"][::-1]
    assert first["base"]["canonical_evidence_ids"] == [row.evidence_id for row in evidence()]
    extraction = prepare_claim_extraction(ANSWER, max_tokens=4096)
    assert extraction["base"]["judge_task_id"] == prepare_claim_extraction(ANSWER, max_tokens=4096)["base"]["judge_task_id"]
    assert "Snippet" not in extraction["prompt"]


# Validators ---------------------------------------------------------------------------------

def test_attribution_validator_rejects_unknown_duplicate_and_conflicting_ids():
    ids = ["EA", "EB"]
    ok = {"claims": [claim("C001", full=["EA"], against=["EB"])]}
    assert validate_attribution(json.dumps(ok), claim_ids=["C001"], evidence_ids=ids)["claims"][0]["contradicting_evidence_ids"] == ["EB"]
    with pytest.raises(JudgeOutputError, match="unknown evidence"):
        validate_attribution({"claims": [claim("C001", full=["EZ"])]}, claim_ids=["C001"], evidence_ids=ids)
    with pytest.raises(JudgeOutputError, match="duplicate"):
        validate_attribution({"claims": [claim("C001", full=["EA", "EA"])]}, claim_ids=["C001"], evidence_ids=ids)
    with pytest.raises(JudgeOutputError, match="two lists"):
        validate_attribution({"claims": [claim("C001", full=["EA"], partial=["EA"])]}, claim_ids=["C001"], evidence_ids=ids)
    with pytest.raises(JudgeOutputError, match="every claim exactly once"):
        validate_attribution({"claims": [claim("C002")]}, claim_ids=["C001"], evidence_ids=ids)


def test_validators_classify_syntax_and_schema_failures():
    with pytest.raises(JudgeOutputSyntaxError):
        validate_fulfilment('{"request_fulfillment": 4')
    with pytest.raises(JudgeOutputSchemaError):
        validate_fulfilment({"request_fulfillment": 6})
    with pytest.raises(JudgeOutputSchemaError):
        validate_fulfilment({"request_fulfillment": 4, "reason": "x"})
    with pytest.raises(JudgeOutputSchemaError):
        validate_attribution({"claims": [{**claim("C001"), "use_score": 3}]}, claim_ids=["C001"], evidence_ids=["EA"])
    with pytest.raises(JudgeOutputError, match="every evidence item"):
        validate_relevance({"ideal_relevance_ranking": ["EA"]}, evidence_ids=["EA", "EB"])
    assert validate_relevance({"ideal_relevance_ranking": ["EB", "EA"]}, evidence_ids=["EA", "EB"]) == {
        "ideal_relevance_ranking": ["EB", "EA"]}


# Deterministic aggregation ------------------------------------------------------------------

def test_one_claim_one_source():
    result = aggregate_claim_attribution({"claims": [claim("C001", full=["EA"])]}, evidence_ids=["EA", "EB"])
    assert result["evidence"]["EA"]["support_share"] == 1.0 and result["evidence"]["EA"]["realized_support_rank"] == 1
    assert result["evidence"]["EB"]["realized_support_rank"] is None
    assert result["realized_support_tiers"] == [["EA"]]


def test_redundant_sources_share_one_claim():
    result = aggregate_claim_attribution(
        {"claims": [claim("C001", full=["EA", "EB", "EC", "ED", "EE"])]}, evidence_ids=["EA", "EB", "EC", "ED", "EE"])
    assert all(result["evidence"][e]["support_credit_exact"] == "1/5" for e in "EA EB EC ED EE".split())
    assert result["realized_support_tiers"] == [["EA", "EB", "EC", "ED", "EE"]]
    assert all(result["evidence"][e]["realized_support_rank"] == 1 for e in "EA EB EC ED EE".split())


def test_several_claims_rank_by_fractional_credit():
    labels = {"claims": [claim("C001", full=["EC"]), claim("C002", full=["EC", "EB"]), claim("C003", full=["EB"]),
                         claim("C004", full=["EC"])]}
    result = aggregate_claim_attribution(labels, evidence_ids=["EA", "EB", "EC"])
    assert result["realized_support_tiers"] == [["EC"], ["EB"]]
    assert result["evidence"]["EC"]["support_credit_exact"] == "5/2"
    assert result["evidence"]["EC"]["support_share"] == pytest.approx(2.5 / 4)
    assert result["evidence"]["EB"]["realized_support_rank"] == 2


def test_partial_unsupported_and_contradiction_counts():
    labels = {"claims": [claim("C001", partial=["EA"]), claim("C002"), claim("C003", full=["EA"], against=["EB"]),
                         claim("C004", against=["EB"])]}
    result = aggregate_claim_attribution(labels, evidence_ids=["EA", "EB"])
    assert (result["fully_supported_claims"], result["partial_only_claims"], result["unsupported_claims"],
            result["contradicted_claims"]) == (1, 1, 2, 2)
    assert result["unsupported_fraction"] == 0.5 and result["contradiction_fraction"] == 0.5
    assert result["evidence"]["EB"]["contradiction_count"] == 2
    assert result["evidence"]["EA"]["partial_support_claim_count"] == 1
    assert result["realized_support_tiers"] == [["EA"]]


def test_exact_ties_stay_tied_and_ignore_presentation_order():
    labels = {"claims": [claim("C001", full=["EB"]), claim("C002", full=["EA"])]}
    one = aggregate_claim_attribution(labels, evidence_ids=["EA", "EB"])
    two = aggregate_claim_attribution(labels, evidence_ids=["EB", "EA"])
    assert one["realized_support_tiers"] == two["realized_support_tiers"] == [["EA", "EB"]]
    assert one["evidence"]["EA"]["realized_support_rank"] == one["evidence"]["EB"]["realized_support_rank"] == 1


def test_aggregation_is_deterministic_and_rejects_unknown_ids():
    labels = {"claims": [claim("C001", full=["EA", "EB"]), claim("C002", full=["EB"])]}
    assert aggregate_claim_attribution(labels, evidence_ids=["EA", "EB"]) == aggregate_claim_attribution(
        labels, evidence_ids=["EA", "EB"])
    with pytest.raises(ValueError):
        aggregate_claim_attribution({"claims": [claim("C001", full=["EZ"])]}, evidence_ids=["EA"])


# Runner path with mocked model responses ----------------------------------------------------

class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.prompts, self.seeds = list(outputs), [], []

    async def complete(self, *, prompt, schema_name, schema, temperature, max_tokens, seed):
        self.prompts.append(prompt)
        self.seeds.append(seed)
        raw, finish = self.outputs.pop(0)
        if isinstance(raw, Exception):
            raise raw
        return raw, {"completion_tokens": 10, "finish_reason": finish}


def run(item, client):
    return asyncio.run(runner._execute_one(item, client=client, fake=False))


def test_schema_retry_uses_the_identical_prompt_and_records_categories():
    item = prepare_fulfilment(prompt_text=REQUEST, answer=ANSWER, max_tokens=64)
    client = FakeClient([('{"request_fulfillment": 9}', "stop"), ('{"request_fulfillment": 4}', "stop")])
    result = run(item, client)
    assert result["ok"] is True and result["parsed_output"] == {"request_fulfillment": 4}
    assert client.prompts[0] == client.prompts[1] == item["prompt"]
    assert client.seeds[0] != client.seeds[1]
    assert result["failure_categories"] == ["schema"] and result["usage"]["finish_reason"] == "stop"


def test_truncated_output_is_a_failure_even_if_it_parses():
    item = prepare_fulfilment(prompt_text=REQUEST, answer=ANSWER, max_tokens=64)
    result = run(item, FakeClient([('{"request_fulfillment": 4}', "length"), ('{"request_fulfillment": 4}', "length")]))
    assert result["ok"] is False and result["failure_category"] == "truncation"
    assert result["failure_categories"] == ["truncation", "truncation"]


def test_transport_failure_is_categorized_and_bounded():
    item = prepare_fulfilment(prompt_text=REQUEST, answer=ANSWER, max_tokens=64)
    result = run(item, FakeClient([(ConnectionError("down"), None)]))
    assert result["ok"] is False and result["failure_category"] == "transport"


def test_legacy_items_keep_feedback_retries_and_no_categories():
    item = prepare_fulfilment(prompt_text=REQUEST, answer=ANSWER, max_tokens=64)
    legacy = {**item, "validation_feedback_contract": "search-experience-validation-feedback-v1"}
    client = FakeClient([('{"request_fulfillment": 9}', "stop"), ('{"request_fulfillment": 3}', "stop")])
    result = run(legacy, client)
    assert result["ok"] is True and "failure_categories" not in result
    assert client.prompts[1] != client.prompts[0] and "VALIDATION_FEEDBACK_JSON" in client.prompts[1]


# Protocol isolation -------------------------------------------------------------------------

def test_v3_outputs_are_not_accepted_as_v1_judgments_and_vice_versa():
    v3 = {"claims": [claim("C001", full=["E1"])]}
    with pytest.raises(ValueError):
        validate_agentic_judgment(v3, allowed_evidence_ids=["E1"])
    v1 = {"request_fulfillment": 3, "evidence_grounding": 3, "ideal_relevance_ranking": ["E1"],
          "realized_support_ranking": [{"evidence_id": "E1", "use_score": 3}], "unsupported_claim_count": 0,
          "judge_confidence": 3}
    with pytest.raises(JudgeOutputSchemaError):
        validate_attribution(v1, claim_ids=["C001"], evidence_ids=["E1"])
    assert CLAIMS_PROTOCOL not in judging.SUPPORTED_FORMAT_VERSIONS


def test_v3_schemas_avoid_keywords_the_vllm_grammar_rejects():
    # vLLM/xgrammar answers HTTP 400 to uniqueItems/prefixItems; validators enforce uniqueness instead.
    def keys(value):
        if isinstance(value, dict):
            yield from value
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    schemas = [judging.claim_extraction_schema(["S001"]), judging.fulfilment_schema(),
               judging.attribution_schema(["C001"], ["E1", "E2"]), judging.attribution_schema(["C001"], []),
               judging.relevance_schema(["E1", "E2"]), judging.relevance_schema([])]
    for schema in schemas:
        assert not {"uniqueItems", "prefixItems"} & set(keys(schema))
    with pytest.raises(judging.JudgeOutputError, match="duplicate"):
        judging.validate_relevance({"ideal_relevance_ranking": ["E1", "E1"]}, evidence_ids=["E1", "E2"])
