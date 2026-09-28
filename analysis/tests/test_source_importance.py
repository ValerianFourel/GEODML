"""Source-importance judge (SI-v1): preprocessing, blinding, validation and code-derived metrics."""

import asyncio
import json

import pytest

from analysis.interpretability.pipeline import source_importance as si
from analysis.interpretability.pipeline.agentic_judging import JudgeOutputError, JudgeOutputSchemaError
from analysis.scripts import run_acl_arr_vllm as runner

REQUEST = "How do I set up a free project board?"
ANSWER = ("## Steps\n- Create a free Trello account.\n- Add a board, then add lists. "
          "Do not pay for Premium yet.\nSee https://www.trello.com/guide for details [S2].")
SOURCE = {"url": "https://trello.com/guide", "title": "Trello guide",
          "text": "Trello’s free plan lets you create boards and lists. Premium is optional."}
URLS = ["https://trello.com/guide", "https://example.org/other"]


def item(**overrides):
    kwargs = dict(request=REQUEST, answer=ANSWER, source=SOURCE, observed_urls=URLS)
    kwargs.update(overrides)
    return si.prepare_source_task(**kwargs)


# Preprocessing -------------------------------------------------------------------------------

def test_units_keep_headings_bullets_negation_and_exact_offsets():
    units = si.unitize_answer(ANSWER)
    assert [u["text"] for u in units][:4] == [
        "## Steps", "- Create a free Trello account.", "- Add a board, then add lists.", "Do not pay for Premium yet."]
    assert all(ANSWER[u["start"]:u["end"]] == u["text"] for u in units)
    assert [u["unit_id"] for u in units] == [f"a{i}" for i in range(1, len(units) + 1)]


def test_masking_hides_only_source_identifying_markers_and_is_reversible():
    answer = "Use Trello [S2] (S1); see www.trello.com/x and https://docs.python.org/3/ for Python."
    masked, spans = si.mask_answer_citations(answer, URLS)
    assert masked == "Use Trello [ref] [ref]; see [ref] and https://docs.python.org/3/ for Python."
    assert [s["kind"] for s in spans] == ["evidence_id_marker", "evidence_id_marker", "source_link"]
    assert si.unmask_answer(masked, spans) == answer
    assert si.mask_answer_citations("No citations here.", URLS) == ("No citations here.", [])


def test_prompt_is_blind_to_the_source_url_and_cell_metadata():
    prepared = item()
    prompt = prepared["prompt"]
    for hidden in ("trello.com", "https://", "[S2]", "example.org", "Parallel", "Reactive", "qwen", "llama",
                   "shuffled", "ablated", "natural", "rank"):
        assert hidden.lower() not in prompt.split("ANSWER UNITS:")[1].lower()
    assert "Title: Trello guide" in prompt and "[ref]" in prompt
    assert prepared["base"]["answer_names_source_url"] is True
    assert prepared["base"]["source_url"] == SOURCE["url"]  # identity kept outside the prompt


def test_identity_ignores_url_but_tracks_every_visible_text_and_cap():
    same_text_other_url = item(source={**SOURCE, "url": "https://mirror.example/guide"},
                               observed_urls=["https://mirror.example/guide"],
                               answer=ANSWER.replace("https://www.trello.com/guide", "[ref]").replace("[S2]", "[ref]"))
    assert same_text_other_url["base"]["judge_task_id"] == item()["base"]["judge_task_id"]
    base = item()["base"]["judge_task_id"]
    for changed in (item(request=REQUEST + " Today."), item(answer=ANSWER + " Extra."),
                    item(source={**SOURCE, "text": SOURCE["text"] + "!"}), item(max_tokens=512)):
        assert changed["base"]["judge_task_id"] != base
    assert item()["seed"] == item()["seed"]


def test_schema_avoids_rejected_keywords_and_enumerates_units():
    schema = item()["schema"]
    text = json.dumps(schema)
    assert "uniqueItems" not in text and "prefixItems" not in text
    assert schema["properties"]["matches"]["items"]["properties"]["answer_unit_id"]["enum"][0] == "a1"
    assert list(schema["properties"]) == ["matches", "importance"]


def test_empty_source_is_unassessable_not_zero():
    assert not si.is_assessable({"title": " ", "text": ""})
    with pytest.raises(ValueError, match="unassessable"):
        item(source={"url": "https://x.org", "title": "", "text": ""})


# Output validation ---------------------------------------------------------------------------

def validate(output):
    prepared = item()
    return prepared["base"]["units"], prepared["validator"](json.dumps(output))


def match(unit="a2", answer_quote="Create a free Trello account", field="text",
          evidence_quote="Trello's free plan lets you create boards"):
    return {"answer_unit_id": unit, "answer_quote": answer_quote, "evidence_field": field,
            "evidence_quote": evidence_quote}


def test_positive_grade_with_exact_quotes_gets_offsets_and_tolerates_typography():
    units, result = validate({"matches": [match()], "importance": 4})
    assert result["importance"] == 4
    row = result["matches"][0]
    assert row["evidence_quote"] == "Trello’s free plan lets you create boards"  # canonical source text
    assert SOURCE["text"][row["evidence_start"]:row["evidence_end"]] == row["evidence_quote"]
    unit = next(u for u in units if u["unit_id"] == "a2")
    assert unit["text"][row["answer_start"]:row["answer_end"]] == "Create a free Trello account"


@pytest.mark.parametrize("output, error", [
    ({"matches": [match()], "importance": 0}, JudgeOutputError),
    ({"matches": [], "importance": 3}, JudgeOutputError),
    ({"matches": [match(answer_quote="Buy Premium now")], "importance": 2}, JudgeOutputError),
    ({"matches": [match(evidence_quote="Premium is required")], "importance": 2}, JudgeOutputError),
    ({"matches": [match(unit="a99")], "importance": 2}, JudgeOutputError),
    ({"matches": [match(), match()], "importance": 2}, JudgeOutputError),
    ({"matches": [], "importance": True}, JudgeOutputSchemaError),
    ({"matches": [], "importance": 6}, JudgeOutputSchemaError),
    ({"matches": [match(field="url")], "importance": 2}, JudgeOutputSchemaError),
    ({"matches": [], "importance": 0, "why": "x"}, JudgeOutputSchemaError),
])
def test_invalid_outputs_are_rejected_with_their_category(output, error):
    with pytest.raises(error):
        validate(output)


def test_zero_with_no_matches_is_a_valid_no_contribution_result():
    assert validate({"matches": [], "importance": 0})[1] == {"importance": 0, "matches": []}


# Runner path (mocked model) ------------------------------------------------------------------

class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.prompts, self.seeds = list(outputs), [], []

    async def complete(self, *, prompt, schema_name, schema, temperature, max_tokens, seed):
        self.prompts.append(prompt)
        self.seeds.append(seed)
        raw, finish = self.outputs.pop(0)
        if isinstance(raw, Exception):
            raise raw
        return raw, {"completion_tokens": 30, "finish_reason": finish}


def run(prepared, client):
    return asyncio.run(runner._execute_one(prepared, client=client, fake=False))


def test_one_identical_retry_after_a_bad_quote_then_success():
    prepared = item()
    bad = json.dumps({"matches": [match(evidence_quote="not in source")], "importance": 3})
    good = json.dumps({"matches": [match()], "importance": 4})
    client = FakeClient([(bad, "stop"), (good, "stop")])
    result = run(prepared, client)
    assert result["ok"] is True and result["parsed_output"]["importance"] == 4
    assert client.prompts == [prepared["prompt"]] * 2 and client.seeds[0] != client.seeds[1]
    assert result["failure_categories"] == ["semantic"]


def test_failures_stay_failures_and_are_never_scored_zero():
    zero = json.dumps({"matches": [], "importance": 0})
    truncated = run(item(), FakeClient([(zero, "length"), (zero, "length")]))
    assert truncated["ok"] is False and truncated["failure_category"] == "truncation"
    assert "parsed_output" not in truncated
    transport = run(item(), FakeClient([(ConnectionError("down"), None)]))
    assert transport["ok"] is False and transport["failure_category"] == "transport"


# Code-derived metrics ------------------------------------------------------------------------

def test_tau_b_matches_a_hand_computed_value_with_ties():
    assert si.kendall_tau_b([3, 2, 1], [2, 2, 1]) == pytest.approx(2 / 6 ** 0.5)
    assert si.kendall_tau_b([1, 2], [5, 5]) is None and si.kendall_tau_b([1], [1]) is None
    scipy = pytest.importorskip("scipy.stats")
    x, y = [0, -1, -2, -3, -4], [4, 4, 1, 0, 2]
    assert si.kendall_tau_b(x, y) == pytest.approx(scipy.kendalltau(x, y, variant="b").statistic)


def test_ties_define_the_top_group_and_zeros_stay_unranked():
    grades = {"s1": 4, "s2": 4, "s3": 0, "s4": 2}
    assert si.rank_groups(grades) == {"groups": [{"grade": 4, "sources": ["s1", "s2"]},
                                                 {"grade": 2, "sources": ["s4"]}], "zero": ["s3"]}
    m = si.cell_metrics(grades, generator_list=["s2", "s4"], presented=["s3", "s1", "s2", "s4"])
    assert m["top_source_alignment"] is True and m["first_presented_alignment"] is False
    assert m["ordering_tau_b"] == pytest.approx(1.0)
    assert m["important_source_omission"] == pytest.approx(0.5) and m["coverage"] == pytest.approx(0.5)
    assert m["list_in_presented_order"] is True


def test_undefined_metrics_have_reasons_and_missing_grades_are_not_zero():
    all_zero = si.cell_metrics({"a": 0, "b": 0}, generator_list=["a", "b"], presented=["a", "b"])
    assert all_zero["top_source_alignment"] is None and all_zero["reasons"]["top"] == "all_grades_zero"
    assert all_zero["ordering_tau_b"] is None and all_zero["important_source_omission"] is None
    empty = si.cell_metrics({"a": 3, "b": 1}, generator_list=[], presented=["a", "b"])
    assert empty["top_source_alignment"] is None and empty["reasons"]["top"] == "empty_generator_list"
    assert empty["coverage"] == 0.0
    missing = si.cell_metrics({"a": 5, "b": None}, generator_list=["a"], presented=["a", "b"])
    assert missing["complete"] is False and missing["reasons"] == {"all": "incomplete_cell"}
    assert missing["top_source_alignment"] is None and "rank_groups" not in missing


def test_generator_list_must_be_a_clean_subset_of_observed_sources():
    with pytest.raises(ValueError):
        si.cell_metrics({"a": 1}, generator_list=["a", "a"], presented=["a"])
    with pytest.raises(ValueError):
        si.cell_metrics({"a": 1}, generator_list=["z"], presented=["a"])
    with pytest.raises(ValueError):
        si.cell_metrics({"a": 1}, generator_list=[], presented=["a", "b"])
