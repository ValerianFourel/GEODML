"""Synthetic contract fixtures; these are not scientific outputs."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import unittest

from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    AclArrExperimentPlan, ConditionAssignment, ExperimentPrompt, FrozenDocument,
    FrozenDocumentSet,
)
from analysis.interpretability.pipeline.search_experience import (
    prepare_case, render_answer_prompt, validate_answer_output, render_answer_display,
    prepare_judge_input, render_judge_prompt, validate_judge_output, SCORE_FIELDS,
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    question = "Which synthetic tool works offline?"
    prompt = ExperimentPrompt("p", "k", "synthetic tools", "t", 0, question,
                              digest(question), .5, 0., 1, .5, 0., 0.)
    texts = ("ABLATION_SENTINEL Tool A works offline.", "Tool B requires a network.",
             "Tool C works offline. Café support.")
    docs = tuple(FrozenDocument(f"C{i:03}", i, f"Synthetic {i}",
                 f"https://example.test/{i}", text, digest(text))
                 for i, text in enumerate(texts, 1))
    ids = tuple(d.document_id for d in docs)
    document_set = FrozenDocumentSet("set", prompt.keyword, prompt.keyword, "synthetic",
                                     "b" * 64, docs)
    assignment = ConditionAssignment("a", "p", "set", ids, ids[0], ids[1:],
                                     ids[1:] + ids[:1], "frozen", "perm")
    plan = AclArrExperimentPlan("plan", "acl-arr-document-experiment-v1", 42, 2,
        None, None, None, None, (prompt,), (document_set,), (assignment,), (), {})
    capture = {"keyword": prompt.keyword, "query": prompt.keyword,
        "query_parameters": {"q": prompt.keyword}, "synthetic": True,
        "raw_results": [{"position": d.natural_position, "title": d.title,
            "url": d.url, "snippet": "Synthetic search snippet", "text": d.text,
            "text_sha256": d.text_sha256} for d in docs]}
    answer = {"status": "answered", "claims": [{"claim_id": "c1",
        "text": "Tool C works offline.", "cited_document_ids": ["C003"]}],
        "uncertainty": ""}
    return plan, capture, answer


class SearchExperienceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("vllm"), "requires cluster vLLM installation")
    def test_installed_backend_accepts_all_search_schemas(self):
        from analysis.scripts.check_search_experience_grammar import main
        self.assertEqual(main(), 0)

    def test_citation_uniqueness_is_validated_outside_generation_schema(self):
        from analysis.interpretability.pipeline.search_experience import answer_schema
        citations = answer_schema()["properties"]["claims"]["items"]["properties"]["cited_document_ids"]
        # vLLM 0.28 rejects uniqueItems even when set to false.
        self.assertNotIn("uniqueItems", citations)
        _, _, answer = fixture()
        answer["claims"][0]["cited_document_ids"] = ["C003", "C003"]
        with self.assertRaises(ValueError):
            validate_answer_output(json.dumps(answer), allowed_document_ids=["C003"])
        answer["claims"][0]["cited_document_ids"] = ["C001"]
        with self.assertRaises(ValueError):
            validate_answer_output(json.dumps(answer), allowed_document_ids=["C003"])

    def test_frozen_capture_and_query_intents(self):
        plan, capture, _ = fixture()
        capture["raw_results"].append(deepcopy(capture["raw_results"][0]))
        capture["fetch_failure"] = {"url": "https://example.test/missing", "error": "timeout"}
        case = prepare_case(plan, "p", capture_rows=[capture])
        self.assertIs(case.assignment, plan.assignments[0])
        self.assertEqual(case.capture_rows[0], capture)
        self.assertEqual(case.query_intent["url_audit"][0]["duplicate_occurrences"], 1)
        self.assertEqual(case.query_intent["url_audit"][0]["selection"], "frozen_evidence")
        self.assertFalse(case.query_intent["query_executed_by_preparation"])
        self.assertEqual(case.query_intent["capture_query_status"], "verified")
        self.assertNotEqual(capture["raw_results"][0]["snippet"], case.document_set.documents[0].text)
        intent = prepare_case(plan, "p", query_contract="full-request-v1")
        self.assertEqual(intent.query_intent["query"], plan.prompts[0].question)
        self.assertEqual(intent.query_intent["capture_query_status"], "unavailable")
        with self.assertRaises(ValueError):
            prepare_case(plan, "p", capture_rows=[capture], query_contract="full-request-v1")
        relabeled = deepcopy(capture)
        relabeled["query"] = plan.prompts[0].question
        relabeled["query_parameters"]["q"] = plan.prompts[0].question
        with self.assertRaisesRegex(ValueError, "frozen search query"):
            prepare_case(plan, "p", capture_rows=[relabeled], query_contract="full-request-v1")
        with self.assertRaises(ValueError):
            prepare_case(plan, "p", capture_rows=[{"keyword": capture["keyword"], "raw_results": []}])

    def test_capture_mutations_rejected(self):
        plan, capture, _ = fixture()
        for field, value in (("text", "changed"), ("title", "changed"),
                             ("url", "https://example.test/changed"), ("url", "javascript:invalid"),
                             ("position", 99)):
            changed = deepcopy(capture)
            changed["raw_results"][0][field] = value
            if field == "text":
                changed["raw_results"][0]["text_sha256"] = digest(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_case(plan, "p", capture_rows=[changed])
        changed = deepcopy(capture)
        changed.update(question="changed", question_sha256=digest("changed"))
        with self.assertRaises(ValueError):
            prepare_case(plan, "p", capture_rows=[changed])
        changed = deepcopy(capture)
        changed["query_parameters"]["q"] = "changed"
        with self.assertRaises(ValueError):
            prepare_case(plan, "p", capture_rows=[changed])
        bad_prompt = replace(plan.prompts[0], question="changed")
        with self.assertRaises(ValueError):
            prepare_case(replace(plan, prompts=(bad_prompt,)), "p")

    def test_cited_display_and_empty_claim_abstention(self):
        _, _, answer = fixture()
        validated = validate_answer_output(json.dumps(answer), allowed_document_ids=["C003"])
        self.assertEqual(render_answer_display(validated), "Tool C works offline. [C003]")
        abstain = {"status": "insufficient_evidence", "claims": [],
                   "uncertainty": "The synthetic records do not establish this."}
        self.assertEqual(validate_answer_output(json.dumps(abstain), allowed_document_ids=[]), abstain)
        for mutation in ({**abstain, "uncertainty": ""}, {**answer, "claims": []}):
            with self.assertRaises(ValueError):
                validate_answer_output(json.dumps(mutation), allowed_document_ids=["C003"])
        for claim_text in ("Hidden [C001]", "Hidden [external]"):
            changed = deepcopy(answer)
            changed["claims"][0]["text"] = claim_text
            with self.assertRaises(ValueError):
                validate_answer_output(json.dumps(changed), allowed_document_ids=["C003"])
        with self.assertRaises(ValueError):
            validate_answer_output(json.dumps(answer), allowed_document_ids=["C002"])

    def test_blind_judge_excludes_ablated_document_and_metadata(self):
        plan, capture, answer = fixture()
        case = prepare_case(plan, "p", capture_rows=[capture])
        visible = prepare_judge_input(case, "ablated", answer, source_task_id="source", master_seed=42)
        self.assertEqual(set(visible), {"question", "answer", "documents"})
        self.assertNotIn("ABLATION_SENTINEL", render_judge_prompt(visible))
        self.assertNotIn("ABLATION_SENTINEL", render_answer_prompt(case, "ablated"))
        self.assertEqual({d["document_id"] for d in visible["documents"]}, {"C002", "C003"})
        self.assertEqual(visible, prepare_judge_input(case, "ablated", answer,
            source_task_id="source", master_seed=42))
        with self.assertRaises(ValueError):
            render_judge_prompt({**visible, "condition": "ablated"})

    def test_judge_exact_claim_and_quote_contract(self):
        plan, _, answer = fixture()
        visible = prepare_judge_input(prepare_case(plan, "p"), "ablated", answer,
                                      source_task_id="source", master_seed=42)
        judgment = {**{k: 5 for k in SCORE_FIELDS}, "claim_assessments": [{
            "claim_id": "c1", "support": "supported", "citation_correctness": "correct",
            "evidence": [{"document_id": "C003", "quote": "Tool C works offline.", "start": 0, "end": 21}]}]}
        self.assertEqual(validate_judge_output(json.dumps(judgment), judge_input=visible), judgment)
        for field, value in (("document_id", "C001"), ("quote", "invented"),
                             ("start", 1), ("end", True)):
            changed = deepcopy(judgment)
            changed["claim_assessments"][0]["evidence"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_judge_output(json.dumps(changed), judge_input=visible)
        for assessments in ([], judgment["claim_assessments"] * 2):
            with self.assertRaises(ValueError):
                validate_judge_output(json.dumps({**judgment, "claim_assessments": assessments}),
                                      judge_input=visible)
        abstain = {"status": "insufficient_evidence", "claims": [], "uncertainty": "Unknown."}
        abstain_input = prepare_judge_input(prepare_case(plan, "p"), "ablated", abstain,
                                            source_task_id="abstain", master_seed=42)
        validate_judge_output(json.dumps({**judgment, "claim_assessments": []}), judge_input=abstain_input)


if __name__ == "__main__":
    unittest.main()
