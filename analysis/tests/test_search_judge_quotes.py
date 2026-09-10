import json
import unittest
from copy import deepcopy

from analysis.interpretability.pipeline import search_experience as contract
from analysis.tests.test_search_experience import fixture


class QuoteJudgeTests(unittest.TestCase):
    def setUp(self):
        plan, _, answer = fixture()
        self.visible = contract.prepare_judge_input(contract.prepare_case(plan, "p"),
            "ablated", answer, source_task_id="source", master_seed=42)
        self.raw = {**{k: 3 for k in contract.SCORE_FIELDS}, "claim_assessments": [{
            "claim_id": "c1", "support": "supported", "citation_correctness": "correct",
            "evidence": [{"document_id": "C003", "quote": "Café support."}]}]}

    def validate(self, value=None):
        return contract.validate_quote_judge_output(json.dumps(self.raw if value is None else value),
                                                    judge_input=self.visible)

    def test_unique_quote_resolves_unicode_offsets_without_changing_scores_or_text(self):
        before = deepcopy(self.visible)
        parsed = self.validate()
        evidence = parsed["claim_assessments"][0]["evidence"][0]
        self.assertEqual(evidence, {"document_id": "C003", "quote": "Café support.",
                                    "start": 22, "end": 35})
        self.assertEqual(contract.validate_judge_output(json.dumps(parsed), judge_input=self.visible), parsed)
        del evidence["start"], evidence["end"]
        self.assertEqual(parsed, self.raw)
        self.assertEqual(self.visible, before)

    def test_absent_ambiguous_and_title_only_quotes_are_not_repaired(self):
        for quote in ("invented", "Synthetic 3", "Cafe support."):
            with self.subTest(quote=quote):
                changed = deepcopy(self.raw)
                changed["claim_assessments"][0]["evidence"][0]["quote"] = quote
                with self.assertRaises(ValueError):
                    self.validate(changed)
        doc = next(d for d in self.visible["documents"] if d["document_id"] == "C003")
        doc["text"] += " Café support."
        with self.assertRaisesRegex(ValueError, "ambiguous|unique"):
            self.validate()

    def test_typographic_mismatch_reports_exact_source_candidate_without_accepting_it(self):
        doc = next(d for d in self.visible["documents"] if d["document_id"] == "C003")
        doc["text"] += ' Visit the "Courses" page.'
        changed = deepcopy(self.raw)
        changed["claim_assessments"][0]["evidence"][0]["quote"] = 'Visit the “Courses” page.'
        with self.assertRaisesRegex(
            ValueError,
            r'nearest_exact_source=\'Visit the "Courses" page\.\'',
        ):
            self.validate(changed)
        doc["text"] = "aaaa"
        self.raw["claim_assessments"][0]["evidence"][0]["quote"] = "aaa"
        with self.assertRaisesRegex(ValueError, "ambiguous|unique"):
            self.validate()

    def test_ambiguous_quote_reports_unique_exact_source_options_without_accepting_it(self):
        doc = next(d for d in self.visible["documents"] if d["document_id"] == "C003")
        doc["text"] = (
            'First instruction: visit the "Courses" page to register. '
            'Second instruction: visit the "Courses" page to review grades.'
        )
        changed = deepcopy(self.raw)
        changed["claim_assessments"][0]["evidence"][0]["quote"] = 'visit the "'
        with self.assertRaisesRegex(
            ValueError,
            r'exact_source_options=.*First instruction.*Second instruction',
        ):
            self.validate(changed)

    def test_wrong_ids_duplicate_missing_claims_and_extra_offsets_rejected(self):
        unknown = deepcopy(self.raw)
        unknown["claim_assessments"][0]["claim_id"] = "1"
        duplicate = deepcopy(self.raw)
        duplicate["claim_assessments"] *= 2
        missing = {**self.raw, "claim_assessments": []}
        removed = deepcopy(self.raw)
        removed["claim_assessments"][0]["evidence"][0]["document_id"] = "C001"
        offsets = deepcopy(self.raw)
        offsets["claim_assessments"][0]["evidence"][0].update(start=0, end=100)
        for changed in (unknown, duplicate, missing, removed, offsets):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.validate(changed)

    def test_schema_constrains_ids_and_abstention_without_empty_enum(self):
        schema = contract.judge_quote_schema(self.visible)
        assessments = schema["properties"]["claim_assessments"]
        self.assertEqual(assessments["minItems"], 1)
        self.assertEqual(assessments["maxItems"], 1)
        properties = assessments["items"]["properties"]
        self.assertEqual(properties["claim_id"]["enum"], ["c1"])
        quote = properties["evidence"]["items"]["properties"]
        self.assertEqual(set(quote["document_id"]["enum"]), {"C002", "C003"})
        self.assertEqual(set(quote), {"document_id", "quote"})
        self.visible["answer"] = {"status": "insufficient_evidence", "claims": [], "uncertainty": "Unknown."}
        abstain = {**self.raw, "claim_assessments": []}
        self.assertEqual(self.validate(abstain), abstain)
        with self.assertRaises(ValueError):
            self.validate()
        empty_schema = contract.judge_quote_schema(self.visible)
        self.assertEqual(empty_schema["properties"]["claim_assessments"]["maxItems"], 0)
        self.assertNotIn('"enum": []', json.dumps(empty_schema))

    def test_v1_stays_strict_and_v2_prompt_keeps_exact_blinded_input(self):
        with self.assertRaises(ValueError):
            contract.validate_judge_output(json.dumps(self.raw), judge_input=self.visible)
        prompt = contract.render_quote_judge_prompt(self.visible)
        self.assertTrue(prompt.endswith(json.dumps(self.visible, ensure_ascii=False)))
        self.assertNotIn("ABLATION_SENTINEL", prompt)
        self.assertNotIn("zero-based Unicode character offsets", prompt)
        self.assertIn("label the claim unsupported and return an empty evidence list", prompt)
        self.assertIn("Do not reconstruct text from headings", prompt)


if __name__ == "__main__":
    unittest.main()
