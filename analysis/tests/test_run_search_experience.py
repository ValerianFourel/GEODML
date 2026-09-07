"""Synthetic execution fixtures never establish scientific results."""
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis.scripts import run_search_experience as runtime


class SearchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = [{"base": {"task_id": f"task-{i}", "pipeline": "answer", "fake_backend": True},
                       "prompt": f"unchanged question {i}", "schema_name": "test", "schema": {},
                       "temperature": 0.0, "max_tokens": 30, "seed": i,
                       "validator": json.loads} for i in range(3)]
        self.identity = {"model_id": "fixture/model", "model_revision": "a" * 40,
                         "pipeline": "search-primary-v1", "source_manifest_sha256": "b" * 64,
                         "fake_backend": True, "pilot_only": True, "maximum_attempts": 1}

    def execute(self, directory, **kwargs):
        class Client:
            async def complete(self, **request):
                return '{"value":1}', {"completion_tokens": 3}
        return asyncio.run(runtime.run_prepared(self.items, directory, client=Client(),
            identity=self.identity, source_hashes={}, **kwargs))

    def test_partial_resume_has_global_coverage_and_pilot_flags(self):
        output = self.root / "results"
        self.assertEqual(self.execute(output, max_tasks=1), 3)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["remaining_count"], 2)
        self.assertFalse(manifest["scientific_result"])
        self.assertEqual(self.execute(output, resume=True), 0)
        rows = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["fake_backend"] and not row["eligible_for_analysis"] for row in rows))

    def test_tampered_resume_preserves_original_bytes(self):
        output = self.root / "results"
        self.execute(output)
        path = output / "outcomes.jsonl"
        path.write_text(path.read_text() + "{")
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(ValueError):
            self.execute(output, resume=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_changed_source_rejected_before_output_creation(self):
        source = self.root / "capture.jsonl"
        source.write_text("changed")
        with self.assertRaises(ValueError):
            asyncio.run(runtime.run_prepared(self.items, self.root / "results", client=None,
                identity=self.identity, source_hashes={str(source): "a" * 64}))
        self.assertFalse((self.root / "results").exists())

    def test_self_judge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            runtime.validate_judge_model("fixture/model", "fixture/model")

    def test_resume_rejects_changed_payload_identity_and_pilot_flags(self):
        output = self.root / "results"
        self.execute(output)
        path = output / "outcomes.jsonl"
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        for field, value in (("raw_output", "{}"), ("parsed_output", {}),
                             ("request_sha256", "a" * 64), ("eligible_for_analysis", True)):
            changed = [dict(row) for row in rows]
            changed[0][field] = value
            path.write_text("".join(json.dumps(row) + "\n" for row in changed))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                self.execute(output, resume=True)
            self.assertEqual(path.read_bytes(), before)
        path.write_text(original + original.splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            self.execute(output, resume=True)

    def test_frozen_bundle_primary_judge_inspection_and_human_packets(self):
        from analysis.interpretability.pipeline.acl_arr_document_experiment import (
            build_acl_arr_experiment_plan, write_acl_arr_experiment_plan, iter_experiment_tasks,
        )
        from analysis.tests.test_acl_arr_document_experiment import _prompts, _axis_rows, _models
        from analysis.interpretability.pipeline import search_experience as contract
        model = _models()[0]
        prompts, axes, document_sets = [], [], []
        for index, label in enumerate(("ordinary", "long", "insufficient")):
            question = f"Explain the synthetic {label} tool's offline compatibility."
            question_hash = hashlib.sha256(question.encode()).hexdigest()
            prompts.append({**_prompts()[0], "candidate_id": label, "keyword": label,
                            "keyword_id": label, "target_id": label,
                            "question": question, "question_sha256": question_hash})
            axes.append({**_axis_rows()[0], "candidate_id": label, "text_sha256": question_hash,
                         "axis_1_rank": index, "axis_1_percentile_0_1": index / 2})
            documents = []
            for position in range(1, 12):
                text = (f"Synthetic {label} source {position} discusses offline support." if label != "insufficient"
                        else f"Synthetic unrelated source {position} describes a garden.")
                if label == "long":
                    suffix = f"END-{position:03d}"
                    text += "x" * (12000 - len(text) - len(suffix)) + suffix
                documents.append({"document_id": f"C{position:03d}", "natural_position": position,
                    "title": f"Synthetic {label} {position}", "url": f"https://example.test/{label}/{position}",
                    "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
            document_sets.append({"candidate_set_id": label, "keyword": label, "search_query": label,
                                  "search_engine": "synthetic", "documents": documents})
        plan = build_acl_arr_experiment_plan(prompts, axes, document_sets, models=(model,), top_n=10)
        artifacts = write_acl_arr_experiment_plan(self.root / "plan", plan=plan)
        captures = [{"keyword": ds.keyword, "query": ds.search_query, "synthetic": True,
                     "raw_results": [{"url": d.url, "title": d.title, "position": d.natural_position,
                         "snippet": "Synthetic distinct snippet", "text": d.text,
                         "text_sha256": d.text_sha256} for d in ds.documents]} for ds in plan.document_sets]
        capture_path = self.root / "captures.jsonl"
        capture_path.write_text("".join(json.dumps(row) + "\n" for row in captures))
        bundle = self.root / "bundle"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            runtime.prepare_bundle(artifacts.manifest_path, [p.prompt_id for p in plan.prompts], capture_path, bundle)
        runtime.prepare_bundle(artifacts.manifest_path, [p.prompt_id for p in plan.prompts], capture_path, bundle, synthetic=True)
        items, identity, hashes = runtime.primary_items(bundle, model.configuration_id)
        self.assertEqual(len(items), 18)
        _, _, cases, _ = runtime.load_bundle(bundle)
        old_tasks = {t.task_id: t for t in iter_experiment_tasks(plan)}
        responses = {}
        for item in items:
            case = cases[item["base"]["prompt_id"]]
            if case.prompt.prompt_id == "long":
                allowed = item["base"]["input_document_ids"]
                for doc in case.document_set.documents:
                    self.assertEqual(len(doc.text), 12000)
                    self.assertEqual(doc.text in item["prompt"], doc.document_id in allowed)
                self.assertGreater(len(item["prompt"]), 120000)
            if item["base"]["pipeline"] == "rerank":
                legacy = runtime._prepare_primary(old_tasks[item["base"]["legacy_task_id"]], prompt=case.prompt,
                    assignment=case.assignment, document_set=case.document_set, model=model)
                for key in ("prompt", "schema", "seed", "max_tokens", "temperature"):
                    self.assertEqual(item[key], legacy[key])
                responses[item["prompt"]] = item["fake_output"]
                self.assertEqual(len(json.loads(item["fake_output"])["ranked_document_ids"]), 10)
            elif case.prompt.prompt_id == "insufficient":
                responses[item["prompt"]] = json.dumps({"status": "insufficient_evidence", "claims": [],
                    "uncertainty": "The synthetic garden records do not establish offline compatibility."})
            else:
                doc_id = item["base"]["input_document_ids"][0]
                document = next(d for d in case.document_set.documents if d.document_id == doc_id)
                responses[item["prompt"]] = json.dumps({"status": "answered", "claims": [
                    {"claim_id": "c1", "text": document.text[:64], "cited_document_ids": [doc_id]}], "uncertainty": ""})
            item["base"]["fake_backend"] = True
        identity["fake_backend"] = True
        class Client:
            async def complete(self, **request):
                return responses[request["prompt"]], {"completion_tokens": 5}
        primary = self.root / "primary"
        self.assertEqual(asyncio.run(runtime.run_prepared(items, primary, client=Client(), identity=identity,
                                                         source_hashes=hashes)), 0)
        before = {p.name: p.read_bytes() for p in primary.iterdir()}
        self.assertEqual(asyncio.run(runtime.run_prepared(items, primary, client=Client(), identity=identity,
                                                         source_hashes=hashes, resume=True)), 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in primary.iterdir()})
        judges, judge_identity, judge_hashes = runtime.judge_items(bundle, primary, "independent/model", "c" * 40)
        self.assertEqual(len(judges), 9)
        again, _, _ = runtime.judge_items(bundle, primary, "independent/model", "c" * 40)
        self.assertEqual([runtime._request_sha256(i) for i in judges], [runtime._request_sha256(i) for i in again])
        for item in judges:
            value = json.loads(item["prompt"].split("\n\n", 1)[1])
            if not value["answer"]["claims"]:
                self.assertEqual(value["answer"]["status"], "insufficient_evidence")
                responses[item["prompt"]] = json.dumps({**{key: 3 for key in contract.SCORE_FIELDS}, "claim_assessments": []})
                continue
            claim = value["answer"]["claims"][0]
            doc = next(d for d in value["documents"] if d["document_id"] == claim["cited_document_ids"][0])
            responses[item["prompt"]] = json.dumps({**{key: 3 for key in contract.SCORE_FIELDS},
                "claim_assessments": [{"claim_id": "c1", "support": "supported", "citation_correctness": "correct",
                    "evidence": [{"document_id": doc["document_id"], "quote": doc["text"][:64], "start": 0, "end": len(doc["text"][:64])}]}]})
        judge_output = self.root / "judge"
        self.assertEqual(asyncio.run(runtime.run_prepared(judges, judge_output, client=Client(), identity=judge_identity,
                                                         source_hashes=judge_hashes)), 0)
        report = self.root / "report"
        records = runtime.inspect_bundle(bundle, primary, judge_output, report)
        self.assertEqual(len(records), 9)
        self.assertEqual({(row["prompt_id"], row["condition"]) for row in records},
                         {(label, condition) for label in ("ordinary", "long", "insufficient")
                          for condition in ("natural", "ablated", "shuffled")})
        self.assertTrue(all(row["complete"] for row in records))
        for row in records:
            self.assertEqual(len(row["ranking"]["ranked_document_ids"]), 10)
            if row["prompt_id"] == "insufficient":
                self.assertEqual(row["answer"]["claims"], [])
                self.assertEqual(row["judgment"]["claim_assessments"], [])
            if row["prompt_id"] == "long":
                self.assertTrue(all(len(doc["text"]) == 12000 for doc in row["documents"]))
        self.assertEqual(records[0]["evidence_representation"], "frozen_page_text")
        self.assertTrue(records[0]["captured_search_records"])
        packets = runtime.export_human(bundle, primary, self.root / "human", sample_size=2)
        self.assertEqual(len(packets), 2)
        self.assertNotIn("generator_model_id", json.dumps(packets))
        self.assertNotIn("condition", packets[0]["input"])
        calibration = json.loads((self.root / "human/calibration_manifest.json").read_text())
        self.assertTrue(calibration["synthetic_inputs"])
        self.assertTrue(calibration["fake_backend"])
        with patch.object(runtime, "VllmChatClient") as client_factory:
            with self.assertRaises(ValueError):
                runtime.main(["run-primary", "--bundle-dir", str(bundle), "--model-configuration-id", model.configuration_id,
                    "--output-dir", str(primary), "--base-url", "http://example.invalid", "--server-model-revision", model.model_revision, "--resume"])
            client_factory.assert_not_called()
