import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from analysis.scripts import run_search_experience as runtime
from analysis.scripts import search_vllm_stage


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

    def serving_profile(
        self,
        path,
        *,
        model_id="fixture/model",
        revision="a" * 40,
        data_parallel_size=1,
        tensor_parallel_size=4,
    ):
        record = search_vllm_stage.build_profile(
            stage="fixture-primary",
            model_id=model_id,
            model_revision=revision,
            vllm_executable="/runtime/bin/vllm",
            vllm_version="0.28.0",
            vllm_help="--data-parallel-size --language-model-only",
            visible_gpus=tuple({
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "NVIDIA GH200 120GB",
                "memory_total_mib": 97871,
            } for index in range(4)),
            cuda_visible_devices="0,1,2,3",
            expected_gpu_name_pattern="GH200",
            max_model_len=41472,
            language_model_only=True,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        search_vllm_stage.create_or_verify_profile(path, record)
        return record

    def serving_runtime(self, root, profile_path, profile, job_id, result_hash):
        approval_path = root / f"approval-{job_id}.json"
        approval_path.write_text(json.dumps({
            "format_version": search_vllm_stage.APPROVAL_FORMAT_VERSION,
            "profile_sha256": profile["profile_sha256"],
            "benchmark_result_sha256": result_hash,
            "approved_for_scientific_use": True,
        }))
        approval = search_vllm_stage.benchmark_approval_binding(
            profile,
            approval_path,
        )
        caches = search_vllm_stage.cache_environment(
            profile,
            job_id=job_id,
            cache_base=root / "cache",
            hostname="jpbo-001-04",
        )
        runtime_record = search_vllm_stage.build_runtime_record(
            profile_path=profile_path,
            profile=profile,
            visible_gpus=tuple({
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "NVIDIA GH200 120GB",
                "memory_total_mib": 97871,
            } for index in range(4)),
            cuda_visible_devices="0,1,2,3",
            cache_paths=caches,
            hostname="jpbo-001-04",
            job_id=job_id,
            step_id=f"{job_id}.1",
            benchmark_approval=approval,
        )
        runtime_record.update(
            server_status="terminated",
            server_exit_code=0,
            controller_status="exited",
            controller_exit_code=0,
            finished_at_epoch_seconds=time.time(),
        )
        server_log = root / f"server-{job_id}.log"
        runtime_path = search_vllm_stage.write_runtime_record(
            server_log,
            runtime_record,
        )
        return runtime_path, approval

    def test_new_manifest_binds_and_revalidates_serving_profile(self):
        output = self.root / "results"
        profile_path = self.root / "results.serving-profile.json"
        profile = self.serving_profile(profile_path)
        self.assertEqual(self.execute(output, serving_profile=profile_path), 0)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["serving_profile"], {
            "path": str(profile_path.resolve()),
            "sha256": profile["profile_sha256"],
        })
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        profile_path.unlink()
        with self.assertRaisesRegex(ValueError, "serving profile"):
            self.execute(output, resume=True)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_manifest_links_runtime_record_and_dp_cannot_bypass_approval(self):
        output = self.root / "results"
        profile_path = self.root / "results.serving-profile.json"
        profile = self.serving_profile(
            profile_path,
            data_parallel_size=2,
            tensor_parallel_size=2,
        )
        with self.assertRaisesRegex(ValueError, "runtime record"):
            self.execute(output, serving_profile=profile_path)
        runtime_path, approval = self.serving_runtime(
            self.root,
            profile_path,
            profile,
            "1725979",
            "d" * 64,
        )
        with patch.dict(
            "os.environ",
            {"GEODML_SERVING_RUNTIME_RECORD": str(runtime_path)},
        ):
            self.assertEqual(
                self.execute(
                    output,
                    max_tasks=1,
                    serving_profile=profile_path,
                ),
                3,
            )
        first_manifest = json.loads((output / "run_manifest.json").read_text())
        first_run_id = first_manifest["run_id"]
        second_runtime_path, second_approval = self.serving_runtime(
            self.root,
            profile_path,
            profile,
            "1725980",
            "e" * 64,
        )
        with patch.dict(
            "os.environ",
            {"GEODML_SERVING_RUNTIME_RECORD": str(second_runtime_path)},
        ):
            self.assertEqual(
                self.execute(
                    output,
                    resume=True,
                    serving_profile=profile_path,
                ),
                0,
            )
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(
            manifest["serving_runtime"]["path"],
            str(second_runtime_path.resolve()),
        )
        invocations = manifest["serving_invocations"]
        self.assertEqual(len(invocations), 2)
        self.assertEqual(invocations[0]["run_id"], first_run_id)
        self.assertEqual(invocations[0]["serving_runtime"]["benchmark_approval"], approval)
        self.assertEqual(invocations[1]["run_id"], manifest["run_id"])
        self.assertEqual(
            invocations[1]["serving_runtime"]["benchmark_approval"],
            second_approval,
        )
        self.assertEqual(
            {entry["serving_runtime"]["slurm_job_id"] for entry in invocations},
            {"1725979", "1725980"},
        )
        outcome_run_ids = {
            json.loads(line)["run_id"]
            for line in (output / "outcomes.jsonl").read_text().splitlines()
        }
        self.assertEqual(outcome_run_ids, {entry["run_id"] for entry in invocations})
        manifest["serving_invocations"] = manifest["serving_invocations"][1:]
        (output / "run_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "journal run ID"):
            runtime._resume(
                output,
                self.items,
                runtime._run_identity(self.items, self.identity),
                serving_profile=manifest["serving_profile"],
            )

    def test_partial_bound_run_requires_same_profile_on_resume(self):
        output = self.root / "results"
        profile_path = self.root / "results.serving-profile.json"
        self.serving_profile(profile_path)
        self.assertEqual(
            self.execute(output, max_tasks=1, serving_profile=profile_path),
            3,
        )
        self.assertEqual(
            runtime._resume(
                output,
                self.items,
                runtime._run_identity(self.items, self.identity),
            ),
            {"task-0"},
        )
        self.assertEqual(
            runtime.preflight_run(
                self.items,
                output,
                self.identity,
                {},
                resume=True,
            ),
            {"task-0"},
        )
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaisesRegex(ValueError, "requires an explicit serving profile"):
            self.execute(output, resume=True)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        self.assertEqual(
            self.execute(output, resume=True, serving_profile=profile_path),
            0,
        )

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
        expanded_items, expanded_identity, expanded_hashes = runtime.primary_items(
            bundle,
            model.configuration_id,
            answer_max_tokens=1536,
        )
        legacy_items, legacy_identity, legacy_hashes = runtime.primary_items(
            bundle,
            model.configuration_id,
            answer_schema_contract=None,
        )
        self.assertEqual(expanded_hashes, hashes)
        self.assertEqual(legacy_hashes, hashes)
        self.assertNotIn("answer_max_tokens_override", identity)
        self.assertEqual(expanded_identity["answer_max_tokens_override"], 1536)
        self.assertEqual(
            identity["answer_schema_contract"],
            runtime._contract().ANSWER_SCHEMA_CONTRACT,
        )
        self.assertNotIn("answer_schema_contract", legacy_identity)
        legacy_answer = next(
            item for item in legacy_items if item["base"]["pipeline"] == "answer"
        )
        self.assertEqual(legacy_answer["schema_name"], "search_experience_answer_v1")
        self.assertNotIn(
            "enum",
            legacy_answer["schema"]["properties"]["claims"]["items"]
            ["properties"]["cited_document_ids"]["items"],
        )
        by_legacy_task = {item["base"]["legacy_task_id"]: item for item in items}
        expanded_by_legacy_task = {
            item["base"]["legacy_task_id"]: item for item in expanded_items
        }
        for legacy_task_id, item in by_legacy_task.items():
            expanded = expanded_by_legacy_task[legacy_task_id]
            if item["base"]["pipeline"] == "answer":
                self.assertEqual(item["max_tokens"], model.answer_max_tokens)
                self.assertEqual(expanded["max_tokens"], 1536)
                self.assertEqual(
                    expanded["schema"]["properties"]["claims"]["items"]
                    ["properties"]["cited_document_ids"]["items"]["enum"],
                    expanded["base"]["input_document_ids"],
                )
                self.assertNotEqual(
                    expanded["base"]["task_id"],
                    item["base"]["task_id"],
                )
            else:
                self.assertEqual(expanded["max_tokens"], item["max_tokens"])
                self.assertEqual(
                    runtime._request_sha256(expanded),
                    runtime._request_sha256(item),
                )
                self.assertEqual(
                    expanded["base"]["task_id"],
                    item["base"]["task_id"],
                )
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
        primary_profile_path = self.root / "primary.serving-profile.json"
        self.serving_profile(
            primary_profile_path,
            model_id=model.model_id,
            revision=model.model_revision,
        )
        self.assertEqual(asyncio.run(runtime.run_prepared(items, primary, client=Client(), identity=identity,
            source_hashes=hashes, serving_profile=primary_profile_path)), 0)
        for item in legacy_items:
            item["base"]["fake_backend"] = True
        legacy_identity["fake_backend"] = True
        legacy_primary = self.root / "legacy-primary"
        self.assertEqual(asyncio.run(runtime.run_prepared(
            legacy_items,
            legacy_primary,
            client=Client(),
            identity=legacy_identity,
            source_hashes=legacy_hashes,
            serving_profile=primary_profile_path,
        )), 0)
        legacy_judges, _, _ = runtime.judge_items(
            bundle,
            legacy_primary,
            "independent/model",
            "c" * 40,
        )
        self.assertEqual(len(legacy_judges), 9)
        for item in expanded_items:
            item["base"]["fake_backend"] = True
        expanded_identity["fake_backend"] = True
        expanded_primary = self.root / "expanded-primary"
        self.assertEqual(asyncio.run(runtime.run_prepared(
            expanded_items,
            expanded_primary,
            client=Client(),
            identity=expanded_identity,
            source_hashes=expanded_hashes,
            serving_profile=primary_profile_path,
        )), 0)
        expanded_judges, _, _ = runtime.judge_items(
            bundle,
            expanded_primary,
            "independent/model",
            "c" * 40,
        )
        self.assertEqual(len(expanded_judges), 9)
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

        frozen_primary = {p.name: p.read_bytes() for p in primary.iterdir()}
        quote_judges, quote_identity, quote_hashes = runtime.judge_items(
            bundle, primary, "independent/model", "c" * 40,
            judge_contract=contract.QUOTE_JUDGE_CONTRACT)
        expanded_quote_judges, expanded_quote_identity, _ = runtime.judge_items(
            bundle, primary, "independent/model", "c" * 40,
            judge_contract=contract.QUOTE_JUDGE_CONTRACT, judge_max_tokens=4096)
        self.assertTrue(all(item["max_tokens"] == 4096 for item in expanded_quote_judges))
        self.assertEqual(expanded_quote_identity["judge_max_tokens_override"], 4096)
        self.assertTrue({item["base"]["task_id"] for item in quote_judges}.isdisjoint(
            {item["base"]["task_id"] for item in expanded_quote_judges}))
        with self.assertRaisesRegex(ValueError, "judge max tokens must be positive"):
            runtime.judge_items(bundle, primary, "independent/model", "c" * 40,
                judge_contract=contract.QUOTE_JUDGE_CONTRACT, judge_max_tokens=0)
        self.assertTrue({i["base"]["task_id"] for i in judges}.isdisjoint(
            {i["base"]["task_id"] for i in quote_judges}))
        for item in quote_judges:
            visible = json.loads(item["prompt"].split("\n\n", 1)[1])
            assessments = []
            for claim in visible["answer"]["claims"]:
                doc = next(d for d in visible["documents"]
                           if d["document_id"] == claim["cited_document_ids"][0])
                assessments.append({"claim_id": claim["claim_id"], "support": "supported",
                    "citation_correctness": "correct", "evidence": [{
                        "document_id": doc["document_id"], "quote": doc["text"][:64]}]})
            responses[item["prompt"]] = json.dumps({**{k: 3 for k in contract.SCORE_FIELDS},
                                                     "claim_assessments": assessments})
        quote_output = self.root / "quote-judge"
        self.assertEqual(asyncio.run(runtime.run_prepared(quote_judges, quote_output,
            client=Client(), identity=quote_identity, source_hashes=quote_hashes, max_tasks=1)), 3)
        self.assertEqual(asyncio.run(runtime.run_prepared(quote_judges, quote_output,
            client=Client(), identity=quote_identity, source_hashes=quote_hashes, resume=True)), 0)
        quote_records = runtime.inspect_bundle(bundle, primary, quote_output, self.root / "quote-report")
        self.assertTrue(all(r["complete"] for r in quote_records))
        expanded_quote_output = self.root / "expanded-quote-judge"
        self.assertEqual(asyncio.run(runtime.run_prepared(
            expanded_quote_judges, expanded_quote_output, client=Client(),
            identity=expanded_quote_identity, source_hashes=quote_hashes)), 0)
        expanded_quote_records = runtime.inspect_bundle(
            bundle, primary, expanded_quote_output, self.root / "expanded-quote-report")
        self.assertTrue(all(record["complete"] for record in expanded_quote_records))
        saved_quote = {p.name: p.read_bytes() for p in quote_output.iterdir()}
        with self.assertRaises(ValueError):
            runtime.preflight_run(judges, quote_output, judge_identity, judge_hashes, resume=True)
        self.assertEqual(saved_quote, {p.name: p.read_bytes() for p in quote_output.iterdir()})
        self.assertEqual(frozen_primary, {p.name: p.read_bytes() for p in primary.iterdir()})
        check_args = ["run-judge", "--bundle-dir", str(bundle), "--primary-output", str(primary),
            "--judge-model-id", "independent/model", "--judge-model-revision", "c" * 40,
            "--judge-contract", contract.QUOTE_JUDGE_CONTRACT, "--base-url", "http://127.0.0.1:1/v1",
            "--output-dir", str(quote_output), "--resume", "--preflight-only"]
        with patch.object(runtime, "VllmChatClient") as client_factory:
            self.assertEqual(runtime.main(check_args), 0)
            fresh_args = list(check_args)
            fresh_args[fresh_args.index(str(quote_output))] = str(self.root / "fresh-judge")
            fresh_args.remove("--resume")
            fresh_args.extend(("--judge-max-tokens", "4096"))
            self.assertEqual(runtime.main(fresh_args), 3)
            self.assertFalse((self.root / "fresh-judge").exists())
            client_factory.assert_not_called()
        self.assertEqual(saved_quote, {p.name: p.read_bytes() for p in quote_output.iterdir()})

        ranking = next(i for i in items if i["base"]["pipeline"] == "rerank")
        responses[ranking["prompt"]] = '{"ranked_document_ids": ["REMOVED"]}'
        incomplete_ranking = self.root / "incomplete-ranking"
        self.assertEqual(asyncio.run(runtime.run_prepared(
            items, incomplete_ranking, client=Client(), identity=identity,
            source_hashes=hashes)), 2)
        saved = {p.name: p.read_bytes() for p in incomplete_ranking.iterdir()}
        partial_judges, _, _ = runtime.judge_items(
            bundle, incomplete_ranking, "independent/model", "c" * 40)
        self.assertEqual(len(partial_judges), 9)
        self.assertEqual({i["base"]["task_id"]: runtime._request_sha256(i) for i in judges},
                         {i["base"]["task_id"]: runtime._request_sha256(i) for i in partial_judges})
        self.assertEqual(saved, {p.name: p.read_bytes() for p in incomplete_ranking.iterdir()})
        record_path = incomplete_ranking / "run_manifest.json"
        record = json.loads(record_path.read_text())
        record["status"] = "running"
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "primary writer must finish"):
            runtime.judge_items(bundle, incomplete_ranking, "independent/model", "c" * 40)
        record_path.write_bytes(saved["run_manifest.json"])
        incomplete_report = runtime.inspect_bundle(
            bundle, incomplete_ranking, None, self.root / "incomplete-report")
        self.assertTrue(any(r["ranking"] is None for r in incomplete_report))
        self.assertFalse(all(r["complete"] for r in incomplete_report))

        answer = next(i for i in items if i["base"]["pipeline"] == "answer")
        responses[answer["prompt"]] = '{}'
        incomplete_answer = self.root / "incomplete-answer"
        self.assertEqual(asyncio.run(runtime.run_prepared(
            items, incomplete_answer, client=Client(), identity=identity,
            source_hashes=hashes)), 2)
        with self.assertRaisesRegex(ValueError, "complete answer coverage"):
            runtime.judge_items(bundle, incomplete_answer, "independent/model", "c" * 40)
        with patch.object(runtime, "VllmChatClient") as client_factory:
            with self.assertRaises(ValueError):
                runtime.main(["run-primary", "--bundle-dir", str(bundle), "--model-configuration-id", model.configuration_id,
                    "--output-dir", str(primary), "--base-url", "http://example.invalid", "--server-model-revision", model.model_revision, "--resume"])
            client_factory.assert_not_called()

        saved_profile = primary_profile_path.read_bytes()
        primary_profile_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "serving profile"):
            runtime.judge_items(bundle, primary, "independent/model", "c" * 40)
        with self.assertRaisesRegex(ValueError, "serving profile"):
            runtime.inspect_bundle(bundle, primary, None, self.root / "tampered-report")
        primary_profile_path.write_bytes(saved_profile)
