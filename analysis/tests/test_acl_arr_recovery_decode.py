"""Bounded, auditable pilot-only duplicate regeneration."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import unittest

from analysis.interpretability.pipeline.acl_arr_document_experiment import validate_rerank_output
from analysis.scripts.run_acl_arr_vllm import _rerank_schema
from analysis.scripts.acl_arr_recovery_decode import recover_one


class RecordingClient:
    maximum_attempts = 1

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        raw = response if isinstance(response, str) else json.dumps({"ranked_document_ids": response})
        return raw, {"prompt_tokens": 123, "completion_tokens": 12}


def prepared_item(count=3):
    allowed = [f"C{i:03d}" for i in range(1, count + 2)]
    return {
        "base": {"task_id": "recovery-task", "pipeline": "rerank", "input_document_ids": allowed},
        "prompt": "Original prompt and frozen evidence.",
        "schema_name": "acl_arr_rerank",
        "schema": _rerank_schema(count),
        "temperature": 0.0,
        "seed": 1903,
        "max_tokens": 256,
        "validator": lambda raw: validate_rerank_output(raw, allowed_document_ids=allowed, output_count=count),
    }


class RecoveryDecodeTests(unittest.TestCase):
    def run_case(self, responses, item=None):
        item = prepared_item() if item is None else item
        client = RecordingClient(responses)
        events = []
        result = asyncio.run(recover_one(item, client=client, emit_attempt=events.append))
        return result, client.calls, events

    def test_valid_first_response_uses_exact_enum_and_original_validator(self):
        result, calls, events = self.run_case([["C003", "C001", "C004"]])
        self.assertTrue(result["ok"])
        self.assertEqual(result["decode_attempt_count"], 1)
        self.assertEqual(result["parsed_output"], {"ranked_document_ids": ["C003", "C001", "C004"]})
        array = calls[0]["schema"]["properties"]["ranked_document_ids"]
        self.assertEqual(array["items"]["enum"], prepared_item()["base"]["input_document_ids"])
        self.assertEqual((array["minItems"], array["maxItems"]), (3, 3))
        self.assertNotIn("uniqueItems", array)
        self.assertIsNone(events[0]["validator_error"])

    def test_duplicate_regeneration_freezes_prefix_and_excludes_prefix_ids(self):
        result, calls, events = self.run_case([
            ["C003", "C003", "C002"],
            ["C003", "C004", "C004"],
            ["C003", "C004", "C001"],
        ])
        self.assertTrue(result["ok"])
        self.assertEqual(result["decode_attempt_count"], 3)
        for index, prefix in [(1, ["C003"]), (2, ["C003", "C004"])]:
            array = calls[index]["schema"]["properties"]["ranked_document_ids"]
            self.assertEqual(array["prefixItems"], [{"const": value} for value in prefix])
            self.assertFalse(set(array["items"]["enum"]) & set(prefix))
            self.assertNotIn("uniqueItems", array)
        self.assertIn("duplicate", events[0]["validator_error"])
        self.assertEqual(events[1]["prefix"], ["C003"])

    def test_recovery_does_not_change_prompt_seed_budget_temperature_or_input(self):
        item = prepared_item()
        before = deepcopy(item)
        _, calls, _ = self.run_case([["C001", "C001", "C002"], ["C001", "C003", "C004"]], item)
        self.assertEqual(item, before)
        for call in calls:
            for name in ("prompt", "schema_name", "temperature", "max_tokens", "seed"):
                self.assertEqual(call[name], item[name])

    def test_unknown_ids_fail_without_fabricating_or_retrying(self):
        result, calls, events = self.run_case([["C001", "C999", "C003"]])
        self.assertFalse(result["ok"])
        self.assertIn("unknown", result["error"])
        self.assertEqual(len(calls), 1)
        self.assertIn("C999", events[0]["raw_output"])

    def test_ablated_id_is_excluded_from_the_schema(self):
        item = prepared_item()
        item["base"]["input_document_ids"] = ["C001", "C003", "C004"]
        result, calls, _ = self.run_case([["C001", "C002", "C003"]], item)
        self.assertFalse(result["ok"])
        self.assertIn("unknown document IDs: C002", result["error"])
        self.assertEqual(calls[0]["schema"]["properties"]["ranked_document_ids"]["items"]["enum"],
                         ["C001", "C003", "C004"])

    def test_prior_prefix_cannot_change(self):
        result, calls, events = self.run_case([["C001", "C001", "C002"], ["C002", "C003", "C004"]])
        self.assertFalse(result["ok"])
        self.assertIn("prefix", result["error"])
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(events[-1]["validator_error"])

    def test_suffix_cannot_repeat_a_frozen_prefix_id(self):
        result, calls, _ = self.run_case([["C001", "C001", "C002"], ["C001", "C003", "C001"]])
        self.assertFalse(result["ok"])
        self.assertIn("prefix", result["error"])
        self.assertEqual(len(calls), 2)

    def test_non_growing_duplicate_response_stops(self):
        result, calls, _ = self.run_case([["C001", "C001", "C002"], ["C001", "C001", "C002"]])
        self.assertFalse(result["ok"])
        self.assertEqual(len(calls), 2)

    def test_wrong_shape_or_length_is_not_repaired(self):
        for raw in ('[]', '{"ranked_document_ids": ["C001"]}',
                    '{"ranked_document_ids": ["C001", "C002", 3]}',
                    '{"ranked_document_ids": ["C001", "C002", "C003"], "extra": 1}',
                    '{"ranked_document_ids": [], "ranked_document_ids": ["C001", "C002", "C003"]}',
                    'not JSON'):
            with self.subTest(raw=raw):
                result, calls, events = self.run_case([raw])
                self.assertFalse(result["ok"])
                self.assertEqual(len(calls), 1)
                self.assertIsNotNone(events[0]["validator_error"])

    def test_http_error_is_logged_once(self):
        result, calls, events = self.run_case([RuntimeError("HTTP 400: context overflow")])
        self.assertFalse(result["ok"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["decode_attempt_count"], 1)
        self.assertIn("HTTP 400", events[0]["validator_error"])
        self.assertIsNone(events[0]["raw_output"])
        self.assertEqual(events[0]["usage"], {})

    def test_attempt_records_include_verifiable_hashes_and_times(self):
        _, calls, events = self.run_case([["C001", "C001", "C002"], ["C001", "C003", "C004"]])
        for number, event in enumerate(events, 1):
            self.assertEqual(event["task_id"], "recovery-task")
            self.assertEqual(event["attempt"], number)
            self.assertEqual(event["schema"], calls[number - 1]["schema"])
            self.assertEqual(event["schema_sha256"], hashlib.sha256(json.dumps(event["schema"], sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            self.assertEqual(event["raw_output_sha256"], hashlib.sha256(event["raw_output"].encode()).hexdigest())
            self.assertLessEqual(event["started_at"], event["finished_at"])
            self.assertEqual(event["usage"]["prompt_tokens"], 123)

    def test_request_bound_is_output_count(self):
        item = prepared_item(5)
        responses = [["C001", "C001", "C001", "C001", "C001"],
                     ["C001", "C002", "C002", "C002", "C002"],
                     ["C001", "C002", "C003", "C003", "C003"],
                     ["C001", "C002", "C003", "C004", "C004"],
                     ["C001", "C002", "C003", "C004", "C005"]]
        result, calls, _ = self.run_case(responses, item)
        self.assertTrue(result["ok"])
        self.assertEqual(len(calls), 5)

    def test_one_item_ranking_needs_at_most_one_attempt(self):
        result, calls, _ = self.run_case([["C002"]], prepared_item(1))
        self.assertTrue(result["ok"])
        self.assertEqual(len(calls), 1)

    def test_infeasible_count_is_rejected_without_requests(self):
        item = prepared_item()
        item["schema"] = _rerank_schema(5)
        result, calls, events = self.run_case([], item)
        self.assertFalse(result["ok"])
        self.assertEqual(result["decode_attempt_count"], 0)
        self.assertEqual(calls, [])
        self.assertEqual(events, [])

    def test_original_validator_failure_is_not_overridden(self):
        item = prepared_item()
        def reject(raw):
            raise ValueError("original validator rejected")
        item["validator"] = reject
        result, calls, _ = self.run_case([["C001", "C002", "C003"]], item)
        self.assertFalse(result["ok"])
        self.assertIn("original validator rejected", result["error"])
        self.assertEqual(len(calls), 1)

    def test_unsafe_network_retry_configuration_is_rejected(self):
        client = RecordingClient([])
        client.maximum_attempts = 3
        result = asyncio.run(recover_one(prepared_item(), client=client, emit_attempt=lambda _: None))
        self.assertFalse(result["ok"])
        self.assertEqual(client.calls, [])

    def test_logging_failure_propagates_before_success_or_retry(self):
        def fail_log(event):
            raise OSError("audit disk full")
        client = RecordingClient([["C001", "C002", "C003"]])
        with self.assertRaisesRegex(OSError, "audit disk full"):
            asyncio.run(recover_one(prepared_item(), client=client, emit_attempt=fail_log))
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
