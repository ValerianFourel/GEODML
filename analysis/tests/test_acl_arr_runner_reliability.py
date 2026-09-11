"""CPU contracts for durable generic execution and bounded scheduling."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace

from analysis.scripts import run_acl_arr_vllm as runner
from analysis.interpretability.pipeline.acl_arr_document_experiment import (
    build_acl_arr_experiment_plan, write_acl_arr_experiment_plan,
)
from analysis.tests.test_acl_arr_document_experiment import _prompts, _axis_rows, _document_sets, _models


def prepared(i):
    return dict(base={"task_id": str(i)}, prompt=str(i), schema_name="test",
                schema={}, temperature=0, max_tokens=10, seed=i,
                validator=json.loads, fake_output="{}")


class ReliabilityTests(unittest.TestCase):
    def test_restart_journal_is_streamed_without_whole_file_read(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'attempts.jsonl'
            path.write_text('{"task_id":"one"}\n{"task_id":"two"}\n')
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole file read')):
                self.assertEqual([r['task_id'] for r in runner._journal_rows(path)], ['one', 'two'])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = _models()[0]
        plan = build_acl_arr_experiment_plan(_prompts(), _axis_rows(), _document_sets(),
                                            models=(model,), top_n=2)
        artifacts = write_acl_arr_experiment_plan(self.root / "plan", plan=plan)
        self.argv = ["primary", "--tasks", str(artifacts.task_files[(model.configuration_id, "rerank")]),
                     "--plan-manifest", str(artifacts.manifest_path), "--output-dir", str(self.root / "out"), "--fake"]

    def run_cli(self, *extra):
        return asyncio.run(runner._run(runner._parser().parse_args([*self.argv, *extra])))

    def test_partial_is_checkpointed_then_resume_completes(self):
        self.assertEqual(self.run_cli("--max-tasks", "1"), 3)
        manifest = json.loads((self.root / "out/run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "checkpointed")
        self.assertGreater(manifest["remaining_count"], 0)
        self.assertEqual(self.run_cli("--resume"), 0)

    def test_pilot_rows_are_explicitly_ineligible(self):
        self.assertEqual(self.run_cli("--pilot-only"), 0)
        for line in (self.root / "out/outcomes.jsonl").read_text().splitlines():
            row = json.loads(line)
            self.assertIs(row["scientific_result"], False)
            self.assertIs(row["eligible_for_analysis"], False)
        self.assertEqual(self.run_cli("--resume", "--pilot-only"), 0)

    def test_corrupt_failure_or_attempt_journal_is_rejected(self):
        self.run_cli()
        for name in ("failures.jsonl", "attempts.jsonl"):
            path = self.root / "out" / name
            path.write_text("{}\n")
            with self.assertRaises(ValueError):
                self.run_cli("--resume")
            self.assertEqual(path.read_text(), "{}\n")
            path.write_text("")

    def test_bad_resume_never_mutates_artifacts(self):
        self.run_cli()
        path = self.root / "out/outcomes.jsonl"
        original = path.read_text()
        first = json.loads(original.splitlines()[0])
        mutations = [original + original.splitlines()[0] + "\n", original + "{",
                     original.rstrip("\n"), json.dumps({**first, "task_id": "foreign"}) + "\n",
                     json.dumps({**first, "fake_backend": False}) + "\n",
                     json.dumps({**first, "raw_output": "invalid"}) + "\n"]
        for value in mutations:
            with self.subTest(value=value[-60:]):
                path.write_text(value)
                before = {p.name: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
                with self.assertRaises(ValueError):
                    self.run_cli("--resume")
                self.assertEqual(before, {p.name: p.read_bytes() for p in path.parent.iterdir() if p.is_file()})

    def test_invalid_response_keeps_raw_and_usage(self):
        class Client:
            async def complete(self, **kwargs):
                return "not JSON", {"completion_tokens": 3}
        results = asyncio.run(runner._execute([prepared(1)], client=Client(), maximum_concurrency=1, fake=False))
        self.assertFalse(results[0]["ok"])
        self.assertEqual(results[0]["raw_output"], "not JSON")
        self.assertEqual(results[0]["usage"], {"completion_tokens": 3})

    def test_resume_identity_and_legacy_rejected_before_mutation(self):
        self.run_cli()
        manifest_path = self.root / "out/run_manifest.json"
        original = json.loads(manifest_path.read_text())
        for key in ("model_revision", "fake_backend", "pilot_only", "maximum_attempts", "source_manifest_sha256"):
            record = json.loads(json.dumps(original))
            record["resume_identity"][key] = "changed"
            manifest_path.write_text(json.dumps(record))
            before = manifest_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "identity"):
                self.run_cli("--resume")
            self.assertEqual(before, manifest_path.read_bytes())
        original.pop("resume_identity")
        manifest_path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.run_cli("--resume")

    def test_writer_lock_excludes_other_owner(self):
        with runner._writer_lock(self.root):
            with self.assertRaisesRegex(ValueError, "writer"):
                with runner._writer_lock(self.root):
                    self.fail("second writer acquired ownership")

    def test_http_identity_failure_closes_session(self):
        session = SimpleNamespace(close=AsyncMock())
        aiohttp = SimpleNamespace(ClientTimeout=lambda **kw: None, ClientSession=lambda **kw: session)
        client = runner.VllmChatClient(base_url="http://example/v1", api_key=None,
            server_model_name="model", timeout_seconds=1, maximum_attempts=3)
        with patch.dict("sys.modules", {"aiohttp": aiohttp}), patch.object(client, "verify_server_identity", AsyncMock(side_effect=RuntimeError("identity"))):
            with self.assertRaisesRegex(RuntimeError, "identity"):
                asyncio.run(client.__aenter__())
        session.close.assert_awaited_once()

    def test_audit_failure_does_not_retry_and_payload_is_unchanged(self):
        payloads, events = [], []
        class Response:
            status = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def text(self):
                return json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}})
        class Session:
            def post(self, url, json):
                payloads.append(json)
                return Response()
        def audit(event):
            events.append(event)
            if event["event"] == "end":
                raise OSError("disk full")
        client = runner.VllmChatClient(base_url="http://example/v1", api_key=None,
            server_model_name="model", timeout_seconds=1, maximum_attempts=3, audit_callback=audit)
        client.session = Session()
        with self.assertRaises(runner.AuditWriteError):
            asyncio.run(runner._execute([prepared(1)], client=client, maximum_concurrency=1, fake=False))
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0], {"model": "model", "messages": [{"role": "user", "content": "1"}],
            "temperature": 0.0, "max_tokens": 10, "seed": 1,
            "response_format": {"type": "json_schema", "json_schema": {"name": "test", "schema": {}, "strict": True}}})
        self.assertEqual([e["event"] for e in events], ["start", "end"])
        self.assertEqual(events[0]["task_context"], {"task_id": "1"})

    def test_chat_template_kwargs_are_sent_and_audited(self):
        payloads, events = [], []

        class Response:
            status = 200

            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def text(self):
                return json.dumps({"choices": [{"message": {"content": "{}"}}]})

        class Session:
            def post(self, url, json):
                payloads.append(json)
                return Response()

        client = runner.VllmChatClient(
            base_url="http://example/v1",
            api_key=None,
            server_model_name="model",
            timeout_seconds=1,
            maximum_attempts=1,
            audit_callback=events.append,
            chat_template_kwargs={"enable_thinking": False},
        )
        client.session = Session()
        asyncio.run(client.complete(
            prompt="test",
            schema_name="test",
            schema={},
            temperature=0.0,
            max_tokens=10,
            seed=1,
        ))

        self.assertEqual(
            payloads[0]["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(
            events[0]["request"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_http_retries_keep_payload_and_emit_each_attempt(self):
        events, payloads = [], []
        class Response:
            status = 503
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def text(self): return "unavailable"
        class Session:
            def post(self, url, json):
                payloads.append(json)
                return Response()
        client = runner.VllmChatClient(base_url="http://example/v1", api_key=None,
            server_model_name="model", timeout_seconds=1, maximum_attempts=3, audit_callback=events.append)
        client.session = Session()
        with patch.object(runner.asyncio, "sleep", AsyncMock()) as sleep:
            result = asyncio.run(runner._execute([prepared(1)], client=client, maximum_concurrency=1, fake=False))
        self.assertFalse(result[0]["ok"])
        self.assertEqual(len(payloads), 3)
        self.assertTrue(all(p == payloads[0] for p in payloads))
        self.assertEqual([e["attempt"] for e in events], [1, 1, 2, 2, 3, 3])
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [0.5, 1.0])

    def test_rolling_refills_before_straggler_and_closes_active_calls(self):
        async def exercise():
            release = asyncio.Event()
            dispatched = []
            active = set()
            class Client:
                async def complete(self, **kwargs):
                    key = kwargs["prompt"]
                    dispatched.append(key)
                    active.add(key)
                    try:
                        if key == "0":
                            await release.wait()
                        return "{}", {}
                    finally:
                        active.remove(key)
            stream = runner._iter_execute((prepared(i) for i in range(8)), client=Client(), maximum_concurrency=2, fake=False)
            try:
                for _ in range(3):
                    await asyncio.wait_for(anext(stream), 1)
                self.assertIn("3", dispatched)
                self.assertIn("0", active)
            finally:
                await stream.aclose()
            self.assertFalse(active)
        asyncio.run(exercise())
